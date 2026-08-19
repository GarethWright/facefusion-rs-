using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/processors/modules/deep_swapper/{core,types,choices}.py</c> — swaps a
/// single fixed identity (a pre-trained DeepFaceLive "DFM" model) onto every selected target
/// face, no source image required.
///
/// <para>
/// <b><c>DeepSwapperModel : TypeAlias = str</c>, not an enum (deliberate divergence from
/// <see cref="FaceSwapperModel"/>).</b> Python's model set is neither a small closed
/// <c>Literal</c> nor fixed at compile time: <see cref="CreateStaticModelSet"/> below also
/// scans <c>.assets/models/custom/*.dfm</c> at call time and adds one <c>"custom/&lt;file
/// name&gt;"</c> entry per file found, so the full key set is only known once the local
/// filesystem is examined. A C# <c>enum</c> cannot represent that, so — matching Python's own
/// choice to keep <c>DeepSwapperModel</c> a bare <c>str</c> rather than a <c>Literal</c> here
/// (unlike <c>FaceSwapperModel</c>) — model identity stays a plain <see cref="string"/>
/// throughout this port too (the <c>"scope/name"</c> or <c>"custom/name"</c> key exactly as
/// Python builds it).
/// </para>
///
/// <para>
/// <b>No global state; sessions and settings taken as parameters (PORT_CONVENTIONS.md rule 5).</b>
/// Every <c>state_manager.get_item(...)</c> read becomes an explicit parameter — see
/// <see cref="DeepSwapperInputs"/>, matching <c>FaceSwapperInputs</c>'s precedent.
/// <c>get_inference_pool</c>'s real downloading is not reproduced for the same reason
/// <c>FaceSwapper</c> doesn't (no <c>facefusion/download.py</c> port); <see cref="PreCheck"/>
/// checks local file presence only. <see cref="ResolveModelsDirectory"/>/<see cref="BuildDownloadUrl"/>
/// reuse the same precedent <c>FaceSwapper</c> established, except the local file layout here
/// nests one directory deeper (<c>.assets/models/&lt;scope&gt;/&lt;name&gt;.dfm</c>, matching
/// Python's <c>resolve_relative_path('../.assets/models/' + model_scope + '/' + model_name +
/// '.dfm')</c>) and the download provider is <see cref="DownloadProvider.Huggingface"/>, not
/// <see cref="DownloadProvider.Github"/> (Python: <c>resolve_download_url_by_provider('huggingface',
/// 'deepfacelive-models-' + model_scope, model_name + '.hash')</c>).
/// </para>
///
/// <para>
/// <b>Model families covered by this port vs. parity-tested end to end.</b> Every one of the
/// ~155 <c>full</c>-scope model catalog entries (<see cref="CreateStaticModelSet"/>, transcribed
/// verbatim from Python's <c>model_config</c> literal lists) is ported; per the assignment's
/// fixture budget, real ONNX-Runtime parity (<c>tools/parity/dump_processors4.py</c>) is run
/// against exactly one family — <c>iperov/elon_musk_224</c>, Python's own configured default
/// (<c>--deep-swapper-model</c> default) and the smallest model in the <c>iperov</c> group,
/// which is present at both <c>lite</c> and <c>full</c> scope. Every preprocessing/postprocessing
/// method below (<see cref="PrepareCropFrame"/>/<see cref="NormalizeCropFrame"/>/
/// <see cref="PrepareCropMask"/>) is model-family-agnostic (the model only changes crop
/// resolution, read dynamically via <see cref="GetModelSize"/>, matching Python's own
/// <c>get_model_size()</c>), so one family's real inference run exercises the same code path
/// every other family would.
/// </para>
///
/// <para>
/// <b>Mat / dtype conventions.</b> <c>VisionFrame</c>s are <see cref="Mat"/>, <c>CV_8UC3</c> BGR,
/// caller-owned on every return, matching <c>FaceHelper</c>/<c>FaceSwapper</c>. Unlike
/// <c>FaceSwapper</c>, <see cref="PrepareCropFrame"/> keeps NHWC layout throughout (no CHW
/// transpose, no BGR-&gt;RGB flip — Python's own <c>prepare_crop_frame</c> does neither; the
/// DFM/TensorFlow-derived ONNX graph's <c>in_face:0</c> input is <c>(1, H, W, 3)</c> BGR), and
/// <see cref="NormalizeCropFrame"/> always narrows to <c>uint8</c> unconditionally (Python: a
/// single <c>.astype(numpy.uint8)</c>, no per-model-kind branch), so <see cref="SwapFace"/>
/// reuses <c>FaceFusion.Face.FaceHelper.PasteBack</c> directly, same as <c>ExpressionRestorer</c>.
/// </para>
/// </summary>
public static class DeepSwapper
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>deep_swapper_morph_range = create_int_range(0, 100, 1)</c>.</summary>
    public static readonly IReadOnlyList<int> DeepSwapperMorphRange =
        FaceFusion.Core.CommonHelper.CreateIntRange(0, 100, 1);

    // -----------------------------------------------------------------
    // create_static_model_set
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: one entry of <c>create_static_model_set</c>'s return value for the
    /// <c>deep_swapper</c> module. <see cref="Hashes"/> is <see langword="null"/> for a
    /// <c>custom/*</c> entry (Python: the custom-model branch's dict has no <c>'hashes'</c> key
    /// at all — <c>get_model_options().get('hashes')</c> returns <see langword="null"/> for
    /// those, which <see cref="PreCheck"/> and <see cref="Processor.PreCheck"/> both handle,
    /// matching Python's <c>if model_hash_set and model_source_set:</c> guard in
    /// <c>pre_check()</c>). <see cref="Types.WarpTemplate.DflWholeFace"/> is the one template
    /// value every entry (built-in or custom) uses. Crop size is not part of this catalog —
    /// Python's own <c>get_model_size()</c> reads it from the live ONNX session's own
    /// <c>in_face:0</c> input shape instead (see <see cref="GetModelSize"/>).
    /// </summary>
    public sealed record DeepSwapperModelOptions(
        IReadOnlyDictionary<string, Download>? Hashes,
        IReadOnlyDictionary<string, Download> Sources,
        WarpTemplate Template);

    private static readonly object ModelCatalogLock = new();
    private static readonly Dictionary<DownloadScope, IReadOnlyDictionary<string, DeepSwapperModelOptions>> CachedModelCatalogs = new();

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c> — cached per
    /// <paramref name="downloadScope"/> value here too, matching <c>lru_cache</c>'s per-argument
    /// cache).
    /// </summary>
    public static IReadOnlyDictionary<string, DeepSwapperModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        lock (ModelCatalogLock)
        {
            if (CachedModelCatalogs.TryGetValue(downloadScope, out var cached))
            {
                return cached;
            }

            var built = BuildModelCatalog(downloadScope);
            CachedModelCatalogs[downloadScope] = built;
            return built;
        }
    }

    // Python: the three `model_config.extend([...])` literal lists in create_static_model_set,
    // transcribed verbatim (scope, name) tuples, gated the same way ('full' only / 'lite'+'full'
    // / 'full' only).
    private static readonly (string Scope, string Name)[] FullOnlyModelsPartA =
    {
        ("druuzil", "adam_levine_320"), ("druuzil", "adrianne_palicki_384"), ("druuzil", "agnetha_falskog_224"),
        ("druuzil", "alan_ritchson_320"), ("druuzil", "alicia_vikander_320"), ("druuzil", "amber_midthunder_320"),
        ("druuzil", "andras_arato_384"), ("druuzil", "andrew_tate_320"), ("druuzil", "angelina_jolie_384"),
        ("druuzil", "anne_hathaway_320"), ("druuzil", "anya_chalotra_320"), ("druuzil", "arnold_schwarzenegger_320"),
        ("druuzil", "benjamin_affleck_320"), ("druuzil", "benjamin_stiller_384"), ("druuzil", "bradley_pitt_224"),
        ("druuzil", "brie_larson_384"), ("druuzil", "bruce_campbell_384"), ("druuzil", "bryan_cranston_320"),
        ("druuzil", "catherine_blanchett_352"), ("druuzil", "christian_bale_320"), ("druuzil", "christopher_hemsworth_320"),
        ("druuzil", "christoph_waltz_384"), ("druuzil", "cillian_murphy_320"), ("druuzil", "cobie_smulders_256"),
        ("druuzil", "dwayne_johnson_384"), ("druuzil", "edward_norton_320"), ("druuzil", "elisabeth_shue_320"),
        ("druuzil", "elizabeth_olsen_384"), ("druuzil", "elon_musk_320"), ("druuzil", "emily_blunt_320"),
        ("druuzil", "emma_stone_384"), ("druuzil", "emma_watson_320"), ("druuzil", "erin_moriarty_384"),
        ("druuzil", "eva_green_320"), ("druuzil", "ewan_mcgregor_320"), ("druuzil", "florence_pugh_320"),
        ("druuzil", "freya_allan_320"), ("druuzil", "gary_cole_224"), ("druuzil", "gigi_hadid_224"),
        ("druuzil", "harrison_ford_384"), ("druuzil", "hayden_christensen_320"), ("druuzil", "heath_ledger_320"),
        ("druuzil", "henry_cavill_448"), ("druuzil", "hugh_jackman_384"), ("druuzil", "idris_elba_320"),
        ("druuzil", "jack_nicholson_320"), ("druuzil", "james_carrey_384"), ("druuzil", "james_mcavoy_320"),
        ("druuzil", "james_varney_320"), ("druuzil", "jason_momoa_320"), ("druuzil", "jason_statham_320"),
        ("druuzil", "jennifer_connelly_384"), ("druuzil", "jimmy_donaldson_320"), ("druuzil", "jordan_peterson_384"),
        ("druuzil", "karl_urban_224"), ("druuzil", "kate_beckinsale_384"), ("druuzil", "laurence_fishburne_384"),
        ("druuzil", "lili_reinhart_320"), ("druuzil", "luke_evans_384"), ("druuzil", "mads_mikkelsen_384"),
        ("druuzil", "mary_winstead_320"), ("druuzil", "margaret_qualley_384"), ("druuzil", "melina_juergens_320"),
        ("druuzil", "michael_fassbender_320"), ("druuzil", "michael_fox_320"), ("druuzil", "millie_bobby_brown_320"),
        ("druuzil", "morgan_freeman_320"), ("druuzil", "patrick_stewart_224"), ("druuzil", "rachel_weisz_384"),
        ("druuzil", "rebecca_ferguson_320"), ("druuzil", "scarlett_johansson_320"), ("druuzil", "shannen_doherty_384"),
        ("druuzil", "seth_macfarlane_384"), ("druuzil", "thomas_cruise_320"), ("druuzil", "thomas_hanks_384"),
        ("druuzil", "william_murray_384"), ("druuzil", "zoe_saldana_384"),
        ("edel", "emma_roberts_224"), ("edel", "ivanka_trump_224"), ("edel", "lize_dzjabrailova_224"),
        ("edel", "sidney_sweeney_224"), ("edel", "winona_ryder_224"),
    };

    private static readonly (string Scope, string Name)[] LiteAndFullModels =
    {
        ("iperov", "alexandra_daddario_224"), ("iperov", "alexei_navalny_224"), ("iperov", "amber_heard_224"),
        ("iperov", "dilraba_dilmurat_224"), ("iperov", "elon_musk_224"), ("iperov", "emilia_clarke_224"),
        ("iperov", "emma_watson_224"), ("iperov", "erin_moriarty_224"), ("iperov", "jackie_chan_224"),
        ("iperov", "james_carrey_224"), ("iperov", "jason_statham_320"), ("iperov", "keanu_reeves_320"),
        ("iperov", "margot_robbie_224"), ("iperov", "natalie_dormer_224"), ("iperov", "nicolas_coppola_224"),
        ("iperov", "robert_downey_224"), ("iperov", "rowan_atkinson_224"), ("iperov", "ryan_reynolds_224"),
        ("iperov", "scarlett_johansson_224"), ("iperov", "sylvester_stallone_224"), ("iperov", "thomas_cruise_224"),
        ("iperov", "thomas_holland_224"), ("iperov", "vin_diesel_224"), ("iperov", "vladimir_putin_224"),
    };

    private static readonly (string Scope, string Name)[] FullOnlyModelsPartB =
    {
        ("jen", "angelica_trae_288"), ("jen", "ella_freya_224"), ("jen", "emma_myers_320"),
        ("jen", "evie_pickerill_224"), ("jen", "kang_hyewon_320"), ("jen", "maddie_mead_224"),
        ("jen", "nicole_turnbull_288"),
        ("mats", "alica_schmidt_320"), ("mats", "ashley_alexiss_224"), ("mats", "billie_eilish_224"),
        ("mats", "brie_larson_224"), ("mats", "cara_delevingne_224"), ("mats", "carolin_kebekus_224"),
        ("mats", "chelsea_clinton_224"), ("mats", "claire_boucher_224"), ("mats", "corinna_kopf_224"),
        ("mats", "florence_pugh_224"), ("mats", "hillary_clinton_224"), ("mats", "jenna_fischer_224"),
        ("mats", "kim_jisoo_320"), ("mats", "mica_suarez_320"), ("mats", "shailene_woodley_224"),
        ("mats", "shraddha_kapoor_320"), ("mats", "yu_jimin_352"),
        ("rumateus", "alison_brie_224"), ("rumateus", "amber_heard_224"), ("rumateus", "angelina_jolie_224"),
        ("rumateus", "aubrey_plaza_224"), ("rumateus", "bridget_regan_224"), ("rumateus", "cobie_smulders_224"),
        ("rumateus", "deborah_woll_224"), ("rumateus", "dua_lipa_224"), ("rumateus", "emma_stone_224"),
        ("rumateus", "hailee_steinfeld_224"), ("rumateus", "hilary_duff_224"), ("rumateus", "jessica_alba_224"),
        ("rumateus", "jessica_biel_224"), ("rumateus", "john_cena_224"), ("rumateus", "kim_kardashian_224"),
        ("rumateus", "kristen_bell_224"), ("rumateus", "lucy_liu_224"), ("rumateus", "margot_robbie_224"),
        ("rumateus", "megan_fox_224"), ("rumateus", "meghan_markle_224"), ("rumateus", "millie_bobby_brown_224"),
        ("rumateus", "natalie_portman_224"), ("rumateus", "nicki_minaj_224"), ("rumateus", "olivia_wilde_224"),
        ("rumateus", "shay_mitchell_224"), ("rumateus", "sophie_turner_224"), ("rumateus", "taylor_swift_224"),
    };

    private static IReadOnlyDictionary<string, DeepSwapperModelOptions> BuildModelCatalog(DownloadScope downloadScope)
    {
        var modelsDirectory = ResolveModelsDirectory();
        var huggingfaceProvider = Choices.DownloadProviderSet[DownloadProvider.Huggingface];

        var modelConfig = new List<(string Scope, string Name)>();

        if (downloadScope == DownloadScope.Full)
        {
            modelConfig.AddRange(FullOnlyModelsPartA);
        }

        if (downloadScope is DownloadScope.Lite or DownloadScope.Full)
        {
            modelConfig.AddRange(LiteAndFullModels);
        }

        if (downloadScope == DownloadScope.Full)
        {
            modelConfig.AddRange(FullOnlyModelsPartB);
        }

        var modelSet = new Dictionary<string, DeepSwapperModelOptions>();

        foreach (var (modelScope, modelName) in modelConfig)
        {
            var modelId = modelScope + "/" + modelName;
            var baseName = "deepfacelive-models-" + modelScope;

            var hashes = new Dictionary<string, Download>
            {
                ["deep_swapper"] = new Download(
                    BuildDownloadUrl(huggingfaceProvider, baseName, modelName + ".hash"),
                    Path.Combine(modelsDirectory, modelScope, modelName + ".hash")),
            };
            var sources = new Dictionary<string, Download>
            {
                ["deep_swapper"] = new Download(
                    BuildDownloadUrl(huggingfaceProvider, baseName, modelName + ".dfm"),
                    Path.Combine(modelsDirectory, modelScope, modelName + ".dfm")),
            };

            modelSet[modelId] = new DeepSwapperModelOptions(hashes, sources, WarpTemplate.DflWholeFace);
        }

        // Python: the `custom_model_file_paths` scan of `.assets/models/custom/*` — each file
        // found becomes a `'custom/<file_name>'` entry with a `sources`-only dict (no `'hashes'`
        // key, no URL — `path` alone, matching a locally supplied .dfm with nothing to fetch).
        var customModelsDirectory = Path.Combine(modelsDirectory, "custom");
        foreach (var customModelFilePath in FaceFusion.Core.FileSystem.ResolveFilePaths(customModelsDirectory))
        {
            var fileName = FaceFusion.Core.FileSystem.GetFileName(customModelFilePath);
            if (fileName is null)
            {
                continue;
            }

            var modelId = "custom/" + fileName;
            var sources = new Dictionary<string, Download>
            {
                ["deep_swapper"] = new Download(string.Empty, customModelFilePath),
            };

            modelSet[modelId] = new DeepSwapperModelOptions(null, sources, WarpTemplate.DflWholeFace);
        }

        return modelSet;
    }

    /// <summary>
    /// Same reasoning as <c>FaceSwapper.ResolveModelsDirectory</c>, walking up to the repository
    /// root marked by <c>FaceFusion.sln</c>.
    /// </summary>
    private static string ResolveModelsDirectory()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return Path.Combine(directory.FullName, ".assets", "models");
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the repository root (FaceFusion.sln) to resolve .assets/models.");
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    // -----------------------------------------------------------------
    // pre_check (file-presence only — see class remarks)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: the <c>deep_swapper</c>-specific half of <c>pre_check</c> — <c>if model_hash_set
    /// and model_source_set: return conditional_download_hashes(...) and
    /// conditional_download_sources(...) / return True</c> (a custom model with no
    /// <see cref="DeepSwapperModelOptions.Hashes"/> is vacuously "present", matching Python's
    /// fallthrough <c>return True</c> when the guard fails — a custom <c>.dfm</c> is either
    /// there or it isn't, and Python does not check that either without a hash to verify
    /// against). The common-module half is the caller's responsibility, per
    /// <see cref="IProcessor.GetCommonModules"/>'s remarks.
    /// </summary>
    public static bool PreCheck(string model)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[model];

        if (modelOptions.Hashes is null)
        {
            return true;
        }

        foreach (var download in modelOptions.Hashes.Values.Concat(modelOptions.Sources.Values))
        {
            if (!File.Exists(download.Path) || new FileInfo(download.Path).Length == 0)
            {
                return false;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------
    // get_model_size / has_morph_input
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>get_model_size</c>. Reads the live <paramref name="deepSwapperSession"/>'s own
    /// <c>in_face:0</c> input shape (Python: <c>deep_swapper_input.shape[1:3]</c> — a <c>(1, H,
    /// W, 3)</c> NHWC input, so indices 1 and 2 are height and width) rather than any static
    /// catalog entry (there is none — see <see cref="DeepSwapperModelOptions"/>'s remarks).
    /// Returns <c>(0, 0)</c> when the session has no such input, matching Python's fallthrough
    /// <c>return 0, 0</c>.
    /// </summary>
    public static Size GetModelSize(InferenceSession deepSwapperSession)
    {
        if (deepSwapperSession.InputMetadata.TryGetValue("in_face:0", out var metadata))
        {
            var height = checked((int)metadata.Dimensions[1]);
            var width = checked((int)metadata.Dimensions[2]);
            return new Size(width, height);
        }

        return new Size(0, 0);
    }

    /// <summary>Python: <c>has_morph_input</c>.</summary>
    public static bool HasMorphInput(InferenceSession deepSwapperSession)
        => deepSwapperSession.InputMetadata.ContainsKey("morph_value:0");

    // -----------------------------------------------------------------
    // prepare_crop_frame / normalize_crop_frame / prepare_crop_mask
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_crop_frame</c> — an unsharp-mask sharpen (<c>cv2.addWeighted(frame,
    /// 1.75, cv2.GaussianBlur(frame, (0, 0), 2), -0.75, 0)</c>, real OpenCV arithmetic, expect
    /// ~0 divergence per PARITY_HARNESS.md) followed by <c>/255.0</c>, <c>expand_dims(axis=0)</c>,
    /// <c>.astype(float32)</c>. Deliberately keeps NHWC/BGR layout throughout — no channel
    /// reversal, no CHW transpose (see the class remarks for why: <c>in_face:0</c> wants BGR
    /// NHWC). Returns a flat length-<c>H*W*3</c> array (batch dim implicit).
    /// </summary>
    public static float[] PrepareCropFrame(Mat cropVisionFrame)
    {
        using var blurred = new Mat();
        Cv2.GaussianBlur(cropVisionFrame, blurred, new Size(0, 0), 2);

        using var sharpened = new Mat();
        Cv2.AddWeighted(cropVisionFrame, 1.75, blurred, -0.75, 0, sharpened);

        var height = sharpened.Rows;
        var width = sharpened.Cols;
        var plane = height * width;
        var hwc = new float[plane * 3];

        sharpened.GetArray(out Vec3b[] pixels);

        for (var index = 0; index < plane; index++)
        {
            var pixel = pixels[index];

            // Python: `crop_vision_frame / 255.0` in float64 (uint8 array / python float),
            // narrowed to float32 only by the trailing `.astype(numpy.float32)`.
            hwc[(index * 3) + 0] = (float)(pixel.Item0 / 255.0); // B
            hwc[(index * 3) + 1] = (float)(pixel.Item1 / 255.0); // G
            hwc[(index * 3) + 2] = (float)(pixel.Item2 / 255.0); // R
        }

        return hwc;
    }

    /// <summary>
    /// Python: <c>normalize_crop_frame</c> — <c>(crop_vision_frame * 255.0).clip(0,
    /// 255).astype(uint8)</c>, no channel reversal (the model already outputs BGR NHWC, same
    /// layout <see cref="PrepareCropFrame"/> fed it). Caller owns the returned
    /// <see cref="Mat"/>.
    /// </summary>
    public static Mat NormalizeCropFrame(ReadOnlySpan<float> modelOutputHwc, int height, int width)
    {
        var plane = height * width;

        if (modelOutputHwc.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputHwc has {modelOutputHwc.Length} elements, expected {3 * plane} for a {width}x{height} HWC frame.", nameof(modelOutputHwc));
        }

        var result = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[plane];

        for (var index = 0; index < plane; index++)
        {
            pixels[index] = new Vec3b(
                ClampToByte255(modelOutputHwc[(index * 3) + 0]),
                ClampToByte255(modelOutputHwc[(index * 3) + 1]),
                ClampToByte255(modelOutputHwc[(index * 3) + 2]));
        }

        result.SetArray(pixels);
        return result;
    }

    private static byte ClampToByte255(float value)
    {
        var scaled = value * 255.0f;
        return scaled <= 0f ? (byte)0 : (scaled >= 255f ? (byte)255 : (byte)scaled);
    }

    /// <summary>
    /// Python: <c>prepare_crop_mask</c>. <paramref name="modelSize"/> is
    /// <see cref="GetModelSize"/>'s <c>(H, W)</c> pair — Python: <c>crop_mask.reshape(model_size)</c>,
    /// reshaping the flat mask back into <c>(H, W)</c> before the morphological ops.
    /// </summary>
    public static Mat PrepareCropMask(ReadOnlySpan<float> cropSourceMask, ReadOnlySpan<float> cropTargetMask, Size modelSize)
    {
        var height = modelSize.Height;
        var width = modelSize.Width;
        var plane = height * width;

        if (cropSourceMask.Length != plane || cropTargetMask.Length != plane)
        {
            throw new ArgumentException($"cropSourceMask/cropTargetMask must have {plane} elements for a {width}x{height} mask.");
        }

        using var reduced = new Mat(height, width, MatType.CV_32FC1);
        var data = new float[plane];
        for (var i = 0; i < plane; i++)
        {
            // Python: `numpy.minimum.reduce([crop_source_mask, crop_target_mask])`, then
            // `.reshape(model_size).clip(0, 1)`.
            data[i] = NumPy.Clip(Math.Min(cropSourceMask[i], cropTargetMask[i]), 0f, 1f);
        }

        reduced.SetArray(data);

        const int kernelSize = 3;
        const double blurSize = 6.25;

        using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(kernelSize, kernelSize));
        using var eroded = new Mat();
        Cv2.Erode(reduced, eroded, kernel, iterations: 2);

        var result = new Mat();
        Cv2.GaussianBlur(eroded, result, new Size(0, 0), blurSize);
        return result;
    }

    // -----------------------------------------------------------------
    // forward
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward</c>. <paramref name="deepSwapperMorph"/> is the length-1 morph value
    /// array (only sent as the <c>morph_value:0</c> input when <see cref="HasMorphInput"/> is
    /// true, matching Python's own conditional input-name loop). Return order deliberately
    /// mirrors Python's own quirky unpack-then-repack exactly:
    /// <code>crop_target_mask, crop_vision_frame, crop_source_mask = deep_swapper.run(None, inputs)</code>
    /// <code>return crop_vision_frame[0], crop_source_mask[0], crop_target_mask[0]</code>
    /// i.e. the three ONNX graph outputs are consumed *positionally* — output 0 is treated as
    /// the target mask, output 1 as the image, output 2 as the source mask — regardless of what
    /// each tensor's name in the graph might suggest, then returned in a different order still
    /// (image, source mask, target mask). Reproduced exactly rather than "fixed" to read output
    /// names, per PORT_CONVENTIONS.md rule 1.
    /// </summary>
    public static (float[] CropVisionFrame, float[] CropSourceMask, float[] CropTargetMask) Forward(
        InferenceSession deepSwapperSession, ReadOnlySpan<float> cropVisionFrameHwc, Size modelSize, float[]? deepSwapperMorph)
    {
        using var inFaceOrtValue = OrtValue.CreateTensorValueFromMemory(
            cropVisionFrameHwc.ToArray(), new long[] { 1, modelSize.Height, modelSize.Width, 3 });

        OrtValue? morphOrtValue = null;
        var inputs = new Dictionary<string, OrtValue>();

        try
        {
            foreach (var inputName in deepSwapperSession.InputNames)
            {
                if (inputName == "in_face:0")
                {
                    inputs[inputName] = inFaceOrtValue;
                }
                else if (inputName == "morph_value:0")
                {
                    if (deepSwapperMorph is null)
                    {
                        throw new ArgumentNullException(nameof(deepSwapperMorph), "the model declares a morph_value:0 input but none was provided.");
                    }

                    morphOrtValue = OrtValue.CreateTensorValueFromMemory(deepSwapperMorph, new long[] { deepSwapperMorph.Length });
                    inputs[inputName] = morphOrtValue;
                }
            }

            using var runOptions = new RunOptions();
            using var results = deepSwapperSession.Run(runOptions, inputs, deepSwapperSession.OutputNames);

            var outputTargetMask = results[0].GetTensorDataAsSpan<float>().ToArray();
            var outputCropVisionFrame = results[1].GetTensorDataAsSpan<float>().ToArray();
            var outputSourceMask = results[2].GetTensorDataAsSpan<float>().ToArray();

            return (outputCropVisionFrame, outputSourceMask, outputTargetMask);
        }
        finally
        {
            morphOrtValue?.Dispose();
        }
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>facefusion.processors.modules.deep_swapper.core</c>'s per-call inputs,
    /// extended per <see cref="IProcessorInputs"/>'s remarks — see each field's comment for the
    /// Python <c>state_manager</c> key/session it replaces. <see cref="Model"/> is the resolved
    /// <c>"scope/name"</c> string key into <see cref="CreateStaticModelSet"/>'s result.
    /// </summary>
    public sealed record DeepSwapperInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        string Model,
        int DeepSwapperMorph,
        InferenceSession DeepSwapperSession,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        double FaceMaskBlur,
        Padding FaceMaskPadding,
        FaceSelectorMode FaceSelectorMode,
        double FaceTrackerScore,
        FaceSelectorOrder FaceSelectorOrder,
        FaceSelectorGender? FaceSelectorGender,
        FaceSelectorRace? FaceSelectorRace,
        int? FaceSelectorAgeStart,
        int? FaceSelectorAgeEnd,
        int ReferenceFacePosition,
        double ReferenceFaceDistance,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> GetStaticFaces,
        Func<IReadOnlyList<FaceFusion.Types.Face?>, IReadOnlyList<FaceFusion.Types.Face>> RefillFaces) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/deep_swapper/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract — see
    /// <c>FaceSwapper.Processor</c> for the same pattern.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "deep_swapper";

        /// <inheritdoc />
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>
        /// Python: the <c>deep_swapper</c>-specific half of <c>pre_check</c>. Needs the chosen
        /// <paramref name="model"/> since <see cref="DeepSwapper.PreCheck"/> has no other way to
        /// learn it (no <c>state_manager</c> — rule 5).
        /// </summary>
        public bool PreCheck(string model) => DeepSwapper.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "deep_swapper.PreCheck requires a model id (no state_manager to read it from — call the string overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Same scope note as <c>FaceSwapper.Processor.PreProcess</c>
        /// (the <c>facefusion/filesystem.py</c> checks are out of this assignment's scope).
        /// </summary>
        public bool PreProcess(ProcessMode mode, ProcessorRunPaths paths)
        {
            _ = mode;
            _ = paths;
            return true;
        }

        /// <inheritdoc />
        public ProcessorOutputs ProcessFrame(IProcessorInputs inputs)
        {
            if (inputs is not DeepSwapperInputs deepSwapperInputs)
            {
                throw new ArgumentException($"expected {nameof(DeepSwapperInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return DeepSwapper.ProcessFrame(deepSwapperInputs);
        }

        /// <inheritdoc />
        public void PostProcess()
        {
        }
    }

    // -----------------------------------------------------------------
    // process_frame / swap_face orchestration
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Returns the (possibly unchanged, if no target face was
    /// found) frame and mask, same ownership convention as <c>FaceSwapper.ProcessFrame</c>.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(DeepSwapperInputs inputs)
    {
        var targetVisionFrame = FaceFusion.Core.CommonHelper.GetMiddle(inputs.TargetVisionFrames);
        var targetFaces = FaceSelector.SelectFaces(
            inputs.ReferenceVisionFrame,
            inputs.SourceVisionFrames,
            inputs.TargetVisionFrames,
            inputs.FaceSelectorMode,
            inputs.FaceTrackerScore,
            inputs.FaceSelectorOrder,
            inputs.FaceSelectorGender,
            inputs.FaceSelectorRace,
            inputs.FaceSelectorAgeStart,
            inputs.FaceSelectorAgeEnd,
            inputs.ReferenceFacePosition,
            inputs.ReferenceFaceDistance,
            inputs.GetStaticFaces,
            inputs.RefillFaces);

        var tempVisionFrame = inputs.TempVisionFrame;

        if (targetFaces.Count > 0 && targetVisionFrame is not null)
        {
            foreach (var rawTargetFace in targetFaces)
            {
                var targetFace = FaceCreator.ScaleFace(rawTargetFace, targetVisionFrame, tempVisionFrame);
                var nextTempVisionFrame = SwapFace(inputs, targetFace, tempVisionFrame);

                if (!ReferenceEquals(tempVisionFrame, inputs.TempVisionFrame))
                {
                    tempVisionFrame.Dispose();
                }

                tempVisionFrame = nextTempVisionFrame;
            }
        }

        return new ProcessorOutputs(tempVisionFrame, inputs.TempVisionMask);
    }

    /// <summary>
    /// Python: <c>swap_face</c>. Caller owns the returned <see cref="Mat"/>. Does not take
    /// ownership of <paramref name="tempVisionFrame"/>. Only <see cref="FaceMaskType.Box"/> is
    /// exercised by this assignment's parity fixtures (same scope note as
    /// <c>FaceSwapper.SwapFace</c>); <see cref="FaceMaskType.Occlusion"/>/<see cref="FaceMaskType.Area"/>/
    /// <see cref="FaceMaskType.Region"/> are implemented (reusing <c>FaceFusion.Face.FaceMasker</c>)
    /// but need their own session pool, all optional and unused when the corresponding
    /// <see cref="FaceMaskType"/> is not requested.
    /// </summary>
    public static Mat SwapFace(
        DeepSwapperInputs inputs,
        FaceFusion.Types.Face targetFace,
        Mat tempVisionFrame,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1,
        IReadOnlyList<FaceMaskArea>? faceMaskAreas = null,
        IReadOnlyDictionary<string, InferenceSession>? parserInferencePool = null,
        FaceParserModel faceParserModel = FaceParserModel.BisenetResnet18,
        IReadOnlyList<FaceMaskRegion>? faceMaskRegions = null)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[inputs.Model];
        var modelTemplate = modelOptions.Template;
        var modelSize = GetModelSize(inputs.DeepSwapperSession);

        var targetLandmark5Of68 = (float[,])targetFace.LandmarkSet.FiveOn68;
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, targetLandmark5Of68, modelTemplate, modelSize);
        using var affineMatrixDisposable = affineMatrix;
        using var cropVisionFrameDisposable = cropVisionFrame;

        using var cropVisionFrameRaw = cropVisionFrame.Clone();

        var cropMasks = new List<Mat>
        {
            FaceMasker.CreateBoxMask(cropVisionFrame, inputs.FaceMaskBlur, inputs.FaceMaskPadding),
        };

        try
        {
            if (inputs.FaceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                if (occluderInferencePool is null)
                {
                    throw new ArgumentNullException(nameof(occluderInferencePool), "FaceMaskType.Occlusion requires occluderInferencePool.");
                }

                cropMasks.Add(FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool));
            }

            var preparedCrop = PrepareCropFrame(cropVisionFrame);

            // Python: `numpy.array([numpy.interp(deep_swapper_morph, [0, 100], [0, 1])]).astype(float32)`.
            var morphValue = (float)((inputs.DeepSwapperMorph - 0.0) / 100.0);
            var deepSwapperMorph = new[] { morphValue };

            var (rawCropVisionFrame, rawCropSourceMask, rawCropTargetMask) = Forward(inputs.DeepSwapperSession, preparedCrop, modelSize, deepSwapperMorph);

            using var normalizedCropVisionFrame = NormalizeCropFrame(rawCropVisionFrame, modelSize.Height, modelSize.Width);
            using var matchedCropVisionFrame = VisionHelper.ConditionalMatchFrameColor(cropVisionFrameRaw, normalizedCropVisionFrame);

            cropMasks.Add(PrepareCropMask(rawCropSourceMask, rawCropTargetMask, modelSize));

            if (inputs.FaceMaskTypes.Contains(FaceMaskType.Area))
            {
                if (faceMaskAreas is null)
                {
                    throw new ArgumentNullException(nameof(faceMaskAreas), "FaceMaskType.Area requires faceMaskAreas.");
                }

                var landmark68 = (float[,])targetFace.LandmarkSet.SixtyEight;
                var transformedLandmark68 = FaceHelper.TransformPoints(landmark68, affineMatrix);
                cropMasks.Add(FaceMasker.CreateAreaMask(matchedCropVisionFrame, transformedLandmark68, faceMaskAreas));
            }

            if (inputs.FaceMaskTypes.Contains(FaceMaskType.Region))
            {
                if (parserInferencePool is null || faceMaskRegions is null)
                {
                    throw new ArgumentNullException(nameof(parserInferencePool), "FaceMaskType.Region requires parserInferencePool and faceMaskRegions.");
                }

                cropMasks.Add(FaceMasker.CreateRegionMask(matchedCropVisionFrame, faceMaskRegions, faceParserModel, parserInferencePool));
            }

            using var cropMask = ReduceMinimumClip01(cropMasks);
            return FaceHelper.PasteBack(tempVisionFrame, matchedCropVisionFrame, cropMask, affineMatrix);
        }
        finally
        {
            foreach (var cropMask in cropMasks)
            {
                cropMask.Dispose();
            }
        }
    }

    /// <summary>Python: <c>numpy.minimum.reduce(crop_masks).clip(0, 1)</c>. Caller owns the returned <see cref="Mat"/> (<c>CV_32FC1</c>).</summary>
    private static Mat ReduceMinimumClip01(IReadOnlyList<Mat> masks)
    {
        if (masks.Count == 0)
        {
            throw new ArgumentException("at least one face mask must be present.", nameof(masks));
        }

        var result = masks[0].Clone();
        for (var i = 1; i < masks.Count; i++)
        {
            Cv2.Min(result, masks[i], result);
        }

        Cv2.Min(result, new Scalar(1.0), result);
        Cv2.Max(result, new Scalar(0.0), result);
        return result;
    }
}
