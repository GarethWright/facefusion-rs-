using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/face_swapper/types.py</c>'s
/// <c>FaceSwapperModel = Literal[...]</c> — the thirteen face-swap model families.
/// </summary>
public enum FaceSwapperModel
{
    [WireName("blendswap_256")]
    BlendSwap256,

    [WireName("ghost_1_256")]
    Ghost1256,

    [WireName("ghost_2_256")]
    Ghost2256,

    [WireName("ghost_3_256")]
    Ghost3256,

    [WireName("hififace_unofficial_256")]
    HifiFaceUnofficial256,

    [WireName("hyperswap_1a_256")]
    Hyperswap1a256,

    [WireName("hyperswap_1b_256")]
    Hyperswap1b256,

    [WireName("hyperswap_1c_256")]
    Hyperswap1c256,

    [WireName("inswapper_128")]
    Inswapper128,

    [WireName("inswapper_128_fp16")]
    Inswapper128Fp16,

    [WireName("simswap_256")]
    Simswap256,

    [WireName("simswap_unofficial_512")]
    SimswapUnofficial512,

    [WireName("uniface_256")]
    Uniface256,
}

/// <summary>
/// Python: each model's <c>'type'</c> entry in <c>create_static_model_set</c> — the seven
/// distinct swap-maths families the thirteen <see cref="FaceSwapperModel"/> values group into.
/// Selects which branch of <c>PrepareSourceFrame</c>/<c>PrepareSourceEmbedding</c>/
/// <c>BalanceSourceEmbedding</c>/<c>NormalizeCropFrame</c> a model uses; see each method's
/// remarks for the exact Python conditionals this reproduces.
/// </summary>
public enum FaceSwapperModelKind
{
    [WireName("blendswap")]
    BlendSwap,

    [WireName("ghost")]
    Ghost,

    [WireName("hififace")]
    HifiFace,

    [WireName("hyperswap")]
    Hyperswap,

    [WireName("inswapper")]
    Inswapper,

    [WireName("simswap")]
    Simswap,

    [WireName("uniface")]
    Uniface,
}

/// <summary>
/// Python: one entry of <c>create_static_model_set</c>'s return value (a <c>ModelOptions</c>
/// dict) for the <c>face_swapper</c> module specifically. A real record rather than the generic
/// <c>IReadOnlyDictionary&lt;string, object?&gt;</c> <c>FaceFusion.Types.TypeAliases.cs</c>
/// documents for <c>ModelOptions</c> in general — matching <c>FaceRecognizer</c>/
/// <c>FaceClassifier</c>'s precedent of exposing a concrete-module's config as typed constants/
/// records rather than a loosely-typed dict, since every real caller in this codebase needs
/// every field and a dict indexer would just be re-cast at every call site. The
/// <c>__metadata__</c> sub-dict (vendor/license/year) is not reproduced — it is display-only in
/// Python (surfaced by the UI's model picker) and nothing in this port's scope reads it.
/// </summary>
public sealed record FaceSwapperModelOptions(
    IReadOnlyDictionary<string, Download> Hashes,
    IReadOnlyDictionary<string, Download> Sources,
    string? Precision,
    FaceSwapperModelKind Type,
    WarpTemplate Template,
    Size Size,
    float[] Mean,
    float[] StandardDeviation);

/// <summary>
/// Port of <c>facefusion/processors/modules/face_swapper/{core,types,choices}.py</c> — the
/// flagship processor. Swaps a source identity onto every selected target face in a frame.
///
/// <para>
/// <b>No global state; sessions and settings taken as parameters (PORT_CONVENTIONS.md rule 5).</b>
/// Every <c>state_manager.get_item(...)</c> read in the Python source becomes an explicit
/// parameter here — model choice, pixel boost, weight, mask settings all flow in through
/// <see cref="SwapFace"/>'s parameter list or through the resolved <c>InferenceSession</c>s a
/// caller passes in, matching how <c>FaceFusion.Face.FaceMasker</c>/<c>FaceRecognizer</c> were
/// ported. <c>get_inference_pool</c>/<c>pre_check</c>'s real downloading is not reproduced for
/// the same reason those two files don't reproduce it either — <c>facefusion/download.py</c> is
/// not ported in this repo. <see cref="ResolveModelsDirectory"/>/<see cref="BuildDownloadUrl"/>
/// below reproduce only the pieces <c>FaceDetector.CreateStaticModelSet</c> already established
/// as this port's precedent for that gap: resolve URLs directly against
/// <c>FaceFusion.Types.Choices.DownloadProviderSet</c>'s github entry (the first provider
/// Python's own <c>resolve_download_url</c> tries), and resolve local paths by walking up from
/// the test/build output directory to the repository root.
/// </para>
///
/// <para>
/// <b>Model families covered by this port vs. parity-tested end to end.</b> Every one of the
/// thirteen models' catalog entries (<see cref="FaceSwapperModelCatalog"/>) and every one of the
/// seven <see cref="FaceSwapperModelKind"/> branches in <see cref="PrepareSourceFrame"/>/
/// <see cref="PrepareSourceEmbedding"/>/<see cref="BalanceSourceEmbedding"/>/
/// <see cref="NormalizeCropFrame"/> is ported and unit-tested against hand-computed values. Real
/// ONNX-Runtime parity (<c>FaceSwapperParityTests</c>, ground truth from
/// <c>tools/parity/dump_face_swapper.py</c>) is run against two families that between them
/// exercise every source-input code path but the embedding-converter-free ones already covered
/// by <c>hyperswap</c>'s straightforward <c>embedding_norm</c> passthrough:
/// <c>inswapper_128</c> (the <c>get_static_model_initializer</c> dot-product path, no
/// renormalize) and <c>ghost_1_256</c> (a real <c>embedding_converter</c> ONNX pass, plus the
/// renormalize branch of <see cref="NormalizeCropFrame"/>). <c>blendswap</c>/<c>uniface</c>'s
/// frame-based source path (<see cref="PrepareSourceFrame"/>) is unit-tested against a
/// hand-verified NumPy computation (no ONNX model needed — it is pure preprocessing) rather than
/// a full model run, since neither family's <c>.onnx</c> was fetched for this fixture set (see
/// the assignment's "keep fixtures lean" instruction). See the port report for the full
/// breakdown.
/// </para>
///
/// <para>
/// <b>Mat / dtype conventions.</b> <c>VisionFrame</c>s are <see cref="Mat"/>, <c>CV_8UC3</c> BGR,
/// caller-owned on every return, matching <c>FaceHelper</c>. <see cref="PrepareCropFrame"/>
/// narrows to <c>float32</c> unconditionally (it builds a model *input*); <see cref="NormalizeCropFrame"/>
/// deliberately does **not** narrow uniformly — Python's <c>normalize_crop_frame</c> promotes to
/// <c>float64</c> for the <c>ghost</c>/<c>hififace</c>/<c>hyperswap</c>/<c>uniface</c> "renormalize"
/// branch (multiplying a <c>float32</c> ndarray by a bare Python <c>list</c> of the model's
/// mean/std upcasts it — verified empirically, see the method's remarks) but stays <c>float32</c>
/// for <c>blendswap</c>/<c>inswapper</c>/<c>simswap</c> (no list arithmetic on that path); this is
/// reproduced exactly via the returned <see cref="Mat"/>'s type (<c>CV_64FC3</c> vs.
/// <c>CV_32FC3</c>) rather than picking one dtype for both, per PORT_CONVENTIONS.md rule 6.
/// </para>
/// </summary>
public static class FaceSwapper
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>face_swapper_models = list(get_args(FaceSwapperModel))</c>.</summary>
    public static readonly IReadOnlyList<FaceSwapperModel> FaceSwapperModels = Enum.GetValues<FaceSwapperModel>();

    /// <summary>Python: <c>face_swapper_weight_range = create_float_range(0.0, 1.0, 0.05)</c>.</summary>
    public static readonly IReadOnlyList<double> FaceSwapperWeightRange =
        FaceFusion.Core.CommonHelper.CreateFloatRange(0.0, 1.0, 0.05);

    /// <summary>
    /// Python: <c>face_swapper_set</c> — the <c>--face-swapper-pixel-boost</c> choices per
    /// model, always the model's own native size first followed by successively larger squares
    /// (so <c>pixel_boost_total == 1</c> — the identity case for <see cref="PixelBoost"/> — is
    /// always a valid, and the default, choice).
    /// </summary>
    public static readonly IReadOnlyDictionary<FaceSwapperModel, IReadOnlyList<string>> FaceSwapperPixelBoostChoices =
        new Dictionary<FaceSwapperModel, IReadOnlyList<string>>
        {
            [FaceSwapperModel.BlendSwap256] = new[] { "256x256", "384x384", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Ghost1256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Ghost2256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Ghost3256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.HifiFaceUnofficial256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Hyperswap1a256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Hyperswap1b256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Hyperswap1c256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Inswapper128] = new[] { "128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Inswapper128Fp16] = new[] { "128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Simswap256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.SimswapUnofficial512] = new[] { "512x512", "768x768", "1024x1024" },
            [FaceSwapperModel.Uniface256] = new[] { "256x256", "512x512", "768x768", "1024x1024" },
        };

    // -----------------------------------------------------------------
    // create_static_model_set
    // -----------------------------------------------------------------

    private static readonly object ModelCatalogLock = new();
    private static IReadOnlyDictionary<FaceSwapperModel, FaceSwapperModelOptions>? _cachedModelCatalog;

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). <paramref name="downloadScope"/>
    /// is accepted for signature parity with Python — every family's entry is identical
    /// regardless of scope, same as <c>FaceDetector.CreateStaticModelSet</c>.
    /// </summary>
    public static IReadOnlyDictionary<FaceSwapperModel, FaceSwapperModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        lock (ModelCatalogLock)
        {
            return _cachedModelCatalog ??= BuildModelCatalog();
        }
    }

    private static IReadOnlyDictionary<FaceSwapperModel, FaceSwapperModelOptions> BuildModelCatalog()
    {
        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];

        (IReadOnlyDictionary<string, Download> Hashes, IReadOnlyDictionary<string, Download> Sources) Component(
            string modelsBaseName, string fileName, string? embeddingConverterBaseName = null, string? embeddingConverterFileName = null)
        {
            var hashes = new Dictionary<string, Download>
            {
                ["face_swapper"] = new Download(
                    BuildDownloadUrl(githubProvider, modelsBaseName, fileName + ".hash"),
                    Path.Combine(modelsDirectory, fileName + ".hash")),
            };
            var sources = new Dictionary<string, Download>
            {
                ["face_swapper"] = new Download(
                    BuildDownloadUrl(githubProvider, modelsBaseName, fileName + ".onnx"),
                    Path.Combine(modelsDirectory, fileName + ".onnx")),
            };

            if (embeddingConverterBaseName is not null && embeddingConverterFileName is not null)
            {
                hashes["embedding_converter"] = new Download(
                    BuildDownloadUrl(githubProvider, embeddingConverterBaseName, embeddingConverterFileName + ".hash"),
                    Path.Combine(modelsDirectory, embeddingConverterFileName + ".hash"));
                sources["embedding_converter"] = new Download(
                    BuildDownloadUrl(githubProvider, embeddingConverterBaseName, embeddingConverterFileName + ".onnx"),
                    Path.Combine(modelsDirectory, embeddingConverterFileName + ".onnx"));
            }

            return (hashes, sources);
        }

        var catalog = new Dictionary<FaceSwapperModel, FaceSwapperModelOptions>();

        void Add(FaceSwapperModel model, string modelsBaseName, string fileName, FaceSwapperModelKind type, WarpTemplate template, int size,
            float[] mean, float[] standardDeviation, string? precision = null, string? embeddingConverterBaseName = null, string? embeddingConverterFileName = null)
        {
            var (hashes, sources) = Component(modelsBaseName, fileName, embeddingConverterBaseName, embeddingConverterFileName);
            catalog[model] = new FaceSwapperModelOptions(hashes, sources, precision, type, template, new Size(size, size), mean, standardDeviation);
        }

        const string crossfaceBaseName = "models-3.4.0";

        Add(FaceSwapperModel.BlendSwap256, "models-3.0.0", "blendswap_256", FaceSwapperModelKind.BlendSwap, WarpTemplate.Ffhq512, 256, new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f });
        Add(FaceSwapperModel.Ghost1256, "models-3.0.0", "ghost_1_256", FaceSwapperModelKind.Ghost, WarpTemplate.Arcface112V1, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, embeddingConverterBaseName: crossfaceBaseName, embeddingConverterFileName: "crossface_ghost");
        Add(FaceSwapperModel.Ghost2256, "models-3.0.0", "ghost_2_256", FaceSwapperModelKind.Ghost, WarpTemplate.Arcface112V1, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, embeddingConverterBaseName: crossfaceBaseName, embeddingConverterFileName: "crossface_ghost");
        Add(FaceSwapperModel.Ghost3256, "models-3.0.0", "ghost_3_256", FaceSwapperModelKind.Ghost, WarpTemplate.Arcface112V1, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, embeddingConverterBaseName: crossfaceBaseName, embeddingConverterFileName: "crossface_ghost");
        Add(FaceSwapperModel.HifiFaceUnofficial256, "models-3.1.0", "hififace_unofficial_256", FaceSwapperModelKind.HifiFace, WarpTemplate.Mtcnn512, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, embeddingConverterBaseName: crossfaceBaseName, embeddingConverterFileName: "crossface_hififace");
        Add(FaceSwapperModel.Hyperswap1a256, "models-3.3.0", "hyperswap_1a_256", FaceSwapperModelKind.Hyperswap, WarpTemplate.Arcface128, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, precision: "fp16");
        Add(FaceSwapperModel.Hyperswap1b256, "models-3.3.0", "hyperswap_1b_256", FaceSwapperModelKind.Hyperswap, WarpTemplate.Arcface128, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, precision: "fp16");
        Add(FaceSwapperModel.Hyperswap1c256, "models-3.3.0", "hyperswap_1c_256", FaceSwapperModelKind.Hyperswap, WarpTemplate.Arcface128, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }, precision: "fp16");
        Add(FaceSwapperModel.Inswapper128, "models-3.0.0", "inswapper_128", FaceSwapperModelKind.Inswapper, WarpTemplate.Arcface128, 128, new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f });
        Add(FaceSwapperModel.Inswapper128Fp16, "models-3.0.0", "inswapper_128_fp16", FaceSwapperModelKind.Inswapper, WarpTemplate.Arcface128, 128, new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f }, precision: "fp16");
        Add(FaceSwapperModel.Simswap256, "models-3.0.0", "simswap_256", FaceSwapperModelKind.Simswap, WarpTemplate.Arcface112V1, 256, new[] { 0.485f, 0.456f, 0.406f }, new[] { 0.229f, 0.224f, 0.225f }, embeddingConverterBaseName: crossfaceBaseName, embeddingConverterFileName: "crossface_simswap");
        Add(FaceSwapperModel.SimswapUnofficial512, "models-3.0.0", "simswap_unofficial_512", FaceSwapperModelKind.Simswap, WarpTemplate.Arcface112V1, 512, new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f }, embeddingConverterBaseName: crossfaceBaseName, embeddingConverterFileName: "crossface_simswap");
        Add(FaceSwapperModel.Uniface256, "models-3.0.0", "uniface_256", FaceSwapperModelKind.Uniface, WarpTemplate.Ffhq512, 256, new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f });

        return catalog;
    }

    /// <summary>
    /// Same reasoning and implementation as <c>FaceDetector.ResolveModelsDirectory</c> — walks
    /// up from the running assembly's base directory to the repository root (marked by
    /// <c>FaceFusion.sln</c>) rather than resolving relative to the build output directory,
    /// which would not reach the real <c>.assets/models</c> from a test assembly's bin folder.
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
    /// Python: the <c>face_swapper</c>-specific half of <c>pre_check</c> (the common-module
    /// half — <c>content_analyser</c>/<c>face_classifier</c>/.../<c>face_recognizer</c> — is the
    /// caller's responsibility; see <see cref="IProcessor.GetCommonModules"/>'s remarks).
    /// Verifies every hash/source file this <paramref name="model"/> needs is already present
    /// locally; unlike Python, does not download a missing one (see class remarks).
    /// </summary>
    public static bool PreCheck(FaceSwapperModel model)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[model];

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
    // prepare_source_frame — blendswap / uniface only
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_source_frame</c>. Only ever called by <c>forward_swap_face</c> for
    /// <see cref="FaceSwapperModelKind.BlendSwap"/>/<see cref="FaceSwapperModelKind.Uniface"/> —
    /// every other kind builds its <c>'source'</c> model input from an embedding instead (see
    /// <see cref="PrepareSourceEmbedding"/>). The warp template/size for each branch
    /// (<c>arcface_112_v2</c> @ 112x112, <c>ffhq_512</c> @ 256x256) are Python literals
    /// independent of the model's own catalog <see cref="FaceSwapperModelOptions.Size"/> — both
    /// 256x256 models still feed a 112/256-sized identity crop into this branch, reproduced
    /// exactly rather than "corrected" to use <c>modelOptions.Size</c>.
    /// </summary>
    public static float[] PrepareSourceFrame(FaceSwapperModelKind modelType, float[,] sourceFaceLandmark5Of68, Mat sourceVisionFrame)
    {
        Mat frame;
        Mat? affineMatrix = null;

        switch (modelType)
        {
            case FaceSwapperModelKind.BlendSwap:
                (frame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(sourceVisionFrame, sourceFaceLandmark5Of68, WarpTemplate.Arcface112V2, new Size(112, 112));
                break;
            case FaceSwapperModelKind.Uniface:
                (frame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(sourceVisionFrame, sourceFaceLandmark5Of68, WarpTemplate.Ffhq512, new Size(256, 256));
                break;
            default:
                frame = sourceVisionFrame;
                break;
        }

        try
        {
            return BuildRgbChwFloat32(frame);
        }
        finally
        {
            if (affineMatrix is not null)
            {
                frame.Dispose();
                affineMatrix.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>crop_vision_frame[:, :, ::-1] / 255.0</c>, transpose(2,0,1), expand_dims,
    /// <c>.astype(float32)</c> — the shared preprocessing tail of both
    /// <see cref="PrepareSourceFrame"/> and (with a different divisor formula)
    /// <see cref="PrepareCropFrame"/>. Computed in <see cref="double"/> per element then narrowed
    /// to <see cref="float"/> only at assignment, matching numpy's true-division promotion
    /// (<c>uint8 array / python float</c> -&gt; float64) exactly, per PORT_CONVENTIONS.md rule 6
    /// (verified empirically against <c>tools/parity/dump_face_swapper.py</c>'s
    /// <c>target_input</c>/<c>source_input</c> fixtures — see <c>FaceSwapperParityTests</c>).
    /// </summary>
    private static float[] BuildRgbChwFloat32(Mat frame)
    {
        var height = frame.Rows;
        var width = frame.Cols;
        var plane = height * width;
        var chw = new float[3 * plane];

        frame.GetArray(out Vec3b[] pixels);

        for (var index = 0; index < plane; index++)
        {
            var pixel = pixels[index];
            chw[index] = (float)(pixel.Item2 / 255.0); // R -> channel 0
            chw[plane + index] = (float)(pixel.Item1 / 255.0); // G -> channel 1
            chw[(2 * plane) + index] = (float)(pixel.Item0 / 255.0); // B -> channel 2
        }

        return chw;
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame — the 'target' model input, every model kind
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_crop_frame</c>. <paramref name="mean"/>/<paramref name="standardDeviation"/>
    /// are <see cref="FaceSwapperModelOptions.Mean"/>/<see cref="FaceSwapperModelOptions.StandardDeviation"/>.
    /// Always narrows to <c>float32</c> at the end regardless of model kind (this builds a model
    /// *input*, unlike <see cref="NormalizeCropFrame"/> below, which builds a model *output* and
    /// does not always narrow — see that method's remarks).
    /// </summary>
    public static float[] PrepareCropFrame(Mat cropVisionFrame, float[] mean, float[] standardDeviation)
    {
        var height = cropVisionFrame.Rows;
        var width = cropVisionFrame.Cols;
        var plane = height * width;
        var chw = new float[3 * plane];

        cropVisionFrame.GetArray(out Vec3b[] pixels);

        for (var index = 0; index < plane; index++)
        {
            var pixel = pixels[index];

            // Python: `crop_vision_frame[:, :, ::-1] / 255.0` then `(x - mean) / std`, both in
            // float64 (uint8/float and float64-array arithmetic), narrowed to float32 only by
            // the final `.astype(float32)` after the transpose/expand_dims.
            var r = (pixel.Item2 / 255.0 - mean[0]) / standardDeviation[0];
            var g = (pixel.Item1 / 255.0 - mean[1]) / standardDeviation[1];
            var b = (pixel.Item0 / 255.0 - mean[2]) / standardDeviation[2];

            chw[index] = (float)r;
            chw[plane + index] = (float)g;
            chw[(2 * plane) + index] = (float)b;
        }

        return chw;
    }

    // -----------------------------------------------------------------
    // prepare_source_embedding / convert_source_embedding / forward_convert_embedding
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_source_embedding</c>. Only called for
    /// <see cref="FaceSwapperModelKind.Ghost"/>/<see cref="FaceSwapperModelKind.HifiFace"/>/
    /// <see cref="FaceSwapperModelKind.Hyperswap"/>/<see cref="FaceSwapperModelKind.Inswapper"/>/
    /// <see cref="FaceSwapperModelKind.Simswap"/> — <c>BlendSwap</c>/<c>Uniface</c> use
    /// <see cref="PrepareSourceFrame"/> instead (see <c>forward_swap_face</c>'s dispatch, this
    /// project's <see cref="Processor.ProcessFrame"/>). Returns a flat length-512 array (Python:
    /// shape <c>(1, 512)</c> — the leading batch dim is implicit here, added back at the ORT
    /// call site).
    /// </summary>
    /// <param name="embeddingConverterSession">
    /// Required (and only used) for <see cref="FaceSwapperModelKind.Ghost"/>/<c>HifiFace</c>/
    /// <c>Simswap</c> — the models whose catalog entry has an <c>embedding_converter</c> source.
    /// </param>
    /// <param name="inswapperModelInitializer">
    /// Required (and only used) for <see cref="FaceSwapperModelKind.Inswapper"/> — Python:
    /// <c>get_static_model_initializer(model_path)</c> on the <c>face_swapper</c> model's own
    /// <c>.onnx</c> file (<see cref="ModelHelper.GetStaticModelInitializer"/>, already ported).
    /// </param>
    public static float[] PrepareSourceEmbedding(
        FaceSwapperModelKind modelType,
        float[] sourceEmbedding,
        float[] sourceEmbeddingNorm,
        InferenceSession? embeddingConverterSession,
        OnnxTensor? inswapperModelInitializer)
    {
        switch (modelType)
        {
            case FaceSwapperModelKind.Ghost:
            {
                if (embeddingConverterSession is null)
                {
                    throw new ArgumentNullException(nameof(embeddingConverterSession), "ghost requires an embedding_converter session.");
                }

                var (embedding, _) = ConvertSourceEmbedding(embeddingConverterSession, sourceEmbedding);
                return embedding;
            }

            case FaceSwapperModelKind.Hyperswap:
                // Python: `source_face.embedding_norm.reshape((1, -1))` — no model pass, no
                // renormalisation, the already-L2-normalised embedding used as-is.
                return sourceEmbeddingNorm;

            case FaceSwapperModelKind.Inswapper:
            {
                if (inswapperModelInitializer is null)
                {
                    throw new ArgumentNullException(nameof(inswapperModelInitializer), "inswapper requires its static model initializer.");
                }

                // Python: `numpy.dot(source_embedding.reshape((1, -1)), model_initializer) /
                // numpy.linalg.norm(source_embedding)` — a (1, 512) row-vector times a (512,
                // 512) matrix, i.e. result[j] = sum_i embedding[i] * initializer[i, j] (the
                // transpose of FaceFusion.Tensors.NumPy.Dot's matrix*vector overload, which
                // multiplies the other way round — hence the small local loop rather than
                // reusing that helper).
                var initializer = inswapperModelInitializer.AsFloats();
                var rows = checked((int)inswapperModelInitializer.Shape[0]);
                var cols = checked((int)inswapperModelInitializer.Shape[1]);

                if (rows != sourceEmbedding.Length)
                {
                    throw new ArgumentException($"model_initializer has {rows} rows, expected {sourceEmbedding.Length} (embedding length).");
                }

                var norm = NumPy.LinalgNorm(sourceEmbedding);
                var result = new float[cols];
                for (var col = 0; col < cols; col++)
                {
                    double sum = 0;
                    for (var row = 0; row < rows; row++)
                    {
                        sum += (double)sourceEmbedding[row] * initializer[(row * cols) + col];
                    }

                    result[col] = (float)(sum / norm);
                }

                return result;
            }

            default:
            {
                // hififace / simswap: the "else" branch of prepare_source_embedding.
                if (embeddingConverterSession is null)
                {
                    throw new ArgumentNullException(nameof(embeddingConverterSession), $"{modelType} requires an embedding_converter session.");
                }

                var (_, embeddingNorm) = ConvertSourceEmbedding(embeddingConverterSession, sourceEmbedding);
                return embeddingNorm;
            }
        }
    }

    /// <summary>
    /// Python: <c>convert_source_embedding</c>. Returns the raw converted embedding and its
    /// L2-normalised counterpart, same shape as <c>FaceRecognizer.CalculateFaceEmbedding</c>'s
    /// return.
    /// </summary>
    public static (float[] Embedding, float[] EmbeddingNorm) ConvertSourceEmbedding(InferenceSession embeddingConverterSession, float[] sourceEmbedding)
    {
        var embedding = ForwardConvertEmbedding(embeddingConverterSession, sourceEmbedding);
        var norm = NumPy.LinalgNorm(embedding);
        var embeddingNorm = new float[embedding.Length];
        for (var i = 0; i < embedding.Length; i++)
        {
            embeddingNorm[i] = embedding[i] / norm;
        }

        return (embedding, embeddingNorm);
    }

    /// <summary>Python: <c>forward_convert_embedding</c>. <paramref name="faceEmbedding"/> is length 512 (batch dim implicit).</summary>
    public static float[] ForwardConvertEmbedding(InferenceSession embeddingConverterSession, float[] faceEmbedding)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(faceEmbedding, new long[] { 1, faceEmbedding.Length });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = embeddingConverterSession.Run(runOptions, inputs, embeddingConverterSession.OutputNames);

        // Python: `.ravel()` — already flat here.
        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    // -----------------------------------------------------------------
    // balance_source_embedding
    // -----------------------------------------------------------------

    private static readonly float[] WeightInterpXp = { 0f, 1f };
    private static readonly float[] WeightInterpFp = { 0.35f, -0.35f };

    private static readonly HashSet<FaceSwapperModelKind> RenormalizeTargetEmbeddingKinds = new()
    {
        FaceSwapperModelKind.HifiFace, FaceSwapperModelKind.Hyperswap, FaceSwapperModelKind.Inswapper, FaceSwapperModelKind.Simswap,
    };

    /// <summary>
    /// Python: <c>balance_source_embedding</c>. Blends the source identity toward the target
    /// face's own embedding by <paramref name="faceSwapperWeight"/> (0 = pure source identity,
    /// 1 = pure target — Python's <c>interp(weight, [0,1], [0.35,-0.35])</c> inverts the sense,
    /// reproduced exactly rather than "fixed"). <paramref name="targetEmbedding"/> is the target
    /// face's *raw* embedding (Python: <c>target_face.embedding</c>, not <c>embedding_norm</c>)
    /// — note <see cref="FaceSwapperModelKind.Ghost"/> is deliberately **not** in
    /// <see cref="RenormalizeTargetEmbeddingKinds"/> (it uses the raw target embedding unnormalised,
    /// matching Python's conditional exactly, even though its own source embedding just went
    /// through <see cref="ConvertSourceEmbedding"/>'s normalisation).
    /// </summary>
    public static float[] BalanceSourceEmbedding(FaceSwapperModelKind modelType, float[] sourceEmbedding, float[] targetEmbedding, double faceSwapperWeight)
    {
        var weight = NumPy.Interp((float)faceSwapperWeight, WeightInterpXp, WeightInterpFp);

        float[] balancedTarget;
        if (RenormalizeTargetEmbeddingKinds.Contains(modelType))
        {
            var norm = NumPy.LinalgNorm(targetEmbedding);
            balancedTarget = new float[targetEmbedding.Length];
            for (var i = 0; i < targetEmbedding.Length; i++)
            {
                balancedTarget[i] = targetEmbedding[i] / norm;
            }
        }
        else
        {
            balancedTarget = targetEmbedding;
        }

        var result = new float[sourceEmbedding.Length];
        for (var i = 0; i < sourceEmbedding.Length; i++)
        {
            result[i] = (sourceEmbedding[i] * (1f - weight)) + (balancedTarget[i] * weight);
        }

        return result;
    }

    // -----------------------------------------------------------------
    // forward_swap_face
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward_swap_face</c>'s ONNX call. <paramref name="sourceInput"/> is either a
    /// length-512 embedding (batch dim implicit -&gt; shape <c>(1, 512)</c>) or a flat CHW frame
    /// from <see cref="PrepareSourceFrame"/> (shape <c>(1, 3, H, W)</c>, <paramref name="sourceInputSize"/>
    /// non-null); <paramref name="targetInput"/> is always a flat CHW frame from
    /// <see cref="PrepareCropFrame"/> (shape <c>(1, 3, modelSize, modelSize)</c>). Input names are
    /// resolved dynamically against <paramref name="faceSwapperSession"/>'s own <c>"source"</c>/
    /// <c>"target"</c> input metadata, matching Python's <c>for face_swapper_input in
    /// face_swapper.get_inputs()</c> loop rather than assuming a fixed input order (verified:
    /// real models order these differently — see the port report). Every output is requested
    /// (Python passes <c>output_names=None</c> to <c>session.run</c>, which computes all of
    /// them) and only the first is used, same as Python's <c>[0][0]</c>.
    /// </summary>
    public static float[] ForwardSwapFace(
        InferenceSession faceSwapperSession, float[] sourceInput, Size? sourceInputSize, float[] targetInput, Size targetInputSize)
    {
        using var sourceOrtValue = sourceInputSize is { } size
            ? OrtValue.CreateTensorValueFromMemory(sourceInput, new long[] { 1, 3, size.Height, size.Width })
            : OrtValue.CreateTensorValueFromMemory(sourceInput, new long[] { 1, sourceInput.Length });
        using var targetOrtValue = OrtValue.CreateTensorValueFromMemory(targetInput, new long[] { 1, 3, targetInputSize.Height, targetInputSize.Width });

        var inputs = new Dictionary<string, OrtValue>();
        foreach (var inputName in faceSwapperSession.InputNames)
        {
            if (inputName == "source")
            {
                inputs[inputName] = sourceOrtValue;
            }
            else if (inputName == "target")
            {
                inputs[inputName] = targetOrtValue;
            }
        }

        using var runOptions = new RunOptions();
        using var results = faceSwapperSession.Run(runOptions, inputs, faceSwapperSession.OutputNames);

        // Python: `face_swapper.run(None, inputs)[0][0]` — first output tensor, batch index 0,
        // leaving a flat (3, H, W) span.
        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    // -----------------------------------------------------------------
    // normalize_crop_frame
    // -----------------------------------------------------------------

    private static readonly HashSet<FaceSwapperModelKind> RenormalizeCropFrameKinds = new()
    {
        FaceSwapperModelKind.Ghost, FaceSwapperModelKind.HifiFace, FaceSwapperModelKind.Hyperswap, FaceSwapperModelKind.Uniface,
    };

    /// <summary>
    /// Python: <c>normalize_crop_frame</c> — turns a raw <c>(3, H, W)</c> model output back into
    /// an 8-bit-range BGR <see cref="Mat"/> (still float; the uint8 cast only happens inside
    /// <see cref="PasteBackFloatCrop"/>, matching Python leaving <c>crop_vision_frame</c> float
    /// all the way to <c>paste_back</c>'s own final <c>.astype</c> — see that method's remarks).
    ///
    /// <para>
    /// <b>Dtype, not narrowed uniformly (deliberate, verified empirically).</b> For
    /// <see cref="FaceSwapperModelKind.Ghost"/>/<c>HifiFace</c>/<c>Hyperswap</c>/<c>Uniface</c>
    /// (<see cref="RenormalizeCropFrameKinds"/>), Python's <c>crop_vision_frame * model_standard_deviation
    /// + model_mean</c> multiplies a <c>float32</c> ndarray by a bare Python <c>list</c> of
    /// floats, which numpy promotes to <c>float64</c> (list -&gt; float64 array, float32 *
    /// float64 -&gt; float64) — every following op stays float64, so the method returns a
    /// <c>CV_64FC3</c> <see cref="Mat"/> for these four kinds. For
    /// <see cref="FaceSwapperModelKind.BlendSwap"/>/<c>Inswapper</c>/<c>Simswap</c> there is no
    /// list arithmetic on this path at all (no renormalise step), so the array never leaves
    /// <c>float32</c> — the method returns <c>CV_32FC3</c> for those three. Reproduced via the
    /// returned <see cref="Mat"/>'s <see cref="MatType"/> rather than always returning
    /// <c>CV_64FC3</c>, per PORT_CONVENTIONS.md rule 6.
    /// </para>
    /// </summary>
    public static Mat NormalizeCropFrame(ReadOnlySpan<float> modelOutputChw, int height, int width, FaceSwapperModelKind modelType, float[] mean, float[] standardDeviation)
    {
        var plane = height * width;

        if (modelOutputChw.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputChw has {modelOutputChw.Length} elements, expected {3 * plane} for a {width}x{height} CHW frame.", nameof(modelOutputChw));
        }

        if (RenormalizeCropFrameKinds.Contains(modelType))
        {
            var result = new Mat(height, width, MatType.CV_64FC3);
            var data = new Vec3d[plane];

            for (var index = 0; index < plane; index++)
            {
                // transpose(1,2,0): channel c, spatial index -> HWC. Then `* std + mean` in
                // float64, `.clip(0,1)`, then `[:, :, ::-1] * 255` (RGB -> BGR).
                var r = Clamp01(((double)modelOutputChw[index] * standardDeviation[0]) + mean[0]);
                var g = Clamp01(((double)modelOutputChw[plane + index] * standardDeviation[1]) + mean[1]);
                var b = Clamp01(((double)modelOutputChw[(2 * plane) + index] * standardDeviation[2]) + mean[2]);

                data[index] = new Vec3d(b * 255.0, g * 255.0, r * 255.0); // BGR
            }

            result.SetArray(data);
            return result;
        }
        else
        {
            var result = new Mat(height, width, MatType.CV_32FC3);
            var data = new Vec3f[plane];

            for (var index = 0; index < plane; index++)
            {
                var r = ClampFloat01(modelOutputChw[index]);
                var g = ClampFloat01(modelOutputChw[plane + index]);
                var b = ClampFloat01(modelOutputChw[(2 * plane) + index]);

                data[index] = new Vec3f(b * 255f, g * 255f, r * 255f); // BGR
            }

            result.SetArray(data);
            return result;
        }
    }

    private static double Clamp01(double value) => value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);

    private static float ClampFloat01(float value) => value < 0f ? 0f : (value > 1f ? 1f : value);

    // -----------------------------------------------------------------
    // paste_back — a float (not necessarily uint8) crop, see class remarks
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>paste_back</c>, called from <c>swap_face</c> with a *float* <c>crop_vision_frame</c>
    /// (the output of <see cref="PixelBoost.ExplodePixelBoost"/> over <see cref="NormalizeCropFrame"/>
    /// results — never cast back to <c>uint8</c> first). <c>FaceFusion.Face.FaceHelper.PasteBack</c>
    /// is not reused here: it deliberately requires a <c>CV_8UC3</c> crop (see its own class
    /// remarks — every *other* real caller in the face pipeline passes one), which this call site
    /// never satisfies, and <c>FaceHelper.cs</c> is out of this module's assignment to modify.
    /// This method reuses <see cref="FaceHelper.CalculatePasteArea"/> (which only reads
    /// <paramref name="cropVisionFrame"/>'s <c>Cols</c>/<c>Rows</c>, not its pixel dtype, so it
    /// is safe to share) for the geometry and reproduces the rest of <c>paste_back</c> generically
    /// for any float crop dtype.
    /// </summary>
    public static Mat PasteBackFloatCrop(Mat tempVisionFrame, Mat cropVisionFrame, Mat cropVisionMask, Mat affineMatrix)
    {
        if (tempVisionFrame.Type() != MatType.CV_8UC3)
        {
            throw new ArgumentException("PasteBackFloatCrop requires a CV_8UC3 tempVisionFrame.", nameof(tempVisionFrame));
        }

        if (cropVisionFrame.Type() != MatType.CV_32FC3 && cropVisionFrame.Type() != MatType.CV_64FC3)
        {
            throw new ArgumentException("PasteBackFloatCrop requires a CV_32FC3 or CV_64FC3 cropVisionFrame.", nameof(cropVisionFrame));
        }

        if (cropVisionMask.Type() != MatType.CV_32FC1)
        {
            throw new ArgumentException("PasteBackFloatCrop requires a CV_32FC1 mask.", nameof(cropVisionMask));
        }

        var (pasteBoundingBox, pasteMatrix) = FaceHelper.CalculatePasteArea(tempVisionFrame, cropVisionFrame, affineMatrix);
        using var _ = pasteMatrix;

        var x1 = pasteBoundingBox[0];
        var y1 = pasteBoundingBox[1];
        var x2 = pasteBoundingBox[2];
        var y2 = pasteBoundingBox[3];
        var pasteWidth = x2 - x1;
        var pasteHeight = y2 - y1;
        var pasteSize = new Size(pasteWidth, pasteHeight);

        using var inverseVisionMaskRaw = new Mat();
        Cv2.WarpAffine(cropVisionMask, inverseVisionMaskRaw, pasteMatrix, pasteSize);

        using var inverseVisionFrame = new Mat();
        Cv2.WarpAffine(cropVisionFrame, inverseVisionFrame, pasteMatrix, pasteSize, InterpolationFlags.Linear, BorderTypes.Replicate);

        var resultVisionFrame = tempVisionFrame.Clone();
        var isDouble = cropVisionFrame.Type() == MatType.CV_64FC3;

        for (var row = 0; row < pasteHeight; row++)
        {
            for (var col = 0; col < pasteWidth; col++)
            {
                var maskValue = NumPy.Clip(inverseVisionMaskRaw.At<float>(row, col), 0f, 1f);
                var oneMinusMask = 1.0 - maskValue;

                var destRow = y1 + row;
                var destCol = x1 + col;
                var original = resultVisionFrame.At<Vec3b>(destRow, destCol);

                Vec3b blended;
                if (isDouble)
                {
                    var warped = inverseVisionFrame.At<Vec3d>(row, col);
                    blended = new Vec3b
                    {
                        Item0 = BlendChannel(original.Item0, warped.Item0, oneMinusMask, maskValue),
                        Item1 = BlendChannel(original.Item1, warped.Item1, oneMinusMask, maskValue),
                        Item2 = BlendChannel(original.Item2, warped.Item2, oneMinusMask, maskValue),
                    };
                }
                else
                {
                    var warped = inverseVisionFrame.At<Vec3f>(row, col);
                    blended = new Vec3b
                    {
                        Item0 = BlendChannel(original.Item0, warped.Item0, oneMinusMask, maskValue),
                        Item1 = BlendChannel(original.Item1, warped.Item1, oneMinusMask, maskValue),
                        Item2 = BlendChannel(original.Item2, warped.Item2, oneMinusMask, maskValue),
                    };
                }

                resultVisionFrame.Set(destRow, destCol, blended);
            }
        }

        return resultVisionFrame;
    }

    /// <summary>
    /// Same blend as <c>FaceHelper.PasteBack</c>'s private <c>BlendChannel</c>:
    /// <c>original * (1 - mask) + warped * mask</c> in double precision, then truncated toward
    /// zero (numpy's <c>.astype(uint8)</c> semantics, not OpenCV's round-to-nearest
    /// <c>saturate_cast</c>).
    /// </summary>
    private static byte BlendChannel(byte original, double warped, double oneMinusMask, double mask)
    {
        var value = ((double)original * oneMinusMask) + (warped * mask);

        if (value <= 0)
        {
            return 0;
        }

        if (value >= 255)
        {
            return 255;
        }

        return (byte)value;
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: the <c>facefusion.processors.modules.face_swapper.core</c> module's per-call
    /// inputs, extended per <see cref="IProcessorInputs"/>'s remarks to also carry every
    /// setting/session Python would have pulled from <c>state_manager</c>/its own
    /// <c>get_inference_pool()</c> for this call — see each field's own comment for the Python
    /// <c>state_manager</c> key or session it replaces.
    /// </summary>
    public sealed record FaceSwapperInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        FaceSwapperModel Model,
        string PixelBoostResolution,
        double Weight,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        double FaceMaskBlur,
        Padding FaceMaskPadding,
        InferenceSession FaceSwapperSession,
        InferenceSession? EmbeddingConverterSession,
        OnnxTensor? InswapperModelInitializer,
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
    /// Python: <c>facefusion/processors/modules/face_swapper/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract. A thin orchestration layer
    /// over the static numeric methods above plus <c>FaceFusion.Face</c>'s existing
    /// <c>FaceSelector</c>/<c>FaceCreator</c> — mirrors <c>process_frame</c>/<c>swap_face</c>
    /// exactly (see <see cref="ProcessFrame"/>/<see cref="SwapFace"/>).
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "face_swapper";

        /// <summary>
        /// Python: <c>get_common_modules()</c> — see <see cref="IProcessor.GetCommonModules"/>'s
        /// remarks for why these are names rather than callable references.
        /// </summary>
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>
        /// Python: the <c>face_swapper</c>-specific half of <c>pre_check</c>. The common-module
        /// half is the caller's responsibility per <see cref="GetCommonModules"/>'s remarks;
        /// this overload needs the chosen <paramref name="model"/> since <see cref="PreCheck"/>
        /// has no other way to learn it (no <c>state_manager</c> — rule 5).
        /// </summary>
        public bool PreCheck(FaceSwapperModel model) => FaceSwapper.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "face_swapper.PreCheck requires a FaceSwapperModel (no state_manager to read it from — call the FaceSwapperModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. <c>has_image</c>/<c>is_image</c>/<c>is_video</c>/
        /// <c>in_directory</c>/<c>same_file_extension</c> are <c>facefusion/filesystem.py</c>
        /// concerns not ported in this assignment (out of scope — see the assignment's file
        /// list); this checks the one condition expressible without them (a source path list is
        /// present) and otherwise trusts <paramref name="paths"/>, documenting the gap rather
        /// than silently under-validating.
        /// </summary>
        public bool PreProcess(ProcessMode mode, ProcessorRunPaths paths)
        {
            _ = mode;
            return paths.SourcePaths.Count > 0;
        }

        /// <inheritdoc />
        public ProcessorOutputs ProcessFrame(IProcessorInputs inputs)
        {
            if (inputs is not FaceSwapperInputs faceSwapperInputs)
            {
                throw new ArgumentException($"expected {nameof(FaceSwapperInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return FaceSwapper.ProcessFrame(faceSwapperInputs);
        }

        /// <summary>
        /// Python: <c>post_process()</c>. Cache clearing (<c>read_static_image.cache_clear()</c>,
        /// the inference pool, ...) is out of scope without <c>download.py</c>/a real pool owner
        /// to clear (rule 5) — a caller that owns those caches/pools clears them itself.
        /// </summary>
        public void PostProcess()
        {
        }
    }

    // -----------------------------------------------------------------
    // process_frame / swap_face / extract_source_face orchestration
    // -----------------------------------------------------------------

    /// <summary>Python: <c>extract_source_face</c>.</summary>
    public static FaceFusion.Types.Face? ExtractSourceFace(
        IReadOnlyList<Mat> sourceVisionFrames,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> getStaticFaces)
    {
        var sourceFaces = new List<FaceFusion.Types.Face>();

        foreach (var sourceVisionFrame in sourceVisionFrames)
        {
            var tempFaces = getStaticFaces(new[] { sourceVisionFrame });
            tempFaces = FaceSelector.SortFacesByOrder(tempFaces, FaceSelectorOrder.LargeSmall);

            if (tempFaces.Count > 0)
            {
                sourceFaces.Add(tempFaces[0]);
            }
        }

        return FaceCreator.AverageFaceIdentity(sourceFaces);
    }

    /// <summary>
    /// Python: <c>process_frame</c>. Returns the (possibly unchanged, if no source/target face
    /// was found — same as Python's early-return-less fallthrough) frame and mask. Caller owns
    /// the returned <see cref="ProcessorOutputs"/>'s <see cref="Mat"/>s; if no swap happened
    /// they are <paramref name="inputs"/>'s own <c>TempVisionFrame</c>/<c>TempVisionMask</c>
    /// (not cloned — Python returns the same object it was given in that case too).
    /// </summary>
    public static ProcessorOutputs ProcessFrame(FaceSwapperInputs inputs)
    {
        var targetVisionFrame = FaceFusion.Core.CommonHelper.GetMiddle(inputs.TargetVisionFrames);
        var sourceFace = ExtractSourceFace(inputs.SourceVisionFrames, inputs.GetStaticFaces);
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

        if (sourceFace is not null && targetFaces.Count > 0 && targetVisionFrame is not null)
        {
            var sourceVisionFrame = inputs.SourceVisionFrames[0];

            foreach (var rawTargetFace in targetFaces)
            {
                var targetFace = FaceCreator.ScaleFace(rawTargetFace, targetVisionFrame, tempVisionFrame);
                var nextTempVisionFrame = SwapFace(inputs, sourceFace, targetFace, sourceVisionFrame, tempVisionFrame);

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
    /// ownership of <paramref name="tempVisionFrame"/>/<paramref name="sourceVisionFrame"/>.
    /// Only <see cref="FaceMaskType.Box"/> is exercised by this assignment's parity fixtures
    /// (see class remarks); <see cref="FaceMaskType.Occlusion"/>/<see cref="FaceMaskType.Area"/>/
    /// <see cref="FaceMaskType.Region"/> are implemented (reusing <c>FaceFusion.Face.FaceMasker</c>,
    /// not reimplemented) but need their own session pool passed via
    /// <paramref name="occluderInferencePool"/>/<paramref name="faceOccluderModel"/>/
    /// <paramref name="faceMaskAreas"/>/<paramref name="parserInferencePool"/>/
    /// <paramref name="faceParserModel"/>/<paramref name="faceMaskRegions"/>, all optional and
    /// unused when the corresponding <see cref="FaceMaskType"/> is not requested.
    /// </summary>
    public static Mat SwapFace(
        FaceSwapperInputs inputs,
        FaceFusion.Types.Face sourceFace,
        FaceFusion.Types.Face targetFace,
        Mat sourceVisionFrame,
        Mat tempVisionFrame,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1,
        IReadOnlyList<FaceMaskArea>? faceMaskAreas = null,
        IReadOnlyDictionary<string, InferenceSession>? parserInferencePool = null,
        FaceParserModel faceParserModel = FaceParserModel.BisenetResnet18,
        IReadOnlyList<FaceMaskRegion>? faceMaskRegions = null)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[inputs.Model];
        var modelSize = modelOptions.Size;
        var pixelBoostSize = VisionHelper.UnpackResolution(inputs.PixelBoostResolution);
        var pixelBoostSizeCv = new Size(pixelBoostSize.Width, pixelBoostSize.Height);
        var pixelBoostTotal = pixelBoostSize.Width / modelSize.Width;

        var targetLandmark5Of68 = (float[,])targetFace.LandmarkSet.FiveOn68;
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, targetLandmark5Of68, modelOptions.Template, pixelBoostSizeCv);
        using var affineMatrixDisposable = affineMatrix;
        using var cropVisionFrameDisposable = cropVisionFrame;

        var cropMasks = new List<Mat>();

        try
        {
            if (inputs.FaceMaskTypes.Contains(FaceMaskType.Box))
            {
                cropMasks.Add(FaceMasker.CreateBoxMask(cropVisionFrame, inputs.FaceMaskBlur, inputs.FaceMaskPadding));
            }

            if (inputs.FaceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                if (occluderInferencePool is null)
                {
                    throw new ArgumentNullException(nameof(occluderInferencePool), "FaceMaskType.Occlusion requires occluderInferencePool.");
                }

                cropMasks.Add(FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool));
            }

            var pixelBoostSubFrames = PixelBoost.ImplodePixelBoost(cropVisionFrame, pixelBoostTotal, modelSize);
            var normalizedSubFrames = new List<Mat>(pixelBoostSubFrames.Count);

            try
            {
                float[] sourceInput;
                Size? sourceInputSize = null;

                if (inputs.Model is FaceSwapperModel.BlendSwap256 or FaceSwapperModel.Uniface256)
                {
                    var sourceLandmark5Of68 = (float[,])sourceFace.LandmarkSet.FiveOn68;
                    sourceInput = PrepareSourceFrame(modelOptions.Type, sourceLandmark5Of68, sourceVisionFrame);
                    sourceInputSize = modelOptions.Type == FaceSwapperModelKind.BlendSwap ? new Size(112, 112) : new Size(256, 256);
                }
                else
                {
                    var preparedSourceEmbedding = PrepareSourceEmbedding(
                        modelOptions.Type,
                        (float[])sourceFace.Embedding,
                        (float[])sourceFace.EmbeddingNorm,
                        inputs.EmbeddingConverterSession,
                        inputs.InswapperModelInitializer);
                    sourceInput = BalanceSourceEmbedding(modelOptions.Type, preparedSourceEmbedding, (float[])targetFace.Embedding, inputs.Weight);
                }

                foreach (var pixelBoostSubFrame in pixelBoostSubFrames)
                {
                    using var subFrameDisposable = pixelBoostSubFrame;

                    var preparedTarget = PrepareCropFrame(pixelBoostSubFrame, modelOptions.Mean, modelOptions.StandardDeviation);
                    var modelOutput = ForwardSwapFace(inputs.FaceSwapperSession, sourceInput, sourceInputSize, preparedTarget, modelSize);
                    normalizedSubFrames.Add(NormalizeCropFrame(modelOutput, modelSize.Height, modelSize.Width, modelOptions.Type, modelOptions.Mean, modelOptions.StandardDeviation));
                }

                using var explodedCropFrame = PixelBoost.ExplodePixelBoost(normalizedSubFrames, pixelBoostTotal, modelSize, pixelBoostSizeCv);

                if (inputs.FaceMaskTypes.Contains(FaceMaskType.Area))
                {
                    if (faceMaskAreas is null)
                    {
                        throw new ArgumentNullException(nameof(faceMaskAreas), "FaceMaskType.Area requires faceMaskAreas.");
                    }

                    var landmark68 = (float[,])targetFace.LandmarkSet.SixtyEight;
                    var transformedLandmark68 = FaceHelper.TransformPoints(landmark68, affineMatrix);
                    cropMasks.Add(FaceMasker.CreateAreaMask(explodedCropFrame, transformedLandmark68, faceMaskAreas));
                }

                if (inputs.FaceMaskTypes.Contains(FaceMaskType.Region))
                {
                    if (parserInferencePool is null || faceMaskRegions is null)
                    {
                        throw new ArgumentNullException(nameof(parserInferencePool), "FaceMaskType.Region requires parserInferencePool and faceMaskRegions.");
                    }

                    cropMasks.Add(FaceMasker.CreateRegionMask(explodedCropFrame, faceMaskRegions, faceParserModel, parserInferencePool));
                }

                using var cropMask = ReduceMinimumClip01(cropMasks);
                return PasteBackFloatCrop(tempVisionFrame, explodedCropFrame, cropMask, affineMatrix);
            }
            finally
            {
                foreach (var subFrame in pixelBoostSubFrames)
                {
                    // Already disposed by the `using subFrameDisposable` above for every frame
                    // actually iterated; this is a defensive no-op double-dispose guard for the
                    // (never expected) case ForwardSwapFace throws mid-loop, matching
                    // FaceMasker.CreateOcclusionMask's own `finally`-dispose-everything pattern.
                    subFrame.Dispose();
                }
            }
        }
        finally
        {
            foreach (var cropMask in cropMasks)
            {
                cropMask.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>numpy.minimum.reduce(crop_masks).clip(0, 1)</c>. Caller owns the returned
    /// <see cref="Mat"/> (<c>CV_32FC1</c>).
    /// </summary>
    private static Mat ReduceMinimumClip01(IReadOnlyList<Mat> masks)
    {
        if (masks.Count == 0)
        {
            throw new ArgumentException("at least one face mask type must be enabled.", nameof(masks));
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
