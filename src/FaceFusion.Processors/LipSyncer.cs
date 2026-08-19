using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/lip_syncer/types.py</c>'s
/// <c>LipSyncerModel = Literal['edtalk_256', 'wav2lip_96', 'wav2lip_gan_96']</c>.
/// </summary>
public enum LipSyncerModel
{
    [WireName("edtalk_256")]
    Edtalk256,

    [WireName("wav2lip_96")]
    Wav2Lip96,

    [WireName("wav2lip_gan_96")]
    Wav2LipGan96,
}

/// <summary>
/// Python: each model's <c>'type'</c> entry in <c>create_static_model_set</c> — the two
/// distinct lip-sync maths families the three <see cref="LipSyncerModel"/> values group into
/// (<c>wav2lip_96</c>/<c>wav2lip_gan_96</c> share the <c>wav2lip</c> branch, differing only in
/// which <c>.onnx</c> weights are loaded). Selects which branch of <see cref="LipSyncer.SyncLip"/>/
/// <see cref="LipSyncer.PrepareAudioFrame"/> a model uses.
/// </summary>
public enum LipSyncerModelKind
{
    [WireName("edtalk")]
    Edtalk,

    [WireName("wav2lip")]
    Wav2Lip,
}

/// <summary>
/// Python: one entry of <c>create_static_model_set</c>'s return value (a <c>ModelOptions</c>
/// dict) for the <c>lip_syncer</c> module specifically. The <c>__metadata__</c> sub-dict
/// (vendor/license/year) is not reproduced — display-only in Python, nothing in this port's
/// scope reads it — matching <c>FaceSwapperModelOptions</c>'s precedent.
/// </summary>
public sealed record LipSyncerModelOptions(
    IReadOnlyDictionary<string, Download> Hashes,
    IReadOnlyDictionary<string, Download> Sources,
    LipSyncerModelKind Type,
    Size Size);

/// <summary>
/// Port of <c>facefusion/processors/modules/lip_syncer/{core,types,choices}.py</c>. Replaces a
/// target face's mouth region with lip movement synchronised to a source voice/audio track,
/// using either the <c>edtalk</c> model family (a single warped 512x512 face crop, a scalar
/// blend weight, and the full crop swapped) or the <c>wav2lip</c> family (a masked lower-face
/// crop cut out, warped separately, run through the model, and pasted back into the 512x512
/// crop before the final <c>paste_back</c>).
///
/// <para>
/// <b>No global state; sessions and settings taken as parameters (PORT_CONVENTIONS.md rule 5).</b>
/// Every <c>state_manager.get_item(...)</c> read in the Python source becomes an explicit
/// parameter — model choice, weight, mask settings all flow in through <see cref="SyncLip"/>'s
/// parameter list or through the resolved <c>InferenceSession</c> a caller passes in, matching
/// how <see cref="FaceSwapper"/> was ported. <c>get_inference_pool</c>/<c>pre_check</c>'s real
/// downloading is not reproduced for the same reason <c>FaceSwapper.cs</c> doesn't reproduce it
/// either — <c>facefusion/download.py</c> is not ported in this repo. <see cref="ResolveModelsDirectory"/>/
/// <see cref="BuildDownloadUrl"/> reuse the same precedent <c>FaceSwapper.cs</c> established for
/// that gap (resolve URLs directly against <c>FaceFusion.Types.Choices.DownloadProviderSet</c>'s
/// github entry; resolve local paths by walking up to the repository root).
/// </para>
///
/// <para>
/// <b>Audio layer reused, not reimplemented.</b> <c>prepare_audio_frame</c>'s *input*
/// (Python: an <c>AudioFrame</c>, shape <c>(80, 16)</c>) is produced entirely by
/// <c>FaceFusion.Media.Audio</c> (<c>ReadVoice</c>/<c>GetVoiceFrame</c>/<c>ExtractAudioFrames</c>
/// — already ported and parity-verified against real scipy, see <c>AudioParityTests</c>); this
/// file only reproduces <c>prepare_audio_frame</c> itself (the small elementwise transform that
/// turns a raw mel-spectrogram slice into the model's <c>'source'</c> input) and takes the
/// <c>double[,]</c> <c>AudioFrame</c> as a plain parameter — no new audio math.
/// </para>
///
/// <para>
/// <b>Mat / dtype conventions.</b> <c>VisionFrame</c>s are <see cref="Mat"/>, <c>CV_8UC3</c> BGR,
/// caller-owned on every return, matching <see cref="FaceHelper"/>/<see cref="FaceSwapper"/>.
/// Unlike <see cref="FaceSwapper.NormalizeCropFrame"/>, <see cref="NormalizeCropFrameEdtalk"/>/
/// <see cref="NormalizeCropFrameWav2Lip"/> always narrow straight to <c>uint8</c> — Python's
/// <c>normalize_crop_frame</c> here has no "renormalize by list mean/std" branch (lip_syncer's
/// model catalog carries no mean/std at all), so there is no float64-promotion case to
/// reproduce, and the resulting crop is <c>CV_8UC3</c> the whole way through
/// <see cref="SyncLip"/> — which lets this file reuse <c>FaceHelper.PasteBack</c> directly
/// (the <c>CV_8UC3</c>-only overload), unlike <c>FaceSwapper.PasteBackFloatCrop</c>'s bespoke
/// float-crop variant.
/// </para>
///
/// <para>
/// <b><c>crop_mask</c> is not clipped (deliberate, verified against Python source).</b> Python:
/// <c>crop_mask = numpy.minimum.reduce(crop_masks)</c> — no trailing <c>.clip(0, 1)</c>, unlike
/// <c>face_swapper.swap_face</c>'s <c>numpy.minimum.reduce(crop_masks).clip(0, 1)</c>. Reproduced
/// exactly via <see cref="ReduceMinimum"/> (no clamp), not "fixed" to match <c>FaceSwapper</c>'s
/// pattern, per PORT_CONVENTIONS.md rule 1.
/// </para>
/// </summary>
public static class LipSyncer
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>lip_syncer_models = list(get_args(LipSyncerModel))</c>.</summary>
    public static readonly IReadOnlyList<LipSyncerModel> LipSyncerModels = Enum.GetValues<LipSyncerModel>();

    /// <summary>Python: <c>lip_syncer_weight_range = create_float_range(0.0, 1.0, 0.05)</c>.</summary>
    public static readonly IReadOnlyList<double> LipSyncerWeightRange =
        CommonHelper.CreateFloatRange(0.0, 1.0, 0.05);

    // -----------------------------------------------------------------
    // create_static_model_set
    // -----------------------------------------------------------------

    private static readonly object ModelCatalogLock = new();
    private static IReadOnlyDictionary<LipSyncerModel, LipSyncerModelOptions>? _cachedModelCatalog;

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). <paramref name="downloadScope"/>
    /// is accepted for signature parity with Python — every family's entry is identical
    /// regardless of scope, same as <c>FaceSwapper.CreateStaticModelSet</c>.
    /// </summary>
    public static IReadOnlyDictionary<LipSyncerModel, LipSyncerModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        lock (ModelCatalogLock)
        {
            return _cachedModelCatalog ??= BuildModelCatalog();
        }
    }

    private static IReadOnlyDictionary<LipSyncerModel, LipSyncerModelOptions> BuildModelCatalog()
    {
        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];

        (IReadOnlyDictionary<string, Download> Hashes, IReadOnlyDictionary<string, Download> Sources) Component(string modelsBaseName, string fileName)
        {
            var hashes = new Dictionary<string, Download>
            {
                ["lip_syncer"] = new Download(
                    BuildDownloadUrl(githubProvider, modelsBaseName, fileName + ".hash"),
                    Path.Combine(modelsDirectory, fileName + ".hash")),
            };
            var sources = new Dictionary<string, Download>
            {
                ["lip_syncer"] = new Download(
                    BuildDownloadUrl(githubProvider, modelsBaseName, fileName + ".onnx"),
                    Path.Combine(modelsDirectory, fileName + ".onnx")),
            };

            return (hashes, sources);
        }

        var catalog = new Dictionary<LipSyncerModel, LipSyncerModelOptions>();

        void Add(LipSyncerModel model, string modelsBaseName, string fileName, LipSyncerModelKind type, int size)
        {
            var (hashes, sources) = Component(modelsBaseName, fileName);
            catalog[model] = new LipSyncerModelOptions(hashes, sources, type, new Size(size, size));
        }

        Add(LipSyncerModel.Edtalk256, "models-3.3.0", "edtalk_256", LipSyncerModelKind.Edtalk, 256);
        Add(LipSyncerModel.Wav2Lip96, "models-3.0.0", "wav2lip_96", LipSyncerModelKind.Wav2Lip, 96);
        Add(LipSyncerModel.Wav2LipGan96, "models-3.0.0", "wav2lip_gan_96", LipSyncerModelKind.Wav2Lip, 96);

        return catalog;
    }

    /// <summary>
    /// Same reasoning and implementation as <c>FaceSwapper.ResolveModelsDirectory</c> — walks
    /// up from the running assembly's base directory to the repository root (marked by
    /// <c>FaceFusion.sln</c>) rather than resolving relative to the build output directory.
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
    /// Python: the <c>lip_syncer</c>-specific half of <c>pre_check</c> (the common-module half
    /// — <c>content_analyser</c>/.../<c>voice_extractor</c> — is the caller's responsibility;
    /// see <see cref="IProcessor.GetCommonModules"/>'s remarks). Verifies every hash/source file
    /// this <paramref name="model"/> needs is already present locally; unlike Python, does not
    /// download a missing one (see class remarks).
    /// </summary>
    public static bool PreCheck(LipSyncerModel model)
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
    // prepare_audio_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_audio_frame</c>. <paramref name="audioFrame"/> is the raw mel-spectrogram
    /// slice (shape <c>(80, 16)</c>, row = mel bin, column = time step — <c>FaceFusion.Media.Audio</c>'s
    /// <c>AudioFrame</c> convention) fed in as <c>double[,]</c> exactly as Python's own
    /// <c>AudioFrame</c> is float64 at this point (see class remarks). Returns a flat length-1280
    /// array in the same row-major (mel, step) order — Python's <c>expand_dims(axis=(0,1))</c>
    /// only inserts leading singleton dims, it does not reorder the underlying buffer, so the
    /// flat layout here is identical to <paramref name="audioFrame"/>'s own row-major layout.
    ///
    /// <para>
    /// <b>Dtype (verified against Python source, NumPy 2 weak-scalar promotion).</b> The
    /// <c>maximum</c>/<c>log10</c>/<c>* 1.6 + 3.2</c>/<c>clip</c> chain runs in float64 (the
    /// input is float64 and every operand is a Python float), narrowed to <c>float32</c> only
    /// at the explicit <c>.astype(numpy.float32)</c> — computed here in <see cref="double"/> and
    /// narrowed to <see cref="float"/> only at that same point, per PORT_CONVENTIONS.md rule 6.
    /// The <c>wav2lip</c>-only <c>* weight * 2.0</c> multiply happens *after* that cast, so it
    /// runs at float32 precision (NumPy 2's weak-scalar promotion keeps an ndarray's own dtype
    /// when multiplied by a bare Python scalar) — reproduced here as two chained
    /// <see cref="float"/> multiplications, not upcast to double.
    /// </para>
    /// </summary>
    public static float[] PrepareAudioFrame(LipSyncerModelKind modelType, double[,] audioFrame, double lipSyncerWeight)
    {
        var melFilterTotal = audioFrame.GetLength(0);
        var stepTotal = audioFrame.GetLength(1);
        var threshold = Math.Exp(-5.0 * Math.Log(10.0));
        var result = new float[melFilterTotal * stepTotal];

        for (var mel = 0; mel < melFilterTotal; mel++)
        {
            for (var step = 0; step < stepTotal; step++)
            {
                var value = Math.Max(threshold, audioFrame[mel, step]);
                value = (Math.Log10(value) * 1.6) + 3.2;
                value = Math.Clamp(value, -4.0, 4.0);

                var narrowed = (float)value;

                if (modelType == LipSyncerModelKind.Wav2Lip)
                {
                    narrowed = narrowed * (float)lipSyncerWeight * 2f;
                }

                result[(mel * stepTotal) + step] = narrowed;
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame — model-specific 'target' preprocessing
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_crop_frame</c>'s <c>edtalk</c> branch. <paramref name="cropVisionFrame"/>
    /// is the 512x512 <c>ffhq_512</c> face crop; resizes to <paramref name="modelSize"/> (256x256)
    /// with <c>cv2.INTER_AREA</c> first. Returns a flat <c>(1, 3, h, w)</c> RGB/255.0 array.
    /// Division precedes the <c>.astype(float32)</c> cast in Python (<c>[:, :, ::-1] / 255.0</c>
    /// then <c>transpose</c>/<c>expand_dims</c>/<c>.astype(float32)</c>), so this computes in
    /// <see cref="double"/> and narrows only at assignment, matching
    /// <c>FaceSwapper.PrepareSourceFrame</c>'s identical tail exactly.
    /// </summary>
    public static float[] PrepareCropFrameEdtalk(Mat cropVisionFrame, Size modelSize)
    {
        using var resized = new Mat();
        Cv2.Resize(cropVisionFrame, resized, modelSize, 0, 0, InterpolationFlags.Area);

        var height = resized.Rows;
        var width = resized.Cols;
        var plane = height * width;
        var chw = new float[3 * plane];

        resized.GetArray(out Vec3b[] pixels);

        for (var index = 0; index < plane; index++)
        {
            var pixel = pixels[index];
            chw[index] = (float)(pixel.Item2 / 255.0); // R -> channel 0
            chw[plane + index] = (float)(pixel.Item1 / 255.0); // G -> channel 1
            chw[(2 * plane) + index] = (float)(pixel.Item0 / 255.0); // B -> channel 2
        }

        return chw;
    }

    /// <summary>
    /// Python: <c>prepare_crop_frame</c>'s <c>wav2lip</c> branch. <paramref name="areaVisionFrame"/>
    /// is the model-size (96x96) bounding-box crop from <c>warp_face_by_bounding_box</c> — still
    /// BGR <c>uint8</c>, no channel flip anywhere in this branch (unlike <c>edtalk</c>). Returns a
    /// flat <c>(1, 6, h, w)</c> array: channels 0-2 are a copy with the lower half of the rows
    /// zeroed (Python: <c>prepare_vision_frame[:, model_size[0] // 2:] = 0</c>, indexing the
    /// height axis of the NHWC array), channels 3-5 are the frame unmodified — Python's
    /// <c>concatenate((prepare_vision_frame, crop_vision_frame), axis=3)</c> (the NHWC channel
    /// axis) followed by <c>transpose(0, 3, 1, 2)</c>.
    ///
    /// <para>
    /// <b>Dtype.</b> Python casts to <c>float32</c> *before* dividing by 255.0
    /// (<c>.astype(numpy.float32) / 255.0</c>), so — unlike <see cref="PrepareCropFrameEdtalk"/> —
    /// the division itself runs at float32 precision (NumPy 2 weak-scalar promotion): reproduced
    /// here as a direct <c>byte / 255f</c> division, not routed through <see cref="double"/>.
    /// </para>
    /// </summary>
    public static float[] PrepareCropFrameWav2Lip(Mat areaVisionFrame)
    {
        var height = areaVisionFrame.Rows;
        var width = areaVisionFrame.Cols;
        var plane = height * width;
        var zeroFromRow = height / 2; // Python: model_size[0] // 2 (square model size, so height/2 == width/2)

        areaVisionFrame.GetArray(out Vec3b[] pixels);

        var chw = new float[6 * plane];

        for (var row = 0; row < height; row++)
        {
            var isZeroed = row >= zeroFromRow;

            for (var col = 0; col < width; col++)
            {
                var index = (row * width) + col;
                var pixel = pixels[index];

                var b = pixel.Item0 / 255f;
                var g = pixel.Item1 / 255f;
                var r = pixel.Item2 / 255f;

                chw[index] = isZeroed ? 0f : b;
                chw[plane + index] = isZeroed ? 0f : g;
                chw[(2 * plane) + index] = isZeroed ? 0f : r;

                chw[(3 * plane) + index] = b;
                chw[(4 * plane) + index] = g;
                chw[(5 * plane) + index] = r;
            }
        }

        return chw;
    }

    // -----------------------------------------------------------------
    // forward_edtalk / forward_wav2lip
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward_edtalk</c>'s ONNX call. <paramref name="sourceAudioInput"/> is the
    /// flat length-1280 array from <see cref="PrepareAudioFrame"/> (shape <c>(1, 1, 80, 16)</c>);
    /// <paramref name="targetInput"/> is the flat CHW frame from <see cref="PrepareCropFrameEdtalk"/>
    /// (shape <c>(1, 3, modelSize, modelSize)</c>). Every output is requested (Python:
    /// <c>output_names=None</c>) and only the first used, same as Python's <c>[0]</c>.
    /// </summary>
    public static float[] ForwardEdtalk(
        InferenceSession lipSyncerSession, float[] sourceAudioInput, float[] targetInput, Size targetInputSize, float lipSyncerWeight)
    {
        using var sourceOrtValue = OrtValue.CreateTensorValueFromMemory(sourceAudioInput, new long[] { 1, 1, 80, 16 });
        using var targetOrtValue = OrtValue.CreateTensorValueFromMemory(targetInput, new long[] { 1, 3, targetInputSize.Height, targetInputSize.Width });

        var weightInput = new[] { lipSyncerWeight };
        using var weightOrtValue = OrtValue.CreateTensorValueFromMemory(weightInput, new long[] { 1 });

        var inputs = new Dictionary<string, OrtValue>
        {
            ["source"] = sourceOrtValue,
            ["target"] = targetOrtValue,
            ["weight"] = weightOrtValue,
        };

        using var runOptions = new RunOptions();
        using var results = lipSyncerSession.Run(runOptions, inputs, lipSyncerSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>
    /// Python: <c>forward_wav2lip</c>'s ONNX call. <paramref name="sourceAudioInput"/> is the
    /// flat length-1280 array from <see cref="PrepareAudioFrame"/> (shape <c>(1, 1, 80, 16)</c>);
    /// <paramref name="targetInput"/> is the flat 6-channel frame from
    /// <see cref="PrepareCropFrameWav2Lip"/> (shape <c>(1, 6, modelSize, modelSize)</c>). No
    /// <c>'weight'</c> input — only <c>edtalk</c>'s model takes one.
    /// </summary>
    public static float[] ForwardWav2Lip(InferenceSession lipSyncerSession, float[] sourceAudioInput, float[] targetInput, Size targetInputSize)
    {
        using var sourceOrtValue = OrtValue.CreateTensorValueFromMemory(sourceAudioInput, new long[] { 1, 1, 80, 16 });
        using var targetOrtValue = OrtValue.CreateTensorValueFromMemory(targetInput, new long[] { 1, 6, targetInputSize.Height, targetInputSize.Width });

        var inputs = new Dictionary<string, OrtValue>
        {
            ["source"] = sourceOrtValue,
            ["target"] = targetOrtValue,
        };

        using var runOptions = new RunOptions();
        using var results = lipSyncerSession.Run(runOptions, inputs, lipSyncerSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    // -----------------------------------------------------------------
    // normalize_crop_frame — model-specific model-output postprocessing
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>normalize_crop_frame</c>'s common tail (<c>transpose(1,2,0)</c>,
    /// <c>clip(0,1)*255</c>, <c>.astype(uint8)</c>) plus the <c>edtalk</c> branch's RGB-&gt;BGR
    /// flip and final <c>cv2.resize(..., (512, 512), interpolation=cv2.INTER_CUBIC)</c>. Caller
    /// owns the returned <see cref="Mat"/> (<c>CV_8UC3</c>, 512x512).
    /// </summary>
    public static Mat NormalizeCropFrameEdtalk(ReadOnlySpan<float> modelOutputChw, int height, int width)
    {
        var plane = height * width;

        if (modelOutputChw.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputChw has {modelOutputChw.Length} elements, expected {3 * plane} for a {width}x{height} CHW frame.", nameof(modelOutputChw));
        }

        using var rgbFrame = new Mat(height, width, MatType.CV_8UC3);
        var data = new Vec3b[plane];

        for (var index = 0; index < plane; index++)
        {
            var r = ClipUnitToByte(modelOutputChw[index]);
            var g = ClipUnitToByte(modelOutputChw[plane + index]);
            var b = ClipUnitToByte(modelOutputChw[(2 * plane) + index]);

            // Write directly in BGR order — equivalent to Python's `[:, :, ::-1]` RGB->BGR flip
            // followed by a separate resize; channel order and resize are independent so this
            // reorders once instead of twice.
            data[index] = new Vec3b { Item0 = b, Item1 = g, Item2 = r };
        }

        rgbFrame.SetArray(data);

        var result = new Mat();
        Cv2.Resize(rgbFrame, result, new Size(512, 512), 0, 0, InterpolationFlags.Cubic);
        return result;
    }

    /// <summary>
    /// Python: <c>normalize_crop_frame</c>'s common tail for the <c>wav2lip</c> branch — no RGB
    /// flip, no resize (the caller warps the result back into the 512x512 crop separately via
    /// <c>cv2.warpAffine</c>). Caller owns the returned <see cref="Mat"/> (<c>CV_8UC3</c>,
    /// <paramref name="width"/>x<paramref name="height"/>).
    /// </summary>
    public static Mat NormalizeCropFrameWav2Lip(ReadOnlySpan<float> modelOutputChw, int height, int width)
    {
        var plane = height * width;

        if (modelOutputChw.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputChw has {modelOutputChw.Length} elements, expected {3 * plane} for a {width}x{height} CHW frame.", nameof(modelOutputChw));
        }

        var result = new Mat(height, width, MatType.CV_8UC3);
        var data = new Vec3b[plane];

        for (var index = 0; index < plane; index++)
        {
            var c0 = ClipUnitToByte(modelOutputChw[index]);
            var c1 = ClipUnitToByte(modelOutputChw[plane + index]);
            var c2 = ClipUnitToByte(modelOutputChw[(2 * plane) + index]);

            data[index] = new Vec3b { Item0 = c0, Item1 = c1, Item2 = c2 };
        }

        result.SetArray(data);
        return result;
    }

    /// <summary>Python: <c>value.clip(0, 1) * 255</c> then <c>.astype(uint8)</c> (truncation toward zero).</summary>
    private static byte ClipUnitToByte(float value)
    {
        var clipped = value < 0f ? 0f : (value > 1f ? 1f : value);
        return (byte)(clipped * 255f);
    }

    // -----------------------------------------------------------------
    // sync_lip
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>sync_lip</c>. Caller owns the returned <see cref="Mat"/>. Does not take
    /// ownership of <paramref name="tempVisionFrame"/>. <paramref name="occluderInferencePool"/>/
    /// <paramref name="faceOccluderModel"/> are only used when <see cref="FaceMaskType.Occlusion"/>
    /// is requested, matching <c>FaceSwapper.SwapFace</c>'s equivalent optional parameters (this
    /// module has no <c>area</c>/<c>region</c> mask types of its own — <c>wav2lip</c> always adds
    /// its own <c>lower-face</c> area mask internally, not driven by
    /// <paramref name="faceMaskTypes"/> — see the Python source's unconditional
    /// <c>create_area_mask</c> call in the <c>wav2lip</c> branch).
    /// </summary>
    public static Mat SyncLip(
        FaceFusion.Types.Face targetFace,
        double[,] sourceVoiceFrame,
        Mat tempVisionFrame,
        LipSyncerModel model,
        double lipSyncerWeight,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        double faceMaskBlur,
        Padding faceMaskPadding,
        InferenceSession lipSyncerSession,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[model];
        var modelType = modelOptions.Type;
        var modelSize = modelOptions.Size;

        var preparedAudioFrame = PrepareAudioFrame(modelType, sourceVoiceFrame, lipSyncerWeight);

        var targetLandmark5Of68 = (float[,])targetFace.LandmarkSet.FiveOn68;
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, targetLandmark5Of68, WarpTemplate.Ffhq512, new Size(512, 512));
        using var affineMatrixDisposable = affineMatrix;
        using var cropVisionFrameDisposable = cropVisionFrame;

        var cropMasks = new List<Mat>();
        Mat? resultCropVisionFrame = null;

        try
        {
            if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                if (occluderInferencePool is null)
                {
                    throw new ArgumentNullException(nameof(occluderInferencePool), "FaceMaskType.Occlusion requires occluderInferencePool.");
                }

                cropMasks.Add(FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool));
            }

            if (modelType == LipSyncerModelKind.Edtalk)
            {
                cropMasks.Add(FaceMasker.CreateBoxMask(cropVisionFrame, faceMaskBlur, faceMaskPadding));

                var preparedCropFrame = PrepareCropFrameEdtalk(cropVisionFrame, modelSize);
                var modelOutput = ForwardEdtalk(lipSyncerSession, preparedAudioFrame, preparedCropFrame, modelSize, (float)lipSyncerWeight);
                resultCropVisionFrame = NormalizeCropFrameEdtalk(modelOutput, modelSize.Height, modelSize.Width);
            }
            else
            {
                var targetLandmark68 = (float[,])targetFace.LandmarkSet.SixtyEight;
                var faceLandmark68 = FaceHelper.TransformPoints(targetLandmark68, affineMatrix);

                cropMasks.Add(FaceMasker.CreateAreaMask(cropVisionFrame, faceLandmark68, new[] { FaceMaskArea.LowerFace }));

                var boundingBox = FaceHelper.CreateBoundingBox(faceLandmark68);
                var (areaVisionFrame, areaMatrix) = FaceHelper.WarpFaceByBoundingBox(cropVisionFrame, boundingBox, modelSize);
                using var areaMatrixDisposable = areaMatrix;
                using var areaVisionFrameDisposable = areaVisionFrame;

                var preparedAreaFrame = PrepareCropFrameWav2Lip(areaVisionFrame);
                var modelOutput = ForwardWav2Lip(lipSyncerSession, preparedAudioFrame, preparedAreaFrame, modelSize);

                using var normalizedAreaVisionFrame = NormalizeCropFrameWav2Lip(modelOutput, modelSize.Height, modelSize.Width);
                using var inverseAreaMatrix = new Mat();
                Cv2.InvertAffineTransform(areaMatrix, inverseAreaMatrix);

                resultCropVisionFrame = new Mat();
                Cv2.WarpAffine(normalizedAreaVisionFrame, resultCropVisionFrame, inverseAreaMatrix, new Size(512, 512), InterpolationFlags.Linear, BorderTypes.Replicate);
            }

            using var cropMask = ReduceMinimum(cropMasks);
            return FaceHelper.PasteBack(tempVisionFrame, resultCropVisionFrame, cropMask, affineMatrix);
        }
        finally
        {
            resultCropVisionFrame?.Dispose();

            foreach (var mask in cropMasks)
            {
                mask.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>numpy.minimum.reduce(crop_masks)</c> — deliberately *not* clipped afterward,
    /// unlike <c>FaceSwapper</c>'s equivalent; see class remarks. Caller owns the returned
    /// <see cref="Mat"/> (<c>CV_32FC1</c>).
    /// </summary>
    private static Mat ReduceMinimum(IReadOnlyList<Mat> masks)
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

        return result;
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: the <c>facefusion.processors.modules.lip_syncer.core</c> module's per-call
    /// inputs, extended per <see cref="IProcessorInputs"/>'s remarks to also carry every
    /// setting/session Python would have pulled from <c>state_manager</c>/its own
    /// <c>get_inference_pool()</c> for this call.
    /// </summary>
    public sealed record LipSyncerInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        double[,] SourceVoiceFrame,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        LipSyncerModel Model,
        double Weight,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        double FaceMaskBlur,
        Padding FaceMaskPadding,
        InferenceSession LipSyncerSession,
        IReadOnlyDictionary<string, InferenceSession>? OccluderInferencePool,
        FaceOccluderModel FaceOccluderModel,
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
    /// Python: <c>facefusion/processors/modules/lip_syncer/core.py</c>'s module-level functions,
    /// adapted to the <see cref="IProcessor"/> contract. Mirrors <see cref="FaceSwapper.Processor"/>'s
    /// shape exactly.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "lip_syncer";

        /// <summary>
        /// Python: <c>get_common_modules()</c> — <c>content_analyser</c>, <c>face_classifier</c>,
        /// <c>face_detector</c>, <c>face_landmarker</c>, <c>face_masker</c>, <c>face_recognizer</c>,
        /// <c>voice_extractor</c> (the last is <c>lip_syncer</c>-only, not shared with
        /// <see cref="FaceSwapper"/>'s list).
        /// </summary>
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer", "voice_extractor" };

        /// <summary>
        /// Python: the <c>lip_syncer</c>-specific half of <c>pre_check</c>. The common-module
        /// half is the caller's responsibility per <see cref="GetCommonModules"/>'s remarks.
        /// </summary>
        public bool PreCheck(LipSyncerModel model) => LipSyncer.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "lip_syncer.PreCheck requires a LipSyncerModel (no state_manager to read it from — call the LipSyncerModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c> — <c>if not has_audio(state_manager.get_item('source_paths')): ... return False</c>,
        /// else <c>return True</c>. <paramref name="mode"/> is unused, matching Python (the
        /// check does not depend on it). The logger/translator error message Python emits
        /// before returning <see langword="false"/> is out of scope (no logger/translator port
        /// in this assignment), same simplification <c>FaceSwapper.Processor.PreProcess</c>
        /// documents for its own gaps.
        /// </summary>
        public bool PreProcess(ProcessMode mode, ProcessorRunPaths paths)
        {
            _ = mode;
            return FileSystem.HasAudio(paths.SourcePaths);
        }

        /// <inheritdoc />
        public ProcessorOutputs ProcessFrame(IProcessorInputs inputs)
        {
            if (inputs is not LipSyncerInputs lipSyncerInputs)
            {
                throw new ArgumentException($"expected {nameof(LipSyncerInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return LipSyncer.ProcessFrame(lipSyncerInputs);
        }

        /// <summary>
        /// Python: <c>post_process()</c>. Cache clearing is out of scope without
        /// <c>download.py</c>/a real pool owner to clear (rule 5), same as
        /// <c>FaceSwapper.Processor.PostProcess</c>.
        /// </summary>
        public void PostProcess()
        {
        }
    }

    // -----------------------------------------------------------------
    // process_frame orchestration
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Returns the (possibly unchanged, if no target face was
    /// found) frame and mask. Caller owns the returned <see cref="ProcessorOutputs"/>'s
    /// <see cref="Mat"/>s; if no sync happened they are <paramref name="inputs"/>'s own
    /// <c>TempVisionFrame</c>/<c>TempVisionMask</c> (not cloned), same as
    /// <c>FaceSwapper.ProcessFrame</c>'s equivalent early-return-less fallthrough.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(LipSyncerInputs inputs)
    {
        var targetVisionFrame = CommonHelper.GetMiddle(inputs.TargetVisionFrames);
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
                var nextTempVisionFrame = SyncLip(
                    targetFace,
                    inputs.SourceVoiceFrame,
                    tempVisionFrame,
                    inputs.Model,
                    inputs.Weight,
                    inputs.FaceMaskTypes,
                    inputs.FaceMaskBlur,
                    inputs.FaceMaskPadding,
                    inputs.LipSyncerSession,
                    inputs.OccluderInferencePool,
                    inputs.FaceOccluderModel);

                if (!ReferenceEquals(tempVisionFrame, inputs.TempVisionFrame))
                {
                    tempVisionFrame.Dispose();
                }

                tempVisionFrame = nextTempVisionFrame;
            }
        }

        return new ProcessorOutputs(tempVisionFrame, inputs.TempVisionMask);
    }
}
