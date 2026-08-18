using FaceFusion.Core;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/background_remover/types.py</c>'s
/// <c>BackgroundRemoverModel = Literal['ben_2', 'birefnet_general', 'birefnet_portrait',
/// 'corridor_key_1024', 'corridor_key_2048', 'isnet_general', 'modnet', 'ormbg', 'rmbg_1.4',
/// 'rmbg_2.0', 'silueta', 'u2net_cloth', 'u2net_general', 'u2net_human', 'u2netp']</c>. Declared
/// here (not <c>FaceFusion.Types</c>) per the file-scope constraint, matching
/// <c>FaceEnhancerModel</c>/<c>FrameColorizerModel</c>'s own local declarations. Members with a
/// wire name that is not a legal C# identifier (<c>rmbg_1.4</c>/<c>rmbg_2.0</c> contain a
/// literal dot) get a plain numeric suffix instead (<c>Rmbg14</c>/<c>Rmbg20</c>) — the
/// <see cref="WireNameAttribute"/> string is the source of truth, the identifier is just a name.
/// </summary>
public enum BackgroundRemoverModel
{
    [WireName("ben_2")]
    Ben2,

    [WireName("birefnet_general")]
    BirefnetGeneral,

    [WireName("birefnet_portrait")]
    BirefnetPortrait,

    [WireName("corridor_key_1024")]
    CorridorKey1024,

    [WireName("corridor_key_2048")]
    CorridorKey2048,

    [WireName("isnet_general")]
    IsnetGeneral,

    [WireName("modnet")]
    Modnet,

    [WireName("ormbg")]
    Ormbg,

    [WireName("rmbg_1.4")]
    Rmbg14,

    [WireName("rmbg_2.0")]
    Rmbg20,

    [WireName("silueta")]
    Silueta,

    [WireName("u2net_cloth")]
    U2netCloth,

    [WireName("u2net_general")]
    U2netGeneral,

    [WireName("u2net_human")]
    U2netHuman,

    [WireName("u2netp")]
    U2netp,
}

/// <summary>
/// Python: the model set's <c>'type'</c> field. Only <see cref="CorridorKey"/> and
/// <see cref="U2netCloth"/> get bespoke handling in <c>remove_background</c>/<c>forward</c> —
/// every other type (<c>ben</c>, <c>birefnet</c>, <c>isnet</c>, <c>modnet</c>, <c>ormbg</c>,
/// <c>rmbg</c>, <c>silueta</c>, <c>u2net</c>, <c>u2netp</c>) shares one code path (a plain
/// single-channel mask model), so this port keeps them as one <see cref="Standard"/> case rather
/// than nine near-identical enum members with no behavioural difference — the model-specific
/// bits that actually vary (size, mean, std) live on <see cref="BackgroundRemover.ModelOptions"/>
/// regardless.
/// </summary>
public enum BackgroundRemoverModelType
{
    Standard,
    CorridorKey,
    U2netCloth,
}

/// <summary>
/// Port of <c>facefusion/processors/modules/background_remover/{core,types,choices}.py</c> —
/// runs an ONNX background-segmentation model to produce an alpha matte, then despills and
/// fills the "removed" area with a configurable colour.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5); plain static methods, no <c>IProcessor</c>
/// wiring yet</b> — same posture as <see cref="FaceEnhancer"/>/<see cref="FrameColorizer"/> at
/// the time this file was written. Every Python <c>state_manager.get_item(...)</c> call
/// (<c>background_remover_model</c>, <c>background_remover_fill_color</c>,
/// <c>background_remover_despill_color</c>) becomes an explicit parameter;
/// <see cref="ProcessFrame"/>'s return type is already <see cref="ProcessorOutputs"/>.
/// </para>
///
/// <para>
/// <b>Reduced-scope <c>pre_check</c>/model set</b> — same shape as every other module in this
/// port: checks only this module's own hash/source files on disk; does not download or verify
/// hashes.
/// </para>
///
/// <para>
/// <b>Mask-scale mismatch in <c>process_frame</c> (reproduced deliberately, per
/// PORT_CONVENTIONS.md rule 1 — flagged loudly, not silently "fixed").</b> Python's
/// <c>process_frame</c> ends with
/// <c>temp_vision_mask = numpy.minimum.reduce([temp_vision_mask, inputs.get('temp_vision_mask')])</c>,
/// where the first <c>temp_vision_mask</c> is <c>remove_background</c>'s own output — a
/// <c>uint8</c> mask scaled <c>0-255</c> (<c>normalize_vision_mask</c>'s <c>* 255</c>,
/// <c>astype(uint8)</c>) — combined elementwise against <c>inputs['temp_vision_mask']</c>, the
/// rest of this codebase's <c>Mask</c> convention (<c>FaceMasker</c>'s <c>CV_32FC1</c>, values
/// in <c>[0, 1]</c>, see its class remarks). Comparing a <c>0-255</c> array against a
/// <c>0-1</c> array with <c>numpy.minimum</c> means the incoming <c>[0, 1]</c> mask wins almost
/// everywhere the local mask is above 1.0 (i.e. almost everywhere it is not exactly 0) — this
/// looks like a real scale-mismatch bug already present in upstream FaceFusion, not something
/// introduced by this port, and is reproduced index-for-index in <see cref="ProcessFrame"/>
/// (upcasting the local <c>uint8</c> mask to <c>CV_32FC1</c> <i>without</i> rescaling to
/// <c>[0, 1]</c>, matching numpy's own dtype-only upcast on <c>numpy.minimum.reduce</c> of a
/// mixed-dtype list) rather than silently rescaled to "what was probably intended".
/// </para>
///
/// <para>
/// <b>VisionFrame / Mask representation</b> — same convention as every other ported module:
/// <see cref="Mat"/>, native memory, every returned <see cref="Mat"/> caller-owned, parameters
/// never disposed by the callee unless documented otherwise.
/// </para>
/// </summary>
public static class BackgroundRemover
{
    private const string ModuleName = "facefusion.processors.modules.background_remover.core";

    /// <summary>One entry of Python's <c>create_static_model_set('full')</c> — the fields this
    /// port needs (<c>type</c>, <c>size</c>, <c>mean</c>, <c>standard_deviation</c>,
    /// <c>hashes.background_remover</c>, <c>sources.background_remover</c>); the
    /// <c>__metadata__</c> vendor/license/year entries are download-manifest bookkeeping with no
    /// behavioural effect, same reduced scope as <see cref="FaceEnhancer.ModelOptions"/>.
    /// <paramref name="Mean"/>/<paramref name="Std"/> are in R, G, B channel order (applied
    /// after Python's BGR-&gt;RGB flip in <see cref="PrepareTempFrame"/>).</summary>
    public sealed record ModelOptions(
        BackgroundRemoverModelType Type, string BaseName, Resolution Size, float[] Mean, float[] Std, Download Hash, Download Source);

    private static readonly IReadOnlyList<BackgroundRemoverModel> AllModels = Enum.GetValues<BackgroundRemoverModel>();

    private static readonly IReadOnlyDictionary<BackgroundRemoverModel, string> ModelFileNames = new Dictionary<BackgroundRemoverModel, string>
    {
        [BackgroundRemoverModel.Ben2] = "ben_2",
        [BackgroundRemoverModel.BirefnetGeneral] = "birefnet_general",
        [BackgroundRemoverModel.BirefnetPortrait] = "birefnet_portrait",
        [BackgroundRemoverModel.CorridorKey1024] = "corridor_key_1024",
        [BackgroundRemoverModel.CorridorKey2048] = "corridor_key_2048",
        [BackgroundRemoverModel.IsnetGeneral] = "isnet_general",
        [BackgroundRemoverModel.Modnet] = "modnet",
        [BackgroundRemoverModel.Ormbg] = "ormbg",
        [BackgroundRemoverModel.Rmbg14] = "rmbg_1.4",
        [BackgroundRemoverModel.Rmbg20] = "rmbg_2.0",
        [BackgroundRemoverModel.Silueta] = "silueta",
        [BackgroundRemoverModel.U2netCloth] = "u2net_cloth",
        [BackgroundRemoverModel.U2netGeneral] = "u2net_general",
        [BackgroundRemoverModel.U2netHuman] = "u2net_human",
        [BackgroundRemoverModel.U2netp] = "u2netp",
    };

    // Python: create_static_model_set's resolve_download_url base_name argument — 'models-3.5.0'
    // for every model except the two corridor_key variants ('models-3.6.0').
    private static readonly IReadOnlyDictionary<BackgroundRemoverModel, string> ModelBaseNames = AllModels.ToDictionary(
        model => model,
        model => model is BackgroundRemoverModel.CorridorKey1024 or BackgroundRemoverModel.CorridorKey2048 ? "models-3.6.0" : "models-3.5.0");

    private static readonly IReadOnlyDictionary<BackgroundRemoverModel, BackgroundRemoverModelType> ModelTypes = new Dictionary<BackgroundRemoverModel, BackgroundRemoverModelType>
    {
        [BackgroundRemoverModel.Ben2] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.BirefnetGeneral] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.BirefnetPortrait] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.CorridorKey1024] = BackgroundRemoverModelType.CorridorKey,
        [BackgroundRemoverModel.CorridorKey2048] = BackgroundRemoverModelType.CorridorKey,
        [BackgroundRemoverModel.IsnetGeneral] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.Modnet] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.Ormbg] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.Rmbg14] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.Rmbg20] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.Silueta] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.U2netCloth] = BackgroundRemoverModelType.U2netCloth,
        [BackgroundRemoverModel.U2netGeneral] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.U2netHuman] = BackgroundRemoverModelType.Standard,
        [BackgroundRemoverModel.U2netp] = BackgroundRemoverModelType.Standard,
    };

    private static readonly IReadOnlyDictionary<BackgroundRemoverModel, Resolution> ModelSizes = new Dictionary<BackgroundRemoverModel, Resolution>
    {
        [BackgroundRemoverModel.Ben2] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.BirefnetGeneral] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.BirefnetPortrait] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.CorridorKey1024] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.CorridorKey2048] = new Resolution(2048, 2048),
        [BackgroundRemoverModel.IsnetGeneral] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.Modnet] = new Resolution(512, 512),
        [BackgroundRemoverModel.Ormbg] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.Rmbg14] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.Rmbg20] = new Resolution(1024, 1024),
        [BackgroundRemoverModel.Silueta] = new Resolution(320, 320),
        [BackgroundRemoverModel.U2netCloth] = new Resolution(768, 768),
        [BackgroundRemoverModel.U2netGeneral] = new Resolution(320, 320),
        [BackgroundRemoverModel.U2netHuman] = new Resolution(320, 320),
        [BackgroundRemoverModel.U2netp] = new Resolution(320, 320),
    };

    private static readonly float[] MeanZero = { 0.0f, 0.0f, 0.0f };
    private static readonly float[] StdOne = { 1.0f, 1.0f, 1.0f };
    private static readonly float[] MeanHalf = { 0.5f, 0.5f, 0.5f };
    private static readonly float[] StdHalf = { 0.5f, 0.5f, 0.5f };
    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    private static readonly IReadOnlyDictionary<BackgroundRemoverModel, (float[] Mean, float[] Std)> ModelNormalization = new Dictionary<BackgroundRemoverModel, (float[], float[])>
    {
        [BackgroundRemoverModel.Ben2] = (MeanZero, StdOne),
        [BackgroundRemoverModel.BirefnetGeneral] = (MeanZero, StdOne),
        [BackgroundRemoverModel.BirefnetPortrait] = (MeanZero, StdOne),
        [BackgroundRemoverModel.CorridorKey1024] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.CorridorKey2048] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.IsnetGeneral] = (MeanHalf, StdOne),
        [BackgroundRemoverModel.Modnet] = (MeanHalf, StdHalf),
        [BackgroundRemoverModel.Ormbg] = (MeanZero, StdOne),
        [BackgroundRemoverModel.Rmbg14] = (MeanHalf, StdOne),
        [BackgroundRemoverModel.Rmbg20] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.Silueta] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.U2netCloth] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.U2netGeneral] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.U2netHuman] = (ImageNetMean, ImageNetStd),
        [BackgroundRemoverModel.U2netp] = (ImageNetMean, ImageNetStd),
    };

    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>background_remover_models</c>.</summary>
    public static IReadOnlyList<BackgroundRemoverModel> BackgroundRemoverModels => AllModels;

    /// <summary>Python: <c>background_remover_color_range</c> (<c>create_int_range(0, 255, 1)</c>).</summary>
    public static readonly IReadOnlyList<int> BackgroundRemoverColorRange = CommonHelper.CreateIntRange(0, 255, 1);

    // -----------------------------------------------------------------
    // Model set / downloads / pre_check
    // -----------------------------------------------------------------

    /// <summary>Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>).</summary>
    public static IReadOnlyDictionary<BackgroundRemoverModel, ModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        var result = new Dictionary<BackgroundRemoverModel, ModelOptions>();

        foreach (var model in AllModels)
        {
            var fileName = ModelFileNames[model];
            var baseName = ModelBaseNames[model];
            var hash = new Download(
                BuildDownloadUrl(githubProvider, baseName, fileName + ".hash"),
                Path.Combine(modelsDirectory, fileName + ".hash"));
            var source = new Download(
                BuildDownloadUrl(githubProvider, baseName, fileName + ".onnx"),
                Path.Combine(modelsDirectory, fileName + ".onnx"));
            var (mean, std) = ModelNormalization[model];

            result[model] = new ModelOptions(ModelTypes[model], baseName, ModelSizes[model], mean, std, hash, source);
        }

        return result;
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    /// <summary>Same repo-root-walking approach as <see cref="FaceFusion.Face.FaceDetector"/>'s
    /// private helper of the same name.</summary>
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

    /// <summary>Python: <c>get_model_options</c>.</summary>
    public static ModelOptions GetModelOptions(BackgroundRemoverModel backgroundRemoverModel)
        => CreateStaticModelSet(DownloadScope.Full)[backgroundRemoverModel];

    /// <summary>Python: <c>pre_check</c>. See the class remarks — checks only this module's own
    /// hash/source files, not the <c>content_analyser</c> common-module pre-check (out of
    /// scope).</summary>
    public static bool PreCheck(BackgroundRemoverModel backgroundRemoverModel)
    {
        var options = GetModelOptions(backgroundRemoverModel);
        return FileSystem.IsFile(options.Hash.Path) && FileSystem.IsFile(options.Source.Path);
    }

    // -----------------------------------------------------------------
    // Inference pool
    // -----------------------------------------------------------------

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    public static IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        InferenceManager inferenceManager,
        BackgroundRemoverModel backgroundRemoverModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var options = GetModelOptions(backgroundRemoverModel);
        var modelSourceSet = new Dictionary<string, Download> { ["background_remover"] = options.Source };
        var modelNames = new[] { backgroundRemoverModel.ToWireName() };
        return inferenceManager.GetInferencePool(ModuleName, modelNames, modelSourceSet, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public static void ClearInferencePool(
        InferenceManager inferenceManager,
        BackgroundRemoverModel backgroundRemoverModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelNames = new[] { backgroundRemoverModel.ToWireName() };
        inferenceManager.ClearInferencePool(ModuleName, modelNames, executionDeviceIds, executionProviders);
    }

    // -----------------------------------------------------------------
    // prepare_temp_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_temp_frame</c>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>. Returns the flat <c>(1, C, modelSize.Height,
    /// modelSize.Width)</c> CHW float32 model input — <c>C = 4</c> (RGB + the coarse key-colour
    /// mask channel) for <see cref="BackgroundRemoverModelType.CorridorKey"/>, <c>C = 3</c>
    /// otherwise.
    /// </summary>
    public static float[] PrepareTempFrame(Mat tempVisionFrame, ModelOptions modelOptions)
    {
        var modelSize = modelOptions.Size;
        var plane = modelSize.Width * modelSize.Height;
        var channels = modelOptions.Type == BackgroundRemoverModelType.CorridorKey ? 4 : 3;

        float[]? coarseVisionMask = modelOptions.Type == BackgroundRemoverModelType.CorridorKey
            ? ComputeCoarseVisionMask(tempVisionFrame, modelSize)
            : null;

        using var resizedBgr = new Mat();
        Cv2.Resize(tempVisionFrame, resizedBgr, new Size(modelSize.Width, modelSize.Height));
        resizedBgr.GetArray(out Vec3b[] pixels);

        var chwData = new float[channels * plane];
        var mean = modelOptions.Mean;
        var std = modelOptions.Std;

        for (var i = 0; i < plane; i++)
        {
            var pixel = pixels[i];
            // Python: `temp_vision_frame[:, :, ::-1] / 255.0` (BGR -> RGB), then
            // `(x - model_mean) / model_standard_deviation` (mean/std in R, G, B order).
            var r = ((pixel.Item2 / 255f) - mean[0]) / std[0];
            var g = ((pixel.Item1 / 255f) - mean[1]) / std[1];
            var b = ((pixel.Item0 / 255f) - mean[2]) / std[2];

            chwData[i] = r;
            chwData[plane + i] = g;
            chwData[(2 * plane) + i] = b;
        }

        if (coarseVisionMask is not null)
        {
            Array.Copy(coarseVisionMask, 0, chwData, 3 * plane, plane);
        }

        return chwData;
    }

    /// <summary>
    /// Python: the <c>corridor_key</c> branch of <c>prepare_temp_frame</c> — computes
    /// <c>coarse_bias = G - max(R, B)</c> (a crude green-screen-style key detector) at the
    /// frame's <b>original</b> resolution, then resizes <c>1.0 - clip(coarse_bias * 2.0, 0, 1)</c>
    /// down to <paramref name="modelSize"/>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>. Returns the flat <c>modelSize.Height *
    /// modelSize.Width</c> single-channel mask.
    /// </summary>
    private static float[] ComputeCoarseVisionMask(Mat tempVisionFrame, Resolution modelSize)
    {
        tempVisionFrame.GetArray(out Vec3b[] pixels);
        var origHeight = tempVisionFrame.Rows;
        var origWidth = tempVisionFrame.Cols;
        var bias = new float[origHeight * origWidth];

        for (var i = 0; i < pixels.Length; i++)
        {
            var pixel = pixels[i];
            var r = pixel.Item2 / 255f;
            var g = pixel.Item1 / 255f;
            var b = pixel.Item0 / 255f;
            var coarseBias = g - MathF.Max(r, b);
            bias[i] = 1f - NumPy.Clip(coarseBias * 2f, 0f, 1f);
        }

        using var biasMat = new Mat(origHeight, origWidth, MatType.CV_32FC1);
        biasMat.SetArray(bias);

        using var resized = new Mat();
        Cv2.Resize(biasMat, resized, new Size(modelSize.Width, modelSize.Height));
        resized.GetArray(out float[] resizedFlat);
        return resizedFlat;
    }

    // -----------------------------------------------------------------
    // forward / forward_corridor_key
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward</c>. Does not take ownership of <paramref name="backgroundRemoverSession"/>.
    /// For <see cref="BackgroundRemoverModelType.U2netCloth"/>, reproduces
    /// <c>numpy.argmax(remove_vision_frame, axis = 1)</c> — the per-pixel class index across the
    /// output's channel axis, first-max-wins on ties (matching <c>numpy.argmax</c>) — collapsing
    /// to a single channel before returning. Every other model type returns the raw first output
    /// tensor's channel/height/width as read from its own shape.
    /// </summary>
    public static (float[] Data, int Channels, int Height, int Width) Forward(
        float[] chwInput, int inputChannels, int height, int width, InferenceSession backgroundRemoverSession, BackgroundRemoverModelType modelType)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(chwInput, new long[] { 1, inputChannels, height, width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = backgroundRemoverSession.Run(runOptions, inputs, backgroundRemoverSession.OutputNames);

        var outputShape = results[0].GetTensorTypeAndShape().Shape;
        var outputChannels = checked((int)outputShape[1]);
        var outputHeight = checked((int)outputShape[2]);
        var outputWidth = checked((int)outputShape[3]);
        var outputSpan = results[0].GetTensorDataAsSpan<float>();

        if (modelType == BackgroundRemoverModelType.U2netCloth)
        {
            var plane = outputHeight * outputWidth;
            var argmaxData = new float[plane];

            for (var i = 0; i < plane; i++)
            {
                var bestChannel = 0;
                var bestValue = outputSpan[i];

                for (var c = 1; c < outputChannels; c++)
                {
                    var value = outputSpan[(c * plane) + i];
                    if (value > bestValue)
                    {
                        bestValue = value;
                        bestChannel = c;
                    }
                }

                argmaxData[i] = bestChannel;
            }

            return (argmaxData, 1, outputHeight, outputWidth);
        }

        return (outputSpan.ToArray(), outputChannels, outputHeight, outputWidth);
    }

    /// <summary>
    /// Python: <c>forward_corridor_key</c>. Does not take ownership of
    /// <paramref name="backgroundRemoverSession"/>. Reads both of the session's two declared
    /// outputs (mask, then frame — matching Python's
    /// <c>remove_vision_mask, remove_vision_frame = background_remover.run(...)</c> unpacking
    /// order) by position rather than by name, since Python unpacks positionally too.
    /// </summary>
    public static (float[] MaskData, int MaskHeight, int MaskWidth, float[] FrameData, int FrameChannels, int FrameHeight, int FrameWidth) ForwardCorridorKey(
        float[] chwInput, Resolution modelSize, InferenceSession backgroundRemoverSession)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(chwInput, new long[] { 1, 4, modelSize.Height, modelSize.Width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = backgroundRemoverSession.Run(runOptions, inputs, backgroundRemoverSession.OutputNames);

        var maskShape = results[0].GetTensorTypeAndShape().Shape;
        var maskHeight = checked((int)maskShape[2]);
        var maskWidth = checked((int)maskShape[3]);
        var maskData = results[0].GetTensorDataAsSpan<float>().ToArray();

        var frameShape = results[1].GetTensorTypeAndShape().Shape;
        var frameChannels = checked((int)frameShape[1]);
        var frameHeight = checked((int)frameShape[2]);
        var frameWidth = checked((int)frameShape[3]);
        var frameData = results[1].GetTensorDataAsSpan<float>().ToArray();

        return (maskData, maskHeight, maskWidth, frameData, frameChannels, frameHeight, frameWidth);
    }

    // -----------------------------------------------------------------
    // normalize_vision_mask / apply_fill_color / apply_despill_color
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>normalize_vision_mask</c>: <c>squeeze().clip(0, 1) * 255</c>, then
    /// <c>clip(0, 255).astype(uint8)</c>. <paramref name="data"/> is already the squeezed
    /// single-channel <c>height * width</c> plane (this port's <see cref="Forward"/> never
    /// leaves a size-&gt;1 channel axis for it to squeeze away). Numpy's <c>astype(uint8)</c> is
    /// a truncating cast, not a rounding one; the input is already clipped to <c>[0, 255]</c>
    /// so truncation vs. rounding is the only observable difference from a saturating convert,
    /// and truncation is what is reproduced here. Caller owns the returned <see cref="Mat"/>
    /// (<c>CV_8UC1</c>).
    /// </summary>
    public static Mat NormalizeVisionMask(float[] data, int height, int width)
    {
        var bytes = new byte[height * width];

        for (var i = 0; i < bytes.Length; i++)
        {
            var value = NumPy.Clip(data[i], 0f, 1f) * 255f;
            bytes[i] = (byte)(int)value;
        }

        var mat = new Mat(height, width, MatType.CV_8UC1);
        mat.SetArray(bytes);
        return mat;
    }

    /// <summary>
    /// Python: <c>apply_fill_color</c>. Does not take ownership of either argument. Caller owns
    /// the returned <see cref="Mat"/> (<c>CV_8UC3</c>). Arithmetic is carried out in
    /// <see langword="double"/> to match numpy's own float64 promotion of a
    /// <c>uint8</c>/<c>float32</c> mix with Python <see langword="int"/> operands; the final
    /// <c>.astype(uint8)</c> truncates rather than rounds.
    /// </summary>
    public static Mat ApplyFillColor(Mat tempVisionFrame, Mat temp8UMask, Color fillColor)
    {
        tempVisionFrame.GetArray(out Vec3b[] framePixels);
        temp8UMask.GetArray(out byte[] maskPixels);

        var alphaFraction = fillColor.Alpha / 255.0;
        var outPixels = new Vec3b[framePixels.Length];

        for (var i = 0; i < framePixels.Length; i++)
        {
            var maskWeight = (1.0 - (maskPixels[i] / 255.0)) * alphaFraction;
            var pixel = framePixels[i];

            var b = (pixel.Item0 * (1.0 - maskWeight)) + (fillColor.Blue * maskWeight);
            var g = (pixel.Item1 * (1.0 - maskWeight)) + (fillColor.Green * maskWeight);
            var r = (pixel.Item2 * (1.0 - maskWeight)) + (fillColor.Red * maskWeight);

            outPixels[i] = new Vec3b { Item0 = (byte)(int)b, Item1 = (byte)(int)g, Item2 = (byte)(int)r };
        }

        var result = new Mat(tempVisionFrame.Rows, tempVisionFrame.Cols, MatType.CV_8UC3);
        result.SetArray(outPixels);
        return result;
    }

    /// <summary>
    /// Python: <c>apply_despill_color</c>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>. Caller owns the returned <see cref="Mat"/>
    /// (<c>CV_8UC3</c>).
    ///
    /// <para>
    /// <b>What the <c>numpy.roll(..., axis = 2)</c> pair actually computes.</b> Python rolls the
    /// channel axis by +1 and by -1 and sums the two rolls: for a BGR pixel, channel <c>c</c>'s
    /// sum ends up being the pixel's <i>other two</i> channels (roll by 1 brings channel
    /// <c>(c - 1) mod 3</c> to position <c>c</c>; roll by -1 brings channel <c>(c + 1) mod 3</c>)
    /// — i.e. <c>color_limit[B] = R + G</c>, <c>color_limit[G] = B + R</c>,
    /// <c>color_limit[R] = G + B</c>. This port writes that out directly per channel rather than
    /// reproducing a generic 3-channel roll, since the image is always BGR (3 channels) here —
    /// behaviourally identical, no roll/sum indirection needed to get the same three sums.
    /// </para>
    /// </summary>
    public static Mat ApplyDespillColor(Mat tempVisionFrame, Color despillColor)
    {
        tempVisionFrame.GetArray(out Vec3b[] pixels);

        var colorAlpha = despillColor.Alpha / 255.0;
        var denom = Math.Max(Math.Max(Math.Max(despillColor.Red, despillColor.Green), despillColor.Blue), 1);
        var weightB = despillColor.Blue / (double)denom;
        var weightG = despillColor.Green / (double)denom;
        var weightR = despillColor.Red / (double)denom;

        var outPixels = new Vec3b[pixels.Length];

        for (var i = 0; i < pixels.Length; i++)
        {
            var pixel = pixels[i];
            double b = pixel.Item0;
            double g = pixel.Item1;
            double r = pixel.Item2;

            var limitB = Math.Min(b, (r + g) * 0.5);
            var limitG = Math.Min(g, (b + r) * 0.5);
            var limitR = Math.Min(r, (g + b) * 0.5);

            var outB = b + ((limitB - b) * colorAlpha * weightB);
            var outG = g + ((limitG - g) * colorAlpha * weightG);
            var outR = r + ((limitR - r) * colorAlpha * weightR);

            outPixels[i] = new Vec3b { Item0 = (byte)(int)outB, Item1 = (byte)(int)outG, Item2 = (byte)(int)outR };
        }

        var result = new Mat(tempVisionFrame.Rows, tempVisionFrame.Cols, MatType.CV_8UC3);
        result.SetArray(outPixels);
        return result;
    }

    // -----------------------------------------------------------------
    // remove_background / process_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>remove_background</c>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>/<paramref name="backgroundRemoverSession"/>. Caller
    /// owns both fields of the returned tuple. For
    /// <see cref="BackgroundRemoverModelType.CorridorKey"/>, the model's own second output
    /// (a full-colour "keyed" image, not just a matte) replaces the working frame that
    /// despill/fill run against — Python: <c>remove_vision_frame[:, :, ::-1]</c> (a channel-order
    /// reversal — swaps channel 0 and channel 2 for a 3-channel image) resized back to the
    /// original frame size.
    /// </summary>
    public static (Mat ResultFrame, Mat ResultMask) RemoveBackground(
        Mat tempVisionFrame, ModelOptions modelOptions, InferenceSession backgroundRemoverSession, Color fillColor, Color despillColor)
    {
        var chwInput = PrepareTempFrame(tempVisionFrame, modelOptions);

        Mat workingFrame;
        bool ownsWorkingFrame;
        float[] maskData;
        int maskHeight;
        int maskWidth;

        if (modelOptions.Type == BackgroundRemoverModelType.CorridorKey)
        {
            var (mData, mHeight, mWidth, fData, fChannels, fHeight, fWidth) =
                ForwardCorridorKey(chwInput, modelOptions.Size, backgroundRemoverSession);
            maskData = mData;
            maskHeight = mHeight;
            maskWidth = mWidth;

            if (fChannels != 3)
            {
                throw new NotSupportedException(
                    $"BackgroundRemover.RemoveBackground: corridor_key frame output has {fChannels} channels; only the observed 3-channel case is implemented.");
            }

            // Python: `numpy.squeeze(remove_vision_frame).transpose(1, 2, 0)` (CHW -> HWC),
            // `numpy.clip(x * 255, 0, 255).astype(uint8)` (truncating cast, values already
            // clipped so this agrees with a saturating cast here), then
            // `remove_vision_frame[:, :, ::-1]` (channel-order reversal) before resizing back to
            // the original frame size.
            var hwcFrame = NumPy.TransposeChwToHwc(fData, fChannels, fHeight, fWidth);
            var plane = fHeight * fWidth;
            var swappedBytePixels = new Vec3b[plane];

            for (var i = 0; i < plane; i++)
            {
                var offset = i * 3;
                var c0 = (byte)(int)NumPy.Clip(hwcFrame[offset] * 255f, 0f, 255f);
                var c1 = (byte)(int)NumPy.Clip(hwcFrame[offset + 1] * 255f, 0f, 255f);
                var c2 = (byte)(int)NumPy.Clip(hwcFrame[offset + 2] * 255f, 0f, 255f);

                // `[:, :, ::-1]` reverses channel order: new[0] = old[2], new[2] = old[0].
                swappedBytePixels[i] = new Vec3b { Item0 = c2, Item1 = c1, Item2 = c0 };
            }

            using var keyedFrame = new Mat(fHeight, fWidth, MatType.CV_8UC3);
            keyedFrame.SetArray(swappedBytePixels);

            workingFrame = new Mat();
            Cv2.Resize(keyedFrame, workingFrame, new Size(tempVisionFrame.Cols, tempVisionFrame.Rows));
            ownsWorkingFrame = true;
        }
        else
        {
            var (mData, mChannels, mHeight, mWidth) = Forward(
                chwInput, modelOptions.Type == BackgroundRemoverModelType.CorridorKey ? 4 : 3,
                modelOptions.Size.Height, modelOptions.Size.Width, backgroundRemoverSession, modelOptions.Type);

            if (mChannels != 1)
            {
                throw new NotSupportedException(
                    $"BackgroundRemover.RemoveBackground: mask output has {mChannels} channels; expected 1 (the u2net_cloth argmax collapse already reduces to 1).");
            }

            maskData = mData;
            maskHeight = mHeight;
            maskWidth = mWidth;
            workingFrame = tempVisionFrame;
            ownsWorkingFrame = false;
        }

        try
        {
            using var rawMask = NormalizeVisionMask(maskData, maskHeight, maskWidth);
            var resizedMask = new Mat();
            Cv2.Resize(rawMask, resizedMask, new Size(workingFrame.Cols, workingFrame.Rows));

            try
            {
                using var despilledFrame = ApplyDespillColor(workingFrame, despillColor);
                var filledFrame = ApplyFillColor(despilledFrame, resizedMask, fillColor);
                return (filledFrame, resizedMask);
            }
            catch
            {
                resizedMask.Dispose();
                throw;
            }
        }
        finally
        {
            if (ownsWorkingFrame)
            {
                workingFrame.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>process_frame</c>. Does not take ownership of <paramref name="tempVisionFrame"/>
    /// or <paramref name="tempVisionMask"/>. Caller owns both fields of the returned
    /// <see cref="ProcessorOutputs"/>. See the class remarks for the deliberately-reproduced
    /// mask-scale mismatch this method's final <c>numpy.minimum.reduce</c> step carries over
    /// from Python.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(
        Mat tempVisionFrame, Mat tempVisionMask, ModelOptions modelOptions, InferenceSession backgroundRemoverSession, Color fillColor, Color despillColor)
    {
        var (resultFrame, resultMask) = RemoveBackground(tempVisionFrame, modelOptions, backgroundRemoverSession, fillColor, despillColor);

        using var _resultMask = resultMask;

        // Python: `numpy.minimum.reduce([temp_vision_mask, inputs.get('temp_vision_mask')])` —
        // `temp_vision_mask` here (resultMask) is uint8 in [0, 255]; `inputs['temp_vision_mask']`
        // is this codebase's usual CV_32FC1 mask in [0, 1] (see the class remarks). Upcast the
        // local mask to float32 *without* rescaling, matching numpy's dtype-only upcast of a
        // mixed uint8/float32 list, then take the elementwise minimum — reproducing the
        // documented scale mismatch rather than "fixing" it.
        using var resultMaskAsFloat = new Mat();
        resultMask.ConvertTo(resultMaskAsFloat, MatType.CV_32FC1);

        var combinedMask = new Mat();
        Cv2.Min(resultMaskAsFloat, tempVisionMask, combinedMask);

        return new ProcessorOutputs(resultFrame, combinedMask);
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: the <c>facefusion.processors.modules.background_remover.core</c> module's
    /// per-call inputs, extended per <see cref="IProcessorInputs"/>'s remarks — mirrors
    /// <c>FaceSwapper.FaceSwapperInputs</c>'s pattern. <c>background_remover</c> has no
    /// source-face concept, so this is a flat wrapper over <see cref="ProcessFrame"/>'s own
    /// parameters.
    /// </summary>
    public sealed record BackgroundRemoverInputs(
        Mat TempVisionFrame,
        Mat TempVisionMask,
        ModelOptions ModelOptions,
        InferenceSession BackgroundRemoverSession,
        Color FillColor,
        Color DespillColor) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/background_remover/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract. Thin orchestration over
    /// <see cref="ProcessFrame"/> — mirrors <c>FaceSwapper.Processor</c>'s shape.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "background_remover";

        /// <summary>Python: <c>get_common_modules()</c> (<c>[content_analyser]</c>).</summary>
        public IReadOnlyList<string> GetCommonModules() => new[] { "content_analyser" };

        /// <summary>
        /// Python: the <c>background_remover</c>-specific half of <c>pre_check</c>. The
        /// common-module half is the caller's responsibility per <see cref="GetCommonModules"/>'s
        /// remarks; this overload needs the chosen <paramref name="model"/> since the
        /// parameterless <see cref="IProcessor.PreCheck"/> member has no <c>state_manager</c> to
        /// read it from.
        /// </summary>
        public bool PreCheck(BackgroundRemoverModel model) => BackgroundRemover.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "background_remover.PreCheck requires a BackgroundRemoverModel (no state_manager to read it from — call the BackgroundRemoverModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Filesystem validation (<c>is_image</c>/<c>is_video</c>/
        /// <c>in_directory</c>/<c>same_file_extension</c>) is out of scope (same gap
        /// <c>FaceSwapper.Processor.PreProcess</c> documents); <c>background_remover</c> has no
        /// source-path requirement of its own, so with that validation unavailable there is
        /// nothing left to check.
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
            if (inputs is not BackgroundRemoverInputs backgroundRemoverInputs)
            {
                throw new ArgumentException($"expected {nameof(BackgroundRemoverInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return BackgroundRemover.ProcessFrame(
                backgroundRemoverInputs.TempVisionFrame,
                backgroundRemoverInputs.TempVisionMask,
                backgroundRemoverInputs.ModelOptions,
                backgroundRemoverInputs.BackgroundRemoverSession,
                backgroundRemoverInputs.FillColor,
                backgroundRemoverInputs.DespillColor);
        }

        /// <summary>
        /// Python: <c>post_process()</c>. Cache clearing is out of scope without a real pool
        /// owner to clear (rule 5), same as <c>FaceSwapper.Processor.PostProcess</c>.
        /// </summary>
        public void PostProcess()
        {
        }
    }
}
