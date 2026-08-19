using FaceFusion.Core;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/frame_enhancer/types.py</c>'s
/// <c>FrameEnhancerModel = Literal['clear_reality_x4', 'face_dat_x4', 'nomos8k_sc_x4',
/// 'real_esrgan_x2', 'real_esrgan_x2_fp16', 'real_esrgan_x4', 'real_esrgan_x4_fp16',
/// 'real_esrgan_x8', 'real_esrgan_x8_fp16', 'real_hatgan_x4', 'real_web_photo_x4',
/// 'realistic_rescaler_x4', 'remacri_x4', 'siax_x4', 'span_kendata_x4', 'swin2_sr_x4',
/// 'tghq_face_x8', 'ultra_sharp_x4', 'ultra_sharp_2_x4']</c>. Declared here rather than
/// <c>FaceFusion.Types</c> — see <see cref="FaceEnhancerModel"/>'s remarks for why.
/// </summary>
public enum FrameEnhancerModel
{
    [WireName("clear_reality_x4")]
    ClearRealityX4,

    [WireName("face_dat_x4")]
    FaceDatX4,

    [WireName("nomos8k_sc_x4")]
    Nomos8kScX4,

    [WireName("real_esrgan_x2")]
    RealEsrganX2,

    [WireName("real_esrgan_x2_fp16")]
    RealEsrganX2Fp16,

    [WireName("real_esrgan_x4")]
    RealEsrganX4,

    [WireName("real_esrgan_x4_fp16")]
    RealEsrganX4Fp16,

    [WireName("real_esrgan_x8")]
    RealEsrganX8,

    [WireName("real_esrgan_x8_fp16")]
    RealEsrganX8Fp16,

    [WireName("real_hatgan_x4")]
    RealHatganX4,

    [WireName("real_web_photo_x4")]
    RealWebPhotoX4,

    [WireName("realistic_rescaler_x4")]
    RealisticRescalerX4,

    [WireName("remacri_x4")]
    RemacriX4,

    [WireName("siax_x4")]
    SiaxX4,

    [WireName("span_kendata_x4")]
    SpanKendataX4,

    [WireName("swin2_sr_x4")]
    Swin2SrX4,

    [WireName("tghq_face_x8")]
    TghqFaceX8,

    [WireName("ultra_sharp_x4")]
    UltraSharpX4,

    [WireName("ultra_sharp_2_x4")]
    UltraSharp2X4,
}

/// <summary>
/// Python: one entry of <c>create_static_model_set('full')</c>'s <c>'precision'</c> field —
/// only <c>'fp16'</c> is ever used (as a marker for <see cref="FrameEnhancer.AdjustInferenceProviders"/>'s
/// CoreML MLProgram branch); every other model omits the key entirely (Python's
/// <c>.get('precision')</c> then returns <see langword="None"/>), represented here as
/// <see langword="null"/>.
/// </summary>
public enum FrameEnhancerPrecision
{
    [WireName("fp16")]
    Fp16,
}

/// <summary>
/// Port of <c>facefusion/processors/modules/frame_enhancer/{core,types,choices}.py</c> — runs a
/// tiled super-resolution ONNX model (Real-ESRGAN/ClearReality/SPAN/Swin2SR/… families) over an
/// entire frame (not just the face region) and blends the up-scaled result back over a
/// resized copy of the original.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Every Python
/// <c>state_manager.get_item(...)</c> call (<c>frame_enhancer_model</c>,
/// <c>frame_enhancer_blend</c>, <c>video_memory_strategy</c>) becomes an explicit parameter.
/// </para>
///
/// <para>
/// <b>Plain static methods, no <c>IProcessor</c> yet</b> — same situation and same reasoning as
/// <see cref="FaceEnhancer"/>: a sibling agent owns <c>IProcessor</c>/<c>ProcessorRegistry</c>
/// concurrently and neither existed when this file was written. <see cref="ProcessFrame"/>'s
/// return shape (<c>(Mat VisionFrame, Mat Mask)</c>) already matches Python's
/// <c>ProcessorOutputs</c>.
/// </para>
///
/// <para>
/// <b>Tiling reused, not re-derived.</b> <see cref="EnhanceFrame"/> calls
/// <see cref="FaceFusion.Vision.Vision.CreateTileFrames"/>/<c>MergeTileFrames</c> exactly as
/// Python's <c>enhance_frame</c> calls <c>vision.create_tile_frames</c>/<c>merge_tile_frames</c>
/// — both are already ported and tested; this file does not touch tiling math.
/// </para>
///
/// <para>
/// <b>Reduced-scope <c>pre_check</c>/model set</b> — same documented gap as
/// <see cref="FaceEnhancer"/>/<see cref="FaceFusion.Face.FaceDetector"/>: Python's
/// <c>pre_check</c> also calls <c>content_analyser.pre_check()</c>; that module's own
/// download/hash verification is out of this file's assignment, so <see cref="PreCheck"/>
/// here checks only the <c>frame_enhancer</c> model's own hash/source files.
/// </para>
///
/// <para>
/// <b>VisionFrame / Mask representation</b> — <see cref="Mat"/>, native memory, every returned
/// <see cref="Mat"/> caller-owned, parameters never disposed by the callee.
/// </para>
/// </summary>
public static class FrameEnhancer
{
    private const string ModuleName = "facefusion.processors.modules.frame_enhancer.core";

    /// <summary>One entry of Python's <c>create_static_model_set('full')</c> — the fields this
    /// port needs (<c>size</c> is really <c>(tile_size, pad_size, overlap_size)</c>, per
    /// <see cref="FaceFusion.Vision.Vision.CreateTileFrames"/>'s remarks on the same
    /// misleadingly-typed Python tuple; <c>scale</c> is the model's upscale factor;
    /// <c>precision</c> drives <see cref="FrameEnhancer.AdjustInferenceProviders"/>).</summary>
    public sealed record ModelOptions(
        (int TileSize, int PadSize, int OverlapSize) Size,
        int Scale,
        FrameEnhancerPrecision? Precision,
        Download Hash,
        Download Source);

    private static readonly IReadOnlyList<FrameEnhancerModel> AllModels = Enum.GetValues<FrameEnhancerModel>();

    private static readonly IReadOnlyDictionary<FrameEnhancerModel, string> ModelFileNames = new Dictionary<FrameEnhancerModel, string>
    {
        [FrameEnhancerModel.ClearRealityX4] = "clear_reality_x4",
        [FrameEnhancerModel.FaceDatX4] = "face_dat_x4",
        [FrameEnhancerModel.Nomos8kScX4] = "nomos8k_sc_x4",
        [FrameEnhancerModel.RealEsrganX2] = "real_esrgan_x2",
        [FrameEnhancerModel.RealEsrganX2Fp16] = "real_esrgan_x2_fp16",
        [FrameEnhancerModel.RealEsrganX4] = "real_esrgan_x4",
        [FrameEnhancerModel.RealEsrganX4Fp16] = "real_esrgan_x4_fp16",
        [FrameEnhancerModel.RealEsrganX8] = "real_esrgan_x8",
        [FrameEnhancerModel.RealEsrganX8Fp16] = "real_esrgan_x8_fp16",
        [FrameEnhancerModel.RealHatganX4] = "real_hatgan_x4",
        [FrameEnhancerModel.RealWebPhotoX4] = "real_web_photo_x4",
        [FrameEnhancerModel.RealisticRescalerX4] = "realistic_rescaler_x4",
        [FrameEnhancerModel.RemacriX4] = "remacri_x4",
        [FrameEnhancerModel.SiaxX4] = "siax_x4",
        [FrameEnhancerModel.SpanKendataX4] = "span_kendata_x4",
        [FrameEnhancerModel.Swin2SrX4] = "swin2_sr_x4",
        [FrameEnhancerModel.TghqFaceX8] = "tghq_face_x8",
        [FrameEnhancerModel.UltraSharpX4] = "ultra_sharp_x4",
        [FrameEnhancerModel.UltraSharp2X4] = "ultra_sharp_2_x4",
    };

    // Python: create_static_model_set's resolve_download_url base_name argument, per model.
    private static readonly IReadOnlyDictionary<FrameEnhancerModel, string> ModelBaseNames = new Dictionary<FrameEnhancerModel, string>
    {
        [FrameEnhancerModel.ClearRealityX4] = "models-3.0.0",
        [FrameEnhancerModel.FaceDatX4] = "models-3.5.0",
        [FrameEnhancerModel.Nomos8kScX4] = "models-3.0.0",
        [FrameEnhancerModel.RealEsrganX2] = "models-3.0.0",
        [FrameEnhancerModel.RealEsrganX2Fp16] = "models-3.0.0",
        [FrameEnhancerModel.RealEsrganX4] = "models-3.0.0",
        [FrameEnhancerModel.RealEsrganX4Fp16] = "models-3.0.0",
        [FrameEnhancerModel.RealEsrganX8] = "models-3.0.0",
        [FrameEnhancerModel.RealEsrganX8Fp16] = "models-3.0.0",
        [FrameEnhancerModel.RealHatganX4] = "models-3.0.0",
        [FrameEnhancerModel.RealWebPhotoX4] = "models-3.1.0",
        [FrameEnhancerModel.RealisticRescalerX4] = "models-3.1.0",
        [FrameEnhancerModel.RemacriX4] = "models-3.1.0",
        [FrameEnhancerModel.SiaxX4] = "models-3.1.0",
        [FrameEnhancerModel.SpanKendataX4] = "models-3.0.0",
        [FrameEnhancerModel.Swin2SrX4] = "models-3.1.0",
        [FrameEnhancerModel.TghqFaceX8] = "models-3.5.0",
        [FrameEnhancerModel.UltraSharpX4] = "models-3.0.0",
        [FrameEnhancerModel.UltraSharp2X4] = "models-3.3.0",
    };

    private static readonly IReadOnlyDictionary<FrameEnhancerModel, (int TileSize, int PadSize, int OverlapSize)> ModelSizes =
        new Dictionary<FrameEnhancerModel, (int, int, int)>
        {
            [FrameEnhancerModel.ClearRealityX4] = (128, 8, 4),
            [FrameEnhancerModel.FaceDatX4] = (128, 8, 4),
            [FrameEnhancerModel.Nomos8kScX4] = (128, 8, 4),
            [FrameEnhancerModel.RealEsrganX2] = (256, 16, 8),
            [FrameEnhancerModel.RealEsrganX2Fp16] = (256, 16, 8),
            [FrameEnhancerModel.RealEsrganX4] = (256, 16, 8),
            [FrameEnhancerModel.RealEsrganX4Fp16] = (256, 16, 8),
            [FrameEnhancerModel.RealEsrganX8] = (256, 16, 8),
            [FrameEnhancerModel.RealEsrganX8Fp16] = (256, 16, 8),
            [FrameEnhancerModel.RealHatganX4] = (256, 16, 8),
            [FrameEnhancerModel.RealWebPhotoX4] = (64, 4, 2),
            [FrameEnhancerModel.RealisticRescalerX4] = (128, 8, 4),
            [FrameEnhancerModel.RemacriX4] = (128, 8, 4),
            [FrameEnhancerModel.SiaxX4] = (128, 8, 4),
            [FrameEnhancerModel.SpanKendataX4] = (128, 8, 4),
            [FrameEnhancerModel.Swin2SrX4] = (128, 8, 4),
            [FrameEnhancerModel.TghqFaceX8] = (128, 8, 4),
            [FrameEnhancerModel.UltraSharpX4] = (128, 8, 4),
            [FrameEnhancerModel.UltraSharp2X4] = (1024, 64, 32),
        };

    private static readonly IReadOnlyDictionary<FrameEnhancerModel, int> ModelScales = new Dictionary<FrameEnhancerModel, int>
    {
        [FrameEnhancerModel.ClearRealityX4] = 4,
        [FrameEnhancerModel.FaceDatX4] = 4,
        [FrameEnhancerModel.Nomos8kScX4] = 4,
        [FrameEnhancerModel.RealEsrganX2] = 2,
        [FrameEnhancerModel.RealEsrganX2Fp16] = 2,
        [FrameEnhancerModel.RealEsrganX4] = 4,
        [FrameEnhancerModel.RealEsrganX4Fp16] = 4,
        [FrameEnhancerModel.RealEsrganX8] = 8,
        [FrameEnhancerModel.RealEsrganX8Fp16] = 8,
        [FrameEnhancerModel.RealHatganX4] = 4,
        [FrameEnhancerModel.RealWebPhotoX4] = 4,
        [FrameEnhancerModel.RealisticRescalerX4] = 4,
        [FrameEnhancerModel.RemacriX4] = 4,
        [FrameEnhancerModel.SiaxX4] = 4,
        [FrameEnhancerModel.SpanKendataX4] = 4,
        [FrameEnhancerModel.Swin2SrX4] = 4,
        [FrameEnhancerModel.TghqFaceX8] = 8,
        [FrameEnhancerModel.UltraSharpX4] = 4,
        [FrameEnhancerModel.UltraSharp2X4] = 4,
    };

    private static readonly IReadOnlyDictionary<FrameEnhancerModel, FrameEnhancerPrecision?> ModelPrecisions = new Dictionary<FrameEnhancerModel, FrameEnhancerPrecision?>
    {
        [FrameEnhancerModel.RealEsrganX2Fp16] = FrameEnhancerPrecision.Fp16,
        [FrameEnhancerModel.RealEsrganX4Fp16] = FrameEnhancerPrecision.Fp16,
        [FrameEnhancerModel.RealEsrganX8Fp16] = FrameEnhancerPrecision.Fp16,
    };

    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>frame_enhancer_models</c> (<c>list(get_args(FrameEnhancerModel))</c>).</summary>
    public static IReadOnlyList<FrameEnhancerModel> FrameEnhancerModels => AllModels;

    /// <summary>Python: <c>frame_enhancer_blend_range</c> (<c>create_int_range(0, 100, 1)</c>).</summary>
    public static readonly IReadOnlyList<int> FrameEnhancerBlendRange = CommonHelper.CreateIntRange(0, 100, 1);

    // -----------------------------------------------------------------
    // Model set / downloads / pre_check
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). <paramref name="downloadScope"/>
    /// is accepted for signature parity with Python (unused — same as every other ported
    /// <c>create_static_model_set</c> in this port).
    /// </summary>
    public static IReadOnlyDictionary<FrameEnhancerModel, ModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        var result = new Dictionary<FrameEnhancerModel, ModelOptions>();

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

            result[model] = new ModelOptions(ModelSizes[model], ModelScales[model], ModelPrecisions.GetValueOrDefault(model), hash, source);
        }

        return result;
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    /// <summary>Same repo-root-walking approach as <see cref="FaceEnhancer"/>'s private helper
    /// of the same name.</summary>
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
    public static ModelOptions GetModelOptions(FrameEnhancerModel frameEnhancerModel)
        => CreateStaticModelSet(DownloadScope.Full)[frameEnhancerModel];

    /// <summary>
    /// Python: <c>pre_check</c>. See the class remarks — checks only the selected
    /// frame_enhancer model's own hash/source files, not <c>content_analyser.pre_check()</c>
    /// (out of scope; documented reduced scope matching every other ported module).
    /// </summary>
    public static bool PreCheck(FrameEnhancerModel frameEnhancerModel)
    {
        var options = GetModelOptions(frameEnhancerModel);
        return FileSystem.IsFile(options.Hash.Path) && FileSystem.IsFile(options.Source.Path);
    }

    // -----------------------------------------------------------------
    // Inference pool (thin wrappers around FaceFusion.Inference.InferenceManager)
    // -----------------------------------------------------------------

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    public static IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        InferenceManager inferenceManager,
        FrameEnhancerModel frameEnhancerModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var options = GetModelOptions(frameEnhancerModel);
        var modelSourceSet = new Dictionary<string, Download> { ["frame_enhancer"] = options.Source };
        var modelNames = new[] { frameEnhancerModel.ToWireName() };
        var adjustInferenceProviders = () => AdjustInferenceProviders(frameEnhancerModel);
        return inferenceManager.GetInferencePool(ModuleName, modelNames, modelSourceSet, executionDeviceIds, executionProviders, adjustInferenceProviders: adjustInferenceProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public static void ClearInferencePool(
        InferenceManager inferenceManager,
        FrameEnhancerModel frameEnhancerModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelNames = new[] { frameEnhancerModel.ToWireName() };
        inferenceManager.ClearInferencePool(ModuleName, modelNames, executionDeviceIds, executionProviders);
    }

    /// <summary>
    /// Python: <c>adjust_inference_providers</c>. Only ever returns a non-empty list on macOS
    /// with the CoreML execution provider active and an <c>fp16</c> model selected.
    /// </summary>
    public static IReadOnlyList<InferenceProviderEntry> AdjustInferenceProviders(FrameEnhancerModel frameEnhancerModel)
    {
        var precision = ModelPrecisions.GetValueOrDefault(frameEnhancerModel);

        if (CommonHelper.IsMacOS() && Execution.HasExecutionProvider(ExecutionProvider.Coreml) && precision == FrameEnhancerPrecision.Fp16)
        {
            return new[]
            {
                new InferenceProviderEntry(
                    Choices.ExecutionProviderSet[ExecutionProvider.Coreml].ToWireName(),
                    new Dictionary<string, object?> { ["ModelFormat"] = "MLProgram" }),
            };
        }

        return Array.Empty<InferenceProviderEntry>();
    }

    // -----------------------------------------------------------------
    // enhance_frame / forward / prepare / normalize / blend
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>enhance_frame</c>. Does not take ownership of <paramref name="tempVisionFrame"/>
    /// or <paramref name="frameEnhancerSession"/>. Caller owns the returned <see cref="Mat"/>.
    /// </summary>
    public static Mat EnhanceFrame(Mat tempVisionFrame, ModelOptions modelOptions, InferenceSession frameEnhancerSession, double frameEnhancerBlend)
    {
        var tempHeight = tempVisionFrame.Rows;
        var tempWidth = tempVisionFrame.Cols;
        var (tileVisionFrames, padWidth, padHeight) = VisionHelper.CreateTileFrames(tempVisionFrame, modelOptions.Size);

        try
        {
            var normalizedTiles = new List<Mat>(tileVisionFrames.Count);
            foreach (var tile in tileVisionFrames)
            {
                var (chwData, tileHeight, tileWidth) = PrepareTileFrame(tile);
                var outputData = Forward(chwData, tileHeight, tileWidth, frameEnhancerSession, out var outputHeight, out var outputWidth);
                normalizedTiles.Add(NormalizeTileFrame(outputData, outputHeight, outputWidth));
            }

            try
            {
                var scaledSize = (modelOptions.Size.TileSize * modelOptions.Scale, modelOptions.Size.PadSize * modelOptions.Scale, modelOptions.Size.OverlapSize * modelOptions.Scale);
                using var mergeVisionFrame = VisionHelper.MergeTileFrames(
                    normalizedTiles, tempWidth * modelOptions.Scale, tempHeight * modelOptions.Scale,
                    padWidth * modelOptions.Scale, padHeight * modelOptions.Scale, scaledSize);

                return BlendMergeFrame(tempVisionFrame, mergeVisionFrame, frameEnhancerBlend);
            }
            finally
            {
                foreach (var tile in normalizedTiles)
                {
                    tile.Dispose();
                }
            }
        }
        finally
        {
            foreach (var tile in tileVisionFrames)
            {
                tile.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>forward</c>. Does not take ownership of <paramref name="frameEnhancerSession"/>.
    /// Reads the model's own output shape (rather than assuming <paramref name="tileHeight"/>/
    /// <paramref name="tileWidth"/> scaled by a fixed factor) since that is exactly what
    /// Python's untyped <c>.run(...)[0]</c> return does — the scale is implied by whatever the
    /// ONNX graph actually produces.
    /// </summary>
    public static float[] Forward(float[] chwData, int tileHeight, int tileWidth, InferenceSession frameEnhancerSession, out int outputHeight, out int outputWidth)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(chwData, new long[] { 1, 3, tileHeight, tileWidth });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = frameEnhancerSession.Run(runOptions, inputs, frameEnhancerSession.OutputNames);

        // Python: `frame_enhancer.run(None, { 'input': tile_vision_frame })[0]` — first output
        // tensor, batch dimension still present (squeezed later in normalize_tile_frame),
        // shape (1, 3, outH, outW).
        var shape = results[0].GetTensorTypeAndShape().Shape;
        outputHeight = checked((int)shape[2]);
        outputWidth = checked((int)shape[3]);
        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>
    /// Python: <c>prepare_tile_frame</c>. <c>tile_vision_frame[:, :, ::-1]</c> (BGR-&gt;RGB),
    /// <c>expand_dims(axis = 0)</c>, <c>transpose(0, 3, 1, 2)</c> (NHWC-&gt;NCHW),
    /// <c>astype(float32) / 255.0</c>. Does not take ownership of
    /// <paramref name="tileVisionFrame"/>. Returns the flat CHW float32 data (batch dimension
    /// implicit) plus its height/width.
    /// </summary>
    public static (float[] ChwData, int Height, int Width) PrepareTileFrame(Mat tileVisionFrame)
    {
        var height = tileVisionFrame.Rows;
        var width = tileVisionFrame.Cols;
        var plane = height * width;

        tileVisionFrame.GetArray(out Vec3b[] pixels);
        var hwcRgb = new float[plane * 3];

        for (var i = 0; i < plane; i++)
        {
            var pixel = pixels[i];
            var offset = i * 3;
            // BGR -> RGB (Python's `[:, :, ::-1]`), then /255.0 (order matches Python: the
            // channel reversal happens before the float cast/divide).
            hwcRgb[offset] = pixel.Item2 / 255f;
            hwcRgb[offset + 1] = pixel.Item1 / 255f;
            hwcRgb[offset + 2] = pixel.Item0 / 255f;
        }

        var chwData = NumPy.TransposeHwcToChw(hwcRgb, height, width, 3);
        return (chwData, height, width);
    }

    /// <summary>
    /// Python: <c>normalize_tile_frame</c>. <c>transpose(0, 2, 3, 1).squeeze(0) * 255</c>
    /// (NCHW-&gt;NHWC, drop batch dim, scale), <c>.clip(0, 255).astype(uint8)</c>, then
    /// <c>[:, :, ::-1]</c> (RGB-&gt;BGR). <paramref name="chwData"/> is the flat CHW output of
    /// <see cref="Forward"/> (batch dimension already squeezed away since it is always 1).
    /// Caller owns the returned <see cref="Mat"/> (<c>CV_8UC3</c>, BGR).
    /// </summary>
    public static Mat NormalizeTileFrame(ReadOnlySpan<float> chwData, int height, int width)
    {
        var hwcRgb = NumPy.TransposeChwToHwc(chwData, 3, height, width);
        var plane = height * width;
        var pixels = new Vec3b[plane];

        for (var i = 0; i < plane; i++)
        {
            var offset = i * 3;
            var r = ScaleAndClipChannel(hwcRgb[offset]);
            var g = ScaleAndClipChannel(hwcRgb[offset + 1]);
            var b = ScaleAndClipChannel(hwcRgb[offset + 2]);

            // RGB -> BGR (Python's trailing `[:, :, ::-1]`).
            pixels[i] = new Vec3b { Item0 = b, Item1 = g, Item2 = r };
        }

        var result = new Mat(height, width, MatType.CV_8UC3);
        result.SetArray(pixels);
        return result;
    }

    /// <summary>Python: <c>(value * 255).clip(0, 255).astype(uint8)</c> for a single channel
    /// value. No <c>.round()</c> call in the Python here (unlike face_enhancer's
    /// <c>normalize_crop_frame</c>) — <c>.astype(uint8)</c> truncates toward zero directly, so
    /// this truncates rather than rounds, deliberately reproducing that asymmetry between the
    /// two sibling processors.</summary>
    private static byte ScaleAndClipChannel(float value)
    {
        var scaled = value * 255f;
        var clipped = Math.Clamp(scaled, 0f, 255f);
        return (byte)clipped; // truncation toward zero, matching numpy .astype(uint8)
    }

    /// <summary>
    /// Python: <c>blend_merge_frame</c>. Does not take ownership of either argument. Caller
    /// owns the returned <see cref="Mat"/>.
    /// </summary>
    public static Mat BlendMergeFrame(Mat tempVisionFrame, Mat mergeVisionFrame, double frameEnhancerBlend)
    {
        var blendFactor = 1 - (frameEnhancerBlend / 100.0);
        using var resizedTempVisionFrame = new Mat();
        Cv2.Resize(tempVisionFrame, resizedTempVisionFrame, new Size(mergeVisionFrame.Cols, mergeVisionFrame.Rows));
        return VisionHelper.BlendFrame(resizedTempVisionFrame, mergeVisionFrame, 1 - blendFactor);
    }

    // -----------------------------------------------------------------
    // process_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Does not take ownership of either <see cref="Mat"/>
    /// parameter. Caller owns both returned <see cref="Mat"/>s.
    /// </summary>
    public static (Mat VisionFrame, Mat Mask) ProcessFrame(
        Mat tempVisionFrame, Mat tempVisionMask, ModelOptions modelOptions, InferenceSession frameEnhancerSession, double frameEnhancerBlend)
    {
        var enhancedVisionFrame = EnhanceFrame(tempVisionFrame, modelOptions, frameEnhancerSession, frameEnhancerBlend);

        var resizedMask = new Mat();
        Cv2.Resize(tempVisionMask, resizedMask, new Size(enhancedVisionFrame.Cols, enhancedVisionFrame.Rows));

        return (enhancedVisionFrame, resizedMask);
    }

    // -----------------------------------------------------------------
    // IProcessor adapter
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>frame_enhancer/core.py</c>'s <c>process_frame</c> inputs. Widened beyond
    /// Python's <c>TypedDict</c> with the settings the module would have read off
    /// <c>state_manager</c> — see <see cref="IProcessorInputs"/>'s remarks.
    /// </summary>
    public sealed record FrameEnhancerInputs(
        Mat TempVisionFrame,
        Mat TempVisionMask,
        ModelOptions ModelOptions,
        InferenceSession FrameEnhancerSession,
        double FrameEnhancerBlend) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/frame_enhancer/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract — see
    /// <c>FrameColorizer.Processor</c> for the same pattern (both are frame-only processors
    /// with no face pipeline).
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "frame_enhancer";

        /// <summary>Python: <c>get_common_modules()</c> (<c>[content_analyser]</c>).</summary>
        public IReadOnlyList<string> GetCommonModules() => new[] { "content_analyser" };

        /// <summary>Python: the <c>frame_enhancer</c>-specific half of <c>pre_check</c>. Takes
        /// the chosen model because the parameterless <see cref="IProcessor.PreCheck"/> member
        /// has no <c>state_manager</c> to read it from.</summary>
        public bool PreCheck(FrameEnhancerModel model) => FrameEnhancer.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "frame_enhancer.PreCheck requires a FrameEnhancerModel (no state_manager to read it from — call the FrameEnhancerModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Filesystem validation is out of scope (the same gap
        /// <c>FrameColorizer.Processor.PreProcess</c> documents); <c>frame_enhancer</c> has no
        /// source-path requirement of its own, so nothing else remains to check.
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
            if (inputs is not FrameEnhancerInputs frameEnhancerInputs)
            {
                throw new ArgumentException($"expected {nameof(FrameEnhancerInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            var (visionFrame, mask) = FrameEnhancer.ProcessFrame(
                frameEnhancerInputs.TempVisionFrame,
                frameEnhancerInputs.TempVisionMask,
                frameEnhancerInputs.ModelOptions,
                frameEnhancerInputs.FrameEnhancerSession,
                frameEnhancerInputs.FrameEnhancerBlend);

            return new ProcessorOutputs(visionFrame, mask);
        }

        /// <summary>Python: <c>post_process()</c>. Cache clearing is out of scope without a real
        /// pool owner to clear (rule 5), same as every other processor here.</summary>
        public void PostProcess()
        {
        }
    }
}
