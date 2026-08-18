using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/face_enhancer/types.py</c>'s
/// <c>FaceEnhancerModel = Literal['codeformer', 'gfpgan_1.2', 'gfpgan_1.3', 'gfpgan_1.4',
/// 'gpen_bfr_256', 'gpen_bfr_512', 'gpen_bfr_1024', 'gpen_bfr_2048',
/// 'restoreformer_plus_plus']</c>. Declared in this file (rather than
/// <c>FaceFusion.Types</c>) per the assignment's file-scope constraint — only
/// <c>FaceEnhancer.cs</c>/<c>FrameEnhancer.cs</c> are this agent's to touch, and
/// <see cref="FaceFusion.Types.EnumNames"/>'s <c>[WireName]</c> convention works for any enum
/// regardless of which assembly declares it.
/// </summary>
public enum FaceEnhancerModel
{
    [WireName("codeformer")]
    Codeformer,

    [WireName("gfpgan_1.2")]
    Gfpgan12,

    [WireName("gfpgan_1.3")]
    Gfpgan13,

    [WireName("gfpgan_1.4")]
    Gfpgan14,

    [WireName("gpen_bfr_256")]
    GpenBfr256,

    [WireName("gpen_bfr_512")]
    GpenBfr512,

    [WireName("gpen_bfr_1024")]
    GpenBfr1024,

    [WireName("gpen_bfr_2048")]
    GpenBfr2048,

    [WireName("restoreformer_plus_plus")]
    RestoreformerPlusPlus,
}

/// <summary>
/// Port of <c>facefusion/processors/modules/face_enhancer/{core,types,choices}.py</c> — restores/
/// sharpens the face region of a frame by warping each selected face into a model-specific
/// canonical crop, running a face-restoration ONNX model (CodeFormer/GFPGAN/GPEN/RestoreFormer++)
/// over it, and pasting the result back through a blended mask.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Every Python
/// <c>state_manager.get_item(...)</c> call (<c>face_enhancer_model</c>,
/// <c>face_enhancer_blend</c>, <c>face_enhancer_weight</c>, <c>face_mask_blur</c>,
/// <c>face_mask_types</c>, <c>face_selector_*</c>, <c>reference_face_*</c>,
/// <c>face_tracker_score</c>) becomes an explicit parameter on <see cref="EnhanceFace"/> /
/// <see cref="ProcessFrame"/> instead of an ambient lookup.
/// </para>
///
/// <para>
/// <b>Plain static methods, no <c>IProcessor</c> yet (per the assignment brief).</b> A sibling
/// agent is concurrently defining <c>IProcessor</c>/<c>ProcessorRegistry</c> in this same
/// project; at the time this file was written neither existed, so this class exposes the
/// model-set/preprocess/inference/blend logic as plain static methods (mirroring
/// <c>FaceFusion.Face.FaceDetector</c>'s shape) rather than inventing a competing interface.
/// <see cref="ProcessFrame"/>'s signature/return shape (<c>(Mat VisionFrame, Mat Mask)</c>)
/// already matches Python's <c>ProcessorOutputs = Tuple[VisionFrame, Mask]</c>
/// (<c>facefusion/processors/types.py</c>) so wiring it into <c>IProcessor.ProcessFrame</c>
/// later should be a thin adapter, not a rewrite.
/// </para>
///
/// <para>
/// <b>Reduced-scope <c>pre_check</c>/model set (documented, consistent with
/// <see cref="FaceFusion.Face.FaceDetector"/> and <see cref="FaceFusion.Face.FaceMasker"/>).</b>
/// Python's <c>pre_check</c> also calls <c>pre_check()</c> on six "common modules"
/// (content_analyser, face_classifier, face_detector, face_landmarker, face_masker,
/// face_recognizer) before checking its own model files. Those modules' own download/hash
/// verification is out of this file's assignment (download.py/hash_helper.py are not ported
/// anywhere in this repo yet — see <see cref="FaceDetector"/>'s remarks for the same gap), so
/// <see cref="PreCheck"/> here checks only the <c>face_enhancer</c> model's own hash/source
/// files, matching the reduced scope already established by every other ported module that
/// touches <c>pre_check</c>.
/// </para>
///
/// <para>
/// <b>Occluder/mask machinery is a caller-supplied dependency, not reproduced here.</b>
/// <c>enhance_face</c> calls <c>face_masker.create_box_mask</c>/<c>create_occlusion_mask</c>,
/// which are already ported in <see cref="FaceFusion.Face.FaceMasker"/> — <see cref="EnhanceFace"/>
/// calls those directly rather than re-implementing mask construction. Similarly
/// <c>face_selector.select_faces</c> and <c>face_creator.scale_face</c> are reused from
/// <see cref="FaceFusion.Face.FaceSelector"/>/<see cref="FaceFusion.Face.FaceCreator"/> exactly
/// as Python's <c>process_frame</c> calls them.
/// </para>
///
/// <para>
/// <b>VisionFrame / Mask representation</b> — same convention as every other ported module in
/// this port: <see cref="Mat"/>, native memory, every returned <see cref="Mat"/> caller-owned,
/// parameters never disposed by the callee unless explicitly documented otherwise.
/// </para>
/// </summary>
public static class FaceEnhancer
{
    /// <summary>Python: the <c>__name__</c> module-path string <c>get_inference_pool</c>/
    /// <c>clear_inference_pool</c> pass to <c>inference_manager</c> as the pool's cache key.</summary>
    private const string ModuleName = "facefusion.processors.modules.face_enhancer.core";

    /// <summary>One entry of Python's <c>create_static_model_set('full')</c> — the fields this
    /// port's <see cref="EnhanceFace"/>/<see cref="PreCheck"/> actually need (<c>template</c>,
    /// <c>size</c>, <c>hashes.face_enhancer</c>, <c>sources.face_enhancer</c>); the
    /// <c>__metadata__</c> vendor/license/year entries are download-manifest bookkeeping with
    /// no behavioural effect and are not reproduced, matching <see cref="FaceMasker"/>'s
    /// documented reduced scope for the same shape of dictionary.</summary>
    public sealed record ModelOptions(WarpTemplate Template, Size Size, Download Hash, Download Source);

    private static readonly IReadOnlyList<FaceEnhancerModel> AllModels = Enum.GetValues<FaceEnhancerModel>();

    private static readonly IReadOnlyDictionary<FaceEnhancerModel, string> ModelFileNames = new Dictionary<FaceEnhancerModel, string>
    {
        [FaceEnhancerModel.Codeformer] = "codeformer",
        [FaceEnhancerModel.Gfpgan12] = "gfpgan_1.2",
        [FaceEnhancerModel.Gfpgan13] = "gfpgan_1.3",
        [FaceEnhancerModel.Gfpgan14] = "gfpgan_1.4",
        [FaceEnhancerModel.GpenBfr256] = "gpen_bfr_256",
        [FaceEnhancerModel.GpenBfr512] = "gpen_bfr_512",
        [FaceEnhancerModel.GpenBfr1024] = "gpen_bfr_1024",
        [FaceEnhancerModel.GpenBfr2048] = "gpen_bfr_2048",
        [FaceEnhancerModel.RestoreformerPlusPlus] = "restoreformer_plus_plus",
    };

    // Python: create_static_model_set's resolve_download_url base_name argument — every
    // face_enhancer model resolves against 'models-3.0.0'.
    private const string ModelBaseName = "models-3.0.0";

    private static readonly IReadOnlyDictionary<FaceEnhancerModel, WarpTemplate> ModelTemplates = new Dictionary<FaceEnhancerModel, WarpTemplate>
    {
        [FaceEnhancerModel.Codeformer] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.Gfpgan12] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.Gfpgan13] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.Gfpgan14] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.GpenBfr256] = WarpTemplate.Arcface128,
        [FaceEnhancerModel.GpenBfr512] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.GpenBfr1024] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.GpenBfr2048] = WarpTemplate.Ffhq512,
        [FaceEnhancerModel.RestoreformerPlusPlus] = WarpTemplate.Ffhq512,
    };

    private static readonly IReadOnlyDictionary<FaceEnhancerModel, Size> ModelSizes = new Dictionary<FaceEnhancerModel, Size>
    {
        [FaceEnhancerModel.Codeformer] = new Size(512, 512),
        [FaceEnhancerModel.Gfpgan12] = new Size(512, 512),
        [FaceEnhancerModel.Gfpgan13] = new Size(512, 512),
        [FaceEnhancerModel.Gfpgan14] = new Size(512, 512),
        [FaceEnhancerModel.GpenBfr256] = new Size(256, 256),
        [FaceEnhancerModel.GpenBfr512] = new Size(512, 512),
        [FaceEnhancerModel.GpenBfr1024] = new Size(1024, 1024),
        [FaceEnhancerModel.GpenBfr2048] = new Size(2048, 2048),
        [FaceEnhancerModel.RestoreformerPlusPlus] = new Size(512, 512),
    };

    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>face_enhancer_models</c> (<c>list(get_args(FaceEnhancerModel))</c>).</summary>
    public static IReadOnlyList<FaceEnhancerModel> FaceEnhancerModels => AllModels;

    /// <summary>Python: <c>face_enhancer_blend_range</c> (<c>create_int_range(0, 100, 1)</c>).</summary>
    public static readonly IReadOnlyList<int> FaceEnhancerBlendRange = CommonHelper.CreateIntRange(0, 100, 1);

    /// <summary>Python: <c>face_enhancer_weight_range</c> (<c>create_float_range(0.0, 1.0, 0.05)</c>).</summary>
    public static readonly IReadOnlyList<double> FaceEnhancerWeightRange = CommonHelper.CreateFloatRange(0.0, 1.0, 0.05);

    // -----------------------------------------------------------------
    // Model set / downloads / pre_check
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). <paramref name="downloadScope"/>
    /// is accepted for signature parity with Python — the dict body does not vary by scope
    /// (same as <see cref="FaceDetector.CreateStaticModelSet"/>).
    /// </summary>
    public static IReadOnlyDictionary<FaceEnhancerModel, ModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        var result = new Dictionary<FaceEnhancerModel, ModelOptions>();

        foreach (var model in AllModels)
        {
            var fileName = ModelFileNames[model];
            var hash = new Download(
                BuildDownloadUrl(githubProvider, ModelBaseName, fileName + ".hash"),
                Path.Combine(modelsDirectory, fileName + ".hash"));
            var source = new Download(
                BuildDownloadUrl(githubProvider, ModelBaseName, fileName + ".onnx"),
                Path.Combine(modelsDirectory, fileName + ".onnx"));

            result[model] = new ModelOptions(ModelTemplates[model], ModelSizes[model], hash, source);
        }

        return result;
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    /// <summary>Same repo-root-walking approach as <see cref="FaceDetector"/>'s private helper
    /// of the same name — see its remarks for why <c>FileSystem.ResolveRelativePath</c> is not
    /// used directly from a test assembly's bin folder.</summary>
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
    public static ModelOptions GetModelOptions(FaceEnhancerModel faceEnhancerModel)
        => CreateStaticModelSet(DownloadScope.Full)[faceEnhancerModel];

    /// <summary>
    /// Python: <c>pre_check</c>. See the class remarks — this checks only the selected
    /// face_enhancer model's own hash/source files, not the six "common module" pre-checks
    /// (out of scope, matching FaceDetector/FaceMasker's documented reduced scope).
    /// </summary>
    public static bool PreCheck(FaceEnhancerModel faceEnhancerModel)
    {
        var options = GetModelOptions(faceEnhancerModel);
        return FileSystem.IsFile(options.Hash.Path) && FileSystem.IsFile(options.Source.Path);
    }

    // -----------------------------------------------------------------
    // Inference pool (thin wrappers around FaceFusion.Inference.InferenceManager)
    // -----------------------------------------------------------------

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    public static IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        InferenceManager inferenceManager,
        FaceEnhancerModel faceEnhancerModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var options = GetModelOptions(faceEnhancerModel);
        var modelSourceSet = new Dictionary<string, Download> { ["face_enhancer"] = options.Source };
        var modelNames = new[] { faceEnhancerModel.ToWireName() };
        return inferenceManager.GetInferencePool(ModuleName, modelNames, modelSourceSet, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public static void ClearInferencePool(
        InferenceManager inferenceManager,
        FaceEnhancerModel faceEnhancerModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelNames = new[] { faceEnhancerModel.ToWireName() };
        inferenceManager.ClearInferencePool(ModuleName, modelNames, executionDeviceIds, executionProviders);
    }

    // -----------------------------------------------------------------
    // enhance_face / forward / prepare / normalize / blend
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>enhance_face</c>. Does not take ownership of <paramref name="tempVisionFrame"/>
    /// or of <paramref name="faceEnhancerSession"/>/<paramref name="occluderInferencePool"/>.
    /// Caller owns the returned <see cref="Mat"/>. <paramref name="targetFace"/>'s
    /// <c>LandmarkSet.FiveOn68</c> must be a <c>float[,]</c> of shape <c>(5, 2)</c> (Python:
    /// <c>target_face.landmark_set.get('5/68')</c>).
    /// </summary>
    public static Mat EnhanceFace(
        Types.Face targetFace,
        Mat tempVisionFrame,
        ModelOptions modelOptions,
        double faceMaskBlur,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        FaceOccluderModel faceOccluderModel,
        IReadOnlyDictionary<string, InferenceSession> occluderInferencePool,
        InferenceSession faceEnhancerSession,
        double faceEnhancerWeight,
        double faceEnhancerBlend)
    {
        var faceLandmark5 = (float[,])targetFace.LandmarkSet.FiveOn68;
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, faceLandmark5, modelOptions.Template, modelOptions.Size);

        using var _affine = affineMatrix;
        using var _crop = cropVisionFrame;

        var cropMasks = new List<Mat>
        {
            FaceMasker.CreateBoxMask(cropVisionFrame, faceMaskBlur, new Padding(0, 0, 0, 0)),
        };

        try
        {
            if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                cropMasks.Add(FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool));
            }

            var (chwData, height, width) = PrepareCropFrame(cropVisionFrame);
            var outputData = Forward(chwData, height, width, faceEnhancerWeight, faceEnhancerSession);
            using var normalizedCropVisionFrame = NormalizeCropFrame(outputData, height, width);

            using var cropMask = ReduceMinimumClip(cropMasks);
            using var pasteVisionFrame = FaceHelper.PasteBack(tempVisionFrame, normalizedCropVisionFrame, cropMask, affineMatrix);

            return BlendPasteFrame(tempVisionFrame, pasteVisionFrame, faceEnhancerBlend);
        }
        finally
        {
            foreach (var mask in cropMasks)
            {
                mask.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>forward</c>. Loops <c>face_enhancer.get_inputs()</c> and wires
    /// <paramref name="chwData"/> to an <c>'input'</c> tensor of shape <c>(1, 3, H, W)</c>, and
    /// (only when the model declares it — CodeFormer does, GFPGAN/GPEN/RestoreFormer++ do not)
    /// <paramref name="faceEnhancerWeight"/> to a <c>'weight'</c> tensor of shape <c>(1,)</c>,
    /// float64 (Python: <c>numpy.array([face_enhancer_weight]).astype(numpy.double)</c>). Does
    /// not take ownership of <paramref name="faceEnhancerSession"/>. Returns the flat CHW
    /// output span copied into a managed array (the underlying <see cref="OrtValue"/> results
    /// are disposed before returning).
    /// </summary>
    public static float[] Forward(float[] chwData, int height, int width, double faceEnhancerWeight, InferenceSession faceEnhancerSession)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(chwData, new long[] { 1, 3, height, width });

        var inputs = new Dictionary<string, OrtValue>();
        var weightArray = new double[] { faceEnhancerWeight };
        OrtValue? weightOrtValue = null;

        try
        {
            foreach (var inputName in faceEnhancerSession.InputNames)
            {
                if (inputName == "input")
                {
                    inputs[inputName] = inputOrtValue;
                }
                else if (inputName == "weight")
                {
                    weightOrtValue = OrtValue.CreateTensorValueFromMemory(weightArray, new long[] { 1 });
                    inputs[inputName] = weightOrtValue;
                }
            }

            using var runOptions = new RunOptions();
            using var results = faceEnhancerSession.Run(runOptions, inputs, faceEnhancerSession.OutputNames);

            // Python: `face_enhancer.run(None, face_enhancer_inputs)[0][0]` — first output
            // tensor, first (only) batch entry, leaving shape (3, H, W).
            return results[0].GetTensorDataAsSpan<float>().ToArray();
        }
        finally
        {
            weightOrtValue?.Dispose();
        }
    }

    /// <summary>
    /// Python: <c>has_weight_input</c>. Does not take ownership of
    /// <paramref name="faceEnhancerSession"/>.
    /// </summary>
    public static bool HasWeightInput(InferenceSession faceEnhancerSession)
        => faceEnhancerSession.InputNames.Contains("weight");

    /// <summary>
    /// Python: <c>prepare_crop_frame</c>. <c>crop_vision_frame[:, :, ::-1] / 255.0</c> (BGR-&gt;RGB,
    /// <c>/255</c>), <c>(x - 0.5) / 0.5</c> (into <c>[-1, 1]</c>), then
    /// <c>transpose(2, 0, 1)</c> (HWC-&gt;CHW) + <c>expand_dims(axis = 0).astype(numpy.float32)</c>.
    /// Does not take ownership of <paramref name="cropVisionFrame"/>. Returns the flat CHW
    /// float32 data (batch dimension is implicit — always 1) plus its height/width.
    ///
    /// <para>
    /// <b>Dtype (float32 vs float64), reproduced exactly.</b> Python divides a <c>uint8</c>
    /// array by the Python float <c>255.0</c> (true division of an integer array always
    /// yields the default float dtype, <c>float64</c>), and the following <c>- 0.5</c> /
    /// <c>/ 0.5</c> stay <c>float64</c> too — only the final <c>.astype(numpy.float32)</c>
    /// narrows, after the HWC-&gt;CHW transpose (a no-op for precision). This mirrors
    /// <see cref="FaceFusion.Face.FaceRecognizer.PrepareInput"/>'s documented
    /// <c>/ 127.5 - 1</c> divergence from <see cref="FaceFusion.Face.FaceClassifier.PrepareInput"/>'s
    /// float32-throughout normalisation — each channel value here is computed as
    /// <c>((byte / 255.0) - 0.5) / 0.5</c> in <see cref="double"/> and only narrowed to
    /// <see cref="float"/> at the point of assignment into the CHW buffer, matching Python's
    /// per-element rounding exactly (per PORT_CONVENTIONS.md rule 6).
    /// </para>
    /// </summary>
    public static (float[] ChwData, int Height, int Width) PrepareCropFrame(Mat cropVisionFrame)
    {
        var height = cropVisionFrame.Rows;
        var width = cropVisionFrame.Cols;
        var plane = height * width;

        cropVisionFrame.GetArray(out Vec3b[] pixels);
        var hwcRgb = new float[plane * 3];

        for (var i = 0; i < plane; i++)
        {
            var pixel = pixels[i];
            var offset = i * 3;
            // BGR -> RGB (Python's `[:, :, ::-1]`), /255.0, then (x - 0.5) / 0.5 — computed in
            // double throughout (see remarks above), narrowed to float only here.
            hwcRgb[offset] = (float)(((pixel.Item2 / 255.0) - 0.5) / 0.5);
            hwcRgb[offset + 1] = (float)(((pixel.Item1 / 255.0) - 0.5) / 0.5);
            hwcRgb[offset + 2] = (float)(((pixel.Item0 / 255.0) - 0.5) / 0.5);
        }

        var chwData = NumPy.TransposeHwcToChw(hwcRgb, height, width, 3);
        return (chwData, height, width);
    }

    /// <summary>
    /// Python: <c>normalize_crop_frame</c>. <c>.clip(-1, 1)</c>, <c>(x + 1) / 2</c>,
    /// <c>transpose(1, 2, 0)</c> (CHW-&gt;HWC), <c>* 255.0</c>, <c>.round()</c>,
    /// <c>.astype(uint8)</c>, then <c>[:, :, ::-1]</c> (RGB-&gt;BGR). <paramref name="chwData"/>
    /// is the flat CHW output of <see cref="Forward"/> (batch dimension already squeezed by
    /// Python's <c>[0]</c> index). Caller owns the returned <see cref="Mat"/> (<c>CV_8UC3</c>,
    /// BGR).
    /// </summary>
    public static Mat NormalizeCropFrame(ReadOnlySpan<float> chwData, int height, int width)
    {
        var hwcRgb = NumPy.TransposeChwToHwc(chwData, 3, height, width);
        var plane = height * width;
        var pixels = new Vec3b[plane];

        for (var i = 0; i < plane; i++)
        {
            var offset = i * 3;
            var r = NormalizeChannel(hwcRgb[offset]);
            var g = NormalizeChannel(hwcRgb[offset + 1]);
            var b = NormalizeChannel(hwcRgb[offset + 2]);

            // RGB -> BGR (Python's trailing `[:, :, ::-1]`).
            pixels[i] = new Vec3b { Item0 = b, Item1 = g, Item2 = r };
        }

        var result = new Mat(height, width, MatType.CV_8UC3);
        result.SetArray(pixels);
        return result;
    }

    /// <summary>Python: <c>((value.clip(-1, 1) + 1) / 2 * 255.0).round().astype(uint8)</c> for
    /// a single channel value. Python's <c>numpy.round</c> uses round-half-to-even, matching
    /// <see cref="MidpointRounding.ToEven"/>; the final <c>.astype(uint8)</c> truncates toward
    /// zero, but the input is already a non-negative integer at that point (clipped to
    /// <c>[0, 255]</c> beforehand by construction), so truncation and saturation agree here.</summary>
    private static byte NormalizeChannel(float value)
    {
        var clipped = NumPy.Clip(value, -1f, 1f);
        var scaled = (clipped + 1f) / 2f * 255f;
        var rounded = MathF.Round(scaled, MidpointRounding.ToEven);
        return (byte)Math.Clamp(rounded, 0f, 255f);
    }

    /// <summary>
    /// Python: <c>blend_paste_frame</c>. Does not take ownership of either argument. Caller
    /// owns the returned <see cref="Mat"/>.
    /// </summary>
    public static Mat BlendPasteFrame(Mat tempVisionFrame, Mat pasteVisionFrame, double faceEnhancerBlend)
    {
        var blendFactor = 1 - (faceEnhancerBlend / 100.0);
        return FaceFusion.Vision.Vision.BlendFrame(tempVisionFrame, pasteVisionFrame, 1 - blendFactor);
    }

    /// <summary>Python: <c>numpy.minimum.reduce(crop_masks).clip(0, 1)</c>. Does not take
    /// ownership of any entry in <paramref name="masks"/>. Caller owns the returned
    /// <see cref="Mat"/> (<c>CV_32FC1</c>).</summary>
    private static Mat ReduceMinimumClip(IReadOnlyList<Mat> masks)
    {
        var result = masks[0].Clone();
        for (var i = 1; i < masks.Count; i++)
        {
            Cv2.Min(result, masks[i], result);
        }

        using var clampedLow = new Mat();
        Cv2.Max(result, 0.0, clampedLow);
        Cv2.Min(clampedLow, 1.0, result);
        return result;
    }

    // -----------------------------------------------------------------
    // process_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Does not take ownership of any <see cref="Mat"/>
    /// parameter. Caller owns the returned <see cref="Mat"/> vision frame; the returned mask
    /// is <paramref name="tempVisionMask"/> itself (Python returns it unmodified), not a copy.
    /// <paramref name="getStaticFaces"/>/<paramref name="refillFaces"/> stand in for
    /// <c>face_creator.get_static_faces</c>/<c>refill_faces</c> — see
    /// <see cref="FaceFusion.Face.FaceSelector.SelectFaces"/>'s remarks for why they are
    /// delegates rather than a hard dependency.
    /// </summary>
    public static (Mat VisionFrame, Mat Mask) ProcessFrame(
        Mat referenceVisionFrame,
        IReadOnlyList<Mat> sourceVisionFrames,
        IReadOnlyList<Mat> targetVisionFrames,
        Mat tempVisionFrame,
        Mat tempVisionMask,
        FaceSelectorMode faceSelectorMode,
        double faceTrackerScore,
        FaceSelectorOrder faceSelectorOrder,
        FaceSelectorGender? faceSelectorGender,
        FaceSelectorRace? faceSelectorRace,
        int? faceSelectorAgeStart,
        int? faceSelectorAgeEnd,
        int referenceFacePosition,
        double referenceFaceDistance,
        Func<IReadOnlyList<Mat>, IReadOnlyList<Types.Face>> getStaticFaces,
        Func<IReadOnlyList<Types.Face?>, IReadOnlyList<Types.Face>> refillFaces,
        double faceMaskBlur,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        FaceOccluderModel faceOccluderModel,
        IReadOnlyDictionary<string, InferenceSession> occluderInferencePool,
        ModelOptions modelOptions,
        InferenceSession faceEnhancerSession,
        double faceEnhancerWeight,
        double faceEnhancerBlend)
    {
        var targetVisionFrame = CommonHelper.GetMiddle(targetVisionFrames);
        var targetFaces = FaceSelector.SelectFaces(
            referenceVisionFrame, sourceVisionFrames, targetVisionFrames,
            faceSelectorMode, faceTrackerScore, faceSelectorOrder, faceSelectorGender, faceSelectorRace,
            faceSelectorAgeStart, faceSelectorAgeEnd, referenceFacePosition, referenceFaceDistance,
            getStaticFaces, refillFaces);

        var currentVisionFrame = tempVisionFrame;
        var ownsCurrentVisionFrame = false;

        if (targetFaces.Count > 0 && targetVisionFrame is not null)
        {
            foreach (var rawTargetFace in targetFaces)
            {
                var scaledTargetFace = FaceCreator.ScaleFace(rawTargetFace, targetVisionFrame, currentVisionFrame);
                var nextVisionFrame = EnhanceFace(
                    scaledTargetFace, currentVisionFrame, modelOptions, faceMaskBlur, faceMaskTypes,
                    faceOccluderModel, occluderInferencePool, faceEnhancerSession, faceEnhancerWeight, faceEnhancerBlend);

                if (ownsCurrentVisionFrame)
                {
                    currentVisionFrame.Dispose();
                }

                currentVisionFrame = nextVisionFrame;
                ownsCurrentVisionFrame = true;
            }
        }

        // Python returns `temp_vision_frame, temp_vision_mask` — if no face was enhanced,
        // `temp_vision_frame` is returned unmodified. This method's ownership contract
        // ("does not take ownership of any Mat parameter; caller owns the returned Mat") means
        // the no-op case must still hand back a fresh caller-owned clone rather than aliasing
        // `tempVisionFrame` itself, so the caller cannot end up disposing the same Mat twice.
        var resultVisionFrame = ownsCurrentVisionFrame ? currentVisionFrame : tempVisionFrame.Clone();
        return (resultVisionFrame, tempVisionMask);
    }

    // -----------------------------------------------------------------
    // IProcessor adapter
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>face_enhancer/core.py</c>'s <c>process_frame</c> inputs, widened with the
    /// settings the module would have read off <c>state_manager</c> — see
    /// <see cref="IProcessorInputs"/>'s remarks.
    /// </summary>
    public sealed record FaceEnhancerInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        ModelOptions ModelOptions,
        InferenceSession FaceEnhancerSession,
        double FaceEnhancerWeight,
        double FaceEnhancerBlend,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        double FaceMaskBlur,
        FaceOccluderModel FaceOccluderModel,
        IReadOnlyDictionary<string, InferenceSession> OccluderInferencePool,
        FaceSelectorMode FaceSelectorMode,
        double FaceTrackerScore,
        FaceSelectorOrder FaceSelectorOrder,
        FaceSelectorGender? FaceSelectorGender,
        FaceSelectorRace? FaceSelectorRace,
        int? FaceSelectorAgeStart,
        int? FaceSelectorAgeEnd,
        int ReferenceFacePosition,
        double ReferenceFaceDistance,
        Func<IReadOnlyList<Mat>, IReadOnlyList<Types.Face>> GetStaticFaces,
        Func<IReadOnlyList<Types.Face?>, IReadOnlyList<Types.Face>> RefillFaces) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/face_enhancer/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract — see
    /// <c>FaceSwapper.Processor</c> for the same pattern.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "face_enhancer";

        /// <inheritdoc />
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>Python: the <c>face_enhancer</c>-specific half of <c>pre_check</c>. Takes the
        /// chosen model because the parameterless <see cref="IProcessor.PreCheck"/> member has no
        /// <c>state_manager</c> to read it from.</summary>
        public bool PreCheck(FaceEnhancerModel model) => FaceEnhancer.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "face_enhancer.PreCheck requires a FaceEnhancerModel (no state_manager to read it from — call the FaceEnhancerModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Filesystem validation is out of scope (the same gap
        /// <c>FaceSwapper.Processor.PreProcess</c> documents); <c>face_enhancer</c> has no
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
            if (inputs is not FaceEnhancerInputs faceEnhancerInputs)
            {
                throw new ArgumentException($"expected {nameof(FaceEnhancerInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            var (visionFrame, mask) = FaceEnhancer.ProcessFrame(
                faceEnhancerInputs.ReferenceVisionFrame,
                faceEnhancerInputs.SourceVisionFrames,
                faceEnhancerInputs.TargetVisionFrames,
                faceEnhancerInputs.TempVisionFrame,
                faceEnhancerInputs.TempVisionMask,
                faceEnhancerInputs.FaceSelectorMode,
                faceEnhancerInputs.FaceTrackerScore,
                faceEnhancerInputs.FaceSelectorOrder,
                faceEnhancerInputs.FaceSelectorGender,
                faceEnhancerInputs.FaceSelectorRace,
                faceEnhancerInputs.FaceSelectorAgeStart,
                faceEnhancerInputs.FaceSelectorAgeEnd,
                faceEnhancerInputs.ReferenceFacePosition,
                faceEnhancerInputs.ReferenceFaceDistance,
                faceEnhancerInputs.GetStaticFaces,
                faceEnhancerInputs.RefillFaces,
                faceEnhancerInputs.FaceMaskBlur,
                faceEnhancerInputs.FaceMaskTypes,
                faceEnhancerInputs.FaceOccluderModel,
                faceEnhancerInputs.OccluderInferencePool,
                faceEnhancerInputs.ModelOptions,
                faceEnhancerInputs.FaceEnhancerSession,
                faceEnhancerInputs.FaceEnhancerWeight,
                faceEnhancerInputs.FaceEnhancerBlend);

            return new ProcessorOutputs(visionFrame, mask);
        }

        /// <summary>Python: <c>post_process()</c>. Cache clearing is out of scope without a real
        /// pool owner to clear (rule 5), same as every other processor here.</summary>
        public void PostProcess()
        {
        }
    }
}
