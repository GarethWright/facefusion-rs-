using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/expression_restorer/types.py</c>'s
/// <c>ExpressionRestorerModel = Literal['live_portrait']</c> — a single-member enum, kept as an
/// enum rather than a constant for symmetry with every other processor's model choice and so a
/// future model addition is a one-line change here, matching <c>FaceSwapperModel</c>'s
/// precedent.
/// </summary>
public enum ExpressionRestorerModel
{
    [WireName("live_portrait")]
    LivePortrait,
}

/// <summary>
/// Python: <c>ExpressionRestorerArea = Literal['upper-face', 'lower-face']</c>.
/// </summary>
public enum ExpressionRestorerArea
{
    [WireName("upper-face")]
    UpperFace,

    [WireName("lower-face")]
    LowerFace,
}

/// <summary>
/// Python: one entry of <c>expression_restorer/core.py</c>'s <c>create_static_model_set</c>.
/// Same shape/reasoning as <c>FaceSwapperModelOptions</c> — a concrete record rather than the
/// generic <c>ModelOptions</c> alias, since every real caller needs every field.
/// </summary>
public sealed record ExpressionRestorerModelOptions(
    IReadOnlyDictionary<string, Download> Hashes,
    IReadOnlyDictionary<string, Download> Sources,
    WarpTemplate Template,
    Size Size);

/// <summary>
/// Port of <c>facefusion/processors/modules/expression_restorer/{core,types,choices}.py</c> —
/// restores a target face's original expression onto the swapped/enhanced version of the same
/// face using the <c>live_portrait</c> motion model (<see cref="LivePortrait"/>).
///
/// <para>
/// <b>No global state; sessions and settings taken as parameters (PORT_CONVENTIONS.md rule 5).</b>
/// Every <c>state_manager.get_item(...)</c> read becomes an explicit parameter — see
/// <see cref="ExpressionRestorerInputs"/>, matching <c>FaceSwapperInputs</c>'s precedent.
/// <c>get_inference_pool</c>'s real downloading is not reproduced for the same reason
/// <c>FaceSwapper</c> doesn't (no <c>facefusion/download.py</c> port); <see cref="PreCheck"/>
/// checks local file presence only.
/// </para>
///
/// <para>
/// <b>Model input tensors matched Python exactly (rtol=atol=0) — see the port report.</b>
/// <see cref="PrepareCropFrame"/> (the <c>feature_extractor</c>/<c>motion_extractor</c> input)
/// and <see cref="NormalizeCropFrame"/> (the <c>generator</c> output turned back into a paste-able
/// frame) are pure managed preprocessing, parity-tested against
/// <c>tools/parity/dump_processors3.py</c>'s fixtures with zero tolerance. The three ONNX
/// forward passes (<see cref="ForwardExtractFeature"/>/<see cref="ForwardExtractMotion"/>/
/// <see cref="ForwardGenerateFrame"/>) are where ONNX Runtime itself does the arithmetic — a
/// tight (not loosened) tolerance per PARITY_HARNESS.md's "expect ~0 divergence" guidance.
/// </para>
///
/// <para>
/// <b>Mat / dtype conventions.</b> <c>VisionFrame</c>s are <see cref="Mat"/>, <c>CV_8UC3</c> BGR,
/// caller-owned on every return, matching <c>FaceHelper</c>/<c>FaceSwapper</c>. Unlike
/// <c>FaceSwapper.NormalizeCropFrame</c> (which sometimes stays float to feed a float
/// <c>paste_back</c>), <see cref="NormalizeCropFrame"/> here always narrows to <c>uint8</c> —
/// Python's own <c>normalize_crop_frame</c> ends with <c>.astype(numpy.uint8)</c> unconditionally
/// (no per-model-kind branch, unlike face_swapper's) — so <see cref="RestoreExpression"/> reuses
/// <c>FaceFusion.Face.FaceHelper.PasteBack</c> directly rather than needing a float-crop variant.
/// </para>
/// </summary>
public static class ExpressionRestorer
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>expression_restorer_models = list(get_args(ExpressionRestorerModel))</c>.</summary>
    public static readonly IReadOnlyList<ExpressionRestorerModel> ExpressionRestorerModels = Enum.GetValues<ExpressionRestorerModel>();

    /// <summary>Python: <c>expression_restorer_areas = list(get_args(ExpressionRestorerArea))</c>.</summary>
    public static readonly IReadOnlyList<ExpressionRestorerArea> ExpressionRestorerAreas = Enum.GetValues<ExpressionRestorerArea>();

    /// <summary>Python: <c>expression_restorer_factor_range = create_int_range(0, 100, 1)</c>.</summary>
    public static readonly IReadOnlyList<int> ExpressionRestorerFactorRange =
        FaceFusion.Core.CommonHelper.CreateIntRange(0, 100, 1);

    // -----------------------------------------------------------------
    // create_static_model_set
    // -----------------------------------------------------------------

    private static readonly object ModelCatalogLock = new();
    private static IReadOnlyDictionary<ExpressionRestorerModel, ExpressionRestorerModelOptions>? _cachedModelCatalog;

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). <paramref name="downloadScope"/>
    /// is accepted for signature parity with Python; the single entry is identical regardless of
    /// scope, matching <c>FaceSwapper.CreateStaticModelSet</c>'s precedent.
    /// </summary>
    public static IReadOnlyDictionary<ExpressionRestorerModel, ExpressionRestorerModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        lock (ModelCatalogLock)
        {
            return _cachedModelCatalog ??= BuildModelCatalog();
        }
    }

    private static IReadOnlyDictionary<ExpressionRestorerModel, ExpressionRestorerModelOptions> BuildModelCatalog()
    {
        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        const string modelsBaseName = "models-3.0.0";

        Download Component(string fileName, string extension) => new(
            BuildDownloadUrl(githubProvider, modelsBaseName, fileName + extension),
            Path.Combine(modelsDirectory, fileName + extension));

        var hashes = new Dictionary<string, Download>
        {
            ["feature_extractor"] = Component("live_portrait_feature_extractor", ".hash"),
            ["motion_extractor"] = Component("live_portrait_motion_extractor", ".hash"),
            ["generator"] = Component("live_portrait_generator", ".hash"),
        };
        var sources = new Dictionary<string, Download>
        {
            ["feature_extractor"] = Component("live_portrait_feature_extractor", ".onnx"),
            ["motion_extractor"] = Component("live_portrait_motion_extractor", ".onnx"),
            ["generator"] = Component("live_portrait_generator", ".onnx"),
        };

        return new Dictionary<ExpressionRestorerModel, ExpressionRestorerModelOptions>
        {
            [ExpressionRestorerModel.LivePortrait] = new ExpressionRestorerModelOptions(
                hashes, sources, WarpTemplate.Arcface128, new Size(512, 512)),
        };
    }

    /// <summary>Same reasoning/implementation as <c>FaceSwapper.ResolveModelsDirectory</c>.</summary>
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
    /// Python: the <c>expression_restorer</c>-specific half of <c>pre_check</c> (the
    /// common-module half is the caller's responsibility, per <c>IProcessor.GetCommonModules</c>).
    /// </summary>
    public static bool PreCheck()
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[ExpressionRestorerModel.LivePortrait];

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
    // prepare_crop_frame / normalize_crop_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_crop_frame</c>. Resizes to half the model's crop size with
    /// <c>cv2.INTER_AREA</c> (Python: <c>prepare_size = (model_size[0] // 2, model_size[1] // 2)</c>
    /// — 256x256 for the 512x512 <c>live_portrait</c> template) before the usual BGR-&gt;RGB,
    /// <c>/255.0</c>, CHW, <c>expand_dims</c> tail (computed in <see cref="double"/> then narrowed
    /// to <see cref="float"/> only at assignment, matching numpy's <c>uint8 / python float -&gt;
    /// float64</c> promotion — same reasoning as <c>FaceSwapper.BuildRgbChwFloat32</c>). Returns a
    /// flat length-<c>3*256*256</c> CHW array (batch dim implicit).
    /// </summary>
    public static float[] PrepareCropFrame(Mat cropVisionFrame)
    {
        using var resized = new Mat();
        Cv2.Resize(cropVisionFrame, resized, new Size(256, 256), 0, 0, InterpolationFlags.Area);

        var plane = resized.Rows * resized.Cols;
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
    /// Python: <c>normalize_crop_frame</c>. Turns the generator's raw <c>(3, 512, 512)</c> CHW
    /// output back into a <c>CV_8UC3</c> BGR <see cref="Mat"/>: <c>transpose(1,2,0).clip(0,1) *
    /// 255.0</c>, then <c>.astype(uint8)</c> (numpy truncation toward zero, not round-to-nearest —
    /// reproduced with a plain truncating cast since every value is non-negative after the clip),
    /// then <c>[:, :, ::-1]</c> (RGB -&gt; BGR). Caller owns the returned <see cref="Mat"/>.
    /// </summary>
    public static Mat NormalizeCropFrame(ReadOnlySpan<float> modelOutputChw, int height, int width)
    {
        var plane = height * width;

        if (modelOutputChw.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputChw has {modelOutputChw.Length} elements, expected {3 * plane} for a {width}x{height} CHW frame.", nameof(modelOutputChw));
        }

        var result = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[plane];

        for (var index = 0; index < plane; index++)
        {
            var r = ClampToByteRange01(modelOutputChw[index]);
            var g = ClampToByteRange01(modelOutputChw[plane + index]);
            var b = ClampToByteRange01(modelOutputChw[(2 * plane) + index]);

            pixels[index] = new Vec3b(b, g, r); // BGR
        }

        result.SetArray(pixels);
        return result;
    }

    private static byte ClampToByteRange01(float value)
    {
        var clipped = value < 0f ? 0f : (value > 1f ? 1f : value);
        return (byte)(clipped * 255.0f);
    }

    // -----------------------------------------------------------------
    // forward_extract_feature / forward_extract_motion / forward_generate_frame
    // -----------------------------------------------------------------

    /// <summary>Python: <c>forward_extract_feature</c>. Returns a flat length-<c>32*16*64*64</c> array (batch dim implicit).</summary>
    public static float[] ForwardExtractFeature(InferenceSession featureExtractorSession, ReadOnlySpan<float> cropVisionFrameChw)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(cropVisionFrameChw.ToArray(), new long[] { 1, 3, 256, 256 });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = featureExtractorSession.Run(runOptions, inputs, featureExtractorSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>
    /// Python: <c>forward_extract_motion</c>. Output order (<c>pitch, yaw, roll, scale,
    /// translation, expression, motion_points</c>) matches
    /// <c>live_portrait_motion_extractor.onnx</c>'s own declared graph output order (verified via
    /// <c>onnx.load(...).graph.output</c> — the same order <c>featureExtractorSession.OutputNames</c>
    /// is passed to <c>Run</c> in, so positional indexing here is safe, matching
    /// <c>FaceSwapper.ForwardSwapFace</c>'s own positional-index precedent for a single-output
    /// model). Scale/translation/expression/motion_points have their batch dimension of 1 dropped
    /// per the class remarks.
    /// </summary>
    public static (float Pitch, float Yaw, float Roll, float Scale, float[] Translation, float[,] Expression, float[,] MotionPoints)
        ForwardExtractMotion(InferenceSession motionExtractorSession, ReadOnlySpan<float> cropVisionFrameChw)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(cropVisionFrameChw.ToArray(), new long[] { 1, 3, 256, 256 });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = motionExtractorSession.Run(runOptions, inputs, motionExtractorSession.OutputNames);

        var pitch = results[0].GetTensorDataAsSpan<float>()[0];
        var yaw = results[1].GetTensorDataAsSpan<float>()[0];
        var roll = results[2].GetTensorDataAsSpan<float>()[0];
        var scale = results[3].GetTensorDataAsSpan<float>()[0];
        var translation = results[4].GetTensorDataAsSpan<float>().ToArray();
        var expressionFlat = results[5].GetTensorDataAsSpan<float>();
        var motionPointsFlat = results[6].GetTensorDataAsSpan<float>();

        var expression = new float[21, 3];
        var motionPoints = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                expression[i, c] = expressionFlat[(i * 3) + c];
                motionPoints[i, c] = motionPointsFlat[(i * 3) + c];
            }
        }

        return (pitch, yaw, roll, scale, translation, expression, motionPoints);
    }

    /// <summary>
    /// Python: <c>forward_generate_frame</c>. <paramref name="targetMotionPoints"/>/
    /// <paramref name="tempMotionPoints"/> map to the model's <c>source</c>/<c>target</c> inputs
    /// respectively (Python: <c>{'feature_volume': ..., 'source': target_motion_points, 'target':
    /// temp_motion_points}</c> — note the swap: the *target* expression donor becomes the model's
    /// <c>source</c> input, reproduced exactly). Returns a flat length-<c>3*512*512</c> CHW array
    /// (Python: <c>run(None, inputs)[0][0]</c> — first output, batch index 0).
    /// </summary>
    public static float[] ForwardGenerateFrame(
        InferenceSession generatorSession, ReadOnlySpan<float> featureVolume, float[,] targetMotionPoints, float[,] tempMotionPoints)
    {
        using var featureVolumeOrtValue = OrtValue.CreateTensorValueFromMemory(featureVolume.ToArray(), new long[] { 1, 32, 16, 64, 64 });
        using var sourceOrtValue = OrtValue.CreateTensorValueFromMemory(FlattenMotionPoints(targetMotionPoints), new long[] { 1, 21, 3 });
        using var targetOrtValue = OrtValue.CreateTensorValueFromMemory(FlattenMotionPoints(tempMotionPoints), new long[] { 1, 21, 3 });

        var inputs = new Dictionary<string, OrtValue>
        {
            ["feature_volume"] = featureVolumeOrtValue,
            ["source"] = sourceOrtValue,
            ["target"] = targetOrtValue,
        };

        using var runOptions = new RunOptions();
        using var results = generatorSession.Run(runOptions, inputs, generatorSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    private static float[] FlattenMotionPoints(float[,] motionPoints)
    {
        var flat = new float[21 * 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                flat[(i * 3) + c] = motionPoints[i, c];
            }
        }

        return flat;
    }

    // -----------------------------------------------------------------
    // restrict_expression_areas
    // -----------------------------------------------------------------

    private static readonly int[] UpperFaceRows = { 1, 2, 6, 10, 11, 12, 13, 15, 16 };
    private static readonly int[] LowerFaceRows = { 3, 7, 14, 17, 18, 19, 20 };
    private static readonly int[] AlwaysRestrictedRows = { 0, 4, 5, 8, 9 };

    /// <summary>
    /// Python: <c>restrict_expression_areas</c>. Returns a new array rather than mutating
    /// <paramref name="targetExpression"/> in place (Python's numpy fancy-index assignment does
    /// mutate its argument, but every caller in this port only ever uses the return value, so a
    /// pure function is behaviourally equivalent and safer). Rows in
    /// <see cref="AlwaysRestrictedRows"/> are copied from <paramref name="tempExpression"/>
    /// unconditionally, matching Python's un-guarded final assignment.
    /// </summary>
    public static float[,] RestrictExpressionAreas(float[,] tempExpression, float[,] targetExpression, IReadOnlyList<ExpressionRestorerArea> expressionRestorerAreas)
    {
        var result = (float[,])targetExpression.Clone();

        if (!expressionRestorerAreas.Contains(ExpressionRestorerArea.UpperFace))
        {
            CopyRows(tempExpression, result, UpperFaceRows);
        }

        if (!expressionRestorerAreas.Contains(ExpressionRestorerArea.LowerFace))
        {
            CopyRows(tempExpression, result, LowerFaceRows);
        }

        CopyRows(tempExpression, result, AlwaysRestrictedRows);

        return result;
    }

    private static void CopyRows(float[,] source, float[,] destination, IReadOnlyList<int> rows)
    {
        foreach (var row in rows)
        {
            destination[row, 0] = source[row, 0];
            destination[row, 1] = source[row, 1];
            destination[row, 2] = source[row, 2];
        }
    }

    // -----------------------------------------------------------------
    // apply_restore
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>apply_restore</c>. <paramref name="targetCropVisionFrameChw"/>/
    /// <paramref name="tempCropVisionFrameChw"/> are <see cref="PrepareCropFrame"/>'s output (the
    /// model *inputs*, already preprocessed — Python re-uses the same "prepared" local variable
    /// names for both the pre- and post-<c>prepare_crop_frame</c> value, which this port keeps
    /// distinct via the parameter name). Returns the generator's raw <c>(3, 512, 512)</c> CHW
    /// output (Python leaves <c>crop_vision_frame</c> un-normalised here —
    /// <see cref="NormalizeCropFrame"/> is the caller's job, matching
    /// <c>restore_expression</c>'s own call order).
    /// </summary>
    public static float[] ApplyRestore(
        InferenceSession featureExtractorSession,
        InferenceSession motionExtractorSession,
        InferenceSession generatorSession,
        ReadOnlySpan<float> targetCropVisionFrameChw,
        ReadOnlySpan<float> tempCropVisionFrameChw,
        double expressionRestorerFactor,
        IReadOnlyList<ExpressionRestorerArea> expressionRestorerAreas)
    {
        var featureVolume = ForwardExtractFeature(featureExtractorSession, tempCropVisionFrameChw);
        var (_, _, _, _, _, targetExpressionRaw, _) = ForwardExtractMotion(motionExtractorSession, targetCropVisionFrameChw);
        var (pitch, yaw, roll, scale, translation, tempExpression, motionPoints) = ForwardExtractMotion(motionExtractorSession, tempCropVisionFrameChw);

        var rotation = LivePortrait.CreateRotation(pitch, yaw, roll);
        var targetExpression = RestrictExpressionAreas(tempExpression, targetExpressionRaw, expressionRestorerAreas);

        // Python: `target_expression * factor + temp_expression * (1 - factor)` — computed in
        // double (expressionRestorerFactor is itself a Python float from `numpy.interp`, not
        // float32) then narrowed to float32 only at assignment, then `limit_expression`.
        var blended = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                blended[i, c] = (float)((targetExpression[i, c] * expressionRestorerFactor) + (tempExpression[i, c] * (1.0 - expressionRestorerFactor)));
            }
        }

        var limitedTargetExpression = LivePortrait.LimitExpression(blended);

        var targetMotionPoints = ComputeMotionPoints(motionPoints, rotation, limitedTargetExpression, scale, translation);
        var tempMotionPoints = ComputeMotionPoints(motionPoints, rotation, tempExpression, scale, translation);

        return ForwardGenerateFrame(generatorSession, featureVolume, targetMotionPoints, tempMotionPoints);
    }

    /// <summary>
    /// Python: <c>scale * (motion_points @ rotation.T + expression) + translation</c>. Since
    /// <c>(p @ R.T)[c] == sum_k p[k] * R[c, k]</c>, this rotates every point by
    /// <paramref name="rotation"/> directly (no explicit transpose needed).
    /// </summary>
    private static float[,] ComputeMotionPoints(float[,] motionPoints, float[,] rotation, float[,] expression, float scale, float[] translation)
    {
        var result = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                double rotated = 0;
                for (var k = 0; k < 3; k++)
                {
                    rotated += (double)motionPoints[i, k] * rotation[c, k];
                }

                result[i, c] = (float)((scale * (rotated + expression[i, c])) + translation[c]);
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // restore_expression
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>restore_expression</c>. Caller owns the returned <see cref="Mat"/>. Does not
    /// take ownership of <paramref name="targetVisionFrame"/>/<paramref name="tempVisionFrame"/>.
    /// Only <see cref="FaceMaskType.Box"/> is exercised by this assignment's parity fixtures (same
    /// scope note as <c>FaceSwapper.SwapFace</c>); <see cref="FaceMaskType.Occlusion"/> is
    /// implemented (reusing <c>FaceFusion.Face.FaceMasker</c>) but needs
    /// <paramref name="occluderInferencePool"/>/<paramref name="faceOccluderModel"/>, both
    /// optional and unused when <see cref="FaceMaskType.Occlusion"/> is not requested.
    /// </summary>
    public static Mat RestoreExpression(
        FaceFusion.Types.Face targetFace,
        Mat targetVisionFrame,
        Mat tempVisionFrame,
        InferenceSession featureExtractorSession,
        InferenceSession motionExtractorSession,
        InferenceSession generatorSession,
        int expressionRestorerFactorSetting,
        IReadOnlyList<ExpressionRestorerArea> expressionRestorerAreas,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        double faceMaskBlur,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[ExpressionRestorerModel.LivePortrait];
        var modelTemplate = modelOptions.Template;
        var modelSize = modelOptions.Size;

        // Python: `float(numpy.interp(float(state_manager.get_item('expression_restorer_factor')),
        // [0, 100], [0, 1.2]))` — a linear map from [0,100] to [0,1.2], computed directly in
        // double rather than via FaceFusion.Tensors.NumPy.Interp (which is float32-only) since
        // Python's own `float(...)` wrapper keeps this a float64 computation throughout.
        var expressionRestorerFactor = expressionRestorerFactorSetting / 100.0 * 1.2;

        var targetLandmark5Of68 = (float[,])targetFace.LandmarkSet.FiveOn68;

        var (targetCropVisionFrame, targetAffineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(targetVisionFrame, targetLandmark5Of68, modelTemplate, modelSize);
        using var targetCropDisposable = targetCropVisionFrame;
        using var targetAffineDisposable = targetAffineMatrix;

        var (tempCropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, targetLandmark5Of68, modelTemplate, modelSize);
        using var tempCropDisposable = tempCropVisionFrame;
        using var affineMatrixDisposable = affineMatrix;

        var cropMasks = new List<Mat>();

        try
        {
            cropMasks.Add(FaceMasker.CreateBoxMask(tempCropVisionFrame, faceMaskBlur, new Padding(0, 0, 0, 0)));

            if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                if (occluderInferencePool is null)
                {
                    throw new ArgumentNullException(nameof(occluderInferencePool), "FaceMaskType.Occlusion requires occluderInferencePool.");
                }

                cropMasks.Add(FaceMasker.CreateOcclusionMask(tempCropVisionFrame, faceOccluderModel, occluderInferencePool));
            }

            var preparedTarget = PrepareCropFrame(targetCropVisionFrame);
            var preparedTemp = PrepareCropFrame(tempCropVisionFrame);

            var rawOutput = ApplyRestore(featureExtractorSession, motionExtractorSession, generatorSession, preparedTarget, preparedTemp, expressionRestorerFactor, expressionRestorerAreas);

            using var normalizedCropVisionFrame = NormalizeCropFrame(rawOutput, modelSize.Height, modelSize.Width);

            using var cropMask = ReduceMinimumClip01(cropMasks);
            return FaceHelper.PasteBack(tempVisionFrame, normalizedCropVisionFrame, cropMask, affineMatrix);
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

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>facefusion.processors.modules.expression_restorer.core</c>'s per-call inputs,
    /// extended per <see cref="IProcessorInputs"/>'s remarks — see each field's comment for the
    /// Python <c>state_manager</c> key/session it replaces. Unlike <c>FaceSwapperInputs</c>, there
    /// is no source-identity concept here (<c>SourceVisionFrames</c> is only used by
    /// <see cref="FaceSelector.SelectFaces"/>'s own <c>faceSelectorMode == 'reference'</c> path,
    /// matching Python's <c>select_faces</c>).
    /// </summary>
    public sealed record ExpressionRestorerInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        int ExpressionRestorerFactor,
        IReadOnlyList<ExpressionRestorerArea> ExpressionRestorerAreas,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        double FaceMaskBlur,
        InferenceSession FeatureExtractorSession,
        InferenceSession MotionExtractorSession,
        InferenceSession GeneratorSession,
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
    /// Python: <c>facefusion/processors/modules/expression_restorer/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract — see
    /// <c>FaceSwapper.Processor</c> for the same pattern.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "expression_restorer";

        /// <inheritdoc />
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>Python: the <c>expression_restorer</c>-specific half of <c>pre_check</c>.</summary>
        public bool PreCheck() => ExpressionRestorer.PreCheck();

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. The <c>mode == 'stream'</c> branch (Python: logs and
        /// returns <see langword="false"/> — expression restoration is not supported live) is the
        /// one condition expressible without <c>facefusion/filesystem.py</c> (out of scope, same
        /// gap <c>FaceSwapper.Processor.PreProcess</c> documents).
        /// </summary>
        public bool PreProcess(ProcessMode mode, ProcessorRunPaths paths)
        {
            _ = paths;
            return mode != ProcessMode.Stream;
        }

        /// <inheritdoc />
        public ProcessorOutputs ProcessFrame(IProcessorInputs inputs)
        {
            if (inputs is not ExpressionRestorerInputs expressionRestorerInputs)
            {
                throw new ArgumentException($"expected {nameof(ExpressionRestorerInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return ExpressionRestorer.ProcessFrame(expressionRestorerInputs);
        }

        /// <inheritdoc />
        public void PostProcess()
        {
        }
    }

    // -----------------------------------------------------------------
    // process_frame orchestration
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Returns the (possibly unchanged, if no target face was
    /// found) frame and mask, same ownership convention as <c>FaceSwapper.ProcessFrame</c>.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(ExpressionRestorerInputs inputs)
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
                var nextTempVisionFrame = RestoreExpression(
                    targetFace,
                    targetVisionFrame,
                    tempVisionFrame,
                    inputs.FeatureExtractorSession,
                    inputs.MotionExtractorSession,
                    inputs.GeneratorSession,
                    inputs.ExpressionRestorerFactor,
                    inputs.ExpressionRestorerAreas,
                    inputs.FaceMaskTypes,
                    inputs.FaceMaskBlur);

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
