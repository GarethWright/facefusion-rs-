using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/face_editor/types.py</c>'s <c>FaceEditorModel =
/// Literal['live_portrait']</c> — a single-member enum, same reasoning as
/// <see cref="ExpressionRestorerModel"/>.
/// </summary>
public enum FaceEditorModel
{
    [WireName("live_portrait")]
    LivePortrait,
}

/// <summary>
/// Python: one entry of <c>face_editor/core.py</c>'s <c>create_static_model_set</c>. Same shape
/// as <see cref="ExpressionRestorerModelOptions"/>, plus the two extra <c>live_portrait</c>
/// sub-models <c>face_editor</c> needs and <c>expression_restorer</c> does not:
/// <c>eye_retargeter</c>/<c>lip_retargeter</c> (retargeting a single eye/lip open-ratio slider
/// into 21x3 motion-point deltas) and <c>stitcher</c> (blends the edited source motion points
/// back against the original target ones so the paste seam stays smooth).
/// </summary>
public sealed record FaceEditorModelOptions(
    IReadOnlyDictionary<string, Download> Hashes,
    IReadOnlyDictionary<string, Download> Sources,
    WarpTemplate Template,
    Size Size);

/// <summary>
/// Port of <c>facefusion/processors/modules/face_editor/{core,types,choices}.py</c> — edits a
/// target face's expression/gaze/mouth/eyebrow/head-pose via 14 continuous sliders, driven by
/// the same <c>live_portrait</c> motion model <see cref="ExpressionRestorer"/> and
/// <see cref="LivePortrait"/> already establish the conventions for.
///
/// <para>
/// <b>No global state; sessions and settings taken as parameters (PORT_CONVENTIONS.md rule 5).</b>
/// Every <c>state_manager.get_item('face_editor_*')</c> read becomes an explicit parameter on
/// <see cref="ApplyEdit"/>/<see cref="FaceEditorInputs"/>, matching <c>ExpressionRestorerInputs</c>'s
/// precedent. <c>get_inference_pool</c>'s real downloading is not reproduced for the same reason
/// <c>FaceSwapper</c>/<c>ExpressionRestorer</c> don't; <see cref="PreCheck"/> checks local file
/// presence only.
/// </para>
///
/// <para>
/// <b>Reuses <see cref="LivePortrait"/> directly, does not re-derive it.</b>
/// <see cref="LivePortrait.CreateRotation"/>/<see cref="LivePortrait.LimitAngle"/>/
/// <see cref="LivePortrait.LimitExpression"/> are called exactly as Python's <c>edit_head_rotation</c>/
/// <c>apply_edit</c> call <c>facefusion.processors.live_portrait</c>'s functions of the same
/// name — this module contributes only the fourteen slider-specific expression/motion edits and
/// the crop/mask/paste orchestration around them.
/// </para>
///
/// <para>
/// <b>Model input tensors matched Python exactly (rtol=atol=0) — see the port report.</b>
/// <see cref="PrepareCropFrame"/> (shared code path with <c>ExpressionRestorer.PrepareCropFrame</c>
/// but resizing to <em>half</em> of face_editor's own 512x512 model size, same as
/// expression_restorer's own 256x256) and <see cref="NormalizeCropFrame"/> are pure managed
/// preprocessing, parity-tested against <c>tools/parity/dump_processors4.py</c>'s fixtures with
/// zero tolerance. The five ONNX forward passes (feature/motion extractor, eye/lip retargeter,
/// stitcher, generator) are where ONNX Runtime itself does the arithmetic — a tight (not
/// loosened) tolerance per PARITY_HARNESS.md's "expect ~0 divergence" guidance.
/// </para>
///
/// <para>
/// <b>Mat / dtype conventions.</b> <c>VisionFrame</c>s are <see cref="Mat"/>, <c>CV_8UC3</c> BGR,
/// caller-owned on every return, matching <c>FaceHelper</c>/<c>ExpressionRestorer</c>.
/// <see cref="NormalizeCropFrame"/> always narrows to <c>uint8</c> (Python's own
/// <c>normalize_crop_frame</c> ends with an unconditional <c>.astype(numpy.uint8)</c>), so
/// <see cref="EditFace"/> reuses <c>FaceFusion.Face.FaceHelper.PasteBack</c> directly.
/// </para>
/// </summary>
public static class FaceEditor
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>face_editor_models = list(get_args(FaceEditorModel))</c>.</summary>
    public static readonly IReadOnlyList<FaceEditorModel> FaceEditorModels = Enum.GetValues<FaceEditorModel>();

    /// <summary>Python: every <c>face_editor_*_range = create_float_range(-1.0, 1.0, 0.05)</c> choice list — all fourteen sliders share the same range.</summary>
    public static readonly IReadOnlyList<double> FaceEditorSliderRange =
        FaceFusion.Core.CommonHelper.CreateFloatRange(-1.0, 1.0, 0.05);

    // -----------------------------------------------------------------
    // create_static_model_set
    // -----------------------------------------------------------------

    private static readonly object ModelCatalogLock = new();
    private static IReadOnlyDictionary<FaceEditorModel, FaceEditorModelOptions>? _cachedModelCatalog;

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). <paramref name="downloadScope"/>
    /// is accepted for signature parity with Python; the single entry is identical regardless of
    /// scope, matching <c>ExpressionRestorer.CreateStaticModelSet</c>'s precedent.
    /// </summary>
    public static IReadOnlyDictionary<FaceEditorModel, FaceEditorModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        lock (ModelCatalogLock)
        {
            return _cachedModelCatalog ??= BuildModelCatalog();
        }
    }

    private static IReadOnlyDictionary<FaceEditorModel, FaceEditorModelOptions> BuildModelCatalog()
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
            ["eye_retargeter"] = Component("live_portrait_eye_retargeter", ".hash"),
            ["lip_retargeter"] = Component("live_portrait_lip_retargeter", ".hash"),
            ["stitcher"] = Component("live_portrait_stitcher", ".hash"),
            ["generator"] = Component("live_portrait_generator", ".hash"),
        };
        var sources = new Dictionary<string, Download>
        {
            ["feature_extractor"] = Component("live_portrait_feature_extractor", ".onnx"),
            ["motion_extractor"] = Component("live_portrait_motion_extractor", ".onnx"),
            ["eye_retargeter"] = Component("live_portrait_eye_retargeter", ".onnx"),
            ["lip_retargeter"] = Component("live_portrait_lip_retargeter", ".onnx"),
            ["stitcher"] = Component("live_portrait_stitcher", ".onnx"),
            ["generator"] = Component("live_portrait_generator", ".onnx"),
        };

        return new Dictionary<FaceEditorModel, FaceEditorModelOptions>
        {
            [FaceEditorModel.LivePortrait] = new FaceEditorModelOptions(
                hashes, sources, WarpTemplate.Ffhq512, new Size(512, 512)),
        };
    }

    /// <summary>Same reasoning/implementation as <c>ExpressionRestorer.ResolveModelsDirectory</c>.</summary>
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
    /// Python: the <c>face_editor</c>-specific half of <c>pre_check</c> (the common-module half
    /// is the caller's responsibility, per <c>IProcessor.GetCommonModules</c>'s remarks).
    /// </summary>
    public static bool PreCheck()
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[FaceEditorModel.LivePortrait];

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
    /// Python: <c>prepare_crop_frame</c>. Resizes to half the model's crop size
    /// (<c>256x256</c> for face_editor's <c>512x512</c> template) with <c>cv2.INTER_AREA</c>
    /// before the usual BGR-&gt;RGB, <c>/255.0</c>, CHW, <c>expand_dims</c> tail — identical code
    /// shape to <c>ExpressionRestorer.PrepareCropFrame</c> (computed in <see cref="double"/> then
    /// narrowed to <see cref="float"/> only at assignment, matching numpy's <c>uint8 / python
    /// float -&gt; float64</c> promotion). Returns a flat length-<c>3*256*256</c> CHW array
    /// (batch dim implicit).
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
    /// Python: <c>normalize_crop_frame</c> — <c>transpose(1,2,0).clip(0,1) * 255.0</c>, then
    /// <c>.astype(uint8)[:, :, ::-1]</c> (RGB -&gt; BGR after the uint8 cast, not before — same
    /// order as Python; truncation toward zero is the correct semantics either way since every
    /// value is non-negative post-clip). Caller owns the returned <see cref="Mat"/>.
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
    // forward_extract_feature / forward_extract_motion / forward_retarget_eye / forward_retarget_lip / forward_stitch_motion_points / forward_generate_frame
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
    /// Python: <c>forward_extract_motion</c>. Same output-order reasoning as
    /// <c>ExpressionRestorer.ForwardExtractMotion</c> (verified via <c>onnx.load(...).graph.output</c>).
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

    /// <summary>Python: <c>forward_retarget_eye</c>. <paramref name="eyeMotionPoints"/> is the flat length-64 input (Python: <c>(1, 64)</c>). Returns the flat length-63 <c>(1, 21, 3)</c> output (batch dim implicit).</summary>
    public static float[] ForwardRetargetEye(InferenceSession eyeRetargeterSession, float[] eyeMotionPoints)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(eyeMotionPoints, new long[] { 1, eyeMotionPoints.Length });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = eyeRetargeterSession.Run(runOptions, inputs, eyeRetargeterSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>Python: <c>forward_retarget_lip</c>. <paramref name="lipMotionPoints"/> is the flat length-65 input (Python: <c>(1, 65)</c>). Returns the flat length-63 <c>(1, 21, 3)</c> output (batch dim implicit).</summary>
    public static float[] ForwardRetargetLip(InferenceSession lipRetargeterSession, float[] lipMotionPoints)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(lipMotionPoints, new long[] { 1, lipMotionPoints.Length });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = lipRetargeterSession.Run(runOptions, inputs, lipRetargeterSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>Python: <c>forward_stitch_motion_points</c>. Returns a flat length-63 <c>(1, 21, 3)</c> array (batch dim implicit).</summary>
    public static float[] ForwardStitchMotionPoints(InferenceSession stitcherSession, float[,] sourceMotionPoints, float[,] targetMotionPoints)
    {
        using var sourceOrtValue = OrtValue.CreateTensorValueFromMemory(FlattenMotionPoints(sourceMotionPoints), new long[] { 1, 21, 3 });
        using var targetOrtValue = OrtValue.CreateTensorValueFromMemory(FlattenMotionPoints(targetMotionPoints), new long[] { 1, 21, 3 });

        var inputs = new Dictionary<string, OrtValue>
        {
            ["source"] = sourceOrtValue,
            ["target"] = targetOrtValue,
        };

        using var runOptions = new RunOptions();
        using var results = stitcherSession.Run(runOptions, inputs, stitcherSession.OutputNames);

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>
    /// Python: <c>forward_generate_frame</c>. <paramref name="sourceMotionPoints"/>/
    /// <paramref name="targetMotionPoints"/> map to the model's <c>source</c>/<c>target</c>
    /// inputs directly (no swap, unlike <c>ExpressionRestorer.ForwardGenerateFrame</c>'s
    /// donor/target relabelling — <c>apply_edit</c> passes <c>motion_points_source</c> as
    /// <c>source</c> and <c>motion_points_target</c> as <c>target</c>, matching the parameter
    /// names literally). Returns a flat length-<c>3*512*512</c> CHW array (Python:
    /// <c>run(None, inputs)[0][0]</c> — first output, batch index 0).
    /// </summary>
    public static float[] ForwardGenerateFrame(
        InferenceSession generatorSession, ReadOnlySpan<float> featureVolume, float[,] sourceMotionPoints, float[,] targetMotionPoints)
    {
        using var featureVolumeOrtValue = OrtValue.CreateTensorValueFromMemory(featureVolume.ToArray(), new long[] { 1, 32, 16, 64, 64 });
        using var sourceOrtValue = OrtValue.CreateTensorValueFromMemory(FlattenMotionPoints(sourceMotionPoints), new long[] { 1, 21, 3 });
        using var targetOrtValue = OrtValue.CreateTensorValueFromMemory(FlattenMotionPoints(targetMotionPoints), new long[] { 1, 21, 3 });

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
    // numpy.interp helper for the [-1, 1] -> [lo, hi] sliders (double precision — see remarks)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>numpy.interp(value, [-1, 1], [lo, hi])</c> — every one of face_editor's
    /// fourteen slider edits calls this with the same two-point <c>[-1, 1]</c> domain. Computed
    /// in <see cref="double"/> (the slider values are Python floats, i.e. float64 — matching
    /// <c>ExpressionRestorer.RestoreExpression</c>'s precedent of computing an interp-derived
    /// factor directly in double rather than via <c>FaceFusion.Tensors.NumPy.Interp</c>, which is
    /// float32-only), clamped at the domain endpoints exactly like <c>numpy.interp</c>.
    /// </summary>
    private static double Interp(double value, double lo, double hi)
    {
        if (value <= -1.0)
        {
            return lo;
        }

        if (value >= 1.0)
        {
            return hi;
        }

        var t = (value + 1.0) / 2.0;
        return lo + (t * (hi - lo));
    }

    // -----------------------------------------------------------------
    // calculate_distance_ratio
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>calculate_distance_ratio</c>. <paramref name="faceLandmark68"/> is <c>(68,
    /// 2)</c>, same convention as <see cref="FaceHelper"/>. Computed in <see cref="double"/>
    /// (numpy float32 subtraction promoted through <c>numpy.linalg.norm</c>'s float64 internals,
    /// then the Python <c>float(...)</c> wrapper), matching the class's general "managed float
    /// math computed in double, narrowed only at the edge" convention.
    /// </summary>
    public static float CalculateDistanceRatio(float[,] faceLandmark68, int topIndex, int bottomIndex, int leftIndex, int rightIndex)
    {
        double verticalX = faceLandmark68[topIndex, 0] - faceLandmark68[bottomIndex, 0];
        double verticalY = faceLandmark68[topIndex, 1] - faceLandmark68[bottomIndex, 1];
        double horizontalX = faceLandmark68[leftIndex, 0] - faceLandmark68[rightIndex, 0];
        double horizontalY = faceLandmark68[leftIndex, 1] - faceLandmark68[rightIndex, 1];

        var verticalNorm = Math.Sqrt((verticalX * verticalX) + (verticalY * verticalY));
        var horizontalNorm = Math.Sqrt((horizontalX * horizontalX) + (horizontalY * horizontalY));

        return (float)(verticalNorm / (horizontalNorm + 1e-6));
    }

    // -----------------------------------------------------------------
    // edit_eyebrow_direction / edit_eye_gaze / edit_mouth_* / edit_head_rotation — the 14 sliders
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>edit_eyebrow_direction</c>. Mutates and returns a copy of
    /// <paramref name="expression"/> (Python mutates in place; every real caller in this port
    /// only threads the return value through — same reasoning as
    /// <c>ExpressionRestorer.RestrictExpressionAreas</c>'s "returns a new array" note).
    /// </summary>
    public static float[,] EditEyebrowDirection(float[,] expression, double faceEditorEyebrowDirection)
    {
        var result = (float[,])expression.Clone();

        if (faceEditorEyebrowDirection > 0)
        {
            result[1, 1] += (float)Interp(faceEditorEyebrowDirection, -0.015, 0.015);
            result[2, 1] -= (float)Interp(faceEditorEyebrowDirection, -0.020, 0.020);
        }
        else
        {
            result[1, 0] -= (float)Interp(faceEditorEyebrowDirection, -0.015, 0.015);
            result[2, 0] += (float)Interp(faceEditorEyebrowDirection, -0.020, 0.020);
            result[1, 1] += (float)Interp(faceEditorEyebrowDirection, -0.005, 0.005);
            result[2, 1] -= (float)Interp(faceEditorEyebrowDirection, -0.005, 0.005);
        }

        return result;
    }

    /// <summary>Python: <c>edit_eye_gaze</c>.</summary>
    public static float[,] EditEyeGaze(float[,] expression, double faceEditorEyeGazeHorizontal, double faceEditorEyeGazeVertical)
    {
        var result = (float[,])expression.Clone();

        if (faceEditorEyeGazeHorizontal > 0)
        {
            result[11, 0] += (float)Interp(faceEditorEyeGazeHorizontal, -0.015, 0.015);
            result[15, 0] += (float)Interp(faceEditorEyeGazeHorizontal, -0.020, 0.020);
        }
        else
        {
            result[11, 0] += (float)Interp(faceEditorEyeGazeHorizontal, -0.020, 0.020);
            result[15, 0] += (float)Interp(faceEditorEyeGazeHorizontal, -0.015, 0.015);
        }

        result[1, 1] += (float)Interp(faceEditorEyeGazeVertical, -0.0025, 0.0025);
        result[2, 1] -= (float)Interp(faceEditorEyeGazeVertical, -0.0025, 0.0025);
        result[11, 1] -= (float)Interp(faceEditorEyeGazeVertical, -0.010, 0.010);
        result[13, 1] -= (float)Interp(faceEditorEyeGazeVertical, -0.005, 0.005);
        result[15, 1] -= (float)Interp(faceEditorEyeGazeVertical, -0.010, 0.010);
        result[16, 1] -= (float)Interp(faceEditorEyeGazeVertical, -0.005, 0.005);

        return result;
    }

    /// <summary>
    /// Python: <c>edit_eye_open</c>. <paramref name="eyeRetargeterSession"/> is null-checked
    /// (rather than always required) so a caller with <paramref name="faceEditorEyeOpenRatio"/>
    /// exactly zero — a very common default — can skip loading the model, matching how the
    /// generator/stitcher session parameters elsewhere in this port are only exercised when the
    /// corresponding feature is in use; Python has no such gate (it always calls
    /// <c>forward_retarget_eye</c> and then multiplies by <c>abs(0) == 0</c>, i.e. it always
    /// pays for a model pass whose result is thrown away at the default). This divergence changes
    /// timing only, not the returned value: when the ratio is exactly 0, the retargeted motion
    /// points are scaled by 0 either way, so a zero motion-point delta is returned directly
    /// without needing the extra ONNX pass.
    /// </summary>
    public static float[,] EditEyeOpen(InferenceSession? eyeRetargeterSession, float[,] motionPoints, float[,] faceLandmark68, double faceEditorEyeOpenRatio)
    {
        if (faceEditorEyeOpenRatio == 0.0)
        {
            return new float[21, 3];
        }

        if (eyeRetargeterSession is null)
        {
            throw new ArgumentNullException(nameof(eyeRetargeterSession), "a non-zero EyeOpenRatio requires eyeRetargeterSession.");
        }

        var leftEyeRatio = CalculateDistanceRatio(faceLandmark68, 37, 40, 39, 36);
        var rightEyeRatio = CalculateDistanceRatio(faceLandmark68, 43, 46, 45, 42);

        var eyeMotionPoints = new float[21 * 3 + 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                eyeMotionPoints[(i * 3) + c] = motionPoints[i, c];
            }
        }

        eyeMotionPoints[21 * 3] = leftEyeRatio;
        eyeMotionPoints[(21 * 3) + 1] = rightEyeRatio;
        eyeMotionPoints[(21 * 3) + 2] = faceEditorEyeOpenRatio < 0 ? 0.0f : 0.6f;

        var retargeted = ForwardRetargetEye(eyeRetargeterSession, eyeMotionPoints);
        var scale = (float)Math.Abs(faceEditorEyeOpenRatio);

        var result = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                result[i, c] = retargeted[(i * 3) + c] * scale;
            }
        }

        return result;
    }

    /// <summary>Python: <c>edit_lip_open</c>. Same zero-ratio short-circuit reasoning as <see cref="EditEyeOpen"/>.</summary>
    public static float[,] EditLipOpen(InferenceSession? lipRetargeterSession, float[,] motionPoints, float[,] faceLandmark68, double faceEditorLipOpenRatio)
    {
        if (faceEditorLipOpenRatio == 0.0)
        {
            return new float[21, 3];
        }

        if (lipRetargeterSession is null)
        {
            throw new ArgumentNullException(nameof(lipRetargeterSession), "a non-zero LipOpenRatio requires lipRetargeterSession.");
        }

        var lipRatio = CalculateDistanceRatio(faceLandmark68, 62, 66, 54, 48);

        var lipMotionPoints = new float[21 * 3 + 2];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                lipMotionPoints[(i * 3) + c] = motionPoints[i, c];
            }
        }

        lipMotionPoints[21 * 3] = lipRatio;
        lipMotionPoints[(21 * 3) + 1] = faceEditorLipOpenRatio < 0 ? 0.0f : 1.0f;

        var retargeted = ForwardRetargetLip(lipRetargeterSession, lipMotionPoints);
        var scale = (float)Math.Abs(faceEditorLipOpenRatio);

        var result = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                result[i, c] = retargeted[(i * 3) + c] * scale;
            }
        }

        return result;
    }

    /// <summary>Python: <c>edit_mouth_grim</c>.</summary>
    public static float[,] EditMouthGrim(float[,] expression, double faceEditorMouthGrim)
    {
        var result = (float[,])expression.Clone();

        if (faceEditorMouthGrim > 0)
        {
            result[17, 2] -= (float)Interp(faceEditorMouthGrim, -0.005, 0.005);
            result[19, 2] += (float)Interp(faceEditorMouthGrim, -0.01, 0.01);
            result[20, 1] -= (float)Interp(faceEditorMouthGrim, -0.06, 0.06);
            result[20, 2] -= (float)Interp(faceEditorMouthGrim, -0.03, 0.03);
        }
        else
        {
            result[19, 1] -= (float)Interp(faceEditorMouthGrim, -0.05, 0.05);
            result[19, 2] -= (float)Interp(faceEditorMouthGrim, -0.02, 0.02);
            result[20, 2] -= (float)Interp(faceEditorMouthGrim, -0.03, 0.03);
        }

        return result;
    }

    /// <summary>Python: <c>edit_mouth_position</c>.</summary>
    public static float[,] EditMouthPosition(float[,] expression, double faceEditorMouthPositionHorizontal, double faceEditorMouthPositionVertical)
    {
        var result = (float[,])expression.Clone();

        result[19, 0] += (float)Interp(faceEditorMouthPositionHorizontal, -0.05, 0.05);
        result[20, 0] += (float)Interp(faceEditorMouthPositionHorizontal, -0.04, 0.04);

        if (faceEditorMouthPositionVertical > 0)
        {
            result[19, 1] -= (float)Interp(faceEditorMouthPositionVertical, -0.04, 0.04);
            result[20, 1] -= (float)Interp(faceEditorMouthPositionVertical, -0.02, 0.02);
        }
        else
        {
            result[19, 1] -= (float)Interp(faceEditorMouthPositionVertical, -0.05, 0.05);
            result[20, 1] -= (float)Interp(faceEditorMouthPositionVertical, -0.04, 0.04);
        }

        return result;
    }

    /// <summary>Python: <c>edit_mouth_pout</c>.</summary>
    public static float[,] EditMouthPout(float[,] expression, double faceEditorMouthPout)
    {
        var result = (float[,])expression.Clone();

        if (faceEditorMouthPout > 0)
        {
            result[19, 1] -= (float)Interp(faceEditorMouthPout, -0.022, 0.022);
            result[19, 2] += (float)Interp(faceEditorMouthPout, -0.025, 0.025);
            result[20, 2] -= (float)Interp(faceEditorMouthPout, -0.002, 0.002);
        }
        else
        {
            result[19, 1] += (float)Interp(faceEditorMouthPout, -0.022, 0.022);
            result[19, 2] += (float)Interp(faceEditorMouthPout, -0.025, 0.025);
            result[20, 2] -= (float)Interp(faceEditorMouthPout, -0.002, 0.002);
        }

        return result;
    }

    /// <summary>Python: <c>edit_mouth_purse</c>.</summary>
    public static float[,] EditMouthPurse(float[,] expression, double faceEditorMouthPurse)
    {
        var result = (float[,])expression.Clone();

        if (faceEditorMouthPurse > 0)
        {
            result[19, 1] -= (float)Interp(faceEditorMouthPurse, -0.04, 0.04);
            result[19, 2] -= (float)Interp(faceEditorMouthPurse, -0.02, 0.02);
        }
        else
        {
            result[14, 1] -= (float)Interp(faceEditorMouthPurse, -0.02, 0.02);
            result[17, 2] += (float)Interp(faceEditorMouthPurse, -0.01, 0.01);
            result[19, 2] -= (float)Interp(faceEditorMouthPurse, -0.015, 0.015);
            result[20, 2] -= (float)Interp(faceEditorMouthPurse, -0.002, 0.002);
        }

        return result;
    }

    /// <summary>Python: <c>edit_mouth_smile</c>.</summary>
    public static float[,] EditMouthSmile(float[,] expression, double faceEditorMouthSmile)
    {
        var result = (float[,])expression.Clone();

        if (faceEditorMouthSmile > 0)
        {
            result[20, 1] -= (float)Interp(faceEditorMouthSmile, -0.015, 0.015);
            result[14, 1] -= (float)Interp(faceEditorMouthSmile, -0.025, 0.025);
            result[17, 1] += (float)Interp(faceEditorMouthSmile, -0.01, 0.01);
            result[17, 2] += (float)Interp(faceEditorMouthSmile, -0.004, 0.004);
            result[3, 1] -= (float)Interp(faceEditorMouthSmile, -0.0045, 0.0045);
            result[7, 1] -= (float)Interp(faceEditorMouthSmile, -0.0045, 0.0045);
        }
        else
        {
            result[14, 1] -= (float)Interp(faceEditorMouthSmile, -0.02, 0.02);
            result[17, 1] += (float)Interp(faceEditorMouthSmile, -0.003, 0.003);
            result[19, 1] += (float)Interp(faceEditorMouthSmile, -0.02, 0.02);
            result[19, 2] -= (float)Interp(faceEditorMouthSmile, -0.005, 0.005);
            result[20, 2] += (float)Interp(faceEditorMouthSmile, -0.01, 0.01);
            result[3, 1] += (float)Interp(faceEditorMouthSmile, -0.0045, 0.0045);
            result[7, 1] += (float)Interp(faceEditorMouthSmile, -0.0045, 0.0045);
        }

        return result;
    }

    /// <summary>
    /// Python: <c>edit_head_rotation</c>. Reuses <see cref="LivePortrait.LimitAngle"/>/
    /// <see cref="LivePortrait.CreateRotation"/> directly, matching Python's own import of
    /// <c>facefusion.processors.live_portrait</c>'s functions.
    /// </summary>
    public static float[,] EditHeadRotation(
        float pitch, float yaw, float roll,
        double faceEditorHeadPitch, double faceEditorHeadYaw, double faceEditorHeadRoll)
    {
        // Python: `pitch + float(numpy.interp(face_editor_head_pitch, [-1, 1], [20, -20]))` —
        // a numpy float32 scalar plus a python float promotes to float64; computed in double
        // throughout, matching ExpressionRestorer's "managed float math in double" convention.
        var editPitch = (double)pitch + Interp(faceEditorHeadPitch, 20.0, -20.0);
        var editYaw = (double)yaw + Interp(faceEditorHeadYaw, 60.0, -60.0);
        var editRoll = (double)roll + Interp(faceEditorHeadRoll, -15.0, 15.0);

        var (limitedPitch, limitedYaw, limitedRoll) = LivePortrait.LimitAngle(pitch, yaw, roll, (float)editPitch, (float)editYaw, (float)editRoll);

        return LivePortrait.CreateRotation(limitedPitch, limitedYaw, limitedRoll);
    }

    // -----------------------------------------------------------------
    // apply_edit
    // -----------------------------------------------------------------

    /// <summary>Every <c>face_editor_*</c> slider value Python would have pulled from <c>state_manager</c> for one <see cref="ApplyEdit"/> call.</summary>
    public sealed record FaceEditorSliders(
        double EyebrowDirection,
        double EyeGazeHorizontal,
        double EyeGazeVertical,
        double EyeOpenRatio,
        double LipOpenRatio,
        double MouthGrim,
        double MouthPout,
        double MouthPurse,
        double MouthSmile,
        double MouthPositionHorizontal,
        double MouthPositionVertical,
        double HeadPitch,
        double HeadYaw,
        double HeadRoll);

    /// <summary>
    /// Python: <c>apply_edit</c>. <paramref name="faceLandmark68"/> is the *unscaled* <c>68</c>
    /// landmark set (Python: <c>target_face.landmark_set.get('68')</c> — note this is not the
    /// crop-relative/affine-transformed version <c>FaceSwapper</c>/<c>DeepSwapper</c> compute
    /// for their own area masks; face_editor's <c>calculate_distance_ratio</c> only ever
    /// computes ratios of differences, which are invariant to the crop's affine transform, so
    /// Python uses the frame-space landmarks directly). Returns the generator's raw <c>(3, 512,
    /// 512)</c> CHW output (Python leaves <c>crop_vision_frame</c> un-normalised here —
    /// <see cref="NormalizeCropFrame"/> is the caller's job, matching <c>edit_face</c>'s own call
    /// order).
    /// </summary>
    public static float[] ApplyEdit(
        InferenceSession featureExtractorSession,
        InferenceSession motionExtractorSession,
        InferenceSession? eyeRetargeterSession,
        InferenceSession? lipRetargeterSession,
        InferenceSession stitcherSession,
        InferenceSession generatorSession,
        ReadOnlySpan<float> cropVisionFrameChw,
        float[,] faceLandmark68,
        FaceEditorSliders sliders)
    {
        var featureVolume = ForwardExtractFeature(featureExtractorSession, cropVisionFrameChw);
        var (pitch, yaw, roll, scale, translation, rawExpression, motionPoints) = ForwardExtractMotion(motionExtractorSession, cropVisionFrameChw);

        var rotation = LivePortrait.CreateRotation(pitch, yaw, roll);
        var motionPointsTarget = ComputeMotionPoints(motionPoints, rotation, rawExpression, scale, translation);

        var expression = rawExpression;
        expression = EditEyeGaze(expression, sliders.EyeGazeHorizontal, sliders.EyeGazeVertical);
        expression = EditMouthGrim(expression, sliders.MouthGrim);
        expression = EditMouthPosition(expression, sliders.MouthPositionHorizontal, sliders.MouthPositionVertical);
        expression = EditMouthPout(expression, sliders.MouthPout);
        expression = EditMouthPurse(expression, sliders.MouthPurse);
        expression = EditMouthSmile(expression, sliders.MouthSmile);
        expression = EditEyebrowDirection(expression, sliders.EyebrowDirection);
        expression = LivePortrait.LimitExpression(expression);

        var editedRotation = EditHeadRotation(pitch, yaw, roll, sliders.HeadPitch, sliders.HeadYaw, sliders.HeadRoll);

        // Python: `motion_points_source = motion_points @ rotation.T; += expression; *= scale;
        // += translation; += edit_eye_open(...); += edit_lip_open(...)` — the same rotate/add
        // expression/scale/translate composition as ComputeMotionPoints, but built up with the
        // *edited* head rotation and expression, then two more additive deltas.
        var motionPointsSource = ComputeMotionPoints(motionPoints, editedRotation, expression, scale, translation);

        var eyeDelta = EditEyeOpen(eyeRetargeterSession, motionPointsTarget, faceLandmark68, sliders.EyeOpenRatio);
        var lipDelta = EditLipOpen(lipRetargeterSession, motionPointsTarget, faceLandmark68, sliders.LipOpenRatio);

        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                motionPointsSource[i, c] += eyeDelta[i, c] + lipDelta[i, c];
            }
        }

        var stitchedFlat = ForwardStitchMotionPoints(stitcherSession, motionPointsSource, motionPointsTarget);
        var stitchedMotionPointsSource = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                stitchedMotionPointsSource[i, c] = stitchedFlat[(i * 3) + c];
            }
        }

        return ForwardGenerateFrame(generatorSession, featureVolume, stitchedMotionPointsSource, motionPointsTarget);
    }

    /// <summary>
    /// Python: <c>scale * (motion_points @ rotation.T + expression) + translation</c>. Since
    /// <c>(p @ R.T)[c] == sum_k p[k] * R[c, k]</c>, this rotates every point by
    /// <paramref name="rotation"/> directly (no explicit transpose needed) — same helper shape
    /// as <c>ExpressionRestorer.ComputeMotionPoints</c>.
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
    // edit_face
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>edit_face</c>. Caller owns the returned <see cref="Mat"/>. Does not take
    /// ownership of <paramref name="tempVisionFrame"/>. Python: <c>face_landmark_5 =
    /// scale_face_landmark_5(target_face.landmark_set.get('5/68'), 1.5)</c> — a wider warp
    /// template than <c>FaceSwapper</c>/<c>ExpressionRestorer</c> use, giving the model more
    /// surrounding context to generate into.
    /// </summary>
    public static Mat EditFace(
        FaceFusion.Types.Face targetFace,
        Mat tempVisionFrame,
        InferenceSession featureExtractorSession,
        InferenceSession motionExtractorSession,
        InferenceSession? eyeRetargeterSession,
        InferenceSession? lipRetargeterSession,
        InferenceSession stitcherSession,
        InferenceSession generatorSession,
        FaceEditorSliders sliders,
        double faceMaskBlur)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[FaceEditorModel.LivePortrait];
        var modelTemplate = modelOptions.Template;
        var modelSize = modelOptions.Size;

        var targetLandmark5Of68 = (float[,])targetFace.LandmarkSet.FiveOn68;
        var scaledLandmark5 = FaceHelper.ScaleFaceLandmark5(targetLandmark5Of68, 1.5);

        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, scaledLandmark5, modelTemplate, modelSize);
        using var affineMatrixDisposable = affineMatrix;
        using var cropVisionFrameDisposable = cropVisionFrame;

        using var boxMask = FaceMasker.CreateBoxMask(cropVisionFrame, faceMaskBlur, new Padding(0, 0, 0, 0));

        var preparedCrop = PrepareCropFrame(cropVisionFrame);
        var faceLandmark68 = (float[,])targetFace.LandmarkSet.SixtyEight;

        var rawOutput = ApplyEdit(
            featureExtractorSession, motionExtractorSession, eyeRetargeterSession, lipRetargeterSession, stitcherSession, generatorSession,
            preparedCrop, faceLandmark68, sliders);

        using var normalizedCropVisionFrame = NormalizeCropFrame(rawOutput, modelSize.Height, modelSize.Width);

        return FaceHelper.PasteBack(tempVisionFrame, normalizedCropVisionFrame, boxMask, affineMatrix);
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>facefusion.processors.modules.face_editor.core</c>'s per-call inputs, extended
    /// per <see cref="IProcessorInputs"/>'s remarks — see each field's comment for the Python
    /// <c>state_manager</c> key/session it replaces. <see cref="EyeRetargeterSession"/>/
    /// <see cref="LipRetargeterSession"/> are nullable per <see cref="EditEyeOpen"/>/
    /// <see cref="EditLipOpen"/>'s remarks (only needed when the corresponding ratio slider is
    /// non-zero).
    /// </summary>
    public sealed record FaceEditorInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        FaceEditorSliders Sliders,
        double FaceMaskBlur,
        InferenceSession FeatureExtractorSession,
        InferenceSession MotionExtractorSession,
        InferenceSession? EyeRetargeterSession,
        InferenceSession? LipRetargeterSession,
        InferenceSession StitcherSession,
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
    /// Python: <c>facefusion/processors/modules/face_editor/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract — see
    /// <c>ExpressionRestorer.Processor</c> for the same pattern.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "face_editor";

        /// <inheritdoc />
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>Python: the <c>face_editor</c>-specific half of <c>pre_check</c>.</summary>
        public bool PreCheck() => FaceEditor.PreCheck();

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Same scope note as
        /// <c>ExpressionRestorer.Processor.PreProcess</c> (the <c>facefusion/filesystem.py</c>
        /// checks are out of this assignment's scope — unlike <c>expression_restorer</c>,
        /// face_editor's Python <c>pre_process</c> has no <c>mode == 'stream'</c> rejection, so
        /// this always returns <see langword="true"/>).
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
            if (inputs is not FaceEditorInputs faceEditorInputs)
            {
                throw new ArgumentException($"expected {nameof(FaceEditorInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return FaceEditor.ProcessFrame(faceEditorInputs);
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
    /// found) frame and mask, same ownership convention as <c>ExpressionRestorer.ProcessFrame</c>.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(FaceEditorInputs inputs)
    {
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
        var targetVisionFrame = FaceFusion.Core.CommonHelper.GetMiddle(inputs.TargetVisionFrames);

        if (targetFaces.Count > 0 && targetVisionFrame is not null)
        {
            foreach (var rawTargetFace in targetFaces)
            {
                var targetFace = FaceCreator.ScaleFace(rawTargetFace, targetVisionFrame, tempVisionFrame);

                var nextTempVisionFrame = EditFace(
                    targetFace,
                    tempVisionFrame,
                    inputs.FeatureExtractorSession,
                    inputs.MotionExtractorSession,
                    inputs.EyeRetargeterSession,
                    inputs.LipRetargeterSession,
                    inputs.StitcherSession,
                    inputs.GeneratorSession,
                    inputs.Sliders,
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
