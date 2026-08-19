using System.Text.Json;
using FaceFusion.Parity;
using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Processors.{DeepSwapper,FaceEditor}</c> against the
/// real Python <c>facefusion.processors.modules.{deep_swapper,face_editor}.core</c>. Ground
/// truth was captured with <c>tools/parity/dump_processors4.py</c>; see that script's docstring
/// for why <c>deep_swapper</c> has no real ONNX-Runtime-backed fixtures (every <c>.dfm</c> is
/// hosted exclusively on Hugging Face, and this sandbox's egress policy blocks huggingface.co /
/// hf-mirror.com outright — confirmed via the agent proxy status endpoint, not a transient
/// failure) while <c>face_editor</c>'s fixtures run all five real <c>live_portrait</c> ONNX
/// models (feature/motion extractor, eye/lip retargeter, stitcher, generator — all hosted on
/// GitHub, reachable here).
///
/// <para>
/// <b>Two tiers, gated differently</b> — same split as <c>FaceSwapperParityTests</c>/
/// <c>ProcessorParityTests3</c>: the preprocessing-tensor tests need only the committed
/// <c>.npy</c>/<c>.json</c> fixtures, so they run once those are present; the end-to-end tests
/// that run a real <see cref="InferenceSession"/> additionally need the corresponding
/// <c>.onnx</c> file(s) under <c>.assets/models/</c> and are gated with
/// <see cref="Processors4ModelFactAttribute"/>.
/// </para>
///
/// <para>
/// <b>Model-input tensors matched Python exactly (rtol=atol=0)</b> — see
/// <see cref="TestDeepSwapperPrepareCropFrameMatchesPythonExactly"/> and
/// <see cref="TestFaceEditorPrepareCropFrameMatchesPythonExactly"/> — for every preprocessing
/// tensor this port could dump against real Python execution (see the class remarks above for
/// the one exception, <c>deep_swapper</c>'s own model-facing <c>in_face:0</c> tensor, which is
/// the *same* <c>prepare_crop_frame</c> output already verified — only the live ONNX Runtime
/// call around it is untestable here for the environmental reason above, not a preprocessing
/// gap). Where ONNX Runtime then does the arithmetic (face_editor only), a tight (not loosened)
/// tolerance is used per PARITY_HARNESS.md's "expect ~0 divergence" guidance.
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class ProcessorParityTests4
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "processors4");

    // -----------------------------------------------------------------
    // Shared fixture / environment helpers
    // -----------------------------------------------------------------

    internal static string? FindRepoRoot()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    internal static string? FindModelPath(string modelFileName)
    {
        var repoRoot = FindRepoRoot();
        return repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models", modelFileName);
    }

    internal static bool ModelAvailable(string modelFileName)
    {
        var modelPath = FindModelPath(modelFileName);
        return modelPath is not null && File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
    }

    internal static bool FixturesAvailable => Directory.Exists(Path.Combine(
        System.AppContext.BaseDirectory, "fixtures", "processors4"));

    private static NpyArray LoadNpy(string family, string name) =>
        NpyReader.Load(Path.Combine(FixturesDirectory, family, name + ".npy"));

    private static double[] LoadJson(string family, string name)
    {
        var path = Path.Combine(FixturesDirectory, family, name + ".json");
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<double[]>(json)!;
    }

    private static Mat MatFromUInt8HwcFixture(NpyArray array)
    {
        Assert.Equal("uint8", array.DType);
        Assert.Equal(3, array.Shape.Count);
        Assert.Equal(3, array.Shape[2]);

        var height = array.Shape[0];
        var width = array.Shape[1];
        var raw = array.RawData;

        var mat = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[height * width];
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Vec3b(raw[i * 3], raw[(i * 3) + 1], raw[(i * 3) + 2]);
        }

        mat.SetArray(pixels);
        return mat;
    }

    private static void AssertAllClose(double[] actual, double[] expected, double rtol, double atol, string label)
    {
        var result = TensorComparison.Compare(actual, expected, relativeTolerance: rtol, absoluteTolerance: atol);
        Assert.True(result.Passed, $"{label}: {result.Describe()}");
    }

    private static float[,] ToPointArray(float[] flat, int rows, int cols)
    {
        var result = new float[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[i, c] = flat[(i * cols) + c];
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // deep_swapper.prepare_crop_frame — model input, no ONNX Runtime required
    // -----------------------------------------------------------------

    [Processors4FixturesFact]
    public void TestDeepSwapperPrepareCropFrameMatchesPythonExactly()
    {
        using var crop = MatFromUInt8HwcFixture(LoadNpy("deep_swapper", "crop_vision_frame"));

        var actual = Array.ConvertAll(DeepSwapper.PrepareCropFrame(crop), v => (double)v);
        var expected = LoadNpy("deep_swapper", "prepared_input").AsDoubles();

        AssertAllClose(actual, expected, rtol: 0, atol: 0, "deep_swapper prepared_input (in_face:0)");
    }

    [Processors4FixturesFact]
    public void TestDeepSwapperNormalizeCropFrameMatchesPython()
    {
        // Python dumped normalize_crop_frame(prepared[0]) — i.e. its own PrepareCropFrame
        // output (already verified byte-exact above) round-tripped straight back through
        // NormalizeCropFrame, standing in for a real forward() output since the model itself
        // cannot be fetched here (see class remarks).
        using var crop = MatFromUInt8HwcFixture(LoadNpy("deep_swapper", "crop_vision_frame"));
        var prepared = DeepSwapper.PrepareCropFrame(crop);

        using var normalized = DeepSwapper.NormalizeCropFrame(prepared, crop.Rows, crop.Cols);

        var expected = LoadNpy("deep_swapper", "normalized_crop_frame");
        Assert.Equal("uint8", expected.DType);
        var expectedRaw = expected.RawData;

        normalized.GetArray(out Vec3b[] actualPixels);
        for (var i = 0; i < actualPixels.Length; i++)
        {
            Assert.Equal(expectedRaw[i * 3], actualPixels[i].Item0);
            Assert.Equal(expectedRaw[(i * 3) + 1], actualPixels[i].Item1);
            Assert.Equal(expectedRaw[(i * 3) + 2], actualPixels[i].Item2);
        }
    }

    [Processors4FixturesFact]
    public void TestDeepSwapperMorphInputFormulaMatchesPython()
    {
        // Python: numpy.array([numpy.interp(65, [0, 100], [0, 1])]).astype(float32) — dumped
        // with deep_swapper_morph = 65.
        var morphValue = (float)((65 - 0.0) / 100.0);
        var expected = LoadNpy("deep_swapper", "morph_input").AsFloats();

        Assert.Single(expected);
        Assert.Equal((double)expected[0], (double)morphValue, 6);
    }

    [Processors4FixturesFact]
    public void TestDeepSwapperPrepareCropMaskMatchesPythonWithinOpenCvTolerance()
    {
        var sourceMask = LoadNpy("deep_swapper", "crop_source_mask_input").AsFloats();
        var targetMask = LoadNpy("deep_swapper", "crop_target_mask_input").AsFloats();
        var modelSize = LoadNpy("deep_swapper", "crop_vision_frame"); // 224x224 crop -> 224x224 model size in this fixture set.

        using var result = DeepSwapper.PrepareCropMask(sourceMask, targetMask, new Size(modelSize.Shape[1], modelSize.Shape[0]));

        var expected = LoadNpy("deep_swapper", "crop_mask").AsDoubles();
        // Rows/Cols are P/Invoke calls, so they are read once rather than per iteration
        // (OpenCvSharp analyzer OCVS002).
        var rowTotal = result.Rows;
        var colTotal = result.Cols;
        var actual = new double[rowTotal * colTotal];

        for (var row = 0; row < rowTotal; row++)
        {
            for (var col = 0; col < colTotal; col++)
            {
                actual[(row * colTotal) + col] = result.At<float>(row, col);
            }
        }

        // Real OpenCV arithmetic on both sides (erode + GaussianBlur) — PARITY_HARNESS.md's
        // "expect ~0" bar, with a small epsilon for float32 accumulation order.
        AssertAllClose(actual, expected, rtol: 1e-5, atol: 1e-5, "deep_swapper prepare_crop_mask");
    }

    // -----------------------------------------------------------------
    // face_editor.prepare_crop_frame — model input, no ONNX Runtime required
    // -----------------------------------------------------------------

    [Processors4FixturesFact]
    public void TestFaceEditorPrepareCropFrameMatchesPythonExactly()
    {
        using var crop = MatFromUInt8HwcFixture(LoadNpy("face_editor", "crop_vision_frame"));

        var actual = Array.ConvertAll(FaceEditor.PrepareCropFrame(crop), v => (double)v);
        var expected = LoadNpy("face_editor", "prepared_input").AsDoubles();

        AssertAllClose(actual, expected, rtol: 0, atol: 0, "face_editor prepared_input");
    }

    // -----------------------------------------------------------------
    // face_editor.calculate_distance_ratio — pure math, no ONNX Runtime required
    // -----------------------------------------------------------------

    [Processors4FixturesFact]
    public void TestCalculateDistanceRatioMatchesPython()
    {
        var landmark68Flat = LoadNpy("face_editor", "face_landmark_68").AsFloats();
        var landmark68 = ToPointArray(landmark68Flat, 68, 2);

        var leftEyeRatio = FaceEditor.CalculateDistanceRatio(landmark68, 37, 40, 39, 36);
        var rightEyeRatio = FaceEditor.CalculateDistanceRatio(landmark68, 43, 46, 45, 42);
        var lipRatio = FaceEditor.CalculateDistanceRatio(landmark68, 62, 66, 54, 48);

        var expectedEyeRatios = LoadJson("face_editor", "eye_ratios");
        var expectedLipRatio = LoadJson("face_editor", "lip_ratio");

        Assert.Equal(expectedEyeRatios[0], (double)leftEyeRatio, 5);
        Assert.Equal(expectedEyeRatios[1], (double)rightEyeRatio, 5);
        Assert.Equal(expectedLipRatio[0], (double)lipRatio, 5);
    }

    // -----------------------------------------------------------------
    // face_editor edit_* sliders (the exact values dump_processors4.py used) — pure math
    // -----------------------------------------------------------------

    private static readonly FaceEditor.FaceEditorSliders DumpedSliders = new(
        EyebrowDirection: 0.6,
        EyeGazeHorizontal: 0.4,
        EyeGazeVertical: -0.3,
        EyeOpenRatio: 0.5,
        LipOpenRatio: -0.4,
        MouthGrim: 0.3,
        MouthPout: -0.2,
        MouthPurse: 0.25,
        MouthSmile: 0.5,
        MouthPositionHorizontal: -0.15,
        MouthPositionVertical: 0.2,
        HeadPitch: 0.3,
        HeadYaw: -0.25,
        HeadRoll: 0.1);

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference) — face_editor / live_portrait
    // -----------------------------------------------------------------

    private const string FeatureExtractorModel = "live_portrait_feature_extractor.onnx";
    private const string MotionExtractorModel = "live_portrait_motion_extractor.onnx";
    private const string EyeRetargeterModel = "live_portrait_eye_retargeter.onnx";
    private const string LipRetargeterModel = "live_portrait_lip_retargeter.onnx";
    private const string StitcherModel = "live_portrait_stitcher.onnx";
    private const string GeneratorModel = "live_portrait_generator.onnx";

    [Processors4ModelFact(MotionExtractorModel)]
    public void TestFaceEditorForwardExtractMotionMatchesPythonRawModelOutput()
    {
        var prepared = LoadNpy("face_editor", "prepared_input").AsFloats();

        using var motionExtractorSession = new InferenceSession(FindModelPath(MotionExtractorModel));
        var (pitch, yaw, roll, scale, translation, expression, motionPoints) = FaceEditor.ForwardExtractMotion(motionExtractorSession, prepared);

        var expectedScalars = LoadJson("face_editor", "motion_scalars");
        Assert.Equal(expectedScalars[0], (double)pitch, 3);
        Assert.Equal(expectedScalars[1], (double)yaw, 3);
        Assert.Equal(expectedScalars[2], (double)roll, 3);

        var expectedScale = LoadNpy("face_editor", "motion_scale").AsDoubles();
        Assert.Equal(expectedScale[0], (double)scale, 3);

        var expectedTranslation = LoadNpy("face_editor", "motion_translation").AsDoubles();
        AssertAllClose(Array.ConvertAll(translation, v => (double)v), expectedTranslation, rtol: 1e-3, atol: 1e-3, "motion_translation");

        var expectedExpression = LoadNpy("face_editor", "motion_expression").AsDoubles();
        var actualExpression = new double[21 * 3];
        var expectedMotionPoints = LoadNpy("face_editor", "motion_points").AsDoubles();
        var actualMotionPoints = new double[21 * 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                actualExpression[(i * 3) + c] = expression[i, c];
                actualMotionPoints[(i * 3) + c] = motionPoints[i, c];
            }
        }

        AssertAllClose(actualExpression, expectedExpression, rtol: 1e-3, atol: 1e-3, "motion_expression");
        AssertAllClose(actualMotionPoints, expectedMotionPoints, rtol: 1e-3, atol: 1e-3, "motion_points");
    }

    [Processors4ModelFact(EyeRetargeterModel)]
    public void TestFaceEditorForwardRetargetEyeMatchesPythonRawModelOutput()
    {
        var eyeInput = LoadNpy("face_editor", "eye_retargeter_input").AsFloats();

        using var eyeRetargeterSession = new InferenceSession(FindModelPath(EyeRetargeterModel));
        var actual = FaceEditor.ForwardRetargetEye(eyeRetargeterSession, eyeInput);

        var expected = LoadNpy("face_editor", "eye_retargeter_output").AsDoubles();
        AssertAllClose(Array.ConvertAll(actual, v => (double)v), expected, rtol: 1e-3, atol: 1e-3, "eye_retargeter_output");
    }

    [Processors4ModelFact(LipRetargeterModel)]
    public void TestFaceEditorForwardRetargetLipMatchesPythonRawModelOutput()
    {
        var lipInput = LoadNpy("face_editor", "lip_retargeter_input").AsFloats();

        using var lipRetargeterSession = new InferenceSession(FindModelPath(LipRetargeterModel));
        var actual = FaceEditor.ForwardRetargetLip(lipRetargeterSession, lipInput);

        var expected = LoadNpy("face_editor", "lip_retargeter_output").AsDoubles();
        AssertAllClose(Array.ConvertAll(actual, v => (double)v), expected, rtol: 1e-3, atol: 1e-3, "lip_retargeter_output");
    }

    [Processors4ModelFact(FeatureExtractorModel, MotionExtractorModel, EyeRetargeterModel, LipRetargeterModel, StitcherModel, GeneratorModel)]
    public void TestFaceEditorApplyEditMatchesPythonRawModelOutput()
    {
        var prepared = LoadNpy("face_editor", "prepared_input").AsFloats();
        var landmark68Flat = LoadNpy("face_editor", "face_landmark_68").AsFloats();
        var landmark68 = ToPointArray(landmark68Flat, 68, 2);

        using var featureExtractorSession = new InferenceSession(FindModelPath(FeatureExtractorModel));
        using var motionExtractorSession = new InferenceSession(FindModelPath(MotionExtractorModel));
        using var eyeRetargeterSession = new InferenceSession(FindModelPath(EyeRetargeterModel));
        using var lipRetargeterSession = new InferenceSession(FindModelPath(LipRetargeterModel));
        using var stitcherSession = new InferenceSession(FindModelPath(StitcherModel));
        using var generatorSession = new InferenceSession(FindModelPath(GeneratorModel));

        var rawOutput = FaceEditor.ApplyEdit(
            featureExtractorSession, motionExtractorSession, eyeRetargeterSession, lipRetargeterSession, stitcherSession, generatorSession,
            prepared, landmark68, DumpedSliders);

        var expectedRawOutput = LoadNpy("face_editor", "apply_edit_raw_output").AsDoubles();

        // Five chained ONNX models plus the edit_*/stitch math in between — a looser (but
        // still tight) tolerance than a single forward pass, matching
        // ExpressionRestorerParityTests' apply_restore_raw_output precedent.
        AssertAllClose(Array.ConvertAll(rawOutput, v => (double)v), expectedRawOutput, rtol: 1e-2, atol: 1e-2, "apply_edit_raw_output");

        using var normalized = FaceEditor.NormalizeCropFrame(rawOutput, 512, 512);
        normalized.GetArray(out Vec3b[] actualPixels);

        var expectedNormalized = LoadNpy("face_editor", "normalized_crop_frame");
        Assert.Equal("uint8", expectedNormalized.DType);
        var expectedRaw = expectedNormalized.RawData;

        var mismatchCount = 0;
        for (var i = 0; i < actualPixels.Length; i++)
        {
            var actualPixel = actualPixels[i];
            var expectedB = expectedRaw[i * 3];
            var expectedG = expectedRaw[(i * 3) + 1];
            var expectedR = expectedRaw[(i * 3) + 2];

            if (Math.Abs(actualPixel.Item0 - expectedB) > 2 || Math.Abs(actualPixel.Item1 - expectedG) > 2 || Math.Abs(actualPixel.Item2 - expectedR) > 2)
            {
                mismatchCount++;
            }
        }

        Assert.True(mismatchCount < actualPixels.Length * 0.01, $"normalized_crop_frame: {mismatchCount} / {actualPixels.Length} pixels differ by more than 2 (byte).");
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the committed <c>processors4</c> fixture
/// directory is not present.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class Processors4FixturesFactAttribute : FactAttribute
{
    public Processors4FixturesFactAttribute()
    {
        if (!ProcessorParityTests4.FixturesAvailable)
        {
            Skip = "requires tests/FaceFusion.ParityTests/fixtures/processors4 (missing from this build output)";
        }
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c>
/// file(s) (or the fixtures) are not present — same reasoning as
/// <c>ProcessorParityTests3.Processors3ModelFactAttribute</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class Processors4ModelFactAttribute : FactAttribute
{
    public Processors4ModelFactAttribute(params string[] modelFileNames)
    {
        if (!ProcessorParityTests4.FixturesAvailable)
        {
            Skip = "requires tests/FaceFusion.ParityTests/fixtures/processors4 (missing from this build output)";
            return;
        }

        foreach (var modelFileName in modelFileNames)
        {
            if (!ProcessorParityTests4.ModelAvailable(modelFileName))
            {
                Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                       "run `FACEFUSION_PARITY_DIR=... python3 tools/parity/dump_processors4.py` once with " +
                       "network access to populate .assets/models via pre_check(), then retry";
                return;
            }
        }
    }
}
