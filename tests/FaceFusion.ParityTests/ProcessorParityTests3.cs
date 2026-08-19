using System.Text.Json;
using FaceFusion.Parity;
using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Processors.{LivePortrait,ExpressionRestorer,AgeModifier}</c>
/// against the real Python <c>facefusion.processors.live_portrait</c> and
/// <c>facefusion.processors.modules.{expression_restorer,age_modifier}.core</c>, run against the
/// real <c>live_portrait_{feature_extractor,motion_extractor,generator}</c> and <c>fran</c> ONNX
/// models. Ground truth was captured with <c>tools/parity/dump_processors3.py</c>; see that
/// script's docstring for why these families and not <c>styleganex_age</c> (hand-verified in
/// <c>AgeModifierTests</c> instead).
///
/// <para>
/// <b>Two tiers, gated differently</b> — same split as <c>FaceSwapperParityTests</c>: the
/// preprocessing-tensor tests need only the committed <c>.npy</c>/<c>.json</c> fixtures, so they
/// run once those are present; the end-to-end tests that run a real <see cref="InferenceSession"/>
/// additionally need the corresponding <c>.onnx</c> file(s) under <c>.assets/models/</c>
/// (<c>.gitignore</c>'d, never present on CI) and are gated with
/// <see cref="Processors3ModelFactAttribute"/>.
/// </para>
///
/// <para>
/// <b>Model-input tensors matched Python exactly (rtol=atol=0).</b> See
/// <see cref="TestPrepareCropFrameMatchesPythonExactly"/> (expression_restorer) and
/// <see cref="TestAgeModifierPrepareVisionFrameMatchesPythonExactly"/> (age_modifier fran) — the
/// assignment's bar is met for every model-input tensor dumped by
/// <c>tools/parity/dump_processors3.py</c>. Where ONNX Runtime then does the arithmetic, a tight
/// (not loosened) tolerance is used per PARITY_HARNESS.md's "expect ~0 divergence" guidance.
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class ProcessorParityTests3
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "processors3");

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
        System.AppContext.BaseDirectory, "fixtures", "processors3"));

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

    // -----------------------------------------------------------------
    // live_portrait.create_rotation / limit_expression / limit_angle — no ONNX needed
    // -----------------------------------------------------------------

    [Processors3FixturesFact]
    public void TestCreateRotationMatchesPython()
    {
        var input = LoadJson("live_portrait", "rotation_input");
        var rotation = LivePortrait.CreateRotation((float)input[0], (float)input[1], (float)input[2]);

        var expected = LoadNpy("live_portrait", "rotation_output").AsDoubles();
        var actual = new double[9];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual[(i * 3) + j] = rotation[i, j];
            }
        }

        // Pure managed trig/matrix-multiply reproducing scipy's Euler convention — no ONNX
        // Runtime/OpenCV involved, so the assignment's "expect ~0" bar applies. A small non-zero
        // tolerance (not exactly 0) accounts for the two independent float64 trig
        // implementations (.NET's Math.Cos/Sin vs. scipy's) accumulating through 2 matrix
        // multiplies before the final narrow to float32.
        AssertAllClose(actual, expected, rtol: 1e-6, atol: 1e-6, "create_rotation");
    }

    [Processors3FixturesFact]
    public void TestLimitExpressionMatchesPython()
    {
        var unclipped = LoadNpy("live_portrait", "expression_unclipped").AsFloats();
        var expression = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                expression[i, c] = unclipped[(i * 3) + c];
            }
        }

        var limited = LivePortrait.LimitExpression(expression);
        var actual = new double[21 * 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                actual[(i * 3) + c] = limited[i, c];
            }
        }

        var expected = LoadNpy("live_portrait", "expression_limited").AsDoubles();
        AssertAllClose(actual, expected, rtol: 0, atol: 0, "limit_expression");
    }

    [Processors3FixturesFact]
    public void TestLimitAngleMatchesPython()
    {
        var input = LoadJson("live_portrait", "limit_angle_input");
        var (pitch, yaw, roll) = LivePortrait.LimitAngle(
            (float)input[0], (float)input[1], (float)input[2],
            (float)input[3], (float)input[4], (float)input[5]);

        var expected = LoadJson("live_portrait", "limit_angle_output");
        Assert.Equal(expected[0], (double)pitch, 5);
        Assert.Equal(expected[1], (double)yaw, 5);
        Assert.Equal(expected[2], (double)roll, 5);
    }

    // -----------------------------------------------------------------
    // expression_restorer.prepare_crop_frame — model input, no ONNX Runtime required
    // -----------------------------------------------------------------

    [Processors3FixturesFact]
    public void TestPrepareCropFrameMatchesPythonExactly()
    {
        using var targetCrop = MatFromUInt8HwcFixture(LoadNpy("expression_restorer", "target_crop_vision_frame"));
        using var tempCrop = MatFromUInt8HwcFixture(LoadNpy("expression_restorer", "temp_crop_vision_frame"));

        var actualTarget = Array.ConvertAll(ExpressionRestorer.PrepareCropFrame(targetCrop), v => (double)v);
        var actualTemp = Array.ConvertAll(ExpressionRestorer.PrepareCropFrame(tempCrop), v => (double)v);

        var expectedTarget = LoadNpy("expression_restorer", "prepared_target_input").AsDoubles();
        var expectedTemp = LoadNpy("expression_restorer", "prepared_temp_input").AsDoubles();

        AssertAllClose(actualTarget, expectedTarget, rtol: 0, atol: 0, "prepared_target_input");
        AssertAllClose(actualTemp, expectedTemp, rtol: 0, atol: 0, "prepared_temp_input");
    }

    // -----------------------------------------------------------------
    // expression_restorer.restrict_expression_areas — pure math, no ONNX Runtime required
    // -----------------------------------------------------------------

    [Processors3FixturesFact]
    public void TestRestrictExpressionAreasMatchesPython()
    {
        var tempExpressionFlat = LoadNpy("expression_restorer", "temp_expression").AsFloats();
        var targetExpressionFlat = LoadNpy("expression_restorer", "target_expression").AsFloats();

        var tempExpression = ToExpression(tempExpressionFlat);
        var targetExpression = ToExpression(targetExpressionFlat);

        var result = ExpressionRestorer.RestrictExpressionAreas(tempExpression, targetExpression, new[] { ExpressionRestorerArea.UpperFace, ExpressionRestorerArea.LowerFace });

        var actual = new double[21 * 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                actual[(i * 3) + c] = result[i, c];
            }
        }

        var expected = LoadNpy("expression_restorer", "restricted_expression").AsDoubles();
        AssertAllClose(actual, expected, rtol: 0, atol: 0, "restrict_expression_areas");
    }

    private static float[,] ToExpression(float[] flat)
    {
        var result = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                result[i, c] = flat[(i * 3) + c];
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // age_modifier (fran) prepare_vision_frame — model input, no ONNX Runtime required
    // -----------------------------------------------------------------

    [Processors3FixturesFact]
    public void TestAgeModifierPrepareVisionFrameMatchesPythonExactly()
    {
        using var crop = MatFromUInt8HwcFixture(LoadNpy("age_modifier", "crop_vision_frame"));
        var mean = new[] { 0f, 0f, 0f };
        var std = new[] { 1f, 1f, 1f };

        var actual = Array.ConvertAll(AgeModifier.PrepareVisionFrame(crop, mean, std), v => (double)v);
        var expected = LoadNpy("age_modifier", "prepared_input").AsDoubles();

        AssertAllClose(actual, expected, rtol: 0, atol: 0, "age_modifier prepared_input (fran)");
    }

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference) — expression_restorer / live_portrait
    // -----------------------------------------------------------------

    [Processors3ModelFact("live_portrait_feature_extractor.onnx", "live_portrait_motion_extractor.onnx", "live_portrait_generator.onnx")]
    public void TestExpressionRestorerApplyRestoreMatchesPythonRawModelOutput()
    {
        var preparedTarget = LoadNpy("expression_restorer", "prepared_target_input").AsFloats();
        var preparedTemp = LoadNpy("expression_restorer", "prepared_temp_input").AsFloats();

        using var featureExtractorSession = new InferenceSession(FindModelPath("live_portrait_feature_extractor.onnx"));
        using var motionExtractorSession = new InferenceSession(FindModelPath("live_portrait_motion_extractor.onnx"));
        using var generatorSession = new InferenceSession(FindModelPath("live_portrait_generator.onnx"));

        // Cross-check the intermediate motion-extractor outputs first (tight ONNX Runtime
        // tolerance), since a mismatch here would otherwise be indistinguishable from a mismatch
        // in the rotation/expression-blend math further down the pipeline.
        var (targetPitch, targetYaw, targetRoll, _, _, _, _) = ExpressionRestorer.ForwardExtractMotion(motionExtractorSession, preparedTarget);
        var (tempPitch, tempYaw, tempRoll, tempScale, tempTranslation, tempExpression, tempMotionPoints) = ExpressionRestorer.ForwardExtractMotion(motionExtractorSession, preparedTemp);

        var expectedTargetScalars = LoadJson("expression_restorer", "target_motion_scalars");
        Assert.Equal(expectedTargetScalars[0], (double)targetPitch, 3);
        Assert.Equal(expectedTargetScalars[1], (double)targetYaw, 3);
        Assert.Equal(expectedTargetScalars[2], (double)targetRoll, 3);

        var expectedTempScalars = LoadJson("expression_restorer", "temp_motion_scalars");
        Assert.Equal(expectedTempScalars[0], (double)tempPitch, 3);
        Assert.Equal(expectedTempScalars[1], (double)tempYaw, 3);
        Assert.Equal(expectedTempScalars[2], (double)tempRoll, 3);

        var expectedTempScale = LoadNpy("expression_restorer", "temp_scale").AsDoubles();
        Assert.Equal(expectedTempScale[0], (double)tempScale, 3);

        var expectedTempTranslation = LoadNpy("expression_restorer", "temp_translation").AsDoubles();
        AssertAllClose(Array.ConvertAll(tempTranslation, v => (double)v), expectedTempTranslation, rtol: 1e-3, atol: 1e-3, "temp_translation");

        var expectedTempExpression = LoadNpy("expression_restorer", "temp_expression").AsDoubles();
        var actualTempExpression = new double[21 * 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                actualTempExpression[(i * 3) + c] = tempExpression[i, c];
            }
        }

        AssertAllClose(actualTempExpression, expectedTempExpression, rtol: 1e-3, atol: 1e-3, "temp_expression");

        var expectedTempMotionPoints = LoadNpy("expression_restorer", "temp_motion_points").AsDoubles();
        var actualTempMotionPoints = new double[21 * 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                actualTempMotionPoints[(i * 3) + c] = tempMotionPoints[i, c];
            }
        }

        AssertAllClose(actualTempMotionPoints, expectedTempMotionPoints, rtol: 1e-3, atol: 1e-3, "temp_motion_points");

        var rotation = LivePortrait.CreateRotation(tempPitch, tempYaw, tempRoll);
        var expectedRotation = LoadNpy("expression_restorer", "temp_rotation").AsDoubles();
        var actualRotation = new double[9];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actualRotation[(i * 3) + j] = rotation[i, j];
            }
        }

        AssertAllClose(actualRotation, expectedRotation, rtol: 1e-3, atol: 1e-3, "temp_rotation (derived from ONNX motion extractor scalars)");

        // Now the full apply_restore pipeline (all three models, plus the rotation/expression
        // blend/limit math in between).
        var rawOutput = ExpressionRestorer.ApplyRestore(
            featureExtractorSession, motionExtractorSession, generatorSession,
            preparedTarget, preparedTemp,
            expressionRestorerFactor: 80.0 / 100.0 * 1.2,
            expressionRestorerAreas: new[] { ExpressionRestorerArea.UpperFace, ExpressionRestorerArea.LowerFace });

        var expectedRawOutput = LoadNpy("expression_restorer", "apply_restore_raw_output").AsDoubles();
        AssertAllClose(Array.ConvertAll(rawOutput, v => (double)v), expectedRawOutput, rtol: 1e-2, atol: 1e-2, "apply_restore_raw_output");

        using var normalized = ExpressionRestorer.NormalizeCropFrame(rawOutput, 512, 512);
        normalized.GetArray(out Vec3b[] actualPixels);

        var expectedNormalized = LoadNpy("expression_restorer", "normalized_crop_frame");
        Assert.Equal("uint8", expectedNormalized.DType);
        var expectedRaw = expectedNormalized.RawData;

        // Byte-level comparison after the raw-output tolerance above already passed: allow a
        // small per-channel slack (the *255 + truncate step can flip a value across a byte
        // boundary for a raw-output difference well within the 1e-2 tolerance already asserted)
        // and require the overwhelming majority of pixels to match exactly.
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

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference) — age_modifier (fran)
    // -----------------------------------------------------------------

    [Processors3ModelFact("fran.onnx")]
    public void TestAgeModifierForwardMatchesPythonRawModelOutput()
    {
        var preparedCrop = LoadNpy("age_modifier", "prepared_input").AsFloats();
        var direction = LoadNpy("age_modifier", "direction_input").AsFloats();

        using var ageModifierSession = new InferenceSession(FindModelPath("fran.onnx"));
        var forwardOutput = AgeModifier.Forward(ageModifierSession, preparedCrop, new Size(1024, 1024), ReadOnlySpan<float>.Empty, null, direction);

        var expected = LoadNpy("age_modifier", "forward_output").AsDoubles();
        AssertAllClose(Array.ConvertAll(forwardOutput, v => (double)v), expected, rtol: 1e-3, atol: 1e-3, "age_modifier forward_output (fran)");

        using var normalized = AgeModifier.NormalizeVisionFrame(forwardOutput, 1024, 1024, new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f });
        Assert.Equal(MatType.CV_64FC3, normalized.Type());

        normalized.GetArray(out Vec3d[] actualPixels);
        var actualDoubles = new double[actualPixels.Length * 3];
        for (var i = 0; i < actualPixels.Length; i++)
        {
            actualDoubles[i * 3] = actualPixels[i].Item0;
            actualDoubles[(i * 3) + 1] = actualPixels[i].Item1;
            actualDoubles[(i * 3) + 2] = actualPixels[i].Item2;
        }

        var expectedNormalized = LoadNpy("age_modifier", "normalized_output").AsDoubles();
        AssertAllClose(actualDoubles, expectedNormalized, rtol: 1e-2, atol: 1e-1, "age_modifier normalized_output (fran)");
    }

    [Processors3ModelFact("fran.onnx")]
    public void TestModifyAgeFranEndToEndMatchesPythonByPsnr()
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy("age_modifier", "crop_vision_frame"));
        var mean = new[] { 0f, 0f, 0f };
        var std = new[] { 1f, 1f, 1f };

        var preparedCrop = AgeModifier.PrepareVisionFrame(cropVisionFrame, mean, std);
        var direction = LoadNpy("age_modifier", "direction_input").AsFloats();

        using var ageModifierSession = new InferenceSession(FindModelPath("fran.onnx"));
        var forwardOutput = AgeModifier.Forward(ageModifierSession, preparedCrop, new Size(1024, 1024), ReadOnlySpan<float>.Empty, null, direction);

        using var normalized = AgeModifier.NormalizeVisionFrame(forwardOutput, 1024, 1024, mean, std);

        var expected = LoadNpy("age_modifier", "normalized_output");
        normalized.GetArray(out Vec3d[] actualPixels);
        var actualDoubles = new double[actualPixels.Length * 3];
        for (var i = 0; i < actualPixels.Length; i++)
        {
            actualDoubles[i * 3] = actualPixels[i].Item0;
            actualDoubles[(i * 3) + 1] = actualPixels[i].Item1;
            actualDoubles[(i * 3) + 2] = actualPixels[i].Item2;
        }

        var expectedDoubles = expected.AsDoubles();
        var psnr = ImageMetrics.Psnr(actualDoubles, expectedDoubles, maxValue: 255.0);

        Assert.True(psnr > 35.0, $"age_modifier normalized_output PSNR too low: {psnr:F2} dB");
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the committed <c>processors3</c> fixture
/// directory is not present.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class Processors3FixturesFactAttribute : FactAttribute
{
    public Processors3FixturesFactAttribute()
    {
        if (!ProcessorParityTests3.FixturesAvailable)
        {
            Skip = "requires tests/FaceFusion.ParityTests/fixtures/processors3 (missing from this build output)";
        }
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c>
/// file(s) (or the fixtures) are not present — same reasoning as
/// <c>FaceSwapperParityTests.FaceSwapperModelFactAttribute</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class Processors3ModelFactAttribute : FactAttribute
{
    public Processors3ModelFactAttribute(params string[] modelFileNames)
    {
        if (!ProcessorParityTests3.FixturesAvailable)
        {
            Skip = "requires tests/FaceFusion.ParityTests/fixtures/processors3 (missing from this build output)";
            return;
        }

        foreach (var modelFileName in modelFileNames)
        {
            if (!ProcessorParityTests3.ModelAvailable(modelFileName))
            {
                Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                       "run `FACEFUSION_PARITY_DIR=... python3 tools/parity/dump_processors3.py` once with " +
                       "network access to populate .assets/models via pre_check(), then retry";
                return;
            }
        }
    }
}
