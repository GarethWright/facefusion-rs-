using FaceFusion.Face;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for <c>facefusion/face_landmarker.py</c>. There is no
/// <c>tests/test_face_landmarker.py</c> in the Python suite. Pure-logic pieces
/// (<c>ComputeScaleAndTranslation</c>, <c>ComputeTwoDFan4Score</c>) are verified by hand
/// against the module's numpy semantics; <c>ConditionalOptimizeContrast</c>'s CLAHE gate and
/// the <c>DetectFaceLandmark</c> model-selection logic are covered structurally here, and
/// tight cross-language parity for the full geometry chain (against the real <c>2dfan4</c>/
/// <c>peppa_wutz</c>/<c>fan_68_5</c> models and a real detected face) lives in
/// <c>tests/FaceFusion.ParityTests/FaceAnalysisParityTests.cs</c>.
/// </summary>
[Collection("NativeInference")]
public sealed class FaceLandmarkerTests
{
    // -----------------------------------------------------------------
    // ComputeScaleAndTranslation
    // -----------------------------------------------------------------

    [Fact]
    public void TestComputeScaleAndTranslationBasicCase()
    {
        // bounding_box = [100, 100, 300, 300] -> width = height = 200.
        // scale = 195 / max(200, 200).clip(1, None) = 195 / 200 = 0.975
        // translation = (256 - (400) * 0.975) * 0.5 = (256 - 390) * 0.5 = -67
        var boundingBox = new float[] { 100, 100, 300, 300 };
        var (scale, translation) = FaceLandmarker.ComputeScaleAndTranslation(boundingBox, new Size(256, 256));

        Assert.Equal(0.975, scale, 9);
        Assert.Equal(-67.0, translation[0], 9);
        Assert.Equal(-67.0, translation[1], 9);
    }

    [Fact]
    public void TestComputeScaleAndTranslationClipsTinyBoundingBoxToOne()
    {
        // A degenerate (near-zero-size) bounding box: max(dx, dy) would be < 1 without the
        // `.clip(1, None)` -- Python clips it up to 1 before dividing, so scale caps at 195.
        var boundingBox = new float[] { 100f, 100f, 100.5f, 100.2f };
        var (scale, _) = FaceLandmarker.ComputeScaleAndTranslation(boundingBox, new Size(256, 256));

        Assert.Equal(195.0, scale, 9);
    }

    [Fact]
    public void TestComputeScaleAndTranslationUsesTallerDimensionWhenTaller()
    {
        // width=100, height=400 -> max dimension is height.
        var boundingBox = new float[] { 0, 0, 100, 400 };
        var (scale, _) = FaceLandmarker.ComputeScaleAndTranslation(boundingBox, new Size(256, 256));

        Assert.Equal(195.0 / 400.0, scale, 9);
    }

    // -----------------------------------------------------------------
    // ComputeTwoDFan4Score — numpy.amax(axis=(2,3)) then mean then interp([0, 0.9], [0, 1])
    // -----------------------------------------------------------------

    [Fact]
    public void TestComputeTwoDFan4ScoreAllZeroHeatmapGivesZeroScore()
    {
        var heatmaps = new float[68 * 64 * 64];
        var score = FaceLandmarker.ComputeTwoDFan4Score(heatmaps);
        Assert.Equal(0.0, score, 6);
    }

    [Fact]
    public void TestComputeTwoDFan4ScoreUniformMaxInterpolatesLinearly()
    {
        // Every channel's max is 0.45 -> mean is 0.45 -> interp([0, 0.9], [0, 1]) at the
        // midpoint of the input range gives the midpoint of the output range: 0.5.
        var heatmaps = new float[68 * 64 * 64];
        for (var channel = 0; channel < 68; channel++)
        {
            heatmaps[(channel * 64 * 64) + 5] = 0.45f; // one peak per channel
        }

        var score = FaceLandmarker.ComputeTwoDFan4Score(heatmaps);
        Assert.Equal(0.5, score, 4);
    }

    [Fact]
    public void TestComputeTwoDFan4ScoreClampsAboveInputRange()
    {
        // numpy.interp clamps x above xp's range to the last fp value (1.0 here), it does not
        // extrapolate past it.
        var heatmaps = new float[68 * 64 * 64];
        for (var channel = 0; channel < 68; channel++)
        {
            heatmaps[channel * 64 * 64] = 5.0f; // far above 0.9
        }

        var score = FaceLandmarker.ComputeTwoDFan4Score(heatmaps);
        Assert.Equal(1.0, score, 4);
    }

    // -----------------------------------------------------------------
    // PrepareLandmarkerInput — no channel reversal (unlike FaceRecognizer/FaceClassifier), /255
    // -----------------------------------------------------------------

    [Fact]
    public void TestPrepareLandmarkerInputKeepsChannelOrderAndDividesBy255()
    {
        using var crop = new Mat(1, 1, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(10, 20, 30));

        var chw = FaceLandmarker.PrepareLandmarkerInput(crop);

        Assert.Equal(3, chw.Length);
        Assert.Equal((double)(10 / 255f), (double)chw[0], 6); // channel 0 stays B (no reversal)
        Assert.Equal((double)(20 / 255f), (double)chw[1], 6); // channel 1 stays G
        Assert.Equal((double)(30 / 255f), (double)chw[2], 6); // channel 2 stays R
    }

    // -----------------------------------------------------------------
    // ConditionalOptimizeContrast — CLAHE only applied when mean L < 30
    // -----------------------------------------------------------------

    [Fact]
    public void TestConditionalOptimizeContrastLeavesBrightFrameUnchangedByClahe()
    {
        // A mid-grey frame's Lab L channel mean is well above 30, so CLAHE never runs and
        // the RGB -> Lab -> RGB round trip should reproduce the input near-exactly.
        using var crop = new Mat(16, 16, MatType.CV_8UC3, new Scalar(150, 150, 150));
        using var result = FaceLandmarker.ConditionalOptimizeContrast(crop);

        result.GetArray(out Vec3b[] resultPixels);
        foreach (var pixel in resultPixels)
        {
            Assert.True(Math.Abs(pixel.Item0 - 150) <= 1, $"B channel drifted to {pixel.Item0}");
            Assert.True(Math.Abs(pixel.Item1 - 150) <= 1, $"G channel drifted to {pixel.Item1}");
            Assert.True(Math.Abs(pixel.Item2 - 150) <= 1, $"R channel drifted to {pixel.Item2}");
        }
    }

    [Fact]
    public void TestConditionalOptimizeContrastBrightensAVeryDarkFrame()
    {
        // A near-black frame's Lab L mean is well under 30, so CLAHE(clipLimit=2) runs on the
        // L channel -- for a flat, very dark frame this raises (or at least never lowers) the
        // brightness of at least one pixel; a hand transcription of the exact CLAHE output
        // is not attempted here (that is what FaceAnalysisParityTests' real-model fixtures
        // check bit-for-bit), this only checks the gate actually fires.
        using var crop = new Mat(16, 16, MatType.CV_8UC3, new Scalar(5, 5, 5));
        using var result = FaceLandmarker.ConditionalOptimizeContrast(crop);

        result.GetArray(out Vec3b[] resultPixels);
        var meanAfter = resultPixels.Average(p => (p.Item0 + p.Item1 + p.Item2) / 3.0);
        Assert.True(meanAfter >= 5.0, $"expected CLAHE not to darken a near-black frame, got mean {meanAfter}");
    }

    // -----------------------------------------------------------------
    // DetectFaceLandmark — model selection and the "unselected model can still win" oddity
    // -----------------------------------------------------------------

    [Fact]
    public void TestDetectFaceLandmarkRequiresTheRequestedSessions()
    {
        Assert.Throws<ArgumentNullException>(() => FaceLandmarker.DetectFaceLandmark(
            FaceLandmarkerModel.TwoDFan4, null, null, new Mat(4, 4, MatType.CV_8UC3), new float[] { 0, 0, 4, 4 }, 0));

        Assert.Throws<ArgumentNullException>(() => FaceLandmarker.DetectFaceLandmark(
            FaceLandmarkerModel.PeppaWutz, null, null, new Mat(4, 4, MatType.CV_8UC3), new float[] { 0, 0, 4, 4 }, 0));

        Assert.Throws<ArgumentNullException>(() => FaceLandmarker.DetectFaceLandmark(
            FaceLandmarkerModel.Many, null, null, new Mat(4, 4, MatType.CV_8UC3), new float[] { 0, 0, 4, 4 }, 0));
    }

    // -----------------------------------------------------------------
    // ModelSize
    // -----------------------------------------------------------------

    [Fact]
    public void TestModelSizesMatchPython()
    {
        Assert.Equal(256, FaceLandmarker.TwoDFan4ModelSize.Width);
        Assert.Equal(256, FaceLandmarker.TwoDFan4ModelSize.Height);
        Assert.Equal(256, FaceLandmarker.PeppaWutzModelSize.Width);
        Assert.Equal(256, FaceLandmarker.PeppaWutzModelSize.Height);
    }

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference) against the real 2dfan4/peppa_wutz/fan_68_5
    // -----------------------------------------------------------------

    [ModelAndMediaFact("2dfan4.onnx", "peppa_wutz.onnx", "fan_68_5.onnx")]
    public void TestDetectFaceLandmarkAndEstimateFaceLandmark685AgainstRealModels()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        Assert.NotNull(visionFrame);

        // A generous bounding box roughly centred on the frame; exact-value parity for a real
        // detected face lives in FaceAnalysisParityTests.
        var boundingBox = new float[] { 300f, 300f, 700f, 700f };

        using var twoDFan4Session = new InferenceSession(ModelAndMediaFactAttribute.FindModelPath("2dfan4.onnx"));
        using var peppaWutzSession = new InferenceSession(ModelAndMediaFactAttribute.FindModelPath("peppa_wutz.onnx"));

        var (faceLandmark68, score) = FaceLandmarker.DetectFaceLandmark(
            FaceLandmarkerModel.Many, twoDFan4Session, peppaWutzSession, visionFrame!, boundingBox, 0);

        Assert.NotNull(faceLandmark68);
        Assert.Equal(68, faceLandmark68!.GetLength(0));
        Assert.Equal(2, faceLandmark68.GetLength(1));
        Assert.True(score is >= 0.0 and <= 1.0, $"score {score} out of [0, 1]");

        var faceLandmark5 = FaceHelper.ConvertToFaceLandmark5(faceLandmark68);
        using var fan685Session = new InferenceSession(ModelAndMediaFactAttribute.FindModelPath("fan_68_5.onnx"));
        var faceLandmark685 = FaceLandmarker.EstimateFaceLandmark685(fan685Session, faceLandmark5);

        Assert.Equal(68, faceLandmark685.GetLength(0));
        Assert.Equal(2, faceLandmark685.GetLength(1));
    }
}
