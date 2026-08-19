using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for <c>facefusion/face_classifier.py</c>. There is no
/// <c>tests/test_face_classifier.py</c> in the Python suite; <c>categorize_gender</c>/
/// <c>categorize_age</c>/<c>categorize_race</c> are pure <c>if</c>-chains transcribed exactly
/// from the Python source (every branch and boundary below), and <c>PrepareInput</c>'s
/// arithmetic is verified against real numpy (see
/// <c>FaceFusion.Face.FaceClassifier</c>'s class remarks for the float32-throughout dtype
/// proof). The end-to-end case is verified against the real <c>fairface</c> ONNX model; tight
/// cross-language parity for the same path lives in
/// <c>tests/FaceFusion.ParityTests/FaceAnalysisParityTests.cs</c>.
/// </summary>
[Collection("NativeInference")]
public sealed class FaceClassifierTests
{
    // -----------------------------------------------------------------
    // CategorizeGender
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(1, Gender.Female)]
    [InlineData(0, Gender.Male)]
    [InlineData(2, Gender.Male)] // Python: `if gender_id == 1: return 'female'` then unconditional `return 'male'`.
    [InlineData(-1, Gender.Male)]
    public void TestCategorizeGender(long genderId, Gender expected)
    {
        Assert.Equal(expected, FaceClassifier.CategorizeGender(genderId));
    }

    // -----------------------------------------------------------------
    // CategorizeAge — every branch, transcribed from categorize_age's if-chain
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(0, 0, 2)]
    [InlineData(1, 3, 9)]
    [InlineData(2, 10, 19)]
    [InlineData(3, 20, 29)]
    [InlineData(4, 30, 39)]
    [InlineData(5, 40, 49)]
    [InlineData(6, 50, 59)]
    [InlineData(7, 60, 69)]
    [InlineData(8, 70, 100)] // falls through every branch to the final `return range(70, 100)`.
    [InlineData(9, 70, 100)]
    [InlineData(-1, 70, 100)]
    public void TestCategorizeAge(long ageId, int expectedStart, int expectedStop)
    {
        var age = FaceClassifier.CategorizeAge(ageId);
        Assert.Equal(expectedStart, age.Start.Value);
        Assert.Equal(expectedStop, age.End.Value);
    }

    // -----------------------------------------------------------------
    // CategorizeRace — every branch including the race_id in {3, 4} -> asian merge
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(0, Race.White)]
    [InlineData(1, Race.Black)]
    [InlineData(2, Race.Latino)]
    [InlineData(3, Race.Asian)]
    [InlineData(4, Race.Asian)]
    [InlineData(5, Race.Indian)]
    [InlineData(6, Race.Arabic)]
    [InlineData(7, Race.White)] // falls through every branch to the final `return 'white'`.
    [InlineData(-1, Race.White)]
    public void TestCategorizeRace(long raceId, Race expected)
    {
        Assert.Equal(expected, FaceClassifier.CategorizeRace(raceId));
    }

    // -----------------------------------------------------------------
    // PrepareInput — channel reversal (BGR -> RGB), /255, mean/std normalisation, all float32
    // -----------------------------------------------------------------

    [Fact]
    public void TestPrepareInputReversesChannelsNormalizesAndStandardizes()
    {
        using var crop = new Mat(1, 1, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(10, 20, 30)); // B=10, G=20, R=30

        var chw = FaceClassifier.PrepareInput(crop);

        Assert.Equal(3, chw.Length);
        var expectedR = (float)((30 / 255f - FaceClassifier.ModelMean[0]) / FaceClassifier.ModelStandardDeviation[0]);
        var expectedG = (float)((20 / 255f - FaceClassifier.ModelMean[1]) / FaceClassifier.ModelStandardDeviation[1]);
        var expectedB = (float)((10 / 255f - FaceClassifier.ModelMean[2]) / FaceClassifier.ModelStandardDeviation[2]);

        Assert.Equal((double)expectedR, (double)chw[0], 5);
        Assert.Equal((double)expectedG, (double)chw[1], 5);
        Assert.Equal((double)expectedB, (double)chw[2], 5);
    }

    [Fact]
    public void TestModelOptionsMatchPython()
    {
        // Python: create_static_model_set('full')['fairface'] -> template 'arcface_112_v2',
        // size (224, 224), mean [0.485, 0.456, 0.406], standard_deviation [0.229, 0.224, 0.225].
        Assert.Equal(WarpTemplate.Arcface112V2, FaceClassifier.ModelTemplate);
        Assert.Equal(224, FaceClassifier.ModelSize.Width);
        Assert.Equal(224, FaceClassifier.ModelSize.Height);
        Assert.Equal(new[] { 0.485f, 0.456f, 0.406f }, FaceClassifier.ModelMean);
        Assert.Equal(new[] { 0.229f, 0.224f, 0.225f }, FaceClassifier.ModelStandardDeviation);
    }

    // -----------------------------------------------------------------
    // ClassifyFace — end to end against the real fairface model
    // -----------------------------------------------------------------

    [ModelAndMediaFact("fairface.onnx")]
    public void TestClassifyFaceAgainstRealModel()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        Assert.NotNull(visionFrame);

        var faceLandmark5 = new float[5, 2]
        {
            { 380f, 380f },
            { 620f, 380f },
            { 500f, 500f },
            { 400f, 620f },
            { 600f, 620f },
        };

        using var session = new InferenceSession(ModelAndMediaFactAttribute.FindModelPath("fairface.onnx"));
        var (gender, age, race) = FaceClassifier.ClassifyFace(session, visionFrame!, faceLandmark5);

        Assert.True(gender is Gender.Female or Gender.Male);
        Assert.True(age.Start.Value is >= 0 and <= 70);
        Assert.True(age.End.Value is >= 2 and <= 100);
        Assert.True(Enum.IsDefined(race));
    }
}
