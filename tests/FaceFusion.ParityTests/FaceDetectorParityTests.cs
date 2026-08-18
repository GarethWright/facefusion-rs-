using System.Text.Json;
using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Parity;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for the face detector (docs/PARITY_HARNESS.md). Ground truth was
/// dumped from the real <c>facefusion.face_detector</c> module (via
/// <c>tools/parity/dump_face_detector.py</c>), running real ONNX inference against the real
/// example media — not synthetic data.
///
/// <para>
/// <b>Fixture layout</b> — <c>fixtures/face_detector/</c>:
/// <list type="bullet">
/// <item><description><c>model_input/source_640x640/{raw,normalized_m1_1,normalized_0_1}.npy</c> —
/// the model-input tensor, the highest-value dump per the assignment: if this matches, any
/// downstream mismatch is ONNX Runtime's own arithmetic (expect ~0 divergence); if it does
/// not, the bug is in <see cref="FaceDetector.PrepareDetectFrame"/>/<see cref="FaceDetector.NormalizeDetectFrame"/>.
/// De-duplicated across families (see <c>dump_face_detector.py</c>'s docstring) since
/// <c>prepare_detect_frame</c> does not depend on the family and three of the four families
/// share a normalize_range with another or with the identity case.</description></item>
/// <item><description><c>&lt;family&gt;/source_640x640/{bounding_boxes,face_scores,face_landmarks_5}.npy</c> —
/// <c>detect_with_&lt;family&gt;</c> raw output, source.jpg, for all four families.</description></item>
/// <item><description><c>&lt;family&gt;/source_640x640/detect_faces_*.npy</c> — end-to-end
/// <c>detect_faces()</c> output (adds the margin-shift/normalize step; default zero margin).</description></item>
/// <item><description><c>retinaface/source_320x320/*</c> — a second face_detector_size,
/// exercising the <c>restrict_frame</c> ratio-scaling path (ratio != 1).</description></item>
/// <item><description><c>yolo_face/video_frame_0_640x640/*</c> plus <c>video/target_240p_frame_0.npy</c> —
/// one video-frame case, end to end. The frame pixels are dumped too (Python-decoded bytes)
/// rather than re-decoded via <see cref="OpenCvSharp.VideoCapture"/> in C#, isolating this
/// case to detector-math parity rather than the documented OpenCvSharp-vs-ffmpeg decode
/// divergence (see VisionParityTests) — and this is also the only case with real zero-padding
/// in <see cref="FaceDetector.PrepareDetectFrame"/> (426x226 restricted into a 640x640 square
/// leaves unfilled rows), since source.jpg is exactly square at every tested size.</description></item>
/// <item><description><c>prepare_margin_cases.json</c> — <c>prepare_margin()</c> case table
/// against source.jpg's real dimensions.</description></item>
/// </list>
/// </para>
///
/// <para>
/// <b>Tolerances</b> (per PARITY_HARNESS.md's "choosing a tolerance"): the model-input tensor
/// comparisons use <c>rtol = atol = 0</c> — pure data movement (pad/copy/transpose/cast), no
/// arithmetic beyond a dtype cast, so this must be bit-exact. The per-family bounding-box/score/
/// landmark comparisons use a small real epsilon (<see cref="BoxRelativeTolerance"/>/
/// <see cref="BoxAbsoluteTolerance"/>) rather than 0: <see cref="FaceHelper.CreateStaticAnchors"/>'s
/// float32 anchor grid diverges in principle from Python's actual int64 anchor grid combined
/// with float64 decode arithmetic (see the divergence documented on <see cref="FaceDetector"/>),
/// which is PARITY_HARNESS.md's "managed float math ... a real epsilon belongs here" case, not
/// the "ORT does the arithmetic" case. In measurement (see the port report) this divergence
/// turns out tiny for every family, including the anchor-based ones — the anchor grid's values
/// are small integers times a small feature stride, all exactly representable in float32, so
/// the theoretical float64-vs-float32 gap does not show up in practice at these coordinate
/// magnitudes; the chosen tolerance sits about an order of magnitude above the measured worst
/// case (~2e-7 relative, ~7e-5 absolute) rather than being padded defensively.
/// </para>
/// </summary>
public sealed class FaceDetectorParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "face_detector");

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    // -----------------------------------------------------------------
    // Model availability (real .onnx files under .assets/models — not fetched in CI).
    // -----------------------------------------------------------------

    private static bool ModelsAvailable => FaceDetector.PreCheck(FaceDetectorModel.Many);

    private const string MissingModelsMessage =
        "requires the four face-detector ONNX models in .assets/models/ (retinaface_10g, " +
        "scrfd_2.5g, yoloface_8n, yunet_2023_mar) — run `FACEFUSION_PARITY_DIR=/tmp/x " +
        "python3 tools/parity/dump_face_detector.py` with network access (or any other way " +
        "of running facefusion.face_detector.pre_check()) to fetch them, then retry";

    private static bool SourceImageAvailable => File.Exists(SourceImage);

    private const string MissingSourceImageMessage =
        "requires /tmp/facefusion-test-examples/source.jpg — run tools/parity/fetch_examples.sh, then retry";

    /// <summary><c>[Fact]</c> that skips at discovery time when source.jpg is not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ImageFactAttribute : FactAttribute
    {
        public ImageFactAttribute()
        {
            if (!SourceImageAvailable)
            {
                Skip = MissingSourceImageMessage;
            }
        }
    }

    /// <summary><c>[Theory]</c> that skips at discovery time when source.jpg is not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ImageTheoryAttribute : TheoryAttribute
    {
        public ImageTheoryAttribute()
        {
            if (!SourceImageAvailable)
            {
                Skip = MissingSourceImageMessage;
            }
        }
    }

    /// <summary><c>[Fact]</c> that skips at discovery time when the four real detector models (and source.jpg) are not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelFactAttribute : FactAttribute
    {
        public ModelFactAttribute()
        {
            if (!ModelsAvailable)
            {
                Skip = MissingModelsMessage;
            }
            else if (!SourceImageAvailable)
            {
                Skip = MissingSourceImageMessage;
            }
        }
    }

    /// <summary><c>[Theory]</c> that skips at discovery time when the four real detector models (and source.jpg) are not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelTheoryAttribute : TheoryAttribute
    {
        public ModelTheoryAttribute()
        {
            if (!ModelsAvailable)
            {
                Skip = MissingModelsMessage;
            }
            else if (!SourceImageAvailable)
            {
                Skip = MissingSourceImageMessage;
            }
        }
    }

    /// <summary><c>[Fact]</c> that skips at discovery time when the four real detector models
    /// are not present — unlike <see cref="ModelFactAttribute"/>, does not also require
    /// source.jpg (for cases driven entirely by a committed .npy frame fixture).</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelOnlyFactAttribute : FactAttribute
    {
        public ModelOnlyFactAttribute()
        {
            if (!ModelsAvailable)
            {
                Skip = MissingModelsMessage;
            }
        }
    }

    // -----------------------------------------------------------------
    // Shared inference sessions (one per family, loaded once for the whole test run).
    // -----------------------------------------------------------------

    private static readonly object SessionLock = new();
    private static readonly Dictionary<FaceDetectorModel, InferenceSession> Sessions = new();

    private static InferenceSession GetSession(FaceDetectorModel family)
    {
        lock (SessionLock)
        {
            if (Sessions.TryGetValue(family, out var existing))
            {
                return existing;
            }

            var modelSet = FaceDetector.CreateStaticModelSet(DownloadScope.Full);
            var modelPath = modelSet[family].Source.Path;
            var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });
            var inferenceManager = new InferenceManager();
            var session = inferenceManager.CreateInferenceSession(modelPath, inferenceProviders);
            Sessions[family] = session;
            return session;
        }
    }

    // -----------------------------------------------------------------
    // model_input — the highest-value dump. Exact (0/0) tolerance: pure data movement.
    // -----------------------------------------------------------------

    [ImageFact]
    public void PrepareDetectFrameMatchesPythonExactly()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var resolution = FaceFusion.Vision.Vision.UnpackResolution("640x640");
        using var tempVisionFrame = FaceFusion.Vision.Vision.RestrictFrame(sourceFrame!, resolution);

        var actual = FaceDetector.PrepareDetectFrame(tempVisionFrame, "640x640");
        var actualAsDouble = Array.ConvertAll(actual, x => (double)x);

        var expected = NpyReader.Load(Path.Combine(FixturesDirectory, "model_input", "source_640x640", "raw.npy")).AsDoubles();

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    [ImageTheory]
    [InlineData(-1f, 1f, "normalized_m1_1")]
    [InlineData(0f, 1f, "normalized_0_1")]
    public void NormalizeDetectFrameMatchesPythonExactly(float rangeLow, float rangeHigh, string fixtureName)
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var resolution = FaceFusion.Vision.Vision.UnpackResolution("640x640");
        using var tempVisionFrame = FaceFusion.Vision.Vision.RestrictFrame(sourceFrame!, resolution);

        var raw = FaceDetector.PrepareDetectFrame(tempVisionFrame, "640x640");
        var actual = FaceDetector.NormalizeDetectFrame(raw, rangeLow, rangeHigh);
        var actualAsDouble = Array.ConvertAll(actual, x => (double)x);

        var expected = NpyReader.Load(Path.Combine(FixturesDirectory, "model_input", "source_640x640", fixtureName + ".npy")).AsDoubles();

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // prepare_margin — pure integer math, exact.
    // -----------------------------------------------------------------

    [ImageFact]
    public void PrepareMarginMatchesPython()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, "prepare_margin_cases.json")));

        foreach (var caseElement in document.RootElement.EnumerateArray())
        {
            var marginElement = caseElement.GetProperty("margin");
            var margin = new[] { marginElement[0].GetInt32(), marginElement[1].GetInt32(), marginElement[2].GetInt32(), marginElement[3].GetInt32() };
            var resultElement = caseElement.GetProperty("result");
            var expected = (Top: resultElement[0].GetInt32(), Right: resultElement[1].GetInt32(), Bottom: resultElement[2].GetInt32(), Left: resultElement[3].GetInt32());

            var actual = FaceDetector.PrepareMargin(sourceFrame!, margin);

            Assert.Equal(expected, actual);
        }
    }

    // -----------------------------------------------------------------
    // Per-family detect_with_* — real ONNX inference, source.jpg @ 640x640.
    // -----------------------------------------------------------------

    // Measured (see the port report): every family's max relative difference against the
    // Python fixture is ~1e-7-2e-7 and max absolute difference ~3e-5-7e-5 — plain float32
    // rounding noise, not a meaningful anchor float32-vs-float64 divergence (the anchor
    // grid's values are small integers × a small feature stride, all exactly representable
    // in float32, so the theoretical float64-vs-float32 gap this class's remarks describe
    // does not actually show up at these magnitudes). rtol/atol below sit about an order of
    // magnitude above the measured worst case for every family, not padded further.
    private const double BoxRelativeTolerance = 1e-4;
    private const double BoxAbsoluteTolerance = 1e-3;

    public static IEnumerable<object[]> FamilyCases()
    {
        yield return new object[] { FaceDetectorModel.Retinaface, "retinaface", BoxRelativeTolerance, BoxAbsoluteTolerance };
        yield return new object[] { FaceDetectorModel.Scrfd, "scrfd", BoxRelativeTolerance, BoxAbsoluteTolerance };
        yield return new object[] { FaceDetectorModel.YoloFace, "yolo_face", BoxRelativeTolerance, BoxAbsoluteTolerance };
        yield return new object[] { FaceDetectorModel.Yunet, "yunet", BoxRelativeTolerance, BoxAbsoluteTolerance };
    }

    [ModelTheory]
    [MemberData(nameof(FamilyCases))]
    public void DetectWithFamilyMatchesPython(FaceDetectorModel family, string wireName, double relativeTolerance, double absoluteTolerance)
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var session = GetSession(family);
        var (boundingBoxes, faceScores, faceLandmarks5) = family switch
        {
            FaceDetectorModel.Retinaface => FaceDetector.DetectWithRetinaface(sourceFrame!, "640x640", 0.5, session),
            FaceDetectorModel.Scrfd => FaceDetector.DetectWithScrfd(sourceFrame!, "640x640", 0.5, session),
            FaceDetectorModel.YoloFace => FaceDetector.DetectWithYoloFace(sourceFrame!, "640x640", 0.5, session),
            FaceDetectorModel.Yunet => FaceDetector.DetectWithYunet(sourceFrame!, "640x640", 0.5, session),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };

        var fixtureDirectory = Path.Combine(FixturesDirectory, wireName, "source_640x640");
        AssertBoundingBoxesMatch(boundingBoxes, Path.Combine(fixtureDirectory, "bounding_boxes.npy"), relativeTolerance, absoluteTolerance);
        AssertScoresMatch(faceScores, Path.Combine(fixtureDirectory, "face_scores.npy"), relativeTolerance, absoluteTolerance);
        AssertLandmarksMatch(faceLandmarks5, Path.Combine(fixtureDirectory, "face_landmarks_5.npy"), relativeTolerance, absoluteTolerance);
    }

    // -----------------------------------------------------------------
    // End-to-end detect_faces() per family — adds normalize_bounding_box + the (zero) margin
    // shift on top of detect_with_<family>.
    // -----------------------------------------------------------------

    [ModelTheory]
    [MemberData(nameof(FamilyCases))]
    public void DetectFacesEndToEndMatchesPython(FaceDetectorModel family, string wireName, double relativeTolerance, double absoluteTolerance)
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var sessions = new Dictionary<string, InferenceSession> { [wireName] = GetSession(family) };
        var (boundingBoxes, faceScores, faceLandmarks5) = FaceDetector.DetectFaces(
            sourceFrame!, family, "640x640", 0.5, new[] { 0, 0, 0, 0 }, sessions);

        var fixtureDirectory = Path.Combine(FixturesDirectory, wireName, "source_640x640");
        AssertBoundingBoxesMatch(boundingBoxes, Path.Combine(fixtureDirectory, "detect_faces_bounding_boxes.npy"), relativeTolerance, absoluteTolerance);
        AssertScoresMatch(faceScores, Path.Combine(fixtureDirectory, "detect_faces_face_scores.npy"), relativeTolerance, absoluteTolerance);
        AssertLandmarksMatch(faceLandmarks5, Path.Combine(fixtureDirectory, "detect_faces_face_landmarks_5.npy"), relativeTolerance, absoluteTolerance);
    }

    // -----------------------------------------------------------------
    // A second face_detector_size (retinaface, 320x320) — exercises restrict_frame's
    // ratio-scaling path (ratio_width/ratio_height != 1).
    // -----------------------------------------------------------------

    [ModelFact]
    public void DetectWithRetinafaceAtAlternateSizeMatchesPython()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var session = GetSession(FaceDetectorModel.Retinaface);
        var (boundingBoxes, faceScores, faceLandmarks5) = FaceDetector.DetectWithRetinaface(sourceFrame!, "320x320", 0.5, session);

        var fixtureDirectory = Path.Combine(FixturesDirectory, "retinaface", "source_320x320");
        AssertBoundingBoxesMatch(boundingBoxes, Path.Combine(fixtureDirectory, "bounding_boxes.npy"), BoxRelativeTolerance, BoxAbsoluteTolerance);
        AssertScoresMatch(faceScores, Path.Combine(fixtureDirectory, "face_scores.npy"), BoxRelativeTolerance, BoxAbsoluteTolerance);
        AssertLandmarksMatch(faceLandmarks5, Path.Combine(fixtureDirectory, "face_landmarks_5.npy"), BoxRelativeTolerance, BoxAbsoluteTolerance);
    }

    // -----------------------------------------------------------------
    // One video-frame case, end to end (also the only case with real zero-padding in
    // PrepareDetectFrame — see class remarks).
    // -----------------------------------------------------------------

    [ModelOnlyFact]
    public void DetectFacesOnVideoFrameMatchesPython()
    {
        using var frame = LoadFrameFixtureAsMat(Path.Combine(FixturesDirectory, "video", "target_240p_frame_0.npy"));

        var session = GetSession(FaceDetectorModel.YoloFace);
        var sessions = new Dictionary<string, InferenceSession> { ["yolo_face"] = session };
        var (boundingBoxes, faceScores, faceLandmarks5) = FaceDetector.DetectFaces(
            frame, FaceDetectorModel.YoloFace, "640x640", 0.5, new[] { 0, 0, 0, 0 }, sessions);

        var fixtureDirectory = Path.Combine(FixturesDirectory, "yolo_face", "video_frame_0_640x640");
        AssertBoundingBoxesMatch(boundingBoxes, Path.Combine(fixtureDirectory, "detect_faces_bounding_boxes.npy"), BoxRelativeTolerance, BoxAbsoluteTolerance);
        AssertScoresMatch(faceScores, Path.Combine(fixtureDirectory, "detect_faces_face_scores.npy"), BoxRelativeTolerance, BoxAbsoluteTolerance);
        AssertLandmarksMatch(faceLandmarks5, Path.Combine(fixtureDirectory, "detect_faces_face_landmarks_5.npy"), BoxRelativeTolerance, BoxAbsoluteTolerance);
    }

    /// <summary>
    /// The model-input tensor for the video-frame case, verified separately for the same
    /// reason as the source.jpg case: real zero-padding is exercised here (426x226 restricted
    /// into a 640x640 square leaves unfilled rows/columns), which the square source.jpg case
    /// never exercises.
    /// </summary>
    [Fact]
    public void PrepareDetectFrameOnVideoFrameZeroPadsCorrectly()
    {
        using var frame = LoadFrameFixtureAsMat(Path.Combine(FixturesDirectory, "video", "target_240p_frame_0.npy"));

        var resolution = FaceFusion.Vision.Vision.UnpackResolution("640x640");
        using var tempVisionFrame = FaceFusion.Vision.Vision.RestrictFrame(frame, resolution);
        var actual = FaceDetector.PrepareDetectFrame(tempVisionFrame, "640x640");

        // Every pixel outside the copied region must be exactly zero (numpy.zeros() default),
        // and the region actually covered by the frame must be non-trivial (i.e. this really
        // does exercise padding, not an accidental exact fit).
        var height = 640;
        var width = 640;
        var copiedHeight = tempVisionFrame.Rows;
        var copiedWidth = tempVisionFrame.Cols;
        Assert.True(copiedHeight < height || copiedWidth < width, "expected the video-frame case to require padding.");

        for (var c = 0; c < 3; c++)
        {
            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    if (h >= copiedHeight || w >= copiedWidth)
                    {
                        Assert.Equal(0f, actual[(c * height * width) + (h * width) + w]);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // The 'many'-excludes-'yunet' dispatch quirk (see FaceDetector's class remarks) — proven
    // by NOT providing a yunet session at all: if DetectFaces(Many, ...) ever looked it up,
    // this would throw KeyNotFoundException.
    // -----------------------------------------------------------------

    [ModelFact]
    public void DetectFacesWithManyNeverLooksUpYunet()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var sessions = new Dictionary<string, InferenceSession>
        {
            ["retinaface"] = GetSession(FaceDetectorModel.Retinaface),
            ["scrfd"] = GetSession(FaceDetectorModel.Scrfd),
            ["yolo_face"] = GetSession(FaceDetectorModel.YoloFace),
            // Deliberately no "yunet" entry.
        };

        var exception = Record.Exception(() => FaceDetector.DetectFaces(sourceFrame!, FaceDetectorModel.Many, "640x640", 0.5, new[] { 0, 0, 0, 0 }, sessions));

        Assert.Null(exception);
    }

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    private static void AssertBoundingBoxesMatch(IReadOnlyList<float[]> boundingBoxes, string fixturePath, double relativeTolerance, double absoluteTolerance)
    {
        var expected = NpyReader.Load(fixturePath);
        Assert.Equal(new[] { boundingBoxes.Count, 4 }, expected.Shape);

        var actual = new double[boundingBoxes.Count * 4];
        for (var i = 0; i < boundingBoxes.Count; i++)
        {
            for (var c = 0; c < 4; c++)
            {
                actual[(i * 4) + c] = boundingBoxes[i][c];
            }
        }

        var result = TensorComparison.Compare(actual, expected.AsDoubles(), relativeTolerance, absoluteTolerance);
        Assert.True(result.Passed, $"{fixturePath}: {result.Describe()}");
    }

    private static void AssertScoresMatch(IReadOnlyList<double> faceScores, string fixturePath, double relativeTolerance, double absoluteTolerance)
    {
        var expected = NpyReader.Load(fixturePath);
        Assert.Equal(new[] { faceScores.Count }, expected.Shape);

        var result = TensorComparison.Compare(faceScores.ToArray(), expected.AsDoubles(), relativeTolerance, absoluteTolerance);
        Assert.True(result.Passed, $"{fixturePath}: {result.Describe()}");
    }

    private static void AssertLandmarksMatch(IReadOnlyList<float[,]> faceLandmarks5, string fixturePath, double relativeTolerance, double absoluteTolerance)
    {
        var expected = NpyReader.Load(fixturePath);
        Assert.Equal(new[] { faceLandmarks5.Count, 5, 2 }, expected.Shape);

        var actual = new double[faceLandmarks5.Count * 5 * 2];
        var index = 0;
        foreach (var landmark in faceLandmarks5)
        {
            for (var k = 0; k < 5; k++)
            {
                actual[index++] = landmark[k, 0];
                actual[index++] = landmark[k, 1];
            }
        }

        var result = TensorComparison.Compare(actual, expected.AsDoubles(), relativeTolerance, absoluteTolerance);
        Assert.True(result.Passed, $"{fixturePath}: {result.Describe()}");
    }

    /// <summary>Builds a CV_8UC3 <see cref="Mat"/> directly from a dumped (H, W, 3) uint8 .npy fixture.</summary>
    private static Mat LoadFrameFixtureAsMat(string fixturePath)
    {
        var array = NpyReader.Load(fixturePath);
        var shape = array.Shape;
        Assert.Equal(3, shape.Count);
        Assert.Equal(3, shape[2]);
        Assert.Equal("uint8", array.DType);

        var height = shape[0];
        var width = shape[1];
        var mat = new Mat(height, width, MatType.CV_8UC3);
        var bytes = array.RawData.ToArray();
        System.Runtime.InteropServices.Marshal.Copy(bytes, 0, mat.Data, bytes.Length);
        return mat;
    }
}
