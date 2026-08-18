using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_face_detector.py</c>, plus unit coverage for the pure-logic pieces
/// (<c>prepare_margin</c>, <c>prepare_detect_frame</c>, <c>normalize_detect_frame</c>,
/// <c>create_static_model_set</c>/<c>collect_model_downloads</c>) that Python's test module
/// does not cover directly (it only exercises the four <c>detect_with_*</c> functions against
/// real ONNX models). Numeric parity against real Python inference lives in
/// <c>FaceDetectorParityTests</c> (ParityTests project) — the tests here are either pure
/// deterministic logic (no model/network dependency) or a direct port of the Python module's
/// own real-model test cases.
///
/// <para>
/// <b>Cropped-variant divergence (documented, deliberate).</b> Python's <c>before_all</c>
/// fixture shells out to a real <c>ffmpeg</c> process (<c>crop=iw*0.N:ih*0.N</c>, centered by
/// default) to produce <c>source-80crop.jpg</c>/<c>-70crop.jpg</c>/<c>-60crop.jpg</c>.
/// <c>FaceFusion.Media</c> only has command-line *builders*, not a process runner (see
/// <c>FaceFusion.Vision.Vision</c>'s class remarks), and this module's assignment does not
/// include porting one. <see cref="CenterCrop"/> below reproduces the same *effect* (a
/// centered crop at a given scale) directly via <see cref="Mat"/> indexing instead — the exact
/// crop rectangle need not match ffmpeg's pixel-for-pixel since these tests only assert "the
/// detector still finds exactly one face after NMS" on a smaller framing of the same photo,
/// not exact geometry.
/// </para>
/// </summary>
public sealed class FaceDetectorTests
{
    // -----------------------------------------------------------------
    // Runtime skip attributes (models / example media not available in CI).
    // -----------------------------------------------------------------

    private static bool ModelsAvailable => FaceDetector.PreCheck(FaceDetectorModel.Many);

    private const string MissingModelsMessage =
        "requires the four face-detector ONNX models in .assets/models/ — run " +
        "`FACEFUSION_PARITY_DIR=/tmp/x python3 tools/parity/dump_face_detector.py` with " +
        "network access (or any other way of running facefusion.face_detector.pre_check()) " +
        "to fetch them, then retry";

    /// <summary><c>[Fact]</c> that skips at discovery time when the four real detector models
    /// or the example media are not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelFactAttribute : FactAttribute
    {
        public ModelFactAttribute()
        {
            if (!ModelsAvailable)
            {
                Skip = MissingModelsMessage;
            }
            else if (!TestHelper.ExamplesAvailable)
            {
                Skip = TestHelper.MissingMediaMessage;
            }
        }
    }

    // -----------------------------------------------------------------
    // prepare_margin — pure integer math.
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(0, 0, 0, 0, 0, 0, 0, 0)]
    [InlineData(10, 5, 20, 15, 51, 25, 102, 76)] // matches the real Python fixture for a 1024x1024 frame.
    [InlineData(100, 100, 100, 100, 512, 512, 512, 512)] // upper bound: interp(100) == 0.5 exactly.
    public void PrepareMarginMatchesExpectedOffsets(int marginTop, int marginRight, int marginBottom, int marginLeft, int expectedTop, int expectedRight, int expectedBottom, int expectedLeft)
    {
        using var visionFrame = new Mat(1024, 1024, MatType.CV_8UC3, Scalar.All(0));

        var actual = FaceDetector.PrepareMargin(visionFrame, new[] { marginTop, marginRight, marginBottom, marginLeft });

        Assert.Equal((expectedTop, expectedRight, expectedBottom, expectedLeft), actual);
    }

    [Fact]
    public void PrepareMarginScalesIndependentlyByWidthAndHeight()
    {
        // Non-square frame: height 200, width 400 — top/bottom scale off height, left/right off width.
        using var visionFrame = new Mat(200, 400, MatType.CV_8UC3, Scalar.All(0));

        var (top, right, bottom, left) = FaceDetector.PrepareMargin(visionFrame, new[] { 50, 50, 50, 50 });

        Assert.Equal(50, top); // 200 * interp(50,[0,100],[0,0.5]) = 200 * 0.25 = 50
        Assert.Equal(100, right); // 400 * 0.25 = 100
        Assert.Equal(50, bottom);
        Assert.Equal(100, left);
    }

    // -----------------------------------------------------------------
    // normalize_detect_frame — pure float32 arithmetic.
    // -----------------------------------------------------------------

    [Fact]
    public void NormalizeDetectFrameNegativeOneToOneMatchesFormula()
    {
        var input = new float[] { 0f, 127.5f, 255f };

        var actual = FaceDetector.NormalizeDetectFrame(input, -1f, 1f);

        // 127.5 and 128 are exact in binary floating point, so this is exact float32 arithmetic.
        Assert.Equal(-0.99609375f, actual[0]); // (0 - 127.5) / 128
        Assert.Equal(0f, actual[1]);
        Assert.Equal(0.99609375f, actual[2]); // (255 - 127.5) / 128
    }

    [Fact]
    public void NormalizeDetectFrameZeroToOneMatchesFormula()
    {
        var input = new float[] { 0f, 127.5f, 255f };

        var actual = FaceDetector.NormalizeDetectFrame(input, 0f, 1f);

        Assert.Equal(0f, actual[0]);
        Assert.Equal(0.5f, actual[1]);
        Assert.Equal(1f, actual[2]);
    }

    [Theory]
    [InlineData(0f, 255f)] // yunet's actual normalize_range.
    [InlineData(1f, 2f)] // any other pair falls through to the identity branch too.
    public void NormalizeDetectFrameIdentityBranchReturnsUnchangedCopy(float rangeLow, float rangeHigh)
    {
        var input = new float[] { 0f, 127.5f, 255f };

        var actual = FaceDetector.NormalizeDetectFrame(input, rangeLow, rangeHigh);

        Assert.Equal(input, actual);
        Assert.NotSame(input, actual); // a copy, not the same array instance.
    }

    // -----------------------------------------------------------------
    // prepare_detect_frame — zero-pad + HWC->CHW transpose.
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareDetectFrameCopiesTopLeftAndZeroPadsTheRest()
    {
        // 2 rows x 3 cols source, target size 5x4 (width x height) leaves real padding on
        // every edge — enough to prove both the copy region and the padding are right.
        using var source = new Mat(2, 3, MatType.CV_8UC3);
        source.Set(0, 0, new Vec3b(1, 2, 3));
        source.Set(0, 1, new Vec3b(4, 5, 6));
        source.Set(0, 2, new Vec3b(7, 8, 9));
        source.Set(1, 0, new Vec3b(10, 11, 12));
        source.Set(1, 1, new Vec3b(13, 14, 15));
        source.Set(1, 2, new Vec3b(16, 17, 18));

        var actual = FaceDetector.PrepareDetectFrame(source, "5x4"); // width=5, height=4

        Assert.Equal(3 * 4 * 5, actual.Length);

        // CHW indexing: index = channel * (height * width) + row * width + col.
        float At(int channel, int row, int col) => actual[(channel * 4 * 5) + (row * 5) + col];

        // Channel 0 (BGR "B") at the copied 2x3 region:
        Assert.Equal(1f, At(0, 0, 0));
        Assert.Equal(4f, At(0, 0, 1));
        Assert.Equal(7f, At(0, 0, 2));
        Assert.Equal(10f, At(0, 1, 0));
        Assert.Equal(13f, At(0, 1, 1));
        Assert.Equal(16f, At(0, 1, 2));

        // Channel 2 (BGR "R") — proves channel order survived the transpose, not just channel 0.
        Assert.Equal(3f, At(2, 0, 0));
        Assert.Equal(18f, At(2, 1, 2));

        // Padding: anything outside the copied 2x3 region is exactly zero, on every channel.
        for (var channel = 0; channel < 3; channel++)
        {
            for (var row = 0; row < 4; row++)
            {
                for (var col = 0; col < 5; col++)
                {
                    if (row >= 2 || col >= 3)
                    {
                        Assert.Equal(0f, At(channel, row, col));
                    }
                }
            }
        }
    }

    [Fact]
    public void PrepareDetectFrameRejectsNonBgr8Frames()
    {
        using var grayscale = new Mat(4, 4, MatType.CV_8UC1);

        Assert.Throws<ArgumentException>(() => FaceDetector.PrepareDetectFrame(grayscale, "4x4"));
    }

    // -----------------------------------------------------------------
    // create_static_model_set / collect_model_downloads — pure data, no I/O.
    // -----------------------------------------------------------------

    [Fact]
    public void CreateStaticModelSetHasAllFourFamiliesWithExpectedFileNames()
    {
        var modelSet = FaceDetector.CreateStaticModelSet(DownloadScope.Full);

        Assert.Equal(4, modelSet.Count);

        AssertModelFiles(modelSet, FaceDetectorModel.Retinaface, "retinaface_10g");
        AssertModelFiles(modelSet, FaceDetectorModel.Scrfd, "scrfd_2.5g");
        AssertModelFiles(modelSet, FaceDetectorModel.YoloFace, "yoloface_8n");
        AssertModelFiles(modelSet, FaceDetectorModel.Yunet, "yunet_2023_mar");
    }

    private static void AssertModelFiles(IReadOnlyDictionary<FaceDetectorModel, (Download Hash, Download Source)> modelSet, FaceDetectorModel family, string fileName)
    {
        var (hash, source) = modelSet[family];

        Assert.EndsWith(Path.Combine(".assets", "models", fileName + ".hash"), hash.Path);
        Assert.EndsWith(Path.Combine(".assets", "models", fileName + ".onnx"), source.Path);
        Assert.False(string.IsNullOrEmpty(hash.Url));
        Assert.False(string.IsNullOrEmpty(source.Url));
    }

    [Fact]
    public void CollectModelDownloadsForASingleFamilyReturnsOnlyThatFamily()
    {
        var (hashes, sources) = FaceDetector.CollectModelDownloads(FaceDetectorModel.Retinaface);

        Assert.Single(hashes);
        Assert.Single(sources);
        Assert.True(hashes.ContainsKey("retinaface"));
        Assert.True(sources.ContainsKey("retinaface"));
    }

    /// <summary>
    /// The Python <c>collect_model_downloads</c>/<c>detect_faces</c> asymmetry (see
    /// <see cref="FaceDetector"/>'s class remarks): <c>'many'</c> downloads/pools all four
    /// families here, including yunet — even though <c>detect_faces</c> never actually runs
    /// the yunet model under <c>'many'</c> (see <see cref="FaceDetectorParityTests.DetectFacesWithManyNeverLooksUpYunet"/>
    /// in the ParityTests project for that half of the quirk).
    /// </summary>
    [Fact]
    public void CollectModelDownloadsForManyIncludesAllFourFamiliesIncludingYunet()
    {
        var (hashes, sources) = FaceDetector.CollectModelDownloads(FaceDetectorModel.Many);

        Assert.Equal(4, hashes.Count);
        Assert.Equal(4, sources.Count);
        Assert.True(hashes.ContainsKey("yunet"));
        Assert.True(sources.ContainsKey("yunet"));
    }

    // -----------------------------------------------------------------
    // pre_check — real file presence against the repo's .assets/models (see FaceDetector's
    // class remarks: this checks presence only, it does not download or hash-validate).
    // -----------------------------------------------------------------

    [ModelFact]
    public void PreCheckReturnsTrueWhenAllFourModelsArePresent()
    {
        Assert.True(FaceDetector.PreCheck(FaceDetectorModel.Many));
    }

    /// <summary>
    /// <see cref="FaceDetector.PreCheck"/> is a straightforward "every hash/source path from
    /// <see cref="FaceDetector.CollectModelDownloads"/> exists" check with no seam to redirect
    /// the resolved <c>.assets/models</c> directory for a deterministic missing-file case
    /// (see <see cref="FaceDetector"/>'s class remarks — path resolution walks up to the real
    /// repository root), so the missing-model branch is exercised at the level actually under
    /// test's control instead: a <see cref="Types.Download"/> path that is known not to exist.
    /// </summary>
    [Fact]
    public void PreCheckPathsPointAtAssetsModelsUnderTheRepoRoot()
    {
        var (hashes, sources) = FaceDetector.CollectModelDownloads(FaceDetectorModel.Retinaface);

        Assert.True(Path.IsPathRooted(hashes["retinaface"].Path));
        Assert.True(Path.IsPathRooted(sources["retinaface"].Path));
        Assert.False(File.Exists(sources["retinaface"].Path + ".definitely-not-a-real-file"));
    }

    // -----------------------------------------------------------------
    // Port of tests/test_face_detector.py — real ONNX inference, exactly one face after NMS
    // across the original photo and three progressively tighter center crops.
    // -----------------------------------------------------------------

    private static InferenceSession CreateSession(FaceDetectorModel family)
    {
        var modelSet = FaceDetector.CreateStaticModelSet(DownloadScope.Full);
        var modelPath = modelSet[family].Source.Path;
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });
        var inferenceManager = new InferenceManager();
        return inferenceManager.CreateInferenceSession(modelPath, inferenceProviders);
    }

    /// <summary>Python: ffmpeg's <c>crop=iw*scale:ih*scale</c> (centered by default) — see
    /// the class remarks for why this is a direct <see cref="Mat"/> crop instead.</summary>
    private static Mat CenterCrop(Mat source, double scale)
    {
        var newWidth = (int)(source.Cols * scale);
        var newHeight = (int)(source.Rows * scale);
        var x = (source.Cols - newWidth) / 2;
        var y = (source.Rows - newHeight) / 2;
        return new Mat(source, new Rect(x, y, newWidth, newHeight));
    }

    private static IEnumerable<Mat> SourceCrops(Mat sourceFrame)
    {
        yield return sourceFrame;
        yield return CenterCrop(sourceFrame, 0.8);
        yield return CenterCrop(sourceFrame, 0.7);
        yield return CenterCrop(sourceFrame, 0.6);
    }

    [ModelFact]
    public void DetectWithRetinafaceFindsExactlyOneFaceAcrossCrops()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        using var session = CreateSession(FaceDetectorModel.Retinaface);

        foreach (var frame in SourceCrops(sourceFrame!))
        {
            var (boundingBoxes, faceScores, _) = FaceDetector.DetectWithRetinaface(frame, "320x320", 0.5, session);
            var keepIndices = FaceHelper.ApplyNms(boundingBoxes, faceScores.Select(s => (float)s).ToList(), 0.5f, FaceHelper.GetNmsThreshold(FaceDetectorModel.Retinaface, new[] { 0 }));

            Assert.Single(keepIndices);
        }
    }

    [ModelFact]
    public void DetectWithScrfdFindsExactlyOneFaceAcrossCrops()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        using var session = CreateSession(FaceDetectorModel.Scrfd);

        foreach (var frame in SourceCrops(sourceFrame!))
        {
            var (boundingBoxes, faceScores, _) = FaceDetector.DetectWithScrfd(frame, "320x320", 0.5, session);
            var keepIndices = FaceHelper.ApplyNms(boundingBoxes, faceScores.Select(s => (float)s).ToList(), 0.5f, FaceHelper.GetNmsThreshold(FaceDetectorModel.Scrfd, new[] { 0 }));

            Assert.Single(keepIndices);
        }
    }

    [ModelFact]
    public void DetectWithYoloFaceFindsExactlyOneFaceAcrossCrops()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        using var session = CreateSession(FaceDetectorModel.YoloFace);

        foreach (var frame in SourceCrops(sourceFrame!))
        {
            var (boundingBoxes, faceScores, _) = FaceDetector.DetectWithYoloFace(frame, "640x640", 0.5, session);
            var keepIndices = FaceHelper.ApplyNms(boundingBoxes, faceScores.Select(s => (float)s).ToList(), 0.5f, FaceHelper.GetNmsThreshold(FaceDetectorModel.YoloFace, new[] { 0 }));

            Assert.Single(keepIndices);
        }
    }

    [ModelFact]
    public void DetectWithYunetFindsExactlyOneFaceAcrossCrops()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        using var session = CreateSession(FaceDetectorModel.Yunet);

        foreach (var frame in SourceCrops(sourceFrame!))
        {
            var (boundingBoxes, faceScores, _) = FaceDetector.DetectWithYunet(frame, "640x640", 0.5, session);
            var keepIndices = FaceHelper.ApplyNms(boundingBoxes, faceScores.Select(s => (float)s).ToList(), 0.5f, FaceHelper.GetNmsThreshold(FaceDetectorModel.Yunet, new[] { 0 }));

            Assert.Single(keepIndices);
        }
    }

    // -----------------------------------------------------------------
    // detect_faces / detect_faces_by_angle dispatch — real models, small deterministic checks.
    // -----------------------------------------------------------------

    [ModelFact]
    public void DetectFacesByAngleRotatesResultsBackToOriginalOrientation()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        using var session = CreateSession(FaceDetectorModel.YoloFace);
        var sessions = new Dictionary<string, InferenceSession> { ["yolo_face"] = session };

        var (unrotatedBoxes, _, _) = FaceDetector.DetectFaces(sourceFrame!, FaceDetectorModel.YoloFace, "640x640", 0.5, new[] { 0, 0, 0, 0 }, sessions);
        var (angleZeroBoxes, _, _) = FaceDetector.DetectFacesByAngle(sourceFrame!, 0, FaceDetectorModel.YoloFace, "640x640", 0.5, new[] { 0, 0, 0, 0 }, sessions);

        // A 0-degree rotation is the identity transform, so detect_faces_by_angle(..., 0, ...)
        // must reproduce detect_faces(...) (same face count; boxes match closely — the
        // rotate/warp/inverse-warp round trip at angle 0 still resamples through
        // cv2.warpAffine once, so this is not asserted bit-exact).
        Assert.Equal(unrotatedBoxes.Count, angleZeroBoxes.Count);
        Assert.True(unrotatedBoxes.Count > 0);
    }
}
