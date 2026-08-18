using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_face_creator.py</c>.
///
/// <para>
/// <b>State-manager settings, taken as explicit locals (per PORT_CONVENTIONS.md rule 5).</b>
/// Python's <c>before_all</c> fixture calls <c>state_manager.init_item(...)</c> for
/// <c>execution_device_ids</c>, <c>execution_providers</c>, <c>download_providers</c>,
/// <c>face_detector_angles</c> = <c>[0]</c>, <c>face_detector_model</c> = <c>'many'</c>,
/// <c>face_detector_size</c> = <c>'640x640'</c>, <c>face_detector_margin</c> = <c>(0,0,0,0)</c>,
/// <c>face_detector_score</c> = <c>0.5</c>, <c>face_landmarker_model</c> = <c>'many'</c>,
/// <c>face_landmarker_score</c> = <c>0.5</c>. Those become the local constants below, passed
/// explicitly into every <see cref="FaceCreator"/> call.
/// </para>
///
/// <para>
/// <b>Not ported: the crop-scale fixture generation.</b> Python's <c>before_all</c> also
/// ffmpeg-crops <c>source.jpg</c> into <c>source-80crop.jpg</c>/<c>-70crop.jpg</c>/
/// <c>-60crop.jpg</c>, but none of the four test bodies in <c>test_face_creator.py</c>
/// (<c>test_get_one_face</c>, <c>test_get_many_faces</c>, <c>test_refill_faces</c>,
/// <c>test_average_face_geometry</c>) reference those files — they appear to be dead fixture
/// setup in the Python suite. Not reproducing unused setup is not a coverage loss; if a later
/// Python test starts using them, port the crop generation then.
/// </para>
///
/// <para>
/// <b>No download-backed model pool (per <see cref="FaceDetector"/>/<see cref="FaceCreator"/>'s
/// own documented divergence).</b> Sessions are loaded directly from
/// <c>.assets/models/*.onnx</c> via <see cref="ModelFactAttribute"/>, which skips cleanly
/// (rather than failing) when a model is not present — matching how
/// <c>FaceFusion.ParityTests.FaceAnalysisParityTests</c> gates its own end-to-end tests, and
/// satisfying PORT_CONVENTIONS.md rule 2 ("mark it Skip, do not silently drop coverage") for
/// this environment-dependent test.
/// </para>
/// </summary>
/// <summary>
/// Loads every session <see cref="FaceCreatorTests"/> needs exactly once and shares it across
/// all test methods in that class (xUnit creates a fresh test-class instance per <c>[Fact]</c>
/// by default; an <c>IClassFixture</c> is the standard way to share expensive setup — these
/// are ~500 MB of ONNX models — across them instead). Loosely stands in for the module-scoped
/// <c>before_all</c>/module-lifetime session pool Python's test gets implicitly from
/// <c>inference_manager</c>'s <c>lru_cache</c>-backed pool.
/// </summary>
public sealed class FaceCreatorSessionFixture : IDisposable
{
    private static readonly string[] RequiredModels =
    {
        "retinaface_10g.onnx", "scrfd_2.5g.onnx", "yoloface_8n.onnx",
        "2dfan4.onnx", "peppa_wutz.onnx", "fan_68_5.onnx",
        "arcface_w600k_r50.onnx", "fairface.onnx",
    };

    internal FaceCreatorTests.Sessions? Sessions { get; }

    public FaceCreatorSessionFixture()
    {
        Sessions = TryCreateSessions();
    }

    private static FaceCreatorTests.Sessions? TryCreateSessions()
    {
        var repoRoot = FaceCreatorTests.FindRepoRoot();

        if (repoRoot is null)
        {
            return null;
        }

        var modelsDirectory = Path.Combine(repoRoot, ".assets", "models");

        foreach (var modelFileName in RequiredModels)
        {
            var modelPath = Path.Combine(modelsDirectory, modelFileName);

            if (!File.Exists(modelPath) || new FileInfo(modelPath).Length == 0)
            {
                return null;
            }
        }

        var detectorSessions = new Dictionary<string, InferenceSession>
        {
            ["retinaface"] = new InferenceSession(Path.Combine(modelsDirectory, "retinaface_10g.onnx")),
            ["scrfd"] = new InferenceSession(Path.Combine(modelsDirectory, "scrfd_2.5g.onnx")),
            ["yolo_face"] = new InferenceSession(Path.Combine(modelsDirectory, "yoloface_8n.onnx")),
        };

        return new FaceCreatorTests.Sessions(
            detectorSessions,
            new InferenceSession(Path.Combine(modelsDirectory, "2dfan4.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "peppa_wutz.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "fan_68_5.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "arcface_w600k_r50.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "fairface.onnx")));
    }

    public void Dispose() => Sessions?.Dispose();
}

[Collection("NativeInference")]
public sealed class FaceCreatorTests : IClassFixture<FaceCreatorSessionFixture>
{
    private const FaceDetectorModel DetectorModel = FaceDetectorModel.Many;
    private const string DetectorSize = "640x640";
    private const double DetectorScore = 0.5;
    private static readonly int[] DetectorMargin = { 0, 0, 0, 0 };
    private static readonly int[] DetectorAngles = { 0 };
    private const FaceLandmarkerModel LandmarkerModel = FaceLandmarkerModel.Many;
    private const double LandmarkerScore = 0.5;

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    private static readonly string[] RequiredModels =
    {
        "retinaface_10g.onnx", "scrfd_2.5g.onnx", "yoloface_8n.onnx",
        "2dfan4.onnx", "peppa_wutz.onnx", "fan_68_5.onnx",
        "arcface_w600k_r50.onnx", "fairface.onnx",
    };

    private readonly FaceCreatorSessionFixture _fixture;

    public FaceCreatorTests(FaceCreatorSessionFixture fixture)
    {
        _fixture = fixture;
    }

    internal sealed record Sessions(
        IReadOnlyDictionary<string, InferenceSession> DetectorSessions,
        InferenceSession TwoDFan4Session,
        InferenceSession PeppaWutzSession,
        InferenceSession Fan685Session,
        InferenceSession RecognizerSession,
        InferenceSession ClassifierSession) : IDisposable
    {
        public void Dispose()
        {
            foreach (var session in DetectorSessions.Values)
            {
                session.Dispose();
            }

            TwoDFan4Session.Dispose();
            PeppaWutzSession.Dispose();
            Fan685Session.Dispose();
            RecognizerSession.Dispose();
            ClassifierSession.Dispose();
        }
    }

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

    internal static bool ModelsAndMediaAvailable()
    {
        if (!File.Exists(SourceImage) || new FileInfo(SourceImage).Length == 0)
        {
            return false;
        }

        var repoRoot = FindRepoRoot();

        if (repoRoot is null)
        {
            return false;
        }

        var modelsDirectory = Path.Combine(repoRoot, ".assets", "models");

        return RequiredModels.All(modelFileName =>
        {
            var modelPath = Path.Combine(modelsDirectory, modelFileName);
            return File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
        });
    }

    private IReadOnlyList<Types.Face> GetManyFacesForFrames(IReadOnlyList<Mat> visionFrames)
    {
        var sessions = _fixture.Sessions!;

        return FaceCreator.GetManyFaces(
            visionFrames,
            DetectorModel,
            DetectorSize,
            DetectorScore,
            DetectorMargin,
            DetectorAngles,
            LandmarkerScore,
            LandmarkerModel,
            sessions.DetectorSessions,
            sessions.Fan685Session,
            sessions.TwoDFan4Session,
            sessions.PeppaWutzSession,
            sessions.RecognizerSession,
            sessions.ClassifierSession);
    }

    // -----------------------------------------------------------------
    // ModelFactAttribute
    // -----------------------------------------------------------------

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelFactAttribute : FactAttribute
    {
        public ModelFactAttribute()
        {
            if (!ModelsAndMediaAvailable())
            {
                Skip = "requires source.jpg in /tmp/facefusion-test-examples (run tools/parity/fetch_examples.sh) " +
                       "and every model in RequiredModels under .assets/models/ (gitignored, not present in CI) — " +
                       "populate via the real Python content_analyser/face_detector/etc. pre_check() with network " +
                       "access, then retry";
            }
        }
    }

    // -----------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------

    /// <summary>Python: <c>test_get_one_face</c>.</summary>
    [ModelFact]
    public void TestGetOneFace()
    {
        using var sourceVisionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var manyFaces = GetManyFacesForFrames(new[] { sourceVisionFrame });
        var face = FaceCreator.GetOneFace(manyFaces);

        Assert.NotNull(face);
        Assert.Equal(4, ((float[])face!.BoundingBox).Length);
    }

    /// <summary>Python: <c>test_get_many_faces</c>.</summary>
    [ModelFact]
    public void TestGetManyFaces()
    {
        using var sourceVisionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var manyFaces = GetManyFacesForFrames(new[] { sourceVisionFrame, sourceVisionFrame, sourceVisionFrame });

        Assert.Equal(3, manyFaces.Count);
    }

    /// <summary>Python: <c>test_refill_faces</c>.</summary>
    [ModelFact]
    public void TestRefillFaces()
    {
        using var sourceVisionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var manyFaces = GetManyFacesForFrames(new[] { sourceVisionFrame });
        var face = FaceCreator.GetOneFace(manyFaces)!;

        var faceFirst = face with { BoundingBox = new float[] { 0, 0, 10, 10 } };
        var faceMiddle = face with { BoundingBox = new float[] { 40, 40, 50, 50 } };
        var faceLast = face with { BoundingBox = new float[] { 80, 80, 90, 90 } };

        var fillFaces = FaceCreator.RefillFaces(new Types.Face?[] { faceFirst, null, faceLast });

        Assert.Equal(new float[] { 0f, 0f, 10f, 10f }, (float[])fillFaces[0].BoundingBox);
        Assert.Equal(new float[] { 40f, 40f, 50f, 50f }, (float[])fillFaces[1].BoundingBox);
        Assert.Equal(new float[] { 80f, 80f, 90f, 90f }, (float[])fillFaces[2].BoundingBox);

        fillFaces = FaceCreator.RefillFaces(new Types.Face?[] { faceFirst, null, null, null, faceLast });

        Assert.Equal(new float[] { 0f, 0f, 10f, 10f }, (float[])fillFaces[0].BoundingBox);
        Assert.Equal(new float[] { 20f, 20f, 30f, 30f }, (float[])fillFaces[1].BoundingBox);
        Assert.Equal(new float[] { 40f, 40f, 50f, 50f }, (float[])fillFaces[2].BoundingBox);
        Assert.Equal(new float[] { 60f, 60f, 70f, 70f }, (float[])fillFaces[3].BoundingBox);
        Assert.Equal(new float[] { 80f, 80f, 90f, 90f }, (float[])fillFaces[4].BoundingBox);

        fillFaces = FaceCreator.RefillFaces(new Types.Face?[] { faceFirst, null, faceMiddle, null, faceLast });

        Assert.Equal(new float[] { 0f, 0f, 10f, 10f }, (float[])fillFaces[0].BoundingBox);
        Assert.Equal(new float[] { 20f, 20f, 30f, 30f }, (float[])fillFaces[1].BoundingBox);
        Assert.Equal(new float[] { 40f, 40f, 50f, 50f }, (float[])fillFaces[2].BoundingBox);
        Assert.Equal(new float[] { 60f, 60f, 70f, 70f }, (float[])fillFaces[3].BoundingBox);
        Assert.Equal(new float[] { 80f, 80f, 90f, 90f }, (float[])fillFaces[4].BoundingBox);
    }

    /// <summary>Python: <c>test_average_face_geometry</c>.</summary>
    [ModelFact]
    public void TestAverageFaceGeometry()
    {
        using var sourceVisionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var manyFaces = GetManyFacesForFrames(new[] { sourceVisionFrame });
        var facePrevious = FaceCreator.GetOneFace(manyFaces)!;
        var faceNext = FaceCreator.GetOneFace(manyFaces)!;

        facePrevious = facePrevious with { BoundingBox = new float[] { 0, 0, 10, 10 } };
        faceNext = faceNext with { BoundingBox = new float[] { 80, 80, 90, 90 } };

        var averaged = FaceCreator.AverageFaceGeometry(new[] { facePrevious, faceNext }, 0.5);
        Assert.Equal(new float[] { 40f, 40f, 50f, 50f }, (float[])averaged.BoundingBox);
        Assert.Equal(faceNext.Angle, averaged.Angle);
        Assert.Same(faceNext.Embedding, averaged.Embedding);

        var averagedLow = FaceCreator.AverageFaceGeometry(new[] { facePrevious, faceNext }, 0.25);
        Assert.Same(facePrevious.Embedding, averagedLow.Embedding);
    }
}
