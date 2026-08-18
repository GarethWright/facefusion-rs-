using System.Linq;
using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_face_tracker.py</c>.
///
/// <para>
/// <b>State-manager settings, taken as explicit locals (per PORT_CONVENTIONS.md rule 5).</b>
/// Python's <c>before_all</c> fixture calls <c>state_manager.init_item(...)</c> for
/// <c>execution_device_ids</c>, <c>execution_providers</c>, <c>download_providers</c>,
/// <c>face_detector_angles</c> = <c>[0]</c>, <c>face_detector_model</c> = <c>'yolo_face'</c>,
/// <c>face_detector_size</c> = <c>'640x640'</c>, <c>face_detector_margin</c> = <c>(0,0,0,0)</c>,
/// <c>face_detector_score</c> = <c>0.5</c>, <c>face_landmarker_model</c> = <c>'many'</c>,
/// <c>face_landmarker_score</c> = <c>0.5</c>, <c>face_tracker_score</c> = <c>0.3</c>. Those
/// become the local constants below, passed explicitly into every call.
/// </para>
///
/// <para>
/// <b><c>get_static_faces</c>/<c>refill_faces</c> collaborators.</b> <see cref="FaceTracker"/>
/// takes both as delegates (see its class remarks). Here they are wired to
/// <see cref="FaceCreator.GetManyFaces"/> (bypassing <see cref="FaceStore"/>'s caching layer —
/// caching only affects performance, never the detected face values, so this is behaviourally
/// identical to Python's <c>face_creator.get_static_faces</c> for the purposes of this test)
/// and <see cref="FaceCreator.RefillFaces"/> directly.
/// </para>
///
/// <para>
/// <b>No download-backed model pool (per <see cref="FaceDetector"/>/<see cref="FaceCreator"/>'s
/// own documented divergence).</b> Sessions are loaded directly from
/// <c>.assets/models/*.onnx</c>, which skips cleanly (rather than failing) when a model or the
/// example video is not present, per PORT_CONVENTIONS.md rule 2.
/// </para>
/// </summary>
public sealed class FaceTrackerSessionFixture : IDisposable
{
    private static readonly string[] RequiredModels =
    {
        "yoloface_8n.onnx", "2dfan4.onnx", "peppa_wutz.onnx", "fan_68_5.onnx",
        "arcface_w600k_r50.onnx", "fairface.onnx",
    };

    internal FaceTrackerTests.Sessions? Sessions { get; }

    public FaceTrackerSessionFixture()
    {
        Sessions = TryCreateSessions();
    }

    private static FaceTrackerTests.Sessions? TryCreateSessions()
    {
        var repoRoot = FaceTrackerTests.FindRepoRoot();

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
            ["yolo_face"] = new InferenceSession(Path.Combine(modelsDirectory, "yoloface_8n.onnx")),
        };

        return new FaceTrackerTests.Sessions(
            detectorSessions,
            new InferenceSession(Path.Combine(modelsDirectory, "2dfan4.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "peppa_wutz.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "fan_68_5.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "arcface_w600k_r50.onnx")),
            new InferenceSession(Path.Combine(modelsDirectory, "fairface.onnx")));
    }

    public void Dispose() => Sessions?.Dispose();
}

public sealed class FaceTrackerTests : IClassFixture<FaceTrackerSessionFixture>
{
    private const FaceDetectorModel DetectorModel = FaceDetectorModel.YoloFace;
    private const string DetectorSize = "640x640";
    private const double DetectorScore = 0.5;
    private static readonly int[] DetectorMargin = { 0, 0, 0, 0 };
    private static readonly int[] DetectorAngles = { 0 };
    private const FaceLandmarkerModel LandmarkerModel = FaceLandmarkerModel.Many;
    private const double LandmarkerScore = 0.5;
    private const double TrackerScore = 0.3;

    private const string TargetVideo = "/tmp/facefusion-test-examples/target-240p.mp4";

    private static readonly string[] RequiredModels =
    {
        "yoloface_8n.onnx", "2dfan4.onnx", "peppa_wutz.onnx", "fan_68_5.onnx",
        "arcface_w600k_r50.onnx", "fairface.onnx",
    };

    private readonly FaceTrackerSessionFixture _fixture;

    public FaceTrackerTests(FaceTrackerSessionFixture fixture)
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
        if (!File.Exists(TargetVideo) || new FileInfo(TargetVideo).Length == 0)
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

    private IReadOnlyList<Types.Face> GetStaticFaces(IReadOnlyList<Mat> visionFrames)
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

    private static IReadOnlyList<Types.Face> RefillFaces(IReadOnlyList<Types.Face?> faces) => FaceCreator.RefillFaces(faces);

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
                Skip = "requires target-240p.mp4 in /tmp/facefusion-test-examples (run tools/parity/fetch_examples.sh) " +
                       "and every model in RequiredModels under .assets/models/ (gitignored, not present in CI) — " +
                       "populate via the real Python face_classifier/face_detector/etc. pre_check() with network " +
                       "access, then retry";
            }
        }
    }

    // -----------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------

    /// <summary>Python: <c>test_track_faces</c>.</summary>
    [ModelFact]
    public void TestTrackFaces()
    {
        var targetVisionFrames = FaceFusion.Vision.Vision.SelectVideoFrames(TargetVideo, 3, 3).ToList();
        try
        {
            using var emptyVisionFrame = new Mat(targetVisionFrames[0].Size(), targetVisionFrames[0].Type(), Scalar.All(0));

            ReplaceFrame(targetVisionFrames, 2, emptyVisionFrame);
            ReplaceFrame(targetVisionFrames, 3, emptyVisionFrame);
            ReplaceFrame(targetVisionFrames, 4, emptyVisionFrame);
            ReplaceFrame(targetVisionFrames, 5, emptyVisionFrame);

            var tracked = FaceTracker.TrackFaces(targetVisionFrames, TrackerScore, GetStaticFaces, RefillFaces);
            Assert.Single(tracked);
        }
        finally
        {
            foreach (var frame in targetVisionFrames)
            {
                frame.Dispose();
            }
        }

        var secondBatch = FaceFusion.Vision.Vision.SelectVideoFrames(TargetVideo, 3, 3).Take(5).ToList();
        try
        {
            using var emptyVisionFrame = new Mat(secondBatch[0].Size(), secondBatch[0].Type(), Scalar.All(0));

            ReplaceFrame(secondBatch, 0, emptyVisionFrame);
            ReplaceFrame(secondBatch, 1, emptyVisionFrame);
            ReplaceFrame(secondBatch, 2, emptyVisionFrame);

            var tracked = FaceTracker.TrackFaces(secondBatch, TrackerScore, GetStaticFaces, RefillFaces);
            Assert.Empty(tracked);
        }
        finally
        {
            foreach (var frame in secondBatch)
            {
                frame.Dispose();
            }
        }
    }

    /// <summary>Replaces <c>frames[index]</c> with a clone of <paramref name="replacement"/>, disposing the original — the C# analogue of Python's <c>target_vision_frames[index] = empty_vision_frame</c> (array-element rebind; no disposal concept in Python).</summary>
    private static void ReplaceFrame(IList<Mat> frames, int index, Mat replacement)
    {
        frames[index].Dispose();
        frames[index] = replacement.Clone();
    }

    /// <summary>Python: <c>test_create_face_tracks</c>.</summary>
    [ModelFact]
    public void TestCreateFaceTracks()
    {
        using var targetVisionFrame = FaceFusion.Vision.Vision.ReadStaticVideoFrame(TargetVideo, 0)!;
        using var multiFaceVisionFrame = new Mat();
        Cv2.HConcat(new[] { targetVisionFrame, targetVisionFrame }, multiFaceVisionFrame);

        var faceTracks = FaceTracker.CreateFaceTracks(new[] { targetVisionFrame, targetVisionFrame }, TrackerScore, GetStaticFaces);
        Assert.Single(faceTracks);
        Assert.Equal(new[] { 0, 1 }, faceTracks[0].Keys.OrderBy(k => k));

        var multiFaceTracks = FaceTracker.CreateFaceTracks(new[] { multiFaceVisionFrame, multiFaceVisionFrame }, TrackerScore, GetStaticFaces);
        Assert.Equal(2, multiFaceTracks.Count);
        Assert.Equal(new[] { 0, 1 }, multiFaceTracks[0].Keys.OrderBy(k => k));
        Assert.Equal(new[] { 0, 1 }, multiFaceTracks[^1].Keys.OrderBy(k => k));

        var strictFaceTracks = FaceTracker.CreateFaceTracks(new[] { targetVisionFrame, targetVisionFrame }, 1.0, GetStaticFaces);
        Assert.Equal(2, strictFaceTracks.Count);
    }

    /// <summary>Python: <c>test_select_face_track</c>.</summary>
    [ModelFact]
    public void TestSelectFaceTrack()
    {
        using var targetVisionFrame = FaceFusion.Vision.Vision.ReadStaticVideoFrame(TargetVideo, 0)!;
        var manyFaces = GetStaticFaces(new[] { targetVisionFrame });
        var face = FaceCreator.GetOneFace(manyFaces);
        Assert.NotNull(face);

        var faceOverlap = face! with { BoundingBox = new float[] { 12, 12, 52, 52 } };
        var faceDistant = face with { BoundingBox = new float[] { 200, 200, 240, 240 } };
        var faceTrackOverlap = new Dictionary<int, Types.Face> { [0] = face with { BoundingBox = new float[] { 10, 10, 50, 50 } } };
        var faceTrackDistant = new Dictionary<int, Types.Face> { [0] = face with { BoundingBox = new float[] { 100, 100, 140, 140 } } };

        var selectedOverlap = FaceTracker.SelectFaceTrack(new[] { faceTrackOverlap, faceTrackDistant }, faceOverlap, 0.3);
        Assert.Same(faceTrackOverlap, selectedOverlap);

        var selectedDistant = FaceTracker.SelectFaceTrack(new[] { faceTrackOverlap, faceTrackDistant }, faceDistant, 0.3);
        Assert.Empty(selectedDistant);
    }
}
