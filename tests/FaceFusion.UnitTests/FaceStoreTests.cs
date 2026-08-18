using FaceFusion.Face;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the behaviour in <c>facefusion/face_store.py</c>. There is no
/// <c>tests/test_face_store.py</c> in the Python suite (per the port instructions: "(none)"),
/// so this exercises <see cref="FaceStore"/>'s public surface directly against the semantics
/// documented in its own class remarks (get/set/resolve-lock/clear, keyed by frame content, not
/// by <see cref="Mat"/> identity — two different <see cref="Mat"/> instances with identical
/// pixel bytes hit the same cache entry, matching Python's
/// <c>create_hash(vision_frame.tobytes())</c> keying).
/// </summary>
public sealed class FaceStoreTests
{
    private static Mat MakeFrame(byte seed)
    {
        var mat = new Mat(4, 4, MatType.CV_8UC3, Scalar.All(seed));
        return mat;
    }

    private static Types.Face MakeFace(string origin = "detect")
    {
        var five = new float[5, 2];
        var landmarkSet = new FaceLandmarkSet(five, five, five, five);
        var scoreSet = new FaceScoreSet(0.9, 0.8);

        return new Types.Face(
            Origin: origin,
            BoundingBox: new float[] { 0, 0, 10, 10 },
            ScoreSet: scoreSet,
            LandmarkSet: landmarkSet,
            Angle: 0,
            Embedding: new float[] { 1f, 2f, 3f },
            EmbeddingNorm: new float[] { 0.26f, 0.53f, 0.80f },
            Age: 20..29,
            Gender: Gender.Female,
            Race: Race.White);
    }

    [Fact]
    public void GetFacesReturnsNullBeforeSet()
    {
        var store = new FaceStore();
        using var frame = MakeFrame(1);

        Assert.Null(store.GetFaces(frame));
    }

    [Fact]
    public void SetThenGetReturnsTheSameFaces()
    {
        var store = new FaceStore();
        using var frame = MakeFrame(2);
        var faces = new[] { MakeFace() };

        store.SetFaces(frame, faces);

        Assert.Same(faces, store.GetFaces(frame));
    }

    [Fact]
    public void GetFacesKeysByFrameContentNotByMatIdentity()
    {
        var store = new FaceStore();
        using var frameA = MakeFrame(7);
        using var frameB = MakeFrame(7); // same pixels, different Mat instance
        var faces = new[] { MakeFace() };

        store.SetFaces(frameA, faces);

        Assert.Same(faces, store.GetFaces(frameB));
    }

    [Fact]
    public void DifferentFrameContentGetsDifferentCacheEntries()
    {
        var store = new FaceStore();
        using var frameA = MakeFrame(3);
        using var frameB = MakeFrame(4);
        var facesA = new[] { MakeFace("detect") };

        store.SetFaces(frameA, facesA);

        Assert.Same(facesA, store.GetFaces(frameA));
        Assert.Null(store.GetFaces(frameB));
    }

    [Fact]
    public void ClearFacesRemovesEveryEntry()
    {
        var store = new FaceStore();
        using var frame = MakeFrame(5);
        store.SetFaces(frame, new[] { MakeFace() });

        store.ClearFaces();

        Assert.Null(store.GetFaces(frame));
    }

    [Fact]
    public void ResolveLockReturnsTheSameObjectForTheSameFrameContent()
    {
        var store = new FaceStore();
        using var frameA = MakeFrame(9);
        using var frameB = MakeFrame(9);

        var lockA = store.ResolveLock(frameA);
        var lockB = store.ResolveLock(frameB);

        Assert.Same(lockA, lockB);
    }

    [Fact]
    public void ResolveLockReturnsDifferentObjectsForDifferentFrameContent()
    {
        var store = new FaceStore();
        using var frameA = MakeFrame(11);
        using var frameB = MakeFrame(12);

        Assert.NotSame(store.ResolveLock(frameA), store.ResolveLock(frameB));
    }

    [Fact]
    public void ResolveLockOnANonVisionFrameReturnsAFreshUnsharedObject()
    {
        var store = new FaceStore();
        using var empty = new Mat();

        var lockA = store.ResolveLock(empty);
        var lockB = store.ResolveLock(empty);

        Assert.NotSame(lockA, lockB);
    }

    [Fact]
    public void GetFacesOnANonVisionFrameReturnsNull()
    {
        var store = new FaceStore();
        using var empty = new Mat();

        Assert.Null(store.GetFaces(empty));
    }

    [Fact]
    public void SetFacesOnANonVisionFrameIsANoOp()
    {
        var store = new FaceStore();
        using var empty = new Mat();

        store.SetFaces(empty, new[] { MakeFace() });

        Assert.Null(store.GetFaces(empty));
    }

    [Fact]
    public void SetFacesOverwritesAPreviousEntryForTheSameFrame()
    {
        var store = new FaceStore();
        using var frame = MakeFrame(6);
        var firstFaces = new[] { MakeFace("detect") };
        var secondFaces = new[] { MakeFace("refill") };

        store.SetFaces(frame, firstFaces);
        store.SetFaces(frame, secondFaces);

        Assert.Same(secondFaces, store.GetFaces(frame));
    }

    [Fact]
    public void StoreInstancesAreIndependent()
    {
        var storeA = new FaceStore();
        var storeB = new FaceStore();
        using var frame = MakeFrame(13);
        var faces = new[] { MakeFace() };

        storeA.SetFaces(frame, faces);

        Assert.Same(faces, storeA.GetFaces(frame));
        Assert.Null(storeB.GetFaces(frame));
    }
}
