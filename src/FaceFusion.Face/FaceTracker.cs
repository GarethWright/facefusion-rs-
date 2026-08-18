using FaceFusion.Core;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_tracker.py</c> — groups per-frame face detections into tracks
/// (a face bounding box followed across consecutive frames) and reconstructs the track
/// covering the middle frame of a clip.
///
/// <para>
/// <b>No global state.</b> Per PORT_CONVENTIONS.md rule 5, every Python
/// <c>state_manager</c>/module-level call becomes an explicit parameter.
/// </para>
///
/// <para>
/// <b><c>FaceTrack</c> representation.</b> Python: <c>FaceTrack : TypeAlias = Dict[int,
/// Face]</c> (frame index -&gt; the face detected in that frame, for whichever frames the
/// track covers). Represented here as <c>Dictionary&lt;int, Face&gt;</c>, matching
/// <c>src/FaceFusion.Types/TypeAliases.cs</c>'s documented mapping.
/// </para>
///
/// <para>
/// <b><c>face_creator</c> collaborators taken as delegates.</b> Python calls
/// <c>facefusion.face_creator.get_static_faces</c> and <c>.refill_faces</c>. Both live in
/// <c>face_creator.py</c> (<c>FaceCreator.cs</c>), which is out of this module's assignment —
/// it is owned by a different, concurrently-running port agent and does not exist in this
/// project yet. <see cref="TrackFaces"/> and <see cref="CreateFaceTracks"/> therefore take
/// <c>getStaticFaces</c> (and, for <see cref="TrackFaces"/>, <c>refillFaces</c>) as delegate
/// parameters rather than a hard compile-time dependency; a caller in a later phase passes
/// <c>FaceCreator.GetStaticFaces</c> / <c>FaceCreator.RefillFaces</c> once that file lands.
/// </para>
///
/// <para>
/// <b>Dictionary iteration order.</b> Python's <c>get_last(face_track)</c> (via
/// <c>common_helper.get_last</c>, <c>next(reversed(face_track), None)</c>) returns the
/// most-recently-inserted key of the dict — dicts are insertion-ordered and reversible since
/// Python 3.8. <see cref="SelectFaceTrack"/> relies on the same guarantee for
/// <see cref="Dictionary{TKey,TValue}"/>: .NET's <see cref="Dictionary{TKey,TValue}"/>
/// preserves insertion order in enumeration as an implementation detail as long as no entries
/// are ever removed, which holds for every <c>FaceTrack</c> built by
/// <see cref="CreateFaceTracks"/> (entries are only ever added or overwritten in place, never
/// removed) — and because <see cref="CreateFaceTracks"/>'s outer loop visits
/// <c>frameIndex</c> strictly ascending, the last-inserted key and the numerically largest key
/// coincide for every track built here, so this is not a fragile coincidence specific to one
/// input.
/// </para>
/// </summary>
public static class FaceTracker
{
    /// <summary>Python: <c>track_faces</c>. Does not take ownership of any <see cref="Mat"/> in <paramref name="visionFrames"/>.</summary>
    public static IReadOnlyList<FaceFusion.Types.Face> TrackFaces(
        IReadOnlyList<Mat> visionFrames,
        double score,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> getStaticFaces,
        Func<IReadOnlyList<FaceFusion.Types.Face?>, IReadOnlyList<FaceFusion.Types.Face>> refillFaces)
    {
        var targetIndex = visionFrames.Count / 2;
        var faceTracks = CreateFaceTracks(visionFrames, score, getStaticFaces);
        var tempFaces = new List<FaceFusion.Types.Face>();

        foreach (var faceTrack in faceTracks)
        {
            var trackIndices = faceTrack.Keys.OrderBy(index => index).ToList();
            var trackIndexFirst = trackIndices[0];
            var trackIndexLast = trackIndices[^1];

            if (targetIndex >= trackIndexFirst && targetIndex <= trackIndexLast)
            {
                var fillFaces = new List<FaceFusion.Types.Face?>();

                for (var index = trackIndexFirst; index <= trackIndexLast; index++)
                {
                    fillFaces.Add(faceTrack.TryGetValue(index, out var face) ? face : null);
                }

                var refilled = refillFaces(fillFaces);
                tempFaces.Add(refilled[targetIndex - trackIndexFirst]);
            }
        }

        return tempFaces;
    }

    /// <summary>Python: <c>create_face_tracks</c>. Does not take ownership of any <see cref="Mat"/> in <paramref name="visionFrames"/>.</summary>
    public static IReadOnlyList<Dictionary<int, FaceFusion.Types.Face>> CreateFaceTracks(
        IReadOnlyList<Mat> visionFrames,
        double score,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> getStaticFaces)
    {
        var faceTracks = new List<Dictionary<int, FaceFusion.Types.Face>>();

        for (var frameIndex = 0; frameIndex < visionFrames.Count; frameIndex++)
        {
            var visionFrame = visionFrames[frameIndex];

            foreach (var face in getStaticFaces(new[] { visionFrame }))
            {
                var faceTrack = SelectFaceTrack(faceTracks, face, score);

                if (faceTrack.Count > 0)
                {
                    faceTrack[frameIndex] = face;
                }
                else
                {
                    faceTracks.Add(new Dictionary<int, FaceFusion.Types.Face> { [frameIndex] = face });
                }
            }
        }

        return faceTracks;
    }

    /// <summary>
    /// Python: <c>select_face_track</c>. Returns an empty <see cref="Dictionary{TKey,TValue}"/>
    /// (matching Python's <c>{}</c>) when no existing track overlaps <paramref name="face"/>
    /// closely enough — never <see langword="null"/>, so a caller can always test
    /// <c>.Count &gt; 0</c> the way Python tests dict truthiness.
    /// </summary>
    public static Dictionary<int, FaceFusion.Types.Face> SelectFaceTrack(
        IReadOnlyList<Dictionary<int, FaceFusion.Types.Face>> faceTracks, FaceFusion.Types.Face face, double score)
    {
        var selectTrack = new Dictionary<int, FaceFusion.Types.Face>();
        var selectScore = score;

        foreach (var faceTrack in faceTracks)
        {
            var lastKey = CommonHelper.GetLast(faceTrack.Keys);
            var trackFace = faceTrack[lastKey];
            var trackScore = FaceHelper.CalculateBoundingBoxOverlap(AsBoundingBox(face), AsBoundingBox(trackFace));

            if (trackScore > selectScore)
            {
                selectScore = trackScore;
                selectTrack = faceTrack;
            }
        }

        return selectTrack;
    }

    /// <summary>
    /// Assumes <c>Face.BoundingBox</c> is a <c>float[4]</c> — see the same assumption
    /// documented in <see cref="FaceSelector"/>'s class remarks (every producer of a
    /// bounding box in this codebase uses <c>float[]</c>).
    /// </summary>
    private static float[] AsBoundingBox(FaceFusion.Types.Face face)
    {
        if (face.BoundingBox is float[] boundingBox)
        {
            return boundingBox;
        }

        throw new ArgumentException("Face.BoundingBox must be a float[4] (see FaceHelper's bounding box helpers).", nameof(face));
    }
}
