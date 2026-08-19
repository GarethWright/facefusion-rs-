using FaceFusion.Core;
using FaceFusion.Tensors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_selector.py</c> — sorting/filtering faces by order, gender,
/// race and age, plus the reference/one/many face-selection orchestration.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Every Python
/// <c>state_manager.get_item(...)</c> call becomes an explicit method parameter here.
/// </para>
///
/// <para>
/// <b>Cross-module collaborators taken as delegates.</b> Python's <c>select_faces</c> calls
/// <c>facefusion.face_creator.get_static_faces</c>/<c>get_one_face</c> and
/// <c>facefusion.face_tracker.track_faces</c>. <c>face_tracker.track_faces</c> is this port's
/// own <see cref="FaceTracker.TrackFaces"/> (same assignment, called directly), but
/// <c>face_creator.py</c> (<c>FaceCreator.cs</c>) is out of this module's assignment — it is
/// owned by a different, concurrently-running port agent and does not exist in this project
/// yet, so there is nothing to call statically. <see cref="SelectFaces"/> therefore takes
/// <c>getStaticFaces</c> and <c>refillFaces</c> (the two <c>face_creator</c> functions
/// <c>face_tracker.py</c> needs) as delegate parameters rather than a hard compile-time
/// dependency; a caller in a later phase passes <c>FaceCreator.GetStaticFaces</c> /
/// <c>FaceCreator.RefillFaces</c> once that file lands. <c>get_one_face</c> is reproduced
/// locally as a private one-line helper instead (see <see cref="GetOneFace"/>) since it is a
/// trivial, model-free selection (no reason to thread a delegate through for it).
/// </para>
///
/// <para>
/// <b><c>Face.BoundingBox</c> / <c>Face.EmbeddingNorm</c> runtime type.</b>
/// <see cref="FaceFusion.Types.Face"/> declares these as <c>object</c> (FaceFusion.Types has
/// no tensor dependency). Every producer in this codebase (<see cref="FaceHelper"/>'s bounding
/// box helpers, the recognizer's embedding output) uses <c>float[]</c> for both, so this file
/// casts to <c>float[]</c> and documents the assumption at each cast site rather than silently
/// trusting <c>object</c>.
/// </para>
/// </summary>
public static class FaceSelector
{
    // -----------------------------------------------------------------
    // Top-level selection
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>select_faces</c>. See the class remarks for why <paramref name="getStaticFaces"/>
    /// and <paramref name="refillFaces"/> are delegates rather than direct calls into
    /// <c>FaceCreator</c>. Does not take ownership of any <see cref="Mat"/> parameter.
    /// </summary>
    public static IReadOnlyList<FaceFusion.Types.Face> SelectFaces(
        Mat referenceVisionFrame,
        IReadOnlyList<Mat> sourceVisionFrames,
        IReadOnlyList<Mat> targetVisionFrames,
        FaceSelectorMode faceSelectorMode,
        double faceTrackerScore,
        FaceSelectorOrder faceSelectorOrder,
        FaceSelectorGender? faceSelectorGender,
        FaceSelectorRace? faceSelectorRace,
        int? faceSelectorAgeStart,
        int? faceSelectorAgeEnd,
        int referenceFacePosition,
        double referenceFaceDistance,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> getStaticFaces,
        Func<IReadOnlyList<FaceFusion.Types.Face?>, IReadOnlyList<FaceFusion.Types.Face>> refillFaces)
    {
        var sourceFaces = getStaticFaces(sourceVisionFrames);

        IReadOnlyList<FaceFusion.Types.Face> targetFaces;
        if (faceTrackerScore > 0)
        {
            targetFaces = FaceTracker.TrackFaces(targetVisionFrames, faceTrackerScore, getStaticFaces, refillFaces);
        }
        else
        {
            var middleFrame = CommonHelper.GetMiddle(targetVisionFrames);
            targetFaces = middleFrame is null
                ? Array.Empty<FaceFusion.Types.Face>()
                : getStaticFaces(new[] { middleFrame });
        }

        if (faceSelectorMode == FaceSelectorMode.Many)
        {
            return SortAndFilterFaces(sourceFaces, targetFaces, faceSelectorOrder, faceSelectorGender, faceSelectorRace, faceSelectorAgeStart, faceSelectorAgeEnd);
        }

        if (faceSelectorMode == FaceSelectorMode.One)
        {
            var filtered = SortAndFilterFaces(sourceFaces, targetFaces, faceSelectorOrder, faceSelectorGender, faceSelectorRace, faceSelectorAgeStart, faceSelectorAgeEnd);
            var targetFace = GetOneFace(filtered, 0);
            if (targetFace is not null)
            {
                return new[] { targetFace };
            }
        }

        if (faceSelectorMode == FaceSelectorMode.Reference)
        {
            var referenceFaces = getStaticFaces(new[] { referenceVisionFrame });
            referenceFaces = SortAndFilterFaces(sourceFaces, referenceFaces, faceSelectorOrder, faceSelectorGender, faceSelectorRace, faceSelectorAgeStart, faceSelectorAgeEnd);
            var referenceFace = GetOneFace(referenceFaces, referenceFacePosition);

            if (referenceFace is not null)
            {
                return FindMatchFaces(new[] { referenceFace }, targetFaces, referenceFaceDistance);
            }
        }

        return Array.Empty<FaceFusion.Types.Face>();
    }

    /// <summary>
    /// Python: <c>face_creator.get_one_face</c>. Reproduced locally — see the class remarks.
    /// </summary>
    private static FaceFusion.Types.Face? GetOneFace(IReadOnlyList<FaceFusion.Types.Face> faces, int position = 0)
    {
        if (faces.Count > 0)
        {
            var clampedPosition = Math.Min(position, faces.Count - 1);
            return faces[clampedPosition];
        }

        return null;
    }

    // -----------------------------------------------------------------
    // Matching
    // -----------------------------------------------------------------

    /// <summary>Python: <c>find_match_faces</c>.</summary>
    public static IReadOnlyList<FaceFusion.Types.Face> FindMatchFaces(
        IReadOnlyList<FaceFusion.Types.Face> referenceFaces, IReadOnlyList<FaceFusion.Types.Face> targetFaces, double faceDistance)
    {
        var matchFaces = new List<FaceFusion.Types.Face>();

        foreach (var referenceFace in referenceFaces)
        {
            // Python: `if reference_face:` — a Face namedtuple is always truthy in Python
            // (it is a non-empty tuple), so this guard never actually filters anything for a
            // real Face; kept as a null-check here for parity with the shape of the loop.
            if (referenceFace is not null)
            {
                foreach (var targetFace in targetFaces)
                {
                    if (CompareFaces(targetFace, referenceFace, faceDistance))
                    {
                        matchFaces.Add(targetFace);
                    }
                }
            }
        }

        return matchFaces;
    }

    /// <summary>Python: <c>compare_faces</c>.</summary>
    public static bool CompareFaces(FaceFusion.Types.Face face, FaceFusion.Types.Face referenceFace, double faceDistance)
    {
        var currentFaceDistance = CalculateFaceDistance(face, referenceFace);
        var interpolated = NumPy.Interp((float)currentFaceDistance, new[] { 0f, 2f }, new[] { 0f, 1f });
        return interpolated < faceDistance;
    }

    /// <summary>
    /// Python: <c>calculate_face_distance</c>. Python's <c>hasattr(face, 'embedding_norm')</c>
    /// is always true for a real <c>Face</c> namedtuple (the field always exists), so the
    /// Python "return 0" branch is effectively dead for real faces; the closest faithful
    /// analogue in a strongly-typed record is a null check on the <c>object</c>-typed field,
    /// which is what is done here.
    /// </summary>
    public static double CalculateFaceDistance(FaceFusion.Types.Face face, FaceFusion.Types.Face referenceFace)
    {
        if (face.EmbeddingNorm is float[] embeddingNorm && referenceFace.EmbeddingNorm is float[] referenceEmbeddingNorm)
        {
            return 1 - NumPy.Dot(embeddingNorm, referenceEmbeddingNorm);
        }

        return 0;
    }

    // -----------------------------------------------------------------
    // Sort / filter
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>sort_and_filter_faces</c>.
    ///
    /// <para>
    /// <b>Deliberately reproduced operator-precedence oddity.</b> Python:
    /// <c>if source_faces and face_selector_gender == 'auto' or face_selector_race == 'auto':</c>.
    /// Python's <c>and</c> binds tighter than <c>or</c>, so this parses as
    /// <c>(source_faces and face_selector_gender == 'auto') or (face_selector_race == 'auto')</c>
    /// — an explicit <c>face_selector_race == 'auto'</c> triggers the auto-resolution block
    /// even when <c>source_faces</c> is empty (in which case <see cref="CommonHelper.GetFirst{T}"/>
    /// on an empty sorted list below returns <see langword="null"/> and nothing actually gets
    /// resolved, so the visible behaviour is unaffected, but the *shape* of the condition is
    /// reproduced exactly per PORT_CONVENTIONS.md rule 1).
    /// </para>
    ///
    /// <para>
    /// <b>Nullable age-filter condition.</b> Python:
    /// <c>if state_manager.get_item('face_selector_age_start') or state_manager.get_item('face_selector_age_end'):</c>.
    /// Python truthiness treats both <c>None</c> (unset) and <c>0</c> (an explicit "from age
    /// zero"/"to age zero" bound) as falsy — the filter only activates when at least one bound
    /// is a real, nonzero int. <see cref="FaceFusion.Types.State"/> types these fields
    /// <c>int?</c> specifically so "unset" (<see langword="null"/>) and "explicitly 0" stay
    /// distinguishable in the type system; the check below is written against that nullable
    /// int directly (<c>value is int v &amp;&amp; v != 0</c>) rather than collapsing to a
    /// plain <c>int</c> defaulted to 0, which would make an explicit 0 and an unset value
    /// indistinguishable in code even though both happen to evaluate the same way here (both
    /// are falsy in Python too) — the nullable form is what makes that "both happen to agree"
    /// an observed fact instead of an assumption baked into the type.
    /// </para>
    /// </summary>
    public static IReadOnlyList<FaceFusion.Types.Face> SortAndFilterFaces(
        IReadOnlyList<FaceFusion.Types.Face> sourceFaces,
        IReadOnlyList<FaceFusion.Types.Face> targetFaces,
        FaceSelectorOrder faceSelectorOrder,
        FaceSelectorGender? faceSelectorGender,
        FaceSelectorRace? faceSelectorRace,
        int? faceSelectorAgeStart,
        int? faceSelectorAgeEnd)
    {
        if (targetFaces.Count == 0)
        {
            return targetFaces;
        }

        var sortedTargetFaces = SortFacesByOrder(targetFaces, faceSelectorOrder);

        // `face_selector_gender`/`face_selector_race` start as the raw selector value, and may
        // be overwritten below with a *resolved* Gender/Race taken from the largest source
        // face. Modelled here as two independent nullable results (null = "no filter"; the
        // FaceSelectorGender/Race 'auto' member is never a member of Gender/Race, so it can
        // never itself become a filter value) rather than mutating a Python-style union local.
        Gender? effectiveGender = faceSelectorGender switch
        {
            FaceSelectorGender.Female => Gender.Female,
            FaceSelectorGender.Male => Gender.Male,
            _ => null
        };
        Race? effectiveRace = faceSelectorRace switch
        {
            FaceSelectorRace.White => Race.White,
            FaceSelectorRace.Black => Race.Black,
            FaceSelectorRace.Latino => Race.Latino,
            FaceSelectorRace.Asian => Race.Asian,
            FaceSelectorRace.Indian => Race.Indian,
            FaceSelectorRace.Arabic => Race.Arabic,
            _ => null
        };

        // See the "Deliberately reproduced operator-precedence oddity" remark above.
        if ((sourceFaces.Count > 0 && faceSelectorGender == FaceSelectorGender.Auto) || faceSelectorRace == FaceSelectorRace.Auto)
        {
            var sourceFace = CommonHelper.GetFirst(SortFacesByOrder(sourceFaces, FaceSelectorOrder.LargeSmall));

            if (sourceFace is not null)
            {
                if (faceSelectorGender == FaceSelectorGender.Auto)
                {
                    effectiveGender = sourceFace.Gender;
                }

                if (faceSelectorRace == FaceSelectorRace.Auto)
                {
                    effectiveRace = sourceFace.Race;
                }
            }
        }

        var filteredTargetFaces = sortedTargetFaces;

        if (effectiveGender is { } gender)
        {
            filteredTargetFaces = FilterFacesByGender(filteredTargetFaces, gender);
        }

        if (effectiveRace is { } race)
        {
            filteredTargetFaces = FilterFacesByRace(filteredTargetFaces, race);
        }

        // See the "Nullable age-filter condition" remark above.
        var ageStartIsTruthy = faceSelectorAgeStart is int startValue && startValue != 0;
        var ageEndIsTruthy = faceSelectorAgeEnd is int endValue && endValue != 0;

        if (ageStartIsTruthy || ageEndIsTruthy)
        {
            // Python passes both raw values straight into `range(age_start, age_end)` even
            // though only one of the two is guaranteed to be a real int here (the other could
            // still be the unset `None`) — a latent `TypeError` in the original if a caller
            // ever sets only one bound. Reproducing a crash is not useful (PORT_CONVENTIONS.md
            // rule 1 is about behavioural oddities, not unhandled exceptions), so an unset
            // bound falls back to 0 here, which is also `range`'s own conceptual "start"/"stop
            // no bound" value for this purpose.
            filteredTargetFaces = FilterFacesByAge(filteredTargetFaces, faceSelectorAgeStart ?? 0, faceSelectorAgeEnd ?? 0);
        }

        return filteredTargetFaces;
    }

    /// <summary>Python: <c>sort_faces_by_order</c>. Stable sort, matching Python's <c>sorted</c>.</summary>
    public static IReadOnlyList<FaceFusion.Types.Face> SortFacesByOrder(IReadOnlyList<FaceFusion.Types.Face> faces, FaceSelectorOrder order)
    {
        return order switch
        {
            FaceSelectorOrder.LeftRight => faces.OrderBy(GetBoundingBoxLeft).ToList(),
            FaceSelectorOrder.RightLeft => faces.OrderByDescending(GetBoundingBoxLeft).ToList(),
            FaceSelectorOrder.TopBottom => faces.OrderBy(GetBoundingBoxTop).ToList(),
            FaceSelectorOrder.BottomTop => faces.OrderByDescending(GetBoundingBoxTop).ToList(),
            FaceSelectorOrder.SmallLarge => faces.OrderBy(GetBoundingBoxArea).ToList(),
            FaceSelectorOrder.LargeSmall => faces.OrderByDescending(GetBoundingBoxArea).ToList(),
            FaceSelectorOrder.BestWorst => faces.OrderByDescending(GetFaceDetectorScore).ToList(),
            FaceSelectorOrder.WorstBest => faces.OrderBy(GetFaceDetectorScore).ToList(),
            _ => faces
        };
    }

    /// <summary>Python: <c>get_bounding_box_left</c>. Assumes <c>Face.BoundingBox</c> is a <c>float[4]</c> — see class remarks.</summary>
    public static float GetBoundingBoxLeft(FaceFusion.Types.Face face) => AsBoundingBox(face)[0];

    /// <summary>Python: <c>get_bounding_box_top</c>.</summary>
    public static float GetBoundingBoxTop(FaceFusion.Types.Face face) => AsBoundingBox(face)[1];

    /// <summary>Python: <c>get_bounding_box_area</c>.</summary>
    public static float GetBoundingBoxArea(FaceFusion.Types.Face face)
    {
        var boundingBox = AsBoundingBox(face);
        return (boundingBox[2] - boundingBox[0]) * (boundingBox[3] - boundingBox[1]);
    }

    /// <summary>Python: <c>get_face_detector_score</c>.</summary>
    public static double GetFaceDetectorScore(FaceFusion.Types.Face face) => face.ScoreSet.Detector;

    private static float[] AsBoundingBox(FaceFusion.Types.Face face)
    {
        if (face.BoundingBox is float[] boundingBox)
        {
            return boundingBox;
        }

        throw new ArgumentException("Face.BoundingBox must be a float[4] (see FaceHelper's bounding box helpers).", nameof(face));
    }

    // -----------------------------------------------------------------
    // Filters
    // -----------------------------------------------------------------

    /// <summary>Python: <c>filter_faces_by_gender</c>.</summary>
    public static IReadOnlyList<FaceFusion.Types.Face> FilterFacesByGender(IReadOnlyList<FaceFusion.Types.Face> faces, Gender gender)
    {
        var filterFaces = new List<FaceFusion.Types.Face>();

        foreach (var face in faces)
        {
            if (face.Gender == gender)
            {
                filterFaces.Add(face);
            }
        }

        return filterFaces;
    }

    /// <summary>
    /// Python: <c>filter_faces_by_age</c>. Python builds <c>age = range(face_selector_age_start,
    /// face_selector_age_end)</c> (a half-open <c>[start, end)</c> integer interval) and keeps a
    /// face when <c>set(face.age) &amp; set(age)</c> is non-empty, i.e. when the face's own
    /// half-open age range overlaps the query range at all — reproduced here as a direct
    /// interval-overlap test rather than materialising and intersecting two `HashSet&lt;int&gt;`
    /// (behaviourally identical for two half-open integer ranges, and does not allocate a set
    /// per face). <see cref="FaceFusion.Types.Face.Age"/> is assumed to hold plain,
    /// not-from-end <see cref="Index"/> values (as every producer of a <see cref="System.Range"/>
    /// "age bucket" in this codebase does), so <c>.Value</c> is read directly.
    /// </summary>
    public static IReadOnlyList<FaceFusion.Types.Face> FilterFacesByAge(IReadOnlyList<FaceFusion.Types.Face> faces, int faceSelectorAgeStart, int faceSelectorAgeEnd)
    {
        var filterFaces = new List<FaceFusion.Types.Face>();

        foreach (var face in faces)
        {
            var faceAgeStart = face.Age.Start.Value;
            var faceAgeEnd = face.Age.End.Value;

            // Two half-open intervals [a0, a1) and [b0, b1) overlap iff
            // max(a0, b0) < min(a1, b1). Unlike the simpler `a0 < b1 && b0 < a1` form, this is
            // also correctly false whenever *either* interval is degenerate/empty (start >=
            // end) — e.g. a0=a1=20 with b=[15,25) satisfies `a0 < b1 && b0 < a1` (20<25 and
            // 15<20) despite the empty interval having no elements to overlap with — matching
            // Python's `range(start, end)` being the empty range whenever start >= end.
            if (Math.Max(faceAgeStart, faceSelectorAgeStart) < Math.Min(faceAgeEnd, faceSelectorAgeEnd))
            {
                filterFaces.Add(face);
            }
        }

        return filterFaces;
    }

    /// <summary>Python: <c>filter_faces_by_race</c>.</summary>
    public static IReadOnlyList<FaceFusion.Types.Face> FilterFacesByRace(IReadOnlyList<FaceFusion.Types.Face> faces, Race race)
    {
        var filterFaces = new List<FaceFusion.Types.Face>();

        foreach (var face in faces)
        {
            if (face.Race == race)
            {
                filterFaces.Add(face);
            }
        }

        return filterFaces;
    }
}
