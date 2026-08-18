using FaceFusion.Face;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the sorting/filtering logic in <c>facefusion/face_selector.py</c>. There is no
/// <c>tests/test_face_selector.py</c> in the Python suite, so these are hand-written property
/// tests exercising every branch directly against fabricated <see cref="Types.Face"/> records
/// (no ONNX models or media needed — this module is pure logic over already-detected faces).
///
/// <para>
/// <b>Nullable age-filter condition — the case this file exists to pin down.</b> See
/// <see cref="FaceSelector.SortAndFilterFaces"/>'s own remarks for the full reasoning; the
/// short version is Python's <c>if age_start or age_end:</c> treats both <c>None</c> (unset)
/// and <c>0</c> (an explicit bound) as falsy, so the filter is skipped in both cases —
/// <see cref="AgeFilterTreatsUnsetAndExplicitZeroAsFalsy"/>,
/// <see cref="AgeFilterAppliesWhenOnlyOneBoundIsNonzero"/> and
/// <see cref="AgeFilterAppliesWhenBothBoundsAreNonzero"/> below cover, respectively: both unset
/// (skip), one bound unset, one nonzero (applies, unset bound falls back to 0), and both
/// nonzero (applies).
/// </para>
/// </summary>
public sealed class FaceSelectorTests
{
    // -----------------------------------------------------------------
    // Test fixtures
    // -----------------------------------------------------------------

    private static Types.Face CreateFace(
        float[] boundingBox,
        double detectorScore = 0.9,
        Gender gender = Gender.Female,
        Race race = Race.White,
        System.Range? age = null,
        float[]? embeddingNorm = null)
    {
        return new Types.Face(
            Origin: "detect",
            BoundingBox: boundingBox,
            ScoreSet: new FaceScoreSet(detectorScore, 0.9),
            LandmarkSet: new FaceLandmarkSet(
                Five: new float[5, 2],
                FiveOn68: new float[5, 2],
                SixtyEight: new float[68, 2],
                SixtyEightOn5: new float[68, 2]),
            Angle: 0,
            Embedding: embeddingNorm ?? new float[] { 1f, 0f, 0f },
            EmbeddingNorm: embeddingNorm ?? new float[] { 1f, 0f, 0f },
            Age: age ?? (25..30),
            Gender: gender,
            Race: race);
    }

    // -----------------------------------------------------------------
    // SortFacesByOrder
    // -----------------------------------------------------------------

    [Fact]
    public void SortFacesByOrderLeftRightAndRightLeft()
    {
        var left = CreateFace(new float[] { 0, 0, 10, 10 });
        var middle = CreateFace(new float[] { 20, 0, 30, 10 });
        var right = CreateFace(new float[] { 40, 0, 50, 10 });
        var faces = new[] { right, left, middle };

        var leftRight = FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.LeftRight);
        Assert.Equal(new[] { left, middle, right }, leftRight);

        var rightLeft = FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.RightLeft);
        Assert.Equal(new[] { right, middle, left }, rightLeft);
    }

    [Fact]
    public void SortFacesByOrderTopBottomAndBottomTop()
    {
        var top = CreateFace(new float[] { 0, 0, 10, 10 });
        var middle = CreateFace(new float[] { 0, 20, 10, 30 });
        var bottom = CreateFace(new float[] { 0, 40, 10, 50 });
        var faces = new[] { bottom, top, middle };

        Assert.Equal(new[] { top, middle, bottom }, FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.TopBottom));
        Assert.Equal(new[] { bottom, middle, top }, FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.BottomTop));
    }

    [Fact]
    public void SortFacesByOrderSmallLargeAndLargeSmall()
    {
        var small = CreateFace(new float[] { 0, 0, 10, 10 });    // area 100
        var medium = CreateFace(new float[] { 0, 0, 20, 20 });   // area 400
        var large = CreateFace(new float[] { 0, 0, 40, 40 });    // area 1600
        var faces = new[] { large, small, medium };

        Assert.Equal(new[] { small, medium, large }, FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.SmallLarge));
        Assert.Equal(new[] { large, medium, small }, FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.LargeSmall));
    }

    [Fact]
    public void SortFacesByOrderBestWorstAndWorstBest()
    {
        var worst = CreateFace(new float[] { 0, 0, 10, 10 }, detectorScore: 0.1);
        var middle = CreateFace(new float[] { 0, 0, 10, 10 }, detectorScore: 0.5);
        var best = CreateFace(new float[] { 0, 0, 10, 10 }, detectorScore: 0.99);
        var faces = new[] { middle, worst, best };

        Assert.Equal(new[] { best, middle, worst }, FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.BestWorst));
        Assert.Equal(new[] { worst, middle, best }, FaceSelector.SortFacesByOrder(faces, FaceSelectorOrder.WorstBest));
    }

    [Fact]
    public void GetBoundingBoxHelpersMatchPython()
    {
        var face = CreateFace(new float[] { 10, 20, 30, 50 });

        Assert.Equal(10, FaceSelector.GetBoundingBoxLeft(face));
        Assert.Equal(20, FaceSelector.GetBoundingBoxTop(face));
        Assert.Equal(20 * 30, FaceSelector.GetBoundingBoxArea(face)); // (30-10)*(50-20)
        Assert.Equal(0.9, FaceSelector.GetFaceDetectorScore(face));
    }

    // -----------------------------------------------------------------
    // Gender / race / age filters
    // -----------------------------------------------------------------

    [Fact]
    public void FilterFacesByGenderKeepsOnlyMatchingGender()
    {
        var female = CreateFace(new float[] { 0, 0, 1, 1 }, gender: Gender.Female);
        var male = CreateFace(new float[] { 0, 0, 1, 1 }, gender: Gender.Male);

        var result = FaceSelector.FilterFacesByGender(new[] { female, male }, Gender.Male);

        Assert.Equal(new[] { male }, result);
    }

    [Fact]
    public void FilterFacesByRaceKeepsOnlyMatchingRace()
    {
        var white = CreateFace(new float[] { 0, 0, 1, 1 }, race: Race.White);
        var asian = CreateFace(new float[] { 0, 0, 1, 1 }, race: Race.Asian);

        var result = FaceSelector.FilterFacesByRace(new[] { white, asian }, Race.Asian);

        Assert.Equal(new[] { asian }, result);
    }

    [Fact]
    public void FilterFacesByAgeKeepsOverlappingRangesOnly()
    {
        var young = CreateFace(new float[] { 0, 0, 1, 1 }, age: 0..10);
        var overlapping = CreateFace(new float[] { 0, 0, 1, 1 }, age: 18..30);
        var old = CreateFace(new float[] { 0, 0, 1, 1 }, age: 60..80);

        // Python: `range(18, 40)` — half-open [18, 40).
        var result = FaceSelector.FilterFacesByAge(new[] { young, overlapping, old }, 18, 40);

        Assert.Equal(new[] { overlapping }, result);
    }

    [Fact]
    public void FilterFacesByAgeExcludesTouchingButNonOverlappingRanges()
    {
        // Python half-open ranges: [0, 10) and [10, 20) share no integers.
        var face = CreateFace(new float[] { 0, 0, 1, 1 }, age: 0..10);

        var result = FaceSelector.FilterFacesByAge(new[] { face }, 10, 20);

        Assert.Empty(result);
    }

    [Fact]
    public void FilterFacesByAgeWithStartEqualEndIsEmptyQueryRange()
    {
        // Python: range(20, 20) is the empty range, so nothing ever matches.
        var face = CreateFace(new float[] { 0, 0, 1, 1 }, age: 15..25);

        var result = FaceSelector.FilterFacesByAge(new[] { face }, 20, 20);

        Assert.Empty(result);
    }

    // -----------------------------------------------------------------
    // Nullable age-filter condition (SortAndFilterFaces)
    // -----------------------------------------------------------------

    [Fact]
    public void AgeFilterTreatsUnsetAndExplicitZeroAsFalsy()
    {
        // A face whose age range would never overlap any real query (age 200..210) — if the
        // filter were (wrongly) applied here it would empty the result; since the condition is
        // falsy in every one of these cases, the filter must never run and the face must
        // survive unfiltered.
        var face = CreateFace(new float[] { 0, 0, 1, 1 }, age: 200..210);
        var faces = new[] { face };

        // Both unset (Python: both None).
        var bothUnset = FaceSelector.SortAndFilterFaces(Array.Empty<Types.Face>(), faces, FaceSelectorOrder.LeftRight, null, null, null, null);
        Assert.Equal(faces, bothUnset);

        // Both explicitly zero (Python: both 0, still falsy).
        var bothZero = FaceSelector.SortAndFilterFaces(Array.Empty<Types.Face>(), faces, FaceSelectorOrder.LeftRight, null, null, 0, 0);
        Assert.Equal(faces, bothZero);

        // One unset, one explicitly zero.
        var mixedFalsy = FaceSelector.SortAndFilterFaces(Array.Empty<Types.Face>(), faces, FaceSelectorOrder.LeftRight, null, null, null, 0);
        Assert.Equal(faces, mixedFalsy);
    }

    [Fact]
    public void AgeFilterAppliesWhenOnlyOneBoundIsNonzero()
    {
        var face = CreateFace(new float[] { 0, 0, 1, 1 }, age: 25..30);
        var faces = new[] { face };

        // ageStart unset (falls back to 0), ageEnd = 10 (nonzero, truthy) -> filter runs with
        // range [0, 10), which does not overlap [25, 30) -> face is dropped.
        var onlyEndSet = FaceSelector.SortAndFilterFaces(Array.Empty<Types.Face>(), faces, FaceSelectorOrder.LeftRight, null, null, null, 10);
        Assert.Empty(onlyEndSet);

        // ageStart = 20 (nonzero, truthy), ageEnd unset (falls back to 0) -> filter runs with
        // range [20, 0), the empty range -> face is dropped too (matches the divergence
        // documented on SortAndFilterFaces: Python would crash on `range(20, None)` here; this
        // port substitutes 0 rather than reproducing the crash).
        var onlyStartSet = FaceSelector.SortAndFilterFaces(Array.Empty<Types.Face>(), faces, FaceSelectorOrder.LeftRight, null, null, 20, null);
        Assert.Empty(onlyStartSet);
    }

    [Fact]
    public void AgeFilterAppliesWhenBothBoundsAreNonzero()
    {
        var inside = CreateFace(new float[] { 0, 0, 1, 1 }, age: 25..30);
        var outside = CreateFace(new float[] { 0, 0, 1, 1 }, age: 60..70);
        var faces = new[] { inside, outside };

        var result = FaceSelector.SortAndFilterFaces(Array.Empty<Types.Face>(), faces, FaceSelectorOrder.LeftRight, null, null, 20, 40);

        Assert.Equal(new[] { inside }, result);
    }

    // -----------------------------------------------------------------
    // Gender/race 'auto' resolution and its operator-precedence oddity
    // -----------------------------------------------------------------

    [Fact]
    public void ExplicitGenderAndRaceFilterWithoutSourceFaces()
    {
        var male = CreateFace(new float[] { 0, 0, 1, 1 }, gender: Gender.Male, race: Race.Asian);
        var female = CreateFace(new float[] { 0, 0, 1, 1 }, gender: Gender.Female, race: Race.Asian);

        var result = FaceSelector.SortAndFilterFaces(
            Array.Empty<Types.Face>(), new[] { male, female }, FaceSelectorOrder.LeftRight,
            FaceSelectorGender.Male, FaceSelectorRace.Asian, null, null);

        Assert.Equal(new[] { male }, result);
    }

    [Fact]
    public void AutoGenderResolvesFromLargestSourceFace()
    {
        var smallSource = CreateFace(new float[] { 0, 0, 10, 10 }, gender: Gender.Female); // area 100
        var largeSource = CreateFace(new float[] { 0, 0, 40, 40 }, gender: Gender.Male);    // area 1600
        var sourceFaces = new[] { smallSource, largeSource };

        var male = CreateFace(new float[] { 0, 0, 1, 1 }, gender: Gender.Male);
        var female = CreateFace(new float[] { 0, 0, 1, 1 }, gender: Gender.Female);

        var result = FaceSelector.SortAndFilterFaces(
            sourceFaces, new[] { male, female }, FaceSelectorOrder.LeftRight,
            FaceSelectorGender.Auto, null, null, null);

        // The largest source face is Male, so 'auto' resolves to Male and only the male target
        // face survives.
        Assert.Equal(new[] { male }, result);
    }

    [Fact]
    public void RaceAutoAloneTriggersResolutionBlockEvenWithEmptySourceFacesPerPythonOperatorPrecedence()
    {
        // Python: `if source_faces and face_selector_gender == 'auto' or face_selector_race == 'auto':`
        // parses as `(source_faces and gender == 'auto') or (race == 'auto')` — an explicit
        // `face_selector_race == 'auto'` alone (no source faces) still satisfies the condition.
        // With no source faces, `get_first(sort_faces_by_order([], 'large-small'))` is null, so
        // nothing actually gets resolved and race stays unresolved (not a member of
        // choices.races) — the filter is then skipped and both faces survive. This test exists
        // to document/pin that shape, not to observe a different outcome than the "condition
        // is false" case would produce.
        var white = CreateFace(new float[] { 0, 0, 1, 1 }, race: Race.White);
        var asian = CreateFace(new float[] { 0, 0, 1, 1 }, race: Race.Asian);

        var result = FaceSelector.SortAndFilterFaces(
            Array.Empty<Types.Face>(), new[] { white, asian }, FaceSelectorOrder.LeftRight,
            null, FaceSelectorRace.Auto, null, null);

        Assert.Equal(new[] { white, asian }, result);
    }

    [Fact]
    public void EmptyTargetFacesShortCircuits()
    {
        var result = FaceSelector.SortAndFilterFaces(
            Array.Empty<Types.Face>(), Array.Empty<Types.Face>(), FaceSelectorOrder.LeftRight,
            FaceSelectorGender.Male, FaceSelectorRace.Asian, 10, 20);

        Assert.Empty(result);
    }

    // -----------------------------------------------------------------
    // Distance / matching
    // -----------------------------------------------------------------

    [Fact]
    public void CalculateFaceDistanceIsOneMinusDotProduct()
    {
        var face = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var reference = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 0f, 1f, 0f });

        // Orthogonal unit vectors -> dot = 0 -> distance = 1.
        Assert.Equal(1.0, FaceSelector.CalculateFaceDistance(face, reference), 6);

        var identical = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        Assert.Equal(0.0, FaceSelector.CalculateFaceDistance(face, identical), 6);
    }

    [Fact]
    public void CompareFacesUsesInterpolatedDistanceThreshold()
    {
        var face = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var closeReference = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var farReference = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { -1f, 0f, 0f });

        Assert.True(FaceSelector.CompareFaces(face, closeReference, 0.6));
        Assert.False(FaceSelector.CompareFaces(face, farReference, 0.6));
    }

    [Fact]
    public void FindMatchFacesReturnsAllTargetsWithinDistance()
    {
        var reference = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var closeTarget = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var farTarget = CreateFace(new float[] { 0, 0, 1, 1 }, embeddingNorm: new float[] { -1f, 0f, 0f });

        var result = FaceSelector.FindMatchFaces(new[] { reference }, new[] { closeTarget, farTarget }, 0.6);

        Assert.Equal(new[] { closeTarget }, result);
    }

    // -----------------------------------------------------------------
    // SelectFaces orchestration (fake delegates — no real detection needed)
    // -----------------------------------------------------------------

    [Fact]
    public void SelectFacesManyModeReturnsAllSortedFilteredFaces()
    {
        using var frame = new OpenCvSharp.Mat(4, 4, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0));

        var male = CreateFace(new float[] { 20, 0, 30, 10 }, gender: Gender.Male);
        var female = CreateFace(new float[] { 0, 0, 10, 10 }, gender: Gender.Female);

        var result = FaceSelector.SelectFaces(
            frame, new[] { frame }, new[] { frame },
            FaceSelectorMode.Many, faceTrackerScore: 0, faceSelectorOrder: FaceSelectorOrder.LeftRight,
            faceSelectorGender: null, faceSelectorRace: null, faceSelectorAgeStart: null, faceSelectorAgeEnd: null,
            referenceFacePosition: 0, referenceFaceDistance: 0.6,
            getStaticFaces: _ => new[] { male, female },
            refillFaces: _ => Array.Empty<Types.Face>());

        Assert.Equal(new[] { female, male }, result); // left-right order
    }

    [Fact]
    public void SelectFacesOneModeReturnsSingleFace()
    {
        using var frame = new OpenCvSharp.Mat(4, 4, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0));

        var first = CreateFace(new float[] { 0, 0, 10, 10 });
        var second = CreateFace(new float[] { 20, 0, 30, 10 });

        var result = FaceSelector.SelectFaces(
            frame, new[] { frame }, new[] { frame },
            FaceSelectorMode.One, faceTrackerScore: 0, faceSelectorOrder: FaceSelectorOrder.LeftRight,
            faceSelectorGender: null, faceSelectorRace: null, faceSelectorAgeStart: null, faceSelectorAgeEnd: null,
            referenceFacePosition: 0, referenceFaceDistance: 0.6,
            getStaticFaces: _ => new[] { second, first },
            refillFaces: _ => Array.Empty<Types.Face>());

        Assert.Equal(new[] { first }, result);
    }

    [Fact]
    public void SelectFacesOneModeReturnsEmptyWhenNoFacesDetected()
    {
        using var frame = new OpenCvSharp.Mat(4, 4, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0));

        var result = FaceSelector.SelectFaces(
            frame, new[] { frame }, new[] { frame },
            FaceSelectorMode.One, faceTrackerScore: 0, faceSelectorOrder: FaceSelectorOrder.LeftRight,
            faceSelectorGender: null, faceSelectorRace: null, faceSelectorAgeStart: null, faceSelectorAgeEnd: null,
            referenceFacePosition: 0, referenceFaceDistance: 0.6,
            getStaticFaces: _ => Array.Empty<Types.Face>(),
            refillFaces: _ => Array.Empty<Types.Face>());

        Assert.Empty(result);
    }

    [Fact]
    public void SelectFacesReferenceModeMatchesByEmbeddingDistance()
    {
        using var sourceFrame = new OpenCvSharp.Mat(4, 4, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0));
        using var targetFrame = new OpenCvSharp.Mat(4, 4, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0));
        using var referenceFrame = new OpenCvSharp.Mat(4, 4, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0));

        var referenceFace = CreateFace(new float[] { 0, 0, 10, 10 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var closeTarget = CreateFace(new float[] { 0, 0, 10, 10 }, embeddingNorm: new float[] { 1f, 0f, 0f });
        var farTarget = CreateFace(new float[] { 20, 0, 30, 10 }, embeddingNorm: new float[] { -1f, 0f, 0f });

        // Three distinct (blank, pixel-content-irrelevant) frames so the fake getStaticFaces
        // below can tell apart Python's three separate call sites (source / target-middle /
        // reference) by which frame it was asked about, exactly as the real
        // FaceCreator.GetStaticFaces would be called with three different frames.
        var result = FaceSelector.SelectFaces(
            referenceFrame, new[] { sourceFrame }, new[] { targetFrame },
            FaceSelectorMode.Reference, faceTrackerScore: 0, faceSelectorOrder: FaceSelectorOrder.LeftRight,
            faceSelectorGender: null, faceSelectorRace: null, faceSelectorAgeStart: null, faceSelectorAgeEnd: null,
            referenceFacePosition: 0, referenceFaceDistance: 0.6,
            getStaticFaces: frames =>
            {
                if (ReferenceEquals(frames[0], referenceFrame))
                {
                    return new[] { referenceFace };
                }

                if (ReferenceEquals(frames[0], targetFrame))
                {
                    return new[] { closeTarget, farTarget };
                }

                return Array.Empty<Types.Face>();
            },
            refillFaces: _ => Array.Empty<Types.Face>());

        Assert.Equal(new[] { closeTarget }, result);
    }
}
