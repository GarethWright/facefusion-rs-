using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/face_debugger/{core,types,choices}.py</c>. There is no
/// <c>tests/test_face_debugger.py</c> in the Python suite (only <c>tests/test_cli_face_debugger.py</c>,
/// a subprocess/CLI end-to-end test that needs <c>FaceFusion.Cli</c>, not built yet — Phase 6),
/// so this file exercises the pure, model-free arithmetic (<c>calculate_scale</c>, the
/// <c>numpy.array_equal</c>/<c>numpy.any</c> landmark-comparison helpers, bounding-box int32
/// truncation, the <see cref="FaceDebuggerItem"/> flags mapping) directly. The real rendered
/// output (bounding box, landmarks, box mask) is verified against a frame rendered by the real
/// <c>facefusion.processors.modules.face_debugger.core.debug_face</c> in
/// <c>tests/FaceFusion.ParityTests/ProcessorParityTests2.cs</c>.
/// </summary>
public sealed class FaceDebuggerTests
{
    // -----------------------------------------------------------------
    // choices.py — face_debugger_items
    // -----------------------------------------------------------------

    [Fact]
    public void FaceDebuggerItemFlagsCoverEveryLiteralValue()
    {
        // Python: FaceDebuggerItem = Literal['bounding-box', 'face-landmark-5',
        // 'face-landmark-5/68', 'face-landmark-68', 'face-landmark-68/5', 'face-mask'].
        Assert.Equal("bounding-box", FaceDebuggerItem.BoundingBox.ToWireName());
        Assert.Equal("face-landmark-5", FaceDebuggerItem.FaceLandmark5.ToWireName());
        Assert.Equal("face-landmark-5/68", FaceDebuggerItem.FaceLandmark5On68.ToWireName());
        Assert.Equal("face-landmark-68", FaceDebuggerItem.FaceLandmark68.ToWireName());
        Assert.Equal("face-landmark-68/5", FaceDebuggerItem.FaceLandmark68On5.ToWireName());
        Assert.Equal("face-mask", FaceDebuggerItem.FaceMask.ToWireName());
    }

    [Fact]
    public void FaceDebuggerItemsCombineAsFlags()
    {
        // CLI default: 'face-landmark-5/68 face-mask'.
        var combined = FaceDebuggerItem.FaceLandmark5On68 | FaceDebuggerItem.FaceMask;
        Assert.True(combined.HasFlag(FaceDebuggerItem.FaceLandmark5On68));
        Assert.True(combined.HasFlag(FaceDebuggerItem.FaceMask));
        Assert.False(combined.HasFlag(FaceDebuggerItem.BoundingBox));
        Assert.False(combined.HasFlag(FaceDebuggerItem.FaceLandmark5));
    }

    // -----------------------------------------------------------------
    // calculate_scale
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(270, 1)] // round(270 / 270) = 1
    [InlineData(2160, 8)] // round(2160 / 270) = 8
    [InlineData(27, 1)] // round(27 / 270) = 0, clamped to min 1
    [InlineData(3000, 10)] // round(3000 / 270) = 11, clamped to max 10
    [InlineData(135, 1)] // round(135 / 270) = round(0.5) = 0 (banker's rounding: half-to-even), clamped to 1
    [InlineData(405, 2)] // round(405 / 270) = round(1.5) = 2 (half-to-even)
    public void CalculateScaleMatchesPython(int frameHeight, int expectedScale)
    {
        using var frame = new Mat(frameHeight, 100, MatType.CV_8UC3, Scalar.Black);
        Assert.Equal(expectedScale, FaceDebugger.CalculateScale(frame));
    }

    // -----------------------------------------------------------------
    // draw_bounding_box — int32 truncation, angle-dependent border line
    // -----------------------------------------------------------------

    [Fact]
    public void DrawBoundingBoxTruncatesFloatCoordinatesTowardZero()
    {
        // Python: bounding_box.astype(numpy.int32) truncates toward zero, not round — 10.9
        // truncates to 10, not 11.
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Black);
        var face = MakeFace(new[] { 10.9f, 10.9f, 100.1f, 100.1f }, angle: 0);

        FaceDebugger.DrawBoundingBox(face, frame);

        // The rectangle's top-left corner pixel should be red (box_color = (0, 0, 255) BGR) at
        // (10, 10), not at (11, 11) — confirms truncation rather than rounding was used.
        var pixelAtTruncated = frame.At<Vec3b>(10, 10);
        Assert.Equal(255, pixelAtTruncated.Item2); // R channel of BGR
    }

    [Theory]
    [InlineData(0)]
    [InlineData(90)]
    [InlineData(180)]
    [InlineData(270)]
    public void DrawBoundingBoxDrawsForEveryKnownAngle(int angle)
    {
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Black);
        var face = MakeFace(new[] { 20f, 20f, 200f, 200f }, angle);

        // Must not throw for any of the four angles debug_face's Python source special-cases.
        var exception = Record.Exception(() => FaceDebugger.DrawBoundingBox(face, frame));
        Assert.Null(exception);
    }

    // -----------------------------------------------------------------
    // draw_face_landmark_* — numpy.any / numpy.array_equal semantics
    // -----------------------------------------------------------------

    [Fact]
    public void DrawFaceLandmark5DoesNothingWhenAllZero()
    {
        // Python: `if numpy.any(face_landmark_5):` — an all-zero landmark array is falsy and
        // skips drawing entirely.
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Black);
        var zeroLandmarks = new float[5, 2];
        var face = MakeFaceWithLandmarks(zeroLandmarks, zeroLandmarks, zeroLandmarks, zeroLandmarks);

        FaceDebugger.DrawFaceLandmark5(face, frame);

        // Frame must remain untouched (still all-black).
        frame.GetArray(out Vec3b[] pixels);
        Assert.All(pixels, pixel => Assert.Equal(0, pixel.Item0 + pixel.Item1 + pixel.Item2));
    }

    [Fact]
    public void DrawFaceLandmark5On68UsesCyanWhenLandmarksAreIdentical()
    {
        // Python: `if numpy.array_equal(face_landmark_5, face_landmark_5_68): point_color =
        // 255, 255, 0` (cyan in BGR).
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Black);
        var landmarks = new float[5, 2];
        for (var i = 0; i < 5; i++)
        {
            landmarks[i, 0] = 50 + i;
            landmarks[i, 1] = 50 + i;
        }

        var face = MakeFaceWithLandmarks(landmarks, landmarks, landmarks, landmarks);

        FaceDebugger.DrawFaceLandmark5On68(face, frame);

        var point = frame.At<Vec3b>(50, 50);
        Assert.Equal(255, point.Item0); // B
        Assert.Equal(255, point.Item1); // G
        Assert.Equal(0, point.Item2);   // R
    }

    [Fact]
    public void DrawFaceLandmark5On68UsesGreenWhenLandmarksDiffer()
    {
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Black);
        var landmark5 = new float[5, 2];
        var landmark5On68 = new float[5, 2];
        for (var i = 0; i < 5; i++)
        {
            landmark5[i, 0] = 50 + i;
            landmark5[i, 1] = 50 + i;
            landmark5On68[i, 0] = 60 + i;
            landmark5On68[i, 1] = 60 + i;
        }

        var face = MakeFaceWithLandmarks(landmark5, landmark5On68, landmark5On68, landmark5On68);

        FaceDebugger.DrawFaceLandmark5On68(face, frame);

        var point = frame.At<Vec3b>(60, 60);
        Assert.Equal(0, point.Item0);
        Assert.Equal(255, point.Item1);
        Assert.Equal(0, point.Item2);
    }

    [Fact]
    public void DrawFaceLandmarkUsesOrangeForRefillOrigin()
    {
        // Python: `if target_face.origin == 'refill': point_color = 0, 165, 255` (orange in
        // BGR), overriding whatever colour numpy.array_equal picked.
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Black);
        var landmarks = new float[5, 2];
        landmarks[0, 0] = 50;
        landmarks[0, 1] = 50;

        var face = MakeFaceWithLandmarks(landmarks, landmarks, landmarks, landmarks) with { Origin = "refill" };

        FaceDebugger.DrawFaceLandmark5On68(face, frame);

        var point = frame.At<Vec3b>(50, 50);
        Assert.Equal(0, point.Item0);
        Assert.Equal(165, point.Item1);
        Assert.Equal(255, point.Item2);
    }

    // -----------------------------------------------------------------
    // draw_face_mask — box mask is model-free, exercises the full mask/warp/contour path.
    // -----------------------------------------------------------------

    [Fact]
    public void DrawFaceMaskWithBoxTypeDrawsSomethingAndDoesNotThrow()
    {
        // face_mask_types = ['box'] (facefusion/program.py's own default) needs no ONNX model,
        // so this exercises DrawFaceMask's full warp/mask-reduce/inverse-warp/contour pipeline
        // without any model dependency. A tighter, real-frame version of this same call is
        // checked pixel-for-pixel against Python in ProcessorParityTests2.
        using var frame = new Mat(270, 270, MatType.CV_8UC3, Scalar.Gray);
        var face = MakePlausibleFace();

        var exception = Record.Exception(() => FaceDebugger.DrawFaceMask(
            face, frame,
            faceMaskTypes: new[] { FaceMaskType.Box },
            faceMaskPadding: new Padding(0, 0, 0, 0),
            faceMaskAreas: Array.Empty<FaceMaskArea>(),
            faceMaskRegions: Array.Empty<FaceMaskRegion>(),
            faceOccluderModel: FaceOccluderModel.Xseg1,
            faceParserModel: FaceParserModel.BisenetResnet34,
            occluderInferencePool: null,
            parserInferencePool: null));

        Assert.Null(exception);

        // The mask contour is drawn in green ((0, 255, 0) BGR) by default (no 'refill' origin,
        // landmark_5 != landmark_5_68 for this synthetic face) — at least one pixel should have
        // changed from the initial gray fill.
        frame.GetArray(out Vec3b[] pixels);
        Assert.Contains(pixels, pixel => pixel.Item0 != 128 || pixel.Item1 != 128 || pixel.Item2 != 128);
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private static Types.Face MakeFace(float[] boundingBox, int angle) =>
        MakeFaceWithLandmarks(new float[5, 2], new float[5, 2], new float[68, 2], new float[68, 2])
            with
            {
                BoundingBox = boundingBox,
                Angle = angle,
            };

    private static Types.Face MakeFaceWithLandmarks(float[,] five, float[,] fiveOn68, float[,] sixtyEight, float[,] sixtyEightOn5) =>
        new Types.Face(
            Origin: "detect",
            BoundingBox: new[] { 0f, 0f, 10f, 10f },
            ScoreSet: new FaceScoreSet(1.0, 1.0),
            LandmarkSet: new FaceLandmarkSet(five, fiveOn68, sixtyEight, sixtyEightOn5),
            Angle: 0,
            Embedding: Array.Empty<float>(),
            EmbeddingNorm: Array.Empty<float>(),
            Age: 20..30,
            Gender: Gender.Male,
            Race: Race.White);

    /// <summary>A synthetic but geometrically plausible face (roughly centred 5-point landmark
    /// layout) so <c>WarpFaceByFaceLandmark5</c>'s affine estimation does not degenerate.</summary>
    private static Types.Face MakePlausibleFace()
    {
        var five = new float[5, 2]
        {
            { 100f, 100f }, // left eye
            { 160f, 100f }, // right eye
            { 130f, 130f }, // nose
            { 105f, 160f }, // left mouth corner
            { 155f, 160f }, // right mouth corner
        };
        var sixtyEight = new float[68, 2];
        for (var i = 0; i < 68; i++)
        {
            sixtyEight[i, 0] = 100f + (i % 8 * 5f);
            sixtyEight[i, 1] = 100f + (i / 8 * 5f);
        }

        return MakeFaceWithLandmarks(five, five, sixtyEight, sixtyEight) with { BoundingBox = new[] { 90f, 90f, 170f, 170f } };
    }
}
