using FaceFusion.Face;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of ground-truth checks for <c>facefusion/face_masker.py</c>. There is no
/// <c>tests/test_face_masker.py</c> upstream, so this file exercises <see cref="FaceMasker"/>
/// against real Python output captured ad hoc from the actual
/// <c>facefusion.face_masker.create_box_mask</c>/<c>create_area_mask</c> functions (opencv-python
/// 5.0.0, numpy 2.4.6) rather than against a ported Python test — see each test's comment for the
/// exact Python invocation used to produce the expected values.
///
/// <para>
/// <b>Model-backed <c>create_occlusion_mask</c>/<c>create_region_mask</c> are not covered here</b>
/// — they need an <see cref="Microsoft.ML.OnnxRuntime.InferenceSession"/>, which this project does
/// not construct (no model-file gating infrastructure here, unlike
/// <c>FaceFusion.ParityTests</c>). Ground truth for those two lives in
/// <c>tests/FaceFusion.ParityTests/FaceMaskerParityTests.cs</c> instead, gated to skip when
/// <c>xseg_1.onnx</c>/<c>bisenet_resnet_18.onnx</c> are not present under <c>.assets/models/</c>.
/// </para>
///
/// <para>
/// <b>Tolerance.</b> Both functions are pure OpenCV arithmetic (<c>cv2.GaussianBlur</c>,
/// <c>cv2.fillConvexPoly</c>/<c>cv2.convexHull</c>), so per PARITY_HARNESS.md the expected
/// divergence from OpenCvSharp's native OpenCV build is ~0 — these assert at <c>1e-4</c>, loose
/// enough to absorb the documented bilinear/blur-kernel float rounding between builds, tight
/// enough to catch a real algorithm mismatch (e.g. a padding-order or threshold-direction bug).
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class FaceMaskerTests
{
    private const float Tolerance = 1e-4f;

    // -----------------------------------------------------------------
    // create_box_mask
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// crop = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
    /// box = create_box_mask(crop, 0.3, (10, 15, 20, 5))  # (top, right, bottom, left)
    /// box[::4, ::4]  # captured below, row-major, 8x8 subsample
    /// </code>
    /// Exercises every one of the four independently-sized padding edges plus the Gaussian
    /// blur — the highest-value spot to catch a swapped top/bottom or left/right padding index.
    /// </summary>
    [Fact]
    public void CreateBoxMaskMatchesPython()
    {
        // Row-major, subsampled every 4th row/col of the real 32x32 output (rows 0,4,...,28;
        // cols 0,4,...,28).
        float[,] expected =
        {
            { 0.001069f, 0.009090f, 0.009131f, 0.009131f, 0.009131f, 0.009131f, 0.009130f, 0.002744f },
            { 0.110256f, 0.937145f, 0.941443f, 0.941443f, 0.941443f, 0.941443f, 0.941317f, 0.282930f },
            { 0.117114f, 0.995434f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.999866f, 0.300528f },
            { 0.117114f, 0.995434f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.999866f, 0.300528f },
            { 0.117114f, 0.995434f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.999866f, 0.300528f },
            { 0.117114f, 0.995434f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 0.999866f, 0.300528f },
            { 0.110256f, 0.937145f, 0.941443f, 0.941443f, 0.941443f, 0.941443f, 0.941317f, 0.282930f },
            { 0.000535f, 0.004545f, 0.004566f, 0.004566f, 0.004566f, 0.004566f, 0.004565f, 0.001372f },
        };

        using var crop = new Mat(32, 32, MatType.CV_8UC3, Scalar.All(0));
        using var box = FaceMasker.CreateBoxMask(crop, 0.3, new Padding(Top: 10, Right: 15, Bottom: 20, Left: 5));

        Assert.Equal(32, box.Rows);
        Assert.Equal(32, box.Cols);
        Assert.Equal(MatType.CV_32FC1, box.Type());

        box.GetArray(out float[] flat);
        for (var r = 0; r < 8; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                var actual = flat[(r * 4 * 32) + (c * 4)];
                Assert.True(
                    Math.Abs(actual - expected[r, c]) <= Tolerance,
                    $"box[{r * 4},{c * 4}] = {actual}, expected {expected[r, c]}");
            }
        }
    }

    /// <summary>Python: <c>blur_amount &lt;= 0</c> skips the Gaussian blur entirely (Python's
    /// own <c>if blur_amount &gt; 0:</c> guard) — <c>face_mask_blur = 0</c> forces
    /// <c>blur_amount = int(32 * 0.5 * 0) = 0</c>, so the mask is a crisp (unblurred) rectangle.</summary>
    [Fact]
    public void CreateBoxMaskWithZeroBlurSkipsGaussianBlur()
    {
        using var crop = new Mat(20, 20, MatType.CV_8UC3, Scalar.All(0));
        using var box = FaceMasker.CreateBoxMask(crop, 0.0, new Padding(0, 0, 0, 0));

        // blur_area = max(0 // 2, 1) = 1, so a 1px border on every edge is still zeroed even
        // with zero blur (Python: box_mask[:max(blur_area, ...)] = 0 always zeros at least 1px).
        box.GetArray(out float[] flat);
        Assert.Equal(0f, flat[0]); // (0,0) corner: inside the 1px zeroed border.
        Assert.Equal(1f, flat[(10 * 20) + 10]); // center: untouched, still exactly 1.0 (no blur).
    }

    // -----------------------------------------------------------------
    // create_area_mask
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// crop = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
    /// landmark_68 = numpy.zeros((68, 2), dtype=numpy.float32)  # every point defaults to (16,16)
    /// landmark_68[48] = (8, 8); landmark_68[51] = (24, 8)
    /// landmark_68[54] = (24, 24); landmark_68[57] = (8, 24)
    /// area = create_area_mask(crop, landmark_68, ['mouth'])
    /// area[::4, ::4]  # captured below, row-major, 8x8 subsample
    /// </code>
    /// <c>'mouth'</c> selects landmark indices 48-67 (facefusion.choices.face_mask_area_set);
    /// every unselected index sits at the crop center (16,16) so only the four placed points
    /// form the convex hull — a diamond spanning (8,8)-(24,8)-(24,24)-(8,24).
    /// </summary>
    [Fact]
    public void CreateAreaMaskMatchesPython()
    {
        float[,] expected =
        {
            { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },
            { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },
            { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },
            { 0f, 0f, 0f, 0.313050f, 0.477009f, 0.313343f, 0f, 0f },
            { 0f, 0f, 0f, 0.477009f, 0.661441f, 0.477338f, 0f, 0f },
            { 0f, 0f, 0f, 0.313343f, 0.477338f, 0.313636f, 0f, 0f },
            { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },
            { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },
        };

        var landmark68 = new float[68, 2];
        for (var i = 0; i < 68; i++)
        {
            landmark68[i, 0] = 16;
            landmark68[i, 1] = 16;
        }

        landmark68[48, 0] = 8; landmark68[48, 1] = 8;
        landmark68[51, 0] = 24; landmark68[51, 1] = 8;
        landmark68[54, 0] = 24; landmark68[54, 1] = 24;
        landmark68[57, 0] = 8; landmark68[57, 1] = 24;

        using var crop = new Mat(32, 32, MatType.CV_8UC3, Scalar.All(0));
        using var area = FaceMasker.CreateAreaMask(crop, landmark68, new[] { FaceMaskArea.Mouth });

        Assert.Equal(32, area.Rows);
        Assert.Equal(32, area.Cols);
        Assert.Equal(MatType.CV_32FC1, area.Type());

        area.GetArray(out float[] flat);
        for (var r = 0; r < 8; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                var actual = flat[(r * 4 * 32) + (c * 4)];
                Assert.True(
                    Math.Abs(actual - expected[r, c]) <= Tolerance,
                    $"area[{r * 4},{c * 4}] = {actual}, expected {expected[r, c]}");
            }
        }
    }

    /// <summary>
    /// <b>Documented divergence, found by this test (not "fixed" — see below for why).</b>
    /// Python: an empty <c>face_mask_areas</c> list leaves <c>landmark_points</c> empty, so
    /// <c>cv2.convexHull(numpy.empty((0, 2)))</c> feeds a zero-point array to
    /// <c>cv2.fillConvexPoly</c>, which is <b>not</b> a graceful no-op — real opencv-python 5.0.0
    /// raises <c>cv2.error: ... Assertion failed) points.checkVector(2, CV_32S) &gt;= 0 in
    /// function 'fillConvexPoly'</c> (verified interactively: <c>create_area_mask(
    /// numpy.zeros((16,16,3), uint8), numpy.zeros((68,2), float32), [])</c> raises). OpenCvSharp's
    /// native build does <b>not</b> raise for the same zero-point <c>Cv2.FillConvexPoly</c> call —
    /// it is a silent no-op, so this port returns an all-zero mask instead of throwing. Not
    /// reproduced as a crash here deliberately: an empty (or entirely-unrecognised)
    /// <c>face_mask_areas</c> list is not a reachable CLI input (every real caller either omits
    /// the flag, getting the non-empty default, or the sanitizer's own choice validation rejects
    /// an unrecognised area name before this function ever runs) — there is no code path in this
    /// port that could pass an empty list here today, so forcing a crash purely for Python-crash
    /// parity on an unreachable input would trade a real robustness improvement for no observable
    /// behavioural benefit. Recorded here so the divergence is visible rather than silently
    /// absorbed.
    /// </summary>
    [Fact]
    public void CreateAreaMaskWithNoAreasIsANoOpInThisPortUnlikePythonWhichCrashes()
    {
        var landmark68 = new float[68, 2];
        using var crop = new Mat(16, 16, MatType.CV_8UC3, Scalar.All(0));

        using var area = FaceMasker.CreateAreaMask(crop, landmark68, Array.Empty<FaceMaskArea>());

        area.GetArray(out float[] flat);
        Assert.All(flat, value => Assert.Equal(0f, value, Tolerance));
    }
}
