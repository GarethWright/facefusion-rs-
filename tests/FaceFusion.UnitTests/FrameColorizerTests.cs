using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of ground-truth checks for
/// <c>facefusion/processors/modules/frame_colorizer/{core,types,choices}.py</c>. There is no
/// <c>tests/test_frame_colorizer.py</c> upstream, so this file exercises
/// <see cref="FrameColorizer"/>'s pure pre/post-processing (<c>prepare_temp_frame</c>,
/// <c>merge_color_frame</c>) against real Python output captured ad hoc from the actual
/// <c>facefusion.processors.modules.frame_colorizer.core</c> functions (opencv-python 5.0.0,
/// numpy 2.4.6) on a tiny crafted 2x2 image — see each test's comment for the exact Python
/// invocation. <b>The Lab colour-space arithmetic is the highest-risk part of this module per
/// the assignment brief</b> — every test below exercises it directly (both the <c>ddcolor</c>
/// gray-&gt;Lab-&gt;RGB round trip in <c>prepare_temp_frame</c> and the a/b-channel recombination
/// in <c>merge_color_frame</c>) rather than only checking shapes.
///
/// <para>
/// <b>Real end-to-end ONNX-model ground truth</b> (the actual <c>ddcolor</c>/<c>deoldify_stable</c>
/// models run against a real frame) lives in
/// <c>tests/FaceFusion.ParityTests/ProcessorParityTests2.cs</c> instead, gated to skip when the
/// model files are not present under <c>.assets/models/</c>.
/// </para>
///
/// <para>
/// <b>Tolerance.</b> This is pure OpenCV arithmetic (<c>cv2.cvtColor</c>, <c>cv2.resize</c>) —
/// per PARITY_HARNESS.md, expect ~0 divergence; asserted at <c>1e-4</c>.
/// </para>
/// </summary>
public sealed class FrameColorizerTests
{
    private const float Tolerance = 1e-4f;

    private static Mat MakeTinyFrame()
    {
        // BGR, 2x2, deliberately saturated/varied per channel so every Lab conversion step
        // actually moves through non-trivial values.
        var mat = new Mat(2, 2, MatType.CV_8UC3);
        var pixels = new[]
        {
            new Vec3b { Item0 = 10, Item1 = 20, Item2 = 200 },
            new Vec3b { Item0 = 230, Item1 = 100, Item2 = 30 },
            new Vec3b { Item0 = 50, Item1 = 200, Item2 = 50 },
            new Vec3b { Item0 = 0, Item1 = 0, Item2 = 0 },
        };
        mat.SetArray(pixels);
        return mat;
    }

    private static void AssertClose(float[] actual, float[] expected, float tolerance = Tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(actual[i] - expected[i]) <= tolerance,
                $"index {i}: actual {actual[i]}, expected {expected[i]}");
        }
    }

    // -----------------------------------------------------------------
    // prepare_temp_frame — ddcolor branch (the gray -> Lab(L only) -> RGB round trip)
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// gray = cv2.cvtColor(frame, COLOR_BGR2GRAY); gray_rgb = cv2.cvtColor(gray, COLOR_GRAY2RGB)
    /// f = gray_rgb.astype(float32) / 255.0; lab = cv2.cvtColor(f, COLOR_RGB2Lab)
    /// l, _, _ = cv2.split(lab); combined = stack([l, zeros, zeros], axis=-1)
    /// rgb = cv2.cvtColor(combined, COLOR_Lab2RGB)  # a = b = 0 -> RGB is achromatic (R=G=B)
    /// resized = cv2.resize(rgb, (2, 2)).astype(float32).transpose(2, 0, 1)
    /// </code>
    /// on the 2x2 frame from <see cref="MakeTinyFrame"/>. Every pixel's three channels come out
    /// equal (the Lab round trip with <c>a = b = 0</c> is achromatic by construction) — the
    /// highest-value check that the L channel (and only the L channel) survived the round trip.
    /// </summary>
    [Fact]
    public void PrepareTempFrameDdcolorMatchesPython()
    {
        float[] expected =
        {
            // CHW: R plane, G plane, B plane (each pixel's 3 channels are equal).
            0.285205f, 0.367305f, 0.541105f, 0.0f,
            0.285205f, 0.367305f, 0.541105f, 0.0f,
            0.285205f, 0.367305f, 0.541105f, 0.0f,
        };

        using var frame = MakeTinyFrame();
        var actual = FrameColorizer.PrepareTempFrame(frame, FrameColorizerModelType.Ddcolor, new Resolution(2, 2));

        AssertClose(actual, expected);
    }

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// gray = cv2.cvtColor(frame, COLOR_BGR2GRAY); gray_rgb = cv2.cvtColor(gray, COLOR_GRAY2RGB)
    /// resized = cv2.resize(gray_rgb, (2, 2)).astype(float32).transpose(2, 0, 1)
    /// </code>
    /// on the same 2x2 frame — the <c>deoldify</c> branch skips the Lab round trip entirely
    /// (Python's <c>if model_type == 'ddcolor':</c> guard), so this is plain grayscale expanded
    /// to 3 identical channels, unscaled (still <c>[0, 255]</c>, not <c>/255.0</c>).
    /// </summary>
    [Fact]
    public void PrepareTempFrameDeoldifyMatchesPython()
    {
        float[] expected = { 73f, 94f, 138f, 0f, 73f, 94f, 138f, 0f, 73f, 94f, 138f, 0f };

        using var frame = MakeTinyFrame();
        var actual = FrameColorizer.PrepareTempFrame(frame, FrameColorizerModelType.Deoldify, new Resolution(2, 2));

        AssertClose(actual, expected, tolerance: 0.5f); // integer grayscale values — exact except for rounding mode.
    }

    // -----------------------------------------------------------------
    // merge_color_frame — ddcolor branch (original L + model a/b -> BGR, round, uint8)
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from a synthetic 2-channel (a, b) "model output" the same size as the
    /// frame (skipping the resize, which is exercised separately in the box/area mask parity
    /// tests and the real end-to-end frame_colorizer parity tests):
    /// <code>
    /// temp = (frame / 255.0).astype(float32); lab = cv2.cvtColor(temp, COLOR_BGR2Lab)
    /// l = lab[:, :, 0]; combined = stack([l, a, b], axis=-1)
    /// bgr = cv2.cvtColor(combined, COLOR_Lab2BGR); result = numpy.round(bgr * 255.0).astype(uint8)
    /// </code>
    /// </summary>
    [Fact]
    public void MergeColorFrameDdcolorMatchesPython()
    {
        // CHW: a-plane then b-plane, row-major H*W = 4.
        float[] abChwData = { 5f, 10f, -8f, 0f, -3f, 2f, 1f, 0f };
        byte[] expected = { 105, 98, 106, 105, 102, 126, 171, 178, 158, 0, 0, 0 };

        using var frame = MakeTinyFrame();
        using var merged = FrameColorizer.MergeColorFrame(frame, abChwData, outputChannels: 2, outputHeight: 2, outputWidth: 2, FrameColorizerModelType.Ddcolor);

        merged.GetArray(out Vec3b[] pixels);
        var actual = new byte[pixels.Length * 3];
        for (var i = 0; i < pixels.Length; i++)
        {
            actual[(i * 3) + 0] = pixels[i].Item0;
            actual[(i * 3) + 1] = pixels[i].Item1;
            actual[(i * 3) + 2] = pixels[i].Item2;
        }

        Assert.Equal(expected, actual);
    }

    // -----------------------------------------------------------------
    // merge_color_frame — deoldify branch (the channel-index quirk, see FrameColorizer's
    // class remarks: BGR2RGB swap, truncating (not rounding) uint8 cast, BGR2Lab again, keep
    // a/b, merge with the ORIGINAL frame's own blue channel as if it were L, LAB2BGR)
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// color = cv2.cvtColor(model_output, COLOR_BGR2RGB); color = color.astype(uint8)  # truncate
    /// lab = cv2.cvtColor(color, COLOR_BGR2Lab); _, a, b = cv2.split(lab)
    /// merged = cv2.merge([frame[:, :, 0], a, b])  # frame's own B channel stands in for L
    /// result = cv2.cvtColor(merged, COLOR_Lab2BGR)
    /// </code>
    /// with a synthetic 3-channel "model output" already at the frame's resolution.
    /// </summary>
    [Fact]
    public void MergeColorFrameDeoldifyReproducesChannelIndexQuirk()
    {
        // CHW: channel0, channel1, channel2 planes, row-major H*W = 4 each.
        float[] chwData =
        {
            100f, 10f, 5f, 0f,
            150f, 250f, 5f, 0f,
            30f, 60f, 5f, 0f,
        };
        byte[] expected = { 0, 27, 0, 71, 255, 52, 47, 48, 47, 0, 0, 0 };

        using var frame = MakeTinyFrame();
        using var merged = FrameColorizer.MergeColorFrame(frame, chwData, outputChannels: 3, outputHeight: 2, outputWidth: 2, FrameColorizerModelType.Deoldify);

        merged.GetArray(out Vec3b[] pixels);
        var actual = new byte[pixels.Length * 3];
        for (var i = 0; i < pixels.Length; i++)
        {
            actual[(i * 3) + 0] = pixels[i].Item0;
            actual[(i * 3) + 1] = pixels[i].Item1;
            actual[(i * 3) + 2] = pixels[i].Item2;
        }

        Assert.Equal(expected, actual);
    }

    // -----------------------------------------------------------------
    // blend_color_frame — the simplified double-negation algebra
    // -----------------------------------------------------------------

    [Fact]
    public void BlendColorFrameWithBlend100ReturnsColorFrameUnchanged()
    {
        using var temp = MakeTinyFrame();
        using var color = new Mat(2, 2, MatType.CV_8UC3, Scalar.All(77));

        using var blended = FrameColorizer.BlendColorFrame(temp, color, frameColorizerBlend: 100);

        blended.GetArray(out Vec3b[] pixels);
        Assert.All(pixels, p => Assert.True(p.Item0 == 77 && p.Item1 == 77 && p.Item2 == 77));
    }

    [Fact]
    public void BlendColorFrameWithBlend0ReturnsTempFrameUnchanged()
    {
        using var temp = MakeTinyFrame();
        using var color = new Mat(2, 2, MatType.CV_8UC3, Scalar.All(77));

        using var blended = FrameColorizer.BlendColorFrame(temp, color, frameColorizerBlend: 0);

        temp.GetArray(out Vec3b[] expectedPixels);
        blended.GetArray(out Vec3b[] actualPixels);
        Assert.Equal(expectedPixels, actualPixels);
    }
}
