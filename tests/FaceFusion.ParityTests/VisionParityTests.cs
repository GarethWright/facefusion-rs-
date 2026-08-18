using System.Globalization;
using System.Text.Json;
using FaceFusion.Parity;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// First real cross-language parity run for the Vision layer (docs/PARITY_HARNESS.md,
/// "What is not built yet"). Ground truth was dumped from the actual Python
/// <c>facefusion.vision</c> module (via <c>tools/parity/dump_vision.py</c>), running
/// against the real example media, not synthetic data.
///
/// <para>
/// <b>Fixture layout</b> — <c>fixtures/vision/</c>:
/// <list type="bullet">
/// <item>video/&lt;name&gt;/{resolution,fps,duration,frame_total}.json — metadata scalars for
/// target-240p.mp4 and target-1080p.mp4.</item>
/// <item>video/predict_video_frame_total.json, video/restrict_trim_frame.json — case tables.</item>
/// <item>image/source_resolution.json, image/source_pixels.npy — DetectImageResolution / ReadStaticImage.</item>
/// <item>resolution/pack_resolution.json, resolution/unpack_resolution.json — case tables.</item>
/// <item>video/frames/target_240p_frame_&lt;n&gt;.npy — decoded frames at n = 0, 1, 50, 150, 269.</item>
/// <item>frame/{fit_contain,fit_cover,restrict}_*.npy — resize helpers, computed from the
/// frame-0 fixture as input (loaded from the .npy, not re-decoded — see below).</item>
/// <item>color/{match_frame_color,equalize_frame_color_*,blend_frame_*}.npy — colour helpers,
/// likewise computed from frame-0/frame-150 fixtures as input.</item>
/// </list>
/// </para>
///
/// <para>
/// <b>Why the frame-processing fixtures load their input from .npy instead of decoding the
/// video in C#.</b> <see cref="Vision"/> decodes video frames via
/// <see cref="OpenCvSharp.VideoCapture"/>, while Python decodes them through an ffmpeg
/// subprocess pipe (<c>facefusion/video_manager.py</c>/<c>ffmpeg.py</c>) — two independent
/// decode paths that, per <see cref="ReadStaticVideoFrameIsCloseToFfmpegDecode"/> below, do
/// not agree pixel-for-pixel (small YUV→BGR conversion differences, PSNR ≈ 49 dB, see that
/// test's remarks). If the resize/colour-math tests fed a C#-decoded frame into
/// e.g. <see cref="Vision.FitContainFrame"/> and compared against a Python fixture computed
/// from the ffmpeg-decoded frame, a mismatch could come from either the decode divergence or
/// a genuine bug in the resize math, and the test could not tell them apart. Loading the same
/// Python-decoded bytes on both sides isolates what these tests are actually meant to check —
/// whether the OpenCV arithmetic in <see cref="Vision"/> matches Python's — which is why they
/// hold to the tight, near-zero tolerance PARITY_HARNESS.md prescribes for OpenCV-arithmetic
/// comparisons.
/// </para>
/// </summary>
public sealed class VisionParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "vision");

    private const string Video240p = "/tmp/facefusion-test-examples/target-240p.mp4";
    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    // -----------------------------------------------------------------
    // Video metadata: OpenCvSharp.VideoCapture vs Python's ffprobe-backed values.
    // -----------------------------------------------------------------

    public static IEnumerable<object[]> VideoMetadataCases()
    {
        yield return new object[] { "target_240p", Video240p };
        yield return new object[] { "target_1080p", "/tmp/facefusion-test-examples/target-1080p.mp4" };
    }

    /// <summary>
    /// Measures the divergence documented in docs/IMPLEMENTATION_STATUS.md ("Open divergences
    /// in the Vision port"): <see cref="Vision"/> sources frame count/fps/resolution/duration
    /// from <see cref="OpenCvSharp.VideoCapture"/> properties, Python sources them from
    /// ffprobe. For both example videos the two demuxers agree exactly (see the port report),
    /// so no fix (e.g. the IVideoMetadataProvider interface sketched in that doc) is applied
    /// here — but this test is what would catch it the day a container shows up where they
    /// disagree, so the tolerance stays exact rather than padded defensively.
    /// </summary>
    [Theory]
    [MemberData(nameof(VideoMetadataCases))]
    public void VideoMetadataMatchesFfprobe(string caseName, string videoPath)
    {

        var expectedResolution = ReadJsonIntPair(Path.Combine(FixturesDirectory, "video", caseName, "resolution.json"));
        var expectedFps = ReadJsonDouble(Path.Combine(FixturesDirectory, "video", caseName, "fps.json"));
        var expectedDuration = ReadJsonDouble(Path.Combine(FixturesDirectory, "video", caseName, "duration.json"));
        var expectedFrameTotal = ReadJsonInt(Path.Combine(FixturesDirectory, "video", caseName, "frame_total.json"));

        var actualResolution = FaceFusion.Vision.Vision.DetectVideoResolution(videoPath);
        var actualFps = FaceFusion.Vision.Vision.DetectVideoFps(videoPath);
        var actualDuration = FaceFusion.Vision.Vision.DetectVideoDuration(videoPath);
        var actualFrameTotal = FaceFusion.Vision.Vision.CountVideoFrameTotal(videoPath);

        Assert.NotNull(actualResolution);
        Assert.Equal(expectedResolution, (actualResolution!.Value.Width, actualResolution.Value.Height));
        Assert.NotNull(actualFps);
        Assert.Equal(expectedFps, actualFps!.Value, 6);
        Assert.Equal(expectedDuration, actualDuration, 6);
        Assert.Equal(expectedFrameTotal, actualFrameTotal);
    }

    [Fact]
    public void PredictVideoFrameTotalMatches()
    {

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, "video", "predict_video_frame_total.json")));

        foreach (var caseElement in document.RootElement.EnumerateArray())
        {
            var fps = caseElement.GetProperty("fps").GetDouble();
            var trimFrameStart = caseElement.GetProperty("trim_frame_start").GetInt32();
            var trimFrameEnd = caseElement.GetProperty("trim_frame_end").GetInt32();
            var expected = caseElement.GetProperty("result").GetInt32();

            var actual = FaceFusion.Vision.Vision.PredictVideoFrameTotal(Video240p, fps, trimFrameStart, trimFrameEnd);

            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void RestrictTrimFrameMatches()
    {

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, "video", "restrict_trim_frame.json")));

        foreach (var caseElement in document.RootElement.EnumerateArray())
        {
            int? trimFrameStart = caseElement.GetProperty("trim_frame_start").ValueKind == JsonValueKind.Null
                ? null
                : caseElement.GetProperty("trim_frame_start").GetInt32();
            int? trimFrameEnd = caseElement.GetProperty("trim_frame_end").ValueKind == JsonValueKind.Null
                ? null
                : caseElement.GetProperty("trim_frame_end").GetInt32();
            var resultElement = caseElement.GetProperty("result");
            var expected = (resultElement[0].GetInt32(), resultElement[1].GetInt32());

            var actual = FaceFusion.Vision.Vision.RestrictTrimFrame(Video240p, trimFrameStart, trimFrameEnd);

            Assert.Equal(expected, (actual.Start, actual.End));
        }
    }

    // -----------------------------------------------------------------
    // Image resolution / resolution packing (pure math — exact).
    // -----------------------------------------------------------------

    [Fact]
    public void DetectImageResolutionMatches()
    {

        var expected = ReadJsonIntPair(Path.Combine(FixturesDirectory, "image", "source_resolution.json"));
        var actual = FaceFusion.Vision.Vision.DetectImageResolution(SourceImage);

        Assert.NotNull(actual);
        Assert.Equal(expected, (actual!.Value.Width, actual.Value.Height));
    }

    [Fact]
    public void PackResolutionMatches()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, "resolution", "pack_resolution.json")));

        foreach (var caseElement in document.RootElement.EnumerateArray())
        {
            var input = caseElement.GetProperty("input");
            var resolution = new Resolution(input[0].GetInt32(), input[1].GetInt32());
            var expected = caseElement.GetProperty("packed").GetString();

            Assert.Equal(expected, FaceFusion.Vision.Vision.PackResolution(resolution));
        }
    }

    [Fact]
    public void UnpackResolutionMatches()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, "resolution", "unpack_resolution.json")));

        foreach (var caseElement in document.RootElement.EnumerateArray())
        {
            var input = caseElement.GetProperty("input").GetString()!;
            var output = caseElement.GetProperty("output");
            var expected = new Resolution(output[0].GetInt32(), output[1].GetInt32());

            Assert.Equal(expected, FaceFusion.Vision.Vision.UnpackResolution(input));
        }
    }

    // -----------------------------------------------------------------
    // Image / video frame decode.
    // -----------------------------------------------------------------

    /// <summary>
    /// JPEG decode: OpenCvSharp's <c>Cv2.ImRead</c> vs Python's <c>cv2.imread</c>. Both bind
    /// libjpeg (whichever build each wheel/package bundles); decode is bit-exact here across
    /// the full 1024x1024x3 example image, so the tolerance is 0/0 — a real divergence would
    /// mean the two builds disagree on JPEG decoding, which is worth knowing about exactly,
    /// not smoothing over.
    /// </summary>
    [Fact]
    public void ReadStaticImageMatchesPythonDecode()
    {

        using var image = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(image);

        var expected = NpyReader.Load(Path.Combine(FixturesDirectory, "image", "source_pixels.npy")).AsDoubles();
        var actual = MatToBgrDoubles(image!);

        var result = TensorComparison.Compare(actual, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    /// <summary>
    /// <b>Real parity defect, measured rather than assumed.</b> <see cref="Vision.ReadVideoFrame"/>
    /// decodes through <see cref="OpenCvSharp.VideoCapture"/> (its own internal ffmpeg-backed
    /// demuxer/decoder); Python decodes the same frame through a plain <c>ffmpeg</c>
    /// subprocess piping raw <c>bgr24</c> (facefusion/ffmpeg.py:create_video_reader). Both
    /// ultimately go through ffmpeg, but via different code paths with (presumably) different
    /// default swscale flags for the YUV→BGR conversion — <c>restrict_color_transfer</c> is a
    /// no-op for this video (target-240p.mp4's color_transfer is smpte170m, not one of the two
    /// HDR transfers that function rewrites), so the difference is not a colour-transfer bug,
    /// it is the two decoders' default chroma conversion disagreeing at the margins.
    ///
    /// <para>
    /// Measured across frames 0, 1, 50, 150, 269 of target-240p.mp4: about 60% of pixels
    /// differ, by at most 2-3 of 255 per channel, giving PSNR ≈ 49.3 dB (SSIM was not usable
    /// here — <see cref="ImageMetrics.Ssim"/> takes a single-channel plane, and this is a
    /// concrete difference in decode backend, not an "OpenCV does the same arithmetic on both
    /// sides" case, so this is exactly the situation PARITY_HARNESS.md's "final frames compare
    /// with PSNR/SSIM ... assert a threshold, not equality" applies to). 45 dB is used as the
    /// threshold: comfortably above the ~49 dB measured, tight enough to fail if the decode
    /// divergence gets meaningfully worse.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(50)]
    [InlineData(150)]
    [InlineData(269)]
    public void ReadStaticVideoFrameIsCloseToFfmpegDecode(int frameNumber)
    {

        using var frame = FaceFusion.Vision.Vision.ReadStaticVideoFrame(Video240p, frameNumber);
        Assert.NotNull(frame);

        var expected = NpyReader.Load(Path.Combine(FixturesDirectory, "video", "frames", $"target_240p_frame_{frameNumber}.npy")).AsDoubles();
        var actual = MatToBgrDoubles(frame!);

        var psnr = ImageMetrics.Psnr(actual, expected);
        Assert.True(psnr > 45.0, $"frame {frameNumber}: PSNR {psnr:F2} dB fell at/below the 45 dB threshold — the OpenCvSharp.VideoCapture vs ffmpeg-pipe decode divergence got worse.");
    }

    // -----------------------------------------------------------------
    // Resize helpers — OpenCV arithmetic, exact tolerance, fed from the .npy-decoded frame
    // (see class remarks for why).
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(300, 300)]
    [InlineData(600, 200)]
    public void FitContainFrameMatchesExactly(int width, int height)
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var actualMat = FaceFusion.Vision.Vision.FitContainFrame(source, new Resolution(width, height));

        AssertExactMatch(actualMat, $"frame/fit_contain_{width}x{height}.npy");
    }

    [Theory]
    [InlineData(300, 300)]
    [InlineData(600, 200)]
    public void FitCoverFrameMatchesExactly(int width, int height)
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var actualMat = FaceFusion.Vision.Vision.FitCoverFrame(source, new Resolution(width, height));

        AssertExactMatch(actualMat, $"frame/fit_cover_{width}x{height}.npy");
    }

    [Fact]
    public void RestrictFrameDownscalesExactly()
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var actualMat = FaceFusion.Vision.Vision.RestrictFrame(source, new Resolution(200, 100));

        AssertExactMatch(actualMat, "frame/restrict_200x100.npy");
    }

    [Fact]
    public void RestrictFrameNoOpMatchesExactly()
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var actualMat = FaceFusion.Vision.Vision.RestrictFrame(source, new Resolution(800, 800));

        AssertExactMatch(actualMat, "frame/restrict_800x800.npy");
    }

    // -----------------------------------------------------------------
    // Colour helpers — OpenCV arithmetic, exact tolerance.
    // -----------------------------------------------------------------

    [Fact]
    public void MatchFrameColorMatchesExactly()
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var target = LoadFrameFixtureAsMat("video/frames/target_240p_frame_150.npy");
        using var actualMat = FaceFusion.Vision.Vision.MatchFrameColor(source, target);

        AssertExactMatch(actualMat, "color/match_frame_color.npy");
    }

    [Fact]
    public void EqualizeFrameColorSmallSizeMatchesExactly()
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var target = LoadFrameFixtureAsMat("video/frames/target_240p_frame_150.npy");
        using var actualMat = FaceFusion.Vision.Vision.EqualizeFrameColor(source, target, new Resolution(16, 16));

        AssertExactMatch(actualMat, "color/equalize_frame_color_small.npy");
    }

    [Fact]
    public void EqualizeFrameColorFullSizeMatchesExactly()
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var target = LoadFrameFixtureAsMat("video/frames/target_240p_frame_150.npy");
        using var actualMat = FaceFusion.Vision.Vision.EqualizeFrameColor(source, target, new Resolution(target.Cols, target.Rows));

        AssertExactMatch(actualMat, "color/equalize_frame_color_full.npy");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.25)]
    [InlineData(0.75)]
    [InlineData(1.0)]
    public void BlendFrameMatchesExactly(double blendFactor)
    {
        using var source = LoadFrameFixtureAsMat("video/frames/target_240p_frame_0.npy");
        using var target = LoadFrameFixtureAsMat("video/frames/target_240p_frame_150.npy");
        using var actualMat = FaceFusion.Vision.Vision.BlendFrame(source, target, blendFactor);

        var fixtureName = blendFactor.ToString("0.0#", CultureInfo.InvariantCulture);
        AssertExactMatch(actualMat, $"color/blend_frame_{fixtureName}.npy");
    }

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    private static void AssertExactMatch(Mat actualMat, string relativeFixturePath)
    {
        var expected = NpyReader.Load(Path.Combine(FixturesDirectory, relativeFixturePath)).AsDoubles();
        var actual = MatToBgrDoubles(actualMat);

        var result = TensorComparison.Compare(actual, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, $"{relativeFixturePath}: {result.Describe()}");
    }

    /// <summary>Builds a CV_8UC3 <see cref="Mat"/> directly from a dumped (H, W, 3) uint8 .npy fixture.</summary>
    private static Mat LoadFrameFixtureAsMat(string relativeFixturePath)
    {
        var array = NpyReader.Load(Path.Combine(FixturesDirectory, relativeFixturePath));
        var shape = array.Shape;
        Assert.Equal(3, shape.Count);
        Assert.Equal(3, shape[2]);
        Assert.Equal("uint8", array.DType);

        var height = shape[0];
        var width = shape[1];
        var mat = new Mat(height, width, MatType.CV_8UC3);
        var bytes = array.RawData.ToArray();
        System.Runtime.InteropServices.Marshal.Copy(bytes, 0, mat.Data, bytes.Length);
        return mat;
    }

    /// <summary>Flattens a CV_8UC3 <see cref="Mat"/> to row-major (H, W, 3) doubles, same layout as the .npy fixtures.</summary>
    private static double[] MatToBgrDoubles(Mat mat)
    {
        mat.GetArray(out Vec3b[] pixels);
        var result = new double[pixels.Length * 3];

        for (var i = 0; i < pixels.Length; i++)
        {
            result[(i * 3) + 0] = pixels[i].Item0;
            result[(i * 3) + 1] = pixels[i].Item1;
            result[(i * 3) + 2] = pixels[i].Item2;
        }

        return result;
    }

    private static (int, int) ReadJsonIntPair(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        return (root[0].GetInt32(), root[1].GetInt32());
    }

    private static double ReadJsonDouble(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.GetDouble();
    }

    private static int ReadJsonInt(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.GetInt32();
    }
}
