using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_vision.py</c>.
///
/// <para>
/// The Python tests download <c>source.jpg</c> / <c>target-240p.mp4</c> / <c>target-1080p.mp4</c>
/// from a GitHub release and derive a whole family of fixture files from them with ffmpeg.
/// Neither the network egress nor an <c>ffmpeg</c>/<c>ffprobe</c> binary is available in this
/// container. Per the assignment brief, every fixture below is synthesised locally instead with
/// <c>OpenCvSharp</c> itself (<see cref="Cv2.ImWrite(string, Mat, int[])"/> for images,
/// <see cref="VideoWriter"/> with the <c>mp4v</c> fourcc — bundled in the OpenCV native build, no
/// external ffmpeg needed — for video). This exercises the real encode/decode path rather than
/// mocking it, and wherever the Python test asserts a specific literal value (resolutions, trim
/// frame math, fps, exact-second durations, cache-of-N-frames lengths), the fixture is
/// constructed so that literal continues to hold — see per-test comments. No test in this file is
/// skipped: nothing here actually requires the missing binaries or network access once the
/// fixtures are synthesised, so there is no honest reason to skip any of it.
/// </para>
/// </summary>
public sealed class VisionFixture : IDisposable
{
    public string Directory { get; }

    public string LandscapeImagePath { get; }
    public string PortraitImagePath { get; }
    public string LargeImagePath { get; }
    public string UnicodeImagePath { get; }

    /// <summary>270 frames @ 25 fps, 64x48 — chosen so CountVideoFrameTotal / DetectVideoFps /
    /// DetectVideoDuration reproduce the Python suite's literal expected values (270, 25.0,
    /// 10.8) exactly, and so PredictVideoFrameTotal / CountTrimFrameTotal / RestrictTrimFrame
    /// can use the same literal assertions as the Python tests. Each frame's pixels are seeded
    /// from the frame index so frame-identity (not just frame-count) is verifiable.</summary>
    public string Video270Path { get; }

    public string VideoFps25Path { get; }
    public string VideoFps30Path { get; }
    public string VideoFps60Path { get; }

    public string VideoWidePath { get; }
    public string VideoTallPath { get; }
    public string VideoLargeWidePath { get; }
    public string VideoLargeTallPath { get; }

    public VisionFixture()
    {
        Directory = Path.Combine(Path.GetTempPath(), "facefusion-vision-tests-" + Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(Directory);

        LandscapeImagePath = WriteImage("landscape.png", 426, 226);
        PortraitImagePath = WriteImage("portrait.png", 226, 426);
        LargeImagePath = WriteImage("large.png", 2048, 1080);
        UnicodeImagePath = WriteImage("目标-240p.png", 426, 226);

        Video270Path = WriteVideo("v270.mp4", frameTotal: 270, fps: 25.0, width: 64, height: 48);

        VideoFps25Path = WriteVideo("v25fps.mp4", frameTotal: 50, fps: 25.0, width: 64, height: 48);
        VideoFps30Path = WriteVideo("v30fps.mp4", frameTotal: 60, fps: 30.0, width: 64, height: 48);
        VideoFps60Path = WriteVideo("v60fps.mp4", frameTotal: 120, fps: 60.0, width: 64, height: 48);

        VideoWidePath = WriteVideo("vwide.mp4", frameTotal: 3, fps: 25.0, width: 426, height: 226);
        VideoTallPath = WriteVideo("vtall.mp4", frameTotal: 3, fps: 25.0, width: 226, height: 426);
        VideoLargeWidePath = WriteVideo("vlargewide.mp4", frameTotal: 3, fps: 25.0, width: 2048, height: 1080);
        VideoLargeTallPath = WriteVideo("vlargetall.mp4", frameTotal: 3, fps: 25.0, width: 1080, height: 2048);
    }

    public void Dispose()
    {
        if (System.IO.Directory.Exists(Directory))
        {
            System.IO.Directory.Delete(Directory, recursive: true);
        }
    }

    private string WriteImage(string fileName, int width, int height)
    {
        var path = Path.Combine(Directory, fileName);
        using var image = new Mat(new Size(width, height), MatType.CV_8UC3, new Scalar(60, 90, 120));
        Cv2.ImWrite(path, image);
        return path;
    }

    private string WriteVideo(string fileName, int frameTotal, double fps, int width, int height)
    {
        var path = Path.Combine(Directory, fileName);
        using var writer = new OpenCvSharp.VideoWriter(path, FourCC.FromFourChars('m', 'p', '4', 'v'), fps, new Size(width, height));

        for (var frameNumber = 0; frameNumber < frameTotal; frameNumber++)
        {
            using var frame = new Mat(new Size(width, height), MatType.CV_8UC3, new Scalar(frameNumber % 256, 50, 100));
            writer.Write(frame);
        }

        return path;
    }
}

public sealed class VisionTests : IClassFixture<VisionFixture>
{
    private readonly VisionFixture _fixture;

    public VisionTests(VisionFixture fixture)
    {
        _fixture = fixture;
    }

    // -----------------------------------------------------------------
    // Image I/O
    // -----------------------------------------------------------------

    [Fact]
    public void ReadImage()
    {
        using (var landscape = FaceFusion.Vision.Vision.ReadImage(_fixture.LandscapeImagePath))
        {
            Assert.NotNull(landscape);
            Assert.Equal(226, landscape!.Rows);
            Assert.Equal(426, landscape.Cols);
            Assert.Equal(3, landscape.Channels());
        }

        using (var unicode = FaceFusion.Vision.Vision.ReadImage(_fixture.UnicodeImagePath))
        {
            Assert.NotNull(unicode);
            Assert.Equal(226, unicode!.Rows);
            Assert.Equal(426, unicode.Cols);
        }

        Assert.Null(FaceFusion.Vision.Vision.ReadImage("invalid"));
    }

    [Fact]
    public void WriteImage()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadImage(_fixture.LandscapeImagePath);
        Assert.NotNull(visionFrame);

        var outputPath = Path.Combine(_fixture.Directory, "output.jpg");
        Assert.True(FaceFusion.Vision.Vision.WriteImage(outputPath, visionFrame!));

        var unicodeOutputPath = Path.Combine(_fixture.Directory, "输出.png");
        Assert.True(FaceFusion.Vision.Vision.WriteImage(unicodeOutputPath, visionFrame!));
    }

    [Fact]
    public void DetectImageResolution()
    {
        Assert.Equal(new Resolution(426, 226), FaceFusion.Vision.Vision.DetectImageResolution(_fixture.LandscapeImagePath));
        Assert.Equal(new Resolution(226, 426), FaceFusion.Vision.Vision.DetectImageResolution(_fixture.PortraitImagePath));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.DetectImageResolution(_fixture.LargeImagePath));
        Assert.Null(FaceFusion.Vision.Vision.DetectImageResolution("invalid"));
    }

    [Fact]
    public void RestrictImageResolution()
    {
        Assert.Equal(new Resolution(426, 226), FaceFusion.Vision.Vision.RestrictImageResolution(_fixture.LargeImagePath, new Resolution(426, 226)));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.RestrictImageResolution(_fixture.LargeImagePath, new Resolution(2048, 1080)));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.RestrictImageResolution(_fixture.LargeImagePath, new Resolution(4096, 2160)));
    }

    // -----------------------------------------------------------------
    // Video I/O
    // -----------------------------------------------------------------

    [Fact]
    public void ReadVideoFrame()
    {
        using (var frame = FaceFusion.Vision.Vision.ReadVideoFrame(_fixture.Video270Path))
        {
            Assert.NotNull(frame);
            Assert.Equal(48, frame!.Rows);
            Assert.Equal(64, frame.Cols);
            Assert.Equal(3, frame.Channels());
        }

        foreach (var frameNumber in new[] { 49, 50, 51 })
        {
            using var direct = FaceFusion.Vision.Vision.ReadVideoFrame(_fixture.Video270Path, frameNumber);
            var selected = FaceFusion.Vision.Vision.SelectVideoFrames(_fixture.Video270Path, frameNumber, 5);
            Assert.NotNull(direct);
            Assert.Equal(11, selected.Count);
            AssertMatEqual(direct!, selected[5]);
            foreach (var frame in selected)
            {
                frame.Dispose();
            }
        }

        Assert.Null(FaceFusion.Vision.Vision.ReadVideoFrame("invalid"));
    }

    [Fact]
    public void SelectVideoFrames()
    {
        foreach (var frameNumber in new[] { 50, 1, 269 })
        {
            var frames = FaceFusion.Vision.Vision.SelectVideoFrames(_fixture.Video270Path, frameNumber, 5);
            Assert.Equal(11, frames.Count);
            foreach (var frame in frames)
            {
                frame.Dispose();
            }
        }

        Assert.Empty(FaceFusion.Vision.Vision.SelectVideoFrames("invalid", 50, 5));
    }

    [Fact]
    public void CountVideoFrameTotal()
    {
        Assert.Equal(270, FaceFusion.Vision.Vision.CountVideoFrameTotal(_fixture.Video270Path));
        Assert.Equal(50, FaceFusion.Vision.Vision.CountVideoFrameTotal(_fixture.VideoFps25Path));
        Assert.Equal(60, FaceFusion.Vision.Vision.CountVideoFrameTotal(_fixture.VideoFps30Path));
        Assert.Equal(120, FaceFusion.Vision.Vision.CountVideoFrameTotal(_fixture.VideoFps60Path));
        Assert.Equal(0, FaceFusion.Vision.Vision.CountVideoFrameTotal("invalid"));
    }

    [Fact]
    public void PredictVideoFrameTotal()
    {
        Assert.Equal(50, FaceFusion.Vision.Vision.PredictVideoFrameTotal(_fixture.Video270Path, 12.5, 0, 100));
        Assert.Equal(100, FaceFusion.Vision.Vision.PredictVideoFrameTotal(_fixture.Video270Path, 25, 0, 100));
        Assert.Equal(200, FaceFusion.Vision.Vision.PredictVideoFrameTotal(_fixture.Video270Path, 25, 0, 200));
        Assert.Equal(0, FaceFusion.Vision.Vision.PredictVideoFrameTotal("invalid", 25, 0, 100));
    }

    [Fact]
    public void DetectVideoFps()
    {
        Assert.Equal(25.0, FaceFusion.Vision.Vision.DetectVideoFps(_fixture.VideoFps25Path));
        Assert.Equal(30.0, FaceFusion.Vision.Vision.DetectVideoFps(_fixture.VideoFps30Path));
        Assert.Equal(60.0, FaceFusion.Vision.Vision.DetectVideoFps(_fixture.VideoFps60Path));
        Assert.Null(FaceFusion.Vision.Vision.DetectVideoFps("invalid"));
    }

    [Fact]
    public void RestrictVideoFps()
    {
        Assert.Equal(20.0, FaceFusion.Vision.Vision.RestrictVideoFps(_fixture.Video270Path, 20.0));
        Assert.Equal(25.0, FaceFusion.Vision.Vision.RestrictVideoFps(_fixture.Video270Path, 25.0));
        Assert.Equal(25.0, FaceFusion.Vision.Vision.RestrictVideoFps(_fixture.Video270Path, 60.0));
    }

    [Fact]
    public void DetectVideoDuration()
    {
        Assert.Equal(10.8, FaceFusion.Vision.Vision.DetectVideoDuration(_fixture.Video270Path), precision: 6);
        Assert.Equal(0, FaceFusion.Vision.Vision.DetectVideoDuration("invalid"));
    }

    [Fact]
    public void CountTrimFrameTotal()
    {
        Assert.Equal(200, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, 0, 200));
        Assert.Equal(200, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, 70, 270));
        Assert.Equal(270, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, -10, null));
        Assert.Equal(0, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, null, -10));
        Assert.Equal(0, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, 280, null));
        Assert.Equal(270, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, null, 280));
        Assert.Equal(270, FaceFusion.Vision.Vision.CountTrimFrameTotal(_fixture.Video270Path, null, null));
    }

    [Fact]
    public void RestrictTrimFrame()
    {
        Assert.Equal((0, 200), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, 0, 200));
        Assert.Equal((70, 270), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, 70, 270));
        Assert.Equal((0, 270), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, -10, null));
        Assert.Equal((0, 0), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, null, -10));
        Assert.Equal((270, 270), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, 280, null));
        Assert.Equal((0, 270), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, null, 280));
        Assert.Equal((0, 270), FaceFusion.Vision.Vision.RestrictTrimFrame(_fixture.Video270Path, null, null));
    }

    [Fact]
    public void DetectVideoResolution()
    {
        Assert.Equal(new Resolution(426, 226), FaceFusion.Vision.Vision.DetectVideoResolution(_fixture.VideoWidePath));
        Assert.Equal(new Resolution(226, 426), FaceFusion.Vision.Vision.DetectVideoResolution(_fixture.VideoTallPath));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.DetectVideoResolution(_fixture.VideoLargeWidePath));
        Assert.Equal(new Resolution(1080, 2048), FaceFusion.Vision.Vision.DetectVideoResolution(_fixture.VideoLargeTallPath));
        Assert.Null(FaceFusion.Vision.Vision.DetectVideoResolution("invalid"));
    }

    [Fact]
    public void RestrictVideoResolution()
    {
        Assert.Equal(new Resolution(426, 226), FaceFusion.Vision.Vision.RestrictVideoResolution(_fixture.VideoLargeWidePath, new Resolution(426, 226)));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.RestrictVideoResolution(_fixture.VideoLargeWidePath, new Resolution(2048, 1080)));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.RestrictVideoResolution(_fixture.VideoLargeWidePath, new Resolution(4096, 2160)));
    }

    // -----------------------------------------------------------------
    // Resolution math
    // -----------------------------------------------------------------

    [Fact]
    public void ScaleResolution()
    {
        Assert.Equal(new Resolution(212, 112), FaceFusion.Vision.Vision.ScaleResolution(new Resolution(426, 226), 0.5));
        Assert.Equal(new Resolution(2048, 1080), FaceFusion.Vision.Vision.ScaleResolution(new Resolution(2048, 1080), 1.0));
        Assert.Equal(new Resolution(8192, 4320), FaceFusion.Vision.Vision.ScaleResolution(new Resolution(4096, 2160), 2.0));
    }

    [Fact]
    public void NormalizeResolution()
    {
        Assert.Equal(new Resolution(2, 2), FaceFusion.Vision.Vision.NormalizeResolution(2.5, 2.5));
        Assert.Equal(new Resolution(4, 4), FaceFusion.Vision.Vision.NormalizeResolution(3.0, 3.0));
        Assert.Equal(new Resolution(6, 6), FaceFusion.Vision.Vision.NormalizeResolution(6.5, 6.5));
    }

    [Fact]
    public void PackResolution()
    {
        Assert.Equal("0x0", FaceFusion.Vision.Vision.PackResolution(new Resolution(1, 1)));
        Assert.Equal("2x2", FaceFusion.Vision.Vision.PackResolution(new Resolution(2, 2)));
    }

    [Fact]
    public void UnpackResolution()
    {
        Assert.Equal(new Resolution(0, 0), FaceFusion.Vision.Vision.UnpackResolution("0x0"));
        Assert.Equal(new Resolution(2, 2), FaceFusion.Vision.Vision.UnpackResolution("2x2"));
    }

    // -----------------------------------------------------------------
    // Colour helpers
    // -----------------------------------------------------------------

    [Fact]
    public void CalcHistogramDifference()
    {
        using var source = CreateHueGradientFrame(64, 64);
        using var target = CreateDesaturatedFrame(source);

        Assert.Equal(1.0, FaceFusion.Vision.Vision.CalculateHistogramDifference(source, source), precision: 6);
        Assert.True(FaceFusion.Vision.Vision.CalculateHistogramDifference(source, target) < 0.5);
    }

    [Fact]
    public void MatchFrameColor()
    {
        using var source = CreateHueGradientFrame(64, 64);
        using var target = CreateDesaturatedFrame(source);
        using var output = FaceFusion.Vision.Vision.MatchFrameColor(source, target);

        Assert.True(FaceFusion.Vision.Vision.CalculateHistogramDifference(source, output) > 0.5);
    }

    // -----------------------------------------------------------------
    // Frame helpers — not covered by tests/test_vision.py (which only exercises the functions
    // above), but exercised here since they are exactly the kind of manual-Mat-indexing code
    // most likely to hide an off-by-one.
    // -----------------------------------------------------------------

    [Fact]
    public void DetectFrameOrientation()
    {
        using var landscape = new Mat(new Size(64, 32), MatType.CV_8UC3, Scalar.All(0));
        using var portrait = new Mat(new Size(32, 64), MatType.CV_8UC3, Scalar.All(0));

        Assert.Equal(Orientation.Landscape, FaceFusion.Vision.Vision.DetectFrameOrientation(landscape));
        Assert.Equal(Orientation.Portrait, FaceFusion.Vision.Vision.DetectFrameOrientation(portrait));
    }

    [Fact]
    public void RestrictFrame()
    {
        using var frame = new Mat(new Size(400, 200), MatType.CV_8UC3, Scalar.All(0));

        using var restricted = FaceFusion.Vision.Vision.RestrictFrame(frame, new Resolution(200, 200));
        Assert.Equal(100, restricted.Rows);
        Assert.Equal(200, restricted.Cols);

        using var unrestricted = FaceFusion.Vision.Vision.RestrictFrame(frame, new Resolution(800, 800));
        Assert.Equal(200, unrestricted.Rows);
        Assert.Equal(400, unrestricted.Cols);
    }

    [Fact]
    public void FitContainFrame()
    {
        using var frame = new Mat(new Size(400, 200), MatType.CV_8UC3, new Scalar(10, 20, 30));
        using var fitted = FaceFusion.Vision.Vision.FitContainFrame(frame, new Resolution(200, 200));

        Assert.Equal(200, fitted.Rows);
        Assert.Equal(200, fitted.Cols);
        // Top padding band should be zero (letterboxed), the resized frame is centred vertically.
        Assert.Equal(new Vec3b(0, 0, 0), fitted.At<Vec3b>(0, 100));
        Assert.Equal(new Vec3b(10, 20, 30), fitted.At<Vec3b>(100, 100));
    }

    [Fact]
    public void FitCoverFrame()
    {
        using var frame = new Mat(new Size(400, 200), MatType.CV_8UC3, new Scalar(10, 20, 30));
        using var covered = FaceFusion.Vision.Vision.FitCoverFrame(frame, new Resolution(200, 200));

        Assert.Equal(200, covered.Rows);
        Assert.Equal(200, covered.Cols);
    }

    [Fact]
    public void ObscureFrame()
    {
        using var frame = new Mat(new Size(64, 64), MatType.CV_8UC3, new Scalar(10, 20, 30));
        using var blurred = FaceFusion.Vision.Vision.ObscureFrame(frame);

        Assert.Equal(frame.Size(), blurred.Size());
        Assert.Equal(frame.Type(), blurred.Type());
    }

    [Fact]
    public void BlendFrame()
    {
        using var source = new Mat(new Size(4, 4), MatType.CV_8UC3, new Scalar(0, 0, 0));
        using var target = new Mat(new Size(4, 4), MatType.CV_8UC3, new Scalar(100, 100, 100));

        using var blended = FaceFusion.Vision.Vision.BlendFrame(source, target, 0.5);
        Assert.Equal(new Vec3b(50, 50, 50), blended.At<Vec3b>(0, 0));

        using var blendedAlias = FaceFusion.Vision.Vision.BlendVisionFrames(source, target, 0.5);
        Assert.Equal(new Vec3b(50, 50, 50), blendedAlias.At<Vec3b>(0, 0));
    }

    [Fact]
    public void CreateEmptyVisionFrame()
    {
        using var frame = FaceFusion.Vision.Vision.CreateEmptyVisionFrame();
        Assert.Equal(1, frame.Rows);
        Assert.Equal(1, frame.Cols);
        Assert.Equal(3, frame.Channels());
        Assert.Equal(new Vec3b(0, 0, 0), frame.At<Vec3b>(0, 0));
    }

    [Fact]
    public void IsVisionFrame()
    {
        using var frame = new Mat(new Size(4, 4), MatType.CV_8UC3, Scalar.All(0));
        using var mask = new Mat(new Size(4, 4), MatType.CV_8UC1, Scalar.All(0));

        Assert.True(FaceFusion.Vision.Vision.IsVisionFrame(frame));
        Assert.False(FaceFusion.Vision.Vision.IsVisionFrame(mask));
        Assert.False(FaceFusion.Vision.Vision.IsVisionFrame(null));
    }

    [Fact]
    public void CreateAndMergeTileFrames()
    {
        using var frame = new Mat(new Size(50, 50), MatType.CV_8UC3, new Scalar(5, 10, 15));
        var size = (TileSize: 16, PadSize: 2, OverlapSize: 1);

        var (tiles, padWidth, padHeight) = FaceFusion.Vision.Vision.CreateTileFrames(frame, size);
        Assert.NotEmpty(tiles);

        using var merged = FaceFusion.Vision.Vision.MergeTileFrames(tiles, frame.Cols, frame.Rows, padWidth, padHeight, size);
        Assert.Equal(frame.Rows, merged.Rows);
        Assert.Equal(frame.Cols, merged.Cols);
        Assert.Equal(new Vec3b(5, 10, 15), merged.At<Vec3b>(25, 25));

        foreach (var tile in tiles)
        {
            tile.Dispose();
        }
    }

    [Fact]
    public void ExtractAndMergeVisionMask()
    {
        using var opaqueFrame = new Mat(new Size(4, 4), MatType.CV_8UC3, Scalar.All(0));
        using var defaultMask = FaceFusion.Vision.Vision.ExtractVisionMask(opaqueFrame);
        Assert.Equal(255, defaultMask.At<byte>(0, 0));

        using var rgbaFrame = new Mat(new Size(4, 4), MatType.CV_8UC4, new Scalar(1, 2, 3, 42));
        using var alphaMask = FaceFusion.Vision.Vision.ExtractVisionMask(rgbaFrame);
        Assert.Equal(42, alphaMask.At<byte>(0, 0));

        using var bgrFrame = new Mat(new Size(4, 4), MatType.CV_8UC3, new Scalar(1, 2, 3));
        using var customMask = new Mat(new Size(4, 4), MatType.CV_8UC1, Scalar.All(200));
        using var merged = FaceFusion.Vision.Vision.MergeVisionMask(bgrFrame, customMask);
        Assert.Equal(4, merged.Channels());
        Assert.Equal(200, merged.At<Vec4b>(0, 0).Item3);
    }

    [Fact]
    public void ConditionalMergeVisionMask()
    {
        using var frame = new Mat(new Size(4, 4), MatType.CV_8UC3, Scalar.All(0));
        using var fullMask = new Mat(new Size(4, 4), MatType.CV_8UC1, Scalar.All(255));
        using var partialMask = new Mat(new Size(4, 4), MatType.CV_8UC1, Scalar.All(200));

        using var unchanged = FaceFusion.Vision.Vision.ConditionalMergeVisionMask(frame, fullMask);
        Assert.Equal(3, unchanged.Channels());

        using var merged = FaceFusion.Vision.Vision.ConditionalMergeVisionMask(frame, partialMask);
        Assert.Equal(4, merged.Channels());
    }

    private static void AssertMatEqual(Mat left, Mat right)
    {
        Assert.Equal(left.Size(), right.Size());
        using var diff = new Mat();
        Cv2.Absdiff(left, right, diff);
        Assert.Equal(0, Cv2.CountNonZero(diff.Reshape(1)));
    }

    private static Mat CreateHueGradientFrame(int width, int height)
    {
        using var hsv = new Mat(new Size(width, height), MatType.CV_8UC3);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var hue = (byte)(x * 180 / width);
                hsv.Set(y, x, new Vec3b(hue, 255, 200));
            }
        }

        var bgr = new Mat();
        Cv2.CvtColor(hsv, bgr, ColorConversionCodes.HSV2BGR);
        return bgr;
    }

    private static Mat CreateDesaturatedFrame(Mat source)
    {
        using var gray = new Mat();
        Cv2.CvtColor(source, gray, ColorConversionCodes.BGR2GRAY);
        var desaturated = new Mat();
        Cv2.CvtColor(gray, desaturated, ColorConversionCodes.GRAY2BGR);
        return desaturated;
    }
}
