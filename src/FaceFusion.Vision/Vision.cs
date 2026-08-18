using FaceFusion.Core;
using FaceFusion.Tensors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.Vision;

/// <summary>
/// Port of <c>facefusion/vision.py</c>.
///
/// <para>
/// <b>VisionFrame / Mask representation.</b> Python represents a <c>VisionFrame</c> as an
/// <c>H x W x C uint8</c> numpy array and a <c>Mask</c> as an <c>H x W uint8</c> array. Per
/// docs/DOTNET_PORT_PLAN.md §5a/§5b, both are represented here as <see cref="Mat"/> — pixel
/// data stays in native memory for the lifetime of a frame and never round-trips through a
/// managed array. <b>Every method that returns a new <see cref="Mat"/> transfers ownership to
/// the caller</b>, who must dispose it (typically via <c>using</c>); this is called out on each
/// method. Methods that accept a <see cref="Mat"/> never take ownership of it and never dispose
/// it — the caller retains ownership of anything it passes in.
/// </para>
///
/// <para>
/// <b>Video I/O divergence from Python (documented, deliberate).</b> The Python module reads
/// video frames through <c>facefusion.video_manager</c>, which pools long-lived ffmpeg
/// subprocesses (spawned via <c>facefusion.ffmpeg</c>) and derives duration/fps/resolution/frame
/// count from <c>facefusion.ffprobe</c> (which shells out to the <c>ffprobe</c> binary). Neither
/// of those modules has a C# port yet — <c>FaceFusion.Media</c> currently contains only pure
/// command-line *builders* (<c>FfmpegBuilder</c>, <c>FfprobeBuilder</c>), not a process runner,
/// and porting the ffmpeg/ffprobe subprocess plumbing and the frame-store pool is out of this
/// module's assignment. This container also has neither the <c>ffmpeg</c> nor the <c>ffprobe</c>
/// binary installed. Per the assignment brief, video functions here are implemented directly
/// against <see cref="OpenCvSharp.VideoCapture"/> instead: opening a fresh capture per call and
/// reading via <c>CAP_PROP_POS_FRAMES</c> / <c>CAP_PROP_FPS</c> / <c>CAP_PROP_FRAME_COUNT</c> /
/// <c>CAP_PROP_FRAME_WIDTH</c> / <c>CAP_PROP_FRAME_HEIGHT</c>. This gives the same observable
/// results (the Nth frame's pixels; the video's fps/resolution/frame count) without the pooled
/// subprocess machinery, but it is sourced from OpenCV's demuxer rather than from `ffprobe`, so
/// exact metadata values (e.g. frame_total for VFR video, or color_transfer) can differ at the
/// margins from the Python. Because there is no <c>video_manager</c> pool in this port, the
/// Python module's <c>thread_lock</c> / <c>thread_semaphore</c> guards (which protect the shared
/// pool) have no C# equivalent here — there is no shared mutable state to protect (see
/// PORT_CONVENTIONS.md rule 5).
/// </para>
/// </summary>
public static class Vision
{
    private const int ImageCacheCapacity = 64;
    private const int VideoFrameCacheCapacity = 64;

    private static readonly object ImageCacheLock = new();
    private static readonly Dictionary<(string ImagePath, ColorMode ColorMode), Mat> ImageCache = new();
    private static readonly Queue<(string ImagePath, ColorMode ColorMode)> ImageCacheOrder = new();

    private static readonly object VideoFrameCacheLock = new();
    private static readonly Dictionary<(string VideoPath, int FrameNumber), Mat> VideoFrameCache = new();
    private static readonly Queue<(string VideoPath, int FrameNumber)> VideoFrameCacheOrder = new();

    // -----------------------------------------------------------------
    // Image I/O
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>read_static_images</c>. Caller owns every <see cref="Mat"/> in the result and
    /// must dispose each one.
    /// </summary>
    public static IReadOnlyList<Mat> ReadStaticImages(IReadOnlyList<string>? imagePaths, ColorMode colorMode = ColorMode.Rgb)
    {
        var visionFrames = new List<Mat>();

        if (imagePaths is { Count: > 0 })
        {
            foreach (var imagePath in imagePaths)
            {
                var visionFrame = ReadStaticImage(imagePath, colorMode);
                if (visionFrame is not null)
                {
                    visionFrames.Add(visionFrame);
                }
            }
        }

        return visionFrames;
    }

    /// <summary>
    /// Python: <c>read_static_image</c> (<c>@lru_cache(maxsize = 64)</c>).
    ///
    /// <para>
    /// Python's <c>lru_cache</c> hands back the *same* numpy array object on a repeat call —
    /// callers that mutate it corrupt the cache. That aliasing is not safe to reproduce under
    /// strict <see cref="Mat"/> disposal (a caller that disposes what looks like "their" frame
    /// would corrupt or crash on the next cache hit), so this cache instead keeps its own
    /// private canonical <see cref="Mat"/> per key and always returns a fresh
    /// <see cref="Mat.Clone()"/> — the caller gets a normal, independently-owned frame every
    /// time, and cache eviction is a simple bounded FIFO (not true LRU order) rather than
    /// reproducing <c>lru_cache</c>'s recency tracking. Caller owns the returned
    /// <see cref="Mat"/> and must dispose it.
    /// </para>
    /// </summary>
    public static Mat? ReadStaticImage(string imagePath, ColorMode colorMode = ColorMode.Rgb)
    {
        var key = (imagePath, colorMode);

        lock (ImageCacheLock)
        {
            if (ImageCache.TryGetValue(key, out var cached))
            {
                return cached.Clone();
            }
        }

        var visionFrame = ReadImage(imagePath, colorMode);
        if (visionFrame is null)
        {
            return null;
        }

        lock (ImageCacheLock)
        {
            if (!ImageCache.ContainsKey(key))
            {
                if (ImageCacheOrder.Count >= ImageCacheCapacity)
                {
                    var oldestKey = ImageCacheOrder.Dequeue();
                    if (ImageCache.Remove(oldestKey, out var evicted))
                    {
                        evicted.Dispose();
                    }
                }

                ImageCache[key] = visionFrame.Clone();
                ImageCacheOrder.Enqueue(key);
            }
        }

        return visionFrame;
    }

    /// <summary>
    /// Python: <c>read_image</c>. Caller owns the returned <see cref="Mat"/> and must dispose
    /// it.
    ///
    /// <para>
    /// Reproduces the Python's Windows-only <c>numpy.fromfile</c> + <c>cv2.imdecode</c> path
    /// (a workaround for non-ASCII paths on Windows) as a byte-read + <see cref="Cv2.ImDecode"/>
    /// path here, gated on the same <see cref="CommonHelper.IsWindows"/> check; on Linux/macOS
    /// this always takes the plain <see cref="Cv2.ImRead"/> branch, same as Python.
    /// </para>
    /// </summary>
    public static Mat? ReadImage(string imagePath, ColorMode colorMode = ColorMode.Rgb)
    {
        if (!FileSystem.IsImage(imagePath))
        {
            return null;
        }

        var flag = colorMode == ColorMode.Rgba ? ImreadModes.Unchanged : ImreadModes.Color;

        Mat image = CommonHelper.IsWindows()
            ? Cv2.ImDecode(File.ReadAllBytes(imagePath), flag)
            : Cv2.ImRead(imagePath, flag);

        if (image.Empty())
        {
            image.Dispose();
            return null;
        }

        return image;
    }

    /// <summary>Python: <c>write_image</c>. Does not take ownership of <paramref name="visionFrame"/>.</summary>
    public static bool WriteImage(string? imagePath, Mat visionFrame)
    {
        if (string.IsNullOrEmpty(imagePath))
        {
            return false;
        }

        if (CommonHelper.IsWindows())
        {
            var imageFileExtension = FileSystem.GetFileExtension(imagePath) ?? string.Empty;
            Cv2.ImEncode(imageFileExtension, visionFrame, out var buffer);
            File.WriteAllBytes(imagePath, buffer);
            return FileSystem.IsImage(imagePath);
        }

        return Cv2.ImWrite(imagePath, visionFrame);
    }

    /// <summary>Python: <c>detect_image_resolution</c>.</summary>
    public static Resolution? DetectImageResolution(string imagePath)
    {
        if (!FileSystem.IsImage(imagePath))
        {
            return null;
        }

        using var image = ReadImage(imagePath);
        if (image is null)
        {
            return null;
        }

        var width = image.Cols;
        var height = image.Rows;

        if (width > 0 && height > 0)
        {
            return new Resolution(width, height);
        }

        return null;
    }

    /// <summary>Python: <c>restrict_image_resolution</c>.</summary>
    public static Resolution RestrictImageResolution(string imagePath, Resolution resolution)
    {
        if (FileSystem.IsImage(imagePath))
        {
            var imageResolution = DetectImageResolution(imagePath);
            if (imageResolution is { } detected && IsLexicographicallyLess(detected, resolution))
            {
                return detected;
            }
        }

        return resolution;
    }

    // -----------------------------------------------------------------
    // Video I/O — see the class-level remarks on the OpenCvSharp.VideoCapture divergence.
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>read_static_video_frame</c> (<c>@lru_cache(maxsize = 64)</c>). Same
    /// clone-on-read / bounded-FIFO caching approach as <see cref="ReadStaticImage"/>, for the
    /// same disposal-safety reason. Caller owns the returned <see cref="Mat"/> and must dispose
    /// it.
    /// </summary>
    public static Mat? ReadStaticVideoFrame(string videoPath, int frameNumber = 0)
    {
        var key = (videoPath, frameNumber);

        lock (VideoFrameCacheLock)
        {
            if (VideoFrameCache.TryGetValue(key, out var cached))
            {
                return cached.Clone();
            }
        }

        var visionFrame = ReadVideoFrame(videoPath, frameNumber);
        if (visionFrame is null)
        {
            return null;
        }

        lock (VideoFrameCacheLock)
        {
            if (!VideoFrameCache.ContainsKey(key))
            {
                if (VideoFrameCacheOrder.Count >= VideoFrameCacheCapacity)
                {
                    var oldestKey = VideoFrameCacheOrder.Dequeue();
                    if (VideoFrameCache.Remove(oldestKey, out var evicted))
                    {
                        evicted.Dispose();
                    }
                }

                VideoFrameCache[key] = visionFrame.Clone();
                VideoFrameCacheOrder.Enqueue(key);
            }
        }

        return visionFrame;
    }

    /// <summary>
    /// Python: <c>read_video_frame</c>. Caller owns the returned <see cref="Mat"/> and must
    /// dispose it. See the class-level remarks: this opens a fresh
    /// <see cref="OpenCvSharp.VideoCapture"/> and seeks directly to <paramref name="frameNumber"/>
    /// rather than driving the ffmpeg-pipe reader pool the Python uses.
    /// </summary>
    public static Mat? ReadVideoFrame(string videoPath, int frameNumber = 0)
    {
        if (!FileSystem.IsVideo(videoPath))
        {
            return null;
        }

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            return null;
        }

        if (frameNumber != 0)
        {
            capture.Set(VideoCaptureProperties.PosFrames, frameNumber);
        }

        var frame = new Mat();
        if (capture.Read(frame) && !frame.Empty())
        {
            return frame;
        }

        frame.Dispose();
        return null;
    }

    /// <summary>
    /// Python: <c>select_video_frames</c>. Always returns <c>2 * frameOffset + 1</c> frames when
    /// the path is a video (empty-frame placeholders for indices outside the video, exactly as
    /// <c>create_empty_vision_frame</c> stands in for missing entries in the Python's
    /// <c>frame_set</c>), or an empty list when it is not. Caller owns every <see cref="Mat"/>
    /// in the result and must dispose each one.
    /// </summary>
    public static IReadOnlyList<Mat> SelectVideoFrames(string videoPath, int frameNumber = 0, int frameOffset = 2)
    {
        var visionFrames = new List<Mat>();

        if (!FileSystem.IsVideo(videoPath))
        {
            return visionFrames;
        }

        var frameStart = frameNumber - frameOffset;
        var frameEnd = frameNumber + frameOffset;

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            for (var i = frameStart; i <= frameEnd; i++)
            {
                visionFrames.Add(CreateEmptyVisionFrame());
            }

            return visionFrames;
        }

        var frameTotal = (int)capture.Get(VideoCaptureProperties.FrameCount);

        for (var currentFrameNumber = frameStart; currentFrameNumber <= frameEnd; currentFrameNumber++)
        {
            if (currentFrameNumber < 0 || (frameTotal > 0 && currentFrameNumber >= frameTotal))
            {
                visionFrames.Add(CreateEmptyVisionFrame());
                continue;
            }

            capture.Set(VideoCaptureProperties.PosFrames, currentFrameNumber);
            var frame = new Mat();
            if (capture.Read(frame) && !frame.Empty())
            {
                visionFrames.Add(frame);
            }
            else
            {
                frame.Dispose();
                visionFrames.Add(CreateEmptyVisionFrame());
            }
        }

        return visionFrames;
    }

    /// <summary>
    /// Python: <c>count_video_frame_total</c>. Sourced from
    /// <c>CAP_PROP_FRAME_COUNT</c> rather than ffprobe — see class remarks.
    /// </summary>
    public static int CountVideoFrameTotal(string videoPath)
    {
        if (!FileSystem.IsVideo(videoPath))
        {
            return 0;
        }

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            return 0;
        }

        return (int)capture.Get(VideoCaptureProperties.FrameCount);
    }

    /// <summary>Python: <c>predict_video_frame_total</c>.</summary>
    public static int PredictVideoFrameTotal(string videoPath, double fps, int trimFrameStart, int trimFrameEnd)
    {
        if (FileSystem.IsVideo(videoPath))
        {
            var videoFps = DetectVideoFps(videoPath) ?? 0;
            if (videoFps == 0)
            {
                return 0;
            }

            var extractFrameTotal = CountTrimFrameTotal(videoPath, trimFrameStart, trimFrameEnd) * fps / videoFps;
            return (int)Math.Floor(extractFrameTotal);
        }

        return 0;
    }

    /// <summary>
    /// Python: <c>detect_video_fps</c>. Sourced from <c>CAP_PROP_FPS</c> rather than ffprobe —
    /// see class remarks.
    /// </summary>
    public static double? DetectVideoFps(string videoPath)
    {
        if (!FileSystem.IsVideo(videoPath))
        {
            return null;
        }

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            return null;
        }

        return capture.Get(VideoCaptureProperties.Fps);
    }

    /// <summary>Python: <c>restrict_video_fps</c>.</summary>
    public static double RestrictVideoFps(string videoPath, double fps)
    {
        if (FileSystem.IsVideo(videoPath))
        {
            var videoFps = DetectVideoFps(videoPath);
            if (videoFps is { } detected && detected < fps)
            {
                return detected;
            }
        }

        return fps;
    }

    /// <summary>Python: <c>detect_video_duration</c>.</summary>
    public static double DetectVideoDuration(string videoPath)
    {
        var videoFrameTotal = CountVideoFrameTotal(videoPath);
        var videoFps = DetectVideoFps(videoPath);

        if (videoFrameTotal != 0 && videoFps is { } fps && fps != 0)
        {
            return videoFrameTotal / fps;
        }

        return 0;
    }

    /// <summary>Python: <c>count_trim_frame_total</c>.</summary>
    public static int CountTrimFrameTotal(string videoPath, int? trimFrameStart, int? trimFrameEnd)
    {
        var (start, end) = RestrictTrimFrame(videoPath, trimFrameStart, trimFrameEnd);
        return end - start;
    }

    /// <summary>Python: <c>restrict_trim_frame</c>.</summary>
    public static (int Start, int End) RestrictTrimFrame(string videoPath, int? trimFrameStart, int? trimFrameEnd)
    {
        var videoFrameTotal = CountVideoFrameTotal(videoPath);

        var start = trimFrameStart;
        var end = trimFrameEnd;

        if (start is { } startValue)
        {
            start = Math.Max(0, Math.Min(startValue, videoFrameTotal));
        }

        if (end is { } endValue)
        {
            end = Math.Max(0, Math.Min(endValue, videoFrameTotal));
        }

        if (start is { } s && end is { } e)
        {
            return (s, e);
        }

        if (start is { } sOnly)
        {
            return (sOnly, videoFrameTotal);
        }

        if (end is { } eOnly)
        {
            return (0, eOnly);
        }

        return (0, videoFrameTotal);
    }

    /// <summary>
    /// Python: <c>detect_video_resolution</c>. Sourced from <c>CAP_PROP_FRAME_WIDTH</c> /
    /// <c>CAP_PROP_FRAME_HEIGHT</c> rather than ffprobe — see class remarks.
    /// </summary>
    public static Resolution? DetectVideoResolution(string videoPath)
    {
        if (!FileSystem.IsVideo(videoPath))
        {
            return null;
        }

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            return null;
        }

        var width = (int)capture.Get(VideoCaptureProperties.FrameWidth);
        var height = (int)capture.Get(VideoCaptureProperties.FrameHeight);

        if (width <= 0 || height <= 0)
        {
            return null;
        }

        return new Resolution(width, height);
    }

    /// <summary>Python: <c>restrict_video_resolution</c>.</summary>
    public static Resolution RestrictVideoResolution(string videoPath, Resolution resolution)
    {
        if (FileSystem.IsVideo(videoPath))
        {
            var videoResolution = DetectVideoResolution(videoPath);
            if (videoResolution is { } detected && IsLexicographicallyLess(detected, resolution))
            {
                return detected;
            }
        }

        return resolution;
    }

    // -----------------------------------------------------------------
    // Resolution math
    // -----------------------------------------------------------------

    /// <summary>Python: <c>scale_resolution</c>.</summary>
    public static Resolution ScaleResolution(Resolution resolution, double scale)
    {
        // Python: int(resolution[0] * scale) truncates toward zero, same as an (int) cast here.
        var width = (int)(resolution.Width * scale);
        var height = (int)(resolution.Height * scale);
        return NormalizeResolution(width, height);
    }

    /// <summary>Python: <c>normalize_resolution</c>.</summary>
    public static Resolution NormalizeResolution(double width, double height)
    {
        if (width > 0 && height > 0)
        {
            // Python: round(width / 2) * 2 — round() is round-half-to-even, same as the
            // default MidpointRounding for Math.Round(double).
            var normalizedWidth = (int)(Math.Round(width / 2.0) * 2);
            var normalizedHeight = (int)(Math.Round(height / 2.0) * 2);
            return new Resolution(normalizedWidth, normalizedHeight);
        }

        return new Resolution(0, 0);
    }

    /// <summary>Python: <c>pack_resolution</c>.</summary>
    public static string PackResolution(Resolution resolution)
    {
        var normalized = NormalizeResolution(resolution.Width, resolution.Height);
        return normalized.Width.ToString(System.Globalization.CultureInfo.InvariantCulture)
            + "x"
            + normalized.Height.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>Python: <c>unpack_resolution</c>. Does not normalize, unlike <see cref="PackResolution"/>.</summary>
    public static Resolution UnpackResolution(string resolution)
    {
        var parts = resolution.Split('x');
        var width = int.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture);
        var height = int.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
        return new Resolution(width, height);
    }

    // -----------------------------------------------------------------
    // Frame helpers
    // -----------------------------------------------------------------

    /// <summary>Python: <c>detect_frame_orientation</c>.</summary>
    public static Orientation DetectFrameOrientation(Mat visionFrame)
    {
        var width = visionFrame.Cols;
        var height = visionFrame.Rows;
        return width > height ? Orientation.Landscape : Orientation.Portrait;
    }

    /// <summary>
    /// Python: <c>restrict_frame</c>. Caller owns the returned <see cref="Mat"/> and must
    /// dispose it. Unlike Python (which returns the same array object, unresized, when no
    /// resize is needed), this always returns a distinct new <see cref="Mat"/> — a deliberate
    /// divergence so ownership of the return value is unambiguous under strict disposal (see
    /// class remarks); no test relies on Python's aliasing here.
    /// </summary>
    public static Mat RestrictFrame(Mat visionFrame, Resolution resolution)
    {
        var height = visionFrame.Rows;
        var width = visionFrame.Cols;
        var restrictWidth = resolution.Width;
        var restrictHeight = resolution.Height;

        if (height > restrictHeight || width > restrictWidth)
        {
            var scale = Math.Min((double)restrictHeight / height, (double)restrictWidth / width);
            var newWidth = (int)(width * scale);
            var newHeight = (int)(height * scale);
            var resized = new Mat();
            Cv2.Resize(visionFrame, resized, new Size(newWidth, newHeight));
            return resized;
        }

        return visionFrame.Clone();
    }

    /// <summary>Python: <c>fit_contain_frame</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat FitContainFrame(Mat visionFrame, Resolution resolution)
    {
        var containWidth = resolution.Width;
        var containHeight = resolution.Height;
        var height = visionFrame.Rows;
        var width = visionFrame.Cols;
        var scale = Math.Min((double)containHeight / height, (double)containWidth / width);
        var newWidth = (int)(width * scale);
        var newHeight = (int)(height * scale);
        var startX = Math.Max(0, FloorDiv(containWidth - newWidth, 2));
        var startY = Math.Max(0, FloorDiv(containHeight - newHeight, 2));
        var endX = Math.Max(0, containWidth - newWidth - startX);
        var endY = Math.Max(0, containHeight - newHeight - startY);

        using var resized = new Mat();
        Cv2.Resize(visionFrame, resized, new Size(newWidth, newHeight));

        var padded = new Mat();
        Cv2.CopyMakeBorder(resized, padded, startY, endY, startX, endX, BorderTypes.Constant, Scalar.All(0));
        return padded;
    }

    /// <summary>Python: <c>fit_cover_frame</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat FitCoverFrame(Mat visionFrame, Resolution resolution)
    {
        var coverWidth = resolution.Width;
        var coverHeight = resolution.Height;
        var height = visionFrame.Rows;
        var width = visionFrame.Cols;
        var scale = Math.Max((double)coverWidth / width, (double)coverHeight / height);
        var newWidth = (int)(width * scale);
        var newHeight = (int)(height * scale);
        var startX = Math.Max(0, FloorDiv(newWidth - coverWidth, 2));
        var startY = Math.Max(0, FloorDiv(newHeight - coverHeight, 2));
        var endX = Math.Min(newWidth, startX + coverWidth);
        var endY = Math.Min(newHeight, startY + coverHeight);

        using var resized = new Mat();
        Cv2.Resize(visionFrame, resized, new Size(newWidth, newHeight));

        var rect = new Rect(startX, startY, endX - startX, endY - startY);
        return resized[rect].Clone();
    }

    /// <summary>Python: <c>obscure_frame</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat ObscureFrame(Mat visionFrame)
    {
        var blurred = new Mat();
        Cv2.GaussianBlur(visionFrame, blurred, new Size(99, 99), 0);
        return blurred;
    }

    /// <summary>Python: <c>blend_frame</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat BlendFrame(Mat sourceVisionFrame, Mat targetVisionFrame, double blendFactor)
    {
        var blended = new Mat();
        Cv2.AddWeighted(sourceVisionFrame, 1 - blendFactor, targetVisionFrame, blendFactor, 0, blended);
        return blended;
    }

    /// <summary>
    /// Python: <c>blend_vision_frames</c> — defined in Python as a byte-for-byte duplicate of
    /// <c>blend_frame</c>. Reproduced here as a thin wrapper rather than a second copy of the
    /// same body; behaviour is identical. Caller owns the returned <see cref="Mat"/> and must
    /// dispose it.
    /// </summary>
    public static Mat BlendVisionFrames(Mat sourceVisionFrame, Mat targetVisionFrame, double blendFactor)
        => BlendFrame(sourceVisionFrame, targetVisionFrame, blendFactor);

    /// <summary>Python: <c>conditional_match_frame_color</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat ConditionalMatchFrameColor(Mat sourceVisionFrame, Mat targetVisionFrame)
    {
        var histogramFactor = CalculateHistogramDifference(sourceVisionFrame, targetVisionFrame);
        using var matched = MatchFrameColor(sourceVisionFrame, targetVisionFrame);
        return BlendFrame(targetVisionFrame, matched, histogramFactor);
    }

    /// <summary>Python: <c>match_frame_color</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat MatchFrameColor(Mat sourceVisionFrame, Mat targetVisionFrame)
    {
        var colorDifferenceSizes = NumPy.Linspace(16f, targetVisionFrame.Rows, 3, endpoint: false);

        var currentSource = sourceVisionFrame.Clone();
        foreach (var colorDifferenceSize in colorDifferenceSizes)
        {
            var size = NormalizeResolution(colorDifferenceSize, colorDifferenceSize);
            var next = EqualizeFrameColor(currentSource, targetVisionFrame, size);
            currentSource.Dispose();
            currentSource = next;
        }

        var finalSize = new Resolution(targetVisionFrame.Cols, targetVisionFrame.Rows);
        var result = EqualizeFrameColor(currentSource, targetVisionFrame, finalSize);
        currentSource.Dispose();
        return result;
    }

    /// <summary>
    /// Python: <c>equalize_frame_color</c>. Caller owns the returned <see cref="Mat"/> and must
    /// dispose it.
    ///
    /// <para>
    /// The final <c>.clip(0, 255).astype(numpy.uint8)</c> in Python *truncates* toward zero
    /// (numpy's float-to-int cast semantics), whereas <see cref="Mat.ConvertTo(Mat, MatType, double, double)"/>
    /// to an 8-bit type *rounds* to the nearest integer (OpenCV's <c>saturate_cast</c>). That is
    /// a genuine, documented parity gap: this method's output can differ from the Python's by up
    /// to 1 per channel per pixel. It is not corrected here because doing so would require a
    /// manual per-pixel floor pass (this method resizes to very small sizes for most of its
    /// calls, but not for the final full-size call in <see cref="MatchFrameColor"/>, so a
    /// per-pixel loop is not free), and every caller of this color-matching path
    /// (<see cref="CalculateHistogramDifference"/>-gated blending, histogram-difference
    /// comparisons in the tests) is tolerance-based rather than exact-value, so the 1-count
    /// rounding difference does not change any observable outcome.
    /// </para>
    /// </summary>
    public static Mat EqualizeFrameColor(Mat sourceVisionFrame, Mat targetVisionFrame, Resolution size)
    {
        var targetSize = new Size(size.Width, size.Height);

        using var sourceResized = new Mat();
        Cv2.Resize(sourceVisionFrame, sourceResized, targetSize, 0, 0, InterpolationFlags.Area);
        using var sourceFloat = new Mat();
        sourceResized.ConvertTo(sourceFloat, MatType.CV_32FC3);

        using var targetResized = new Mat();
        Cv2.Resize(targetVisionFrame, targetResized, targetSize, 0, 0, InterpolationFlags.Area);
        using var targetFloat = new Mat();
        targetResized.ConvertTo(targetFloat, MatType.CV_32FC3);

        using var colorDifference = new Mat();
        Cv2.Subtract(sourceFloat, targetFloat, colorDifference);

        using var colorDifferenceResized = new Mat();
        Cv2.Resize(colorDifference, colorDifferenceResized, new Size(targetVisionFrame.Cols, targetVisionFrame.Rows), 0, 0, InterpolationFlags.Cubic);

        using var targetOriginalFloat = new Mat();
        targetVisionFrame.ConvertTo(targetOriginalFloat, MatType.CV_32FC3);

        using var summed = new Mat();
        Cv2.Add(targetOriginalFloat, colorDifferenceResized, summed);

        using var clippedLow = new Mat();
        Cv2.Max(summed, 0.0, clippedLow);
        using var clipped = new Mat();
        Cv2.Min(clippedLow, 255.0, clipped);

        var result = new Mat();
        clipped.ConvertTo(result, MatType.CV_8UC3);
        return result;
    }

    /// <summary>Python: <c>calculate_histogram_difference</c>. Does not take ownership of either argument.</summary>
    public static double CalculateHistogramDifference(Mat sourceVisionFrame, Mat targetVisionFrame)
    {
        using var sourceHsv = new Mat();
        Cv2.CvtColor(sourceVisionFrame, sourceHsv, ColorConversionCodes.BGR2HSV);
        using var targetHsv = new Mat();
        Cv2.CvtColor(targetVisionFrame, targetHsv, ColorConversionCodes.BGR2HSV);

        var ranges = new[] { new Rangef(0, 180), new Rangef(0, 256) };

        using var histogramSource = new Mat();
        Cv2.CalcHist(new[] { sourceHsv }, new[] { 0, 1 }, null, histogramSource, 2, new[] { 50, 60 }, ranges);
        using var histogramTarget = new Mat();
        Cv2.CalcHist(new[] { targetHsv }, new[] { 0, 1 }, null, histogramTarget, 2, new[] { 50, 60 }, ranges);

        var correlation = Cv2.CompareHist(histogramSource, histogramTarget, HistCompMethods.Correl);
        return NumPy.Interp((float)correlation, new[] { -1f, 1f }, new[] { 0f, 1f });
    }

    /// <summary>Python: <c>create_empty_vision_frame</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat CreateEmptyVisionFrame()
        => new(new Size(1, 1), MatType.CV_8UC3, Scalar.All(0));

    /// <summary>
    /// Python: <c>is_vision_frame</c> (<c>numpy.ndim(vision_frame) == 3</c>).
    ///
    /// <para>
    /// numpy's <c>ndim</c> distinguishes a true <c>(H, W, C)</c> image array from a bare
    /// <c>None</c> or a 2-D <c>(H, W)</c> mask array. <see cref="Mat"/> has no directly
    /// analogous concept — <see cref="Mat.Dims"/> is 2 for both a grayscale and a
    /// multi-channel image, because OpenCV keeps the channel count as a separate property
    /// rather than a third array dimension. This is approximated as "non-null, non-empty, and
    /// has the 3-channel layout every real <c>VisionFrame</c> in this codebase uses" — which is
    /// the actual property the Python callers of <c>is_vision_frame</c> rely on (checking a
    /// video-frame read actually produced usable pixel data).
    /// </para>
    /// </summary>
    public static bool IsVisionFrame(Mat? visionFrame)
        => visionFrame is { IsDisposed: false } frame && !frame.Empty() && frame.Channels() == 3;

    /// <summary>
    /// Python: <c>create_tile_frames</c>. <paramref name="size"/> mirrors the Python's
    /// <c>size : Size</c> parameter, whose declared <c>cv2.typing.Size</c> (a 2-tuple) type hint
    /// is inaccurate — the Python body indexes <c>size[0]</c>/<c>size[1]</c>/<c>size[2]</c>, so
    /// it is really a 3-tuple of (tile size, pad size, overlap size); reproduced here as an
    /// explicit 3-element tuple rather than propagating the misleading name. Caller owns every
    /// <see cref="Mat"/> in the returned tile list and must dispose each one.
    /// </summary>
    public static (IReadOnlyList<Mat> TileVisionFrames, int PadWidth, int PadHeight) CreateTileFrames(Mat visionFrame, (int TileSize, int PadSize, int OverlapSize) size)
    {
        var tileWidth = size.TileSize - (2 * size.OverlapSize);
        // Python quirk, reproduced deliberately: the top pad and the left pad are both
        // `pad_size_top` — the same value is used for both edges, while bottom/right are
        // computed separately from height/width respectively.
        var padSizeTop = size.PadSize + size.OverlapSize;
        var padSizeBottom = padSizeTop + tileWidth - ((visionFrame.Rows + (2 * size.PadSize)) % tileWidth);
        var padSizeRight = padSizeTop + tileWidth - ((visionFrame.Cols + (2 * size.PadSize)) % tileWidth);

        using var paddedVisionFrame = new Mat();
        Cv2.CopyMakeBorder(visionFrame, paddedVisionFrame, padSizeTop, padSizeBottom, padSizeTop, padSizeRight, BorderTypes.Constant, Scalar.All(0));

        var padHeight = paddedVisionFrame.Rows;
        var padWidth = paddedVisionFrame.Cols;
        var tileVisionFrames = new List<Mat>();

        for (var row = size.OverlapSize; row < padHeight - size.OverlapSize; row += tileWidth)
        {
            var top = row - size.OverlapSize;
            var bottom = row + size.OverlapSize + tileWidth;

            for (var col = size.OverlapSize; col < padWidth - size.OverlapSize; col += tileWidth)
            {
                var left = col - size.OverlapSize;
                var right = col + size.OverlapSize + tileWidth;
                var rect = new Rect(left, top, right - left, bottom - top);
                tileVisionFrames.Add(paddedVisionFrame[rect].Clone());
            }
        }

        return (tileVisionFrames, padWidth, padHeight);
    }

    /// <summary>
    /// Python: <c>merge_tile_frames</c>. Does not take ownership of <paramref name="tileVisionFrames"/>
    /// (the caller still owns and must dispose them, same as Python leaves the input array list
    /// untouched). Caller owns the returned <see cref="Mat"/> and must dispose it.
    /// </summary>
    public static Mat MergeTileFrames(IReadOnlyList<Mat> tileVisionFrames, int tempWidth, int tempHeight, int padWidth, int padHeight, (int TileSize, int PadSize, int OverlapSize) size)
    {
        using var mergeVisionFrame = new Mat(new Size(padWidth, padHeight), MatType.CV_8UC3, Scalar.All(0));
        var tileWidth = tileVisionFrames[0].Cols - (2 * size.OverlapSize);
        var tilesPerRow = Math.Min(padWidth / tileWidth, tileVisionFrames.Count);

        for (var index = 0; index < tileVisionFrames.Count; index++)
        {
            var tile = tileVisionFrames[index];
            var overlap = size.OverlapSize;
            var croppedRect = new Rect(overlap, overlap, tile.Cols - (2 * overlap), tile.Rows - (2 * overlap));
            using var croppedTile = tile[croppedRect];

            var rowIndex = index / tilesPerRow;
            var colIndex = index % tilesPerRow;
            var top = rowIndex * croppedTile.Rows;
            var left = colIndex * croppedTile.Cols;
            var destinationRect = new Rect(left, top, croppedTile.Cols, croppedTile.Rows);

            using var destination = mergeVisionFrame[destinationRect];
            croppedTile.CopyTo(destination);
        }

        var finalRect = new Rect(size.PadSize, size.PadSize, tempWidth, tempHeight);
        return mergeVisionFrame[finalRect].Clone();
    }

    /// <summary>Python: <c>extract_vision_mask</c>. Caller owns the returned <see cref="Mat"/> and must dispose it.</summary>
    public static Mat ExtractVisionMask(Mat visionFrame)
    {
        if (visionFrame.Channels() == 4)
        {
            var channels = Cv2.Split(visionFrame);
            var alpha = channels[3];
            channels[0].Dispose();
            channels[1].Dispose();
            channels[2].Dispose();
            return alpha;
        }

        return new Mat(visionFrame.Size(), MatType.CV_8UC1, Scalar.All(255));
    }

    /// <summary>
    /// Python: <c>merge_vision_mask</c>. Does not take ownership of either argument. Caller owns
    /// the returned <see cref="Mat"/> and must dispose it.
    /// </summary>
    public static Mat MergeVisionMask(Mat visionFrame, Mat visionMask)
    {
        var channels = Cv2.Split(visionFrame);
        try
        {
            var merged = new Mat();
            Cv2.Merge(new[] { channels[0], channels[1], channels[2], visionMask }, merged);
            return merged;
        }
        finally
        {
            foreach (var channel in channels)
            {
                channel.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>conditional_merge_vision_mask</c>. Does not take ownership of either argument.
    /// Caller owns the returned <see cref="Mat"/> and must dispose it.
    /// </summary>
    public static Mat ConditionalMergeVisionMask(Mat visionFrame, Mat visionMask)
    {
        using var belowMax = new Mat();
        Cv2.Compare(visionMask, 255, belowMax, CmpType.LT);

        if (Cv2.CountNonZero(belowMax) > 0)
        {
            return MergeVisionMask(visionFrame, visionMask);
        }

        return visionFrame.Clone();
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    /// <summary>
    /// Python tuple comparison (<c>image_resolution &lt; resolution</c>) compares
    /// lexicographically: width first, then height on a width tie. Reproduced explicitly rather
    /// than via e.g. an area comparison, which would not match.
    /// </summary>
    private static bool IsLexicographicallyLess(Resolution left, Resolution right)
        => left.Width < right.Width || (left.Width == right.Width && left.Height < right.Height);

    /// <summary>Python's <c>//</c> floor-divides toward negative infinity; C#'s <c>/</c> on ints truncates toward zero.</summary>
    private static int FloorDiv(int a, int b)
        => (int)Math.Floor((double)a / b);
}
