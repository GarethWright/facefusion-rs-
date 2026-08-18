using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Media;
using FaceFusion.Face;
using FaceFusion.Processors;
using FaceFusion.Types;
using FaceFusion.Vision;
using FaceFusion.Workflows;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Coverage for the Phase 8 streaming path (<see cref="Streamer"/>, <see cref="CameraManager"/>).
///
/// <para>
/// <b>How this is testable without a webcam.</b> OpenCV's <c>VideoCapture</c> takes any string
/// source, and Python's own <c>get_remote_camera_capture</c> exists precisely to pass one — a
/// file path is a valid source. Pointing the camera pool at an example video exercises every
/// part of the live path except the device open itself: capture loop, per-frame processor
/// chain, the bounded worker queue, and teardown.
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class StreamerTests
{
    private static string? FindRepoRoot()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string ModelsDirectory()
        => Path.Combine(FindRepoRoot() ?? ".", ".assets", "models");

    private static bool ModelAvailable(string fileName)
    {
        var path = Path.Combine(ModelsDirectory(), fileName);
        return File.Exists(path) && new FileInfo(path).Length > 0;
    }

    private sealed class StreamTestAttribute : FactAttribute
    {
        public StreamTestAttribute(params string[] requiredModels)
        {
            if (!TestHelper.ExamplesAvailable)
            {
                Skip = TestHelper.MissingMediaMessage;
            }
            else if (requiredModels.Any(model => !ModelAvailable(model)))
            {
                Skip = $"requires {string.Join(", ", requiredModels.Select(m => $".assets/models/{m}"))} (gitignored, not present in CI)";
            }
        }
    }

    [Fact]
    public void CameraManagerReturnsNullForADeviceThatWillNotOpen()
    {
        using var cameraManager = new CameraManager();

        // Python returns None here too: get_local_camera_capture only caches a capture that
        // reports isOpened(), so the trailing dict lookup misses.
        Assert.Null(cameraManager.GetRemoteCameraCapture("/does/not/exist.mp4"));
    }

    [StreamTest]
    public void CameraManagerPoolsAndReleasesACaptureBySource()
    {
        var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
        using var cameraManager = new CameraManager();

        var first = cameraManager.GetRemoteCameraCapture(targetPath);
        var second = cameraManager.GetRemoteCameraCapture(targetPath);

        Assert.NotNull(first);
        // Python's pool is keyed by the source, so the second call returns the same object.
        Assert.Same(first, second);

        cameraManager.ClearCameraPool();

        // A released capture is gone from the pool, so the next call opens a new one.
        var third = cameraManager.GetRemoteCameraCapture(targetPath);
        Assert.NotNull(third);
        Assert.NotSame(first, third);
    }

    [StreamTest("ddcolor.onnx", "nsfw_1.onnx", "nsfw_2.onnx", "nsfw_3.onnx")]
    public void MultiProcessCaptureProcessesEveryFrameInCaptureOrder()
    {
        var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
        var args = new Dictionary<string, object?> { ["processors"] = new[] { "frame_colorizer" } };

        using var cameraManager = new CameraManager();
        var capture = cameraManager.GetRemoteCameraCapture(targetPath);
        Assert.NotNull(capture);

        var built = ProcessorStepFactory.Build("frame_colorizer", args);

        try
        {
            var frames = new List<Mat>();
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(5));

            foreach (var frame in Streamer.MultiProcessCapture(
                capture!,
                cameraFps: 25.0,
                new[] { built.Step },
                Array.Empty<string>(),
                executionThreadCount: 2,
                new ContentAnalyser(),
                ModelsDirectory(),
                new[] { 0 },
                new[] { ExecutionProvider.Cpu },
                cancellation.Token))
            {
                frames.Add(frame);

                // Enough to prove the loop, the workers and the ordering without colorizing a
                // whole clip on CPU.
                if (frames.Count == 6)
                {
                    break;
                }
            }

            try
            {
                Assert.Equal(6, frames.Count);

                foreach (var frame in frames)
                {
                    Assert.False(frame.Empty());
                    Assert.Equal(226, frame.Rows);
                    Assert.Equal(426, frame.Cols);
                }

                // The processor actually ran: a colorized frame is not the source frame.
                using var sourceCapture = new VideoCapture(targetPath);
                using var firstSourceFrame = new Mat();
                Assert.True(sourceCapture.Read(firstSourceFrame));

                using var difference = new Mat();
                Cv2.Absdiff(firstSourceFrame, frames[0], difference);
                Assert.True(Cv2.Sum(difference).Val0 > 0, "the streamed frame is identical to the captured frame — the processor did not run");
            }
            finally
            {
                foreach (var frame in frames)
                {
                    frame.Dispose();
                }
            }
        }
        finally
        {
            built.Resource.Dispose();
        }
    }

    /// <summary>
    /// Python's <c>process_stream_frame</c> skips any processor whose <c>pre_process('stream')</c>
    /// says it is not supported live, rather than failing. Verified here through
    /// <c>expression_restorer</c>, whose Python <c>pre_process</c> returns False for stream mode.
    /// </summary>
    [StreamTest("live_portrait_generator.onnx", "live_portrait_feature_extractor.onnx", "live_portrait_motion_extractor.onnx",
        "yoloface_8n.onnx", "fan_68_5.onnx", "2dfan4.onnx", "arcface_w600k_r50.onnx", "fairface.onnx")]
    public void ProcessStreamFrameSkipsAProcessorThatDoesNotSupportStreaming()
    {
        var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
        var args = new Dictionary<string, object?> { ["processors"] = new[] { "expression_restorer" } };

        using var faceResources = FacePipelineFactory.Build(args);
        var built = ProcessorStepFactory.Build("expression_restorer", args, faceResources);

        try
        {
            using var capture = new VideoCapture(targetPath);
            using var captured = new Mat();
            Assert.True(capture.Read(captured));

            using var result = Streamer.ProcessStreamFrame(
                new[] { built.Step },
                Array.Empty<Mat>(),
                captured,
                new ProcessorRunPaths(Array.Empty<string>(), targetPath, null));

            // Skipped, so the frame comes back unchanged — but as a distinct Mat the caller owns.
            Assert.NotSame(captured, result);

            using var difference = new Mat();
            Cv2.Absdiff(captured, result, difference);
            Assert.Equal(0.0, Cv2.Sum(difference).Val0);
        }
        finally
        {
            built.Resource.Dispose();
        }
    }

    /// <summary>
    /// Python: <c>open_stream('udp', ...)</c>. Checks the sink end to end rather than just the
    /// command builder: ffmpeg is started, raw frames are written to its stdin, and a second
    /// ffmpeg listening on the UDP port records what arrives. A command that is merely
    /// well-formed but wrong (bad pixel format, wrong resolution) fails here and would not fail
    /// a string comparison.
    /// </summary>
    [StreamTest]
    public void OpenStreamPushesFramesToAUdpSink()
    {
        if (!TestHelper.HasFfmpeg)
        {
            return;
        }

        const string resolution = "320x240";
        var recordedPath = Path.Combine(Path.GetTempPath(), $"facefusion-udp-{Guid.NewGuid():N}.mp4");

        // The listener has to be up before the sender starts, or the first datagrams are lost.
        var listener = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("ffmpeg")
        {
            ArgumentList = { "-y", "-hide_banner", "-loglevel", "error", "-i", "udp://localhost:27000?listen=1&timeout=15000000", "-t", "2", recordedPath },
            RedirectStandardError = true,
            UseShellExecute = false,
        });

        Assert.NotNull(listener);

        try
        {
            Thread.Sleep(1000);

            var stream = Streamer.OpenStream(StreamMode.Udp, resolution, 25.0);
            Assert.NotNull(stream);

            try
            {
                using var frame = new Mat(240, 320, MatType.CV_8UC3, new Scalar(30, 60, 90));
                var byteCount = (int)(frame.Total() * frame.ElemSize());
                var bytes = new byte[byteCount];
                System.Runtime.InteropServices.Marshal.Copy(frame.Data, bytes, 0, byteCount);

                for (var index = 0; index < 40; index++)
                {
                    stream!.StandardInput.BaseStream.Write(bytes, 0, byteCount);
                }

                stream!.StandardInput.BaseStream.Flush();
                stream.StandardInput.Close();
                stream.WaitForExit(30_000);
            }
            finally
            {
                stream?.Dispose();
            }

            listener!.WaitForExit(30_000);

            Assert.True(File.Exists(recordedPath), "the UDP listener produced no file — nothing arrived at the sink");
            Assert.True(new FileInfo(recordedPath).Length > 0, "the UDP listener's file is empty");

            var metadata = Ffprobe.ExtractVideoMetadata(recordedPath);
            Assert.Equal(320, metadata.Resolution.Width);
            Assert.Equal(240, metadata.Resolution.Height);
        }
        finally
        {
            if (!listener!.HasExited)
            {
                listener.Kill();
            }

            listener.Dispose();

            if (File.Exists(recordedPath))
            {
                File.Delete(recordedPath);
            }
        }
    }
}
