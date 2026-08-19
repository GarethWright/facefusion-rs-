using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Media;
using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Workflows;

/// <summary>
/// Port of <c>facefusion/streamer.py</c> — the live path: read a camera, run the processor
/// chain over each captured frame, and either hand the result back for display or push it into
/// an ffmpeg process for UDP/v4l2 output.
///
/// <para>
/// <b>Why this does not reuse <see cref="WorkflowCore.ProcessTempFrame"/>.</b> Python's
/// <c>process_stream_frame</c> is deliberately a smaller function than <c>process_temp_frame</c>:
/// there is no reference frame, no audio or voice frame read from a file (both are empty), no
/// target-frame window (<c>target_vision_frames</c> is the single captured frame), and a
/// processor whose <c>pre_process('stream')</c> returns False is silently skipped rather than
/// failing the run — <c>expression_restorer</c> and the other non-live processors take that
/// path on every frame. Reproducing that shape is the parity-correct choice; routing streaming
/// through the file pipeline would change which processors run.
/// </para>
///
/// <para>
/// <b>No global state (rule 5).</b> Python reads <c>state_manager</c> for the source images,
/// thread count and log level; those are parameters here, and the camera pool belongs to the
/// caller's <see cref="FaceFusion.Vision.CameraManager"/>.
/// </para>
/// </summary>
public static class Streamer
{
    private const string ModuleName = "facefusion.streamer";

    /// <summary>
    /// Python: <c>process_stream_frame</c>. Caller owns the returned <see cref="Mat"/>;
    /// <paramref name="targetVisionFrame"/> is not taken over.
    /// </summary>
    public static Mat ProcessStreamFrame(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        IReadOnlyList<Mat> sourceVisionFrames,
        Mat targetVisionFrame,
        ProcessorRunPaths paths,
        Logger? logger = null)
    {
        var sourceAudioFrame = Audio.CreateEmptyAudioFrame();
        var sourceVoiceFrame = Audio.CreateEmptyAudioFrame();

        var tempVisionFrame = targetVisionFrame.Clone();
        var tempVisionMask = VisionHelper.ExtractVisionMask(tempVisionFrame);

        try
        {
            foreach (var step in processorSteps)
            {
                // Python disables the logger around pre_process so a non-live processor's
                // "not supported in stream mode" message is not printed once per frame.
                logger?.Disable();
                var supported = step.Processor.PreProcess(ProcessMode.Stream, paths);
                logger?.Enable();

                if (!supported)
                {
                    continue;
                }

                var context = new ProcessorFrameContext(
                    // Python passes no reference frame at all in stream mode; the captured
                    // frame stands in, and face_selector_mode is forced to 'one' by the webcam
                    // component before streaming starts, so nothing reads it.
                    targetVisionFrame,
                    sourceVisionFrames,
                    sourceAudioFrame,
                    sourceVoiceFrame,
                    new[] { targetVisionFrame },
                    tempVisionFrame,
                    tempVisionMask);

                var outputs = step.Processor.ProcessFrame(step.BuildInputs(context));

                if (!ReferenceEquals(outputs.VisionFrame, tempVisionFrame))
                {
                    tempVisionFrame.Dispose();
                    tempVisionFrame = outputs.VisionFrame;
                }

                if (!ReferenceEquals(outputs.Mask, tempVisionMask))
                {
                    tempVisionMask.Dispose();
                    tempVisionMask = outputs.Mask;
                }
            }

            return tempVisionFrame;
        }
        catch
        {
            tempVisionFrame.Dispose();
            throw;
        }
        finally
        {
            tempVisionMask.Dispose();
        }
    }

    /// <summary>
    /// Python: <c>multi_process_capture</c>. Reads frames as fast as the camera delivers them,
    /// processes each on a worker, and yields results.
    ///
    /// <para>
    /// <b>Deliberate difference: results are yielded in capture order.</b> Python appends each
    /// completed future to a deque as it finishes and yields from there, so under uneven
    /// per-frame processing time a later frame can reach the screen before an earlier one —
    /// visible as a stutter that jumps backwards. This keeps a bounded, ordered queue instead,
    /// which is the same shape <see cref="ToVideo"/> already uses for its own frame workers and
    /// what makes its output deterministic. The concurrency is identical; only the emission
    /// order is pinned.
    /// </para>
    /// </summary>
    public static IEnumerable<Mat> MultiProcessCapture(
        VideoCapture cameraCapture,
        double cameraFps,
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        IReadOnlyList<string> sourcePaths,
        int executionThreadCount,
        ContentAnalyser contentAnalyser,
        string modelsDirectory,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders,
        CancellationToken cancellationToken,
        Logger? logger = null)
    {
        var sourceVisionFrames = VisionHelper.ReadStaticImages(sourcePaths);
        var paths = new ProcessorRunPaths(sourcePaths, null, null);
        var pending = new Queue<Task<Mat>>();

        try
        {
            while (cameraCapture.IsOpened() && !cancellationToken.IsCancellationRequested)
            {
                var captureVisionFrame = new Mat();

                if (!cameraCapture.Read(captureVisionFrame) || !VisionHelper.IsVisionFrame(captureVisionFrame))
                {
                    captureVisionFrame.Dispose();

                    // Python's `while camera_capture.isOpened()` spins forever on a source that
                    // has ended; a file-backed capture (which is how this port is testable
                    // without a device) does end, so a failed read stops the loop.
                    break;
                }

                if (contentAnalyser.AnalyseStream(captureVisionFrame, modelsDirectory, executionDeviceIds, executionProviders, cameraFps))
                {
                    captureVisionFrame.Dispose();
                    cameraCapture.Release();
                    break;
                }

                var frame = captureVisionFrame;
                pending.Enqueue(Task.Run(() =>
                {
                    try
                    {
                        return ProcessStreamFrame(processorSteps, sourceVisionFrames, frame, paths, logger);
                    }
                    finally
                    {
                        frame.Dispose();
                    }
                }, cancellationToken));

                // Bounded: never hold more than executionThreadCount frames in flight, which is
                // what keeps a live stream from growing without limit when processing is slower
                // than capture (docs/IMPLEMENTATION_STATUS.md records how large one in-flight
                // frame actually is).
                while (pending.Count >= Math.Max(1, executionThreadCount))
                {
                    yield return pending.Dequeue().GetAwaiter().GetResult();
                }
            }

            while (pending.Count > 0)
            {
                yield return pending.Dequeue().GetAwaiter().GetResult();
            }
        }
        finally
        {
            // Drain anything still queued after an early exit, so no worker is left writing to
            // a Mat nobody will dispose.
            while (pending.Count > 0)
            {
                try
                {
                    pending.Dequeue().GetAwaiter().GetResult().Dispose();
                }
                catch (Exception)
                {
                    // The frame failed to process; there is nothing to dispose and the stream
                    // is already ending.
                }
            }

            foreach (var sourceVisionFrame in sourceVisionFrames)
            {
                sourceVisionFrame.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>open_stream</c>. Builds the ffmpeg command for a UDP or v4l2 sink and starts
    /// it; the caller writes raw frames to the returned process's stdin.
    /// </summary>
    public static System.Diagnostics.Process? OpenStream(StreamMode streamMode, string streamResolution, double streamFps, Logger? logger = null)
    {
        var commands = new List<string>();
        commands.AddRange(FfmpegBuilder.CaptureVideo());
        commands.AddRange(FfmpegBuilder.SetMediaResolution(streamResolution));
        commands.AddRange(FfmpegBuilder.SetInputFps(streamFps));

        if (streamMode == StreamMode.Udp)
        {
            commands.AddRange(FfmpegBuilder.SetInput("-"));
            commands.AddRange(FfmpegBuilder.SetStreamMode("udp"));
            commands.AddRange(FfmpegBuilder.SetStreamQuality(2000));
            commands.AddRange(FfmpegBuilder.SetOutput("udp://localhost:27000?pkt_size=1316"));
        }

        if (streamMode == StreamMode.V4l2)
        {
            const string deviceDirectoryPath = "/sys/devices/virtual/video4linux";
            commands.AddRange(FfmpegBuilder.SetInput("-"));
            commands.AddRange(FfmpegBuilder.SetStreamMode("v4l2"));

            if (Directory.Exists(deviceDirectoryPath))
            {
                foreach (var deviceName in Directory.GetFileSystemEntries(deviceDirectoryPath).Select(Path.GetFileName))
                {
                    commands.AddRange(FfmpegBuilder.SetOutput("/dev/" + deviceName));
                }
            }
            else
            {
                logger?.Error(
                    Translator.Get("stream_not_loaded", ("stream_mode", streamMode.ToWireName()))
                        ?? $"stream {streamMode.ToWireName()} not loaded",
                    ModuleName);
            }
        }

        return Ffmpeg.OpenFfmpeg(commands);
    }
}
