using System.Buffers;
using System.Runtime.InteropServices;
using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Media;
using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;
using LogLevel = FaceFusion.Types.LogLevel;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Workflows;

/// <summary>
/// Port of <c>facefusion/workflows/to_video.py</c> — the video-workflow stages
/// <see cref="ImageToVideo"/> chains: NSFW analysis, frame extraction, the two frame-processing
/// strategies (disk and in-memory), merge, audio restore, and finalize.
///
/// <para>
/// <b>Concurrency (assignment-mandated design note).</b> Python fans frame processing out with
/// a <c>ThreadPoolExecutor(max_workers = execution_thread_count)</c>, submits every frame up
/// front into a <c>deque</c> of futures, then pops them strictly in submission order —
/// <c>future.result()</c> blocks until *that* future is done even if a later one finished
/// first, so writes/disk-reads happen in frame order while up to <c>execution_thread_count</c>
/// frames are in flight at once. <c>docs/DOTNET_PORT_PLAN.md</c> §2 suggests a TPL Dataflow
/// <c>TransformBlock</c> with <c>EnsureOrdered</c> for this shape, but
/// <c>System.Threading.Tasks.Dataflow</c> is a separate NuGet package this port may not add
/// (PORT_CONVENTIONS.md), so <see cref="ProcessDiskFrames"/>/<see cref="ProcessMemoryFrames"/>
/// instead reproduce Python's own submit-ahead/pop-in-order shape directly with a bounded
/// <see cref="Queue{T}"/> of <see cref="Task{Mat}"/>: at most <c>executionThreadCount</c> frames
/// are ever queued/decoded/processed at once (bounded memory — a 1080p frame is ~6&#160;MB, so an
/// unbounded queue would exhaust RAM on a long video, exactly the failure mode the assignment
/// calls out), and results are dequeued and written in the same order they were submitted
/// (<see cref="Queue{T}"/> is FIFO), which is frame order. This is the same guarantee a bounded
/// <c>Parallel.ForEachAsync</c> would give and needs no extra dependency.
/// </para>
/// </summary>
public static class ToVideo
{
    private const string ModuleName = "facefusion.workflows.to_video";

    /// <summary>Python: <c>analyse_video</c>. See <see cref="ToImage.AnalyseImage"/>'s remarks
    /// on why <paramref name="contentAnalyser"/> is nullable.</summary>
    public static int AnalyseVideo(
        ContentAnalyser? contentAnalyser,
        string targetPath,
        int? trimFrameStart,
        int? trimFrameEnd,
        string modelsDirectory,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var (start, end) = VisionHelper.RestrictTrimFrame(targetPath, trimFrameStart, trimFrameEnd);

        if (contentAnalyser is not null && contentAnalyser.AnalyseVideo(targetPath, start, end, modelsDirectory, executionDeviceIds, executionProviders))
        {
            return 3;
        }

        return 0;
    }

    /// <summary>Python: <c>extract_frames</c>.</summary>
    public static int ExtractFrames(
        string targetPath,
        double outputVideoScale,
        double outputVideoFps,
        int? trimFrameStart,
        int? trimFrameEnd,
        string tempPath,
        string tempFrameFormat,
        UpdateProgress? updateProgress = null,
        ProcessManager? processManager = null,
        Logger? logger = null,
        LogLevel? logLevel = null)
    {
        var (start, end) = VisionHelper.RestrictTrimFrame(targetPath, trimFrameStart, trimFrameEnd);
        var videoResolution = VisionHelper.DetectVideoResolution(targetPath)
            ?? throw new InvalidOperationException($"could not detect the video resolution of '{targetPath}'.");
        var outputVideoResolution = VisionHelper.ScaleResolution(videoResolution, outputVideoScale);
        var tempVideoResolution = VisionHelper.RestrictVideoResolution(targetPath, outputVideoResolution);
        var tempVideoFps = VisionHelper.RestrictVideoFps(targetPath, outputVideoFps);

        logger?.Info(
            Translator.Get("extracting_frames", ("resolution", VisionHelper.PackResolution(tempVideoResolution)), ("fps", tempVideoFps)) ?? "extracting_frames",
            ModuleName);

        if (Ffmpeg.ExtractFrames(targetPath, tempVideoResolution, tempVideoFps, start, end, tempPath, tempFrameFormat, updateProgress ?? (_ => { }), processManager, logger, logLevel))
        {
            logger?.Debug(Translator.Get("extracting_frames_succeeded") ?? "extracting_frames_succeeded", ModuleName);
        }
        else
        {
            if (processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
            {
                return 4;
            }

            logger?.Error(Translator.Get("extracting_frames_failed") ?? "extracting_frames_failed", ModuleName);
            return 1;
        }

        return 0;
    }

    /// <summary>Python: <c>process_disk_frame</c>. Reads <paramref name="tempFramePath"/>, runs
    /// <paramref name="processorSteps"/> over it, writes the result back over the same path.
    /// </summary>
    public static bool ProcessDiskFrame(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        string tempFramePath,
        int frameNumber,
        int targetFrameAmount)
    {
        var targetVisionFrames = WorkflowCore.ConditionalGetTargetVisionFrames(context.WorkflowMode, context.TargetPath, frameNumber, targetFrameAmount);

        try
        {
            using var tempVisionFrame = VisionHelper.ReadStaticImage(tempFramePath, ColorMode.Rgba)
                ?? throw new InvalidOperationException($"could not read temp frame '{tempFramePath}'.");
            using var processedVisionFrame = WorkflowCore.ProcessTempFrame(processorSteps, context, targetVisionFrames, tempVisionFrame, frameNumber);

            return VisionHelper.WriteImage(tempFramePath, processedVisionFrame);
        }
        finally
        {
            foreach (var targetVisionFrame in targetVisionFrames)
            {
                targetVisionFrame.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>process_disk_frames</c>. See the class remarks for the bounded/ordered
    /// concurrency shape. Unlike Python's <c>tqdm</c> progress bar (a CLI display concern
    /// skipped throughout this port — see <c>Ffmpeg</c>'s own remarks), <paramref name="updateProgress"/>
    /// is invoked once per completed frame with the running completed count, and
    /// <c>execution_providers</c>' postfix display has no equivalent here.
    /// </summary>
    public static int ProcessDiskFrames(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        string tempPath,
        string tempFrameFormat,
        int targetFrameAmount,
        int executionThreadCount,
        UpdateProgress? updateProgress = null,
        ProcessManager? processManager = null,
        Logger? logger = null)
    {
        var tempFrameSet = TempHelper.ResolveTempFrameSet(context.TargetPath, tempPath, tempFrameFormat);

        if (tempFrameSet.Count == 0)
        {
            logger?.Error(Translator.Get("temp_frames_not_found") ?? "temp_frames_not_found", ModuleName);
            return 1;
        }

        // Python: `read_static_video_frame(target_path, reference_frame_number)` here just
        // warms the lru_cache before the thread pool starts hammering it concurrently; the
        // C# `Vision` cache is thread-safe (guarded by a lock) either way, so this call is
        // reproduced for parity of intent but is not load-bearing for correctness here.
        VisionHelper.ReadStaticVideoFrame(context.TargetPath, context.ReferenceFrameNumber)?.Dispose();

        var frameNumbers = tempFrameSet.Keys.OrderBy(frameNumber => frameNumber).ToList();
        var maxInFlight = Math.Max(1, executionThreadCount);
        var window = new Queue<(int FrameNumber, Task<bool> Task)>();
        var index = 0;
        var completed = 0;
        var stopping = false;

        while (index < frameNumbers.Count || window.Count > 0)
        {
            while (!stopping && window.Count < maxInFlight && index < frameNumbers.Count)
            {
                var frameNumber = frameNumbers[index++];
                var tempFramePath = tempFrameSet[frameNumber];
                var task = Task.Run(() => ProcessDiskFrame(processorSteps, context, tempFramePath, frameNumber, targetFrameAmount));
                window.Enqueue((frameNumber, task));
            }

            if (window.Count == 0)
            {
                break;
            }

            var (_, frameTask) = window.Dequeue();

            if (!stopping && processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
            {
                stopping = true;
            }

            if (stopping)
            {
                // Python cancels every still-pending future and clears the deque without
                // calling result() on them, but the ThreadPoolExecutor context manager still
                // blocks on every already-submitted (including already-running) future when
                // the `with` block exits. Draining (observing, not discarding) each task here
                // reproduces that same "wait for in-flight work, stop scheduling more" shape.
                frameTask.GetAwaiter().GetResult();
                index = frameNumbers.Count;
                continue;
            }

            frameTask.GetAwaiter().GetResult();
            completed++;
            updateProgress?.Invoke(completed);
        }

        foreach (var step in processorSteps)
        {
            step.Processor.PostProcess();
        }

        if (processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
        {
            return 4;
        }

        return 0;
    }

    /// <summary>
    /// Python: <c>process_memory_frame</c>. Caller owns the returned <see cref="Mat"/> and must
    /// dispose it.
    /// </summary>
    public static Mat ProcessMemoryFrame(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        int frameNumber,
        Resolution tempVideoResolution,
        Resolution outputVideoResolution,
        TempPixelFormat tempPixelFormat,
        int targetFrameAmount)
    {
        var targetVisionFrames = VisionHelper.SelectVideoFrames(context.TargetPath, frameNumber, targetFrameAmount);

        try
        {
            var targetVisionFrame = CommonHelper.GetMiddle(targetVisionFrames)
                ?? throw new InvalidOperationException($"select_video_frames returned no frames for '{context.TargetPath}' at frame {frameNumber}.");

            var tempVisionFrame = targetVisionFrame.Clone();

            try
            {
                if (!MatchesResolution(targetVisionFrame, tempVideoResolution))
                {
                    tempVisionFrame = Resize(tempVisionFrame, tempVideoResolution);
                }

                var processedVisionFrame = WorkflowCore.ProcessTempFrame(processorSteps, context, targetVisionFrames, tempVisionFrame, frameNumber);
                tempVisionFrame.Dispose();
                tempVisionFrame = processedVisionFrame;

                if (!MatchesResolution(tempVisionFrame, outputVideoResolution))
                {
                    tempVisionFrame = Resize(tempVisionFrame, outputVideoResolution);
                }

                if (tempPixelFormat == TempPixelFormat.Bgra)
                {
                    var converted = new Mat();
                    Cv2.CvtColor(tempVisionFrame, converted, ColorConversionCodes.BGR2BGRA);
                    tempVisionFrame.Dispose();
                    tempVisionFrame = converted;
                }
                else if (tempPixelFormat == TempPixelFormat.Bgr24)
                {
                    var trimmed = WorkflowCore.ExtractFirst3Channels(tempVisionFrame);
                    tempVisionFrame.Dispose();
                    tempVisionFrame = trimmed;
                }

                // Python: `numpy.ascontiguousarray(temp_vision_frame)`. Every Mat produced
                // above (Clone/Resize/CvtColor/split-merge) is already a fresh, continuous
                // allocation, so there is nothing further to normalize here.
                return tempVisionFrame;
            }
            catch
            {
                tempVisionFrame.Dispose();
                throw;
            }
        }
        finally
        {
            foreach (var targetVisionFrame in targetVisionFrames)
            {
                targetVisionFrame.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>process_memory_frames</c>. See the class remarks for the bounded/ordered
    /// concurrency shape.
    /// </summary>
    public static int ProcessMemoryFrames(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        string tempPath,
        double outputVideoScale,
        VideoEncoder outputVideoEncoder,
        int outputVideoQuality,
        VideoPreset outputVideoPreset,
        TempPixelFormat tempPixelFormat,
        int targetFrameAmount,
        int executionThreadCount,
        UpdateProgress? updateProgress = null,
        ProcessManager? processManager = null,
        Logger? logger = null)
    {
        var (trimFrameStart, trimFrameEnd) = VisionHelper.RestrictTrimFrame(context.TargetPath, context.TrimFrameStart, context.TrimFrameEnd);
        var videoResolution = VisionHelper.DetectVideoResolution(context.TargetPath)
            ?? throw new InvalidOperationException($"could not detect the video resolution of '{context.TargetPath}'.");
        var outputVideoResolution = VisionHelper.ScaleResolution(videoResolution, outputVideoScale);
        var tempVideoResolution = VisionHelper.RestrictVideoResolution(context.TargetPath, outputVideoResolution);
        var tempVideoFps = VisionHelper.RestrictVideoFps(context.TargetPath, context.OutputVideoFps);

        if (trimFrameEnd <= trimFrameStart)
        {
            logger?.Error(Translator.Get("temp_frames_not_found") ?? "temp_frames_not_found", ModuleName);
            return 1;
        }

        // Python passes `output_video_resolution` for BOTH the writer's temp- and
        // output-resolution parameters here (`video_manager.get_writer(target_path,
        // temp_video_fps, output_video_resolution, output_video_resolution,
        // output_video_fps)`) — not a transcription bug, reproduced deliberately per
        // PORT_CONVENTIONS.md rule 1: process_memory_frame (see above) always resizes its
        // result to output_video_resolution before handing it to the writer, so the raw
        // frames arriving on the writer's stdin are already at that size, not at
        // temp_video_resolution.
        using var videoWriter = Ffmpeg.CreateVideoWriter(
            context.TargetPath, tempVideoFps, outputVideoResolution, outputVideoResolution, context.OutputVideoFps,
            outputVideoEncoder, outputVideoQuality, outputVideoPreset, tempPixelFormat, tempPath);

        if (!videoWriter.IsAvailable || videoWriter.StandardInput is null)
        {
            logger?.Error(Translator.Get("temp_frames_not_found") ?? "temp_frames_not_found", ModuleName);
            return 1;
        }

        VisionHelper.ReadStaticVideoFrame(context.TargetPath, context.ReferenceFrameNumber)?.Dispose();

        var frameNumbers = Enumerable.Range(trimFrameStart, trimFrameEnd - trimFrameStart).ToList();
        var maxInFlight = Math.Max(1, executionThreadCount);
        var window = new Queue<(int FrameNumber, Task<Mat> Task)>();
        var index = 0;
        var completed = 0;
        var stopping = false;

        while (index < frameNumbers.Count || window.Count > 0)
        {
            while (!stopping && window.Count < maxInFlight && index < frameNumbers.Count)
            {
                var frameNumber = frameNumbers[index++];
                var task = Task.Run(() => ProcessMemoryFrame(processorSteps, context, frameNumber, tempVideoResolution, outputVideoResolution, tempPixelFormat, targetFrameAmount));
                window.Enqueue((frameNumber, task));
            }

            if (window.Count == 0)
            {
                break;
            }

            var (_, frameTask) = window.Dequeue();

            if (!stopping && processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
            {
                stopping = true;
            }

            if (stopping)
            {
                // Same drain-without-writing shape as ProcessDiskFrames — see its remarks.
                frameTask.GetAwaiter().GetResult().Dispose();
                index = frameNumbers.Count;
                continue;
            }

            using var frame = frameTask.GetAwaiter().GetResult();
            WriteFrameToStream(frame, videoWriter.StandardInput);
            completed++;
            updateProgress?.Invoke(completed);
        }

        videoWriter.FinishWriting();

        if (videoWriter.ExitCode != 0)
        {
            processManager?.Stop();
        }

        foreach (var step in processorSteps)
        {
            step.Processor.PostProcess();
        }

        if (processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
        {
            return 4;
        }

        return 0;
    }

    /// <summary>Python: <c>merge_frames</c>.</summary>
    public static int MergeFrames(
        string targetPath,
        double outputVideoScale,
        double outputVideoFps,
        int? trimFrameStart,
        int? trimFrameEnd,
        VideoEncoder outputVideoEncoder,
        int outputVideoQuality,
        VideoPreset outputVideoPreset,
        string tempPath,
        string tempFrameFormat,
        UpdateProgress? updateProgress = null,
        ProcessManager? processManager = null,
        Logger? logger = null,
        LogLevel? logLevel = null)
    {
        var (start, end) = VisionHelper.RestrictTrimFrame(targetPath, trimFrameStart, trimFrameEnd);
        var videoResolution = VisionHelper.DetectVideoResolution(targetPath)
            ?? throw new InvalidOperationException($"could not detect the video resolution of '{targetPath}'.");
        var outputVideoResolution = VisionHelper.ScaleResolution(videoResolution, outputVideoScale);
        var tempVideoFps = VisionHelper.RestrictVideoFps(targetPath, outputVideoFps);

        logger?.Info(
            Translator.Get("merging_video", ("resolution", VisionHelper.PackResolution(outputVideoResolution)), ("fps", outputVideoFps)) ?? "merging_video",
            ModuleName);

        if (Ffmpeg.MergeVideo(targetPath, tempVideoFps, outputVideoResolution, outputVideoFps, start, end, outputVideoEncoder, outputVideoQuality, outputVideoPreset, tempPath, tempFrameFormat, updateProgress ?? (_ => { }), processManager, logger, logLevel))
        {
            logger?.Debug(Translator.Get("merging_video_succeeded") ?? "merging_video_succeeded", ModuleName);
        }
        else
        {
            if (processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
            {
                return 4;
            }

            logger?.Error(Translator.Get("merging_video_failed") ?? "merging_video_failed", ModuleName);
            return 1;
        }

        return 0;
    }

    /// <summary>Python: <c>restore_audio</c>. Unlike Python, there is no <c>video_manager</c>
    /// reader/writer pool to clear (see <see cref="ProcessMemoryFrames"/>'s writer, opened and
    /// disposed per call rather than pooled) — the <c>video_manager.clear_video_pool()</c> calls
    /// have no equivalent here and are simply omitted.</summary>
    public static int RestoreAudio(
        string targetPath,
        string outputPath,
        IReadOnlyList<string> sourcePaths,
        int? trimFrameStart,
        int? trimFrameEnd,
        int outputAudioVolume,
        AudioEncoder outputAudioEncoder,
        int outputAudioQuality,
        string tempPath,
        ProcessManager? processManager = null,
        Logger? logger = null,
        LogLevel? logLevel = null)
    {
        var (start, end) = VisionHelper.RestrictTrimFrame(targetPath, trimFrameStart, trimFrameEnd);

        if (outputAudioVolume == 0)
        {
            logger?.Info(Translator.Get("skipping_audio") ?? "skipping_audio", ModuleName);
            TempHelper.MoveTempFile(targetPath, outputPath, tempPath);
            return 0;
        }

        var sourceAudioPath = CommonHelper.GetFirst(FileSystem.FilterAudioPaths(sourcePaths));

        if (sourceAudioPath is not null)
        {
            if (Ffmpeg.ReplaceAudio(targetPath, sourceAudioPath, outputPath, outputAudioEncoder, outputAudioQuality, outputAudioVolume, tempPath, processManager, logger, logLevel))
            {
                logger?.Debug(Translator.Get("replacing_audio_succeeded") ?? "replacing_audio_succeeded", ModuleName);
            }
            else
            {
                if (processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
                {
                    return 4;
                }

                logger?.Warn(Translator.Get("replacing_audio_skipped") ?? "replacing_audio_skipped", ModuleName);
                TempHelper.MoveTempFile(targetPath, outputPath, tempPath);
            }
        }
        else
        {
            if (Ffmpeg.RestoreAudio(targetPath, outputPath, start, end, outputAudioEncoder, outputAudioQuality, outputAudioVolume, tempPath, processManager, logger, logLevel))
            {
                logger?.Debug(Translator.Get("restoring_audio_succeeded") ?? "restoring_audio_succeeded", ModuleName);
            }
            else
            {
                if (processManager is not null && WorkflowCore.IsProcessStopping(processManager, logger))
                {
                    return 4;
                }

                logger?.Warn(Translator.Get("restoring_audio_skipped") ?? "restoring_audio_skipped", ModuleName);
                TempHelper.MoveTempFile(targetPath, outputPath, tempPath);
            }
        }

        return 0;
    }

    /// <summary>Python: <c>finalize_video</c>.</summary>
    public static int FinalizeVideo(string outputPath, double startTime, Logger? logger = null)
    {
        if (FileSystem.IsVideo(outputPath))
        {
            logger?.Info(Translator.Get("processing_video_succeeded", ("seconds", TimeHelper.CalculateEndTime(startTime))) ?? "processing_video_succeeded", ModuleName);
        }
        else
        {
            logger?.Error(Translator.Get("processing_video_failed") ?? "processing_video_failed", ModuleName);
            return 1;
        }

        return 0;
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    private static bool MatchesResolution(Mat frame, Resolution resolution)
        => frame.Cols == resolution.Width && frame.Rows == resolution.Height;

    private static Mat Resize(Mat frame, Resolution resolution)
    {
        var resized = new Mat();
        Cv2.Resize(frame, resized, new OpenCvSharp.Size(resolution.Width, resolution.Height));
        return resized;
    }

    /// <summary>
    /// Writes one <see cref="Mat"/>'s raw contiguous pixel bytes to the ffmpeg writer's stdin —
    /// the C# side of Python's <c>video_writer.get('process').stdin.write(vision_frame.data)</c>.
    /// Uses a pooled buffer (docs/DOTNET_PORT_PLAN.md §5a rule 4) rather than a fresh managed
    /// array per frame, since this runs once per frame for the whole length of a video.
    /// </summary>
    private static void WriteFrameToStream(Mat frame, Stream stream)
    {
        using var continuousFrame = frame.IsContinuous() ? null : frame.Clone();
        var source = continuousFrame ?? frame;
        var byteCount = checked((int)(source.Total() * source.ElemSize()));
        var buffer = ArrayPool<byte>.Shared.Rent(byteCount);

        try
        {
            Marshal.Copy(source.Data, buffer, 0, byteCount);
            stream.Write(buffer, 0, byteCount);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
