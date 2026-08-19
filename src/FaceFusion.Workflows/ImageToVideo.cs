using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Types;

namespace FaceFusion.Workflows;

/// <summary>
/// Port of <c>facefusion/workflows/image_to_video.py</c> — chains <see cref="ToVideo"/>'s
/// stages exactly as Python's <c>tasks</c> list does, branching on <paramref name="workflowStrategy"/>
/// between the disk and in-memory frame-processing paths, same as Python's
/// <c>state_manager.get_item('workflow_strategy') == 'disk' | 'memory'</c> checks.
///
/// <para>
/// <b>Which strategy to prefer.</b> <c>docs/DOTNET_PORT_PLAN.md</c> §5b: "Where
/// <c>video_memory_strategy</c> allows, the .NET port should prefer the memory path". Python's
/// own CLI default (<c>facefusion/program.py</c>: <c>default =
/// config.get_str_value('workflow', 'workflow_strategy', 'memory')</c>) already agrees —
/// <see cref="WorkflowStrategy.Memory"/> is the default a caller should pass unless it has a
/// specific reason (e.g. a very high resolution/frame count where the disk path's lower peak
/// memory — one frame at a time on disk vs. up to <c>executionThreadCount</c> resident frames —
/// matters more than the extra encode-time disk I/O) to choose <see cref="WorkflowStrategy.Disk"/>
/// instead; this port reproduces Python's existing selection logic (a plain caller-supplied
/// choice) rather than inventing a new automatic policy.
/// </para>
/// </summary>
public static class ImageToVideo
{
    /// <summary>Python: <c>process(start_time)</c>. See <see cref="ImageToImage.Process"/>'s
    /// remarks on why <paramref name="processManager"/> is not nullable.</summary>
    public static int Process(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        WorkflowStrategy workflowStrategy,
        string outputPath,
        string tempPath,
        string tempFrameFormat,
        double outputVideoScale,
        VideoEncoder outputVideoEncoder,
        int outputVideoQuality,
        VideoPreset outputVideoPreset,
        TempPixelFormat tempPixelFormat,
        int targetFrameAmount,
        int executionThreadCount,
        int outputAudioVolume,
        AudioEncoder outputAudioEncoder,
        int outputAudioQuality,
        double startTime,
        ContentAnalyser? contentAnalyser,
        string modelsDirectory,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders,
        ProcessManager processManager,
        UpdateProgress? updateProgress = null,
        Logger? logger = null,
        LogLevel? logLevel = null)
    {
        ArgumentNullException.ThrowIfNull(processManager);

        var tasks = new List<Func<int>>
        {
            () => ToVideo.AnalyseVideo(contentAnalyser, context.TargetPath, context.TrimFrameStart, context.TrimFrameEnd, modelsDirectory, executionDeviceIds, executionProviders),
            () => WorkflowCore.Clear(context.TargetPath, tempPath, logger),
            () => WorkflowCore.Setup(context.TargetPath, tempPath, logger),
        };

        if (workflowStrategy == WorkflowStrategy.Disk)
        {
            tasks.Add(() => ToVideo.ExtractFrames(context.TargetPath, outputVideoScale, context.OutputVideoFps, context.TrimFrameStart, context.TrimFrameEnd, tempPath, tempFrameFormat, updateProgress, processManager, logger, logLevel));
            tasks.Add(() => ToVideo.ProcessDiskFrames(processorSteps, context, tempPath, tempFrameFormat, targetFrameAmount, executionThreadCount, updateProgress, processManager, logger));
            tasks.Add(() => ToVideo.MergeFrames(context.TargetPath, outputVideoScale, context.OutputVideoFps, context.TrimFrameStart, context.TrimFrameEnd, outputVideoEncoder, outputVideoQuality, outputVideoPreset, tempPath, tempFrameFormat, updateProgress, processManager, logger, logLevel));
        }

        if (workflowStrategy == WorkflowStrategy.Memory)
        {
            tasks.Add(() => ToVideo.ProcessMemoryFrames(processorSteps, context, tempPath, outputVideoScale, outputVideoEncoder, outputVideoQuality, outputVideoPreset, tempPixelFormat, targetFrameAmount, executionThreadCount, updateProgress, processManager, logger));
        }

        tasks.Add(() => ToVideo.RestoreAudio(context.TargetPath, outputPath, context.SourcePaths, context.TrimFrameStart, context.TrimFrameEnd, outputAudioVolume, outputAudioEncoder, outputAudioQuality, tempPath, processManager, logger, logLevel));
        tasks.Add(() => ToVideo.FinalizeVideo(outputPath, startTime, logger));
        tasks.Add(() => WorkflowCore.Clear(context.TargetPath, tempPath, logger));

        processManager.Start();

        foreach (var task in tasks)
        {
            var errorCode = task();

            if (errorCode > 0)
            {
                processManager.End();
                return errorCode;
            }
        }

        processManager.End();
        return 0;
    }
}
