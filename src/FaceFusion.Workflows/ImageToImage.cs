using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Types;

namespace FaceFusion.Workflows;

/// <summary>
/// Port of <c>facefusion/workflows/image_to_image.py</c> — chains <see cref="ToImage"/>'s stages
/// exactly as Python's <c>tasks</c> list does, stopping at the first stage that reports an
/// <c>ErrorCode &gt; 0</c>.
/// </summary>
public static class ImageToImage
{
    /// <summary>
    /// Python: <c>process(start_time)</c>. <paramref name="processManager"/> is not nullable
    /// here (unlike every individual stage's own optional logging/stop-check parameter) because
    /// Python's <c>process()</c> itself unconditionally calls <c>process_manager.start()</c>/
    /// <c>end()</c> — there is no "no process manager" case at this orchestration level.
    /// </summary>
    public static int Process(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        string outputPath,
        string tempPath,
        double outputImageScale,
        int outputImageQuality,
        double startTime,
        ContentAnalyser? contentAnalyser,
        string modelsDirectory,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders,
        ProcessManager processManager,
        Logger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(processManager);

        var tasks = new List<Func<int>>
        {
            () => ToImage.AnalyseImage(contentAnalyser, context.TargetPath, modelsDirectory, executionDeviceIds, executionProviders),
            () => WorkflowCore.Clear(context.TargetPath, tempPath, logger),
            () => WorkflowCore.Setup(context.TargetPath, tempPath, logger),
            () => ToImage.PrepareImage(context.TargetPath, tempPath, outputImageScale, processManager, logger),
            () => ToImage.ProcessImage(processorSteps, context, tempPath, processManager),
            () => ToImage.FinalizeImage(context.TargetPath, outputPath, tempPath, outputImageScale, outputImageQuality, startTime, processManager, logger),
            () => WorkflowCore.Clear(context.TargetPath, tempPath, logger),
        };

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
