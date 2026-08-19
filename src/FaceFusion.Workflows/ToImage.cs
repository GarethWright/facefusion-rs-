using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Media;
using FaceFusion.Processors;
using FaceFusion.Types;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Workflows;

/// <summary>
/// Port of <c>facefusion/workflows/to_image.py</c> — the four image-workflow stages
/// <see cref="ImageToImage"/> chains: NSFW analysis, ffmpeg copy-to-temp, the processor chain
/// over the single temp frame, and ffmpeg finalize-to-output.
///
/// <para>
/// Every Python <c>state_manager.get_item(...)</c> read is an explicit parameter here (rule 5).
/// <see cref="ProcessManager"/>/<see cref="Logger"/> are nullable, matching the convention every
/// <see cref="Ffmpeg"/> pipeline function already uses.
/// </para>
/// </summary>
public static class ToImage
{
    private const string ModuleName = "facefusion.workflows.to_image";

    /// <summary>
    /// Python: <c>analyse_image</c>. <paramref name="contentAnalyser"/> is nullable because
    /// <c>FaceFusion.Face.ContentAnalyser</c> needs real NSFW model files on disk (see its own
    /// class remarks, Deviation 2) that are not guaranteed present in every environment this
    /// port runs in; passing <see langword="null"/> skips the check entirely rather than
    /// crashing (a deliberate divergence — Python always runs it — documented here since a
    /// caller that wants full Python parity should always pass a real instance).
    /// </summary>
    public static int AnalyseImage(
        ContentAnalyser? contentAnalyser,
        string targetPath,
        string modelsDirectory,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        if (contentAnalyser is not null && contentAnalyser.AnalyseImage(targetPath, modelsDirectory, executionDeviceIds, executionProviders))
        {
            return 3;
        }

        return 0;
    }

    /// <summary>Python: <c>prepare_image</c>.</summary>
    public static int PrepareImage(
        string targetPath,
        string tempPath,
        double outputImageScale,
        ProcessManager? processManager = null,
        Logger? logger = null)
    {
        var imageResolution = VisionHelper.DetectImageResolution(targetPath)
            ?? throw new InvalidOperationException($"could not detect the image resolution of '{targetPath}'.");
        var outputImageResolution = VisionHelper.ScaleResolution(imageResolution, outputImageScale);
        var tempImageResolution = VisionHelper.RestrictImageResolution(targetPath, outputImageResolution);

        logger?.Info(Translator.Get("copying_image", ("resolution", VisionHelper.PackResolution(tempImageResolution))) ?? "copying_image", ModuleName);

        if (Ffmpeg.CopyImage(targetPath, tempImageResolution, tempPath, processManager, logger))
        {
            logger?.Debug(Translator.Get("copying_image_succeeded") ?? "copying_image_succeeded", ModuleName);
        }
        else
        {
            logger?.Error(Translator.Get("copying_image_failed") ?? "copying_image_failed", ModuleName);
            processManager?.End();
            return 1;
        }

        return 0;
    }

    /// <summary>
    /// Python: <c>process_image</c>. Reads the temp frame ffmpeg copied in <see cref="PrepareImage"/>,
    /// runs <paramref name="processorSteps"/> over it via <see cref="WorkflowCore.ProcessTempFrame"/>,
    /// writes the merged result back over the same temp file, then calls every processor's
    /// <see cref="IProcessor.PostProcess"/> — same order as Python.
    /// </summary>
    public static int ProcessImage(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        string tempPath,
        ProcessManager? processManager = null)
    {
        var tempImagePath = TempHelper.GetTempFilePath(context.TargetPath, tempPath);
        var targetVisionFrames = WorkflowCore.ConditionalGetTargetVisionFrames(context.WorkflowMode, context.TargetPath, 0, 0);

        try
        {
            using var tempVisionFrame = VisionHelper.ReadStaticImage(tempImagePath, ColorMode.Rgba)
                ?? throw new InvalidOperationException($"could not read the copied temp image '{tempImagePath}'.");
            using var processedVisionFrame = WorkflowCore.ProcessTempFrame(processorSteps, context, targetVisionFrames, tempVisionFrame, 0);

            VisionHelper.WriteImage(tempImagePath, processedVisionFrame);
        }
        finally
        {
            foreach (var targetVisionFrame in targetVisionFrames)
            {
                targetVisionFrame.Dispose();
            }
        }

        foreach (var step in processorSteps)
        {
            step.Processor.PostProcess();
        }

        if (processManager is not null && WorkflowCore.IsProcessStopping(processManager))
        {
            return 4;
        }

        return 0;
    }

    /// <summary>Python: <c>finalize_image</c>.</summary>
    public static int FinalizeImage(
        string targetPath,
        string outputPath,
        string tempPath,
        double outputImageScale,
        int outputImageQuality,
        double startTime,
        ProcessManager? processManager = null,
        Logger? logger = null)
    {
        var imageResolution = VisionHelper.DetectImageResolution(targetPath)
            ?? throw new InvalidOperationException($"could not detect the image resolution of '{targetPath}'.");
        var outputImageResolution = VisionHelper.ScaleResolution(imageResolution, outputImageScale);

        logger?.Info(Translator.Get("finalizing_image", ("resolution", VisionHelper.PackResolution(outputImageResolution))) ?? "finalizing_image", ModuleName);

        if (Ffmpeg.FinalizeImage(targetPath, outputPath, outputImageResolution, outputImageQuality, tempPath, processManager, logger))
        {
            logger?.Debug(Translator.Get("finalizing_image_succeeded") ?? "finalizing_image_succeeded", ModuleName);
        }
        else
        {
            logger?.Warn(Translator.Get("finalizing_image_skipped") ?? "finalizing_image_skipped", ModuleName);
        }

        if (FileSystem.IsImage(outputPath))
        {
            logger?.Info(Translator.Get("processing_image_succeeded", ("seconds", TimeHelper.CalculateEndTime(startTime))) ?? "processing_image_succeeded", ModuleName);
        }
        else
        {
            logger?.Error(Translator.Get("processing_image_failed") ?? "processing_image_failed", ModuleName);
            return 1;
        }

        return 0;
    }
}
