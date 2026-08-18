using FaceFusion.Core;
using FaceFusion.Media;
using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Workflows;

/// <summary>
/// Per-call inputs a <see cref="WorkflowProcessorStep"/>'s <see cref="WorkflowProcessorStep.BuildInputs"/>
/// factory receives from <see cref="WorkflowCore.ProcessTempFrame"/> — the seven values Python's
/// <c>process_temp_frame</c> assembles into the dict it hands each processor module's
/// <c>process_frame</c>: <c>reference_vision_frame</c>, <c>source_vision_frames</c>,
/// <c>source_audio_frame</c>, <c>source_voice_frame</c>, <c>target_vision_frames</c>,
/// <c>temp_vision_frame</c> (already sliced to its first 3 channels, matching Python's
/// <c>temp_vision_frame[:, :, :3]</c>), <c>temp_vision_mask</c>. None of the <see cref="Mat"/>
/// fields are owned by the factory or by whatever <see cref="IProcessorInputs"/> it builds —
/// <see cref="WorkflowCore.ProcessTempFrame"/> disposes every one of them itself.
/// </summary>
public sealed record ProcessorFrameContext(
    Mat ReferenceVisionFrame,
    IReadOnlyList<Mat> SourceVisionFrames,
    double[,] SourceAudioFrame,
    double[,] SourceVoiceFrame,
    IReadOnlyList<Mat> TargetVisionFrames,
    Mat TempVisionFrame,
    Mat TempVisionMask);

/// <summary>
/// Builds one processor's concrete <see cref="IProcessorInputs"/> (e.g.
/// <c>FaceSwapper.FaceSwapperInputs</c>) out of the shared per-frame <see cref="ProcessorFrameContext"/>.
/// See <see cref="WorkflowProcessorStep"/>'s remarks for why this indirection exists at all.
/// </summary>
public delegate IProcessorInputs ProcessorInputsFactory(ProcessorFrameContext context);

/// <summary>
/// One resolved entry of Python's <c>get_processors_modules(state_manager.get_item('processors'))</c>
/// list, paired with a factory that turns the shared <see cref="ProcessorFrameContext"/> into
/// that specific <see cref="IProcessor"/>'s own concrete <see cref="IProcessorInputs"/>.
///
/// <para>
/// <b>Why this pairing exists (and is not just <c>IReadOnlyList&lt;IProcessor&gt;</c>).</b>
/// Python's <c>process_temp_frame</c> builds one shared, untyped dict per processor call and
/// lets each module's <c>process_frame</c> pull whatever keys it needs out of it (plus whatever
/// it separately reads off <c>state_manager</c> — model choice, weights, mask settings, the
/// resolved <c>InferenceSession</c>s, ...). This port has no such dict and no
/// <c>state_manager</c> (PORT_CONVENTIONS.md rule 5): every processor's <c>*Inputs</c> record
/// (see <c>IProcessorInputs</c>'s remarks) instead demands its own full, strongly-typed settings
/// bundle up front — <c>FaceSwapperInputs</c> alone has 20 fields beyond the five shared frame
/// values. There is therefore no single shape <see cref="WorkflowCore.ProcessTempFrame"/> could
/// hand every processor directly; the caller that already knows which processors are running and
/// with which settings/sessions (the eventual CLI/UI layer, not built in this phase) supplies one
/// closure per processor that closes over that processor's settings and turns the shared frame
/// context into its concrete inputs. This is the same "the caller resolves what Python read from
/// <c>state_manager</c>" shape every other processor and <c>Ffmpeg</c> pipeline function in this
/// port already takes as explicit parameters — applied here because a per-processor inputs type
/// cannot be a plain parameter list.
/// </para>
/// </summary>
public sealed record WorkflowProcessorStep(IProcessor Processor, ProcessorInputsFactory BuildInputs);

/// <summary>
/// The state-manager values every one of <c>workflows/core.py</c>'s <c>conditional_get_*</c>
/// helpers and <c>process_temp_frame</c> itself read, bundled per PORT_CONVENTIONS.md rule 5
/// (no global mutable state — take settings as parameters) instead of being re-read from a
/// shared store on every call. <see cref="ToImage"/>/<see cref="ToVideo"/>/<see cref="ImageToImage"/>/
/// <see cref="ImageToVideo"/> all thread the same instance through a single run.
/// </summary>
public sealed record WorkflowRunContext(
    WorkflowMode WorkflowMode,
    string TargetPath,
    IReadOnlyList<string> SourcePaths,
    int ReferenceFrameNumber,
    int? TrimFrameStart,
    int? TrimFrameEnd,
    double OutputVideoFps,
    Audio.ExtractVoiceDelegate ExtractVoice);

/// <summary>
/// Port of <c>facefusion/workflows/core.py</c> — the shared helpers <c>to_image.py</c>/
/// <c>to_video.py</c> both build on: the process-stopping check, temp directory setup/teardown,
/// the four <c>conditional_get_*</c> frame/audio lookups, and <c>process_temp_frame</c> itself
/// (the function that actually drives the processor chain over one frame).
///
/// <para>
/// <b>No global state (rule 5).</b> Every Python <c>state_manager.get_item(...)</c> call in this
/// module becomes an explicit parameter here — <see cref="WorkflowRunContext"/> for the seven
/// values <c>process_temp_frame</c>'s own callees read, plus <see cref="ProcessManager"/>/
/// <see cref="Logger"/> instances (nullable, matching every <c>FaceFusion.Media.Ffmpeg</c>
/// pipeline function's own convention) in place of the bare <c>process_manager</c>/<c>logger</c>
/// module singletons.
/// </para>
/// </summary>
public static class WorkflowCore
{
    private const string ModuleName = "facefusion.workflows.core";

    /// <summary>Python: <c>is_process_stopping</c>.</summary>
    public static bool IsProcessStopping(ProcessManager processManager, Logger? logger = null)
    {
        if (processManager.IsStopping())
        {
            processManager.End();
            logger?.Info(Translator.Get("processing_stopped") ?? "processing_stopped", ModuleName);
        }

        return processManager.IsPending();
    }

    /// <summary>Python: <c>setup</c>. Returns Python's plain-<c>int</c> <c>ErrorCode</c> (see
    /// <c>FaceFusion.Types.TypeAliases</c>'s remarks on why that alias is not a C# enum).</summary>
    public static int Setup(string targetPath, string tempPath, Logger? logger = null)
    {
        if (TempHelper.CreateTempDirectory(targetPath, tempPath))
        {
            logger?.Debug(Translator.Get("creating_temp") ?? "creating_temp", ModuleName);
        }

        return 0;
    }

    /// <summary>Python: <c>clear</c>.</summary>
    public static int Clear(string targetPath, string tempPath, Logger? logger = null)
    {
        if (TempHelper.ClearTempDirectory(targetPath, tempPath))
        {
            logger?.Debug(Translator.Get("clearing_temp") ?? "clearing_temp", ModuleName);
        }

        return 0;
    }

    /// <summary>
    /// Python: <c>conditional_get_reference_vision_frame</c>. Caller owns the returned
    /// <see cref="Mat"/> and must dispose it (mirrors <c>Vision.ReadStaticImage</c>/
    /// <c>ReadStaticVideoFrame</c>'s own ownership convention, since this just wraps whichever
    /// one is called). Null exactly when the underlying read fails — same as Python passing
    /// <c>None</c> through to a caller that always assumes it succeeded.
    /// </summary>
    public static Mat? ConditionalGetReferenceVisionFrame(WorkflowMode workflowMode, string targetPath, int referenceFrameNumber)
    {
        if (workflowMode == WorkflowMode.ImageToVideo)
        {
            return VisionHelper.ReadStaticVideoFrame(targetPath, referenceFrameNumber);
        }

        return VisionHelper.ReadStaticImage(targetPath);
    }

    /// <summary>Python: <c>conditional_get_source_audio_frame</c>.</summary>
    public static double[,] ConditionalGetSourceAudioFrame(
        WorkflowMode workflowMode,
        string targetPath,
        IReadOnlyList<string> sourcePaths,
        int? trimFrameStart,
        int? trimFrameEnd,
        double outputVideoFps,
        int frameNumber)
    {
        if (workflowMode == WorkflowMode.ImageToVideo)
        {
            var (start, _) = VisionHelper.RestrictTrimFrame(targetPath, trimFrameStart, trimFrameEnd);
            var tempVideoFps = VisionHelper.RestrictVideoFps(targetPath, outputVideoFps);
            var sourceAudioPath = CommonHelper.GetFirst(FileSystem.FilterAudioPaths(sourcePaths));

            if (sourceAudioPath is not null)
            {
                var sourceAudioFrame = Audio.GetAudioFrame(sourceAudioPath, tempVideoFps, frameNumber - start);

                if (sourceAudioFrame is not null && AnyNonZero(sourceAudioFrame))
                {
                    return sourceAudioFrame;
                }
            }
        }

        return Audio.CreateEmptyAudioFrame();
    }

    /// <summary>Python: <c>conditional_get_source_voice_frame</c>.</summary>
    public static double[,] ConditionalGetSourceVoiceFrame(
        WorkflowMode workflowMode,
        string targetPath,
        IReadOnlyList<string> sourcePaths,
        int? trimFrameStart,
        int? trimFrameEnd,
        double outputVideoFps,
        int frameNumber,
        Audio.ExtractVoiceDelegate extractVoice)
    {
        if (workflowMode == WorkflowMode.ImageToVideo)
        {
            var (start, _) = VisionHelper.RestrictTrimFrame(targetPath, trimFrameStart, trimFrameEnd);
            var tempVideoFps = VisionHelper.RestrictVideoFps(targetPath, outputVideoFps);
            var sourceAudioPath = CommonHelper.GetFirst(FileSystem.FilterAudioPaths(sourcePaths));

            if (sourceAudioPath is not null)
            {
                var sourceVoiceFrame = Audio.GetVoiceFrame(sourceAudioPath, tempVideoFps, extractVoice, frameNumber - start);

                if (sourceVoiceFrame is not null && AnyNonZero(sourceVoiceFrame))
                {
                    return sourceVoiceFrame;
                }
            }
        }

        return Audio.CreateEmptyAudioFrame();
    }

    /// <summary>
    /// Python: <c>conditional_get_target_vision_frames</c>. Caller owns every <see cref="Mat"/>
    /// in the result and must dispose each one. <paramref name="targetFrameAmount"/> is passed
    /// straight through to <c>Vision.SelectVideoFrames</c>'s <c>frameOffset</c> parameter — same
    /// (confusingly-named-in-Python-too) reuse the real <c>select_video_frames(target_path,
    /// frame_number, target_frame_amount)</c> call site already makes.
    /// </summary>
    public static IReadOnlyList<Mat> ConditionalGetTargetVisionFrames(WorkflowMode workflowMode, string targetPath, int frameNumber, int targetFrameAmount)
    {
        if (workflowMode == WorkflowMode.ImageToVideo)
        {
            return VisionHelper.SelectVideoFrames(targetPath, frameNumber, targetFrameAmount);
        }

        var image = VisionHelper.ReadStaticImage(targetPath);
        return image is null ? Array.Empty<Mat>() : new[] { image };
    }

    /// <summary>
    /// Python: <c>process_temp_frame</c>. Runs the resolved processor chain over one frame and
    /// returns the merged result. Does not take ownership of <paramref name="targetVisionFrames"/>
    /// or <paramref name="tempVisionFrame"/> (mirrors every processor's own <c>process_frame</c>
    /// convention — see <c>ProcessorTypes.cs</c>'s remarks); caller owns the returned
    /// <see cref="Mat"/> and must dispose it.
    ///
    /// <para>
    /// Every intermediate <see cref="Mat"/> this method itself creates (the reference frame, the
    /// source frames, the 3-channel slice of <paramref name="tempVisionFrame"/>, the extracted
    /// mask, and each processor's replacement frame/mask along the way) is disposed before
    /// returning — see docs/DOTNET_PORT_PLAN.md §5a: this runs once per frame for the whole
    /// length of a video, so a leak here is not a one-off, it is a leak times the frame count.
    /// </para>
    /// </summary>
    public static Mat ProcessTempFrame(
        IReadOnlyList<WorkflowProcessorStep> processorSteps,
        WorkflowRunContext context,
        IReadOnlyList<Mat> targetVisionFrames,
        Mat tempVisionFrame,
        int frameNumber)
    {
        using var referenceVisionFrame = ConditionalGetReferenceVisionFrame(context.WorkflowMode, context.TargetPath, context.ReferenceFrameNumber)
            ?? throw new InvalidOperationException($"could not read the reference vision frame for '{context.TargetPath}'.");

        var sourceVisionFrames = VisionHelper.ReadStaticImages(context.SourcePaths);
        var sourceAudioFrame = ConditionalGetSourceAudioFrame(context.WorkflowMode, context.TargetPath, context.SourcePaths, context.TrimFrameStart, context.TrimFrameEnd, context.OutputVideoFps, frameNumber);
        var sourceVoiceFrame = ConditionalGetSourceVoiceFrame(context.WorkflowMode, context.TargetPath, context.SourcePaths, context.TrimFrameStart, context.TrimFrameEnd, context.OutputVideoFps, frameNumber, context.ExtractVoice);

        try
        {
            var currentFrame = ExtractFirst3Channels(tempVisionFrame);
            var currentMask = VisionHelper.ExtractVisionMask(tempVisionFrame);

            try
            {
                foreach (var step in processorSteps)
                {
                    var frameContext = new ProcessorFrameContext(referenceVisionFrame, sourceVisionFrames, sourceAudioFrame, sourceVoiceFrame, targetVisionFrames, currentFrame, currentMask);
                    var inputs = step.BuildInputs(frameContext);
                    var outputs = step.Processor.ProcessFrame(inputs);

                    if (!ReferenceEquals(outputs.VisionFrame, currentFrame))
                    {
                        currentFrame.Dispose();
                        currentFrame = outputs.VisionFrame;
                    }

                    if (!ReferenceEquals(outputs.Mask, currentMask))
                    {
                        currentMask.Dispose();
                        currentMask = outputs.Mask;
                    }
                }

                return VisionHelper.ConditionalMergeVisionMask(currentFrame, currentMask);
            }
            finally
            {
                currentFrame.Dispose();
                currentMask.Dispose();
            }
        }
        finally
        {
            foreach (var sourceVisionFrame in sourceVisionFrames)
            {
                sourceVisionFrame.Dispose();
            }
        }
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>temp_vision_frame[:, :, :3]</c> — a numpy view onto the first 3 channels, used
    /// wherever <paramref name="frame"/> may carry an alpha channel (read with
    /// <c>ColorMode.Rgba</c>) that must not reach the processor chain. <see cref="Mat"/> has no
    /// disposal-safe view equivalent (see <c>Vision.ExtractVisionMask</c>'s own channel-split
    /// pattern), so a 3-channel frame is cloned and a 4-channel one has its alpha channel
    /// dropped via split/merge; either way the result is an independently owned copy. Caller
    /// owns the returned <see cref="Mat"/> and must dispose it.
    /// </summary>
    internal static Mat ExtractFirst3Channels(Mat frame)
    {
        if (frame.Channels() <= 3)
        {
            return frame.Clone();
        }

        var channels = Cv2.Split(frame);
        try
        {
            var merged = new Mat();
            Cv2.Merge(new[] { channels[0], channels[1], channels[2] }, merged);
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

    /// <summary>Python: <c>numpy.any(audio_frame)</c> — true when at least one sample is non-zero.</summary>
    private static bool AnyNonZero(double[,] audioFrame)
    {
        for (var row = 0; row < audioFrame.GetLength(0); row++)
        {
            for (var column = 0; column < audioFrame.GetLength(1); column++)
            {
                if (audioFrame[row, column] != 0)
                {
                    return true;
                }
            }
        }

        return false;
    }
}
