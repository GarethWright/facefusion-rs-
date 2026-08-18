using FaceFusion.Processors;
using FaceFusion.Types;
using FaceFusion.Workflows;
using Microsoft.ML.OnnxRuntime;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Cli;

/// <summary>
/// Turns a <c>--processors</c> name plus the step's flat args bag into a real
/// <see cref="WorkflowProcessorStep"/> — the CLI-layer closure <see cref="WorkflowProcessorStep"/>'s
/// own remarks describe as "the eventual CLI/UI layer, not built in this phase". Two phases,
/// matching <c>process_step</c>'s own ordering (<c>processors_pre_check()</c> runs before any
/// processor module is actually used):
///
/// <list type="bullet">
/// <item><description><see cref="PreCheck"/> — cheap, no <see cref="InferenceSession"/>
/// allocation; local file-presence check only (matches every processor's own reduced-scope
/// <c>PreCheck</c>, see e.g. <c>FrameColorizer.PreCheck</c>'s remarks).</description></item>
/// <item><description><see cref="Build"/> — called only after every processor's
/// <see cref="PreCheck"/> has passed; opens the model's <see cref="InferenceSession"/> and
/// returns it alongside the step so the caller can dispose it once the run is done.</description></item>
/// </list>
///
/// <para>
/// <b>Coverage.</b> Only <c>frame_colorizer</c> is wired end to end and verified against the
/// real Python CLI (see the assignment report). <c>background_remover</c> has a real
/// <c>Processor : IProcessor</c> adapter and no face-pipeline dependency, so it was attempted
/// next, but its <c>ProcessFrame</c> hits a pre-existing OpenCvSharp size-mismatch bug in
/// <c>BackgroundRemover.cs</c> (outside this assignment's file scope to fix) and was dropped
/// rather than shipped broken. Every other name in the Phase-6 brief's ordered list
/// (<c>face_enhancer</c>, <c>frame_enhancer</c> — no <see cref="IProcessor"/> adapter exists at
/// all yet; <c>face_debugger</c>, <c>face_swapper</c>, <c>age_modifier</c>,
/// <c>expression_restorer</c>, <c>lip_syncer</c>, <c>deep_swapper</c>, <c>face_editor</c> — an
/// adapter exists but its <c>*Inputs</c> record demands the full face pipeline (detector,
/// landmarker, recognizer, masker — resolving faces per frame, none of which this phase builds)
/// on top of its own model, which is a materially larger job than one model + one frame) throws
/// <see cref="NotSupportedException"/> naming the processor, per the assignment's "never return
/// a silently-wrong step" instruction.
/// </para>
/// </summary>
public static class ProcessorStepFactory
{
	/// <summary>Cheap, allocation-free (no <see cref="InferenceSession"/>) local pre-check for
	/// one processor. Python: the processor-specific half of <c>pre_check()</c>.</summary>
	public static bool PreCheck(string processorName, IReadOnlyDictionary<string, object?> args)
	{
		return processorName switch
		{
			"frame_colorizer" => FrameColorizer.PreCheck(ReadFrameColorizerModel(args)),
			_ => throw Unsupported(processorName),
		};
	}

	/// <summary>One built processor step plus the native resource (an <see cref="InferenceSession"/>)
	/// the caller must dispose once the run finishes.</summary>
	public sealed record BuiltStep(WorkflowProcessorStep Step, IDisposable Resource);

	/// <summary>Builds the real step. Only call after <see cref="PreCheck"/> has returned
	/// <see langword="true"/> for this processor — matches <c>process_step</c>'s own ordering.</summary>
	public static BuiltStep Build(string processorName, IReadOnlyDictionary<string, object?> args)
	{
		return processorName switch
		{
			"frame_colorizer" => BuildFrameColorizer(args),
			_ => throw Unsupported(processorName),
		};
	}

	private static NotSupportedException Unsupported(string processorName)
		=> new($"processor '{processorName}' is not wired into headless-run yet (ProcessorStepFactory does not build it) — " +
			"see ProcessorStepFactory's class remarks for why.");

	// -----------------------------------------------------------------
	// frame_colorizer
	// -----------------------------------------------------------------

	private static FrameColorizerModel ReadFrameColorizerModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<FrameColorizerModel>(StepArgsReader.GetString(args, "frame_colorizer_model", "ddcolor"));

	private static BuiltStep BuildFrameColorizer(IReadOnlyDictionary<string, object?> args)
	{
		var model = ReadFrameColorizerModel(args);
		var modelSize = VisionHelper.UnpackResolution(StepArgsReader.GetString(args, "frame_colorizer_size", "256x256"));
		var blend = StepArgsReader.GetInt(args, "frame_colorizer_blend", 100);
		var options = FrameColorizer.GetModelOptions(model);

		var session = new InferenceSession(options.Source.Path);
		var processor = new FrameColorizer.Processor();

		var step = new WorkflowProcessorStep(processor, context => new FrameColorizer.FrameColorizerInputs(
			context.TempVisionFrame,
			context.TempVisionMask,
			options.Type,
			modelSize,
			session,
			blend));

		return new BuiltStep(step, session);
	}

	// background_remover was attempted and dropped: its Processor.ProcessFrame hits a
	// pre-existing OpenCvSharp size-mismatch in Cv2.Min (BackgroundRemover.cs:759) that is
	// out of this assignment's file scope to fix (only FaceFusion.Cli/ files and the two test
	// files are in scope). See the class remarks and the final report for the exact failure.
}
