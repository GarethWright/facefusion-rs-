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
/// <b>Coverage (Phase 6).</b> Wired and verified against the real Python CLI:
/// <c>frame_colorizer</c>, <c>background_remover</c> (a real Cv2.Min type-mismatch bug fixed in
/// <c>BackgroundRemover.cs</c> — see its class remarks) and <c>face_debugger</c>, which validates
/// the shared face pipeline (<see cref="FacePipelineFactory"/>) end to end since it has no
/// output model of its own to also get right.
/// </para>
///
/// <para>
/// <b>face_swapper attempted and dropped — a real, pre-existing bug, out of this assignment's
/// file scope to fix.</b> <c>FaceSwapper.SwapFace</c> calls <c>PixelBoost.ExplodePixelBoost</c>
/// on the float (<c>CV_32FC3</c>/<c>CV_64FC3</c>) Mats <c>NormalizeCropFrame</c> produces —
/// <c>NormalizeCropFrame</c>'s own remarks are explicit that this is deliberate ("never cast
/// back to <c>uint8</c> first") — but <c>PixelBoost.ExplodePixelBoost</c> hard-asserts every
/// input Mat is <c>CV_8UC3</c> and throws <see cref="ArgumentException"/> otherwise. This fires
/// on every real frame for every <see cref="FaceSwapperModelKind"/> (confirmed with
/// <c>inswapper_128</c>, whose <c>NormalizeCropFrame</c> branch returns <c>CV_32FC3</c>) — the
/// full <c>face_swapper</c> pipeline (<c>ProcessFrame</c> → <c>SwapFace</c> → ExplodePixelBoost)
/// has evidently never been exercised end to end before (the existing parity tests call
/// <c>ForwardSwapFace</c> directly, never the full pipeline). The fix belongs in
/// <c>PixelBoost.cs</c> or <c>FaceSwapper.cs</c>, neither of which this assignment's file scope
/// covers (Task 2 needed no new adapter for <c>face_swapper</c> — one already existed — and
/// Task 3 names only <c>BackgroundRemover.cs</c>), so per the assignment's own "never return a
/// silently-wrong step" instruction <c>face_swapper</c> is reported here rather than wired
/// broken. See the assignment report for the exact repro and stack trace.
/// </para>
///
/// <para>
/// Every other name in the Phase-6 brief's ordered list (<c>face_enhancer</c>,
/// <c>frame_enhancer</c> — no <see cref="IProcessor"/> adapter exists yet, and adding one plus
/// wiring it is a materially larger job than the remaining budget covers; <c>age_modifier</c>,
/// <c>expression_restorer</c>, <c>lip_syncer</c>, <c>deep_swapper</c>, <c>face_editor</c> — an
/// adapter exists, not yet wired here) throws <see cref="NotSupportedException"/> naming the
/// processor, per the same instruction.
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
			"background_remover" => BackgroundRemover.PreCheck(ReadBackgroundRemoverModel(args)),
			"face_debugger" => FacePipelineFactory.PreCheck(args),
			_ => throw Unsupported(processorName),
		};
	}

	/// <summary>One built processor step plus the native resource (an <see cref="InferenceSession"/>)
	/// the caller must dispose once the run finishes.</summary>
	public sealed record BuiltStep(WorkflowProcessorStep Step, IDisposable Resource);

	/// <summary>Builds the real step. Only call after <see cref="PreCheck"/> has returned
	/// <see langword="true"/> for this processor — matches <c>process_step</c>'s own ordering.
	/// <paramref name="faceResources"/> is required for any processor
	/// <see cref="FacePipelineFactory.Requires"/> names — build it once per run (see
	/// <see cref="FacePipelineFactory"/>'s class remarks on why it is shared rather than rebuilt
	/// per processor) and pass the same instance for every processor in the step's
	/// <c>--processors</c> list.</summary>
	public static BuiltStep Build(string processorName, IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources? faceResources = null)
	{
		return processorName switch
		{
			"frame_colorizer" => BuildFrameColorizer(args),
			"background_remover" => BuildBackgroundRemover(args),
			"face_debugger" => BuildFaceDebugger(args, faceResources ?? throw MissingFaceResources(processorName)),
			_ => throw Unsupported(processorName),
		};
	}

	private static InvalidOperationException MissingFaceResources(string processorName)
		=> new($"processor '{processorName}' needs a FacePipelineFactory.Resources instance — " +
			"pass one built via FacePipelineFactory.Build (see ProcessorStepFactory.Build's remarks).");

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

	// -----------------------------------------------------------------
	// background_remover
	// -----------------------------------------------------------------

	private static BackgroundRemoverModel ReadBackgroundRemoverModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<BackgroundRemoverModel>(StepArgsReader.GetString(args, "background_remover_model", "modnet"));

	/// <summary>Python: <c>normalize_color</c> — the 1/2/3/4-length int list a
	/// <c>--*-color</c> flag accepts, defaulting (like every processor's own config default)
	/// to fully-transparent black when the flag is absent.</summary>
	private static Types.Color ReadColor(IReadOnlyDictionary<string, object?> args, string key)
	{
		var channels = StepArgsReader.GetIntList(args, key, Array.Empty<int>());

		return channels.Count switch
		{
			1 => new Types.Color(channels[0], channels[0], channels[0], 255),
			2 => new Types.Color(channels[0], channels[1], channels[0], 255),
			3 => new Types.Color(channels[0], channels[1], channels[2], 255),
			4 => new Types.Color(channels[0], channels[1], channels[2], channels[3]),
			_ => new Types.Color(0, 0, 0, 0),
		};
	}

	private static BuiltStep BuildBackgroundRemover(IReadOnlyDictionary<string, object?> args)
	{
		var model = ReadBackgroundRemoverModel(args);
		var options = BackgroundRemover.GetModelOptions(model);
		var fillColor = ReadColor(args, "background_remover_fill_color");
		var despillColor = ReadColor(args, "background_remover_despill_color");

		var session = new InferenceSession(options.Source.Path);
		var processor = new BackgroundRemover.Processor();

		var step = new WorkflowProcessorStep(processor, context => new BackgroundRemover.BackgroundRemoverInputs(
			context.TempVisionFrame,
			context.TempVisionMask,
			options,
			session,
			fillColor,
			despillColor));

		return new BuiltStep(step, session);
	}

	// -----------------------------------------------------------------
	// face_debugger
	// -----------------------------------------------------------------

	/// <summary>No native resource of its own — <c>face_debugger</c> has no ONNX model (see
	/// <c>FaceDebugger.Processor.GetCommonModules</c>'s remarks); every session it uses belongs
	/// to the shared <see cref="FacePipelineFactory.Resources"/> the caller owns and disposes
	/// separately (see <see cref="Build"/>'s remarks). <see cref="BuiltStep.Resource"/> is
	/// non-nullable, so this stands in for "nothing to dispose here".</summary>
	private sealed class NoopDisposable : IDisposable
	{
		public static readonly NoopDisposable Instance = new();

		public void Dispose()
		{
		}
	}

	/// <summary>Python: <c>normalize_color</c>'s int-list reading, reused for the flags-style
	/// <c>--face-debugger-items</c>/<c>--face-mask-types</c>/<c>--face-mask-areas</c>/
	/// <c>--face-mask-regions</c> multi-value string flags — ORs every parsed wire name into one
	/// flags value (<see cref="FaceDebuggerItem"/> is the only one of these that is itself a
	/// <see cref="FlagsAttribute"/> enum; the others come back as a plain list).</summary>
	private static FaceDebuggerItem ReadFaceDebuggerItems(IReadOnlyDictionary<string, object?> args)
	{
		var names = StepArgsReader.GetStringList(args, "face_debugger_items", new[] { "face-landmark-5/68", "face-mask" });
		var items = FaceDebuggerItem.None;

		foreach (var name in names)
		{
			items |= EnumNames.FromWireName<FaceDebuggerItem>(name);
		}

		return items;
	}

	private static IReadOnlyList<T> ReadEnumList<T>(IReadOnlyDictionary<string, object?> args, string key, IReadOnlyList<string> fallback) where T : struct, Enum
		=> StepArgsReader.GetStringList(args, key, fallback).Select(EnumNames.FromWireName<T>).ToArray();

	private static BuiltStep BuildFaceDebugger(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var faceDebuggerItems = ReadFaceDebuggerItems(args);
		var faceSelectorMode = EnumNames.FromWireName<FaceSelectorMode>(StepArgsReader.GetString(args, "face_selector_mode", "reference"));
		var faceTrackerScore = StepArgsReader.GetDouble(args, "face_tracker_score", 0.0);
		var faceSelectorOrder = EnumNames.FromWireName<FaceSelectorOrder>(StepArgsReader.GetString(args, "face_selector_order", "large-small"));
		var faceSelectorGenderName = StepArgsReader.GetStringOrNull(args, "face_selector_gender");
		var faceSelectorGender = faceSelectorGenderName is null ? (FaceSelectorGender?)null : EnumNames.FromWireName<FaceSelectorGender>(faceSelectorGenderName);
		var faceSelectorRaceName = StepArgsReader.GetStringOrNull(args, "face_selector_race");
		var faceSelectorRace = faceSelectorRaceName is null ? (FaceSelectorRace?)null : EnumNames.FromWireName<FaceSelectorRace>(faceSelectorRaceName);
		var faceSelectorAgeStart = StepArgsReader.GetIntOrNull(args, "face_selector_age_start");
		var faceSelectorAgeEnd = StepArgsReader.GetIntOrNull(args, "face_selector_age_end");
		var referenceFacePosition = StepArgsReader.GetInt(args, "reference_face_position", 0);
		var referenceFaceDistance = StepArgsReader.GetDouble(args, "reference_face_distance", 0.3);
		var faceMaskTypes = ReadEnumList<FaceMaskType>(args, "face_mask_types", new[] { "box" });
		var faceMaskPaddingValues = StepArgsReader.GetIntList(args, "face_mask_padding", new[] { 0, 0, 0, 0 });
		var faceMaskPadding = new Padding(faceMaskPaddingValues[0], faceMaskPaddingValues[1], faceMaskPaddingValues[2], faceMaskPaddingValues[3]);
		var faceMaskAreas = ReadEnumList<FaceMaskArea>(args, "face_mask_areas", new[] { "upper-face", "lower-face", "mouth" });
		var faceMaskRegions = ReadEnumList<FaceMaskRegion>(args, "face_mask_regions", new[]
		{
			"skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "glasses", "nose", "mouth", "upper-lip", "lower-lip",
		});
		var faceOccluderModel = EnumNames.FromWireName<FaceOccluderModel>(StepArgsReader.GetString(args, "face_occluder_model", "xseg_1"));
		var faceParserModel = EnumNames.FromWireName<FaceParserModel>(StepArgsReader.GetString(args, "face_parser_model", "bisenet_resnet_34"));

		// Occlusion/region masks need their own ONNX sessions (FaceMasker.CreateOcclusionMask/
		// CreateRegionMask) — only opened when actually requested, matching face_debugger's
		// default face_mask_types = ['box'] (no masker session needed at all).
		IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null;
		IReadOnlyDictionary<string, InferenceSession>? parserInferencePool = null;
		var modelsDirectory = HeadlessRunner.ResolveModelsDirectory();
		var extraSessions = new List<InferenceSession>();

		if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
		{
			var occluderFileName = faceOccluderModel.ToWireName();
			var occluderSession = new InferenceSession(Path.Combine(modelsDirectory, occluderFileName + ".onnx"));
			extraSessions.Add(occluderSession);
			occluderInferencePool = new Dictionary<string, InferenceSession> { [occluderFileName] = occluderSession };
		}

		if (faceMaskTypes.Contains(FaceMaskType.Region))
		{
			var parserFileName = faceParserModel.ToWireName();
			var parserSession = new InferenceSession(Path.Combine(modelsDirectory, parserFileName + ".onnx"));
			extraSessions.Add(parserSession);
			parserInferencePool = new Dictionary<string, InferenceSession> { [parserFileName] = parserSession };
		}

		var processor = new FaceDebugger.Processor();

		var step = new WorkflowProcessorStep(processor, context => new FaceDebugger.FaceDebuggerInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			faceDebuggerItems,
			faceSelectorMode,
			faceTrackerScore,
			faceSelectorOrder,
			faceSelectorGender,
			faceSelectorRace,
			faceSelectorAgeStart,
			faceSelectorAgeEnd,
			referenceFacePosition,
			referenceFaceDistance,
			faceMaskTypes,
			faceMaskPadding,
			faceMaskAreas,
			faceMaskRegions,
			faceOccluderModel,
			faceParserModel,
			occluderInferencePool,
			parserInferencePool,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		var resource = extraSessions.Count == 0
			? (IDisposable)NoopDisposable.Instance
			: new CompositeDisposable(extraSessions);

		return new BuiltStep(step, resource);
	}

	/// <summary>Disposes a fixed set of <see cref="InferenceSession"/>s together — used for a
	/// processor (like <see cref="BuildFaceDebugger"/>'s optional occluder/parser sessions) that
	/// opens more than one session not already owned by <see cref="FacePipelineFactory.Resources"/>.</summary>
	private sealed class CompositeDisposable : IDisposable
	{
		private readonly IReadOnlyList<IDisposable> _disposables;

		public CompositeDisposable(IReadOnlyList<IDisposable> disposables) => _disposables = disposables;

		public void Dispose()
		{
			foreach (var disposable in _disposables)
			{
				disposable.Dispose();
			}
		}
	}

}
