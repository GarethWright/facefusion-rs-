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
/// <b>Coverage.</b> All eleven processors are wired. Ten are verified the way this phase
/// demands — both CLIs run on the same 8-frame clip and the output pixels compared, every one
/// landing in the 42–43 dB band that two independent libx264 encodes of identical pixels
/// produce (see docs/IMPLEMENTATION_STATUS.md for the per-processor table).
/// </para>
///
/// <para>
/// <b>The one exception is <c>deep_swapper</c>,</b> whose <c>.dfm</c> models live on
/// huggingface.co — blocked by this environment's proxy with a 403, so neither implementation
/// can run it here to be compared. Both refuse the run rather than emitting wrong output
/// (Python fails its hash validation after attempting a download; this port fails its
/// file-presence pre-check, since <c>download.py</c> is deliberately not ported), which is
/// what has actually been checked.
/// </para>
///
/// <para>
/// Two defects were found by running the binary rather than by the test suite, and both are
/// fixed: <c>PixelBoost.ExplodePixelBoost</c> hard-asserted <c>CV_8UC3</c> while
/// <c>NormalizeCropFrame</c> deliberately hands it float Mats (so <c>face_swapper</c>'s
/// assembled pipeline had never once executed), and <c>HeadlessRunner.BuildRunContext</c>
/// supplied an <c>ExtractVoice</c> delegate that threw unconditionally, which made
/// <c>lip_syncer</c> — any processor reading source audio — impossible to run.
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
			// face_swapper needs its own model AND the shared face pipeline, since it
			// resolves both a source face and the target faces before swapping.
			"face_swapper" => FaceSwapper.PreCheck(ReadFaceSwapperModel(args)) && FacePipelineFactory.PreCheck(args),
			"age_modifier" => AgeModifier.PreCheck(ReadAgeModifierModel(args)) && FacePipelineFactory.PreCheck(args),
			"expression_restorer" => ExpressionRestorer.PreCheck() && FacePipelineFactory.PreCheck(args),
			"face_editor" => FaceEditor.PreCheck() && FacePipelineFactory.PreCheck(args),
			"deep_swapper" => DeepSwapper.PreCheck(ReadDeepSwapperModel(args)) && FacePipelineFactory.PreCheck(args),
			"lip_syncer" => LipSyncer.PreCheck(ReadLipSyncerModel(args)) && FacePipelineFactory.PreCheck(args),
			"frame_enhancer" => FrameEnhancer.PreCheck(ReadFrameEnhancerModel(args)),
			"face_enhancer" => FaceEnhancer.PreCheck(ReadFaceEnhancerModel(args)) && FacePipelineFactory.PreCheck(args),
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
			"face_swapper" => BuildFaceSwapper(args, faceResources ?? throw MissingFaceResources(processorName)),
			"age_modifier" => BuildAgeModifier(args, faceResources ?? throw MissingFaceResources(processorName)),
			"expression_restorer" => BuildExpressionRestorer(args, faceResources ?? throw MissingFaceResources(processorName)),
			"face_editor" => BuildFaceEditor(args, faceResources ?? throw MissingFaceResources(processorName)),
			"deep_swapper" => BuildDeepSwapper(args, faceResources ?? throw MissingFaceResources(processorName)),
			"lip_syncer" => BuildLipSyncer(args, faceResources ?? throw MissingFaceResources(processorName)),
			"frame_enhancer" => BuildFrameEnhancer(args),
			"face_enhancer" => BuildFaceEnhancer(args, faceResources ?? throw MissingFaceResources(processorName)),
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


	// -----------------------------------------------------------------
	// face_swapper
	// -----------------------------------------------------------------

	// Defaults taken from face_swapper/core.py's own register_args, NOT guessed: the model
	// defaults to hyperswap_1a_256 and the weight to 0.5. An earlier version of this file
	// used inswapper_128 and 1.0, which made a comparison against the Python CLI compare
	// two different models and look like a 33 dB parity failure.
	private static FaceSwapperModel ReadFaceSwapperModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<FaceSwapperModel>(StepArgsReader.GetString(args, "face_swapper_model", "hyperswap_1a_256"));

	private static BuiltStep BuildFaceSwapper(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var model = ReadFaceSwapperModel(args);
		var options = FaceSwapper.CreateStaticModelSet(DownloadScope.Full)[model];
		// Python: default = get_first(face_swapper_pixel_boost_choices), i.e. the first entry
		// of the CHOSEN model's own list, not a fixed literal.
		var pixelBoostChoices = FaceSwapper.FaceSwapperPixelBoostChoices[model];
		var pixelBoost = StepArgsReader.GetString(args, "face_swapper_pixel_boost", pixelBoostChoices[0]);
		var weight = StepArgsReader.GetDouble(args, "face_swapper_weight", 0.5);

		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);

		var sessions = new List<InferenceSession>();
		var faceSwapperSession = new InferenceSession(options.Sources["face_swapper"].Path);
		sessions.Add(faceSwapperSession);

		// Only the ghost/hyperswap families ship an embedding converter; the rest have no
		// such source and Python simply never looks one up.
		InferenceSession? embeddingConverterSession = null;

		if (options.Sources.TryGetValue("embedding_converter", out var embeddingConverter))
		{
			embeddingConverterSession = new InferenceSession(embeddingConverter.Path);
			sessions.Add(embeddingConverterSession);
		}

		// Python: get_static_model_initializer(model_path) — inswapper's swap maths needs the
		// model's last graph initializer as a matrix. Only the inswapper family reads it.
		var inswapperInitializer = options.Type == FaceSwapperModelKind.Inswapper
			? FaceFusion.Inference.ModelHelper.GetStaticModelInitializer(options.Sources["face_swapper"].Path)
			: null;

		var processor = new FaceSwapper.Processor();

		var step = new WorkflowProcessorStep(processor, context => new FaceSwapper.FaceSwapperInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			model,
			pixelBoost,
			weight,
			masks.Types,
			masks.Blur,
			masks.Padding,
			faceSwapperSession,
			embeddingConverterSession,
			inswapperInitializer,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, new CompositeDisposable(sessions));
	}

	/// <summary>The face-selector settings every face-pipeline processor reads, in one place
	/// rather than repeated per processor.</summary>
	private sealed record FaceSelectorSettings(
		FaceSelectorMode Mode,
		double TrackerScore,
		FaceSelectorOrder Order,
		FaceSelectorGender? Gender,
		FaceSelectorRace? Race,
		int? AgeStart,
		int? AgeEnd,
		int ReferenceFacePosition,
		double ReferenceFaceDistance);

	private static FaceSelectorSettings ReadFaceSelectorSettings(IReadOnlyDictionary<string, object?> args)
	{
		var genderName = StepArgsReader.GetStringOrNull(args, "face_selector_gender");
		var raceName = StepArgsReader.GetStringOrNull(args, "face_selector_race");

		return new FaceSelectorSettings(
			EnumNames.FromWireName<FaceSelectorMode>(StepArgsReader.GetString(args, "face_selector_mode", "reference")),
			StepArgsReader.GetDouble(args, "face_tracker_score", 0.0),
			EnumNames.FromWireName<FaceSelectorOrder>(StepArgsReader.GetString(args, "face_selector_order", "large-small")),
			genderName is null ? null : EnumNames.FromWireName<FaceSelectorGender>(genderName),
			raceName is null ? null : EnumNames.FromWireName<FaceSelectorRace>(raceName),
			StepArgsReader.GetIntOrNull(args, "face_selector_age_start"),
			StepArgsReader.GetIntOrNull(args, "face_selector_age_end"),
			StepArgsReader.GetInt(args, "reference_face_position", 0),
			StepArgsReader.GetDouble(args, "reference_face_distance", 0.3));
	}

	private sealed record FaceMaskSettings(IReadOnlyList<FaceMaskType> Types, double Blur, Padding Padding);

	private static FaceMaskSettings ReadFaceMaskSettings(IReadOnlyDictionary<string, object?> args)
	{
		var paddingValues = StepArgsReader.GetIntList(args, "face_mask_padding", new[] { 0, 0, 0, 0 });

		return new FaceMaskSettings(
			ReadEnumList<FaceMaskType>(args, "face_mask_types", new[] { "box" }),
			StepArgsReader.GetDouble(args, "face_mask_blur", 0.3),
			new Padding(paddingValues[0], paddingValues[1], paddingValues[2], paddingValues[3]));
	}


	// -----------------------------------------------------------------
	// age_modifier
	// -----------------------------------------------------------------

	// Defaults from age_modifier/core.py's register_args: model 'fran', direction 0.
	private static AgeModifierModel ReadAgeModifierModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<AgeModifierModel>(StepArgsReader.GetString(args, "age_modifier_model", "fran"));

	private static BuiltStep BuildAgeModifier(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var model = ReadAgeModifierModel(args);
		var options = AgeModifier.CreateStaticModelSet(DownloadScope.Full)[model];
		var direction = StepArgsReader.GetInt(args, "age_modifier_direction", 0);
		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);

		var session = new InferenceSession(options.Sources["age_modifier"].Path);
		var processor = new AgeModifier.Processor();

		var step = new WorkflowProcessorStep(processor, context => new AgeModifier.AgeModifierInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			model,
			direction,
			masks.Types,
			masks.Blur,
			session,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, session);
	}


	// -----------------------------------------------------------------
	// Shared: the optional face-masker sessions
	// -----------------------------------------------------------------

	/// <summary>The occluder/parser inference pools <c>FaceMasker.CreateOcclusionMask</c> and
	/// <c>CreateRegionMask</c> need, opened only when the run's <c>--face-mask-types</c> actually
	/// asks for them — matching Python, where <c>face_masker.get_inference_pool()</c> is reached
	/// only from inside the corresponding <c>create_*_mask</c> call. The returned sessions are
	/// owned by the caller and belong in the step's <see cref="BuiltStep.Resource"/>.</summary>
	private sealed record MaskerSessions(
		IReadOnlyDictionary<string, InferenceSession>? OccluderPool,
		IReadOnlyDictionary<string, InferenceSession>? ParserPool,
		IReadOnlyList<InferenceSession> Sessions);

	private static MaskerSessions BuildMaskerSessions(IReadOnlyList<FaceMaskType> faceMaskTypes, FaceOccluderModel occluderModel, FaceParserModel parserModel)
	{
		IReadOnlyDictionary<string, InferenceSession>? occluderPool = null;
		IReadOnlyDictionary<string, InferenceSession>? parserPool = null;
		var modelsDirectory = HeadlessRunner.ResolveModelsDirectory();
		var sessions = new List<InferenceSession>();

		if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
		{
			var occluderFileName = occluderModel.ToWireName();
			var occluderSession = new InferenceSession(Path.Combine(modelsDirectory, occluderFileName + ".onnx"));
			sessions.Add(occluderSession);
			occluderPool = new Dictionary<string, InferenceSession> { [occluderFileName] = occluderSession };
		}

		if (faceMaskTypes.Contains(FaceMaskType.Region))
		{
			var parserFileName = parserModel.ToWireName();
			var parserSession = new InferenceSession(Path.Combine(modelsDirectory, parserFileName + ".onnx"));
			sessions.Add(parserSession);
			parserPool = new Dictionary<string, InferenceSession> { [parserFileName] = parserSession };
		}

		return new MaskerSessions(occluderPool, parserPool, sessions);
	}

	private static FaceOccluderModel ReadFaceOccluderModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<FaceOccluderModel>(StepArgsReader.GetString(args, "face_occluder_model", "xseg_1"));

	private static FaceParserModel ReadFaceParserModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<FaceParserModel>(StepArgsReader.GetString(args, "face_parser_model", "bisenet_resnet_34"));

	// -----------------------------------------------------------------
	// expression_restorer
	// -----------------------------------------------------------------

	// Defaults from expression_restorer/core.py's register_args: model 'live_portrait',
	// factor 80, areas = every ExpressionRestorerArea (Python: ' '.join(choices)).
	private static BuiltStep BuildExpressionRestorer(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var options = ExpressionRestorer.CreateStaticModelSet(DownloadScope.Full)[ExpressionRestorerModel.LivePortrait];
		var factor = StepArgsReader.GetInt(args, "expression_restorer_factor", 80);
		var areas = ReadEnumList<ExpressionRestorerArea>(args, "expression_restorer_areas", new[] { "upper-face", "lower-face" });
		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);

		var featureExtractorSession = new InferenceSession(options.Sources["feature_extractor"].Path);
		var motionExtractorSession = new InferenceSession(options.Sources["motion_extractor"].Path);
		var generatorSession = new InferenceSession(options.Sources["generator"].Path);
		var sessions = new List<InferenceSession> { featureExtractorSession, motionExtractorSession, generatorSession };

		var processor = new ExpressionRestorer.Processor();

		var step = new WorkflowProcessorStep(processor, context => new ExpressionRestorer.ExpressionRestorerInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			factor,
			areas,
			masks.Types,
			masks.Blur,
			featureExtractorSession,
			motionExtractorSession,
			generatorSession,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, new CompositeDisposable(sessions));
	}

	// -----------------------------------------------------------------
	// face_editor
	// -----------------------------------------------------------------

	/// <summary>Python: face_editor/core.py's register_args — fourteen float sliders, every one
	/// defaulting to 0 (i.e. "no edit"), so a bare <c>--processors face_editor</c> run is a
	/// no-op pass-through in Python too.</summary>
	private static FaceEditor.FaceEditorSliders ReadFaceEditorSliders(IReadOnlyDictionary<string, object?> args)
		=> new(
			StepArgsReader.GetDouble(args, "face_editor_eyebrow_direction", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_eye_gaze_horizontal", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_eye_gaze_vertical", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_eye_open_ratio", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_lip_open_ratio", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_mouth_grim", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_mouth_pout", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_mouth_purse", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_mouth_smile", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_mouth_position_horizontal", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_mouth_position_vertical", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_head_pitch", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_head_yaw", 0.0),
			StepArgsReader.GetDouble(args, "face_editor_head_roll", 0.0));

	private static BuiltStep BuildFaceEditor(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var options = FaceEditor.CreateStaticModelSet(DownloadScope.Full)[FaceEditorModel.LivePortrait];
		var sliders = ReadFaceEditorSliders(args);
		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);

		var featureExtractorSession = new InferenceSession(options.Sources["feature_extractor"].Path);
		var motionExtractorSession = new InferenceSession(options.Sources["motion_extractor"].Path);
		var eyeRetargeterSession = new InferenceSession(options.Sources["eye_retargeter"].Path);
		var lipRetargeterSession = new InferenceSession(options.Sources["lip_retargeter"].Path);
		var stitcherSession = new InferenceSession(options.Sources["stitcher"].Path);
		var generatorSession = new InferenceSession(options.Sources["generator"].Path);
		var sessions = new List<InferenceSession>
		{
			featureExtractorSession, motionExtractorSession, eyeRetargeterSession,
			lipRetargeterSession, stitcherSession, generatorSession,
		};

		var processor = new FaceEditor.Processor();

		var step = new WorkflowProcessorStep(processor, context => new FaceEditor.FaceEditorInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			sliders,
			masks.Blur,
			featureExtractorSession,
			motionExtractorSession,
			eyeRetargeterSession,
			lipRetargeterSession,
			stitcherSession,
			generatorSession,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, new CompositeDisposable(sessions));
	}

	// -----------------------------------------------------------------
	// deep_swapper
	// -----------------------------------------------------------------

	// Defaults from deep_swapper/core.py's register_args: model 'iperov/elon_musk_224',
	// morph 100. The model key is a "scope/name" string, not an enum (the catalog is built at
	// runtime and can include user-supplied models) — see DeepSwapper's class remarks.
	private static string ReadDeepSwapperModel(IReadOnlyDictionary<string, object?> args)
		=> StepArgsReader.GetString(args, "deep_swapper_model", "iperov/elon_musk_224");

	private static BuiltStep BuildDeepSwapper(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var model = ReadDeepSwapperModel(args);
		var options = DeepSwapper.CreateStaticModelSet(DownloadScope.Full)[model];
		var morph = StepArgsReader.GetInt(args, "deep_swapper_morph", 100);
		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);

		var session = new InferenceSession(options.Sources["deep_swapper"].Path);
		var processor = new DeepSwapper.Processor();

		var step = new WorkflowProcessorStep(processor, context => new DeepSwapper.DeepSwapperInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			model,
			morph,
			session,
			masks.Types,
			masks.Blur,
			masks.Padding,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, session);
	}

	// -----------------------------------------------------------------
	// lip_syncer
	// -----------------------------------------------------------------

	// Defaults from lip_syncer/core.py's register_args: model 'wav2lip_gan_96', weight 0.5.
	private static LipSyncerModel ReadLipSyncerModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<LipSyncerModel>(StepArgsReader.GetString(args, "lip_syncer_model", "wav2lip_gan_96"));

	private static BuiltStep BuildLipSyncer(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var model = ReadLipSyncerModel(args);
		var options = LipSyncer.CreateStaticModelSet(DownloadScope.Full)[model];
		var weight = StepArgsReader.GetDouble(args, "lip_syncer_weight", 0.5);
		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);
		var occluderModel = ReadFaceOccluderModel(args);
		var maskerSessions = BuildMaskerSessions(masks.Types, occluderModel, ReadFaceParserModel(args));

		var lipSyncerSession = new InferenceSession(options.Sources["lip_syncer"].Path);
		var sessions = new List<InferenceSession> { lipSyncerSession };
		sessions.AddRange(maskerSessions.Sessions);

		var processor = new LipSyncer.Processor();

		var step = new WorkflowProcessorStep(processor, context => new LipSyncer.LipSyncerInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.SourceVoiceFrame,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			model,
			weight,
			masks.Types,
			masks.Blur,
			masks.Padding,
			lipSyncerSession,
			maskerSessions.OccluderPool,
			occluderModel,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, new CompositeDisposable(sessions));
	}


	// -----------------------------------------------------------------
	// frame_enhancer
	// -----------------------------------------------------------------

	// Defaults from frame_enhancer/core.py's register_args: model 'span_kendata_x4', blend 80.
	private static FrameEnhancerModel ReadFrameEnhancerModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<FrameEnhancerModel>(StepArgsReader.GetString(args, "frame_enhancer_model", "span_kendata_x4"));

	private static BuiltStep BuildFrameEnhancer(IReadOnlyDictionary<string, object?> args)
	{
		var model = ReadFrameEnhancerModel(args);
		var options = FrameEnhancer.GetModelOptions(model);
		var blend = StepArgsReader.GetInt(args, "frame_enhancer_blend", 80);

		var session = new InferenceSession(options.Source.Path);
		var processor = new FrameEnhancer.Processor();

		var step = new WorkflowProcessorStep(processor, context => new FrameEnhancer.FrameEnhancerInputs(
			context.TempVisionFrame,
			context.TempVisionMask,
			options,
			session,
			blend));

		return new BuiltStep(step, session);
	}

	// -----------------------------------------------------------------
	// face_enhancer
	// -----------------------------------------------------------------

	// Defaults from face_enhancer/core.py's register_args: model 'gfpgan_1.4', blend 80,
	// weight 0.5.
	private static FaceEnhancerModel ReadFaceEnhancerModel(IReadOnlyDictionary<string, object?> args)
		=> EnumNames.FromWireName<FaceEnhancerModel>(StepArgsReader.GetString(args, "face_enhancer_model", "gfpgan_1.4"));

	private static BuiltStep BuildFaceEnhancer(IReadOnlyDictionary<string, object?> args, FacePipelineFactory.Resources faceResources)
	{
		var model = ReadFaceEnhancerModel(args);
		var options = FaceEnhancer.GetModelOptions(model);
		var blend = StepArgsReader.GetInt(args, "face_enhancer_blend", 80);
		var weight = StepArgsReader.GetDouble(args, "face_enhancer_weight", 0.5);
		var selector = ReadFaceSelectorSettings(args);
		var masks = ReadFaceMaskSettings(args);
		var occluderModel = ReadFaceOccluderModel(args);
		var maskerSessions = BuildMaskerSessions(masks.Types, occluderModel, ReadFaceParserModel(args));

		var faceEnhancerSession = new InferenceSession(options.Source.Path);
		var sessions = new List<InferenceSession> { faceEnhancerSession };
		sessions.AddRange(maskerSessions.Sessions);

		// FaceEnhancer.ProcessFrame takes a non-nullable pool (it only reaches into it when
		// face_mask_types includes 'occlusion'); an empty dictionary is the "not requested"
		// case, matching BuildMaskerSessions returning null for it.
		var occluderPool = maskerSessions.OccluderPool ?? new Dictionary<string, InferenceSession>();

		var processor = new FaceEnhancer.Processor();

		var step = new WorkflowProcessorStep(processor, context => new FaceEnhancer.FaceEnhancerInputs(
			context.ReferenceVisionFrame,
			context.SourceVisionFrames,
			context.TargetVisionFrames,
			context.TempVisionFrame,
			context.TempVisionMask,
			options,
			faceEnhancerSession,
			weight,
			blend,
			masks.Types,
			masks.Blur,
			occluderModel,
			occluderPool,
			selector.Mode,
			selector.TrackerScore,
			selector.Order,
			selector.Gender,
			selector.Race,
			selector.AgeStart,
			selector.AgeEnd,
			selector.ReferenceFacePosition,
			selector.ReferenceFaceDistance,
			faceResources.GetStaticFaces,
			faceResources.RefillFaces));

		return new BuiltStep(step, new CompositeDisposable(sessions));
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
