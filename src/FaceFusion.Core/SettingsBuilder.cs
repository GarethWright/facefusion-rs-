using System.Collections.Generic;
using System.IO;
using System.Linq;
using FaceFusion.Types;

namespace FaceFusion.Core;

/// <summary>
/// Ported from the default-value expressions in facefusion/program.py (the
/// <c>ArgumentParser.add_argument(..., default = config.get_xxx_value(section, option,
/// hardcoded_fallback))</c> calls), NOT from argparse itself. program.py mixes three things
/// per argument: an ini-backed default, a CLI flag, and (for a handful of options) a value
/// computed from the runtime environment (installed ffmpeg encoders, available ONNX Runtime
/// execution providers, processor/UI-layout files present on disk). Only the first of those
/// three is this port's job — Phase 6 owns the CLI flags (`System.CommandLine`), and the
/// environment-probing defaults belong to FaceFusion.Media / FaceFusion.Inference, which do
/// not exist yet in this phase.
///
/// <see cref="Build"/> therefore produces the "ini value, falling back to program.py's
/// hardcoded literal" layer only, as a plain <see cref="FaceFusion.Types.State"/>. Two kinds
/// of field do not get a faithful default here and are called out below and at each site with
/// a "PLACEHOLDER" comment:
///
/// 1. **Environment-probed defaults** (output_audio_encoder, output_video_encoder,
///    execution_providers) — Python computes these from `get_available_encoder_set()` /
///    `get_available_execution_providers()` at parse time. This builder uses a fixed,
///    deterministic stand-in (the first declared enum member) instead, and a later phase
///    that owns those subsystems should overwrite the field with `state with { ... }` once it
///    has a real answer — that `with` expression is the "seam".
/// 2. **Fields with no Python default at all** (None): e.g. `target_path`,
///    `face_selector_age_start`, `output_video_fps`, and the three CLI-positional-only fields
///    `job_id` / `job_status` / `step_index`. Python represents "unset" as `None`; every field
///    in the current <see cref="State"/> record is non-nullable, so there is no exact
///    equivalent to fall back to here. This builder uses each type's natural zero value
///    (`""`, `0`, `0.0`, empty list, first enum member) and documents it inline. This is a
///    pre-existing shape gap in <see cref="State"/> (built by an earlier agent, out of this
///    assignment's scope to change) rather than a decision made by this file — see the report
///    for the recommendation to widen the affected fields to nullable in a follow-up.
///
/// CLI args are Phase 6: apply them, once parsed, as `with`-expressions layered on top of the
/// <see cref="State"/> this method returns — mirroring how Python's `args.py::apply_args`
/// layers `args.get(...)` over whatever `program.py` already put in `default=`.
/// </summary>
public static class SettingsBuilder
{
	public static State Build(Config config)
	{
		var faceDetectorModel = ParseEnum<FaceDetectorModel>(
			config.GetStrValue("face_detector", "face_detector_model", "yolo_face")!);

		// face_detector_size's hardcoded fallback is `get_last(face_detector_set[face_detector_model])`
		// in Python, i.e. it depends on the (possibly ini-overridden) face_detector_model chosen
		// just above, not a fixed literal like the other options.
		var faceDetectorSizeChoices = Choices.FaceDetectorSet[faceDetectorModel];
		var faceDetectorSizeFallback = faceDetectorSizeChoices[faceDetectorSizeChoices.Count - 1];

		return new State(
			Command: string.Empty, // PLACEHOLDER: CLI subcommand, Phase 6 seam.
			ConfigPath: "facefusion.ini", // Python default in create_config_path_program; not itself ini-driven.
			TempPath: config.GetStrValue("paths", "temp_path", Path.GetTempPath())!,
			JobsPath: config.GetStrValue("paths", "jobs_path", ".jobs")!,
			SourcePaths: config.GetStrList("paths", "source_paths") ?? System.Array.Empty<string>(), // Python default: None.
			TargetPath: config.GetStrValue("paths", "target_path") ?? string.Empty, // Python default: None.
			OutputPath: config.GetStrValue("paths", "output_path") ?? string.Empty, // Python default: None.
			SourcePattern: config.GetStrValue("patterns", "source_pattern") ?? string.Empty, // Python default: None.
			TargetPattern: config.GetStrValue("patterns", "target_pattern") ?? string.Empty, // Python default: None.
			OutputPattern: config.GetStrValue("patterns", "output_pattern") ?? string.Empty, // Python default: None.
			DownloadProviders: ParseEnumList<DownloadProvider>(
				config.GetStrList("download", "download_providers", string.Join(' ', EnumNames.AllWireNames<DownloadProvider>()))!),
			DownloadScope: ParseEnum<DownloadScope>(config.GetStrValue("download", "download_scope", "lite")!),
			BenchmarkMode: ParseEnum<BenchmarkMode>(config.GetStrValue("benchmark", "benchmark_mode", "warm")!),
			BenchmarkResolutions: ParseEnumList<BenchmarkResolution>(
				config.GetStrList("benchmark", "benchmark_resolutions", EnumNames.AllWireNames<BenchmarkResolution>()[0])!),
			BenchmarkCycleCount: config.GetIntValue("benchmark", "benchmark_cycle_count", "5")!.Value,
			FaceDetectorModel: faceDetectorModel,
			FaceDetectorSize: config.GetStrValue("face_detector", "face_detector_size", faceDetectorSizeFallback)!,
			FaceDetectorMargin: ToMargin(config.GetIntList("face_detector", "face_detector_margin", "0 0 0 0")!),
			FaceDetectorAngles: config.GetIntList("face_detector", "face_detector_angles", "0")!,
			FaceDetectorScore: config.GetFloatValue("face_detector", "face_detector_score", "0.5")!.Value,
			FaceLandmarkerModel: ParseEnum<FaceLandmarkerModel>(config.GetStrValue("face_landmarker", "face_landmarker_model", "2dfan4")!),
			FaceLandmarkerScore: config.GetFloatValue("face_landmarker", "face_landmarker_score", "0.5")!.Value,
			FaceSelectorMode: ParseEnum<FaceSelectorMode>(config.GetStrValue("face_selector", "face_selector_mode", "reference")!),
			FaceSelectorOrder: ParseEnum<FaceSelectorOrder>(config.GetStrValue("face_selector", "face_selector_order", "large-small")!),
			FaceSelectorRace: ParseEnumOrFirst<FaceSelectorRace>(config.GetStrValue("face_selector", "face_selector_race")), // Python default: None.
			FaceSelectorGender: ParseEnumOrFirst<FaceSelectorGender>(config.GetStrValue("face_selector", "face_selector_gender")), // Python default: None.
			FaceSelectorAgeStart: config.GetIntValue("face_selector", "face_selector_age_start") ?? 0, // Python default: None.
			FaceSelectorAgeEnd: config.GetIntValue("face_selector", "face_selector_age_end") ?? 0, // Python default: None.
			ReferenceFacePosition: config.GetIntValue("face_selector", "reference_face_position", "0")!.Value,
			ReferenceFaceDistance: config.GetFloatValue("face_selector", "reference_face_distance", "0.3")!.Value,
			ReferenceFrameNumber: config.GetIntValue("face_selector", "reference_frame_number", "0")!.Value,
			FaceTrackerScore: config.GetFloatValue("face_tracker", "face_tracker_score", "0.0")!.Value,
			FaceOccluderModel: ParseEnum<FaceOccluderModel>(config.GetStrValue("face_masker", "face_occluder_model", "xseg_1")!),
			FaceParserModel: ParseEnum<FaceParserModel>(config.GetStrValue("face_masker", "face_parser_model", "bisenet_resnet_34")!),
			FaceMaskTypes: ParseEnumList<FaceMaskType>(config.GetStrList("face_masker", "face_mask_types", "box")!),
			FaceMaskAreas: ParseEnumList<FaceMaskArea>(
				config.GetStrList("face_masker", "face_mask_areas", string.Join(' ', EnumNames.AllWireNames<FaceMaskArea>()))!),
			FaceMaskRegions: ParseEnumList<FaceMaskRegion>(
				config.GetStrList("face_masker", "face_mask_regions", string.Join(' ', EnumNames.AllWireNames<FaceMaskRegion>()))!),
			FaceMaskBlur: config.GetFloatValue("face_masker", "face_mask_blur", "0.3")!.Value,
			FaceMaskPadding: ToPadding(config.GetIntList("face_masker", "face_mask_padding", "0 0 0 0")!),
			VoiceExtractorModel: ParseEnum<VoiceExtractorModel>(config.GetStrValue("voice_extractor", "voice_extractor_model", "kim_vocal_2")!),
			TrimFrameStart: config.GetIntValue("frame_extraction", "trim_frame_start") ?? 0, // Python default: None.
			TrimFrameEnd: config.GetIntValue("frame_extraction", "trim_frame_end") ?? 0, // Python default: None.
			TempFrameFormat: ParseEnum<TempFrameFormat>(config.GetStrValue("frame_extraction", "temp_frame_format", "png")!),
			TempPixelFormat: ParseEnum<TempPixelFormat>(config.GetStrValue("frame_extraction", "temp_pixel_format", "bgr24")!),
			TargetFrameAmount: config.GetIntValue("frame_distribution", "target_frame_amount", "2")!.Value,
			OutputImageQuality: config.GetIntValue("output_creation", "output_image_quality", "80")!.Value,
			OutputImageScale: config.GetFloatValue("output_creation", "output_image_scale", "1.0")!.Value,
			// PLACEHOLDER: Python default is get_first(get_available_encoder_set()['audio']),
			// probed via ffmpeg (FaceFusion.Media, not available in this phase). Falls back to
			// the ini value only; the environment-probed default is the first declared enum
			// member, to be overwritten by the Media layer's real detection.
			OutputAudioEncoder: ParseEnumOrFirst<AudioEncoder>(config.GetStrValue("output_creation", "output_audio_encoder")),
			OutputAudioQuality: config.GetIntValue("output_creation", "output_audio_quality", "80")!.Value,
			OutputAudioVolume: config.GetIntValue("output_creation", "output_audio_volume", "100")!.Value,
			// PLACEHOLDER: same as OutputAudioEncoder, for get_available_encoder_set()['video'].
			OutputVideoEncoder: ParseEnumOrFirst<VideoEncoder>(config.GetStrValue("output_creation", "output_video_encoder")),
			OutputVideoPreset: ParseEnum<VideoPreset>(config.GetStrValue("output_creation", "output_video_preset", "veryfast")!),
			OutputVideoQuality: config.GetIntValue("output_creation", "output_video_quality", "80")!.Value,
			OutputVideoScale: config.GetFloatValue("output_creation", "output_video_scale", "1.0")!.Value,
			OutputVideoFps: config.GetFloatValue("output_creation", "output_video_fps") ?? 0.0, // Python default: None (falls back to detect_video_fps at use-site).
			WorkflowMode: ParseEnum<WorkflowMode>(config.GetStrValue("workflow", "workflow_mode", "auto")!),
			WorkflowStrategy: ParseEnum<WorkflowStrategy>(config.GetStrValue("workflow", "workflow_strategy", "memory")!),
			Processors: config.GetStrList("processors", "processors", "face_swapper")!,
			OpenBrowser: config.GetBoolValue("uis", "open_browser") ?? false, // Python default: None (store_true).
			UiLayouts: config.GetStrList("uis", "ui_layouts", "default")!,
			UiWorkflow: ParseEnum<UiWorkflow>(config.GetStrValue("uis", "ui_workflow", "instant_runner")!),
			ExecutionDeviceIds: config.GetIntList("execution", "execution_device_ids", "0")!,
			// PLACEHOLDER: Python default is get_first(get_available_execution_providers()),
			// probed via ONNX Runtime (FaceFusion.Inference, not available in this phase).
			ExecutionProviders: ParseEnumList<ExecutionProvider>(
				config.GetStrList("execution", "execution_providers", EnumNames.AllWireNames<ExecutionProvider>()[^1])!), // last = "cpu": always available.
			ExecutionThreadCount: config.GetIntValue("execution", "execution_thread_count", "8")!.Value,
			VideoMemoryStrategy: ParseEnum<VideoMemoryStrategy>(config.GetStrValue("memory", "video_memory_strategy", "strict")!),
			LogLevel: ParseEnum<LogLevel>(config.GetStrValue("misc", "log_level", "info")!),
			HaltOnError: config.GetBoolValue("misc", "halt_on_error") ?? false, // Python default: None (store_true).
			JobId: string.Empty, // PLACEHOLDER: CLI positional argument, Phase 6 seam.
			JobStatus: default, // PLACEHOLDER: CLI positional argument, Phase 6 seam.
			StepIndex: 0); // PLACEHOLDER: CLI positional argument, Phase 6 seam.
	}

	private static Margin ToMargin(IReadOnlyList<int> values)
	{
		return new Margin(values[0], values[1], values[2], values[3]);
	}

	private static Padding ToPadding(IReadOnlyList<int> values)
	{
		return new Padding(values[0], values[1], values[2], values[3]);
	}

	private static T ParseEnum<T>(string wireName) where T : struct, System.Enum
	{
		return EnumNames.FromWireName<T>(wireName);
	}

	/// <summary>
	/// Parses a wire name if present, otherwise returns the first declared enum member. Used
	/// for fields whose Python default is <c>None</c> but whose <see cref="State"/> field is a
	/// non-nullable enum — see the "known gaps" note on <see cref="Build"/>.
	/// </summary>
	private static T ParseEnumOrFirst<T>(string? wireName) where T : struct, System.Enum
	{
		if (wireName != null)
		{
			return EnumNames.FromWireName<T>(wireName);
		}
		return default;
	}

	private static IReadOnlyList<T> ParseEnumList<T>(IReadOnlyList<string> wireNames) where T : struct, System.Enum
	{
		return wireNames.Select(EnumNames.FromWireName<T>).ToArray();
	}
}
