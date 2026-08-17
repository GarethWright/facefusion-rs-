namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>StateKey</c>: the closed set of keys that appear
/// in <see cref="State"/> / <c>facefusion.ini</c> / CLI args. Wire names are the exact
/// snake_case key strings used by state_manager.py, config.py, and job JSON.
/// </summary>
public enum StateKey
{
	[WireName("command")]
	Command,

	[WireName("config_path")]
	ConfigPath,

	[WireName("temp_path")]
	TempPath,

	[WireName("jobs_path")]
	JobsPath,

	[WireName("source_paths")]
	SourcePaths,

	[WireName("target_path")]
	TargetPath,

	[WireName("output_path")]
	OutputPath,

	[WireName("source_pattern")]
	SourcePattern,

	[WireName("target_pattern")]
	TargetPattern,

	[WireName("output_pattern")]
	OutputPattern,

	[WireName("download_providers")]
	DownloadProviders,

	[WireName("download_scope")]
	DownloadScope,

	[WireName("benchmark_mode")]
	BenchmarkMode,

	[WireName("benchmark_resolutions")]
	BenchmarkResolutions,

	[WireName("benchmark_cycle_count")]
	BenchmarkCycleCount,

	[WireName("face_detector_model")]
	FaceDetectorModel,

	[WireName("face_detector_size")]
	FaceDetectorSize,

	[WireName("face_detector_margin")]
	FaceDetectorMargin,

	[WireName("face_detector_angles")]
	FaceDetectorAngles,

	[WireName("face_detector_score")]
	FaceDetectorScore,

	[WireName("face_landmarker_model")]
	FaceLandmarkerModel,

	[WireName("face_landmarker_score")]
	FaceLandmarkerScore,

	[WireName("face_selector_mode")]
	FaceSelectorMode,

	[WireName("face_selector_order")]
	FaceSelectorOrder,

	[WireName("face_selector_gender")]
	FaceSelectorGender,

	[WireName("face_selector_race")]
	FaceSelectorRace,

	[WireName("face_selector_age_start")]
	FaceSelectorAgeStart,

	[WireName("face_selector_age_end")]
	FaceSelectorAgeEnd,

	[WireName("reference_face_position")]
	ReferenceFacePosition,

	[WireName("reference_face_distance")]
	ReferenceFaceDistance,

	[WireName("reference_frame_number")]
	ReferenceFrameNumber,

	[WireName("face_tracker_score")]
	FaceTrackerScore,

	[WireName("face_occluder_model")]
	FaceOccluderModel,

	[WireName("face_parser_model")]
	FaceParserModel,

	[WireName("face_mask_types")]
	FaceMaskTypes,

	[WireName("face_mask_areas")]
	FaceMaskAreas,

	[WireName("face_mask_regions")]
	FaceMaskRegions,

	[WireName("face_mask_blur")]
	FaceMaskBlur,

	[WireName("face_mask_padding")]
	FaceMaskPadding,

	[WireName("voice_extractor_model")]
	VoiceExtractorModel,

	[WireName("trim_frame_start")]
	TrimFrameStart,

	[WireName("trim_frame_end")]
	TrimFrameEnd,

	[WireName("temp_frame_format")]
	TempFrameFormat,

	[WireName("temp_pixel_format")]
	TempPixelFormat,

	[WireName("target_frame_amount")]
	TargetFrameAmount,

	[WireName("output_image_quality")]
	OutputImageQuality,

	[WireName("output_image_scale")]
	OutputImageScale,

	[WireName("output_audio_encoder")]
	OutputAudioEncoder,

	[WireName("output_audio_quality")]
	OutputAudioQuality,

	[WireName("output_audio_volume")]
	OutputAudioVolume,

	[WireName("output_video_encoder")]
	OutputVideoEncoder,

	[WireName("output_video_preset")]
	OutputVideoPreset,

	[WireName("output_video_quality")]
	OutputVideoQuality,

	[WireName("output_video_scale")]
	OutputVideoScale,

	[WireName("output_video_fps")]
	OutputVideoFps,

	[WireName("workflow_mode")]
	WorkflowMode,

	[WireName("workflow_strategy")]
	WorkflowStrategy,

	[WireName("processors")]
	Processors,

	[WireName("open_browser")]
	OpenBrowser,

	[WireName("ui_layouts")]
	UiLayouts,

	[WireName("ui_workflow")]
	UiWorkflow,

	[WireName("execution_device_ids")]
	ExecutionDeviceIds,

	[WireName("execution_providers")]
	ExecutionProviders,

	[WireName("execution_thread_count")]
	ExecutionThreadCount,

	[WireName("video_memory_strategy")]
	VideoMemoryStrategy,

	[WireName("log_level")]
	LogLevel,

	[WireName("halt_on_error")]
	HaltOnError,

	[WireName("job_id")]
	JobId,

	[WireName("job_status")]
	JobStatus,

	[WireName("step_index")]
	StepIndex
}
