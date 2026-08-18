using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>State</c> TypedDict — the full set of values
/// keyed by <see cref="StateKey"/>. Field order matches the Python declaration.
///
/// The TypedDict itself types every field non-optionally, but several of the corresponding
/// <c>facefusion/program.py</c> argparse defaults are <c>config.get_*_value(section, option)</c>
/// calls with no third (fallback) argument, which return <c>None</c> when the ini key is
/// absent — i.e. the field's real Python default is <c>None</c>, not a hardcoded literal.
/// Per PORT_CONVENTIONS.md rule 6 (<c>Optional[X]</c> → <c>X?</c>) those fields are nullable
/// here: <see cref="SourcePaths"/>, <see cref="TargetPath"/>, <see cref="OutputPath"/>,
/// <see cref="SourcePattern"/>, <see cref="TargetPattern"/>, <see cref="OutputPattern"/>,
/// <see cref="FaceSelectorGender"/>, <see cref="FaceSelectorRace"/>,
/// <see cref="FaceSelectorAgeStart"/>, <see cref="FaceSelectorAgeEnd"/>,
/// <see cref="TrimFrameStart"/>, <see cref="TrimFrameEnd"/>, <see cref="OutputVideoFps"/>.
/// Several consumers depend on the None/zero distinction: e.g. <c>vision.py</c>'s
/// <c>count_trim_frame_total</c>/<c>restrict_trim_frame</c> take
/// <c>Optional[int]</c> and use <c>isinstance(x, int)</c> to tell "unset" from frame 0;
/// <c>face_selector.py</c>'s <c>sort_and_filter_faces</c> only applies age filtering when
/// <c>face_selector_age_start or face_selector_age_end</c> is truthy, which age 0 would
/// otherwise defeat; and the same file compares <c>face_selector_gender == 'auto'</c>, a real,
/// distinct value from "unset" (unset does not trigger the auto-inference-from-source-face
/// path that an explicit 'auto' does). <c>open_browser</c> and <c>halt_on_error</c> are also
/// <c>None</c>-by-default at the argparse layer (both are <c>store_true</c> flags whose
/// default is <c>config.get_bool_value(...)</c> with no fallback), but every consumer
/// (<c>job_manager.submit_jobs</c>/<c>delete_jobs</c>, <c>job_runner.run_jobs</c>/
/// <c>retry_jobs</c>) declares a plain non-Optional <c>bool</c> parameter and none of them
/// distinguish <c>None</c> from <c>False</c>, so they stay non-nullable <c>bool</c> here.
/// </summary>
public sealed record State(
	string Command,
	string ConfigPath,
	string TempPath,
	string JobsPath,
	IReadOnlyList<string>? SourcePaths,
	string? TargetPath,
	string? OutputPath,
	string? SourcePattern,
	string? TargetPattern,
	string? OutputPattern,
	IReadOnlyList<DownloadProvider> DownloadProviders,
	DownloadScope DownloadScope,
	BenchmarkMode BenchmarkMode,
	IReadOnlyList<BenchmarkResolution> BenchmarkResolutions,
	int BenchmarkCycleCount,
	FaceDetectorModel FaceDetectorModel,
	string FaceDetectorSize,
	Margin FaceDetectorMargin,
	IReadOnlyList<int> FaceDetectorAngles,
	double FaceDetectorScore,
	FaceLandmarkerModel FaceLandmarkerModel,
	double FaceLandmarkerScore,
	FaceSelectorMode FaceSelectorMode,
	FaceSelectorOrder FaceSelectorOrder,
	FaceSelectorRace? FaceSelectorRace,
	FaceSelectorGender? FaceSelectorGender,
	int? FaceSelectorAgeStart,
	int? FaceSelectorAgeEnd,
	int ReferenceFacePosition,
	double ReferenceFaceDistance,
	int ReferenceFrameNumber,
	double FaceTrackerScore,
	FaceOccluderModel FaceOccluderModel,
	FaceParserModel FaceParserModel,
	IReadOnlyList<FaceMaskType> FaceMaskTypes,
	IReadOnlyList<FaceMaskArea> FaceMaskAreas,
	IReadOnlyList<FaceMaskRegion> FaceMaskRegions,
	double FaceMaskBlur,
	Padding FaceMaskPadding,
	VoiceExtractorModel VoiceExtractorModel,
	int? TrimFrameStart,
	int? TrimFrameEnd,
	TempFrameFormat TempFrameFormat,
	TempPixelFormat TempPixelFormat,
	int TargetFrameAmount,
	int OutputImageQuality,
	double OutputImageScale,
	AudioEncoder OutputAudioEncoder,
	int OutputAudioQuality,
	int OutputAudioVolume,
	VideoEncoder OutputVideoEncoder,
	VideoPreset OutputVideoPreset,
	int OutputVideoQuality,
	double OutputVideoScale,
	double? OutputVideoFps,
	WorkflowMode WorkflowMode,
	WorkflowStrategy WorkflowStrategy,
	IReadOnlyList<string> Processors,
	bool OpenBrowser,
	IReadOnlyList<string> UiLayouts,
	UiWorkflow UiWorkflow,
	IReadOnlyList<int> ExecutionDeviceIds,
	IReadOnlyList<ExecutionProvider> ExecutionProviders,
	int ExecutionThreadCount,
	VideoMemoryStrategy VideoMemoryStrategy,
	LogLevel LogLevel,
	bool HaltOnError,
	string JobId,
	JobStatus JobStatus,
	int StepIndex);
