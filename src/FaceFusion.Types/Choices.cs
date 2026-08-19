using System;
using System.Collections.Generic;
using System.Globalization;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/choices.py. Python builds many of its lists via
/// <c>list(get_args(SomeLiteral))</c>; in C# that becomes <c>Enum.GetValues&lt;T&gt;()</c>,
/// which enumerates in declaration order (matching the Literal's tuple order) — used directly
/// below rather than duplicated as a stored list.
///
/// The two numeric-range builders (<c>create_int_range</c>/<c>create_float_range</c>) live in
/// facefusion/common_helper.py, which is out of this port's scope (owned by FaceFusion.Core).
/// They are reproduced here as tiny private helpers so choices.py's range constants can be
/// ported faithfully; FaceFusion.Core.CommonHelper already carries an equivalent public copy
/// (CreateIntRange/CreateFloatRange) for its own module — de-duplicate the two once
/// FaceFusion.Core is allowed to depend on FaceFusion.Types-side helpers, or vice versa.
/// </summary>
public static class Choices
{
	// face_detector_set
	public static readonly IReadOnlyDictionary<FaceDetectorModel, IReadOnlyList<string>> FaceDetectorSet = new Dictionary<FaceDetectorModel, IReadOnlyList<string>>
	{
		[FaceDetectorModel.Many] = new[] { "640x640" },
		[FaceDetectorModel.Retinaface] = new[] { "160x160", "320x320", "480x480", "512x512", "640x640" },
		[FaceDetectorModel.Scrfd] = new[] { "160x160", "320x320", "480x480", "512x512", "640x640" },
		[FaceDetectorModel.YoloFace] = new[] { "640x640" },
		[FaceDetectorModel.Yunet] = new[] { "640x640" }
	};

	// face_detector_models, face_landmarker_models, face_selector_modes, face_selector_orders,
	// genders, races, face_selector_genders, face_selector_races, face_occluder_models,
	// face_parser_models, face_mask_types are all `list(get_args(...))` in Python — use
	// Enum.GetValues<T>() directly at call sites instead of duplicating a stored list here.

	// face_mask_area_set
	public static readonly IReadOnlyDictionary<FaceMaskArea, IReadOnlyList<int>> FaceMaskAreaSet = new Dictionary<FaceMaskArea, IReadOnlyList<int>>
	{
		[FaceMaskArea.UpperFace] = new[] { 0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17 },
		[FaceMaskArea.LowerFace] = new[] { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31 },
		[FaceMaskArea.Mouth] = new[] { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67 }
	};

	// face_mask_region_set
	public static readonly IReadOnlyDictionary<FaceMaskRegion, int> FaceMaskRegionSet = new Dictionary<FaceMaskRegion, int>
	{
		[FaceMaskRegion.Skin] = 1,
		[FaceMaskRegion.LeftEyebrow] = 2,
		[FaceMaskRegion.RightEyebrow] = 3,
		[FaceMaskRegion.LeftEye] = 4,
		[FaceMaskRegion.RightEye] = 5,
		[FaceMaskRegion.Glasses] = 6,
		[FaceMaskRegion.Nose] = 10,
		[FaceMaskRegion.Mouth] = 11,
		[FaceMaskRegion.UpperLip] = 12,
		[FaceMaskRegion.LowerLip] = 13
	};

	// audio_type_set
	public static readonly IReadOnlyDictionary<AudioFormat, string> AudioTypeSet = new Dictionary<AudioFormat, string>
	{
		[AudioFormat.Flac] = "audio/flac",
		[AudioFormat.M4a] = "audio/mp4",
		[AudioFormat.Mp3] = "audio/mpeg",
		[AudioFormat.Ogg] = "audio/ogg",
		[AudioFormat.Opus] = "audio/opus",
		[AudioFormat.Wav] = "audio/x-wav"
	};

	// image_type_set
	public static readonly IReadOnlyDictionary<ImageFormat, string> ImageTypeSet = new Dictionary<ImageFormat, string>
	{
		[ImageFormat.Bmp] = "image/bmp",
		[ImageFormat.Jpeg] = "image/jpeg",
		[ImageFormat.Png] = "image/png",
		[ImageFormat.Tiff] = "image/tiff",
		[ImageFormat.Webp] = "image/webp"
	};

	// video_type_set
	public static readonly IReadOnlyDictionary<VideoFormat, string> VideoTypeSet = new Dictionary<VideoFormat, string>
	{
		[VideoFormat.Avi] = "video/x-msvideo",
		[VideoFormat.M4v] = "video/mp4",
		[VideoFormat.Mkv] = "video/x-matroska",
		[VideoFormat.Mp4] = "video/mp4",
		[VideoFormat.Mpeg] = "video/mpeg",
		[VideoFormat.Mov] = "video/quicktime",
		[VideoFormat.Mxf] = "application/mxf",
		[VideoFormat.Webm] = "video/webm",
		[VideoFormat.Wmv] = "video/x-ms-wmv"
	};

	// output_encoder_set
	public static readonly EncoderSet OutputEncoderSet = new EncoderSet(
		Audio: Enum.GetValues<AudioEncoder>(),
		Video: Enum.GetValues<VideoEncoder>());

	// benchmark_set
	public static readonly IReadOnlyDictionary<BenchmarkResolution, string> BenchmarkSet = new Dictionary<BenchmarkResolution, string>
	{
		[BenchmarkResolution.R240p] = ".assets/examples/target-240p.mp4",
		[BenchmarkResolution.R360p] = ".assets/examples/target-360p.mp4",
		[BenchmarkResolution.R540p] = ".assets/examples/target-540p.mp4",
		[BenchmarkResolution.R720p] = ".assets/examples/target-720p.mp4",
		[BenchmarkResolution.R1080p] = ".assets/examples/target-1080p.mp4",
		[BenchmarkResolution.R1440p] = ".assets/examples/target-1440p.mp4",
		[BenchmarkResolution.R2160p] = ".assets/examples/target-2160p.mp4"
	};

	// execution_provider_set
	public static readonly IReadOnlyDictionary<ExecutionProvider, ExecutionProviderValue> ExecutionProviderSet = new Dictionary<ExecutionProvider, ExecutionProviderValue>
	{
		[ExecutionProvider.Cuda] = ExecutionProviderValue.CudaExecutionProvider,
		[ExecutionProvider.Tensorrt] = ExecutionProviderValue.TensorrtExecutionProvider,
		[ExecutionProvider.Rocm] = ExecutionProviderValue.RocmExecutionProvider,
		[ExecutionProvider.Migraphx] = ExecutionProviderValue.MiGraphXExecutionProvider,
		[ExecutionProvider.Coreml] = ExecutionProviderValue.CoreMlExecutionProvider,
		[ExecutionProvider.Openvino] = ExecutionProviderValue.OpenVinoExecutionProvider,
		[ExecutionProvider.Qnn] = ExecutionProviderValue.QnnExecutionProvider,
		[ExecutionProvider.Directml] = ExecutionProviderValue.DmlExecutionProvider,
		[ExecutionProvider.Cpu] = ExecutionProviderValue.CpuExecutionProvider
	};

	// download_provider_set
	public static readonly IReadOnlyDictionary<DownloadProvider, DownloadProviderValue> DownloadProviderSet = new Dictionary<DownloadProvider, DownloadProviderValue>
	{
		[DownloadProvider.Github] = new DownloadProviderValue(
			Urls: new[] { "https://github.com" },
			Path: "/facefusion/facefusion-assets/releases/download/{base_name}/{file_name}"),
		[DownloadProvider.Huggingface] = new DownloadProviderValue(
			Urls: new[] { "https://huggingface.co", "https://hf-mirror.com" },
			Path: "/facefusion/{base_name}/resolve/main/{file_name}")
	};

	// log_level_set — values are Python's logging module levels (logging.ERROR/WARNING/INFO/DEBUG).
	public static readonly IReadOnlyDictionary<LogLevel, int> LogLevelSet = new Dictionary<LogLevel, int>
	{
		[LogLevel.Error] = 40,
		[LogLevel.Warn] = 30,
		[LogLevel.Info] = 20,
		[LogLevel.Debug] = 10
	};

	// benchmark_cycle_count_range
	public static readonly IReadOnlyList<int> BenchmarkCycleCountRange = CreateIntRange(1, 10, 1);

	// execution_thread_count_range
	public static readonly IReadOnlyList<int> ExecutionThreadCountRange = CreateIntRange(1, 32, 1);

	// face_detector_margin_range
	public static readonly IReadOnlyList<int> FaceDetectorMarginRange = CreateIntRange(0, 100, 1);

	// face_detector_angles (choices.py's default angle set, distinct from State.FaceDetectorAngles)
	public static readonly IReadOnlyList<int> FaceDetectorAngles = CreateIntRange(0, 270, 90);

	// face_detector_score_range
	public static readonly IReadOnlyList<double> FaceDetectorScoreRange = CreateFloatRange(0.0, 1.0, 0.05);

	// face_landmarker_score_range
	public static readonly IReadOnlyList<double> FaceLandmarkerScoreRange = CreateFloatRange(0.0, 1.0, 0.05);

	// face_mask_blur_range
	public static readonly IReadOnlyList<double> FaceMaskBlurRange = CreateFloatRange(0.0, 1.0, 0.05);

	// face_mask_padding_range
	public static readonly IReadOnlyList<int> FaceMaskPaddingRange = CreateIntRange(0, 100, 1);

	// face_selector_age_range
	public static readonly IReadOnlyList<int> FaceSelectorAgeRange = CreateIntRange(0, 100, 1);

	// reference_face_distance_range
	public static readonly IReadOnlyList<double> ReferenceFaceDistanceRange = CreateFloatRange(0.0, 1.0, 0.05);

	// face_tracker_score_range
	public static readonly IReadOnlyList<double> FaceTrackerScoreRange = CreateFloatRange(0.0, 0.5, 0.05);

	// target_frame_amount_range
	public static readonly IReadOnlyList<int> TargetFrameAmountRange = CreateIntRange(0, 10, 1);

	// output_image_quality_range
	public static readonly IReadOnlyList<int> OutputImageQualityRange = CreateIntRange(0, 100, 1);

	// output_image_scale_range
	public static readonly IReadOnlyList<double> OutputImageScaleRange = CreateFloatRange(0.25, 8.0, 0.25);

	// output_audio_quality_range
	public static readonly IReadOnlyList<int> OutputAudioQualityRange = CreateIntRange(0, 100, 1);

	// output_audio_volume_range
	public static readonly IReadOnlyList<int> OutputAudioVolumeRange = CreateIntRange(0, 100, 1);

	// output_video_quality_range
	public static readonly IReadOnlyList<int> OutputVideoQualityRange = CreateIntRange(0, 100, 1);

	// output_video_scale_range
	public static readonly IReadOnlyList<double> OutputVideoScaleRange = CreateFloatRange(0.25, 8.0, 0.25);

	/// <summary>
	/// Ported from facefusion/common_helper.py's <c>create_int_range</c>. Private and local to
	/// this class — see the type-level doc comment for why it is duplicated rather than shared.
	/// </summary>
	private static IReadOnlyList<int> CreateIntRange(int start, int end, int step)
	{
		var intRange = new List<int>();
		var current = start;

		while (current <= end)
		{
			intRange.Add(current);
			current += step;
		}

		return intRange;
	}

	/// <summary>
	/// Ported from facefusion/common_helper.py's <c>create_float_range</c>, including its
	/// round-to-2-decimal-places behaviour at every step.
	/// </summary>
	private static IReadOnlyList<double> CreateFloatRange(double start, double end, double step)
	{
		var floatRange = new List<double>();
		var current = start;

		while (current <= end)
		{
			floatRange.Add(Math.Round(current, 2, MidpointRounding.ToEven));
			current = Math.Round(current + step, 2, MidpointRounding.ToEven);
		}

		return floatRange;
	}
}
