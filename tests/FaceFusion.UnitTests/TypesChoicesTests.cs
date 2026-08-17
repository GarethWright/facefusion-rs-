using System.Linq;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Spot-checks the dictionaries and ranges ported from facefusion/choices.py against the
/// Python source values.
/// </summary>
public class TypesChoicesTests
{
	[Fact]
	public void FaceDetectorSet_MatchesPython()
	{
		Assert.Equal(new[] { "640x640" }, Choices.FaceDetectorSet[FaceDetectorModel.Many]);
		Assert.Equal(
			new[] { "160x160", "320x320", "480x480", "512x512", "640x640" },
			Choices.FaceDetectorSet[FaceDetectorModel.Retinaface]);
		Assert.Equal(
			new[] { "160x160", "320x320", "480x480", "512x512", "640x640" },
			Choices.FaceDetectorSet[FaceDetectorModel.Scrfd]);
		Assert.Equal(new[] { "640x640" }, Choices.FaceDetectorSet[FaceDetectorModel.YoloFace]);
		Assert.Equal(new[] { "640x640" }, Choices.FaceDetectorSet[FaceDetectorModel.Yunet]);
	}

	[Fact]
	public void FaceMaskRegionSet_MatchesPython()
	{
		Assert.Equal(1, Choices.FaceMaskRegionSet[FaceMaskRegion.Skin]);
		Assert.Equal(2, Choices.FaceMaskRegionSet[FaceMaskRegion.LeftEyebrow]);
		Assert.Equal(3, Choices.FaceMaskRegionSet[FaceMaskRegion.RightEyebrow]);
		Assert.Equal(4, Choices.FaceMaskRegionSet[FaceMaskRegion.LeftEye]);
		Assert.Equal(5, Choices.FaceMaskRegionSet[FaceMaskRegion.RightEye]);
		Assert.Equal(6, Choices.FaceMaskRegionSet[FaceMaskRegion.Glasses]);
		Assert.Equal(10, Choices.FaceMaskRegionSet[FaceMaskRegion.Nose]);
		Assert.Equal(11, Choices.FaceMaskRegionSet[FaceMaskRegion.Mouth]);
		Assert.Equal(12, Choices.FaceMaskRegionSet[FaceMaskRegion.UpperLip]);
		Assert.Equal(13, Choices.FaceMaskRegionSet[FaceMaskRegion.LowerLip]);
		Assert.Equal(10, Choices.FaceMaskRegionSet.Count);
	}

	[Fact]
	public void FaceMaskAreaSet_MatchesPython()
	{
		Assert.Equal(
			new[] { 0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17 },
			Choices.FaceMaskAreaSet[FaceMaskArea.UpperFace]);
		Assert.Equal(
			new[] { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31 },
			Choices.FaceMaskAreaSet[FaceMaskArea.LowerFace]);
		Assert.Equal(
			new[] { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67 },
			Choices.FaceMaskAreaSet[FaceMaskArea.Mouth]);
	}

	[Fact]
	public void AudioTypeSet_MatchesPython()
	{
		Assert.Equal("audio/flac", Choices.AudioTypeSet[AudioFormat.Flac]);
		Assert.Equal("audio/mp4", Choices.AudioTypeSet[AudioFormat.M4a]);
		Assert.Equal("audio/mpeg", Choices.AudioTypeSet[AudioFormat.Mp3]);
		Assert.Equal("audio/ogg", Choices.AudioTypeSet[AudioFormat.Ogg]);
		Assert.Equal("audio/opus", Choices.AudioTypeSet[AudioFormat.Opus]);
		Assert.Equal("audio/x-wav", Choices.AudioTypeSet[AudioFormat.Wav]);
	}

	[Fact]
	public void ImageTypeSet_MatchesPython()
	{
		Assert.Equal("image/bmp", Choices.ImageTypeSet[ImageFormat.Bmp]);
		Assert.Equal("image/jpeg", Choices.ImageTypeSet[ImageFormat.Jpeg]);
		Assert.Equal("image/png", Choices.ImageTypeSet[ImageFormat.Png]);
		Assert.Equal("image/tiff", Choices.ImageTypeSet[ImageFormat.Tiff]);
		Assert.Equal("image/webp", Choices.ImageTypeSet[ImageFormat.Webp]);
	}

	[Fact]
	public void VideoTypeSet_MatchesPython()
	{
		Assert.Equal("video/x-msvideo", Choices.VideoTypeSet[VideoFormat.Avi]);
		Assert.Equal("video/mp4", Choices.VideoTypeSet[VideoFormat.M4v]);
		Assert.Equal("video/x-matroska", Choices.VideoTypeSet[VideoFormat.Mkv]);
		Assert.Equal("video/mp4", Choices.VideoTypeSet[VideoFormat.Mp4]);
		Assert.Equal("video/mpeg", Choices.VideoTypeSet[VideoFormat.Mpeg]);
		Assert.Equal("video/quicktime", Choices.VideoTypeSet[VideoFormat.Mov]);
		Assert.Equal("application/mxf", Choices.VideoTypeSet[VideoFormat.Mxf]);
		Assert.Equal("video/webm", Choices.VideoTypeSet[VideoFormat.Webm]);
		Assert.Equal("video/x-ms-wmv", Choices.VideoTypeSet[VideoFormat.Wmv]);
	}

	[Fact]
	public void BenchmarkSet_MatchesPython()
	{
		Assert.Equal(".assets/examples/target-240p.mp4", Choices.BenchmarkSet[BenchmarkResolution.R240p]);
		Assert.Equal(".assets/examples/target-2160p.mp4", Choices.BenchmarkSet[BenchmarkResolution.R2160p]);
	}

	[Fact]
	public void ExecutionProviderSet_MatchesPython()
	{
		Assert.Equal(ExecutionProviderValue.CudaExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Cuda]);
		Assert.Equal(ExecutionProviderValue.TensorrtExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Tensorrt]);
		Assert.Equal(ExecutionProviderValue.RocmExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Rocm]);
		Assert.Equal(ExecutionProviderValue.MiGraphXExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Migraphx]);
		Assert.Equal(ExecutionProviderValue.CoreMlExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Coreml]);
		Assert.Equal(ExecutionProviderValue.OpenVinoExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Openvino]);
		Assert.Equal(ExecutionProviderValue.QnnExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Qnn]);
		Assert.Equal(ExecutionProviderValue.DmlExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Directml]);
		Assert.Equal(ExecutionProviderValue.CpuExecutionProvider, Choices.ExecutionProviderSet[ExecutionProvider.Cpu]);
	}

	[Fact]
	public void DownloadProviderSet_MatchesPython()
	{
		var github = Choices.DownloadProviderSet[DownloadProvider.Github];
		Assert.Equal(new[] { "https://github.com" }, github.Urls);
		Assert.Equal("/facefusion/facefusion-assets/releases/download/{base_name}/{file_name}", github.Path);

		var huggingface = Choices.DownloadProviderSet[DownloadProvider.Huggingface];
		Assert.Equal(new[] { "https://huggingface.co", "https://hf-mirror.com" }, huggingface.Urls);
		Assert.Equal("/facefusion/{base_name}/resolve/main/{file_name}", huggingface.Path);
	}

	[Fact]
	public void LogLevelSet_MatchesPythonLoggingLevels()
	{
		Assert.Equal(40, Choices.LogLevelSet[LogLevel.Error]);
		Assert.Equal(30, Choices.LogLevelSet[LogLevel.Warn]);
		Assert.Equal(20, Choices.LogLevelSet[LogLevel.Info]);
		Assert.Equal(10, Choices.LogLevelSet[LogLevel.Debug]);
	}

	[Fact]
	public void OutputEncoderSet_ContainsAllEncoderValues()
	{
		Assert.Equal(Enum.GetValues<AudioEncoder>(), Choices.OutputEncoderSet.Audio);
		Assert.Equal(Enum.GetValues<VideoEncoder>(), Choices.OutputEncoderSet.Video);
	}

	[Fact]
	public void CreateIntRange_MatchesPython()
	{
		Assert.Equal(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, Choices.BenchmarkCycleCountRange);
		Assert.Equal(Enumerable.Range(1, 32), Choices.ExecutionThreadCountRange);
		Assert.Equal(Enumerable.Range(0, 101), Choices.FaceDetectorMarginRange);
		Assert.Equal(new[] { 0, 90, 180, 270 }, Choices.FaceDetectorAngles);
		Assert.Equal(Enumerable.Range(0, 11), Choices.TargetFrameAmountRange);
	}

	[Fact]
	public void CreateFloatRange_MatchesPython()
	{
		// create_float_range(0.0, 1.0, 0.05) -> 21 values: 0.00, 0.05, ..., 1.00
		Assert.Equal(21, Choices.FaceDetectorScoreRange.Count);
		Assert.Equal(0.0, Choices.FaceDetectorScoreRange[0]);
		Assert.Equal(0.05, Choices.FaceDetectorScoreRange[1]);
		Assert.Equal(1.0, Choices.FaceDetectorScoreRange[^1]);
		Assert.Equal(Choices.FaceDetectorScoreRange, Choices.FaceLandmarkerScoreRange);
		Assert.Equal(Choices.FaceDetectorScoreRange, Choices.FaceMaskBlurRange);
		Assert.Equal(Choices.FaceDetectorScoreRange, Choices.ReferenceFaceDistanceRange);

		// create_float_range(0.0, 0.5, 0.05) -> 11 values: 0.00, 0.05, ..., 0.50
		Assert.Equal(11, Choices.FaceTrackerScoreRange.Count);
		Assert.Equal(0.5, Choices.FaceTrackerScoreRange[^1]);

		// create_float_range(0.25, 8.0, 0.25) -> 32 values: 0.25, 0.50, ..., 8.00
		Assert.Equal(32, Choices.OutputImageScaleRange.Count);
		Assert.Equal(0.25, Choices.OutputImageScaleRange[0]);
		Assert.Equal(8.0, Choices.OutputImageScaleRange[^1]);
		Assert.Equal(Choices.OutputImageScaleRange, Choices.OutputVideoScaleRange);
	}

	[Fact]
	public void EnumGetValues_MatchesPythonGetArgsLists()
	{
		// Mirrors choices.py's `list(get_args(...))` constants: face_detector_models,
		// genders, races, etc. — these are derived directly from the enum rather than stored.
		Assert.Equal(5, Enum.GetValues<FaceDetectorModel>().Length);
		Assert.Equal(2, Enum.GetValues<Gender>().Length);
		Assert.Equal(6, Enum.GetValues<Race>().Length);
		Assert.Equal(3, Enum.GetValues<FaceSelectorGender>().Length);
		Assert.Equal(7, Enum.GetValues<FaceSelectorRace>().Length);
	}
}
