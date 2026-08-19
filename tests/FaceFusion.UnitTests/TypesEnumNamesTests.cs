using System;
using System.Linq;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Wire-string round-tripping for every enum ported from a Python
/// <c>Literal['a', 'b', ...]</c> string union in facefusion/types.py. Each enum must convert
/// to its Python wire string and back without loss, since these values appear verbatim on CLI
/// args, in facefusion.ini, and in job JSON.
/// </summary>
public class TypesEnumNamesTests
{
	private static void AssertRoundTrips<T>(params (T Value, string Wire)[] expected) where T : struct, Enum
	{
		// Every declared enum member must be covered by the expected pairs.
		var allValues = Enum.GetValues<T>();
		Assert.Equal(allValues.Length, expected.Length);

		foreach (var (value, wire) in expected)
		{
			Assert.Equal(wire, value.ToWireName());
			Assert.Equal(value, EnumNames.FromWireName<T>(wire));
			Assert.True(EnumNames.TryFromWireName<T>(wire, out var parsed));
			Assert.Equal(value, parsed);
		}
	}

	[Fact]
	public void Gender_RoundTrips()
	{
		AssertRoundTrips(
			(Gender.Female, "female"),
			(Gender.Male, "male"));
	}

	[Fact]
	public void Race_RoundTrips()
	{
		AssertRoundTrips(
			(Race.White, "white"),
			(Race.Black, "black"),
			(Race.Latino, "latino"),
			(Race.Asian, "asian"),
			(Race.Indian, "indian"),
			(Race.Arabic, "arabic"));
	}

	[Fact]
	public void FaceSelectorGender_RoundTrips()
	{
		AssertRoundTrips(
			(FaceSelectorGender.Auto, "auto"),
			(FaceSelectorGender.Female, "female"),
			(FaceSelectorGender.Male, "male"));
	}

	[Fact]
	public void FaceSelectorRace_RoundTrips()
	{
		AssertRoundTrips(
			(FaceSelectorRace.Auto, "auto"),
			(FaceSelectorRace.White, "white"),
			(FaceSelectorRace.Black, "black"),
			(FaceSelectorRace.Latino, "latino"),
			(FaceSelectorRace.Asian, "asian"),
			(FaceSelectorRace.Indian, "indian"),
			(FaceSelectorRace.Arabic, "arabic"));
	}

	[Fact]
	public void Language_RoundTrips()
	{
		AssertRoundTrips((Language.En, "en"));
	}

	[Fact]
	public void WorkflowMode_RoundTrips()
	{
		AssertRoundTrips(
			(WorkflowMode.Auto, "auto"),
			(WorkflowMode.ImageToImage, "image-to-image"),
			(WorkflowMode.ImageToVideo, "image-to-video"));
	}

	[Fact]
	public void WorkflowStrategy_RoundTrips()
	{
		AssertRoundTrips(
			(WorkflowStrategy.Disk, "disk"),
			(WorkflowStrategy.Memory, "memory"));
	}

	[Fact]
	public void ColorMode_RoundTrips()
	{
		AssertRoundTrips(
			(ColorMode.Rgb, "rgb"),
			(ColorMode.Rgba, "rgba"));
	}

	[Fact]
	public void ColorSpace_RoundTrips()
	{
		AssertRoundTrips(
			(ColorSpace.Bt601, "bt601"),
			(ColorSpace.Bt709, "bt709"),
			(ColorSpace.Bt2020, "bt2020"));
	}

	[Fact]
	public void Orientation_RoundTrips()
	{
		AssertRoundTrips(
			(Orientation.Landscape, "landscape"),
			(Orientation.Portrait, "portrait"));
	}

	[Fact]
	public void ProcessState_RoundTrips()
	{
		AssertRoundTrips(
			(ProcessState.Checking, "checking"),
			(ProcessState.Processing, "processing"),
			(ProcessState.Stopping, "stopping"),
			(ProcessState.Pending, "pending"));
	}

	[Fact]
	public void WarpTemplate_RoundTrips()
	{
		AssertRoundTrips(
			(WarpTemplate.Arcface112V1, "arcface_112_v1"),
			(WarpTemplate.Arcface112V2, "arcface_112_v2"),
			(WarpTemplate.Arcface128, "arcface_128"),
			(WarpTemplate.DflWholeFace, "dfl_whole_face"),
			(WarpTemplate.Ffhq512, "ffhq_512"),
			(WarpTemplate.Mtcnn512, "mtcnn_512"),
			(WarpTemplate.Styleganex384, "styleganex_384"));
	}

	[Fact]
	public void ProcessMode_RoundTrips()
	{
		AssertRoundTrips(
			(ProcessMode.Output, "output"),
			(ProcessMode.Preview, "preview"),
			(ProcessMode.Stream, "stream"));
	}

	[Fact]
	public void LogLevel_RoundTrips()
	{
		AssertRoundTrips(
			(LogLevel.Error, "error"),
			(LogLevel.Warn, "warn"),
			(LogLevel.Info, "info"),
			(LogLevel.Debug, "debug"));
	}

	[Fact]
	public void FaceDetectorModel_RoundTrips()
	{
		AssertRoundTrips(
			(FaceDetectorModel.Many, "many"),
			(FaceDetectorModel.Retinaface, "retinaface"),
			(FaceDetectorModel.Scrfd, "scrfd"),
			(FaceDetectorModel.YoloFace, "yolo_face"),
			(FaceDetectorModel.Yunet, "yunet"));
	}

	[Fact]
	public void FaceLandmarkerModel_RoundTrips()
	{
		AssertRoundTrips(
			(FaceLandmarkerModel.Many, "many"),
			(FaceLandmarkerModel.TwoDFan4, "2dfan4"),
			(FaceLandmarkerModel.PeppaWutz, "peppa_wutz"));
	}

	[Fact]
	public void FaceSelectorMode_RoundTrips()
	{
		AssertRoundTrips(
			(FaceSelectorMode.Many, "many"),
			(FaceSelectorMode.One, "one"),
			(FaceSelectorMode.Reference, "reference"));
	}

	[Fact]
	public void FaceSelectorOrder_RoundTrips()
	{
		AssertRoundTrips(
			(FaceSelectorOrder.LeftRight, "left-right"),
			(FaceSelectorOrder.RightLeft, "right-left"),
			(FaceSelectorOrder.TopBottom, "top-bottom"),
			(FaceSelectorOrder.BottomTop, "bottom-top"),
			(FaceSelectorOrder.SmallLarge, "small-large"),
			(FaceSelectorOrder.LargeSmall, "large-small"),
			(FaceSelectorOrder.BestWorst, "best-worst"),
			(FaceSelectorOrder.WorstBest, "worst-best"));
	}

	[Fact]
	public void FaceOccluderModel_RoundTrips()
	{
		AssertRoundTrips(
			(FaceOccluderModel.Many, "many"),
			(FaceOccluderModel.Xseg1, "xseg_1"),
			(FaceOccluderModel.Xseg2, "xseg_2"),
			(FaceOccluderModel.Xseg3, "xseg_3"));
	}

	[Fact]
	public void FaceParserModel_RoundTrips()
	{
		AssertRoundTrips(
			(FaceParserModel.BisenetResnet18, "bisenet_resnet_18"),
			(FaceParserModel.BisenetResnet34, "bisenet_resnet_34"));
	}

	[Fact]
	public void FaceMaskType_RoundTrips()
	{
		AssertRoundTrips(
			(FaceMaskType.Box, "box"),
			(FaceMaskType.Occlusion, "occlusion"),
			(FaceMaskType.Area, "area"),
			(FaceMaskType.Region, "region"));
	}

	[Fact]
	public void FaceMaskArea_RoundTrips()
	{
		AssertRoundTrips(
			(FaceMaskArea.UpperFace, "upper-face"),
			(FaceMaskArea.LowerFace, "lower-face"),
			(FaceMaskArea.Mouth, "mouth"));
	}

	[Fact]
	public void FaceMaskRegion_RoundTrips()
	{
		AssertRoundTrips(
			(FaceMaskRegion.Skin, "skin"),
			(FaceMaskRegion.LeftEyebrow, "left-eyebrow"),
			(FaceMaskRegion.RightEyebrow, "right-eyebrow"),
			(FaceMaskRegion.LeftEye, "left-eye"),
			(FaceMaskRegion.RightEye, "right-eye"),
			(FaceMaskRegion.Glasses, "glasses"),
			(FaceMaskRegion.Nose, "nose"),
			(FaceMaskRegion.Mouth, "mouth"),
			(FaceMaskRegion.UpperLip, "upper-lip"),
			(FaceMaskRegion.LowerLip, "lower-lip"));
	}

	[Fact]
	public void VoiceExtractorModel_RoundTrips()
	{
		AssertRoundTrips(
			(VoiceExtractorModel.KimVocal1, "kim_vocal_1"),
			(VoiceExtractorModel.KimVocal2, "kim_vocal_2"),
			(VoiceExtractorModel.UvrMdxnet, "uvr_mdxnet"));
	}

	[Fact]
	public void AudioFormat_RoundTrips()
	{
		AssertRoundTrips(
			(AudioFormat.Flac, "flac"),
			(AudioFormat.M4a, "m4a"),
			(AudioFormat.Mp3, "mp3"),
			(AudioFormat.Ogg, "ogg"),
			(AudioFormat.Opus, "opus"),
			(AudioFormat.Wav, "wav"));
	}

	[Fact]
	public void ImageFormat_RoundTrips()
	{
		AssertRoundTrips(
			(ImageFormat.Bmp, "bmp"),
			(ImageFormat.Jpeg, "jpeg"),
			(ImageFormat.Png, "png"),
			(ImageFormat.Tiff, "tiff"),
			(ImageFormat.Webp, "webp"));
	}

	[Fact]
	public void VideoFormat_RoundTrips()
	{
		AssertRoundTrips(
			(VideoFormat.Avi, "avi"),
			(VideoFormat.M4v, "m4v"),
			(VideoFormat.Mkv, "mkv"),
			(VideoFormat.Mov, "mov"),
			(VideoFormat.Mp4, "mp4"),
			(VideoFormat.Mpeg, "mpeg"),
			(VideoFormat.Mxf, "mxf"),
			(VideoFormat.Webm, "webm"),
			(VideoFormat.Wmv, "wmv"));
	}

	[Fact]
	public void TempFrameFormat_RoundTrips()
	{
		AssertRoundTrips(
			(TempFrameFormat.Bmp, "bmp"),
			(TempFrameFormat.Jpeg, "jpeg"),
			(TempFrameFormat.Png, "png"),
			(TempFrameFormat.Tiff, "tiff"));
	}

	[Fact]
	public void TempPixelFormat_RoundTrips()
	{
		AssertRoundTrips(
			(TempPixelFormat.Bgr24, "bgr24"),
			(TempPixelFormat.Bgra, "bgra"));
	}

	[Fact]
	public void AudioEncoder_RoundTrips()
	{
		AssertRoundTrips(
			(AudioEncoder.Flac, "flac"),
			(AudioEncoder.Aac, "aac"),
			(AudioEncoder.Libmp3lame, "libmp3lame"),
			(AudioEncoder.Libopus, "libopus"),
			(AudioEncoder.Libvorbis, "libvorbis"),
			(AudioEncoder.PcmS16le, "pcm_s16le"),
			(AudioEncoder.PcmS32le, "pcm_s32le"));
	}

	[Fact]
	public void VideoEncoder_RoundTrips()
	{
		AssertRoundTrips(
			(VideoEncoder.Libx264, "libx264"),
			(VideoEncoder.Libx264rgb, "libx264rgb"),
			(VideoEncoder.Libx265, "libx265"),
			(VideoEncoder.LibvpxVp9, "libvpx-vp9"),
			(VideoEncoder.H264Nvenc, "h264_nvenc"),
			(VideoEncoder.HevcNvenc, "hevc_nvenc"),
			(VideoEncoder.H264Amf, "h264_amf"),
			(VideoEncoder.HevcAmf, "hevc_amf"),
			(VideoEncoder.H264Qsv, "h264_qsv"),
			(VideoEncoder.HevcQsv, "hevc_qsv"),
			(VideoEncoder.H264Videotoolbox, "h264_videotoolbox"),
			(VideoEncoder.HevcVideotoolbox, "hevc_videotoolbox"),
			(VideoEncoder.Rawvideo, "rawvideo"));
	}

	[Fact]
	public void VideoPreset_RoundTrips()
	{
		AssertRoundTrips(
			(VideoPreset.Ultrafast, "ultrafast"),
			(VideoPreset.Superfast, "superfast"),
			(VideoPreset.Veryfast, "veryfast"),
			(VideoPreset.Faster, "faster"),
			(VideoPreset.Fast, "fast"),
			(VideoPreset.Medium, "medium"),
			(VideoPreset.Slow, "slow"),
			(VideoPreset.Slower, "slower"),
			(VideoPreset.Veryslow, "veryslow"));
	}

	[Fact]
	public void BenchmarkMode_RoundTrips()
	{
		AssertRoundTrips(
			(BenchmarkMode.Warm, "warm"),
			(BenchmarkMode.Cold, "cold"));
	}

	[Fact]
	public void BenchmarkResolution_RoundTrips()
	{
		AssertRoundTrips(
			(BenchmarkResolution.R240p, "240p"),
			(BenchmarkResolution.R360p, "360p"),
			(BenchmarkResolution.R540p, "540p"),
			(BenchmarkResolution.R720p, "720p"),
			(BenchmarkResolution.R1080p, "1080p"),
			(BenchmarkResolution.R1440p, "1440p"),
			(BenchmarkResolution.R2160p, "2160p"));
	}

	[Fact]
	public void WebcamMode_RoundTrips()
	{
		AssertRoundTrips(
			(WebcamMode.Inline, "inline"),
			(WebcamMode.Udp, "udp"),
			(WebcamMode.V4l2, "v4l2"));
	}

	[Fact]
	public void StreamMode_RoundTrips()
	{
		AssertRoundTrips(
			(StreamMode.Udp, "udp"),
			(StreamMode.V4l2, "v4l2"));
	}

	[Fact]
	public void ExecutionProvider_RoundTrips()
	{
		AssertRoundTrips(
			(ExecutionProvider.Cuda, "cuda"),
			(ExecutionProvider.Tensorrt, "tensorrt"),
			(ExecutionProvider.Rocm, "rocm"),
			(ExecutionProvider.Migraphx, "migraphx"),
			(ExecutionProvider.Coreml, "coreml"),
			(ExecutionProvider.Openvino, "openvino"),
			(ExecutionProvider.Qnn, "qnn"),
			(ExecutionProvider.Directml, "directml"),
			(ExecutionProvider.Cpu, "cpu"));
	}

	[Fact]
	public void ExecutionProviderValue_RoundTrips()
	{
		AssertRoundTrips(
			(ExecutionProviderValue.CpuExecutionProvider, "CPUExecutionProvider"),
			(ExecutionProviderValue.CoreMlExecutionProvider, "CoreMLExecutionProvider"),
			(ExecutionProviderValue.CudaExecutionProvider, "CUDAExecutionProvider"),
			(ExecutionProviderValue.DmlExecutionProvider, "DmlExecutionProvider"),
			(ExecutionProviderValue.OpenVinoExecutionProvider, "OpenVINOExecutionProvider"),
			(ExecutionProviderValue.MiGraphXExecutionProvider, "MIGraphXExecutionProvider"),
			(ExecutionProviderValue.QnnExecutionProvider, "QNNExecutionProvider"),
			(ExecutionProviderValue.RocmExecutionProvider, "ROCMExecutionProvider"),
			(ExecutionProviderValue.TensorrtExecutionProvider, "TensorrtExecutionProvider"));
	}

	[Fact]
	public void DownloadProvider_RoundTrips()
	{
		AssertRoundTrips(
			(DownloadProvider.Github, "github"),
			(DownloadProvider.Huggingface, "huggingface"));
	}

	[Fact]
	public void DownloadScope_RoundTrips()
	{
		AssertRoundTrips(
			(DownloadScope.Lite, "lite"),
			(DownloadScope.Full, "full"));
	}

	[Fact]
	public void VideoMemoryStrategy_RoundTrips()
	{
		AssertRoundTrips(
			(VideoMemoryStrategy.Strict, "strict"),
			(VideoMemoryStrategy.Moderate, "moderate"),
			(VideoMemoryStrategy.Tolerant, "tolerant"));
	}

	[Fact]
	public void AppContext_RoundTrips()
	{
		// FaceFusion.Types.AppContext is ambiguous with System.AppContext once both
		// namespaces are in scope — qualify it explicitly, and note the collision on the
		// type itself for anyone hitting this later.
		AssertRoundTrips(
			(FaceFusion.Types.AppContext.Cli, "cli"),
			(FaceFusion.Types.AppContext.Ui, "ui"));
	}

	[Fact]
	public void UiWorkflow_RoundTrips()
	{
		AssertRoundTrips(
			(UiWorkflow.InstantRunner, "instant_runner"),
			(UiWorkflow.JobRunner, "job_runner"),
			(UiWorkflow.JobManager, "job_manager"));
	}

	[Fact]
	public void JobStatus_RoundTrips()
	{
		AssertRoundTrips(
			(JobStatus.Drafted, "drafted"),
			(JobStatus.Queued, "queued"),
			(JobStatus.Completed, "completed"),
			(JobStatus.Failed, "failed"));
	}

	[Fact]
	public void JobStepStatus_RoundTrips()
	{
		AssertRoundTrips(
			(JobStepStatus.Drafted, "drafted"),
			(JobStepStatus.Queued, "queued"),
			(JobStepStatus.Started, "started"),
			(JobStepStatus.Completed, "completed"),
			(JobStepStatus.Failed, "failed"));
	}

	[Fact]
	public void StateKey_RoundTrips()
	{
		// StateKey is large (70 members mirroring facefusion/types.py's StateKey Literal) —
		// spot-check a representative sample plus full coverage of the round-trip contract
		// (every declared member has a working wire name) rather than enumerating every pair.
		var allValues = Enum.GetValues<StateKey>();
		Assert.Equal(70, allValues.Length);

		foreach (var value in allValues)
		{
			var wire = value.ToWireName();
			Assert.Equal(value, EnumNames.FromWireName<StateKey>(wire));
		}

		Assert.Equal("command", StateKey.Command.ToWireName());
		Assert.Equal("config_path", StateKey.ConfigPath.ToWireName());
		Assert.Equal("face_detector_model", StateKey.FaceDetectorModel.ToWireName());
		Assert.Equal("face_selector_age_start", StateKey.FaceSelectorAgeStart.ToWireName());
		Assert.Equal("output_video_fps", StateKey.OutputVideoFps.ToWireName());
		Assert.Equal("video_memory_strategy", StateKey.VideoMemoryStrategy.ToWireName());
		Assert.Equal("step_index", StateKey.StepIndex.ToWireName());
		Assert.Equal(StateKey.HaltOnError, EnumNames.FromWireName<StateKey>("halt_on_error"));
	}

	[Fact]
	public void FromWireName_ThrowsOnUnknownName()
	{
		Assert.Throws<ArgumentException>(() => EnumNames.FromWireName<Gender>("nonbinary"));
	}

	[Fact]
	public void TryFromWireName_ReturnsFalseOnUnknownName()
	{
		Assert.False(EnumNames.TryFromWireName<Gender>("nonbinary", out _));
	}

	[Fact]
	public void AllWireNames_MatchesGetArgsOrder()
	{
		// Mirrors Python's `list(get_args(Gender))` == ['female', 'male'].
		Assert.Equal(new[] { "female", "male" }, EnumNames.AllWireNames<Gender>());
	}
}
