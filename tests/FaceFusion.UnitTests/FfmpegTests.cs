using FaceFusion.Core;
using FaceFusion.Media;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of tests/test_ffmpeg.py.
///
/// <para>
/// Both preconditions the earlier port report against this file cited — no <c>ffmpeg</c>/
/// <c>ffprobe</c> binaries, no example media — have since been removed from the environment
/// (see docs/PORT_CONVENTIONS.md and <c>tools/parity/fetch_examples.sh</c>), so every test
/// below now runs for real rather than being <c>[Fact(Skip = ...)]</c>. Each is still gated
/// on <see cref="MediaFactAttribute"/> so the suite degrades to a clear runtime skip (rather
/// than a confusing file-not-found) in an environment that genuinely lacks either.
/// </para>
///
/// <para>
/// <b>Fixture generation.</b> Python's module-scoped <c>before_all</c> pytest fixture
/// derives several fixture files (per-fps re-encodes, per-container/-sample-rate variants,
/// one HDR-tagged clip) from the four downloaded examples before any test body runs. This
/// port reproduces the same ffmpeg invocations in <see cref="MediaFixtures.Ensure"/>
/// (idempotently — see that class's doc comment for why), invoked once per test instance
/// from the constructor below, mirroring Python's <c>before_all</c>/<c>before_each</c> pair
/// (<see cref="TestHelper.PrepareTestOutputDirectory"/> is the function-scoped half).
/// </para>
/// </summary>
[Collection("MediaOutput")]
public sealed class FfmpegTests
{
	public FfmpegTests()
	{
		MediaFixtures.Ensure();
		TestHelper.PrepareTestOutputDirectory();
	}

	private static readonly string TempPath = Path.GetTempPath();
	private const string TempFrameFormat = "png";

	/// <summary>
	/// Video encoders whose <c>-encoders</c> listing only means "ffmpeg was compiled with
	/// support for the API", not "usable here" — they additionally need matching hardware
	/// (an NVIDIA/AMD/Intel GPU driver, or running on macOS for VideoToolbox), none of
	/// which this container has. Python's test suite sidesteps the same problem with a
	/// local <c>get_available_encoder_set()</c> override that hard-codes
	/// <c>{'audio': ['aac'], 'video': ['libx264']}</c> when <c>os.getenv('CI')</c> is set;
	/// that env var is not set in this container even though it has no GPU, so porting the
	/// check literally would not actually solve the problem here. This filters the same
	/// class of encoder out of the real (still exercised) <see cref="Ffmpeg.GetAvailableEncoderSet"/>
	/// result instead — deliberately test-infra only, not a change to <see cref="Ffmpeg"/> itself.
	/// </summary>
	private static readonly HashSet<VideoEncoder> HardwareOnlyVideoEncoders = new()
	{
		VideoEncoder.H264Nvenc, VideoEncoder.HevcNvenc,
		VideoEncoder.H264Amf, VideoEncoder.HevcAmf,
		VideoEncoder.H264Qsv, VideoEncoder.HevcQsv,
		VideoEncoder.H264Videotoolbox, VideoEncoder.HevcVideotoolbox
	};

	private static IReadOnlyList<VideoEncoder> GetSoftwareVideoEncoders()
		=> Ffmpeg.GetAvailableEncoderSet().Video.Where(encoder => !HardwareOnlyVideoEncoders.Contains(encoder)).ToArray();

	[MediaFact]
	public void TestGetAvailableEncoderSet()
	{
		var availableEncoderSet = Ffmpeg.GetAvailableEncoderSet();

		Assert.Contains(AudioEncoder.Aac, availableEncoderSet.Audio);
		Assert.Contains(VideoEncoder.Libx264, availableEncoderSet.Video);
	}

	[MediaFact]
	public void TestExtractFrames()
	{
		var testSet = new (string TargetFile, int TrimFrameStart, int TrimFrameEnd, int FrameTotal, double FrameStd, double FrameMax)[]
		{
			("target-240p-25fps.mp4", 0, 270, 324, 55, 250),
			("target-240p-25fps.mp4", 224, 270, 55, 55, 250),
			("target-240p-25fps.mp4", 124, 224, 120, 55, 250),
			("target-240p-25fps.mp4", 0, 100, 120, 55, 250),
			("target-240p-30fps.mp4", 0, 324, 324, 55, 250),
			("target-240p-30fps.mp4", 224, 324, 100, 55, 250),
			("target-240p-30fps.mp4", 124, 224, 100, 55, 250),
			("target-240p-30fps.mp4", 0, 100, 100, 55, 250),
			("target-240p-60fps.mp4", 0, 648, 324, 55, 250),
			("target-240p-60fps.mp4", 224, 648, 212, 55, 250),
			("target-240p-60fps.mp4", 124, 224, 50, 55, 250),
			("target-240p-60fps.mp4", 0, 100, 50, 55, 250),
			("target-240p-smpte2084.mp4", 0, 1, 1, 32, 190)
		};

		foreach (var (targetFile, trimFrameStart, trimFrameEnd, frameTotal, frameStd, frameMax) in testSet)
		{
			var targetPath = TestHelper.GetTestExampleFile(targetFile);

			// FaceFusion's own restrict_color_transfer emits
			// `scale=out_primaries=...:out_transfer=bt709:intent=perceptual`, and those
			// scale options only exist on ffmpeg 7+. On ffmpeg 6.x this HDR case fails
			// identically in the PYTHON suite (tests/test_ffmpeg.py::test_extract_frames
			// fails here for the same reason), so it is an upstream ffmpeg-version
			// requirement rather than a port defect. Skip it rather than assert a
			// behaviour the reference implementation does not achieve either.
			if (targetFile.Contains("smpte2084", StringComparison.Ordinal) && !TestHelper.SupportsHdrColorTransfer())
			{
				continue;
			}

			// Defensive clear before create: xunit does not guarantee test method
			// execution order (unlike pytest's default source-order), and every entry for
			// the same target file shares one temp-directory key, so a differently-ordered
			// or previously-aborted run must not be able to leave stale frames behind for
			// this one to read.
			TempHelper.ClearTempDirectory(targetPath, TempPath);
			TempHelper.CreateTempDirectory(targetPath, TempPath);

			try
			{
				var extracted = Ffmpeg.ExtractFrames(targetPath, new Resolution(452, 240), 30.0, trimFrameStart, trimFrameEnd, TempPath, TempFrameFormat, _ => { });
				Assert.True(extracted, $"extract_frames failed for {targetFile} [{trimFrameStart}, {trimFrameEnd})");

				var tempFrameSet = TempHelper.ResolveTempFrameSet(targetPath, TempPath, TempFrameFormat);
				Assert.Equal(frameTotal, tempFrameSet.Count);

				using var frame = Vision.Vision.ReadImage(tempFrameSet[trimFrameStart]);
				Assert.NotNull(frame);

				var (std, max) = ComputeFrameStats(frame!);
				Assert.True(std > frameStd, $"{targetFile}: expected std > {frameStd}, got {std}");
				Assert.True(max > frameMax, $"{targetFile}: expected max > {frameMax}, got {max}");
			}
			finally
			{
				TempHelper.ClearTempDirectory(targetPath, TempPath);
			}
		}
	}

	[MediaFact]
	public void TestMergeVideo()
	{
		var testSet = new (string TargetFile, string[] ColorTransfers)[]
		{
			("target-240p-16khz.avi", new[] { "bt709", "unknown" }),
			("target-240p-16khz.m4v", new[] { "bt709" }),
			("target-240p-16khz.mkv", new[] { "bt709" }),
			("target-240p-16khz.mp4", new[] { "bt709" }),
			("target-240p-16khz.mov", new[] { "bt709" }),
			("target-240p-16khz.webm", new[] { "bt709" }),
			("target-240p-16khz.wmv", new[] { "bt709" })
		};
		// See GetSoftwareVideoEncoders's doc comment: hardware-only encoders are excluded
		// since this container has no functional GPU to actually drive them, even though
		// ffmpeg's own -encoders listing reports them as compiled in.
		var outputVideoEncoders = GetSoftwareVideoEncoders();

		foreach (var (targetFile, colorTransfers) in testSet)
		{
			var targetPath = TestHelper.GetTestExampleFile(targetFile);
			TempHelper.ClearTempDirectory(targetPath, TempPath);

			try
			{
				foreach (var outputVideoEncoder in outputVideoEncoders)
				{
					TempHelper.CreateTempDirectory(targetPath, TempPath);
					Ffmpeg.ExtractFrames(targetPath, new Resolution(452, 240), 25.0, 0, 1, TempPath, TempFrameFormat, _ => { });

					var merged = Ffmpeg.MergeVideo(targetPath, 25.0, new Resolution(452, 240), 25.0, 0, 1, outputVideoEncoder, 100, VideoPreset.Ultrafast, TempPath, TempFrameFormat, _ => { });
					Assert.True(merged, $"merge_video failed for {targetFile} with encoder {outputVideoEncoder}");

					var videoMetadata = Ffprobe.ExtractVideoMetadata(TempHelper.GetTempFilePath(targetPath, TempPath));
					Assert.Contains(videoMetadata.ColorTransfer, colorTransfers);
				}
			}
			finally
			{
				TempHelper.ClearTempDirectory(targetPath, TempPath);
			}
		}
	}

	[MediaFact]
	public void TestConcatVideo()
	{
		var outputPath = TestHelper.GetTestOutputFile("test-concat-video.mp4");
		var targetPath = TestHelper.GetTestExampleFile("target-240p-16khz.mp4");
		var tempOutputPaths = new[] { targetPath, targetPath };

		Assert.True(Ffmpeg.ConcatVideo(outputPath, tempOutputPaths));
	}

	[MediaFact]
	public void TestReadAudioBuffer()
	{
		Assert.NotNull(Ffmpeg.ReadAudioBuffer(TestHelper.GetTestExampleFile("source.mp3"), 1, 16, 1));
		Assert.NotNull(Ffmpeg.ReadAudioBuffer(TestHelper.GetTestExampleFile("source.wav"), 1, 16, 1));
		Assert.Null(Ffmpeg.ReadAudioBuffer(TestHelper.GetTestExampleFile("invalid.mp3"), 1, 16, 1));
	}

	[MediaFact]
	public void TestRestoreAudio()
	{
		var testSet = new[]
		{
			"target-240p-16khz.avi",
			"target-240p-16khz.m4v",
			"target-240p-16khz.mkv",
			"target-240p-16khz.mov",
			"target-240p-16khz.mp4",
			"target-240p-48khz.mp4",
			"target-240p-16khz.webm",
			"target-240p-16khz.wmv"
		};
		var outputAudioEncoders = Ffmpeg.GetAvailableEncoderSet().Audio;

		foreach (var fileName in testSet)
		{
			var targetPath = TestHelper.GetTestExampleFile(fileName);
			var outputPath = TestHelper.GetTestOutputFile(fileName);
			TempHelper.ClearTempDirectory(targetPath, TempPath);
			TempHelper.CreateTempDirectory(targetPath, TempPath);

			try
			{
				foreach (var outputAudioEncoder in outputAudioEncoders)
				{
					FileSystem.CopyFile(targetPath, TempHelper.GetTempFilePath(targetPath, TempPath));

					var restored = Ffmpeg.RestoreAudio(targetPath, outputPath, 0, 270, outputAudioEncoder, 100, 100, TempPath);
					Assert.True(restored, $"restore_audio failed for {fileName} with encoder {outputAudioEncoder}");
				}
			}
			finally
			{
				TempHelper.ClearTempDirectory(targetPath, TempPath);
			}
		}
	}

	[MediaFact]
	public void TestReplaceAudio()
	{
		var testSet = new[]
		{
			"target-240p-16khz.avi",
			"target-240p-16khz.m4v",
			"target-240p-16khz.mkv",
			"target-240p-16khz.mov",
			"target-240p-16khz.mp4",
			"target-240p-48khz.mp4",
			"target-240p-16khz.webm"
		};
		var outputAudioEncoders = Ffmpeg.GetAvailableEncoderSet().Audio;
		var sourceMp3 = TestHelper.GetTestExampleFile("source.mp3");
		var sourceWav = TestHelper.GetTestExampleFile("source.wav");

		foreach (var fileName in testSet)
		{
			var targetPath = TestHelper.GetTestExampleFile(fileName);
			var outputPath = TestHelper.GetTestOutputFile(fileName);
			TempHelper.ClearTempDirectory(targetPath, TempPath);
			TempHelper.CreateTempDirectory(targetPath, TempPath);

			try
			{
				foreach (var outputAudioEncoder in outputAudioEncoders)
				{
					FileSystem.CopyFile(targetPath, TempHelper.GetTempFilePath(targetPath, TempPath));

					Assert.True(Ffmpeg.ReplaceAudio(targetPath, sourceMp3, outputPath, outputAudioEncoder, 100, 100, TempPath), $"replace_audio(mp3) failed for {fileName} with encoder {outputAudioEncoder}");
					Assert.True(Ffmpeg.ReplaceAudio(targetPath, sourceWav, outputPath, outputAudioEncoder, 100, 100, TempPath), $"replace_audio(wav) failed for {fileName} with encoder {outputAudioEncoder}");
				}
			}
			finally
			{
				TempHelper.ClearTempDirectory(targetPath, TempPath);
			}
		}
	}

	/// <summary>
	/// Python: <c>vision_frame.std()</c> / <c>.max()</c> on a <c>(H, W, C) uint8</c> numpy
	/// array — both flatten across every channel of every pixel into one population
	/// (numpy's default <c>ddof = 0</c>). <see cref="Cv2.MeanStdDev"/> computes std
	/// per-channel instead, which is not the same statistic when channel variances differ,
	/// so this reads every pixel back and reduces over the flattened byte values directly to
	/// match numpy's semantics exactly.
	/// </summary>
	private static (double Std, double Max) ComputeFrameStats(Mat frame)
	{
		frame.GetArray(out Vec3b[] pixels);
		var count = pixels.Length * 3;
		double sum = 0;
		byte max = 0;

		foreach (var pixel in pixels)
		{
			sum += pixel.Item0 + pixel.Item1 + pixel.Item2;
			max = Math.Max(max, Math.Max(pixel.Item0, Math.Max(pixel.Item1, pixel.Item2)));
		}

		var mean = sum / count;
		double squaredDiffSum = 0;

		foreach (var pixel in pixels)
		{
			squaredDiffSum += Sq(pixel.Item0 - mean) + Sq(pixel.Item1 - mean) + Sq(pixel.Item2 - mean);
		}

		var std = Math.Sqrt(squaredDiffSum / count);
		return (std, max);
	}

	private static double Sq(double value) => value * value;

	// --- Ffmpeg.FixAudioEncoder / FixVideoEncoder (pure, no I/O) -----------------------

	[Fact]
	public void TestFixAudioEncoder()
	{
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Avi, AudioEncoder.Libopus));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.M4v, AudioEncoder.Libopus));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Mpeg, AudioEncoder.Libopus));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Wmv, AudioEncoder.Libopus));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Mov, AudioEncoder.Flac));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Mov, AudioEncoder.Libopus));
		Assert.Equal(AudioEncoder.PcmS16le, Ffmpeg.FixAudioEncoder(VideoFormat.Mxf, AudioEncoder.Libopus));
		Assert.Equal(AudioEncoder.Libopus, Ffmpeg.FixAudioEncoder(VideoFormat.Webm, AudioEncoder.Aac));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Mp4, AudioEncoder.Aac));
		Assert.Equal(AudioEncoder.Aac, Ffmpeg.FixAudioEncoder(VideoFormat.Avi, AudioEncoder.Aac));
	}

	[Fact]
	public void TestFixVideoEncoder()
	{
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.M4v, VideoEncoder.Libx265));
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.Mpeg, VideoEncoder.Libx265));
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.Mxf, VideoEncoder.Libx265));
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.Wmv, VideoEncoder.Libx265));
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.Mkv, VideoEncoder.Rawvideo));
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.Mp4, VideoEncoder.Rawvideo));
		Assert.Equal(VideoEncoder.Libx264, Ffmpeg.FixVideoEncoder(VideoFormat.Mov, VideoEncoder.LibvpxVp9));
		Assert.Equal(VideoEncoder.LibvpxVp9, Ffmpeg.FixVideoEncoder(VideoFormat.Webm, VideoEncoder.Libx264));
		Assert.Equal(VideoEncoder.Libx265, Ffmpeg.FixVideoEncoder(VideoFormat.Mp4, VideoEncoder.Libx265));
		Assert.Equal(VideoEncoder.Rawvideo, Ffmpeg.FixVideoEncoder(VideoFormat.Avi, VideoEncoder.Rawvideo));
	}

	// --- Ffmpeg.TryParseFrameNumber (pulled out of run_ffmpeg_with_progress) ---------

	[Fact]
	public void TestTryParseFrameNumberBasicLine()
	{
		Assert.Equal(123, Ffmpeg.TryParseFrameNumber("frame=123"));
	}

	[Fact]
	public void TestTryParseFrameNumberWithTrailingNewline()
	{
		// ReadLine() itself strips the newline, but guard the parser against one anyway
		// since Python's decoded readline() keeps it and int() tolerates it.
		Assert.Equal(45, Ffmpeg.TryParseFrameNumber("frame=45\n"));
	}

	[Fact]
	public void TestTryParseFrameNumberZero()
	{
		Assert.Equal(0, Ffmpeg.TryParseFrameNumber("frame=0"));
	}

	[Fact]
	public void TestTryParseFrameNumberNoMarker()
	{
		Assert.Null(Ffmpeg.TryParseFrameNumber("fps=25.00"));
		Assert.Null(Ffmpeg.TryParseFrameNumber("progress=continue"));
	}

	[Fact]
	public void TestTryParseFrameNumberMalformedTrailer()
	{
		// Deviation from Python (documented on the method): a malformed trailer returns
		// null instead of throwing.
		Assert.Null(Ffmpeg.TryParseFrameNumber("frame=not-a-number"));
	}

	// --- Ffmpeg.ParseEncoderLine (pulled out of get_available_encoder_set) -----------

	[Fact]
	public void TestParseEncoderLineAudio()
	{
		var result = Ffmpeg.ParseEncoderLine(" a..... aac                  aac (advanced audio coding)");

		Assert.NotNull(result);
		Assert.Equal(Ffmpeg.EncoderLineKind.Audio, result.Value.Kind);
		Assert.Equal("aac", result.Value.EncoderName);
	}

	[Fact]
	public void TestParseEncoderLineVideo()
	{
		var result = Ffmpeg.ParseEncoderLine(" v..... libx264              h.264 / avc / mpeg-4 avc");

		Assert.NotNull(result);
		Assert.Equal(Ffmpeg.EncoderLineKind.Video, result.Value.Kind);
		Assert.Equal("libx264", result.Value.EncoderName);
	}

	[Fact]
	public void TestParseEncoderLineUnrelatedLine()
	{
		Assert.Null(Ffmpeg.ParseEncoderLine("encoders:"));
		Assert.Null(Ffmpeg.ParseEncoderLine(" s..... some_subtitle_encoder"));
	}

	[Fact]
	public void TestParseEncoderLineTooFewTokens()
	{
		Assert.Null(Ffmpeg.ParseEncoderLine(" a"));
	}

	// --- process-launching surface, binary forced absent via the ffmpegPath override ---
	// (see FfmpegBuilder.Run's doc comment: ffmpeg may genuinely be installed now, so the
	// "not found" branch is forced deterministically rather than assumed from the machine.)

	[Fact]
	public void TestRunFfmpegReturnsNullWhenBinaryNotFound()
	{
		var process = Ffmpeg.RunFfmpeg(FfmpegBuilder.SetInput("media.mp4"), ffmpegPath: TestHelper.BogusBinaryPath);

		Assert.Null(process);
	}

	[Fact]
	public void TestRunFfmpegWithProgressReturnsNullWhenBinaryNotFound()
	{
		var updates = new List<int>();
		var process = Ffmpeg.RunFfmpegWithProgress(FfmpegBuilder.SetInput("media.mp4"), updates.Add, ffmpegPath: TestHelper.BogusBinaryPath);

		Assert.Null(process);
		Assert.Empty(updates);
	}

	[Fact]
	public void TestOpenFfmpegReturnsNullWhenBinaryNotFound()
	{
		var process = Ffmpeg.OpenFfmpeg(FfmpegBuilder.SetInput("media.mp4"), TestHelper.BogusBinaryPath);

		Assert.Null(process);
	}

	[Fact]
	public void TestGetAvailableEncoderSetGracefulWhenBinaryNotFound()
	{
		var encoderSet = Ffmpeg.GetAvailableEncoderSet(ffmpegPath: TestHelper.BogusBinaryPath);

		Assert.Empty(encoderSet.Audio);
		Assert.Empty(encoderSet.Video);
	}
}
