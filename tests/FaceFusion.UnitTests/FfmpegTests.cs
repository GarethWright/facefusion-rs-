using FaceFusion.Media;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of tests/test_ffmpeg.py.
///
/// Almost every Python test in this suite downloads example media over the network,
/// invokes the real ffmpeg/ffprobe binaries, and asserts on the resulting files or pixel
/// data. Neither the binaries nor the example media are available in this container
/// (network egress restricted, binaries not installed), so those tests are ported below
/// but marked <c>Skip</c> per port convention rule 2 — not silently dropped.
///
/// <c>test_fix_audio_encoder</c> and <c>test_fix_video_encoder</c> are pure lookup tables
/// with no I/O, so they are ported and run for real. The <c>-progress</c> line parsing and
/// <c>-encoders</c> line parsing pulled out of <c>run_ffmpeg_with_progress</c> and
/// <c>get_available_encoder_set</c> have no standalone Python test (they were only
/// exercised indirectly through real ffmpeg output) but are, per the assignment brief, the
/// part of this module most worth testing directly — they are covered below too.
/// </summary>
public sealed class FfmpegTests
{
	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestGetAvailableEncoderSet()
	{
		// Python: test_get_available_encoder_set — asserts 'aac' in audio encoders and
		// 'libx264' in video encoders.
	}

	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestExtractFrames()
	{
		// Python: test_extract_frames — extracts trimmed frame ranges at several fps and
		// asserts frame counts/pixel statistics.
	}

	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestMergeVideo()
	{
		// Python: test_merge_video — merges extracted frames back into a video per
		// available video encoder and asserts the resulting color_transfer.
	}

	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestConcatVideo()
	{
		// Python: test_concat_video — concatenates two copies of the same clip.
	}

	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestReadAudioBuffer()
	{
		// Python: test_read_audio_buffer — asserts a bytes buffer for valid audio and
		// None for a missing/invalid source.
	}

	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestRestoreAudio()
	{
		// Python: test_restore_audio — restores audio into re-encoded video across every
		// available audio encoder and output container.
	}

	[Fact(Skip = "requires ffmpeg and example media (network download restricted in this container)")]
	public void TestReplaceAudio()
	{
		// Python: test_replace_audio — replaces audio in-place across every available
		// audio encoder and output container.
	}

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

	// --- process-launching surface, without a binary ----------------------------------

	[Fact]
	public void TestRunFfmpegReturnsNullWhenBinaryNotFound()
	{
		// ffmpeg is not installed in this container, so FfmpegBuilder.Run resolves the
		// executable path to null. RunFfmpeg must degrade to null instead of throwing.
		var process = Ffmpeg.RunFfmpeg(FfmpegBuilder.SetInput("media.mp4"));

		Assert.Null(process);
	}

	[Fact]
	public void TestRunFfmpegWithProgressReturnsNullWhenBinaryNotFound()
	{
		var updates = new List<int>();
		var process = Ffmpeg.RunFfmpegWithProgress(FfmpegBuilder.SetInput("media.mp4"), updates.Add);

		Assert.Null(process);
		Assert.Empty(updates);
	}

	[Fact]
	public void TestOpenFfmpegReturnsNullWhenBinaryNotFound()
	{
		var process = Ffmpeg.OpenFfmpeg(FfmpegBuilder.SetInput("media.mp4"));

		Assert.Null(process);
	}

	[Fact]
	public void TestGetAvailableEncoderSetGracefulWhenBinaryNotFound()
	{
		var encoderSet = Ffmpeg.GetAvailableEncoderSet();

		Assert.Empty(encoderSet.Audio);
		Assert.Empty(encoderSet.Video);
	}
}
