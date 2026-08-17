using System;
using System.Collections.Generic;
using System.IO;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

public class FfmpegBuilderTests
{
	// Mirrors shutil.which('ffmpeg') from the Python test. ffmpeg is not installed in the
	// test environment, so this resolves to null on both sides.
	private static string? WhichFfmpeg()
	{
		var pathVariable = Environment.GetEnvironmentVariable("PATH");

		if (string.IsNullOrEmpty(pathVariable))
		{
			return null;
		}

		foreach (var directory in pathVariable.Split(Path.PathSeparator))
		{
			if (string.IsNullOrEmpty(directory))
			{
				continue;
			}

			var candidatePath = Path.Combine(directory, "ffmpeg");

			if (File.Exists(candidatePath))
			{
				return candidatePath;
			}
		}

		return null;
	}

	[Fact]
	public void TestRun()
	{
		Assert.Equal(new[] { WhichFfmpeg(), "-loglevel", "error" }, FfmpegBuilder.Run(Array.Empty<string>()));
	}

	[Fact]
	public void TestChain()
	{
		Assert.Equal(
			new[] { "-i", "input.mp4", "output.mp4" },
			FfmpegBuilder.Chain(
				FfmpegBuilder.SetInput("input.mp4"),
				FfmpegBuilder.SetOutput("output.mp4")));

		Assert.Equal(
			new[] { "-c:v", "libx264", "-vf", "fps=30", "-c:a", "aac" },
			FfmpegBuilder.Chain(
				FfmpegBuilder.SetVideoEncoder("libx264"),
				FfmpegBuilder.SetVideoFps(30),
				FfmpegBuilder.SetAudioEncoder("aac")));
	}

	[Fact]
	public void TestConcat()
	{
		Assert.Equal(
			new[] { "-c:v", "libvpx-vp9", "-vf", "fps=30" },
			FfmpegBuilder.Concat(
				FfmpegBuilder.SetVideoEncoder("libvpx-vp9"),
				FfmpegBuilder.SetVideoFps(30)));

		Assert.Equal(
			new[] { "-c:v", "libvpx-vp9", "-vf", "fps=30,format=yuva420p" },
			FfmpegBuilder.Concat(
				FfmpegBuilder.SetVideoEncoder("libvpx-vp9"),
				FfmpegBuilder.SetVideoFps(30),
				FfmpegBuilder.KeepVideoAlpha("libvpx-vp9")));

		Assert.Equal(
			new[] { "-vf", "trim=start_frame=0:end_frame=100,fps=30,format=yuva420p" },
			FfmpegBuilder.Concat(
				FfmpegBuilder.SelectFrameRange(0, 100, 30),
				FfmpegBuilder.KeepVideoAlpha("libvpx-vp9")));
	}

	[Fact]
	public void TestSetStreamMode()
	{
		Assert.Equal(new[] { "-f", "mpegts" }, FfmpegBuilder.SetStreamMode("udp"));
		Assert.Equal(new[] { "-f", "v4l2" }, FfmpegBuilder.SetStreamMode("v4l2"));
	}

	[Fact]
	public void TestSeekTo()
	{
		Assert.Equal(new[] { "-ss", "0.0" }, FfmpegBuilder.SeekTo(0.0));
		Assert.Equal(new[] { "-ss", "1.5" }, FfmpegBuilder.SeekTo(1.5));
	}

	[Fact]
	public void TestSetOutputFormat()
	{
		Assert.Equal(new[] { "-f", "rawvideo" }, FfmpegBuilder.SetOutputFormat("rawvideo"));
	}

	[Fact]
	public void TestSelectFrameRange()
	{
		Assert.Equal(new[] { "-vf", "trim=start_frame=0,fps=30" }, FfmpegBuilder.SelectFrameRange(0, null, 30));
		Assert.Equal(new[] { "-vf", "trim=end_frame=100,fps=30" }, FfmpegBuilder.SelectFrameRange(null, 100, 30));
		Assert.Equal(new[] { "-vf", "trim=start_frame=0:end_frame=100,fps=30" }, FfmpegBuilder.SelectFrameRange(0, 100, 30));
		Assert.Equal(new[] { "-vf", "fps=30" }, FfmpegBuilder.SelectFrameRange(null, null, 30));
	}

	[Fact]
	public void TestRestrictColorTransfer()
	{
		Assert.Equal(new[] { "-vf", "scale=out_primaries=bt709:out_transfer=bt709:intent=perceptual" }, FfmpegBuilder.RestrictColorTransfer("smpte2084"));
		Assert.Equal(new[] { "-vf", "scale=out_primaries=bt709:out_transfer=bt709:intent=perceptual" }, FfmpegBuilder.RestrictColorTransfer("arib-std-b67"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.RestrictColorTransfer("invalid"));
	}

	[Fact]
	public void TestConvertColorSpace()
	{
		Assert.Equal(new[] { "-vf", "scale=out_color_matrix=bt601:out_range=tv,setparams=colorspace=bt601:color_primaries=bt601:color_trc=bt601" }, FfmpegBuilder.ConvertColorSpace("bt601"));
		Assert.Equal(new[] { "-vf", "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709" }, FfmpegBuilder.ConvertColorSpace("bt709"));
		Assert.Equal(new[] { "-vf", "scale=out_color_matrix=bt2020:out_range=tv,setparams=colorspace=bt2020:color_primaries=bt2020:color_trc=bt2020" }, FfmpegBuilder.ConvertColorSpace("bt2020"));
	}

	[Fact]
	public void TestSetAudioSampleSize()
	{
		Assert.Equal(new[] { "-f", "s16le" }, FfmpegBuilder.SetAudioSampleSize(16));
		Assert.Equal(new[] { "-f", "s32le" }, FfmpegBuilder.SetAudioSampleSize(32));
	}

	[Fact]
	public void TestSetAudioQuality()
	{
		Assert.Equal(new[] { "-q:a", "0.1" }, FfmpegBuilder.SetAudioQuality("aac", 0));
		Assert.Equal(new[] { "-q:a", "1.0" }, FfmpegBuilder.SetAudioQuality("aac", 50));
		Assert.Equal(new[] { "-q:a", "2.0" }, FfmpegBuilder.SetAudioQuality("aac", 100));
		Assert.Equal(new[] { "-q:a", "9" }, FfmpegBuilder.SetAudioQuality("libmp3lame", 0));
		Assert.Equal(new[] { "-q:a", "4" }, FfmpegBuilder.SetAudioQuality("libmp3lame", 50));
		Assert.Equal(new[] { "-q:a", "0" }, FfmpegBuilder.SetAudioQuality("libmp3lame", 100));
		Assert.Equal(new[] { "-b:a", "64k" }, FfmpegBuilder.SetAudioQuality("libopus", 0));
		Assert.Equal(new[] { "-b:a", "160k" }, FfmpegBuilder.SetAudioQuality("libopus", 50));
		Assert.Equal(new[] { "-b:a", "256k" }, FfmpegBuilder.SetAudioQuality("libopus", 100));
		Assert.Equal(new[] { "-q:a", "-1.0" }, FfmpegBuilder.SetAudioQuality("libvorbis", 0));
		Assert.Equal(new[] { "-q:a", "4.5" }, FfmpegBuilder.SetAudioQuality("libvorbis", 50));
		Assert.Equal(new[] { "-q:a", "10.0" }, FfmpegBuilder.SetAudioQuality("libvorbis", 100));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetAudioQuality("flac", 0));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetAudioQuality("flac", 50));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetAudioQuality("flac", 100));
	}

	[Fact]
	public void TestSetThreadCount()
	{
		Assert.Equal(new[] { "-threads", "8" }, FfmpegBuilder.SetThreadCount(8));
		Assert.Equal(new[] { "-threads", "16" }, FfmpegBuilder.SetThreadCount(16));
	}

	[Fact]
	public void TestSetFaststart()
	{
		Assert.Equal(new[] { "-movflags", "+faststart" }, FfmpegBuilder.SetFaststart("m4v"));
		Assert.Equal(new[] { "-movflags", "+faststart" }, FfmpegBuilder.SetFaststart("mov"));
		Assert.Equal(new[] { "-movflags", "+faststart" }, FfmpegBuilder.SetFaststart("mp4"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetFaststart("mkv"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetFaststart("webm"));
	}

	[Fact]
	public void TestSetVideoTag()
	{
		Assert.Equal(new[] { "-tag:v", "hvc1" }, FfmpegBuilder.SetVideoTag("libx265", "m4v"));
		Assert.Equal(new[] { "-tag:v", "hvc1" }, FfmpegBuilder.SetVideoTag("hevc_nvenc", "mov"));
		Assert.Equal(new[] { "-tag:v", "hvc1" }, FfmpegBuilder.SetVideoTag("hevc_videotoolbox", "mp4"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetVideoTag("libx265", "mkv"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetVideoTag("libx265", "webm"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetVideoTag("libx264", "mp4"));
		Assert.Equal(Array.Empty<string>(), FfmpegBuilder.SetVideoTag("h264_nvenc", "mp4"));
	}

	[Fact]
	public void TestSetVideoQuality()
	{
		Assert.Equal(new[] { "-crf", "51" }, FfmpegBuilder.SetVideoQuality("libx264", 0));
		Assert.Equal(new[] { "-crf", "26" }, FfmpegBuilder.SetVideoQuality("libx264", 50));
		Assert.Equal(new[] { "-crf", "0" }, FfmpegBuilder.SetVideoQuality("libx264", 100));
		Assert.Equal(new[] { "-crf", "51" }, FfmpegBuilder.SetVideoQuality("libx264rgb", 0));
		Assert.Equal(new[] { "-crf", "26" }, FfmpegBuilder.SetVideoQuality("libx264rgb", 50));
		Assert.Equal(new[] { "-crf", "0" }, FfmpegBuilder.SetVideoQuality("libx264rgb", 100));
		Assert.Equal(new[] { "-crf", "51" }, FfmpegBuilder.SetVideoQuality("libx265", 0));
		Assert.Equal(new[] { "-crf", "26" }, FfmpegBuilder.SetVideoQuality("libx265", 50));
		Assert.Equal(new[] { "-crf", "0" }, FfmpegBuilder.SetVideoQuality("libx265", 100));
		Assert.Equal(new[] { "-crf", "63" }, FfmpegBuilder.SetVideoQuality("libvpx-vp9", 0));
		Assert.Equal(new[] { "-crf", "32" }, FfmpegBuilder.SetVideoQuality("libvpx-vp9", 50));
		Assert.Equal(new[] { "-crf", "0" }, FfmpegBuilder.SetVideoQuality("libvpx-vp9", 100));
		Assert.Equal(new[] { "-cq", "51" }, FfmpegBuilder.SetVideoQuality("h264_nvenc", 0));
		Assert.Equal(new[] { "-cq", "26" }, FfmpegBuilder.SetVideoQuality("h264_nvenc", 50));
		Assert.Equal(new[] { "-cq", "0" }, FfmpegBuilder.SetVideoQuality("h264_nvenc", 100));
		Assert.Equal(new[] { "-cq", "51" }, FfmpegBuilder.SetVideoQuality("hevc_nvenc", 0));
		Assert.Equal(new[] { "-cq", "26" }, FfmpegBuilder.SetVideoQuality("hevc_nvenc", 50));
		Assert.Equal(new[] { "-cq", "0" }, FfmpegBuilder.SetVideoQuality("hevc_nvenc", 100));
		Assert.Equal(new[] { "-qp_i", "51", "-qp_p", "51", "-qp_b", "51" }, FfmpegBuilder.SetVideoQuality("h264_amf", 0));
		Assert.Equal(new[] { "-qp_i", "26", "-qp_p", "26", "-qp_b", "26" }, FfmpegBuilder.SetVideoQuality("h264_amf", 50));
		Assert.Equal(new[] { "-qp_i", "0", "-qp_p", "0", "-qp_b", "0" }, FfmpegBuilder.SetVideoQuality("h264_amf", 100));
		Assert.Equal(new[] { "-qp_i", "51", "-qp_p", "51", "-qp_b", "51" }, FfmpegBuilder.SetVideoQuality("hevc_amf", 0));
		Assert.Equal(new[] { "-qp_i", "26", "-qp_p", "26", "-qp_b", "26" }, FfmpegBuilder.SetVideoQuality("hevc_amf", 50));
		Assert.Equal(new[] { "-qp_i", "0", "-qp_p", "0", "-qp_b", "0" }, FfmpegBuilder.SetVideoQuality("hevc_amf", 100));
		Assert.Equal(new[] { "-qp", "51" }, FfmpegBuilder.SetVideoQuality("h264_qsv", 0));
		Assert.Equal(new[] { "-qp", "26" }, FfmpegBuilder.SetVideoQuality("h264_qsv", 50));
		Assert.Equal(new[] { "-qp", "0" }, FfmpegBuilder.SetVideoQuality("h264_qsv", 100));
		Assert.Equal(new[] { "-qp", "51" }, FfmpegBuilder.SetVideoQuality("hevc_qsv", 0));
		Assert.Equal(new[] { "-qp", "26" }, FfmpegBuilder.SetVideoQuality("hevc_qsv", 50));
		Assert.Equal(new[] { "-qp", "0" }, FfmpegBuilder.SetVideoQuality("hevc_qsv", 100));
		Assert.Equal(new[] { "-b:v", "1024k" }, FfmpegBuilder.SetVideoQuality("h264_videotoolbox", 0));
		Assert.Equal(new[] { "-b:v", "25768k" }, FfmpegBuilder.SetVideoQuality("h264_videotoolbox", 50));
		Assert.Equal(new[] { "-b:v", "50512k" }, FfmpegBuilder.SetVideoQuality("h264_videotoolbox", 100));
		Assert.Equal(new[] { "-b:v", "1024k" }, FfmpegBuilder.SetVideoQuality("hevc_videotoolbox", 0));
		Assert.Equal(new[] { "-b:v", "25768k" }, FfmpegBuilder.SetVideoQuality("hevc_videotoolbox", 50));
		Assert.Equal(new[] { "-b:v", "50512k" }, FfmpegBuilder.SetVideoQuality("hevc_videotoolbox", 100));
	}
}
