using System;
using System.Collections.Generic;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

public class FfprobeBuilderTests
{
	/// <summary>
	/// Helper equivalent to Python's shutil.which()
	/// </summary>
	private static string? Which(string executable)
	{
		var pathEnvVar = Environment.GetEnvironmentVariable("PATH") ?? "";
		var pathDirs = pathEnvVar.Split(Path.PathSeparator);

		foreach (var dir in pathDirs)
		{
			var fullPath = Path.Combine(dir, executable);
			if (File.Exists(fullPath))
			{
				return fullPath;
			}
		}

		return null;
	}

	[Fact]
	public void TestRun()
	{
		var result = FfprobeBuilder.Run(new[] { "-v", "error" });
		var expected = new List<string?> { Which("ffprobe"), "-loglevel", "error", "-v", "error" };

		Assert.Equal(expected, result);
	}

	[Fact]
	public void TestChain()
	{
		var result = FfprobeBuilder.Chain(
			FfprobeBuilder.SelectStream("a:0"),
			FfprobeBuilder.ShowStreamEntries(new[] { "sample_rate" }),
			FfprobeBuilder.FormatToKeyValue(),
			FfprobeBuilder.SetInput("audio.mp3")
		);

		var expected = new[] { "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1", "-i", "audio.mp3" };

		Assert.Equal(expected, result);
	}

	[Fact]
	public void TestSelectStream()
	{
		Assert.Equal(new[] { "-select_streams", "a:0" }, FfprobeBuilder.SelectStream("a:0"));
		Assert.Equal(new[] { "-select_streams", "v:0" }, FfprobeBuilder.SelectStream("v:0"));
	}

	[Fact]
	public void TestShowStreamEntries()
	{
		Assert.Equal(
			new[] { "-show_entries", "stream=duration" },
			FfprobeBuilder.ShowStreamEntries(new[] { "duration" })
		);

		Assert.Equal(
			new[] { "-show_entries", "stream=duration,sample_rate" },
			FfprobeBuilder.ShowStreamEntries(new[] { "duration", "sample_rate" })
		);
	}

	[Fact]
	public void TestFormatToKeyValue()
	{
		Assert.Equal(
			new[] { "-of", "default=noprint_wrappers=1" },
			FfprobeBuilder.FormatToKeyValue()
		);
	}

	[Fact]
	public void TestSetInput()
	{
		Assert.Equal(new[] { "-i", "input.mp3" }, FfprobeBuilder.SetInput("input.mp3"));
		Assert.Equal(new[] { "-i", "input.wav" }, FfprobeBuilder.SetInput("input.wav"));
	}
}
