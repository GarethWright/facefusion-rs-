using System;
using System.IO;
using System.Linq;
using FaceFusion.Core;

namespace FaceFusion.UnitTests;

public class ProcessHelperTests
{
	[Fact]
	public void TestWhichFindsExecutableOnPath()
	{
		// curl is present on PATH in the test/CI container.
		var result = ProcessHelper.Which("curl");

		Assert.NotNull(result);
		Assert.True(File.Exists(result));
		Assert.Equal("curl", Path.GetFileName(result));
	}

	[Fact]
	public void TestWhichFindsFfmpegAndFfprobeWhenInstalled()
	{
		// ffmpeg/ffprobe are installed in this container (see docs/PORT_CONVENTIONS.md /
		// the Phase 2 milestone), so the "not found" case for these two specific names is
		// no longer a property of this environment and cannot be asserted here — see
		// TestWhichReturnsNullForBogusExecutable below for the "not found" path, exercised
		// with a name that genuinely can never resolve on any machine.
		if (TestHelper.HasFfmpeg)
		{
			Assert.NotNull(ProcessHelper.Which("ffmpeg"));
		}

		if (TestHelper.HasFfprobe)
		{
			Assert.NotNull(ProcessHelper.Which("ffprobe"));
		}
	}

	[Fact]
	public void TestWhichReturnsNullForBogusExecutable()
	{
		// A GUID-based name can never collide with a real installed tool, unlike a
		// plausible-sounding placeholder string, so this holds regardless of what happens
		// to be on PATH in any given environment.
		var bogusExecutableName = "this-executable-does-not-exist-" + Guid.NewGuid().ToString("N");

		Assert.Null(ProcessHelper.Which(bogusExecutableName));
	}

	[Fact]
	public void TestWhichReturnsFullPathNotBareName()
	{
		var result = ProcessHelper.Which("curl");

		Assert.NotNull(result);
		Assert.True(Path.IsPathRooted(result));
	}

	[Fact]
	public void TestWhichIsConsistentWithManualPathSearch()
	{
		// Cross-check against an independent PATH walk (not the production implementation)
		// so this test is not vacuous against ProcessHelper.Which itself.
		var pathVariable = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
		var expected = pathVariable
			.Split(Path.PathSeparator)
			.Where(directory => !string.IsNullOrEmpty(directory))
			.Select(directory => Path.Combine(directory, "curl"))
			.FirstOrDefault(File.Exists);

		Assert.Equal(expected, ProcessHelper.Which("curl"));
	}
}
