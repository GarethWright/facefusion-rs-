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

		// On Windows the resolved file carries a PATHEXT extension (curl.EXE), because
		// ProcessHelper.Which appends PATHEXT candidates exactly as shutil.which does. Asserting
		// the bare name here is a POSIX assumption, not a property of Which.
		Assert.Equal("curl", Path.GetFileNameWithoutExtension(result), ignoreCase: true);
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
		// so this test is not vacuous against ProcessHelper.Which itself. The walk has to model
		// PATHEXT on Windows for the same reason Which does: there is no bare "curl" on PATH
		// there, only curl.EXE, so a POSIX-only walk found nothing and expected null while Which
		// correctly returned C:\Windows\system32\curl.EXE.
		var candidateNames = OperatingSystem.IsWindows()
			? new[] { "curl" }
				.Concat((Environment.GetEnvironmentVariable("PATHEXT") ?? ".COM;.EXE;.BAT;.CMD")
					.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
					.Select(extension => "curl" + extension))
				.ToArray()
			: new[] { "curl" };

		var pathVariable = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
		var expected = pathVariable
			.Split(Path.PathSeparator)
			.Where(directory => !string.IsNullOrEmpty(directory))
			.SelectMany(directory => candidateNames.Select(name => Path.Combine(directory, name)))
			.FirstOrDefault(File.Exists);

		Assert.Equal(expected, ProcessHelper.Which("curl"));
	}
}
