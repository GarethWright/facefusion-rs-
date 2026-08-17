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

	[Theory]
	[InlineData("ffmpeg")]
	[InlineData("ffprobe")]
	public void TestWhichReturnsNullWhenNotFound(string executable)
	{
		// ffmpeg/ffprobe are deliberately absent from this container so both the found
		// and not-found paths of shutil.which get exercised.
		Assert.Null(ProcessHelper.Which(executable));
	}

	[Fact]
	public void TestWhichReturnsNullForBogusExecutable()
	{
		Assert.Null(ProcessHelper.Which("this-executable-does-not-exist-anywhere"));
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
