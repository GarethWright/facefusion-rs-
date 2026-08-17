using System;
using System.Collections.Generic;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

public class CurlBuilderTests
{
	// Metadata values mirroring facefusion/metadata.py
	private const string MetadataName = "FaceFusion";
	private const string MetadataVersion = "3.8.2";
	private const string MetadataUrl = "https://facefusion.io";

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
		var userAgent = MetadataName + "/" + MetadataVersion;
		var result = CurlBuilder.Run(Array.Empty<string>());
		var expected = new List<string?>
		{
			Which("curl"),
			"--user-agent",
			userAgent,
			"--location",
			"--silent",
			"--ssl-no-revoke"
		};

		Assert.Equal(expected, result);
	}

	[Fact]
	public void TestChain()
	{
		var result = CurlBuilder.Chain(
			CurlBuilder.Ping(MetadataUrl),
			CurlBuilder.SetTimeout(5)
		);

		var expected = new[] { "-I", MetadataUrl, "--connect-timeout", "5" };

		Assert.Equal(expected, result);
	}
}
