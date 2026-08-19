using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port-adjacent coverage for <see cref="BatchRunner"/> — Phase 6's <c>batch-run</c> path
/// (<c>process_batch</c>): glob expansion via <c>--target-pattern</c>/<c>--source-pattern</c>
/// and <c>--output-pattern</c> placeholder formatting, including the "unknown placeholder
/// returns 1 instead of throwing" behaviour Python gets from
/// <c>except KeyError: return 1</c>.
/// </summary>
[Collection("NativeInference")]
public sealed class BatchRunnerTests
{
	public BatchRunnerTests()
	{
		TestHelper.PrepareTestOutputDirectory();
	}

	private static string? FindRepoRoot()
	{
		var directory = new DirectoryInfo(AppContext.BaseDirectory);

		while (directory is not null)
		{
			if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
			{
				return directory.FullName;
			}

			directory = directory.Parent;
		}

		return null;
	}

	private static bool ModelAvailable(string modelFileName)
	{
		var repoRoot = FindRepoRoot();
		var modelPath = repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models", modelFileName);
		return modelPath is not null && File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
	}

	private sealed class EndToEndTestAttribute : FactAttribute
	{
		public EndToEndTestAttribute()
		{
			if (!TestHelper.HasFfmpeg || !TestHelper.HasFfprobe || !TestHelper.ExamplesAvailable)
			{
				Skip = TestHelper.MissingMediaMessage;
			}
			else if (!ModelAvailable("ddcolor.onnx") || !ModelAvailable("nsfw_1.onnx") || !ModelAvailable("nsfw_2.onnx") || !ModelAvailable("nsfw_3.onnx"))
			{
				Skip = "requires .assets/models/ddcolor.onnx and the nsfw_*.onnx content-analyser models (gitignored, not present in CI)";
			}
		}
	}

	private static string NewJobsPath()
		=> Path.Combine(Path.GetTempPath(), $"facefusion-batch-runner-jobs-{Guid.NewGuid():N}");

	[EndToEndTest]
	public void ProcessBatchExpandsTargetsOnlyPatternAndFormatsOutputPath()
	{
		var glob = Path.Combine(Path.GetTempPath(), $"facefusion-batch-runner-targets-{Guid.NewGuid():N}");
		Directory.CreateDirectory(glob);
		var targetPath = Path.Combine(glob, "target-240p.mp4");
		File.Copy(TestHelper.GetTestExampleFile("target-240p.mp4"), targetPath);

		var outputDirectory = TestHelper.GetTestOutputsDirectory();
		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["target_pattern"] = Path.Combine(glob, "*.mp4"),
				["output_pattern"] = Path.Combine(outputDirectory, "{target_name}-{index}{target_extension}"),
				["processors"] = new[] { "frame_colorizer" },
				["trim_frame_end"] = 4,
			};

			var errorCode = BatchRunner.ProcessBatch(args, jobManager, new Logger());

			Assert.Equal(0, errorCode);

			var expectedOutputPath = Path.Combine(outputDirectory, "target-240p-0.mp4");
			Assert.True(FileSystem.IsFile(expectedOutputPath), $"expected an output file at '{expectedOutputPath}'");
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
			FileSystem.RemoveDirectory(glob);
		}
	}

	[Fact]
	public void ProcessBatchReturnsOneForAnUnknownOutputPatternPlaceholder()
	{
		var glob = Path.Combine(Path.GetTempPath(), $"facefusion-batch-runner-targets-{Guid.NewGuid():N}");
		Directory.CreateDirectory(glob);
		var targetPath = Path.Combine(glob, "target.txt");
		File.WriteAllBytes(targetPath, Array.Empty<byte>());

		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["target_pattern"] = Path.Combine(glob, "*.txt"),
				["output_pattern"] = "/tmp/{not_a_real_placeholder}.mp4",
			};

			var errorCode = BatchRunner.ProcessBatch(args, jobManager, new Logger());

			Assert.Equal(1, errorCode);
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
			FileSystem.RemoveDirectory(glob);
		}
	}

	[Fact]
	public void ProcessBatchReturnsOneWhenNeitherPatternResolvesAnyFiles()
	{
		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["target_pattern"] = Path.Combine(Path.GetTempPath(), $"no-such-directory-{Guid.NewGuid():N}", "*.mp4"),
			};

			var errorCode = BatchRunner.ProcessBatch(args, jobManager, new Logger());

			Assert.Equal(1, errorCode);
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
		}
	}
}
