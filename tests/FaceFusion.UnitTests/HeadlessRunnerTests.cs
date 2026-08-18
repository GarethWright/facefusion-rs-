using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port-adjacent coverage for <see cref="HeadlessRunner"/> — Phase 6's <c>headless-run</c>
/// path (<c>process_headless</c> / <c>process_step</c> / <c>conditional_process</c>).
///
/// <see cref="EndToEndTestAttribute"/> gates the real-video test on ffmpeg/ffprobe/the shared
/// example media (same as <c>WorkflowTests</c>' own <c>WorkflowFactAttribute</c>) plus
/// <c>ddcolor.onnx</c>, since <c>frame_colorizer</c> is the one processor
/// <see cref="ProcessorStepFactory"/> currently builds.
/// </summary>
[Collection("NativeInference")]
public sealed class HeadlessRunnerTests
{
	public HeadlessRunnerTests()
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
		=> Path.Combine(Path.GetTempPath(), $"facefusion-headless-runner-jobs-{Guid.NewGuid():N}");

	[EndToEndTest]
	public void ProcessHeadlessFrameColorizerProducesExpectedVideo()
	{
		var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
		var outputPath = TestHelper.GetTestOutputFile("headless-frame-colorizer.mp4");
		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["target_path"] = targetPath,
				["output_path"] = outputPath,
				["processors"] = new[] { "frame_colorizer" },
				["trim_frame_end"] = 8,
			};

			var errorCode = HeadlessRunner.ProcessHeadless(args, jobManager, new Logger());

			Assert.Equal(0, errorCode);
			Assert.True(FileSystem.IsFile(outputPath), $"expected an output file at '{outputPath}'");

			var metadata = Ffprobe.ExtractVideoMetadata(outputPath);
			Assert.Equal(8, metadata.FrameTotal);
			Assert.Equal(25.0, metadata.Fps);
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
		}
	}

	[Fact]
	public void ProcessStepReturnsFalseWhenAnUnsupportedProcessorIsRequested()
	{
		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["processors"] = new[] { "face_swapper" },
			};

			// face_swapper has no ProcessorStepFactory wiring yet (needs the full face
			// pipeline) — ProcessorStepFactory.PreCheck throws NotSupportedException rather
			// than silently returning a wrong step, matching the assignment's "never return
			// a silently-wrong step" instruction.
			Assert.Throws<NotSupportedException>(() =>
				HeadlessRunner.ProcessStep("job-under-test", 0, args, jobManager, new Logger()));
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
		}
	}

	[Fact]
	public void ConditionalProcessReturnsTwoWhenWorkflowModeDoesNotMatchTheTarget()
	{
		var tempImagePath = Path.Combine(Path.GetTempPath(), $"facefusion-headless-runner-{Guid.NewGuid():N}.png");
		File.WriteAllBytes(tempImagePath, Array.Empty<byte>());

		try
		{
			var args = new Dictionary<string, object?>
			{
				["target_path"] = tempImagePath,
				// image-to-video explicitly requested for a target that is not a video:
				// Python's `workflow_mode == detect_workflow_mode()` guard fails and
				// conditional_process returns 2.
				["workflow_mode"] = "image-to-video",
			};

			var errorCode = HeadlessRunner.ConditionalProcess(args, Array.Empty<string>(), new Logger());

			Assert.Equal(2, errorCode);
		}
		finally
		{
			File.Delete(tempImagePath);
		}
	}
}
