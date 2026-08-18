using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port-adjacent coverage for <see cref="ProcessorStepFactory"/> and
/// <see cref="FacePipelineFactory"/> — Phase 6's face-pipeline processor wiring
/// (<c>background_remover</c>, <c>face_debugger</c>) plus the still-unsupported names
/// (<c>face_swapper</c> and everything after it — see <see cref="ProcessorStepFactory"/>'s
/// class remarks for exactly why each is not wired).
/// </summary>
[Collection("NativeInference")]
public sealed class ProcessorStepFactoryTests
{
	public ProcessorStepFactoryTests()
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

	private static readonly string[] FacePipelineModels =
	{
		"yoloface_8n.onnx", "fan_68_5.onnx", "2dfan4.onnx", "arcface_w600k_r50.onnx", "fairface.onnx",
	};

	/// <summary>Gates a real-video test on ffmpeg/ffprobe/the shared example media (same as
	/// <c>HeadlessRunnerTests.EndToEndTestAttribute</c>) plus whichever <c>.onnx</c> files the
	/// specific processor under test needs — none of these ship in this repo (gitignored), so
	/// every one of these tests skips cleanly rather than failing when they are absent.</summary>
	private sealed class EndToEndTestAttribute : FactAttribute
	{
		public EndToEndTestAttribute(params string[] requiredModels)
		{
			if (!TestHelper.HasFfmpeg || !TestHelper.HasFfprobe || !TestHelper.ExamplesAvailable)
			{
				Skip = TestHelper.MissingMediaMessage;
			}
			else if (requiredModels.Any(model => !ModelAvailable(model)))
			{
				Skip = $"requires {string.Join(", ", requiredModels.Select(m => $".assets/models/{m}"))} (gitignored, not present in CI)";
			}
		}
	}

	private static string NewJobsPath()
		=> Path.Combine(Path.GetTempPath(), $"facefusion-processor-step-factory-jobs-{Guid.NewGuid():N}");

	// -----------------------------------------------------------------
	// Unsupported processors — "never return a silently-wrong step"
	// -----------------------------------------------------------------

	[Theory]
	[InlineData("face_swapper")]
	[InlineData("age_modifier")]
	[InlineData("expression_restorer")]
	[InlineData("deep_swapper")]
	[InlineData("face_editor")]
	[InlineData("lip_syncer")]
	[InlineData("face_enhancer")]
	[InlineData("frame_enhancer")]
	[InlineData("not_a_real_processor")]
	public void PreCheckThrowsNotSupportedForEveryProcessorNotYetWired(string processorName)
	{
		Assert.Throws<NotSupportedException>(() => ProcessorStepFactory.PreCheck(processorName, new Dictionary<string, object?>()));
	}

	[Theory]
	[InlineData("face_swapper")]
	[InlineData("age_modifier")]
	[InlineData("not_a_real_processor")]
	public void BuildThrowsNotSupportedForEveryProcessorNotYetWired(string processorName)
	{
		Assert.Throws<NotSupportedException>(() => ProcessorStepFactory.Build(processorName, new Dictionary<string, object?>()));
	}

	[Fact]
	public void BuildFaceDebuggerWithoutFaceResourcesThrowsInvalidOperation()
	{
		// face_debugger needs a FacePipelineFactory.Resources instance (see
		// ProcessorStepFactory.Build's remarks) — omitting it is a caller bug, not an
		// unsupported-processor situation, so this is a different exception type than the
		// NotSupportedException every genuinely-unwired processor throws.
		var exception = Assert.Throws<InvalidOperationException>(
			() => ProcessorStepFactory.Build("face_debugger", new Dictionary<string, object?>()));
		Assert.Contains("FacePipelineFactory.Resources", exception.Message);
	}

	// -----------------------------------------------------------------
	// PreCheck — file-presence only, no InferenceSession allocation
	// -----------------------------------------------------------------

	[Fact]
	public void PreCheckReturnsFalseWhenTheChosenModelIsNotPresentLocally()
	{
		// modnet.onnx may or may not be present in this environment; background_remover's
		// PreCheck is a pure file check either way (see class remarks — no auto-download),
		// so pointing it at a model family that is never present locally proves the check
		// actually inspects the filesystem instead of trivially returning true.
		var args = new Dictionary<string, object?> { ["background_remover_model"] = "silueta" };
		var repoRoot = FindRepoRoot();

		if (repoRoot is not null && ModelAvailable("silueta.onnx"))
		{
			return; // this one environment happens to have it — nothing to assert either way
		}

		Assert.False(ProcessorStepFactory.PreCheck("background_remover", args));
	}

	[Fact]
	public void FacePipelineFactoryPreCheckAndBuildAgreeOnModelPresence()
	{
		if (!FacePipelineModels.All(ModelAvailable))
		{
			return; // skip cleanly — see class remarks
		}

		var args = new Dictionary<string, object?>();
		Assert.True(FacePipelineFactory.PreCheck(args));

		using var resources = FacePipelineFactory.Build(args);
		Assert.NotEmpty(resources.FaceDetectorSessions);
		Assert.NotNull(resources.Fan685Session);
		Assert.NotNull(resources.FaceRecognizerSession);
		Assert.NotNull(resources.FaceClassifierSession);
	}

	// -----------------------------------------------------------------
	// End-to-end (real headless-run, real video) — Task 3's background_remover fix and
	// Task 2's face_debugger wiring, the two processors this phase actually ships.
	// -----------------------------------------------------------------

	[EndToEndTest("modnet.onnx", "nsfw_1.onnx", "nsfw_2.onnx", "nsfw_3.onnx")]
	public void ProcessHeadlessBackgroundRemoverProducesExpectedVideo()
	{
		var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
		var outputPath = TestHelper.GetTestOutputFile("headless-background-remover.mp4");
		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["target_path"] = targetPath,
				["output_path"] = outputPath,
				["processors"] = new[] { "background_remover" },
				["trim_frame_end"] = 8,
			};

			// Regression coverage for the Cv2.Min type-mismatch bug fixed in
			// BackgroundRemover.cs — this used to throw OpenCvSharp.OpenCVException (and, in
			// the multi-threaded video path, segfault the whole process) on every real frame.
			var errorCode = HeadlessRunner.ProcessHeadless(args, jobManager, new Logger());

			Assert.Equal(0, errorCode);
			Assert.True(FileSystem.IsFile(outputPath), $"expected an output file at '{outputPath}'");

			var metadata = Ffprobe.ExtractVideoMetadata(outputPath);
			Assert.Equal(8, metadata.FrameTotal);
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
		}
	}

	[EndToEndTest("yoloface_8n.onnx", "fan_68_5.onnx", "2dfan4.onnx", "arcface_w600k_r50.onnx", "fairface.onnx", "nsfw_1.onnx", "nsfw_2.onnx", "nsfw_3.onnx")]
	public void ProcessHeadlessFaceDebuggerProducesExpectedVideo()
	{
		var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
		var outputPath = TestHelper.GetTestOutputFile("headless-face-debugger.mp4");
		var jobsPath = NewJobsPath();

		try
		{
			var jobManager = new JobManager(jobsPath);
			Assert.True(jobManager.InitJobs());

			var args = new Dictionary<string, object?>
			{
				["target_path"] = targetPath,
				["output_path"] = outputPath,
				["processors"] = new[] { "face_debugger" },
				["trim_frame_end"] = 8,
			};

			// Exercises the shared face pipeline (FacePipelineFactory) end to end: detector,
			// landmarker, recognizer, classifier all run against every frame of a real video.
			var errorCode = HeadlessRunner.ProcessHeadless(args, jobManager, new Logger());

			Assert.Equal(0, errorCode);
			Assert.True(FileSystem.IsFile(outputPath), $"expected an output file at '{outputPath}'");

			var metadata = Ffprobe.ExtractVideoMetadata(outputPath);
			Assert.Equal(8, metadata.FrameTotal);
		}
		finally
		{
			FileSystem.RemoveDirectory(jobsPath);
		}
	}
}
