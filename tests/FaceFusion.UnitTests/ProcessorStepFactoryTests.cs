using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port-adjacent coverage for <see cref="ProcessorStepFactory"/> and
/// <see cref="FacePipelineFactory"/> — Phase 6's processor wiring. All eleven of Python's
/// processor modules are now recognised; see <see cref="ProcessorStepFactory"/>'s class
/// remarks for how each was verified against the real Python CLI.
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
	// Processor-name coverage — every module Python ships must be recognised
	// -----------------------------------------------------------------

	/// <summary>Every directory under <c>facefusion/processors/modules/</c>, i.e. exactly the
	/// set <c>--processors</c> accepts. Hardcoded rather than read off disk so the assertion
	/// still means something in CI, where the Python tree is not guaranteed to be present;
	/// <see cref="WiredProcessorsMatchThePythonModuleDirectory"/> catches drift when it
	/// is.</summary>
	public static readonly string[] PythonProcessorNames =
	{
		"age_modifier", "background_remover", "deep_swapper", "expression_restorer", "face_debugger",
		"face_editor", "face_enhancer", "face_swapper", "frame_colorizer", "frame_enhancer", "lip_syncer",
	};

	public static TheoryData<string> ProcessorNames()
	{
		var data = new TheoryData<string>();

		foreach (var name in PythonProcessorNames)
		{
			data.Add(name);
		}

		return data;
	}

	/// <summary>
	/// PreCheck must recognise every processor. It returns false when the model files are
	/// absent (which is the normal case in CI) — what must never happen is
	/// <see cref="NotSupportedException"/>, which would mean the name is unwired.
	/// </summary>
	[Theory]
	[MemberData(nameof(ProcessorNames))]
	public void PreCheckRecognisesEveryPythonProcessor(string processorName)
	{
		var exception = Record.Exception(() => ProcessorStepFactory.PreCheck(processorName, new Dictionary<string, object?>()));

		Assert.False(exception is NotSupportedException, $"'{processorName}' is not wired into ProcessorStepFactory.PreCheck");
	}

	/// <summary>An unknown name is still a named failure rather than a silently-wrong step.</summary>
	[Fact]
	public void PreCheckThrowsNotSupportedForAnUnknownProcessor()
	{
		var exception = Assert.Throws<NotSupportedException>(
			() => ProcessorStepFactory.PreCheck("not_a_real_processor", new Dictionary<string, object?>()));
		Assert.Contains("not_a_real_processor", exception.Message);
	}

	[Fact]
	public void BuildThrowsNotSupportedForAnUnknownProcessor()
	{
		Assert.Throws<NotSupportedException>(
			() => ProcessorStepFactory.Build("not_a_real_processor", new Dictionary<string, object?>()));
	}

	/// <summary>
	/// Drift guard: if Python gains or loses a processor module, this fails rather than the
	/// port silently accepting a stale list. Skips when the Python tree is not checked out.
	/// </summary>
	[Fact]
	public void WiredProcessorsMatchThePythonModuleDirectory()
	{
		var repoRoot = FindRepoRoot();
		var modulesPath = repoRoot is null ? null : Path.Combine(repoRoot, "facefusion", "processors", "modules");

		if (modulesPath is null || !Directory.Exists(modulesPath))
		{
			return; // the Python tree is not present — nothing to compare against
		}

		var pythonNames = Directory.GetDirectories(modulesPath)
			.Select(Path.GetFileName)
			.Where(name => name is not null && !name.StartsWith("__", StringComparison.Ordinal))
			.OrderBy(name => name, StringComparer.Ordinal)
			.ToArray();

		Assert.Equal(PythonProcessorNames.OrderBy(name => name, StringComparer.Ordinal).ToArray(), pythonNames);
	}

	[Fact]
	public void BuildFaceDebuggerWithoutFaceResourcesThrowsInvalidOperation()
	{
		// face_debugger needs a FacePipelineFactory.Resources instance (see
		// ProcessorStepFactory.Build's remarks) — omitting it is a caller bug, not an
		// unknown-processor situation, so this is a different exception type than the
		// NotSupportedException an unrecognised name throws.
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
