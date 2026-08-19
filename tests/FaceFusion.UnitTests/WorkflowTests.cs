using FaceFusion.Core;
using FaceFusion.Media;
using FaceFusion.Processors;
using FaceFusion.Types;
using FaceFusion.Workflows;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port-only coverage for <c>facefusion/workflows/{core,to_image,to_video,image_to_image,
/// image_to_video}.py</c> (no <c>tests/test_workflows*.py</c> exists upstream — there is no
/// Python test suite for this module to port 1:1).
///
/// <para>
/// <see cref="ProcessTempFrameWithNoProcessorsRoundTripsTheFrame"/>/
/// <see cref="ConditionalGetTargetVisionFramesImageModeReturnsSingleFrame"/> exercise
/// <see cref="WorkflowCore"/>'s pure frame-plumbing directly against a tiny synthetic image, no
/// ffmpeg/model dependency.
/// </para>
///
/// <para>
/// <see cref="MemoryStrategyEndToEndProducesExpectedVideo"/>/
/// <see cref="DiskStrategyEndToEndProducesExpectedVideo"/> are the genuine end-to-end checks the
/// assignment calls for: <see cref="ImageToVideo.Process"/> driven over the real
/// <c>target-240p.mp4</c> example with a real <c>frame_colorizer</c> processor
/// (<c>ddcolor.onnx</c>) in the chain — chosen over e.g. <c>face_swapper</c> because it needs no
/// face-pipeline models (<c>GetCommonModules()</c> is just <c>[content_analyser]</c>, itself
/// skipped here via a <see langword="null"/> <c>ContentAnalyser</c> — see
/// <see cref="ToImage.AnalyseImage"/>'s remarks), so the test's model dependency is exactly one
/// <c>.onnx</c> file. Both gate on <see cref="WorkflowFactAttribute"/> (media + ddcolor.onnx),
/// matching every other model-dependent test in this suite (skip with a clear message rather
/// than fail when either is absent, e.g. in CI).
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class WorkflowTests
{
	private const string ModelFileName = "ddcolor.onnx";
	private static readonly Resolution FrameColorizerModelSize = new(256, 256);

	public WorkflowTests()
	{
		TestHelper.PrepareTestOutputDirectory();
	}

	// -----------------------------------------------------------------
	// Pure frame-plumbing (no ffmpeg/model dependency)
	// -----------------------------------------------------------------

	private static Mat MakeTinyBgrFrame()
	{
		var mat = new Mat(2, 2, MatType.CV_8UC3);
		mat.SetArray(new[]
		{
			new Vec3b { Item0 = 10, Item1 = 20, Item2 = 200 },
			new Vec3b { Item0 = 230, Item1 = 100, Item2 = 30 },
			new Vec3b { Item0 = 50, Item1 = 200, Item2 = 50 },
			new Vec3b { Item0 = 0, Item1 = 0, Item2 = 0 },
		});
		return mat;
	}

	[Fact]
	public void ProcessTempFrameWithNoProcessorsRoundTripsTheFrame()
	{
		var tempImagePath = Path.Combine(Path.GetTempPath(), $"facefusion-workflow-test-{Guid.NewGuid():N}.png");

		using var sourceFrame = MakeTinyBgrFrame();
		Assert.True(FaceFusion.Vision.Vision.WriteImage(tempImagePath, sourceFrame));

		try
		{
			var context = new WorkflowRunContext(
				WorkflowMode: WorkflowMode.Auto,
				TargetPath: tempImagePath,
				SourcePaths: Array.Empty<string>(),
				ReferenceFrameNumber: 0,
				TrimFrameStart: null,
				TrimFrameEnd: null,
				OutputVideoFps: 25.0,
				ExtractVoice: (_, _, _) => throw new InvalidOperationException("not expected to be called: no source audio path"));

			var targetVisionFrames = WorkflowCore.ConditionalGetTargetVisionFrames(context.WorkflowMode, context.TargetPath, 0, 2);
			try
			{
				Assert.Single(targetVisionFrames);

				using var tempVisionFrame = FaceFusion.Vision.Vision.ReadStaticImage(tempImagePath, ColorMode.Rgba)!;
				using var result = WorkflowCore.ProcessTempFrame(Array.Empty<WorkflowProcessorStep>(), context, targetVisionFrames, tempVisionFrame, 0);

				Assert.Equal(sourceFrame.Rows, result.Rows);
				Assert.Equal(sourceFrame.Cols, result.Cols);
				// No processor ran, so the mask stayed fully opaque (255) end to end —
				// conditional_merge_vision_mask therefore returns the 3-channel frame
				// unchanged rather than merging in an alpha channel (Vision.cs's own
				// ConditionalMergeVisionMask remarks).
				Assert.Equal(3, result.Channels());

				for (var row = 0; row < 2; row++)
				{
					for (var col = 0; col < 2; col++)
					{
						Assert.Equal(sourceFrame.Get<Vec3b>(row, col), result.Get<Vec3b>(row, col));
					}
				}
			}
			finally
			{
				foreach (var frame in targetVisionFrames)
				{
					frame.Dispose();
				}
			}
		}
		finally
		{
			File.Delete(tempImagePath);
		}
	}

	[Fact]
	public void ConditionalGetTargetVisionFramesImageModeReturnsSingleFrame()
	{
		var tempImagePath = Path.Combine(Path.GetTempPath(), $"facefusion-workflow-test-{Guid.NewGuid():N}.png");
		using var sourceFrame = MakeTinyBgrFrame();
		Assert.True(FaceFusion.Vision.Vision.WriteImage(tempImagePath, sourceFrame));

		try
		{
			var frames = WorkflowCore.ConditionalGetTargetVisionFrames(WorkflowMode.ImageToImage, tempImagePath, 0, 2);
			try
			{
				Assert.Single(frames);
				Assert.Equal(2, frames[0].Rows);
				Assert.Equal(2, frames[0].Cols);
			}
			finally
			{
				foreach (var frame in frames)
				{
					frame.Dispose();
				}
			}
		}
		finally
		{
			File.Delete(tempImagePath);
		}
	}

	// -----------------------------------------------------------------
	// End-to-end: ImageToVideo.Process over a real video with a real processor
	// -----------------------------------------------------------------

	private static string? FindRepoRoot()
	{
		var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

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

	private static string? FindModelPath(string modelFileName)
	{
		var repoRoot = FindRepoRoot();
		return repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models", modelFileName);
	}

	private static bool ModelAvailable(string modelFileName)
	{
		var modelPath = FindModelPath(modelFileName);
		return modelPath is not null && File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
	}

	/// <summary>
	/// Skips when either the example media/ffmpeg/ffprobe or <c>ddcolor.onnx</c> are missing —
	/// both are gitignored/network-fetched and not guaranteed present (e.g. in CI).
	/// </summary>
	private sealed class WorkflowFactAttribute : FactAttribute
	{
		public WorkflowFactAttribute()
		{
			if (!TestHelper.HasFfmpeg || !TestHelper.HasFfprobe || !TestHelper.ExamplesAvailable)
			{
				Skip = TestHelper.MissingMediaMessage;
			}
			else if (!ModelAvailable(ModelFileName))
			{
				Skip = $"requires .assets/models/{ModelFileName} (gitignored, not present in CI) — " +
					   "populate via the real Python frame_colorizer pre_check() with network access, then retry";
			}
		}
	}

	private static IReadOnlyList<WorkflowProcessorStep> BuildFrameColorizerSteps(InferenceSession session)
	{
		var processor = new FrameColorizer.Processor();

		return new[]
		{
			new WorkflowProcessorStep(processor, context => new FrameColorizer.FrameColorizerInputs(
				context.TempVisionFrame,
				context.TempVisionMask,
				FrameColorizerModelType.Ddcolor,
				FrameColorizerModelSize,
				session,
				FrameColorizerBlend: 100)),
		};
	}

	[WorkflowFact]
	public void MemoryStrategyEndToEndProducesExpectedVideo()
	{
		var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
		var outputPath = TestHelper.GetTestOutputFile("workflow-memory-e2e.mp4");
		const int trimFrameStart = 0;
		const int trimFrameEnd = 8;
		const double outputVideoFps = 25.0;
		var expectedResolution = new Resolution(426, 226);

		using var session = new InferenceSession(FindModelPath(ModelFileName));
		var processorSteps = BuildFrameColorizerSteps(session);

		var context = new WorkflowRunContext(
			WorkflowMode: WorkflowMode.ImageToVideo,
			TargetPath: targetPath,
			SourcePaths: Array.Empty<string>(),
			ReferenceFrameNumber: 0,
			TrimFrameStart: trimFrameStart,
			TrimFrameEnd: trimFrameEnd,
			OutputVideoFps: outputVideoFps,
			ExtractVoice: (_, _, _) => throw new InvalidOperationException("not expected to be called: no source audio path"));

		var processManager = new ProcessManager();
		var tempPath = Path.GetTempPath();

		TempHelper.ClearTempDirectory(targetPath, tempPath);
		try
		{
			var errorCode = ImageToVideo.Process(
				processorSteps,
				context,
				WorkflowStrategy.Memory,
				outputPath,
				tempPath,
				tempFrameFormat: "png",
				outputVideoScale: 1.0,
				outputVideoEncoder: VideoEncoder.Libx264,
				outputVideoQuality: 80,
				outputVideoPreset: VideoPreset.Ultrafast,
				tempPixelFormat: TempPixelFormat.Bgr24,
				targetFrameAmount: 2,
				executionThreadCount: 2,
				outputAudioVolume: 100,
				outputAudioEncoder: AudioEncoder.Aac,
				outputAudioQuality: 80,
				startTime: 0,
				contentAnalyser: null,
				modelsDirectory: string.Empty,
				executionDeviceIds: Array.Empty<int>(),
				executionProviders: Array.Empty<ExecutionProvider>(),
				processManager: processManager);

			Assert.Equal(0, errorCode);
			Assert.True(FileSystem.IsFile(outputPath), $"expected an output file at '{outputPath}'");

			var outputMetadata = Ffprobe.ExtractVideoMetadata(outputPath);
			Assert.Equal(trimFrameEnd - trimFrameStart, outputMetadata.FrameTotal);
			Assert.Equal(expectedResolution, outputMetadata.Resolution);
			Assert.Equal(outputVideoFps, outputMetadata.Fps);

			var expectedDuration = (trimFrameEnd - trimFrameStart) / outputVideoFps;
			Assert.True(Math.Abs(outputMetadata.Duration - expectedDuration) < 0.2,
				$"expected duration near {expectedDuration}s, got {outputMetadata.Duration}s");
		}
		finally
		{
			TempHelper.ClearTempDirectory(targetPath, tempPath);
		}
	}

	[WorkflowFact]
	public void DiskStrategyEndToEndProducesExpectedVideo()
	{
		var targetPath = TestHelper.GetTestExampleFile("target-240p.mp4");
		var outputPath = TestHelper.GetTestOutputFile("workflow-disk-e2e.mp4");
		const int trimFrameStart = 0;
		const int trimFrameEnd = 8;
		const double outputVideoFps = 25.0;
		var expectedResolution = new Resolution(426, 226);

		using var session = new InferenceSession(FindModelPath(ModelFileName));
		var processorSteps = BuildFrameColorizerSteps(session);

		var context = new WorkflowRunContext(
			WorkflowMode: WorkflowMode.ImageToVideo,
			TargetPath: targetPath,
			SourcePaths: Array.Empty<string>(),
			ReferenceFrameNumber: 0,
			TrimFrameStart: trimFrameStart,
			TrimFrameEnd: trimFrameEnd,
			OutputVideoFps: outputVideoFps,
			ExtractVoice: (_, _, _) => throw new InvalidOperationException("not expected to be called: no source audio path"));

		var processManager = new ProcessManager();
		var tempPath = Path.GetTempPath();

		TempHelper.ClearTempDirectory(targetPath, tempPath);
		try
		{
			var errorCode = ImageToVideo.Process(
				processorSteps,
				context,
				WorkflowStrategy.Disk,
				outputPath,
				tempPath,
				tempFrameFormat: "png",
				outputVideoScale: 1.0,
				outputVideoEncoder: VideoEncoder.Libx264,
				outputVideoQuality: 80,
				outputVideoPreset: VideoPreset.Ultrafast,
				tempPixelFormat: TempPixelFormat.Bgr24,
				targetFrameAmount: 2,
				executionThreadCount: 2,
				outputAudioVolume: 100,
				outputAudioEncoder: AudioEncoder.Aac,
				outputAudioQuality: 80,
				startTime: 0,
				contentAnalyser: null,
				modelsDirectory: string.Empty,
				executionDeviceIds: Array.Empty<int>(),
				executionProviders: Array.Empty<ExecutionProvider>(),
				processManager: processManager);

			Assert.Equal(0, errorCode);
			Assert.True(FileSystem.IsFile(outputPath), $"expected an output file at '{outputPath}'");

			var outputMetadata = Ffprobe.ExtractVideoMetadata(outputPath);
			Assert.Equal(trimFrameEnd - trimFrameStart, outputMetadata.FrameTotal);
			Assert.Equal(expectedResolution, outputMetadata.Resolution);
			Assert.Equal(outputVideoFps, outputMetadata.Fps);

			var expectedDuration = (trimFrameEnd - trimFrameStart) / outputVideoFps;
			Assert.True(Math.Abs(outputMetadata.Duration - expectedDuration) < 0.2,
				$"expected duration near {expectedDuration}s, got {outputMetadata.Duration}s");
		}
		finally
		{
			TempHelper.ClearTempDirectory(targetPath, tempPath);
		}
	}
}
