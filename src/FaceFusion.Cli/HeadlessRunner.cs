using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Jobs;
using FaceFusion.Media;
using FaceFusion.Processors;
using FaceFusion.Types;
using FaceFusion.Workflows;
using Microsoft.ML.OnnxRuntime;

namespace FaceFusion.Cli;

/// <summary>
/// Port of the headless slice of <c>facefusion/core.py</c>: <c>process_headless</c>,
/// <c>process_step</c> and <c>conditional_process</c>.
///
/// <para>
/// <b>No global <c>state_manager</c> (PORT_CONVENTIONS.md rule 5).</b> Python's
/// <c>process_step</c> writes the merged step args into the global <c>state_manager</c> via
/// <c>apply_args</c>, then every downstream function (<c>conditional_process</c>,
/// <c>image_to_image.process</c>, each processor's <c>pre_process</c>/<c>process_frame</c>)
/// reads it back out. This port instead threads the same flat args bag
/// (<c>IReadOnlyDictionary&lt;string, object?&gt;</c>, exactly Python's own <c>Args</c> shape)
/// straight through the call chain and reads it with <see cref="StepArgsReader"/> wherever
/// Python would have called <c>state_manager.get_item</c> — there is no intermediate
/// <see cref="State"/> record built here because <see cref="State"/> has no fields for the
/// per-processor settings (<c>frame_colorizer_model</c>, ...) this path also needs; see
/// <c>SettingsBuilder</c>'s remarks for why those fields do not exist yet.
/// </para>
/// </summary>
public static class HeadlessRunner
{
	private const string ModuleName = "facefusion.core";

	/// <summary>Python: <c>process_headless</c>. Returns Python's <c>ErrorCode</c> (0 success,
	/// 1 failure — job creation/submission/run itself failed, not a processing-stage code).</summary>
	public static int ProcessHeadless(IReadOnlyDictionary<string, object?> args, JobManager jobManager, Logger logger)
	{
		var jobId = JobHelper.SuggestJobId("headless");

		if (jobManager.CreateJob(jobId)
			&& jobManager.AddStep(jobId, args)
			&& jobManager.SubmitJob(jobId)
			&& JobRunner.RunJob(jobManager, jobId, (id, index, stepArgs) => ProcessStep(id, index, stepArgs, jobManager, logger), ConcatVideoStep))
		{
			return 0;
		}

		return 1;
	}

	/// <summary>Python: <c>facefusion.ffmpeg.concat_video</c>, as used by
	/// <c>job_runner.finalize_steps</c> — see <c>JobRouter</c>'s own wiring of the same delegate
	/// for job-run/job-retry.</summary>
	public static bool ConcatVideoStep(string outputPath, IReadOnlyList<string> tempOutputPaths)
		=> Ffmpeg.ConcatVideo(outputPath, tempOutputPaths);

	/// <summary>
	/// Python: <c>process_step(job_id, step_index, step_args)</c>. Runs the content-analyser
	/// integrity gate and every processor's own pre-check <b>before</b> any processing, exactly
	/// where Python calls <c>common_pre_check()</c>/<c>processors_pre_check()</c> — this is the
	/// NSFW gate and it must run unconditionally on this path.
	/// </summary>
	public static bool ProcessStep(string jobId, int stepIndex, IReadOnlyDictionary<string, object?> stepArgs, JobManager jobManager, Logger logger)
	{
		var stepTotal = jobManager.CountStepTotal(jobId);
		logger.Info(
			Translator.Get("processing_step", ("step_current", stepIndex + 1), ("step_total", stepTotal))
				?? $"processing step {stepIndex + 1} of {stepTotal}",
			ModuleName);

		var processorNames = StepArgsReader.GetStringList(stepArgs, "processors", new[] { "face_swapper" });

		if (!PreCheck.CommonPreCheck(PreCheck.ContentAnalyserHash))
		{
			// Python's common_pre_check failure is equally quiet, but a silent exit is
			// unhelpful when the gate is the thing blocking a run.
			logger.Error("content analyser integrity check failed", ModuleName);
			return false;
		}

		foreach (var name in processorNames)
		{
			// Report WHICH processor failed its pre-check. Reporting only "false" left the
			// CLI exiting 1 after "creating temporary resources" with no clue why.
			if (!ProcessorStepFactory.PreCheck(name, stepArgs))
			{
				logger.Error($"processor '{name}' pre-check failed — its model files are missing or unreadable", ModuleName);
				return false;
			}
		}

		var errorCode = ConditionalProcess(stepArgs, processorNames, logger);

		if (errorCode != 0)
		{
			// Name the code. Python's error codes are documented in core.py; a bare
			// non-zero exit told the user nothing about which stage refused.
			var reason = errorCode switch
			{
				2 => "workflow mode did not match the target (image vs video)",
				3 => "content analyser rejected the media",
				4 => "processing was stopped",
				_ => "processing failed"
			};

			logger.Error($"{reason} (error code {errorCode})", ModuleName);
		}

		return errorCode == 0;
	}

	/// <summary>Python: <c>conditional_process</c>. Detects image-vs-video when
	/// <c>workflow_mode</c> is <c>auto</c>, then dispatches to <see cref="ImageToImage.Process"/>/
	/// <see cref="ImageToVideo.Process"/>. Returns 2 (Python's <c>ErrorCode</c> for this branch)
	/// when the (possibly explicit) workflow mode does not match the target's real kind — same
	/// as Python's own <c>state_manager.get_item('workflow_mode') == detect_workflow_mode()</c>
	/// guard.</summary>
	public static int ConditionalProcess(IReadOnlyDictionary<string, object?> args, IReadOnlyList<string> processorNames, Logger logger)
	{
		var startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds() / 1000.0;
		var targetPath = StepArgsReader.GetStringOrNull(args, "target_path");
		var requestedMode = EnumNames.FromWireName<WorkflowMode>(StepArgsReader.GetString(args, "workflow_mode", "auto"));
		var detectedMode = FileSystem.IsVideo(targetPath) ? WorkflowMode.ImageToVideo : WorkflowMode.ImageToImage;
		var workflowMode = requestedMode == WorkflowMode.Auto ? detectedMode : requestedMode;

		if (workflowMode != detectedMode)
		{
			return 2;
		}

		var built = new List<ProcessorStepFactory.BuiltStep>();
		FacePipelineFactory.Resources? faceResources = null;
		// Python: --voice-extractor-model, default kim_vocal_2 (program.py).
		using var voiceExtraction = new VoiceExtraction(
			EnumNames.FromWireName<VoiceExtractorModel>(StepArgsReader.GetString(args, "voice_extractor_model", "kim_vocal_2")));

		try
		{
			// Built once and shared across every processor in this step's --processors list —
			// see FacePipelineFactory's class remarks on why (Python: one process-wide
			// InferencePool per model, not one per processor module).
			if (processorNames.Any(FacePipelineFactory.Requires))
			{
				faceResources = FacePipelineFactory.Build(args);
			}

			foreach (var name in processorNames)
			{
				built.Add(ProcessorStepFactory.Build(name, args, faceResources));
			}

			var processorSteps = built.Select(b => b.Step).ToArray();
			var context = BuildRunContext(args, workflowMode, targetPath!, voiceExtraction);
			var processManager = new ProcessManager();
			var contentAnalyser = new ContentAnalyser();
			var modelsDirectory = ResolveModelsDirectory();
			var executionDeviceIds = StepArgsReader.GetIntList(args, "execution_device_ids", new[] { 0 });
			var executionProviders = ReadExecutionProviders(args);
			var tempPath = StepArgsReader.GetString(args, "temp_path", Path.GetTempPath());
			var outputPath = StepArgsReader.GetStringOrNull(args, "output_path") ?? targetPath!;

			if (workflowMode == WorkflowMode.ImageToImage)
			{
				return ImageToImage.Process(
					processorSteps,
					context,
					outputPath,
					tempPath,
					StepArgsReader.GetDouble(args, "output_image_scale", 1.0),
					StepArgsReader.GetInt(args, "output_image_quality", 80),
					startTime,
					contentAnalyser,
					modelsDirectory,
					executionDeviceIds,
					executionProviders,
					processManager,
					logger);
			}

			var encoderSet = Ffmpeg.GetAvailableEncoderSet(processManager, logger);
			var outputVideoEncoder = CommonHelper.GetFirst(encoderSet.Video) is { } videoEncoder
				? videoEncoder
				: EnumNames.FromWireName<VideoEncoder>("libx264");
			var outputAudioEncoder = CommonHelper.GetFirst(encoderSet.Audio) is { } audioEncoder
				? audioEncoder
				: EnumNames.FromWireName<AudioEncoder>("aac");

			return ImageToVideo.Process(
				processorSteps,
				context,
				EnumNames.FromWireName<WorkflowStrategy>(StepArgsReader.GetString(args, "workflow_strategy", "memory")),
				outputPath,
				tempPath,
				StepArgsReader.GetString(args, "temp_frame_format", "png"),
				StepArgsReader.GetDouble(args, "output_video_scale", 1.0),
				outputVideoEncoder,
				StepArgsReader.GetInt(args, "output_video_quality", 80),
				EnumNames.FromWireName<VideoPreset>(StepArgsReader.GetString(args, "output_video_preset", "veryfast")),
				EnumNames.FromWireName<TempPixelFormat>(StepArgsReader.GetString(args, "temp_pixel_format", "bgr24")),
				StepArgsReader.GetInt(args, "target_frame_amount", 2),
				StepArgsReader.GetInt(args, "execution_thread_count", 8),
				StepArgsReader.GetInt(args, "output_audio_volume", 100),
				outputAudioEncoder,
				StepArgsReader.GetInt(args, "output_audio_quality", 80),
				startTime,
				contentAnalyser,
				modelsDirectory,
				executionDeviceIds,
				executionProviders,
				processManager,
				updateProgress: null,
				logger: logger);
		}
		catch (Exception exception)
		{
			// Python lets the traceback reach the terminal. Swallowing it here produced a
			// CLI that exited 1 after "creating temporary resources" with no indication of
			// what went wrong, which is far worse to debug than a stack trace.
			logger.Error($"{exception.GetType().Name}: {exception.Message}", ModuleName);
			logger.Debug(exception.ToString(), ModuleName);
			return 1;
		}
		finally
		{
			foreach (var step in built)
			{
				step.Resource.Dispose();
			}

			faceResources?.Dispose();
		}
	}

	/// <summary>
	/// Owns the <c>voice_extractor</c> <see cref="InferenceSession"/> for one run, opened on
	/// first use rather than up front. Python reaches <c>voice_extractor.get_inference_pool()</c>
	/// only from inside <c>audio.read_voice</c>, which only <c>lip_syncer</c> ever triggers — so
	/// a run without it must not pay for loading a ~50 MB model. <see cref="Lazy{T}"/> is
	/// thread-safe by default, which matters: <c>ToVideo.ProcessMemoryFrames</c> calls this from
	/// several worker threads at once.
	/// </summary>
	private sealed class VoiceExtraction : IDisposable
	{
		private readonly Lazy<InferenceSession> _session;

		public VoiceExtraction(VoiceExtractorModel model)
		{
			var source = VoiceExtractor.CreateStaticModelSet(DownloadScope.Full)[model].Source;
			_session = new Lazy<InferenceSession>(() => new InferenceSession(source.Path));
		}

		/// <summary>Python: <c>voice_extractor.batch_extract_voice</c>, which
		/// <c>audio.read_voice</c> calls directly. See
		/// <c>FaceFusion.Media.Audio.ExtractVoiceDelegate</c>'s remarks for why the port passes
		/// it as a delegate instead of importing it.</summary>
		public double[,] Extract(double[,] audio, int chunkSize, int stepSize)
			=> VoiceExtractor.BatchExtractVoice(audio, chunkSize, stepSize, _session.Value);

		public void Dispose()
		{
			if (_session.IsValueCreated)
			{
				_session.Value.Dispose();
			}
		}
	}

	private static WorkflowRunContext BuildRunContext(IReadOnlyDictionary<string, object?> args, WorkflowMode workflowMode, string targetPath, VoiceExtraction voiceExtraction)
	{
		var trimFrameStart = StepArgsReader.GetIntOrNull(args, "trim_frame_start");
		var trimFrameEnd = StepArgsReader.GetIntOrNull(args, "trim_frame_end");
		var outputVideoFps = StepArgsReader.GetDoubleOrNull(args, "output_video_fps")
			?? FaceFusion.Vision.Vision.DetectVideoFps(targetPath)
			?? 25.0;
		var sourcePaths = StepArgsReader.GetStringList(args, "source_paths", Array.Empty<string>());

		return new WorkflowRunContext(
			WorkflowMode: workflowMode,
			TargetPath: targetPath,
			SourcePaths: sourcePaths,
			ReferenceFrameNumber: StepArgsReader.GetInt(args, "reference_frame_number", 0),
			TrimFrameStart: trimFrameStart,
			TrimFrameEnd: trimFrameEnd,
			OutputVideoFps: outputVideoFps,
			ExtractVoice: voiceExtraction.Extract);
	}

	private static IReadOnlyList<ExecutionProvider> ReadExecutionProviders(IReadOnlyDictionary<string, object?> args)
	{
		var wireNames = StepArgsReader.GetStringList(args, "execution_providers", Array.Empty<string>());

		if (wireNames.Count > 0)
		{
			return wireNames.Select(EnumNames.FromWireName<ExecutionProvider>).ToArray();
		}

		var available = Execution.GetAvailableExecutionProviders();
		return available.Count > 0 ? new[] { available[0] } : new[] { ExecutionProvider.Cpu };
	}

	internal static string ResolveModelsDirectory()
	{
		var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

		while (directory is not null)
		{
			if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
			{
				return Path.Combine(directory.FullName, ".assets", "models");
			}

			directory = directory.Parent;
		}

		throw new InvalidOperationException("Could not locate the repository root (FaceFusion.sln) to resolve .assets/models.");
	}
}
