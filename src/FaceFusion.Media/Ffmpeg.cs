using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using FaceFusion.Core;
using FaceFusion.Types;
using FaceFusion.Vision;

// Grants the test project access to the internal parsing helpers on Ffmpeg/Ffprobe
// (TryParseFrameNumber, ParseEncoderLine, ParseEntries, ExtractVideoFps) so they can be
// unit-tested directly against canned strings without a real process. A C# assembly
// attribute, not a project-file edit.
[assembly: InternalsVisibleTo("FaceFusion.UnitTests")]

namespace FaceFusion.Media;

/// <summary>
/// Port of facefusion/ffmpeg.py — the ffmpeg process runners plus the higher-level
/// pipeline functions built on them (frame extraction/merging, image copy/finalize, audio
/// buffer read/restore/replace, video concat).
///
/// <para>
/// State-manager parameters: the Python module reads several values from
/// <c>state_manager</c> (<c>output_video_encoder</c>, <c>output_video_quality</c>,
/// <c>output_video_preset</c>, <c>output_audio_encoder</c>, <c>output_audio_quality</c>,
/// <c>output_audio_volume</c>, <c>output_image_quality</c>, <c>temp_pixel_format</c>,
/// <c>temp_path</c>, <c>log_level</c>). Per port convention rule 5 (no global mutable
/// state) every one of those is taken as an explicit parameter here instead.
/// </para>
///
/// <para>
/// Progress reporting: <c>extract_frames</c>/<c>merge_video</c> in Python wrap the caller's
/// <c>update_progress</c> in a <c>tqdm</c> progress bar (<c>partial(update_progress,
/// progress)</c>) sized from <c>vision.predict_video_frame_total</c> and gated on
/// <c>translator</c>/<c>state_manager.get_item('log_level')</c>. <c>tqdm</c> is a terminal
/// UI concern with no bearing on the ffmpeg command or the boolean result, and this layer
/// has no CLI/UI dependency (same reasoning <see cref="RunFfmpegWithProgress"/> already
/// applies below) — so <see cref="ExtractFrames"/>/<see cref="MergeVideo"/> take an
/// <see cref="UpdateProgress"/> delegate directly and pass it straight through to
/// <see cref="RunFfmpegWithProgress"/>, same as that method's own callers already do.
/// </para>
///
/// <para>
/// Design note (deliberate, see port report): line/output parsing lives in methods that
/// take plain strings and have no <see cref="Process"/> dependency —
/// <see cref="TryParseFrameNumber"/> (the <c>-progress</c> "frame=" line ported out of
/// <c>run_ffmpeg_with_progress</c>) and <see cref="ParseEncoderLine"/> (the
/// <c>-encoders</c> line ported out of <c>get_available_encoder_set</c>) — so they are
/// unit-testable with canned output with no ffmpeg binary present.
/// </para>
/// </summary>
public static class Ffmpeg
{
	/// <summary>
	/// Python: <c>run_ffmpeg_with_progress</c>.
	///
	/// Python reads <c>state_manager.get_item('log_level')</c> and polls
	/// <c>process_manager.is_processing()</c> / <c>is_stopping()</c>. Per port convention
	/// rule 5 (no global mutable state) these become explicit optional parameters:
	/// <paramref name="processManager"/> null means "always processing, never stopping"
	/// (the common case — a caller that never requested a stop), and
	/// <paramref name="logLevel"/> null means debug logging is off.
	/// </summary>
	public static Process? RunFfmpegWithProgress(
		IReadOnlyList<string> commands,
		UpdateProgress updateProgress,
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null,
		string? ffmpegPath = null)
	{
		var extendedCommands = new List<string>(commands);
		extendedCommands.AddRange(FfmpegBuilder.SetProgress());
		extendedCommands.AddRange(FfmpegBuilder.CastStream());
		var fullCommands = FfmpegBuilder.Run(extendedCommands, ffmpegPath);
		var process = ProcessRunner.TryStart(fullCommands);

		if (process is null)
		{
			return null;
		}

		while (processManager is null || processManager.IsProcessing())
		{
			// Python: `while __line__ := process.stdout.readline().decode().lower():`.
			// ReadLine() returns null at genuine EOF; a blank line (rare, mid-stream)
			// stops this port's inner loop one iteration earlier than Python's decoded
			// "\n" would, which is immaterial here since blank lines never carry a
			// "frame=" token.
			string? line;

			while (!string.IsNullOrEmpty(line = process.StandardOutput.ReadLine()?.ToLowerInvariant()))
			{
				if (processManager?.IsStopping() == true)
				{
					TryTerminate(process);
				}

				var frameNumber = TryParseFrameNumber(line);

				if (frameNumber.HasValue)
				{
					updateProgress(frameNumber.Value);
				}
			}

			if (logLevel == LogLevel.Debug)
			{
				LogDebug(process, logger);
			}

			if (process.WaitForExit(500))
			{
				return process;
			}
		}

		return process;
	}

	/// <summary>Python: <c>run_ffmpeg</c>. See <see cref="RunFfmpegWithProgress"/> for the parameter deviations.</summary>
	public static Process? RunFfmpeg(
		IReadOnlyList<string> commands,
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null,
		string? ffmpegPath = null)
	{
		var fullCommands = FfmpegBuilder.Run(commands, ffmpegPath);
		var process = ProcessRunner.TryStart(fullCommands);

		if (process is null)
		{
			return null;
		}

		while (processManager is null || processManager.IsProcessing())
		{
			if (logLevel == LogLevel.Debug)
			{
				LogDebug(process, logger);
			}

			if (process.WaitForExit(500))
			{
				return process;
			}
		}

		if (processManager?.IsStopping() == true)
		{
			TryTerminate(process);
		}

		return process;
	}

	/// <summary>
	/// Python: <c>open_ffmpeg</c>. Stdin is piped in, stderr is discarded (not redirected,
	/// mirroring Python's <c>stderr = subprocess.DEVNULL</c>), stdout is piped out for the
	/// caller to read raw frame/audio bytes from.
	/// </summary>
	public static Process? OpenFfmpeg(IReadOnlyList<string> commands, string? ffmpegPath = null)
	{
		var fullCommands = FfmpegBuilder.Run(commands, ffmpegPath);
		return ProcessRunner.TryStart(fullCommands, redirectStdin: true, redirectStderr: false);
	}

	/// <summary>
	/// Python: <c>log_debug</c>. Drains stdout and stderr concurrently (never one to
	/// completion before the other) and waits for exit, then logs each non-blank stderr
	/// line at <see cref="LogLevel.Debug"/>. Python's <c>__name__</c> for this module is
	/// <c>'facefusion.ffmpeg'</c>; that literal string is kept (not a C# namespace) so the
	/// logged message prefix matches the Python output exactly.
	/// </summary>
	public static void LogDebug(Process process, Logger? logger)
	{
		var (_, stderr, _) = ProcessRunner.Communicate(process);
		var errors = stderr.Split(Environment.NewLine);

		foreach (var error in errors)
		{
			var trimmed = error.Trim();

			if (trimmed.Length > 0)
			{
				logger?.Debug(trimmed, "facefusion.ffmpeg");
			}
		}
	}

	/// <summary>Python: <c>get_available_encoder_set</c>.</summary>
	public static EncoderSet GetAvailableEncoderSet(
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null,
		string? ffmpegPath = null)
	{
		var audioEncoders = new List<AudioEncoder>();
		var videoEncoders = new List<VideoEncoder>();
		var commands = FfmpegBuilder.Chain(FfmpegBuilder.GetEncoders());
		// Disposed via `using` (deliberate port-only fix, not present in Python — a bare
		// subprocess.Popen has no explicit .close() call to port either — see port report):
		// every RunFfmpeg/RunFfmpegWithProgress call site in this file used to leak the
		// returned Process (and its redirected stdout/stderr pipe handles) until the next
		// GC, which is exactly the kind of leak that only shows up under sustained load —
		// confirmed via a long real test run that spawns several hundred ffmpeg/ffprobe
		// subprocesses back to back, where a handful of otherwise-correct calls started
		// failing sporadically partway through. Disposing here (and at every other
		// fire-and-forget call site below) keeps each subprocess's OS handles bounded to
		// its own call instead of accumulating for the process's lifetime.
		using var process = RunFfmpeg(commands, processManager, logger, logLevel, ffmpegPath);

		if (process is null)
		{
			return new EncoderSet(audioEncoders, videoEncoders);
		}

		var outputAudioEncoders = Choices.OutputEncoderSet.Audio;
		var outputVideoEncoders = Choices.OutputEncoderSet.Video;

		string? rawLine;

		// Python: `while line := process.stdout.readline().decode().lower():`. This reads
		// ffmpeg's `-encoders` listing from stdout without having drained it during
		// run_ffmpeg()'s wait loop above — the same latent full-pipe-buffer hazard exists
		// in the Python source (reproduced faithfully per port convention rule 1, not
		// fixed here).
		while (!string.IsNullOrEmpty(rawLine = process.StandardOutput.ReadLine()))
		{
			var line = rawLine.ToLowerInvariant();
			var parsed = ParseEncoderLine(line);

			if (parsed is null)
			{
				continue;
			}

			var (kind, encoderName) = parsed.Value;

			if (kind == EncoderLineKind.Audio && EnumNames.TryFromWireName<AudioEncoder>(encoderName, out var audioEncoder))
			{
				var index = IndexOf(outputAudioEncoders, audioEncoder);

				if (index >= 0)
				{
					InsertAt(audioEncoders, index, audioEncoder);
				}
			}
			else if (kind == EncoderLineKind.Video && EnumNames.TryFromWireName<VideoEncoder>(encoderName, out var videoEncoder))
			{
				var index = IndexOf(outputVideoEncoders, videoEncoder);

				if (index >= 0)
				{
					InsertAt(videoEncoders, index, videoEncoder);
				}
			}
		}

		return new EncoderSet(audioEncoders, videoEncoders);
	}

	/// <summary>Python: <c>fix_audio_encoder</c>. Pure — testable without a binary.</summary>
	public static AudioEncoder FixAudioEncoder(VideoFormat videoFormat, AudioEncoder audioEncoder)
	{
		if (videoFormat == VideoFormat.Avi && audioEncoder == AudioEncoder.Libopus)
		{
			return AudioEncoder.Aac;
		}

		if (videoFormat is VideoFormat.M4v or VideoFormat.Mpeg or VideoFormat.Wmv)
		{
			return AudioEncoder.Aac;
		}

		if (videoFormat == VideoFormat.Mov && audioEncoder is AudioEncoder.Flac or AudioEncoder.Libopus)
		{
			return AudioEncoder.Aac;
		}

		if (videoFormat == VideoFormat.Mxf)
		{
			return AudioEncoder.PcmS16le;
		}

		if (videoFormat == VideoFormat.Webm)
		{
			return AudioEncoder.Libopus;
		}

		return audioEncoder;
	}

	/// <summary>Python: <c>fix_video_encoder</c>. Pure — testable without a binary.</summary>
	public static VideoEncoder FixVideoEncoder(VideoFormat videoFormat, VideoEncoder videoEncoder)
	{
		if (videoFormat is VideoFormat.M4v or VideoFormat.Mpeg or VideoFormat.Mxf or VideoFormat.Wmv)
		{
			return VideoEncoder.Libx264;
		}

		if (videoFormat is VideoFormat.Mkv or VideoFormat.Mp4 && videoEncoder == VideoEncoder.Rawvideo)
		{
			return VideoEncoder.Libx264;
		}

		if (videoFormat == VideoFormat.Mov && videoEncoder == VideoEncoder.LibvpxVp9)
		{
			return VideoEncoder.Libx264;
		}

		if (videoFormat == VideoFormat.Webm)
		{
			return VideoEncoder.LibvpxVp9;
		}

		return videoEncoder;
	}

	// -----------------------------------------------------------------
	// Pipeline functions (frame extraction/merging, image copy/finalize, audio, concat)
	// -----------------------------------------------------------------

	/// <summary>
	/// Python: <c>create_video_reader</c>. Opens a long-lived ffmpeg subprocess that
	/// streams raw <c>bgr24</c> frame bytes to stdout starting at <paramref name="frameNumber"/>.
	/// Returns a <see cref="VideoReaderProcess"/> wrapping the (possibly absent, if ffmpeg
	/// is not on PATH) process; the caller must dispose it to guarantee the subprocess is
	/// terminated, including when disposed mid-pump on an exception path.
	/// </summary>
	public static VideoReaderProcess CreateVideoReader(string videoPath, int frameNumber, VideoMetadata videoMetadata, string? ffmpegPath = null)
	{
		var commands = BuildCreateVideoReaderCommands(videoPath, frameNumber, videoMetadata);
		var process = OpenFfmpeg(commands, ffmpegPath);
		return new VideoReaderProcess(process, videoPath, videoMetadata, frameNumber);
	}

	/// <summary>Command construction pulled out of <see cref="CreateVideoReader"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildCreateVideoReaderCommands(string videoPath, int frameNumber, VideoMetadata videoMetadata)
	{
		return FfmpegBuilder.Chain(
			FfmpegBuilder.SeekTo(frameNumber / videoMetadata.Fps),
			FfmpegBuilder.SetInput(videoPath),
			FfmpegBuilder.RestrictColorTransfer(videoMetadata.ColorTransfer),
			FfmpegBuilder.PreventFrameDrop(),
			FfmpegBuilder.EnforcePixelFormat("bgr24"),
			FfmpegBuilder.SetOutputFormat("rawvideo"),
			FfmpegBuilder.CastStream());
	}

	/// <summary>
	/// Python: <c>create_video_writer</c>. Opens a long-lived ffmpeg subprocess that reads
	/// raw frame bytes from stdin and encodes them to <paramref name="targetPath"/>'s temp
	/// video file. Returns a <see cref="VideoWriterProcess"/> wrapping the (possibly
	/// absent) process; the caller must dispose it to guarantee the subprocess is
	/// terminated (including flushing/closing stdin), including on an exception path.
	/// </summary>
	public static VideoWriterProcess CreateVideoWriter(
		string targetPath,
		double tempVideoFps,
		Resolution tempVideoResolution,
		Resolution outputVideoResolution,
		double outputVideoFps,
		VideoEncoder outputVideoEncoder,
		int outputVideoQuality,
		VideoPreset outputVideoPreset,
		TempPixelFormat tempPixelFormat,
		string tempPath,
		string? ffmpegPath = null)
	{
		var tempVideoPath = TempHelper.GetTempFilePath(targetPath, tempPath);
		var tempVideoFormat = FileSystem.GetFileFormat(tempVideoPath);
		var resolvedVideoEncoder = ResolveVideoEncoder(tempVideoFormat, outputVideoEncoder);

		var commands = BuildCreateVideoWriterCommands(tempVideoPath, tempVideoFormat, tempVideoFps, tempVideoResolution, outputVideoResolution, outputVideoFps, resolvedVideoEncoder, outputVideoQuality, outputVideoPreset, tempPixelFormat);
		var process = OpenFfmpeg(commands, ffmpegPath);
		return new VideoWriterProcess(process, targetPath, tempVideoPath, tempVideoFps, outputVideoResolution, outputVideoFps);
	}

	/// <summary>Command construction pulled out of <see cref="CreateVideoWriter"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildCreateVideoWriterCommands(string tempVideoPath, string? tempVideoFormat, double tempVideoFps, Resolution tempVideoResolution, Resolution outputVideoResolution, double outputVideoFps, VideoEncoder resolvedVideoEncoder, int outputVideoQuality, VideoPreset outputVideoPreset, TempPixelFormat tempPixelFormat)
	{
		var videoEncoderName = resolvedVideoEncoder.ToWireName();

		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetOutputFormat("rawvideo"),
			FfmpegBuilder.EnforcePixelFormat(tempPixelFormat.ToWireName()),
			FfmpegBuilder.SetMediaResolution(Vision.Vision.PackResolution(tempVideoResolution)),
			FfmpegBuilder.SetInputFps(tempVideoFps),
			FfmpegBuilder.SetInput("pipe:0"),
			FfmpegBuilder.SetMediaResolution(Vision.Vision.PackResolution(outputVideoResolution)),
			FfmpegBuilder.SetVideoEncoder(videoEncoderName),
			FfmpegBuilder.SetThreadCount(16),
			FfmpegBuilder.SetVideoTag(videoEncoderName, tempVideoFormat ?? string.Empty),
			FfmpegBuilder.SetVideoQuality(videoEncoderName, outputVideoQuality),
			AssertNoNullPreset(FfmpegBuilder.SetVideoPreset(videoEncoderName, outputVideoPreset.ToWireName())),
			FfmpegBuilder.Concat(
				FfmpegBuilder.SetVideoFps(outputVideoFps),
				FfmpegBuilder.ConvertColorSpace("bt709")),
			FfmpegBuilder.SetPixelFormat(videoEncoderName),
			FfmpegBuilder.ForceOutput(tempVideoPath));
	}

	/// <summary>
	/// Python: <c>extract_frames</c>. Extracts the trimmed/rescaled frame range of
	/// <paramref name="targetPath"/> to numbered temp frame files. See the class remarks
	/// for why the <c>tqdm</c> progress-bar wrapping is not reproduced —
	/// <paramref name="updateProgress"/> is passed straight through to
	/// <see cref="RunFfmpegWithProgress"/>.
	/// </summary>
	public static bool ExtractFrames(
		string targetPath,
		Resolution tempVideoResolution,
		double tempVideoFps,
		int trimFrameStart,
		int trimFrameEnd,
		string tempPath,
		string tempFrameFormat,
		UpdateProgress updateProgress,
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null)
	{
		var colorTransfer = Ffprobe.ExtractStaticVideoMetadata(targetPath).ColorTransfer;
		var tempFramePattern = TempHelper.GetTempFramePattern(targetPath, "%08d", tempPath, tempFrameFormat);

		var commands = BuildExtractFramesCommands(targetPath, tempVideoResolution, tempVideoFps, trimFrameStart, trimFrameEnd, colorTransfer, tempFramePattern);
		using var process = RunFfmpegWithProgress(commands, updateProgress, processManager, logger, logLevel);
		return process is not null && process.ExitCode == 0;
	}

	/// <summary>
	/// Command construction pulled out of <see cref="ExtractFrames"/> for direct unit
	/// testing (no process/ffprobe dependency) — takes <paramref name="colorTransfer"/> and
	/// <paramref name="tempFramePattern"/> as plain values rather than deriving them
	/// internally via <see cref="Ffprobe"/>/<see cref="TempHelper"/>.
	/// </summary>
	internal static IReadOnlyList<string> BuildExtractFramesCommands(string targetPath, Resolution tempVideoResolution, double tempVideoFps, int trimFrameStart, int trimFrameEnd, string colorTransfer, string tempFramePattern)
	{
		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(targetPath),
			FfmpegBuilder.SetMediaResolution(Vision.Vision.PackResolution(tempVideoResolution)),
			FfmpegBuilder.SetFrameQuality(0),
			FfmpegBuilder.EnforcePixelFormat("rgb24"),
			FfmpegBuilder.Concat(
				FfmpegBuilder.SelectFrameRange(trimFrameStart, trimFrameEnd, tempVideoFps),
				FfmpegBuilder.RestrictColorTransfer(colorTransfer)),
			FfmpegBuilder.PreventFrameDrop(),
			FfmpegBuilder.SetStartNumber(trimFrameStart),
			FfmpegBuilder.SetOutput(tempFramePattern));
	}

	/// <summary>Python: <c>copy_image</c>.</summary>
	public static bool CopyImage(string targetPath, Resolution tempImageResolution, string tempPath, ProcessManager? processManager = null, Logger? logger = null, LogLevel? logLevel = null)
	{
		var tempImagePath = TempHelper.GetTempFilePath(targetPath, tempPath);
		var commands = BuildCopyImageCommands(targetPath, tempImageResolution, tempImagePath);
		using var process = RunFfmpeg(commands, processManager, logger, logLevel);
		return process is not null && process.ExitCode == 0;
	}

	/// <summary>Command construction pulled out of <see cref="CopyImage"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildCopyImageCommands(string targetPath, Resolution tempImageResolution, string tempImagePath)
	{
		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(targetPath),
			FfmpegBuilder.SetMediaResolution(Vision.Vision.PackResolution(tempImageResolution)),
			FfmpegBuilder.SetImageQuality(targetPath, 100),
			FfmpegBuilder.ForceOutput(tempImagePath));
	}

	/// <summary>Python: <c>finalize_image</c>.</summary>
	public static bool FinalizeImage(string targetPath, string outputPath, Resolution outputImageResolution, int outputImageQuality, string tempPath, ProcessManager? processManager = null, Logger? logger = null, LogLevel? logLevel = null)
	{
		var tempImagePath = TempHelper.GetTempFilePath(targetPath, tempPath);
		var commands = BuildFinalizeImageCommands(targetPath, tempImagePath, outputPath, outputImageResolution, outputImageQuality);
		using var process = RunFfmpeg(commands, processManager, logger, logLevel);
		return process is not null && process.ExitCode == 0;
	}

	/// <summary>Command construction pulled out of <see cref="FinalizeImage"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildFinalizeImageCommands(string targetPath, string tempImagePath, string outputPath, Resolution outputImageResolution, int outputImageQuality)
	{
		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(tempImagePath),
			FfmpegBuilder.SetMediaResolution(Vision.Vision.PackResolution(outputImageResolution)),
			// Python note (reproduced): quality is derived from target_path's format,
			// even though the input here is temp_image_path.
			FfmpegBuilder.SetImageQuality(targetPath, outputImageQuality),
			FfmpegBuilder.ForceOutput(outputPath));
	}

	/// <summary>
	/// Python: <c>read_audio_buffer</c>. Unlike <see cref="ProcessRunner.Communicate"/>
	/// (text-oriented, used by the metadata-probing paths elsewhere in this file), this
	/// reads stdout as raw bytes off <see cref="Process.StandardOutput"/>'s
	/// <see cref="StreamReader.BaseStream"/> — decoding a raw PCM/rawvideo byte stream
	/// through a text <see cref="StreamReader"/> would corrupt it.
	/// </summary>
	public static byte[]? ReadAudioBuffer(string targetPath, int audioSampleRate, int audioSampleSize, int audioChannelTotal)
	{
		var commands = BuildReadAudioBufferCommands(targetPath, audioSampleRate, audioSampleSize, audioChannelTotal);
		using var process = OpenFfmpeg(commands);

		if (process is null)
		{
			return null;
		}

		using var buffer = new MemoryStream();
		process.StandardOutput.BaseStream.CopyTo(buffer);
		process.WaitForExit();

		return process.ExitCode == 0 ? buffer.ToArray() : null;
	}

	/// <summary>Command construction pulled out of <see cref="ReadAudioBuffer"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildReadAudioBufferCommands(string targetPath, int audioSampleRate, int audioSampleSize, int audioChannelTotal)
	{
		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(targetPath),
			FfmpegBuilder.IgnoreVideoStream(),
			FfmpegBuilder.SetAudioSampleRate(audioSampleRate),
			FfmpegBuilder.SetAudioSampleSize(audioSampleSize),
			FfmpegBuilder.SetAudioChannelTotal(audioChannelTotal),
			FfmpegBuilder.CastStream());
	}

	/// <summary>Python: <c>restore_audio</c>.</summary>
	public static bool RestoreAudio(
		string targetPath,
		string outputPath,
		int trimFrameStart,
		int trimFrameEnd,
		AudioEncoder outputAudioEncoder,
		int outputAudioQuality,
		int outputAudioVolume,
		string tempPath,
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null)
	{
		var targetVideoFps = Vision.Vision.DetectVideoFps(targetPath) ?? 0;
		var tempVideoPath = TempHelper.GetTempFilePath(targetPath, tempPath);
		var tempVideoFormat = FileSystem.GetFileFormat(tempVideoPath);
		var tempVideoDuration = Vision.Vision.DetectVideoDuration(tempVideoPath);
		var outputVideoFormat = FileSystem.GetFileFormat(outputPath) ?? string.Empty;
		var resolvedAudioEncoder = ResolveAudioEncoder(tempVideoFormat, outputAudioEncoder);

		var commands = BuildRestoreAudioCommands(tempVideoPath, targetPath, outputPath, trimFrameStart, trimFrameEnd, targetVideoFps, resolvedAudioEncoder, outputAudioQuality, outputAudioVolume, tempVideoDuration, outputVideoFormat);
		using var process = RunFfmpeg(commands, processManager, logger, logLevel);
		return process is not null && process.ExitCode == 0;
	}

	/// <summary>Command construction pulled out of <see cref="RestoreAudio"/> for direct unit testing (no process/OpenCV dependency).</summary>
	internal static IReadOnlyList<string> BuildRestoreAudioCommands(string tempVideoPath, string targetPath, string outputPath, int trimFrameStart, int trimFrameEnd, double targetVideoFps, AudioEncoder resolvedAudioEncoder, int outputAudioQuality, int outputAudioVolume, double tempVideoDuration, string outputVideoFormat)
	{
		var audioEncoderName = resolvedAudioEncoder.ToWireName();

		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(tempVideoPath),
			FfmpegBuilder.SelectMediaRange(trimFrameStart, trimFrameEnd, targetVideoFps),
			FfmpegBuilder.SetInput(targetPath),
			FfmpegBuilder.CopyVideoEncoder(),
			FfmpegBuilder.SetAudioEncoder(audioEncoderName),
			FfmpegBuilder.SetAudioQuality(audioEncoderName, outputAudioQuality),
			FfmpegBuilder.SetAudioVolume(outputAudioVolume),
			FfmpegBuilder.SelectMediaStream("0:v:0"),
			FfmpegBuilder.SelectMediaStream("1:a:0"),
			FfmpegBuilder.SetVideoDuration(tempVideoDuration),
			FfmpegBuilder.SetFaststart(outputVideoFormat),
			FfmpegBuilder.ForceOutput(outputPath));
	}

	/// <summary>Python: <c>replace_audio</c>.</summary>
	public static bool ReplaceAudio(
		string targetPath,
		string audioPath,
		string outputPath,
		AudioEncoder outputAudioEncoder,
		int outputAudioQuality,
		int outputAudioVolume,
		string tempPath,
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null)
	{
		var tempVideoPath = TempHelper.GetTempFilePath(targetPath, tempPath);
		var tempVideoFormat = FileSystem.GetFileFormat(tempVideoPath);
		var tempVideoDuration = Vision.Vision.DetectVideoDuration(tempVideoPath);
		var outputVideoFormat = FileSystem.GetFileFormat(outputPath) ?? string.Empty;
		var resolvedAudioEncoder = ResolveAudioEncoder(tempVideoFormat, outputAudioEncoder);

		var commands = BuildReplaceAudioCommands(tempVideoPath, audioPath, outputPath, resolvedAudioEncoder, outputAudioQuality, outputAudioVolume, tempVideoDuration, outputVideoFormat);
		using var process = RunFfmpeg(commands, processManager, logger, logLevel);
		return process is not null && process.ExitCode == 0;
	}

	/// <summary>Command construction pulled out of <see cref="ReplaceAudio"/> for direct unit testing (no process/OpenCV dependency).</summary>
	internal static IReadOnlyList<string> BuildReplaceAudioCommands(string tempVideoPath, string audioPath, string outputPath, AudioEncoder resolvedAudioEncoder, int outputAudioQuality, int outputAudioVolume, double tempVideoDuration, string outputVideoFormat)
	{
		var audioEncoderName = resolvedAudioEncoder.ToWireName();

		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(tempVideoPath),
			FfmpegBuilder.SetInput(audioPath),
			FfmpegBuilder.CopyVideoEncoder(),
			FfmpegBuilder.SetAudioEncoder(audioEncoderName),
			FfmpegBuilder.SetAudioQuality(audioEncoderName, outputAudioQuality),
			FfmpegBuilder.SetAudioVolume(outputAudioVolume),
			FfmpegBuilder.SetVideoDuration(tempVideoDuration),
			FfmpegBuilder.SetFaststart(outputVideoFormat),
			FfmpegBuilder.ForceOutput(outputPath));
	}

	/// <summary>
	/// Python: <c>merge_video</c>. See the class remarks for why the <c>tqdm</c>
	/// progress-bar wrapping is not reproduced.
	/// </summary>
	public static bool MergeVideo(
		string targetPath,
		double tempVideoFps,
		Resolution outputVideoResolution,
		double outputVideoFps,
		int trimFrameStart,
		int trimFrameEnd,
		VideoEncoder outputVideoEncoder,
		int outputVideoQuality,
		VideoPreset outputVideoPreset,
		string tempPath,
		string tempFrameFormat,
		UpdateProgress updateProgress,
		ProcessManager? processManager = null,
		Logger? logger = null,
		LogLevel? logLevel = null)
	{
		var tempVideoPath = TempHelper.GetTempFilePath(targetPath, tempPath);
		var tempVideoFormat = FileSystem.GetFileFormat(tempVideoPath);
		var tempFramePattern = TempHelper.GetTempFramePattern(targetPath, "%08d", tempPath, tempFrameFormat);
		var resolvedVideoEncoder = ResolveVideoEncoder(tempVideoFormat, outputVideoEncoder);

		var commands = BuildMergeVideoCommands(tempVideoPath, tempVideoFormat, tempVideoFps, trimFrameStart, tempFramePattern, outputVideoResolution, resolvedVideoEncoder, outputVideoQuality, outputVideoPreset, outputVideoFps);
		using var process = RunFfmpegWithProgress(commands, updateProgress, processManager, logger, logLevel);
		return process is not null && process.ExitCode == 0;
	}

	/// <summary>Command construction pulled out of <see cref="MergeVideo"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildMergeVideoCommands(string tempVideoPath, string? tempVideoFormat, double tempVideoFps, int trimFrameStart, string tempFramePattern, Resolution outputVideoResolution, VideoEncoder resolvedVideoEncoder, int outputVideoQuality, VideoPreset outputVideoPreset, double outputVideoFps)
	{
		var videoEncoderName = resolvedVideoEncoder.ToWireName();

		return FfmpegBuilder.Chain(
			FfmpegBuilder.SetInputFps(tempVideoFps),
			FfmpegBuilder.SetStartNumber(trimFrameStart),
			FfmpegBuilder.SetInput(tempFramePattern),
			FfmpegBuilder.SetMediaResolution(Vision.Vision.PackResolution(outputVideoResolution)),
			FfmpegBuilder.SetVideoEncoder(videoEncoderName),
			FfmpegBuilder.SetVideoTag(videoEncoderName, tempVideoFormat ?? string.Empty),
			FfmpegBuilder.SetVideoQuality(videoEncoderName, outputVideoQuality),
			AssertNoNullPreset(FfmpegBuilder.SetVideoPreset(videoEncoderName, outputVideoPreset.ToWireName())),
			FfmpegBuilder.Concat(
				FfmpegBuilder.SetVideoFps(outputVideoFps),
				FfmpegBuilder.KeepVideoAlpha(videoEncoderName),
				FfmpegBuilder.ConvertColorSpace("bt709")),
			FfmpegBuilder.SetPixelFormat(videoEncoderName),
			FfmpegBuilder.ForceOutput(tempVideoPath));
	}

	/// <summary>
	/// Python: <c>concat_video</c>. Writes an ffmpeg concat-demuxer list file to a real
	/// temp file (Python: <c>tempfile.mkstemp()</c>; here: <see cref="Path.GetTempFileName"/>),
	/// runs the concat, then removes the list file regardless of the result — same as
	/// Python's unconditional <c>remove_file(concat_video_path)</c> after <c>communicate()</c>.
	/// </summary>
	public static bool ConcatVideo(string outputPath, IReadOnlyList<string> tempOutputPaths, ProcessManager? processManager = null, Logger? logger = null, LogLevel? logLevel = null)
	{
		var concatVideoPath = Path.GetTempFileName();

		using (var concatVideoFile = new StreamWriter(concatVideoPath, append: false))
		{
			foreach (var tempOutputPath in tempOutputPaths)
			{
				concatVideoFile.Write("file '" + Path.GetFullPath(tempOutputPath) + "'" + Environment.NewLine);
			}
		}

		var resolvedOutputPath = Path.GetFullPath(outputPath);
		var outputVideoFormat = FileSystem.GetFileFormat(resolvedOutputPath) ?? string.Empty;

		var commands = BuildConcatVideoCommands(concatVideoPath, resolvedOutputPath, outputVideoFormat);
		using var process = RunFfmpeg(commands, processManager, logger, logLevel);
		// Python: process.communicate() drains any output run_ffmpeg's wait loop left
		// unread before checking returncode; ProcessRunner.Communicate does the same here
		// without the deadlock risk of a synchronous read (see its doc comment).
		var (_, _, exitCode) = ProcessRunner.Communicate(process);
		FileSystem.RemoveFile(concatVideoPath);

		return process is not null && exitCode == 0;
	}

	/// <summary>Command construction pulled out of <see cref="ConcatVideo"/> for direct unit testing (no process dependency).</summary>
	internal static IReadOnlyList<string> BuildConcatVideoCommands(string concatVideoPath, string resolvedOutputPath, string outputVideoFormat)
	{
		return FfmpegBuilder.Chain(
			FfmpegBuilder.UnsafeConcat(),
			FfmpegBuilder.SetInput(concatVideoPath),
			FfmpegBuilder.CopyVideoEncoder(),
			FfmpegBuilder.CopyAudioEncoder(),
			FfmpegBuilder.SetFaststart(outputVideoFormat),
			FfmpegBuilder.ForceOutput(resolvedOutputPath));
	}

	/// <summary>
	/// Python: <c>cast(VideoFormat, get_file_format(temp_video_path))</c> followed by
	/// <c>fix_video_encoder(temp_video_format, output_video_encoder)</c>. Python's
	/// <c>cast</c> is a type-only annotation with no runtime check; if the file's format
	/// string is not one of the <c>VideoFormat</c> literals (e.g. no extension), every
	/// <c>video_format ==</c>/<c>in</c> comparison inside <c>fix_video_encoder</c> simply
	/// fails to match and the original encoder passes through unchanged. Reproduced here by
	/// treating an unparseable format the same way — skip the fix-up — rather than by
	/// throwing.
	/// </summary>
	internal static VideoEncoder ResolveVideoEncoder(string? tempVideoFormat, VideoEncoder outputVideoEncoder)
	{
		if (tempVideoFormat is not null && EnumNames.TryFromWireName<VideoFormat>(tempVideoFormat, out var videoFormat))
		{
			return FixVideoEncoder(videoFormat, outputVideoEncoder);
		}

		return outputVideoEncoder;
	}

	/// <summary>See <see cref="ResolveVideoEncoder"/>; same reasoning for <c>fix_audio_encoder</c>.</summary>
	internal static AudioEncoder ResolveAudioEncoder(string? tempVideoFormat, AudioEncoder outputAudioEncoder)
	{
		if (tempVideoFormat is not null && EnumNames.TryFromWireName<VideoFormat>(tempVideoFormat, out var videoFormat))
		{
			return FixAudioEncoder(videoFormat, outputAudioEncoder);
		}

		return outputAudioEncoder;
	}

	/// <summary>
	/// <see cref="FfmpegBuilder.SetVideoPreset"/> only produces a null second element when
	/// its <c>videoPreset</c> string does not match any <see cref="VideoPreset"/> literal —
	/// see that method's own doc comment. Every call site in this file passes
	/// <c>videoPreset.ToWireName()</c> for a real <see cref="VideoPreset"/> enum value, and
	/// <c>MapNvencPreset</c>/<c>MapAmfPreset</c>/<c>MapQsvPreset</c> each cover all nine
	/// <see cref="VideoPreset"/> members exhaustively, so this can never actually observe a
	/// null. The throw documents that invariant instead of silently swallowing it with `!`.
	/// </summary>
	private static IReadOnlyList<string> AssertNoNullPreset(IReadOnlyList<string?> values)
	{
		var result = new string[values.Count];

		for (var index = 0; index < values.Count; index++)
		{
			result[index] = values[index] ?? throw new InvalidOperationException(
				"FfmpegBuilder.SetVideoPreset returned a null element for a real VideoPreset value; this should be unreachable.");
		}

		return result;
	}

	/// <summary>
	/// Python: the <c>if 'frame=' in __line__: _, frame_number = __line__.split('frame=')</c>
	/// / <c>update_progress(int(frame_number))</c> fragment inside
	/// <c>run_ffmpeg_with_progress</c>, pulled out as a pure function. Deviation from
	/// Python: Python's <c>int(...)</c> throws on a malformed trailer and would crash the
	/// whole run; this returns null instead so one unexpected progress line cannot abort
	/// an otherwise-successful encode.
	/// </summary>
	internal static int? TryParseFrameNumber(string line)
	{
		const string marker = "frame=";
		var index = line.IndexOf(marker, StringComparison.Ordinal);

		if (index < 0)
		{
			return null;
		}

		var frameNumberText = line[(index + marker.Length)..];

		return int.TryParse(frameNumberText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var frameNumber)
			? frameNumber
			: null;
	}

	internal enum EncoderLineKind
	{
		Audio,
		Video
	}

	/// <summary>
	/// Python: the <c>if line.startswith(' a'): audio_encoder = line.split()[1]</c> /
	/// <c>if line.startswith(' v'): video_encoder = line.split()[1]</c> fragment inside
	/// <c>get_available_encoder_set</c>, pulled out as a pure function. Deviation from
	/// Python: Python's bare <c>line.split()[1]</c> throws <c>IndexError</c> on a line with
	/// fewer than two tokens; this returns null instead.
	/// </summary>
	internal static (EncoderLineKind Kind, string EncoderName)? ParseEncoderLine(string lowerLine)
	{
		if (lowerLine.StartsWith(" a", StringComparison.Ordinal))
		{
			var tokens = lowerLine.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

			if (tokens.Length > 1)
			{
				return (EncoderLineKind.Audio, tokens[1]);
			}
		}

		if (lowerLine.StartsWith(" v", StringComparison.Ordinal))
		{
			var tokens = lowerLine.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

			if (tokens.Length > 1)
			{
				return (EncoderLineKind.Video, tokens[1]);
			}
		}

		return null;
	}

	private static void TryTerminate(Process process)
	{
		try
		{
			if (!process.HasExited)
			{
				// .NET has no cross-platform equivalent of Python's Popen.terminate()
				// (SIGTERM); Process.Kill() is the closest available primitive (SIGKILL
				// on POSIX). Documented divergence — not behaviourally significant for a
				// caller that only observes "did the process stop".
				process.Kill();
			}
		}
		catch (InvalidOperationException)
		{
			// Process exited between the check and Kill(); best-effort, same as Python's
			// terminate() racing process exit.
		}
	}

	private static int IndexOf<T>(IReadOnlyList<T> list, T value) where T : struct, Enum
	{
		for (var index = 0; index < list.Count; index++)
		{
			if (EqualityComparer<T>.Default.Equals(list[index], value))
			{
				return index;
			}
		}

		return -1;
	}

	/// <summary>Python: <c>list.insert(index, item)</c>, which clamps an out-of-range index to the list length.</summary>
	private static void InsertAt<T>(List<T> list, int index, T value)
	{
		list.Insert(Math.Min(index, list.Count), value);
	}
}

/// <summary>
/// Shared process-launch/capture plumbing for <see cref="Ffmpeg"/> and <see cref="Ffprobe"/>.
/// Internal, not itself a port of any single Python function — both modules call
/// <c>subprocess.Popen(...)</c> directly; this factors out the launch-and-capture mechanics
/// they have in common (including graceful "binary not found" handling) so it is written
/// once. Kept in this file rather than a new one per the assignment's file-scope constraint.
/// </summary>
internal static class ProcessRunner
{
	/// <summary>
	/// Starts a process from a command list whose first element is an executable path (as
	/// produced by <see cref="FfmpegBuilder.Run"/> / <see cref="FfprobeBuilder.Run"/>).
	/// Returns null instead of throwing when the executable could not be located (a null
	/// first element, mirroring <c>shutil.which</c> returning <c>None</c>) or failed to
	/// start, so callers can treat "no binary" as "no output" rather than crash.
	/// </summary>
	public static Process? TryStart(IReadOnlyList<string?> commands, bool redirectStdin = false, bool redirectStderr = true)
	{
		if (commands.Count == 0 || commands[0] is null)
		{
			return null;
		}

		var startInfo = new ProcessStartInfo
		{
			FileName = commands[0],
			RedirectStandardOutput = true,
			RedirectStandardError = redirectStderr,
			RedirectStandardInput = redirectStdin,
			UseShellExecute = false,
			CreateNoWindow = true
		};

		for (var index = 1; index < commands.Count; index++)
		{
			startInfo.ArgumentList.Add(commands[index] ?? string.Empty);
		}

		try
		{
			var process = new Process { StartInfo = startInfo };
			process.Start();
			return process;
		}
		catch (System.ComponentModel.Win32Exception)
		{
			// e.g. the resolved path stopped being executable between Which() and Start().
			return null;
		}
	}

	/// <summary>
	/// Reads stdout and stderr to completion concurrently — never one stream fully before
	/// the other, which deadlocks once the unread pipe's OS buffer fills — then waits for
	/// exit. Mirrors Python's <c>Popen.communicate()</c>.
	/// </summary>
	public static (string Stdout, string Stderr, int ExitCode) Communicate(Process? process)
	{
		if (process is null)
		{
			return (string.Empty, string.Empty, -1);
		}

		var stdoutTask = process.StandardOutput.ReadToEndAsync();
		var stderrTask = process.StartInfo.RedirectStandardError
			? process.StandardError.ReadToEndAsync()
			: Task.FromResult(string.Empty);

		Task.WaitAll(stdoutTask, stderrTask);
		process.WaitForExit();

		return (stdoutTask.Result, stderrTask.Result, process.ExitCode);
	}
}

/// <summary>
/// Wraps the long-lived ffmpeg subprocess <see cref="Ffmpeg.CreateVideoReader"/> opens.
/// Python's <c>create_video_reader</c> just returns the bare <c>subprocess.Popen</c>; this
/// port adds an explicit <see cref="IDisposable"/> wrapper (per docs/DOTNET_PORT_PLAN.md §5
/// disposal discipline) so a caller pumping frames off <see cref="StandardOutput"/> is
/// guaranteed to terminate and dispose the subprocess — including on an exception path
/// partway through the pump — via a single <c>using</c>/<c>Dispose()</c>, rather than
/// having to remember to kill a bare <see cref="Process"/> by hand. <see cref="IsAvailable"/>
/// is false (and <see cref="StandardOutput"/> null) when ffmpeg was not found on PATH — see
/// <see cref="Ffmpeg.OpenFfmpeg"/>.
/// </summary>
public sealed class VideoReaderProcess : IDisposable
{
	private readonly Process? _process;
	private bool _disposed;

	internal VideoReaderProcess(Process? process, string videoPath, VideoMetadata metadata, int frameNumber)
	{
		_process = process;
		VideoPath = videoPath;
		Metadata = metadata;
		FrameNumber = frameNumber;
	}

	public string VideoPath { get; }

	public VideoMetadata Metadata { get; }

	public int FrameNumber { get; }

	/// <summary>False when ffmpeg could not be located on PATH; no subprocess was started.</summary>
	public bool IsAvailable => _process is not null;

	/// <summary>
	/// Raw <c>bgr24</c>/<c>rawvideo</c> frame bytes ffmpeg writes to stdout. Null when
	/// <see cref="IsAvailable"/> is false.
	/// </summary>
	public Stream? StandardOutput => _process?.StandardOutput.BaseStream;

	/// <summary>Null while the process is still running, or when <see cref="IsAvailable"/> is false.</summary>
	public int? ExitCode => _process is { HasExited: true } ? _process.ExitCode : null;

	public void Dispose()
	{
		if (_disposed)
		{
			return;
		}

		_disposed = true;

		if (_process is null)
		{
			return;
		}

		try
		{
			if (!_process.HasExited)
			{
				try
				{
					_process.Kill();
				}
				catch (InvalidOperationException)
				{
					// Process exited between the HasExited check and Kill(); best-effort.
				}
			}
		}
		finally
		{
			_process.Dispose();
		}
	}
}

/// <summary>
/// Wraps the long-lived ffmpeg subprocess <see cref="Ffmpeg.CreateVideoWriter"/> opens. See
/// <see cref="VideoReaderProcess"/>'s doc comment for why this wrapper exists beyond the
/// bare <see cref="Process"/> Python's <c>create_video_writer</c> returns. Disposing closes
/// stdin (so ffmpeg can flush and exit cleanly if the caller already wrote every frame) and
/// otherwise terminates the process, same disposal guarantee as
/// <see cref="VideoReaderProcess"/>.
/// </summary>
public sealed class VideoWriterProcess : IDisposable
{
	private readonly Process? _process;
	private bool _disposed;

	internal VideoWriterProcess(Process? process, string targetPath, string tempVideoPath, double tempVideoFps, Resolution outputVideoResolution, double outputVideoFps)
	{
		_process = process;
		TargetPath = targetPath;
		TempVideoPath = tempVideoPath;
		Metadata = new VideoWriterMetadata(outputVideoFps, outputVideoResolution);
		TempVideoFps = tempVideoFps;
	}

	public string TargetPath { get; }

	public string TempVideoPath { get; }

	public double TempVideoFps { get; }

	public VideoWriterMetadata Metadata { get; }

	/// <summary>False when ffmpeg could not be located on PATH; no subprocess was started.</summary>
	public bool IsAvailable => _process is not null;

	/// <summary>
	/// Raw frame bytes the caller writes for ffmpeg to encode. Null when
	/// <see cref="IsAvailable"/> is false.
	/// </summary>
	public Stream? StandardInput => _process?.StandardInput.BaseStream;

	/// <summary>Null while the process is still running, or when <see cref="IsAvailable"/> is false.</summary>
	public int? ExitCode => _process is { HasExited: true } ? _process.ExitCode : null;

	/// <summary>
	/// Closes stdin so ffmpeg observes end-of-input and flushes/exits, then waits for it to
	/// exit. Callers that have finished writing every frame should call this (rather than
	/// going straight to <see cref="Dispose"/>) so ffmpeg finalizes the output file instead
	/// of being killed mid-encode.
	/// </summary>
	public void FinishWriting()
	{
		if (_process is null)
		{
			return;
		}

		_process.StandardInput.Close();
		_process.WaitForExit();
	}

	public void Dispose()
	{
		if (_disposed)
		{
			return;
		}

		_disposed = true;

		if (_process is null)
		{
			return;
		}

		try
		{
			if (!_process.HasExited)
			{
				try
				{
					_process.Kill();
				}
				catch (InvalidOperationException)
				{
					// Process exited between the HasExited check and Kill(); best-effort.
				}
			}
		}
		finally
		{
			_process.Dispose();
		}
	}
}
