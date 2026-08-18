using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using FaceFusion.Core;
using FaceFusion.Types;

// Grants the test project access to the internal parsing helpers on Ffmpeg/Ffprobe
// (TryParseFrameNumber, ParseEncoderLine, ParseEntries, ExtractVideoFps) so they can be
// unit-tested directly against canned strings without a real process. A C# assembly
// attribute, not a project-file edit.
[assembly: InternalsVisibleTo("FaceFusion.UnitTests")]

namespace FaceFusion.Media;

/// <summary>
/// Port of facefusion/ffmpeg.py — the ffmpeg process runners.
///
/// Scope note: this file ports the process-launching/parsing surface of ffmpeg.py
/// (<c>run_ffmpeg</c>, <c>run_ffmpeg_with_progress</c>, <c>open_ffmpeg</c>,
/// <c>log_debug</c>, <c>get_available_encoder_set</c>, <c>fix_audio_encoder</c>,
/// <c>fix_video_encoder</c>) plus the pure parsing helpers pulled out of them. The
/// higher-level pipeline functions in ffmpeg.py (<c>create_video_reader</c>,
/// <c>create_video_writer</c>, <c>extract_frames</c>, <c>copy_image</c>,
/// <c>finalize_image</c>, <c>read_audio_buffer</c>, <c>restore_audio</c>,
/// <c>replace_audio</c>, <c>merge_video</c>, <c>concat_video</c>) call into
/// <c>facefusion.vision</c> and <c>facefusion.state_manager</c>. Neither has a C# port
/// yet — <c>FaceFusion.Media.csproj</c> has no project reference to a Vision project (none
/// exists in this solution yet) and there is no state-manager port to take values from —
/// so those functions are out of scope here per the assignment ("port the ffmpeg/ffprobe
/// process runners... command builders are already ported"; the pipeline functions are
/// orchestration on top of a runner, not the runner itself) and are left for whoever ports
/// facefusion/vision.py to pick up alongside it.
///
/// Design note (deliberate, see port report): line/output parsing lives in methods that
/// take plain strings and have no <see cref="Process"/> dependency —
/// <see cref="TryParseFrameNumber"/> (the <c>-progress</c> "frame=" line ported out of
/// <c>run_ffmpeg_with_progress</c>) and <see cref="ParseEncoderLine"/> (the
/// <c>-encoders</c> line ported out of <c>get_available_encoder_set</c>) — so they are
/// unit-testable with canned output with no ffmpeg binary present.
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
		LogLevel? logLevel = null)
	{
		var extendedCommands = new List<string>(commands);
		extendedCommands.AddRange(FfmpegBuilder.SetProgress());
		extendedCommands.AddRange(FfmpegBuilder.CastStream());
		var fullCommands = FfmpegBuilder.Run(extendedCommands);
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
		LogLevel? logLevel = null)
	{
		var fullCommands = FfmpegBuilder.Run(commands);
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
	public static Process? OpenFfmpeg(IReadOnlyList<string> commands)
	{
		var fullCommands = FfmpegBuilder.Run(commands);
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
		LogLevel? logLevel = null)
	{
		var audioEncoders = new List<AudioEncoder>();
		var videoEncoders = new List<VideoEncoder>();
		var commands = FfmpegBuilder.Chain(FfmpegBuilder.GetEncoders());
		var process = RunFfmpeg(commands, processManager, logger, logLevel);

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
