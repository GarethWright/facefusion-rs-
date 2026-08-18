using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using FaceFusion.Types;

namespace FaceFusion.Media;

/// <summary>
/// Port of facefusion/ffprobe.py. Runs ffprobe as a child process and parses its
/// key=value output into metadata.
///
/// Design note (deliberate, see port report): all string parsing lives in methods that
/// take plain strings (<see cref="ParseEntries"/>, <see cref="ExtractVideoFps"/>) and have
/// no dependency on <see cref="Process"/> at all, so they are testable with canned output
/// with no ffprobe binary present. Only <see cref="RunFfprobe"/> and the
/// <c>Probe*Entries</c>/<c>Extract*Metadata</c> methods above them touch the process
/// launcher.
/// </summary>
public static class Ffprobe
{
	// Python: @lru_cache(maxsize = 128) on extract_static_audio_metadata /
	// extract_static_video_metadata. This is a pure memoization cache (not shared mutable
	// state read from elsewhere), so per port convention rule 5 it is fine to keep as a
	// module-level cache like Python's decorator does; unlike lru_cache it has no eviction
	// bound, which is an accepted, documented divergence (a 128-entry LRU has no direct
	// BCL equivalent without hand-rolling one, and eviction policy is not behaviourally
	// significant here).
	private static readonly ConcurrentDictionary<string, AudioMetadata> StaticAudioMetadataCache = new(StringComparer.Ordinal);
	private static readonly ConcurrentDictionary<string, VideoMetadata> StaticVideoMetadataCache = new(StringComparer.Ordinal);

	/// <summary>
	/// Python: <c>run_ffprobe</c>. Starts ffprobe with stdout/stderr redirected. Returns
	/// null when ffprobe cannot be located on PATH (Python's <c>shutil.which</c> returning
	/// <c>None</c> would make <c>subprocess.Popen</c> raise; this port instead fails
	/// gracefully so callers can treat "binary not found" the same as "no output").
	/// </summary>
	public static Process? RunFfprobe(IReadOnlyList<string> commands)
	{
		var fullCommands = FfprobeBuilder.Run(commands);
		return ProcessRunner.TryStart(fullCommands, redirectStdin: false, redirectStderr: true);
	}

	/// <summary>
	/// Python: <c>parse_entries</c>. Pure string parsing, no process access — the piece of
	/// this module most worth unit-testing directly.
	/// </summary>
	internal static IReadOnlyDictionary<string, string> ParseEntries(string? output)
	{
		var mediaEntries = new Dictionary<string, string>(StringComparer.Ordinal);

		if (string.IsNullOrEmpty(output))
		{
			return mediaEntries;
		}

		// Python: output.decode().strip().splitlines()
		var lines = output.Trim().Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

		foreach (var line in lines)
		{
			var separatorIndex = line.IndexOf('=');

			if (separatorIndex >= 0)
			{
				var key = line[..separatorIndex];
				var value = line[(separatorIndex + 1)..];
				mediaEntries[key] = value;
			}
		}

		return mediaEntries;
	}

	/// <summary>Python: <c>probe_audio_entries</c>.</summary>
	public static IReadOnlyDictionary<string, string> ProbeAudioEntries(string audioPath, IReadOnlyList<string> entries)
	{
		var commands = FfprobeBuilder.Chain(
			FfprobeBuilder.SelectStream("a:0"),
			FfprobeBuilder.ShowStreamEntries(entries),
			FfprobeBuilder.FormatToKeyValue(),
			FfprobeBuilder.SetInput(audioPath));

		var output = ProcessRunner.Communicate(RunFfprobe(commands)).Stdout;
		return ParseEntries(output);
	}

	/// <summary>Python: <c>probe_video_entries</c>.</summary>
	public static IReadOnlyDictionary<string, string> ProbeVideoEntries(string videoPath, IReadOnlyList<string> entries)
	{
		var commands = FfprobeBuilder.Chain(
			FfprobeBuilder.SelectStream("v:0"),
			FfprobeBuilder.ShowStreamEntries(entries),
			FfprobeBuilder.FormatToKeyValue(),
			FfprobeBuilder.SetInput(videoPath));

		var output = ProcessRunner.Communicate(RunFfprobe(commands)).Stdout;
		return ParseEntries(output);
	}

	/// <summary>Python: <c>probe_format_entries</c>.</summary>
	public static IReadOnlyDictionary<string, string> ProbeFormatEntries(string mediaPath, IReadOnlyList<string> entries)
	{
		var commands = FfprobeBuilder.Chain(
			FfprobeBuilder.ShowFormatEntries(entries),
			FfprobeBuilder.FormatToKeyValue(),
			FfprobeBuilder.SetInput(mediaPath));

		var output = ProcessRunner.Communicate(RunFfprobe(commands)).Stdout;
		return ParseEntries(output);
	}

	/// <summary>Python: <c>extract_static_audio_metadata</c> (memoized).</summary>
	public static AudioMetadata ExtractStaticAudioMetadata(string audioPath)
		=> StaticAudioMetadataCache.GetOrAdd(audioPath, ExtractAudioMetadata);

	/// <summary>Python: <c>extract_audio_metadata</c>.</summary>
	public static AudioMetadata ExtractAudioMetadata(string audioPath)
	{
		var audioEntries = ProbeAudioEntries(audioPath, new[] { "sample_rate", "channels" });
		var formatEntries = ProbeFormatEntries(audioPath, new[] { "duration", "bit_rate" });

		var duration = double.Parse(formatEntries["duration"], CultureInfo.InvariantCulture);
		var sampleRate = int.Parse(audioEntries["sample_rate"], CultureInfo.InvariantCulture);
		var frameTotal = (int)Math.Round(duration * sampleRate, MidpointRounding.ToEven);
		var channelTotal = int.Parse(audioEntries["channels"], CultureInfo.InvariantCulture);
		var bitRate = int.Parse(formatEntries["bit_rate"], CultureInfo.InvariantCulture);

		return new AudioMetadata(duration, frameTotal, channelTotal, sampleRate, bitRate);
	}

	/// <summary>Python: <c>extract_static_video_metadata</c> (memoized).</summary>
	public static VideoMetadata ExtractStaticVideoMetadata(string videoPath)
		=> StaticVideoMetadataCache.GetOrAdd(videoPath, ExtractVideoMetadata);

	/// <summary>Python: <c>extract_video_metadata</c>.</summary>
	public static VideoMetadata ExtractVideoMetadata(string videoPath)
	{
		var videoEntries = ProbeVideoEntries(videoPath, new[] { "width", "height", "r_frame_rate", "color_transfer" });
		var formatEntries = ProbeFormatEntries(videoPath, new[] { "duration", "bit_rate" });

		var duration = double.Parse(formatEntries["duration"], CultureInfo.InvariantCulture);
		var fps = ExtractVideoFps(videoEntries.GetValueOrDefault("r_frame_rate"));
		var frameTotal = (int)Math.Round(duration * fps, MidpointRounding.ToEven);
		var width = int.Parse(videoEntries["width"], CultureInfo.InvariantCulture);
		var height = int.Parse(videoEntries["height"], CultureInfo.InvariantCulture);
		var bitRate = int.Parse(formatEntries["bit_rate"], CultureInfo.InvariantCulture);
		// Python: video_entries.get('color_transfer', 'unknown') — only this field defaults.
		var colorTransfer = videoEntries.GetValueOrDefault("color_transfer", "unknown");

		return new VideoMetadata(duration, frameTotal, fps, new Resolution(width, height), bitRate, colorTransfer);
	}

	/// <summary>
	/// Python: <c>extract_video_fps</c>. Handles the "30000/1001" rational form ffprobe
	/// emits for <c>r_frame_rate</c>. Pure and testable without a binary.
	/// </summary>
	internal static double ExtractVideoFps(string? frameRate)
	{
		if (!string.IsNullOrEmpty(frameRate) && frameRate.Contains('/'))
		{
			var parts = frameRate.Split('/');
			var numerator = int.Parse(parts[0], CultureInfo.InvariantCulture);
			var denominator = int.Parse(parts[1], CultureInfo.InvariantCulture);

			if (numerator != 0 && denominator != 0)
			{
				return (double)numerator / denominator;
			}
		}

		return 0.0;
	}
}
