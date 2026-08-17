using System.Globalization;

namespace FaceFusion.Media;

/// <summary>
/// Port of facefusion/ffmpeg_builder.py. Pure functions that build ffmpeg command
/// argument lists. No I/O beyond locating the ffmpeg executable on PATH, no global state.
/// </summary>
public static class FfmpegBuilder
{
	/// <summary>
	/// Prepends the ffmpeg executable path (as found via a PATH search, mirroring Python's
	/// shutil.which) and the base logging flags. When ffmpeg is not on PATH, Python's
	/// shutil.which('ffmpeg') returns None and that None ends up as the first list entry;
	/// we reproduce that by allowing a null first element.
	/// </summary>
	public static IReadOnlyList<string?> Run(IReadOnlyList<string> commands)
	{
		var result = new List<string?> { Which("ffmpeg"), "-loglevel", "error" };
		result.AddRange(commands);
		return result;
	}

	public static IReadOnlyList<string> Chain(params IReadOnlyList<string>[] commands)
	{
		var result = new List<string>();

		foreach (var command in commands)
		{
			result.AddRange(command);
		}

		return result;
	}

	public static IReadOnlyList<string> Concat(params IReadOnlyList<string>[] commands)
	{
		var result = new List<string>();
		var order = new List<string>();
		var commandSet = new Dictionary<string, List<string>>();

		foreach (var command in commands)
		{
			for (var index = 0; index + 1 < command.Count; index += 2)
			{
				var argument = command[index];
				var value = command[index + 1];

				if (!commandSet.TryGetValue(argument, out var values))
				{
					values = new List<string>();
					commandSet[argument] = values;
					order.Add(argument);
				}

				values.Add(value);
			}
		}

		foreach (var argument in order)
		{
			result.Add(argument);
			result.Add(string.Join(",", commandSet[argument]));
		}

		return result;
	}

	public static IReadOnlyList<string> GetEncoders()
	{
		return new[] { "-encoders" };
	}

	public static IReadOnlyList<string> SetHardwareAccelerator(string value)
	{
		return new[] { "-hwaccel", value };
	}

	public static IReadOnlyList<string> SetProgress()
	{
		return new[] { "-progress" };
	}

	public static IReadOnlyList<string> SetInput(string inputPath)
	{
		return new[] { "-i", inputPath };
	}

	public static IReadOnlyList<string> SetInputFps(double inputFps)
	{
		return new[] { "-r", FormatFps(inputFps) };
	}

	public static IReadOnlyList<string> SetStartNumber(int frameNumber)
	{
		return new[] { "-start_number", frameNumber.ToString(CultureInfo.InvariantCulture) };
	}

	public static IReadOnlyList<string> SetOutput(string outputPath)
	{
		return new[] { outputPath };
	}

	public static IReadOnlyList<string> ForceOutput(string outputPath)
	{
		return new[] { "-y", outputPath };
	}

	public static IReadOnlyList<string> CastStream()
	{
		return new[] { "-" };
	}

	// TODO(types): tighten streamMode to FaceFusion.Types.StreamMode once available
	public static IReadOnlyList<string> SetStreamMode(string streamMode)
	{
		if (streamMode == "udp")
		{
			return new[] { "-f", "mpegts" };
		}

		if (streamMode == "v4l2")
		{
			return new[] { "-f", "v4l2" };
		}

		return Array.Empty<string>();
	}

	public static IReadOnlyList<string> SetStreamQuality(int streamQuality)
	{
		return new[] { "-b:v", streamQuality.ToString(CultureInfo.InvariantCulture) + "k" };
	}

	public static IReadOnlyList<string> UnsafeConcat()
	{
		return new[] { "-f", "concat", "-safe", "0" };
	}

	public static IReadOnlyList<string> SeekTo(double time)
	{
		return new[] { "-ss", FormatPyFloat(time) };
	}

	public static IReadOnlyList<string> SetOutputFormat(string outputFormat)
	{
		return new[] { "-f", outputFormat };
	}

	public static IReadOnlyList<string> EnforcePixelFormat(string pixelFormat)
	{
		return new[] { "-pix_fmt", pixelFormat };
	}

	// TODO(types): tighten videoEncoder to FaceFusion.Types.VideoEncoder once available
	public static IReadOnlyList<string> SetPixelFormat(string videoEncoder)
	{
		if (videoEncoder == "rawvideo")
		{
			return new[] { "-pix_fmt", "rgb24" };
		}

		if (videoEncoder == "libvpx-vp9")
		{
			return new[] { "-pix_fmt", "yuva420p" };
		}

		return new[] { "-pix_fmt", "yuv420p" };
	}

	public static IReadOnlyList<string> SetFrameQuality(int frameQuality)
	{
		return new[] { "-q:v", frameQuality.ToString(CultureInfo.InvariantCulture) };
	}

	public static IReadOnlyList<string> SelectFrameRange(int? frameStart, int? frameEnd, double videoFps)
	{
		if (frameStart.HasValue && frameEnd.HasValue)
		{
			return new[] { "-vf", "trim=start_frame=" + frameStart.Value.ToString(CultureInfo.InvariantCulture) + ":end_frame=" + frameEnd.Value.ToString(CultureInfo.InvariantCulture) + ",fps=" + FormatFps(videoFps) };
		}

		if (frameStart.HasValue)
		{
			return new[] { "-vf", "trim=start_frame=" + frameStart.Value.ToString(CultureInfo.InvariantCulture) + ",fps=" + FormatFps(videoFps) };
		}

		if (frameEnd.HasValue)
		{
			return new[] { "-vf", "trim=end_frame=" + frameEnd.Value.ToString(CultureInfo.InvariantCulture) + ",fps=" + FormatFps(videoFps) };
		}

		return new[] { "-vf", "fps=" + FormatFps(videoFps) };
	}

	public static IReadOnlyList<string> PreventFrameDrop()
	{
		return new[] { "-fps_mode", "passthrough" };
	}

	// TODO(types): tighten colorTransfer to FaceFusion.Types.ColorTransfer once available
	public static IReadOnlyList<string> RestrictColorTransfer(string colorTransfer)
	{
		if (colorTransfer == "smpte2084" || colorTransfer == "arib-std-b67")
		{
			return new[] { "-vf", "scale=out_primaries=bt709:out_transfer=bt709:intent=perceptual" };
		}

		return Array.Empty<string>();
	}

	// TODO(types): tighten colorSpace to FaceFusion.Types.ColorSpace once available
	public static IReadOnlyList<string> ConvertColorSpace(string colorSpace)
	{
		return new[] { "-vf", "scale=out_color_matrix=" + colorSpace + ":out_range=tv,setparams=colorspace=" + colorSpace + ":color_primaries=" + colorSpace + ":color_trc=" + colorSpace };
	}

	public static IReadOnlyList<string> SelectMediaRange(int? frameStart, int? frameEnd, double mediaFps)
	{
		var result = new List<string>();

		if (frameStart.HasValue)
		{
			result.Add("-ss");
			result.Add(FormatPyFloat(frameStart.Value / mediaFps));
		}

		if (frameEnd.HasValue)
		{
			result.Add("-to");
			result.Add(FormatPyFloat(frameEnd.Value / mediaFps));
		}

		return result;
	}

	public static IReadOnlyList<string> SelectMediaStream(string mediaStream)
	{
		return new[] { "-map", mediaStream };
	}

	public static IReadOnlyList<string> SetMediaResolution(string videoResolution)
	{
		return new[] { "-s", videoResolution };
	}

	public static IReadOnlyList<string> SetImageQuality(string imagePath, int imageQuality)
	{
		if (GetFileFormat(imagePath) == "webp")
		{
			return new[] { "-q:v", imageQuality.ToString(CultureInfo.InvariantCulture) };
		}

		var imageCompression = (int)Math.Round(31 - (imageQuality * 0.31), MidpointRounding.ToEven);
		return new[] { "-q:v", imageCompression.ToString(CultureInfo.InvariantCulture) };
	}

	// TODO(types): tighten audioCodec to FaceFusion.Types.AudioEncoder once available
	public static IReadOnlyList<string> SetAudioEncoder(string audioCodec)
	{
		return new[] { "-c:a", audioCodec };
	}

	public static IReadOnlyList<string> CopyAudioEncoder()
	{
		return SetAudioEncoder("copy");
	}

	public static IReadOnlyList<string> SetAudioSampleRate(int audioSampleRate)
	{
		return new[] { "-ar", audioSampleRate.ToString(CultureInfo.InvariantCulture) };
	}

	public static IReadOnlyList<string> SetAudioSampleSize(int audioSampleSize)
	{
		if (audioSampleSize == 16)
		{
			return new[] { "-f", "s16le" };
		}

		if (audioSampleSize == 32)
		{
			return new[] { "-f", "s32le" };
		}

		return Array.Empty<string>();
	}

	public static IReadOnlyList<string> SetAudioChannelTotal(int audioChannelTotal)
	{
		return new[] { "-ac", audioChannelTotal.ToString(CultureInfo.InvariantCulture) };
	}

	// TODO(types): tighten audioEncoder to FaceFusion.Types.AudioEncoder once available
	public static IReadOnlyList<string> SetAudioQuality(string audioEncoder, int audioQuality)
	{
		if (audioEncoder == "aac")
		{
			var audioCompression = Math.Round(Interp(audioQuality, 0, 100, 0.1, 2.0), 1, MidpointRounding.ToEven);
			return new[] { "-q:a", FormatPyFloat(audioCompression) };
		}

		if (audioEncoder == "libmp3lame")
		{
			var audioCompression = (int)Math.Round(Interp(audioQuality, 0, 100, 9, 0), MidpointRounding.ToEven);
			return new[] { "-q:a", audioCompression.ToString(CultureInfo.InvariantCulture) };
		}

		if (audioEncoder == "libopus")
		{
			var audioBitRate = (int)Math.Round(Interp(audioQuality, 0, 100, 64, 256), MidpointRounding.ToEven);
			return new[] { "-b:a", audioBitRate.ToString(CultureInfo.InvariantCulture) + "k" };
		}

		if (audioEncoder == "libvorbis")
		{
			var audioCompression = Math.Round(Interp(audioQuality, 0, 100, -1, 10), 1, MidpointRounding.ToEven);
			return new[] { "-q:a", FormatPyFloat(audioCompression) };
		}

		return Array.Empty<string>();
	}

	public static IReadOnlyList<string> SetAudioVolume(int audioVolume)
	{
		return new[] { "-filter:a", "volume=" + FormatPyFloat(audioVolume / 100.0) };
	}

	public static IReadOnlyList<string> SetThreadCount(int threadCount)
	{
		return new[] { "-threads", threadCount.ToString(CultureInfo.InvariantCulture) };
	}

	public static IReadOnlyList<string> SetVideoEncoder(string videoEncoder)
	{
		return new[] { "-c:v", videoEncoder };
	}

	public static IReadOnlyList<string> CopyVideoEncoder()
	{
		return SetVideoEncoder("copy");
	}

	// TODO(types): tighten videoFormat to FaceFusion.Types.VideoFormat once available
	public static IReadOnlyList<string> SetFaststart(string videoFormat)
	{
		if (videoFormat is "m4v" or "mov" or "mp4")
		{
			return new[] { "-movflags", "+faststart" };
		}

		return Array.Empty<string>();
	}

	// TODO(types): tighten videoEncoder/videoFormat to FaceFusion.Types once available
	public static IReadOnlyList<string> SetVideoTag(string videoEncoder, string videoFormat)
	{
		if (videoFormat is "m4v" or "mov" or "mp4" && videoEncoder is "libx265" or "hevc_nvenc" or "hevc_amf" or "hevc_qsv" or "hevc_videotoolbox")
		{
			return new[] { "-tag:v", "hvc1" };
		}

		return Array.Empty<string>();
	}

	// TODO(types): tighten videoEncoder to FaceFusion.Types.VideoEncoder once available
	public static IReadOnlyList<string> SetVideoQuality(string videoEncoder, int videoQuality)
	{
		if (videoEncoder is "libx264" or "libx264rgb" or "libx265")
		{
			var videoCompression = (int)Math.Round(Interp(videoQuality, 0, 100, 51, 0), MidpointRounding.ToEven);
			return new[] { "-crf", videoCompression.ToString(CultureInfo.InvariantCulture) };
		}

		if (videoEncoder == "libvpx-vp9")
		{
			var videoCompression = (int)Math.Round(Interp(videoQuality, 0, 100, 63, 0), MidpointRounding.ToEven);
			return new[] { "-crf", videoCompression.ToString(CultureInfo.InvariantCulture) };
		}

		if (videoEncoder is "h264_nvenc" or "hevc_nvenc")
		{
			var videoCompression = (int)Math.Round(Interp(videoQuality, 0, 100, 51, 0), MidpointRounding.ToEven);
			return new[] { "-cq", videoCompression.ToString(CultureInfo.InvariantCulture) };
		}

		if (videoEncoder is "h264_amf" or "hevc_amf")
		{
			var videoCompression = (int)Math.Round(Interp(videoQuality, 0, 100, 51, 0), MidpointRounding.ToEven);
			var videoCompressionText = videoCompression.ToString(CultureInfo.InvariantCulture);
			return new[] { "-qp_i", videoCompressionText, "-qp_p", videoCompressionText, "-qp_b", videoCompressionText };
		}

		if (videoEncoder is "h264_qsv" or "hevc_qsv")
		{
			var videoCompression = (int)Math.Round(Interp(videoQuality, 0, 100, 51, 0), MidpointRounding.ToEven);
			return new[] { "-qp", videoCompression.ToString(CultureInfo.InvariantCulture) };
		}

		if (videoEncoder is "h264_videotoolbox" or "hevc_videotoolbox")
		{
			var videoBitRate = (int)Math.Round(Interp(videoQuality, 0, 100, 1024, 50512), MidpointRounding.ToEven);
			return new[] { "-b:v", videoBitRate.ToString(CultureInfo.InvariantCulture) + "k" };
		}

		return Array.Empty<string>();
	}

	// TODO(types): tighten videoEncoder/videoPreset to FaceFusion.Types once available
	//
	// Python builds e.g. [ '-preset', map_nvenc_preset(video_preset) ] directly, and the
	// map_*_preset helpers can return None. Unlike the other builders here, this means the
	// resulting command list can genuinely contain a null second element (Python allows
	// None in a list) instead of omitting it, so the return type is nullable to match.
	public static IReadOnlyList<string?> SetVideoPreset(string videoEncoder, string videoPreset)
	{
		if (videoEncoder is "libx264" or "libx264rgb" or "libx265")
		{
			return new[] { "-preset", videoPreset };
		}

		if (videoEncoder is "h264_nvenc" or "hevc_nvenc")
		{
			return new[] { "-preset", MapNvencPreset(videoPreset) };
		}

		if (videoEncoder is "h264_amf" or "hevc_amf")
		{
			return new[] { "-quality", MapAmfPreset(videoPreset) };
		}

		if (videoEncoder is "h264_qsv" or "hevc_qsv")
		{
			return new[] { "-preset", MapQsvPreset(videoPreset) };
		}

		return Array.Empty<string?>();
	}

	public static IReadOnlyList<string> SetVideoFps(double videoFps)
	{
		return new[] { "-vf", "fps=" + FormatFps(videoFps) };
	}

	public static IReadOnlyList<string> SetVideoDuration(double videoDuration)
	{
		return new[] { "-t", FormatPyFloat(videoDuration) };
	}

	// TODO(types): tighten videoEncoder to FaceFusion.Types.VideoEncoder once available
	public static IReadOnlyList<string> KeepVideoAlpha(string videoEncoder)
	{
		if (videoEncoder == "libvpx-vp9")
		{
			return new[] { "-vf", "format=yuva420p" };
		}

		return Array.Empty<string>();
	}

	public static IReadOnlyList<string> CaptureVideo()
	{
		return new[] { "-f", "rawvideo", "-pix_fmt", "rgb24" };
	}

	public static IReadOnlyList<string> IgnoreVideoStream()
	{
		return new[] { "-vn" };
	}

	// TODO(types): tighten videoPreset to FaceFusion.Types.VideoPreset once available
	public static string? MapNvencPreset(string videoPreset)
	{
		if (videoPreset is "ultrafast" or "superfast" or "veryfast" or "faster" or "fast")
		{
			return "fast";
		}

		if (videoPreset == "medium")
		{
			return "medium";
		}

		if (videoPreset is "slow" or "slower" or "veryslow")
		{
			return "slow";
		}

		return null;
	}

	// TODO(types): tighten videoPreset to FaceFusion.Types.VideoPreset once available
	public static string? MapAmfPreset(string videoPreset)
	{
		if (videoPreset is "ultrafast" or "superfast" or "veryfast")
		{
			return "speed";
		}

		if (videoPreset is "faster" or "fast" or "medium")
		{
			return "balanced";
		}

		if (videoPreset is "slow" or "slower" or "veryslow")
		{
			return "quality";
		}

		return null;
	}

	// TODO(types): tighten videoPreset to FaceFusion.Types.VideoPreset once available
	public static string? MapQsvPreset(string videoPreset)
	{
		if (videoPreset is "ultrafast" or "superfast" or "veryfast")
		{
			return "veryfast";
		}

		if (videoPreset is "faster" or "fast" or "medium" or "slow" or "slower" or "veryslow")
		{
			return videoPreset;
		}

		return null;
	}

	/// <summary>
	/// Locates an executable on PATH, mirroring Python's shutil.which. Returns null when
	/// not found (as Python's shutil.which does) rather than throwing.
	///
	/// This is a minimal local stand-in for a shared PATH-search helper; it duplicates
	/// logic that other ported modules may also need and should be de-duplicated into a
	/// shared location once one exists.
	/// </summary>
	private static string? Which(string executable)
	{
		var pathVariable = Environment.GetEnvironmentVariable("PATH");

		if (string.IsNullOrEmpty(pathVariable))
		{
			return null;
		}

		var candidateNames = OperatingSystem.IsWindows()
			? new[] { executable + ".exe", executable + ".cmd", executable + ".bat", executable }
			: new[] { executable };

		foreach (var directory in pathVariable.Split(Path.PathSeparator))
		{
			if (string.IsNullOrEmpty(directory))
			{
				continue;
			}

			foreach (var candidateName in candidateNames)
			{
				var candidatePath = Path.Combine(directory, candidateName);

				if (File.Exists(candidatePath))
				{
					return candidatePath;
				}
			}
		}

		return null;
	}

	/// <summary>
	/// Minimal local port of facefusion/filesystem.py's get_file_format, limited to what
	/// SetImageQuality needs (the 'webp' comparison). Do not extend this into a general
	/// filesystem helper here; filesystem.py is a separate, unported module.
	/// </summary>
	private static string? GetFileFormat(string filePath)
	{
		var fileExtension = Path.GetExtension(filePath);

		if (string.IsNullOrEmpty(fileExtension))
		{
			return null;
		}

		fileExtension = fileExtension.ToLowerInvariant();

		return fileExtension switch
		{
			".jpg" => "jpeg",
			".tif" => "tiff",
			".mpg" => "mpeg",
			_ => fileExtension.TrimStart('.'),
		};
	}

	/// <summary>
	/// Linear interpolation matching numpy.interp for a scalar x against a two-point
	/// (x0, x1) -> (y0, y1) table, clamping at the boundaries.
	/// </summary>
	private static double Interp(double x, double x0, double x1, double y0, double y1)
	{
		if (x <= x0)
		{
			return y0;
		}

		if (x >= x1)
		{
			return y1;
		}

		return y0 + ((y1 - y0) * (x - x0) / (x1 - x0));
	}

	/// <summary>
	/// Formats an Fps-typed value the way the Python call sites in this module do: those
	/// pass plain str() over whatever numeric value they hold, which in practice (and in
	/// the ported tests) is often an integral value expected to render without a decimal
	/// point (e.g. "30", not "30.0"). Matches .NET's default double formatting.
	/// </summary>
	private static string FormatFps(double value)
	{
		return value.ToString(CultureInfo.InvariantCulture);
	}

	/// <summary>
	/// Formats a value the way Python's str() formats an actual float: always with a
	/// decimal point, e.g. str(0.0) == "0.0", str(1.5) == "1.5".
	/// </summary>
	private static string FormatPyFloat(double value)
	{
		var formatted = value.ToString("R", CultureInfo.InvariantCulture);

		if (!formatted.Contains('.') && !formatted.Contains('E') && !formatted.Contains('e'))
		{
			formatted += ".0";
		}

		return formatted;
	}
}
