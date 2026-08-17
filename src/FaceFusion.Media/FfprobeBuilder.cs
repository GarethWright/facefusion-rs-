using System;
using System.Collections.Generic;
using System.Linq;

namespace FaceFusion.Media
{
	/// <summary>
	/// Builds ffprobe command-line arguments.
	/// Ported from facefusion/ffprobe_builder.py
	/// </summary>
	public static class FfprobeBuilder
	{
		/// <summary>
		/// Prepends ffprobe executable path and error logging level to commands.
		/// Note: Returns a list that may contain null (matching Python's shutil.which behavior).
		/// </summary>
		public static IReadOnlyList<string?> Run(IReadOnlyList<string> commands)
		{
			// TODO(types): Command type alias should be resolved when FaceFusion.Types is available
			var result = new List<string?>();
			result.Add(Which("ffprobe"));
			result.Add("-loglevel");
			result.Add("error");
			result.AddRange(commands);
			return result;
		}

		/// <summary>
		/// Chains multiple command lists together.
		/// Equivalent to itertools.chain(*commands) in Python.
		/// </summary>
		public static IReadOnlyList<string> Chain(params IReadOnlyList<string>[] commands)
		{
			return commands.SelectMany(c => c).ToList();
		}

		/// <summary>
		/// Selects a specific stream.
		/// </summary>
		public static IReadOnlyList<string> SelectStream(string stream)
		{
			return new[] { "-select_streams", stream };
		}

		/// <summary>
		/// Shows specific stream entries.
		/// </summary>
		public static IReadOnlyList<string> ShowStreamEntries(IReadOnlyList<string> entries)
		{
			return new[] { "-show_entries", "stream=" + string.Join(",", entries) };
		}

		/// <summary>
		/// Shows specific format entries.
		/// </summary>
		public static IReadOnlyList<string> ShowFormatEntries(IReadOnlyList<string> entries)
		{
			return new[] { "-show_entries", "format=" + string.Join(",", entries) };
		}

		/// <summary>
		/// Formats output to key-value format.
		/// </summary>
		public static IReadOnlyList<string> FormatToKeyValue()
		{
			return new[] { "-of", "default=noprint_wrappers=1" };
		}

		/// <summary>
		/// Sets the input file path.
		/// </summary>
		public static IReadOnlyList<string> SetInput(string inputPath)
		{
			return new[] { "-i", inputPath };
		}

		/// <summary>
		/// Equivalent of Python's shutil.which() - finds an executable in PATH.
		/// Returns null if not found, matching Python's behavior.
		/// Deliberate reproduction of Python's behavior where None can appear in command lists.
		/// </summary>
		private static string? Which(string executable)
		{
			var pathEnvVar = Environment.GetEnvironmentVariable("PATH") ?? "";
			var pathDirs = pathEnvVar.Split(Path.PathSeparator);

			foreach (var dir in pathDirs)
			{
				var fullPath = Path.Combine(dir, executable);
				if (File.Exists(fullPath))
				{
					return fullPath;
				}
			}

			return null;
		}
	}
}
