using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace FaceFusion.Media
{
	/// <summary>
	/// Builds curl command-line arguments.
	/// Ported from facefusion/curl_builder.py
	/// </summary>
	public static class CurlBuilder
	{
		// Metadata values mirroring facefusion/metadata.py
		// These are hardcoded since we cannot depend on Python's metadata module
		private const string MetadataName = "FaceFusion";
		private const string MetadataVersion = "3.8.2";

		/// <summary>
		/// Prepends curl executable path, user agent, and common flags to commands.
		/// Note: Returns a list that may contain null (matching Python's shutil.which behavior).
		/// </summary>
		public static IReadOnlyList<string?> Run(IReadOnlyList<string> commands)
		{
			// TODO(types): Command type alias should be resolved when FaceFusion.Types is available
			var userAgent = MetadataName + "/" + MetadataVersion;
			var result = new List<string?>();
			result.Add(Which("curl"));
			result.Add("--user-agent");
			result.Add(userAgent);
			result.Add("--location");
			result.Add("--silent");
			result.Add("--ssl-no-revoke");
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
		/// Pings a URL (HEAD request).
		/// </summary>
		public static IReadOnlyList<string> Ping(string url)
		{
			return new[] { "-I", url };
		}

		/// <summary>
		/// Downloads a file from a URL with resume capability.
		/// </summary>
		public static IReadOnlyList<string> Download(string url, string downloadFilePath)
		{
			return new[] { "--create-dirs", "--continue-at", "-", "--output", downloadFilePath, url };
		}

		/// <summary>
		/// Sets the connection timeout.
		/// </summary>
		public static IReadOnlyList<string> SetTimeout(int timeout)
		{
			return new[] { "--connect-timeout", timeout.ToString(CultureInfo.InvariantCulture) };
		}

		/// <summary>
		/// Sets the number of retries.
		/// </summary>
		public static IReadOnlyList<string> SetRetry(int retry)
		{
			return new[] { "--retry", retry.ToString(CultureInfo.InvariantCulture) };
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
