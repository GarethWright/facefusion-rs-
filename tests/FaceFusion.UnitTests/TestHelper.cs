using FaceFusion.Core;
using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/helper.py</c> — path helpers for the shared example-media directory
/// and the per-run test-output directory.
///
/// <para>
/// <see cref="Path.GetTempPath"/> already honours <c>TMPDIR</c> on Linux/macOS the same
/// way Python's <c>tempfile.gettempdir()</c> does (both fall back to <c>/tmp</c> only when
/// the environment variable is unset), so no manual <c>TMPDIR</c> lookup is needed here —
/// this resolves to the exact same directory
/// (<c>&lt;tempdir&gt;/facefusion-test-examples</c>) the Python suite's <c>tests/helper.py</c>
/// uses, which is the point: both suites share one copy of the downloaded media.
/// </para>
/// </summary>
public static class TestHelper
{
	/// <summary>Python: <c>get_test_examples_directory</c>.</summary>
	public static string GetTestExamplesDirectory()
		=> Path.Combine(Path.GetTempPath(), "facefusion-test-examples");

	/// <summary>Python: <c>get_test_example_file</c>.</summary>
	public static string GetTestExampleFile(string fileName)
		=> Path.Combine(GetTestExamplesDirectory(), fileName);

	/// <summary>Python: <c>get_test_outputs_directory</c>.</summary>
	public static string GetTestOutputsDirectory()
		=> Path.Combine(Path.GetTempPath(), "facefusion-test-outputs");

	/// <summary>Python: <c>get_test_output_file</c>.</summary>
	public static string GetTestOutputFile(string fileName)
		=> Path.Combine(GetTestOutputsDirectory(), fileName);

	/// <summary>Python: <c>prepare_test_output_directory</c>.</summary>
	public static bool PrepareTestOutputDirectory()
	{
		var testOutputsDirectory = GetTestOutputsDirectory();
		FileSystem.RemoveDirectory(testOutputsDirectory);
		FileSystem.CreateDirectory(testOutputsDirectory);
		return FileSystem.IsDirectory(testOutputsDirectory);
	}

	/// <summary>The four example media files every media-dependent test in this suite needs.</summary>
	private static readonly string[] RequiredExampleFiles =
	{
		"source.jpg",
		"source.mp3",
		"target-240p.mp4",
		"target-1080p.mp4"
	};

	/// <summary>
	/// True when every file <c>tools/parity/fetch_examples.sh</c> downloads is present
	/// (and non-empty) in the shared examples directory.
	/// </summary>
	public static bool ExamplesAvailable
		=> RequiredExampleFiles.All(fileName =>
		{
			var path = GetTestExampleFile(fileName);
			return File.Exists(path) && new FileInfo(path).Length > 0;
		});

	/// <summary>True when the <c>ffmpeg</c> binary can be located on PATH.</summary>
	public static bool HasFfmpeg => ProcessHelper.Which("ffmpeg") is not null;

	/// <summary>True when the <c>ffprobe</c> binary can be located on PATH.</summary>
	public static bool HasFfprobe => ProcessHelper.Which("ffprobe") is not null;

	/// <summary>
	/// Message shown for a runtime skip when the example media and/or ffmpeg/ffprobe are
	/// not available in this environment. Points the developer at the fetch script rather
	/// than surfacing a confusing file-not-found deep inside a test body.
	/// </summary>
	public const string MissingMediaMessage =
		"requires ffmpeg/ffprobe on PATH and the example media in " +
		"$TMPDIR/facefusion-test-examples (or /tmp if TMPDIR is unset) — " +
		"run tools/parity/fetch_examples.sh, and install ffmpeg/ffprobe, then retry";

	/// <summary>
	/// A bogus absolute path that can never resolve to a real executable, for tests that
	/// need to force the "binary not found" branch of <see cref="Ffmpeg"/>/<see cref="Ffprobe"/>
	/// deterministically (see <see cref="FfmpegBuilder.Run"/>'s <c>executablePath</c> doc
	/// comment) rather than relying on ffmpeg/ffprobe being absent from the machine, which
	/// is not a property of the code under test.
	/// </summary>
	public static string BogusBinaryPath
		=> Path.Combine(Path.GetTempPath(), "facefusion-tests-nonexistent-binary-" + Guid.NewGuid().ToString("N"));

	private static bool? _supportsHdrColorTransfer;

	/// <summary>
	/// True when the installed ffmpeg's `scale` filter accepts `out_transfer`, which
	/// FaceFusion's <c>restrict_color_transfer</c> depends on. Added in ffmpeg 7; Ubuntu
	/// 24.04 ships 6.1.1, where the Python suite fails the same cases.
	/// </summary>
	public static bool SupportsHdrColorTransfer()
	{
		if (_supportsHdrColorTransfer.HasValue)
		{
			return _supportsHdrColorTransfer.Value;
		}

		var ffmpegPath = FaceFusion.Core.ProcessHelper.Which("ffmpeg");

		if (ffmpegPath is null)
		{
			_supportsHdrColorTransfer = false;
			return false;
		}

		try
		{
			using var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
			{
				FileName = ffmpegPath,
				ArgumentList = { "-hide_banner", "-h", "filter=scale" },
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				UseShellExecute = false
			});

			if (process is null)
			{
				_supportsHdrColorTransfer = false;
				return false;
			}

			var help = process.StandardOutput.ReadToEnd();
			process.WaitForExit();
			_supportsHdrColorTransfer = help.Contains("out_transfer", StringComparison.Ordinal);
		}
		catch (System.ComponentModel.Win32Exception)
		{
			_supportsHdrColorTransfer = false;
		}

		return _supportsHdrColorTransfer.Value;
	}
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time (documented conditional-fact attribute, per
/// port convention rule 2 — xunit 2.4.2 has no <c>Assert.Skip</c>, which only arrived in
/// xunit v3) with <see cref="TestHelper.MissingMediaMessage"/> when either ffmpeg/ffprobe
/// are not on PATH or the example media have not been fetched, instead of failing with a
/// confusing file-not-found partway through the test body.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class MediaFactAttribute : FactAttribute
{
	public MediaFactAttribute()
	{
		if (!TestHelper.HasFfmpeg || !TestHelper.HasFfprobe || !TestHelper.ExamplesAvailable)
		{
			Skip = TestHelper.MissingMediaMessage;
		}
	}
}
