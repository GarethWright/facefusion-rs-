using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the module-scoped <c>before_all</c> pytest fixtures in <c>tests/test_ffmpeg.py</c>
/// and <c>tests/test_ffprobe.py</c> — both derive a handful of fixture media files from the
/// four downloaded examples (different fps, sample rates, containers, one HDR-tagged clip)
/// so the real test bodies have concrete files with known properties to assert against.
///
/// <para>
/// Deviation from Python (documented, deliberate): pytest's <c>before_all</c> always
/// re-runs every ffmpeg invocation unconditionally, because the Python test container is
/// assumed fresh per run. This suite's example directory is a persistent, shared
/// <c>$TMPDIR</c> location (deliberately, so both suites reuse one download — see
/// <see cref="TestHelper"/>), so re-encoding every derived file on every <c>dotnet test</c>
/// run would be slow and would also fail outright: <c>ffmpeg_builder.set_output</c> (unlike
/// <c>force_output</c>) never passes <c>-y</c>, and non-interactive ffmpeg refuses to
/// overwrite an existing file. This port instead skips generating a derived file that
/// already exists and is non-empty — idempotent, same end state, no behavioural difference
/// to the code under test (this is test infrastructure, not production code).
/// </para>
/// </summary>
internal static class MediaFixtures
{
	private static readonly object Lock = new();
	private static bool _ensured;

	/// <summary>
	/// Generates every derived fixture file used by <see cref="FfmpegTests"/> and
	/// <see cref="FfprobeTests"/>, if not already present. No-op (and safe to call
	/// unconditionally) when ffmpeg is missing or the base example media have not been
	/// fetched — callers gate their own tests on <see cref="MediaFactAttribute"/> already,
	/// this is just cheap enough to call from every test's setup without re-checking.
	/// </summary>
	public static void Ensure()
	{
		if (_ensured)
		{
			return;
		}

		lock (Lock)
		{
			if (_ensured)
			{
				return;
			}

			if (TestHelper.HasFfmpeg && TestHelper.ExamplesAvailable)
			{
				GenerateAll();
			}

			_ensured = true;
		}
	}

	private static void GenerateAll()
	{
		var sourceMp3 = TestHelper.GetTestExampleFile("source.mp3");
		var target240p = TestHelper.GetTestExampleFile("target-240p.mp4");

		// source.wav
		RunIfMissing("source.wav", () => FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(sourceMp3),
			FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile("source.wav"))));

		// target-240p-{25,30,60}fps.mp4
		foreach (var videoFps in new[] { 25, 30, 60 })
		{
			var fileName = "target-240p-" + videoFps.ToString(System.Globalization.CultureInfo.InvariantCulture) + "fps.mp4";
			RunIfMissing(fileName, () => FfmpegBuilder.Chain(
				FfmpegBuilder.SetInput(target240p),
				FfmpegBuilder.SetVideoFps(videoFps),
				FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile(fileName))));
		}

		// target-240p-smpte2084.mp4
		//
		// Python's before_all uses the literal filter `scale=out_transfer=smpte2084` — but
		// that (and every other `scale` out_primaries/out_transfer/intent option
		// FfmpegBuilder.RestrictColorTransfer also relies on) requires a newer ffmpeg than
		// the 6.1.1 build installed here supports: confirmed directly (`ffmpeg -vf
		// scale=out_transfer=smpte2084 ...` on this build fails with "Option not found").
		// `zscale=transfer=smpte2084` tags the same color_transfer metadata via a filter
		// this build does support, so the fixture itself can still be produced reliably;
		// see MediaFactRequiringScaleColorPrimariesAttribute in TestHelper.cs for where the
		// *use* of that fixture (FfmpegTests.TestExtractFramesHdrColorTransferRestriction)
		// is skipped instead of "fixed", since RestrictColorTransfer's command is faithful
		// Python parity that this ffmpeg build genuinely cannot run — not a bug to patch.
		RunIfMissing("target-240p-smpte2084.mp4", () => FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(target240p),
			new[] { "-vf", "zscale=transfer=smpte2084" },
			FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile("target-240p-smpte2084.mp4"))));

		// target-240p-16khz.{avi,m4v,mkv,mov,mp4,webm,wmv}
		foreach (var outputVideoFormat in new[] { "avi", "m4v", "mkv", "mov", "mp4", "webm", "wmv" })
		{
			var fileName = "target-240p-16khz." + outputVideoFormat;
			RunIfMissing(fileName, () => FfmpegBuilder.Chain(
				FfmpegBuilder.SetInput(sourceMp3),
				FfmpegBuilder.SetInput(target240p),
				FfmpegBuilder.SetAudioSampleRate(16000),
				FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile(fileName))));
		}

		// target-240p-48khz.mp4
		RunIfMissing("target-240p-48khz.mp4", () => FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(sourceMp3),
			FfmpegBuilder.SetInput(target240p),
			FfmpegBuilder.SetAudioSampleRate(48000),
			FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile("target-240p-48khz.mp4"))));

		// source-48000khz-2ch.wav (test_ffprobe.py before_all)
		RunIfMissing("source-48000khz-2ch.wav", () => FfmpegBuilder.Chain(
			FfmpegBuilder.SetInput(sourceMp3),
			FfmpegBuilder.SetVideoDuration(1.9),
			FfmpegBuilder.SetAudioSampleRate(48000),
			FfmpegBuilder.SetAudioChannelTotal(2),
			FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile("source-48000khz-2ch.wav"))));

		// target-240p-1s.{mkv,mov} (test_ffprobe.py before_all)
		foreach (var videoFormat in new[] { "mkv", "mov" })
		{
			var fileName = "target-240p-1s." + videoFormat;
			RunIfMissing(fileName, () => FfmpegBuilder.Chain(
				FfmpegBuilder.SetInput(target240p),
				FfmpegBuilder.SetVideoDuration(1),
				FfmpegBuilder.SetOutput(TestHelper.GetTestExampleFile(fileName))));
		}
	}

	private static void RunIfMissing(string exampleFileName, Func<IReadOnlyList<string>> buildCommands)
	{
		var outputPath = TestHelper.GetTestExampleFile(exampleFileName);

		if (File.Exists(outputPath) && new FileInfo(outputPath).Length > 0)
		{
			return;
		}

		using var process = Ffmpeg.RunFfmpeg(buildCommands());
		process?.WaitForExit();
	}
}
