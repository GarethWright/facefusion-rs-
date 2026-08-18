using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of tests/test_ffprobe.py.
///
/// <para>
/// Both preconditions the earlier port report cited against this file — no ffprobe
/// binary, no example media — no longer hold (see docs/PORT_CONVENTIONS.md and
/// <c>tools/parity/fetch_examples.sh</c>), so <see cref="TestExtractAudioMetadata"/> and
/// <see cref="TestExtractVideoMetadata"/> now run for real, gated on
/// <see cref="MediaFactAttribute"/> so the suite still degrades to a clear runtime skip in
/// an environment that genuinely lacks either. <see cref="MediaFixtures.Ensure"/> (called
/// from the constructor below) generates <c>source-48000khz-2ch.wav</c> and
/// <c>target-240p-1s.{mkv,mov}</c>, the same derived fixtures Python's module-scoped
/// <c>before_all</c> produces for this file.
/// </para>
///
/// The parsing logic (<see cref="Ffprobe.ParseEntries"/>, <see cref="Ffprobe.ExtractVideoFps"/>)
/// stays covered directly against canned ffprobe output below too — it has no standalone
/// Python test (only exercised indirectly through real ffprobe output) and needs no binary
/// at all.
/// </summary>
public sealed class FfprobeTests
{
	public FfprobeTests()
	{
		MediaFixtures.Ensure();
	}

	[MediaFact]
	public void TestExtractAudioMetadata()
	{
		var audioMetadata = Ffprobe.ExtractAudioMetadata(TestHelper.GetTestExampleFile("source.mp3"));

		Assert.Equal(44100, audioMetadata.SampleRate);
		Assert.Equal(1, audioMetadata.ChannelTotal);
		Assert.Equal(167040, audioMetadata.FrameTotal);
		Assert.Equal(128000, audioMetadata.BitRate);

		audioMetadata = Ffprobe.ExtractAudioMetadata(TestHelper.GetTestExampleFile("source-48000khz-2ch.wav"));

		Assert.Equal(48000, audioMetadata.SampleRate);
		Assert.Equal(2, audioMetadata.ChannelTotal);
		Assert.Equal(91200, audioMetadata.FrameTotal);
		Assert.Equal(1536328, audioMetadata.BitRate);
	}

	[MediaFact]
	public void TestExtractVideoMetadata()
	{
		var videoMetadata = Ffprobe.ExtractVideoMetadata(TestHelper.GetTestExampleFile("target-240p.mp4"));

		Assert.Equal(25.0, videoMetadata.Fps);
		Assert.Equal(10.8, videoMetadata.Duration);
		Assert.Equal(new FaceFusion.Types.Resolution(426, 226), videoMetadata.Resolution);
		Assert.Equal(141981, videoMetadata.BitRate);
		Assert.Equal("smpte170m", videoMetadata.ColorTransfer);

		videoMetadata = Ffprobe.ExtractVideoMetadata(TestHelper.GetTestExampleFile("target-240p-1s.mkv"));

		Assert.Equal(25.0, videoMetadata.Fps);
		Assert.Equal(1.0, videoMetadata.Duration);
		Assert.Equal(25, videoMetadata.FrameTotal);
		Assert.Equal(new FaceFusion.Types.Resolution(426, 226), videoMetadata.Resolution);

		videoMetadata = Ffprobe.ExtractVideoMetadata(TestHelper.GetTestExampleFile("target-240p-1s.mov"));

		Assert.Equal(25.0, videoMetadata.Fps);
		Assert.Equal(1.0, videoMetadata.Duration);
		Assert.Equal(new FaceFusion.Types.Resolution(426, 226), videoMetadata.Resolution);
	}

	// --- missing-binary contract: crash is faithful Python parity, not a bug -----------
	// See Ffprobe.ExtractAudioMetadata/ExtractVideoMetadata's doc comments. Forced
	// deterministically via the ffprobePath override (see FfprobeBuilder.Run's doc
	// comment) rather than assumed from the machine, since ffprobe may genuinely be
	// installed here.

	[Fact]
	public void TestExtractAudioMetadataThrowsWhenFfprobeBinaryMissing()
	{
		// Python: format_entries.get('duration') is None when ffprobe produced no output,
		// and float(None) raises TypeError — extract_audio_metadata was never designed to
		// degrade gracefully when the binary itself is missing (as opposed to the empty
		// dict path ProbeFormatEntries/ProbeAudioEntries/ProbeVideoEntries already handle
		// gracefully, e.g. TestProbeFormatEntriesGracefulWhenBinaryNotFound below). The C#
		// port throws KeyNotFoundException from the same dictionary access instead of
		// TypeError — a different exception type (C# has no equivalent to indexing None),
		// but the same "this is not a supported configuration, it throws" contract. Per
		// port convention rule 1, this is faithful parity and is deliberately not "fixed"
		// into a graceful null/default return.
		Assert.Throws<KeyNotFoundException>(() =>
			Ffprobe.ExtractAudioMetadata(TestHelper.GetTestExampleFile("source.mp3"), TestHelper.BogusBinaryPath));
	}

	[Fact]
	public void TestExtractVideoMetadataThrowsWhenFfprobeBinaryMissing()
	{
		Assert.Throws<KeyNotFoundException>(() =>
			Ffprobe.ExtractVideoMetadata(TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.BogusBinaryPath));
	}

	// --- Ffprobe.ParseEntries -------------------------------------------------------
	// Not present as a standalone Python test (parse_entries is only exercised
	// indirectly via extract_*_metadata against real ffprobe output above), but it is
	// the single piece of this module most worth unit-testing directly per the
	// assignment brief, since it needs no binary.

	[Fact]
	public void TestParseEntriesNull()
	{
		Assert.Empty(Ffprobe.ParseEntries(null));
	}

	[Fact]
	public void TestParseEntriesEmpty()
	{
		Assert.Empty(Ffprobe.ParseEntries(string.Empty));
	}

	[Fact]
	public void TestParseEntriesSingleLine()
	{
		var entries = Ffprobe.ParseEntries("sample_rate=44100\n");

		Assert.Equal("44100", entries["sample_rate"]);
		Assert.Single(entries);
	}

	[Fact]
	public void TestParseEntriesMultipleLines()
	{
		var output = "width=426\nheight=226\nr_frame_rate=25/1\ncolor_transfer=smpte170m\n";
		var entries = Ffprobe.ParseEntries(output);

		Assert.Equal("426", entries["width"]);
		Assert.Equal("226", entries["height"]);
		Assert.Equal("25/1", entries["r_frame_rate"]);
		Assert.Equal("smpte170m", entries["color_transfer"]);
	}

	[Fact]
	public void TestParseEntriesCarriageReturnLineEndings()
	{
		var entries = Ffprobe.ParseEntries("duration=10.800000\r\nbit_rate=141981\r\n");

		Assert.Equal("10.800000", entries["duration"]);
		Assert.Equal("141981", entries["bit_rate"]);
	}

	[Fact]
	public void TestParseEntriesIgnoresLinesWithoutEquals()
	{
		var entries = Ffprobe.ParseEntries("not-a-key-value-line\nsample_rate=44100\n");

		Assert.Single(entries);
		Assert.Equal("44100", entries["sample_rate"]);
	}

	[Fact]
	public void TestParseEntriesValueContainingEquals()
	{
		// Python: line.split('=', 1) — only the first '=' splits key from value.
		var entries = Ffprobe.ParseEntries("tag:some=weird=value\n");

		Assert.Equal("weird=value", entries["tag:some"]);
	}

	// --- Ffprobe.ExtractVideoFps -----------------------------------------------------
	// Not present as a standalone Python test either (only reached indirectly through
	// extract_video_metadata against real ffprobe output), tested here directly.

	[Fact]
	public void TestExtractVideoFpsRational()
	{
		Assert.Equal(25.0, Ffprobe.ExtractVideoFps("25/1"));
		Assert.Equal(30.0, Ffprobe.ExtractVideoFps("30/1"));
	}

	[Fact]
	public void TestExtractVideoFpsNonIntegerResult()
	{
		// ffprobe's classic NTSC rational form.
		Assert.Equal(30000.0 / 1001.0, Ffprobe.ExtractVideoFps("30000/1001"));
		Assert.Equal(23.976023976023978, Ffprobe.ExtractVideoFps("24000/1001"), 12);
	}

	[Fact]
	public void TestExtractVideoFpsZeroNumeratorOrDenominator()
	{
		Assert.Equal(0.0, Ffprobe.ExtractVideoFps("0/1"));
		Assert.Equal(0.0, Ffprobe.ExtractVideoFps("25/0"));
	}

	[Fact]
	public void TestExtractVideoFpsMissingOrNonRational()
	{
		Assert.Equal(0.0, Ffprobe.ExtractVideoFps(null));
		Assert.Equal(0.0, Ffprobe.ExtractVideoFps(string.Empty));
		Assert.Equal(0.0, Ffprobe.ExtractVideoFps("25"));
	}

	// --- process-launching surface, without a binary ----------------------------------

	[Fact]
	public void TestRunFfprobeReturnsNullWhenBinaryNotFound()
	{
		// ffprobe may genuinely be installed in this environment (see
		// docs/PORT_CONVENTIONS.md), so the "not found" path is forced deterministically
		// via the ffprobePath override (see FfprobeBuilder.Run's doc comment) instead of
		// relying on the machine's PATH. This exercises the "binary not found" path for
		// real: RunFfprobe must degrade to null instead of throwing an unhandled exception
		// starting the process.
		var process = Ffprobe.RunFfprobe(FfprobeBuilder.SetInput("media.mp4"), TestHelper.BogusBinaryPath);

		Assert.Null(process);
	}

	[Fact]
	public void TestProbeFormatEntriesGracefulWhenBinaryNotFound()
	{
		// probe_format_entries -> run_ffprobe -> communicate() -> parse_entries. With no
		// ffprobe binary present this must come back as an empty dictionary rather than
		// throw, all the way through the public entry point.
		var entries = Ffprobe.ProbeFormatEntries("media.mp4", new[] { "duration", "bit_rate" }, TestHelper.BogusBinaryPath);

		Assert.Empty(entries);
	}
}
