using FaceFusion.Media;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of tests/test_ffprobe.py.
///
/// The Python suite downloads example media over the network and then asserts on
/// ffprobe's real output (`test_extract_audio_metadata`, `test_extract_video_metadata`).
/// Neither ffprobe nor the example media are available in this container (network egress
/// restricted, binary not installed), so those two are ported but skipped per port
/// convention rule 2. In their place, the parsing logic they exercise
/// (<see cref="Ffprobe.ParseEntries"/>, <see cref="Ffprobe.ExtractVideoFps"/>) is tested
/// directly against canned ffprobe output, which needs no binary at all — this is the
/// part of the module most worth testing per the assignment brief, and it now has
/// coverage the Python suite (which only exercises it indirectly through real output)
/// does not.
/// </summary>
public sealed class FfprobeTests
{
	[Fact(Skip = "requires example media (network download restricted in this container)")]
	public void TestExtractAudioMetadata()
	{
		// Python: test_extract_audio_metadata — asserts sample_rate/channel_total/
		// frame_total/bit_rate for source.mp3 and source-48000khz-2ch.wav.
	}

	[Fact(Skip = "requires example media (network download restricted in this container)")]
	public void TestExtractVideoMetadata()
	{
		// Python: test_extract_video_metadata — asserts fps/duration/resolution/bit_rate/
		// color_transfer for target-240p.mp4, target-240p-1s.mkv, target-240p-1s.mov.
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
		// ffprobe is not installed in this container, so FfprobeBuilder.Run resolves the
		// executable path to null (Which() returning null, mirroring shutil.which). This
		// exercises the "binary not found" path for real: RunFfprobe must degrade to null
		// instead of throwing an unhandled exception starting the process.
		var process = Ffprobe.RunFfprobe(FfprobeBuilder.SetInput("media.mp4"));

		Assert.Null(process);
	}

	[Fact]
	public void TestProbeFormatEntriesGracefulWhenBinaryNotFound()
	{
		// probe_format_entries -> run_ffprobe -> communicate() -> parse_entries. With no
		// ffprobe binary present this must come back as an empty dictionary rather than
		// throw, all the way through the public entry point.
		var entries = Ffprobe.ProbeFormatEntries("media.mp4", new[] { "duration", "bit_rate" });

		Assert.Empty(entries);
	}
}
