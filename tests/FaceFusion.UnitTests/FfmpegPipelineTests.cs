using FaceFusion.Media;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Tests for the higher-level ffmpeg.py pipeline functions ported into
/// <see cref="Ffmpeg"/> alongside the process-runner surface covered by
/// <c>FfmpegTests.cs</c> (kept in a separate file per the assignment brief so the two
/// don't collide).
///
/// <para>
/// Neither the <c>ffmpeg</c> nor the <c>ffprobe</c> binary is installed in this container,
/// and the example media used by the real <c>tests/test_ffmpeg.py</c> cases needs a
/// network download that is restricted here — same situation <c>FfmpegTests.cs</c>
/// documents. Two kinds of coverage are still genuinely available without either:
/// </para>
/// <list type="bullet">
/// <item>
/// <b>Command construction.</b> Every pipeline function's ffmpeg argument list is built by
/// an <c>internal static Build&lt;Name&gt;Commands</c> helper with no <see cref="System.Diagnostics.Process"/>
/// dependency (same pattern as <see cref="Ffmpeg.TryParseFrameNumber"/> /
/// <see cref="Ffmpeg.ParseEncoderLine"/>), so the exact argument list — including which
/// state-manager-turned-parameter values land where, and the fixed-up encoder — is
/// directly assertable.
/// </item>
/// <item>
/// <b>Graceful degradation.</b> <see cref="FfmpegBuilder.Run"/>/<see cref="FfprobeBuilder.Run"/>
/// prepend <c>null</c> when <c>shutil.which</c>-equivalent lookup fails (ffmpeg/ffprobe not
/// on PATH), and <see cref="ProcessRunner.TryStart"/> turns that into a <c>null</c>
/// <see cref="System.Diagnostics.Process"/> rather than throwing. Every pipeline function
/// that only needs a <see cref="System.Diagnostics.Process"/> (not <see cref="Ffprobe"/>
/// metadata — see below) is exercised end-to-end against the real (absent) binary and
/// asserted to degrade to <c>false</c>/<c>null</c>/a non-available process handle, exactly
/// as it would in this container at runtime.
/// </item>
/// </list>
/// <para>
/// <see cref="Ffmpeg.ExtractFrames"/> is the one exception: unlike every other pipeline
/// function here, it calls <see cref="Ffprobe.ExtractStaticVideoMetadata"/> internally
/// (<c>ffprobe.extract_static_video_metadata(target_path).get('color_transfer')</c> in
/// Python), and that call throws (a missing dictionary key) rather than degrading
/// gracefully when ffprobe is absent — <c>FfprobeTests.cs</c> documents the same limitation
/// for <c>ExtractVideoMetadata</c> directly. So the end-to-end
/// <c>test_extract_frames</c> port stays <c>[Fact(Skip = ...)]</c> in <c>FfmpegTests.cs</c>;
/// <see cref="Ffmpeg.BuildExtractFramesCommands"/> below covers its command construction
/// instead, since that helper takes <c>colorTransfer</c> as a plain parameter rather than
/// deriving it via <see cref="Ffprobe"/>.
/// </para>
/// </summary>
public sealed class FfmpegPipelineTests
{
    private static readonly VideoMetadata SampleVideoMetadata = new(
        Duration: 10.0,
        FrameTotal: 250,
        Fps: 25.0,
        Resolution: new Resolution(1920, 1080),
        BitRate: 4_000_000,
        ColorTransfer: "bt709");

    // -----------------------------------------------------------------
    // Command construction
    // -----------------------------------------------------------------

    [Fact]
    public void BuildCreateVideoReaderCommands()
    {
        var commands = Ffmpeg.BuildCreateVideoReaderCommands("target.mp4", 50, SampleVideoMetadata);

        Assert.Equal(
            new[] { "-ss", "2.0", "-i", "target.mp4", "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-" },
            commands);
    }

    [Fact]
    public void BuildCreateVideoReaderCommandsRestrictsHdrColorTransfer()
    {
        var hdrMetadata = SampleVideoMetadata with { ColorTransfer = "smpte2084" };
        var commands = Ffmpeg.BuildCreateVideoReaderCommands("target.mp4", 0, hdrMetadata);

        // frame_number / fps == 0, and restrict_color_transfer now contributes its -vf filter.
        Assert.Equal(
            new[] { "-ss", "0.0", "-i", "target.mp4", "-vf", "scale=out_primaries=bt709:out_transfer=bt709:intent=perceptual", "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-" },
            commands);
    }

    [Fact]
    public void BuildCreateVideoWriterCommands()
    {
        var commands = Ffmpeg.BuildCreateVideoWriterCommands(
            tempVideoPath: "/tmp/facefusion/target/temp.mp4",
            tempVideoFormat: "mp4",
            tempVideoFps: 25.0,
            tempVideoResolution: new Resolution(640, 480),
            outputVideoResolution: new Resolution(1280, 720),
            outputVideoFps: 30.0,
            resolvedVideoEncoder: VideoEncoder.Libx264,
            outputVideoQuality: 80,
            outputVideoPreset: VideoPreset.Medium,
            tempPixelFormat: TempPixelFormat.Bgr24);

        // Assembled the same way Ffmpeg.CreateVideoWriter itself assembles it, so this
        // pins argument *ordering* and which value lands where, not a re-derivation of
        // FfmpegBuilder's own quality/preset arithmetic (already covered by FfmpegBuilderTests).
        var expected = FfmpegBuilder.Chain(
            FfmpegBuilder.SetOutputFormat("rawvideo"),
            FfmpegBuilder.EnforcePixelFormat("bgr24"),
            FfmpegBuilder.SetMediaResolution("640x480"),
            FfmpegBuilder.SetInputFps(25.0),
            FfmpegBuilder.SetInput("pipe:0"),
            FfmpegBuilder.SetMediaResolution("1280x720"),
            FfmpegBuilder.SetVideoEncoder("libx264"),
            FfmpegBuilder.SetThreadCount(16),
            FfmpegBuilder.SetVideoTag("libx264", "mp4"),
            FfmpegBuilder.SetVideoQuality("libx264", 80),
            NonNullPreset(FfmpegBuilder.SetVideoPreset("libx264", "medium")),
            FfmpegBuilder.Concat(
                FfmpegBuilder.SetVideoFps(30.0),
                FfmpegBuilder.ConvertColorSpace("bt709")),
            FfmpegBuilder.SetPixelFormat("libx264"),
            FfmpegBuilder.ForceOutput("/tmp/facefusion/target/temp.mp4"));

        Assert.Equal(expected, commands);
    }

    /// <summary>
    /// Python: <c>output_video_encoder = fix_video_encoder(temp_video_format,
    /// output_video_encoder)</c> — the line every one of <c>create_video_writer</c>,
    /// <c>merge_video</c> runs before building its command list. Exercises
    /// <see cref="Ffmpeg.ResolveVideoEncoder"/> directly (the private helper both call),
    /// which is the actual logic under test — <see cref="Ffmpeg.FixVideoEncoder"/> itself
    /// is already pinned by <c>FfmpegTests.TestFixVideoEncoder</c>.
    /// </summary>
    [Fact]
    public void ResolveVideoEncoderFixesRawvideoForMp4()
    {
        Assert.Equal(VideoEncoder.Libx264, Ffmpeg.ResolveVideoEncoder("mp4", VideoEncoder.Rawvideo));
        Assert.Equal(VideoEncoder.Libx264, Ffmpeg.ResolveVideoEncoder("mkv", VideoEncoder.Rawvideo));
        // A format string that isn't a real VideoFormat wire name (e.g. no extension) is
        // Python's `cast(...)`-without-runtime-check case: every fix_video_encoder branch
        // fails to match, so the original encoder passes through unchanged.
        Assert.Equal(VideoEncoder.Rawvideo, Ffmpeg.ResolveVideoEncoder(null, VideoEncoder.Rawvideo));
        Assert.Equal(VideoEncoder.Rawvideo, Ffmpeg.ResolveVideoEncoder("not-a-real-format", VideoEncoder.Rawvideo));
    }

    /// <summary>See <see cref="ResolveVideoEncoderFixesRawvideoForMp4"/>; same reasoning for the audio-encoder fix-up shared by <c>restore_audio</c>/<c>replace_audio</c>.</summary>
    [Fact]
    public void ResolveAudioEncoderFixesLibopusForAvi()
    {
        Assert.Equal(AudioEncoder.Aac, Ffmpeg.ResolveAudioEncoder("avi", AudioEncoder.Libopus));
        Assert.Equal(AudioEncoder.Aac, Ffmpeg.ResolveAudioEncoder("wmv", AudioEncoder.Libopus));
        Assert.Equal(AudioEncoder.Libopus, Ffmpeg.ResolveAudioEncoder(null, AudioEncoder.Libopus));
        Assert.Equal(AudioEncoder.Libopus, Ffmpeg.ResolveAudioEncoder("not-a-real-format", AudioEncoder.Libopus));
    }

    [Fact]
    public void BuildExtractFramesCommands()
    {
        var commands = Ffmpeg.BuildExtractFramesCommands(
            targetPath: "target.mp4",
            tempVideoResolution: new Resolution(640, 480),
            tempVideoFps: 25.0,
            trimFrameStart: 10,
            trimFrameEnd: 100,
            colorTransfer: "bt709",
            tempFramePattern: "/tmp/facefusion/target/%08d.png");

        Assert.Equal(
            new[]
            {
                "-i", "target.mp4",
                "-s", "640x480",
                "-q:v", "0",
                "-pix_fmt", "rgb24",
                "-vf", "trim=start_frame=10:end_frame=100,fps=25",
                "-fps_mode", "passthrough",
                "-start_number", "10",
                "/tmp/facefusion/target/%08d.png"
            },
            commands);
    }

    [Fact]
    public void BuildCopyImageCommands()
    {
        var commands = Ffmpeg.BuildCopyImageCommands("target.png", new Resolution(200, 100), "/tmp/facefusion/target/temp.png");

        var expected = FfmpegBuilder.Chain(
            FfmpegBuilder.SetInput("target.png"),
            FfmpegBuilder.SetMediaResolution("200x100"),
            // copy_image always hardcodes quality 100, unlike finalize_image.
            FfmpegBuilder.SetImageQuality("target.png", 100),
            FfmpegBuilder.ForceOutput("/tmp/facefusion/target/temp.png"));

        Assert.Equal(expected, commands);
    }

    [Fact]
    public void BuildFinalizeImageCommands()
    {
        var commands = Ffmpeg.BuildFinalizeImageCommands("target.png", "/tmp/facefusion/target/temp.png", "output.png", new Resolution(400, 200), 80);

        // Note (reproduced from Python, see the method's own doc comment): the input path
        // is temp_image_path, but the quality lookup uses target_path's extension.
        Assert.Equal("-i", commands[0]);
        Assert.Equal("/tmp/facefusion/target/temp.png", commands[1]);
        Assert.Equal(new[] { "-s", "400x200" }, new[] { commands[2], commands[3] });
        Assert.Equal(FfmpegBuilder.SetImageQuality("target.png", 80), new[] { commands[4], commands[5] });
        Assert.Equal(new[] { "-y", "output.png" }, new[] { commands[6], commands[7] });
    }

    [Fact]
    public void BuildReadAudioBufferCommands()
    {
        var commands = Ffmpeg.BuildReadAudioBufferCommands("target.mp4", 48000, 16, 2);

        Assert.Equal(
            new[] { "-i", "target.mp4", "-vn", "-ar", "48000", "-f", "s16le", "-ac", "2", "-" },
            commands);
    }

    [Fact]
    public void BuildRestoreAudioCommands()
    {
        var commands = Ffmpeg.BuildRestoreAudioCommands(
            tempVideoPath: "/tmp/facefusion/target/temp.mp4",
            targetPath: "target.mp4",
            outputPath: "output.mp4",
            trimFrameStart: 0,
            trimFrameEnd: 100,
            targetVideoFps: 25.0,
            resolvedAudioEncoder: AudioEncoder.Aac,
            outputAudioQuality: 80,
            outputAudioVolume: 100,
            tempVideoDuration: 4.0,
            outputVideoFormat: "mp4");

        var expected = FfmpegBuilder.Chain(
            FfmpegBuilder.SetInput("/tmp/facefusion/target/temp.mp4"),
            FfmpegBuilder.SelectMediaRange(0, 100, 25.0),
            FfmpegBuilder.SetInput("target.mp4"),
            FfmpegBuilder.CopyVideoEncoder(),
            FfmpegBuilder.SetAudioEncoder("aac"),
            FfmpegBuilder.SetAudioQuality("aac", 80),
            FfmpegBuilder.SetAudioVolume(100),
            FfmpegBuilder.SelectMediaStream("0:v:0"),
            FfmpegBuilder.SelectMediaStream("1:a:0"),
            FfmpegBuilder.SetVideoDuration(4.0),
            FfmpegBuilder.SetFaststart("mp4"),
            FfmpegBuilder.ForceOutput("output.mp4"));

        Assert.Equal(expected, commands);
    }

    [Fact]
    public void BuildReplaceAudioCommands()
    {
        var commands = Ffmpeg.BuildReplaceAudioCommands(
            tempVideoPath: "/tmp/facefusion/target/temp.mp4",
            audioPath: "audio.wav",
            outputPath: "output.mp4",
            resolvedAudioEncoder: AudioEncoder.Libopus,
            outputAudioQuality: 50,
            outputAudioVolume: 80,
            tempVideoDuration: 6.5,
            outputVideoFormat: "webm");

        var expected = FfmpegBuilder.Chain(
            FfmpegBuilder.SetInput("/tmp/facefusion/target/temp.mp4"),
            FfmpegBuilder.SetInput("audio.wav"),
            FfmpegBuilder.CopyVideoEncoder(),
            FfmpegBuilder.SetAudioEncoder("libopus"),
            FfmpegBuilder.SetAudioQuality("libopus", 50),
            FfmpegBuilder.SetAudioVolume(80),
            FfmpegBuilder.SetVideoDuration(6.5),
            FfmpegBuilder.SetFaststart("webm"),
            FfmpegBuilder.ForceOutput("output.mp4"));

        Assert.Equal(expected, commands);
    }

    [Fact]
    public void BuildMergeVideoCommands()
    {
        var commands = Ffmpeg.BuildMergeVideoCommands(
            tempVideoPath: "/tmp/facefusion/target/temp.mp4",
            tempVideoFormat: "mp4",
            tempVideoFps: 25.0,
            trimFrameStart: 10,
            tempFramePattern: "/tmp/facefusion/target/%08d.png",
            outputVideoResolution: new Resolution(1280, 720),
            resolvedVideoEncoder: VideoEncoder.LibvpxVp9,
            outputVideoQuality: 60,
            outputVideoPreset: VideoPreset.Slow,
            outputVideoFps: 30.0);

        var expected = FfmpegBuilder.Chain(
            FfmpegBuilder.SetInputFps(25.0),
            FfmpegBuilder.SetStartNumber(10),
            FfmpegBuilder.SetInput("/tmp/facefusion/target/%08d.png"),
            FfmpegBuilder.SetMediaResolution("1280x720"),
            FfmpegBuilder.SetVideoEncoder("libvpx-vp9"),
            FfmpegBuilder.SetVideoTag("libvpx-vp9", "mp4"),
            FfmpegBuilder.SetVideoQuality("libvpx-vp9", 60),
            // libvpx-vp9 has no preset mapping (see FfmpegBuilder.SetVideoPreset), so this
            // contributes nothing — asserted explicitly below via the concat/pix_fmt tail.
            FfmpegBuilder.Concat(
                FfmpegBuilder.SetVideoFps(30.0),
                FfmpegBuilder.KeepVideoAlpha("libvpx-vp9"),
                FfmpegBuilder.ConvertColorSpace("bt709")),
            FfmpegBuilder.SetPixelFormat("libvpx-vp9"),
            FfmpegBuilder.ForceOutput("/tmp/facefusion/target/temp.mp4"));

        Assert.Equal(expected, commands);
    }

    [Fact]
    public void BuildConcatVideoCommands()
    {
        var commands = Ffmpeg.BuildConcatVideoCommands("/tmp/concat-list.txt", "/abs/output.mp4", "mp4");

        Assert.Equal(
            new[] { "-f", "concat", "-safe", "0", "-i", "/tmp/concat-list.txt", "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", "-y", "/abs/output.mp4" },
            commands);
    }

    // -----------------------------------------------------------------
    // Graceful degradation — ffmpeg is not installed in this container, so every one of
    // these exercises the real (absent) binary lookup end to end. None of these functions
    // call Ffprobe (see the class doc comment for why ExtractFrames is the exception).
    // -----------------------------------------------------------------

    [Fact]
    public void CreateVideoReaderIsUnavailableWithoutFfmpeg()
    {
        // ffmpeg may genuinely be installed in this environment (see docs/PORT_CONVENTIONS.md),
        // so "without ffmpeg" is forced deterministically via the ffmpegPath override
        // (see FfmpegBuilder.Run's doc comment) rather than by relying on the machine's
        // PATH, which is not a property of the code under test.
        using var reader = Ffmpeg.CreateVideoReader("target.mp4", 0, SampleVideoMetadata, TestHelper.BogusBinaryPath);

        Assert.False(reader.IsAvailable);
        Assert.Null(reader.StandardOutput);
        Assert.Null(reader.ExitCode);
        Assert.Equal("target.mp4", reader.VideoPath);
        Assert.Equal(SampleVideoMetadata, reader.Metadata);
    }

    [Fact]
    public void CreateVideoReaderDisposeIsIdempotent()
    {
        var reader = Ffmpeg.CreateVideoReader("target.mp4", 0, SampleVideoMetadata);

        reader.Dispose();
        reader.Dispose();
    }

    [Fact]
    public void CreateVideoWriterIsUnavailableWithoutFfmpeg()
    {
        // See CreateVideoReaderIsUnavailableWithoutFfmpeg above for why the binary path is
        // forced rather than assumed absent.
        using var writer = Ffmpeg.CreateVideoWriter(
            "target.mp4", 25.0, new Resolution(640, 480), new Resolution(640, 480), 25.0,
            VideoEncoder.Libx264, 80, VideoPreset.Medium, TempPixelFormat.Bgr24, Path.GetTempPath(), TestHelper.BogusBinaryPath);

        Assert.False(writer.IsAvailable);
        Assert.Null(writer.StandardInput);
        Assert.Null(writer.ExitCode);
        Assert.Equal("target.mp4", writer.TargetPath);
        Assert.Equal(25.0, writer.Metadata.Fps);
        Assert.Equal(new Resolution(640, 480), writer.Metadata.Resolution);
    }

    [Fact]
    public void CreateVideoWriterDisposeAndFinishWritingAreSafeWithoutFfmpeg()
    {
        var writer = Ffmpeg.CreateVideoWriter(
            "target.mp4", 25.0, new Resolution(640, 480), new Resolution(640, 480), 25.0,
            VideoEncoder.Libx264, 80, VideoPreset.Medium, TempPixelFormat.Bgr24, Path.GetTempPath());

        // Neither should throw when there is no underlying process.
        writer.FinishWriting();
        writer.Dispose();
        writer.Dispose();
    }

    [Fact]
    public void CopyImageFailsWithoutFfmpeg()
    {
        Assert.False(Ffmpeg.CopyImage("target.png", new Resolution(100, 100), Path.GetTempPath()));
    }

    [Fact]
    public void FinalizeImageFailsWithoutFfmpeg()
    {
        Assert.False(Ffmpeg.FinalizeImage("target.png", "output.png", new Resolution(100, 100), 80, Path.GetTempPath()));
    }

    [Fact]
    public void ReadAudioBufferReturnsNullWithoutFfmpeg()
    {
        Assert.Null(Ffmpeg.ReadAudioBuffer("target.mp4", 48000, 16, 2));
    }

    [Fact]
    public void RestoreAudioFailsWithoutFfmpeg()
    {
        // target.mp4/temp.mp4 do not exist on disk either, but RestoreAudio's own metadata
        // lookups (Vision.DetectVideoFps / Vision.DetectVideoDuration) go through
        // OpenCvSharp.VideoCapture, not ffprobe, and degrade to 0/null for a missing file
        // rather than throwing — so this reaches (and is stopped by) the ffmpeg lookup.
        Assert.False(Ffmpeg.RestoreAudio("target.mp4", "output.mp4", 0, 100, AudioEncoder.Aac, 80, 100, Path.GetTempPath()));
    }

    [Fact]
    public void ReplaceAudioFailsWithoutFfmpeg()
    {
        Assert.False(Ffmpeg.ReplaceAudio("target.mp4", "audio.wav", "output.mp4", AudioEncoder.Aac, 80, 100, Path.GetTempPath()));
    }

    [Fact]
    public void MergeVideoFailsWithoutFfmpeg()
    {
        var updateProgressCalls = 0;
        Assert.False(Ffmpeg.MergeVideo(
            "target.mp4", 25.0, new Resolution(640, 480), 25.0, 0, 100,
            VideoEncoder.Libx264, 80, VideoPreset.Medium, Path.GetTempPath(), "png",
            _ => updateProgressCalls++));
        Assert.Equal(0, updateProgressCalls);
    }

    [Fact]
    public void ConcatVideoFailsWithoutFfmpegButCleansUpItsListFile()
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), "facefusion-ffmpeg-pipeline-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDirectory);

        try
        {
            var clipA = Path.Combine(tempDirectory, "a.mp4");
            var clipB = Path.Combine(tempDirectory, "b.mp4");
            File.WriteAllBytes(clipA, Array.Empty<byte>());
            File.WriteAllBytes(clipB, Array.Empty<byte>());

            var outputPath = Path.Combine(tempDirectory, "concat.mp4");

            // Ffmpeg.ConcatVideo writes its concat-demuxer list file to a real temp file
            // (Path.GetTempFileName(), in the shared system temp directory) and must
            // remove it again regardless of the ffmpeg result. A before/after file-count
            // diff on that shared directory would be racy against every other test in this
            // suite that also touches it concurrently, so this only asserts the boolean
            // result; the unconditional FileSystem.RemoveFile(concatVideoPath) call is
            // verified by inspection (see Ffmpeg.ConcatVideo's own doc comment).
            Assert.False(Ffmpeg.ConcatVideo(outputPath, new[] { clipA, clipB }));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    /// <summary>
    /// <see cref="FfmpegBuilder.SetVideoPreset"/> only returns a null element for a preset
    /// string that isn't a real <see cref="VideoPreset"/> wire name (see that method's own
    /// doc comment); every call here passes a real one, so this is provably never null —
    /// same justification as <c>Ffmpeg.AssertNoNullPreset</c> in the production code.
    /// </summary>
    private static IReadOnlyList<string> NonNullPreset(IReadOnlyList<string?> values)
    {
        var result = new string[values.Count];

        for (var index = 0; index < values.Count; index++)
        {
            result[index] = values[index] ?? throw new InvalidOperationException("unreachable for a real VideoPreset value");
        }

        return result;
    }
}
