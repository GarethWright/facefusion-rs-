using FaceFusion.Media;
using FaceFusion.Parity;
using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Processors.LipSyncer</c> against the real Python
/// <c>facefusion.processors.modules.lip_syncer.core</c>, run against the real
/// <c>edtalk_256</c> and <c>wav2lip_gan_96</c> ONNX models and the largest detected face on the
/// real <c>source.jpg</c> example image, lip-synced to a real slice of <c>source.mp3</c>'s
/// extracted voice track (frame 10 of <c>get_voice_frame</c> at 25 fps — source == target ==
/// reference frame, per the dumper's docstring). Ground truth was captured with
/// <c>tools/parity/dump_lip_syncer.py</c>; see that script's docstring and
/// docs/PARITY_HARNESS.md for why these two families and not all three.
///
/// <para>
/// <b>Mel-spectrogram vs. model-input split (per the assignment).</b>
/// <see cref="TestSourceVoiceFrameMatchesTheAudioLayer"/> checks the *mel-spectrogram itself* —
/// <c>FaceFusion.Media.Audio.GetVoiceFrame</c>'s output — against Python's
/// <c>audio.get_voice_frame()</c>, entirely independent of anything in this file's own
/// preprocessing. It matches to <c>rtol = 1e-4, atol = 1e-4</c>, the same tolerance
/// <c>AudioParityTests</c> already established for <c>read_voice</c>'s output (a float32
/// <c>batch_extract_voice</c> precision divergence upstream of <c>lfilter</c>, not anything new
/// introduced here — see <c>AudioParityTests</c>' remarks and <c>Audio.cs</c>'s class remarks).
/// <see cref="TestPrepareAudioFrameMatchesPythonExactlyForEdtalk"/>/<c>ForWav2Lip</c> then take
/// the *fixture's own* <c>source_voice_frame</c> (not the audio layer's live output) through
/// <see cref="LipSyncer.PrepareAudioFrame"/> and compare at <c>rtol = atol = 0</c> — this
/// isolates lip_syncer's own preprocessing from the audio layer's precision, per the
/// assignment's "split those two, that's the point" instruction. Both stages independently
/// matched Python: the mel-spectrogram to audio-layer tolerance, the model-input tensor
/// exactly.
/// </para>
///
/// <para>
/// <b>Every model-input tensor matched Python exactly (rtol=atol=0), including the one with an
/// OpenCV resize inside it.</b> <see cref="TestPrepareAudioFrameMatchesPythonExactlyForEdtalk"/>/
/// <c>ForWav2Lip</c> and <see cref="TestPrepareCropFrameWav2LipMatchesPythonExactly"/> were
/// never going to need slack (pure array arithmetic, no OpenCV call in either).
/// <see cref="TestPrepareCropFrameEdtalkMatchesPython"/> was the one case expected to need the
/// project's already-documented OpenCvSharp/opencv-python interpolation gap (its
/// <c>cv2.INTER_AREA</c> resize is the untested-until-now case of that gap) — measured, not
/// assumed, by first running it at a loosened tolerance and then tightening back to
/// <c>rtol=atol=0</c> to confirm; it passed at zero tolerance, so <c>INTER_AREA</c> did not
/// reproduce the bilinear gap here and the test stays at <c>rtol=atol=0</c> like every other
/// case in this file.
/// </para>
///
/// <para>
/// <b>Two tiers, gated differently</b> — same split as <c>FaceSwapperParityTests</c>: the
/// preprocessing-tensor tests need only the committed <c>.npy</c> fixtures and the source
/// image/audio, so they run once that media is present; the end-to-end tests that run a real
/// <see cref="InferenceSession"/> additionally need the corresponding <c>.onnx</c> file(s)
/// under <c>.assets/models/</c> (<c>.gitignore</c>'d, never present on CI) and are gated with
/// <see cref="LipSyncerModelFactAttribute"/>.
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class LipSyncerParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "lip_syncer");

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";
    private const string SourceAudio = "/tmp/facefusion-test-examples/source.mp3";
    private const double Fps = 25.0;
    private const int FrameNumber = 10;

    // -----------------------------------------------------------------
    // Shared fixture / environment helpers
    // -----------------------------------------------------------------

    internal static string? FindRepoRoot()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    internal static string? FindModelPath(string modelFileName)
    {
        var repoRoot = FindRepoRoot();
        return repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models", modelFileName);
    }

    internal static bool ModelAvailable(string modelFileName)
    {
        var modelPath = FindModelPath(modelFileName);
        return modelPath is not null && File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
    }

    internal static bool SourceImageAvailable =>
        File.Exists(SourceImage) && new FileInfo(SourceImage).Length > 0;

    internal static bool SourceAudioAvailable =>
        File.Exists(SourceAudio) && new FileInfo(SourceAudio).Length > 0;

    private static NpyArray LoadNpy(string family, string name) =>
        NpyReader.Load(Path.Combine(FixturesDirectory, family, name + ".npy"));

    private static Mat MatFromUInt8HwcFixture(NpyArray array)
    {
        Assert.Equal("uint8", array.DType);
        Assert.Equal(3, array.Shape.Count);
        Assert.Equal(3, array.Shape[2]);

        var height = array.Shape[0];
        var width = array.Shape[1];
        var raw = array.RawData;

        var mat = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[height * width];
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Vec3b(raw[i * 3], raw[(i * 3) + 1], raw[(i * 3) + 2]);
        }

        mat.SetArray(pixels);
        return mat;
    }

    private static double[,] LoadAudioFrame(string family)
    {
        var array = LoadNpy(family, "source_voice_frame");
        Assert.Equal(new[] { 80, 16 }, array.Shape);

        var flat = array.AsDoubles();
        var result = new double[80, 16];
        for (var mel = 0; mel < 80; mel++)
        {
            for (var step = 0; step < 16; step++)
            {
                result[mel, step] = flat[(mel * 16) + step];
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // The mel-spectrogram itself (FaceFusion.Media.Audio, not lip_syncer's own code)
    // -----------------------------------------------------------------

    /// <summary>
    /// Checks <c>FaceFusion.Media.Audio.GetVoiceFrame</c> — the audio layer's own live output,
    /// nothing in <c>LipSyncer.cs</c> — against Python's <c>audio.get_voice_frame()</c> fixture.
    /// See the class remarks for why this is a separate test from
    /// <see cref="TestPrepareAudioFrameMatchesPythonExactlyForEdtalk"/>.
    /// </summary>
    [LipSyncerAudioModelFact]
    public void TestSourceVoiceFrameMatchesTheAudioLayer()
    {
        using var voiceExtractorSession = new InferenceSession(FindModelPath("kim_vocal_2.onnx"));

        double[,] ExtractVoice(double[,] audio, int chunkSize, int stepSize) =>
            VoiceExtractor.BatchExtractVoice(audio, chunkSize, stepSize, voiceExtractorSession);

        var actual = Audio.GetVoiceFrame(SourceAudio, Fps, ExtractVoice, FrameNumber);
        Assert.NotNull(actual);

        var expected = LoadNpy("edtalk_256", "source_voice_frame").AsDoubles();

        var actualFlat = new double[80 * 16];
        for (var mel = 0; mel < 80; mel++)
        {
            for (var step = 0; step < 16; step++)
            {
                actualFlat[(mel * 16) + step] = actual![mel, step];
            }
        }

        // Same tolerance AudioParityTests already established for read_voice's output — the
        // float32 batch_extract_voice precision divergence upstream of lfilter (Audio.cs's
        // documented, deliberate approximation), not anything new to lip_syncer.
        var result = TensorComparison.Compare(actualFlat, expected, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // prepare_audio_frame — the 'source' model input, no ONNX Runtime required
    // -----------------------------------------------------------------

    [LipSyncerSourceMediaFact]
    public void TestPrepareAudioFrameMatchesPythonExactlyForEdtalk()
    {
        var audioFrame = LoadAudioFrame("edtalk_256");
        var actual = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Edtalk, audioFrame, lipSyncerWeight: 0.5);

        var expected = LoadNpy("edtalk_256", "prepared_audio_frame").AsDoubles();
        var actualDoubles = Array.ConvertAll(actual, value => (double)value);

        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    [LipSyncerSourceMediaFact]
    public void TestPrepareAudioFrameMatchesPythonExactlyForWav2Lip()
    {
        var audioFrame = LoadAudioFrame("wav2lip_gan_96");
        var actual = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Wav2Lip, audioFrame, lipSyncerWeight: 0.3);

        var expected = LoadNpy("wav2lip_gan_96", "prepared_audio_frame").AsDoubles();
        var actualDoubles = Array.ConvertAll(actual, value => (double)value);

        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame — the 'target' model input
    // -----------------------------------------------------------------

    /// <summary>
    /// <c>PrepareCropFrameEdtalk</c> resizes the 512x512 crop down to 256x256 with
    /// <c>cv2.INTER_AREA</c> before the RGB/255.0 conversion — the one model-input tensor in
    /// this file with an OpenCV resize inside it, and so the one candidate for the project's
    /// already-documented OpenCvSharp/opencv-python interpolation gap (measured elsewhere for
    /// bilinear: up to 2/255 on ~9% of pixels, ~62 dB PSNR). Measured directly against this
    /// fixture at a loosened tolerance first, then tightened back down: it matches at
    /// <c>rtol=atol=0</c>, so <c>INTER_AREA</c> does not reproduce that gap for this input and
    /// no tolerance loosening was needed here after all.
    /// </summary>
    [LipSyncerSourceMediaFact]
    public void TestPrepareCropFrameEdtalkMatchesPython()
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy("edtalk_256", "crop_vision_frame"));
        var modelSize = new Size(256, 256);

        var actual = LipSyncer.PrepareCropFrameEdtalk(cropVisionFrame, modelSize);
        var expected = LoadNpy("edtalk_256", "target_input").AsDoubles();
        var actualDoubles = Array.ConvertAll(actual, value => (double)value);

        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    [LipSyncerSourceMediaFact]
    public void TestPrepareCropFrameWav2LipMatchesPythonExactly()
    {
        using var areaVisionFrame = MatFromUInt8HwcFixture(LoadNpy("wav2lip_gan_96", "area_vision_frame"));

        var actual = LipSyncer.PrepareCropFrameWav2Lip(areaVisionFrame);
        var expected = LoadNpy("wav2lip_gan_96", "target_input").AsDoubles();
        var actualDoubles = Array.ConvertAll(actual, value => (double)value);

        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // box_mask — reused FaceMasker, sanity-checked (not re-verifying FaceMasker's own parity)
    // -----------------------------------------------------------------

    [LipSyncerSourceMediaFact]
    public void TestBoxMaskMatchesPython()
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy("edtalk_256", "crop_vision_frame"));
        using var actual = FaceFusion.Face.FaceMasker.CreateBoxMask(cropVisionFrame, faceMaskBlur: 0.3, faceMaskPadding: new Padding(0, 0, 0, 0));

        var expected = LoadNpy("edtalk_256", "box_mask");
        Assert.Equal(new[] { actual.Rows, actual.Cols }, expected.Shape);

        actual.GetArray(out float[] actualValues);
        var expectedValues = expected.AsDoubles();

        var result = TensorComparison.Compare(Array.ConvertAll(actualValues, v => (double)v), expectedValues, relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference)
    // -----------------------------------------------------------------

    [LipSyncerModelFact("edtalk_256.onnx")]
    public void TestForwardEdtalkMatchesPythonRawModelOutputAndNormalize()
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy("edtalk_256", "crop_vision_frame"));
        var modelSize = new Size(256, 256);
        var targetInput = LipSyncer.PrepareCropFrameEdtalk(cropVisionFrame, modelSize);
        var sourceInput = LoadNpy("edtalk_256", "prepared_audio_frame").AsFloats();

        using var lipSyncerSession = new InferenceSession(FindModelPath("edtalk_256.onnx"));
        var rawOutput = LipSyncer.ForwardEdtalk(lipSyncerSession, sourceInput, targetInput, modelSize, lipSyncerWeight: 0.5f);

        var expectedRawOutput = LoadNpy("edtalk_256", "raw_model_output").AsDoubles();
        var rawResult = TensorComparison.Compare(Array.ConvertAll(rawOutput, v => (double)v), expectedRawOutput, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(rawResult.Passed, $"raw model output: {rawResult.Describe()}");

        using var normalized = LipSyncer.NormalizeCropFrameEdtalk(rawOutput, modelSize.Height, modelSize.Width);
        var expectedNormalized = LoadNpy("edtalk_256", "normalized_crop_frame");
        Assert.Equal(new[] { normalized.Rows, normalized.Cols, 3 }, expectedNormalized.Shape);

        normalized.GetArray(out Vec3b[] actualPixels);
        var expectedBytes = expectedNormalized.RawData;

        // The final cv2.INTER_CUBIC upscale to 512x512 is a second OpenCV resize on top of the
        // ONNX output — measured against the project's documented OpenCvSharp/opencv-python
        // interpolation gap the same way TestPrepareCropFrameEdtalkMatchesPython was: it turned
        // out byte-exact (PSNR = +Infinity), so this asserts that directly rather than reaching
        // for PSNR as an unearned safety margin.
        var mismatches = 0;
        for (var i = 0; i < actualPixels.Length; i++)
        {
            var pixel = actualPixels[i];
            if (pixel.Item0 != expectedBytes[i * 3] || pixel.Item1 != expectedBytes[(i * 3) + 1] || pixel.Item2 != expectedBytes[(i * 3) + 2])
            {
                mismatches++;
            }
        }

        Assert.True(mismatches == 0, $"normalized_crop_frame: {mismatches}/{actualPixels.Length} pixels differ (expected byte-exact).");
    }

    [LipSyncerModelFact("wav2lip_gan_96.onnx")]
    public void TestForwardWav2LipMatchesPythonRawModelOutputAndNormalize()
    {
        using var areaVisionFrame = MatFromUInt8HwcFixture(LoadNpy("wav2lip_gan_96", "area_vision_frame"));
        var modelSize = new Size(96, 96);
        var targetInput = LipSyncer.PrepareCropFrameWav2Lip(areaVisionFrame);
        var sourceInput = LoadNpy("wav2lip_gan_96", "prepared_audio_frame").AsFloats();

        using var lipSyncerSession = new InferenceSession(FindModelPath("wav2lip_gan_96.onnx"));
        var rawOutput = LipSyncer.ForwardWav2Lip(lipSyncerSession, sourceInput, targetInput, modelSize);

        var expectedRawOutput = LoadNpy("wav2lip_gan_96", "raw_model_output").AsDoubles();
        var rawResult = TensorComparison.Compare(Array.ConvertAll(rawOutput, v => (double)v), expectedRawOutput, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(rawResult.Passed, $"raw model output: {rawResult.Describe()}");

        using var normalized = LipSyncer.NormalizeCropFrameWav2Lip(rawOutput, modelSize.Height, modelSize.Width);
        var expectedNormalized = LoadNpy("wav2lip_gan_96", "normalized_area_vision_frame");
        Assert.Equal(new[] { normalized.Rows, normalized.Cols, 3 }, expectedNormalized.Shape);

        normalized.GetArray(out Vec3b[] actualPixels);
        var expectedBytes = expectedNormalized.RawData;

        // No resize anywhere in this branch (no cv2.resize inside prepare_crop_frame/
        // normalize_crop_frame for wav2lip) — measured byte-exact, so asserted byte-exact
        // rather than leaving slack for a gap this branch cannot hit.
        var mismatches = 0;
        for (var i = 0; i < actualPixels.Length; i++)
        {
            var pixel = actualPixels[i];
            if (pixel.Item0 != expectedBytes[i * 3] || pixel.Item1 != expectedBytes[(i * 3) + 1] || pixel.Item2 != expectedBytes[(i * 3) + 2])
            {
                mismatches++;
            }
        }

        Assert.True(mismatches == 0, $"normalized_area_vision_frame: {mismatches}/{actualPixels.Length} pixels differ (expected byte-exact).");
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the example media
/// (<c>/tmp/facefusion-test-examples/source.jpg</c>) is not present — same reasoning as
/// <c>FaceSwapperParityTests.FaceSwapperSourceImageFactAttribute</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class LipSyncerSourceMediaFactAttribute : FactAttribute
{
    public LipSyncerSourceMediaFactAttribute()
    {
        if (!LipSyncerParityTests.SourceImageAvailable)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.jpg) — run tools/parity/fetch_examples.sh, then retry";
        }
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c>
/// file(s) (or the example image) are not present.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class LipSyncerModelFactAttribute : FactAttribute
{
    public LipSyncerModelFactAttribute(params string[] modelFileNames)
    {
        if (!LipSyncerParityTests.SourceImageAvailable)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.jpg) — run tools/parity/fetch_examples.sh, then retry";
            return;
        }

        foreach (var modelFileName in modelFileNames)
        {
            if (!LipSyncerParityTests.ModelAvailable(modelFileName))
            {
                Skip = $"requires .assets/models/{modelFileName} — run lip_syncer.pre_check() from Python (see tools/parity/dump_lip_syncer.py) to fetch it, then retry";
                return;
            }
        }
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when either the example audio
/// (<c>source.mp3</c>) or the real <c>kim_vocal_2.onnx</c> voice-extractor model is not
/// present — only <see cref="LipSyncerParityTests.TestSourceVoiceFrameMatchesTheAudioLayer"/>
/// needs both (a real end-to-end <c>read_voice</c> call).
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class LipSyncerAudioModelFactAttribute : FactAttribute
{
    public LipSyncerAudioModelFactAttribute()
    {
        if (!LipSyncerParityTests.SourceAudioAvailable)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.mp3) — run tools/parity/fetch_examples.sh, then retry";
            return;
        }

        if (!LipSyncerParityTests.ModelAvailable("kim_vocal_2.onnx"))
        {
            Skip = "requires .assets/models/kim_vocal_2.onnx — run voice_extractor.pre_check() from Python to fetch it, then retry";
        }
    }
}
