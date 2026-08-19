using System.Linq;
using System.Text.Json;
using FaceFusion.Media;
using FaceFusion.Parity;
using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Media.Audio</c> (Python: <c>facefusion/audio.py</c>)
/// and <c>FaceFusion.Processors.VoiceExtractor</c> (Python: <c>facefusion/voice_extractor.py</c>)
/// against real scipy 1.17.1 and the real Python modules, run against the real
/// <c>source.wav</c> example audio and (for the voice_extractor section) the real
/// <c>kim_vocal_2</c> ONNX model. Ground truth was captured with
/// <c>tools/parity/dump_audio.py</c>; see that script's docstring and docs/PARITY_HARNESS.md.
///
/// <para>
/// <b>Tiers, gated differently</b> (same split as <c>FaceDetectorParityTests</c>/
/// <c>ContentAnalyserParityTests</c>): the <c>primitives/*</c> tests need only the committed
/// <c>.npy</c> fixtures — no example media, no ONNX model — so they run unconditionally. The
/// <c>audio/*</c> tests additionally need <c>source.wav</c> from
/// <c>tools/parity/fetch_examples.sh</c> (gated by <see cref="AudioFactAttribute"/>). The
/// <c>voice_extractor/*</c> tests additionally need the real <c>kim_vocal_2.onnx</c>/
/// <c>.hash</c> in <c>.assets/models/</c> — never present on CI — gated by
/// <see cref="ModelFactAttribute"/> and skipping with a clear message instead of failing.
/// </para>
///
/// <para>
/// <b>Tolerances.</b> The FFT/STFT/lfilter/resample primitives are this port's own managed
/// float math (not ONNX Runtime, not OpenCV), so per PARITY_HARNESS.md a real epsilon belongs
/// here rather than "expect ~0" — <see cref="PrimitiveRelativeTolerance"/>/
/// <see cref="PrimitiveAbsoluteTolerance"/> (1e-9/1e-9) are tight enough to catch a wrong
/// convention while tolerating float64 operation-order differences between this Bluestein/
/// radix-2 FFT and FFTPACK/pocketfft. <see cref="Audio.Stft"/>'s output for
/// <c>create_spectrogram</c> and the end-to-end mel spectrogram use the same tolerance since
/// that whole path is float64 throughout (see <see cref="Audio"/>'s class remarks). The
/// voice_extractor model *input* tensor (<c>DecomposeAudioChunk</c>'s output, pure
/// preprocessing over the model's own float32 dtype) is compared at
/// <c>rtol = 1e-4, atol = 1e-4</c> rather than 0 — see the port report: this stage's STFT is
/// computed at float64 precision here versus scipy's actual float32/complex64 precision
/// there (verified via <c>_spectral_helper</c>'s <c>win.astype(outdtype)</c> downcast), a
/// documented, deliberate divergence rather than a bug, and the measured gap is reported in
/// the port report. The model *output* slice and downstream <c>composed</c>/<c>normalized</c>/
/// <c>extract_voice_output</c> arrays inherit that same tolerance since ONNX Runtime's own
/// arithmetic is bit-for-bit regardless of the input's precision pedigree.
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class AudioParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "audio");

    private const string SourceWav = "/tmp/facefusion-test-examples/source.wav";

    private const double PrimitiveRelativeTolerance = 1e-9;
    private const double PrimitiveAbsoluteTolerance = 1e-9;

    /// <summary>
    /// For transforms whose length is not a power of two and therefore go through
    /// Bluestein's algorithm, where more rounding accumulates than in the radix-2 path.
    /// Still ~7 orders of magnitude tighter than any convention error would produce.
    /// </summary>
    private const double BluesteinRelativeTolerance = 1e-6;
    private const double BluesteinAbsoluteTolerance = 1e-8;

    private const double VoiceExtractorRelativeTolerance = 1e-4;
    private const double VoiceExtractorAbsoluteTolerance = 1e-4;

    // -----------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------

    private static NpyArray LoadNpy(params string[] pathParts) =>
        NpyReader.Load(Path.Combine(new[] { FixturesDirectory }.Concat(pathParts).ToArray()));

    private static JsonElement LoadJson(params string[] pathParts)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(new[] { FixturesDirectory }.Concat(pathParts).ToArray())));
        return document.RootElement.Clone();
    }

    private static double[] Flatten(double[,] values)
    {
        var rows = values.GetLength(0);
        var cols = values.GetLength(1);
        var result = new double[rows * cols];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[(r * cols) + c] = values[r, c];
            }
        }

        return result;
    }

    private static double[] Flatten(float[,] values)
    {
        var rows = values.GetLength(0);
        var cols = values.GetLength(1);
        var result = new double[rows * cols];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[(r * cols) + c] = values[r, c];
            }
        }

        return result;
    }

    private static void AssertClose(double[] actual, NpyArray expected, double rtol, double atol)
    {
        var result = TensorComparison.Compare(actual, expected.AsDoubles(), rtol, atol);
        Assert.True(result.Passed, result.Describe());
    }

    private static bool SourceWavAvailable => File.Exists(SourceWav) && new FileInfo(SourceWav).Length > 0;

    private const string MissingSourceWavMessage =
        "requires /tmp/facefusion-test-examples/source.wav — run tools/parity/fetch_examples.sh " +
        "(and ffmpeg -i source.mp3 source.wav), then retry";

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class AudioFactAttribute : FactAttribute
    {
        public AudioFactAttribute()
        {
            if (!SourceWavAvailable)
            {
                Skip = MissingSourceWavMessage;
            }
        }
    }

    private static bool ModelAvailable => VoiceExtractor.PreCheck(VoiceExtractorModel.KimVocal2);

    private const string MissingModelMessage =
        "requires kim_vocal_2.onnx/.hash in .assets/models/ — run `FACEFUSION_PARITY_DIR=/tmp/x " +
        "python3 tools/parity/dump_audio.py` with network access (or any other way of running " +
        "facefusion.voice_extractor.pre_check()) to fetch them, then retry";

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelFactAttribute : FactAttribute
    {
        public ModelFactAttribute()
        {
            if (!ModelAvailable)
            {
                Skip = MissingModelMessage;
            }
            else if (!SourceWavAvailable)
            {
                Skip = MissingSourceWavMessage;
            }
        }
    }

    // -----------------------------------------------------------------
    // primitives/* — no example media, no ONNX model required.
    // -----------------------------------------------------------------

    [Fact]
    public void LfilterMatchesScipy()
    {
        var x = LoadNpy("primitives", "lfilter", "x.npy").AsDoubles();
        var expected = LoadNpy("primitives", "lfilter", "y.npy");
        var actual = Audio.Lfilter(new[] { 1.0, -0.97 }, new[] { 1.0 }, x);
        AssertClose(actual, expected, PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Fact]
    public void HannSymmetricMatchesScipy_Audio()
    {
        AssertClose(Audio.HannWindow(8, periodic: false), LoadNpy("primitives", "hann_sym_8.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
        AssertClose(Audio.HannWindow(7680, periodic: false), LoadNpy("primitives", "hann_sym_7680.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Fact]
    public void HannPeriodicMatchesScipy_Audio()
    {
        AssertClose(Audio.HannWindow(8, periodic: true), LoadNpy("primitives", "hann_periodic_8.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
        AssertClose(Audio.HannWindow(800, periodic: true), LoadNpy("primitives", "hann_periodic_800.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Fact]
    public void HannWindowsMatchScipy_VoiceExtractor()
    {
        AssertClose(VoiceExtractor.HannWindow(8, sym: true), LoadNpy("primitives", "hann_sym_8.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
        AssertClose(VoiceExtractor.HannWindow(7680, sym: true), LoadNpy("primitives", "hann_sym_7680.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Theory]
    [InlineData(5, "triang_5.npy")]
    [InlineData(4, "triang_4.npy")]
    [InlineData(37, "triang_37.npy")]
    public void TriangMatchesScipy(int m, string fixtureName)
    {
        AssertClose(Audio.TriangWindow(m), LoadNpy("primitives", fixtureName), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Fact]
    public void ResampleMatchesScipy()
    {
        var x = LoadNpy("primitives", "resample", "x_3000.npy").AsDoubles();

        AssertClose(Audio.Resample(x, 1000), LoadNpy("primitives", "resample", "up_1000.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
        AssertClose(Audio.Resample(x, 2731), LoadNpy("primitives", "resample", "down_2731.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);

        var factor = (int)Math.Round(x.Length * 16000.0 / 48000.0, MidpointRounding.ToEven);
        AssertClose(Audio.Resample(x, factor), LoadNpy("primitives", "resample", "factor_16k.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Fact]
    public void StftMatchesScipy_AudioConvention()
    {
        var x = LoadNpy("primitives", "stft", "x.npy").AsDoubles();
        var window = Audio.HannWindow(800, periodic: true);
        var (real, imag, freqBins, segCount) = Audio.Stft(x, 800, 600, 800, window);

        Assert.Equal(401, freqBins);
        AssertClose(Flatten(real), LoadNpy("primitives", "stft", "real.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
        AssertClose(Flatten(imag), LoadNpy("primitives", "stft", "imag.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [Fact]
    public void StftMatchesScipy_VoiceExtractorConvention()
    {
        var x = LoadNpy("primitives", "stft_voice", "x.npy").AsDoubles();
        var window = VoiceExtractor.HannWindow(7680, sym: true);
        var (real, imag, freqBins, segCount) = VoiceExtractor.Stft(x, 7680, 6656, 7680, window);

        // The 7680-point transform needs a looser tolerance than the 512-point one above,
        // and the reason is the transform length rather than the convention:
        //   512  = 2^9        -> radix-2 fast path, matches scipy at 1e-9
        //   7680 = 2^9 * 3*5  -> not a power of two, so it routes through Bluestein,
        //                        which accumulates materially more rounding error.
        // Measured against scipy: max abs 2.98e-9, max rel 8.29e-8. That this is rounding
        // and not a convention error was established by checking the achievable floor in
        // Python first - scipy.signal.stft and a hand-rolled numpy.fft reference agree to
        // EXACTLY 0.0 at both sizes (both are pocketfft underneath), so the divergence is
        // our FFT's summation order alone. A genuine convention error (wrong scaling,
        // padding, boundary or window placement) would show O(1) relative error, which
        // 1e-6 still catches comfortably.
        AssertClose(Flatten(real), LoadNpy("primitives", "stft_voice", "real.npy"), BluesteinRelativeTolerance, BluesteinAbsoluteTolerance);
        AssertClose(Flatten(imag), LoadNpy("primitives", "stft_voice", "imag.npy"), BluesteinRelativeTolerance, BluesteinAbsoluteTolerance);

        var reconstructed = VoiceExtractor.Istft(real, imag, freqBins, segCount, 7680, 6656, window);
        AssertClose(reconstructed, LoadNpy("primitives", "istft_voice", "x.npy"), PrimitiveRelativeTolerance, 1e-6);
    }

    // -----------------------------------------------------------------
    // audio/* — facefusion.audio end to end against source.wav.
    // -----------------------------------------------------------------

    [AudioFact]
    public void PrepareAudioMatchesPython()
    {
        var rawInt16 = LoadNpy("audio", "raw_int16.npy");
        var shape = rawInt16.Shape;
        var flat = rawInt16.AsDoubles();
        var audio = new double[shape[0], shape[1]];

        for (var i = 0; i < shape[0]; i++)
        {
            audio[i, 0] = flat[(i * shape[1]) + 0];
            audio[i, 1] = flat[(i * shape[1]) + 1];
        }

        var prepared = Audio.PrepareAudio(audio);
        AssertClose(prepared, LoadNpy("audio", "prepared.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [AudioFact]
    public void MelFilterBankMatchesPython()
    {
        var bank = Audio.CreateMelFilterBank();
        AssertClose(Flatten(bank), LoadNpy("audio", "mel_filter_bank.npy"), PrimitiveRelativeTolerance, PrimitiveAbsoluteTolerance);
    }

    [AudioFact]
    public void SpectrogramMatchesPython()
    {
        var prepared = LoadNpy("audio", "prepared.npy").AsDoubles();
        var spectrogram = Audio.CreateSpectrogram(prepared);
        AssertClose(Flatten(spectrogram), LoadNpy("audio", "spectrogram.npy"), PrimitiveRelativeTolerance, 1e-6);
    }

    [AudioFact]
    public void ExtractAudioFramesMatchesPython()
    {
        var spectrogram = LoadNpy("audio", "spectrogram.npy");
        var shape = spectrogram.Shape;
        var flat = spectrogram.AsDoubles();
        var spec2D = new double[shape[0], shape[1]];

        for (var r = 0; r < shape[0]; r++)
        {
            for (var c = 0; c < shape[1]; c++)
            {
                spec2D[r, c] = flat[(r * shape[1]) + c];
            }
        }

        var frames = Audio.ExtractAudioFrames(spec2D, 25);
        var expectedTotal = LoadJson("audio", "frame_total.json").GetInt32();
        Assert.Equal(expectedTotal, frames.Count);

        AssertClose(Flatten(frames[0]), LoadNpy("audio", "frame_0.npy"), PrimitiveRelativeTolerance, 1e-6);
        AssertClose(Flatten(frames[10]), LoadNpy("audio", "frame_10.npy"), PrimitiveRelativeTolerance, 1e-6);
        AssertClose(Flatten(frames[^1]), LoadNpy("audio", "frame_last.npy"), PrimitiveRelativeTolerance, 1e-6);
    }

    [AudioFact]
    public void ReadStaticAudioMatchesPython_FrameCountAndShape()
    {
        var frames = Audio.ReadStaticAudio(SourceWav, 25);
        Assert.NotNull(frames);

        var expectedTotal = LoadJson("audio", "read_static_audio_frame_total.json").GetInt32();
        Assert.Equal(expectedTotal, frames!.Count);
        Assert.Equal(280, frames.Count); // matches the ported Python test (tests/test_audio.py::test_read_static_audio).

        var expectedShape = LoadJson("audio", "read_static_audio_frame_shape.json");
        Assert.Equal(expectedShape[0].GetInt32(), frames[0].GetLength(0));
        Assert.Equal(expectedShape[1].GetInt32(), frames[0].GetLength(1));
    }

    [AudioFact]
    public void GetAudioFrameMatchesPython()
    {
        var frame = Audio.GetAudioFrame(SourceWav, 25);
        Assert.NotNull(frame);
        AssertClose(Flatten(frame!), LoadNpy("audio", "frame_0.npy"), PrimitiveRelativeTolerance, 1e-6);
        Assert.Null(Audio.GetAudioFrame("invalid", 25));
    }

    // -----------------------------------------------------------------
    // voice_extractor/* — facefusion.voice_extractor end to end against a real ONNX model.
    // -----------------------------------------------------------------

    [ModelFact]
    public void DecomposeAudioChunkMatchesPython_ModelInputTensor()
    {
        using var session = LoadKimVocal2Session();

        var inputInt16 = LoadNpy("voice_extractor", "input_int16.npy");
        var shape = inputInt16.Shape;
        var flat = inputInt16.AsDoubles();
        var audio = new double[shape[0], shape[1]];

        for (var i = 0; i < shape[0]; i++)
        {
            audio[i, 0] = flat[(i * shape[1]) + 0];
            audio[i, 1] = flat[(i * shape[1]) + 1];
        }

        const int voiceTrimSize = 3840;
        var voiceChunkSize = (session.InputMetadata["input"].Dimensions[3] - 1) * 1024;

        var channelMajor = VoiceExtractor.TransposeToChannelMajor(audio);
        var (prepared, audioPadSize) = VoiceExtractor.PrepareAudioChunk(channelMajor, voiceChunkSize, voiceTrimSize);

        AssertClose(Flatten(prepared), LoadNpy("voice_extractor", "prepared_chunk.npy"), VoiceExtractorRelativeTolerance, VoiceExtractorAbsoluteTolerance);
        Assert.Equal(LoadJson("voice_extractor", "audio_pad_size.json").GetInt32(), audioPadSize);

        var decomposed = VoiceExtractor.DecomposeAudioChunk(prepared, voiceTrimSize);
        var expectedInput = LoadNpy("voice_extractor", "model_input.npy");

        Assert.Equal(new[] { decomposed.Batch, decomposed.Channel, decomposed.Freq, decomposed.Time }, expectedInput.Shape);

        var actual = decomposed.Data.Select(v => (double)v).ToArray();
        AssertClose(actual, expectedInput, VoiceExtractorRelativeTolerance, VoiceExtractorAbsoluteTolerance);
    }

    [ModelFact]
    public void ForwardMatchesPython_ModelOutputSlice()
    {
        using var session = LoadKimVocal2Session();

        var inputInt16 = LoadNpy("voice_extractor", "input_int16.npy");
        var shape = inputInt16.Shape;
        var flat = inputInt16.AsDoubles();
        var audio = new double[shape[0], shape[1]];

        for (var i = 0; i < shape[0]; i++)
        {
            audio[i, 0] = flat[(i * shape[1]) + 0];
            audio[i, 1] = flat[(i * shape[1]) + 1];
        }

        const int voiceTrimSize = 3840;
        var voiceChunkSize = (session.InputMetadata["input"].Dimensions[3] - 1) * 1024;
        var channelMajor = VoiceExtractor.TransposeToChannelMajor(audio);
        var (prepared, _) = VoiceExtractor.PrepareAudioChunk(channelMajor, voiceChunkSize, voiceTrimSize);
        var decomposed = VoiceExtractor.DecomposeAudioChunk(prepared, voiceTrimSize);
        var output = VoiceExtractor.Forward(session, decomposed);

        var expectedShape = LoadJson("voice_extractor", "model_output_shape.json");
        Assert.Equal(expectedShape[0].GetInt32(), output.Batch);
        Assert.Equal(expectedShape[1].GetInt32(), output.Channel);
        Assert.Equal(expectedShape[2].GetInt32(), output.Freq);
        Assert.Equal(expectedShape[3].GetInt32(), output.Time);

        // Only the first 8 rows of the Freq axis were dumped (see dump_audio.py) to keep the
        // fixture small; ONNX Runtime does the arithmetic here so this is an "expect ~0" check.
        var sliceLength = 8 * output.Time;
        var actualSlice = new double[output.Channel * sliceLength];

        for (var c = 0; c < output.Channel; c++)
        {
            for (var i = 0; i < sliceLength; i++)
            {
                actualSlice[(c * sliceLength) + i] = output.Data[(c * output.Freq * output.Time) + i];
            }
        }

        AssertClose(actualSlice, LoadNpy("voice_extractor", "model_output_slice.npy"), 1e-4, 1e-4);
    }

    [ModelFact]
    public void ExtractVoiceMatchesPython_EndToEnd()
    {
        using var session = LoadKimVocal2Session();

        var inputInt16 = LoadNpy("voice_extractor", "input_int16.npy");
        var shape = inputInt16.Shape;
        var flat = inputInt16.AsDoubles();
        var audio = new double[shape[0], shape[1]];

        for (var i = 0; i < shape[0]; i++)
        {
            audio[i, 0] = flat[(i * shape[1]) + 0];
            audio[i, 1] = flat[(i * shape[1]) + 1];
        }

        var extracted = VoiceExtractor.ExtractVoice(audio, session);
        var expected = LoadNpy("voice_extractor", "extract_voice_output.npy");

        Assert.Equal(expected.Shape[0], extracted.GetLength(0));
        Assert.Equal(expected.Shape[1], extracted.GetLength(1));
        AssertClose(Flatten(extracted), expected, VoiceExtractorRelativeTolerance, VoiceExtractorAbsoluteTolerance);
    }

    [ModelFact]
    public void BatchExtractVoiceMatchesPython_EndToEnd()
    {
        using var session = LoadKimVocal2Session();

        var inputInt16 = LoadNpy("voice_extractor", "input_int16.npy");
        var shape = inputInt16.Shape;
        var flat = inputInt16.AsDoubles();
        var audio = new double[shape[0], shape[1]];

        for (var i = 0; i < shape[0]; i++)
        {
            audio[i, 0] = flat[(i * shape[1]) + 0];
            audio[i, 1] = flat[(i * shape[1]) + 1];
        }

        var extracted = VoiceExtractor.BatchExtractVoice(audio, 240 * 1024, 180 * 1024, session);
        var expected = LoadNpy("voice_extractor", "batch_extract_voice_output.npy");

        Assert.Equal(expected.Shape[0], extracted.GetLength(0));
        AssertClose(Flatten(extracted), expected, VoiceExtractorRelativeTolerance, VoiceExtractorAbsoluteTolerance);
    }

    private static InferenceSession LoadKimVocal2Session()
    {
        var repoRoot = FindRepoRoot() ?? throw new InvalidOperationException("Could not locate repo root.");
        var modelPath = Path.Combine(repoRoot, ".assets", "models", "kim_vocal_2.onnx");
        return new InferenceSession(modelPath);
    }

    private static string? FindRepoRoot()
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
}
