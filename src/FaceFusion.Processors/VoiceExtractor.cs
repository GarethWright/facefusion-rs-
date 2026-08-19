using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using FaceFusion.Inference;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;

// Grants the parity-test project access to the internal numeric primitives and pipeline
// stages (PrepareAudioChunk, DecomposeAudioChunk, ComposeAudioChunk, Stft, Istft, Fft, ...)
// so they can be verified directly against real scipy/voice_extractor.py ground truth. A
// C# assembly attribute, not a project-file edit — see the analogous one in
// FaceFusion.Media/Audio.cs.
[assembly: InternalsVisibleTo("FaceFusion.ParityTests")]
[assembly: InternalsVisibleTo("FaceFusion.UnitTests")]

namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/voice_extractor.py</c> — separates vocals from a stereo audio
/// signal via an MDX-net-family ONNX model (<c>kim_vocal_1</c>/<c>kim_vocal_2</c>/
/// <c>uvr_mdxnet</c>), by chunking the signal, taking a windowed STFT of each chunk, running
/// the model over the complex spectrogram, taking the inverse STFT, and averaging overlapping
/// chunks back into a continuous waveform.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Every Python function here reads
/// <c>state_manager.get_item('voice_extractor_model')</c> and calls
/// <c>get_inference_pool().get(...)</c> internally. Every one of those becomes an explicit
/// <see cref="InferenceSession"/> parameter here instead, same convention as
/// <c>FaceFusion.Face.FaceDetector</c>.
/// </para>
///
/// <para>
/// <b>Duplicated FFT/STFT/ISTFT/Hann-window machinery (deliberate, not an oversight).</b>
/// <c>FaceFusion.Media.Audio</c> (this port's <c>facefusion/audio.py</c>) needs the same
/// primitives, but per PORT_CONVENTIONS.md project references are not something this phase
/// may add, and the natural dependency direction in this solution already has
/// <c>FaceFusion.Processors</c> as the *higher* layer — Python's own
/// <c>facefusion/audio.py</c> imports <c>facefusion/voice_extractor.py</c>, not the reverse,
/// so even a same-direction reference would not help here (<c>FaceFusion.Media</c>, the lower
/// layer, would need to depend on <c>FaceFusion.Processors</c>, which is backwards). The two
/// copies are not identical: <see cref="Audio.PrepareAudio"/>'s (in the sibling
/// project) pipeline runs at float64 throughout (matching Python's actual numpy promotion
/// there), while this file's pipeline runs at float32 throughout (matching Python's explicit
/// <c>.astype(numpy.float32)</c> calls here, since the array is an ONNX model's input/output).
/// See the port report for the measured consequence of computing the FFT itself in double
/// precision rather than reproducing scipy's actual float32/complex64 arithmetic path exactly
/// (a deliberate, documented approximation — the alternative was a second, single-precision
/// FFT implementation purely to chase bit-exactness on an intermediate signal-processing
/// stage, which was judged not worth the doubled surface area to keep correct).
/// </para>
///
/// <para>
/// <b>Model set / <c>pre_check</c> — same reduced-scope port as <c>FaceDetector</c>.</b> See
/// that class's remarks: <see cref="PreCheck"/> here checks file presence only (no hash
/// verification, no network download — <c>facefusion/download.py</c>/<c>hash_helper.py</c>
/// are out of this module's assignment).
/// </para>
/// </summary>
public static class VoiceExtractor
{
    private static readonly IReadOnlyList<VoiceExtractorModel> AllModels = new[]
    {
        VoiceExtractorModel.KimVocal1, VoiceExtractorModel.KimVocal2, VoiceExtractorModel.UvrMdxnet,
    };

    private static readonly IReadOnlyDictionary<VoiceExtractorModel, string> ModelFileNames = new Dictionary<VoiceExtractorModel, string>
    {
        [VoiceExtractorModel.KimVocal1] = "kim_vocal_1",
        [VoiceExtractorModel.KimVocal2] = "kim_vocal_2",
        [VoiceExtractorModel.UvrMdxnet] = "uvr_mdxnet",
    };

    // Python: create_static_model_set's per-model resolve_download_url base_name argument.
    private static readonly IReadOnlyDictionary<VoiceExtractorModel, string> ModelBaseNames = new Dictionary<VoiceExtractorModel, string>
    {
        [VoiceExtractorModel.KimVocal1] = "models-3.4.0",
        [VoiceExtractorModel.KimVocal2] = "models-3.0.0",
        [VoiceExtractorModel.UvrMdxnet] = "models-3.4.0",
    };

    // -----------------------------------------------------------------
    // Model set / downloads / pre_check
    // -----------------------------------------------------------------

    /// <summary>Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). See <c>FaceDetector.CreateStaticModelSet</c>'s remarks for why the download URL is built directly.</summary>
    public static IReadOnlyDictionary<VoiceExtractorModel, (Download Hash, Download Source)> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        var result = new Dictionary<VoiceExtractorModel, (Download, Download)>();

        foreach (var model in AllModels)
        {
            var fileName = ModelFileNames[model];
            var baseName = ModelBaseNames[model];

            var hash = new Download(
                BuildDownloadUrl(githubProvider, baseName, fileName + ".hash"),
                Path.Combine(modelsDirectory, fileName + ".hash"));
            var source = new Download(
                BuildDownloadUrl(githubProvider, baseName, fileName + ".onnx"),
                Path.Combine(modelsDirectory, fileName + ".onnx"));

            result[model] = (hash, source);
        }

        return result;
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    /// <summary>Python: <c>resolve_relative_path('../.assets/models')</c>. See <c>FaceDetector.ResolveModelsDirectory</c>'s remarks — duplicated here for the same reason as the FFT primitives (no shared project to place it in without a .csproj edit).</summary>
    private static string ResolveModelsDirectory()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return Path.Combine(directory.FullName, ".assets", "models");
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the repository root (FaceFusion.sln) to resolve .assets/models.");
    }

    /// <summary>Python: <c>collect_model_downloads</c>.</summary>
    public static (IReadOnlyDictionary<string, Download> Hashes, IReadOnlyDictionary<string, Download> Sources) CollectModelDownloads(VoiceExtractorModel voiceExtractorModel)
    {
        var modelSet = CreateStaticModelSet(DownloadScope.Full);
        var hashes = new Dictionary<string, Download>();
        var sources = new Dictionary<string, Download>();

        var (hash, source) = modelSet[voiceExtractorModel];
        hashes[voiceExtractorModel.ToWireName()] = hash;
        sources[voiceExtractorModel.ToWireName()] = source;

        return (hashes, sources);
    }

    /// <summary>Python: <c>pre_check</c>. See the class remarks — checks file presence only.</summary>
    public static bool PreCheck(VoiceExtractorModel voiceExtractorModel)
    {
        var (hashes, sources) = CollectModelDownloads(voiceExtractorModel);
        return hashes.Values.All(download => FaceFusion.Core.FileSystem.IsFile(download.Path))
            && sources.Values.All(download => FaceFusion.Core.FileSystem.IsFile(download.Path));
    }

    // -----------------------------------------------------------------
    // Inference pool (thin wrappers around FaceFusion.Inference.InferenceManager)
    // -----------------------------------------------------------------

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    public static IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        InferenceManager inferenceManager,
        VoiceExtractorModel voiceExtractorModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var (_, modelSourceSet) = CollectModelDownloads(voiceExtractorModel);
        var modelNames = new[] { voiceExtractorModel.ToWireName() };
        return inferenceManager.GetInferencePool("facefusion.voice_extractor", modelNames, modelSourceSet, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public static void ClearInferencePool(
        InferenceManager inferenceManager,
        VoiceExtractorModel voiceExtractorModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelNames = new[] { voiceExtractorModel.ToWireName() };
        inferenceManager.ClearInferencePool("facefusion.voice_extractor", modelNames, executionDeviceIds, executionProviders);
    }

    // -----------------------------------------------------------------
    // batch_extract_voice / extract_voice
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>batch_extract_voice</c>. <paramref name="audio"/> is the raw stereo signal,
    /// shape <c>(samples, 2)</c>, values in the int16 range but represented as double (matches
    /// <see cref="FaceFusion.Media.Audio.ExtractVoiceDelegate"/>'s contract exactly, so a
    /// caller can pass <c>VoiceExtractor.BatchExtractVoice</c> — closed over an
    /// <see cref="InferenceSession"/> — straight into
    /// <see cref="FaceFusion.Media.Audio.ReadVoice"/>). Internally this widens to/from float32
    /// only around <see cref="ExtractVoice"/>, which is where Python's own array is float32
    /// (it is the ONNX model's input/output dtype).
    /// </summary>
    public static double[,] BatchExtractVoice(double[,] audio, int chunkSize, int stepSize, InferenceSession voiceExtractorSession)
    {
        var sampleTotal = audio.GetLength(0);
        var tempVoice = new double[sampleTotal, 2];
        var tempVoiceChunk = new double[sampleTotal, 2];

        for (var start = 0; start < sampleTotal; start += stepSize)
        {
            var end = Math.Min(start + chunkSize, sampleTotal);
            var slice = new double[end - start, 2];

            for (var i = 0; i < end - start; i++)
            {
                slice[i, 0] = audio[start + i, 0];
                slice[i, 1] = audio[start + i, 1];
            }

            var extracted = ExtractVoice(slice, voiceExtractorSession);

            for (var i = 0; i < end - start; i++)
            {
                tempVoice[start + i, 0] += extracted[i, 0];
                tempVoice[start + i, 1] += extracted[i, 1];
                tempVoiceChunk[start + i, 0] += 1;
                tempVoiceChunk[start + i, 1] += 1;
            }
        }

        var voice = new double[sampleTotal, 2];

        for (var i = 0; i < sampleTotal; i++)
        {
            voice[i, 0] = tempVoice[i, 0] / tempVoiceChunk[i, 0];
            voice[i, 1] = tempVoice[i, 1] / tempVoiceChunk[i, 1];
        }

        return voice;
    }

    /// <summary>Python: <c>extract_voice</c>. <paramref name="audioChunk"/> is <c>(samples, 2)</c>; the returned array has the same shape.</summary>
    public static double[,] ExtractVoice(double[,] audioChunk, InferenceSession voiceExtractorSession)
    {
        const int voiceTrimSize = 3840;

        // Python: `voice_extractor.get_inputs()[0].shape[3]` — the static (non-batch) axis of
        // the ONNX input, e.g. 256 for kim_vocal_2, giving voice_chunk_size = 255 * 1024.
        var voiceChunkSize = (voiceExtractorSession.InputMetadata["input"].Dimensions[3] - 1) * 1024;

        var channelMajor = TransposeToChannelMajor(audioChunk);
        var (prepared, audioPadSize) = PrepareAudioChunk(channelMajor, voiceChunkSize, voiceTrimSize);
        var decomposed = DecomposeAudioChunk(prepared, voiceTrimSize);
        var modelOutput = Forward(voiceExtractorSession, decomposed);
        var composed = ComposeAudioChunk(modelOutput, voiceTrimSize, voiceChunkSize);
        var normalized = NormalizeAudioChunk(composed, voiceChunkSize, voiceTrimSize, audioPadSize);
        return WidenToDouble(normalized);
    }

    /// <summary>Python: <c>temp_audio_chunk.T</c> (called just before <c>prepare_audio_chunk</c> in <c>extract_voice</c>) plus the float32 cast that lands inside <c>prepare_audio_chunk</c> itself, folded together here since nothing observes the int16-range intermediate.</summary>
    internal static float[,] TransposeToChannelMajor(double[,] audioChunk)
    {
        var sampleTotal = audioChunk.GetLength(0);
        var result = new float[2, sampleTotal];

        for (var i = 0; i < sampleTotal; i++)
        {
            result[0, i] = (float)audioChunk[i, 0];
            result[1, i] = (float)audioChunk[i, 1];
        }

        return result;
    }

    private static double[,] WidenToDouble(float[,] values)
    {
        var rows = values.GetLength(0);
        var cols = values.GetLength(1);
        var result = new double[rows, cols];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = values[r, c];
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // prepare_audio_chunk / decompose_audio_chunk / forward / compose_audio_chunk /
    // normalize_audio_chunk
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_audio_chunk</c>. <paramref name="channelMajor"/> is <c>(2, samples)</c>
    /// (channel-major, matching Python's <c>.T</c>). Returns the concatenated-and-reshaped
    /// <c>(numChunks * 2, chunkSize)</c> array Python builds via
    /// <c>numpy.concatenate(chunks, axis=0).reshape((-1, chunk_size))</c> — row <c>2*c + ch</c>
    /// holds channel <c>ch</c> (0 or 1) of window <c>c</c>, exactly matching Python's row order
    /// (a plain `concatenate` of `(2, chunkSize)` slices interleaves the two channel rows per
    /// window before moving to the next window).
    /// </summary>
    internal static (float[,] Data, int PadSize) PrepareAudioChunk(float[,] channelMajor, int chunkSize, int audioTrimSize)
    {
        const float Int16Max = 32767f;

        var channelTotal = channelMajor.GetLength(0);
        var sampleTotal = channelMajor.GetLength(1);
        var audioStepSize = chunkSize - (2 * audioTrimSize);
        var audioPadSize = audioStepSize - (sampleTotal % audioStepSize);
        var audioChunkSize = sampleTotal + audioPadSize;
        var numChunks = audioChunkSize / audioStepSize;

        var padded = new float[channelTotal, audioTrimSize + sampleTotal + audioTrimSize + audioPadSize];

        for (var c = 0; c < channelTotal; c++)
        {
            for (var t = 0; t < sampleTotal; t++)
            {
                padded[c, audioTrimSize + t] = channelMajor[c, t] / Int16Max;
            }
        }

        var result = new float[numChunks * channelTotal, chunkSize];

        for (var chunkIndex = 0; chunkIndex < numChunks; chunkIndex++)
        {
            var offset = chunkIndex * audioStepSize;

            for (var c = 0; c < channelTotal; c++)
            {
                var row = (chunkIndex * channelTotal) + c;

                for (var t = 0; t < chunkSize; t++)
                {
                    result[row, t] = padded[c, offset + t];
                }
            }
        }

        return (result, audioPadSize);
    }

    /// <summary>
    /// Python: <c>decompose_audio_chunk</c>. Takes each row of <paramref name="rows"/>
    /// (shape <c>(numChunks * 2, chunkSize)</c>, row <c>2c+ch</c> = window c, channel ch — see
    /// <see cref="PrepareAudioChunk"/>) through a windowed STFT, then reshapes/interleaves the
    /// real/imaginary parts of the two channels into 4 "channels" (Re(ch0), Im(ch0), Re(ch1),
    /// Im(ch1)) and trims the frequency axis to <c>audioFrameTotal = 3072</c> bins (from the
    /// full <c>audioTrimSize + 1 = 3841</c>), matching the ONNX model's <c>(batch, 4, 3072,
    /// 256)</c> input exactly.
    /// </summary>
    internal static (float[] Data, int Batch, int Channel, int Freq, int Time) DecomposeAudioChunk(float[,] rows, int audioTrimSize)
    {
        const int audioFrameSize = 7680;
        const int audioFrameOverlap = 6656;
        const int audioFrameTotal = 3072;
        const int audioBinTotal = 256;
        const int audioChannelTotal = 4;

        var totalRows = rows.GetLength(0);
        var chunkSize = rows.GetLength(1);
        var numChunks = totalRows / 2;
        var window = HannWindow(audioFrameSize, sym: true);
        var windowSum = window.Sum();

        var data = new float[numChunks * audioChannelTotal * audioFrameTotal * audioBinTotal];

        for (var r = 0; r < totalRows; r++)
        {
            var signal = new double[chunkSize];

            for (var t = 0; t < chunkSize; t++)
            {
                signal[t] = rows[r, t];
            }

            var (real, imag, _, segCount) = Stft(signal, audioFrameSize, audioFrameOverlap, audioFrameSize, window);
            var chunkIndex = r / 2;
            var channelOrig = r % 2;
            var reChannel = channelOrig * 2;
            var imChannel = reChannel + 1;

            for (var f = 0; f < audioFrameTotal; f++)
            {
                for (var s = 0; s < segCount; s++)
                {
                    var reValue = (float)(real[f, s] * windowSum);
                    var imValue = (float)(imag[f, s] * windowSum);
                    data[FlatIndex4(chunkIndex, reChannel, f, s, audioChannelTotal, audioFrameTotal, audioBinTotal)] = reValue;
                    data[FlatIndex4(chunkIndex, imChannel, f, s, audioChannelTotal, audioFrameTotal, audioBinTotal)] = imValue;
                }
            }
        }

        return (data, numChunks, audioChannelTotal, audioFrameTotal, audioBinTotal);
    }

    private static int FlatIndex4(int b, int c, int h, int w, int channelTotal, int height, int width)
        => (((((b * channelTotal) + c) * height) + h) * width) + w;

    /// <summary>Python: <c>forward</c> — runs the ONNX model via the zero-copy <see cref="OrtValue"/> calling convention (DOTNET_PORT_PLAN.md §5.3, matching <c>FaceDetector.RunSession</c>'s established pattern).</summary>
    internal static (float[] Data, int Batch, int Channel, int Freq, int Time) Forward(InferenceSession voiceExtractorSession, (float[] Data, int Batch, int Channel, int Freq, int Time) input)
    {
        var shape = new long[] { input.Batch, input.Channel, input.Freq, input.Time };

        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(input.Data, shape);
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = voiceExtractorSession.Run(runOptions, inputs, voiceExtractorSession.OutputNames);

        var outputShape = results[0].GetTensorTypeAndShape().Shape;
        var outputData = results[0].GetTensorDataAsSpan<float>().ToArray();

        return (outputData, (int)outputShape[0], (int)outputShape[1], (int)outputShape[2], (int)outputShape[3]);
    }

    /// <summary>
    /// Python: <c>compose_audio_chunk</c>. Inverse of <see cref="DecomposeAudioChunk"/>: zero-pads
    /// the frequency axis back to <c>audioTrimSize + 1</c>, recombines the 4 "channels" back
    /// into 2 complex spectrograms, takes the inverse STFT of each, and returns
    /// <c>(numChunks * 2, chunkSize)</c> — the same row layout <see cref="PrepareAudioChunk"/> produced.
    /// </summary>
    internal static float[,] ComposeAudioChunk((float[] Data, int Batch, int Channel, int Freq, int Time) input, int audioTrimSize, int chunkSize)
    {
        const int audioFrameSize = 7680;
        const int audioFrameOverlap = 6656;

        var numChunks = input.Batch;
        var channelTotal = input.Channel;
        var freq = input.Freq;
        var time = input.Time;
        var fullFreq = audioTrimSize + 1;

        var window = HannWindow(audioFrameSize, sym: true);
        var windowSum = window.Sum();

        var rows = new float[numChunks * 2, chunkSize];

        for (var c = 0; c < numChunks; c++)
        {
            for (var channelOrig = 0; channelOrig < 2; channelOrig++)
            {
                var reChannel = channelOrig * 2;
                var imChannel = reChannel + 1;

                var real = new double[fullFreq, time];
                var imag = new double[fullFreq, time];

                for (var f = 0; f < freq; f++)
                {
                    for (var s = 0; s < time; s++)
                    {
                        real[f, s] = input.Data[FlatIndex4(c, reChannel, f, s, channelTotal, freq, time)];
                        imag[f, s] = input.Data[FlatIndex4(c, imChannel, f, s, channelTotal, freq, time)];
                    }
                }

                // Remaining frequency bins [freq, fullFreq) stay zero — matches Python's
                // numpy.pad(..., (0, audio_trim_size + 1 - audio_frame_total)) exactly.
                var timeSignal = Istft(real, imag, fullFreq, time, audioFrameSize, audioFrameOverlap, window);
                var rowIndex = (c * 2) + channelOrig;

                for (var t = 0; t < chunkSize; t++)
                {
                    rows[rowIndex, t] = (float)(timeSignal[t] / windowSum);
                }
            }
        }

        return rows;
    }

    /// <summary>Python: <c>normalize_audio_chunk</c>.</summary>
    internal static float[,] NormalizeAudioChunk(float[,] rows, int chunkSize, int audioTrimSize, int audioPadSize)
    {
        var totalRows = rows.GetLength(0);
        var numChunks = totalRows / 2;
        var audioStepSize = chunkSize - (2 * audioTrimSize);
        var totalLength = (numChunks * audioStepSize) - audioPadSize;

        var result = new float[totalLength, 2];

        for (var c = 0; c < numChunks; c++)
        {
            for (var channelOrig = 0; channelOrig < 2; channelOrig++)
            {
                var rowIndex = (c * 2) + channelOrig;

                for (var i = 0; i < audioStepSize; i++)
                {
                    var flatIndex = (c * audioStepSize) + i;

                    if (flatIndex >= totalLength)
                    {
                        continue;
                    }

                    result[flatIndex, channelOrig] = rows[rowIndex, audioTrimSize + i];
                }
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // Numeric primitives (scipy.signal ports — see the class remarks for why this duplicates
    // FaceFusion.Media.Audio's copy of the same primitives, and the port report for the
    // conventions each one reproduces, discovered/verified against real scipy 1.17.1).
    // -----------------------------------------------------------------

    /// <summary>Python: <c>scipy.signal.windows.hann(m, sym=...)</c>. <c>sym=true</c> is the only variant used in this file (the explicit <c>scipy.signal.windows.hann(audio_frame_size)</c> call — note the *default* <c>sym=True</c>, unlike <c>FaceFusion.Media.Audio</c>'s periodic-by-default <c>stft</c> window).</summary>
    internal static double[] HannWindow(int m, bool sym)
    {
        if (m <= 0)
        {
            return Array.Empty<double>();
        }

        if (m == 1)
        {
            return new[] { 1.0 };
        }

        var denominator = sym ? m - 1 : m;
        var window = new double[m];

        for (var i = 0; i < m; i++)
        {
            window[i] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / denominator));
        }

        return window;
    }

    /// <summary>Python: <c>scipy.signal.stft(x, nperseg=..., noverlap=..., window=...)</c> with this codebase's fixed defaults (<c>detrend=False</c>, <c>boundary='zeros'</c>, <c>padded=True</c>, <c>scaling='spectrum'</c>) — see <c>FaceFusion.Media.Audio.Stft</c>'s remarks for how each was verified.</summary>
    internal static (double[,] Real, double[,] Imag, int FreqBins, int SegCount) Stft(double[] x, int nperseg, int noverlap, int nfft, double[] window)
    {
        var nstep = nperseg - noverlap;
        var pad = nperseg / 2;

        var extended = new double[x.Length + (2 * pad)];
        Array.Copy(x, 0, extended, pad, x.Length);

        var nadd = PositiveMod(PositiveMod(-(extended.Length - nperseg), nstep), nperseg);
        var padded = nadd == 0 ? extended : extended.Concat(new double[nadd]).ToArray();

        var freqBins = (nfft / 2) + 1;
        var segCount = 1 + ((padded.Length - nperseg) / nstep);
        var winSum = window.Sum();
        var scale = 1.0 / winSum;

        var real = new double[freqBins, segCount];
        var imag = new double[freqBins, segCount];
        var segment = new double[nperseg];

        for (var s = 0; s < segCount; s++)
        {
            var offset = s * nstep;

            for (var i = 0; i < nperseg; i++)
            {
                segment[i] = padded[offset + i] * window[i];
            }

            var spectrum = Rfft(segment, nfft);

            for (var f = 0; f < freqBins; f++)
            {
                real[f, s] = spectrum[f].Real * scale;
                imag[f, s] = spectrum[f].Imaginary * scale;
            }
        }

        return (real, imag, freqBins, segCount);
    }

    /// <summary>Python: <c>scipy.signal.istft(Zxx, nperseg=..., noverlap=..., window=...)</c> with this codebase's fixed defaults (<c>input_onesided=True</c>, <c>boundary=True</c>, <c>scaling='spectrum'</c>). <paramref name="real"/>/<paramref name="imag"/> are <c>(freqBins, segCount)</c>. Verified against real scipy — see the port report.</summary>
    internal static double[] Istft(double[,] real, double[,] imag, int freqBins, int segCount, int nperseg, int noverlap, double[] window)
    {
        var nstep = nperseg - noverlap;
        var outputLength = nperseg + ((segCount - 1) * nstep);
        var winSum = window.Sum();

        var x = new double[outputLength];
        var norm = new double[outputLength];
        var half = new Complex[freqBins];

        for (var s = 0; s < segCount; s++)
        {
            for (var f = 0; f < freqBins; f++)
            {
                half[f] = new Complex(real[f, s], imag[f, s]);
            }

            var xsubs = Irfft(half, nperseg);
            var offset = s * nstep;

            for (var i = 0; i < nperseg; i++)
            {
                x[offset + i] += xsubs[i] * winSum * window[i];
                norm[offset + i] += window[i] * window[i];
            }
        }

        var pad = nperseg / 2;
        var result = new double[outputLength - (2 * pad)];

        for (var i = 0; i < result.Length; i++)
        {
            var normValue = norm[pad + i];
            result[i] = x[pad + i] / (normValue > 1e-10 ? normValue : 1.0);
        }

        return result;
    }

    private static int PositiveMod(int value, int modulus) => (((value % modulus) + modulus) % modulus);

    // ---- FFT (System.Numerics.Complex-based Bluestein-on-radix-2; see FaceFusion.Media.Audio's copy for the full remarks — duplicated here per this class's project-layering remarks) ----

    internal static Complex[] Fft(Complex[] x)
    {
        var n = x.Length;

        if (n == 0)
        {
            return Array.Empty<Complex>();
        }

        if ((n & (n - 1)) == 0)
        {
            var result = (Complex[])x.Clone();
            FftRadix2InPlace(result, inverse: false);
            return result;
        }

        return FftBluestein(x);
    }

    internal static Complex[] Ifft(Complex[] x)
    {
        var n = x.Length;

        if (n == 0)
        {
            return Array.Empty<Complex>();
        }

        if ((n & (n - 1)) == 0)
        {
            var result = (Complex[])x.Clone();
            FftRadix2InPlace(result, inverse: true);
            return result;
        }

        var conjugated = new Complex[n];

        for (var i = 0; i < n; i++)
        {
            conjugated[i] = Complex.Conjugate(x[i]);
        }

        var forward = Fft(conjugated);
        var result2 = new Complex[n];

        for (var i = 0; i < n; i++)
        {
            result2[i] = Complex.Conjugate(forward[i]) / n;
        }

        return result2;
    }

    internal static Complex[] Rfft(double[] x, int n)
    {
        var padded = new Complex[n];
        var copyLength = Math.Min(n, x.Length);

        for (var i = 0; i < copyLength; i++)
        {
            padded[i] = new Complex(x[i], 0);
        }

        var full = Fft(padded);
        var halfLength = (n / 2) + 1;
        var result = new Complex[halfLength];
        Array.Copy(full, result, halfLength);
        return result;
    }

    internal static double[] Irfft(Complex[] half, int n)
    {
        var full = new Complex[n];
        var halfLength = half.Length;

        for (var i = 0; i < halfLength && i < n; i++)
        {
            full[i] = half[i];
        }

        for (var k = halfLength; k < n; k++)
        {
            full[k] = Complex.Conjugate(full[n - k]);
        }

        var timeDomain = Ifft(full);
        var result = new double[n];

        for (var i = 0; i < n; i++)
        {
            result[i] = timeDomain[i].Real;
        }

        return result;
    }

    private static void FftRadix2InPlace(Complex[] a, bool inverse)
    {
        var n = a.Length;

        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;

            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }

            j ^= bit;

            if (i < j)
            {
                (a[i], a[j]) = (a[j], a[i]);
            }
        }

        for (var len = 2; len <= n; len <<= 1)
        {
            var angle = (2 * Math.PI / len) * (inverse ? 1 : -1);
            var wlen = new Complex(Math.Cos(angle), Math.Sin(angle));

            for (var i = 0; i < n; i += len)
            {
                var w = Complex.One;

                for (var j = 0; j < len / 2; j++)
                {
                    var u = a[i + j];
                    var v = a[i + j + (len / 2)] * w;
                    a[i + j] = u + v;
                    a[i + j + (len / 2)] = u - v;
                    w *= wlen;
                }
            }
        }

        if (inverse)
        {
            for (var i = 0; i < n; i++)
            {
                a[i] /= n;
            }
        }
    }

    private static Complex[] FftBluestein(Complex[] x)
    {
        var n = x.Length;
        var m = 1;

        while (m < (2 * n) + 1)
        {
            m <<= 1;
        }

        var chirp = new Complex[n];

        for (var i = 0; i < n; i++)
        {
            var iSquared = (double)((long)i * i);
            var angle = -Math.PI * iSquared / n;
            chirp[i] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }

        var a = new Complex[m];

        for (var i = 0; i < n; i++)
        {
            a[i] = x[i] * chirp[i];
        }

        var b = new Complex[m];
        b[0] = Complex.Conjugate(chirp[0]);

        for (var i = 1; i < n; i++)
        {
            var conjugated = Complex.Conjugate(chirp[i]);
            b[i] = conjugated;
            b[m - i] = conjugated;
        }

        var fa = (Complex[])a.Clone();
        FftRadix2InPlace(fa, inverse: false);
        var fb = (Complex[])b.Clone();
        FftRadix2InPlace(fb, inverse: false);

        var fc = new Complex[m];

        for (var i = 0; i < m; i++)
        {
            fc[i] = fa[i] * fb[i];
        }

        FftRadix2InPlace(fc, inverse: true);

        var result = new Complex[n];

        for (var i = 0; i < n; i++)
        {
            result[i] = fc[i] * chirp[i];
        }

        return result;
    }
}
