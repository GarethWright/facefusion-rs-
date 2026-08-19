using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using FaceFusion.Core;

// Grants the parity-test project access to the internal numeric primitives (Lfilter,
// Resample, HannWindow, TriangWindow, Stft, Fft, Rfft, Irfft) so they can be verified
// directly against real scipy.signal ground truth without going through the full audio
// pipeline. A C# assembly attribute, not a project-file edit (see the existing one for
// FaceFusion.UnitTests in Ffmpeg.cs — InternalsVisibleTo allows multiple instances).
[assembly: InternalsVisibleTo("FaceFusion.ParityTests")]
[assembly: InternalsVisibleTo("FaceFusion.UnitTests")]

namespace FaceFusion.Media;

/// <summary>
/// Port of <c>facefusion/audio.py</c> — reads an audio file via ffmpeg, builds a mel
/// spectrogram, and slices it into per-video-frame windows for the lip syncer.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Python's <c>read_voice</c>/
/// <c>read_static_voice</c> call <c>facefusion.voice_extractor.batch_extract_voice</c>
/// directly (a real Python import). In this port <c>facefusion/audio.py</c> lands in
/// <c>FaceFusion.Media</c> and <c>facefusion/voice_extractor.py</c> lands in
/// <c>FaceFusion.Processors</c> (per the assignment table) — and per PORT_CONVENTIONS.md,
/// project references are not something this phase may add (".csproj files" are off
/// limits), while the natural dependency direction in this solution has the higher-level
/// <c>FaceFusion.Processors</c> depend on the lower-level <c>FaceFusion.Media</c>, never the
/// reverse. So <see cref="ReadVoice"/>/<see cref="ReadStaticVoice"/>/<see cref="GetVoiceFrame"/>
/// take the voice-extraction step as an explicit <see cref="ExtractVoiceDelegate"/> parameter
/// instead of importing it — <c>FaceFusion.Processors.VoiceExtractor.BatchExtractVoice</c> is
/// the real implementation a caller wires in, and this is consistent with how every other
/// state_manager-backed call in this port becomes an explicit parameter.
/// </para>
///
/// <para>
/// <b>Precision (float64 throughout, matching Python exactly for the <see cref="ReadAudio"/>
/// path).</b> Python's <c>numpy.frombuffer(..., dtype=numpy.int16)</c> promotes to float64 the
/// moment it hits <c>numpy.mean</c>/division by a Python float, and <c>scipy.signal.lfilter</c>
/// upcasts to float64 unconditionally (its <c>[1.0, -0.97]</c>/<c>[1.0]</c> coefficients are
/// Python floats, i.e. float64, and numpy always promotes mixed-precision array arithmetic to
/// the wider type) — verified empirically against real scipy 1.17.1 (see the port report).
/// So every stage from <see cref="PrepareAudio"/> onward is float64 (<see cref="double"/>) in
/// this port, matching <c>AudioFrame</c>'s actual runtime dtype exactly, not just its nominal
/// <c>NDArray[Any]</c> type alias. The one path where this is an approximation rather than a
/// bit-exact match is <see cref="ReadVoice"/>: Python's <c>batch_extract_voice</c> output is
/// float32 (it feeds an ONNX model), and <c>prepare_voice</c>'s <c>resample</c>/<c>mean</c>/
/// normalize steps run at that float32 precision before <c>lfilter</c> upcasts the final
/// result to float64 — this port widens the float32 voice array to double at the
/// <see cref="ExtractVoiceDelegate"/> boundary (an exact, lossless upcast) and then runs
/// <see cref="PrepareVoice"/> entirely in double, so intermediate rounding differs from
/// Python's float32 arithmetic by up to float32 epsilon (~1e-7 relative) rather than being
/// bit-identical. This is deliberate: duplicating every primitive at both precisions for one
/// intermediate stage was not judged worth the risk of the two copies drifting apart, and the
/// deviation is well inside PARITY_HARNESS.md's "managed float math ... a real epsilon
/// belongs here" guidance. See AudioParityTests for the measured divergence.
/// </para>
///
/// <para>
/// <b>Representation.</b> <c>Audio</c>/<c>AudioChunk</c> (2-channel, interleaved by column)
/// are <c>double[,]</c> of shape <c>(samples, channels)</c>, matching numpy's row-major
/// layout exactly (row = sample, column = channel). <c>AudioFrame</c>/<c>Spectrogram</c> are
/// <c>double[,]</c> of shape <c>(melFilterTotal, columns)</c> = <c>(80, columns)</c>,
/// matching Python's <c>(80, N)</c> layout (row = mel bin, column = time step) exactly — no
/// transpose anywhere in this file relative to Python.
/// </para>
///
/// <para>
/// <b>scipy.signal conventions reproduced here (not the textbook definitions — see the port
/// report for how each was discovered/verified against real scipy 1.17.1).</b>
/// <list type="bullet">
/// <item><description><see cref="Stft"/>: default window is the *periodic* Hann
/// (<c>scipy.signal.stft</c>'s <c>window='hann_periodic'</c> default — <c>get_window</c>'s
/// <c>fftbins=True</c>, i.e. <c>sym=False</c>, which is a different window than
/// <c>scipy.signal.windows.hann(M)</c>'s own <c>sym=True</c> default). <c>detrend=False</c> is
/// <c>stft</c>'s own default (not the <c>'constant'</c> default of the internal
/// <c>_spectral_helper</c> it wraps — verified by reading <c>stft</c>'s actual signature, not
/// assumed from the helper's docstring). <c>boundary='zeros'</c> zero-extends the signal by
/// <c>nperseg // 2</c> samples at each end before segmenting (not reflection/even/odd
/// extension). <c>padded=True</c> then zero-pads the end to an integer number of hop-sized
/// segments. <c>scaling='spectrum'</c> divides every segment's FFT by <c>window.sum()</c>
/// (verified: <c>_spectral_helper</c> computes <c>scale = 1/window.sum()**2</c> for
/// <c>scaling='spectrum'</c>, then <c>stft</c>'s <c>mode='stft'</c> branch takes
/// <c>sqrt(scale)</c>, i.e. <c>1/window.sum()</c> — not <c>1/sqrt(N)</c> or any other textbook
/// normalisation).</description></item>
/// <item><description><see cref="Resample"/>: FFT-based (via one-sided real FFT since the
/// input is always real here), not polyphase. Trims or zero-pads the frequency-domain
/// representation to <c>min(num, n)//2 + 1</c> bins, and — only when the shorter of the two
/// lengths is even and resampling actually changes the length — rescales the single unpaired
/// Nyquist bin by 2x (downsampling) or 0.5x (upsampling) rather than leaving it as-is; this
/// bin-pair handling is the part a naive rfft/irfft reimplementation gets wrong, and is
/// reproduced exactly per scipy 1.17.1's actual algorithm (read from source, then verified
/// numerically against real <c>scipy.signal.resample</c> output for several odd/even,
/// up/down combinations — see the port report).</description></item>
/// </list>
/// </para>
/// </summary>
public static class Audio
{
    private const int MelFilterTotal = 80;
    private const int AudioFrameStepSize = 16;

    // ---------------------------------------------------------------------
    // Caching (Python: @lru_cache(maxsize = 64)). Per PORT_CONVENTIONS.md rule 5 this is not
    // "global mutable state" in the state_manager sense — it is pure memoisation over
    // arguments the caller controls, and the existing FaceFusion.Vision.Vision.ReadStaticImage
    // established the same bounded-FIFO-cache-over-a-static-class pattern for the equivalent
    // Python lru_cache in that module. Cache eviction is a simple bounded FIFO (not true LRU
    // recency order), same documented simplification as ReadStaticImage.
    // ---------------------------------------------------------------------

    private const int CacheCapacity = 64;
    private static readonly object AudioCacheLock = new();
    private static readonly Dictionary<(string, double), IReadOnlyList<double[,]>?> AudioCache = new();
    private static readonly Queue<(string, double)> AudioCacheOrder = new();

    private static readonly object VoiceCacheLock = new();
    private static readonly Dictionary<(string, double), IReadOnlyList<double[,]>?> VoiceCache = new();
    private static readonly Queue<(string, double)> VoiceCacheOrder = new();

    /// <summary>
    /// Stands in for Python's <c>facefusion.voice_extractor.batch_extract_voice(audio,
    /// chunk_size, step_size)</c> — see the class remarks for why this is a parameter rather
    /// than a direct call. <paramref name="audio"/> is the raw stereo signal (shape
    /// <c>(samples, 2)</c>, values in the int16 range but represented as double); the returned
    /// array has the same shape.
    /// </summary>
    public delegate double[,] ExtractVoiceDelegate(double[,] audio, int chunkSize, int stepSize);

    // -----------------------------------------------------------------
    // read_static_audio / read_audio / read_static_voice / read_voice
    // -----------------------------------------------------------------

    /// <summary>Python: <c>read_static_audio</c> (<c>@lru_cache(maxsize = 64)</c>).</summary>
    public static IReadOnlyList<double[,]>? ReadStaticAudio(string audioPath, double fps)
    {
        var key = (audioPath, fps);

        lock (AudioCacheLock)
        {
            if (AudioCache.TryGetValue(key, out var cached))
            {
                return cached;
            }
        }

        var result = ReadAudio(audioPath, fps);

        lock (AudioCacheLock)
        {
            if (!AudioCache.ContainsKey(key))
            {
                if (AudioCacheOrder.Count >= CacheCapacity)
                {
                    var oldestKey = AudioCacheOrder.Dequeue();
                    AudioCache.Remove(oldestKey);
                }

                AudioCache[key] = result;
                AudioCacheOrder.Enqueue(key);
            }
        }

        return result;
    }

    /// <summary>Python: <c>read_audio</c>.</summary>
    public static IReadOnlyList<double[,]>? ReadAudio(string audioPath, double fps)
    {
        const int audioSampleRate = 48000;
        const int audioSampleSize = 16;
        const int audioChannelTotal = 2;

        if (!FileSystem.IsAudio(audioPath))
        {
            return null;
        }

        var audioBuffer = Ffmpeg.ReadAudioBuffer(audioPath, audioSampleRate, audioSampleSize, audioChannelTotal);

        if (audioBuffer is null)
        {
            return null;
        }

        var audio = BufferToStereo(audioBuffer);
        var prepared = PrepareAudio(audio);
        var spectrogram = CreateSpectrogram(prepared);
        return ExtractAudioFrames(spectrogram, fps);
    }

    /// <summary>Python: <c>read_static_voice</c> (<c>@lru_cache(maxsize = 64)</c>).</summary>
    public static IReadOnlyList<double[,]>? ReadStaticVoice(string audioPath, double fps, ExtractVoiceDelegate extractVoice)
    {
        var key = (audioPath, fps);

        lock (VoiceCacheLock)
        {
            if (VoiceCache.TryGetValue(key, out var cached))
            {
                return cached;
            }
        }

        var result = ReadVoice(audioPath, fps, extractVoice);

        lock (VoiceCacheLock)
        {
            if (!VoiceCache.ContainsKey(key))
            {
                if (VoiceCacheOrder.Count >= CacheCapacity)
                {
                    var oldestKey = VoiceCacheOrder.Dequeue();
                    VoiceCache.Remove(oldestKey);
                }

                VoiceCache[key] = result;
                VoiceCacheOrder.Enqueue(key);
            }
        }

        return result;
    }

    /// <summary>Python: <c>read_voice</c>.</summary>
    public static IReadOnlyList<double[,]>? ReadVoice(string audioPath, double fps, ExtractVoiceDelegate extractVoice)
    {
        const int voiceSampleRate = 48000;
        const int voiceSampleSize = 16;
        const int voiceChannelTotal = 2;
        const int voiceChunkSize = 240 * 1024;
        const int voiceStepSize = 180 * 1024;

        if (!FileSystem.IsAudio(audioPath))
        {
            return null;
        }

        var audioBuffer = Ffmpeg.ReadAudioBuffer(audioPath, voiceSampleRate, voiceSampleSize, voiceChannelTotal);

        if (audioBuffer is null)
        {
            return null;
        }

        var audio = BufferToStereo(audioBuffer);
        var voice = extractVoice(audio, voiceChunkSize, voiceStepSize);
        var prepared = PrepareVoice(voice);
        var spectrogram = CreateSpectrogram(prepared);
        return ExtractAudioFrames(spectrogram, fps);
    }

    /// <summary>Python: <c>get_audio_frame</c>.</summary>
    public static double[,]? GetAudioFrame(string audioPath, double fps, int frameNumber = 0)
    {
        if (!FileSystem.IsAudio(audioPath))
        {
            return null;
        }

        var audioFrames = ReadStaticAudio(audioPath, fps);

        if (audioFrames is not null && frameNumber >= 0 && frameNumber < audioFrames.Count)
        {
            return audioFrames[frameNumber];
        }

        return null;
    }

    /// <summary>Python: <c>get_voice_frame</c>.</summary>
    public static double[,]? GetVoiceFrame(string audioPath, double fps, ExtractVoiceDelegate extractVoice, int frameNumber = 0)
    {
        if (!FileSystem.IsAudio(audioPath))
        {
            return null;
        }

        var voiceFrames = ReadStaticVoice(audioPath, fps, extractVoice);

        if (voiceFrames is not null && frameNumber >= 0 && frameNumber < voiceFrames.Count)
        {
            return voiceFrames[frameNumber];
        }

        return null;
    }

    /// <summary>
    /// Python: <c>create_empty_audio_frame</c> (<c>numpy.zeros((80, 16)).astype(numpy.int16)</c>).
    /// The Python source casts to int16 here (unlike every other <c>AudioFrame</c> in this
    /// file, which is float64) — reproduced faithfully per PORT_CONVENTIONS.md rule 1 even
    /// though it looks inconsistent, since an all-zero array is numerically identical in
    /// either dtype and no arithmetic downstream of this function depends on the distinction.
    /// </summary>
    public static double[,] CreateEmptyAudioFrame() => new double[MelFilterTotal, AudioFrameStepSize];

    /// <summary>Python: <c>extract_audio_frames</c>.</summary>
    public static List<double[,]> ExtractAudioFrames(double[,] spectrogram, double fps)
    {
        var rows = spectrogram.GetLength(0);
        var cols = spectrogram.GetLength(1);
        var step = MelFilterTotal / fps;
        var audioFrames = new List<double[,]>();

        // Python: `numpy.arange(0, cols, step).astype(numpy.int16)`. numpy.arange's length is
        // ceil((stop - start) / step); each sample is then truncated toward zero by the int16
        // cast (not rounded) — both reproduced explicitly here rather than relying on a
        // C# Range/loop condition that would silently do something else at the boundary.
        var count = (int)Math.Ceiling(cols / step);

        for (var i = 0; i < count; i++)
        {
            var raw = i * step;

            if (raw >= cols)
            {
                break;
            }

            var index = (int)raw; // truncation toward zero, matching numpy's float -> int16 cast (raw is always >= 0 here).

            if (index < AudioFrameStepSize)
            {
                continue;
            }

            var start = Math.Max(0, index - AudioFrameStepSize);
            var frame = new double[rows, index - start];

            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < index - start; c++)
                {
                    frame[r, c] = spectrogram[r, start + c];
                }
            }

            audioFrames.Add(frame);
        }

        return audioFrames;
    }

    // -----------------------------------------------------------------
    // prepare_audio / prepare_voice / mel filter bank / spectrogram
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_audio</c>. Always takes the 2-channel <c>(samples, 2)</c> shape used
    /// at every call site in this file (Python's <c>audio.ndim &gt; 1</c> branch); the
    /// 1-D-input branch is never reached by any caller here and is not ported.
    /// </summary>
    internal static double[] PrepareAudio(double[,] audio)
    {
        var sampleTotal = audio.GetLength(0);
        var channelTotal = audio.GetLength(1);
        var mono = new double[sampleTotal];

        for (var i = 0; i < sampleTotal; i++)
        {
            double sum = 0;

            for (var c = 0; c < channelTotal; c++)
            {
                sum += audio[i, c];
            }

            mono[i] = sum / channelTotal;
        }

        var maxAbs = 0.0;

        for (var i = 0; i < sampleTotal; i++)
        {
            var abs = Math.Abs(mono[i]);

            if (abs > maxAbs)
            {
                maxAbs = abs;
            }
        }

        var normalized = new double[sampleTotal];

        for (var i = 0; i < sampleTotal; i++)
        {
            normalized[i] = mono[i] / maxAbs;
        }

        return Lfilter(new[] { 1.0, -0.97 }, new[] { 1.0 }, normalized);
    }

    /// <summary>Python: <c>prepare_voice</c>.</summary>
    internal static double[] PrepareVoice(double[,] audio)
    {
        const int audioSampleRate = 48000;
        const int audioResampleRate = 16000;

        var sampleTotal = audio.GetLength(0);
        var audioResampleFactor = (int)Math.Round(sampleTotal * (double)audioResampleRate / audioSampleRate, MidpointRounding.ToEven);
        var resampled = Resample2D(audio, audioResampleFactor);
        return PrepareAudio(resampled);
    }

    /// <summary>Python: <c>convert_hertz_to_mel</c>.</summary>
    public static double ConvertHertzToMel(double hertz) => 2595 * Math.Log10(1 + (hertz / 700));

    /// <summary>Python: <c>convert_mel_to_hertz</c>.</summary>
    public static double ConvertMelToHertz(double mel) => 700 * (Math.Pow(10, mel / 2595) - 1);

    /// <summary>Python: <c>create_mel_filter_bank</c>.</summary>
    internal static double[,] CreateMelFilterBank()
    {
        const double audioSampleRate = 16000;
        const double audioFrequencyMin = 55.0;
        const double audioFrequencyMax = 7600.0;
        const int melFilterTotal = MelFilterTotal;
        const int melBinTotal = 800;

        var melFilterBank = new double[melFilterTotal, (melBinTotal / 2) + 1];
        var melFrequencyRange = Linspace(ConvertHertzToMel(audioFrequencyMin), ConvertHertzToMel(audioFrequencyMax), melFilterTotal + 2);
        var indices = new int[melFrequencyRange.Length];

        for (var i = 0; i < melFrequencyRange.Length; i++)
        {
            // Python: `numpy.floor(...).astype(numpy.int16)` — floor first (values are always
            // non-negative here), then the int16 cast is a no-op at these magnitudes.
            indices[i] = (int)Math.Floor((melBinTotal + 1) * ConvertMelToHertz(melFrequencyRange[i]) / audioSampleRate);
        }

        for (var index = 0; index < melFilterTotal; index++)
        {
            var start = indices[index];
            var end = indices[index + 1];
            var triangle = TriangWindow(end - start);

            for (var j = 0; j < triangle.Length; j++)
            {
                melFilterBank[index, start + j] = triangle[j];
            }
        }

        return melFilterBank;
    }

    /// <summary>Python: <c>create_spectrogram</c>.</summary>
    internal static double[,] CreateSpectrogram(double[] audio)
    {
        const int melBinTotal = 800;
        const int melBinOverlap = 600;

        var melFilterBank = CreateMelFilterBank();
        var window = HannWindow(melBinTotal, periodic: true); // scipy.signal.stft's default 'hann_periodic' window.
        var (real, imag, freqBins, segCount) = Stft(audio, melBinTotal, melBinOverlap, melBinTotal, window);

        var magnitude = new double[freqBins, segCount];

        for (var f = 0; f < freqBins; f++)
        {
            for (var s = 0; s < segCount; s++)
            {
                magnitude[f, s] = Math.Sqrt((real[f, s] * real[f, s]) + (imag[f, s] * imag[f, s]));
            }
        }

        var filterTotal = melFilterBank.GetLength(0);
        var spectrogram = new double[filterTotal, segCount];

        for (var row = 0; row < filterTotal; row++)
        {
            for (var s = 0; s < segCount; s++)
            {
                double sum = 0;

                for (var f = 0; f < freqBins; f++)
                {
                    sum += melFilterBank[row, f] * magnitude[f, s];
                }

                spectrogram[row, s] = sum;
            }
        }

        return spectrogram;
    }

    // -----------------------------------------------------------------
    // Buffer plumbing
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>numpy.frombuffer(audio_buffer, dtype = numpy.int16).reshape(-1, 2)</c>. The
    /// int16 -&gt; double widening is exact (every int16 value is exactly representable in
    /// double), so this is bit-identical to Python's array before any arithmetic runs.
    /// </summary>
    internal static double[,] BufferToStereo(byte[] audioBuffer)
    {
        const int channelTotal = 2;
        var sampleTotal = audioBuffer.Length / 2 / channelTotal;
        var result = new double[sampleTotal, channelTotal];
        var offset = 0;

        for (var i = 0; i < sampleTotal; i++)
        {
            for (var c = 0; c < channelTotal; c++)
            {
                var value = (short)(audioBuffer[offset] | (audioBuffer[offset + 1] << 8));
                result[i, c] = value;
                offset += 2;
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // Numeric primitives (scipy.signal ports — see class remarks for the conventions
    // reproduced here, verified against real scipy 1.17.1; see the port report for how each
    // was discovered).
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>scipy.signal.lfilter(b, a, x)</c> — Direct Form II Transposed, matching
    /// scipy's own implementation exactly (initial conditions zero, as scipy's default
    /// <c>zi=None</c>). Only ever called in this file with <c>b = [1.0, -0.97]</c>,
    /// <c>a = [1.0]</c> (pre-emphasis: <c>y[n] = x[n] - 0.97 * x[n-1]</c>), but implemented
    /// generally and verified against that exact coefficient pair plus scipy ground truth.
    /// </summary>
    internal static double[] Lfilter(double[] b, double[] a, double[] x)
    {
        if (a.Length == 0 || a[0] == 0)
        {
            throw new ArgumentException("a[0] must be non-zero.", nameof(a));
        }

        var a0 = a[0];
        var order = Math.Max(b.Length, a.Length) - 1;
        var bn = new double[order + 1];
        var an = new double[order + 1];

        for (var i = 0; i < b.Length; i++)
        {
            bn[i] = b[i] / a0;
        }

        for (var i = 0; i < a.Length; i++)
        {
            an[i] = a[i] / a0;
        }

        var z = new double[order];
        var y = new double[x.Length];

        for (var n = 0; n < x.Length; n++)
        {
            var xn = x[n];
            var yn = (bn[0] * xn) + (order > 0 ? z[0] : 0.0);

            for (var i = 1; i < order; i++)
            {
                z[i - 1] = (bn[i] * xn) + z[i] - (an[i] * yn);
            }

            if (order > 0)
            {
                z[order - 1] = (bn[order] * xn) - (an[order] * yn);
            }

            y[n] = yn;
        }

        return y;
    }

    /// <summary>
    /// Python: <c>scipy.signal.resample(x, num)</c> (real input, <c>window=None</c>,
    /// <c>domain='time'</c> — the only combination used in this file). See the class remarks
    /// for the unpaired-Nyquist-bin handling this reproduces.
    /// </summary>
    internal static double[] Resample(double[] x, int num)
    {
        var n = x.Length;

        if (num <= 0)
        {
            return Array.Empty<double>();
        }

        var m = Math.Min(num, n);
        var m2 = (m / 2) + 1;
        var sFac = (double)n / num;

        var spectrum = Rfft(x, n);
        var trimmed = new Complex[m2];
        Array.Copy(spectrum, trimmed, m2);

        if (m % 2 == 0 && num != n)
        {
            trimmed[m / 2] *= num < n ? 2.0 : 0.5;
        }

        for (var i = 0; i < trimmed.Length; i++)
        {
            trimmed[i] /= sFac;
        }

        return Irfft(trimmed, num);
    }

    /// <summary>Per-channel <see cref="Resample"/> over a <c>(samples, channels)</c> array (Python's default <c>axis=0</c>).</summary>
    internal static double[,] Resample2D(double[,] x, int num)
    {
        var sampleTotal = x.GetLength(0);
        var channelTotal = x.GetLength(1);
        var result = new double[num, channelTotal];

        for (var c = 0; c < channelTotal; c++)
        {
            var channel = new double[sampleTotal];

            for (var i = 0; i < sampleTotal; i++)
            {
                channel[i] = x[i, c];
            }

            var resampled = Resample(channel, num);

            for (var i = 0; i < num; i++)
            {
                result[i, c] = resampled[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Python: <c>scipy.signal.windows.hann(m, sym=...)</c> / the periodic Hann implicitly
    /// used by <c>scipy.signal.stft</c>'s default <c>window='hann_periodic'</c>. Verified
    /// against real scipy for both parities — see the port report.
    /// </summary>
    internal static double[] HannWindow(int m, bool periodic)
    {
        if (m <= 0)
        {
            return Array.Empty<double>();
        }

        if (m == 1)
        {
            return new[] { 1.0 };
        }

        var denominator = periodic ? m : m - 1;
        var window = new double[m];

        for (var i = 0; i < m; i++)
        {
            window[i] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / denominator));
        }

        return window;
    }

    /// <summary>
    /// Python: <c>scipy.signal.windows.triang(m)</c> (default <c>sym=True</c>, the only
    /// variant used in this file). Verified against real scipy for both odd and even
    /// <paramref name="m"/> — see the port report.
    /// </summary>
    internal static double[] TriangWindow(int m)
    {
        if (m <= 0)
        {
            return Array.Empty<double>();
        }

        if (m == 1)
        {
            return new[] { 1.0 };
        }

        var window = new double[m];

        if (m % 2 == 0)
        {
            var half = m / 2;

            for (var n = 1; n <= half; n++)
            {
                var value = ((2.0 * n) - 1.0) / m;
                window[n - 1] = value;
                window[m - n] = value;
            }
        }
        else
        {
            var half = (m + 1) / 2;

            for (var n = 1; n <= half; n++)
            {
                var value = 2.0 * n / (m + 1.0);
                window[n - 1] = value;
                window[m - n] = value;
            }
        }

        return window;
    }

    /// <summary>
    /// Python: <c>scipy.signal.stft(x, nperseg=..., noverlap=..., nfft=..., window=...)</c>
    /// with the fixed defaults this codebase always uses: <c>detrend=False</c>,
    /// <c>boundary='zeros'</c>, <c>padded=True</c>, <c>return_onesided=True</c>,
    /// <c>scaling='spectrum'</c> — see the class remarks for how each of those was verified.
    /// Returns the real/imaginary parts separately (rather than a complex array) purely so
    /// callers that only need magnitude, like <see cref="CreateSpectrogram"/>, do not need
    /// <see cref="System.Numerics.Complex"/> at their call site; <see cref="Fft"/>/
    /// <see cref="Rfft"/>/<see cref="Irfft"/> underneath still operate on <see cref="Complex"/>.
    /// </summary>
    internal static (double[,] Real, double[,] Imag, int FreqBins, int SegCount) Stft(double[] x, int nperseg, int noverlap, int nfft, double[] window)
    {
        var nstep = nperseg - noverlap;
        var pad = nperseg / 2;

        // boundary='zeros': zero_ext(x, nperseg // 2).
        var extended = new double[x.Length + (2 * pad)];
        Array.Copy(x, 0, extended, pad, x.Length);

        // padded=True: pad to an integer number of hop-sized segments.
        var nadd = PositiveMod(PositiveMod(-(extended.Length - nperseg), nstep), nperseg);
        var padded = nadd == 0 ? extended : extended.Concat(new double[nadd]).ToArray();

        var freqBins = (nfft / 2) + 1;
        var segCount = 1 + ((padded.Length - nperseg) / nstep);
        double winSum = 0;

        foreach (var w in window)
        {
            winSum += w;
        }

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

    /// <summary>Python's <c>%</c> operator (always non-negative for a positive divisor, unlike C#'s <c>%</c>).</summary>
    private static int PositiveMod(int value, int modulus) => (((value % modulus) + modulus) % modulus);

    // -----------------------------------------------------------------
    // FFT (System.Numerics.Complex-based; no third-party FFT library per the port
    // constraints). Sizes used in this file (800, and 7680/512 in FaceFusion.Processors'
    // VoiceExtractor) are not powers of two, so an arbitrary-length transform is genuinely
    // required, not merely for generality — implemented as Bluestein's algorithm (chirp-z
    // transform) layered on an iterative radix-2 Cooley-Tukey FFT, verified against
    // numpy.fft for both the power-of-two fast path and non-power-of-two sizes (see the
    // port report).
    // -----------------------------------------------------------------

    /// <summary>Forward DFT of arbitrary length, dispatching to a radix-2 fast path when <paramref name="x"/>'s length is a power of two.</summary>
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

    /// <summary>Inverse DFT of arbitrary length (includes the <c>1/n</c> normalisation).</summary>
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

        // Standard conjugate trick for a generic-length inverse built on the forward transform.
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

    /// <summary>
    /// Python: <c>numpy.fft.rfft(x, n)</c> — zero-pads or truncates <paramref name="x"/> to
    /// length <paramref name="n"/>, then returns only the first <c>n / 2 + 1</c> bins (the
    /// non-redundant half of a real signal's spectrum).
    /// </summary>
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

    /// <summary>
    /// Python: <c>numpy.fft.irfft(x, n)</c> — reconstructs the full length-<paramref name="n"/>
    /// spectrum from its non-redundant half <paramref name="half"/> via conjugate (Hermitian)
    /// symmetry, then returns the real part of the inverse DFT.
    /// </summary>
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
            // e^{-i * pi * i^2 / n}. i*i can exceed 2^31 only for n far beyond any size used
            // in this codebase; using long avoids overflow regardless.
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

    // -----------------------------------------------------------------
    // Small numeric helper (double precision — the FaceFusion.Tensors.NumPy layer is
    // float32-only and this file's pipeline is deliberately float64 throughout per the
    // class remarks, so it is not reused here).
    // -----------------------------------------------------------------

    private static double[] Linspace(double start, double stop, int count)
    {
        var result = new double[count];

        if (count == 0)
        {
            return result;
        }

        if (count == 1)
        {
            result[0] = start;
            return result;
        }

        var step = (stop - start) / (count - 1);

        for (var i = 0; i < count; i++)
        {
            result[i] = start + (step * i);
        }

        result[count - 1] = stop;
        return result;
    }
}
