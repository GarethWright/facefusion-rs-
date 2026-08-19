using FaceFusion.Processors;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Unit-level checks for <see cref="VoiceExtractor"/> (Python: <c>facefusion/voice_extractor.py</c>).
///
/// <para>
/// <b>The heavy lifting already lives elsewhere.</b> There is no <c>tests/test_voice_extractor.py</c>
/// upstream, but <c>tests/FaceFusion.ParityTests/AudioParityTests.cs</c> already carries real
/// ground truth (real scipy 1.17.1 plus the real <c>kim_vocal_2</c> ONNX model) for every
/// numerically interesting stage of this pipeline — <c>PrepareAudioChunk</c>,
/// <c>DecomposeAudioChunk</c>, <c>Forward</c>, <c>ComposeAudioChunk</c>,
/// <c>NormalizeAudioChunk</c>, <c>ExtractVoice</c>, and <c>BatchExtractVoice</c> end to end — so
/// that file is this module's real parity coverage; see its own class remarks for tolerances
/// and gating. This file adds a small amount of additional, narrower ground truth
/// (<see cref="PrepareAudioChunkPaddingAndTilingMatchesPython"/>, computed from a tiny synthetic
/// input that needs only numpy, no scipy/ONNX) plus a few pure-logic/property checks
/// (<see cref="TransposeToChannelMajorTransposesAndWidensToFloat"/>,
/// <see cref="CreateStaticModelSetCoversEveryModel"/>,
/// <see cref="PreCheckReturnsFalseWhenModelFilesAreMissing"/>) that
/// <c>AudioParityTests.cs</c> does not already cover.
/// </para>
/// </summary>
public sealed class VoiceExtractorTests
{
    // -----------------------------------------------------------------
    // prepare_audio_chunk — padding/tiling arithmetic
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// channel_major = numpy.array([[100,200,300,400,500],[-100,-200,-300,-400,-500]], float32)
    /// prepare_audio_chunk(channel_major, chunk_size=6, audio_trim_size=1)
    /// </code>
    /// (Python's own <c>prepare_audio_chunk</c>, reproduced inline with numpy since it needs no
    /// scipy). 2 channels x 5 samples, <c>chunk_size = 6</c>, <c>audio_trim_size = 1</c> (so
    /// <c>audio_step_size = 4</c>) — small enough to hand-verify the pad amount (3), the
    /// int16-normalisation (<c>/32767</c>), and the sliding-window tiling (2 tiles of 4 samples
    /// each, stacked into 4 rows) all in one shot.
    /// </summary>
    [Fact]
    public void PrepareAudioChunkPaddingAndTilingMatchesPython()
    {
        var channelMajor = new float[2, 5]
        {
            { 100, 200, 300, 400, 500 },
            { -100, -200, -300, -400, -500 },
        };

        var (data, padSize) = VoiceExtractor.PrepareAudioChunk(channelMajor, chunkSize: 6, audioTrimSize: 1);

        Assert.Equal(3, padSize);
        Assert.Equal(4, data.GetLength(0));
        Assert.Equal(6, data.GetLength(1));

        float[,] expected =
        {
            { 0.0f, 0.003052f, 0.006104f, 0.009156f, 0.012207f, 0.015259f },
            { 0.0f, -0.003052f, -0.006104f, -0.009156f, -0.012207f, -0.015259f },
            { 0.012207f, 0.015259f, 0.0f, 0.0f, 0.0f, 0.0f },
            { -0.012207f, -0.015259f, 0.0f, 0.0f, 0.0f, 0.0f },
        };

        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < 6; c++)
            {
                Assert.True(
                    Math.Abs(data[r, c] - expected[r, c]) <= 1e-5f,
                    $"[{r},{c}] = {data[r, c]}, expected {expected[r, c]}");
            }
        }
    }

    // -----------------------------------------------------------------
    // Pure logic / property checks not already covered by AudioParityTests.cs
    // -----------------------------------------------------------------

    /// <summary>Python: <c>temp_audio_chunk.T</c> plus the eventual <c>.astype(float32)</c> —
    /// (samples, 2) in, (2, samples) out, values widened losslessly for the small integers used
    /// here.</summary>
    [Fact]
    public void TransposeToChannelMajorTransposesAndWidensToFloat()
    {
        var audioChunk = new double[3, 2]
        {
            { 1.0, -1.0 },
            { 2.0, -2.0 },
            { 3.0, -3.0 },
        };

        var result = VoiceExtractor.TransposeToChannelMajor(audioChunk);

        Assert.Equal(2, result.GetLength(0));
        Assert.Equal(3, result.GetLength(1));
        Assert.Equal(new float[] { 1, 2, 3 }, new[] { result[0, 0], result[0, 1], result[0, 2] });
        Assert.Equal(new float[] { -1, -2, -3 }, new[] { result[1, 0], result[1, 1], result[1, 2] });
    }

    /// <summary>Python: <c>create_static_model_set('full')</c> has exactly the three
    /// <c>voice_extractor_models</c> entries (<c>kim_vocal_1</c>, <c>kim_vocal_2</c>,
    /// <c>uvr_mdxnet</c>), each with a hash and a source download.</summary>
    [Fact]
    public void CreateStaticModelSetCoversEveryModel()
    {
        var modelSet = VoiceExtractor.CreateStaticModelSet(DownloadScope.Full);

        Assert.Equal(3, modelSet.Count);
        Assert.Contains(VoiceExtractorModel.KimVocal1, modelSet.Keys);
        Assert.Contains(VoiceExtractorModel.KimVocal2, modelSet.Keys);
        Assert.Contains(VoiceExtractorModel.UvrMdxnet, modelSet.Keys);

        foreach (var (hash, source) in modelSet.Values)
        {
            Assert.False(string.IsNullOrEmpty(hash.Url));
            Assert.False(string.IsNullOrEmpty(source.Url));
            Assert.EndsWith(".hash", hash.Path);
            Assert.EndsWith(".onnx", source.Path);
        }
    }

    /// <summary>Python: <c>pre_check</c> returns <see langword="false"/> when
    /// <c>conditional_download_hashes</c>/<c>conditional_download_sources</c> cannot find the
    /// file on disk and cannot download it (no network in this test) — this port's reduced-scope
    /// <c>PreCheck</c> only checks the local file's presence (see
    /// <see cref="VoiceExtractor"/>'s own class remarks), so it returns <see langword="false"/>
    /// deterministically whenever the model file is absent, which it always is on a bare
    /// checkout of this repo (<c>.assets/models/</c> is <c>.gitignore</c>'d).
    /// </summary>
    [Fact]
    public void PreCheckReturnsFalseWhenModelFilesAreMissing()
    {
        var repoRoot = FindRepoRoot();
        if (repoRoot is not null)
        {
            var modelsDirectory = Path.Combine(repoRoot, ".assets", "models");
            if (File.Exists(Path.Combine(modelsDirectory, "kim_vocal_1.onnx")))
            {
                return; // this model happens to be present locally — nothing to assert here.
            }
        }

        Assert.False(VoiceExtractor.PreCheck(VoiceExtractorModel.KimVocal1));
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
