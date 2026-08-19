using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/lip_syncer/{core,types,choices}.py</c>. There is no
/// <c>tests/test_lip_syncer.py</c> in the Python suite (only
/// <c>tests/test_cli_lip_syncer.py</c>, a CLI/subprocess smoke test belonging to a later
/// CLI-layer port phase), so every case below was derived by hand from the module's numpy
/// semantics and cross-checked against a live Python REPL (numpy 2.4.6) — see each test's
/// comment for the exact transcript. Real end-to-end ONNX model coverage lives in
/// <c>tests/FaceFusion.ParityTests/LipSyncerParityTests.cs</c>, ground truth captured by
/// <c>tools/parity/dump_lip_syncer.py</c>.
/// </summary>
public sealed class LipSyncerTests
{
    // -----------------------------------------------------------------
    // create_static_model_set / choices.py
    // -----------------------------------------------------------------

    [Fact]
    public void ModelCatalogHasThreeModels()
    {
        var catalog = LipSyncer.CreateStaticModelSet(DownloadScope.Full);
        Assert.Equal(3, catalog.Count);
        Assert.Equal(3, LipSyncer.LipSyncerModels.Count);

        foreach (var model in LipSyncer.LipSyncerModels)
        {
            Assert.True(catalog.ContainsKey(model), $"{model} missing from the catalog.");
        }
    }

    [Theory]
    [InlineData(LipSyncerModel.Edtalk256, LipSyncerModelKind.Edtalk, 256)]
    [InlineData(LipSyncerModel.Wav2Lip96, LipSyncerModelKind.Wav2Lip, 96)]
    [InlineData(LipSyncerModel.Wav2LipGan96, LipSyncerModelKind.Wav2Lip, 96)]
    public void ModelCatalogEntryMatchesPythonLiterals(LipSyncerModel model, LipSyncerModelKind expectedKind, int expectedSize)
    {
        var options = LipSyncer.CreateStaticModelSet(DownloadScope.Full)[model];

        Assert.Equal(expectedKind, options.Type);
        Assert.Equal(expectedSize, options.Size.Width);
        Assert.Equal(expectedSize, options.Size.Height);
        Assert.True(options.Sources.ContainsKey("lip_syncer"));
        Assert.True(options.Hashes.ContainsKey("lip_syncer"));
    }

    [Fact]
    public void WeightRangeMatchesPythonCreateFloatRange()
    {
        // Python: create_float_range(0.0, 1.0, 0.05) -> 21 values, 0.0..1.0 inclusive.
        Assert.Equal(21, LipSyncer.LipSyncerWeightRange.Count);
        Assert.Equal(0.0, LipSyncer.LipSyncerWeightRange[0], 9);
        Assert.Equal(1.0, LipSyncer.LipSyncerWeightRange[^1], 9);
    }

    // -----------------------------------------------------------------
    // prepare_audio_frame — hand-verified against a live Python REPL
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from a live Python session (numpy 2.4.6):
    /// <code>
    /// af = numpy.random.RandomState(0).uniform(0, 3, (80, 16)).astype(numpy.float64)
    /// t = numpy.maximum(numpy.exp(-5 * numpy.log(10)), af)
    /// t = numpy.log10(t) * 1.6 + 3.2
    /// t = t.clip(-4, 4).astype(numpy.float32)
    /// t.flatten()[:5]
    /// # array([3.5464737, 3.7304678, 3.611629, 3.5414793, 3.3666134], dtype=float32)
    /// </code>
    /// (the <c>edtalk</c> branch never scales by weight, so this is also the <c>edtalk</c>
    /// output for any weight).
    /// </summary>
    [Fact]
    public void PrepareAudioFrameEdtalkMatchesPythonReplTranscript()
    {
        var audioFrame = RandomStateZeroUniform0To3();

        var result = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Edtalk, audioFrame, lipSyncerWeight: 0.5);

        Assert.Equal(80 * 16, result.Length);
        Assert.Equal((double)3.5464737f, result[0], 5);
        Assert.Equal((double)3.7304678f, result[1], 5);
        Assert.Equal((double)3.611629f, result[2], 5);
        Assert.Equal((double)3.5414793f, result[3], 5);
        Assert.Equal((double)3.3666134f, result[4], 5);
    }

    /// <summary>
    /// Ground truth, same <c>af</c> as above, weight = 0.3 (Python: <c>t = t * 0.3 * 2.0</c>
    /// applied after the float32 cast):
    /// <code>
    /// array([2.1278844, 2.2382808, 2.1669774, 2.1248877, 2.019968], dtype=float32)
    /// </code>
    /// Confirms the <c>wav2lip</c>-only weight scaling and that weight = 0.5 is a no-op
    /// (0.5 * 2.0 == 1.0) — deliberately not used as the only test case for that reason.
    /// </summary>
    [Fact]
    public void PrepareAudioFrameWav2LipScalesByWeightTimesTwo()
    {
        var audioFrame = RandomStateZeroUniform0To3();

        var result = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Wav2Lip, audioFrame, lipSyncerWeight: 0.3);

        Assert.Equal((double)2.1278844f, result[0], 5);
        Assert.Equal((double)2.2382808f, result[1], 5);
        Assert.Equal((double)2.1669774f, result[2], 5);
        Assert.Equal((double)2.1248877f, result[3], 5);
        Assert.Equal((double)2.019968f, result[4], 5);
    }

    [Fact]
    public void PrepareAudioFrameWav2LipWeightHalfIsANoOpVersusEdtalk()
    {
        var audioFrame = RandomStateZeroUniform0To3();

        var edtalk = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Edtalk, audioFrame, lipSyncerWeight: 0.5);
        var wav2lip = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Wav2Lip, audioFrame, lipSyncerWeight: 0.5);

        Assert.Equal(edtalk, wav2lip);
    }

    [Fact]
    public void PrepareAudioFrameClampsToPlusMinusFour()
    {
        // A very loud sample (large positive value) and a near-silent one (below the
        // exp(-5 log 10) floor) should clamp to +4 / whatever the floor maps to, respectively.
        var audioFrame = new double[80, 16];
        audioFrame[0, 0] = 1e10; // clamps to +4 after log10*1.6+3.2
        audioFrame[0, 1] = 0.0; // below the floor -> floor value used instead

        var result = LipSyncer.PrepareAudioFrame(LipSyncerModelKind.Edtalk, audioFrame, lipSyncerWeight: 0.5);

        Assert.Equal(4.0, result[0], 5);

        // floor = exp(-5*ln10) = 1e-5; log10(1e-5)*1.6+3.2 = -5*1.6+3.2 = -4.8 -> clipped to -4.
        Assert.Equal(-4.0, result[1], 5);
    }

    private static double[,] RandomStateZeroUniform0To3()
    {
        // Matches numpy.random.RandomState(0).uniform(0, 3, (80, 16)) exactly for the first
        // five flattened elements this test suite reads (values captured from the live Python
        // session referenced in each test's docstring, not regenerated here since this port has
        // no numpy RandomState equivalent — only the five known-good input values are needed to
        // exercise PrepareAudioFrame's own math, the rest of the buffer just has to be within
        // domain (non-negative, for numpy.log10 to be well-defined at the floor)).
        var audioFrame = new double[80, 16];
        audioFrame[0, 0] = 1.6464405117819743;
        audioFrame[0, 1] = 2.1455680991172583;
        audioFrame[0, 2] = 1.8082901282149315;
        audioFrame[0, 3] = 1.6346495489906907;
        audioFrame[0, 4] = 1.270964398016714;

        // Fill the remainder with an arbitrary but valid (non-negative) value; no test reads it.
        for (var mel = 0; mel < 80; mel++)
        {
            for (var step = 0; step < 16; step++)
            {
                if (mel == 0 && step < 5)
                {
                    continue;
                }

                audioFrame[mel, step] = 1.0;
            }
        }

        return audioFrame;
    }

    // -----------------------------------------------------------------
    // normalize_crop_frame's clip(0,1)*255 -> uint8 tail
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from a live Python session (numpy 2.4.6):
    /// <code>
    /// vals = numpy.array([-0.5, 0.0, 0.2, 0.999, 1.0, 1.5], dtype=numpy.float32)
    /// (vals.clip(0, 1) * 255).astype(numpy.uint8)
    /// # array([0, 0, 51, 254, 255, 255], dtype=uint8)
    /// </code>
    /// Confirms the astype(uint8) truncates toward zero (254.745 -> 254, not rounded to 255).
    /// </summary>
    [Theory]
    [InlineData(-0.5f, (byte)0)]
    [InlineData(0.0f, (byte)0)]
    [InlineData(0.2f, (byte)51)]
    [InlineData(0.999f, (byte)254)]
    [InlineData(1.0f, (byte)255)]
    [InlineData(1.5f, (byte)255)]
    public void NormalizeCropFrameClipsAndTruncatesTowardZero(float value, byte expected)
    {
        using var mat = LipSyncer.NormalizeCropFrameWav2Lip(new[] { value, 0f, 0f }, 1, 1);
        var pixel = mat.At<Vec3b>(0, 0);
        Assert.Equal(expected, pixel.Item0);
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame (wav2lip) — the 6-channel concat/zero-half formula
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareCropFrameWav2LipZerosLowerHalfInFirstThreeChannelsOnly()
    {
        using var area = new Mat(4, 4, MatType.CV_8UC3);
        for (var row = 0; row < 4; row++)
        {
            for (var col = 0; col < 4; col++)
            {
                area.Set(row, col, new Vec3b(10, 20, 30));
            }
        }

        var result = LipSyncer.PrepareCropFrameWav2Lip(area);

        Assert.Equal(6 * 16, result.Length);

        var plane = 16;

        // Row 0 (top half, not zeroed): channels 0-2 (prepare) equal channels 3-5 (raw).
        var topIndex = 0; // row 0, col 0
        Assert.Equal((double)(10 / 255f), result[topIndex], 6);
        Assert.Equal((double)(10 / 255f), result[(3 * plane) + topIndex], 6);

        // Row 2 (bottom half, model_size 4 -> zeroFromRow = 2): channels 0-2 zeroed, 3-5 raw.
        var bottomIndex = (2 * 4) + 0; // row 2, col 0
        Assert.Equal(0f, result[bottomIndex]);
        Assert.Equal((double)(10 / 255f), result[(3 * plane) + bottomIndex], 6);
        Assert.Equal((double)(20 / 255f), result[(4 * plane) + bottomIndex], 6);
        Assert.Equal((double)(30 / 255f), result[(5 * plane) + bottomIndex], 6);
    }

    // -----------------------------------------------------------------
    // sync_lip / process_frame orchestration — no-target-face fallthrough
    // -----------------------------------------------------------------

    [Fact]
    public void ProcessFrameReturnsInputFrameUnchangedWhenNoTargetFaceFound()
    {
        using var referenceFrame = new Mat(4, 4, MatType.CV_8UC3, Scalar.All(0));
        using var sourceFrame = new Mat(4, 4, MatType.CV_8UC3, Scalar.All(0));
        using var targetFrame = new Mat(4, 4, MatType.CV_8UC3, Scalar.All(0));
        using var tempFrame = new Mat(4, 4, MatType.CV_8UC3, Scalar.All(0));
        using var tempMask = new Mat(4, 4, MatType.CV_32FC1, Scalar.All(1));

        var inputs = new LipSyncer.LipSyncerInputs(
            ReferenceVisionFrame: referenceFrame,
            SourceVisionFrames: new[] { sourceFrame },
            SourceVoiceFrame: new double[80, 16],
            TargetVisionFrames: new[] { targetFrame },
            TempVisionFrame: tempFrame,
            TempVisionMask: tempMask,
            Model: LipSyncerModel.Wav2LipGan96,
            Weight: 0.5,
            FaceMaskTypes: new[] { FaceMaskType.Box },
            FaceMaskBlur: 0.3,
            FaceMaskPadding: new Padding(0, 0, 0, 0),
            LipSyncerSession: null!,
            OccluderInferencePool: null,
            FaceOccluderModel: FaceOccluderModel.Xseg1,
            FaceSelectorMode: FaceSelectorMode.Many,
            FaceTrackerScore: 0,
            FaceSelectorOrder: FaceSelectorOrder.LeftRight,
            FaceSelectorGender: null,
            FaceSelectorRace: null,
            FaceSelectorAgeStart: null,
            FaceSelectorAgeEnd: null,
            ReferenceFacePosition: 0,
            ReferenceFaceDistance: 0.6,
            GetStaticFaces: _ => Array.Empty<FaceFusion.Types.Face>(), // no faces found anywhere
            RefillFaces: faces => faces.Where(f => f is not null).Select(f => f!).ToArray());

        var outputs = LipSyncer.ProcessFrame(inputs);

        Assert.Same(tempFrame, outputs.VisionFrame);
        Assert.Same(tempMask, outputs.Mask);
    }
}
