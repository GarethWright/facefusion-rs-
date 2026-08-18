using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/deep_swapper/{core,types,choices}.py</c>. There is no
/// <c>tests/test_deep_swapper.py</c> in the Python suite, so every case below was derived by
/// hand from the module's own numpy/cv2 semantics. Real end-to-end preprocessing coverage
/// against real Python output (no ONNX Runtime — see <c>tools/parity/dump_processors4.py</c>'s
/// docstring for why) lives in <c>tests/FaceFusion.ParityTests/ProcessorParityTests4.cs</c>.
/// </summary>
public sealed class DeepSwapperTests
{
    // -----------------------------------------------------------------
    // choices.py / create_static_model_set
    // -----------------------------------------------------------------

    [Fact]
    public void MorphRangeMatchesPythonCreateIntRange()
    {
        // Python: create_int_range(0, 100, 1) -> 101 values, 0..100 inclusive.
        Assert.Equal(101, DeepSwapper.DeepSwapperMorphRange.Count);
        Assert.Equal(0, DeepSwapper.DeepSwapperMorphRange[0]);
        Assert.Equal(100, DeepSwapper.DeepSwapperMorphRange[^1]);
    }

    [Fact]
    public void ModelCatalogFullScopeContainsEveryPythonScope()
    {
        var catalog = DeepSwapper.CreateStaticModelSet(DownloadScope.Full);

        Assert.True(catalog.ContainsKey("druuzil/adam_levine_320"));
        Assert.True(catalog.ContainsKey("edel/winona_ryder_224"));
        Assert.True(catalog.ContainsKey("iperov/elon_musk_224"));
        Assert.True(catalog.ContainsKey("jen/ella_freya_224"));
        Assert.True(catalog.ContainsKey("mats/billie_eilish_224"));
        Assert.True(catalog.ContainsKey("rumateus/taylor_swift_224"));

        // Python: 79 druuzil + 5 edel + 24 iperov + 7 jen + 17 mats + 26 rumateus (custom/*
        // entries depend on the local filesystem, so this only asserts a lower bound).
        Assert.True(catalog.Count >= 79 + 5 + 24 + 7 + 17 + 26);
    }

    [Fact]
    public void ModelCatalogLiteScopeOnlyContainsIperov()
    {
        var catalog = DeepSwapper.CreateStaticModelSet(DownloadScope.Lite);

        Assert.True(catalog.ContainsKey("iperov/elon_musk_224"));
        Assert.False(catalog.ContainsKey("druuzil/adam_levine_320"));
        Assert.False(catalog.ContainsKey("jen/ella_freya_224"));
        Assert.Equal(24, catalog.Count);
    }

    [Fact]
    public void ModelCatalogEntryUsesDflWholeFaceTemplateAndHuggingfaceUrls()
    {
        var options = DeepSwapper.CreateStaticModelSet(DownloadScope.Full)["iperov/elon_musk_224"];

        Assert.Equal(WarpTemplate.DflWholeFace, options.Template);
        Assert.NotNull(options.Hashes);
        Assert.True(options.Hashes!.ContainsKey("deep_swapper"));
        Assert.True(options.Sources.ContainsKey("deep_swapper"));
        Assert.Contains("huggingface.co", options.Sources["deep_swapper"].Url);
        Assert.Contains("deepfacelive-models-iperov", options.Sources["deep_swapper"].Url);
        Assert.EndsWith("elon_musk_224.dfm", options.Sources["deep_swapper"].Url);
        Assert.EndsWith(Path.Combine("iperov", "elon_musk_224.dfm"), options.Sources["deep_swapper"].Path);
    }

    [Fact]
    public void PreCheckReturnsFalseForAMissingBuiltInModel()
    {
        // iperov/elon_musk_224's .dfm is never fetched by this test suite (hosted on
        // Hugging Face, gitignored, not present in CI) — matches Python's own `pre_check`
        // returning False for an absent file.
        Assert.False(DeepSwapper.PreCheck("iperov/elon_musk_224"));
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame — synthetic, hand-verified (no cv2 sharpen effect for a flat crop)
    // -----------------------------------------------------------------

    /// <summary>
    /// A constant-colour crop is unaffected by the unsharp-mask sharpen (blurring a flat
    /// region reproduces the same flat region, so <c>1.75 * flat - 0.75 * flat == flat</c>),
    /// so the expected HWC output reduces to a plain <c>channel / 255.0</c> formula.
    /// </summary>
    [Fact]
    public void PrepareCropFrameMatchesHandComputedValuesForAConstantColorCrop()
    {
        using var crop = new Mat(64, 64, MatType.CV_8UC3, new Scalar(40, 80, 200)); // BGR
        var hwc = DeepSwapper.PrepareCropFrame(crop);

        Assert.Equal(64 * 64 * 3, hwc.Length);

        var expectedB = (float)(40 / 255.0);
        var expectedG = (float)(80 / 255.0);
        var expectedR = (float)(200 / 255.0);

        // NHWC/BGR layout, no channel reversal (unlike FaceSwapper/ExpressionRestorer).
        Assert.Equal((double)expectedB, (double)hwc[0], 5);
        Assert.Equal((double)expectedG, (double)hwc[1], 5);
        Assert.Equal((double)expectedR, (double)hwc[2], 5);

        var lastPixel = (64 * 64 - 1) * 3;
        Assert.Equal((double)expectedB, (double)hwc[lastPixel], 5);
        Assert.Equal((double)expectedG, (double)hwc[lastPixel + 1], 5);
        Assert.Equal((double)expectedR, (double)hwc[lastPixel + 2], 5);
    }

    // -----------------------------------------------------------------
    // normalize_crop_frame — synthetic, hand-verified
    // -----------------------------------------------------------------

    [Fact]
    public void NormalizeCropFrameRoundTripsPrepareCropFrameForAConstantColorCrop()
    {
        const int size = 8;
        var hwc = new float[size * size * 3];
        for (var i = 0; i < size * size; i++)
        {
            hwc[(i * 3) + 0] = 40f / 255f; // B
            hwc[(i * 3) + 1] = 80f / 255f; // G
            hwc[(i * 3) + 2] = 200f / 255f; // R
        }

        using var normalized = DeepSwapper.NormalizeCropFrame(hwc, size, size);

        Assert.Equal(MatType.CV_8UC3, normalized.Type());
        normalized.GetArray(out Vec3b[] pixels);

        // No channel reversal — BGR in, BGR out.
        Assert.Equal(40, pixels[0].Item0);
        Assert.Equal(80, pixels[0].Item1);
        Assert.Equal(200, pixels[0].Item2);
    }

    [Fact]
    public void NormalizeCropFrameClipsOutOfRangeValues()
    {
        const int size = 4;
        var hwc = new float[size * size * 3];
        for (var i = 0; i < size * size; i++)
        {
            hwc[(i * 3) + 0] = 2.0f; // B: above 1 -> clip to 1 -> 255
            hwc[(i * 3) + 1] = -1.0f; // G: below 0 -> clip to 0 -> 0
            hwc[(i * 3) + 2] = 0.5f; // R: in range -> 127.5 -> truncated to 127
        }

        using var normalized = DeepSwapper.NormalizeCropFrame(hwc, size, size);
        normalized.GetArray(out Vec3b[] pixels);

        Assert.Equal(255, pixels[0].Item0);
        Assert.Equal(0, pixels[0].Item1);
        Assert.Equal(127, pixels[0].Item2);
    }

    // -----------------------------------------------------------------
    // prepare_crop_mask — synthetic, hand-verified against a simpler property
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareCropMaskReducesToTheSmallerMaskWhenBothInputsAreEqual()
    {
        const int size = 16;
        var mask = new float[size * size];
        for (var i = 0; i < mask.Length; i++)
        {
            mask[i] = 0.5f;
        }

        using var result = DeepSwapper.PrepareCropMask(mask, mask, new Size(size, size));

        Assert.Equal(MatType.CV_32FC1, result.Type());
        Assert.Equal(size, result.Rows);
        Assert.Equal(size, result.Cols);

        // A uniform 0.5 mask survives erode (nothing to shrink into) and Gaussian blur
        // (nothing to smooth) unchanged.
        Assert.Equal(0.5, (double)result.At<float>(size / 2, size / 2), 4);
    }

    [Fact]
    public void PrepareCropMaskThrowsForAMismatchedLength()
    {
        var mask = new float[10];
        Assert.Throws<ArgumentException>(() => DeepSwapper.PrepareCropMask(mask, mask, new Size(16, 16)));
    }
}
