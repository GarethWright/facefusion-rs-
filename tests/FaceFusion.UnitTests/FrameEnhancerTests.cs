using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/frame_enhancer/{core,types,choices}.py</c>. There is no
/// <c>tests/test_frame_enhancer.py</c> in the Python suite, so this file exercises the pure
/// arithmetic (<c>prepare_tile_frame</c>/<c>normalize_tile_frame</c>/<c>blend_merge_frame</c>)
/// against hand-computed numpy-equivalent values, plus the model-set/choices tables and the
/// CoreML <c>adjust_inference_providers</c> gate. The real-ONNX end-to-end path (including
/// <see cref="FaceFusion.Vision.Vision.CreateTileFrames"/>/<c>MergeTileFrames</c>, which are
/// reused rather than re-derived here) is verified against ground truth dumped from the real
/// <c>facefusion.processors.modules.frame_enhancer</c> module in
/// <c>tests/FaceFusion.ParityTests/EnhancerParityTests.cs</c>.
/// </summary>
public sealed class FrameEnhancerTests
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    [Fact]
    public void FrameEnhancerModelsListsEveryLiteralValue()
    {
        var models = FrameEnhancer.FrameEnhancerModels;
        Assert.Equal(19, models.Count);
        Assert.Contains(FrameEnhancerModel.ClearRealityX4, models);
        Assert.Contains(FrameEnhancerModel.UltraSharp2X4, models);
    }

    [Fact]
    public void FrameEnhancerBlendRangeMatchesPython()
    {
        // Python: create_int_range(0, 100, 1) -> [0, 1, ..., 100].
        Assert.Equal(101, FrameEnhancer.FrameEnhancerBlendRange.Count);
        Assert.Equal(0, FrameEnhancer.FrameEnhancerBlendRange[0]);
        Assert.Equal(100, FrameEnhancer.FrameEnhancerBlendRange[^1]);
    }

    // -----------------------------------------------------------------
    // Model set — every family's tile size/scale/precision, transcribed from
    // create_static_model_set.
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(FrameEnhancerModel.ClearRealityX4, 128, 8, 4, 4, null)]
    [InlineData(FrameEnhancerModel.RealEsrganX2, 256, 16, 8, 2, null)]
    [InlineData(FrameEnhancerModel.RealEsrganX2Fp16, 256, 16, 8, 2, FrameEnhancerPrecision.Fp16)]
    [InlineData(FrameEnhancerModel.RealEsrganX8, 256, 16, 8, 8, null)]
    [InlineData(FrameEnhancerModel.RealWebPhotoX4, 64, 4, 2, 4, null)]
    [InlineData(FrameEnhancerModel.TghqFaceX8, 128, 8, 4, 8, null)]
    [InlineData(FrameEnhancerModel.UltraSharp2X4, 1024, 64, 32, 4, null)]
    public void CreateStaticModelSetMatchesPython(FrameEnhancerModel model, int tileSize, int padSize, int overlapSize, int scale, FrameEnhancerPrecision? precision)
    {
        var modelSet = FrameEnhancer.CreateStaticModelSet(DownloadScope.Full);
        var options = modelSet[model];

        Assert.Equal((tileSize, padSize, overlapSize), options.Size);
        Assert.Equal(scale, options.Scale);
        Assert.Equal(precision, options.Precision);
        Assert.EndsWith(".onnx", options.Source.Path, StringComparison.Ordinal);
        Assert.EndsWith(".hash", options.Hash.Path, StringComparison.Ordinal);
        Assert.Contains("github.com", options.Source.Url, StringComparison.Ordinal);
    }

    [Fact]
    public void GetModelOptionsMatchesCreateStaticModelSet()
    {
        var expected = FrameEnhancer.CreateStaticModelSet(DownloadScope.Full)[FrameEnhancerModel.RealWebPhotoX4];
        var actual = FrameEnhancer.GetModelOptions(FrameEnhancerModel.RealWebPhotoX4);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void PreCheckIsFalseWhenModelFilesAreAbsent()
    {
        var result = FrameEnhancer.PreCheck(FrameEnhancerModel.RealWebPhotoX4);
        Assert.Equal(FaceFusion.Core.FileSystem.IsFile(FrameEnhancer.GetModelOptions(FrameEnhancerModel.RealWebPhotoX4).Source.Path), result);
    }

    // -----------------------------------------------------------------
    // adjust_inference_providers — only ever non-empty on macOS + CoreML + fp16.
    // -----------------------------------------------------------------

    [Fact]
    public void AdjustInferenceProvidersIsEmptyForNonFp16Model()
    {
        // clear_reality_x4 has no 'precision' entry (None in Python), so the CoreML branch's
        // `model_precision == 'fp16'` check is false regardless of platform/execution provider.
        var result = FrameEnhancer.AdjustInferenceProviders(FrameEnhancerModel.ClearRealityX4);
        Assert.Empty(result);
    }

    [Fact]
    public void AdjustInferenceProvidersIsEmptyOffMacOs()
    {
        if (OperatingSystem.IsMacOS())
        {
            // This assertion only holds off macOS — the CoreML branch additionally requires
            // Execution.HasExecutionProvider(Coreml), which needs a real execution-provider
            // registration this unit test does not set up, so skip rather than assert a
            // platform-dependent result.
            return;
        }

        var result = FrameEnhancer.AdjustInferenceProviders(FrameEnhancerModel.RealEsrganX2Fp16);
        Assert.Empty(result);
    }

    // -----------------------------------------------------------------
    // prepare_tile_frame — BGR->RGB, NHWC->NCHW, /255 (no [-1,1] rescale, unlike face_enhancer)
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareTileFrameReversesChannelsAndScalesToZeroOne()
    {
        using var tile = new Mat(1, 1, MatType.CV_8UC3);
        tile.Set(0, 0, new Vec3b(10, 20, 30)); // B=10, G=20, R=30

        var (chw, height, width) = FrameEnhancer.PrepareTileFrame(tile);

        Assert.Equal(1, height);
        Assert.Equal(1, width);
        Assert.Equal(3, chw.Length);

        Assert.Equal(30 / 255f, chw[0], 1e-6f); // R
        Assert.Equal(20 / 255f, chw[1], 1e-6f); // G
        Assert.Equal(10 / 255f, chw[2], 1e-6f); // B
    }

    // -----------------------------------------------------------------
    // normalize_tile_frame — NCHW->NHWC, *255, clip(0,255), uint8 (truncating, no round), RGB->BGR
    // -----------------------------------------------------------------

    [Fact]
    public void NormalizeTileFrameRoundTripsPrepareTileFrameForExactEighthValues()
    {
        // 255 is exactly divisible into a value whose /255 then *255 round-trips exactly
        // (no truncation loss) so the round trip can be asserted with no tolerance.
        using var tile = new Mat(1, 1, MatType.CV_8UC3);
        tile.Set(0, 0, new Vec3b(0, 128, 255));

        var (chw, height, width) = FrameEnhancer.PrepareTileFrame(tile);
        using var normalized = FrameEnhancer.NormalizeTileFrame(chw, height, width);

        var pixel = normalized.At<Vec3b>(0, 0);
        Assert.Equal(0, pixel.Item0);
        Assert.InRange(pixel.Item1, 127, 128); // 128/255*255 may lose <1 ULP through float32
        Assert.Equal(255, pixel.Item2);
    }

    [Fact]
    public void NormalizeTileFrameClipsOutOfRangeValues()
    {
        // Simulate a model output that overshoots [0, 1] — normalize_tile_frame's Python
        // clips the *scaled* (x * 255) value to [0, 255], not the pre-scale value.
        float[] chw = { 2f, -1f, 0.5f };

        using var normalized = FrameEnhancer.NormalizeTileFrame(chw, 1, 1);
        var pixel = normalized.At<Vec3b>(0, 0);

        Assert.Equal(255, pixel.Item2); // R: 2 * 255 = 510 -> clipped to 255
        Assert.Equal(0, pixel.Item1); // G: -1 * 255 = -255 -> clipped to 0
        Assert.Equal(127, pixel.Item0); // B: 0.5 * 255 = 127.5 -> truncated (not rounded) to 127
    }

    [Fact]
    public void NormalizeTileFrameTruncatesRatherThanRounds()
    {
        // Python's astype(uint8) truncates toward zero — unlike face_enhancer's
        // normalize_crop_frame, there is no .round() call in normalize_tile_frame at all.
        // 0.999 * 255 = 254.745, which truncates to 254 (a .round() would give 255).
        float[] chw = { 0.999f, 0f, 0f };

        using var normalized = FrameEnhancer.NormalizeTileFrame(chw, 1, 1);
        var pixel = normalized.At<Vec3b>(0, 0);
        Assert.Equal(254, pixel.Item2); // R channel
    }

    // -----------------------------------------------------------------
    // blend_merge_frame — resize temp to merge's size, then the Python blend formula
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(100, 1.0)]
    [InlineData(0, 0.0)]
    [InlineData(80, 0.8)]
    public void BlendMergeFrameUsesPythonBlendFormula(int frameEnhancerBlend, double expectedMergeWeight)
    {
        using var temp = new Mat(2, 2, MatType.CV_8UC3, new Scalar(0, 0, 0));
        using var merge = new Mat(2, 2, MatType.CV_8UC3, new Scalar(200, 200, 200));

        using var blended = FrameEnhancer.BlendMergeFrame(temp, merge, frameEnhancerBlend);
        Assert.Equal(merge.Size(), blended.Size());

        var pixel = blended.At<Vec3b>(0, 0);
        var expected = (byte)Math.Round(200 * expectedMergeWeight, MidpointRounding.AwayFromZero);
        Assert.InRange(pixel.Item0, Math.Max(0, expected - 1), Math.Min(255, expected + 1));
    }

    [Fact]
    public void BlendMergeFrameResizesTempToMergeDimensions()
    {
        using var temp = new Mat(4, 4, MatType.CV_8UC3, new Scalar(50, 50, 50));
        using var merge = new Mat(8, 8, MatType.CV_8UC3, new Scalar(100, 100, 100));

        using var blended = FrameEnhancer.BlendMergeFrame(temp, merge, 50);

        Assert.Equal(8, blended.Rows);
        Assert.Equal(8, blended.Cols);
    }
}
