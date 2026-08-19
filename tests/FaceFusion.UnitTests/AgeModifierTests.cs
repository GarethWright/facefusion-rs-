using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/age_modifier/{core,types,choices}.py</c>. There is no
/// <c>tests/test_age_modifier.py</c> in the Python suite, so every case below was derived by
/// hand from the module's own numpy semantics. Real end-to-end ONNX model coverage for the
/// <c>fran</c> family lives in <c>tests/FaceFusion.ParityTests/ProcessorParityTests3.cs</c>,
/// ground truth captured by <c>tools/parity/dump_processors3.py</c> — <c>styleganex_age</c> has
/// no ONNX fixture (see that script's docstring) and is exercised only by the hand-computed
/// cases below.
/// </summary>
public sealed class AgeModifierTests
{
    // -----------------------------------------------------------------
    // choices.py / create_static_model_set
    // -----------------------------------------------------------------

    [Fact]
    public void ModelCatalogHasTwoModels()
    {
        var catalog = AgeModifier.CreateStaticModelSet(DownloadScope.Full);
        Assert.Equal(2, catalog.Count);
        Assert.Equal(2, AgeModifier.AgeModifierModels.Count);
        Assert.True(catalog.ContainsKey(AgeModifierModel.Fran));
        Assert.True(catalog.ContainsKey(AgeModifierModel.StyleganexAge));
    }

    [Fact]
    public void FranCatalogEntryMatchesPythonLiterals()
    {
        var options = AgeModifier.CreateStaticModelSet(DownloadScope.Full)[AgeModifierModel.Fran];

        Assert.Equal(WarpTemplate.Ffhq512, options.TargetTemplate);
        Assert.Equal(1024, options.TargetSize.Width);
        Assert.Equal(1024, options.TargetSize.Height);
        Assert.Null(options.TargetWithBackgroundTemplate);
        Assert.Null(options.TargetWithBackgroundSize);
        Assert.Equal(new[] { 0f, 0f, 0f }, options.Mean);
        Assert.Equal(new[] { 1f, 1f, 1f }, options.StandardDeviation);
    }

    [Fact]
    public void StyleganexAgeCatalogEntryMatchesPythonLiterals()
    {
        var options = AgeModifier.CreateStaticModelSet(DownloadScope.Full)[AgeModifierModel.StyleganexAge];

        Assert.Equal(WarpTemplate.Ffhq512, options.TargetTemplate);
        Assert.Equal(256, options.TargetSize.Width);
        Assert.Equal(WarpTemplate.Styleganex384, options.TargetWithBackgroundTemplate);
        Assert.Equal(384, options.TargetWithBackgroundSize!.Value.Width);
        Assert.Equal(new[] { 0.5f, 0.5f, 0.5f }, options.Mean);
        Assert.Equal(new[] { 0.5f, 0.5f, 0.5f }, options.StandardDeviation);
    }

    [Fact]
    public void DirectionRangeMatchesPythonCreateIntRange()
    {
        // Python: create_int_range(-100, 100, 1) -> 201 values, -100..100 inclusive.
        Assert.Equal(201, AgeModifier.AgeModifierDirectionRange.Count);
        Assert.Equal(-100, AgeModifier.AgeModifierDirectionRange[0]);
        Assert.Equal(100, AgeModifier.AgeModifierDirectionRange[^1]);
    }

    // -----------------------------------------------------------------
    // prepare_vision_frame — fran's mean=[0,0,0]/std=[1,1,1] reduces to a plain /255.0
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareVisionFrameMatchesHandComputedValuesForFranMeanStd()
    {
        using var crop = new Mat(4, 4, MatType.CV_8UC3, new Scalar(40, 80, 200)); // BGR
        var mean = new[] { 0f, 0f, 0f };
        var std = new[] { 1f, 1f, 1f };

        var chw = AgeModifier.PrepareVisionFrame(crop, mean, std);
        var plane = 16;

        Assert.Equal((double)((float)(200 / 255.0)), (double)(chw[0]), 6);
        Assert.Equal((double)((float)(80 / 255.0)), (double)(chw[plane]), 6);
        Assert.Equal((double)((float)(40 / 255.0)), (double)(chw[2 * plane]), 6);
    }

    /// <summary>
    /// Ground truth by hand: Python's <c>(x - mean) / std</c> with a non-trivial mean/std (the
    /// <c>styleganex_age</c> values) on a known BGR pixel.
    /// </summary>
    [Fact]
    public void PrepareVisionFrameAppliesMeanAndStandardDeviation()
    {
        using var crop = new Mat(2, 2, MatType.CV_8UC3, new Scalar(0, 0, 255)); // pure red, BGR
        var mean = new[] { 0.5f, 0.5f, 0.5f };
        var std = new[] { 0.5f, 0.5f, 0.5f };

        var chw = AgeModifier.PrepareVisionFrame(crop, mean, std);
        var plane = 4;

        var expectedR = (float)(((255 / 255.0) - 0.5) / 0.5); // 1.0
        var expectedG = (float)(((0 / 255.0) - 0.5) / 0.5); // -1.0
        var expectedB = (float)(((0 / 255.0) - 0.5) / 0.5); // -1.0

        Assert.Equal((double)(expectedR), (double)(chw[0]), 5);
        Assert.Equal((double)(expectedG), (double)(chw[plane]), 5);
        Assert.Equal((double)(expectedB), (double)(chw[2 * plane]), 5);
    }

    // -----------------------------------------------------------------
    // normalize_vision_frame (fran) — stays float64 (CV_64FC3), matching FaceSwapper's
    // documented list-arithmetic promotion rule
    // -----------------------------------------------------------------

    [Fact]
    public void NormalizeVisionFrameForFranReturnsFloat64AndRoundTripsPrepareVisionFrame()
    {
        var mean = new[] { 0f, 0f, 0f };
        var std = new[] { 1f, 1f, 1f };
        var plane = 4;
        var chw = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            chw[i] = 200f / 255f; // R
            chw[plane + i] = 80f / 255f; // G
            chw[(2 * plane) + i] = 40f / 255f; // B
        }

        using var normalized = AgeModifier.NormalizeVisionFrame(chw, 2, 2, mean, std);

        Assert.Equal(MatType.CV_64FC3, normalized.Type());
        normalized.GetArray(out Vec3d[] pixels);

        Assert.Equal((double)(40.0), (double)(pixels[0].Item0), 3); // B
        Assert.Equal((double)(80.0), (double)(pixels[0].Item1), 3); // G
        Assert.Equal((double)(200.0), (double)(pixels[0].Item2), 3); // R
    }

    [Fact]
    public void NormalizeVisionFrameClipsBeforeScaling()
    {
        var mean = new[] { 0f, 0f, 0f };
        var std = new[] { 1f, 1f, 1f };
        var plane = 4;
        var chw = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            chw[i] = 2.0f; // R above 1 -> clip to 1 -> 255
            chw[plane + i] = -1.0f; // G below 0 -> clip to 0 -> 0
            chw[(2 * plane) + i] = 0.5f; // B in range -> 127.5
        }

        using var normalized = AgeModifier.NormalizeVisionFrame(chw, 2, 2, mean, std);
        normalized.GetArray(out Vec3d[] pixels);

        Assert.Equal((double)(127.5), (double)(pixels[0].Item0), 3);
        Assert.Equal((double)(0.0), (double)(pixels[0].Item1), 3);
        Assert.Equal((double)(255.0), (double)(pixels[0].Item2), 3);
    }

    // -----------------------------------------------------------------
    // normalize_extend_frame (styleganex_age) — narrows to uint8, matching Python's astype
    // -----------------------------------------------------------------

    [Fact]
    public void NormalizeExtendFrameMapsMinusOneToOneRangeToByteRange()
    {
        var plane = 4;
        var chw = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            chw[i] = 1.0f; // R -> (1+1)/2=1 -> 255
            chw[plane + i] = -1.0f; // G -> (−1+1)/2=0 -> 0
            chw[(2 * plane) + i] = 0.0f; // B -> (0+1)/2=0.5 -> 127
        }

        using var normalized = AgeModifier.NormalizeExtendFrame(chw, 2, 2, new Size(8, 8));

        Assert.Equal(MatType.CV_8UC3, normalized.Type());
        Assert.Equal(8, normalized.Cols);
        Assert.Equal(8, normalized.Rows);

        normalized.GetArray(out Vec3b[] pixels);

        // Upscaled via INTER_AREA from a uniform 2x2 source, so every output pixel keeps the
        // same constant BGR value: Item0=B=127, Item1=G=0, Item2=R=255.
        Assert.Equal(127, pixels[0].Item0);
        Assert.Equal(0, pixels[0].Item1);
        Assert.Equal(255, pixels[0].Item2);
    }

    [Fact]
    public void NormalizeExtendFrameClipsBeyondMinusOneToOneRange()
    {
        var plane = 4;
        var chw = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            chw[i] = 5.0f; // R clipped to 1 -> 255
            chw[plane + i] = -5.0f; // G clipped to -1 -> 0
            chw[(2 * plane) + i] = 0.0f; // B -> 127
        }

        using var normalized = AgeModifier.NormalizeExtendFrame(chw, 2, 2, new Size(4, 4));
        normalized.GetArray(out Vec3b[] pixels);

        Assert.Equal(127, pixels[0].Item0);
        Assert.Equal(0, pixels[0].Item1);
        Assert.Equal(255, pixels[0].Item2);
    }
}
