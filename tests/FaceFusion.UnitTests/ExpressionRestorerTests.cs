using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/expression_restorer/{core,types,choices}.py</c>. There is no
/// <c>tests/test_expression_restorer.py</c> in the Python suite, so every case below was derived
/// by hand from the module's own numpy semantics. Real end-to-end ONNX model coverage (including
/// <see cref="ExpressionRestorer.PrepareCropFrame"/>/<see cref="ExpressionRestorer.NormalizeCropFrame"/>
/// against real Python output) lives in <c>tests/FaceFusion.ParityTests/ProcessorParityTests3.cs</c>,
/// ground truth captured by <c>tools/parity/dump_processors3.py</c>.
/// </summary>
public sealed class ExpressionRestorerTests
{
    // -----------------------------------------------------------------
    // choices.py / create_static_model_set
    // -----------------------------------------------------------------

    [Fact]
    public void ModelCatalogHasOneModel()
    {
        var catalog = ExpressionRestorer.CreateStaticModelSet(DownloadScope.Full);
        Assert.Single(catalog);
        Assert.Single(ExpressionRestorer.ExpressionRestorerModels);
        Assert.True(catalog.ContainsKey(ExpressionRestorerModel.LivePortrait));
    }

    [Fact]
    public void ModelCatalogEntryMatchesPythonLiterals()
    {
        var options = ExpressionRestorer.CreateStaticModelSet(DownloadScope.Full)[ExpressionRestorerModel.LivePortrait];

        Assert.Equal(WarpTemplate.Arcface128, options.Template);
        Assert.Equal(512, options.Size.Width);
        Assert.Equal(512, options.Size.Height);
        Assert.True(options.Sources.ContainsKey("feature_extractor"));
        Assert.True(options.Sources.ContainsKey("motion_extractor"));
        Assert.True(options.Sources.ContainsKey("generator"));
        Assert.True(options.Hashes.ContainsKey("feature_extractor"));
        Assert.True(options.Hashes.ContainsKey("motion_extractor"));
        Assert.True(options.Hashes.ContainsKey("generator"));
    }

    [Fact]
    public void AreasChoicesMatchPython()
    {
        Assert.Equal(new[] { ExpressionRestorerArea.UpperFace, ExpressionRestorerArea.LowerFace }, ExpressionRestorer.ExpressionRestorerAreas);
    }

    [Fact]
    public void FactorRangeMatchesPythonCreateIntRange()
    {
        // Python: create_int_range(0, 100, 1) -> 101 values, 0..100 inclusive.
        Assert.Equal(101, ExpressionRestorer.ExpressionRestorerFactorRange.Count);
        Assert.Equal(0, ExpressionRestorer.ExpressionRestorerFactorRange[0]);
        Assert.Equal(100, ExpressionRestorer.ExpressionRestorerFactorRange[^1]);
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame / normalize_crop_frame — synthetic, hand-verified
    // -----------------------------------------------------------------

    /// <summary>
    /// A constant-colour crop survives <c>cv2.INTER_AREA</c> resize unchanged (area-averaging a
    /// uniform region reproduces the same uniform value), so the expected CHW output reduces to a
    /// plain <c>channel / 255.0</c> formula that can be verified by hand rather than needing a
    /// live Python process.
    /// </summary>
    [Fact]
    public void PrepareCropFrameMatchesHandComputedValuesForAConstantColorCrop()
    {
        using var crop = new Mat(512, 512, MatType.CV_8UC3, new Scalar(40, 80, 200)); // BGR
        var chw = ExpressionRestorer.PrepareCropFrame(crop);

        Assert.Equal(3 * 256 * 256, chw.Length);

        var plane = 256 * 256;
        var expectedR = (float)(200 / 255.0);
        var expectedG = (float)(80 / 255.0);
        var expectedB = (float)(40 / 255.0);

        Assert.Equal((double)(expectedR), (double)(chw[0]), 6);
        Assert.Equal((double)(expectedR), (double)(chw[plane - 1]), 6);
        Assert.Equal((double)(expectedG), (double)(chw[plane]), 6);
        Assert.Equal((double)(expectedG), (double)(chw[(2 * plane) - 1]), 6);
        Assert.Equal((double)(expectedB), (double)(chw[2 * plane]), 6);
        Assert.Equal((double)(expectedB), (double)(chw[(3 * plane) - 1]), 6);
    }

    [Fact]
    public void NormalizeCropFrameRoundTripsPrepareCropFrameForAConstantColorCrop()
    {
        var plane = 512 * 512;
        var chw = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            chw[i] = 200f / 255f; // R
            chw[plane + i] = 80f / 255f; // G
            chw[(2 * plane) + i] = 40f / 255f; // B
        }

        using var normalized = ExpressionRestorer.NormalizeCropFrame(chw, 512, 512);

        Assert.Equal(MatType.CV_8UC3, normalized.Type());
        normalized.GetArray(out Vec3b[] pixels);

        // BGR order in the output Mat.
        Assert.Equal(40, pixels[0].Item0);
        Assert.Equal(80, pixels[0].Item1);
        Assert.Equal(200, pixels[0].Item2);
    }

    [Fact]
    public void NormalizeCropFrameClipsOutOfRangeValues()
    {
        var plane = 4 * 4;
        var chw = new float[3 * plane];
        for (var i = 0; i < plane; i++)
        {
            chw[i] = 2.0f; // R: above 1 -> clip to 1 -> 255
            chw[plane + i] = -1.0f; // G: below 0 -> clip to 0 -> 0
            chw[(2 * plane) + i] = 0.5f; // B: in range -> 127 (truncated, not rounded)
        }

        using var normalized = ExpressionRestorer.NormalizeCropFrame(chw, 4, 4);
        normalized.GetArray(out Vec3b[] pixels);

        // BGR order: Item0 = B (in range: 0.5 * 255 = 127.5 -> truncated, not rounded, to 127),
        // Item1 = G (clipped to 0 -> 0), Item2 = R (clipped to 1 -> 255).
        Assert.Equal(127, pixels[0].Item0);
        Assert.Equal(0, pixels[0].Item1);
        Assert.Equal(255, pixels[0].Item2);
    }

    // -----------------------------------------------------------------
    // restrict_expression_areas
    // -----------------------------------------------------------------

    private static float[,] BuildDistinctExpression(float baseValue)
    {
        var expression = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                expression[i, c] = baseValue + i + (c * 0.01f);
            }
        }

        return expression;
    }

    [Fact]
    public void RestrictExpressionAreasWithBothAreasEnabledOnlyOverwritesAlwaysRestrictedRows()
    {
        var temp = BuildDistinctExpression(0f);
        var target = BuildDistinctExpression(100f);

        var result = ExpressionRestorer.RestrictExpressionAreas(temp, target, new[] { ExpressionRestorerArea.UpperFace, ExpressionRestorerArea.LowerFace });

        int[] alwaysRestricted = { 0, 4, 5, 8, 9 };
        for (var i = 0; i < 21; i++)
        {
            var expectSource = alwaysRestricted.Contains(i) ? temp : target;
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal((double)expectSource[i, c], (double)result[i, c], 6);
            }
        }
    }

    [Fact]
    public void RestrictExpressionAreasWithoutUpperFaceOverwritesUpperFaceRows()
    {
        var temp = BuildDistinctExpression(0f);
        var target = BuildDistinctExpression(100f);

        var result = ExpressionRestorer.RestrictExpressionAreas(temp, target, new[] { ExpressionRestorerArea.LowerFace });

        int[] upperFaceRows = { 1, 2, 6, 10, 11, 12, 13, 15, 16 };
        int[] alwaysRestricted = { 0, 4, 5, 8, 9 };
        int[] restrictedFromTemp = upperFaceRows.Concat(alwaysRestricted).Distinct().ToArray();

        for (var i = 0; i < 21; i++)
        {
            var expectSource = restrictedFromTemp.Contains(i) ? temp : target;
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal((double)expectSource[i, c], (double)result[i, c], 6);
            }
        }
    }

    [Fact]
    public void RestrictExpressionAreasWithoutLowerFaceOverwritesLowerFaceRows()
    {
        var temp = BuildDistinctExpression(0f);
        var target = BuildDistinctExpression(100f);

        var result = ExpressionRestorer.RestrictExpressionAreas(temp, target, new[] { ExpressionRestorerArea.UpperFace });

        int[] lowerFaceRows = { 3, 7, 14, 17, 18, 19, 20 };
        int[] alwaysRestricted = { 0, 4, 5, 8, 9 };
        int[] restrictedFromTemp = lowerFaceRows.Concat(alwaysRestricted).Distinct().ToArray();

        for (var i = 0; i < 21; i++)
        {
            var expectSource = restrictedFromTemp.Contains(i) ? temp : target;
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal((double)expectSource[i, c], (double)result[i, c], 6);
            }
        }
    }

    [Fact]
    public void RestrictExpressionAreasDoesNotMutateItsInputArrays()
    {
        var temp = BuildDistinctExpression(0f);
        var target = BuildDistinctExpression(100f);
        var targetCopy = (float[,])target.Clone();

        ExpressionRestorer.RestrictExpressionAreas(temp, target, Array.Empty<ExpressionRestorerArea>());

        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal(targetCopy[i, c], target[i, c]);
            }
        }
    }
}
