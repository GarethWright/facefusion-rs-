using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/face_editor/{core,types,choices}.py</c>. There is no
/// <c>tests/test_face_editor.py</c> in the Python suite, so every case below was derived by
/// hand from the module's own numpy semantics. Real end-to-end ONNX model coverage (feature/
/// motion extractor, eye/lip retargeter, stitcher, generator) against real Python output lives
/// in <c>tests/FaceFusion.ParityTests/ProcessorParityTests4.cs</c>, ground truth captured by
/// <c>tools/parity/dump_processors4.py</c>.
/// </summary>
public sealed class FaceEditorTests
{
    // -----------------------------------------------------------------
    // choices.py / create_static_model_set
    // -----------------------------------------------------------------

    [Fact]
    public void ModelCatalogHasOneModel()
    {
        var catalog = FaceEditor.CreateStaticModelSet(DownloadScope.Full);
        Assert.Single(catalog);
        Assert.Single(FaceEditor.FaceEditorModels);
        Assert.True(catalog.ContainsKey(FaceEditorModel.LivePortrait));
    }

    [Fact]
    public void ModelCatalogEntryMatchesPythonLiterals()
    {
        var options = FaceEditor.CreateStaticModelSet(DownloadScope.Full)[FaceEditorModel.LivePortrait];

        Assert.Equal(WarpTemplate.Ffhq512, options.Template);
        Assert.Equal(512, options.Size.Width);
        Assert.Equal(512, options.Size.Height);

        foreach (var key in new[] { "feature_extractor", "motion_extractor", "eye_retargeter", "lip_retargeter", "stitcher", "generator" })
        {
            Assert.True(options.Sources.ContainsKey(key), $"missing source '{key}'");
            Assert.True(options.Hashes.ContainsKey(key), $"missing hash '{key}'");
        }
    }

    [Fact]
    public void SliderRangeMatchesPythonCreateFloatRange()
    {
        // Python: create_float_range(-1.0, 1.0, 0.05) -> 41 values, -1.0..1.0 inclusive.
        Assert.Equal(41, FaceEditor.FaceEditorSliderRange.Count);
        Assert.Equal(-1.0, FaceEditor.FaceEditorSliderRange[0], 6);
        Assert.Equal(1.0, FaceEditor.FaceEditorSliderRange[^1], 6);
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame / normalize_crop_frame — synthetic, hand-verified
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareCropFrameMatchesHandComputedValuesForAConstantColorCrop()
    {
        using var crop = new Mat(512, 512, MatType.CV_8UC3, new Scalar(40, 80, 200)); // BGR
        var chw = FaceEditor.PrepareCropFrame(crop);

        Assert.Equal(3 * 256 * 256, chw.Length);

        var plane = 256 * 256;
        var expectedR = (float)(200 / 255.0);
        var expectedG = (float)(80 / 255.0);
        var expectedB = (float)(40 / 255.0);

        Assert.Equal((double)expectedR, (double)chw[0], 6);
        Assert.Equal((double)expectedR, (double)chw[plane - 1], 6);
        Assert.Equal((double)expectedG, (double)chw[plane], 6);
        Assert.Equal((double)expectedG, (double)chw[(2 * plane) - 1], 6);
        Assert.Equal((double)expectedB, (double)chw[2 * plane], 6);
        Assert.Equal((double)expectedB, (double)chw[(3 * plane) - 1], 6);
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

        using var normalized = FaceEditor.NormalizeCropFrame(chw, 512, 512);

        Assert.Equal(MatType.CV_8UC3, normalized.Type());
        normalized.GetArray(out Vec3b[] pixels);

        Assert.Equal(40, pixels[0].Item0); // B
        Assert.Equal(80, pixels[0].Item1); // G
        Assert.Equal(200, pixels[0].Item2); // R
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
            chw[(2 * plane) + i] = 0.5f; // B: in range -> 127.5 -> truncated to 127
        }

        using var normalized = FaceEditor.NormalizeCropFrame(chw, 4, 4);
        normalized.GetArray(out Vec3b[] pixels);

        Assert.Equal(127, pixels[0].Item0); // B
        Assert.Equal(0, pixels[0].Item1); // G
        Assert.Equal(255, pixels[0].Item2); // R
    }

    // -----------------------------------------------------------------
    // calculate_distance_ratio
    // -----------------------------------------------------------------

    [Fact]
    public void CalculateDistanceRatioMatchesHandComputedValue()
    {
        var landmark68 = new float[68, 2];
        // Vertical: top(0,0) - bottom(0,3) -> length 3. Horizontal: left(3,0) - right(0,0) -> length 3.
        landmark68[37, 0] = 0;
        landmark68[37, 1] = 0;
        landmark68[40, 0] = 0;
        landmark68[40, 1] = 3;
        landmark68[39, 0] = 4;
        landmark68[39, 1] = 0;
        landmark68[36, 0] = 0;
        landmark68[36, 1] = 0;

        var ratio = FaceEditor.CalculateDistanceRatio(landmark68, 37, 40, 39, 36);

        // vertical norm = 3, horizontal norm = 4 -> ratio = 3 / (4 + 1e-6) ~= 0.75.
        Assert.Equal(0.75, (double)ratio, 5);
    }

    // -----------------------------------------------------------------
    // edit_* sliders — hand-verified against the Python numpy.interp formulas
    // -----------------------------------------------------------------

    private static float[,] ZeroExpression() => new float[21, 3];

    [Fact]
    public void EditEyebrowDirectionPositiveBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditEyebrowDirection(ZeroExpression(), 0.5);

        // numpy.interp(0.5, [-1, 1], [-0.015, 0.015]) == 0.0075
        Assert.Equal(0.0075, (double)result[1, 1], 5);
        // numpy.interp(0.5, [-1, 1], [-0.020, 0.020]) == 0.01, negated
        Assert.Equal(-0.01, (double)result[2, 1], 5);
        Assert.Equal(0.0, (double)result[1, 0], 5);
    }

    [Fact]
    public void EditEyebrowDirectionNegativeBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditEyebrowDirection(ZeroExpression(), -0.5);

        Assert.Equal(0.0075, (double)result[1, 0], 5); // -interp(-0.5,[-0.015,0.015]) = -(-0.0075)
        Assert.Equal(-0.01, (double)result[2, 0], 5);
        Assert.Equal(-0.0025, (double)result[1, 1], 5);
        Assert.Equal(0.0025, (double)result[2, 1], 5);
    }

    [Fact]
    public void EditEyeGazeVerticalAppliesRegardlessOfHorizontalSign()
    {
        var result = FaceEditor.EditEyeGaze(ZeroExpression(), 0.0, 1.0);

        // numpy.interp(1, [-1, 1], [-0.0025, 0.0025]) == 0.0025.
        Assert.Equal(0.0025, (double)result[1, 1], 5);
        Assert.Equal(-0.0025, (double)result[2, 1], 5);
        Assert.Equal(-0.010, (double)result[11, 1], 5);
        Assert.Equal(-0.005, (double)result[13, 1], 5);
    }

    [Fact]
    public void EditEyeOpenReturnsZeroDeltaWithoutCallingTheModelWhenRatioIsZero()
    {
        var landmark68 = new float[68, 2];
        var result = FaceEditor.EditEyeOpen(null, ZeroExpression(), landmark68, 0.0);

        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal(0f, result[i, c]);
            }
        }
    }

    [Fact]
    public void EditEyeOpenThrowsWithoutASessionWhenRatioIsNonZero()
    {
        var landmark68 = new float[68, 2];
        Assert.Throws<ArgumentNullException>(() => FaceEditor.EditEyeOpen(null, ZeroExpression(), landmark68, 0.5));
    }

    [Fact]
    public void EditLipOpenReturnsZeroDeltaWithoutCallingTheModelWhenRatioIsZero()
    {
        var landmark68 = new float[68, 2];
        var result = FaceEditor.EditLipOpen(null, ZeroExpression(), landmark68, 0.0);

        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal(0f, result[i, c]);
            }
        }
    }

    [Fact]
    public void EditLipOpenThrowsWithoutASessionWhenRatioIsNonZero()
    {
        var landmark68 = new float[68, 2];
        Assert.Throws<ArgumentNullException>(() => FaceEditor.EditLipOpen(null, ZeroExpression(), landmark68, -0.3));
    }

    [Fact]
    public void EditMouthGrimPositiveBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditMouthGrim(ZeroExpression(), 1.0);

        Assert.Equal(-0.005, (double)result[17, 2], 5);
        Assert.Equal(0.01, (double)result[19, 2], 5);
        Assert.Equal(-0.06, (double)result[20, 1], 5);
        Assert.Equal(-0.03, (double)result[20, 2], 5);
    }

    [Fact]
    public void EditMouthGrimNegativeBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditMouthGrim(ZeroExpression(), -1.0);

        Assert.Equal(0.05, (double)result[19, 1], 5);
        Assert.Equal(0.02, (double)result[19, 2], 5);
        Assert.Equal(0.03, (double)result[20, 2], 5);
        // Positive-branch-only field stays untouched on the negative branch.
        Assert.Equal(0.0, (double)result[17, 2], 5);
    }

    [Fact]
    public void EditMouthPositionHorizontalAlwaysAppliesRegardlessOfVerticalSign()
    {
        var result = FaceEditor.EditMouthPosition(ZeroExpression(), 1.0, 0.0);

        Assert.Equal(0.05, (double)result[19, 0], 5);
        Assert.Equal(0.04, (double)result[20, 0], 5);
    }

    [Fact]
    public void EditMouthPoutPositiveBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditMouthPout(ZeroExpression(), 1.0);

        Assert.Equal(-0.022, (double)result[19, 1], 5);
        Assert.Equal(0.025, (double)result[19, 2], 5);
        Assert.Equal(-0.002, (double)result[20, 2], 5);
    }

    [Fact]
    public void EditMouthPurseNegativeBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditMouthPurse(ZeroExpression(), -1.0);

        Assert.Equal(0.02, (double)result[14, 1], 5);
        Assert.Equal(-0.01, (double)result[17, 2], 5);
        Assert.Equal(0.015, (double)result[19, 2], 5);
        Assert.Equal(0.002, (double)result[20, 2], 5);
    }

    [Fact]
    public void EditMouthSmilePositiveBranchMatchesPythonInterp()
    {
        var result = FaceEditor.EditMouthSmile(ZeroExpression(), 1.0);

        Assert.Equal(-0.015, (double)result[20, 1], 5);
        Assert.Equal(-0.025, (double)result[14, 1], 5);
        Assert.Equal(0.01, (double)result[17, 1], 5);
        Assert.Equal(0.004, (double)result[17, 2], 5);
        Assert.Equal(-0.0045, (double)result[3, 1], 5);
        Assert.Equal(-0.0045, (double)result[7, 1], 5);
    }

    [Fact]
    public void EditHeadRotationClampsToPythonLimitAngleRange()
    {
        // face_editor_head_yaw = 1 -> interp(1, [-1,1], [60,-60]) == -60, so edit_yaw = yaw - 60.
        // With yaw = 0, limit_angle clamps edit_yaw to the wider of {-60, min(yaw,-60)} == -60
        // (calculate_euler_limits widens the negative bound to at most -60 when yaw >= 0).
        var rotation = FaceEditor.EditHeadRotation(0f, 0f, 0f, 0.0, 1.0, 0.0);

        var expectedRotation = LivePortrait.CreateRotation(0f, -60f, 0f);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Assert.Equal((double)expectedRotation[i, j], (double)rotation[i, j], 5);
            }
        }
    }
}
