using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for
/// <c>facefusion/processors/modules/face_enhancer/{core,types,choices}.py</c>. There is no
/// <c>tests/test_face_enhancer.py</c> in the Python suite, so this file exercises the pure
/// arithmetic (<c>prepare_crop_frame</c>/<c>normalize_crop_frame</c>/<c>blend_paste_frame</c>)
/// against hand-computed numpy-equivalent values, plus the model-set/choices tables. The
/// real-ONNX end-to-end path is verified against ground truth dumped from the real
/// <c>facefusion.processors.modules.face_enhancer</c> module in
/// <c>tests/FaceFusion.ParityTests/EnhancerParityTests.cs</c>.
/// </summary>
public sealed class FaceEnhancerTests
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    [Fact]
    public void FaceEnhancerModelsListsEveryLiteralValue()
    {
        var models = FaceEnhancer.FaceEnhancerModels;
        Assert.Equal(9, models.Count);
        Assert.Contains(FaceEnhancerModel.Codeformer, models);
        Assert.Contains(FaceEnhancerModel.RestoreformerPlusPlus, models);
    }

    [Fact]
    public void FaceEnhancerBlendRangeMatchesPython()
    {
        // Python: create_int_range(0, 100, 1) -> [0, 1, ..., 100].
        Assert.Equal(101, FaceEnhancer.FaceEnhancerBlendRange.Count);
        Assert.Equal(0, FaceEnhancer.FaceEnhancerBlendRange[0]);
        Assert.Equal(100, FaceEnhancer.FaceEnhancerBlendRange[^1]);
    }

    [Fact]
    public void FaceEnhancerWeightRangeMatchesPython()
    {
        // Python: create_float_range(0.0, 1.0, 0.05) -> [0.0, 0.05, ..., 1.0] (21 values).
        Assert.Equal(21, FaceEnhancer.FaceEnhancerWeightRange.Count);
        Assert.Equal(0.0, FaceEnhancer.FaceEnhancerWeightRange[0]);
        Assert.Equal(1.0, FaceEnhancer.FaceEnhancerWeightRange[^1]);
    }

    // -----------------------------------------------------------------
    // Model set — every family's template/size, transcribed from create_static_model_set.
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(FaceEnhancerModel.Codeformer, WarpTemplate.Ffhq512, 512, 512)]
    [InlineData(FaceEnhancerModel.Gfpgan12, WarpTemplate.Ffhq512, 512, 512)]
    [InlineData(FaceEnhancerModel.Gfpgan13, WarpTemplate.Ffhq512, 512, 512)]
    [InlineData(FaceEnhancerModel.Gfpgan14, WarpTemplate.Ffhq512, 512, 512)]
    [InlineData(FaceEnhancerModel.GpenBfr256, WarpTemplate.Arcface128, 256, 256)]
    [InlineData(FaceEnhancerModel.GpenBfr512, WarpTemplate.Ffhq512, 512, 512)]
    [InlineData(FaceEnhancerModel.GpenBfr1024, WarpTemplate.Ffhq512, 1024, 1024)]
    [InlineData(FaceEnhancerModel.GpenBfr2048, WarpTemplate.Ffhq512, 2048, 2048)]
    [InlineData(FaceEnhancerModel.RestoreformerPlusPlus, WarpTemplate.Ffhq512, 512, 512)]
    public void CreateStaticModelSetMatchesPython(FaceEnhancerModel model, WarpTemplate expectedTemplate, int expectedWidth, int expectedHeight)
    {
        var modelSet = FaceEnhancer.CreateStaticModelSet(DownloadScope.Full);
        var options = modelSet[model];

        Assert.Equal(expectedTemplate, options.Template);
        Assert.Equal(expectedWidth, options.Size.Width);
        Assert.Equal(expectedHeight, options.Size.Height);
        Assert.EndsWith(".onnx", options.Source.Path, StringComparison.Ordinal);
        Assert.EndsWith(".hash", options.Hash.Path, StringComparison.Ordinal);
        Assert.Contains("github.com", options.Source.Url, StringComparison.Ordinal);
    }

    [Fact]
    public void GetModelOptionsMatchesCreateStaticModelSet()
    {
        var expected = FaceEnhancer.CreateStaticModelSet(DownloadScope.Full)[FaceEnhancerModel.GpenBfr256];
        var actual = FaceEnhancer.GetModelOptions(FaceEnhancerModel.GpenBfr256);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void PreCheckIsFalseWhenModelFilesAreAbsent()
    {
        // No real .onnx download happens in this test container by default; if the fixture
        // download already ran (parity tests), this documents the true/false split rather
        // than asserting a fixed value.
        var result = FaceEnhancer.PreCheck(FaceEnhancerModel.GpenBfr256);
        Assert.Equal(FaceFusion.Core.FileSystem.IsFile(FaceEnhancer.GetModelOptions(FaceEnhancerModel.GpenBfr256).Source.Path), result);
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame — BGR->RGB, /255, (x - 0.5) / 0.5, HWC->CHW
    // -----------------------------------------------------------------

    [Fact]
    public void PrepareCropFrameReversesChannelsAndNormalizesToMinusOneOne()
    {
        using var crop = new Mat(1, 1, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(10, 20, 30)); // B=10, G=20, R=30

        var (chw, height, width) = FaceEnhancer.PrepareCropFrame(crop);

        Assert.Equal(1, height);
        Assert.Equal(1, width);
        Assert.Equal(3, chw.Length);

        var expectedR = ((30 / 255f) - 0.5f) / 0.5f;
        var expectedG = ((20 / 255f) - 0.5f) / 0.5f;
        var expectedB = ((10 / 255f) - 0.5f) / 0.5f;

        // CHW layout for a 1x1 crop: channel 0 = R, channel 1 = G, channel 2 = B (post BGR->RGB).
        Assert.Equal(expectedR, chw[0], 1e-6f);
        Assert.Equal(expectedG, chw[1], 1e-6f);
        Assert.Equal(expectedB, chw[2], 1e-6f);
    }

    [Fact]
    public void PrepareCropFrameBoundaryValuesMapToExactlyPlusMinusOne()
    {
        using var crop = new Mat(1, 1, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(0, 0, 255)); // black->0, white channel->255

        var (chw, _, _) = FaceEnhancer.PrepareCropFrame(crop);

        Assert.Equal(1f, chw[0], 1e-6f); // R = 255 -> 1.0
        Assert.Equal(-1f, chw[1], 1e-6f); // G = 0 -> -1.0
        Assert.Equal(-1f, chw[2], 1e-6f); // B = 0 -> -1.0
    }

    // -----------------------------------------------------------------
    // normalize_crop_frame — clip(-1,1), (x+1)/2, CHW->HWC, *255, round, uint8, RGB->BGR
    // -----------------------------------------------------------------

    [Fact]
    public void NormalizeCropFrameRoundTripsPrepareCropFrame()
    {
        using var crop = new Mat(1, 1, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(10, 20, 30));

        var (chw, height, width) = FaceEnhancer.PrepareCropFrame(crop);
        using var normalized = FaceEnhancer.NormalizeCropFrame(chw, height, width);

        // prepare -> normalize is a lossless round trip for 8-bit inputs (up to the /255
        // quantization, which prepare_crop_frame/normalize_crop_frame are mutual inverses
        // for), so the original pixel should come back exactly.
        var pixel = normalized.At<Vec3b>(0, 0);
        Assert.Equal(new Vec3b(10, 20, 30), pixel);
    }

    [Fact]
    public void NormalizeCropFrameClipsOutOfRangeValues()
    {
        // Simulate a model output that overshoots [-1, 1] — normalize_crop_frame must clip
        // before scaling, not saturate afterward (these differ for values beyond +-3).
        float[] chw = { 5f, -5f, 0f };

        using var normalized = FaceEnhancer.NormalizeCropFrame(chw, 1, 1);
        var pixel = normalized.At<Vec3b>(0, 0);

        // R channel clipped to 1 -> 255; G channel clipped to -1 -> 0; B channel 0 -> 127 or 128
        // depending on round-half-to-even ((0 + 1) / 2 * 255 = 127.5 -> rounds to 128, even).
        Assert.Equal(128, pixel.Item0); // B (post RGB->BGR, this is the third CHW channel)
        Assert.Equal(0, pixel.Item1); // G
        Assert.Equal(255, pixel.Item2); // R
    }

    // -----------------------------------------------------------------
    // blend_paste_frame — AddWeighted with the Python 1 - (blend/100) factor
    // -----------------------------------------------------------------

    [Theory]
    [InlineData(100, 1.0)] // full blend towards paste_vision_frame
    [InlineData(0, 0.0)] // no blend: pure temp_vision_frame
    [InlineData(80, 0.8)]
    public void BlendPasteFrameUsesPythonBlendFormula(int faceEnhancerBlend, double expectedPasteWeight)
    {
        using var temp = new Mat(1, 1, MatType.CV_8UC3, new Scalar(0, 0, 0));
        using var paste = new Mat(1, 1, MatType.CV_8UC3, new Scalar(200, 200, 200));

        using var blended = FaceEnhancer.BlendPasteFrame(temp, paste, faceEnhancerBlend);
        var pixel = blended.At<Vec3b>(0, 0);

        var expected = (byte)Math.Round(200 * expectedPasteWeight, MidpointRounding.AwayFromZero);
        // cv2.addWeighted rounds to nearest (saturate_cast), small tolerance for rounding mode.
        Assert.InRange(pixel.Item0, Math.Max(0, expected - 1), Math.Min(255, expected + 1));
    }

    // -----------------------------------------------------------------
    // has_weight_input
    // -----------------------------------------------------------------

    [Fact(Skip = "requires a loaded InferenceSession (real .onnx model) — exercised via the real gpen_bfr_256 model in EnhancerParityTests")]
    public void HasWeightInputReadsSessionInputNames()
    {
        // Left as a documented skip: constructing an InferenceSession needs a real .onnx
        // model file, which this unit test file (unlike the parity tests) is not gated on.
        _ = typeof(FaceEnhancer).GetMethod(nameof(FaceEnhancer.HasWeightInput));
    }
}
