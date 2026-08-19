using FaceFusion.Processors;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of ground-truth checks for
/// <c>facefusion/processors/modules/background_remover/{core,types,choices}.py</c>. There is no
/// <c>tests/test_background_remover.py</c> upstream, so this file exercises
/// <see cref="BackgroundRemover"/>'s pure per-pixel post-processing
/// (<c>normalize_vision_mask</c>, <c>apply_fill_color</c>, <c>apply_despill_color</c>) against
/// real Python output captured ad hoc from the actual
/// <c>facefusion.processors.modules.background_remover.core</c> functions (opencv-python 5.0.0,
/// numpy 2.4.6) — see each test's comment for the exact Python invocation used.
///
/// <para>
/// <b>Real end-to-end ONNX-model ground truth</b> (the actual <c>modnet</c>/<c>u2net_cloth</c>
/// models run against a real frame, including <c>PrepareTempFrame</c>'s model-input tensor)
/// lives in <c>tests/FaceFusion.ParityTests/ProcessorParityTests2.cs</c> instead, gated to skip
/// when the model files are not present under <c>.assets/models/</c>. The
/// <c>corridor_key</c> model type (a second model output, its own prepare/merge path) is
/// documented as uncovered in the port report — <c>tools/parity/dump_processors2.py</c>
/// deliberately does not dump it either (see that script's own docstring), and this file does
/// not add coverage for it beyond what its shared helpers below already exercise.
/// </para>
///
/// <para>
/// <b>Tolerance.</b> This is plain scalar/per-pixel arithmetic with a final truncating
/// <c>uint8</c> cast — per PARITY_HARNESS.md, expect exact equality (not even the usual
/// OpenCV-resize-interpolation caveat applies, since none of these three functions resize).
/// </para>
/// </summary>
public sealed class BackgroundRemoverTests
{
    private static Mat MakeTinyFrame()
    {
        var mat = new Mat(2, 2, MatType.CV_8UC3);
        var pixels = new[]
        {
            new Vec3b { Item0 = 10, Item1 = 20, Item2 = 200 },
            new Vec3b { Item0 = 230, Item1 = 100, Item2 = 30 },
            new Vec3b { Item0 = 50, Item1 = 200, Item2 = 50 },
            new Vec3b { Item0 = 0, Item1 = 0, Item2 = 0 },
        };
        mat.SetArray(pixels);
        return mat;
    }

    private static byte[] MatToBgrBytes(Mat mat)
    {
        mat.GetArray(out Vec3b[] pixels);
        var result = new byte[pixels.Length * 3];
        for (var i = 0; i < pixels.Length; i++)
        {
            result[(i * 3) + 0] = pixels[i].Item0;
            result[(i * 3) + 1] = pixels[i].Item1;
            result[(i * 3) + 2] = pixels[i].Item2;
        }

        return result;
    }

    // -----------------------------------------------------------------
    // apply_fill_color
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// state_manager.init_item('background_remover_fill_color', (0, 255, 0, 200))  # R,G,B,A
    /// apply_fill_color(frame, mask)  # mask = [[0, 128], [255, 64]]
    /// </code>
    /// on the 2x2 frame from <see cref="MakeTinyFrame"/>. Exercises every one of the four mask
    /// values (fully background, half, fully foreground, mostly background) against a
    /// non-trivial alpha, so a swapped R/B channel or an inverted mask direction would show up
    /// immediately.
    /// </summary>
    [Fact]
    public void ApplyFillColorMatchesPython()
    {
        byte[] expected = { 2, 204, 43, 140, 160, 18, 50, 200, 50, 0, 149, 0 };

        using var frame = MakeTinyFrame();
        using var mask = new Mat(2, 2, MatType.CV_8UC1);
        mask.SetArray(new byte[] { 0, 128, 255, 64 });

        var fillColor = new Color(Red: 0, Green: 255, Blue: 0, Alpha: 200);
        using var filled = BackgroundRemover.ApplyFillColor(frame, mask, fillColor);

        Assert.Equal(expected, MatToBgrBytes(filled));
    }

    // -----------------------------------------------------------------
    // apply_despill_color
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// state_manager.init_item('background_remover_despill_color', (10, 240, 30, 150))  # R,G,B,A
    /// apply_despill_color(frame)
    /// </code>
    /// A non-uniform despill colour (<c>R != G != B</c>) so the per-channel
    /// <c>color_weight</c>/<c>color_limit</c> pairing (see <see cref="BackgroundRemover"/>'s
    /// class remarks on the <c>numpy.roll</c> channel arithmetic) is actually distinguishable
    /// from a symmetric one.
    /// </summary>
    [Fact]
    public void ApplyDespillColorMatchesPython()
    {
        byte[] expected = { 10, 20, 195, 217, 100, 30, 50, 111, 50, 0, 0, 0 };

        using var frame = MakeTinyFrame();
        var despillColor = new Color(Red: 10, Green: 240, Blue: 30, Alpha: 150);
        using var despilled = BackgroundRemover.ApplyDespillColor(frame, despillColor);

        Assert.Equal(expected, MatToBgrBytes(despilled));
    }

    // -----------------------------------------------------------------
    // normalize_vision_mask
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth from:
    /// <code>
    /// raw_mask = numpy.array([[-0.5, 0.3], [1.5, 0.7]], dtype=float32)
    /// normalize_vision_mask(raw_mask)  # -&gt; [0, 76, 255, 178]
    /// </code>
    /// Exercises both clip boundaries (<c>-0.5 -&gt; 0</c>, <c>1.5 -&gt; 1 -&gt; 255</c>) plus two
    /// interior values, confirming the <c>clip(0, 1) * 255</c> order (not <c>* 255</c> then
    /// clip to <c>[0, 255]</c> — though for these inputs the two orders agree, a genuinely wrong
    /// clip range would not produce <c>[0, 76, 255, 178]</c>).
    /// </summary>
    [Fact]
    public void NormalizeVisionMaskMatchesPython()
    {
        byte[] expected = { 0, 76, 255, 178 };
        float[] rawMask = { -0.5f, 0.3f, 1.5f, 0.7f };

        using var mask = BackgroundRemover.NormalizeVisionMask(rawMask, height: 2, width: 2);

        mask.GetArray(out byte[] actual);
        Assert.Equal(expected, actual);
    }
}
