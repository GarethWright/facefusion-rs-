using FaceFusion.Processors;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Regression for a defect that blocked the flagship processor.
///
/// Python's <c>explode_pixel_boost</c> is a pure numpy stack/reshape/transpose and is
/// therefore dtype-agnostic. The C# port added a <c>CV_8UC3</c> assertion with no
/// counterpart in the Python, which threw on every real frame: <c>FaceSwapper.SwapFace</c>
/// passes the FLOAT output of <c>NormalizeCropFrame</c> straight in, deliberately — that
/// method's own remarks say never to cast back to uint8 first.
///
/// The existing face_swapper parity tests never caught it because they exercise
/// <c>ForwardSwapFace</c> directly rather than the full
/// <c>ProcessFrame → SwapFace → ExplodePixelBoost</c> path, so the whole pipeline had
/// never run end to end.
/// </summary>
public sealed class PixelBoostDtypeTests
{
    private const int PixelBoostTotal = 2;
    private static readonly Size ModelSize = new(4, 4);
    private static readonly Size BoostSize = new(8, 8);

    /// <summary>
    /// The interleave must be identical whatever the element type — that is what makes it
    /// a faithful port of a numpy reshape.
    /// </summary>
    [Theory]
    [InlineData(0)] // CV_8UC3
    [InlineData(1)] // CV_32FC3
    [InlineData(2)] // CV_64FC3
    public void ExplodeAcceptsEveryChannelTypeAndPreservesIt(int typeIndex)
    {
        var matType = typeIndex switch
        {
            0 => MatType.CV_8UC3,
            1 => MatType.CV_32FC3,
            _ => MatType.CV_64FC3
        };

        var frames = new List<Mat>();

        try
        {
            for (var index = 0; index < PixelBoostTotal * PixelBoostTotal; index++)
            {
                // A distinct constant per sub-image, so the interleave is checkable.
                frames.Add(new Mat(ModelSize.Height, ModelSize.Width, matType, Scalar.All(index + 1)));
            }

            using var exploded = PixelBoost.ExplodePixelBoost(frames, PixelBoostTotal, ModelSize, BoostSize);

            Assert.Equal(matType, exploded.Type());
            Assert.Equal(BoostSize.Height, exploded.Rows);
            Assert.Equal(BoostSize.Width, exploded.Cols);

            // Sub-image (pbh, pbw) owns destination pixels where row % total == pbh and
            // col % total == pbw, mirroring numpy's transpose(2, 0, 3, 1, 4).
            // At<T> reinterprets raw bytes, so read through a common depth rather than
            // calling At<double> on an 8U or 32F Mat.
            using var firstChannel = new Mat();
            Cv2.ExtractChannel(exploded, firstChannel, 0);
            using var channels = new Mat();
            firstChannel.ConvertTo(channels, MatType.CV_64FC1);

            for (var pbh = 0; pbh < PixelBoostTotal; pbh++)
            {
                for (var pbw = 0; pbw < PixelBoostTotal; pbw++)
                {
                    var expected = (pbh * PixelBoostTotal) + pbw + 1;
                    var actual = channels.At<double>(pbh, pbw);

                    Assert.Equal(expected, (int)Math.Round(actual));
                }
            }
        }
        finally
        {
            foreach (var frame in frames)
            {
                frame.Dispose();
            }
        }
    }

    /// <summary>A genuinely unsupported type still fails loudly rather than silently.</summary>
    [Fact]
    public void ExplodeRejectsUnsupportedType()
    {
        var frames = new List<Mat>();

        try
        {
            for (var index = 0; index < PixelBoostTotal * PixelBoostTotal; index++)
            {
                frames.Add(new Mat(ModelSize.Height, ModelSize.Width, MatType.CV_16UC3, Scalar.All(1)));
            }

            Assert.Throws<ArgumentException>(() =>
                PixelBoost.ExplodePixelBoost(frames, PixelBoostTotal, ModelSize, BoostSize));
        }
        finally
        {
            foreach (var frame in frames)
            {
                frame.Dispose();
            }
        }
    }
}
