using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/processors/pixel_boost.py</c> — the "pixel boost" trick <c>face_swapper</c>
/// (and, per its docstring, any processor with a fixed model input resolution) uses to run a
/// small-resolution model against a larger crop without literally resizing it down: the crop is
/// decomposed into <c>pixel_boost_total ** 2</c> phase-shifted sub-images at the model's native
/// resolution (implode), each is run through the model independently, and the outputs are
/// reassembled (explode) into one full-resolution frame.
///
/// <para>
/// <b>What the reshape/transpose actually computes.</b> Python's
/// <c>crop_vision_frame.reshape(model_size[0], pixel_boost_total, model_size[1], pixel_boost_total, 3)</c>
/// splits each spatial axis of length <c>model_size[axis] * pixel_boost_total</c> as
/// <c>(outer, inner)</c> in C order, i.e. a source pixel at row <c>h</c> maps to
/// <c>(outer = h / pixel_boost_total, inner = h % pixel_boost_total)</c> — <b>not</b> a
/// coarse block tiling. The following <c>transpose(1, 3, 0, 2, 4)</c> moves the two
/// <c>inner</c> axes (the phase offsets <c>pbh</c>/<c>pbw</c>) to the front, so
/// <c>implode_pixel_boost</c> is a strided "pixel unshuffle": sub-image <c>i = pbh *
/// pixel_boost_total + pbw</c> is <c>crop[pbh::pixel_boost_total, pbw::pixel_boost_total]</c> —
/// every <c>pixel_boost_total</c>-th pixel starting at offset <c>(pbh, pbw)</c>, covering every
/// phase exactly once across the <c>pixel_boost_total ** 2</c> sub-images.
/// <see cref="ExplodePixelBoost"/> is the exact inverse: it scatters each sub-image's pixels
/// back to their strided positions in the full-resolution frame.
/// </para>
///
/// <para>
/// When <c>pixel_boost_total == 1</c> (the default/first pixel-boost choice for every model —
/// see <c>FaceSwapper.FaceSwapperPixelBoostChoices</c> — is always the model's own native size,
/// e.g. <c>"256x256"</c> for a 256x256 model), both functions degenerate to the identity: one
/// sub-image, itself the whole crop.
/// </para>
///
/// <para>
/// <b>Mat / ownership convention.</b> Same as <c>FaceFusion.Face.FaceHelper</c>: pixel data is
/// <see cref="Mat"/> (<c>CV_8UC3</c>, matching every real caller — <c>crop_vision_frame</c> is
/// always an 8-bit BGR frame at this point in the swap pipeline), native memory, caller-owned,
/// parameters are never disposed by these methods.
/// </para>
/// </summary>
public static class PixelBoost
{
    /// <summary>
    /// Python: <c>implode_pixel_boost</c>. Returns <paramref name="pixelBoostTotal"/> squared
    /// sub-images of size <paramref name="modelSize"/>, in Python's
    /// <c>pbh * pixel_boost_total + pbw</c> order (see class remarks). Caller owns every
    /// returned <see cref="Mat"/> and must dispose them; does not take ownership of
    /// <paramref name="cropVisionFrame"/>.
    /// </summary>
    public static IReadOnlyList<Mat> ImplodePixelBoost(Mat cropVisionFrame, int pixelBoostTotal, Size modelSize)
    {
        if (cropVisionFrame.Type() != MatType.CV_8UC3)
        {
            throw new ArgumentException("ImplodePixelBoost requires a CV_8UC3 vision frame.", nameof(cropVisionFrame));
        }

        if (cropVisionFrame.Rows != modelSize.Height * pixelBoostTotal || cropVisionFrame.Cols != modelSize.Width * pixelBoostTotal)
        {
            throw new ArgumentException(
                $"cropVisionFrame ({cropVisionFrame.Cols}x{cropVisionFrame.Rows}) does not match modelSize ({modelSize.Width}x{modelSize.Height}) * pixelBoostTotal ({pixelBoostTotal}).",
                nameof(cropVisionFrame));
        }

        cropVisionFrame.GetArray(out Vec3b[] source); // row-major (H, W)

        var sourceWidth = cropVisionFrame.Cols;
        var modelHeight = modelSize.Height;
        var modelWidth = modelSize.Width;

        var result = new Mat[pixelBoostTotal * pixelBoostTotal];

        for (var pbh = 0; pbh < pixelBoostTotal; pbh++)
        {
            for (var pbw = 0; pbw < pixelBoostTotal; pbw++)
            {
                var subImage = new Vec3b[modelHeight * modelWidth];

                for (var mh = 0; mh < modelHeight; mh++)
                {
                    var sourceRow = (mh * pixelBoostTotal) + pbh;
                    var sourceRowBase = sourceRow * sourceWidth;
                    var destRowBase = mh * modelWidth;

                    for (var mw = 0; mw < modelWidth; mw++)
                    {
                        var sourceCol = (mw * pixelBoostTotal) + pbw;
                        subImage[destRowBase + mw] = source[sourceRowBase + sourceCol];
                    }
                }

                var mat = new Mat(modelHeight, modelWidth, MatType.CV_8UC3);
                mat.SetArray(subImage);
                result[(pbh * pixelBoostTotal) + pbw] = mat;
            }
        }

        return result;
    }

    /// <summary>
    /// Python: <c>explode_pixel_boost</c>. Reassembles <paramref name="tempVisionFrames"/>
    /// (Python: the list built by running the model on each <see cref="ImplodePixelBoost"/>
    /// sub-image) back into one <paramref name="pixelBoostSize"/> frame — the exact inverse of
    /// <see cref="ImplodePixelBoost"/>. Caller owns the returned <see cref="Mat"/>; does not
    /// take ownership of any element of <paramref name="tempVisionFrames"/>.
    /// </summary>
    public static Mat ExplodePixelBoost(
        IReadOnlyList<Mat> tempVisionFrames, int pixelBoostTotal, Size modelSize, Size pixelBoostSize)
    {
        if (tempVisionFrames.Count != pixelBoostTotal * pixelBoostTotal)
        {
            throw new ArgumentException(
                $"tempVisionFrames has {tempVisionFrames.Count} entries, expected pixelBoostTotal^2 = {pixelBoostTotal * pixelBoostTotal}.",
                nameof(tempVisionFrames));
        }

        var modelHeight = modelSize.Height;
        var modelWidth = modelSize.Width;
        var destWidth = pixelBoostSize.Width;

        var dest = new Vec3b[pixelBoostSize.Height * pixelBoostSize.Width];

        for (var pbh = 0; pbh < pixelBoostTotal; pbh++)
        {
            for (var pbw = 0; pbw < pixelBoostTotal; pbw++)
            {
                var subImage = tempVisionFrames[(pbh * pixelBoostTotal) + pbw];

                if (subImage.Type() != MatType.CV_8UC3 || subImage.Rows != modelHeight || subImage.Cols != modelWidth)
                {
                    throw new ArgumentException(
                        $"tempVisionFrames[{(pbh * pixelBoostTotal) + pbw}] must be a CV_8UC3 {modelWidth}x{modelHeight} frame.",
                        nameof(tempVisionFrames));
                }

                subImage.GetArray(out Vec3b[] sourcePixels);

                for (var mh = 0; mh < modelHeight; mh++)
                {
                    var destRow = (mh * pixelBoostTotal) + pbh;
                    var destRowBase = destRow * destWidth;
                    var sourceRowBase = mh * modelWidth;

                    for (var mw = 0; mw < modelWidth; mw++)
                    {
                        var destCol = (mw * pixelBoostTotal) + pbw;
                        dest[destRowBase + destCol] = sourcePixels[sourceRowBase + mw];
                    }
                }
            }
        }

        var result = new Mat(pixelBoostSize.Height, pixelBoostSize.Width, MatType.CV_8UC3);
        result.SetArray(dest);
        return result;
    }
}
