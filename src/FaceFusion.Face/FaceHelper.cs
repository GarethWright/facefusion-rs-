using System.Collections.Concurrent;
using FaceFusion.Tensors;
using FaceFusion.Types;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_helper.py</c> — the geometric heart of the face pipeline: warping
/// faces into a canonical crop frame and pasting the processed crop back into the original
/// frame, plus the small numpy-only geometry helpers (anchors, distance decoding, NMS, angle
/// estimation) used by the detector/landmarker stages.
///
/// <para>
/// <b>Representation choices</b> (per docs/DOTNET_PORT_PLAN.md §4 — .NET has no ndarray, so each
/// category of array data is routed to the tool that fits it best):
/// <list type="bullet">
/// <item><description><b>VisionFrame / Mask</b> -&gt; <see cref="Mat"/>, matching
/// <c>FaceFusion.Vision.Vision</c>'s convention: pixel data stays in native memory, every
/// returned <see cref="Mat"/> is caller-owned and documented as such, parameters are never
/// disposed by the callee.</description></item>
/// <item><description><b>Affine/rotation/homogeneous matrices</b> (the Python <c>Matrix</c>
/// alias) -&gt; <see cref="Mat"/> (2x3 or 3x3, <c>CV_64F</c>) — these matrices are produced by and
/// fed straight back into OpenCV calls (<c>WarpAffine</c>, <c>InvertAffineTransform</c>, …), so
/// keeping them as <see cref="Mat"/> avoids a marshal round-trip and keeps the same double
/// precision cv2 uses internally for every affine matrix. Caller-owned, same as VisionFrame.
/// </description></item>
/// <item><description><b>Batches of 2-D points</b> (the Python <c>Points</c>/<c>Anchors</c>/
/// <c>FaceLandmark5</c>/<c>FaceLandmark68</c> aliases) -&gt; <c>float[,]</c> of shape
/// <c>(N, 2)</c>, row-major, mirroring the numpy array shape exactly (landmarks arriving from a
/// detector are float32 in Python, so these stay <see cref="float"/>, never promoted to
/// <see cref="double"/>, per PORT_CONVENTIONS.md rule 6 / the assignment's float32-vs-float64
/// instruction).</description></item>
/// <item><description><b>A single bounding box</b> (the Python <c>BoundingBox</c> alias) -&gt;
/// <c>float[]</c> of length 4, <c>[x1, y1, x2, y2]</c>.</description></item>
/// <item><description><b>A batch of per-anchor distances</b> (the Python <c>Distance</c> alias)
/// -&gt; <c>float[,]</c> of shape <c>(N, 4)</c> or <c>(N, 10)</c> depending on the function, same
/// as Points.</description></item>
/// <item><description><b>WARP_TEMPLATE_SET constants</b> -&gt; <c>double[,]</c>. These are
/// literal float64 constants in Python (<c>numpy.array</c> of Python floats defaults to
/// float64) that define alignment — every digit is copied exactly, and they stay double rather
/// than being narrowed to float32, matching the Python dtype.</description></item>
/// </list>
/// </para>
///
/// <para>
/// <b>A real float32/int64 promotion bug reproduced deliberately.</b> Python's
/// <c>calculate_paste_area</c> builds <c>crop_points</c> via a bare
/// <c>numpy.array([[0,0], …])</c> — with no explicit dtype this is <c>int64</c>, not float. When
/// that int64 array is fed through <c>cv2.transform</c> (inside <c>transform_points</c>) against
/// a float64 affine matrix, OpenCV downcasts the (unsupported) int64 input to <c>CV_32S</c> and
/// performs the transform with the output rounded to the nearest int32 (ties-to-even, i.e.
/// <c>cvRound</c>) — the fractional part of the transform is thrown away as part of the corner
/// computation, not after it. This is confirmed empirically (see FaceHelperTests) against real
/// <c>cv2.transform</c>: the numpy <c>floor</c>/<c>ceil</c> calls that follow in Python are then
/// no-ops (their input is already integral). This is reproduced exactly by
/// <see cref="CalculatePasteArea"/> below via <see cref="RoundToEvenInt"/> rather than doing the
/// corner transform in double precision throughout, per PORT_CONVENTIONS.md rule 1
/// ("reproduce the oddity"). <see cref="TransformPoints"/> itself stays float64/float32 (its
/// other caller, <see cref="TransformBoundingBox"/>, passes float landmark-derived coordinates
/// in Python, never integers) — the int-rounding behaviour is local to how
/// <c>calculate_paste_area</c> calls it, so it is inlined there rather than baked into the
/// shared helper.
/// </para>
///
/// <para>
/// <b>PasteBack's dtype assumption.</b> Every real call site in the Python pipeline passes an
/// 8-bit BGR <c>VisionFrame</c> (<c>CV_8UC3</c>) and a float32 single-channel <c>Mask</c> in
/// <c>[0, 1]</c> (<c>CV_32FC1</c>), so <see cref="PasteBack"/> requires exactly those types and
/// throws <see cref="ArgumentException"/> otherwise, rather than silently mishandling an
/// unexpected layout. The final <c>.astype(temp_vision_frame.dtype)</c> in Python truncates
/// toward zero (numpy float-&gt;uint8 cast semantics), which is different from OpenCV's
/// <c>saturate_cast</c> (round-to-nearest) — <see cref="PasteBack"/> replicates numpy's
/// truncation exactly via a manual per-pixel loop rather than an OpenCV <c>ConvertTo</c>.
/// </para>
/// </summary>
public static class FaceHelper
{
    private static readonly IReadOnlyDictionary<WarpTemplate, double[,]> WarpTemplateSetData =
        new Dictionary<WarpTemplate, double[,]>
        {
            [WarpTemplate.Arcface112V1] = new[,]
            {
                { 0.35473214, 0.45658929 },
                { 0.64526786, 0.45658929 },
                { 0.50000000, 0.61154464 },
                { 0.37913393, 0.77687500 },
                { 0.62086607, 0.77687500 },
            },
            [WarpTemplate.Arcface112V2] = new[,]
            {
                { 0.34191607, 0.46157411 },
                { 0.65653393, 0.45983393 },
                { 0.50022500, 0.64050536 },
                { 0.37097589, 0.82469196 },
                { 0.63151696, 0.82325089 },
            },
            [WarpTemplate.Arcface128] = new[,]
            {
                { 0.36167656, 0.40387734 },
                { 0.63696719, 0.40235469 },
                { 0.50019687, 0.56044219 },
                { 0.38710391, 0.72160547 },
                { 0.61507734, 0.72034453 },
            },
            [WarpTemplate.DflWholeFace] = new[,]
            {
                { 0.35342266, 0.39285716 },
                { 0.62797622, 0.39285716 },
                { 0.48660713, 0.54017860 },
                { 0.38839287, 0.68750011 },
                { 0.59821427, 0.68750011 },
            },
            [WarpTemplate.Ffhq512] = new[,]
            {
                { 0.37691676, 0.46864664 },
                { 0.62285697, 0.46912813 },
                { 0.50123859, 0.61331904 },
                { 0.39308822, 0.72541100 },
                { 0.61150205, 0.72490465 },
            },
            [WarpTemplate.Mtcnn512] = new[,]
            {
                { 0.36562865, 0.46733799 },
                { 0.63305391, 0.46585885 },
                { 0.50019127, 0.61942959 },
                { 0.39032951, 0.77598822 },
                { 0.61178945, 0.77476328 },
            },
            [WarpTemplate.Styleganex384] = new[,]
            {
                { 0.42353745, 0.52289879 },
                { 0.57725008, 0.52319972 },
                { 0.50123859, 0.61331904 },
                { 0.43364461, 0.68337652 },
                { 0.57015325, 0.68306005 },
            },
        };

    /// <summary>
    /// Python: <c>WARP_TEMPLATE_SET</c>. Every digit copied exactly from the Python literals;
    /// see the class-level remarks for why these stay <see cref="double"/>. Returns a defensive
    /// copy of the requested template so callers cannot mutate the shared constant.
    /// </summary>
    public static double[,] GetWarpTemplate(WarpTemplate warpTemplate)
    {
        var source = WarpTemplateSetData[warpTemplate];
        var copy = new double[source.GetLength(0), source.GetLength(1)];
        Array.Copy(source, copy, source.Length);
        return copy;
    }

    // -----------------------------------------------------------------
    // Warping
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>estimate_matrix_by_face_landmark_5</c>. Caller owns the returned
    /// <see cref="Mat"/> (2x3, <c>CV_64F</c>) and must dispose it. <paramref name="faceLandmark5"/>
    /// is <c>(5, 2)</c>, float32 (matching detector output dtype).
    /// </summary>
    public static Mat EstimateMatrixByFaceLandmark5(float[,] faceLandmark5, WarpTemplate warpTemplate, Size cropSize)
    {
        if (faceLandmark5.GetLength(0) != 5 || faceLandmark5.GetLength(1) != 2)
        {
            throw new ArgumentException("faceLandmark5 must have shape (5, 2).", nameof(faceLandmark5));
        }

        var template = WarpTemplateSetData[warpTemplate];

        // `warp_template_norm = WARP_TEMPLATE_SET.get(warp_template) * crop_size` — this
        // multiply happens in float64 (the template's own dtype), matching Python exactly;
        // only the result is narrowed to float32 immediately before the cv2 call below (see
        // the class remarks: mixed float32/float64 point sets feed cv2.estimateAffinePartial2D
        // identically to both-float32, verified empirically).
        var from = new Point2f[5];
        var to = new Point2f[5];
        for (var i = 0; i < 5; i++)
        {
            from[i] = new Point2f(faceLandmark5[i, 0], faceLandmark5[i, 1]);
            to[i] = new Point2f((float)(template[i, 0] * cropSize.Width), (float)(template[i, 1] * cropSize.Height));
        }

        using var fromMat = InputArray.Create(from);
        using var toMat = InputArray.Create(to);
        var affineMatrix = Cv2.EstimateAffinePartial2D(
            fromMat,
            toMat,
            null,
            RobustEstimationAlgorithms.RANSAC,
            ransacReprojThreshold: 100);

        // Python: cv2.estimateAffinePartial2D(...)[0] can be None if a transform could not be
        // estimated; with 5 real landmark points this practically never happens, but guard
        // explicitly rather than silently returning a null Mat.
        return affineMatrix ?? throw new InvalidOperationException("cv2.estimateAffinePartial2D could not estimate a transform for the given landmarks.");
    }

    /// <summary>
    /// Python: <c>warp_face_by_face_landmark_5</c>. Caller owns both returned <see cref="Mat"/>
    /// values (crop frame and affine matrix). Does not take ownership of
    /// <paramref name="tempVisionFrame"/>.
    /// </summary>
    public static (Mat CropVisionFrame, Mat AffineMatrix) WarpFaceByFaceLandmark5(
        Mat tempVisionFrame, float[,] faceLandmark5, WarpTemplate warpTemplate, Size cropSize)
    {
        var affineMatrix = EstimateMatrixByFaceLandmark5(faceLandmark5, warpTemplate, cropSize);
        var cropVisionFrame = new Mat();
        Cv2.WarpAffine(tempVisionFrame, cropVisionFrame, affineMatrix, cropSize, InterpolationFlags.Area, BorderTypes.Replicate);
        return (cropVisionFrame, affineMatrix);
    }

    /// <summary>
    /// Python: <c>warp_face_by_bounding_box</c>. Caller owns both returned <see cref="Mat"/>
    /// values. <paramref name="boundingBox"/> is <c>[x1, y1, x2, y2]</c>.
    /// </summary>
    public static (Mat CropVisionFrame, Mat AffineMatrix) WarpFaceByBoundingBox(
        Mat tempVisionFrame, float[] boundingBox, Size cropSize)
    {
        if (boundingBox.Length != 4)
        {
            throw new ArgumentException("boundingBox must have length 4.", nameof(boundingBox));
        }

        var source = new[]
        {
            new Point2f(boundingBox[0], boundingBox[1]),
            new Point2f(boundingBox[2], boundingBox[1]),
            new Point2f(boundingBox[0], boundingBox[3]),
        };
        var target = new[]
        {
            new Point2f(0, 0),
            new Point2f(cropSize.Width, 0),
            new Point2f(0, cropSize.Height),
        };

        var affineMatrix = Cv2.GetAffineTransform(source, target);

        var interpolation = (boundingBox[2] - boundingBox[0] > cropSize.Width) || (boundingBox[3] - boundingBox[1] > cropSize.Height)
            ? InterpolationFlags.Area
            : InterpolationFlags.Linear;

        var cropVisionFrame = new Mat();
        Cv2.WarpAffine(tempVisionFrame, cropVisionFrame, affineMatrix, cropSize, interpolation);
        return (cropVisionFrame, affineMatrix);
    }

    /// <summary>
    /// Python: <c>warp_face_by_translation</c>. Caller owns both returned <see cref="Mat"/>
    /// values. <paramref name="translation"/> is <c>[tx, ty]</c>.
    /// </summary>
    public static (Mat CropVisionFrame, Mat AffineMatrix) WarpFaceByTranslation(
        Mat tempVisionFrame, double[] translation, double scale, Size cropSize)
    {
        if (translation.Length != 2)
        {
            throw new ArgumentException("translation must have length 2.", nameof(translation));
        }

        var affineMatrix = new Mat(2, 3, MatType.CV_64FC1);
        affineMatrix.Set(0, 0, scale);
        affineMatrix.Set(0, 1, 0d);
        affineMatrix.Set(0, 2, translation[0]);
        affineMatrix.Set(1, 0, 0d);
        affineMatrix.Set(1, 1, scale);
        affineMatrix.Set(1, 2, translation[1]);

        var cropVisionFrame = new Mat();
        Cv2.WarpAffine(tempVisionFrame, cropVisionFrame, affineMatrix, cropSize);
        return (cropVisionFrame, affineMatrix);
    }

    // -----------------------------------------------------------------
    // Paste back
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>paste_back</c>. Caller owns the returned <see cref="Mat"/>. Does not take
    /// ownership of any parameter. Requires <paramref name="tempVisionFrame"/> and
    /// <paramref name="cropVisionFrame"/> to be <c>CV_8UC3</c> and <paramref name="cropVisionMask"/>
    /// to be <c>CV_32FC1</c> — see the class-level remarks for why.
    /// </summary>
    public static Mat PasteBack(Mat tempVisionFrame, Mat cropVisionFrame, Mat cropVisionMask, Mat affineMatrix)
    {
        if (tempVisionFrame.Type() != MatType.CV_8UC3 || cropVisionFrame.Type() != MatType.CV_8UC3)
        {
            throw new ArgumentException("PasteBack requires CV_8UC3 vision frames.");
        }

        if (cropVisionMask.Type() != MatType.CV_32FC1)
        {
            throw new ArgumentException("PasteBack requires a CV_32FC1 mask.", nameof(cropVisionMask));
        }

        var (pasteBoundingBox, pasteMatrix) = CalculatePasteArea(tempVisionFrame, cropVisionFrame, affineMatrix);
        using var _ = pasteMatrix;

        var x1 = pasteBoundingBox[0];
        var y1 = pasteBoundingBox[1];
        var x2 = pasteBoundingBox[2];
        var y2 = pasteBoundingBox[3];
        var pasteWidth = x2 - x1;
        var pasteHeight = y2 - y1;
        var pasteSize = new Size(pasteWidth, pasteHeight);

        using var inverseVisionMaskRaw = new Mat();
        Cv2.WarpAffine(cropVisionMask, inverseVisionMaskRaw, pasteMatrix, pasteSize);

        using var inverseVisionFrame = new Mat();
        Cv2.WarpAffine(cropVisionFrame, inverseVisionFrame, pasteMatrix, pasteSize, InterpolationFlags.Linear, BorderTypes.Replicate);

        var resultVisionFrame = tempVisionFrame.Clone();

        for (var row = 0; row < pasteHeight; row++)
        {
            for (var col = 0; col < pasteWidth; col++)
            {
                // `.clip(0, 1)` from Python, applied per read rather than materialising a
                // separate clipped mask Mat.
                var maskValue = NumPy.Clip(inverseVisionMaskRaw.At<float>(row, col), 0f, 1f);
                var oneMinusMask = 1f - maskValue;

                var destRow = y1 + row;
                var destCol = x1 + col;
                var original = resultVisionFrame.At<Vec3b>(destRow, destCol);
                var warped = inverseVisionFrame.At<Vec3b>(row, col);

                var blended = new Vec3b
                {
                    Item0 = BlendChannel(original.Item0, warped.Item0, oneMinusMask, maskValue),
                    Item1 = BlendChannel(original.Item1, warped.Item1, oneMinusMask, maskValue),
                    Item2 = BlendChannel(original.Item2, warped.Item2, oneMinusMask, maskValue),
                };
                resultVisionFrame.Set(destRow, destCol, blended);
            }
        }

        return resultVisionFrame;
    }

    /// <summary>
    /// Blends one channel the way numpy does: <c>original * (1 - mask) + warped * mask</c> in
    /// float64 (numpy's default promotion for a uint8 array combined with a float array), then
    /// <c>.astype(uint8)</c> — which truncates toward zero, not round-to-nearest. Matches
    /// Python's <c>paste_vision_frame.astype(temp_vision_frame.dtype)</c> exactly (as opposed to
    /// OpenCV's own <c>saturate_cast</c>, which rounds).
    /// </summary>
    private static byte BlendChannel(byte original, byte warped, float oneMinusMask, float mask)
    {
        var value = ((double)original * oneMinusMask) + ((double)warped * mask);

        if (value <= 0)
        {
            return 0;
        }

        if (value >= 255)
        {
            return 255;
        }

        return (byte)value; // truncation toward zero, matching numpy .astype(uint8)
    }

    /// <summary>
    /// Python: <c>calculate_paste_area</c>. Returns <c>[x1, y1, x2, y2]</c> and a caller-owned
    /// <see cref="Mat"/> paste matrix. See the class-level remarks for the deliberate
    /// int32-rounding reproduction of Python's <c>cv2.transform</c> call on integer
    /// <c>crop_points</c>.
    /// </summary>
    public static (int[] PasteBoundingBox, Mat PasteMatrix) CalculatePasteArea(
        Mat tempVisionFrame, Mat cropVisionFrame, Mat affineMatrix)
    {
        var tempWidth = tempVisionFrame.Cols;
        var tempHeight = tempVisionFrame.Rows;
        var cropWidth = cropVisionFrame.Cols;
        var cropHeight = cropVisionFrame.Rows;

        var inverseMatrix = new Mat();
        Cv2.InvertAffineTransform(affineMatrix, inverseMatrix);

        var m00 = inverseMatrix.At<double>(0, 0);
        var m01 = inverseMatrix.At<double>(0, 1);
        var m02 = inverseMatrix.At<double>(0, 2);
        var m10 = inverseMatrix.At<double>(1, 0);
        var m11 = inverseMatrix.At<double>(1, 1);
        var m12 = inverseMatrix.At<double>(1, 2);

        Span<(int X, int Y)> corners = stackalloc (int, int)[4]
        {
            (0, 0), (cropWidth, 0), (cropWidth, cropHeight), (0, cropHeight),
        };

        var minX = int.MaxValue;
        var minY = int.MaxValue;
        var maxX = int.MinValue;
        var maxY = int.MinValue;

        foreach (var (cx, cy) in corners)
        {
            // cv2.transform on an int64 crop_points array against a float64 matrix downcasts
            // to CV_32S and rounds (ties-to-even) as part of the transform — see class remarks.
            var px = RoundToEvenInt((m00 * cx) + (m01 * cy) + m02);
            var py = RoundToEvenInt((m10 * cx) + (m11 * cy) + m12);

            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
        }

        // numpy.floor/.ceil are no-ops here since the values are already integral (see class
        // remarks), so they are intentionally omitted rather than ported literally.
        var x1 = Math.Clamp(minX, 0, tempWidth);
        var y1 = Math.Clamp(minY, 0, tempHeight);
        var x2 = Math.Clamp(maxX, 0, tempWidth);
        var y2 = Math.Clamp(maxY, 0, tempHeight);

        var pasteMatrix = inverseMatrix.Clone();
        pasteMatrix.Set(0, 2, pasteMatrix.At<double>(0, 2) - x1);
        pasteMatrix.Set(1, 2, pasteMatrix.At<double>(1, 2) - y1);
        inverseMatrix.Dispose();

        return (new[] { x1, y1, x2, y2 }, pasteMatrix);
    }

    /// <summary>Equivalent of OpenCV's <c>cvRound</c> (round-to-nearest, ties-to-even), saturated
    /// to the <see cref="int"/> range the way <c>saturate_cast&lt;int&gt;</c> would.</summary>
    private static int RoundToEvenInt(double value)
    {
        var rounded = Math.Round(value, MidpointRounding.ToEven);
        if (rounded >= int.MaxValue)
        {
            return int.MaxValue;
        }

        if (rounded <= int.MinValue)
        {
            return int.MinValue;
        }

        return (int)rounded;
    }

    // -----------------------------------------------------------------
    // Anchors / rotation
    // -----------------------------------------------------------------

    private static readonly ConcurrentDictionary<(int FeatureStride, int AnchorTotal, int StrideHeight, int StrideWidth), float[,]> StaticAnchorsCache = new();

    /// <summary>
    /// Python: <c>create_static_anchors</c> (<c>@lru_cache()</c>). Returns a fresh copy of the
    /// cached array on every call (same aliasing-safety reasoning as
    /// <c>FaceFusion.Vision.Vision.ReadStaticImage</c>'s clone-on-read cache).
    /// </summary>
    public static float[,] CreateStaticAnchors(int featureStride, int anchorTotal, int strideHeight, int strideWidth)
    {
        var key = (featureStride, anchorTotal, strideHeight, strideWidth);
        var cached = StaticAnchorsCache.GetOrAdd(key, static k =>
        {
            var (stride, total, height, width) = k;

            // numpy.mgrid[:stride_width, :stride_height] -> x[i, j] = i, y[i, j] = j (shape
            // (stride_width, stride_height)); anchors = stack((y, x), axis=-1) so
            // anchors[i, j] = [j, i]; reshape(-1, 2) flattens row-major (i outer, j inner).
            var baseAnchors = new float[width * height, 2];
            var idx = 0;
            for (var i = 0; i < width; i++)
            {
                for (var j = 0; j < height; j++)
                {
                    baseAnchors[idx, 0] = j * stride;
                    baseAnchors[idx, 1] = i * stride;
                    idx++;
                }
            }

            // stack([anchors] * anchor_total, axis=1).reshape((-1, 2)) -> each base anchor
            // repeated `total` times consecutively.
            var anchors = new float[baseAnchors.GetLength(0) * total, 2];
            var outIdx = 0;
            for (var n = 0; n < baseAnchors.GetLength(0); n++)
            {
                for (var a = 0; a < total; a++)
                {
                    anchors[outIdx, 0] = baseAnchors[n, 0];
                    anchors[outIdx, 1] = baseAnchors[n, 1];
                    outIdx++;
                }
            }

            return anchors;
        });

        var copy = new float[cached.GetLength(0), cached.GetLength(1)];
        Array.Copy(cached, copy, cached.Length);
        return copy;
    }

    /// <summary>
    /// Python: <c>create_rotation_matrix_and_size</c>. Caller owns the returned
    /// <see cref="Mat"/> (2x3, <c>CV_64F</c>).
    /// </summary>
    public static (Mat RotationMatrix, Size RotationSize) CreateRotationMatrixAndSize(int angle, Size size)
    {
        var center = new Point2f(size.Width / 2f, size.Height / 2f);
        var rotationMatrix = Cv2.GetRotationMatrix2D(center, angle, 1);

        var m00 = Math.Abs(rotationMatrix.At<double>(0, 0));
        var m01 = Math.Abs(rotationMatrix.At<double>(0, 1));
        var m10 = Math.Abs(rotationMatrix.At<double>(1, 0));
        var m11 = Math.Abs(rotationMatrix.At<double>(1, 1));

        var rotationSizeX = (m00 * size.Width) + (m01 * size.Height);
        var rotationSizeY = (m10 * size.Width) + (m11 * size.Height);

        rotationMatrix.Set(0, 2, rotationMatrix.At<double>(0, 2) + ((rotationSizeX - size.Width) * 0.5));
        rotationMatrix.Set(1, 2, rotationMatrix.At<double>(1, 2) + ((rotationSizeY - size.Height) * 0.5));

        // Python: int(rotation_size[0]), int(rotation_size[1]) — truncates toward zero.
        var rotationSize = new Size((int)rotationSizeX, (int)rotationSizeY);
        return (rotationMatrix, rotationSize);
    }

    // -----------------------------------------------------------------
    // Bounding boxes / points
    // -----------------------------------------------------------------

    /// <summary>Python: <c>create_bounding_box</c>. <paramref name="faceLandmark68"/> is <c>(68, 2)</c>.</summary>
    public static float[] CreateBoundingBox(float[,] faceLandmark68)
    {
        if (faceLandmark68.GetLength(0) != 68 || faceLandmark68.GetLength(1) != 2)
        {
            throw new ArgumentException("faceLandmark68 must have shape (68, 2).", nameof(faceLandmark68));
        }

        var minX = float.PositiveInfinity;
        var minY = float.PositiveInfinity;
        var maxX = float.NegativeInfinity;
        var maxY = float.NegativeInfinity;

        for (var i = 0; i < 68; i++)
        {
            var x = faceLandmark68[i, 0];
            var y = faceLandmark68[i, 1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        return NormalizeBoundingBox(new[] { minX, minY, maxX, maxY });
    }

    /// <summary>Python: <c>normalize_bounding_box</c>.</summary>
    public static float[] NormalizeBoundingBox(float[] boundingBox)
    {
        if (boundingBox.Length != 4)
        {
            throw new ArgumentException("boundingBox must have length 4.", nameof(boundingBox));
        }

        var x1 = boundingBox[0];
        var x2 = boundingBox[2];
        var y1 = boundingBox[1];
        var y2 = boundingBox[3];

        if (x1 > x2)
        {
            (x1, x2) = (x2, x1);
        }

        if (y1 > y2)
        {
            (y1, y2) = (y2, y1);
        }

        return new[] { x1, y1, x2, y2 };
    }

    /// <summary>
    /// Python: <c>transform_points</c>. <paramref name="points"/> is <c>(N, 2)</c> float32/64.
    /// This is the float path used by <see cref="TransformBoundingBox"/>; the int-array,
    /// round-to-nearest path used by <see cref="CalculatePasteArea"/> is deliberately not
    /// routed through this method — see the class-level remarks.
    /// </summary>
    public static float[,] TransformPoints(float[,] points, Mat matrix)
    {
        var count = points.GetLength(0);
        var result = new float[count, 2];

        var m00 = matrix.At<double>(0, 0);
        var m01 = matrix.At<double>(0, 1);
        var m02 = matrix.At<double>(0, 2);
        var m10 = matrix.At<double>(1, 0);
        var m11 = matrix.At<double>(1, 1);
        var m12 = matrix.At<double>(1, 2);

        for (var i = 0; i < count; i++)
        {
            var x = points[i, 0];
            var y = points[i, 1];
            result[i, 0] = (float)((m00 * x) + (m01 * y) + m02);
            result[i, 1] = (float)((m10 * x) + (m11 * y) + m12);
        }

        return result;
    }

    /// <summary>Python: <c>transform_bounding_box</c>.</summary>
    public static float[] TransformBoundingBox(float[] boundingBox, Mat matrix)
    {
        var points = new float[4, 2]
        {
            { boundingBox[0], boundingBox[1] },
            { boundingBox[2], boundingBox[1] },
            { boundingBox[2], boundingBox[3] },
            { boundingBox[0], boundingBox[3] },
        };

        var transformed = TransformPoints(points, matrix);

        var minX = float.PositiveInfinity;
        var minY = float.PositiveInfinity;
        var maxX = float.NegativeInfinity;
        var maxY = float.NegativeInfinity;
        for (var i = 0; i < 4; i++)
        {
            var x = transformed[i, 0];
            var y = transformed[i, 1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        return NormalizeBoundingBox(new[] { minX, minY, maxX, maxY });
    }

    /// <summary>
    /// Python: <c>distance_to_bounding_box</c>. <paramref name="points"/> is <c>(N, 2)</c>,
    /// <paramref name="distance"/> is <c>(N, 4)</c>; returns <c>(N, 4)</c>.
    /// </summary>
    public static float[,] DistanceToBoundingBox(float[,] points, float[,] distance)
    {
        var count = points.GetLength(0);
        var result = new float[count, 4];
        for (var i = 0; i < count; i++)
        {
            result[i, 0] = points[i, 0] - distance[i, 0];
            result[i, 1] = points[i, 1] - distance[i, 1];
            result[i, 2] = points[i, 0] + distance[i, 2];
            result[i, 3] = points[i, 1] + distance[i, 3];
        }

        return result;
    }

    /// <summary>
    /// Python: <c>distance_to_face_landmark_5</c>. <paramref name="points"/> is <c>(N, 2)</c>,
    /// <paramref name="distance"/> is <c>(N, 10)</c>; returns <c>(N, 5, 2)</c>.
    /// </summary>
    public static float[,,] DistanceToFaceLandmark5(float[,] points, float[,] distance)
    {
        var count = points.GetLength(0);
        var result = new float[count, 5, 2];
        for (var i = 0; i < count; i++)
        {
            for (var k = 0; k < 5; k++)
            {
                result[i, k, 0] = points[i, 0] + distance[i, 2 * k];
                result[i, k, 1] = points[i, 1] + distance[i, (2 * k) + 1];
            }
        }

        return result;
    }

    /// <summary>
    /// Python: <c>scale_face_landmark_5</c>. Kept in float32 throughout (matching numpy's
    /// in-place <c>*=</c>/<c>+=</c> on a float32 array, with <paramref name="scale"/> narrowed
    /// to float32 for the multiply, per PORT_CONVENTIONS.md rule 6 / the float32-vs-float64
    /// assignment instruction).
    /// </summary>
    public static float[,] ScaleFaceLandmark5(float[,] faceLandmark5, double scale)
    {
        var originX = faceLandmark5[2, 0];
        var originY = faceLandmark5[2, 1];
        var scaleF = (float)scale;

        var result = new float[5, 2];
        for (var i = 0; i < 5; i++)
        {
            result[i, 0] = ((faceLandmark5[i, 0] - originX) * scaleF) + originX;
            result[i, 1] = ((faceLandmark5[i, 1] - originY) * scaleF) + originY;
        }

        return result;
    }

    /// <summary>
    /// Python: <c>convert_to_face_landmark_5</c>. Reuses <see cref="NumPy.Mean(ReadOnlySpan{float})"/>
    /// for the two eye-corner means (Python: <c>numpy.mean(face_landmark_68[36:42], axis = 0)</c> /
    /// <c>[42:48]</c>) rather than hand-rolling the reduction, per the assignment's guidance to
    /// reuse FaceFusion.Tensors instead of writing new maths.
    /// </summary>
    public static float[,] ConvertToFaceLandmark5(float[,] faceLandmark68)
    {
        if (faceLandmark68.GetLength(0) != 68 || faceLandmark68.GetLength(1) != 2)
        {
            throw new ArgumentException("faceLandmark68 must have shape (68, 2).", nameof(faceLandmark68));
        }

        var leftEyeX = new float[6];
        var leftEyeY = new float[6];
        for (var i = 0; i < 6; i++)
        {
            leftEyeX[i] = faceLandmark68[36 + i, 0];
            leftEyeY[i] = faceLandmark68[36 + i, 1];
        }

        var rightEyeX = new float[6];
        var rightEyeY = new float[6];
        for (var i = 0; i < 6; i++)
        {
            rightEyeX[i] = faceLandmark68[42 + i, 0];
            rightEyeY[i] = faceLandmark68[42 + i, 1];
        }

        return new float[5, 2]
        {
            { NumPy.Mean(leftEyeX), NumPy.Mean(leftEyeY) },
            { NumPy.Mean(rightEyeX), NumPy.Mean(rightEyeY) },
            { faceLandmark68[30, 0], faceLandmark68[30, 1] },
            { faceLandmark68[48, 0], faceLandmark68[48, 1] },
            { faceLandmark68[54, 0], faceLandmark68[54, 1] },
        };
    }

    /// <summary>
    /// Python: <c>estimate_face_angle</c>. Reuses <see cref="NumPy.Linspace"/> for
    /// <c>numpy.linspace(0, 360, 5)</c>.
    /// </summary>
    public static int EstimateFaceAngle(float[,] faceLandmark68)
    {
        var x1 = faceLandmark68[0, 0];
        var y1 = faceLandmark68[0, 1];
        var x2 = faceLandmark68[16, 0];
        var y2 = faceLandmark68[16, 1];

        var theta = Math.Atan2(y2 - y1, x2 - x1);
        var thetaDegrees = theta * 180.0 / Math.PI;
        thetaDegrees %= 360.0;
        if (thetaDegrees < 0)
        {
            thetaDegrees += 360.0; // Python's % always returns a non-negative result.
        }

        var angles = NumPy.Linspace(0f, 360f, 5);
        var bestIndex = 0;
        var bestDiff = double.PositiveInfinity;
        for (var i = 0; i < angles.Length; i++)
        {
            var diff = Math.Abs(angles[i] - thetaDegrees);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                bestIndex = i;
            }
        }

        var faceAngle = (int)angles[bestIndex] % 360;
        return faceAngle;
    }

    // -----------------------------------------------------------------
    // NMS
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>apply_nms</c>. <paramref name="boundingBoxes"/> entries are
    /// <c>[x1, y1, x2, y2]</c>; converted to <c>[x, y, width, height]</c> (Python:
    /// <c>bounding_boxes_norm</c>) before calling into <c>cv2.dnn.NMSBoxes</c>, matching the
    /// Python conversion exactly.
    /// </summary>
    public static int[] ApplyNms(
        IReadOnlyList<float[]> boundingBoxes, IReadOnlyList<float> scores, float scoreThreshold, float nmsThreshold)
    {
        var rects = new Rect2d[boundingBoxes.Count];
        for (var i = 0; i < boundingBoxes.Count; i++)
        {
            var box = boundingBoxes[i];
            rects[i] = new Rect2d(box[0], box[1], box[2] - box[0], box[3] - box[1]);
        }

        CvDnn.NMSBoxes(rects, scores, scoreThreshold, nmsThreshold, out int[] indices);
        return indices;
    }

    /// <summary>Python: <c>get_nms_threshold</c>.</summary>
    public static float GetNmsThreshold(FaceDetectorModel faceDetectorModel, IReadOnlyList<int> faceDetectorAngles)
    {
        if (faceDetectorModel == FaceDetectorModel.Many)
        {
            return 0.1f;
        }

        return faceDetectorAngles.Count switch
        {
            2 => 0.3f,
            3 => 0.2f,
            4 => 0.1f,
            _ => 0.4f,
        };
    }

    // -----------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>merge_matrix</c>. Pure 3x3 homogeneous-matrix composition (no cv2 call), done
    /// in double precision throughout. Caller owns the returned <see cref="Mat"/> (2x3,
    /// <c>CV_64F</c>).
    /// </summary>
    public static Mat MergeMatrix(IReadOnlyList<Mat> tempMatrices)
    {
        if (tempMatrices.Count == 0)
        {
            throw new ArgumentException("tempMatrices must not be empty.", nameof(tempMatrices));
        }

        var matrix = ToHomogeneous(tempMatrices[0]);

        for (var i = 1; i < tempMatrices.Count; i++)
        {
            var next = ToHomogeneous(tempMatrices[i]);
            matrix = Multiply3X3(next, matrix);
        }

        var result = new Mat(2, 3, MatType.CV_64FC1);
        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < 3; c++)
            {
                result.Set(r, c, matrix[r, c]);
            }
        }

        return result;
    }

    private static double[,] ToHomogeneous(Mat affineMatrix)
    {
        var homogeneous = new double[3, 3];
        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < 3; c++)
            {
                homogeneous[r, c] = affineMatrix.At<double>(r, c);
            }
        }

        homogeneous[2, 0] = 0;
        homogeneous[2, 1] = 0;
        homogeneous[2, 2] = 1;
        return homogeneous;
    }

    private static double[,] Multiply3X3(double[,] a, double[,] b)
    {
        var result = new double[3, 3];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 3; c++)
            {
                double sum = 0;
                for (var k = 0; k < 3; k++)
                {
                    sum += a[r, k] * b[k, c];
                }

                result[r, c] = sum;
            }
        }

        return result;
    }

    /// <summary>Python: <c>calculate_bounding_box_overlap</c>.</summary>
    public static double CalculateBoundingBoxOverlap(float[] boundingBoxA, float[] boundingBoxB)
    {
        var intersectionX1 = Math.Max(boundingBoxA[0], boundingBoxB[0]);
        var intersectionY1 = Math.Max(boundingBoxA[1], boundingBoxB[1]);
        var intersectionX2 = Math.Min(boundingBoxA[2], boundingBoxB[2]);
        var intersectionY2 = Math.Min(boundingBoxA[3], boundingBoxB[3]);

        var intersection = Math.Max(0f, intersectionX2 - intersectionX1) * Math.Max(0f, intersectionY2 - intersectionY1);
        var boundingBoxAreaA = (boundingBoxA[2] - boundingBoxA[0]) * (boundingBoxA[3] - boundingBoxA[1]);
        var boundingBoxAreaB = (boundingBoxB[2] - boundingBoxB[0]) * (boundingBoxB[3] - boundingBoxB[1]);
        var union = boundingBoxAreaA + boundingBoxAreaB - intersection;

        if (union > 0)
        {
            return intersection / union;
        }

        return 0.0;
    }

    /// <summary>Python: <c>average_points</c>. <paramref name="pointsPrevious"/> and
    /// <paramref name="pointsNext"/> are <c>(N, 2)</c>.</summary>
    public static float[,] AveragePoints(float[,] pointsPrevious, float[,] pointsNext, double averageFactor)
    {
        var rows = pointsPrevious.GetLength(0);
        var cols = pointsPrevious.GetLength(1);
        var result = new float[rows, cols];
        var factor = (float)averageFactor;
        var oneMinusFactor = 1f - factor;

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = (pointsPrevious[r, c] * oneMinusFactor) + (pointsNext[r, c] * factor);
            }
        }

        return result;
    }
}
