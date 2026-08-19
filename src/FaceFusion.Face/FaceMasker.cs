using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_masker.py</c> — builds the four kinds of blend mask the face
/// pipeline pastes a processed crop back through: a padded/blurred box mask, an ONNX
/// occlusion mask, a landmark-68 convex-hull area mask, and an ONNX face-parser region mask.
///
/// <para>
/// <b>No global state; ONNX sessions taken as parameters.</b> Per PORT_CONVENTIONS.md rule 5,
/// every Python <c>state_manager.get_item(...)</c> call becomes an explicit method parameter.
/// Python's <c>get_inference_pool()</c>/<c>collect_model_downloads()</c>/<c>pre_check()</c>
/// depend on <c>facefusion.download</c> (hash verification, conditional downloading), which is
/// out of this module's assignment and not ported anywhere in this repo yet. This file
/// therefore does not reproduce <c>get_inference_pool</c>/<c>pre_check</c> at all — callers are
/// expected to obtain an <c>InferenceSession</c> pool themselves (e.g. via
/// <see cref="FaceFusion.Inference.InferenceManager.GetInferencePool"/>, whose model-file
/// existence check subsumes what <c>pre_check</c> would have gated) and pass it to
/// <see cref="CreateOcclusionMask"/>/<see cref="CreateRegionMask"/>, keyed by model name exactly
/// as Python's <c>get_inference_pool().get(model_name)</c> does. What *is* reproduced from
/// <c>create_static_model_set</c> is the one piece these two methods actually need — each
/// model's fixed input resolution (<see cref="OccluderModelSizes"/> /
/// <see cref="ParserModelSizes"/>) — the <c>hashes</c>/<c>sources</c>/<c>__metadata__</c>
/// entries are download-manager concerns and are not reproduced.
/// </para>
///
/// <para>
/// <b>VisionFrame / Mask representation.</b> Same convention as <c>FaceHelper</c>/
/// <c>FaceFusion.Vision.Vision</c>: pixel data and masks are <see cref="Mat"/>, native memory,
/// caller-owned on every return, parameters never disposed by the callee.
/// <see cref="CreateBoxMask"/>/<see cref="CreateOcclusionMask"/>/<see cref="CreateAreaMask"/>/
/// <see cref="CreateRegionMask"/> all return a single-channel <c>CV_32FC1</c> mask, matching
/// Python's <c>Mask : TypeAlias = NDArray[Any]</c> float32 <c>(H, W)</c> array.
/// </para>
///
/// <para>
/// <b><c>crop_size = shape[:2][::-1]</c> axis-order quirk (reproduced, then simplified because
/// it is provably a no-op for every real caller).</b> Python computes
/// <c>crop_size = crop_vision_frame.shape[:2][::-1]</c> — i.e. <c>(W, H)</c>, the reverse of
/// the array's own <c>(H, W)</c> shape — and then builds <c>numpy.ones(crop_size)</c> /
/// <c>numpy.zeros(crop_size)</c> masks shaped <c>(W, H)</c> instead of the more natural
/// <c>(H, W)</c>. The threshold expressions that slice into that mask consistently pair
/// <c>crop_size[1]</c> (= H) with the top/bottom (row) slices and <c>crop_size[0]</c> (= W)
/// with the left/right (column) slices — i.e. despite the array being shaped <c>(W, H)</c>,
/// every slice is sized as though it were shaped <c>(H, W)</c>. This is only self-consistent
/// (does not throw / does not silently zero the wrong edge) when <c>W == H</c>. Every real
/// crop size in this codebase — every <see cref="WarpTemplate"/> target and every occluder/
/// parser model input — is square, so the quirk is unobservable; <see cref="CreateBoxMask"/>
/// and <see cref="CreateAreaMask"/> below build a natural <c>(H, W)</c> <see cref="Mat"/>
/// throughout rather than reproducing the reversed shape, which is behaviourally identical for
/// every square input and is verified against a real square crop in the parity tests.
/// </para>
/// </summary>
public static class FaceMasker
{
    /// <summary>
    /// Python: the <c>'size'</c> entry of <c>create_static_model_set('full')</c> for the three
    /// <c>xseg_*</c> occluder models — the only piece of that model set this file needs (see
    /// class remarks). Keyed by the model's wire name, matching
    /// <c>get_inference_pool()</c>'s dictionary keys.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, Size> OccluderModelSizes = new Dictionary<string, Size>
    {
        ["xseg_1"] = new Size(256, 256),
        ["xseg_2"] = new Size(256, 256),
        ["xseg_3"] = new Size(256, 256),
    };

    /// <summary>Same as <see cref="OccluderModelSizes"/>, for the two <c>bisenet_*</c> face-parser models.</summary>
    private static readonly IReadOnlyDictionary<string, Size> ParserModelSizes = new Dictionary<string, Size>
    {
        ["bisenet_resnet_18"] = new Size(512, 512),
        ["bisenet_resnet_34"] = new Size(512, 512),
    };

    // -----------------------------------------------------------------
    // Box mask
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_box_mask</c>. Caller owns the returned <see cref="Mat"/> (<c>CV_32FC1</c>)
    /// and must dispose it. Does not take ownership of <paramref name="cropVisionFrame"/>.
    /// </summary>
    public static Mat CreateBoxMask(Mat cropVisionFrame, double faceMaskBlur, Padding faceMaskPadding)
    {
        var cropWidth = cropVisionFrame.Cols;
        var cropHeight = cropVisionFrame.Rows;

        var blurAmount = (int)(cropWidth * 0.5 * faceMaskBlur);
        var blurArea = Math.Max(blurAmount / 2, 1);

        var boxMask = new Mat(cropHeight, cropWidth, MatType.CV_32FC1, Scalar.All(1.0));

        var topZero = Math.Min(Math.Max(blurArea, (int)(cropHeight * faceMaskPadding.Top / 100.0)), cropHeight);
        var bottomZero = Math.Min(Math.Max(blurArea, (int)(cropHeight * faceMaskPadding.Bottom / 100.0)), cropHeight);
        var leftZero = Math.Min(Math.Max(blurArea, (int)(cropWidth * faceMaskPadding.Left / 100.0)), cropWidth);
        var rightZero = Math.Min(Math.Max(blurArea, (int)(cropWidth * faceMaskPadding.Right / 100.0)), cropWidth);

        ZeroRegion(boxMask, new Rect(0, 0, cropWidth, topZero));
        ZeroRegion(boxMask, new Rect(0, cropHeight - bottomZero, cropWidth, bottomZero));
        ZeroRegion(boxMask, new Rect(0, 0, leftZero, cropHeight));
        ZeroRegion(boxMask, new Rect(cropWidth - rightZero, 0, rightZero, cropHeight));

        if (blurAmount > 0)
        {
            var blurred = new Mat();
            Cv2.GaussianBlur(boxMask, blurred, new Size(0, 0), blurAmount * 0.25);
            boxMask.Dispose();
            return blurred;
        }

        return boxMask;
    }

    private static void ZeroRegion(Mat mat, Rect rect)
    {
        if (rect.Width <= 0 || rect.Height <= 0)
        {
            return;
        }

        using var region = new Mat(mat, rect);
        region.SetTo(Scalar.All(0));
    }

    // -----------------------------------------------------------------
    // Occlusion mask (ONNX)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_occlusion_mask</c>. Caller owns the returned <see cref="Mat"/>
    /// (<c>CV_32FC1</c>) and must dispose it. Does not take ownership of
    /// <paramref name="cropVisionFrame"/> or of any session in
    /// <paramref name="occluderInferencePool"/>. <paramref name="occluderInferencePool"/> is
    /// keyed by model wire name (<c>"xseg_1"</c>/<c>"xseg_2"</c>/<c>"xseg_3"</c>), matching
    /// Python's <c>get_inference_pool()</c> — pass a pool containing just the one selected
    /// model, or all three when <paramref name="faceOccluderModel"/> is
    /// <see cref="FaceOccluderModel.Many"/>.
    /// </summary>
    public static Mat CreateOcclusionMask(
        Mat cropVisionFrame,
        FaceOccluderModel faceOccluderModel,
        IReadOnlyDictionary<string, InferenceSession> occluderInferencePool)
    {
        var modelNames = faceOccluderModel == FaceOccluderModel.Many
            ? new[] { "xseg_1", "xseg_2", "xseg_3" }
            : new[] { faceOccluderModel.ToWireName() };

        var tempMasks = new List<Mat>();

        try
        {
            foreach (var modelName in modelNames)
            {
                var modelSize = OccluderModelSizes[modelName];

                using var prepareVisionFrame = new Mat();
                Cv2.Resize(cropVisionFrame, prepareVisionFrame, modelSize);

                // Python: expand_dims(axis=0).astype(float32)/255.0, then a no-op
                // transpose(0,1,2,3) — i.e. the NHWC layout cv2.resize already produced is fed
                // to the model unchanged (no BGR->RGB flip for the occluder, unlike the parser
                // below).
                var inputData = BuildNhwcInput(prepareVisionFrame);

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
                    inputData, new long[] { 1, modelSize.Height, modelSize.Width, 3 });
                var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

                var inferenceSession = occluderInferencePool[modelName];
                using var runOptions = new RunOptions();
                using var results = inferenceSession.Run(runOptions, inputs, inferenceSession.OutputNames);

                // Python: `forward_occlude_face(...)[0][0]` selects the first output tensor and
                // its single batch entry, leaving shape (H, W, 1) — a flat span of H*W floats.
                var outputSpan = results[0].GetTensorDataAsSpan<float>();

                using var tempMask = new Mat(modelSize.Height, modelSize.Width, MatType.CV_32FC1);
                var clipped = new float[outputSpan.Length];
                for (var index = 0; index < outputSpan.Length; index++)
                {
                    // Python: `.clip(0, 1)`.
                    clipped[index] = Math.Clamp(outputSpan[index], 0f, 1f);
                }

                tempMask.SetArray(clipped);

                var resizedMask = new Mat();
                Cv2.Resize(tempMask, resizedMask, new Size(cropVisionFrame.Cols, cropVisionFrame.Rows));
                tempMasks.Add(resizedMask);
            }

            // Python: `numpy.minimum.reduce(temp_masks)`.
            var occlusionMask = tempMasks[0].Clone();
            for (var index = 1; index < tempMasks.Count; index++)
            {
                Cv2.Min(occlusionMask, tempMasks[index], occlusionMask);
            }

            var result = SoftenMask(occlusionMask);
            occlusionMask.Dispose();
            return result;
        }
        finally
        {
            foreach (var tempMask in tempMasks)
            {
                tempMask.Dispose();
            }
        }
    }

    // -----------------------------------------------------------------
    // Area mask (landmark-68 convex hull)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_area_mask</c>. Caller owns the returned <see cref="Mat"/>
    /// (<c>CV_32FC1</c>) and must dispose it. Does not take ownership of
    /// <paramref name="cropVisionFrame"/>. <paramref name="faceLandmark68"/> is <c>(68, 2)</c>,
    /// same shape/dtype convention as <see cref="FaceHelper"/>. Region index lists come from
    /// <see cref="Choices.FaceMaskAreaSet"/> (reused, not re-transcribed).
    /// </summary>
    public static Mat CreateAreaMask(Mat cropVisionFrame, float[,] faceLandmark68, IReadOnlyList<FaceMaskArea> faceMaskAreas)
    {
        var landmarkPoints = new List<int>();

        foreach (var faceMaskArea in faceMaskAreas)
        {
            if (Choices.FaceMaskAreaSet.TryGetValue(faceMaskArea, out var indices))
            {
                landmarkPoints.AddRange(indices);
            }
        }

        var points = new Point[landmarkPoints.Count];
        for (var i = 0; i < landmarkPoints.Count; i++)
        {
            var landmarkIndex = landmarkPoints[i];
            // Python: `.astype(numpy.int32)` — truncates toward zero, matching a plain (int) cast.
            points[i] = new Point((int)faceLandmark68[landmarkIndex, 0], (int)faceLandmark68[landmarkIndex, 1]);
        }

        var convexHull = Cv2.ConvexHull(points);

        var areaMask = new Mat(cropVisionFrame.Rows, cropVisionFrame.Cols, MatType.CV_32FC1, Scalar.All(0));
        Cv2.FillConvexPoly(areaMask, convexHull, Scalar.All(1.0));

        var result = SoftenMask(areaMask);
        areaMask.Dispose();
        return result;
    }

    // -----------------------------------------------------------------
    // Region mask (ONNX face parser)
    // -----------------------------------------------------------------

    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    /// <summary>
    /// Python: <c>create_region_mask</c>. Caller owns the returned <see cref="Mat"/>
    /// (<c>CV_32FC1</c>) and must dispose it. Does not take ownership of
    /// <paramref name="cropVisionFrame"/> or of any session in
    /// <paramref name="parserInferencePool"/>. <paramref name="parserInferencePool"/> is keyed
    /// by model wire name (<c>"bisenet_resnet_18"</c>/<c>"bisenet_resnet_34"</c>), matching
    /// Python's <c>get_inference_pool()</c>. Region ids come from
    /// <see cref="Choices.FaceMaskRegionSet"/> (reused, not re-transcribed).
    /// </summary>
    public static Mat CreateRegionMask(
        Mat cropVisionFrame,
        IReadOnlyList<FaceMaskRegion> faceMaskRegions,
        FaceParserModel faceParserModel,
        IReadOnlyDictionary<string, InferenceSession> parserInferencePool)
    {
        var modelName = faceParserModel.ToWireName();
        var modelSize = ParserModelSizes[modelName];

        using var prepareVisionFrame = new Mat();
        Cv2.Resize(cropVisionFrame, prepareVisionFrame, modelSize);

        // Python: `prepare_vision_frame[:, :, ::-1].astype(float32)/255.0`, ImageNet
        // mean/std normalisation, `expand_dims(axis=0)`, then `transpose(0, 3, 1, 2)` (NHWC ->
        // NCHW). BuildNchwNormalizedInput folds the BGR->RGB channel reversal, the
        // normalisation and the transpose into one pass.
        var inputData = BuildNchwNormalizedInput(prepareVisionFrame, ImageNetMean, ImageNetStd);

        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
            inputData, new long[] { 1, 3, modelSize.Height, modelSize.Width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        var inferenceSession = parserInferencePool[modelName];
        using var runOptions = new RunOptions();
        using var results = inferenceSession.Run(runOptions, inputs, inferenceSession.OutputNames);

        // Python: `forward_parse_face(...)[0][0]` selects the first output tensor and its
        // single batch entry, leaving shape (classCount, H, W) — read the shape dynamically
        // rather than hardcoding the class count, same as Python's `.argmax(0)` does not care
        // how many classes there are.
        var outputInfo = results[0].GetTensorTypeAndShape();
        var outputShape = outputInfo.Shape;
        var classCount = checked((int)outputShape[1]);
        var outputHeight = checked((int)outputShape[2]);
        var outputWidth = checked((int)outputShape[3]);
        var plane = outputHeight * outputWidth;

        var outputSpan = results[0].GetTensorDataAsSpan<float>();

        var requestedRegionIds = new HashSet<int>();
        foreach (var faceMaskRegion in faceMaskRegions)
        {
            if (Choices.FaceMaskRegionSet.TryGetValue(faceMaskRegion, out var regionId))
            {
                requestedRegionIds.Add(regionId);
            }
        }

        // Python: `numpy.isin(region_mask.argmax(0), [...])` — per-pixel argmax over the
        // channel axis, then membership test against the requested region ids.
        var regionMaskData = new float[plane];
        for (var spatialIndex = 0; spatialIndex < plane; spatialIndex++)
        {
            var bestClass = 0;
            var bestValue = outputSpan[spatialIndex];
            for (var classIndex = 1; classIndex < classCount; classIndex++)
            {
                var value = outputSpan[(classIndex * plane) + spatialIndex];
                if (value > bestValue)
                {
                    bestValue = value;
                    bestClass = classIndex;
                }
            }

            regionMaskData[spatialIndex] = requestedRegionIds.Contains(bestClass) ? 1f : 0f;
        }

        using var regionMaskSmall = new Mat(outputHeight, outputWidth, MatType.CV_32FC1);
        regionMaskSmall.SetArray(regionMaskData);

        using var regionMask = new Mat();
        Cv2.Resize(regionMaskSmall, regionMask, new Size(cropVisionFrame.Cols, cropVisionFrame.Rows));

        return SoftenMask(regionMask);
    }

    // -----------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------

    /// <summary>
    /// The common tail shared by <see cref="CreateOcclusionMask"/>, <see cref="CreateAreaMask"/>
    /// and <see cref="CreateRegionMask"/>: Python's
    /// <c>(cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2</c>. Does not
    /// take ownership of <paramref name="mask"/>. Caller owns the returned <see cref="Mat"/>.
    /// </summary>
    private static Mat SoftenMask(Mat mask)
    {
        using var clippedLow = ClipMat(mask, 0.0, 1.0);
        using var blurred = new Mat();
        Cv2.GaussianBlur(clippedLow, blurred, new Size(0, 0), 5);
        using var clippedHigh = ClipMat(blurred, 0.5, 1.0);

        var shifted = new Mat();
        Cv2.Subtract(clippedHigh, new Scalar(0.5), shifted);
        Cv2.Multiply(shifted, new Scalar(2.0), shifted);
        return shifted;
    }

    private static Mat ClipMat(Mat mat, double low, double high)
    {
        using var clampedLow = new Mat();
        Cv2.Max(mat, low, clampedLow);
        var result = new Mat();
        Cv2.Min(clampedLow, high, result);
        return result;
    }

    /// <summary>
    /// Builds the occluder's NHWC float32 model input: <c>expand_dims(axis=0).astype(float32) /
    /// 255.0</c>, no channel reordering (the occluder consumes the BGR channel order
    /// <see cref="Mat"/> already has, same as Python leaves it untouched). Returns a flat
    /// <c>H*W*3</c> array in row-major NHWC order (channel fastest).
    /// </summary>
    private static float[] BuildNhwcInput(Mat bgrVisionFrame)
    {
        bgrVisionFrame.GetArray(out Vec3b[] pixels);
        var data = new float[pixels.Length * 3];

        for (var i = 0; i < pixels.Length; i++)
        {
            var pixel = pixels[i];
            var offset = i * 3;
            data[offset] = pixel.Item0 / 255f;
            data[offset + 1] = pixel.Item1 / 255f;
            data[offset + 2] = pixel.Item2 / 255f;
        }

        return data;
    }

    /// <summary>
    /// Builds the face parser's NCHW float32 model input: BGR-&gt;RGB channel reversal
    /// (Python: <c>[:, :, ::-1]</c>), <c>/255.0</c>, per-channel mean/std normalisation, then
    /// the <c>expand_dims(axis=0)</c> + <c>transpose(0, 3, 1, 2)</c> collapsed into building the
    /// NCHW layout directly (channel-plane-major) instead of building NHWC then transposing.
    /// <paramref name="mean"/>/<paramref name="std"/> are in output-channel order (R, G, B).
    /// </summary>
    private static float[] BuildNchwNormalizedInput(Mat bgrVisionFrame, float[] mean, float[] std)
    {
        bgrVisionFrame.GetArray(out Vec3b[] pixels);
        var plane = pixels.Length;
        var data = new float[plane * 3];

        for (var i = 0; i < plane; i++)
        {
            var pixel = pixels[i];
            var r = ((pixel.Item2 / 255f) - mean[0]) / std[0];
            var g = ((pixel.Item1 / 255f) - mean[1]) / std[1];
            var b = ((pixel.Item0 / 255f) - mean[2]) / std[2];

            data[i] = r;
            data[plane + i] = g;
            data[(2 * plane) + i] = b;
        }

        return data;
    }
}
