using System.Runtime.CompilerServices;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

// Exposes this project's `internal` preprocessing helpers (ComputeScaleAndTranslation,
// ConditionalOptimizeContrast, PrepareLandmarkerInput, Forward*) to the two test assemblies,
// the same pattern FaceFusion.Media.Ffmpeg already uses. These stay `internal` rather than
// `public` because they are implementation details of the DetectWith*/ClassifyFace/
// CalculateFaceEmbedding methods, not part of the module's intended public surface — but the
// parity tests need to call them directly to isolate exactly which preprocessing step a
// mismatch comes from (see docs/PARITY_HARNESS.md).
[assembly: InternalsVisibleTo("FaceFusion.UnitTests")]
[assembly: InternalsVisibleTo("FaceFusion.ParityTests")]

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_landmarker.py</c> — the 68-point (and 68-from-5) landmark
/// stage, covering the <c>2dfan4</c> and <c>peppa_wutz</c> heatmap-style detectors plus the
/// <c>fan_68_5</c> landmark-5-&gt;landmark-68 regressor.
///
/// <para>
/// <b>Model / session wiring (documented divergence).</b> Same reasoning as
/// <c>FaceFusion.Face.FaceRecognizer</c>/<c>FaceClassifier</c>: <c>facefusion/download.py</c>
/// has no C# port yet, so every <c>DetectWith*</c>/<c>Forward*</c> method here takes an
/// already-created <see cref="InferenceSession"/> for its model rather than owning a
/// download-backed <c>InferenceManager</c> pool keyed by <c>state_manager</c>'s
/// <c>face_landmarker_model</c>.
/// </para>
///
/// <para>
/// <b>VisionFrame channel order.</b> Per <c>FaceHelper</c>'s BGR convention. Unlike
/// <c>FaceRecognizer</c>/<c>FaceClassifier</c>, Python's <c>detect_with_2dfan4</c>/
/// <c>detect_with_peppa_wutz</c> do <em>not</em> reverse the channel order before feeding the
/// model — the crop stays in the same (BGR) order it was warped in. <c>conditional_optimize_
/// contrast</c> calls <c>cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)</c> on that
/// (actually-BGR) buffer, which is a real oddity in the Python source: the conversion code
/// only produces a colorimetrically-correct Lab image when its input really is RGB, so on the
/// BGR-ordered frame these calls silently swap the R and B channels' contribution to the L/a/b
/// planes. Reproduced verbatim per PORT_CONVENTIONS.md rule 1 ("reproduce the oddity") by
/// calling the same, equally-mislabelled <see cref="ColorConversionCodes.RGB2Lab"/>/
/// <see cref="ColorConversionCodes.Lab2RGB"/> codes on the BGR <see cref="Mat"/> in
/// <see cref="ConditionalOptimizeContrast"/> — both sides apply the identical mislabeled
/// transform to identically-ordered bytes, so the output matches bit-for-bit; "fixing" the
/// color code here would be a real behavioural divergence, not a correction.
/// </para>
/// </summary>
public static class FaceLandmarker
{
    /// <summary>
    /// Python: <c>create_static_model_set('full').get('2dfan4').get('size')</c> and the
    /// <c>peppa_wutz</c> entry's <c>size</c> (both <c>(256, 256)</c>).
    /// </summary>
    public static readonly Size TwoDFan4ModelSize = new(256, 256);

    /// <summary>See <see cref="TwoDFan4ModelSize"/> — <c>peppa_wutz</c> uses the same size.</summary>
    public static readonly Size PeppaWutzModelSize = new(256, 256);

    /// <summary>
    /// Python: <c>detect_face_landmark</c>. <paramref name="twoDFan4Session"/>/
    /// <paramref name="peppaWutzSession"/> are only required (non-null) when
    /// <paramref name="faceLandmarkerModel"/> requests that model (<see cref="FaceLandmarkerModel.Many"/>
    /// requests both); passing null for a requested session throws.
    ///
    /// <para>
    /// <b>Oddity reproduced verbatim.</b> When only one of the two models is requested, the
    /// *other* model's score stays at Python's <c>0.0</c> default rather than being undefined,
    /// so the final <c>face_landmark_score_2dfan4 &gt; face_landmark_score_peppa_wutz - 0.2</c>
    /// comparison can still select the model that was never run (whose landmark array is
    /// <see langword="null"/>) — e.g. requesting only <see cref="FaceLandmarkerModel.PeppaWutz"/>
    /// with a resulting score &lt;= 0.2 makes this method return a <see langword="null"/>
    /// landmark array with score 0.0. This is exactly what the Python returns in that case
    /// (<c>None, 0.0</c>); per PORT_CONVENTIONS.md rule 1 it is reproduced, not "fixed".
    /// </para>
    /// </summary>
    public static (float[,]? FaceLandmark68, double Score) DetectFaceLandmark(
        FaceLandmarkerModel faceLandmarkerModel,
        InferenceSession? twoDFan4Session,
        InferenceSession? peppaWutzSession,
        Mat visionFrame,
        float[] boundingBox,
        int faceAngle)
    {
        float[,]? faceLandmark2dFan4 = null;
        float[,]? faceLandmarkPeppaWutz = null;
        var faceLandmarkScore2dFan4 = 0.0;
        var faceLandmarkScorePeppaWutz = 0.0;

        if (faceLandmarkerModel is FaceLandmarkerModel.Many or FaceLandmarkerModel.TwoDFan4)
        {
            if (twoDFan4Session is null)
            {
                throw new ArgumentNullException(nameof(twoDFan4Session), "twoDFan4Session is required when faceLandmarkerModel requests 2dfan4.");
            }

            (faceLandmark2dFan4, faceLandmarkScore2dFan4) = DetectWith2dFan4(twoDFan4Session, visionFrame, boundingBox, faceAngle);
        }

        if (faceLandmarkerModel is FaceLandmarkerModel.Many or FaceLandmarkerModel.PeppaWutz)
        {
            if (peppaWutzSession is null)
            {
                throw new ArgumentNullException(nameof(peppaWutzSession), "peppaWutzSession is required when faceLandmarkerModel requests peppa_wutz.");
            }

            (faceLandmarkPeppaWutz, faceLandmarkScorePeppaWutz) = DetectWithPeppaWutz(peppaWutzSession, visionFrame, boundingBox, faceAngle);
        }

        if (faceLandmarkScore2dFan4 > faceLandmarkScorePeppaWutz - 0.2)
        {
            return (faceLandmark2dFan4, faceLandmarkScore2dFan4);
        }

        return (faceLandmarkPeppaWutz, faceLandmarkScorePeppaWutz);
    }

    /// <summary>Python: <c>detect_with_2dfan4</c>.</summary>
    public static (float[,] FaceLandmark68, double Score) DetectWith2dFan4(
        InferenceSession twoDFan4Session, Mat tempVisionFrame, float[] boundingBox, int faceAngle)
    {
        var modelSize = TwoDFan4ModelSize;
        var (scale, translation) = ComputeScaleAndTranslation(boundingBox, modelSize);

        var (rotationMatrix, rotationSize) = FaceHelper.CreateRotationMatrixAndSize(faceAngle, modelSize);
        using var rotationMatrixDisposable = rotationMatrix;

        var (translatedCrop, affineMatrix) = FaceHelper.WarpFaceByTranslation(tempVisionFrame, translation, scale, modelSize);
        using var affineMatrixDisposable = affineMatrix;
        using var translatedCropDisposable = translatedCrop;

        using var rotatedCrop = new Mat();
        Cv2.WarpAffine(translatedCrop, rotatedCrop, rotationMatrix, rotationSize);

        using var contrastCrop = ConditionalOptimizeContrast(rotatedCrop);

        var inputTensor = PrepareLandmarkerInput(contrastCrop);
        var (landmarks, heatmaps) = ForwardWith2dFan4(twoDFan4Session, inputTensor, modelSize);

        var faceLandmark68 = new float[68, 2];
        for (var i = 0; i < 68; i++)
        {
            faceLandmark68[i, 0] = landmarks[(i * 3) + 0] / 64f * 256f;
            faceLandmark68[i, 1] = landmarks[(i * 3) + 1] / 64f * 256f;
        }

        faceLandmark68 = TransformByInverse(faceLandmark68, rotationMatrix);
        faceLandmark68 = TransformByInverse(faceLandmark68, affineMatrix);

        var score = ComputeTwoDFan4Score(heatmaps);
        return (faceLandmark68, score);
    }

    /// <summary>Python: <c>detect_with_peppa_wutz</c>.</summary>
    public static (float[,] FaceLandmark68, double Score) DetectWithPeppaWutz(
        InferenceSession peppaWutzSession, Mat tempVisionFrame, float[] boundingBox, int faceAngle)
    {
        var modelSize = PeppaWutzModelSize;
        var (scale, translation) = ComputeScaleAndTranslation(boundingBox, modelSize);

        var (rotationMatrix, rotationSize) = FaceHelper.CreateRotationMatrixAndSize(faceAngle, modelSize);
        using var rotationMatrixDisposable = rotationMatrix;

        var (translatedCrop, affineMatrix) = FaceHelper.WarpFaceByTranslation(tempVisionFrame, translation, scale, modelSize);
        using var affineMatrixDisposable = affineMatrix;
        using var translatedCropDisposable = translatedCrop;

        using var rotatedCrop = new Mat();
        Cv2.WarpAffine(translatedCrop, rotatedCrop, rotationMatrix, rotationSize);

        using var contrastCrop = ConditionalOptimizeContrast(rotatedCrop);

        var inputTensor = PrepareLandmarkerInput(contrastCrop);
        var prediction = ForwardWithPeppaWutz(peppaWutzSession, inputTensor, modelSize);

        var faceLandmark68 = new float[68, 2];
        var scoreSum = 0f;
        for (var i = 0; i < 68; i++)
        {
            faceLandmark68[i, 0] = prediction[(i * 3) + 0] / 64f * modelSize.Width;
            faceLandmark68[i, 1] = prediction[(i * 3) + 1] / 64f * modelSize.Width;
            scoreSum += prediction[(i * 3) + 2];
        }

        faceLandmark68 = TransformByInverse(faceLandmark68, rotationMatrix);
        faceLandmark68 = TransformByInverse(faceLandmark68, affineMatrix);

        var rawScore = scoreSum / 68f;
        var score = NumPy.Interp(rawScore, new[] { 0f, 0.95f }, new[] { 0f, 1f });
        return (faceLandmark68, score);
    }

    /// <summary>
    /// Python: <c>estimate_face_landmark_68_5</c> — regresses a landmark-68 set from a
    /// landmark-5 set via the <c>fan_68_5</c> model, working in the normalised
    /// <c>ffhq_512</c>-template space.
    /// </summary>
    public static float[,] EstimateFaceLandmark685(InferenceSession fan685Session, float[,] faceLandmark5)
    {
        var affineMatrix = FaceHelper.EstimateMatrixByFaceLandmark5(faceLandmark5, WarpTemplate.Ffhq512, new Size(1, 1));
        using var affineMatrixDisposable = affineMatrix;

        var normalizedLandmark5 = FaceHelper.TransformPoints(faceLandmark5, affineMatrix);
        var faceLandmark685 = ForwardFan685(fan685Session, normalizedLandmark5);

        return TransformByInverse(faceLandmark685, affineMatrix);
    }

    // -----------------------------------------------------------------
    // Preprocessing
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)</c>
    /// for <c>scale</c>, and <c>(model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2])
    /// * scale) * 0.5</c> for <c>translation</c>. Computed in <see cref="double"/> throughout —
    /// the real pipeline's bounding boxes are <c>float64</c> by the time they reach the
    /// landmarker (produced by <c>face_detector.detect_faces</c>'s margin/normalisation math),
    /// so this matches Python's actual arithmetic precision here, not just its dtype-promotion
    /// rules for an assumed float32 input.
    /// </summary>
    internal static (double Scale, double[] Translation) ComputeScaleAndTranslation(float[] boundingBox, Size modelSize)
    {
        var boxWidth = (double)boundingBox[2] - boundingBox[0];
        var boxHeight = (double)boundingBox[3] - boundingBox[1];
        var maxDimension = Math.Max(boxWidth, boxHeight);
        maxDimension = Math.Max(maxDimension, 1.0); // .clip(1, None)

        var scale = 195.0 / maxDimension;
        var translationX = (modelSize.Width - (((double)boundingBox[2] + boundingBox[0]) * scale)) * 0.5;
        var translationY = (modelSize.Width - (((double)boundingBox[3] + boundingBox[1]) * scale)) * 0.5; // model_size[0] used for both, per Python

        return (scale, new[] { translationX, translationY });
    }

    /// <summary>
    /// Python: <c>conditional_optimize_contrast</c>. Caller owns the returned
    /// <see cref="Mat"/>; does not take ownership of <paramref name="cropVisionFrame"/>.
    /// </summary>
    internal static Mat ConditionalOptimizeContrast(Mat cropVisionFrame)
    {
        using var lab = new Mat();
        Cv2.CvtColor(cropVisionFrame, lab, ColorConversionCodes.RGB2Lab);

        var channels = Cv2.Split(lab);
        try
        {
            if (Cv2.Mean(channels[0]).Val0 < 30)
            {
                using var clahe = Cv2.CreateCLAHE(2, new Size(8, 8));
                clahe.Apply(channels[0], channels[0]);
            }

            using var merged = new Mat();
            Cv2.Merge(channels, merged);

            var result = new Mat();
            Cv2.CvtColor(merged, result, ColorConversionCodes.Lab2RGB);
            return result;
        }
        finally
        {
            foreach (var channel in channels)
            {
                channel.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0</c> —
    /// shared by both <c>detect_with_2dfan4</c> and <c>detect_with_peppa_wutz</c>. Unlike
    /// <c>FaceRecognizer</c>/<c>FaceClassifier</c>, there is no channel reversal here (see
    /// class remarks). Returns a flat <c>(3, H, W)</c> buffer in C order.
    /// </summary>
    internal static float[] PrepareLandmarkerInput(Mat cropVisionFrame)
    {
        var height = cropVisionFrame.Rows;
        var width = cropVisionFrame.Cols;
        var plane = height * width;
        var chw = new float[3 * plane];

        for (var row = 0; row < height; row++)
        {
            for (var col = 0; col < width; col++)
            {
                var pixel = cropVisionFrame.At<Vec3b>(row, col);
                var index = (row * width) + col;

                chw[index] = pixel.Item0 / 255f;
                chw[plane + index] = pixel.Item1 / 255f;
                chw[(2 * plane) + index] = pixel.Item2 / 255f;
            }
        }

        return chw;
    }

    // -----------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward_with_2dfan4</c>. <paramref name="crop"/> is the flat <c>(3, H, W)</c>
    /// buffer from <see cref="PrepareLandmarkerInput"/> — Python wraps the crop in a
    /// single-element list (<c>{'input': [crop_vision_frame]}</c>), which ONNX Runtime turns
    /// into an implicit <c>(1, 3, H, W)</c> batch tensor identical to explicitly adding the
    /// batch dimension here.
    /// </summary>
    internal static (float[] Landmarks, float[] Heatmaps) ForwardWith2dFan4(InferenceSession twoDFan4Session, float[] crop, Size modelSize)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(crop, new long[] { 1, 3, modelSize.Height, modelSize.Width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = twoDFan4Session.Run(runOptions, inputs, new[] { "landmarks", "heatmaps" });

        var landmarks = results[0].GetTensorDataAsSpan<float>().ToArray(); // (1, 68, 3)
        var heatmaps = results[1].GetTensorDataAsSpan<float>().ToArray(); // (1, 68, 64, 64)
        return (landmarks, heatmaps);
    }

    /// <summary>Python: <c>forward_with_peppa_wutz</c>.</summary>
    internal static float[] ForwardWithPeppaWutz(InferenceSession peppaWutzSession, float[] crop, Size modelSize)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(crop, new long[] { 1, 3, modelSize.Height, modelSize.Width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = peppaWutzSession.Run(runOptions, inputs, new[] { "output" });

        return results[0].GetTensorDataAsSpan<float>().ToArray(); // (1, 68, 3)
    }

    /// <summary>Python: <c>forward_fan_68_5</c>. <paramref name="faceLandmark5"/> is <c>(5, 2)</c>.</summary>
    internal static float[,] ForwardFan685(InferenceSession fan685Session, float[,] faceLandmark5)
    {
        var flat = new float[10];
        for (var i = 0; i < 5; i++)
        {
            flat[i * 2] = faceLandmark5[i, 0];
            flat[(i * 2) + 1] = faceLandmark5[i, 1];
        }

        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(flat, new long[] { 1, 5, 2 });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = fan685Session.Run(runOptions, inputs, new[] { "output" });

        var span = results[0].GetTensorDataAsSpan<float>(); // (1, 68, 2)
        var output = new float[68, 2];
        for (var i = 0; i < 68; i++)
        {
            output[i, 0] = span[i * 2];
            output[i, 1] = span[(i * 2) + 1];
        }

        return output;
    }

    // -----------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>numpy.amax(face_heatmap, axis = (2, 3))</c> then <c>numpy.mean(...)</c> then
    /// <c>numpy.interp(..., [0, 0.9], [0, 1])</c>. <paramref name="heatmaps"/> is the flat
    /// <c>(1, 68, 64, 64)</c> model output in C order.
    /// </summary>
    internal static double ComputeTwoDFan4Score(float[] heatmaps)
    {
        const int channelCount = 68;
        const int spatialCount = 64 * 64;

        var channelMax = new float[channelCount];
        for (var channel = 0; channel < channelCount; channel++)
        {
            var offset = channel * spatialCount;
            channelMax[channel] = NumPy.Amax(heatmaps.AsSpan(offset, spatialCount));
        }

        var meanScore = NumPy.Mean(channelMax);
        return NumPy.Interp(meanScore, new[] { 0f, 0.9f }, new[] { 0f, 1f });
    }

    // -----------------------------------------------------------------
    // Small local helpers
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>transform_points(points, cv2.invertAffineTransform(matrix))</c> — inverts
    /// <paramref name="matrix"/> and applies it via <see cref="FaceHelper.TransformPoints"/>.
    /// Does not take ownership of <paramref name="matrix"/>.
    /// </summary>
    private static float[,] TransformByInverse(float[,] points, Mat matrix)
    {
        using var inverseMatrix = new Mat();
        Cv2.InvertAffineTransform(matrix, inverseMatrix);
        return FaceHelper.TransformPoints(points, inverseMatrix);
    }
}
