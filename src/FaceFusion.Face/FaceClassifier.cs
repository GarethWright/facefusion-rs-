using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_classifier.py</c> — the FairFace gender/age/race stage.
///
/// <para>
/// <b>Model / session wiring (documented divergence).</b> Same reasoning as
/// <c>FaceFusion.Face.FaceRecognizer</c>: <c>facefusion/download.py</c> has no C# port yet, so
/// <see cref="ClassifyFace"/> takes an already-created <see cref="InferenceSession"/> for the
/// <c>fairface</c> model rather than owning a download-backed <c>InferenceManager</c> pool.
/// </para>
///
/// <para>
/// <b>VisionFrame channel order.</b> Same BGR convention as <c>FaceHelper</c>/
/// <c>FaceRecognizer</c> (see their remarks). Python reverses it with
/// <c>crop_vision_frame[:, :, ::-1]</c> before feeding the model; <see cref="PrepareInput"/>
/// reproduces that the same way <c>FaceRecognizer.PrepareInput</c> does.
/// </para>
///
/// <para>
/// <b>Dtype, reproduced exactly.</b> Python: <c>crop_vision_frame.astype(numpy.float32)[:, :,
/// ::-1] / 255.0</c> casts to <c>float32</c> *first*, so the whole normalisation
/// (divide-by-255, subtract mean, divide by standard deviation) happens in single precision
/// throughout — unlike <c>FaceRecognizer</c>, where the equivalent division happens in
/// <c>float64</c> before the later cast. Verified against real numpy (see the port report);
/// <see cref="PrepareInput"/> does every step in <see cref="float"/>.
/// </para>
///
/// <para>
/// <b>Output plumbing, not a bug.</b> Python's <c>forward</c> destructures
/// <c>session.run(...)</c> positionally into locals named <c>race_id, gender_id, age_id</c>
/// (matching the model's actual graph output order: <c>race_id</c>, <c>gender_id</c>,
/// <c>age_id</c>) and then *returns* them reordered as <c>(gender_id, age_id, race_id)</c>;
/// <c>classify_face</c> destructures that returned tuple back into
/// <c>gender_id, age_id, race_id</c>. The two reorderings cancel out — each local ends up
/// holding the model output with the matching semantic name, not a mismatched one. This port
/// sidesteps the shuffle entirely by requesting outputs by name (<c>race_id</c>,
/// <c>gender_id</c>, <c>age_id</c>) from ONNX Runtime directly; see <see cref="Forward"/>.
/// </para>
/// </summary>
public static class FaceClassifier
{
    /// <summary>Python: <c>get_model_options().get('template')</c>.</summary>
    public const WarpTemplate ModelTemplate = WarpTemplate.Arcface112V2;

    /// <summary>Python: <c>get_model_options().get('size')</c>.</summary>
    public static readonly Size ModelSize = new(224, 224);

    /// <summary>Python: <c>get_model_options().get('mean')</c>.</summary>
    public static readonly float[] ModelMean = { 0.485f, 0.456f, 0.406f };

    /// <summary>Python: <c>get_model_options().get('standard_deviation')</c>.</summary>
    public static readonly float[] ModelStandardDeviation = { 0.229f, 0.224f, 0.225f };

    /// <summary>
    /// Python: <c>classify_face</c>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>.
    /// </summary>
    public static (Gender Gender, System.Range Age, Race Race) ClassifyFace(
        InferenceSession faceClassifierSession, Mat tempVisionFrame, float[,] faceLandmark5)
    {
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, faceLandmark5, ModelTemplate, ModelSize);
        using var cropDisposable = cropVisionFrame;
        using var matrixDisposable = affineMatrix;

        var inputTensor = PrepareInput(cropVisionFrame);
        var (genderId, ageId, raceId) = Forward(faceClassifierSession, inputTensor);

        return (CategorizeGender(genderId), CategorizeAge(ageId), CategorizeRace(raceId));
    }

    /// <summary>
    /// Python: the preprocessing half of <c>classify_face</c> (<c>astype(float32)</c>,
    /// channel reversal, <c>/ 255.0</c>, subtract mean, divide by standard deviation,
    /// HWC-&gt;CHW, batch dim). Returns a flat <c>(1, 3, 224, 224)</c> buffer in C order.
    /// Exposed for parity tests that need to assert the exact tensor handed to
    /// <c>session.Run</c>.
    /// </summary>
    public static float[] PrepareInput(Mat cropVisionFrame)
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

                // BGR -> RGB reversal (Python's `[:, :, ::-1]`), everything in float32.
                var r = (pixel.Item2 / 255f - ModelMean[0]) / ModelStandardDeviation[0];
                var g = (pixel.Item1 / 255f - ModelMean[1]) / ModelStandardDeviation[1];
                var b = (pixel.Item0 / 255f - ModelMean[2]) / ModelStandardDeviation[2];

                chw[index] = r; // -> channel 0
                chw[plane + index] = g; // -> channel 1
                chw[(2 * plane) + index] = b; // -> channel 2
            }
        }

        return chw;
    }

    /// <summary>
    /// Python: <c>forward</c>. Requests outputs by name (<c>race_id</c>, <c>gender_id</c>,
    /// <c>age_id</c>) — see the class remarks on why Python's positional destructuring is not
    /// reproduced literally.
    /// </summary>
    public static (long GenderId, long AgeId, long RaceId) Forward(InferenceSession faceClassifierSession, float[] cropVisionFrame)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(cropVisionFrame, new long[] { 1, 3, ModelSize.Height, ModelSize.Width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = faceClassifierSession.Run(runOptions, inputs, new[] { "race_id", "gender_id", "age_id" });

        var raceId = results[0].GetTensorDataAsSpan<long>()[0];
        var genderId = results[1].GetTensorDataAsSpan<long>()[0];
        var ageId = results[2].GetTensorDataAsSpan<long>()[0];

        return (genderId, ageId, raceId);
    }

    /// <summary>Python: <c>categorize_gender</c>.</summary>
    public static Gender CategorizeGender(long genderId)
    {
        if (genderId == 1)
        {
            return Gender.Female;
        }

        return Gender.Male;
    }

    /// <summary>Python: <c>categorize_age</c>.</summary>
    public static System.Range CategorizeAge(long ageId)
    {
        if (ageId == 0)
        {
            return 0..2;
        }

        if (ageId == 1)
        {
            return 3..9;
        }

        if (ageId == 2)
        {
            return 10..19;
        }

        if (ageId == 3)
        {
            return 20..29;
        }

        if (ageId == 4)
        {
            return 30..39;
        }

        if (ageId == 5)
        {
            return 40..49;
        }

        if (ageId == 6)
        {
            return 50..59;
        }

        if (ageId == 7)
        {
            return 60..69;
        }

        return 70..100;
    }

    /// <summary>Python: <c>categorize_race</c>.</summary>
    public static Race CategorizeRace(long raceId)
    {
        if (raceId == 1)
        {
            return Race.Black;
        }

        if (raceId == 2)
        {
            return Race.Latino;
        }

        if (raceId == 3 || raceId == 4)
        {
            return Race.Asian;
        }

        if (raceId == 5)
        {
            return Race.Indian;
        }

        if (raceId == 6)
        {
            return Race.Arabic;
        }

        return Race.White;
    }
}
