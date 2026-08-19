using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_recognizer.py</c> — the ArcFace face-embedding stage.
///
/// <para>
/// <b>Model / session wiring (documented divergence).</b> Python's <c>get_inference_pool</c>
/// resolves model URLs via <c>facefusion.download.resolve_download_url</c> and downloads
/// through <c>conditional_download_hashes</c>/<c>conditional_download_sources</c> —
/// <c>facefusion/download.py</c> has no C# port yet (out of this module's assignment; see
/// PORT_CONVENTIONS.md rule 5 / "do not port a module that is not in your assignment"), so
/// there is no way to resolve those URLs from here. Per port convention rule 5 ("if your
/// module reads state_manager, take the value as a parameter instead"), the same treatment is
/// extended to the inference session itself: <see cref="CalculateFaceEmbedding"/> takes an
/// already-created <see cref="InferenceSession"/> for the <c>arcface_w600k_r50</c> model
/// (loaded from <c>.assets/models/arcface_w600k_r50.onnx</c> the same way Python's
/// <c>pre_check()</c> would place it) rather than owning an <c>InferenceManager</c> pool
/// keyed by a download URL it cannot resolve. A later phase that ports <c>download.py</c> can
/// add a thin <c>GetInferencePool</c> wrapper around <see cref="ModelTemplate"/>/
/// <see cref="ModelSize"/> without touching the math here.
/// </para>
///
/// <para>
/// <b>VisionFrame channel order.</b> Per <c>FaceFusion.Face.FaceHelper</c>'s convention, a
/// <c>VisionFrame</c> <see cref="Mat"/> is 8-bit BGR (matching what <c>Cv2.ImRead</c>/
/// <c>Cv2.WarpAffine</c> actually produce), the same order Python's numpy <c>VisionFrame</c>
/// carries despite being nominally created with <c>color_mode='rgb'</c> (see
/// <c>facefusion/vision.py</c>'s <c>read_image</c>: the color_mode flag only ever selects
/// <c>cv2.IMREAD_COLOR</c> vs <c>cv2.IMREAD_UNCHANGED</c>; it never calls
/// <c>cv2.cvtColor</c>). Python's <c>calculate_face_embedding</c> reverses that BGR channel
/// order with <c>crop_vision_frame[:, :, ::-1]</c> before feeding the model (i.e. it feeds
/// RGB); <see cref="PrepareInput"/> reproduces the same reversal by reading
/// <see cref="Vec3b"/>.Item2/Item1/Item0 (R/G/B) into channel-major order 0/1/2.
/// </para>
///
/// <para>
/// <b>Dtype (float32 vs float64), reproduced exactly.</b> Python: <c>crop_vision_frame /
/// 127.5 - 1</c> divides a <c>uint8</c> array by a Python <c>float</c>, which numpy promotes
/// to <c>float64</c> (true division of an integer array always yields the default float
/// dtype); only the *next* line's <c>.astype(numpy.float32)</c> narrows it, after the
/// channel-reversal and HWC-&gt;CHW transpose (both no-ops for precision). So the
/// <c>127.5</c>/<c>-1</c> normalisation itself happens in double precision in Python, and
/// <see cref="PrepareInput"/> reproduces that: each channel value is computed as
/// <c>(byte / 127.5) - 1</c> in <see cref="double"/> and only narrowed to <see cref="float"/>
/// at the point of assignment into the CHW buffer, matching Python's per-element rounding
/// exactly (verified empirically — see PORT_CONVENTIONS.md rule 6).
/// </para>
/// </summary>
public static class FaceRecognizer
{
    /// <summary>Python: <c>get_model_options().get('template')</c>.</summary>
    public const WarpTemplate ModelTemplate = WarpTemplate.Arcface112V2;

    /// <summary>Python: <c>get_model_options().get('size')</c>.</summary>
    public static readonly Size ModelSize = new(112, 112);

    /// <summary>
    /// Python: <c>calculate_face_embedding</c>. Returns the raw embedding and its L2-normed
    /// counterpart. Does not take ownership of <paramref name="tempVisionFrame"/>.
    /// </summary>
    public static (float[] Embedding, float[] EmbeddingNorm) CalculateFaceEmbedding(
        InferenceSession faceRecognizerSession, Mat tempVisionFrame, float[,] faceLandmark5)
    {
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, faceLandmark5, ModelTemplate, ModelSize);
        using var cropDisposable = cropVisionFrame;
        using var matrixDisposable = affineMatrix;

        var inputTensor = PrepareInput(cropVisionFrame);
        var faceEmbedding = Forward(faceRecognizerSession, inputTensor);

        // Python: `face_embedding.ravel()` — already flat here (Forward returns the raw
        // (1, 512) output as a 512-length span), so no reshape is needed.
        var norm = NumPy.LinalgNorm(faceEmbedding);
        var faceEmbeddingNorm = new float[faceEmbedding.Length];
        for (var i = 0; i < faceEmbedding.Length; i++)
        {
            faceEmbeddingNorm[i] = faceEmbedding[i] / norm;
        }

        return (faceEmbedding, faceEmbeddingNorm);
    }

    /// <summary>
    /// Python: the preprocessing half of <c>calculate_face_embedding</c>
    /// (<c>crop_vision_frame / 127.5 - 1</c>, channel reversal, HWC-&gt;CHW, <c>float32</c>
    /// cast, batch dim). Returns a flat <c>(1, 3, 112, 112)</c> buffer in C order. Exposed for
    /// parity tests that need to assert the exact tensor handed to <c>session.Run</c>.
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

                // BGR -> RGB reversal (Python's `[:, :, ::-1]`), normalised in double
                // precision then narrowed to float32 (see class remarks).
                chw[index] = (float)((pixel.Item2 / 127.5) - 1.0); // R -> channel 0
                chw[plane + index] = (float)((pixel.Item1 / 127.5) - 1.0); // G -> channel 1
                chw[(2 * plane) + index] = (float)((pixel.Item0 / 127.5) - 1.0); // B -> channel 2
            }
        }

        return chw;
    }

    /// <summary>Python: <c>forward</c>.</summary>
    public static float[] Forward(InferenceSession faceRecognizerSession, float[] cropVisionFrame)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(cropVisionFrame, new long[] { 1, 3, ModelSize.Height, ModelSize.Width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = faceRecognizerSession.Run(runOptions, inputs, new[] { "output" });

        return results[0].GetTensorDataAsSpan<float>().ToArray();
    }
}
