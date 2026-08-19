using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for <c>facefusion/face_recognizer.py</c>. There is no
/// <c>tests/test_face_recognizer.py</c> in the Python suite, so the pure-logic cases below
/// were derived by hand from the module's numpy semantics, and the end-to-end case was
/// verified against the real <c>arcface_w600k_r50</c> ONNX model — see
/// <see cref="ModelAndMediaFactAttribute"/> and <c>tests/FaceFusion.ParityTests/
/// FaceAnalysisParityTests.cs</c> for the tight cross-language parity coverage of the same
/// path.
/// </summary>
[Collection("NativeInference")]
public sealed class FaceRecognizerTests
{
    // -----------------------------------------------------------------
    // PrepareInput — channel reversal (BGR -> RGB) and /127.5 - 1 normalisation
    // -----------------------------------------------------------------

    [Fact]
    public void TestPrepareInputReversesChannelsAndNormalizes()
    {
        using var crop = new Mat(1, 1, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(10, 20, 30)); // B=10, G=20, R=30 (OpenCV/FaceHelper BGR convention)

        var chw = FaceRecognizer.PrepareInput(crop);

        Assert.Equal(3, chw.Length);
        // Python: crop_vision_frame[:, :, ::-1] reverses BGR -> RGB, so channel 0 is R.
        AssertClose((30 / 127.5) - 1.0, chw[0]); // R -> channel 0
        AssertClose((20 / 127.5) - 1.0, chw[1]); // G -> channel 1
        AssertClose((10 / 127.5) - 1.0, chw[2]); // B -> channel 2
    }

    [Fact]
    public void TestPrepareInputProducesCorrectShapeAndChannelPlaneLayout()
    {
        using var crop = new Mat(2, 3, MatType.CV_8UC3);
        for (var row = 0; row < 2; row++)
        {
            for (var col = 0; col < 3; col++)
            {
                crop.Set(row, col, new Vec3b((byte)(row * 10), (byte)(col * 10), (byte)(row + col)));
            }
        }

        var chw = FaceRecognizer.PrepareInput(crop);

        // (3, H, W) flat C-order: channel-major, then row-major within a channel.
        Assert.Equal(3 * 2 * 3, chw.Length);
        var plane = 2 * 3;
        for (var row = 0; row < 2; row++)
        {
            for (var col = 0; col < 3; col++)
            {
                var index = (row * 3) + col;
                var pixel = crop.At<Vec3b>(row, col);
                AssertClose((pixel.Item2 / 127.5) - 1.0, chw[index]);
                AssertClose((pixel.Item1 / 127.5) - 1.0, chw[plane + index]);
                AssertClose((pixel.Item0 / 127.5) - 1.0, chw[(2 * plane) + index]);
            }
        }
    }

    [Fact]
    public void TestPrepareInputExtremesMapToMinusOneAndOne()
    {
        using var crop = new Mat(1, 2, MatType.CV_8UC3);
        crop.Set(0, 0, new Vec3b(0, 0, 0));
        crop.Set(0, 1, new Vec3b(255, 255, 255));

        var chw = FaceRecognizer.PrepareInput(crop);

        // pixel 0 -> all channels 0/127.5 - 1 = -1
        Assert.Equal(-1.0, (double)chw[0], 6);
        // pixel 1 (index 1) -> 255/127.5 - 1 = 1
        var plane = 2;
        Assert.Equal(1.0, (double)chw[plane + 1], 5);
    }

    // -----------------------------------------------------------------
    // CalculateFaceEmbedding — geometry + normalisation, model options
    // -----------------------------------------------------------------

    [Fact]
    public void TestModelOptionsMatchPython()
    {
        // Python: create_static_model_set('full')['arcface']['template'] == 'arcface_112_v2',
        // ['size'] == (112, 112).
        Assert.Equal(WarpTemplate.Arcface112V2, FaceRecognizer.ModelTemplate);
        Assert.Equal(112, FaceRecognizer.ModelSize.Width);
        Assert.Equal(112, FaceRecognizer.ModelSize.Height);
    }

    [ModelAndMediaFact("arcface_w600k_r50.onnx")]
    public void TestCalculateFaceEmbeddingAgainstRealModel()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(TestHelper.GetTestExampleFile("source.jpg"));
        Assert.NotNull(visionFrame);

        // A plausible landmark-5 set (roughly centred, arcface_112_v2-template-shaped) is
        // enough to exercise the full CalculateFaceEmbedding path end to end; exact-value
        // parity against the real Python pipeline is FaceAnalysisParityTests's job.
        var faceLandmark5 = new float[5, 2]
        {
            { 380f, 380f },
            { 620f, 380f },
            { 500f, 500f },
            { 400f, 620f },
            { 600f, 620f },
        };

        using var session = new InferenceSession(ModelAndMediaFactAttribute.FindModelPath("arcface_w600k_r50.onnx"));
        var (embedding, embeddingNorm) = FaceRecognizer.CalculateFaceEmbedding(session, visionFrame!, faceLandmark5);

        Assert.Equal(512, embedding.Length);
        Assert.Equal(512, embeddingNorm.Length);

        // embedding_norm must be unit-length (Python: face_embedding / numpy.linalg.norm(face_embedding)).
        var normSquared = 0.0;
        foreach (var value in embeddingNorm)
        {
            normSquared += (double)value * value;
        }

        Assert.True(Math.Abs(Math.Sqrt(normSquared) - 1.0) < 1e-4, $"embedding_norm L2 norm was {Math.Sqrt(normSquared)}, expected ~1.0");

        // embeddingNorm must point the same direction as embedding (it's embedding / ||embedding||).
        var norm = (float)Math.Sqrt(embedding.Sum(value => (double)value * value));
        for (var i = 0; i < embedding.Length; i++)
        {
            Assert.Equal((double)(embedding[i] / norm), (double)embeddingNorm[i], 4);
        }
    }

    private static void AssertClose(double expected, double actual) => Assert.True(Math.Abs(expected - actual) < 1e-4, $"expected {expected}, actual {actual}");
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time unless the example media (see
/// <see cref="TestHelper.ExamplesAvailable"/>) and every named <c>.assets/models/*.onnx</c>
/// file are present — shared by <c>FaceRecognizerTests</c>, <c>FaceClassifierTests</c> and
/// <c>FaceLandmarkerTests</c>. Same reasoning as <see cref="MediaFactAttribute"/>, extended to
/// cover the (`.gitignore`'d, network-downloaded, never present in CI) ONNX model files these
/// three modules need for their end-to-end cases.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class ModelAndMediaFactAttribute : FactAttribute
{
    public ModelAndMediaFactAttribute(params string[] modelFileNames)
    {
        if (!TestHelper.ExamplesAvailable)
        {
            Skip = TestHelper.MissingMediaMessage;
            return;
        }

        foreach (var modelFileName in modelFileNames)
        {
            var modelPath = FindModelPath(modelFileName);
            if (modelPath is null || !File.Exists(modelPath) || new FileInfo(modelPath).Length == 0)
            {
                Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                       "run the Python pre_check() for this model (e.g. via " +
                       "tools/parity/dump_face_analysis.py with network access), then retry";
                return;
            }
        }
    }

    /// <summary>
    /// Walks up from the test assembly's output directory looking for <c>FaceFusion.sln</c>,
    /// then returns <c>&lt;repo root&gt;/.assets/models/&lt;modelFileName&gt;</c> — the same
    /// path Python's <c>resolve_relative_path</c> + <c>pre_check()</c> populate. Mirrors
    /// <c>ModelHelperTests</c>'s repo-root walk for locating its own fixtures.
    /// </summary>
    public static string? FindModelPath(string modelFileName)
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return Path.Combine(directory.FullName, ".assets", "models", modelFileName);
            }

            directory = directory.Parent;
        }

        return null;
    }
}
