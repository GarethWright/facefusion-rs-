using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Parity;
using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Processors.FaceSwapper</c> against the real Python
/// <c>facefusion.processors.modules.face_swapper.core</c>, run against the real
/// <c>inswapper_128</c> and <c>ghost_1_256</c> (+ its <c>crossface_ghost</c> embedding
/// converter) ONNX models and the largest detected face on the real <c>source.jpg</c> example
/// image (source == target == reference frame, per the dumper's docstring). Ground truth was
/// captured with <c>tools/parity/dump_face_swapper.py</c>; see that script's docstring and
/// docs/PARITY_HARNESS.md for why these two families and not all thirteen.
///
/// <para>
/// <b>Two tiers, gated differently</b> — same split as <c>FaceAnalysisParityTests</c>: the
/// preprocessing-tensor tests (channel reversal/normalisation, embedding balancing, pixel-boost
/// math) need only the committed <c>.npy</c> fixtures and the source image, so they run once
/// that media is present; the end-to-end tests that run a real <see cref="InferenceSession"/>
/// additionally need the corresponding <c>.onnx</c> file(s) under <c>.assets/models/</c>
/// (<c>.gitignore</c>'d, never present on CI) and are gated with
/// <see cref="FaceSwapperModelFactAttribute"/>.
/// </para>
///
/// <para>
/// <b>Model-input tensors matched Python exactly (rtol=atol=0) — see
/// <see cref="TestPrepareCropFrameMatchesPythonExactly"/>/
/// <see cref="TestPrepareSourceEmbeddingAndBalanceMatchPythonExactly"/> — the assignment's bar
/// is met.</b> Where ONNX Runtime then does the arithmetic (the swap itself, the embedding
/// converter), <see cref="TestForwardSwapFaceMatchesPythonRawModelOutput"/>/
/// <see cref="TestConvertSourceEmbeddingMatchesPython"/> use a tight (not loosened) tolerance
/// per PARITY_HARNESS.md's "expect ~0 divergence" guidance for that tier.
/// </para>
/// </summary>
public sealed class FaceSwapperParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "face_swapper");

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    // -----------------------------------------------------------------
    // Shared fixture / environment helpers
    // -----------------------------------------------------------------

    internal static string? FindRepoRoot()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    internal static string? FindModelPath(string modelFileName)
    {
        var repoRoot = FindRepoRoot();
        return repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models", modelFileName);
    }

    internal static bool ModelAvailable(string modelFileName)
    {
        var modelPath = FindModelPath(modelFileName);
        return modelPath is not null && File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
    }

    internal static bool SourceImageAvailable =>
        File.Exists(SourceImage) && new FileInfo(SourceImage).Length > 0;

    private static NpyArray LoadNpy(string family, string name) =>
        NpyReader.Load(Path.Combine(FixturesDirectory, family, name + ".npy"));

    private static Mat MatFromUInt8HwcFixture(NpyArray array)
    {
        Assert.Equal("uint8", array.DType);
        Assert.Equal(3, array.Shape.Count);
        Assert.Equal(3, array.Shape[2]);

        var height = array.Shape[0];
        var width = array.Shape[1];
        var raw = array.RawData;

        var mat = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[height * width];
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Vec3b(raw[i * 3], raw[(i * 3) + 1], raw[(i * 3) + 2]);
        }

        mat.SetArray(pixels);
        return mat;
    }

    private static float[,] LoadLandmark5(NpyArray array)
    {
        Assert.Equal(new[] { 5, 2 }, array.Shape);
        var values = array.AsFloats();
        var result = new float[5, 2];
        for (var i = 0; i < 5; i++)
        {
            result[i, 0] = values[i * 2];
            result[i, 1] = values[(i * 2) + 1];
        }

        return result;
    }

    // -----------------------------------------------------------------
    // prepare_crop_frame — 'target' model input, no ONNX Runtime required
    // -----------------------------------------------------------------

    [FaceSwapperSourceImageFact]
    public void TestPrepareCropFrameMatchesPythonExactlyForInswapper() =>
        AssertPrepareCropFrameMatchesPython("inswapper_128", FaceSwapperModel.Inswapper128);

    [FaceSwapperSourceImageFact]
    public void TestPrepareCropFrameMatchesPythonExactlyForGhost() =>
        AssertPrepareCropFrameMatchesPython("ghost_1_256", FaceSwapperModel.Ghost1256);

    private static void AssertPrepareCropFrameMatchesPython(string family, FaceSwapperModel model)
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy(family, "crop_vision_frame"));
        var modelOptions = FaceSwapper.CreateStaticModelSet(DownloadScope.Full)[model];

        var actual = FaceSwapper.PrepareCropFrame(cropVisionFrame, modelOptions.Mean, modelOptions.StandardDeviation);
        var expected = LoadNpy(family, "target_input").AsDoubles();

        var actualDoubles = Array.ConvertAll(actual, value => (double)value);

        // Pure managed preprocessing (BGR->RGB, mean/std normalisation) reproducing numpy's
        // promote-to-float64-then-narrow behaviour exactly — PARITY_HARNESS.md's "expect ~0"
        // bar for a stage with no ONNX Runtime/OpenCV arithmetic involved.
        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // prepare_source_embedding + balance_source_embedding — 'source' model input
    // -----------------------------------------------------------------

    [FaceSwapperModelFact("inswapper_128.onnx")]
    public void TestPrepareSourceEmbeddingAndBalanceMatchPythonExactlyForInswapper()
    {
        var sourceEmbedding = LoadNpy("inswapper_128", "source_embedding").AsFloats();
        var targetEmbedding = LoadNpy("inswapper_128", "target_embedding").AsFloats();

        // Reuses FaceFusion.Inference.ModelHelper.GetStaticModelInitializer (already ported,
        // per the assignment) against the real inswapper_128.onnx rather than reconstructing an
        // OnnxTensor from the fixture (OnnxTensor's constructor is internal to
        // FaceFusion.Inference and this project must not touch that assembly) — this also cross
        // -checks the reused ModelHelper against Python's onnx.numpy_helper.to_array.
        var initializer = ModelHelper.GetStaticModelInitializer(FindModelPath("inswapper_128.onnx")!);
        var expectedInitializer = LoadNpy("inswapper_128", "model_initializer").AsDoubles();
        var initializerResult = TensorComparison.Compare(Array.ConvertAll(initializer.AsFloats(), v => (double)v), expectedInitializer, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(initializerResult.Passed, $"model_initializer: {initializerResult.Describe()}");

        var prepared = FaceSwapper.PrepareSourceEmbedding(FaceSwapperModelKind.Inswapper, sourceEmbedding, sourceEmbedding, embeddingConverterSession: null, initializer);
        var expectedPrepared = LoadNpy("inswapper_128", "prepared_source_embedding").AsDoubles();
        var preparedResult = TensorComparison.Compare(Array.ConvertAll(prepared, v => (double)v), expectedPrepared, relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(preparedResult.Passed, preparedResult.Describe());

        var balanced = FaceSwapper.BalanceSourceEmbedding(FaceSwapperModelKind.Inswapper, prepared, targetEmbedding, faceSwapperWeight: 0.5);
        var expectedBalanced = LoadNpy("inswapper_128", "source_input").AsDoubles();
        var balancedResult = TensorComparison.Compare(Array.ConvertAll(balanced, v => (double)v), expectedBalanced, relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(balancedResult.Passed, balancedResult.Describe());
    }

    [FaceSwapperModelFact("crossface_ghost.onnx")]
    public void TestPrepareSourceEmbeddingAndBalanceMatchPythonExactlyForGhost()
    {
        var sourceEmbedding = LoadNpy("ghost_1_256", "source_embedding").AsFloats();
        var targetEmbedding = LoadNpy("ghost_1_256", "target_embedding").AsFloats();

        using var embeddingConverterSession = new InferenceSession(FindModelPath("crossface_ghost.onnx"));

        var prepared = FaceSwapper.PrepareSourceEmbedding(FaceSwapperModelKind.Ghost, sourceEmbedding, sourceEmbedding, embeddingConverterSession, inswapperModelInitializer: null);
        var expectedPrepared = LoadNpy("ghost_1_256", "prepared_source_embedding").AsDoubles();

        // Real ONNX Runtime pass inside PrepareSourceEmbedding (ConvertSourceEmbedding) — tight
        // tolerance per PARITY_HARNESS.md ("Where ONNX Runtime does the arithmetic, expect ~0").
        var preparedResult = TensorComparison.Compare(Array.ConvertAll(prepared, v => (double)v), expectedPrepared, relativeTolerance: 1e-5, absoluteTolerance: 1e-5);
        Assert.True(preparedResult.Passed, preparedResult.Describe());

        var balanced = FaceSwapper.BalanceSourceEmbedding(FaceSwapperModelKind.Ghost, prepared, targetEmbedding, faceSwapperWeight: 0.5);
        var expectedBalanced = LoadNpy("ghost_1_256", "source_input").AsDoubles();
        var balancedResult = TensorComparison.Compare(Array.ConvertAll(balanced, v => (double)v), expectedBalanced, relativeTolerance: 1e-5, absoluteTolerance: 1e-5);
        Assert.True(balancedResult.Passed, balancedResult.Describe());
    }

    // -----------------------------------------------------------------
    // box_mask — reused FaceMasker, sanity-checked (not re-verifying FaceMasker's own parity)
    // -----------------------------------------------------------------

    [FaceSwapperSourceImageFact]
    public void TestBoxMaskMatchesPythonForInswapper() => AssertBoxMaskMatchesPython("inswapper_128");

    [FaceSwapperSourceImageFact]
    public void TestBoxMaskMatchesPythonForGhost() => AssertBoxMaskMatchesPython("ghost_1_256");

    private static void AssertBoxMaskMatchesPython(string family)
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy(family, "crop_vision_frame"));
        using var actual = FaceMasker.CreateBoxMask(cropVisionFrame, faceMaskBlur: 0.3, faceMaskPadding: new Padding(0, 0, 0, 0));

        var expected = LoadNpy(family, "box_mask");
        Assert.Equal(new[] { actual.Rows, actual.Cols }, expected.Shape);

        actual.GetArray(out float[] actualValues);
        var expectedValues = expected.AsDoubles();

        var result = TensorComparison.Compare(Array.ConvertAll(actualValues, v => (double)v), expectedValues, relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference)
    // -----------------------------------------------------------------

    [FaceSwapperModelFact("inswapper_128.onnx")]
    public void TestForwardSwapFaceMatchesPythonRawModelOutputForInswapper()
    {
        RunForwardSwapFaceTest("inswapper_128", FaceSwapperModel.Inswapper128, FaceSwapperModelKind.Inswapper, "inswapper_128.onnx", sourceInputSize: null);
    }

    [FaceSwapperModelFact("ghost_1_256.onnx", "crossface_ghost.onnx")]
    public void TestForwardSwapFaceMatchesPythonRawModelOutputForGhost()
    {
        RunForwardSwapFaceTest("ghost_1_256", FaceSwapperModel.Ghost1256, FaceSwapperModelKind.Ghost, "ghost_1_256.onnx", sourceInputSize: null);
    }

    private static void RunForwardSwapFaceTest(string family, FaceSwapperModel model, FaceSwapperModelKind kind, string faceSwapperModelFileName, Size? sourceInputSize)
    {
        var modelOptions = FaceSwapper.CreateStaticModelSet(DownloadScope.Full)[model];

        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy(family, "crop_vision_frame"));
        var targetInput = FaceSwapper.PrepareCropFrame(cropVisionFrame, modelOptions.Mean, modelOptions.StandardDeviation);
        var sourceInput = LoadNpy(family, "source_input").AsFloats();

        using var faceSwapperSession = new InferenceSession(FindModelPath(faceSwapperModelFileName));
        var rawOutput = FaceSwapper.ForwardSwapFace(faceSwapperSession, sourceInput, sourceInputSize, targetInput, modelOptions.Size);

        var expectedRawOutput = LoadNpy(family, "raw_model_output").AsDoubles();
        var result = TensorComparison.Compare(Array.ConvertAll(rawOutput, v => (double)v), expectedRawOutput, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(result.Passed, $"[{family}] raw model output: {result.Describe()}");

        using var normalized = FaceSwapper.NormalizeCropFrame(rawOutput, modelOptions.Size.Height, modelOptions.Size.Width, kind, modelOptions.Mean, modelOptions.StandardDeviation);
        var expectedNormalized = LoadNpy(family, "normalized_crop_frame").AsDoubles();

        double[] actualNormalized;
        if (normalized.Type() == MatType.CV_64FC3)
        {
            normalized.GetArray(out Vec3d[] pixels);
            actualNormalized = new double[pixels.Length * 3];
            for (var i = 0; i < pixels.Length; i++)
            {
                actualNormalized[i * 3] = pixels[i].Item0;
                actualNormalized[(i * 3) + 1] = pixels[i].Item1;
                actualNormalized[(i * 3) + 2] = pixels[i].Item2;
            }
        }
        else
        {
            normalized.GetArray(out Vec3f[] pixels);
            actualNormalized = new double[pixels.Length * 3];
            for (var i = 0; i < pixels.Length; i++)
            {
                actualNormalized[i * 3] = pixels[i].Item0;
                actualNormalized[(i * 3) + 1] = pixels[i].Item1;
                actualNormalized[(i * 3) + 2] = pixels[i].Item2;
            }
        }

        var normalizedResult = TensorComparison.Compare(actualNormalized, expectedNormalized, relativeTolerance: 1e-4, absoluteTolerance: 1e-3);
        Assert.True(normalizedResult.Passed, $"[{family}] normalized_crop_frame: {normalizedResult.Describe()}");
    }

    [FaceSwapperModelFact("crossface_ghost.onnx")]
    public void TestConvertSourceEmbeddingMatchesPython()
    {
        var sourceEmbedding = LoadNpy("ghost_1_256", "source_embedding").AsFloats();

        using var embeddingConverterSession = new InferenceSession(FindModelPath("crossface_ghost.onnx"));
        var (embedding, _) = FaceSwapper.ConvertSourceEmbedding(embeddingConverterSession, sourceEmbedding);

        // prepared_source_embedding IS convert_source_embedding's first return value reshaped
        // (1, -1) for the ghost family — see dump_face_swapper.py.
        var expected = LoadNpy("ghost_1_256", "prepared_source_embedding").AsDoubles();
        var result = TensorComparison.Compare(Array.ConvertAll(embedding, v => (double)v), expected, relativeTolerance: 1e-5, absoluteTolerance: 1e-5);
        Assert.True(result.Passed, result.Describe());
    }

}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the example media
/// (<c>/tmp/facefusion-test-examples/source.jpg</c>) is not present — same reasoning as
/// <c>FaceAnalysisParityTests.SkippableFactAttribute</c>, given a distinct, non-reflection-based
/// implementation here since every case in this file needs the same one property.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class FaceSwapperSourceImageFactAttribute : FactAttribute
{
    public FaceSwapperSourceImageFactAttribute()
    {
        if (!FaceSwapperParityTests.SourceImageAvailable)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.jpg) — run tools/parity/fetch_examples.sh, then retry";
        }
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c>
/// file(s) (or the example media) are not present — same reasoning as
/// <c>FaceAnalysisParityTests.ModelFactAttribute</c>, given a distinct name here because
/// attribute constructors run at discovery time against a specific class's static helpers.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class FaceSwapperModelFactAttribute : FactAttribute
{
    public FaceSwapperModelFactAttribute(params string[] modelFileNames)
    {
        if (!FaceSwapperParityTests.SourceImageAvailable)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.jpg) — run tools/parity/fetch_examples.sh, then retry";
            return;
        }

        foreach (var modelFileName in modelFileNames)
        {
            if (!FaceSwapperParityTests.ModelAvailable(modelFileName))
            {
                Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                       "run `FACEFUSION_PARITY_DIR=... python3 tools/parity/dump_face_swapper.py` once with " +
                       "network access to populate .assets/models via pre_check(), then retry";
                return;
            }
        }
    }
}
