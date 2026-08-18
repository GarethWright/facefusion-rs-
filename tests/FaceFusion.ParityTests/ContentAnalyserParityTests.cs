using System.Text.Json;
using FaceFusion.Face;
using FaceFusion.Parity;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Face.ContentAnalyser</c> against the real Python
/// <c>facefusion.content_analyser</c>, run against the real <c>nsfw_1</c>/<c>nsfw_2</c>/
/// <c>nsfw_3</c> ONNX models and the real <c>source.jpg</c> example image. Ground truth was
/// captured with <c>tools/parity/dump_content_analyser.py</c>; see that script's docstring
/// and docs/PARITY_HARNESS.md.
///
/// <para>
/// <b>Two tiers of tests, gated differently</b> — same split as
/// <c>FaceFusion.ParityTests.FaceAnalysisParityTests</c>. The preprocessing-tensor tests
/// (<see cref="PrepareDetectFrameMatchesPython"/>) need only the committed <c>.npy</c>
/// fixtures and <c>source.jpg</c> — no <c>.onnx</c> model — so they run unconditionally once
/// the example media is present. The end-to-end tests that run a real
/// <see cref="Microsoft.ML.OnnxRuntime.InferenceSession"/> additionally need the corresponding
/// model file under <c>.assets/models/</c>, which is <c>.gitignore</c>'d and never present on
/// CI; those are gated with <see cref="ContentAnalyserModelFactAttribute"/> and skip with a clear message
/// instead of failing. If the preprocessing test passes and an end-to-end test still disagrees
/// beyond ~0, the bug is ONNX Runtime's own arithmetic diverging (unexpected — both sides call
/// the same kernels) rather than this port's tensor construction; if the preprocessing test
/// fails, the bug is in <see cref="ContentAnalyser.PrepareDetectFrame"/>.
/// </para>
///
/// <para>
/// <b>Tolerance.</b> Preprocessing tensors are compared at <c>rtol = 1e-6, atol = 1e-6</c> —
/// tighter than <see cref="TensorComparison"/>'s numpy-matching default
/// (<c>1e-5</c>/<c>1e-8</c>), because both sides compute the exact same per-pixel double
/// arithmetic (see <see cref="ContentAnalyser.PrepareDetectFrame"/>'s own remarks) and a real
/// preprocessing bug should show up as far more than a 1e-6 divergence — this is "expect ~0"
/// per PARITY_HARNESS.md, not a loosened tolerance to force a pass. Raw model outputs
/// (<c>nsfw_1</c>'s (1,9,8400) tensor) use the harness default
/// (<c>rtol=1e-5, atol=1e-8</c>) since ONNX Runtime is the one doing that arithmetic. Scalar
/// scores (already a handful of derived floating-point operations on top of the raw output)
/// are compared with <see cref="Assert.Equal(double, double, int)"/> at 4 decimal places.
/// </para>
/// </summary>
public sealed class ContentAnalyserParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "content_analyser");

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    private static readonly (string ModelName, float ScoreThreshold)[] Models =
    {
        ("nsfw_1", 0.2f), ("nsfw_2", 0.25f), ("nsfw_3", 10.5f),
    };

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

    private static NpyArray LoadNpy(params string[] pathParts) =>
        NpyReader.Load(Path.Combine(new[] { FixturesDirectory }.Concat(pathParts).ToArray()));

    private static double ReadJsonDouble(params string[] pathParts)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(new[] { FixturesDirectory }.Concat(pathParts).ToArray())));
        return document.RootElement.GetDouble();
    }

    private static bool ReadJsonBool(params string[] pathParts)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(new[] { FixturesDirectory }.Concat(pathParts).ToArray())));
        return document.RootElement.GetBoolean();
    }

    /// <summary>
    /// <see cref="TensorComparison.Compare"/> works in <see cref="double"/> (matching
    /// <see cref="NpyArray.AsDoubles"/>'s ground-truth side); the C# port's tensors are
    /// <see cref="float"/> throughout (matching ONNX Runtime's own <c>float32</c> I/O), so this
    /// widens for the comparison call only — the widening itself introduces no precision loss.
    /// </summary>
    private static double[] ToDoubles(float[] values)
    {
        var result = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            result[i] = values[i];
        }

        return result;
    }

    // -----------------------------------------------------------------
    // Tier 1: preprocessing tensor (no model required)
    // -----------------------------------------------------------------

    public static IEnumerable<object[]> ModelNames()
    {
        foreach (var (modelName, _) in Models)
        {
            yield return new object[] { modelName };
        }
    }

    [Theory]
    [MemberData(nameof(ModelNames))]
    public void PrepareDetectFrameMatchesPython(string modelName)
    {
        if (!SourceImageAvailable)
        {
            return; // see SkippableFactAttribute below for [Fact]s; Theory needs this manual guard.
        }

        var expected = LoadNpy(modelName, "input.npy");
        Assert.Equal("float32", expected.DType);

        var (size, mean, standardDeviation) = ContentAnalyser.GetModelPreprocessingOptions(modelName);

        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var actual = ContentAnalyser.PrepareDetectFrame(sourceFrame, size, mean, standardDeviation);

        var result = TensorComparison.Compare(ToDoubles(actual), expected.AsDoubles(), relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // Tier 2: end-to-end (model-gated)
    // -----------------------------------------------------------------

    [ContentAnalyserModelFact("nsfw_1.onnx")]
    public void Nsfw1RawOutputAndScoreMatchPython()
    {
        using var analyser = new DisposableContentAnalyser();
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var modelsDirectory = Path.GetDirectoryName(FindModelPath("nsfw_1.onnx"))!;

        var (data, shape) = analyser.Instance.ForwardNsfw(sourceFrame, "nsfw_1", modelsDirectory, new[] { 0 }, new[] { ExecutionProvider.Cpu });

        var expectedRaw = LoadNpy("nsfw_1", "raw_output.npy");
        var result = TensorComparison.Compare(ToDoubles(data), expectedRaw.AsDoubles());
        Assert.True(result.Passed, result.Describe());

        var expectedShape = expectedRaw.Shape;
        Assert.Equal(expectedShape.Count, shape.Length);
        for (var i = 0; i < expectedShape.Count; i++)
        {
            Assert.Equal(expectedShape[i], (int)shape[i]);
        }

        var score = ContentAnalyser.ComputeNsfw1Score(data, shape);
        var expectedScore = ReadJsonDouble("nsfw_1", "score.json");
        Assert.Equal(expectedScore, score, 4);

        var expectedIsNsfw = ReadJsonBool("nsfw_1", "is_nsfw.json");
        Assert.Equal(expectedIsNsfw, score > 0.2f);
    }

    [ContentAnalyserModelFact("nsfw_2.onnx")]
    public void Nsfw2RawOutputAndScoreMatchPython()
    {
        using var analyser = new DisposableContentAnalyser();
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var modelsDirectory = Path.GetDirectoryName(FindModelPath("nsfw_2.onnx"))!;

        var (data, _) = analyser.Instance.ForwardNsfw(sourceFrame, "nsfw_2", modelsDirectory, new[] { 0 }, new[] { ExecutionProvider.Cpu });

        var expectedRaw = LoadNpy("nsfw_2", "raw_output.npy");
        var result = TensorComparison.Compare(ToDoubles(data), expectedRaw.AsDoubles());
        Assert.True(result.Passed, result.Describe());

        var score = ContentAnalyser.ComputeNsfw2Score(data);
        var expectedScore = ReadJsonDouble("nsfw_2", "score.json");
        Assert.Equal(expectedScore, score, 4);

        var expectedIsNsfw = ReadJsonBool("nsfw_2", "is_nsfw.json");
        Assert.Equal(expectedIsNsfw, score > 0.25f);
    }

    [ContentAnalyserModelFact("nsfw_3.onnx")]
    public void Nsfw3RawOutputAndScoreMatchPython()
    {
        using var analyser = new DisposableContentAnalyser();
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var modelsDirectory = Path.GetDirectoryName(FindModelPath("nsfw_3.onnx"))!;

        var (data, _) = analyser.Instance.ForwardNsfw(sourceFrame, "nsfw_3", modelsDirectory, new[] { 0 }, new[] { ExecutionProvider.Cpu });

        var expectedRaw = LoadNpy("nsfw_3", "raw_output.npy");
        var result = TensorComparison.Compare(ToDoubles(data), expectedRaw.AsDoubles());
        Assert.True(result.Passed, result.Describe());

        var score = ContentAnalyser.ComputeNsfw3Score(data);
        var expectedScore = ReadJsonDouble("nsfw_3", "score.json");
        Assert.Equal(expectedScore, score, 4);

        var expectedIsNsfw = ReadJsonBool("nsfw_3", "is_nsfw.json");
        Assert.Equal(expectedIsNsfw, score > 10.5f);
    }

    [ContentAnalyserModelFact("nsfw_1.onnx", "nsfw_2.onnx", "nsfw_3.onnx")]
    public void DetectNsfwAndAnalyseFrameMatchPythonOnSourceImage()
    {
        var analyser = new ContentAnalyser();
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var modelsDirectory = Path.GetDirectoryName(FindModelPath("nsfw_1.onnx"))!;

        var isNsfw = analyser.AnalyseFrame(sourceFrame, modelsDirectory, new[] { 0 }, new[] { ExecutionProvider.Cpu });

        var expectedDetectNsfw = ReadJsonBool("overall", "detect_nsfw.json");
        var expectedAnalyseFrame = ReadJsonBool("overall", "analyse_frame.json");

        Assert.Equal(expectedAnalyseFrame, isNsfw);
        Assert.Equal(expectedDetectNsfw, isNsfw); // Python: analyse_frame == detect_nsfw today.
    }

    /// <summary>Disposes the pooled <see cref="Microsoft.ML.OnnxRuntime.InferenceSession"/>s a test creates.</summary>
    private sealed class DisposableContentAnalyser : IDisposable
    {
        public ContentAnalyser Instance { get; } = new();

        private readonly List<int> _executionDeviceIds = new() { 0 };
        private readonly List<ExecutionProvider> _executionProviders = new() { ExecutionProvider.Cpu };

        public void Dispose() => Instance.ClearInferencePool(_executionDeviceIds, _executionProviders);
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c>
/// file(s) are not present — same reasoning as
/// <c>FaceFusion.ParityTests.FaceAnalysisParityTests.ModelFactAttribute</c>, given a distinct name here
/// (rather than shared) because attribute constructors run at discovery time against a
/// specific class's static helpers, and this file intentionally does not depend on
/// <c>FaceAnalysisParityTests</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class ContentAnalyserModelFactAttribute : FactAttribute
{
    public ContentAnalyserModelFactAttribute(params string[] modelFileNames)
    {
        if (!ContentAnalyserParityTests.SourceImageAvailable)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.jpg) — run tools/parity/fetch_examples.sh, then retry";
            return;
        }

        foreach (var modelFileName in modelFileNames)
        {
            if (!ContentAnalyserParityTests.ModelAvailable(modelFileName))
            {
                Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                       "run `FACEFUSION_PARITY_DIR=... python3 tools/parity/dump_content_analyser.py` once with " +
                       "network access to populate .assets/models via pre_check(), then retry";
                return;
            }
        }
    }
}
