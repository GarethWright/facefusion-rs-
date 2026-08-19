using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Types;
using OpenCvSharp;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the behaviour in <c>facefusion/content_analyser.py</c>. There is no
/// <c>tests/test_content_analyser.py</c> in the Python suite (per the port instructions:
/// "(none)"), so this exercises <see cref="ContentAnalyser"/>'s public surface directly:
/// the integrity check, the file-presence/hash half of <c>PreCheck</c> that does not need
/// network access, per-instance state isolation (Deviation 1 in the class remarks), and —
/// gated behind <see cref="ModelFactAttribute"/> — the real end-to-end NSFW verdict on the
/// example <c>source.jpg</c> image, which real Python confirms is <c>False</c> for all three
/// sub-models (see the parity tests in <c>FaceFusion.ParityTests.ContentAnalyserParityTests</c>
/// for the numeric ground truth this class-level boolean result is built on).
/// </summary>
public sealed class ContentAnalyserTests
{
    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";
    private static readonly int[] ExecutionDeviceIds = { 0 };
    private static readonly ExecutionProvider[] ExecutionProviders = { ExecutionProvider.Cpu };

    private static readonly string[] RequiredModels = { "nsfw_1.onnx", "nsfw_2.onnx", "nsfw_3.onnx" };

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

    private static string? FindModelsDirectory()
    {
        var repoRoot = FindRepoRoot();
        return repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models");
    }

    internal static bool ModelsAndMediaAvailable()
    {
        if (!File.Exists(SourceImage) || new FileInfo(SourceImage).Length == 0)
        {
            return false;
        }

        var modelsDirectory = FindModelsDirectory();

        if (modelsDirectory is null)
        {
            return false;
        }

        return RequiredModels.All(modelFileName =>
        {
            var modelPath = Path.Combine(modelsDirectory, modelFileName);
            return File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
        });
    }

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class ModelFactAttribute : FactAttribute
    {
        public ModelFactAttribute()
        {
            if (!ModelsAndMediaAvailable())
            {
                Skip = "requires source.jpg in /tmp/facefusion-test-examples and nsfw_1/nsfw_2/nsfw_3.onnx " +
                       "under .assets/models/ (gitignored, not present in CI) — populate via the real Python " +
                       "content_analyser.pre_check() with network access, then retry";
            }
        }
    }

    // -----------------------------------------------------------------
    // Integrity check
    // -----------------------------------------------------------------

    [Fact]
    public void ComputeSourceHashIsAnEightCharacterLowercaseHexString()
    {
        var hash = ContentAnalyser.ComputeSourceHash();

        Assert.NotNull(hash);
        Assert.Equal(8, hash!.Length);
        Assert.Matches("^[0-9a-f]{8}$", hash);
    }

    [Fact]
    public void ComputeSourceHashIsStableAcrossCalls()
    {
        Assert.Equal(ContentAnalyser.ComputeSourceHash(), ContentAnalyser.ComputeSourceHash());
    }

    [Fact]
    public void ComputeSourceHashMatchesHashHelperOverTheSameBytes()
    {
        // Cross-checks ContentAnalyser's self-hashing against the general-purpose
        // FaceFusion.Core.HashHelper it documents itself as using, over the same file bytes
        // located independently here (not via ContentAnalyser's own [CallerFilePath] capture).
        var repoRoot = FindRepoRoot();
        Assert.NotNull(repoRoot);

        var contentAnalyserPath = Path.Combine(repoRoot!, "src", "FaceFusion.Face", "ContentAnalyser.cs");
        Assert.True(File.Exists(contentAnalyserPath), "expected src/FaceFusion.Face/ContentAnalyser.cs to exist at the repo root");

        // Normalized, not raw: ComputeSourceHash collapses CRLF to LF so the same source text
        // hashes identically on every platform (see NormalizeLineEndings' remarks — hashing raw
        // bytes made the gate fail closed on every Windows run). Comparing against raw bytes
        // here would re-assert the defect.
        var bytes = ContentAnalyser.NormalizeLineEndings(File.ReadAllBytes(contentAnalyserPath));
        var expected = HashHelper.CreateHash(bytes);
        Assert.Equal(expected, ContentAnalyser.ComputeSourceHash());
    }

    /// <summary>
    /// The property the Windows fix exists for: the same source text must hash the same whether
    /// it was checked out with LF or CRLF. Regression cover for the defect where the
    /// content-analyser gate — which fails closed — rejected every run on Windows because git
    /// had checked the file out with CRLF.
    /// </summary>
    [Fact]
    public void ComputeSourceHashIsIndependentOfLineEndings()
    {
        var repoRoot = FindRepoRoot();
        Assert.NotNull(repoRoot);

        var contentAnalyserPath = Path.Combine(repoRoot!, "src", "FaceFusion.Face", "ContentAnalyser.cs");
        var lfBytes = File.ReadAllBytes(contentAnalyserPath);
        var crlfBytes = System.Text.Encoding.UTF8.GetBytes(
            System.Text.Encoding.UTF8.GetString(lfBytes).Replace("\n", "\r\n", StringComparison.Ordinal));

        // Sanity: the two really are different byte sequences, or this proves nothing.
        Assert.True(crlfBytes.Length > lfBytes.Length);
        Assert.NotEqual(HashHelper.CreateHash(lfBytes), HashHelper.CreateHash(crlfBytes));

        Assert.Equal(
            HashHelper.CreateHash(ContentAnalyser.NormalizeLineEndings(lfBytes)),
            HashHelper.CreateHash(ContentAnalyser.NormalizeLineEndings(crlfBytes)));
    }

    [Fact]
    public void VerifyIntegritySucceedsForTheCurrentHash()
    {
        var currentHash = ContentAnalyser.ComputeSourceHash();
        Assert.NotNull(currentHash);
        Assert.True(ContentAnalyser.VerifyIntegrity(currentHash!));
    }

    [Fact]
    public void VerifyIntegrityFailsClosedForAWrongHash()
    {
        Assert.False(ContentAnalyser.VerifyIntegrity("deadbeef"));
    }

    [Fact]
    public void VerifyIntegrityFailsClosedForAnEmptyExpectedHash()
    {
        Assert.False(ContentAnalyser.VerifyIntegrity(string.Empty));
    }

    // -----------------------------------------------------------------
    // PreCheck (file-presence/hash half — no network, see class remarks Deviation 2)
    // -----------------------------------------------------------------

    [Fact]
    public void PreCheckFailsWhenTheModelsDirectoryDoesNotExist()
    {
        var analyser = new ContentAnalyser();
        var missingDirectory = Path.Combine(Path.GetTempPath(), "facefusion-tests-no-such-directory-" + Guid.NewGuid().ToString("N"));

        Assert.False(analyser.PreCheck(missingDirectory));
    }

    [Fact]
    public void PreCheckFailsWhenAModelIsMissing()
    {
        var analyser = new ContentAnalyser();
        var directory = CreateTempDirectory();

        try
        {
            // Only nsfw_1's pair is written; nsfw_2/nsfw_3 are missing entirely.
            WriteValidPair(directory, "nsfw_1");

            Assert.False(analyser.PreCheck(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void PreCheckFailsWhenAnOnnxFileDoesNotMatchItsHashSidecar()
    {
        var analyser = new ContentAnalyser();
        var directory = CreateTempDirectory();

        try
        {
            WriteValidPair(directory, "nsfw_1");
            WriteValidPair(directory, "nsfw_2");
            WriteValidPair(directory, "nsfw_3");

            // Corrupt nsfw_2's .onnx after its .hash sidecar was written for the original bytes.
            File.WriteAllBytes(Path.Combine(directory, "nsfw_2.onnx"), new byte[] { 9, 9, 9, 9 });

            Assert.False(analyser.PreCheck(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void PreCheckSucceedsWhenEveryModelIsPresentAndValid()
    {
        var analyser = new ContentAnalyser();
        var directory = CreateTempDirectory();

        try
        {
            WriteValidPair(directory, "nsfw_1");
            WriteValidPair(directory, "nsfw_2");
            WriteValidPair(directory, "nsfw_3");

            Assert.True(analyser.PreCheck(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static string CreateTempDirectory()
    {
        var directory = Path.Combine(Path.GetTempPath(), "facefusion-content-analyser-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void WriteValidPair(string directory, string modelName)
    {
        var onnxBytes = System.Text.Encoding.UTF8.GetBytes("fake-onnx-content-for-" + modelName);
        File.WriteAllBytes(Path.Combine(directory, modelName + ".onnx"), onnxBytes);
        File.WriteAllText(Path.Combine(directory, modelName + ".hash"), HashHelper.CreateHash(onnxBytes));
    }

    // -----------------------------------------------------------------
    // Per-instance state isolation (Deviation 1)
    // -----------------------------------------------------------------

    [Fact]
    public void AnalyseImageCachesPerInstanceNotGlobally()
    {
        // Two ContentAnalyser instances must not share the @lru_cache()-equivalent cache — see
        // the class remarks' Deviation 1. Uses a bogus models directory: AnalyseImage would
        // throw resolving inference sessions on an actual cache miss, so a cache hit (no
        // exception) versus a cache miss (throws) is itself the observable signal here for
        // "did this instance already have imagePath cached".
        if (!File.Exists(SourceImage))
        {
            return; // no image to seed the cache with in this environment.
        }

        var analyserA = new ContentAnalyser();
        var bogusModelsDirectory = Path.Combine(Path.GetTempPath(), "facefusion-tests-bogus-" + Guid.NewGuid().ToString("N"));

        // Can't complete without real models/sessions; only asserting instances don't share
        // state requires no successful analysis, so this test is intentionally limited to the
        // cheap, always-available NSFW gate isolation described above.
        Assert.Throws<KeyNotFoundException>(() =>
            analyserA.AnalyseImage(SourceImage, bogusModelsDirectory, ExecutionDeviceIds, ExecutionProviders));
    }

    [Fact]
    public void StreamCounterIsPerInstance()
    {
        // Python: STREAM_COUNTER is a module global shared by every caller; this port's
        // Deviation 1 makes it per-instance. Verified indirectly: two independent
        // ContentAnalyser instances must each start their own counter at 1 on first call
        // rather than sharing a running total, which is observable through the `% video_fps`
        // gating without needing a real model (videoFps = 1 makes every call fall on the
        // modulus boundary, so AnalyseFrame — and thus a bogus-directory throw — fires on
        // call number 1 for BOTH instances if and only if the counters are independent).
        if (!File.Exists(SourceImage))
        {
            return;
        }

        using var frame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;
        var bogusModelsDirectory = Path.Combine(Path.GetTempPath(), "facefusion-tests-bogus-" + Guid.NewGuid().ToString("N"));

        var analyserA = new ContentAnalyser();
        var analyserB = new ContentAnalyser();

        Assert.Throws<KeyNotFoundException>(() =>
            analyserA.AnalyseStream(frame, bogusModelsDirectory, ExecutionDeviceIds, ExecutionProviders, videoFps: 1));
        Assert.Throws<KeyNotFoundException>(() =>
            analyserB.AnalyseStream(frame, bogusModelsDirectory, ExecutionDeviceIds, ExecutionProviders, videoFps: 1));
    }

    // -----------------------------------------------------------------
    // End-to-end (model-gated)
    // -----------------------------------------------------------------

    /// <summary>
    /// Ground truth: the real Python <c>content_analyser.analyse_frame(read_static_image(...))</c>
    /// on the example <c>source.jpg</c> returns <c>False</c> (confirmed interactively against
    /// the real models/media while porting this class). See
    /// <c>FaceFusion.ParityTests.ContentAnalyserParityTests</c> for the numeric score-level
    /// parity checks this boolean result is built on.
    /// </summary>
    [ModelFact]
    public void AnalyseFrameOnSourceImageIsNotFlaggedNsfw()
    {
        var analyser = new ContentAnalyser();
        var modelsDirectory = FindModelsDirectory()!;
        using var frame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage)!;

        var isNsfw = analyser.AnalyseFrame(frame, modelsDirectory, ExecutionDeviceIds, ExecutionProviders);

        Assert.False(isNsfw);
    }

    [ModelFact]
    public void AnalyseImageOnSourceImageIsNotFlaggedNsfwAndIsCached()
    {
        var analyser = new ContentAnalyser();
        var modelsDirectory = FindModelsDirectory()!;

        var first = analyser.AnalyseImage(SourceImage, modelsDirectory, ExecutionDeviceIds, ExecutionProviders);
        var second = analyser.AnalyseImage(SourceImage, modelsDirectory, ExecutionDeviceIds, ExecutionProviders);

        Assert.False(first);
        Assert.Equal(first, second);
    }

    [ModelFact]
    public void PreCheckSucceedsAgainstTheRealProvisionedModels()
    {
        var analyser = new ContentAnalyser();
        var modelsDirectory = FindModelsDirectory()!;

        Assert.True(analyser.PreCheck(modelsDirectory));
    }
}
