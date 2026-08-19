using System.Runtime.CompilerServices;
using FaceFusion.Core;
using FaceFusion.Inference;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/content_analyser.py</c> — FaceFusion's NSFW gate. Three independent
/// classifiers (<c>nsfw_1</c>/<c>nsfw_2</c>/<c>nsfw_3</c>) each vote yes/no; the frame is
/// flagged NSFW when at least two of the three agree (<see cref="DetectNsfw"/>).
///
/// <para>
/// <b>This is deliberately not weakened, stubbed, bypassed, or made optional.</b> Every model,
/// every threshold, and every combination rule below is copied from the Python source with no
/// shortcuts. There is no flag anywhere in this class (or its tests) that disables the check.
/// </para>
///
/// <para>
/// <b>Deviation 1 — instance class, not a module global.</b> Python keeps
/// <c>STREAM_COUNTER</c> as a bare module global mutated by <see cref="AnalyseStream"/>'s
/// Python equivalent, and relies on <c>@lru_cache()</c> for <c>analyse_image</c>/
/// <c>analyse_video</c>. Per PORT_CONVENTIONS.md rule 5, this port is an instance class: the
/// stream counter is a private field guarded by a lock, and the two caches are private
/// <see cref="Dictionary{TKey,TValue}"/> fields guarded by the same lock — consistent with how
/// <see cref="InferenceManager"/> and <see cref="ProcessManager"/> were ported. Callers that
/// want Python's process-global sharing/caching behaviour should hold and share one
/// <see cref="ContentAnalyser"/> instance (e.g. as a DI singleton). Unlike Python's
/// <c>functools.lru_cache</c> (process-lifetime, unbounded, shared by every caller in the
/// process), these caches live and die with the owning instance.
/// </para>
///
/// <para>
/// <b>Deviation 2 — no network download.</b> <c>facefusion/download.py</c>
/// (<c>conditional_download_hashes</c>/<c>conditional_download_sources</c>) is a separate
/// module that has not been ported in this phase (out of this assignment's scope — see the
/// port instructions). <see cref="PreCheck"/> therefore cannot fetch missing model files the
/// way Python's <c>pre_check()</c> does; it instead verifies that the <c>.hash</c>/<c>.onnx</c>
/// pairs for all three models are already present under the supplied models directory and
/// that each <c>.onnx</c> file's CRC32 matches its <c>.hash</c> sidecar (via
/// <see cref="HashHelper.ValidateHash"/>), returning <see langword="false"/> — not throwing,
/// not silently proceeding — when a model is missing or fails that check. A later phase that
/// ports <c>download.py</c> can extend this to actually fetch; until then, provisioning the
/// <c>.assets/models</c> directory (e.g. by running the real Python <c>pre_check()</c> once,
/// as the port instructions describe) is a precondition for this class's inference paths to
/// work at all.
/// </para>
///
/// <para>
/// <b>Deviation 3 — no <c>video_manager</c>/tqdm.</b> <c>facefusion/video_manager.py</c> (the
/// persistent ffmpeg-pipe reader pool) is not ported either (also out of scope; see
/// <c>FaceFusion.Vision.ReadVideoFrame</c>'s own doc comment for the same, already
/// established, divergence). <see cref="AnalyseVideo"/> therefore reads each frame in the trim
/// range via <c>VisionHelper.ReadVideoFrame(videoPath, frameNumber)</c> (a fresh
/// <see cref="OpenCvSharp.VideoCapture"/> seek per frame) instead of driving a persistent
/// reader process — slower, not less correct: the counting/threshold algorithm below
/// (<c>rate &gt; 10.0</c>) is copied exactly. The <c>tqdm</c> progress bar is a CLI display
/// concern with no equivalent needed in a library method and is not ported; nothing it would
/// have displayed affects the returned value.
/// </para>
///
/// <para>
/// <b>The integrity check — what <see cref="VerifyIntegrity"/> covers and does not cover.</b>
/// Python's <c>facefusion/core.py:common_pre_check()</c> refuses to run at all unless
/// <c>hash_helper.create_hash(inspect.getsource(content_analyser).encode()) == '3c6ce25e'</c>
/// — the gate is tamper-evident: editing <c>content_analyser.py</c> on disk (to weaken a
/// threshold, change the vote rule, or stub a detector out) changes that hash and
/// <c>common_pre_check()</c> then refuses to proceed. C# has no <c>inspect.getsource</c> (there
/// is no runtime API that hands back "the literal source text of this compiled type"), so this
/// is not a mechanical translation; <see cref="VerifyIntegrity"/> reproduces the same *intent*
/// — detect on-disk tampering with this file and refuse to vouch for the gate when it cannot
/// verify — by the closest available mechanism:
/// <list type="bullet">
/// <item><description>At class-load time, a <c>[CallerFilePath]</c>-captured constant (see
/// <see cref="ThisSourceFilePath"/>) records the absolute path of <i>this very file</i> as it
/// existed on the machine that compiled it — the C# compiler burns this path into the IL as a
/// literal string, so it cannot be redirected by a caller.</description></item>
/// <item><description><see cref="ComputeSourceHash"/> re-reads that path at call time and
/// hashes its raw bytes with <see cref="HashHelper.CreateHash"/> — the same CRC32-as-8-hex-char
/// algorithm Python's <c>hash_helper.create_hash</c> uses, applied to the same kind of input
/// (the literal source text of the content-analyser module), so the mechanism is the closest
/// possible analogue of <c>inspect.getsource(...).encode()</c>.</description></item>
/// <item><description><see cref="VerifyIntegrity"/> compares that hash against a caller-supplied
/// expected value and returns <see langword="false"/> — never <see langword="true"/> — when the
/// file cannot be found or read. This is the fail-closed property: an environment that cannot
/// verify this file is treated as unverified, not as trusted by default.</description></item>
/// <item><description>The expected-hash constant is deliberately <b>not</b> stored inside this
/// file. Python's own constant (<c>'3c6ce25e'</c>) lives in <c>core.py</c>, not in
/// <c>content_analyser.py</c> — hashing a file that also contains its own expected hash is a
/// fixed-point problem (the constant is part of what gets hashed, so embedding it here would
/// mean every edit to the constant changes the hash the constant needs to equal). The same
/// split is preserved here: a later phase's <c>common_pre_check</c> equivalent should hold the
/// known-good expected hash (recomputed and updated whenever this file legitimately changes,
/// exactly as Python's <c>3c6ce25e</c> must be updated whenever <c>content_analyser.py</c>
/// legitimately changes) and call <c>ContentAnalyser.VerifyIntegrity(thatConstant)</c> as one
/// of its gating conditions, the same role <c>common_pre_check()</c> plays in
/// <c>core.py</c>.</description></item>
/// </list>
/// <b>What this cannot cover, stated plainly:</b>
/// <list type="bullet">
/// <item><description><b>A deployment that ships only the compiled assembly, without this
/// <c>.cs</c> source file alongside it.</b> Python's own check has no such gap — FaceFusion
/// ships and runs as source, so <c>content_analyser.py</c> is unconditionally present wherever
/// <c>common_pre_check()</c> runs. A compiled .NET publish is not obligated to include source
/// files, so a production deployment that strips them would make
/// <see cref="ComputeSourceHash"/> return <see langword="null"/> and
/// <see cref="VerifyIntegrity"/> return <see langword="false"/> unconditionally — correctly
/// fail-closed, but unable to distinguish "tampered" from "source not shipped". A packaging
/// step that guarantees this file travels with the published output (or an assembly-embedded
/// copy of its text, added in a later phase without touching this file's own project settings)
/// is required to make this check meaningful in that deployment shape.</description></item>
/// <item><description><b>Tampering with the compiled IL/DLL directly, independent of this
/// source file.</b> Hashing source text says nothing about whether the assembly actually
/// running was built from that exact text — a modified DLL with this untouched <c>.cs</c> file
/// sitting next to it on disk would still verify. Detecting that requires hashing the shipped
/// binary itself (assembly bytes or IL), which was deliberately not chosen as the primary
/// mechanism here because compiled IL is not stable across build configurations/toolchain
/// versions for identical source (Debug vs Release, differing SDK patch versions), which would
/// make a single hard-coded expected constant fail closed for entirely innocent reasons across
/// environments — the opposite of what a tamper-evident check should train developers to
/// trust. Source hashing avoids that instability at the cost of the gap above.</description></item>
/// <item><description><b>In-memory/runtime patching after the process has started</b> (e.g. via
/// reflection) is undetectable by any static hash of source or binary, in either
/// language.</description></item>
/// </list>
/// A perfectly faithful C# equivalent of <c>inspect.getsource</c> is therefore not achievable
/// in a standard compiled .NET deployment — this is a genuine, structural difference between
/// an interpreted-from-source application and a compiled one, not an oversight. What is
/// implemented here is the closest available mechanism that still fails closed, and the gaps
/// above are recorded so a later phase can decide whether to also add binary-hash verification
/// as a second, complementary layer.
/// </para>
/// </summary>
public sealed class ContentAnalyser
{
    /// <summary>
    /// Python: <c>ModelSet</c> entry for one of <c>nsfw_1</c>/<c>nsfw_2</c>/<c>nsfw_3</c>.
    /// Not public: this is an internal shape for this class only (see PORT_CONVENTIONS.md
    /// "one public type per file" — a supporting private record does not need its own file).
    /// </summary>
    private sealed record ModelOptions(
        string Vendor,
        string License,
        int Year,
        string HashPath,
        string SourcePath,
        Resolution Size,
        double[] Mean,
        double[] StandardDeviation);

    private static readonly string[] ModelNames = { "nsfw_1", "nsfw_2", "nsfw_3" };

    private readonly InferenceManager _inferenceManager;
    private readonly object _lock = new();
    private int _streamCounter;

    // Python: @lru_cache() on analyse_image/analyse_video. See Deviation 1 above: these are
    // per-instance, not process-global.
    private readonly Dictionary<string, bool> _analyseImageCache = new(StringComparer.Ordinal);
    private readonly Dictionary<(string VideoPath, int TrimFrameStart, int TrimFrameEnd), bool> _analyseVideoCache = new();

    public ContentAnalyser(InferenceManager? inferenceManager = null)
    {
        _inferenceManager = inferenceManager ?? new InferenceManager();
    }

    // -----------------------------------------------------------------
    // Model set
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c> — a pure function of
    /// <paramref name="modelsDirectory"/>, so no instance state is needed; matches
    /// PORT_CONVENTIONS.md rule 5's "pure memoization is fine" carve-out the same way
    /// <c>FaceFusion.Inference.ModelHelper</c>'s cache does). <paramref name="modelsDirectory"/>
    /// stands in for Python's <c>resolve_relative_path('../.assets/models/...')</c> — taken as
    /// an explicit parameter rather than resolved against a package-relative path, per
    /// PORT_CONVENTIONS.md rule 5 ("no global state, take it as a parameter"). The
    /// <c>download_scope</c> parameter is dropped: Python's is unused by every value it could
    /// take here (each model always has exactly one hash and one source, so 'lite' vs 'full'
    /// makes no difference for this particular model set — it exists only because
    /// <c>ModelSet</c> is a general shape reused across processors that do have per-scope
    /// variants).
    /// </summary>
    private static IReadOnlyDictionary<string, ModelOptions> CreateStaticModelSet(string modelsDirectory)
    {
        return new Dictionary<string, ModelOptions>(StringComparer.Ordinal)
        {
            ["nsfw_1"] = new ModelOptions(
                Vendor: "EraX",
                License: "Apache-2.0",
                Year: 2024,
                HashPath: Path.Combine(modelsDirectory, "nsfw_1.hash"),
                SourcePath: Path.Combine(modelsDirectory, "nsfw_1.onnx"),
                Size: new Resolution(640, 640),
                Mean: new[] { 0.0, 0.0, 0.0 },
                StandardDeviation: new[] { 1.0, 1.0, 1.0 }),
            ["nsfw_2"] = new ModelOptions(
                Vendor: "Marqo",
                License: "Apache-2.0",
                Year: 2024,
                HashPath: Path.Combine(modelsDirectory, "nsfw_2.hash"),
                SourcePath: Path.Combine(modelsDirectory, "nsfw_2.onnx"),
                Size: new Resolution(384, 384),
                Mean: new[] { 0.5, 0.5, 0.5 },
                StandardDeviation: new[] { 0.5, 0.5, 0.5 }),
            ["nsfw_3"] = new ModelOptions(
                Vendor: "Freepik",
                License: "MIT",
                Year: 2025,
                HashPath: Path.Combine(modelsDirectory, "nsfw_3.hash"),
                SourcePath: Path.Combine(modelsDirectory, "nsfw_3.onnx"),
                Size: new Resolution(448, 448),
                Mean: new[] { 0.48145466, 0.4578275, 0.40821073 },
                StandardDeviation: new[] { 0.26862954, 0.26130258, 0.27577711 }),
        };
    }

    /// <summary>
    /// The <c>size</c>/<c>mean</c>/<c>standard_deviation</c> half of Python's
    /// <c>create_static_model_set(...).get(model_name)</c> — the values
    /// <see cref="PrepareDetectFrame"/> needs — without requiring a models directory (this
    /// does not touch the filesystem). Exposed for parity tests that need to call
    /// <see cref="PrepareDetectFrame"/> directly against a specific sub-model's preprocessing
    /// parameters; see the class remarks on <see cref="ComputeNsfw1Score"/> for why this kind
    /// of accessor is public.
    /// </summary>
    public static (Resolution Size, double[] Mean, double[] StandardDeviation) GetModelPreprocessingOptions(string modelName)
    {
        // modelsDirectory is irrelevant to size/mean/standard_deviation (only the hash/source
        // paths depend on it), so an arbitrary placeholder is fine here.
        var options = CreateStaticModelSet(string.Empty)[modelName];
        return (options.Size, options.Mean, options.StandardDeviation);
    }

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    public IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        string modelsDirectory,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelSourceSet = CollectModelSources(modelsDirectory);
        return _inferenceManager.GetInferencePool("facefusion.content_analyser", ModelNames, modelSourceSet, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public void ClearInferencePool(IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        _inferenceManager.ClearInferencePool("facefusion.content_analyser", ModelNames, executionDeviceIds, executionProviders);
    }

    /// <summary>
    /// Python: <c>collect_model_downloads</c>, split into its two halves since C# has no tuple
    /// unpacking convention as terse as Python's — <see cref="CollectModelHashes"/> for the
    /// hash sidecars, this for the <c>.onnx</c> sources (the half <see cref="GetInferencePool"/>
    /// actually needs).
    /// </summary>
    private static IReadOnlyDictionary<string, Download> CollectModelSources(string modelsDirectory)
    {
        var modelSet = CreateStaticModelSet(modelsDirectory);
        var modelSourceSet = new Dictionary<string, Download>(StringComparer.Ordinal);

        foreach (var modelName in ModelNames)
        {
            var options = modelSet[modelName];
            // Url is unused: no download.py port exists in this phase (Deviation 2). Download's
            // Url field is not read anywhere on this path (InferenceManager only checks Path).
            modelSourceSet[modelName] = new Download(string.Empty, options.SourcePath);
        }

        return modelSourceSet;
    }

    private static IReadOnlyDictionary<string, string> CollectModelHashes(string modelsDirectory)
    {
        var modelSet = CreateStaticModelSet(modelsDirectory);
        var modelHashSet = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var modelName in ModelNames)
        {
            modelHashSet[modelName] = modelSet[modelName].HashPath;
        }

        return modelHashSet;
    }

    /// <summary>
    /// Python: <c>pre_check</c>. See Deviation 2 above — this cannot download; it verifies
    /// that every model's <c>.hash</c>/<c>.onnx</c> pair is already present under
    /// <paramref name="modelsDirectory"/> and that the <c>.onnx</c> file's CRC32 matches its
    /// <c>.hash</c> sidecar. Returns <see langword="false"/> (never throws) for a missing or
    /// invalid model — callers must provision <c>.assets/models</c> out of band.
    /// </summary>
    public bool PreCheck(string modelsDirectory)
    {
        var modelSet = CreateStaticModelSet(modelsDirectory);

        foreach (var modelName in ModelNames)
        {
            var options = modelSet[modelName];

            if (!FileSystem.IsFile(options.HashPath) || !FileSystem.IsFile(options.SourcePath))
            {
                return false;
            }

            if (!HashHelper.ValidateHash(options.SourcePath))
            {
                return false;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------
    // Analysis
    // -----------------------------------------------------------------

    /// <summary>Python: <c>analyse_stream</c>.</summary>
    public bool AnalyseStream(Mat visionFrame, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders, double videoFps)
    {
        int counter;

        lock (_lock)
        {
            _streamCounter++;
            counter = _streamCounter;
        }

        if (counter % (int)videoFps == 0)
        {
            return AnalyseFrame(visionFrame, modelsDirectory, executionDeviceIds, executionProviders);
        }

        return false;
    }

    /// <summary>Python: <c>analyse_frame</c>.</summary>
    public bool AnalyseFrame(Mat visionFrame, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        return DetectNsfw(visionFrame, modelsDirectory, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>analyse_image</c> (<c>@lru_cache()</c> — see Deviation 1).</summary>
    public bool AnalyseImage(string imagePath, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        lock (_lock)
        {
            if (_analyseImageCache.TryGetValue(imagePath, out var cached))
            {
                return cached;
            }
        }

        using var visionFrame = VisionHelper.ReadImage(imagePath)
            ?? throw new InvalidOperationException($"could not read image '{imagePath}'.");
        var result = AnalyseFrame(visionFrame, modelsDirectory, executionDeviceIds, executionProviders);

        lock (_lock)
        {
            _analyseImageCache[imagePath] = result;
        }

        return result;
    }

    /// <summary>
    /// Python: <c>analyse_video</c> (<c>@lru_cache()</c> — see Deviation 1). See Deviation 3
    /// above for why this reads frames one at a time via <c>Vision.ReadVideoFrame</c> rather
    /// than driving a persistent <c>video_manager</c> reader; the counting/threshold algorithm
    /// itself (<c>rate &gt; 10.0</c>, sampled once per <c>int(video_fps)</c> frames) is copied
    /// exactly. No progress bar is rendered (tqdm is a CLI display concern; see Deviation 3).
    /// </summary>
    public bool AnalyseVideo(string videoPath, int trimFrameStart, int trimFrameEnd, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var cacheKey = (videoPath, trimFrameStart, trimFrameEnd);

        lock (_lock)
        {
            if (_analyseVideoCache.TryGetValue(cacheKey, out var cached))
            {
                return cached;
            }
        }

        var videoFps = VisionHelper.DetectVideoFps(videoPath) ?? 25.0;
        var rate = 0.0;
        var total = 0;
        var counter = 0;

        for (var frameNumber = trimFrameStart; frameNumber < trimFrameEnd; frameNumber++)
        {
            using var visionFrame = VisionHelper.ReadVideoFrame(videoPath, frameNumber);

            if (frameNumber % (int)videoFps == 0 && VisionHelper.IsVisionFrame(visionFrame))
            {
                total++;

                if (AnalyseFrame(visionFrame!, modelsDirectory, executionDeviceIds, executionProviders))
                {
                    counter++;
                }
            }

            if (counter > 0 && total > 0)
            {
                rate = (double)counter / total * 100;
            }
        }

        var result = rate > 10.0;

        lock (_lock)
        {
            _analyseVideoCache[cacheKey] = result;
        }

        return result;
    }

    /// <summary>Python: <c>detect_nsfw</c>.</summary>
    private bool DetectNsfw(Mat visionFrame, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var isNsfw1 = DetectWithNsfw1(visionFrame, modelsDirectory, executionDeviceIds, executionProviders);
        var isNsfw2 = DetectWithNsfw2(visionFrame, modelsDirectory, executionDeviceIds, executionProviders);
        var isNsfw3 = DetectWithNsfw3(visionFrame, modelsDirectory, executionDeviceIds, executionProviders);

        return (isNsfw1 && isNsfw2) || (isNsfw1 && isNsfw3) || (isNsfw2 && isNsfw3);
    }

    /// <summary>Python: <c>detect_with_nsfw_1</c>.</summary>
    private bool DetectWithNsfw1(Mat visionFrame, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var (data, shape) = ForwardNsfw(visionFrame, "nsfw_1", modelsDirectory, executionDeviceIds, executionProviders);
        return ComputeNsfw1Score(data, shape) > 0.2f;
    }

    /// <summary>Python: <c>detect_with_nsfw_2</c>.</summary>
    private bool DetectWithNsfw2(Mat visionFrame, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var (data, _) = ForwardNsfw(visionFrame, "nsfw_2", modelsDirectory, executionDeviceIds, executionProviders);
        return ComputeNsfw2Score(data) > 0.25f;
    }

    /// <summary>Python: <c>detect_with_nsfw_3</c>.</summary>
    private bool DetectWithNsfw3(Mat visionFrame, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var (data, _) = ForwardNsfw(visionFrame, "nsfw_3", modelsDirectory, executionDeviceIds, executionProviders);
        return ComputeNsfw3Score(data) > 10.5f;
    }

    /// <summary>
    /// Python: the <c>numpy.max(numpy.amax(detection[:, 4:], axis = 1))</c> half of
    /// <c>detect_with_nsfw_1</c> — the max, across every one of the 8400 anchors, of that
    /// anchor's own max across the 5 class-score channels (indices 4..8; the first 4 are box
    /// coordinates). Batch size is always 1, so <paramref name="rawOutput"/>'s flat buffer is
    /// already channel-major (<c>channel * anchorTotal + anchor</c>) with no batch-dim offset
    /// to account for. Exposed (public, alongside <see cref="ComputeNsfw2Score"/>/
    /// <see cref="ComputeNsfw3Score"/> and <see cref="PrepareDetectFrame"/>) for parity tests
    /// that need to assert the exact score computed from a real model's raw output — same
    /// reasoning as <c>FaceRecognizer.PrepareInput</c>/<c>FaceClassifier.PrepareInput</c>.
    /// </summary>
    public static float ComputeNsfw1Score(float[] rawOutput, long[] shape)
    {
        var channelTotal = (int)shape[1];
        var anchorTotal = (int)shape[2];
        var detectionScore = float.NegativeInfinity;

        for (var channel = 4; channel < channelTotal; channel++)
        {
            var channelOffset = channel * anchorTotal;

            for (var anchor = 0; anchor < anchorTotal; anchor++)
            {
                var value = rawOutput[channelOffset + anchor];

                if (value > detectionScore)
                {
                    detectionScore = value;
                }
            }
        }

        return detectionScore;
    }

    /// <summary>
    /// Python: <c>detection[0] - detection[1]</c> on the batch-squeezed (2,) output — batch
    /// size is always 1, so indices 0/1 of <paramref name="rawOutput"/>'s flat buffer are
    /// exactly <c>detection[0]</c>/<c>detection[1]</c>. See <see cref="ComputeNsfw1Score"/>.
    /// </summary>
    public static float ComputeNsfw2Score(float[] rawOutput) => rawOutput[0] - rawOutput[1];

    /// <summary>
    /// Python: <c>(detection[2] + detection[3]) - (detection[0] + detection[1])</c> on the
    /// batch-squeezed (4,) output. See <see cref="ComputeNsfw1Score"/>.
    /// </summary>
    public static float ComputeNsfw3Score(float[] rawOutput) => (rawOutput[2] + rawOutput[3]) - (rawOutput[0] + rawOutput[1]);

    /// <summary>
    /// Python: <c>forward_nsfw</c>. Runs the named model via the ORT <c>OrtValue</c> zero-copy
    /// calling convention established by <see cref="InferenceManager"/> (DOTNET_PORT_PLAN.md
    /// §5.3) and returns the raw output tensor's flat data plus its shape. The Python
    /// <c>detection[0]</c> squeeze for <c>nsfw_2</c>/<c>nsfw_3</c> is not performed here — batch
    /// size is always 1, so it is a no-op on the flat buffer; callers index the flat array
    /// directly (see <see cref="ComputeNsfw2Score"/>/<see cref="ComputeNsfw3Score"/>). Public
    /// for the same parity-test reason as <see cref="ComputeNsfw1Score"/>.
    /// </summary>
    public (float[] Data, long[] Shape) ForwardNsfw(Mat visionFrame, string modelName, string modelsDirectory, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelSet = CreateStaticModelSet(modelsDirectory);
        var options = modelSet[modelName];
        var inputData = PrepareDetectFrame(visionFrame, options.Size, options.Mean, options.StandardDeviation);
        var inputShape = new long[] { 1, 3, options.Size.Height, options.Size.Width };

        var inferencePool = GetInferencePool(modelsDirectory, executionDeviceIds, executionProviders);
        var inferenceSession = inferencePool[modelName];

        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, inputShape);
        var inputs = new Dictionary<string, OrtValue> { [inferenceSession.InputNames[0]] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = inferenceSession.Run(runOptions, inputs, inferenceSession.OutputNames);

        var output = results[0];
        var shape = output.GetTensorTypeAndShape().Shape;
        var data = output.GetTensorDataAsSpan<float>().ToArray();
        return (data, shape);
    }

    /// <summary>
    /// Python: <c>prepare_detect_frame</c>. Returns the flat NCHW <c>float32</c> tensor data
    /// (row-major, matching <c>numpy.expand_dims(..., axis=0)</c>'s layout exactly).
    ///
    /// <para>
    /// Computed per-pixel in <see cref="double"/> precision, matching Python's evaluation order
    /// exactly (<c>frame[:, :, ::-1] / 255.0</c>, then <c>-= mean</c>, then <c>/= std</c>, all
    /// of which numpy promotes to float64 since <c>model_mean</c>/<c>model_standard_deviation</c>
    /// are Python <c>float</c> tuples; only the very last step, <c>.astype(numpy.float32)</c>,
    /// narrows to float32) — not via a single-precision <see cref="Mat.ConvertTo"/>/OpenCV
    /// arithmetic chain, which would accumulate rounding differently. A manual per-pixel loop
    /// is used rather than <c>Cv2.Split</c>/elementwise <see cref="Mat"/> ops for the same
    /// reason <see cref="FaceHelper.PasteBack"/> uses one: exact floating-point-order parity
    /// with numpy takes priority here over the DOTNET_PORT_PLAN.md §5b preference for
    /// vectorised preprocessing, and this is not a per-frame hot path (the NSFW gate runs once
    /// per sampled frame, not once per face per frame).
    /// </para>
    /// </summary>
    public static float[] PrepareDetectFrame(Mat tempVisionFrame, Resolution modelSize, double[] modelMean, double[] modelStandardDeviation)
    {
        using var detectVisionFrame = VisionHelper.FitContainFrame(tempVisionFrame, modelSize);

        var height = modelSize.Height;
        var width = modelSize.Width;
        var planeSize = height * width;
        var data = new float[3 * planeSize];

        for (var row = 0; row < height; row++)
        {
            for (var col = 0; col < width; col++)
            {
                var pixel = detectVisionFrame.At<Vec3b>(row, col);
                var planeIndex = row * width + col;

                // BGR -> RGB (Python: `frame[:, :, ::-1]`), then per-channel normalise.
                // channel 0 = R (pixel.Item2), 1 = G (pixel.Item1), 2 = B (pixel.Item0).
                data[planeIndex] = (float)((pixel.Item2 / 255.0 - modelMean[0]) / modelStandardDeviation[0]);
                data[planeSize + planeIndex] = (float)((pixel.Item1 / 255.0 - modelMean[1]) / modelStandardDeviation[1]);
                data[(2 * planeSize) + planeIndex] = (float)((pixel.Item0 / 255.0 - modelMean[2]) / modelStandardDeviation[2]);
            }
        }

        return data;
    }

    // -----------------------------------------------------------------
    // Integrity check (tamper-evidence) — see the class-level remarks above.
    // -----------------------------------------------------------------

    /// <summary>
    /// The absolute path of this source file as it existed on the machine that compiled this
    /// assembly, captured by the C# compiler at compile time via <c>[CallerFilePath]</c> on
    /// the private call below (the call site is fixed, inside this very file, so this always
    /// names <c>ContentAnalyser.cs</c> regardless of who later calls
    /// <see cref="VerifyIntegrity"/> or from where — unlike putting <c>[CallerFilePath]</c>
    /// directly on a public method's parameter, which would capture the *caller's* file
    /// instead).
    /// </summary>
    private static readonly string ThisSourceFilePath = CaptureThisFilePath();

    private static string CaptureThisFilePath([CallerFilePath] string path = "") => path;

    /// <summary>
    /// Hashes the literal source text of this file with the same CRC32-as-8-hex-char algorithm
    /// Python's <c>hash_helper.create_hash</c> uses. Returns <see langword="null"/> — never
    /// throws — when the source file cannot be found or read, so callers have an explicit
    /// "could not verify" signal distinct from any real hash value.
    /// </summary>
    /// <summary>
    /// Collapses CRLF to LF before hashing, so the same source text hashes identically on every
    /// platform.
    ///
    /// <para>
    /// <b>Why this is necessary, and why it does not weaken the gate.</b> Git checks this file
    /// out with CRLF on Windows by default, so hashing the raw bytes produced a different value
    /// there — a different value against the pinned one — and the gate tripped on
    /// every run. Because it fails closed, the effect was that <c>common_pre_check</c> refused
    /// to start *any* processing on Windows at all. Line endings are a representation of the
    /// same source text, not part of it: an edit that weakens a threshold, changes the vote rule
    /// or stubs out a detector still changes this hash, which is the property the gate exists
    /// for. Only the choice of newline is neutralised.
    /// </para>
    ///
    /// <para>
    /// A lone CR (old-Mac line endings) is deliberately left alone — it is not a line ending any
    /// checkout produces today, and rewriting it would widen what the hash treats as equivalent
    /// beyond the one case that actually occurs.
    /// </para>
    /// </summary>
    internal static byte[] NormalizeLineEndings(byte[] bytes)
    {
        const byte carriageReturn = 0x0D;
        const byte lineFeed = 0x0A;

        var normalized = new byte[bytes.Length];
        var length = 0;

        for (var index = 0; index < bytes.Length; index++)
        {
            if (bytes[index] == carriageReturn && index + 1 < bytes.Length && bytes[index + 1] == lineFeed)
            {
                continue;
            }

            normalized[length++] = bytes[index];
        }

        return length == bytes.Length ? normalized : normalized[..length];
    }

    public static string? ComputeSourceHash()
    {
        try
        {
            if (string.IsNullOrEmpty(ThisSourceFilePath) || !File.Exists(ThisSourceFilePath))
            {
                return null;
            }

            var bytes = File.ReadAllBytes(ThisSourceFilePath);
            return HashHelper.CreateHash(NormalizeLineEndings(bytes));
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// The C# equivalent of Python's
    /// <c>hash_helper.create_hash(inspect.getsource(content_analyser).encode()) == '3c6ce25e'</c>
    /// gate in <c>facefusion/core.py:common_pre_check()</c>. Returns <see langword="true"/> only
    /// when this file's current on-disk source hashes to exactly
    /// <paramref name="expectedHash"/>; returns <see langword="false"/> (fails closed) for a
    /// mismatch or when the source could not be read at all. See the class-level remarks for
    /// why <paramref name="expectedHash"/> is a caller-supplied parameter rather than a
    /// constant baked into this file, and for what this check does and does not cover.
    ///
    /// <para>
    /// <b>Wiring for a later phase:</b> a ported <c>common_pre_check</c> equivalent (this
    /// phase does not include <c>core.py</c>) should hold the known-good expected hash as its
    /// own constant — recomputed via <see cref="ComputeSourceHash"/> and updated whenever this
    /// file legitimately changes, exactly as Python's <c>3c6ce25e</c> must be updated whenever
    /// <c>content_analyser.py</c> legitimately changes — and require
    /// <c>ContentAnalyser.VerifyIntegrity(thatConstant)</c> to return <see langword="true"/>
    /// before allowing the rest of the pipeline to run, the same gating role
    /// <c>common_pre_check()</c> plays today.
    /// </para>
    /// </summary>
    public static bool VerifyIntegrity(string expectedHash)
    {
        var actualHash = ComputeSourceHash();
        return actualHash is not null && string.Equals(actualHash, expectedHash, StringComparison.Ordinal);
    }
}
