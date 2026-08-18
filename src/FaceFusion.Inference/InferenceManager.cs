using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using FaceFusion.Core;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;

namespace FaceFusion.Inference;

/// <summary>
/// Port of <c>facefusion/inference_manager.py</c> — a pool of <see cref="InferenceSession"/>
/// keyed by <c>module_name + model_names + device_id + providers</c> (see
/// <see cref="GetInferenceContext"/>).
///
/// <para>
/// <b>Deviation 1 — no <c>cli</c>/<c>ui</c> app-context split.</b> Python keeps
/// <c>INFERENCE_POOL_SET : Dict['cli' | 'ui', Dict[context, InferencePool]]</c>, two separate
/// top-level dicts reached via <c>detect_app_context()</c>. Per DOTNET_PORT_PLAN.md §3, that
/// split exists purely to work around Gradio's threading model and disappears in this port —
/// there is a single pool, keyed directly by inference context.
/// </para>
///
/// <para>
/// <b>Deviation 2 — the CUDA arena-leak workaround is dropped, and is correctly dropped as a
/// consequence of deviation 1, not silently.</b> Python's
/// <c>has_arena_leak = has_execution_provider('cuda') and get_onnxruntime_version() &gt; (1, 24, 4)</c>
/// gates a block that, when <c>False</c> (no known leak), copies the session dict *by
/// reference* from <c>INFERENCE_POOL_SET['ui'][context]</c> into
/// <c>INFERENCE_POOL_SET['cli'][context]</c> (and back), so both app contexts reuse the same
/// live sessions instead of loading the model twice — except on ORT versions newer than
/// 1.24.4 with CUDA active, where a real arena double-free bug in the CUDA EP makes sharing a
/// session across the two contexts unsafe, so the copy is skipped and each context gets its
/// own sessions. With the two top-level dicts collapsed into one (deviation 1), there is only
/// ever one dict entry per inference context to begin with — "copy the reference between the
/// two dicts" has no second dict to copy into, so the sharing this workaround enabled happens
/// unconditionally and for free, and the double-free hazard it guarded against cannot arise
/// either, since there is no second, independently-populated pool entry left for a leaking
/// CUDA arena to be shared into. The bug the flag protects against is therefore moot here, not
/// forgotten; if a future phase reintroduces per-scope pools (e.g. for the Blazor UI, §6),
/// this reasoning — and the version gate itself, preserved in
/// <see cref="Execution.GetOnnxRuntimeVersion"/> and <see cref="Execution.HasExecutionProvider"/>
/// — should be revisited.
/// </para>
///
/// <para>
/// <b>Deviation 3 — <c>resolve_static_inference_providers</c>'s <c>importlib</c> module hook.</b>
/// Python resolves <c>module_name</c> via <c>importlib.import_module</c> and looks for
/// optional <c>override_inference_providers</c> / <c>adjust_inference_providers</c> functions
/// on the imported module (used today by <c>background_remover</c>, <c>frame_colorizer</c>,
/// <c>face_swapper</c> and <c>frame_enhancer</c> — all Phase 5 processor modules that do not
/// exist yet in this port). Per DOTNET_PORT_PLAN.md §3/§9.5, processors become DI-resolved
/// <c>IProcessor</c> implementations rather than importlib-scanned modules, so there is no
/// module to import by string here. <see cref="ResolveStaticInferenceProviders"/> keeps the
/// same call shape and caching but takes the two hooks as explicit optional delegates instead
/// — a Phase 5 processor passes its own override/adjust callback when it calls
/// <see cref="GetInferencePool"/>, which forwards them here. <c>module_name</c> is kept as the
/// cache key (as in Python) even though it no longer names an importable module.
/// </para>
///
/// <para>
/// <b>Deviation 4 — instance class, not a module global.</b> Per PORT_CONVENTIONS.md rule 5,
/// this is an instance class with the pool held in a private field guarded by a lock,
/// consistent with how <see cref="ProcessManager"/> was ported. Callers that want
/// module-global behaviour should share one instance (e.g. via DI, as a singleton).
/// </para>
///
/// <para>
/// <b>Deviation 5 — no <c>fatal_exit</c>.</b> Python's <c>create_inference_session</c> calls
/// <c>fatal_exit(1)</c> (<c>os._exit(1)</c>, an unconditional, un-catchable process kill) on a
/// load failure. <c>facefusion/exit_helper.py</c> is out of this phase's scope and killing the
/// host process from inside a library method is not testable and not appropriate for a class
/// library consumed via DI. <see cref="CreateInferenceSession"/> logs the same error message
/// and then throws <see cref="InvalidOperationException"/> instead; a CLI entry point built in
/// a later phase can catch that and call a ported <c>fatal_exit</c>/<c>hard_exit</c> if that
/// behaviour is still wanted at the process boundary.
/// </para>
///
/// <para>
/// <b>Deviation 6 — the <c>OrtValue</c> zero-copy calling convention.</b> Per
/// DOTNET_PORT_PLAN.md §5.3, this is where that convention is established for the rest of the
/// port: sessions are handed out by this pool and callers run them with
/// <c>InferenceSession.Run(RunOptions, IReadOnlyDictionary&lt;string, OrtValue&gt;, IReadOnlyList&lt;string&gt;)</c>
/// over <c>OrtValue.CreateTensorValueFromMemory</c>, never the older
/// <c>NamedOnnxValue</c>/<c>DenseTensor</c> path. <c>inference_manager.py</c> itself never
/// calls <c>Run</c> (only Phase 4/5 processor code does), so there is nothing to convert here;
/// see <c>InferenceManagerTests.TestRunSessionWithOrtValues</c> for a worked example against a
/// real ONNX model, exercised end to end through this pool.
/// </para>
/// </summary>
public sealed class InferenceManager : IDisposable
{
    private readonly object _lock = new();
    private readonly Dictionary<string, Dictionary<string, InferenceSession>> _pool = new();
    private readonly Dictionary<(string ModuleName, int ExecutionDeviceId), IReadOnlyList<InferenceProviderEntry>> _staticInferenceProvidersCache = new();
    private readonly ProcessManager _processManager;
    private readonly Logger _logger;
    private readonly Random _random = new();
    private bool _disposed;

    public InferenceManager(ProcessManager? processManager = null, Logger? logger = null)
    {
        _processManager = processManager ?? new ProcessManager();
        _logger = logger ?? new Logger();
    }

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    /// <param name="overrideInferenceProviders">
    /// Stands in for a Phase-5 processor module's <c>override_inference_providers()</c> hook
    /// (deviation 3 above). Pass <see langword="null"/> when the caller has none.
    /// </param>
    /// <param name="adjustInferenceProviders">
    /// Stands in for a Phase-5 processor module's <c>adjust_inference_providers()</c> hook
    /// (deviation 3 above). Pass <see langword="null"/> when the caller has none.
    /// </param>
    public IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        string moduleName,
        IReadOnlyList<string> modelNames,
        IReadOnlyDictionary<string, Download> modelSourceSet,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders,
        Func<IReadOnlyList<InferenceProviderEntry>>? overrideInferenceProviders = null,
        Func<IReadOnlyList<InferenceProviderEntry>>? adjustInferenceProviders = null)
    {
        while (_processManager.IsChecking())
        {
            Thread.Sleep(500);
        }

        lock (_lock)
        {
            foreach (var executionDeviceId in executionDeviceIds)
            {
                var inferenceContext = GetInferenceContext(moduleName, modelNames, executionDeviceId, executionProviders);

                if (!_pool.ContainsKey(inferenceContext))
                {
                    var inferenceProviders = ResolveStaticInferenceProviders(moduleName, executionDeviceId, executionProviders, overrideInferenceProviders, adjustInferenceProviders);
                    _pool[inferenceContext] = CreateInferencePoolDictionary(modelSourceSet, inferenceProviders);
                }
            }

            var chosenExecutionDeviceId = executionDeviceIds[_random.Next(executionDeviceIds.Count)];
            var currentInferenceContext = GetInferenceContext(moduleName, modelNames, chosenExecutionDeviceId, executionProviders);
            return _pool[currentInferenceContext];
        }
    }

    /// <summary>Python: <c>create_inference_pool</c>.</summary>
    public IReadOnlyDictionary<string, InferenceSession> CreateInferencePool(IReadOnlyDictionary<string, Download> modelSourceSet, IReadOnlyList<InferenceProviderEntry> inferenceProviders)
    {
        return CreateInferencePoolDictionary(modelSourceSet, inferenceProviders);
    }

    private Dictionary<string, InferenceSession> CreateInferencePoolDictionary(IReadOnlyDictionary<string, Download> modelSourceSet, IReadOnlyList<InferenceProviderEntry> inferenceProviders)
    {
        var inferencePool = new Dictionary<string, InferenceSession>();

        foreach (var (modelName, download) in modelSourceSet)
        {
            if (FileSystem.IsFile(download.Path))
            {
                inferencePool[modelName] = CreateInferenceSession(download.Path, inferenceProviders);
            }
        }

        return inferencePool;
    }

    /// <summary>
    /// Python: <c>clear_inference_pool</c>. Disposes every <see cref="InferenceSession"/> it
    /// removes (plan §5a: the pool owns sessions and must dispose them on clear — Python
    /// relies on the CPython refcount/GC finalizer for this instead).
    /// </summary>
    public void ClearInferencePool(string moduleName, IReadOnlyList<string> modelNames, IReadOnlyList<int> executionDeviceIds, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        lock (_lock)
        {
            if (CommonHelper.IsWindows() && Execution.HasExecutionProvider(ExecutionProvider.Directml))
            {
                foreach (var inferencePool in _pool.Values)
                {
                    foreach (var inferenceSession in inferencePool.Values)
                    {
                        inferenceSession.Dispose();
                    }
                }

                _pool.Clear();
            }

            foreach (var executionDeviceId in executionDeviceIds)
            {
                var inferenceContext = GetInferenceContext(moduleName, modelNames, executionDeviceId, executionProviders);

                if (_pool.TryGetValue(inferenceContext, out var inferencePool))
                {
                    foreach (var inferenceSession in inferencePool.Values)
                    {
                        inferenceSession.Dispose();
                    }

                    _pool.Remove(inferenceContext);
                }
            }
        }
    }

    /// <summary>Python: <c>create_inference_session</c>.</summary>
    public InferenceSession CreateInferenceSession(string modelPath, IReadOnlyList<InferenceProviderEntry> inferenceProviders)
    {
        var modelFileName = FileSystem.GetFileName(modelPath) ?? modelPath;
        var startTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;

        try
        {
            using var sessionOptions = CreateSessionOptions(inferenceProviders);
            var inferenceSession = new InferenceSession(modelPath, sessionOptions);
            var seconds = TimeHelper.CalculateEndTime(startTime);

            _logger.Debug(
                Translator.Get("loading_model_succeeded", ("model_name", modelFileName), ("seconds", seconds))
                    ?? $"loading model {modelFileName} succeeded in {seconds.ToString(CultureInfo.InvariantCulture)} seconds",
                nameof(InferenceManager));

            return inferenceSession;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            _logger.Error(
                Translator.Get("loading_model_failed", ("model_name", modelFileName))
                    ?? $"loading model {modelFileName} failed",
                nameof(InferenceManager));

            // Python: fatal_exit(1) — see deviation 5 above.
            throw new InvalidOperationException($"loading model {modelFileName} failed", exception);
        }
    }

    /// <summary>Python: <c>get_inference_context</c>.</summary>
    public static string GetInferenceContext(string moduleName, IReadOnlyList<string> modelNames, int executionDeviceId, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var fragments = new List<string> { moduleName };
        fragments.AddRange(modelNames);
        fragments.Add(executionDeviceId.ToString(CultureInfo.InvariantCulture));
        fragments.AddRange(executionProviders.Select(executionProvider => executionProvider.ToWireName()));

        return string.Join(".", fragments);
    }

    /// <summary>
    /// Python: <c>resolve_static_inference_providers</c> (<c>@lru_cache()</c>). See deviation
    /// 3 in the class doc comment for how the <c>importlib</c> module hook was translated.
    /// </summary>
    public IReadOnlyList<InferenceProviderEntry> ResolveStaticInferenceProviders(
        string moduleName,
        int executionDeviceId,
        IReadOnlyList<ExecutionProvider> executionProviders,
        Func<IReadOnlyList<InferenceProviderEntry>>? overrideInferenceProviders = null,
        Func<IReadOnlyList<InferenceProviderEntry>>? adjustInferenceProviders = null)
    {
        var cacheKey = (moduleName, executionDeviceId);

        lock (_lock)
        {
            if (_staticInferenceProvidersCache.TryGetValue(cacheKey, out var cachedResult))
            {
                return cachedResult;
            }

            var result = ComputeStaticInferenceProviders(executionDeviceId, executionProviders, overrideInferenceProviders, adjustInferenceProviders);
            _staticInferenceProvidersCache[cacheKey] = result;
            return result;
        }
    }

    private static IReadOnlyList<InferenceProviderEntry> ComputeStaticInferenceProviders(
        int executionDeviceId,
        IReadOnlyList<ExecutionProvider> executionProviders,
        Func<IReadOnlyList<InferenceProviderEntry>>? overrideInferenceProviders,
        Func<IReadOnlyList<InferenceProviderEntry>>? adjustInferenceProviders)
    {
        var overridden = overrideInferenceProviders?.Invoke();

        if (overridden is { Count: > 0 })
        {
            return overridden;
        }

        var adjustments = adjustInferenceProviders?.Invoke();

        if (adjustments is { Count: > 0 })
        {
            var inferenceProviders = Execution.CreateInferenceProviders(executionDeviceId, executionProviders)
                .Select(inferenceProvider => new InferenceProviderEntry(
                    inferenceProvider.ProviderName,
                    inferenceProvider.Options is null ? null : new Dictionary<string, object?>(inferenceProvider.Options)))
                .ToList();

            foreach (var adjustment in adjustments)
            {
                foreach (var inferenceProvider in inferenceProviders)
                {
                    // Python: `if inference_provider[0] == adjust_inference_provider[0] and inference_provider[1]:`.
                    // Python's create_inference_providers mixes bare provider-name strings (the
                    // trailing 'cpu' entry, which has no options dict) with (name, dict) tuples
                    // in the same list, so `inference_provider[1]` truthiness there also
                    // happens to filter those out. This port always uses the uniform
                    // InferenceProviderEntry shape, so the equivalent filter is simply "has an
                    // options dictionary to update".
                    if (inferenceProvider.ProviderName == adjustment.ProviderName
                        && inferenceProvider.Options is Dictionary<string, object?> mutableOptions
                        && adjustment.Options is not null)
                    {
                        foreach (var (key, value) in adjustment.Options)
                        {
                            mutableOptions[key] = value;
                        }
                    }
                }
            }

            return inferenceProviders;
        }

        return Execution.CreateInferenceProviders(executionDeviceId, executionProviders);
    }

    private static SessionOptions CreateSessionOptions(IReadOnlyList<InferenceProviderEntry> inferenceProviders)
    {
        var sessionOptions = new SessionOptions();

        foreach (var inferenceProvider in inferenceProviders)
        {
            var providerOptions = new Dictionary<string, string>();

            if (inferenceProvider.Options is not null)
            {
                foreach (var (key, value) in inferenceProvider.Options)
                {
                    providerOptions[key] = ConvertProviderOptionValue(value);
                }
            }

            sessionOptions.AppendExecutionProvider(inferenceProvider.ProviderName, providerOptions);
        }

        return sessionOptions;
    }

    /// <summary>
    /// <c>SessionOptions.AppendExecutionProvider(string, Dictionary&lt;string, string&gt;)</c>
    /// only accepts string values; Python's provider option dicts hold native <c>int</c>,
    /// <c>bool</c> and <c>str</c> values (see <see cref="Execution.CreateInferenceProviders"/>),
    /// converted to their ONNX Runtime provider-option string form here (lowercase
    /// <c>"true"</c>/<c>"false"</c> for booleans, matching the convention ORT's own C/C++ option
    /// parsing expects).
    /// </summary>
    private static string ConvertProviderOptionValue(object? value)
    {
        return value switch
        {
            null => string.Empty,
            bool booleanValue => booleanValue ? "true" : "false",
            string stringValue => stringValue,
            int intValue => intValue.ToString(CultureInfo.InvariantCulture),
            IFormattable formattableValue => formattableValue.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? string.Empty
        };
    }

    /// <summary>
    /// Not present in Python (which relies on the GC to reclaim <c>InferenceSession</c>
    /// instances). Added per plan §5a — <c>InferenceSession</c> wraps native memory and the
    /// pool that owns it must dispose it. Disposes every pooled session.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_lock)
        {
            foreach (var inferencePool in _pool.Values)
            {
                foreach (var inferenceSession in inferencePool.Values)
                {
                    inferenceSession.Dispose();
                }
            }

            _pool.Clear();
        }

        _disposed = true;
    }
}
