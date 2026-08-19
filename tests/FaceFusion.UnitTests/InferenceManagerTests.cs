using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using FaceFusion.Core;
using FaceFusion.Inference;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_inference_manager.py</c>.
///
/// There is no <c>content_analyser</c> model set available in this environment (no network,
/// no <c>.assets/models</c>), so these tests build their own model source set around a tiny,
/// hand-generated, valid ONNX model (a single <c>Identity</c> node, 123 bytes, embedded below
/// as base64) instead of skipping session-creation coverage outright. It was produced with:
/// <code>
/// import onnx
/// from onnx import helper, TensorProto
/// node = helper.make_node('Identity', ['input'], ['output'])
/// graph = helper.make_graph([node], 'tiny_identity',
///     [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])],
///     [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])])
/// model = helper.make_model(graph, producer_name='facefusion-rs-test', opset_imports=[helper.make_opsetid('', 13)])
/// model.ir_version = 8
/// </code>
/// This lets <see cref="InferenceManager.CreateInferenceSession"/>,
/// <see cref="InferenceManager.CreateInferencePool"/>,
/// <see cref="InferenceManager.GetInferencePool"/> and
/// <see cref="InferenceManager.ClearInferencePool"/> all be exercised against a real
/// <see cref="InferenceSession"/> on the CPU execution provider, and lets the
/// <c>OrtValue</c>-based zero-copy Run() convention (DOTNET_PORT_PLAN.md §5.3) be demonstrated
/// end to end rather than only asserted by type.
/// </summary>
[Collection("NativeInference")]
public sealed class InferenceManagerTests
{
    private const string TinyIdentityModelBase64 =
        "CAgSEmZhY2VmdXNpb24tcnMtdGVzdDpdChkKBWlucHV0EgZvdXRwdXQiCElkZW50aXR5Eg10aW55X2lkZW50aXR5WhcKBWlucHV0Eg4KDAgBEggKAggBCgIIA2IYCgZvdXRwdXQSDgoMCAESCAoCCAEKAggDQgQKABAN";

    private static string WriteTinyIdentityModel()
    {
        var modelPath = Path.Combine(Path.GetTempPath(), $"facefusion-tiny-identity-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(modelPath, Convert.FromBase64String(TinyIdentityModelBase64));
        return modelPath;
    }

    // --- get_inference_context --------------------------------------------------------------

    [Fact]
    public void TestGetInferenceContext()
    {
        var inferenceContext = InferenceManager.GetInferenceContext(
            "facefusion.content_analyser",
            new[] { "nsfw_1", "nsfw_2", "nsfw_3" },
            0,
            new[] { ExecutionProvider.Cpu });

        Assert.Equal("facefusion.content_analyser.nsfw_1.nsfw_2.nsfw_3.0.cpu", inferenceContext);
    }

    [Fact]
    public void TestGetInferenceContextVariesByDeviceIdAndProviders()
    {
        var baseContext = InferenceManager.GetInferenceContext("mod", new[] { "m1" }, 0, new[] { ExecutionProvider.Cpu });
        var otherDevice = InferenceManager.GetInferenceContext("mod", new[] { "m1" }, 1, new[] { ExecutionProvider.Cpu });
        var otherProvider = InferenceManager.GetInferenceContext("mod", new[] { "m1" }, 0, new[] { ExecutionProvider.Cuda, ExecutionProvider.Cpu });

        Assert.NotEqual(baseContext, otherDevice);
        Assert.NotEqual(baseContext, otherProvider);
    }

    // --- resolve_static_inference_providers --------------------------------------------------
    // Python's test patches `facefusion.inference_manager.importlib` to hand back a fake
    // module exposing `override_inference_providers` / `adjust_inference_providers`; here the
    // equivalent is passing those two hooks as delegates directly (see the "Deviation 3" note
    // on InferenceManager).

    [Fact]
    public void TestResolveStaticInferenceProvidersOverrideHookWins()
    {
        var inferenceManager = new InferenceManager();
        var executionProviders = new[] { ExecutionProvider.Coreml };

        var inferenceProviders = inferenceManager.ResolveStaticInferenceProviders(
            "override_module",
            0,
            executionProviders,
            overrideInferenceProviders: () => new[]
            {
                new InferenceProviderEntry("CoreMLExecutionProvider", new Dictionary<string, object?> { ["ModelFormat"] = "MLProgram" })
            });

        Assert.Single(inferenceProviders);
        Assert.Equal("CoreMLExecutionProvider", inferenceProviders[0].ProviderName);
        Assert.Equal("MLProgram", inferenceProviders[0].Options!["ModelFormat"]);
        Assert.Single(inferenceProviders[0].Options!);
    }

    [Fact]
    public void TestResolveStaticInferenceProvidersAdjustHookMergesIntoDefaults()
    {
        var inferenceManager = new InferenceManager();
        var executionProviders = new[] { ExecutionProvider.Coreml };

        var inferenceProviders = inferenceManager.ResolveStaticInferenceProviders(
            "adjust_module",
            0,
            executionProviders,
            adjustInferenceProviders: () => new[]
            {
                new InferenceProviderEntry("CoreMLExecutionProvider", new Dictionary<string, object?> { ["ModelFormat"] = "MLProgram" })
            });

        Assert.Single(inferenceProviders);
        var options = inferenceProviders[0].Options!;
        Assert.Equal("FastPrediction", options["SpecializationStrategy"]);
        Assert.Equal(Execution.ResolveCachePath(), options["ModelCacheDirectory"]);
        Assert.Equal("MLProgram", options["ModelFormat"]);
    }

    [Fact]
    public void TestResolveStaticInferenceProvidersWithNoHooksFallsBackToCreateInferenceProviders()
    {
        var inferenceManager = new InferenceManager();
        var executionProviders = new[] { ExecutionProvider.Coreml };

        var inferenceProviders = inferenceManager.ResolveStaticInferenceProviders("plain_module", 0, executionProviders);

        Assert.Single(inferenceProviders);
        var options = inferenceProviders[0].Options!;
        Assert.Equal("FastPrediction", options["SpecializationStrategy"]);
        Assert.Equal(Execution.ResolveCachePath(), options["ModelCacheDirectory"]);
        Assert.False(options.ContainsKey("ModelFormat"));
    }

    [Fact]
    public void TestResolveStaticInferenceProvidersIsCachedPerModuleAndDevice()
    {
        var inferenceManager = new InferenceManager();
        var executionProviders = new[] { ExecutionProvider.Cpu };
        var callCount = 0;

        IReadOnlyList<InferenceProviderEntry> Adjust()
        {
            callCount++;
            return new[] { new InferenceProviderEntry("CPUExecutionProvider", new Dictionary<string, object?> { ["marker"] = callCount }) };
        }

        inferenceManager.ResolveStaticInferenceProviders("cached_module", 0, executionProviders, adjustInferenceProviders: Adjust);
        inferenceManager.ResolveStaticInferenceProviders("cached_module", 0, executionProviders, adjustInferenceProviders: Adjust);

        // Python: @lru_cache() on (module_name, execution_device_id) — the second call must
        // not re-invoke the hook.
        Assert.Equal(1, callCount);
    }

    // --- create_inference_session / create_inference_pool, against a real CPU session --------

    [Fact]
    public void TestCreateInferenceSessionWithRealCpuSession()
    {
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var inferenceManager = new InferenceManager();
            var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });

            using var inferenceSession = inferenceManager.CreateInferenceSession(modelPath, inferenceProviders);

            Assert.Contains("input", inferenceSession.InputMetadata.Keys);
            Assert.Contains("output", inferenceSession.OutputMetadata.Keys);
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact]
    public void TestCreateInferenceSessionFailureThrowsInsteadOfFatalExit()
    {
        // Python: create_inference_session catches any exception, logs loading_model_failed,
        // and calls fatal_exit(1) — an unconditional os._exit(1). See "Deviation 5" on
        // InferenceManager for why this port throws instead of killing the process.
        var badModelPath = Path.Combine(Path.GetTempPath(), $"facefusion-bad-model-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(badModelPath, new byte[] { 0x00, 0x01, 0x02, 0x03 });

        try
        {
            var inferenceManager = new InferenceManager();
            var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });

            var exception = Assert.Throws<InvalidOperationException>(() => inferenceManager.CreateInferenceSession(badModelPath, inferenceProviders));

            Assert.IsType<OnnxRuntimeException>(exception.InnerException);
        }
        finally
        {
            File.Delete(badModelPath);
        }
    }

    [Fact]
    public void TestCreateInferencePoolSkipsMissingModelFiles()
    {
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var modelSourceSet = new Dictionary<string, Download>
            {
                ["tiny"] = new Download("https://example.invalid/tiny.onnx", modelPath),
                ["missing"] = new Download("https://example.invalid/missing.onnx", Path.Combine(Path.GetTempPath(), $"facefusion-missing-{Guid.NewGuid():N}.onnx"))
            };
            var inferenceManager = new InferenceManager();
            var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });

            var inferencePool = inferenceManager.CreateInferencePool(modelSourceSet, inferenceProviders);

            Assert.True(inferencePool.ContainsKey("tiny"));
            Assert.False(inferencePool.ContainsKey("missing"));

            foreach (var session in inferencePool.Values)
            {
                session.Dispose();
            }
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    // --- get_inference_pool / clear_inference_pool -------------------------------------------

    [Fact]
    public void TestGetInferencePoolReturnsPooledSessionsAndSharesAcrossCalls()
    {
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var modelSourceSet = new Dictionary<string, Download> { ["tiny"] = new Download("https://example.invalid/tiny.onnx", modelPath) };
            var inferenceManager = new InferenceManager();

            var firstPool = inferenceManager.GetInferencePool("test_module_pool", new[] { "tiny" }, modelSourceSet, new[] { 0 }, new[] { ExecutionProvider.Cpu });
            var secondPool = inferenceManager.GetInferencePool("test_module_pool", new[] { "tiny" }, modelSourceSet, new[] { 0 }, new[] { ExecutionProvider.Cpu });

            Assert.True(firstPool.ContainsKey("tiny"));
            // Same inference context (module + models + device id + providers) must return the
            // exact same InferenceSession instance, not a freshly loaded one — this is the
            // entire point of the pool. With deviation 1 (single pool, no cli/ui split) this
            // is also what makes the old cli/ui sharing copy unconditional rather than
            // arena-leak-gated: see the "Deviation 2" note on InferenceManager.
            Assert.Same(firstPool["tiny"], secondPool["tiny"]);

            inferenceManager.ClearInferencePool("test_module_pool", new[] { "tiny" }, new[] { 0 }, new[] { ExecutionProvider.Cpu });

            var thirdPool = inferenceManager.GetInferencePool("test_module_pool", new[] { "tiny" }, modelSourceSet, new[] { 0 }, new[] { ExecutionProvider.Cpu });

            // After a clear, the pooled session was disposed and removed, so a fresh call
            // loads a brand-new InferenceSession.
            Assert.NotSame(firstPool["tiny"], thirdPool["tiny"]);
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact]
    public void TestGetInferencePoolCreatesAnEntryPerExecutionDeviceId()
    {
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var modelSourceSet = new Dictionary<string, Download> { ["tiny"] = new Download("https://example.invalid/tiny.onnx", modelPath) };
            var inferenceManager = new InferenceManager();

            var pool = inferenceManager.GetInferencePool("multi_device_module", new[] { "tiny" }, modelSourceSet, new[] { 0, 1 }, new[] { ExecutionProvider.Cpu });

            Assert.True(pool.ContainsKey("tiny"));

            // Both per-device contexts were populated by the loop in GetInferencePool (device
            // ids 0 and 1), so clearing either one independently must succeed without error.
            inferenceManager.ClearInferencePool("multi_device_module", new[] { "tiny" }, new[] { 0 }, new[] { ExecutionProvider.Cpu });
            inferenceManager.ClearInferencePool("multi_device_module", new[] { "tiny" }, new[] { 1 }, new[] { ExecutionProvider.Cpu });
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact]
    public void TestGetInferencePoolWaitsWhileProcessManagerIsChecking()
    {
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var processManager = new ProcessManager();
            processManager.Check();
            var inferenceManager = new InferenceManager(processManager);
            var modelSourceSet = new Dictionary<string, Download> { ["tiny"] = new Download("https://example.invalid/tiny.onnx", modelPath) };

            var releaseTask = Task.Run(() =>
            {
                Task.Delay(300).Wait();
                processManager.End();
            });

            var stopwatch = Stopwatch.StartNew();
            var pool = inferenceManager.GetInferencePool("waiting_module", new[] { "tiny" }, modelSourceSet, new[] { 0 }, new[] { ExecutionProvider.Cpu });
            stopwatch.Stop();

            Assert.True(pool.ContainsKey("tiny"));
            // Python: `while process_manager.is_checking(): sleep(0.5)` — must have actually
            // waited for the process manager to leave the checking state.
            Assert.True(stopwatch.ElapsedMilliseconds >= 250, $"expected GetInferencePool to block for ~300ms, only took {stopwatch.ElapsedMilliseconds}ms");

            releaseTask.Wait();
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact(Skip = "requires DirectML execution provider on Windows")]
    public void TestClearInferencePoolClearsEntirePoolOnWindowsWithDirectMl()
    {
        // Python: `if is_windows() and has_execution_provider('directml'): INFERENCE_POOL_SET[app_context].clear()`.
        // Ported verbatim in InferenceManager.ClearInferencePool (see the CommonHelper.IsWindows()
        // + Execution.HasExecutionProvider(Directml) branch there) but not exercisable in this
        // Linux, CPU-only container.
    }

    // --- OrtValue zero-copy calling convention (DOTNET_PORT_PLAN.md §5.3) --------------------

    [Fact]
    public void TestRunSessionFromPoolWithOrtValues()
    {
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var modelSourceSet = new Dictionary<string, Download> { ["tiny"] = new Download("https://example.invalid/tiny.onnx", modelPath) };
            var inferenceManager = new InferenceManager();

            var pool = inferenceManager.GetInferencePool("ortvalue_module", new[] { "tiny" }, modelSourceSet, new[] { 0 }, new[] { ExecutionProvider.Cpu });
            var inferenceSession = pool["tiny"];

            var inputData = new float[] { 1f, 2f, 3f };
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, new long[] { 1, 3 });
            var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

            using var runOptions = new RunOptions();
            using var results = inferenceSession.Run(runOptions, inputs, inferenceSession.OutputNames);

            var outputSpan = results[0].GetTensorDataAsSpan<float>();

            Assert.Equal(inputData, outputSpan.ToArray());
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    // --- Dispose --------------------------------------------------------------------------

    [Fact]
    public void TestDisposeDisposesAllPooledSessions()
    {
        // Note: this deliberately does not probe the returned InferenceSession after
        // Dispose() to confirm it was released — ONNX Runtime's managed wrapper does not
        // consistently guard against use-after-dispose (several members reach straight into
        // freed native memory and crash the process rather than throwing), so doing that from
        // a unit test would make the whole suite flaky/crash-prone. Disposing the pool and
        // then disposing it again (idempotency) is what is actually verified here.
        var modelPath = WriteTinyIdentityModel();

        try
        {
            var modelSourceSet = new Dictionary<string, Download> { ["tiny"] = new Download("https://example.invalid/tiny.onnx", modelPath) };
            var inferenceManager = new InferenceManager();

            var pool = inferenceManager.GetInferencePool("dispose_module", new[] { "tiny" }, modelSourceSet, new[] { 0 }, new[] { ExecutionProvider.Cpu });
            Assert.True(pool.ContainsKey("tiny"));

            inferenceManager.Dispose();
            inferenceManager.Dispose();
        }
        finally
        {
            File.Delete(modelPath);
        }
    }
}
