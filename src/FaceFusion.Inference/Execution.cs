using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using FaceFusion.Core;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;

namespace FaceFusion.Inference;

/// <summary>
/// Port of <c>facefusion/execution.py</c>.
///
/// Static class per PORT_CONVENTIONS.md rule 4 (a Python module of free functions becomes a
/// <c>public static class</c>). Every function here is pure aside from two Python
/// <c>@lru_cache()</c>-memoized functions (<c>get_onnxruntime_version</c>,
/// <c>detect_static_execution_devices</c>), which are reproduced as small process-lifetime
/// caches guarded by a lock — these are memoization of a deterministic external answer (the
/// installed ORT version, the attached NVIDIA hardware), not mutable application state, so
/// they are not in scope of port convention rule 5 the way <c>state_manager</c> values are.
/// </summary>
public static class Execution
{
    private static readonly string[] LowPowerNvidiaProductNames =
    {
        "GeForce GTX 1630", "GeForce GTX 1650", "GeForce GTX 1660"
    };

    private static readonly object VersionLock = new();
    private static (int Major, int Minor, int Patch)? _cachedOnnxRuntimeVersion;

    private static readonly object DevicesLock = new();
    private static IReadOnlyList<ExecutionDevice>? _cachedStaticExecutionDevices;

    static Execution()
    {
        // Python: onnxruntime.set_default_logger_severity(3) at module import time.
        // ORT_LOGGING_LEVEL_ERROR == severity 3 in the native C API that both bindings wrap.
        OrtEnv.Instance().EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
    }

    /// <summary>Python: <c>has_execution_provider</c>.</summary>
    public static bool HasExecutionProvider(ExecutionProvider executionProvider)
    {
        return GetAvailableExecutionProviders().Contains(executionProvider);
    }

    /// <summary>
    /// Python: <c>get_onnxruntime_version</c>. Python parses <c>onnxruntime.__version__</c>
    /// (the installed package version, which for GPU packages can carry a suffix like
    /// <c>1.20.1+cu124</c>); here the equivalent is ONNX Runtime's own
    /// <c>OrtEnv.GetVersionString()</c>, which on this 1.29.0 CPU package reports a plain
    /// <c>"1.29.0"</c> but is parsed defensively for the same <c>+suffix</c> shape on the
    /// patch component, matching Python's <c>version_split[2].split('+')[0]</c>.
    /// </summary>
    public static (int Major, int Minor, int Patch) GetOnnxRuntimeVersion()
    {
        lock (VersionLock)
        {
            if (_cachedOnnxRuntimeVersion is { } cached)
            {
                return cached;
            }

            var result = ParseOnnxRuntimeVersion(OrtEnv.Instance().GetVersionString());
            _cachedOnnxRuntimeVersion = result;
            return result;
        }
    }

    /// <summary>
    /// The parsing half of <see cref="GetOnnxRuntimeVersion"/>, split out so the
    /// <c>+suffix</c> handling can be unit-tested without depending on which ONNX Runtime
    /// package/build is installed. Python: <c>version_split = onnxruntime.__version__.split('.')</c>
    /// then <c>int(version_split[2].split('+')[0])</c> for the patch component.
    /// </summary>
    public static (int Major, int Minor, int Patch) ParseOnnxRuntimeVersion(string versionString)
    {
        var versionSplit = versionString.Split('.');
        var majorVersion = int.Parse(versionSplit[0], CultureInfo.InvariantCulture);
        var minorVersion = int.Parse(versionSplit[1], CultureInfo.InvariantCulture);
        var patchVersion = int.Parse(versionSplit[2].Split('+')[0], CultureInfo.InvariantCulture);

        return (majorVersion, minorVersion, patchVersion);
    }

    /// <summary>Python: <c>get_available_execution_providers</c>.</summary>
    public static IReadOnlyList<ExecutionProvider> GetAvailableExecutionProviders()
    {
        var inferenceSessionProviders = OrtEnv.Instance().GetAvailableProviders();
        var availableExecutionProviders = new List<ExecutionProvider>();

        // Python builds this by inserting into the result at
        // `execution_providers.index(execution_provider)` while iterating
        // `execution_provider_set` in its declared order. Choices.ExecutionProviderSet and the
        // ExecutionProvider enum are both declared in exactly that same canonical order here
        // (see facefusion/choices.py: execution_provider_set and execution_providers), so a
        // single ordered pass with no explicit index bookkeeping produces an identical result.
        foreach (var executionProvider in Enum.GetValues<ExecutionProvider>())
        {
            var executionProviderValue = Choices.ExecutionProviderSet[executionProvider];

            if (inferenceSessionProviders.Contains(executionProviderValue.ToWireName()))
            {
                availableExecutionProviders.Add(executionProvider);
            }
        }

        return availableExecutionProviders;
    }

    /// <summary>Python: <c>create_inference_providers</c>.</summary>
    public static IReadOnlyList<InferenceProviderEntry> CreateInferenceProviders(int executionDeviceId, IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var inferenceProviders = new List<InferenceProviderEntry>();
        var cachePath = ResolveCachePath();

        foreach (var executionProvider in executionProviders)
        {
            if (executionProvider == ExecutionProvider.Cuda)
            {
                inferenceProviders.Add(new InferenceProviderEntry(
                    Choices.ExecutionProviderSet[executionProvider].ToWireName(),
                    new Dictionary<string, object?>
                    {
                        ["device_id"] = executionDeviceId,
                        ["cudnn_conv_algo_search"] = ResolveCudnnConvAlgoSearch()
                    }));
            }

            if (executionProvider == ExecutionProvider.Tensorrt)
            {
                var inferenceOptionSet = new Dictionary<string, object?>
                {
                    ["device_id"] = executionDeviceId
                };

                if (FileSystem.IsDirectory(cachePath) || FileSystem.CreateDirectory(cachePath))
                {
                    inferenceOptionSet["trt_engine_cache_enable"] = true;
                    inferenceOptionSet["trt_engine_cache_path"] = cachePath;
                    inferenceOptionSet["trt_timing_cache_enable"] = true;
                    inferenceOptionSet["trt_timing_cache_path"] = cachePath;
                    inferenceOptionSet["trt_builder_optimization_level"] = 4;
                }

                inferenceProviders.Add(new InferenceProviderEntry(Choices.ExecutionProviderSet[executionProvider].ToWireName(), inferenceOptionSet));
            }

            if (executionProvider is ExecutionProvider.Directml or ExecutionProvider.Rocm)
            {
                inferenceProviders.Add(new InferenceProviderEntry(
                    Choices.ExecutionProviderSet[executionProvider].ToWireName(),
                    new Dictionary<string, object?>
                    {
                        ["device_id"] = executionDeviceId
                    }));
            }

            if (executionProvider == ExecutionProvider.Migraphx)
            {
                var inferenceOptionSet = new Dictionary<string, object?>
                {
                    ["device_id"] = executionDeviceId
                };

                if (FileSystem.IsDirectory(cachePath) || FileSystem.CreateDirectory(cachePath))
                {
                    inferenceOptionSet["migraphx_model_cache_dir"] = cachePath;
                }

                inferenceProviders.Add(new InferenceProviderEntry(Choices.ExecutionProviderSet[executionProvider].ToWireName(), inferenceOptionSet));
            }

            if (executionProvider == ExecutionProvider.Coreml)
            {
                var inferenceOptionSet = new Dictionary<string, object?>
                {
                    ["SpecializationStrategy"] = "FastPrediction"
                };

                if (FileSystem.IsDirectory(cachePath) || FileSystem.CreateDirectory(cachePath))
                {
                    inferenceOptionSet["ModelCacheDirectory"] = cachePath;
                }

                inferenceProviders.Add(new InferenceProviderEntry(Choices.ExecutionProviderSet[executionProvider].ToWireName(), inferenceOptionSet));
            }

            if (executionProvider == ExecutionProvider.Openvino)
            {
                inferenceProviders.Add(new InferenceProviderEntry(
                    Choices.ExecutionProviderSet[executionProvider].ToWireName(),
                    new Dictionary<string, object?>
                    {
                        ["device_type"] = ResolveOpenVinoDeviceType(executionDeviceId),
                        ["precision"] = "FP32"
                    }));
            }

            if (executionProvider == ExecutionProvider.Qnn)
            {
                inferenceProviders.Add(new InferenceProviderEntry(
                    Choices.ExecutionProviderSet[executionProvider].ToWireName(),
                    new Dictionary<string, object?>
                    {
                        ["device_id"] = executionDeviceId,
                        ["backend_type"] = "htp"
                    }));
            }
        }

        if (executionProviders.Contains(ExecutionProvider.Cpu))
        {
            inferenceProviders.Add(new InferenceProviderEntry(Choices.ExecutionProviderSet[ExecutionProvider.Cpu].ToWireName(), null));
        }

        return inferenceProviders;
    }

    /// <summary>Python: <c>resolve_cache_path</c>.</summary>
    public static string ResolveCachePath()
    {
        return Path.Combine(".caches", OrtEnv.Instance().GetVersionString());
    }

    /// <summary>Python: <c>resolve_cudnn_conv_algo_search</c>.</summary>
    public static string ResolveCudnnConvAlgoSearch()
    {
        var executionDevices = DetectStaticExecutionDevices();

        foreach (var executionDevice in executionDevices)
        {
            if (LowPowerNvidiaProductNames.Any(productName => executionDevice.Product.Name.StartsWith(productName, StringComparison.Ordinal)))
            {
                return "DEFAULT";
            }
        }

        return "EXHAUSTIVE";
    }

    /// <summary>Python: <c>resolve_openvino_device_type</c>.</summary>
    public static string ResolveOpenVinoDeviceType(int executionDeviceId)
    {
        if (executionDeviceId == 0)
        {
            return "GPU";
        }

        return "GPU." + executionDeviceId.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Python: <c>run_nvidia_smi</c>. Python returns the live <c>Popen</c> handle for the
    /// caller to <c>.communicate()</c>; the direct C# analogue is the started
    /// <see cref="Process"/>, left for the caller to read stdout from and dispose. Returns
    /// <see langword="null"/> where Python's <c>shutil.which('nvidia-smi')</c> would return
    /// <see langword="None"/> (binary not on PATH) — Python would then pass <c>None</c> as the
    /// first argument to <c>Popen</c>, which raises, which is caught by the try/except in
    /// <c>detect_execution_devices</c>; returning <see langword="null"/> here lets
    /// <see cref="DetectExecutionDevices"/> take the equivalent no-GPU-found path without an
    /// exception in the common (no <c>nvidia-smi</c> installed) case.
    /// </summary>
    public static Process? RunNvidiaSmi()
    {
        var executablePath = ProcessHelper.Which("nvidia-smi");

        if (executablePath is null)
        {
            return null;
        }

        var startInfo = new ProcessStartInfo(executablePath, "--query --xml-format")
        {
            RedirectStandardOutput = true,
            UseShellExecute = false
        };

        return Process.Start(startInfo);
    }

    /// <summary>Python: <c>detect_static_execution_devices</c> (<c>@lru_cache()</c>).</summary>
    public static IReadOnlyList<ExecutionDevice> DetectStaticExecutionDevices()
    {
        lock (DevicesLock)
        {
            if (_cachedStaticExecutionDevices is { } cached)
            {
                return cached;
            }

            var result = DetectExecutionDevices();
            _cachedStaticExecutionDevices = result;
            return result;
        }
    }

    /// <summary>Python: <c>detect_execution_devices</c>.</summary>
    public static IReadOnlyList<ExecutionDevice> DetectExecutionDevices()
    {
        string outputXml;

        try
        {
            using var process = RunNvidiaSmi();

            if (process is null)
            {
                outputXml = "<xml></xml>";
            }
            else
            {
                outputXml = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
            }
        }
        catch
        {
            // Python: `except Exception: root_element = ElementTree.Element('xml')` — any
            // failure launching or reading nvidia-smi is treated as "no GPU found".
            outputXml = "<xml></xml>";
        }

        return ParseExecutionDevicesXml(outputXml);
    }

    /// <summary>
    /// The XML-parsing half of Python's <c>detect_execution_devices</c>, split out (per the
    /// assignment) so it is testable without a real <c>nvidia-smi</c> binary. Takes the raw
    /// <c>nvidia-smi --query --xml-format</c> output and returns one <see cref="ExecutionDevice"/>
    /// per <c>&lt;gpu&gt;</c> element. Malformed XML is treated the same as Python's
    /// try/except — an empty device list — since Python's <c>ElementTree.fromstring</c> failure
    /// is caught by the same outer try/except as the subprocess call.
    /// </summary>
    public static IReadOnlyList<ExecutionDevice> ParseExecutionDevicesXml(string outputXml)
    {
        XElement rootElement;

        try
        {
            rootElement = XElement.Parse(outputXml);
        }
        catch
        {
            return Array.Empty<ExecutionDevice>();
        }

        var driverVersion = rootElement.Element("driver_version")?.Value ?? string.Empty;
        var cudaVersion = rootElement.Element("cuda_version")?.Value ?? string.Empty;
        var executionDevices = new List<ExecutionDevice>();

        foreach (var gpuElement in rootElement.Elements("gpu"))
        {
            var productName = gpuElement.Element("product_name")?.Value ?? string.Empty;

            executionDevices.Add(new ExecutionDevice(
                DriverVersion: driverVersion,
                Framework: new ExecutionDeviceFramework("CUDA", cudaVersion),
                Product: new ExecutionDeviceProduct("NVIDIA", productName.Replace("NVIDIA", string.Empty).Trim()),
                VideoMemory: new ExecutionDeviceVideoMemory(
                    CreateValueAndUnit(FindElementText(gpuElement, "fb_memory_usage", "total")),
                    CreateValueAndUnit(FindElementText(gpuElement, "fb_memory_usage", "free"))),
                Temperature: new ExecutionDeviceTemperature(
                    CreateValueAndUnit(FindElementText(gpuElement, "temperature", "gpu_temp")),
                    CreateValueAndUnit(FindElementText(gpuElement, "temperature", "memory_temp"))),
                Utilization: new ExecutionDeviceUtilization(
                    CreateValueAndUnit(FindElementText(gpuElement, "utilization", "gpu_util")),
                    CreateValueAndUnit(FindElementText(gpuElement, "utilization", "memory_util")))));
        }

        return executionDevices;
    }

    /// <summary>
    /// Python: <c>gpu_element.findtext('parent/child')</c> — a one-level XPath-style lookup.
    /// Returns <see langword="null"/> where Python's <c>findtext</c> would return
    /// <see langword="None"/> (element missing).
    /// </summary>
    private static string? FindElementText(XElement element, string parentName, string childName)
    {
        return element.Element(parentName)?.Element(childName)?.Value;
    }

    /// <summary>Python: <c>create_value_and_unit</c>.</summary>
    public static ValueAndUnit? CreateValueAndUnit(string? text)
    {
        // Python: `if ' ' in text` — if findtext() returned None (element missing), this
        // raises TypeError, uncaught, and propagates out of detect_execution_devices(). Real
        // nvidia-smi XML always includes these elements, so this is unreachable in practice;
        // reproduced here as a NullReferenceException (via the null-forgiving `!`) rather than
        // a silent null-check that would diverge from Python's crash-on-malformed-input
        // behaviour for genuinely malformed nvidia-smi output.
        if (!text!.Contains(' '))
        {
            return null;
        }

        // Python: `value, unit = text.split()` — default whitespace split, unpacked into
        // exactly two names; a text with more or fewer than two whitespace-separated tokens
        // raises ValueError there, reproduced here as the FormatException from an unmatched
        // array pattern below.
        var parts = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        if (parts.Length != 2)
        {
            throw new FormatException($"expected exactly two whitespace-separated tokens in '{text}'");
        }

        var value = int.Parse(parts[0], CultureInfo.InvariantCulture);
        var unit = parts[1];

        return new ValueAndUnit(value, unit);
    }
}
