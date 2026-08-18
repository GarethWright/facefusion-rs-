using System.Collections.Generic;
using System.Linq;
using FaceFusion.Inference;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_execution.py</c>, plus additional pure-function coverage the
/// assignment calls out explicitly (per-provider option dictionaries, nvidia-smi XML
/// parsing, onnxruntime version parsing) since none of it needs a GPU or a model file.
/// </summary>
public sealed class ExecutionTests
{
    // --- Direct ports of tests/test_execution.py ---------------------------------------

    [Fact]
    public void TestHasExecutionProvider()
    {
        Assert.True(Execution.HasExecutionProvider(ExecutionProvider.Cpu));
        Assert.False(Execution.HasExecutionProvider(ExecutionProvider.Openvino));
    }

    [Fact]
    public void TestGetAvailableExecutionProviders()
    {
        Assert.Contains(ExecutionProvider.Cpu, Execution.GetAvailableExecutionProviders());
    }

    [Fact]
    public void TestCreateInferenceProviders()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(1, new[] { ExecutionProvider.Cpu, ExecutionProvider.Cuda });

        Assert.Equal(2, inferenceProviders.Count);

        Assert.Equal("CUDAExecutionProvider", inferenceProviders[0].ProviderName);
        Assert.NotNull(inferenceProviders[0].Options);
        Assert.Equal(1, inferenceProviders[0].Options!["device_id"]);
        // No nvidia-smi in this environment, so DetectStaticExecutionDevices() is always
        // empty and resolve_cudnn_conv_algo_search() always falls through to 'EXHAUSTIVE' —
        // matches the Python test, which asserts this exact value under the same condition.
        Assert.Equal("EXHAUSTIVE", inferenceProviders[0].Options!["cudnn_conv_algo_search"]);

        Assert.Equal("CPUExecutionProvider", inferenceProviders[1].ProviderName);
        Assert.Null(inferenceProviders[1].Options);
    }

    // --- Provider-name mapping -----------------------------------------------------------

    [Theory]
    [InlineData(ExecutionProvider.Cuda, "CUDAExecutionProvider")]
    [InlineData(ExecutionProvider.Tensorrt, "TensorrtExecutionProvider")]
    [InlineData(ExecutionProvider.Rocm, "ROCMExecutionProvider")]
    [InlineData(ExecutionProvider.Migraphx, "MIGraphXExecutionProvider")]
    [InlineData(ExecutionProvider.Coreml, "CoreMLExecutionProvider")]
    [InlineData(ExecutionProvider.Openvino, "OpenVINOExecutionProvider")]
    [InlineData(ExecutionProvider.Qnn, "QNNExecutionProvider")]
    [InlineData(ExecutionProvider.Directml, "DmlExecutionProvider")]
    [InlineData(ExecutionProvider.Cpu, "CPUExecutionProvider")]
    public void TestProviderNameMapping(ExecutionProvider executionProvider, string expectedProviderValue)
    {
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { executionProvider });

        Assert.Equal(expectedProviderValue, inferenceProviders[0].ProviderName);
    }

    // --- Per-provider option dictionaries --------------------------------------------------

    [Fact]
    public void TestCreateInferenceProvidersCuda()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(3, new[] { ExecutionProvider.Cuda });
        var options = inferenceProviders[0].Options!;

        Assert.Equal(2, options.Count);
        Assert.Equal(3, options["device_id"]);
        Assert.Equal("EXHAUSTIVE", options["cudnn_conv_algo_search"]);
    }

    [Fact]
    public void TestCreateInferenceProvidersTensorrt()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(2, new[] { ExecutionProvider.Tensorrt });
        var options = inferenceProviders[0].Options!;
        var cachePath = Execution.ResolveCachePath();

        Assert.Equal(2, options["device_id"]);
        // The cache directory always exists or is creatable under the repo working
        // directory in this environment, so every trt_* key is populated — matches Python,
        // which takes the same branch under the same condition.
        Assert.Equal(true, options["trt_engine_cache_enable"]);
        Assert.Equal(cachePath, options["trt_engine_cache_path"]);
        Assert.Equal(true, options["trt_timing_cache_enable"]);
        Assert.Equal(cachePath, options["trt_timing_cache_path"]);
        Assert.Equal(4, options["trt_builder_optimization_level"]);
    }

    [Theory]
    [InlineData(ExecutionProvider.Directml)]
    [InlineData(ExecutionProvider.Rocm)]
    public void TestCreateInferenceProvidersDirectmlAndRocm(ExecutionProvider executionProvider)
    {
        var inferenceProviders = Execution.CreateInferenceProviders(7, new[] { executionProvider });
        var options = inferenceProviders[0].Options!;

        Assert.Single(options);
        Assert.Equal(7, options["device_id"]);
    }

    [Fact]
    public void TestCreateInferenceProvidersMigraphx()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(5, new[] { ExecutionProvider.Migraphx });
        var options = inferenceProviders[0].Options!;

        Assert.Equal(5, options["device_id"]);
        Assert.Equal(Execution.ResolveCachePath(), options["migraphx_model_cache_dir"]);
    }

    [Fact]
    public void TestCreateInferenceProvidersCoreml()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Coreml });
        var options = inferenceProviders[0].Options!;

        Assert.Equal("FastPrediction", options["SpecializationStrategy"]);
        Assert.Equal(Execution.ResolveCachePath(), options["ModelCacheDirectory"]);
    }

    [Fact]
    public void TestCreateInferenceProvidersOpenvino()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Openvino });
        var options = inferenceProviders[0].Options!;

        Assert.Equal("GPU", options["device_type"]);
        Assert.Equal("FP32", options["precision"]);

        inferenceProviders = Execution.CreateInferenceProviders(2, new[] { ExecutionProvider.Openvino });
        options = inferenceProviders[0].Options!;

        Assert.Equal("GPU.2", options["device_type"]);
    }

    [Fact]
    public void TestCreateInferenceProvidersQnn()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Qnn });
        var options = inferenceProviders[0].Options!;

        Assert.Equal(0, options["device_id"]);
        Assert.Equal("htp", options["backend_type"]);
    }

    [Fact]
    public void TestCreateInferenceProvidersOrderMatchesInputWithCpuLast()
    {
        // Python: cpu is only appended after the main loop, regardless of where it sits in
        // the input list — see tests/test_execution.py::test_create_inference_providers.
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu, ExecutionProvider.Openvino, ExecutionProvider.Coreml });

        Assert.Equal(new[] { "OpenVINOExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider" }, inferenceProviders.Select(p => p.ProviderName));
    }

    [Fact]
    public void TestCreateInferenceProvidersWithoutCpuOmitsCpuEntry()
    {
        var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Openvino });

        Assert.DoesNotContain(inferenceProviders, p => p.ProviderName == "CPUExecutionProvider");
    }

    // --- resolve_openvino_device_type / resolve_cache_path --------------------------------

    [Fact]
    public void TestResolveOpenVinoDeviceType()
    {
        Assert.Equal("GPU", Execution.ResolveOpenVinoDeviceType(0));
        Assert.Equal("GPU.1", Execution.ResolveOpenVinoDeviceType(1));
        Assert.Equal("GPU.3", Execution.ResolveOpenVinoDeviceType(3));
    }

    [Fact]
    public void TestResolveCachePath()
    {
        var cachePath = Execution.ResolveCachePath();

        Assert.StartsWith(".caches", cachePath);
    }

    // --- get_onnxruntime_version -----------------------------------------------------------

    [Fact]
    public void TestParseOnnxRuntimeVersionPlain()
    {
        Assert.Equal((1, 29, 0), Execution.ParseOnnxRuntimeVersion("1.29.0"));
    }

    [Fact]
    public void TestParseOnnxRuntimeVersionWithCudaSuffix()
    {
        // A GPU onnxruntime package's __version__ can carry a build suffix, e.g. "1.20.1+cu124".
        Assert.Equal((1, 20, 1), Execution.ParseOnnxRuntimeVersion("1.20.1+cu124"));
        Assert.Equal((1, 24, 4), Execution.ParseOnnxRuntimeVersion("1.24.4+cuda12"));
    }

    [Fact]
    public void TestGetOnnxRuntimeVersionAgainstInstalledPackage()
    {
        var (major, minor, patch) = Execution.GetOnnxRuntimeVersion();

        // Microsoft.ML.OnnxRuntime 1.29.0 is what this project references.
        Assert.Equal((1, 29, 0), (major, minor, patch));
    }

    // --- nvidia-smi XML parsing (no real nvidia-smi in this environment) ------------------

    private const string CannedNvidiaSmiXml = """
        <?xml version="1.0" ?>
        <!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v11.dtd">
        <nvidia_smi_log>
            <timestamp>Tue Aug 18 12:00:00 2026</timestamp>
            <driver_version>535.129.03</driver_version>
            <cuda_version>12.2</cuda_version>
            <attached_gpus>2</attached_gpus>
            <gpu id="00000000:01:00.0">
                <product_name>NVIDIA GeForce RTX 3080</product_name>
                <product_brand>GeForce</product_brand>
                <fb_memory_usage>
                    <total>10240 MiB</total>
                    <reserved>203 MiB</reserved>
                    <used>1234 MiB</used>
                    <free>9006 MiB</free>
                </fb_memory_usage>
                <temperature>
                    <gpu_temp>45 C</gpu_temp>
                    <gpu_temp_max_threshold>98 C</gpu_temp_max_threshold>
                    <memory_temp>N/A</memory_temp>
                </temperature>
                <utilization>
                    <gpu_util>12 %</gpu_util>
                    <memory_util>5 %</memory_util>
                </utilization>
            </gpu>
            <gpu id="00000000:02:00.0">
                <product_name>NVIDIA GeForce GTX 1650</product_name>
                <product_brand>GeForce</product_brand>
                <fb_memory_usage>
                    <total>4096 MiB</total>
                    <reserved>64 MiB</reserved>
                    <used>512 MiB</used>
                    <free>3520 MiB</free>
                </fb_memory_usage>
                <temperature>
                    <gpu_temp>52 C</gpu_temp>
                    <gpu_temp_max_threshold>95 C</gpu_temp_max_threshold>
                    <memory_temp>N/A</memory_temp>
                </temperature>
                <utilization>
                    <gpu_util>0 %</gpu_util>
                    <memory_util>0 %</memory_util>
                </utilization>
            </gpu>
        </nvidia_smi_log>
        """;

    [Fact]
    public void TestParseExecutionDevicesXml()
    {
        var executionDevices = Execution.ParseExecutionDevicesXml(CannedNvidiaSmiXml);

        Assert.Equal(2, executionDevices.Count);

        var first = executionDevices[0];
        Assert.Equal("535.129.03", first.DriverVersion);
        Assert.Equal("CUDA", first.Framework.Name);
        Assert.Equal("12.2", first.Framework.Version);
        Assert.Equal("NVIDIA", first.Product.Vendor);
        // Python: gpu_element.findtext('product_name').replace('NVIDIA', '').strip()
        Assert.Equal("GeForce RTX 3080", first.Product.Name);
        Assert.Equal(10240, first.VideoMemory.Total!.Value);
        Assert.Equal("MiB", first.VideoMemory.Total!.Unit);
        Assert.Equal(9006, first.VideoMemory.Free!.Value);
        Assert.Equal(45, first.Temperature.Gpu!.Value);
        Assert.Equal("C", first.Temperature.Gpu!.Unit);
        // "N/A" has no space, so create_value_and_unit returns None.
        Assert.Null(first.Temperature.Memory);
        Assert.Equal(12, first.Utilization.Gpu!.Value);
        Assert.Equal(5, first.Utilization.Memory!.Value);

        var second = executionDevices[1];
        Assert.Equal("GeForce GTX 1650", second.Product.Name);
        Assert.Equal(4096, second.VideoMemory.Total!.Value);
    }

    [Fact]
    public void TestParseExecutionDevicesXmlWithNoGpuElementsReturnsEmpty()
    {
        var executionDevices = Execution.ParseExecutionDevicesXml("<nvidia_smi_log><driver_version>1.0</driver_version></nvidia_smi_log>");

        Assert.Empty(executionDevices);
    }

    [Fact]
    public void TestParseExecutionDevicesXmlWithMalformedXmlReturnsEmpty()
    {
        // Python: ElementTree.fromstring raising is caught by the same try/except as the
        // subprocess call, and treated as "no GPU found".
        var executionDevices = Execution.ParseExecutionDevicesXml("not xml at all <<<");

        Assert.Empty(executionDevices);
    }

    [Fact]
    public void TestParseExecutionDevicesXmlEmptyStringReturnsEmpty()
    {
        Assert.Empty(Execution.ParseExecutionDevicesXml(string.Empty));
    }

    // --- create_value_and_unit -------------------------------------------------------------

    [Fact]
    public void TestCreateValueAndUnit()
    {
        var valueAndUnit = Execution.CreateValueAndUnit("1234 MiB");

        Assert.NotNull(valueAndUnit);
        Assert.Equal(1234, valueAndUnit!.Value);
        Assert.Equal("MiB", valueAndUnit.Unit);
    }

    [Fact]
    public void TestCreateValueAndUnitWithoutSpaceReturnsNull()
    {
        Assert.Null(Execution.CreateValueAndUnit("N/A"));
        Assert.Null(Execution.CreateValueAndUnit(""));
    }

    // --- detect_execution_devices / detect_static_execution_devices (no nvidia-smi here) --

    [Fact]
    public void TestDetectExecutionDevicesReturnsEmptyWithoutNvidiaSmi()
    {
        // This container has no nvidia-smi binary, so RunNvidiaSmi() returns null and
        // DetectExecutionDevices() takes the "no GPU found" path — matches what Python's
        // detect_execution_devices() does when shutil.which('nvidia-smi') is None.
        Assert.Empty(Execution.DetectExecutionDevices());
        Assert.Empty(Execution.DetectStaticExecutionDevices());
    }

    [Fact]
    public void TestResolveCudnnConvAlgoSearchWithoutGpuIsExhaustive()
    {
        Assert.Equal("EXHAUSTIVE", Execution.ResolveCudnnConvAlgoSearch());
    }

    [Fact(Skip = "requires nvidia-smi / a real GPU")]
    public void TestRunNvidiaSmiAgainstRealHardware()
    {
        // Python: run_nvidia_smi() shells out to the real nvidia-smi binary. Not present in
        // this container (confirmed: ProcessHelper.Which("nvidia-smi") returns null here) —
        // ParseExecutionDevicesXml above covers the parsing logic against canned output
        // instead. This test documents the coverage gap rather than silently dropping it.
    }
}
