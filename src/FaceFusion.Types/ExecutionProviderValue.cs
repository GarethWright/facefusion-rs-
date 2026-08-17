namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionProviderValue = Literal['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider', 'QNNExecutionProvider', 'ROCMExecutionProvider', 'TensorrtExecutionProvider']</c>.
/// These are the literal ONNX Runtime provider names, distinct from <see cref="ExecutionProvider"/>
/// (FaceFusion's own short names). <see cref="Choices.ExecutionProviderSet"/> maps one to the other.
/// </summary>
public enum ExecutionProviderValue
{
	[WireName("CPUExecutionProvider")]
	CpuExecutionProvider,

	[WireName("CoreMLExecutionProvider")]
	CoreMlExecutionProvider,

	[WireName("CUDAExecutionProvider")]
	CudaExecutionProvider,

	[WireName("DmlExecutionProvider")]
	DmlExecutionProvider,

	[WireName("OpenVINOExecutionProvider")]
	OpenVinoExecutionProvider,

	[WireName("MIGraphXExecutionProvider")]
	MiGraphXExecutionProvider,

	[WireName("QNNExecutionProvider")]
	QnnExecutionProvider,

	[WireName("ROCMExecutionProvider")]
	RocmExecutionProvider,

	[WireName("TensorrtExecutionProvider")]
	TensorrtExecutionProvider
}
