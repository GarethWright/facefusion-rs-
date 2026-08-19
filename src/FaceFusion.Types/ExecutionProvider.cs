namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionProvider = Literal['cuda', 'tensorrt', 'rocm', 'migraphx', 'coreml', 'openvino', 'qnn', 'directml', 'cpu']</c>.
/// </summary>
public enum ExecutionProvider
{
	[WireName("cuda")]
	Cuda,

	[WireName("tensorrt")]
	Tensorrt,

	[WireName("rocm")]
	Rocm,

	[WireName("migraphx")]
	Migraphx,

	[WireName("coreml")]
	Coreml,

	[WireName("openvino")]
	Openvino,

	[WireName("qnn")]
	Qnn,

	[WireName("directml")]
	Directml,

	[WireName("cpu")]
	Cpu
}
