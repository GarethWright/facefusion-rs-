namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionDeviceUtilization = TypedDict('ExecutionDeviceUtilization', { 'gpu': Optional[ValueAndUnit], 'memory': Optional[ValueAndUnit] })</c>.
/// </summary>
public sealed record ExecutionDeviceUtilization(ValueAndUnit? Gpu, ValueAndUnit? Memory);
