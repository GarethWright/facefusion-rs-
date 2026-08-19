namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionDeviceTemperature = TypedDict('ExecutionDeviceTemperature', { 'gpu': Optional[ValueAndUnit], 'memory': Optional[ValueAndUnit] })</c>.
/// </summary>
public sealed record ExecutionDeviceTemperature(ValueAndUnit? Gpu, ValueAndUnit? Memory);
