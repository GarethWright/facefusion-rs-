namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionDeviceVideoMemory = TypedDict('ExecutionDeviceVideoMemory', { 'total': Optional[ValueAndUnit], 'free': Optional[ValueAndUnit] })</c>.
/// </summary>
public sealed record ExecutionDeviceVideoMemory(ValueAndUnit? Total, ValueAndUnit? Free);
