namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>ExecutionDevice</c> TypedDict.
/// </summary>
public sealed record ExecutionDevice(
	string DriverVersion,
	ExecutionDeviceFramework Framework,
	ExecutionDeviceProduct Product,
	ExecutionDeviceVideoMemory VideoMemory,
	ExecutionDeviceTemperature Temperature,
	ExecutionDeviceUtilization Utilization);
