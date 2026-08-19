namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionDeviceFramework = TypedDict('ExecutionDeviceFramework', { 'name': str, 'version': str })</c>.
/// </summary>
public sealed record ExecutionDeviceFramework(string Name, string Version);
