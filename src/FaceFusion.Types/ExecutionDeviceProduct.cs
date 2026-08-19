namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ExecutionDeviceProduct = TypedDict('ExecutionDeviceProduct', { 'vendor': str, 'name': str })</c>.
/// </summary>
public sealed record ExecutionDeviceProduct(string Vendor, string Name);
