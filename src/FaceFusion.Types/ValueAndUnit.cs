namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ValueAndUnit = TypedDict('ValueAndUnit', { 'value': int, 'unit': str })</c>.
/// </summary>
public sealed record ValueAndUnit(int Value, string Unit);
