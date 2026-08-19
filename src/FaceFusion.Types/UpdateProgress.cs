namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>UpdateProgress : TypeAlias = Callable[[int], None]</c>.
/// </summary>
public delegate void UpdateProgress(int value);
