namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ApplyStateItem : TypeAlias = Callable[[Any, Any], None]</c>.
/// </summary>
public delegate void ApplyStateItem(object? target, object? value);
