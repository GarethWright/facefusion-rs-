namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>Resolution : TypeAlias = Tuple[int, int]</c>.
/// A real struct (rather than a bare tuple) because a raw <c>(int, int)</c> invites mixing up
/// width/height with other 2-int pairs in the codebase.
/// </summary>
public readonly record struct Resolution(int Width, int Height);
