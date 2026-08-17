namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>Color : TypeAlias = Tuple[int, int, int, int]</c>,
/// an (R, G, B, A) tuple as produced by <c>normalizer.normalize_color</c>. A real struct
/// (rather than a bare tuple) so it cannot be confused with <see cref="Padding"/> or
/// <see cref="Margin"/>, which are the same shape but a different meaning.
/// </summary>
public readonly record struct Color(int Red, int Green, int Blue, int Alpha);
