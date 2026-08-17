namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>Padding : TypeAlias = Tuple[int, int, int, int]</c>,
/// a (top, right, bottom, left) tuple as produced by <c>normalizer.normalize_space</c> — see
/// face_masker.py's use of <c>face_mask_padding[0..3]</c> in that order. A real struct so it
/// cannot be confused with <see cref="Color"/> or <see cref="Margin"/>, which are the same
/// shape but a different meaning.
/// </summary>
public readonly record struct Padding(int Top, int Right, int Bottom, int Left);
