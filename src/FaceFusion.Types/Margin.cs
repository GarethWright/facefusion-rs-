namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>Margin : TypeAlias = Tuple[int, int, int, int]</c>,
/// a (top, right, bottom, left) tuple — see face_detector.py's use of
/// <c>face_detector_margin[0..3]</c> in that order. A real struct so it cannot be confused
/// with <see cref="Color"/> or <see cref="Padding"/>, which are the same shape but a
/// different meaning.
/// </summary>
public readonly record struct Margin(int Top, int Right, int Bottom, int Left);
