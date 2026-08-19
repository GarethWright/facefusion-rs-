namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>FaceLandmarkSet</c> TypedDict, keyed by the
/// literal string keys '5', '5/68', '68', '68/5' (which are not legal C# identifiers, hence
/// the renamed properties below). Each value is a <c>FaceLandmark5</c>/<c>FaceLandmark68</c>
/// (<c>NDArray[Any]</c> in Python) — represented as <c>object</c> here since FaceFusion.Types
/// has no tensor dependency; FaceFusion.Tensors supplies the concrete landmark array type.
/// </summary>
public sealed record FaceLandmarkSet(object Five, object FiveOn68, object SixtyEight, object SixtyEightOn5);
