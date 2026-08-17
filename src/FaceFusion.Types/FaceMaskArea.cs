namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceMaskArea = Literal['upper-face', 'lower-face', 'mouth']</c>.
/// </summary>
public enum FaceMaskArea
{
	[WireName("upper-face")]
	UpperFace,

	[WireName("lower-face")]
	LowerFace,

	[WireName("mouth")]
	Mouth
}
