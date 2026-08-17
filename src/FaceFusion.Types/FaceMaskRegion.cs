namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceMaskRegion = Literal['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']</c>.
/// </summary>
public enum FaceMaskRegion
{
	[WireName("skin")]
	Skin,

	[WireName("left-eyebrow")]
	LeftEyebrow,

	[WireName("right-eyebrow")]
	RightEyebrow,

	[WireName("left-eye")]
	LeftEye,

	[WireName("right-eye")]
	RightEye,

	[WireName("glasses")]
	Glasses,

	[WireName("nose")]
	Nose,

	[WireName("mouth")]
	Mouth,

	[WireName("upper-lip")]
	UpperLip,

	[WireName("lower-lip")]
	LowerLip
}
