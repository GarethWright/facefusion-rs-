namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceOccluderModel = Literal['many', 'xseg_1', 'xseg_2', 'xseg_3']</c>.
/// </summary>
public enum FaceOccluderModel
{
	[WireName("many")]
	Many,

	[WireName("xseg_1")]
	Xseg1,

	[WireName("xseg_2")]
	Xseg2,

	[WireName("xseg_3")]
	Xseg3
}
