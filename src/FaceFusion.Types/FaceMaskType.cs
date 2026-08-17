namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceMaskType = Literal['box', 'occlusion', 'area', 'region']</c>.
/// </summary>
public enum FaceMaskType
{
	[WireName("box")]
	Box,

	[WireName("occlusion")]
	Occlusion,

	[WireName("area")]
	Area,

	[WireName("region")]
	Region
}
