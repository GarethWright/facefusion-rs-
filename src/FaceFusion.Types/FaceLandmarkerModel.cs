namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceLandmarkerModel = Literal['many', '2dfan4', 'peppa_wutz']</c>.
/// The wire name '2dfan4' cannot be a C# identifier, so the member is named TwoDFan4.
/// </summary>
public enum FaceLandmarkerModel
{
	[WireName("many")]
	Many,

	[WireName("2dfan4")]
	TwoDFan4,

	[WireName("peppa_wutz")]
	PeppaWutz
}
