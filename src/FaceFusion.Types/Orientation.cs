namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>Orientation = Literal['landscape', 'portrait']</c>.
/// </summary>
public enum Orientation
{
	[WireName("landscape")]
	Landscape,

	[WireName("portrait")]
	Portrait
}
