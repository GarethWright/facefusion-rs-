namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>ColorMode = Literal['rgb', 'rgba']</c>.
/// </summary>
public enum ColorMode
{
	[WireName("rgb")]
	Rgb,

	[WireName("rgba")]
	Rgba
}
