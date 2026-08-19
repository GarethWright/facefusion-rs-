namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>TempPixelFormat = Literal['bgr24', 'bgra']</c>.
/// </summary>
public enum TempPixelFormat
{
	[WireName("bgr24")]
	Bgr24,

	[WireName("bgra")]
	Bgra
}
