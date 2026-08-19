namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>WebcamMode = Literal['inline', 'udp', 'v4l2']</c>.
/// </summary>
public enum WebcamMode
{
	[WireName("inline")]
	Inline,

	[WireName("udp")]
	Udp,

	[WireName("v4l2")]
	V4l2
}
