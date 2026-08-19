namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>StreamMode = Literal['udp', 'v4l2']</c>.
/// </summary>
public enum StreamMode
{
	[WireName("udp")]
	Udp,

	[WireName("v4l2")]
	V4l2
}
