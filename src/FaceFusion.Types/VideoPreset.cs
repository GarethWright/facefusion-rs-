namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VideoPreset = Literal['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']</c>.
/// </summary>
public enum VideoPreset
{
	[WireName("ultrafast")]
	Ultrafast,

	[WireName("superfast")]
	Superfast,

	[WireName("veryfast")]
	Veryfast,

	[WireName("faster")]
	Faster,

	[WireName("fast")]
	Fast,

	[WireName("medium")]
	Medium,

	[WireName("slow")]
	Slow,

	[WireName("slower")]
	Slower,

	[WireName("veryslow")]
	Veryslow
}
