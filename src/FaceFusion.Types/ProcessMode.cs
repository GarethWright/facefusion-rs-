namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ProcessMode = Literal['output', 'preview', 'stream']</c>.
/// </summary>
public enum ProcessMode
{
	[WireName("output")]
	Output,

	[WireName("preview")]
	Preview,

	[WireName("stream")]
	Stream
}
