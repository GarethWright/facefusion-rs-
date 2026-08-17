namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>DownloadScope = Literal['lite', 'full']</c>.
/// </summary>
public enum DownloadScope
{
	[WireName("lite")]
	Lite,

	[WireName("full")]
	Full
}
