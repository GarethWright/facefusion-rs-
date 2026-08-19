namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VideoFormat = Literal['avi', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mxf', 'webm', 'wmv']</c>.
/// </summary>
public enum VideoFormat
{
	[WireName("avi")]
	Avi,

	[WireName("m4v")]
	M4v,

	[WireName("mkv")]
	Mkv,

	[WireName("mov")]
	Mov,

	[WireName("mp4")]
	Mp4,

	[WireName("mpeg")]
	Mpeg,

	[WireName("mxf")]
	Mxf,

	[WireName("webm")]
	Webm,

	[WireName("wmv")]
	Wmv
}
