namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>TempFrameFormat = Literal['bmp', 'jpeg', 'png', 'tiff']</c>.
/// </summary>
public enum TempFrameFormat
{
	[WireName("bmp")]
	Bmp,

	[WireName("jpeg")]
	Jpeg,

	[WireName("png")]
	Png,

	[WireName("tiff")]
	Tiff
}
