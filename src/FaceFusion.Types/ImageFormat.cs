namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ImageFormat = Literal['bmp', 'jpeg', 'png', 'tiff', 'webp']</c>.
/// </summary>
public enum ImageFormat
{
	[WireName("bmp")]
	Bmp,

	[WireName("jpeg")]
	Jpeg,

	[WireName("png")]
	Png,

	[WireName("tiff")]
	Tiff,

	[WireName("webp")]
	Webp
}
