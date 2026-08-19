namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ColorSpace = Literal['bt601', 'bt709', 'bt2020']</c>.
/// </summary>
public enum ColorSpace
{
	[WireName("bt601")]
	Bt601,

	[WireName("bt709")]
	Bt709,

	[WireName("bt2020")]
	Bt2020
}
