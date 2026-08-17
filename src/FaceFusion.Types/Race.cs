namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>Race = Literal['white', 'black', 'latino', 'asian', 'indian', 'arabic']</c>.
/// </summary>
public enum Race
{
	[WireName("white")]
	White,

	[WireName("black")]
	Black,

	[WireName("latino")]
	Latino,

	[WireName("asian")]
	Asian,

	[WireName("indian")]
	Indian,

	[WireName("arabic")]
	Arabic
}
