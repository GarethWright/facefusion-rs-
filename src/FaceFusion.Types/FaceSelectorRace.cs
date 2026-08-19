namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceSelectorRace = Literal['auto', 'white', 'black', 'latino', 'asian', 'indian', 'arabic']</c>.
/// </summary>
public enum FaceSelectorRace
{
	[WireName("auto")]
	Auto,

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
