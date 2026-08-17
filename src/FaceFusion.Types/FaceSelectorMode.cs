namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceSelectorMode = Literal['many', 'one', 'reference']</c>.
/// </summary>
public enum FaceSelectorMode
{
	[WireName("many")]
	Many,

	[WireName("one")]
	One,

	[WireName("reference")]
	Reference
}
