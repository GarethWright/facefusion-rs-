namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceSelectorGender = Literal['auto', 'female', 'male']</c>.
/// </summary>
public enum FaceSelectorGender
{
	[WireName("auto")]
	Auto,

	[WireName("female")]
	Female,

	[WireName("male")]
	Male
}
