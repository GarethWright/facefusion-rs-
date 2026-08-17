namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>Gender = Literal['female', 'male']</c>.
/// </summary>
public enum Gender
{
	[WireName("female")]
	Female,

	[WireName("male")]
	Male
}
