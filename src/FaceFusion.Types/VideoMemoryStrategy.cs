namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VideoMemoryStrategy = Literal['strict', 'moderate', 'tolerant']</c>.
/// </summary>
public enum VideoMemoryStrategy
{
	[WireName("strict")]
	Strict,

	[WireName("moderate")]
	Moderate,

	[WireName("tolerant")]
	Tolerant
}
