namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>BenchmarkMode = Literal['warm', 'cold']</c>.
/// </summary>
public enum BenchmarkMode
{
	[WireName("warm")]
	Warm,

	[WireName("cold")]
	Cold
}
