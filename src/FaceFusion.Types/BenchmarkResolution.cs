namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>BenchmarkResolution = Literal['240p', '360p', '540p', '720p', '1080p', '1440p', '2160p']</c>.
/// The wire names start with a digit, which is not a legal C# identifier, so members are
/// named with a leading "R" (Resolution).
/// </summary>
public enum BenchmarkResolution
{
	[WireName("240p")]
	R240p,

	[WireName("360p")]
	R360p,

	[WireName("540p")]
	R540p,

	[WireName("720p")]
	R720p,

	[WireName("1080p")]
	R1080p,

	[WireName("1440p")]
	R1440p,

	[WireName("2160p")]
	R2160p
}
