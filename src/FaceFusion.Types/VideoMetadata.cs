namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>VideoMetadata</c> TypedDict.
/// Also stands in for <c>VideoReaderMetadata</c>, which Python defines as a bare alias of
/// this same TypedDict (<c>VideoReaderMetadata : TypeAlias = VideoMetadata</c>).
/// </summary>
public sealed record VideoMetadata(
	double Duration,
	int FrameTotal,
	double Fps,
	Resolution Resolution,
	int BitRate,
	string ColorTransfer);
