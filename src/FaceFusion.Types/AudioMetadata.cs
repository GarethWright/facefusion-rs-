namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>AudioMetadata</c> TypedDict.
/// </summary>
public sealed record AudioMetadata(
	double Duration,
	int FrameTotal,
	int ChannelTotal,
	int SampleRate,
	int BitRate);
