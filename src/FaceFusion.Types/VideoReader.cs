namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>VideoReader</c> TypedDict.
/// <c>Process</c> stands in for Python's <c>subprocess.Popen[bytes]</c> (the ffmpeg process) —
/// FaceFusion.Types has no dependency on process-management types, so this is <c>object</c>
/// until FaceFusion.Media (which owns process/ffmpeg plumbing) supplies a concrete handle.
/// </summary>
public sealed record VideoReader(
	string Id,
	string FilePath,
	object Process,
	VideoMetadata Metadata,
	int FrameNumber);
