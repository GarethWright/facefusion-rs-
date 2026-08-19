namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>VideoWriter</c> TypedDict.
/// <c>Process</c> stands in for Python's <c>subprocess.Popen[bytes]</c> — see the note on
/// <see cref="VideoReader"/>.
/// </summary>
public sealed record VideoWriter(
	string Id,
	string FilePath,
	object Process,
	VideoWriterMetadata Metadata);
