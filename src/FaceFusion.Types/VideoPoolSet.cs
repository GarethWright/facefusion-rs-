using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VideoPoolSet = TypedDict('VideoPoolSet', { 'reader': VideoReaderSet, 'writer': VideoWriterSet })</c>,
/// where <c>VideoReaderSet : TypeAlias = Dict[str, VideoReader]</c> and
/// <c>VideoWriterSet : TypeAlias = Dict[str, VideoWriter]</c>.
/// </summary>
public sealed record VideoPoolSet(
	IReadOnlyDictionary<string, VideoReader> Reader,
	IReadOnlyDictionary<string, VideoWriter> Writer);
