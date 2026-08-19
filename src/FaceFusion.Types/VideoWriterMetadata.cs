namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VideoWriterMetadata = TypedDict('VideoWriterMetadata', { 'fps': Fps, 'resolution': Resolution })</c>.
/// </summary>
public sealed record VideoWriterMetadata(double Fps, Resolution Resolution);
