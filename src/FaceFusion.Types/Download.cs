namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>Download = TypedDict('Download', { 'url': str, 'path': str })</c>.
/// </summary>
public sealed record Download(string Url, string Path);
