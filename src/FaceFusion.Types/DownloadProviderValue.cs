using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>DownloadProviderValue = TypedDict('DownloadProviderValue', { 'urls': List[str], 'path': str })</c>.
/// </summary>
public sealed record DownloadProviderValue(IReadOnlyList<string> Urls, string Path);
