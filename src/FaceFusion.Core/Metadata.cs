namespace FaceFusion.Core;

/// <summary>
/// Port of <c>facefusion/metadata.py</c>. Values are transcribed rather than read from the
/// assembly's own attributes so they stay identical to the Python's — the version string in
/// particular appears in the UI header and in <c>--version</c>, and is the version of the
/// upstream application this port tracks, not of the port's assemblies.
/// </summary>
public static class Metadata
{
    private static readonly Dictionary<string, string> Values = new(StringComparer.Ordinal)
    {
        { "name", "FaceFusion" },
        { "description", "Industry leading face manipulation platform" },
        { "version", "3.8.2" },
        { "license", "OpenRAIL-AS" },
        { "author", "Henry Ruhs" },
        { "url", "https://facefusion.io" },
    };

    /// <summary>Python: <c>get(key)</c> — returns null for an unknown key, matching
    /// <c>dict.get</c>.</summary>
    public static string? Get(string key) => Values.TryGetValue(key, out var value) ? value : null;
}
