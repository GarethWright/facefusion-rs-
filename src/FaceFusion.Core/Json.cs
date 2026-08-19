using System.Text.Encodings.Web;
using System.Text.Json;

namespace FaceFusion.Core;

/// <summary>
/// Port of <c>facefusion/json.py</c>.
///
/// Output must stay interchangeable with the Python implementation: job files written
/// by either side are read by the other (see docs/DOTNET_PORT_PLAN.md §9.3), so this
/// reproduces <c>json.dump(content, file, indent = 4)</c> rather than merely producing
/// valid JSON.
/// </summary>
public static class Json
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        // System.Text.Json's default encoder escapes '+', '<', '>' and '&' for HTML
        // safety, so a job timestamp came out as "...\u002B00:00" where Python writes
        // "...+00:00". Every job file carries a timestamp, so the default encoder made
        // every job file differ from Python's. "Unsafe" here means unsafe to drop into
        // HTML unescaped; job JSON is never embedded in a page.
        //
        // Residual, deliberate difference: Python's json.dump defaults to
        // ensure_ascii=True and escapes non-ASCII as lowercase \u00e9, while this writes
        // the character raw as UTF-8. Both decode to the same string, so files still
        // round-trip between the implementations; only a byte comparison of a path
        // containing non-ASCII would differ.
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        // Deliberately NOT JsonIgnoreCondition.WhenWritingNull: Python's json.dump
        // emits null-valued keys, and job files rely on it — job_manager.create_job
        // writes "date_updated": null. Dropping the key changes the file's shape.
        PropertyNamingPolicy = null,
        PropertyNameCaseInsensitive = false
    };

    /// <summary>
    /// Python: <c>read_json</c>. Returns null when the file is missing or unparseable.
    /// </summary>
    public static JsonElement? ReadJson(string? jsonPath)
    {
        // Python guards with is_file(json_path), which is False for None, so a null
        // path returns None rather than raising.
        if (!FileSystem.IsFile(jsonPath))
        {
            return null;
        }

        try
        {
            var content = File.ReadAllText(jsonPath!);
            return JsonSerializer.Deserialize<JsonElement>(content);
        }
        catch (JsonException)
        {
            return null;
        }
        catch (IOException)
        {
            return null;
        }
    }

    /// <summary>
    /// Python: <c>write_json</c>. Writes with four-space indentation and '\n' line
    /// endings, matching <c>json.dump(..., indent = 4)</c>.
    /// </summary>
    public static bool WriteJson(string jsonPath, object content)
    {
        ArgumentNullException.ThrowIfNull(jsonPath);
        ArgumentNullException.ThrowIfNull(content);

        try
        {
            var jsonString = JsonSerializer.Serialize(content, JsonOptions);
            File.WriteAllText(jsonPath, ExpandIndentation(jsonString));
            return FileSystem.IsFile(jsonPath);
        }
        catch (JsonException)
        {
            return false;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Converts System.Text.Json's two-space indentation to Python's four-space
    /// indentation, and normalises line endings to '\n'.
    ///
    /// .NET 8 has no JsonSerializerOptions.IndentSize (added in .NET 9), so the
    /// indentation is widened after serialisation. This is safe rather than a hack:
    /// System.Text.Json escapes control characters inside string values, so a raw
    /// newline never appears within a JSON string and every physical line's leading
    /// spaces are therefore structural indentation.
    ///
    /// TODO(net9): replace with JsonSerializerOptions.IndentSize = 4 once the target
    /// framework moves past net8.0.
    /// </summary>
    public static string ExpandIndentation(string json)
    {
        var lines = json.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');

        for (var index = 0; index < lines.Length; index++)
        {
            var line = lines[index];
            var spaceCount = 0;

            while (spaceCount < line.Length && line[spaceCount] == ' ')
            {
                spaceCount++;
            }

            if (spaceCount > 0)
            {
                lines[index] = new string(' ', spaceCount * 2) + line[spaceCount..];
            }
        }

        return string.Join('\n', lines);
    }
}
