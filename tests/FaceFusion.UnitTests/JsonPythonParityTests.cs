using System.Text.Json;
using FaceFusion.Core;

namespace FaceFusion.UnitTests;

/// <summary>
/// Pins <see cref="Json.WriteJson"/> against the exact bytes Python produces, because
/// job files are written and read by both implementations during the transition
/// (docs/DOTNET_PORT_PLAN.md §9.3).
///
/// Ground truth generated with:
///   python3 -c "import json; print(json.dumps(job, indent=4))"
/// </summary>
public sealed class JsonPythonParityTests
{
    /// <summary>
    /// A job as <c>job_manager.create_job</c> builds it, including the null
    /// <c>date_updated</c> that a WhenWritingNull serialiser would silently drop.
    /// </summary>
    private const string ExpectedJobJson = """
        {
            "version": "1",
            "date_created": "2026-08-17T12:00:00",
            "date_updated": null,
            "steps": [
                {
                    "args": {
                        "source_path": "a.jpg"
                    },
                    "status": "drafted"
                }
            ]
        }
        """;

    [Fact]
    public void WriteJson_MatchesPythonJsonDumpIndentFour()
    {
        var job = new Dictionary<string, object?>
        {
            ["version"] = "1",
            ["date_created"] = "2026-08-17T12:00:00",
            ["date_updated"] = null,
            ["steps"] = new object[]
            {
                new Dictionary<string, object?>
                {
                    ["args"] = new Dictionary<string, object?> { ["source_path"] = "a.jpg" },
                    ["status"] = "drafted"
                }
            }
        };

        var tempPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".json");

        try
        {
            Assert.True(Json.WriteJson(tempPath, job));

            var written = File.ReadAllText(tempPath).Replace("\r\n", "\n");

            Assert.Equal(ExpectedJobJson.Replace("\r\n", "\n"), written);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    /// <summary>
    /// A null-valued key must survive the round trip as an explicit null rather than
    /// vanishing — Python reads job['date_updated'] and would KeyError on a missing key.
    /// </summary>
    [Fact]
    public void WriteJson_KeepsNullValuedKeys()
    {
        var content = new Dictionary<string, object?> { ["present"] = "yes", ["absent"] = null };
        var tempPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".json");

        try
        {
            Assert.True(Json.WriteJson(tempPath, content));

            var element = Json.ReadJson(tempPath);

            Assert.NotNull(element);
            Assert.True(element!.Value.TryGetProperty("absent", out var absent));
            Assert.Equal(JsonValueKind.Null, absent.ValueKind);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    /// <summary>
    /// Indentation is widened after serialisation; verify a string containing an escaped
    /// newline and leading spaces is not corrupted by that pass.
    /// </summary>
    [Fact]
    public void ExpandIndentation_DoesNotTouchStringContents()
    {
        var content = new Dictionary<string, object?> { ["text"] = "line1\n    line2" };

        var serialised = JsonSerializer.Serialize(content, new JsonSerializerOptions { WriteIndented = true });
        var expanded = Json.ExpandIndentation(serialised);

        // The escaped newline stays escaped, so its following spaces are inside the
        // string literal and must not be doubled.
        Assert.Contains(@"line1\n    line2", expanded, StringComparison.Ordinal);
    }

    /// <summary>Python's read_json returns None for a missing or null path.</summary>
    [Fact]
    public void ReadJson_ReturnsNullForMissingOrNullPath()
    {
        Assert.Null(Json.ReadJson(null));
        Assert.Null(Json.ReadJson("does-not-exist.json"));
    }
}
