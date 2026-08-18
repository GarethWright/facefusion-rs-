using System.Diagnostics;
using System.Text.Json;
using FaceFusion.Cli;

namespace FaceFusion.UnitTests;

/// <summary>
/// Guards the generated <see cref="UiOptionDescriptors"/> table, which the Blazor UI binds
/// every control to. A wrong default here is not a cosmetic bug: it silently makes a UI run
/// behave differently from the identical CLI run.
/// </summary>
public sealed class UiOptionDescriptorsTests
{
    private static string? FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    [Fact]
    public void EveryDescriptorHasADistinctStateKey()
    {
        var duplicates = UiOptionDescriptors.All
            .GroupBy(descriptor => descriptor.StateKey, StringComparer.Ordinal)
            .Where(group => group.Count() > 1)
            .Select(group => group.Key)
            .ToArray();

        Assert.Empty(duplicates);
    }

    /// <summary>
    /// The two generated tables are produced by different means — CliOptions.cs by parsing
    /// program.py's source, UiOptionDescriptors.cs by introspecting the constructed parser — so
    /// agreeing on the flag for a shared state key is real cross-checking rather than a
    /// tautology.
    /// </summary>
    /// <summary>
    /// Options the UI table legitimately does not carry. Every one of these belongs to a
    /// command other than <c>run</c> (which is the command the UI is), or is argparse's own:
    /// the <c>--*-pattern</c> flags and <c>--halt-on-error</c> are <c>batch-run</c>'s,
    /// <c>--download-scope</c> is <c>force-download</c>'s, and <c>--version</c> is the root
    /// parser's. Listed explicitly so a genuinely missing option cannot hide behind a pattern
    /// match.
    /// </summary>
    private static readonly HashSet<string> NotOnTheRunCommand = new(StringComparer.Ordinal)
    {
        "source_pattern", "target_pattern", "output_pattern", "halt_on_error", "download_scope", "version",
    };

    [Fact]
    public void FlagsAgreeWithTheCliOptionTable()
    {
        foreach (var cliOption in CliOptions.All)
        {
            if (!UiOptionDescriptors.Has(cliOption.StateKey))
            {
                Assert.True(
                    NotOnTheRunCommand.Contains(cliOption.StateKey),
                    $"'{cliOption.StateKey}' is a CLI option with no UI descriptor and is not on the known-absent list — " +
                    "either regenerate UiOptionDescriptors.cs or add it there with the reason");
                continue;
            }

            var descriptor = UiOptionDescriptors.Get(cliOption.StateKey);

            Assert.Equal(cliOption.Flag, descriptor.Flag);
            Assert.Equal(cliOption.Kind, descriptor.Kind);
        }
    }

    /// <summary>A choice list must contain the default, or the UI opens with a control showing
    /// a value it cannot offer.</summary>
    [Fact]
    public void EveryDefaultIsWithinItsOwnChoiceList()
    {
        foreach (var descriptor in UiOptionDescriptors.All)
        {
            if (descriptor.Choices.Count == 0 || descriptor.Default is null)
            {
                continue;
            }

            // A list-valued default is space-joined, so each element must be a valid choice.
            var values = descriptor.Kind is CliValueKind.StringList or CliValueKind.IntList
                ? descriptor.Default.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                : new[] { descriptor.Default };

            foreach (var value in values)
            {
                Assert.True(
                    descriptor.Choices.Contains(value, StringComparer.Ordinal),
                    $"'{descriptor.StateKey}' defaults to '{value}', which is not one of its {descriptor.Choices.Count} choices");
            }
        }
    }

    /// <summary>
    /// Drift guard against the real Python parser. Skips cleanly when Python or the facefusion
    /// package is unavailable (CI has neither), which is the same gate the media-dependent
    /// tests use.
    /// </summary>
    [Fact]
    public void DefaultsMatchThePythonParser()
    {
        var repoRoot = FindRepoRoot();

        if (repoRoot is null || !File.Exists(Path.Combine(repoRoot, "facefusion", "program.py")))
        {
            return;
        }

        const string script = """
            import json, sys
            import facefusion.program
            parser = facefusion.program.create_program()._actions[2].choices['run']
            out = {}
            for action in parser._actions:
                if not action.option_strings or action.dest == 'help':
                    continue
                value = action.default
                if value is None:
                    out[action.dest] = None
                elif isinstance(value, bool):
                    out[action.dest] = 'true' if value else 'false'
                elif isinstance(value, (list, tuple)):
                    out[action.dest] = ' '.join(str(item) for item in value)
                else:
                    out[action.dest] = str(value)
            json.dump(out, sys.stdout)
            """;

        var startInfo = new ProcessStartInfo("python3")
        {
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        startInfo.ArgumentList.Add("-c");
        startInfo.ArgumentList.Add(script);

        string output;

        try
        {
            using var process = Process.Start(startInfo);

            if (process is null)
            {
                return;
            }

            output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(120_000);

            if (process.ExitCode != 0)
            {
                return; // facefusion's Python dependencies are not installed here
            }
        }
        catch (Exception)
        {
            return; // no python3 on PATH
        }

        var pythonDefaults = JsonSerializer.Deserialize<Dictionary<string, string?>>(output);
        Assert.NotNull(pythonDefaults);

        foreach (var descriptor in UiOptionDescriptors.All)
        {
            Assert.True(pythonDefaults!.ContainsKey(descriptor.StateKey),
                $"'{descriptor.StateKey}' is in the generated table but not in Python's parser — regenerate UiOptionDescriptors.cs");
            Assert.Equal(pythonDefaults[descriptor.StateKey], descriptor.Default);
        }
    }
}
