namespace FaceFusion.Core;

/// <summary>
/// Port of Python's <c>shutil.which</c> as used throughout <c>facefusion</c> (imported
/// directly from the standard library, not a project module).
/// </summary>
public static class ProcessHelper
{
    /// <summary>
    /// Locates an executable on <c>PATH</c>, mirroring Python's <c>shutil.which(cmd)</c>
    /// (called with the default <c>mode</c> and <c>path</c> arguments). Returns null when
    /// the executable cannot be found, matching Python's behaviour, so that callers can
    /// reproduce facefusion's deliberate <c>None</c>-in-command-list oddity rather than
    /// throwing.
    /// </summary>
    public static string? Which(string executable)
    {
        // shutil.which: if the command contains a path separator, it is checked directly
        // rather than being searched for on PATH.
        var hasPathSeparator = executable.Contains(Path.DirectorySeparatorChar)
            || executable.Contains(Path.AltDirectorySeparatorChar);

        if (hasPathSeparator)
        {
            return IsExecutableFile(executable) ? executable : null;
        }

        var pathVariable = Environment.GetEnvironmentVariable("PATH");

        if (string.IsNullOrEmpty(pathVariable))
        {
            return null;
        }

        // shutil.which appends PATHEXT candidates on Windows (e.g. ".EXE", ".CMD"); on
        // POSIX the executable name is used as-is.
        var candidateNames = BuildCandidateNames(executable);

        foreach (var directory in pathVariable.Split(Path.PathSeparator))
        {
            if (string.IsNullOrEmpty(directory))
            {
                continue;
            }

            foreach (var candidateName in candidateNames)
            {
                var candidatePath = Path.Combine(directory, candidateName);

                if (IsExecutableFile(candidatePath))
                {
                    return candidatePath;
                }
            }
        }

        return null;
    }

    private static IReadOnlyList<string> BuildCandidateNames(string executable)
    {
        if (!OperatingSystem.IsWindows())
        {
            return new[] { executable };
        }

        // Mirrors shutil.which: if the name already ends with one of PATHEXT's
        // extensions, only that exact name is tried; otherwise every PATHEXT suffix is
        // tried in order, falling back to the bare name.
        var pathExt = Environment.GetEnvironmentVariable("PATHEXT");
        var extensions = string.IsNullOrEmpty(pathExt)
            ? new[] { ".COM", ".EXE", ".BAT", ".CMD" }
            : pathExt.Split(Path.PathSeparator);

        foreach (var extension in extensions)
        {
            if (executable.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
            {
                return new[] { executable };
            }
        }

        var candidates = new List<string>(extensions.Length + 1) { executable };
        candidates.AddRange(extensions.Select(extension => executable + extension));
        return candidates;
    }

    private static bool IsExecutableFile(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        if (!OperatingSystem.IsWindows())
        {
            // POSIX shutil.which additionally requires os.X_OK; .NET has no direct
            // equivalent of os.access, so approximate it via the unix file mode bits.
            var mode = File.GetUnixFileMode(path);
            const UnixFileMode executeBits = UnixFileMode.UserExecute | UnixFileMode.GroupExecute | UnixFileMode.OtherExecute;
            return (mode & executeBits) != 0;
        }

        return true;
    }
}
