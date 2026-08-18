using FaceFusion.Types;

namespace FaceFusion.Core;

/// <summary>
/// Port of <c>facefusion/logger.py</c>.
///
/// The Python module wraps the standard library <c>logging</c> package (a module-level
/// singleton logger named <c>'facefusion'</c>). <c>Microsoft.Extensions.Logging</c> is
/// not referenced by any project in this solution and is not part of the net8.0 shared
/// framework (<c>dotnet list package</c> against <c>FaceFusion.Core</c> shows no package
/// references at all), and port convention forbids adding a NuGet package without asking.
/// So this is a minimal, self-contained logger with the same level semantics as Python's
/// <c>logging</c>, written to an injectable <see cref="TextWriter"/> (defaults to
/// <see cref="Console.Out"/>).
///
/// Deviation from Python: the Python module keeps its state (level, disabled flag) on a
/// process-wide <c>logging.Logger</c> singleton reached via <c>getLogger('facefusion')</c>.
/// Per port convention rule 5 (no global mutable state), this is an instance class;
/// callers that want module-global behaviour should share one instance.
///
/// The Python file has no <c>clear_line</c> or table-formatting helpers to port — it
/// contains exactly the members below.
/// </summary>
public sealed class Logger
{
    private readonly TextWriter _writer;

    // Python: getLogger('facefusion') without an explicit setLevel() has an effective
    // level inherited from the root logger, which basicConfig() leaves at WARNING. So
    // before Init() is called, warn/error are emitted and info/debug are not.
    private LogLevel _logLevel = LogLevel.Warn;
    private bool _disabled;

    public Logger(TextWriter? writer = null)
    {
        _writer = writer ?? Console.Out;
    }

    /// <summary>Python: <c>init</c>.</summary>
    public void Init(LogLevel logLevel)
    {
        _logLevel = logLevel;
    }

    /// <summary>Python: <c>debug</c>.</summary>
    public void Debug(string message, string moduleName) => Log(LogLevel.Debug, message, moduleName);

    /// <summary>Python: <c>info</c>.</summary>
    public void Info(string message, string moduleName) => Log(LogLevel.Info, message, moduleName);

    /// <summary>Python: <c>warn</c>.</summary>
    public void Warn(string message, string moduleName) => Log(LogLevel.Warn, message, moduleName);

    /// <summary>Python: <c>error</c>.</summary>
    public void Error(string message, string moduleName) => Log(LogLevel.Error, message, moduleName);

    /// <summary>Python: <c>create_message</c>.</summary>
    public static string CreateMessage(string message, string moduleName)
    {
        var moduleNames = moduleName.Split('.');
        var firstModuleName = CommonHelper.GetFirst(moduleNames);
        var lastModuleName = CommonHelper.GetLast(moduleNames);

        if (!string.IsNullOrEmpty(firstModuleName) && !string.IsNullOrEmpty(lastModuleName))
        {
            return "[" + firstModuleName.ToUpperInvariant() + "." + lastModuleName.ToUpperInvariant() + "] " + message;
        }

        return message;
    }

    /// <summary>Python: <c>enable</c>.</summary>
    public void Enable()
    {
        _disabled = false;
    }

    /// <summary>Python: <c>disable</c>.</summary>
    public void Disable()
    {
        _disabled = true;
    }

    private void Log(LogLevel level, string message, string moduleName)
    {
        if (_disabled)
        {
            return;
        }

        if (Choices.LogLevelSet[level] < Choices.LogLevelSet[_logLevel])
        {
            return;
        }

        _writer.WriteLine(CreateMessage(message, moduleName));
    }
}
