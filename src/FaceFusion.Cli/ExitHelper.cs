namespace FaceFusion.Cli;

/// <summary>
/// Port of <c>facefusion/exit_helper.py</c>.
///
/// Python's <c>hard_exit</c> calls <c>sys.exit</c> and <c>fatal_exit</c> calls
/// <c>os._exit</c>. A library should not kill its host, so the error codes are returned
/// up to <c>Main</c> and only the executable's entry point actually exits — the same
/// reasoning already applied to <c>InferenceManager</c>, which throws where Python calls
/// <c>fatal_exit</c>.
/// </summary>
public static class ExitHelper
{
    /// <summary>
    /// Python: <c>graceful_exit</c>. Stops processing, waits for in-flight work to
    /// finish, then clears the temp directory for the target.
    /// </summary>
    public static int GracefulExit(
        int errorCode,
        FaceFusion.Core.ProcessManager processManager,
        string? targetPath,
        string? tempPath)
    {
        processManager.Stop();

        while (processManager.IsProcessing())
        {
            Thread.Sleep(500);
        }

        if (!string.IsNullOrEmpty(targetPath) && !string.IsNullOrEmpty(tempPath))
        {
            FaceFusion.Core.TempHelper.ClearTempDirectory(targetPath, tempPath);
        }

        return errorCode;
    }
}
