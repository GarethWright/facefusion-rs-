namespace FaceFusion.Core;

/// <summary>
/// Port of <c>facefusion/temp_helper.py</c>.
///
/// The Python module reads two values from <c>state_manager</c>: <c>temp_path</c> and
/// <c>temp_frame_format</c>. Per port convention rule 5 (no global mutable state), both
/// are taken as explicit parameters here instead of being read from shared state.
/// </summary>
public static class TempHelper
{
    /// <summary>Python: <c>get_temp_file_path</c>.</summary>
    public static string GetTempFilePath(string filePath, string tempPath)
    {
        var tempDirectoryPath = GetTempDirectoryPath(filePath, tempPath);
        var tempFileExtension = FileSystem.GetFileExtension(filePath) ?? string.Empty;

        return Path.Combine(tempDirectoryPath, "temp" + tempFileExtension);
    }

    /// <summary>Python: <c>move_temp_file</c>.</summary>
    public static bool MoveTempFile(string filePath, string movePath, string tempPath)
    {
        var tempFilePath = GetTempFilePath(filePath, tempPath);
        return FileSystem.MoveFile(tempFilePath, movePath);
    }

    /// <summary>Python: <c>resolve_temp_frame_set</c>.</summary>
    public static IReadOnlyDictionary<int, string> ResolveTempFrameSet(
        string targetPath, string tempPath, string tempFrameFormat)
    {
        var tempFramePattern = GetTempFramePattern(targetPath, "*", tempPath, tempFrameFormat);
        var tempFrameSet = new Dictionary<int, string>();

        foreach (var tempFramePath in FileSystem.ResolveFilePattern(tempFramePattern))
        {
            var fileName = FileSystem.GetFileName(tempFramePath);

            if (fileName is not null && int.TryParse(fileName, out var frameNumber))
            {
                tempFrameSet[frameNumber] = tempFramePath;
            }
        }

        return tempFrameSet;
    }

    /// <summary>Python: <c>get_temp_frame_pattern</c>.</summary>
    public static string GetTempFramePattern(
        string targetPath, string tempFramePrefix, string tempPath, string tempFrameFormat)
    {
        var tempDirectoryPath = GetTempDirectoryPath(targetPath, tempPath);
        return Path.Combine(tempDirectoryPath, tempFramePrefix + "." + tempFrameFormat);
    }

    /// <summary>Python: <c>get_temp_directory_path</c>.</summary>
    public static string GetTempDirectoryPath(string filePath, string tempPath)
    {
        var tempFileName = FileSystem.GetFileName(filePath) ?? string.Empty;
        return Path.Combine(tempPath, "facefusion", tempFileName);
    }

    /// <summary>Python: <c>create_temp_directory</c>.</summary>
    public static bool CreateTempDirectory(string filePath, string tempPath)
    {
        var tempDirectoryPath = GetTempDirectoryPath(filePath, tempPath);
        return FileSystem.CreateDirectory(tempDirectoryPath);
    }

    /// <summary>Python: <c>clear_temp_directory</c>.</summary>
    public static bool ClearTempDirectory(string filePath, string tempPath)
    {
        var tempDirectoryPath = GetTempDirectoryPath(filePath, tempPath);
        return FileSystem.RemoveDirectory(tempDirectoryPath);
    }
}
