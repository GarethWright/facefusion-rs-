using System.Collections.Frozen;

namespace FaceFusion.Core;

/// <summary>
/// Port of <c>facefusion/filesystem.py</c>.
/// </summary>
public static class FileSystem
{
    // TODO(types): source these from FaceFusion.Types.Choices once that port lands.
    // Mirrors AudioFormat / ImageFormat / VideoFormat in facefusion/types.py.
    private static readonly FrozenSet<string> AudioFormats =
        new[] { "flac", "m4a", "mp3", "ogg", "opus", "wav" }.ToFrozenSet(StringComparer.Ordinal);

    private static readonly FrozenSet<string> ImageFormats =
        new[] { "bmp", "jpeg", "png", "tiff", "webp" }.ToFrozenSet(StringComparer.Ordinal);

    private static readonly FrozenSet<string> VideoFormats =
        new[] { "avi", "m4v", "mkv", "mov", "mp4", "mpeg", "mxf", "webm", "wmv" }.ToFrozenSet(StringComparer.Ordinal);

    /// <summary>
    /// Reproduces Python's <c>os.path.splitext</c> exactly, including its handling of
    /// leading dots: <c>".gitignore"</c> splits to <c>(".gitignore", "")</c>, not to an
    /// extension. .NET's <see cref="Path.GetExtension(string)"/> does not agree with
    /// Python on every input, so the algorithm is reimplemented here rather than
    /// delegated — file extensions drive format detection and must match the Python.
    /// </summary>
    public static (string Root, string Extension) SplitExt(string path)
    {
        var separatorIndex = path.LastIndexOfAny(new[] { '/', Path.DirectorySeparatorChar });
        var dotIndex = path.LastIndexOf('.');

        if (dotIndex > separatorIndex)
        {
            // Skip a run of leading dots in the file name; a name that is all dots has
            // no extension.
            var fileNameIndex = separatorIndex + 1;

            while (fileNameIndex < dotIndex)
            {
                if (path[fileNameIndex] != '.')
                {
                    return (path[..dotIndex], path[dotIndex..]);
                }

                fileNameIndex++;
            }
        }

        return (path, string.Empty);
    }

    /// <summary>Python: <c>get_file_size</c>.</summary>
    public static long GetFileSize(string? filePath)
        => IsFile(filePath) ? new FileInfo(filePath!).Length : 0;

    /// <summary>Python: <c>get_file_name</c>. Returns null when the name is empty.</summary>
    public static string? GetFileName(string? filePath)
    {
        if (filePath is null)
        {
            return null;
        }

        var baseName = GetBaseName(filePath);
        var (fileName, _) = SplitExt(baseName);

        return string.IsNullOrEmpty(fileName) ? null : fileName;
    }

    /// <summary>Python: <c>get_file_extension</c>. Lower-cased, includes the dot.</summary>
    public static string? GetFileExtension(string? filePath)
    {
        if (filePath is null)
        {
            return null;
        }

        var (_, fileExtension) = SplitExt(filePath);

        return string.IsNullOrEmpty(fileExtension) ? null : fileExtension.ToLowerInvariant();
    }

    /// <summary>Python: <c>get_file_format</c>.</summary>
    public static string? GetFileFormat(string? filePath)
    {
        var fileExtension = GetFileExtension(filePath);

        if (string.IsNullOrEmpty(fileExtension))
        {
            return null;
        }

        return fileExtension switch
        {
            ".jpg" => "jpeg",
            ".tif" => "tiff",
            ".mpg" => "mpeg",
            _ => fileExtension.TrimStart('.')
        };
    }

    /// <summary>Python: <c>same_file_extension</c>.</summary>
    public static bool SameFileExtension(string? firstFilePath, string? secondFilePath)
    {
        var first = GetFileExtension(firstFilePath);
        var second = GetFileExtension(secondFilePath);

        if (!string.IsNullOrEmpty(first) && !string.IsNullOrEmpty(second))
        {
            return string.Equals(first, second, StringComparison.Ordinal);
        }

        return false;
    }

    /// <summary>Python: <c>is_file</c>.</summary>
    public static bool IsFile(string? filePath)
        => !string.IsNullOrEmpty(filePath) && File.Exists(filePath);

    /// <summary>Python: <c>is_directory</c>.</summary>
    public static bool IsDirectory(string? directoryPath)
        => !string.IsNullOrEmpty(directoryPath) && Directory.Exists(directoryPath);

    /// <summary>Python: <c>is_audio</c>.</summary>
    public static bool IsAudio(string? audioPath)
        => IsFile(audioPath) && GetFileFormat(audioPath) is { } format && AudioFormats.Contains(format);

    /// <summary>Python: <c>is_image</c>.</summary>
    public static bool IsImage(string? imagePath)
        => IsFile(imagePath) && GetFileFormat(imagePath) is { } format && ImageFormats.Contains(format);

    /// <summary>Python: <c>is_video</c>.</summary>
    public static bool IsVideo(string? videoPath)
        => IsFile(videoPath) && GetFileFormat(videoPath) is { } format && VideoFormats.Contains(format);

    /// <summary>Python: <c>has_audio</c>.</summary>
    public static bool HasAudio(IReadOnlyList<string>? audioPaths)
        => audioPaths is { Count: > 0 } && audioPaths.Any(IsAudio);

    /// <summary>Python: <c>are_audios</c>.</summary>
    public static bool AreAudios(IReadOnlyList<string>? audioPaths)
        => audioPaths is { Count: > 0 } && audioPaths.All(IsAudio);

    /// <summary>Python: <c>has_image</c>.</summary>
    public static bool HasImage(IReadOnlyList<string>? imagePaths)
        => imagePaths is { Count: > 0 } && imagePaths.Any(IsImage);

    /// <summary>Python: <c>are_images</c>.</summary>
    public static bool AreImages(IReadOnlyList<string>? imagePaths)
        => imagePaths is { Count: > 0 } && imagePaths.All(IsImage);

    /// <summary>Python: <c>has_video</c>.</summary>
    public static bool HasVideo(IReadOnlyList<string>? videoPaths)
        => videoPaths is { Count: > 0 } && videoPaths.Any(IsVideo);

    /// <summary>Python: <c>are_videos</c>.</summary>
    public static bool AreVideos(IReadOnlyList<string>? videoPaths)
        => videoPaths is { Count: > 0 } && videoPaths.All(IsVideo);

    /// <summary>Python: <c>filter_audio_paths</c>.</summary>
    public static IReadOnlyList<string> FilterAudioPaths(IReadOnlyList<string>? paths)
        => paths is { Count: > 0 } ? paths.Where(IsAudio).ToArray() : Array.Empty<string>();

    /// <summary>Python: <c>filter_image_paths</c>.</summary>
    public static IReadOnlyList<string> FilterImagePaths(IReadOnlyList<string>? paths)
        => paths is { Count: > 0 } ? paths.Where(IsImage).ToArray() : Array.Empty<string>();

    /// <summary>Python: <c>copy_file</c>.</summary>
    public static bool CopyFile(string? filePath, string? movePath)
    {
        if (IsFile(filePath) && !string.IsNullOrEmpty(movePath))
        {
            File.Copy(filePath!, movePath, overwrite: true);
            return IsFile(movePath);
        }

        return false;
    }

    /// <summary>Python: <c>move_file</c>.</summary>
    public static bool MoveFile(string? filePath, string? movePath)
    {
        if (IsFile(filePath) && !string.IsNullOrEmpty(movePath))
        {
            File.Move(filePath!, movePath, overwrite: true);
            return !IsFile(filePath) && IsFile(movePath);
        }

        return false;
    }

    /// <summary>Python: <c>remove_file</c>.</summary>
    public static bool RemoveFile(string? filePath)
    {
        if (IsFile(filePath))
        {
            File.Delete(filePath!);
            return !IsFile(filePath);
        }

        return false;
    }

    /// <summary>
    /// Python: <c>resolve_file_paths</c>. Sorted, skipping entries that start with
    /// '.' or '__'.
    /// </summary>
    public static IReadOnlyList<string> ResolveFilePaths(string? directoryPath)
    {
        if (!IsDirectory(directoryPath))
        {
            return Array.Empty<string>();
        }

        var entries = Directory.GetFileSystemEntries(directoryPath!)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrEmpty(name))
            .Select(name => name!)
            .Where(name => !name.StartsWith('.') && !name.StartsWith("__", StringComparison.Ordinal))
            .ToList();

        // Python sorts the raw names before joining, so sort on the name, not the path.
        entries.Sort(StringComparer.Ordinal);

        return entries.Select(name => Path.Combine(directoryPath!, name)).ToArray();
    }

    /// <summary>Python: <c>in_directory</c>.</summary>
    public static bool InDirectory(string? filePath)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            return false;
        }

        var directoryPath = GetDirectoryName(filePath);

        if (!string.IsNullOrEmpty(directoryPath))
        {
            return !IsDirectory(filePath) && IsDirectory(directoryPath);
        }

        return false;
    }

    /// <summary>Python: <c>resolve_file_pattern</c> (glob).</summary>
    public static IReadOnlyList<string> ResolveFilePattern(string? filePattern)
    {
        if (!InDirectory(filePattern))
        {
            return Array.Empty<string>();
        }

        var directoryPath = GetDirectoryName(filePattern!);
        var searchPattern = Path.GetFileName(filePattern!);

        if (string.IsNullOrEmpty(searchPattern))
        {
            return Array.Empty<string>();
        }

        var matches = Directory
            .GetFileSystemEntries(directoryPath, searchPattern, SearchOption.TopDirectoryOnly)
            .ToList();

        matches.Sort(StringComparer.Ordinal);

        return matches;
    }

    /// <summary>Python: <c>create_directory</c>.</summary>
    public static bool CreateDirectory(string? directoryPath)
    {
        if (!string.IsNullOrEmpty(directoryPath) && !IsFile(directoryPath))
        {
            Directory.CreateDirectory(directoryPath);
            return IsDirectory(directoryPath);
        }

        return false;
    }

    /// <summary>Python: <c>remove_directory</c>.</summary>
    public static bool RemoveDirectory(string? directoryPath)
    {
        if (IsDirectory(directoryPath))
        {
            try
            {
                Directory.Delete(directoryPath!, recursive: true);
            }
            catch (IOException)
            {
                // Python passes ignore_errors = True to shutil.rmtree.
            }
            catch (UnauthorizedAccessException)
            {
                // As above.
            }

            return !IsDirectory(directoryPath);
        }

        return false;
    }

    /// <summary>
    /// Python: <c>resolve_relative_path</c>, which resolves against the directory of the
    /// facefusion package. Here it resolves against the application base directory.
    /// </summary>
    public static string ResolveRelativePath(string path)
        => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, path));

    private static string GetBaseName(string path)
    {
        var separatorIndex = path.LastIndexOfAny(new[] { '/', Path.DirectorySeparatorChar });
        return separatorIndex < 0 ? path : path[(separatorIndex + 1)..];
    }

    private static string GetDirectoryName(string path)
    {
        var separatorIndex = path.LastIndexOfAny(new[] { '/', Path.DirectorySeparatorChar });
        return separatorIndex < 0 ? string.Empty : path[..separatorIndex];
    }
}
