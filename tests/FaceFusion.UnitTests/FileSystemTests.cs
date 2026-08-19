using FaceFusion.Core;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_filesystem.py</c>.
///
/// The Python tests download example media from GitHub releases in a module fixture.
/// Network egress is restricted here, so the fixtures are synthesised locally instead:
/// every assertion these tests make depends on the file's *name* and *existence*, not on
/// its content. The one exception is <c>test_get_file_size</c>, which asserts the exact
/// byte count of a downloaded asset; that case is reproduced against a file of known
/// length written here.
/// </summary>
public sealed class FileSystemTests : IDisposable
{
    private readonly string _examplesDirectory;

    public FileSystemTests()
    {
        _examplesDirectory = Path.Combine(Path.GetTempPath(), "facefusion-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_examplesDirectory);

        foreach (var fileName in new[] { "source.jpg", "source.mp3", "target-240p.mp4" })
        {
            File.WriteAllText(ExampleFile(fileName), "fixture");
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_examplesDirectory))
        {
            Directory.Delete(_examplesDirectory, recursive: true);
        }
    }

    private string ExampleFile(string fileName) => Path.Combine(_examplesDirectory, fileName);

    [Fact]
    public void GetFileSize()
    {
        var filePath = ExampleFile("sized.bin");
        File.WriteAllBytes(filePath, new byte[549458]);

        Assert.Equal(549458, FileSystem.GetFileSize(filePath));
        Assert.Equal(0, FileSystem.GetFileSize("invalid"));
    }

    [Fact]
    public void GetFileExtension()
    {
        Assert.Equal(".jpg", FileSystem.GetFileExtension("source.jpg"));
        Assert.Equal(".mp3", FileSystem.GetFileExtension("source.mp3"));
        Assert.Null(FileSystem.GetFileExtension("invalid"));
    }

    [Fact]
    public void GetFileFormat()
    {
        Assert.Equal("jpeg", FileSystem.GetFileFormat("source.jpg"));
        Assert.Equal("jpeg", FileSystem.GetFileFormat("source.jpeg"));
        Assert.Equal("mp3", FileSystem.GetFileFormat("source.mp3"));
        Assert.Null(FileSystem.GetFileFormat("invalid"));

        // The remaining aliases from get_file_format, not covered by the Python test.
        Assert.Equal("tiff", FileSystem.GetFileFormat("source.tif"));
        Assert.Equal("mpeg", FileSystem.GetFileFormat("source.mpg"));
    }

    [Fact]
    public void SameFileExtension()
    {
        Assert.True(FileSystem.SameFileExtension("source.jpg", "source.jpg"));
        Assert.False(FileSystem.SameFileExtension("source.jpg", "source.mp3"));
        Assert.False(FileSystem.SameFileExtension("invalid", "invalid"));
    }

    [Fact]
    public void IsFile()
    {
        Assert.True(FileSystem.IsFile(ExampleFile("source.jpg")));
        Assert.False(FileSystem.IsFile(_examplesDirectory));
        Assert.False(FileSystem.IsFile("invalid"));
    }

    [Fact]
    public void IsAudio()
    {
        Assert.True(FileSystem.IsAudio(ExampleFile("source.mp3")));
        Assert.False(FileSystem.IsAudio(ExampleFile("source.jpg")));
        Assert.False(FileSystem.IsAudio("invalid"));
    }

    [Fact]
    public void HasAudio()
    {
        Assert.True(FileSystem.HasAudio(new[] { ExampleFile("source.mp3") }));
        Assert.True(FileSystem.HasAudio(new[] { ExampleFile("source.mp3"), ExampleFile("source.jpg") }));
        Assert.False(FileSystem.HasAudio(new[] { ExampleFile("source.jpg"), ExampleFile("source.jpg") }));
        Assert.False(FileSystem.HasAudio(new[] { "invalid" }));
    }

    [Fact]
    public void IsImage()
    {
        Assert.True(FileSystem.IsImage(ExampleFile("source.jpg")));
        Assert.False(FileSystem.IsImage(ExampleFile("source.mp3")));
        Assert.False(FileSystem.IsImage("invalid"));
    }

    [Fact]
    public void HasImage()
    {
        Assert.True(FileSystem.HasImage(new[] { ExampleFile("source.jpg") }));
        Assert.True(FileSystem.HasImage(new[] { ExampleFile("source.jpg"), ExampleFile("source.mp3") }));
        Assert.False(FileSystem.HasImage(new[] { ExampleFile("source.mp3"), ExampleFile("source.mp3") }));
        Assert.False(FileSystem.HasImage(new[] { "invalid" }));
    }

    [Fact]
    public void IsVideo()
    {
        Assert.True(FileSystem.IsVideo(ExampleFile("target-240p.mp4")));
        Assert.False(FileSystem.IsVideo(ExampleFile("source.jpg")));
        Assert.False(FileSystem.IsVideo("invalid"));
    }

    [Fact]
    public void HasVideo()
    {
        Assert.True(FileSystem.HasVideo(new[] { ExampleFile("target-240p.mp4") }));
        Assert.True(FileSystem.HasVideo(new[] { ExampleFile("target-240p.mp4"), ExampleFile("source.jpg") }));
        Assert.False(FileSystem.HasVideo(new[] { ExampleFile("source.jpg") }));
        Assert.False(FileSystem.HasVideo(new[] { "invalid" }));
    }

    [Fact]
    public void FilterAudioPaths()
    {
        var jpg = ExampleFile("source.jpg");
        var mp3 = ExampleFile("source.mp3");

        Assert.Equal(new[] { mp3 }, FileSystem.FilterAudioPaths(new[] { jpg, mp3 }));
        Assert.Empty(FileSystem.FilterAudioPaths(new[] { jpg, jpg }));
        Assert.Empty(FileSystem.FilterAudioPaths(Array.Empty<string>()));
    }

    [Fact]
    public void FilterImagePaths()
    {
        var jpg = ExampleFile("source.jpg");
        var mp3 = ExampleFile("source.mp3");

        Assert.Equal(new[] { jpg }, FileSystem.FilterImagePaths(new[] { jpg, mp3 }));
        Assert.Empty(FileSystem.FilterImagePaths(new[] { mp3, mp3 }));
        Assert.Empty(FileSystem.FilterImagePaths(Array.Empty<string>()));
    }

    [Fact]
    public void IsDirectory()
    {
        Assert.True(FileSystem.IsDirectory(_examplesDirectory));
        Assert.False(FileSystem.IsDirectory(ExampleFile("source.jpg")));
        Assert.False(FileSystem.IsDirectory("invalid"));
    }

    [Fact]
    public void InDirectory()
    {
        Assert.True(FileSystem.InDirectory(ExampleFile("source.jpg")));
        Assert.False(FileSystem.InDirectory(_examplesDirectory));
        Assert.False(FileSystem.InDirectory("invalid"));
    }

    [Fact]
    public void CreateAndRemoveDirectory()
    {
        var directoryPath = Path.Combine(_examplesDirectory, "nested", "deeper");

        Assert.True(FileSystem.CreateDirectory(directoryPath));
        Assert.True(FileSystem.IsDirectory(directoryPath));
        Assert.False(FileSystem.CreateDirectory(ExampleFile("source.jpg")));

        Assert.True(FileSystem.RemoveDirectory(directoryPath));
        Assert.False(FileSystem.RemoveDirectory(directoryPath));
        Assert.False(FileSystem.RemoveDirectory("invalid"));
    }

    [Fact]
    public void ResolveFilePaths()
    {
        var directoryPath = Path.Combine(_examplesDirectory, "resolve");
        Directory.CreateDirectory(directoryPath);

        File.WriteAllText(Path.Combine(directoryPath, "b.txt"), string.Empty);
        File.WriteAllText(Path.Combine(directoryPath, "a.txt"), string.Empty);
        File.WriteAllText(Path.Combine(directoryPath, ".hidden"), string.Empty);
        File.WriteAllText(Path.Combine(directoryPath, "__dunder__.py"), string.Empty);

        var filePaths = FileSystem.ResolveFilePaths(directoryPath);

        // Sorted, with dot- and dunder-prefixed entries excluded.
        Assert.Equal(
            new[] { Path.Combine(directoryPath, "a.txt"), Path.Combine(directoryPath, "b.txt") },
            filePaths);
        Assert.Empty(FileSystem.ResolveFilePaths("invalid"));
    }

    [Fact]
    public void ResolveFilePattern()
    {
        var directoryPath = Path.Combine(_examplesDirectory, "pattern");
        Directory.CreateDirectory(directoryPath);

        File.WriteAllText(Path.Combine(directoryPath, "one.txt"), string.Empty);
        File.WriteAllText(Path.Combine(directoryPath, "two.txt"), string.Empty);
        File.WriteAllText(Path.Combine(directoryPath, "three.md"), string.Empty);

        var matches = FileSystem.ResolveFilePattern(Path.Combine(directoryPath, "*.txt"));

        Assert.Equal(
            new[] { Path.Combine(directoryPath, "one.txt"), Path.Combine(directoryPath, "two.txt") },
            matches);
        Assert.Empty(FileSystem.ResolveFilePattern("invalid"));
    }

    [Fact]
    public void CopyMoveAndRemoveFile()
    {
        var sourcePath = ExampleFile("source.jpg");
        var copyPath = ExampleFile("copy.jpg");
        var movePath = ExampleFile("moved.jpg");

        Assert.True(FileSystem.CopyFile(sourcePath, copyPath));
        Assert.True(FileSystem.IsFile(copyPath));
        Assert.False(FileSystem.CopyFile("invalid", copyPath));

        Assert.True(FileSystem.MoveFile(copyPath, movePath));
        Assert.False(FileSystem.IsFile(copyPath));
        Assert.True(FileSystem.IsFile(movePath));
        Assert.False(FileSystem.MoveFile("invalid", movePath));

        Assert.True(FileSystem.RemoveFile(movePath));
        Assert.False(FileSystem.RemoveFile(movePath));
    }

    /// <summary>
    /// Not in the Python test suite, but guards the reimplementation of
    /// <c>os.path.splitext</c> in <see cref="FileSystem.SplitExt"/>. Ground truth
    /// produced with <c>python3 -c "import os.path; print(os.path.splitext(...))"</c>.
    /// </summary>
    [Theory]
    [InlineData("source.jpg", "source", ".jpg")]
    [InlineData(".gitignore", ".gitignore", "")]
    [InlineData("..", "..", "")]
    [InlineData(".", ".", "")]
    [InlineData("archive.tar.gz", "archive.tar", ".gz")]
    [InlineData("noextension", "noextension", "")]
    [InlineData("dir.with.dots/file", "dir.with.dots/file", "")]
    [InlineData("/tmp/.hidden.txt", "/tmp/.hidden", ".txt")]
    public void SplitExtMatchesPython(string path, string expectedRoot, string expectedExtension)
    {
        var (root, extension) = FileSystem.SplitExt(path);

        Assert.Equal(expectedRoot, root);
        Assert.Equal(expectedExtension, extension);
    }
}
