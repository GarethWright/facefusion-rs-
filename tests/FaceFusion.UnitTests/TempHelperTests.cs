using FaceFusion.Core;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_temp_helper.py</c>.
///
/// The Python tests download example media from GitHub releases and read
/// <c>temp_path</c>/<c>temp_frame_format</c> from <c>state_manager</c>. Network egress is
/// restricted here, and per port convention rule 5 those state values are taken as plain
/// parameters instead, so this reproduces the same assertions against a locally
/// synthesised example file and explicit <c>tempPath</c>/<c>tempFrameFormat</c> arguments.
/// </summary>
public sealed class TempHelperTests : IDisposable
{
    private readonly string _tempPath;
    private readonly string _exampleFile;
    private const string TempFrameFormat = "png";

    public TempHelperTests()
    {
        _tempPath = Path.Combine(Path.GetTempPath(), "facefusion-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempPath);

        _exampleFile = Path.Combine(_tempPath, "target-240p.mp4");
        File.WriteAllText(_exampleFile, "fixture");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempPath))
        {
            Directory.Delete(_tempPath, recursive: true);
        }
    }

    [Fact]
    public void TestGetTempFilePath()
    {
        var expected = Path.Combine(_tempPath, "facefusion", "target-240p", "temp.mp4");

        Assert.Equal(expected, TempHelper.GetTempFilePath(_exampleFile, _tempPath));
    }

    [Fact]
    public void TestGetTempDirectoryPath()
    {
        var expected = Path.Combine(_tempPath, "facefusion", "target-240p");

        Assert.Equal(expected, TempHelper.GetTempDirectoryPath(_exampleFile, _tempPath));
    }

    [Fact]
    public void TestGetTempFramePattern()
    {
        var expected = Path.Combine(_tempPath, "facefusion", "target-240p", "%04d.png");

        Assert.Equal(expected, TempHelper.GetTempFramePattern(_exampleFile, "%04d", _tempPath, TempFrameFormat));
    }

    [Fact]
    public void TestCreateAndClearTempDirectory()
    {
        var tempDirectoryPath = TempHelper.GetTempDirectoryPath(_exampleFile, _tempPath);

        Assert.True(TempHelper.CreateTempDirectory(_exampleFile, _tempPath));
        Assert.True(Directory.Exists(tempDirectoryPath));

        Assert.True(TempHelper.ClearTempDirectory(_exampleFile, _tempPath));
        Assert.False(Directory.Exists(tempDirectoryPath));
    }

    [Fact]
    public void TestResolveTempFrameSet()
    {
        var tempDirectoryPath = TempHelper.GetTempDirectoryPath(_exampleFile, _tempPath);
        Directory.CreateDirectory(tempDirectoryPath);
        File.WriteAllText(Path.Combine(tempDirectoryPath, "0001.png"), "fixture");
        File.WriteAllText(Path.Combine(tempDirectoryPath, "0002.png"), "fixture");

        var tempFrameSet = TempHelper.ResolveTempFrameSet(_exampleFile, _tempPath, TempFrameFormat);

        Assert.Equal(2, tempFrameSet.Count);
        Assert.Equal(Path.Combine(tempDirectoryPath, "0001.png"), tempFrameSet[1]);
        Assert.Equal(Path.Combine(tempDirectoryPath, "0002.png"), tempFrameSet[2]);
    }
}
