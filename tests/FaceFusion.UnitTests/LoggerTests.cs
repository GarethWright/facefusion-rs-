using FaceFusion.Core;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Tests for <see cref="Logger"/> (there is no Python test module for
/// <c>facefusion/logger.py</c> to port).
/// </summary>
public sealed class LoggerTests
{
    [Fact]
    public void TestCreateMessagePrefixesUpperCasedFirstAndLastModule()
    {
        Assert.Equal(
            "[FACEFUSION.CORE] hello",
            Logger.CreateMessage("hello", "facefusion.core"));
    }

    [Fact]
    public void TestCreateMessageSingleModuleNameUsesItForBothParts()
    {
        Assert.Equal("[FACEFUSION.FACEFUSION] hello", Logger.CreateMessage("hello", "facefusion"));
    }

    [Fact]
    public void TestInfoIsSuppressedBelowDefaultWarnLevel()
    {
        using var writer = new StringWriter();
        var logger = new Logger(writer);

        logger.Info("hidden", "facefusion.core");

        Assert.Equal(string.Empty, writer.ToString());
    }

    [Fact]
    public void TestWarnIsEmittedAtDefaultLevel()
    {
        using var writer = new StringWriter();
        var logger = new Logger(writer);

        logger.Warn("shown", "facefusion.core");

        Assert.Contains("shown", writer.ToString());
    }

    [Fact]
    public void TestInitToDebugAllowsDebugMessages()
    {
        using var writer = new StringWriter();
        var logger = new Logger(writer);
        logger.Init(LogLevel.Debug);

        logger.Debug("shown", "facefusion.core");

        Assert.Contains("shown", writer.ToString());
    }

    [Fact]
    public void TestDisableSuppressesAllLevels()
    {
        using var writer = new StringWriter();
        var logger = new Logger(writer);
        logger.Init(LogLevel.Debug);
        logger.Disable();

        logger.Error("hidden", "facefusion.core");

        Assert.Equal(string.Empty, writer.ToString());
    }

    [Fact]
    public void TestEnableAfterDisableResumesLogging()
    {
        using var writer = new StringWriter();
        var logger = new Logger(writer);
        logger.Disable();
        logger.Enable();

        logger.Error("shown", "facefusion.core");

        Assert.Contains("shown", writer.ToString());
    }
}
