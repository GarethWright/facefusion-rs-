using FaceFusion.Core;

namespace FaceFusion.UnitTests;

/// <summary>
/// Regression for a defect the existing translator tests could not catch.
///
/// Python's <c>translator.get()</c> calls <c>__autoload__</c> on a miss, importing the
/// module's locale table. The C# port originally required an explicit <c>Load()</c> and
/// nothing in <c>src/</c> ever called it, so every <c>Get</c> returned null in production
/// and every log line printed its raw key — <c>restoring_audio_skipped</c> instead of
/// "restoring audio skipped". It surfaced only by running the CLI and diffing its output
/// against the Python CLI's.
///
/// <c>TranslatorTests</c> missed it because those tests call <c>Load()</c> themselves
/// first, which production never did. These tests deliberately do NOT call
/// <c>Load()</c> — that is the whole point.
/// </summary>
public sealed class TranslatorAutoloadTests
{
    [Fact]
    public void GetResolvesWithoutAnExplicitLoad()
    {
        var message = Translator.Get("restoring_audio_skipped");

        Assert.Equal("restoring audio skipped", message);
    }

    /// <summary>
    /// A spread of keys across the message table, so a partially-populated pool cannot
    /// pass. Values are the literal strings in facefusion/locales.py.
    /// </summary>
    [Theory]
    [InlineData("processing", "processing")]
    [InlineData("downloading", "downloading")]
    [InlineData("temp_frames_not_found", "temporary frames not found")]
    [InlineData("skipping_audio", "skipping audio")]
    [InlineData("merging_video_succeeded", "merging video succeeded")]
    public void KnownKeysResolveToTheirPythonText(string key, string expected)
    {
        Assert.Equal(expected, Translator.Get(key));
    }

    /// <summary>An unknown key still returns null, so callers can fall back.</summary>
    [Fact]
    public void UnknownKeyReturnsNull()
    {
        Assert.Null(Translator.Get("this_key_does_not_exist"));
    }

    /// <summary>
    /// Formatted messages must substitute rather than emit the placeholder, which is the
    /// other half of what users see in the log.
    /// </summary>
    [Fact]
    public void FormattedMessageSubstitutesPlaceholders()
    {
        var template = Translator.Get("processing_step");
        Assert.NotNull(template);

        var formatted = Translator.Format(template!, ("step_current", 2), ("step_total", 5));

        Assert.Contains("2", formatted, StringComparison.Ordinal);
        Assert.Contains("5", formatted, StringComparison.Ordinal);
        Assert.DoesNotContain("{step_current}", formatted, StringComparison.Ordinal);
    }
}
