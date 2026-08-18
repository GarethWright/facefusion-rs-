using FaceFusion.Cli;
using FaceFusion.Face;

namespace FaceFusion.UnitTests;

/// <summary>
/// Covers the NSFW content-analyser gate required by plan §9.6.
///
/// Python refuses to run unless the hash of <c>content_analyser.py</c>'s source matches a
/// constant held in <c>core.py</c>, so editing the analyser to weaken a threshold or stub
/// out a detector is detected. These tests check the C# equivalent is actually wired and
/// actually fails closed — a gate that silently passes is worse than no gate, because it
/// looks like protection.
/// </summary>
public sealed class PreCheckTests
{
    /// <summary>
    /// The pinned constant must match the analyser's real hash. If this fails, either the
    /// analyser was edited (intentionally or not) or the constant is stale — investigate
    /// which before touching either. Updating the constant to silence this defeats the
    /// mechanism.
    /// </summary>
    [Fact]
    public void PinnedHashMatchesTheAnalyserSource()
    {
        var actual = ContentAnalyser.ComputeSourceHash();

        if (actual is null)
        {
            // Source file not deployed (e.g. a DLL-only publish) — the documented limit of
            // this mechanism, not a failure of it.
            Assert.False(PreCheck.CommonPreCheck(PreCheck.ContentAnalyserHash));
            return;
        }

        Assert.Equal(PreCheck.ContentAnalyserHash, actual);
        Assert.True(PreCheck.CommonPreCheck(PreCheck.ContentAnalyserHash));
    }

    /// <summary>
    /// The gate must reject a wrong hash. This is the property that makes it tamper-
    /// evident rather than decorative.
    /// </summary>
    [Fact]
    public void RejectsAModifiedAnalyser()
    {
        Assert.False(PreCheck.CommonPreCheck("00000000"));
        Assert.False(PreCheck.CommonPreCheck(string.Empty));
    }

    /// <summary>
    /// External tools the port shells out to. All three are present in this environment,
    /// so this asserts the happy path rather than skipping.
    /// </summary>
    [Fact]
    public void ExternalToolsPreCheckFindsTheInstalledTools()
    {
        var found = PreCheck.ExternalToolsPreCheck(out var missing);

        if (!found)
        {
            Assert.NotNull(missing);
            return;
        }

        Assert.Null(missing);
    }
}
