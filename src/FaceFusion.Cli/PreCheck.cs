using FaceFusion.Face;
using FaceFusion.Processors;

namespace FaceFusion.Cli;

/// <summary>
/// Port of <c>pre_check</c>, <c>common_pre_check</c> and <c>processors_pre_check</c> from
/// <c>facefusion/core.py</c>.
///
/// <para>
/// <b>The content-analyser gate.</b> Python refuses to run unless the hash of
/// <c>content_analyser.py</c>'s source equals a constant held in <c>core.py</c>, making
/// the NSFW gate tamper-evident: editing the analyser to weaken a threshold or stub a
/// detector changes the hash and the run is refused. Plan §9.6 makes carrying an
/// equivalent a hard requirement, so <see cref="CommonPreCheck"/> is the wiring point for
/// <see cref="ContentAnalyser.VerifyIntegrity"/>.
/// </para>
/// <para>
/// It fails closed and there is deliberately no flag to bypass it. The expected hash lives
/// here rather than inside <c>ContentAnalyser.cs</c>, mirroring Python's own split — the
/// constant is in <c>core.py</c>, not in the module being hashed, so the module cannot
/// certify itself.
/// </para>
/// </summary>
public static class PreCheck
{
    /// <summary>
    /// Expected hash of the content-analyser source. Python pins <c>'3c6ce25e'</c> for its
    /// own file; the C# file is different text, so its own hash is recorded here.
    ///
    /// Regenerate deliberately, and only when the analyser is intentionally changed:
    /// <c>ContentAnalyser.ComputeSourceHash()</c> returns the current value. Updating it to
    /// silence a failure defeats the entire mechanism.
    /// </summary>
    public const string ContentAnalyserHash = "e5aab047";

    /// <summary>
    /// Python: <c>pre_check</c> — verifies the external tools the port shells out to are
    /// present. Python also checks its own interpreter version, which has no analogue.
    /// </summary>
    public static bool ExternalToolsPreCheck(out string? missingTool)
    {
        foreach (var tool in new[] { "curl", "ffmpeg", "ffprobe" })
        {
            if (FaceFusion.Core.ProcessHelper.Which(tool) is null)
            {
                missingTool = tool;
                return false;
            }
        }

        missingTool = null;
        return true;
    }

    /// <summary>
    /// Python: <c>common_pre_check</c>. Returns false when the content analyser cannot be
    /// verified, which blocks the run.
    /// </summary>
    public static bool CommonPreCheck(string expectedHash)
        => ContentAnalyser.VerifyIntegrity(expectedHash);

    /// <summary>Python: <c>processors_pre_check</c>.</summary>
    public static bool ProcessorsPreCheck(IEnumerable<IProcessor> processors)
        => processors.All(processor => processor.PreCheck());
}
