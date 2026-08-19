using FaceFusion.Cli;
using FaceFusion.Types;
using FaceFusion.Core;
using OpenCvSharp;

namespace FaceFusion.Ui;

/// <summary>
/// Port of <c>facefusion/uis/components/preview.py</c> — renders one frame through the
/// configured processor chain so the user can see the effect of a setting without starting a
/// run. Delegates to <see cref="HeadlessRunner.RenderPreviewFrame"/> so the preview and the run
/// build their steps from the same argument bag through the same factory.
///
/// <para>
/// Frames are handed to the browser as JPEG bytes rendered into a <c>data:</c> URI rather than
/// being passed through component parameters — plan §6's "do not round-trip large frames
/// through component parameters" note. One decoded 1080p frame is ~6 MB as raw BGR and ~200 KB
/// as JPEG.
/// </para>
///
/// <para>
/// Renders are serialised: each one opens its own <c>InferenceSession</c>s, and two concurrent
/// renders would double an already large memory footprint (see the memory defect recorded in
/// docs/IMPLEMENTATION_STATUS.md). A render requested while one is in flight is dropped rather
/// than queued, which is what a slider being dragged should do.
/// </para>
/// </summary>
public sealed class UiPreview
{
    private readonly UiState _state;
    private readonly UiTerminal _terminal;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public UiPreview(UiState state, UiTerminal terminal)
    {
        _state = state;
        _terminal = terminal;
    }

    /// <summary>The most recent rendered frame as a <c>data:image/jpeg;base64,...</c> URI, or
    /// null if nothing has rendered yet.</summary>
    public string? ImageDataUri { get; private set; }

    public bool IsRendering { get; private set; }

    public string? LastError { get; private set; }

    public event Action? Changed;

    public async Task RenderAsync(int frameNumber)
    {
        if (!await _gate.WaitAsync(0).ConfigureAwait(false))
        {
            return; // a render is already in flight — drop this one
        }

        IsRendering = true;
        LastError = null;
        Changed?.Invoke();

        try
        {
            var args = _state.BuildArgs();

            var dataUri = await Task.Run(() =>
            {
                var logger = new Logger(_terminal);
                logger.Init(EnumNames.FromWireName<FaceFusion.Types.LogLevel>(_state.GetString("log_level") ?? "info"));

                using var visionFrame = HeadlessRunner.RenderPreviewFrame(args, frameNumber, logger);

                if (visionFrame is null)
                {
                    return null;
                }

                Cv2.ImEncode(".jpg", visionFrame, out var jpeg, new[] { (int)ImwriteFlags.JpegQuality, 90 });
                return "data:image/jpeg;base64," + Convert.ToBase64String(jpeg);
            }).ConfigureAwait(false);

            if (dataUri is null)
            {
                LastError = "could not render a preview — check the target path and the processors' model files (details in the terminal).";
            }
            else
            {
                ImageDataUri = dataUri;
            }
        }
        catch (Exception exception)
        {
            LastError = $"{exception.GetType().Name}: {exception.Message}";
        }
        finally
        {
            IsRendering = false;
            _gate.Release();
            Changed?.Invoke();
        }
    }
}
