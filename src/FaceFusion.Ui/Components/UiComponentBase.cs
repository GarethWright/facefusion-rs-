using Microsoft.AspNetCore.Components;

namespace FaceFusion.Ui.Components;

/// <summary>
/// Base for every component whose markup depends on <see cref="UiState"/>, including the ones
/// that only *read* a value another panel writes.
///
/// <para>
/// Python wires this explicitly: each Gradio component registers itself in
/// <c>UI_COMPONENTS</c>, and a component that needs to react to another's value looks it up and
/// attaches a <c>.change()</c> handler — <c>frame_colorizer_options.listen()</c>, for instance,
/// subscribes to the processors checkbox group so its block can appear and disappear.
/// </para>
///
/// <para>
/// <b>Why subscribing in the layout alone is not enough.</b> An initial version subscribed only
/// in <c>Default.razor</c> and let the re-render cascade down. It did not: a Blazor event
/// handler re-renders the component that owns it, so a value edited inside
/// <see cref="Controls.OptionText"/> updated the store and the run used it, while the sibling
/// panel whose <c>@if</c> depended on it kept its old markup. Ticking a processor left its
/// options block hidden and a valid target path never showed as found — the state was right and
/// the screen was wrong, which is the worst of the two. Subscribing per component is the
/// direct equivalent of Python's per-component <c>.change()</c> handlers.
/// </para>
/// </summary>
public abstract class UiComponentBase : ComponentBase, IDisposable
{
    [Inject]
    protected UiState State { get; set; } = default!;

    protected override void OnInitialized() => State.Changed += OnStateChanged;

    public virtual void Dispose()
    {
        State.Changed -= OnStateChanged;
        GC.SuppressFinalize(this);
    }

    // The change may arrive from a worker thread (a run writing state), so the re-render is
    // marshalled onto the circuit's synchronisation context.
    private void OnStateChanged() => _ = InvokeAsync(StateHasChanged);
}
