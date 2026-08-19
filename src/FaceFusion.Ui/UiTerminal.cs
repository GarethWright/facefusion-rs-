using System.Text;

namespace FaceFusion.Ui;

/// <summary>
/// Port of <c>facefusion/uis/components/terminal.py</c>: the log pane. Python redirects the
/// logger's output into a Gradio textbox; this is the same idea with a
/// <see cref="TextWriter"/>, which is exactly what <c>FaceFusion.Core.Logger</c>'s constructor
/// already accepts — so the UI gets the identical stream the CLI prints, not a re-implementation
/// of it.
/// </summary>
public sealed class UiTerminal : TextWriter
{
    private const int MaxLines = 500;

    // System.Threading.Lock is .NET 9+; net8.0 locks on a plain object (see PORT_CONVENTIONS.md
    // on why this repo targets net8.0).
    private readonly object _lock = new();
    private readonly Queue<string> _lines = new();
    private readonly StringBuilder _pending = new();

    public override Encoding Encoding => Encoding.UTF8;

    /// <summary>Raised on every completed line so the pane can re-render. Handlers run on
    /// whichever thread wrote the log line — a Blazor component must marshal via
    /// <c>InvokeAsync</c>.</summary>
    public event Action? LineWritten;

    public override void Write(char value)
    {
        string? completed = null;

        lock (_lock)
        {
            if (value == '\n')
            {
                completed = _pending.ToString().TrimEnd('\r');
                _pending.Clear();
                _lines.Enqueue(completed);

                while (_lines.Count > MaxLines)
                {
                    _lines.Dequeue();
                }
            }
            else
            {
                _pending.Append(value);
            }
        }

        if (completed is not null)
        {
            LineWritten?.Invoke();
        }
    }

    public IReadOnlyList<string> Snapshot()
    {
        lock (_lock)
        {
            return _lines.ToArray();
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _lines.Clear();
            _pending.Clear();
        }

        LineWritten?.Invoke();
    }
}
