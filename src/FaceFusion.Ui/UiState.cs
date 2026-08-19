using System.Globalization;
using FaceFusion.Cli;

namespace FaceFusion.Ui;

/// <summary>
/// Port of <c>facefusion/state_manager.py</c> as the UI uses it, and of the
/// <c>UI_COMPONENTS</c> registry's purpose: one place every component reads and writes, so a
/// change in one panel is visible to another without them referencing each other.
///
/// <para>
/// <b>Why this is global state when the rest of the port refuses it.</b> PORT_CONVENTIONS.md
/// rule 5 bans global mutable state because a *library* that reads hidden globals cannot be
/// tested or run concurrently. A UI's control values are genuinely shared mutable state — they
/// are what the user is editing — and Gradio models them the same way. The important part is
/// that this store stops at the UI boundary: <see cref="BuildArgs"/> materialises a plain args
/// dictionary and hands it to <see cref="HeadlessRunner"/>, which is the same flat bag the CLI
/// builds from <c>argv</c>. Nothing below <c>FaceFusion.Ui</c> can see this class.
/// </para>
///
/// <para>
/// <b>Values are stored as strings,</b> exactly as the CLI receives them from argv, and parsed
/// on the way out by the same <see cref="CliValueKind"/> rules. That is deliberate: a UI that
/// parsed early would have two parsers to keep in step with Python's, and the difference would
/// show up as a run that behaves differently depending on whether it was started from the UI or
/// the CLI. Every default comes from <see cref="UiOptionDescriptors"/>, which is generated from
/// the real argparse parser.
/// </para>
/// </summary>
public sealed class UiState
{
    private readonly Dictionary<string, string?> _values = new(StringComparer.Ordinal);

    public UiState()
    {
        Reset();
    }

    /// <summary>Raised after any value changes, so panels that depend on another panel's value
    /// (Python: the <c>UI_COMPONENTS</c> cross-wiring, e.g. the processor checkboxes showing and
    /// hiding each processor's option block) can re-render.</summary>
    public event Action? Changed;

    /// <summary>Restores every option to the default Python's parser resolves.</summary>
    public void Reset()
    {
        _values.Clear();

        foreach (var descriptor in UiOptionDescriptors.All)
        {
            _values[descriptor.StateKey] = descriptor.Default;
        }

        Changed?.Invoke();
    }

    public string? GetString(string stateKey)
        => _values.TryGetValue(stateKey, out var value) ? value : null;

    public void SetString(string stateKey, string? value)
    {
        // Guard against a typo in a Razor file binding to a key that does not exist, which
        // would otherwise silently do nothing at all.
        _ = UiOptionDescriptors.Get(stateKey);

        if (_values.TryGetValue(stateKey, out var existing) && existing == value)
        {
            return;
        }

        _values[stateKey] = value;
        Changed?.Invoke();
    }

    public double GetDouble(string stateKey)
        => double.TryParse(GetString(stateKey), NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ? value : 0.0;

    public void SetDouble(string stateKey, double value)
        => SetString(stateKey, value.ToString("0.####", CultureInfo.InvariantCulture));

    public int GetInt(string stateKey)
        => int.TryParse(GetString(stateKey), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) ? value : 0;

    public void SetInt(string stateKey, int value)
        => SetString(stateKey, value.ToString(CultureInfo.InvariantCulture));

    public bool GetFlag(string stateKey)
        => string.Equals(GetString(stateKey), "true", StringComparison.OrdinalIgnoreCase);

    public void SetFlag(string stateKey, bool value)
        => SetString(stateKey, value ? "true" : "false");

    /// <summary>A space-separated list value split the way argparse's <c>nargs='+'</c> receives
    /// it from a shell.</summary>
    public IReadOnlyList<string> GetList(string stateKey)
    {
        var value = GetString(stateKey);
        return string.IsNullOrWhiteSpace(value)
            ? Array.Empty<string>()
            : value.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    public void SetList(string stateKey, IEnumerable<string> values)
        => SetString(stateKey, string.Join(' ', values));

    /// <summary>Adds or removes one entry of a multi-select list, keeping the descriptor's own
    /// choice order rather than click order — Gradio's CheckboxGroup does the same, and the
    /// order reaches the CLI (e.g. <c>--face-mask-types</c> is order-sensitive downstream).</summary>
    public void ToggleListValue(string stateKey, string value, bool selected)
    {
        var descriptor = UiOptionDescriptors.Get(stateKey);
        var current = GetList(stateKey).ToHashSet(StringComparer.Ordinal);

        if (selected)
        {
            current.Add(value);
        }
        else
        {
            current.Remove(value);
        }

        var ordered = descriptor.Choices.Count > 0
            ? descriptor.Choices.Where(current.Contains)
            : current.AsEnumerable();

        SetList(stateKey, ordered);
    }

    /// <summary>
    /// Materialises the flat args bag <see cref="HeadlessRunner.ProcessHeadless"/> expects —
    /// the same shape <c>CliCommands</c> builds from argv, so a run started here and the
    /// equivalent <c>headless-run</c> command reach identical code with identical values.
    /// Keys whose value is still null (an unset optional, e.g. <c>face_selector_gender</c>) are
    /// omitted rather than sent as null, matching argparse leaving them out.
    /// </summary>
    public Dictionary<string, object?> BuildArgs()
    {
        var args = new Dictionary<string, object?>(StringComparer.Ordinal);

        foreach (var descriptor in UiOptionDescriptors.All)
        {
            var raw = GetString(descriptor.StateKey);

            if (string.IsNullOrWhiteSpace(raw))
            {
                continue;
            }

            args[descriptor.StateKey] = descriptor.Kind switch
            {
                CliValueKind.Int => int.Parse(raw, CultureInfo.InvariantCulture),
                CliValueKind.Float => double.Parse(raw, CultureInfo.InvariantCulture),
                CliValueKind.Flag => string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase),
                CliValueKind.StringList => raw.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
                CliValueKind.IntList => raw.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(item => int.Parse(item, CultureInfo.InvariantCulture)).ToArray(),
                _ => raw,
            };
        }

        return args;
    }
}
