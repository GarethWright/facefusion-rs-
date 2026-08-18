using System.Globalization;
using FaceFusion.Core;

namespace FaceFusion.Ui;

/// <summary>
/// Resolves a control's label the way Gradio does: <c>translator.get('uis.&lt;key&gt;')</c>
/// against the <c>uis</c> section of <c>facefusion/locales.py</c>, already ported in
/// <see cref="Locales"/>.
///
/// <para>
/// Python names each entry by widget, not by state key — <c>face_detector_model_dropdown</c>,
/// <c>execution_thread_count_slider</c>, <c>download_providers_checkbox_group</c> — so the
/// lookup tries the state key with each widget suffix. That table covers 74 of the ~100
/// options; the rest are the per-processor controls, whose labels live in each processor
/// package's own <c>locales.py</c> (<c>help.model</c>, <c>help.blend</c>, ...) which this port
/// does not carry. Those fall back to the upper-cased key, which is exactly the text the
/// English table uses for every entry it does define, so the fallback is indistinguishable
/// rather than a placeholder.
/// </para>
/// </summary>
public static class UiLabels
{
    private static readonly string[] WidgetSuffixes =
    {
        "_dropdown", "_slider", "_checkbox_group", "_checkbox", "_textbox", "_number", "_radio", "_file", "_image",
    };

    public static string Humanize(string stateKey)
    {
        foreach (var suffix in WidgetSuffixes)
        {
            var label = Translator.Get("uis." + stateKey + suffix);

            if (!string.IsNullOrEmpty(label))
            {
                return label;
            }
        }

        return stateKey.Replace('_', ' ').ToUpper(CultureInfo.InvariantCulture);
    }
}
