using System.Text;

namespace FaceFusion.Cli;

/// <summary>
/// Port of <c>facefusion/cli_helper.py</c>. Renders the ASCII table used by
/// <c>job-list</c>.
/// </summary>
public static class CliHelper
{
    /// <summary>
    /// Python: <c>create_table_parts</c>. Returns the row format string and the separator
    /// line. Column widths are the widest of the header and every value in that column.
    /// </summary>
    public static (string Column, string Separator) CreateTableParts(
        IReadOnlyList<string> headers,
        IReadOnlyList<IReadOnlyList<object?>> contents)
    {
        var widths = headers.Select(header => header.Length).ToArray();

        foreach (var content in contents)
        {
            for (var index = 0; index < content.Count && index < widths.Length; index++)
            {
                widths[index] = Math.Max(widths[index], Stringify(content[index]).Length);
            }
        }

        var columnParts = widths.Select(width => "{0,-" + width + "}").ToArray();
        var separatorParts = widths.Select(width => new string('-', width));

        // Python builds '{:<width}' placeholders and formats positionally; the C#
        // equivalent needs explicit indices, so they are renumbered here.
        var column = new StringBuilder("| ");

        for (var index = 0; index < columnParts.Length; index++)
        {
            column.Append(columnParts[index].Replace("{0,", "{" + index + ",", StringComparison.Ordinal));
            column.Append(index == columnParts.Length - 1 ? " |" : " | ");
        }

        return (column.ToString(), "+-" + string.Join("-+-", separatorParts) + "-+");
    }

    /// <summary>Python: <c>render_table</c>.</summary>
    public static IReadOnlyList<string> ComposeTable(
        IReadOnlyList<string> headers,
        IReadOnlyList<IReadOnlyList<object?>> contents)
    {
        var (column, separator) = CreateTableParts(headers, contents);
        var lines = new List<string>
        {
            separator,
            string.Format(System.Globalization.CultureInfo.InvariantCulture, column, headers.Cast<object?>().ToArray()),
            separator
        };

        foreach (var content in contents)
        {
            var values = content.Select(value => (object?)Stringify(value)).ToArray();
            lines.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture, column, values));
        }

        lines.Add(separator);
        return lines;
    }

    /// <summary>
    /// Python renders None as the string "None" via str(); reproduced so a job list with
    /// an unset date column looks the same on both implementations.
    /// </summary>
    private static string Stringify(object? value) => value switch
    {
        null => "None",
        bool boolean => boolean ? "True" : "False",
        _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty
    };
}
