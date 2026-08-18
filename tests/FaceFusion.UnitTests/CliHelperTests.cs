using FaceFusion.Cli;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the table rendering in <c>facefusion/cli_helper.py</c>. Expected strings are
/// real output from <c>create_table_parts</c>, captured with:
///   python3 -c "from facefusion.cli_helper import create_table_parts; ..."
/// This is user-visible output for <c>job-list</c>, so it is pinned exactly rather than
/// checked loosely.
/// </summary>
public sealed class CliHelperTests
{
    private static readonly string[] Headers =
        { "job id", "status", "date created", "date updated", "steps" };

    private static readonly IReadOnlyList<IReadOnlyList<object?>> Contents = new[]
    {
        new object?[] { "job-alpha", "drafted", "2026-08-18 12:00:00", null, 3 },
        new object?[] { "j", "queued", "x", null, 10 }
    };

    [Fact]
    public void ComposeTableMatchesPython()
    {
        var lines = CliHelper.ComposeTable(Headers, Contents);

        Assert.Equal("+-----------+---------+---------------------+--------------+-------+", lines[0]);
        Assert.Equal("| job id    | status  | date created        | date updated | steps |", lines[1]);
        Assert.Equal("+-----------+---------+---------------------+--------------+-------+", lines[2]);
        Assert.Equal("| job-alpha | drafted | 2026-08-18 12:00:00 | None         | 3     |", lines[3]);
        Assert.Equal("| j         | queued  | x                   | None         | 10    |", lines[4]);
        Assert.Equal("+-----------+---------+---------------------+--------------+-------+", lines[5]);
    }

    /// <summary>
    /// Column widths come from the widest value, not the header, when a value is longer.
    /// </summary>
    [Fact]
    public void ColumnWidthFollowsWidestValue()
    {
        var (column, separator) = CliHelper.CreateTableParts(
            new[] { "a" },
            new[] { new object?[] { "a-much-longer-value" } });

        Assert.Equal("+---------------------+", separator);
        Assert.Equal("| {0,-19} |", column);
    }

    /// <summary>Python renders None as "None" and booleans as "True"/"False" via str().</summary>
    [Fact]
    public void RendersPythonSpellingsForNullAndBool()
    {
        var lines = CliHelper.ComposeTable(
            new[] { "value" },
            new[] { new object?[] { null }, new object?[] { true }, new object?[] { false } });

        Assert.Equal("| None  |", lines[3]);
        Assert.Equal("| True  |", lines[4]);
        Assert.Equal("| False |", lines[5]);
    }
}
