using System.Globalization;
using System.Linq;

namespace FaceFusion.Cli;

/// <summary>
/// Typed accessors over the flat <c>IReadOnlyDictionary&lt;string, object?&gt;</c> step-args
/// bag every job step carries (Python: <c>Args = Dict[str, Any]</c>). <see cref="CliOptions"/>
/// hands values through with a small, fixed set of CLR shapes per <see cref="CliValueKind"/>
/// (<c>int?</c>, <c>double?</c>, <c>bool</c>, <c>int[]</c>, <c>string[]</c>, <c>string?</c>);
/// these helpers unbox that shape and apply the same hardcoded fallback program.py's
/// <c>config.get_xxx_value(section, option, fallback)</c> calls use when the ini/CLI value is
/// absent, so <see cref="HeadlessRunner"/>/<see cref="ProcessorStepFactory"/> read step args the
/// same way Python's <c>state_manager.get_item</c> reads a value nobody explicitly set.
/// </summary>
public static class StepArgsReader
{
	// A step args bag reaching HeadlessRunner/ProcessorStepFactory can be in one of two
	// shapes: the raw CLR types System.CommandLine hands CliOptions.Kind (int?, double?, bool,
	// int[], string[], string?) when a step runs straight off a freshly-parsed command line
	// (headless-run before it is ever written to a job file), or the plain-CLR shapes
	// JobManager.ArgsFromJsonElement hands back (string, bool, long/double, List<object?>,
	// null) after a step has round-tripped through job JSON (every job-run/job-retry path, and
	// headless-run/batch-run too — process_headless/process_batch always create + submit +
	// run a real job). Every getter below accepts both.

	public static string GetString(IReadOnlyDictionary<string, object?> args, string key, string fallback)
		=> args.TryGetValue(key, out var value) && value is string { Length: > 0 } text ? text : fallback;

	public static string? GetStringOrNull(IReadOnlyDictionary<string, object?> args, string key)
		=> args.TryGetValue(key, out var value) ? value as string : null;

	public static int GetInt(IReadOnlyDictionary<string, object?> args, string key, int fallback)
		=> args.TryGetValue(key, out var value) ? ToInt(value) ?? fallback : fallback;

	public static int? GetIntOrNull(IReadOnlyDictionary<string, object?> args, string key)
		=> args.TryGetValue(key, out var value) ? ToInt(value) : null;

	public static double GetDouble(IReadOnlyDictionary<string, object?> args, string key, double fallback)
		=> args.TryGetValue(key, out var value) ? ToDouble(value) ?? fallback : fallback;

	public static double? GetDoubleOrNull(IReadOnlyDictionary<string, object?> args, string key)
		=> args.TryGetValue(key, out var value) ? ToDouble(value) : null;

	public static bool GetBool(IReadOnlyDictionary<string, object?> args, string key)
		=> args.TryGetValue(key, out var value) && value is bool flag && flag;

	public static IReadOnlyList<string> GetStringList(IReadOnlyDictionary<string, object?> args, string key, IReadOnlyList<string> fallback)
	{
		if (!args.TryGetValue(key, out var value) || value is null)
		{
			return fallback;
		}

		var list = value switch
		{
			string[] array => array,
			IEnumerable<object?> objects => objects.Select(item => item as string).Where(item => item is not null).Cast<string>().ToArray(),
			_ => null,
		};

		return list is { Length: > 0 } ? list : fallback;
	}

	public static IReadOnlyList<int> GetIntList(IReadOnlyDictionary<string, object?> args, string key, IReadOnlyList<int> fallback)
	{
		if (!args.TryGetValue(key, out var value) || value is null)
		{
			return fallback;
		}

		var list = value switch
		{
			int[] array => array,
			IEnumerable<object?> objects => objects.Select(ToInt).Where(item => item.HasValue).Select(item => item!.Value).ToArray(),
			_ => null,
		};

		return list is { Length: > 0 } ? list : fallback;
	}

	private static int? ToInt(object? value) => value switch
	{
		int i => i,
		long l => (int)l,
		double d => (int)d,
		string s when int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) => parsed,
		_ => null,
	};

	private static double? ToDouble(object? value) => value switch
	{
		double d => d,
		int i => i,
		long l => l,
		string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) => parsed,
		_ => null,
	};

	/// <summary>Formats a value into a <c>{placeholder}</c> string.Format-style template the
	/// same way Python's <c>str.format(**kwargs)</c> does, raising the same shape of error
	/// (<c>KeyError</c>) as a <see cref="FormatUnknownPlaceholderException"/> for an unknown
	/// placeholder, so <see cref="BatchRunner"/> can reproduce Python's "bad --output-pattern
	/// returns 1" behaviour instead of throwing out of the command.</summary>
	public static string FormatPattern(string pattern, IReadOnlyDictionary<string, string?> values)
	{
		var result = new System.Text.StringBuilder();
		var index = 0;

		while (index < pattern.Length)
		{
			var open = pattern.IndexOf('{', index);

			if (open < 0)
			{
				result.Append(pattern, index, pattern.Length - index);
				break;
			}

			result.Append(pattern, index, open - index);
			var close = pattern.IndexOf('}', open);

			if (close < 0)
			{
				throw new FormatUnknownPlaceholderException(pattern[(open + 1)..]);
			}

			var placeholder = pattern[(open + 1)..close];

			if (!values.TryGetValue(placeholder, out var replacement))
			{
				throw new FormatUnknownPlaceholderException(placeholder);
			}

			result.Append(replacement);
			index = close + 1;
		}

		return result.ToString();
	}
}

/// <summary>Python: <c>KeyError</c> raised by <c>str.format(**kwargs)</c> for a placeholder not
/// present in the kwargs — <see cref="BatchRunner"/> catches this and returns error code 1,
/// matching <c>process_batch</c>'s <c>except KeyError: return 1</c>.</summary>
public sealed class FormatUnknownPlaceholderException : Exception
{
	public FormatUnknownPlaceholderException(string placeholder)
		: base($"'{placeholder}'")
	{
	}
}
