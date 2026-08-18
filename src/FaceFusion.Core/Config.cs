using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace FaceFusion.Core;

/// <summary>
/// Ported from facefusion/config.py. Python uses <c>configparser.ConfigParser</c>, which is
/// not available in .NET (Microsoft.Extensions.Configuration.Ini has different comment,
/// duplicate-key, and interpolation semantics), so this is a small hand-written INI parser
/// that reproduces configparser's observed behaviour:
///
/// - Option names are lower-cased (configparser's default <c>optionxform</c>); section names
///   are case-sensitive.
/// - Comments start with <c>#</c> or <c>;</c> at the start of a (stripped) line.
/// - <c>key = value</c> and <c>key : value</c> are both accepted.
/// - A key that exists but whose value is empty or whitespace-only is treated as absent by
///   every getter below (this mirrors the <c>.strip()</c> check in every function in
///   config.py) — the fallback is used instead.
///
/// Unlike Python's module-level <c>@lru_cache</c>d <c>get_static_config_parser()</c>, this is
/// an instance built from a path so tests can construct several independent parsers without
/// sharing cached state.
/// </summary>
public sealed class Config
{
	private readonly Dictionary<string, Dictionary<string, string>> _sections;

	private Config(Dictionary<string, Dictionary<string, string>> sections)
	{
		_sections = sections;
	}

	/// <summary>
	/// Reads and parses an ini file. Mirrors <c>ConfigParser().read(path, encoding='utf-8')</c>:
	/// a missing file simply yields an empty config (configparser.read silently ignores files
	/// that do not exist).
	/// </summary>
	public static Config FromFile(string path)
	{
		var sections = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);

		if (File.Exists(path))
		{
			using var reader = new StreamReader(path, Encoding.UTF8);
			Parse(reader, sections);
		}

		return new Config(sections);
	}

	/// <summary>
	/// Parses ini text directly. Useful for tests that want to avoid touching disk, and
	/// mirrors <c>ConfigParser().read_string(text)</c>.
	/// </summary>
	public static Config FromText(string text)
	{
		var sections = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
		using var reader = new StringReader(text);
		Parse(reader, sections);
		return new Config(sections);
	}

	private static void Parse(TextReader reader, Dictionary<string, Dictionary<string, string>> sections)
	{
		string? currentSection = null;
		string? line;

		while ((line = reader.ReadLine()) != null)
		{
			var trimmed = line.Trim();

			if (trimmed.Length == 0)
			{
				continue;
			}

			if (trimmed[0] == '#' || trimmed[0] == ';')
			{
				continue;
			}

			if (trimmed[0] == '[' && trimmed[^1] == ']')
			{
				currentSection = trimmed.Substring(1, trimmed.Length - 2);
				if (!sections.ContainsKey(currentSection))
				{
					sections[currentSection] = new Dictionary<string, string>(StringComparer.Ordinal);
				}
				continue;
			}

			if (currentSection == null)
			{
				// configparser raises MissingSectionHeaderError here; ini files in this repo
				// always start with a section, so we simply skip stray lines rather than throw.
				continue;
			}

			var separatorIndex = FindSeparator(trimmed);
			if (separatorIndex < 0)
			{
				continue;
			}

			var key = trimmed.Substring(0, separatorIndex).Trim().ToLowerInvariant();
			var value = trimmed.Substring(separatorIndex + 1).Trim();
			sections[currentSection][key] = value;
		}
	}

	private static int FindSeparator(string line)
	{
		var equalsIndex = line.IndexOf('=');
		var colonIndex = line.IndexOf(':');

		if (equalsIndex < 0)
		{
			return colonIndex;
		}
		if (colonIndex < 0)
		{
			return equalsIndex;
		}
		return Math.Min(equalsIndex, colonIndex);
	}

	private bool TryGetRawValue(string section, string option, out string value)
	{
		var lowerOption = option.ToLowerInvariant();

		if (_sections.TryGetValue(section, out var options) &&
			options.TryGetValue(lowerOption, out var rawValue))
		{
			value = rawValue;
			return true;
		}

		value = string.Empty;
		return false;
	}

	/// <summary>
	/// True when the option exists AND its value is non-empty after stripping — the same
	/// "present and non-blank" check every getter below performs before reading a value.
	/// </summary>
	private bool HasNonBlankOption(string section, string option, out string value)
	{
		if (TryGetRawValue(section, option, out value) && value.Trim().Length > 0)
		{
			return true;
		}

		value = string.Empty;
		return false;
	}

	public string? GetStrValue(string section, string option, string? fallback = null)
	{
		if (HasNonBlankOption(section, option, out var value))
		{
			return value;
		}
		return fallback;
	}

	public int? GetIntValue(string section, string option, string? fallback = null)
	{
		if (HasNonBlankOption(section, option, out var value))
		{
			return ParseConfigInt(value);
		}
		return CommonHelper.CastInt(fallback);
	}

	public double? GetFloatValue(string section, string option, string? fallback = null)
	{
		if (HasNonBlankOption(section, option, out var value))
		{
			return ParseConfigFloat(value);
		}
		return CommonHelper.CastFloat(fallback);
	}

	public bool? GetBoolValue(string section, string option, string? fallback = null)
	{
		if (HasNonBlankOption(section, option, out var value))
		{
			return ParseConfigBool(value);
		}
		return CommonHelper.CastBool(fallback);
	}

	/// <summary>
	/// Splits on arbitrary whitespace, matching Python's argument-less <c>str.split()</c>
	/// (runs of spaces/tabs/newlines collapse, and leading/trailing whitespace is stripped) —
	/// not a single-space split.
	/// </summary>
	public IReadOnlyList<string>? GetStrList(string section, string option, string? fallback = null)
	{
		if (HasNonBlankOption(section, option, out var value))
		{
			return SplitWhitespace(value);
		}
		if (!string.IsNullOrEmpty(fallback))
		{
			return SplitWhitespace(fallback);
		}
		return null;
	}

	public IReadOnlyList<int>? GetIntList(string section, string option, string? fallback = null)
	{
		if (HasNonBlankOption(section, option, out var value))
		{
			return SplitWhitespaceAsInt(value);
		}
		if (!string.IsNullOrEmpty(fallback))
		{
			return SplitWhitespaceAsInt(fallback);
		}
		return null;
	}

	private static IReadOnlyList<string> SplitWhitespace(string value)
	{
		return value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
	}

	private static IReadOnlyList<int> SplitWhitespaceAsInt(string value)
	{
		var parts = SplitWhitespace(value);
		var result = new int[parts.Count];
		for (var i = 0; i < parts.Count; i++)
		{
			// Mirrors Python's int(str) — throws on failure, same as config.py's
			// `list(map(int, ...))`, which is not guarded by cast_int.
			result[i] = int.Parse(parts[i], CultureInfo.InvariantCulture);
		}
		return result;
	}

	private static int ParseConfigInt(string value)
	{
		// Mirrors ConfigParser.getint, which is not guarded against bad values either
		// (config.py calls it directly, not through cast_int).
		return int.Parse(value, CultureInfo.InvariantCulture);
	}

	private static double ParseConfigFloat(string value)
	{
		return double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
	}

	private static readonly HashSet<string> TrueValues = new(StringComparer.OrdinalIgnoreCase)
	{
		"1", "yes", "true", "on"
	};

	private static readonly HashSet<string> FalseValues = new(StringComparer.OrdinalIgnoreCase)
	{
		"0", "no", "false", "off"
	};

	/// <summary>
	/// Ported from ConfigParser.BOOLEAN_STATES / getboolean: case-insensitively accepts
	/// 1/yes/true/on and 0/no/false/off, and raises on anything else.
	/// </summary>
	private static bool ParseConfigBool(string value)
	{
		if (TrueValues.Contains(value))
		{
			return true;
		}
		if (FalseValues.Contains(value))
		{
			return false;
		}
		throw new FormatException($"Not a boolean: {value}");
	}
}
