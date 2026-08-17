using System;
using System.Collections.Generic;
using System.Reflection;

namespace FaceFusion.Types;

/// <summary>
/// Generic string round-tripping for enums that were ported from Python
/// <c>Literal['a', 'b', ...]</c> string unions. Every enum member must carry a
/// <see cref="WireNameAttribute"/>; the attribute value is the exact string that appears on
/// CLI args, in <c>facefusion.ini</c>, and in job JSON — this is the single place that
/// conversion is implemented, rather than a hand-written switch per enum.
/// </summary>
public static class EnumNames
{
	private static class Cache<T> where T : struct, Enum
	{
		public static readonly Dictionary<T, string> ToWireMap = new();
		public static readonly Dictionary<string, T> FromWireMap = new(StringComparer.Ordinal);

		static Cache()
		{
			foreach (var field in typeof(T).GetFields(BindingFlags.Public | BindingFlags.Static))
			{
				var value = (T)field.GetValue(null)!;
				var attribute = field.GetCustomAttribute<WireNameAttribute>();

				if (attribute == null)
				{
					throw new InvalidOperationException(
						$"Enum member {typeof(T).Name}.{field.Name} is missing a [WireName] attribute.");
				}

				ToWireMap[value] = attribute.Name;
				FromWireMap[attribute.Name] = value;
			}
		}
	}

	/// <summary>
	/// Returns the exact wire string for an enum value, as it would appear on the CLI, in
	/// facefusion.ini, or in job JSON.
	/// </summary>
	public static string ToWireName<T>(this T value) where T : struct, Enum
	{
		if (Cache<T>.ToWireMap.TryGetValue(value, out var name))
		{
			return name;
		}

		throw new ArgumentOutOfRangeException(nameof(value), value, $"No wire name registered for {typeof(T).Name}.{value}.");
	}

	/// <summary>
	/// Parses a wire string (case-sensitive, exact match) back into its enum value.
	/// Throws <see cref="ArgumentException"/> when the string does not match any member.
	/// </summary>
	public static T FromWireName<T>(string name) where T : struct, Enum
	{
		if (TryFromWireName<T>(name, out var value))
		{
			return value;
		}

		throw new ArgumentException($"'{name}' is not a valid wire name for {typeof(T).Name}.", nameof(name));
	}

	/// <summary>
	/// Attempts to parse a wire string (case-sensitive, exact match) back into its enum value.
	/// </summary>
	public static bool TryFromWireName<T>(string name, out T value) where T : struct, Enum
	{
		return Cache<T>.FromWireMap.TryGetValue(name, out value);
	}

	/// <summary>
	/// All wire names for an enum type, in declaration order — the C# equivalent of Python's
	/// <c>list(get_args(SomeLiteral))</c>.
	/// </summary>
	public static IReadOnlyList<string> AllWireNames<T>() where T : struct, Enum
	{
		var names = new List<string>();

		foreach (var value in Enum.GetValues<T>())
		{
			names.Add(value.ToWireName());
		}

		return names;
	}
}
