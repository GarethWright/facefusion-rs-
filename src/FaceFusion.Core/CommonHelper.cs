using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace FaceFusion.Core;

/// <summary>
/// Common helper functions for platform detection and range/value operations.
/// Ported from facefusion/common_helper.py.
/// </summary>
public static class CommonHelper
{
	public static bool IsLinux()
	{
		return OperatingSystem.IsLinux();
	}

	public static bool IsMacOS()
	{
		return OperatingSystem.IsMacOS();
	}

	public static bool IsWindows()
	{
		return OperatingSystem.IsWindows();
	}

	public static string CreateIntMetavar(IReadOnlyList<int> intRange)
	{
		if (intRange == null || intRange.Count == 0)
		{
			throw new ArgumentException("int_range must not be empty", nameof(intRange));
		}

		var step = CalculateIntStep(intRange);
		return $"[{intRange[0]}..{intRange[intRange.Count - 1]}:{step}]";
	}

	public static string CreateFloatMetavar(IReadOnlyList<double> floatRange)
	{
		if (floatRange == null || floatRange.Count == 0)
		{
			throw new ArgumentException("float_range must not be empty", nameof(floatRange));
		}

		var step = CalculateFloatStep(floatRange);
		return $"[{floatRange[0].ToString(CultureInfo.InvariantCulture)}..{floatRange[floatRange.Count - 1].ToString(CultureInfo.InvariantCulture)}:{step.ToString(CultureInfo.InvariantCulture)}]";
	}

	public static IReadOnlyList<int> CreateIntRange(int start, int end, int step)
	{
		var intRange = new List<int>();
		var current = start;

		while (current <= end)
		{
			intRange.Add(current);
			current += step;
		}

		return intRange;
	}

	public static IReadOnlyList<double> CreateFloatRange(double start, double end, double step)
	{
		var floatRange = new List<double>();
		var current = start;

		while (current <= end)
		{
			floatRange.Add(Math.Round(current, 2));
			current = Math.Round(current + step, 2);
		}

		return floatRange;
	}

	public static int CalculateIntStep(IReadOnlyList<int> intRange)
	{
		if (intRange == null || intRange.Count < 2)
		{
			throw new ArgumentException("int_range must have at least 2 elements", nameof(intRange));
		}

		return intRange[1] - intRange[0];
	}

	public static double CalculateFloatStep(IReadOnlyList<double> floatRange)
	{
		if (floatRange == null || floatRange.Count < 2)
		{
			throw new ArgumentException("float_range must have at least 2 elements", nameof(floatRange));
		}

		return Math.Round(floatRange[1] - floatRange[0], 2);
	}

	public static int? CastInt(object? value)
	{
		try
		{
			if (value == null)
			{
				return null;
			}

			return value switch
			{
				int i => i,
				double d => (int)d,
				float f => (int)f,
				string s => int.Parse(s, CultureInfo.InvariantCulture),
				_ => int.Parse(value.ToString() ?? "", CultureInfo.InvariantCulture)
			};
		}
		catch (FormatException)
		{
			return null;
		}
		catch (OverflowException)
		{
			return null;
		}
	}

	public static double? CastFloat(object? value)
	{
		try
		{
			if (value == null)
			{
				return null;
			}

			return value switch
			{
				double d => d,
				float f => f,
				int i => i,
				string s => double.Parse(s, CultureInfo.InvariantCulture),
				_ => double.Parse(value.ToString() ?? "", CultureInfo.InvariantCulture)
			};
		}
		catch (FormatException)
		{
			return null;
		}
		catch (OverflowException)
		{
			return null;
		}
	}

	public static bool? CastBool(object? value)
	{
		if (value == null)
		{
			return null;
		}

		if (value is bool b)
		{
			return b;
		}

		var strValue = value.ToString();
		return strValue switch
		{
			"True" => true,
			"False" => false,
			_ => null
		};
	}

	/// <summary>
	/// Get the first item from an enumerable. Returns null if empty or null.
	/// IMPORTANT: For value types (struct T), this returns default(T), NOT null.
	/// For value types, use GetFirstOrNull instead to get null on empty.
	/// </summary>
	public static T? GetFirst<T>(IEnumerable<T>? list)
	{
		if (list != null)
		{
			foreach (var item in list)
			{
				return item;
			}
		}

		return default;
	}

	/// <summary>
	/// Get the first item from an enumerable of value types.
	/// Returns null (as Nullable{T}) if empty or null.
	/// </summary>
	public static T? GetFirstOrNull<T>(IEnumerable<T>? list) where T : struct
	{
		if (list != null)
		{
			foreach (var item in list)
			{
				return item;
			}
		}

		return null;
	}

	/// <summary>
	/// Get the middle item from a list. Returns null if empty or null.
	/// For a list with odd length, returns the item at position length/2.
	/// IMPORTANT: For value types (struct T), this returns default(T), NOT null.
	/// For value types, use GetMiddleOrNull instead to get null on empty.
	/// </summary>
	public static T? GetMiddle<T>(IReadOnlyList<T>? list)
	{
		if (list != null && list.Count > 0)
		{
			return list[list.Count / 2];
		}

		return default;
	}

	/// <summary>
	/// Get the middle item from a list of value types.
	/// Returns null (as Nullable{T}) if empty or null.
	/// For a list with odd length, returns the item at position length/2.
	/// </summary>
	public static T? GetMiddleOrNull<T>(IReadOnlyList<T>? list) where T : struct
	{
		if (list != null && list.Count > 0)
		{
			return list[list.Count / 2];
		}

		return null;
	}

	/// <summary>
	/// Get the last item from an enumerable. Returns null if empty or null.
	/// IMPORTANT: For value types (struct T), this returns default(T), NOT null.
	/// For value types, use GetLastOrNull instead to get null on empty.
	/// </summary>
	public static T? GetLast<T>(IEnumerable<T>? list)
	{
		if (list == null)
		{
			return default;
		}

		// For IReadOnlyList, use direct indexing (efficient)
		if (list is IReadOnlyList<T> readOnlyList && readOnlyList.Count > 0)
		{
			return readOnlyList[readOnlyList.Count - 1];
		}

		// For other enumerables, iterate to the last item
		T? last = default;
		foreach (var item in list)
		{
			last = item;
		}

		return last;
	}

	/// <summary>
	/// Get the last item from an enumerable of value types.
	/// Returns null (as Nullable{T}) if empty or null.
	/// </summary>
	public static T? GetLastOrNull<T>(IEnumerable<T>? list) where T : struct
	{
		if (list == null)
		{
			return null;
		}

		// For IReadOnlyList, use direct indexing (efficient)
		if (list is IReadOnlyList<T> readOnlyList && readOnlyList.Count > 0)
		{
			return readOnlyList[readOnlyList.Count - 1];
		}

		// For other enumerables, iterate to the last item
		T? last = null;
		foreach (var item in list)
		{
			last = item;
		}

		return last;
	}
}
