using System;
using System.Collections.Generic;

namespace FaceFusion.Core;

/// <summary>
/// Normalization functions for colors, spaces, and FPS values.
/// Ported from facefusion/normalizer.py.
/// </summary>
public static class Normalizer
{
	/// <summary>
	/// Normalize a list of color channels to a 4-tuple (R, G, B, A).
	/// - 1 channel: replicate to RGB with full alpha
	/// - 2 channels: use as R, G with alpha=255 and repeat first for B
	/// - 3 channels: use as RGB with alpha=255
	/// - 4 channels: use as-is (RGBA)
	/// </summary>
	public static (int, int, int, int)? NormalizeColor(IReadOnlyList<int>? channels)
	{
		if (channels == null)
		{
			return null;
		}

		return channels.Count switch
		{
			1 => (channels[0], channels[0], channels[0], 255),
			2 => (channels[0], channels[1], channels[0], 255),
			3 => (channels[0], channels[1], channels[2], 255),
			4 => (channels[0], channels[1], channels[2], channels[3]),
			_ => null
		};
	}

	/// <summary>
	/// Normalize a list of space values to a 4-tuple (top, right, bottom, left) padding.
	/// - 1 value: replicate to all sides
	/// - 2 values: use as vertical and horizontal (top/bottom, left/right)
	/// - 3 values: use as top, horizontal, bottom with repeat of horizontal for left
	/// - 4 values: use as-is (top, right, bottom, left)
	/// </summary>
	public static (int, int, int, int)? NormalizeSpace(IReadOnlyList<int>? spaces)
	{
		if (spaces == null)
		{
			return null;
		}

		return spaces.Count switch
		{
			1 => (spaces[0], spaces[0], spaces[0], spaces[0]),
			2 => (spaces[0], spaces[1], spaces[0], spaces[1]),
			3 => (spaces[0], spaces[1], spaces[2], spaces[1]),
			4 => (spaces[0], spaces[1], spaces[2], spaces[3]),
			_ => null
		};
	}

	/// <summary>
	/// Normalize FPS to be within [1.0, 60.0] range.
	/// </summary>
	public static double? NormalizeFps(double? fps)
	{
		if (fps == null)
		{
			return null;
		}

		return Math.Max(1.0, Math.Min(fps.Value, 60.0));
	}
}
