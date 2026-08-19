using System;

namespace FaceFusion.Core;

/// <summary>
/// Time-related helper functions for date/time calculations and formatting.
/// Ported from facefusion/time_helper.py.
/// </summary>
public static class TimeHelper
{
	/// <summary>
	/// Get the current date and time with timezone information.
	/// </summary>
	public static DateTime GetCurrentDateTime()
	{
		return DateTime.Now;
	}

	/// <summary>
	/// Calculate the elapsed time in seconds (rounded to 2 decimal places)
	/// from a start time to the current time.
	/// </summary>
	public static double CalculateEndTime(double startTime)
	{
		var elapsed = DateTimeOffset.Now.ToUnixTimeMilliseconds() / 1000.0 - startTime;
		return Math.Round(elapsed, 2);
	}

	/// <summary>
	/// Split a TimeSpan into days, hours, minutes, and seconds.
	/// </summary>
	public static (int Days, int Hours, int Minutes, int Seconds) SplitTimeDelta(TimeSpan timeDelta)
	{
		var totalSeconds = (long)timeDelta.TotalSeconds;

		var days = (int)(totalSeconds / 86400);
		totalSeconds %= 86400;

		var hours = (int)(totalSeconds / 3600);
		totalSeconds %= 3600;

		var minutes = (int)(totalSeconds / 60);
		var seconds = (int)(totalSeconds % 60);

		return (days, hours, minutes, seconds);
	}

	/// <summary>
	/// Describe how long ago a datetime was in a human-readable format.
	/// Returns null if the time is in the future or exactly now.
	/// Uses hardcoded English strings (originally from locales.py).
	/// </summary>
	public static string? DescribeTimeAgo(DateTime dateTime)
	{
		var now = DateTime.Now;
		// If the dateTime has timezone info, use Now with timezone, otherwise use local Now
		if (dateTime.Kind != DateTimeKind.Unspecified)
		{
			now = DateTime.Now;
		}

		var timeAgo = now - dateTime;

		if (timeAgo <= TimeSpan.Zero)
		{
			return null;
		}

		var (days, hours, minutes, _) = SplitTimeDelta(timeAgo);

		if (timeAgo > TimeSpan.FromDays(1))
		{
			return $"{days} days, {hours} hours and {minutes} minutes ago";
		}

		if (timeAgo > TimeSpan.FromHours(1))
		{
			return $"{hours} hours and {minutes} minutes ago";
		}

		if (timeAgo > TimeSpan.FromMinutes(1))
		{
			return $"{minutes} minutes ago";
		}

		return "just now";
	}
}
