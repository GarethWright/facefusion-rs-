using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace FaceFusion.Core;

/// <summary>
/// Sanitization functions for job IDs and value ranges.
/// Ported from facefusion/sanitizer.py.
/// </summary>
public static class Sanitizer
{
	/// <summary>
	/// Sanitize a job ID to ensure it is alphanumeric. If the job_id is already
	/// alphanumeric (after removing dashes), return it unchanged.
	/// Otherwise, return the SHA1 hash of the job_id in hex format.
	/// </summary>
	public static string SanitizeJobId(string jobId)
	{
		if (jobId == null)
		{
			throw new ArgumentNullException(nameof(jobId));
		}

		var cleanedJobId = jobId.Replace("-", "");

		if (cleanedJobId.All(char.IsLetterOrDigit))
		{
			return jobId;
		}

		using (var sha1 = SHA1.Create())
		{
			var hash = sha1.ComputeHash(Encoding.UTF8.GetBytes(jobId));
			return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
		}
	}

	/// <summary>
	/// Sanitize an integer value to be within a given range.
	/// If the value is within the range, return it; otherwise return the first value of the range.
	/// </summary>
	public static int SanitizeIntRange(object? value, IReadOnlyList<int> intRange)
	{
		if (intRange == null || intRange.Count == 0)
		{
			throw new ArgumentException("int_range must not be empty", nameof(intRange));
		}

		var castValue = CommonHelper.CastInt(value);

		if (castValue.HasValue && intRange.Contains(castValue.Value))
		{
			return castValue.Value;
		}

		return intRange[0];
	}
}
