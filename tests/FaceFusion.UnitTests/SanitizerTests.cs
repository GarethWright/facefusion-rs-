using FaceFusion.Core;

namespace FaceFusion.UnitTests;

public class SanitizerTests
{
	[Fact]
	public void TestSanitizeIntRange()
	{
		var range = new[] { 0, 1, 2 };

		// Value within range is returned
		Assert.Equal(0, Sanitizer.SanitizeIntRange(0, range));
		Assert.Equal(2, Sanitizer.SanitizeIntRange(2, range));

		// Value outside range returns first value
		Assert.Equal(0, Sanitizer.SanitizeIntRange(-1, range));
		Assert.Equal(0, Sanitizer.SanitizeIntRange(3, range));

		// Test with different range
		var range2 = new[] { 0, 1 };
		Assert.Equal(0, Sanitizer.SanitizeIntRange(2, range2));
	}

	[Fact]
	public void TestSanitizeJobIdAlphanumeric()
	{
		// Job ID that is already alphanumeric (after removing dashes) should be returned as-is
		var jobId = "abc-123-def";
		Assert.Equal(jobId, Sanitizer.SanitizeJobId(jobId));

		var jobId2 = "abcdef123";
		Assert.Equal(jobId2, Sanitizer.SanitizeJobId(jobId2));
	}

	[Fact]
	public void TestSanitizeJobIdNonAlphanumeric()
	{
		// Job ID with non-alphanumeric characters should be hashed with SHA1
		var jobId = "invalid@job#id";
		var result = Sanitizer.SanitizeJobId(jobId);

		// SHA1 hash should be 40 hex characters
		Assert.Equal(40, result.Length);
		Assert.Matches("^[a-f0-9]+$", result);

		// Same input should produce same hash
		var result2 = Sanitizer.SanitizeJobId(jobId);
		Assert.Equal(result, result2);
	}
}
