using FaceFusion.Core;

namespace FaceFusion.UnitTests;

public class TimeHelperTests
{
	private DateTime GetTimeAgo(int days, int hours, int minutes)
	{
		return DateTime.Now - TimeSpan.FromDays(days) - TimeSpan.FromHours(hours) - TimeSpan.FromMinutes(minutes);
	}

	[Fact]
	public void TestDescribeTimeAgoJustNow()
	{
		var result = TimeHelper.DescribeTimeAgo(GetTimeAgo(0, 0, 0));
		Assert.Equal("just now", result);
	}

	[Fact]
	public void TestDescribeTimeAgoMinutes()
	{
		var result = TimeHelper.DescribeTimeAgo(GetTimeAgo(0, 0, 10));
		Assert.Equal("10 minutes ago", result);
	}

	[Fact]
	public void TestDescribeTimeAgoHours()
	{
		var result = TimeHelper.DescribeTimeAgo(GetTimeAgo(0, 5, 10));
		Assert.Equal("5 hours and 10 minutes ago", result);
	}

	[Fact]
	public void TestDescribeTimeAgoDays()
	{
		var result = TimeHelper.DescribeTimeAgo(GetTimeAgo(1, 5, 10));
		Assert.Equal("1 days, 5 hours and 10 minutes ago", result);
	}

	[Fact]
	public void TestSplitTimeDelta()
	{
		var timeDelta = TimeSpan.FromDays(1) + TimeSpan.FromHours(5) + TimeSpan.FromMinutes(30) + TimeSpan.FromSeconds(45);
		var (days, hours, minutes, seconds) = TimeHelper.SplitTimeDelta(timeDelta);

		Assert.Equal(1, days);
		Assert.Equal(5, hours);
		Assert.Equal(30, minutes);
		Assert.Equal(45, seconds);
	}

	[Fact]
	public void TestSplitTimeDeltaZero()
	{
		var (days, hours, minutes, seconds) = TimeHelper.SplitTimeDelta(TimeSpan.Zero);

		Assert.Equal(0, days);
		Assert.Equal(0, hours);
		Assert.Equal(0, minutes);
		Assert.Equal(0, seconds);
	}

	[Fact]
	public void TestGetCurrentDateTime()
	{
		var before = DateTime.Now;
		var result = TimeHelper.GetCurrentDateTime();
		var after = DateTime.Now;

		Assert.True(result >= before);
		Assert.True(result <= after);
	}
}
