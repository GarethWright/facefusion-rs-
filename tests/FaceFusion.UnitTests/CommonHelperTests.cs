using FaceFusion.Core;

namespace FaceFusion.UnitTests;

public class CommonHelperTests
{
	[Fact]
	public void TestCreateIntMetavar()
	{
		Assert.Equal("[1..5:1]", CommonHelper.CreateIntMetavar(new[] { 1, 2, 3, 4, 5 }));
	}

	[Fact]
	public void TestCreateFloatMetavar()
	{
		Assert.Equal("[0.1..0.5:0.1]", CommonHelper.CreateFloatMetavar(new[] { 0.1, 0.2, 0.3, 0.4, 0.5 }));
	}

	[Fact]
	public void TestCreateIntRange()
	{
		Assert.Equal(new[] { 0, 1, 2 }, CommonHelper.CreateIntRange(0, 2, 1));
	}

	[Fact]
	public void TestCreateFloatRange()
	{
		var result1 = CommonHelper.CreateFloatRange(0.0, 1.0, 0.5);
		Assert.Equal(3, result1.Count);
		Assert.Equal(0.0, result1[0]);
		Assert.Equal(0.5, result1[1]);
		Assert.Equal(1.0, result1[2]);

		var result2 = CommonHelper.CreateFloatRange(0.0, 0.5, 0.05);
		Assert.Equal(11, result2.Count);
		Assert.Equal(0.0, result2[0]);
		Assert.Equal(0.05, result2[1]);
		Assert.Equal(0.1, result2[2]);
		Assert.Equal(0.5, result2[10]);
	}

	[Fact]
	public void TestCalculateIntStep()
	{
		Assert.Equal(1, CommonHelper.CalculateIntStep(new[] { 0, 1 }));
	}

	[Fact]
	public void TestCalculateFloatStep()
	{
		Assert.Equal(0.1, CommonHelper.CalculateFloatStep(new[] { 0.1, 0.2 }));
	}

	[Fact]
	public void TestGetMiddle()
	{
		Assert.Equal(3, CommonHelper.GetMiddle(new[] { 1, 2, 3, 4, 5 }));
		Assert.Equal(1, CommonHelper.GetMiddle(new[] { 1 }));
	}

	[Fact]
	public void TestGetFirst()
	{
		Assert.Equal(1, CommonHelper.GetFirst(new[] { 1, 2, 3 }));
		Assert.Equal(default(int?), CommonHelper.GetFirst(Array.Empty<int>()));
		Assert.Equal(default(int?), CommonHelper.GetFirst((int[]?)null));
	}

	[Fact]
	public void TestGetLast()
	{
		Assert.Equal(3, CommonHelper.GetLast(new[] { 1, 2, 3 }));
		Assert.Equal(default(int?), CommonHelper.GetLast(Array.Empty<int>()));
		Assert.Equal(default(int?), CommonHelper.GetLast((int[]?)null));
	}

	[Fact]
	public void TestCastInt()
	{
		Assert.Equal(42, CommonHelper.CastInt(42));
		Assert.Equal(42, CommonHelper.CastInt(42.0));
		Assert.Equal(42, CommonHelper.CastInt("42"));
		Assert.Null(CommonHelper.CastInt("not an int"));
		Assert.Null(CommonHelper.CastInt(null));
	}

	[Fact]
	public void TestCastFloat()
	{
		Assert.Equal(3.14, CommonHelper.CastFloat(3.14));
		Assert.Equal(42.0, CommonHelper.CastFloat(42));
		Assert.Equal(42.0, CommonHelper.CastFloat("42"));
		Assert.Equal(3.14, CommonHelper.CastFloat("3.14"));
		Assert.Null(CommonHelper.CastFloat("not a float"));
		Assert.Null(CommonHelper.CastFloat(null));
	}

	[Fact]
	public void TestCastBool()
	{
		Assert.True(CommonHelper.CastBool("True"));
		Assert.False(CommonHelper.CastBool("False"));
		Assert.Null(CommonHelper.CastBool("maybe"));
		Assert.Null(CommonHelper.CastBool(null));
	}
}
