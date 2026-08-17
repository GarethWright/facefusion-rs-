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
		// Test with reference type - returns null on empty
		Assert.Null(CommonHelper.GetFirst(Array.Empty<string>()));
		Assert.Null(CommonHelper.GetFirst((string[]?)null));

		// Test with value type unconstrained - returns default(T) on empty (0 for int)
		Assert.Equal(1, CommonHelper.GetFirst(new[] { 1, 2, 3 }));
		Assert.Equal(0, CommonHelper.GetFirst(Array.Empty<int>()));

		// Test with value type OrNull variant - returns null on empty
		Assert.Equal(1, CommonHelper.GetFirstOrNull(new[] { 1, 2, 3 }));
		Assert.Null(CommonHelper.GetFirstOrNull(Array.Empty<int>()));
		Assert.Null(CommonHelper.GetFirstOrNull((int[]?)null));
	}

	[Fact]
	public void TestGetLast()
	{
		// Test with reference type - returns null on empty
		Assert.Null(CommonHelper.GetLast(Array.Empty<string>()));
		Assert.Null(CommonHelper.GetLast((string[]?)null));

		// Test with value type unconstrained - returns default(T) on empty (0 for int)
		Assert.Equal(3, CommonHelper.GetLast(new[] { 1, 2, 3 }));
		Assert.Equal(0, CommonHelper.GetLast(Array.Empty<int>()));

		// Test with value type OrNull variant - returns null on empty
		Assert.Equal(3, CommonHelper.GetLastOrNull(new[] { 1, 2, 3 }));
		Assert.Null(CommonHelper.GetLastOrNull(Array.Empty<int>()));
		Assert.Null(CommonHelper.GetLastOrNull((int[]?)null));
	}

	[Fact]
	public void TestGetMiddleOrNull()
	{
		// Test with value type OrNull variant
		Assert.Equal(3, CommonHelper.GetMiddleOrNull(new[] { 1, 2, 3, 4, 5 }));
		Assert.Equal(1, CommonHelper.GetMiddleOrNull(new[] { 1 }));
		Assert.Null(CommonHelper.GetMiddleOrNull(Array.Empty<int>()));
		Assert.Null(CommonHelper.GetMiddleOrNull((int[]?)null));
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
