using FaceFusion.Core;

namespace FaceFusion.UnitTests;

public class NormalizerTests
{
	[Fact]
	public void TestNormalizeColor()
	{
		// 1 channel: replicate to RGB with alpha=255
		Assert.Equal((0, 0, 0, 255), Normalizer.NormalizeColor(new[] { 0 }));

		// 2 channels: R, G, R with alpha=255
		Assert.Equal((0, 128, 0, 255), Normalizer.NormalizeColor(new[] { 0, 128 }));

		// 3 channels: RGB with alpha=255
		Assert.Equal((0, 128, 255, 255), Normalizer.NormalizeColor(new[] { 0, 128, 255 }));

		// 4 channels: RGBA as-is
		Assert.Equal((0, 128, 255, 0), Normalizer.NormalizeColor(new[] { 0, 128, 255, 0 }));

		// null: return null
		Assert.Null(Normalizer.NormalizeColor(null));

		// 0 or invalid length: return null
		Assert.Null(Normalizer.NormalizeColor(new int[] { }));
		Assert.Null(Normalizer.NormalizeColor(new[] { 0, 128, 255, 0, 255 }));
	}

	[Fact]
	public void TestNormalizeSpace()
	{
		// 1 value: replicate to all 4 sides
		Assert.Equal((1, 1, 1, 1), Normalizer.NormalizeSpace(new[] { 1 }));

		// 2 values: vertical and horizontal (top/bottom, left/right)
		Assert.Equal((1, 2, 1, 2), Normalizer.NormalizeSpace(new[] { 1, 2 }));

		// 3 values: top, horizontal, bottom, horizontal
		Assert.Equal((1, 2, 3, 2), Normalizer.NormalizeSpace(new[] { 1, 2, 3 }));

		// 4 values: top, right, bottom, left as-is
		Assert.Equal((0, 0, 0, 0), Normalizer.NormalizeSpace(new[] { 0, 0, 0, 0 }));

		// null: return null
		Assert.Null(Normalizer.NormalizeSpace(null));

		// Invalid length: return null
		Assert.Null(Normalizer.NormalizeSpace(new int[] { }));
		Assert.Null(Normalizer.NormalizeSpace(new[] { 1, 2, 3, 4, 5 }));
	}

	[Fact]
	public void TestNormalizeFps()
	{
		// FPS < 1 should be clamped to 1
		Assert.Equal(1.0, Normalizer.NormalizeFps(0.0));

		// Normal FPS within range
		Assert.Equal(25.0, Normalizer.NormalizeFps(25.0));

		// FPS > 60 should be clamped to 60
		Assert.Equal(60.0, Normalizer.NormalizeFps(61.0));

		// null should return null
		Assert.Null(Normalizer.NormalizeFps(null));
	}
}
