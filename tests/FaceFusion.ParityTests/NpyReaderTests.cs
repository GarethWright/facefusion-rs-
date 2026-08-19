using System.Globalization;
using System.Text.Json;
using FaceFusion.Parity;

namespace FaceFusion.ParityTests;

/// <summary>
/// Drives <see cref="NpyReader"/> against every fixture in
/// <c>tests/FaceFusion.ParityTests/fixtures</c>, comparing dtype, shape and values against
/// <c>manifest.json</c>. Each fixture is its own reported test case via <see cref="MemberDataAttribute"/>.
/// </summary>
public class NpyReaderTests
{
	private static readonly string FixturesDirectory = Path.Combine(AppContext.BaseDirectory, "fixtures");

	public static IEnumerable<object[]> FixtureCases()
	{
		using var document = JsonDocument.Parse(File.ReadAllText(ManifestPath));

		foreach (var property in document.RootElement.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
		{
			yield return new object[] { property.Name };
		}
	}

	private static string ManifestPath => Path.Combine(FixturesDirectory, "manifest.json");

	[Theory]
	[MemberData(nameof(FixtureCases))]
	public void Loads_fixture_with_expected_metadata_and_values(string fixtureName)
	{
		using var document = JsonDocument.Parse(File.ReadAllText(ManifestPath));
		var meta = document.RootElement.GetProperty(fixtureName);

		var npyPath = Path.Combine(FixturesDirectory, fixtureName + ".npy");
		var array = NpyReader.Load(npyPath);

		Assert.Equal(meta.GetProperty("dtype").GetString(), array.DType);

		var expectedShape = meta.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
		Assert.Equal(expectedShape, array.Shape.ToArray());

		Assert.Equal(meta.GetProperty("element_count").GetInt32(), array.ElementCount);

		var expectedValues = meta.GetProperty("values_c_order").EnumerateArray().ToArray();
		Assert.Equal(expectedValues.Length, array.ElementCount);

		var actual = array.AsDoubles();
		Assert.Equal(expectedValues.Length, actual.Length);

		if (array.DType == "bool")
		{
			for (var index = 0; index < expectedValues.Length; index++)
			{
				var expectedBool = expectedValues[index].GetBoolean();
				Assert.Equal(expectedBool ? 1.0 : 0.0, actual[index]);
			}
		}
		else if (array.DType is "float32" or "float64")
		{
			for (var index = 0; index < expectedValues.Length; index++)
			{
				AssertFloatBitwiseEqual(expectedValues[index].GetString()!, actual[index]);
			}
		}
		else
		{
			for (var index = 0; index < expectedValues.Length; index++)
			{
				Assert.Equal(expectedValues[index].GetDouble(), actual[index]);
			}
		}

		// Also verify the float accessor for float dtypes, and that RawData length matches.
		if (array.DType is "float32" or "float64")
		{
			var actualFloats = array.AsFloats();
			Assert.Equal(expectedValues.Length, actualFloats.Length);
		}

		Assert.Equal(array.ElementCount * array.ItemSize, array.RawData.Length);
	}

	/// <summary>
	/// Compares a Python <c>repr(float(...))</c> string against a widened double, bit-exact so
	/// that NaN, +/-Inf and the sign of zero all round-trip precisely.
	/// </summary>
	private static void AssertFloatBitwiseEqual(string expectedRepr, double actual)
	{
		var expected = expectedRepr switch
		{
			"nan" => double.NaN,
			"inf" => double.PositiveInfinity,
			"-inf" => double.NegativeInfinity,
			_ => double.Parse(expectedRepr, NumberStyles.Float, CultureInfo.InvariantCulture)
		};

		if (double.IsNaN(expected))
		{
			Assert.True(double.IsNaN(actual), $"Expected NaN, got {actual}.");
			return;
		}

		Assert.True(
			BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual),
			$"Expected {expectedRepr} (bits {BitConverter.DoubleToInt64Bits(expected):X}), got {actual} (bits {BitConverter.DoubleToInt64Bits(actual):X}).");
	}
}
