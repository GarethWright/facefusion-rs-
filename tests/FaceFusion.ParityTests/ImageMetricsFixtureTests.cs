using System.Globalization;
using System.Text.Json;
using FaceFusion.Parity;

namespace FaceFusion.ParityTests;

/// <summary>
/// Fixture-driven checks for <see cref="ImageMetrics"/>, complementing the hand-written
/// cases in <c>ImageMetricsTests</c>.
///
/// The expected values come from the independent NumPy reference in
/// <c>tools/parity/generate_fixtures.py</c> and are regenerated alongside the image pairs,
/// so extending the corpus does not mean hand-transcribing constants. The pairs are stored
/// as .npy and loaded through <see cref="NpyReader"/>, which also exercises the reader on
/// real 2-D float data rather than only on synthetic fixtures.
///
/// scikit-image is not installed in this environment, so these values verify the maths
/// against a second implementation — not exact parity with skimage. See the note in
/// generate_fixtures.py.
/// </summary>
public sealed class ImageMetricsFixtureTests
{
    private static string FixturesDirectory =>
        Path.Combine(AppContext.BaseDirectory, "fixtures");

    public static IEnumerable<object[]> ImageCases()
    {
        var manifestPath = Path.Combine(FixturesDirectory, "images_manifest.json");
        using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));

        foreach (var entry in document.RootElement.EnumerateObject())
        {
            yield return new object[] { entry.Name };
        }
    }

    [Theory]
    [MemberData(nameof(ImageCases))]
    public void MatchesIndependentNumPyReference(string caseName)
    {
        var manifestPath = Path.Combine(FixturesDirectory, "images_manifest.json");
        using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var entry = document.RootElement.GetProperty(caseName);

        var width = entry.GetProperty("width").GetInt32();
        var height = entry.GetProperty("height").GetInt32();
        var expectedSsim = ParsePythonRepr(entry.GetProperty("ssim").GetString()!);
        var expectedPsnr = ParsePythonRepr(entry.GetProperty("psnr").GetString()!);

        var first = NpyReader.Load(Path.Combine(FixturesDirectory, "images", caseName + "_a.npy")).AsDoubles();
        var second = NpyReader.Load(Path.Combine(FixturesDirectory, "images", caseName + "_b.npy")).AsDoubles();

        var actualSsim = ImageMetrics.Ssim(first, second, width, height);
        var actualPsnr = ImageMetrics.Psnr(first, second);

        Assert.Equal(expectedSsim, actualSsim, 9);

        if (double.IsPositiveInfinity(expectedPsnr))
        {
            // Pixel-identical images: PSNR is infinite, not merely large.
            Assert.Equal(double.PositiveInfinity, actualPsnr);
        }
        else
        {
            Assert.Equal(expectedPsnr, actualPsnr, 9);
        }
    }

    /// <summary>
    /// SSIM is bounded on [-1, 1] and must reach both ends: identical images score 1, and
    /// an image against its own inversion scores close to -1. A implementation that
    /// silently clamped or took an absolute value would pass the identical-image test but
    /// fail here.
    /// </summary>
    [Fact]
    public void SpansTheFullRange()
    {
        var identical = LoadPair("identical_gradient");
        var inverted = LoadPair("checker_vs_inverted");

        Assert.Equal(1.0, ImageMetrics.Ssim(identical.First, identical.Second, 16, 16), 9);

        var invertedSsim = ImageMetrics.Ssim(inverted.First, inverted.Second, 16, 16);
        Assert.InRange(invertedSsim, -1.0, -0.9);
    }

    /// <summary>SSIM is symmetric in its two arguments, unlike TensorComparison.</summary>
    [Theory]
    [InlineData("gradient_vs_noisy")]
    [InlineData("constant_vs_different")]
    [InlineData("checker_vs_inverted")]
    public void IsSymmetric(string caseName)
    {
        var pair = LoadPair(caseName);

        var forward = ImageMetrics.Ssim(pair.First, pair.Second, 16, 16);
        var backward = ImageMetrics.Ssim(pair.Second, pair.First, 16, 16);

        Assert.Equal(forward, backward, 12);
    }

    private static (double[] First, double[] Second) LoadPair(string caseName)
    {
        var first = NpyReader.Load(Path.Combine(FixturesDirectory, "images", caseName + "_a.npy")).AsDoubles();
        var second = NpyReader.Load(Path.Combine(FixturesDirectory, "images", caseName + "_b.npy")).AsDoubles();

        return (first, second);
    }

    /// <summary>
    /// The manifest stores floats as Python repr strings so nan/inf/-0.0 are unambiguous;
    /// C# spells the non-finite ones differently.
    /// </summary>
    private static double ParsePythonRepr(string text) => text switch
    {
        "inf" => double.PositiveInfinity,
        "-inf" => double.NegativeInfinity,
        "nan" => double.NaN,
        _ => double.Parse(text, CultureInfo.InvariantCulture)
    };
}
