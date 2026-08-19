namespace FaceFusion.Parity;

/// <summary>
/// Image-level parity metrics (docs/DOTNET_PORT_PLAN.md section 7.4): PSNR and SSIM computed
/// directly over flat <see cref="double"/> pixel arrays, since images cross the .NET/Python
/// boundary as plain <c>.npy</c> arrays and there is no image/OpenCV dependency in this phase.
/// </summary>
public static class ImageMetrics
{
    /// <summary>
    /// Peak Signal-to-Noise Ratio in dB: <c>10 * log10(maxValue^2 / mse)</c>. Returns
    /// <see cref="double.PositiveInfinity"/> when the two images are pixel-identical
    /// (MSE == 0), matching the mathematical limit and what
    /// <c>skimage.metrics.peak_signal_noise_ratio</c> returns for identical images.
    /// </summary>
    /// <param name="actual">Flattened pixel values produced by the .NET port.</param>
    /// <param name="expected">Flattened ground-truth pixel values from the Python reference.</param>
    /// <param name="maxValue">Dynamic range of the pixel values (255.0 for 8-bit images).</param>
    public static double Psnr(ReadOnlySpan<double> actual, ReadOnlySpan<double> expected, double maxValue = 255.0)
    {
        if (actual.Length != expected.Length)
        {
            throw new ArgumentException(
                $"PSNR requires equal-length arrays; actual has {actual.Length} elements, expected has {expected.Length}.");
        }

        if (actual.Length == 0)
        {
            throw new ArgumentException("PSNR is undefined for empty arrays.");
        }

        var sumSquaredError = 0.0;
        for (var i = 0; i < actual.Length; i++)
        {
            var diff = actual[i] - expected[i];
            sumSquaredError += diff * diff;
        }

        var mse = sumSquaredError / actual.Length;
        if (mse == 0.0)
        {
            return double.PositiveInfinity;
        }

        return 10.0 * Math.Log10(maxValue * maxValue / mse);
    }

    /// <summary>
    /// Structural Similarity Index (Wang, Bovik, Sheikh &amp; Simoncelli, 2004) for a single
    /// 2-D single-channel image, computed with an 11x11 Gaussian window (sigma = 1.5),
    /// K1 = 0.01, K2 = 0.03, and a "valid" (non-padded) convolution - i.e. the window slides
    /// only over positions where it fully overlaps the image, so the per-pixel SSIM map is
    /// (height - 10) x (width - 10) and the returned value is its mean.
    ///
    /// This is intended to match
    /// <c>skimage.metrics.structural_similarity(im1, im2, gaussian_weights=True, sigma=1.5,
    /// use_sample_covariance=False, data_range=maxValue)</c>: local means/variances/covariance
    /// are Gaussian-weighted (not box-filtered), and the variance/covariance normalization uses
    /// the population (N) denominator rather than the sample (N-1) denominator, matching
    /// skimage's <c>use_sample_covariance=False</c> path. skimage itself was not installed in
    /// this environment (verified: `python3 -c "import skimage"` fails to import), so this was
    /// NOT cross-checked against a live skimage run - it is implemented directly from the paper
    /// formula and skimage's documented parameter semantics. One known, deliberate difference
    /// from skimage's default call (`gaussian_weights=False`): skimage's default uses a 7x7 box
    /// filter, not a Gaussian window; this implementation always uses the Gaussian window
    /// variant requested in the task, which corresponds to skimage only when
    /// `gaussian_weights=True` is passed explicitly, as specified above.
    /// </summary>
    /// <param name="actual">Flattened row-major pixel values produced by the .NET port.</param>
    /// <param name="expected">Flattened row-major ground-truth pixel values.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="maxValue">Dynamic range of the pixel values (255.0 for 8-bit images).</param>
    public static double Ssim(ReadOnlySpan<double> actual, ReadOnlySpan<double> expected, int width, int height, double maxValue = 255.0)
    {
        const int windowSize = 11;
        const double sigma = 1.5;
        const double k1 = 0.01;
        const double k2 = 0.03;

        if (width <= 0 || height <= 0)
        {
            throw new ArgumentException($"SSIM requires positive dimensions; got width={width}, height={height}.");
        }

        if (actual.Length != expected.Length)
        {
            throw new ArgumentException(
                $"SSIM requires equal-length arrays; actual has {actual.Length} elements, expected has {expected.Length}.");
        }

        if (actual.Length != width * height)
        {
            throw new ArgumentException(
                $"SSIM array length {actual.Length} does not match width*height ({width}*{height}={width * height}).");
        }

        if (width < windowSize || height < windowSize)
        {
            throw new ArgumentException(
                $"SSIM window is {windowSize}x{windowSize}; image is {width}x{height}, which is smaller than the window.");
        }

        var kernel = GaussianKernel1D(windowSize, sigma);

        var c1 = (k1 * maxValue) * (k1 * maxValue);
        var c2 = (k2 * maxValue) * (k2 * maxValue);

        // Separable Gaussian-weighted local statistics via two 1-D passes (horizontal then
        // vertical), computed for actual, expected, actual^2, expected^2 and actual*expected -
        // the standard trick to get local mean/variance/covariance without an O(window^2) loop
        // per pixel.
        var actualSq = Multiply(actual, actual);
        var expectedSq = Multiply(expected, expected);
        var crossProduct = Multiply(actual, expected);

        var muActual = SeparableGaussianFilterValid(actual, width, height, kernel);
        var muExpected = SeparableGaussianFilterValid(expected, width, height, kernel);
        var muActualSq = SeparableGaussianFilterValid(actualSq, width, height, kernel);
        var muExpectedSq = SeparableGaussianFilterValid(expectedSq, width, height, kernel);
        var muCross = SeparableGaussianFilterValid(crossProduct, width, height, kernel);

        var outWidth = width - windowSize + 1;
        var outHeight = height - windowSize + 1;
        var outCount = outWidth * outHeight;

        var sum = 0.0;
        for (var i = 0; i < outCount; i++)
        {
            var ma = muActual[i];
            var me = muExpected[i];

            // Population variance/covariance (use_sample_covariance=False): E[X^2] - E[X]^2,
            // not the Bessel-corrected N/(N-1) form skimage uses when
            // use_sample_covariance=True (its default for gaussian_weights=True call sites is
            // actually use_sample_covariance=False per the task spec, which is what this
            // matches).
            var varActual = muActualSq[i] - ma * ma;
            var varExpected = muExpectedSq[i] - me * me;
            var covar = muCross[i] - ma * me;

            var numerator = (2 * ma * me + c1) * (2 * covar + c2);
            var denominator = (ma * ma + me * me + c1) * (varActual + varExpected + c2);

            sum += numerator / denominator;
        }

        return sum / outCount;
    }

    private static double[] Multiply(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var result = new double[a.Length];
        for (var i = 0; i < a.Length; i++)
        {
            result[i] = a[i] * b[i];
        }

        return result;
    }

    private static double[] GaussianKernel1D(int size, double sigma)
    {
        var kernel = new double[size];
        var center = (size - 1) / 2.0;
        var sum = 0.0;

        for (var i = 0; i < size; i++)
        {
            var x = i - center;
            var value = Math.Exp(-(x * x) / (2 * sigma * sigma));
            kernel[i] = value;
            sum += value;
        }

        for (var i = 0; i < size; i++)
        {
            kernel[i] /= sum;
        }

        return kernel;
    }

    /// <summary>
    /// Applies a separable 2-D Gaussian filter to a row-major image using "valid" boundary
    /// handling (no padding): the output is (height - kernel.Length + 1) x (width -
    /// kernel.Length + 1). This matches skimage's SSIM, which crops the window border
    /// (`mode='valid'` filtering) rather than padding it, so no border pixels are included in
    /// the mean SSIM.
    /// </summary>
    private static double[] SeparableGaussianFilterValid(ReadOnlySpan<double> image, int width, int height, double[] kernel)
    {
        var k = kernel.Length;

        // Horizontal pass: full height, valid width.
        var validWidth = width - k + 1;
        var horizontal = new double[height * validWidth];
        for (var y = 0; y < height; y++)
        {
            var rowOffset = y * width;
            var outRowOffset = y * validWidth;
            for (var x = 0; x < validWidth; x++)
            {
                var acc = 0.0;
                for (var j = 0; j < k; j++)
                {
                    acc += image[rowOffset + x + j] * kernel[j];
                }

                horizontal[outRowOffset + x] = acc;
            }
        }

        // Vertical pass: valid height, valid width.
        var validHeight = height - k + 1;
        var result = new double[validHeight * validWidth];
        for (var y = 0; y < validHeight; y++)
        {
            var outRowOffset = y * validWidth;
            for (var x = 0; x < validWidth; x++)
            {
                var acc = 0.0;
                for (var j = 0; j < k; j++)
                {
                    acc += horizontal[(y + j) * validWidth + x] * kernel[j];
                }

                result[outRowOffset + x] = acc;
            }
        }

        return result;
    }
}
