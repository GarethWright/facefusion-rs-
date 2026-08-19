using System.Globalization;
using System.Text;

namespace FaceFusion.Parity;

/// <summary>
/// Result of comparing two tensors (flattened as <see cref="double"/> arrays) for parity.
/// Carries enough diagnostics to debug a mismatch without re-running the comparison.
/// </summary>
public sealed class ComparisonResult
{
    /// <summary>True when every element passed the tolerance check (and shapes matched).</summary>
    public bool Passed { get; }

    /// <summary>True when the failure was a shape mismatch rather than a value mismatch.</summary>
    public bool ShapeMismatch { get; }

    /// <summary>Number of elements in the "actual" array (0 when shapes mismatch).</summary>
    public int ActualLength { get; }

    /// <summary>Number of elements in the "expected" array.</summary>
    public int ExpectedLength { get; }

    /// <summary>Total number of elements compared (equal to both lengths when shapes match).</summary>
    public int TotalCount { get; }

    /// <summary>Number of elements that failed the tolerance check.</summary>
    public int MismatchCount { get; }

    /// <summary>Largest |actual - expected| observed, over all compared elements.</summary>
    public double MaxAbsoluteDifference { get; }

    /// <summary>
    /// Largest relative difference observed, computed as |actual - expected| / |expected|
    /// (0/0 treated as 0, x/0 treated as +Infinity) purely for reporting purposes. This is
    /// diagnostic only: the pass/fail decision uses the combined numpy.allclose formula, not
    /// this value in isolation.
    /// </summary>
    public double MaxRelativeDifference { get; }

    /// <summary>Flat index of the element with the largest absolute difference.</summary>
    public int MaxDifferenceIndex { get; }

    /// <summary>Actual value at <see cref="MaxDifferenceIndex"/>.</summary>
    public double ActualAtMaxDifference { get; }

    /// <summary>Expected value at <see cref="MaxDifferenceIndex"/>.</summary>
    public double ExpectedAtMaxDifference { get; }

    /// <summary>Relative tolerance used for the comparison.</summary>
    public double RelativeTolerance { get; }

    /// <summary>Absolute tolerance used for the comparison.</summary>
    public double AbsoluteTolerance { get; }

    private ComparisonResult(
        bool passed,
        bool shapeMismatch,
        int actualLength,
        int expectedLength,
        int totalCount,
        int mismatchCount,
        double maxAbsoluteDifference,
        double maxRelativeDifference,
        int maxDifferenceIndex,
        double actualAtMaxDifference,
        double expectedAtMaxDifference,
        double relativeTolerance,
        double absoluteTolerance)
    {
        Passed = passed;
        ShapeMismatch = shapeMismatch;
        ActualLength = actualLength;
        ExpectedLength = expectedLength;
        TotalCount = totalCount;
        MismatchCount = mismatchCount;
        MaxAbsoluteDifference = maxAbsoluteDifference;
        MaxRelativeDifference = maxRelativeDifference;
        MaxDifferenceIndex = maxDifferenceIndex;
        ActualAtMaxDifference = actualAtMaxDifference;
        ExpectedAtMaxDifference = expectedAtMaxDifference;
        RelativeTolerance = relativeTolerance;
        AbsoluteTolerance = absoluteTolerance;
    }

    internal static ComparisonResult ForShapeMismatch(int actualLength, int expectedLength, double rtol, double atol)
    {
        return new ComparisonResult(
            passed: false,
            shapeMismatch: true,
            actualLength: actualLength,
            expectedLength: expectedLength,
            totalCount: 0,
            mismatchCount: 0,
            maxAbsoluteDifference: double.NaN,
            maxRelativeDifference: double.NaN,
            maxDifferenceIndex: -1,
            actualAtMaxDifference: double.NaN,
            expectedAtMaxDifference: double.NaN,
            relativeTolerance: rtol,
            absoluteTolerance: atol);
    }

    internal static ComparisonResult ForElementwise(
        int totalCount,
        int mismatchCount,
        double maxAbsoluteDifference,
        double maxRelativeDifference,
        int maxDifferenceIndex,
        double actualAtMaxDifference,
        double expectedAtMaxDifference,
        double rtol,
        double atol)
    {
        return new ComparisonResult(
            passed: mismatchCount == 0,
            shapeMismatch: false,
            actualLength: totalCount,
            expectedLength: totalCount,
            totalCount: totalCount,
            mismatchCount: mismatchCount,
            maxAbsoluteDifference: maxAbsoluteDifference,
            maxRelativeDifference: maxRelativeDifference,
            maxDifferenceIndex: maxDifferenceIndex,
            actualAtMaxDifference: actualAtMaxDifference,
            expectedAtMaxDifference: expectedAtMaxDifference,
            relativeTolerance: rtol,
            absoluteTolerance: atol);
    }

    /// <summary>Human-readable summary suitable for test failure messages.</summary>
    public string Describe()
    {
        var sb = new StringBuilder();

        if (Passed)
        {
            sb.Append(CultureInfo.InvariantCulture, $"PASS: {TotalCount} elements within tolerance (rtol={RelativeTolerance}, atol={AbsoluteTolerance}).");
            return sb.ToString();
        }

        if (ShapeMismatch)
        {
            sb.Append(CultureInfo.InvariantCulture, $"FAIL: shape mismatch - actual has {ActualLength} elements, expected has {ExpectedLength} elements.");
            return sb.ToString();
        }

        sb.Append(CultureInfo.InvariantCulture, $"FAIL: {MismatchCount}/{TotalCount} elements outside tolerance (rtol={RelativeTolerance}, atol={AbsoluteTolerance}). ");
        sb.Append(CultureInfo.InvariantCulture, $"Max absolute difference = {MaxAbsoluteDifference} at flat index {MaxDifferenceIndex} ");
        sb.Append(CultureInfo.InvariantCulture, $"(actual={ActualAtMaxDifference}, expected={ExpectedAtMaxDifference}). ");
        sb.Append(CultureInfo.InvariantCulture, $"Max relative difference = {MaxRelativeDifference}.");
        return sb.ToString();
    }

    public override string ToString() => Describe();
}

/// <summary>
/// Element-wise tensor comparison matching <c>numpy.allclose</c> semantics
/// (docs/DOTNET_PORT_PLAN.md section 7.3). Used to compare .NET pipeline intermediates
/// against arrays dumped from the reference Python pipeline.
/// </summary>
public static class TensorComparison
{
    /// <summary>
    /// Compares <paramref name="actual"/> against <paramref name="expected"/> using the same
    /// formula as <c>numpy.allclose</c>: an element passes when
    /// <c>|actual - expected| &lt;= atol + rtol * |expected|</c>. Note the asymmetry: the
    /// relative term is scaled by the expected (second/ground-truth) value, not the actual
    /// value, exactly as numpy does it - so <c>Compare(a, b)</c> and <c>Compare(b, a)</c> can
    /// disagree when |a| and |b| differ.
    /// </summary>
    /// <param name="actual">Values produced by the .NET port.</param>
    /// <param name="expected">Ground-truth values from the Python reference.</param>
    /// <param name="relativeTolerance">rtol, defaults to numpy's default of 1e-5.</param>
    /// <param name="absoluteTolerance">atol, defaults to numpy's default of 1e-8.</param>
    /// <param name="equalNan">
    /// When true, NaN is considered equal to NaN at the same position (matches
    /// <c>numpy.allclose(..., equal_nan=True)</c>). Defaults to false, matching numpy's own
    /// default of <c>equal_nan=False</c>, under which NaN vs NaN fails.
    /// </param>
    public static ComparisonResult Compare(
        ReadOnlySpan<double> actual,
        ReadOnlySpan<double> expected,
        double relativeTolerance = 1e-5,
        double absoluteTolerance = 1e-8,
        bool equalNan = false)
    {
        if (actual.Length != expected.Length)
        {
            return ComparisonResult.ForShapeMismatch(actual.Length, expected.Length, relativeTolerance, absoluteTolerance);
        }

        var totalCount = actual.Length;
        var mismatchCount = 0;
        var maxAbsoluteDifference = double.NegativeInfinity;
        var maxRelativeDifference = double.NaN;
        var maxDifferenceIndex = -1;
        var actualAtMax = double.NaN;
        var expectedAtMax = double.NaN;

        for (var i = 0; i < totalCount; i++)
        {
            var a = actual[i];
            var e = expected[i];

            var elementPasses = ElementCloses(a, e, relativeTolerance, absoluteTolerance, equalNan);

            var absoluteDifference = ElementAbsoluteDifference(a, e);
            var relativeDifference = ElementRelativeDifference(a, e, absoluteDifference);

            // Track the element with the largest absolute difference for diagnostics. NaN
            // differences (from a NaN actual/expected) never win over a real numeric
            // difference, since "absoluteDifference > maxAbsoluteDifference" is always false
            // when absoluteDifference is NaN - they only get reported when nothing else has
            // set a candidate yet (maxDifferenceIndex == -1), e.g. an all-NaN array.
            if (maxDifferenceIndex == -1 || absoluteDifference > maxAbsoluteDifference)
            {
                maxAbsoluteDifference = absoluteDifference;
                maxRelativeDifference = relativeDifference;
                maxDifferenceIndex = i;
                actualAtMax = a;
                expectedAtMax = e;
            }

            if (!elementPasses)
            {
                mismatchCount++;
            }
        }

        return ComparisonResult.ForElementwise(
            totalCount,
            mismatchCount,
            maxAbsoluteDifference,
            maxRelativeDifference,
            maxDifferenceIndex,
            actualAtMax,
            expectedAtMax,
            relativeTolerance,
            absoluteTolerance);
    }

    /// <summary>
    /// Single-element version of the numpy.allclose formula, exposed for callers that only
    /// need a boolean and not the full diagnostic report.
    /// </summary>
    public static bool ElementCloses(double actual, double expected, double relativeTolerance, double absoluteTolerance, bool equalNan = false)
    {
        if (double.IsNaN(actual) || double.IsNaN(expected))
        {
            return equalNan && double.IsNaN(actual) && double.IsNaN(expected);
        }

        if (double.IsInfinity(actual) || double.IsInfinity(expected))
        {
            // numpy.isclose treats matching-sign infinities as equal, and anything else
            // (including +inf vs finite) as not equal.
            return actual == expected;
        }

        return Math.Abs(actual - expected) <= absoluteTolerance + relativeTolerance * Math.Abs(expected);
    }

    private static double ElementAbsoluteDifference(double actual, double expected)
    {
        if (double.IsNaN(actual) || double.IsNaN(expected))
        {
            return double.NaN;
        }

        if (double.IsInfinity(actual) || double.IsInfinity(expected))
        {
            return actual == expected ? 0.0 : double.PositiveInfinity;
        }

        return Math.Abs(actual - expected);
    }

    private static double ElementRelativeDifference(double actual, double expected, double absoluteDifference)
    {
        if (double.IsNaN(absoluteDifference))
        {
            return double.NaN;
        }

        if (expected == 0.0)
        {
            return absoluteDifference == 0.0 ? 0.0 : double.PositiveInfinity;
        }

        return absoluteDifference / Math.Abs(expected);
    }
}
