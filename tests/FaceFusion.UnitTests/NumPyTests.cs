using FaceFusion.Tensors;
using Xunit;

namespace FaceFusion.UnitTests;

/// <summary>
/// Tests for <see cref="NumPy"/>. Expected values are taken from real NumPy output
/// (numpy 2.4.6, generated via <c>python3 -c "import numpy; ..."</c> in this
/// container) unless a comment says the value was derived by hand for a case NumPy
/// itself does not need special-casing to explain (e.g. trivial shape arithmetic).
/// </summary>
public class NumPyTests
{
    // ---------------------------------------------------------------------
    // Interp — the highest-priority operation (79 call sites in Python).
    // ---------------------------------------------------------------------

    private static readonly float[] InterpXp = { 0f, 10f, 20f, 30f };
    private static readonly float[] InterpFp = { 0f, 100f, 50f, 200f };

    [Theory]
    // Verified against: numpy.interp(numpy.float32(x), xp, fp)
    [InlineData(-5f, 0f)]     // below range -> clamps to fp[0]
    [InlineData(0f, 0f)]      // exactly on first knot
    [InlineData(5f, 50f)]     // midpoint of first segment
    [InlineData(10f, 100f)]   // exactly on interior knot
    [InlineData(15f, 75f)]    // midpoint of a descending segment
    [InlineData(20f, 50f)]    // exactly on interior knot
    [InlineData(25f, 125f)]   // midpoint of last segment
    [InlineData(30f, 200f)]   // exactly on last knot
    [InlineData(35f, 200f)]   // above range -> clamps to fp[-1]
    public void Interp_Scalar_MatchesNumPy(float x, float expected)
    {
        var actual = NumPy.Interp(x, InterpXp, InterpFp);
        Assert.Equal(expected, actual, precision: 4);
    }

    [Fact]
    public void Interp_SingleElementXp_ClampsToTheOnlyValue()
    {
        // Verified against: numpy.interp(x, [5.0], [42.0]) for x in {0, 5, 10} -> always 42.0
        float[] xp = { 5f };
        float[] fp = { 42f };

        Assert.Equal(42f, NumPy.Interp(0f, xp, fp));
        Assert.Equal(42f, NumPy.Interp(5f, xp, fp));
        Assert.Equal(42f, NumPy.Interp(10f, xp, fp));
    }

    [Fact]
    public void Interp_ArrayOverload_MatchesNumPy()
    {
        // Verified against: numpy.interp([-5, 12.5, 40], xp, fp) -> [0, 87.5, 200]
        float[] x = { -5f, 12.5f, 40f };
        var actual = NumPy.Interp(x, InterpXp, InterpFp);

        Assert.Equal(new[] { 0f, 87.5f, 200f }, actual, EqualityComparerF(1e-4f));
    }

    [Fact]
    public void Interp_DuplicateXpValues_TakesRightHandValue()
    {
        // numpy documents that with duplicate x-coordinates the right-hand fp value
        // wins (searchsorted 'left' semantics land on the segment after the duplicate).
        float[] xp = { 0f, 5f, 5f, 10f };
        float[] fp = { 0f, 1f, 2f, 3f };

        var actual = NumPy.Interp(5f, xp, fp);
        Assert.Equal(2f, actual);
    }

    [Fact]
    public void Interp_ThrowsOnMismatchedLengths()
    {
        Assert.Throws<ArgumentException>(() => NumPy.Interp(1f, new float[] { 1f, 2f }, new float[] { 1f }));
    }

    // ---------------------------------------------------------------------
    // Clip
    // ---------------------------------------------------------------------

    [Fact]
    public void Clip_Array_MatchesNumPy()
    {
        // Verified against: numpy.clip([-5, 0, 3, 10, 15], 0, 10) -> [0, 0, 3, 10, 10]
        float[] values = { -5f, 0f, 3f, 10f, 15f };
        var actual = NumPy.Clip(values, 0f, 10f);

        Assert.Equal(new[] { 0f, 0f, 3f, 10f, 10f }, actual);
    }

    [Theory]
    [InlineData(-5f, 0f)]
    [InlineData(15f, 10f)]
    [InlineData(3f, 3f)]
    public void Clip_Scalar_MatchesNumPy(float value, float expected)
    {
        Assert.Equal(expected, NumPy.Clip(value, 0f, 10f));
    }

    // ---------------------------------------------------------------------
    // Round — banker's rounding is the classic porting bug this test locks down.
    // ---------------------------------------------------------------------

    [Theory]
    // Verified against: numpy.round(numpy.float32(v)) for each v below.
    [InlineData(0.5f, 0f)]
    [InlineData(1.5f, 2f)]
    [InlineData(2.5f, 2f)]
    [InlineData(-0.5f, -0f)]
    [InlineData(-1.5f, -2f)]
    [InlineData(-2.5f, -2f)]
    [InlineData(0.25f, 0f)]
    [InlineData(2.675f, 3f)] // float32(2.675) rounds up to 3, unlike the famous Python float64 gotcha
    [InlineData(3.5f, 4f)]
    public void Round_UsesBankersRounding_MatchesNumPy(float value, float expected)
    {
        Assert.Equal(expected, NumPy.Round(value));
    }

    [Fact]
    public void Round_MatchesCSharpMathRoundDefault()
    {
        // Documents explicitly that numpy.round and Math.Round share the same
        // round-half-to-even convention, so NumPy.Round can safely delegate to it.
        for (var i = -10; i <= 10; i++)
        {
            var v = i + 0.5f;
            Assert.Equal((float)Math.Round((double)v, MidpointRounding.ToEven), NumPy.Round(v));
        }
    }

    [Theory]
    // Verified against: numpy.round(numpy.float32(v), 2)
    [InlineData(1.005f, 1.0f)]
    [InlineData(2.675f, 2.680000066757202f)]
    [InlineData(0.125f, 0.11999999731779099f)]
    public void Round_WithDecimals_MatchesNumPy(float value, float expected)
    {
        Assert.Equal(expected, NumPy.Round(value, 2), precision: 5);
    }

    [Fact]
    public void Round_Array_MatchesNumPy()
    {
        float[] values = { 0.5f, 1.5f, 2.5f, -0.5f };
        var actual = NumPy.Round(values);
        Assert.Equal(new[] { 0f, 2f, 2f, -0f }, actual);
    }

    // ---------------------------------------------------------------------
    // Mean / Min / Max / Amax / ArgMax
    // ---------------------------------------------------------------------

    private static readonly float[] Sample = { 1f, 5f, 3f, 5f, -2f };

    [Fact]
    public void Mean_MatchesNumPy()
    {
        // Verified against: numpy.mean([1,5,3,5,-2], dtype=float32) -> 2.4000000953674316
        Assert.Equal(2.4000000953674316f, NumPy.Mean(Sample), precision: 5);
    }

    [Fact]
    public void Min_MatchesNumPy()
    {
        Assert.Equal(-2f, NumPy.Min(Sample));
    }

    [Fact]
    public void Max_MatchesNumPy()
    {
        Assert.Equal(5f, NumPy.Max(Sample));
    }

    [Fact]
    public void Amax_MatchesNumPy()
    {
        Assert.Equal(5f, NumPy.Amax(Sample));
    }

    [Fact]
    public void ArgMax_ReturnsFirstOccurrence()
    {
        // Verified against: numpy.argmax([1,5,5,3]) -> 1 (first occurrence of the max)
        float[] values = { 1f, 5f, 5f, 3f };
        Assert.Equal(1, NumPy.ArgMax(values));
    }

    [Fact]
    public void Mean_Min_Max_ArgMax_ThrowOnEmpty()
    {
        Assert.Throws<ArgumentException>(() => NumPy.Mean(Array.Empty<float>()));
        Assert.Throws<ArgumentException>(() => NumPy.Min(Array.Empty<float>()));
        Assert.Throws<ArgumentException>(() => NumPy.Max(Array.Empty<float>()));
        Assert.Throws<ArgumentException>(() => NumPy.ArgMax(Array.Empty<float>()));
    }

    // ---------------------------------------------------------------------
    // Shape manipulation
    // ---------------------------------------------------------------------

    [Theory]
    // Verified against: numpy.expand_dims(numpy.zeros((3,4)), axis=A).shape
    [InlineData(0, new[] { 1, 3, 4 })]
    [InlineData(2, new[] { 3, 4, 1 })]
    [InlineData(-1, new[] { 3, 4, 1 })]
    public void ExpandDims_MatchesNumPyShape(int axis, int[] expectedShape)
    {
        var data = new float[12];
        var (_, shape) = NumPy.ExpandDims(data, new[] { 3, 4 }, axis);
        Assert.Equal(expectedShape, shape);
    }

    [Fact]
    public void Squeeze_RemovesLengthOneAxes_MatchesNumPy()
    {
        // Verified against: numpy.squeeze(numpy.zeros((1,3,1,4))).shape -> (3, 4)
        var data = new float[12];
        var (_, shape) = NumPy.Squeeze(data, new[] { 1, 3, 1, 4 });
        Assert.Equal(new[] { 3, 4 }, shape);
    }

    [Fact]
    public void Concatenate_JoinsArraysInOrder()
    {
        var actual = NumPy.Concatenate(new float[] { 1f, 2f }, new float[] { 3f }, new float[] { 4f, 5f });
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f }, actual);
    }

    [Fact]
    public void Stack_ProducesRowMajorBufferAndShape()
    {
        var (data, shape) = NumPy.Stack(new float[] { 1f, 2f }, new float[] { 3f, 4f }, new float[] { 5f, 6f });
        Assert.Equal(new[] { 3, 2 }, shape);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, data);
    }

    [Fact]
    public void HStack_IsSameAsConcatenateFor1D()
    {
        var actual = NumPy.HStack(new float[] { 1f, 2f }, new float[] { 3f });
        Assert.Equal(new[] { 1f, 2f, 3f }, actual);
    }

    [Fact]
    public void VStack_IsSameAsStackFor1D()
    {
        var (data, shape) = NumPy.VStack(new float[] { 1f, 2f }, new float[] { 3f, 4f });
        Assert.Equal(new[] { 2, 2 }, shape);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, data);
    }

    [Fact]
    public void Pad_ConstantZero_MatchesNumPy()
    {
        // Verified against: numpy.pad([1,2,3], (2,1)) -> [0,0,1,2,3,0]
        var actual = NumPy.Pad(new float[] { 1f, 2f, 3f }, 2, 1);
        Assert.Equal(new[] { 0f, 0f, 1f, 2f, 3f, 0f }, actual);
    }

    [Fact]
    public void Pad_ConstantNonZero_MatchesNumPy()
    {
        // Verified against: numpy.pad([1,2,3], (1,1), constant_values=9) -> [9,1,2,3,9]
        var actual = NumPy.Pad(new float[] { 1f, 2f, 3f }, 1, 1, 9f);
        Assert.Equal(new[] { 9f, 1f, 2f, 3f, 9f }, actual);
    }

    [Fact]
    public void Linspace_Endpoint_MatchesNumPy()
    {
        // Verified against: numpy.linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1]
        var actual = NumPy.Linspace(0f, 1f, 5);
        Assert.Equal(new[] { 0f, 0.25f, 0.5f, 0.75f, 1f }, actual, EqualityComparerF(1e-6f));
    }

    [Fact]
    public void Linspace_SinglePoint_ReturnsStart()
    {
        // Verified against: numpy.linspace(0, 1, 1) -> [0.0]
        var actual = NumPy.Linspace(0f, 1f, 1);
        Assert.Equal(new[] { 0f }, actual);
    }

    [Fact]
    public void Linspace_NoEndpoint_MatchesNumPy()
    {
        // Verified against: numpy.linspace(0, 10, 4, endpoint=False) -> [0, 2.5, 5, 7.5]
        var actual = NumPy.Linspace(0f, 10f, 4, endpoint: false);
        Assert.Equal(new[] { 0f, 2.5f, 5f, 7.5f }, actual, EqualityComparerF(1e-5f));
    }

    [Fact]
    public void Zeros_ReturnsAllZeros()
    {
        Assert.Equal(new float[5], NumPy.Zeros(5));
    }

    [Fact]
    public void ZerosLike_MatchesInputLength()
    {
        var actual = NumPy.ZerosLike(new float[] { 1f, 2f, 3f });
        Assert.Equal(new[] { 0f, 0f, 0f }, actual);
    }

    [Fact]
    public void Where_SelectsBetweenTwoArrays_MatchesNumPy()
    {
        // Verified against: numpy.where([True, False, True], [1,2,3], [10,20,30]) -> [1, 20, 3]
        bool[] condition = { true, false, true };
        float[] x = { 1f, 2f, 3f };
        float[] y = { 10f, 20f, 30f };

        var actual = NumPy.Where(condition, x, y);
        Assert.Equal(new[] { 1f, 20f, 3f }, actual);
    }

    [Fact]
    public void Where_ReturnsIndices_MatchesNumPy()
    {
        // Verified against: numpy.where([False, True, False, True]) -> (array([1, 3]),)
        bool[] condition = { false, true, false, true };
        var actual = NumPy.Where(condition);
        Assert.Equal(new[] { 1, 3 }, actual);
    }

    // ---------------------------------------------------------------------
    // Linear algebra
    // ---------------------------------------------------------------------

    [Fact]
    public void Dot_Vectors_MatchesNumPy()
    {
        // Verified against: numpy.dot([1,2,3], [4,5,6]) -> 32.0
        Assert.Equal(32f, NumPy.Dot(new float[] { 1f, 2f, 3f }, new float[] { 4f, 5f, 6f }));
    }

    [Fact]
    public void Dot_MatrixVector_MatchesNumPy()
    {
        // Verified against: numpy.dot([[1,2,3],[4,5,6]], [1,0,-1]) -> [-2, -2]
        float[] matrix = { 1f, 2f, 3f, 4f, 5f, 6f };
        float[] vector = { 1f, 0f, -1f };

        var actual = NumPy.Dot(matrix, rows: 2, inner: 3, vector);
        Assert.Equal(new[] { -2f, -2f }, actual);
    }

    [Fact]
    public void LinalgNorm_MatchesNumPy()
    {
        // Verified against: numpy.linalg.norm([3, 4]) -> 5.0 ; numpy.linalg.norm([1, 2, 2]) -> 3.0
        Assert.Equal(5f, NumPy.LinalgNorm(new float[] { 3f, 4f }));
        Assert.Equal(3f, NumPy.LinalgNorm(new float[] { 1f, 2f, 2f }));
    }

    // ---------------------------------------------------------------------
    // Layout conversion
    // ---------------------------------------------------------------------

    [Fact]
    public void TransposeHwcToChw_MatchesNumPy()
    {
        // Verified against:
        //   img = numpy.arange(2*3*4, dtype=float32).reshape(2,3,4)  # H=2,W=3,C=4
        //   numpy.transpose(img, (2,0,1)).flatten()
        var hwc = new float[24];
        for (var i = 0; i < 24; i++)
        {
            hwc[i] = i;
        }

        var chw = NumPy.TransposeHwcToChw(hwc, height: 2, width: 3, channels: 4);

        float[] expected =
        {
            0f, 4f, 8f, 12f, 16f, 20f,
            1f, 5f, 9f, 13f, 17f, 21f,
            2f, 6f, 10f, 14f, 18f, 22f,
            3f, 7f, 11f, 15f, 19f, 23f,
        };

        Assert.Equal(expected, chw);
    }

    [Fact]
    public void TransposeChwToHwc_IsInverseOfHwcToChw()
    {
        var hwc = new float[24];
        for (var i = 0; i < 24; i++)
        {
            hwc[i] = i;
        }

        var chw = NumPy.TransposeHwcToChw(hwc, height: 2, width: 3, channels: 4);
        var roundTripped = NumPy.TransposeChwToHwc(chw, channels: 4, height: 2, width: 3);

        Assert.Equal(hwc, roundTripped);
    }

    // ---------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------

    private static IEqualityComparer<float> EqualityComparerF(float tolerance) =>
        EqualityComparer<float>.Create((a, b) => MathF.Abs(a - b) <= tolerance);
}
