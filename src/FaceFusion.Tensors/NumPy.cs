using System;
using System.Collections.Generic;

namespace FaceFusion.Tensors;

/// <summary>
/// A small, closed-ended numpy-compatibility layer. Arrays are represented as
/// <see cref="float"/>[] / <see cref="ReadOnlySpan{T}"/> with explicit shape parameters
/// where needed, rather than a general N-dimensional array type. See
/// docs/DOTNET_PORT_PLAN.md section 4: this covers only the ~20 numpy operations the
/// FaceFusion Python codebase actually uses. Do not extend this into a general array
/// library.
///
/// <see cref="System.Numerics.Tensors.TensorPrimitives"/> is not available on net8.0
/// without adding a NuGet package (it ships in the BCL reference assemblies starting
/// with net9.0), and the port conventions forbid adding packages without asking, so this
/// class uses plain loops throughout. Correctness (verified against real NumPy output)
/// takes priority over SIMD here.
/// </summary>
public static class NumPy
{
    // ---------------------------------------------------------------------
    // interp — numpy.interp(x, xp, fp)
    // ---------------------------------------------------------------------

    /// <summary>
    /// One-dimensional piecewise linear interpolation, equivalent to
    /// <c>numpy.interp(x, xp, fp)</c>. <paramref name="xp"/> is assumed to be increasing
    /// (not verified, matching numpy's default <c>period=None</c> behaviour). Values of
    /// <paramref name="x"/> below <c>xp[0]</c> clamp to <c>fp[0]</c>; values above
    /// <c>xp[^1]</c> clamp to <c>fp[^1]</c>.
    /// </summary>
    public static float Interp(float x, ReadOnlySpan<float> xp, ReadOnlySpan<float> fp)
    {
        if (xp.Length != fp.Length)
        {
            throw new ArgumentException("xp and fp must have the same length.");
        }

        if (xp.Length == 0)
        {
            throw new ArgumentException("xp must not be empty.");
        }

        if (xp.Length == 1)
        {
            // numpy: with a single knot every x maps to fp[0].
            return fp[0];
        }

        if (x <= xp[0])
        {
            return fp[0];
        }

        if (x >= xp[^1])
        {
            return fp[^1];
        }

        // Find the interval [xp[i], xp[i + 1]] that contains x. numpy uses binary
        // search internally (via searchsorted); we do the same for parity on ties.
        var lo = 0;
        var hi = xp.Length - 1;
        while (lo < hi - 1)
        {
            var mid = (lo + hi) / 2;
            if (xp[mid] <= x)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }

        var x0 = xp[lo];
        var x1 = xp[hi];
        var y0 = fp[lo];
        var y1 = fp[hi];

        if (x1 == x0)
        {
            // numpy: duplicate x-coordinates — take the right-hand value.
            return y1;
        }

        var slope = (y1 - y0) / (x1 - x0);
        return y0 + (slope * (x - x0));
    }

    /// <summary>Array overload of <see cref="Interp(float, ReadOnlySpan{float}, ReadOnlySpan{float})"/>.</summary>
    public static float[] Interp(ReadOnlySpan<float> x, ReadOnlySpan<float> xp, ReadOnlySpan<float> fp)
    {
        var result = new float[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            result[i] = Interp(x[i], xp, fp);
        }

        return result;
    }

    // ---------------------------------------------------------------------
    // clip / round
    // ---------------------------------------------------------------------

    /// <summary>Equivalent of <c>numpy.clip(a, a_min, a_max)</c> for a scalar.</summary>
    public static float Clip(float value, float min, float max)
    {
        if (value < min)
        {
            return min;
        }

        if (value > max)
        {
            return max;
        }

        return value;
    }

    /// <summary>Equivalent of <c>numpy.clip(a, a_min, a_max)</c> for an array.</summary>
    public static float[] Clip(ReadOnlySpan<float> values, float min, float max)
    {
        var result = new float[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            result[i] = Clip(values[i], min, max);
        }

        return result;
    }

    /// <summary>
    /// Equivalent of <c>numpy.round(a)</c> (round-half-to-even / "banker's rounding").
    /// This matches the default rounding mode of <see cref="Math.Round(double)"/>, but the
    /// match is deliberate and load-bearing, not incidental — see NumPyTests for proof.
    /// </summary>
    public static float Round(float value)
    {
        return (float)Math.Round(value, MidpointRounding.ToEven);
    }

    /// <summary>
    /// Equivalent of <c>numpy.round(a, decimals)</c>. numpy implements this as
    /// <c>round(a * 10**decimals) / 10**decimals</c> entirely in the array's own dtype
    /// (float32 here), which means the scaling step can itself introduce float32
    /// rounding error before the round-half-to-even step runs — e.g.
    /// <c>numpy.round(numpy.float32(2.675), 2)</c> is <c>2.68</c>, not <c>2.67</c>,
    /// because <c>float32(2.675) * 100 == 267.5</c> exactly in float32. Doing this
    /// computation in double (as <see cref="Math.Round(double, int)"/> would) does not
    /// reproduce that error and gives the "more correct" but numpy-incompatible 2.67, so
    /// we deliberately replicate numpy's float32 arithmetic here instead of using
    /// <see cref="Math.Round(double, int)"/>.
    /// </summary>
    public static float Round(float value, int decimals)
    {
        var scale = MathF.Pow(10f, decimals);
        var scaled = value * scale;
        var rounded = MathF.Round(scaled, MidpointRounding.ToEven);
        return rounded / scale;
    }

    /// <summary>Array overload of <see cref="Round(float)"/>.</summary>
    public static float[] Round(ReadOnlySpan<float> values)
    {
        var result = new float[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            result[i] = Round(values[i]);
        }

        return result;
    }

    // ---------------------------------------------------------------------
    // mean / min / max / amax / argmax
    // ---------------------------------------------------------------------

    /// <summary>Equivalent of <c>numpy.mean(a)</c>.</summary>
    public static float Mean(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("values must not be empty.");
        }

        // numpy accumulates in double precision internally for float32 input's
        // intermediate sum in many paths; to keep this simple and precise we sum in
        // double and cast back, which matches numpy's float32 mean closely.
        double sum = 0;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
        }

        return (float)(sum / values.Length);
    }

    /// <summary>Equivalent of <c>numpy.min(a)</c>.</summary>
    public static float Min(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("values must not be empty.");
        }

        var min = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] < min)
            {
                min = values[i];
            }
        }

        return min;
    }

    /// <summary>Equivalent of <c>numpy.max(a)</c>.</summary>
    public static float Max(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("values must not be empty.");
        }

        var max = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
            }
        }

        return max;
    }

    /// <summary>Equivalent of <c>numpy.amax(a)</c> — alias of max for a flat array.</summary>
    public static float Amax(ReadOnlySpan<float> values) => Max(values);

    /// <summary>
    /// Equivalent of <c>numpy.argmax(a)</c>: index of the first occurrence of the
    /// maximum value.
    /// </summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("values must not be empty.");
        }

        var maxIndex = 0;
        var max = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    // ---------------------------------------------------------------------
    // shape manipulation (all logical no-ops on a flat float[] buffer + reported shape)
    // ---------------------------------------------------------------------

    /// <summary>
    /// Equivalent of <c>numpy.expand_dims(a, axis)</c>. Since arrays here are flat
    /// buffers with an explicit shape, this returns the new shape; the underlying data
    /// is unchanged (a copy is still returned for value-type safety/ownership clarity).
    /// </summary>
    public static (float[] Data, int[] Shape) ExpandDims(ReadOnlySpan<float> data, ReadOnlySpan<int> shape, int axis)
    {
        var rank = shape.Length + 1;
        var normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis > shape.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(axis));
        }

        var newShape = new int[rank];
        var si = 0;
        for (var i = 0; i < rank; i++)
        {
            newShape[i] = i == normalizedAxis ? 1 : shape[si++];
        }

        return (data.ToArray(), newShape);
    }

    /// <summary>
    /// Equivalent of <c>numpy.squeeze(a)</c>: removes all axes of length 1 from
    /// <paramref name="shape"/>. Data is unchanged (flat buffers have no per-axis
    /// layout to collapse).
    /// </summary>
    public static (float[] Data, int[] Shape) Squeeze(ReadOnlySpan<float> data, ReadOnlySpan<int> shape)
    {
        var newShapeList = new List<int>(shape.Length);
        foreach (var dim in shape)
        {
            if (dim != 1)
            {
                newShapeList.Add(dim);
            }
        }

        return (data.ToArray(), newShapeList.ToArray());
    }

    /// <summary>
    /// Equivalent of <c>numpy.concatenate(arrays)</c> for 1-D arrays (the only case this
    /// codebase uses concatenate for on flat buffers). For higher-rank concatenation
    /// along axis 0 of row-major buffers with identical trailing shape, this is the same
    /// operation as a flat concatenation.
    /// </summary>
    public static float[] Concatenate(params float[][] arrays)
    {
        var totalLength = 0;
        foreach (var array in arrays)
        {
            totalLength += array.Length;
        }

        var result = new float[totalLength];
        var offset = 0;
        foreach (var array in arrays)
        {
            array.CopyTo(result.AsSpan(offset));
            offset += array.Length;
        }

        return result;
    }

    /// <summary>
    /// Equivalent of <c>numpy.stack(arrays)</c>: stacks equal-length 1-D arrays along a
    /// new leading axis, returning the flat row-major buffer and the resulting 2-D shape
    /// <c>(count, length)</c>.
    /// </summary>
    public static (float[] Data, int[] Shape) Stack(params float[][] arrays)
    {
        if (arrays.Length == 0)
        {
            throw new ArgumentException("arrays must not be empty.");
        }

        var length = arrays[0].Length;
        foreach (var array in arrays)
        {
            if (array.Length != length)
            {
                throw new ArgumentException("all input arrays must have the same length.");
            }
        }

        var result = new float[arrays.Length * length];
        for (var i = 0; i < arrays.Length; i++)
        {
            arrays[i].CopyTo(result.AsSpan(i * length, length));
        }

        return (result, new[] { arrays.Length, length });
    }

    /// <summary>Equivalent of <c>numpy.hstack(arrays)</c> for 1-D arrays: same as concatenate.</summary>
    public static float[] HStack(params float[][] arrays) => Concatenate(arrays);

    /// <summary>
    /// Equivalent of <c>numpy.vstack(arrays)</c> for 1-D arrays: stacks them as rows of a
    /// 2-D array, i.e. the same layout as <see cref="Stack"/>.
    /// </summary>
    public static (float[] Data, int[] Shape) VStack(params float[][] arrays) => Stack(arrays);

    /// <summary>
    /// Equivalent of <c>numpy.pad(a, pad_width, mode='constant', constant_values=0)</c>
    /// for a 1-D array.
    /// </summary>
    public static float[] Pad(ReadOnlySpan<float> values, int padBefore, int padAfter, float constantValue = 0f)
    {
        if (padBefore < 0 || padAfter < 0)
        {
            throw new ArgumentOutOfRangeException(padBefore < 0 ? nameof(padBefore) : nameof(padAfter));
        }

        var result = new float[padBefore + values.Length + padAfter];
        if (constantValue != 0f)
        {
            Array.Fill(result, constantValue, 0, padBefore);
            Array.Fill(result, constantValue, padBefore + values.Length, padAfter);
        }

        values.CopyTo(result.AsSpan(padBefore));
        return result;
    }

    /// <summary>
    /// Equivalent of <c>numpy.linspace(start, stop, num, endpoint=True)</c>.
    /// </summary>
    public static float[] Linspace(float start, float stop, int num, bool endpoint = true)
    {
        if (num < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(num));
        }

        var result = new float[num];
        if (num == 0)
        {
            return result;
        }

        if (num == 1)
        {
            result[0] = start;
            return result;
        }

        var divisor = endpoint ? num - 1 : num;
        var step = (stop - start) / divisor;
        for (var i = 0; i < num; i++)
        {
            result[i] = start + (step * i);
        }

        if (endpoint)
        {
            // numpy explicitly sets the last sample to `stop` to avoid float drift.
            result[num - 1] = stop;
        }

        return result;
    }

    /// <summary>Equivalent of <c>numpy.zeros(count)</c>.</summary>
    public static float[] Zeros(int count) => new float[count];

    /// <summary>Equivalent of <c>numpy.zeros_like(a)</c>.</summary>
    public static float[] ZerosLike(ReadOnlySpan<float> array) => new float[array.Length];

    /// <summary>
    /// Equivalent of <c>numpy.where(condition, x, y)</c> for equal-length 1-D arrays.
    /// </summary>
    public static float[] Where(ReadOnlySpan<bool> condition, ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (condition.Length != x.Length || condition.Length != y.Length)
        {
            throw new ArgumentException("condition, x and y must have the same length.");
        }

        var result = new float[condition.Length];
        for (var i = 0; i < condition.Length; i++)
        {
            result[i] = condition[i] ? x[i] : y[i];
        }

        return result;
    }

    /// <summary>
    /// Equivalent of <c>numpy.where(condition)</c> for a 1-D array: returns the indices
    /// where <paramref name="condition"/> is true.
    /// </summary>
    public static int[] Where(ReadOnlySpan<bool> condition)
    {
        var indices = new List<int>();
        for (var i = 0; i < condition.Length; i++)
        {
            if (condition[i])
            {
                indices.Add(i);
            }
        }

        return indices.ToArray();
    }

    // ---------------------------------------------------------------------
    // linear algebra
    // ---------------------------------------------------------------------

    /// <summary>Equivalent of <c>numpy.dot(a, b)</c> for two 1-D arrays (inner product).</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("a and b must have the same length.");
        }

        double sum = 0;
        for (var i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }

        return (float)sum;
    }

    /// <summary>
    /// Equivalent of <c>numpy.dot(a, b)</c> for a matrix (row-major, shape
    /// <c>rows x inner</c>) times a vector (length <c>inner</c>), producing a vector of
    /// length <c>rows</c>.
    /// </summary>
    public static float[] Dot(ReadOnlySpan<float> matrix, int rows, int inner, ReadOnlySpan<float> vector)
    {
        if (matrix.Length != rows * inner)
        {
            throw new ArgumentException("matrix length must equal rows * inner.");
        }

        if (vector.Length != inner)
        {
            throw new ArgumentException("vector length must equal inner.");
        }

        var result = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            double sum = 0;
            var rowOffset = r * inner;
            for (var c = 0; c < inner; c++)
            {
                sum += (double)matrix[rowOffset + c] * vector[c];
            }

            result[r] = (float)sum;
        }

        return result;
    }

    /// <summary>
    /// Equivalent of <c>numpy.linalg.norm(a)</c> with the default order (Frobenius / L2
    /// norm over all elements).
    /// </summary>
    public static float LinalgNorm(ReadOnlySpan<float> values)
    {
        double sumOfSquares = 0;
        for (var i = 0; i < values.Length; i++)
        {
            sumOfSquares += (double)values[i] * values[i];
        }

        return (float)Math.Sqrt(sumOfSquares);
    }

    // ---------------------------------------------------------------------
    // layout conversion
    // ---------------------------------------------------------------------

    /// <summary>
    /// Converts an HWC-layout image buffer (height, width, channels) to CHW layout
    /// (channels, height, width), as used by <c>numpy.transpose(image, (2, 0, 1))</c>
    /// followed by <c>numpy.expand_dims(image, axis=0)</c> in model preprocessing.
    /// </summary>
    public static float[] TransposeHwcToChw(ReadOnlySpan<float> hwc, int height, int width, int channels)
    {
        if (hwc.Length != height * width * channels)
        {
            throw new ArgumentException("hwc length must equal height * width * channels.");
        }

        var result = new float[hwc.Length];
        var hw = height * width;
        for (var h = 0; h < height; h++)
        {
            for (var w = 0; w < width; w++)
            {
                var hwcOffset = ((h * width) + w) * channels;
                for (var c = 0; c < channels; c++)
                {
                    // CHW index: c * (H*W) + h * W + w
                    result[(c * hw) + (h * width) + w] = hwc[hwcOffset + c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts a CHW-layout tensor (channels, height, width) back to HWC layout
    /// (height, width, channels), the inverse of <see cref="TransposeHwcToChw"/>.
    /// </summary>
    public static float[] TransposeChwToHwc(ReadOnlySpan<float> chw, int channels, int height, int width)
    {
        if (chw.Length != channels * height * width)
        {
            throw new ArgumentException("chw length must equal channels * height * width.");
        }

        var result = new float[chw.Length];
        var hw = height * width;
        for (var c = 0; c < channels; c++)
        {
            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    var chwIndex = (c * hw) + (h * width) + w;
                    var hwcIndex = ((h * width) + w) * channels + c;
                    result[hwcIndex] = chw[chwIndex];
                }
            }
        }

        return result;
    }
}
