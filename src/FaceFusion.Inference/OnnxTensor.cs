namespace FaceFusion.Inference;

/// <summary>
/// The in-memory result of decoding one ONNX <c>TensorProto</c> — the C# analogue of what
/// <c>onnx.numpy_helper.to_array</c> returns in Python. Mirrors the shape of
/// <c>FaceFusion.Parity.NpyArray</c> (data is always exposed in the host's native byte
/// order, with <see cref="AsFloats"/> / <see cref="AsDoubles"/> conversion helpers), but is
/// defined locally rather than referencing the Parity project, per the port assignment.
/// </summary>
public sealed class OnnxTensor
{
    private readonly byte[] data;

    /// <summary>The tensor shape, taken from <c>TensorProto.dims</c> (an <c>int64</c> per
    /// the proto definition, hence <see cref="long"/> here rather than <see cref="int"/>).
    /// Empty for a 0-d scalar.</summary>
    public IReadOnlyList<long> Shape { get; }

    /// <summary>
    /// The element dtype name: one of <c>"float32"</c>, <c>"float16"</c>, <c>"double"</c>,
    /// <c>"uint8"</c>, <c>"int8"</c>, <c>"int32"</c>, <c>"int64"</c> — the subset of
    /// <c>onnx.TensorProto.DataType</c> this reader supports (see
    /// <see cref="OnnxProtoReader"/>).
    /// </summary>
    public string DType { get; }

    /// <summary>Total element count (the product of <see cref="Shape"/>; 1 for a 0-d scalar).</summary>
    public int ElementCount { get; }

    /// <summary>Size in bytes of a single element for <see cref="DType"/>.</summary>
    public int ItemSize { get; }

    /// <summary>
    /// The raw element bytes, in row-major (C) order and host-native (little-endian) byte
    /// order — the same layout ONNX's <c>raw_data</c> field uses on disk. Length is
    /// <c>ElementCount * ItemSize</c>.
    /// </summary>
    public ReadOnlySpan<byte> RawData => data;

    internal OnnxTensor(IReadOnlyList<long> shape, string dtype, int itemSize, byte[] data)
    {
        Shape = shape;
        DType = dtype;
        ItemSize = itemSize;
        this.data = data;

        long elementCount = 1;
        foreach (var dimension in shape)
        {
            elementCount *= dimension;
        }

        ElementCount = checked((int)elementCount);
    }

    /// <summary>Converts every element to <see cref="double"/>, widening as necessary.</summary>
    public double[] AsDoubles()
    {
        var result = new double[ElementCount];
        var span = data.AsSpan();

        for (var index = 0; index < ElementCount; index++)
        {
            var element = span.Slice(index * ItemSize, ItemSize);
            result[index] = DType switch
            {
                "float32" => BitConverter.ToSingle(element),
                "float16" => (double)BitConverter.ToHalf(element),
                "double" => BitConverter.ToDouble(element),
                "int8" => (sbyte)element[0],
                "uint8" => element[0],
                "int32" => BitConverter.ToInt32(element),
                "int64" => BitConverter.ToInt64(element),
                _ => throw new NotSupportedException($"Cannot convert dtype '{DType}' to double.")
            };
        }

        return result;
    }

    /// <summary>Converts every element to <see cref="float"/>, narrowing as necessary.</summary>
    public float[] AsFloats()
    {
        var result = new float[ElementCount];
        var span = data.AsSpan();

        for (var index = 0; index < ElementCount; index++)
        {
            var element = span.Slice(index * ItemSize, ItemSize);
            result[index] = DType switch
            {
                "float32" => BitConverter.ToSingle(element),
                "float16" => (float)BitConverter.ToHalf(element),
                "double" => (float)BitConverter.ToDouble(element),
                "int8" => (sbyte)element[0],
                "uint8" => element[0],
                "int32" => BitConverter.ToInt32(element),
                "int64" => BitConverter.ToInt64(element),
                _ => throw new NotSupportedException($"Cannot convert dtype '{DType}' to float.")
            };
        }

        return result;
    }
}
