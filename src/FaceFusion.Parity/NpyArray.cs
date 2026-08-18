namespace FaceFusion.Parity;

/// <summary>
/// The in-memory result of loading a NumPy <c>.npy</c> file via <see cref="NpyReader"/>.
/// Data is always exposed in C (row-major) order, regardless of how the source file
/// stored it, and always in the host's native byte order.
/// </summary>
public sealed class NpyArray
{
	private readonly byte[] data;

	/// <summary>The array shape, e.g. <c>[2, 3]</c>. Empty for a 0-d scalar.</summary>
	public IReadOnlyList<int> Shape { get; }

	/// <summary>The NumPy dtype name, e.g. <c>"float32"</c>, <c>"int64"</c>, <c>"bool"</c>.</summary>
	public string DType { get; }

	/// <summary>Total element count (the product of <see cref="Shape"/>; 1 for a 0-d scalar).</summary>
	public int ElementCount { get; }

	/// <summary>Size in bytes of a single element for <see cref="DType"/>.</summary>
	public int ItemSize { get; }

	/// <summary>
	/// The raw element bytes, in C order and host-native byte order. Length is
	/// <c>ElementCount * ItemSize</c>.
	/// </summary>
	public ReadOnlySpan<byte> RawData => data;

	internal NpyArray(IReadOnlyList<int> shape, string dtype, int itemSize, byte[] data)
	{
		Shape = shape;
		DType = dtype;
		ItemSize = itemSize;
		this.data = data;

		var elementCount = 1;
		foreach (var dimension in shape)
		{
			elementCount *= dimension;
		}

		ElementCount = elementCount;
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
				"float64" => BitConverter.ToDouble(element),
				"int8" => (sbyte)element[0],
				"int16" => BitConverter.ToInt16(element),
				"int32" => BitConverter.ToInt32(element),
				"int64" => BitConverter.ToInt64(element),
				"uint8" => element[0],
				"uint16" => BitConverter.ToUInt16(element),
				"uint32" => BitConverter.ToUInt32(element),
				"uint64" => BitConverter.ToUInt64(element),
				"bool" => element[0] != 0 ? 1.0 : 0.0,
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
				"float64" => (float)BitConverter.ToDouble(element),
				"int8" => (sbyte)element[0],
				"int16" => BitConverter.ToInt16(element),
				"int32" => BitConverter.ToInt32(element),
				"int64" => BitConverter.ToInt64(element),
				"uint8" => element[0],
				"uint16" => BitConverter.ToUInt16(element),
				"uint32" => BitConverter.ToUInt32(element),
				"uint64" => BitConverter.ToUInt64(element),
				"bool" => element[0] != 0 ? 1f : 0f,
				_ => throw new NotSupportedException($"Cannot convert dtype '{DType}' to float.")
			};
		}

		return result;
	}
}
