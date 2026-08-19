namespace FaceFusion.Inference;

/// <summary>
/// A minimal, hand-written protobuf wire-format reader for the one thing
/// <c>facefusion/model_helper.py</c> needs out of an <c>.onnx</c> file: the last
/// <c>TensorProto</c> in <c>ModelProto.graph.initializer</c>.
///
/// <para>
/// Deliberately not built on generated <c>onnx.proto</c> message classes — see
/// docs/DOTNET_PORT_PLAN.md section 2: "A minimal <c>TensorProto</c> decode is enough — do
/// not pull in a full ONNX graph library." A generated <c>ModelProto</c>/<c>GraphProto</c>
/// class pair would force <c>Google.Protobuf</c> to materialise every node, every
/// value_info, and every initializer's bytes into memory just to reach the last one, and
/// real ONNX weight files run into the hundreds of megabytes. Instead this reader walks
/// the wire format field-by-field, using <see cref="Stream.Seek"/> to skip anything that
/// is not <c>graph</c> (field 7) or, inside the graph, not <c>initializer</c> (field 5),
/// and skips every initializer except the last one by seeking over its bytes instead of
/// parsing them. Only the final <c>TensorProto</c> is ever fully decoded, and only its
/// <c>raw_data</c>/typed-data payload is copied into memory.
/// </para>
///
/// <para>
/// Protobuf wire format reference (proto3, as used by onnx.proto):
/// a field is a varint tag (<c>field_number &lt;&lt; 3 | wire_type</c>) followed by a
/// payload whose shape depends on <c>wire_type</c>: 0 = varint, 1 = 8-byte fixed64,
/// 2 = length-delimited (string/bytes/embedded message/packed repeated scalar),
/// 5 = 4-byte fixed32. Wire types 3/4 (deprecated groups) are not used by onnx.proto and
/// are not supported here.
/// </para>
/// </summary>
internal static class OnnxProtoReader
{
    private const int WireVarint = 0;
    private const int WireFixed64 = 1;
    private const int WireLengthDelimited = 2;
    private const int WireFixed32 = 5;

    // ModelProto field numbers (onnx.proto3).
    private const int ModelGraphField = 7;

    // GraphProto field numbers.
    private const int GraphInitializerField = 5;

    // TensorProto field numbers.
    private const int TensorDimsField = 1;
    private const int TensorDataTypeField = 2;
    private const int TensorSegmentField = 3;
    private const int TensorFloatDataField = 4;
    private const int TensorInt32DataField = 5;
    private const int TensorStringDataField = 6;
    private const int TensorInt64DataField = 7;
    private const int TensorNameField = 8;
    private const int TensorRawDataField = 9;
    private const int TensorDoubleDataField = 10;
    private const int TensorUint64DataField = 11;
    private const int TensorDocStringField = 12;
    private const int TensorExternalDataField = 13;
    private const int TensorDataLocationField = 14;

    // onnx.TensorProto.DataType values this reader understands. FaceFusion's static
    // model initializers are plain numeric matrices, so this deliberately does not cover
    // STRING, BOOL, UINT16/INT16, UINT32/UINT64, COMPLEX64/128, BFLOAT16 or the FLOAT8*
    // family — an unsupported data_type raises NotSupportedException rather than
    // returning garbage.
    private const int DataTypeFloat = 1;
    private const int DataTypeUInt8 = 2;
    private const int DataTypeInt8 = 3;
    private const int DataTypeInt32 = 6;
    private const int DataTypeInt64 = 7;
    private const int DataTypeFloat16 = 10;
    private const int DataTypeDouble = 11;

    /// <summary>
    /// Reads <paramref name="stream"/> as a <c>ModelProto</c> and returns the last tensor
    /// in <c>graph.initializer</c>, decoded to an <see cref="OnnxTensor"/>. The stream
    /// must be seekable (a <see cref="FileStream"/> opened for reading qualifies).
    /// </summary>
    public static OnnxTensor ReadLastInitializer(Stream stream)
    {
        if (!stream.CanSeek)
        {
            throw new ArgumentException("OnnxProtoReader requires a seekable stream.", nameof(stream));
        }

        var graphRange = FindGraph(stream);
        if (graphRange is null)
        {
            throw new InvalidDataException("ONNX file has no top-level 'graph' field (ModelProto.graph).");
        }

        var lastInitializerRange = FindLastInitializer(stream, graphRange.Value);
        if (lastInitializerRange is null)
        {
            throw new InvalidDataException("ONNX graph has no initializer tensors (GraphProto.initializer is empty).");
        }

        stream.Position = lastInitializerRange.Value.Start;
        return ParseTensorProto(stream, lastInitializerRange.Value.End);
    }

    // -----------------------------------------------------------------
    // ModelProto / GraphProto traversal — only ever reads bytes for tags and lengths;
    // everything else is skipped with Seek.
    // -----------------------------------------------------------------

    private static (long Start, long End)? FindGraph(Stream stream)
    {
        var end = stream.Length;
        stream.Position = 0;

        while (stream.Position < end)
        {
            var (fieldNumber, wireType) = ReadTag(stream);

            if (fieldNumber == ModelGraphField)
            {
                if (wireType != WireLengthDelimited)
                {
                    throw new InvalidDataException($"ONNX ModelProto.graph has unexpected wire type {wireType} (expected length-delimited).");
                }

                var length = (long)ReadVarint(stream);
                var start = stream.Position;
                return (start, start + length);
            }

            SkipField(stream, wireType);
        }

        return null;
    }

    private static (long Start, long End)? FindLastInitializer(Stream stream, (long Start, long End) graphRange)
    {
        stream.Position = graphRange.Start;
        (long Start, long End)? last = null;

        while (stream.Position < graphRange.End)
        {
            var (fieldNumber, wireType) = ReadTag(stream);

            if (fieldNumber == GraphInitializerField)
            {
                if (wireType != WireLengthDelimited)
                {
                    throw new InvalidDataException($"ONNX GraphProto.initializer has unexpected wire type {wireType} (expected length-delimited).");
                }

                var length = (long)ReadVarint(stream);
                var start = stream.Position;
                last = (start, start + length);
                stream.Position = start + length;
            }
            else
            {
                SkipField(stream, wireType);
            }
        }

        return last;
    }

    // -----------------------------------------------------------------
    // TensorProto decoding — the only message this reader fully parses.
    // -----------------------------------------------------------------

    private static OnnxTensor ParseTensorProto(Stream stream, long end)
    {
        var dims = new List<long>();
        int? dataType = null;
        string? name = null;
        byte[]? rawData = null;
        List<float>? floatData = null;
        List<int>? int32Data = null;
        List<long>? int64Data = null;
        List<double>? doubleData = null;
        var hasSegment = false;
        var hasExternalData = false;

        while (stream.Position < end)
        {
            var (fieldNumber, wireType) = ReadTag(stream);

            switch (fieldNumber)
            {
                case TensorDimsField:
                    ReadPackedInt64(stream, wireType, dims);
                    break;

                case TensorDataTypeField:
                    RequireWireType(fieldNumber, wireType, WireVarint);
                    dataType = unchecked((int)ReadVarint(stream));
                    break;

                case TensorSegmentField:
                    // Segment splits a single tensor's data across multiple TensorProto
                    // entries; none of FaceFusion's models use it, and silently ignoring
                    // it would return a truncated tensor, so this is a hard error instead.
                    hasSegment = true;
                    SkipField(stream, wireType);
                    break;

                case TensorFloatDataField:
                    floatData ??= new List<float>();
                    ReadPackedFloat(stream, wireType, floatData);
                    break;

                case TensorInt32DataField:
                    int32Data ??= new List<int>();
                    ReadPackedInt32(stream, wireType, int32Data);
                    break;

                case TensorStringDataField:
                    // Only relevant for STRING tensors, which this reader does not
                    // support (see DataType* constants above); skip.
                    SkipField(stream, wireType);
                    break;

                case TensorInt64DataField:
                    int64Data ??= new List<long>();
                    ReadPackedInt64(stream, wireType, int64Data);
                    break;

                case TensorNameField:
                    RequireWireType(fieldNumber, wireType, WireLengthDelimited);
                    name = ReadLengthDelimitedString(stream);
                    break;

                case TensorRawDataField:
                    RequireWireType(fieldNumber, wireType, WireLengthDelimited);
                    rawData = ReadLengthDelimitedBytes(stream);
                    break;

                case TensorDoubleDataField:
                    doubleData ??= new List<double>();
                    ReadPackedDouble(stream, wireType, doubleData);
                    break;

                case TensorUint64DataField:
                    // Not in the supported data_type set; skip.
                    SkipField(stream, wireType);
                    break;

                case TensorDocStringField:
                    SkipField(stream, wireType);
                    break;

                case TensorExternalDataField:
                    hasExternalData = true;
                    SkipField(stream, wireType);
                    break;

                case TensorDataLocationField:
                    RequireWireType(fieldNumber, wireType, WireVarint);
                    var location = ReadVarint(stream);
                    if (location != 0)
                    {
                        // onnx.TensorProto.DataLocation: DEFAULT = 0, EXTERNAL = 1.
                        hasExternalData = true;
                    }
                    break;

                default:
                    SkipField(stream, wireType);
                    break;
            }
        }

        var tensorLabel = name is null ? "<unnamed>" : $"'{name}'";

        if (hasExternalData)
        {
            throw new NotSupportedException(
                $"ONNX tensor {tensorLabel} stores its data externally (TensorProto.data_location = EXTERNAL / " +
                "external_data is set). This minimal reader only supports tensors whose data is embedded in the " +
                "model file (raw_data or the typed *_data fields).");
        }

        if (hasSegment)
        {
            throw new NotSupportedException(
                $"ONNX tensor {tensorLabel} splits its data across TensorProto.segment entries, which this " +
                "minimal reader does not support.");
        }

        if (dataType is null)
        {
            throw new InvalidDataException($"ONNX tensor {tensorLabel} has no data_type.");
        }

        var (dtypeName, itemSize) = DescribeDataType(dataType.Value, tensorLabel);

        long elementCount = 1;
        foreach (var dimension in dims)
        {
            elementCount *= dimension;
        }

        var data = BuildElementBytes(dataType.Value, dtypeName, itemSize, elementCount, tensorLabel, rawData, floatData, int32Data, int64Data, doubleData);

        return new OnnxTensor(dims, dtypeName, itemSize, data);
    }

    private static (string Name, int ItemSize) DescribeDataType(int dataType, string tensorLabel) => dataType switch
    {
        DataTypeFloat => ("float32", 4),
        DataTypeUInt8 => ("uint8", 1),
        DataTypeInt8 => ("int8", 1),
        DataTypeInt32 => ("int32", 4),
        DataTypeInt64 => ("int64", 8),
        DataTypeFloat16 => ("float16", 2),
        DataTypeDouble => ("double", 8),
        _ => throw new NotSupportedException(
            $"ONNX tensor {tensorLabel} has data_type {dataType}, which this minimal reader does not support " +
            "(supported: FLOAT=1, UINT8=2, INT8=3, INT32=6, INT64=7, FLOAT16=10, DOUBLE=11).")
    };

    private static byte[] BuildElementBytes(
        int dataType,
        string dtypeName,
        int itemSize,
        long elementCount,
        string tensorLabel,
        byte[]? rawData,
        List<float>? floatData,
        List<int>? int32Data,
        List<long>? int64Data,
        List<double>? doubleData)
    {
        var expectedByteLength = checked((int)(elementCount * itemSize));

        // ONNX allows a tensor's values to live in EITHER raw_data OR the type-specific
        // repeated field, never both meaningfully at once; raw_data wins if present,
        // matching onnx.numpy_helper.to_array's own precedence.
        if (rawData is not null)
        {
            if (rawData.Length != expectedByteLength)
            {
                throw new InvalidDataException(
                    $"ONNX tensor {tensorLabel} has raw_data of {rawData.Length} bytes, expected " +
                    $"{expectedByteLength} bytes for {elementCount} '{dtypeName}' elements.");
            }

            // raw_data is little-endian per the ONNX spec; .NET runs little-endian on
            // every platform this project targets, so no byte-swap is needed (same
            // assumption FaceFusion.Parity.NpyArray makes for .npy files).
            return rawData;
        }

        switch (dataType)
        {
            case DataTypeFloat:
                return FromTypedList(floatData, elementCount, tensorLabel, "float_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    Buffer.BlockCopy(values.ToArray(), 0, bytes, 0, expectedByteLength);
                    return bytes;
                });

            case DataTypeDouble:
                return FromTypedList(doubleData, elementCount, tensorLabel, "double_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    Buffer.BlockCopy(values.ToArray(), 0, bytes, 0, expectedByteLength);
                    return bytes;
                });

            case DataTypeInt64:
                return FromTypedList(int64Data, elementCount, tensorLabel, "int64_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    Buffer.BlockCopy(values.ToArray(), 0, bytes, 0, expectedByteLength);
                    return bytes;
                });

            case DataTypeInt32:
                // Plain int32: each widened value maps directly to a 4-byte LE word.
                return FromTypedList(int32Data, elementCount, tensorLabel, "int32_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    Buffer.BlockCopy(values.ToArray(), 0, bytes, 0, expectedByteLength);
                    return bytes;
                });

            case DataTypeUInt8:
                // Per onnx.proto: uint8 values are widened into int32_data.
                return FromTypedList(int32Data, elementCount, tensorLabel, "int32_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    for (var i = 0; i < values.Count; i++)
                    {
                        bytes[i] = unchecked((byte)values[i]);
                    }

                    return bytes;
                });

            case DataTypeInt8:
                // Per onnx.proto: int8 values are widened into int32_data; truncating
                // back to a byte reproduces the original two's-complement bit pattern.
                return FromTypedList(int32Data, elementCount, tensorLabel, "int32_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    for (var i = 0; i < values.Count; i++)
                    {
                        bytes[i] = unchecked((byte)values[i]);
                    }

                    return bytes;
                });

            case DataTypeFloat16:
                // Per onnx.proto: float16 values are bit-cast to uint16 and widened into
                // int32_data; take the low 16 bits back out as the half's bit pattern.
                return FromTypedList(int32Data, elementCount, tensorLabel, "int32_data", values =>
                {
                    var bytes = new byte[expectedByteLength];
                    for (var i = 0; i < values.Count; i++)
                    {
                        var bits = unchecked((ushort)values[i]);
                        bytes[i * 2] = (byte)(bits & 0xFF);
                        bytes[i * 2 + 1] = (byte)(bits >> 8);
                    }

                    return bytes;
                });

            default:
                // Unreachable: DescribeDataType already validated dataType above.
                throw new NotSupportedException($"ONNX tensor {tensorLabel} has unsupported data_type {dataType}.");
        }
    }

    private static byte[] FromTypedList<T>(List<T>? values, long elementCount, string tensorLabel, string fieldName, Func<List<T>, byte[]> convert)
    {
        if (values is null || values.Count != elementCount)
        {
            var actual = values is null ? "no" : values.Count.ToString(System.Globalization.CultureInfo.InvariantCulture);
            throw new InvalidDataException(
                $"ONNX tensor {tensorLabel} has neither raw_data nor a matching {fieldName} (expected " +
                $"{elementCount} elements, found {actual}).");
        }

        return convert(values);
    }

    // -----------------------------------------------------------------
    // Packed / unpacked repeated scalar fields. proto3 defaults repeated scalar fields to
    // packed encoding (a single length-delimited field containing back-to-back values),
    // but the wire format still accepts the legacy unpacked form (one tag+value per
    // element) for compatibility, so both are handled here.
    // -----------------------------------------------------------------

    private static void ReadPackedInt64(Stream stream, int wireType, List<long> target)
    {
        if (wireType == WireLengthDelimited)
        {
            var length = (long)ReadVarint(stream);
            var end = stream.Position + length;
            while (stream.Position < end)
            {
                target.Add(unchecked((long)ReadVarint(stream)));
            }
        }
        else if (wireType == WireVarint)
        {
            target.Add(unchecked((long)ReadVarint(stream)));
        }
        else
        {
            throw new InvalidDataException($"Unexpected wire type {wireType} for an int64 field.");
        }
    }

    private static void ReadPackedInt32(Stream stream, int wireType, List<int> target)
    {
        if (wireType == WireLengthDelimited)
        {
            var length = (long)ReadVarint(stream);
            var end = stream.Position + length;
            while (stream.Position < end)
            {
                target.Add(unchecked((int)(long)ReadVarint(stream)));
            }
        }
        else if (wireType == WireVarint)
        {
            target.Add(unchecked((int)(long)ReadVarint(stream)));
        }
        else
        {
            throw new InvalidDataException($"Unexpected wire type {wireType} for an int32 field.");
        }
    }

    private static void ReadPackedFloat(Stream stream, int wireType, List<float> target)
    {
        if (wireType == WireLengthDelimited)
        {
            var length = (long)ReadVarint(stream);
            var end = stream.Position + length;
            while (stream.Position < end)
            {
                target.Add(BitConverter.ToSingle(ReadExact(stream, 4)));
            }
        }
        else if (wireType == WireFixed32)
        {
            target.Add(BitConverter.ToSingle(ReadExact(stream, 4)));
        }
        else
        {
            throw new InvalidDataException($"Unexpected wire type {wireType} for a float field.");
        }
    }

    private static void ReadPackedDouble(Stream stream, int wireType, List<double> target)
    {
        if (wireType == WireLengthDelimited)
        {
            var length = (long)ReadVarint(stream);
            var end = stream.Position + length;
            while (stream.Position < end)
            {
                target.Add(BitConverter.ToDouble(ReadExact(stream, 8)));
            }
        }
        else if (wireType == WireFixed64)
        {
            target.Add(BitConverter.ToDouble(ReadExact(stream, 8)));
        }
        else
        {
            throw new InvalidDataException($"Unexpected wire type {wireType} for a double field.");
        }
    }

    // -----------------------------------------------------------------
    // Low-level wire format primitives.
    // -----------------------------------------------------------------

    private static (int FieldNumber, int WireType) ReadTag(Stream stream)
    {
        var tag = ReadVarint(stream);
        return (unchecked((int)(tag >> 3)), unchecked((int)(tag & 0x7)));
    }

    private static void RequireWireType(int fieldNumber, int actual, int expected)
    {
        if (actual != expected)
        {
            throw new InvalidDataException($"ONNX TensorProto field {fieldNumber} has unexpected wire type {actual} (expected {expected}).");
        }
    }

    private static string ReadLengthDelimitedString(Stream stream)
    {
        return System.Text.Encoding.UTF8.GetString(ReadLengthDelimitedBytes(stream));
    }

    private static byte[] ReadLengthDelimitedBytes(Stream stream)
    {
        var length = (long)ReadVarint(stream);
        if (length == 0)
        {
            return Array.Empty<byte>();
        }

        return ReadExact(stream, checked((int)length));
    }

    private static void SkipField(Stream stream, int wireType)
    {
        switch (wireType)
        {
            case WireVarint:
                ReadVarint(stream);
                break;

            case WireFixed64:
                Seek(stream, 8);
                break;

            case WireLengthDelimited:
                var length = (long)ReadVarint(stream);
                Seek(stream, length);
                break;

            case WireFixed32:
                Seek(stream, 4);
                break;

            default:
                throw new NotSupportedException($"Unsupported protobuf wire type {wireType} (groups are not supported).");
        }
    }

    private static void Seek(Stream stream, long byteCount)
    {
        var target = stream.Position + byteCount;
        if (target > stream.Length || byteCount < 0)
        {
            throw new EndOfStreamException("Truncated ONNX protobuf stream: field length runs past end of file.");
        }

        stream.Position = target;
    }

    private static byte[] ReadExact(Stream stream, int byteCount)
    {
        var buffer = new byte[byteCount];
        var offset = 0;

        while (offset < byteCount)
        {
            var read = stream.Read(buffer, offset, byteCount - offset);
            if (read == 0)
            {
                throw new EndOfStreamException("Truncated ONNX protobuf stream while reading a field payload.");
            }

            offset += read;
        }

        return buffer;
    }

    private static ulong ReadVarint(Stream stream)
    {
        ulong result = 0;
        var shift = 0;

        for (var i = 0; i < 10; i++)
        {
            var b = stream.ReadByte();
            if (b < 0)
            {
                throw new EndOfStreamException("Truncated ONNX protobuf stream while reading a varint.");
            }

            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return result;
            }

            shift += 7;
        }

        throw new InvalidDataException("Malformed protobuf varint (more than 10 continuation bytes).");
    }
}
