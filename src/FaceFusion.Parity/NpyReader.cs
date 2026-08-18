using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace FaceFusion.Parity;

/// <summary>
/// Reads NumPy <c>.npy</c> files (format versions 1.0, 2.0 and 3.0) into <see cref="NpyArray"/>.
/// See https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html for the format.
/// </summary>
public static class NpyReader
{
	private static readonly byte[] Magic = { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

	private static readonly Regex DescrPattern = new("'descr'\\s*:\\s*'([^']*)'", RegexOptions.Compiled);
	private static readonly Regex FortranOrderPattern = new("'fortran_order'\\s*:\\s*(True|False)", RegexOptions.Compiled);
	private static readonly Regex ShapePattern = new("'shape'\\s*:\\s*\\(([^)]*)\\)", RegexOptions.Compiled);

	/// <summary>Loads a <c>.npy</c> array from a file path.</summary>
	public static NpyArray Load(string path)
	{
		// Committed fixtures may be gzipped: model-input tensors are multi-megabyte
		// float32 arrays that compress about 4x, which matters because the fixture corpus
		// grows with every ported stage. A caller still asks for "foo.npy"; if only
		// "foo.npy.gz" is on disk it is transparently decompressed, so compressing a
		// fixture never means touching the test that reads it.
		if (!File.Exists(path) && File.Exists(path + ".gz"))
		{
			path += ".gz";
		}

		using var fileStream = File.OpenRead(path);

		if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
		{
			using var gzipStream = new System.IO.Compression.GZipStream(fileStream, System.IO.Compression.CompressionMode.Decompress);
			// GZipStream is forward-only and the reader seeks, so materialise first.
			using var buffer = new MemoryStream();
			gzipStream.CopyTo(buffer);
			buffer.Position = 0;
			return Load(buffer);
		}

		return Load(fileStream);
	}

	/// <summary>Loads a <c>.npy</c> array from a stream. The stream is read from its current position.</summary>
	public static NpyArray Load(Stream stream)
	{
		var magic = ReadExact(stream, 6);
		if (!magic.AsSpan().SequenceEqual(Magic))
		{
			throw new InvalidDataException("Not a .npy file: missing '\\x93NUMPY' magic.");
		}

		var versionBytes = ReadExact(stream, 2);
		var majorVersion = versionBytes[0];
		var minorVersion = versionBytes[1];

		var headerLengthFieldSize = majorVersion == 1 ? 2 : 4;
		var headerLengthBytes = ReadExact(stream, headerLengthFieldSize);
		var headerLength = headerLengthFieldSize == 2
			? BinaryPrimitivesToUInt16LittleEndian(headerLengthBytes)
			: BinaryPrimitivesToUInt32LittleEndian(headerLengthBytes);

		var headerBytes = ReadExact(stream, (int)headerLength);
		// v1.0/v2.0 headers are latin-1; v3.0 headers are utf-8. Both decode identically for
		// the ascii-only dict literals numpy.lib.format actually emits, so utf-8 is used throughout.
		var header = Encoding.UTF8.GetString(headerBytes);

		var (dtype, itemSize, byteOrder) = ParseDescr(header);
		var fortranOrder = ParseFortranOrder(header);
		var shape = ParseShape(header);

		var elementCount = 1;
		foreach (var dimension in shape)
		{
			elementCount *= dimension;
		}

		var byteCount = elementCount * itemSize;
		var raw = ReadExact(stream, byteCount);

		NormalizeByteOrder(raw, itemSize, byteOrder);

		var ordered = fortranOrder
			? FortranToCOrder(raw, shape, itemSize)
			: raw;

		return new NpyArray(shape, dtype, itemSize, ordered);
	}

	private static byte[] ReadExact(Stream stream, int count)
	{
		var buffer = new byte[count];
		var offset = 0;

		while (offset < count)
		{
			var read = stream.Read(buffer, offset, count - offset);
			if (read == 0)
			{
				throw new EndOfStreamException("Unexpected end of stream while reading .npy data.");
			}

			offset += read;
		}

		return buffer;
	}

	private static ushort BinaryPrimitivesToUInt16LittleEndian(byte[] bytes)
	{
		return (ushort)(bytes[0] | (bytes[1] << 8));
	}

	private static uint BinaryPrimitivesToUInt32LittleEndian(byte[] bytes)
	{
		return (uint)(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
	}

	private static (string DType, int ItemSize, char ByteOrder) ParseDescr(string header)
	{
		var match = DescrPattern.Match(header);
		if (!match.Success)
		{
			throw new InvalidDataException("Malformed .npy header: missing 'descr' field.");
		}

		var descr = match.Groups[1].Value;
		if (descr.Length < 2)
		{
			throw new InvalidDataException($"Malformed .npy 'descr' value: '{descr}'.");
		}

		var byteOrder = descr[0];
		var typeChar = descr[1];
		var sizeText = descr[2..];

		if (typeChar == 'O')
		{
			throw new NotSupportedException(
				"Object-dtype .npy arrays (allow_pickle) are not supported by NpyReader.");
		}

		var size = sizeText.Length > 0 ? int.Parse(sizeText, CultureInfo.InvariantCulture) : 1;

		string dtype = (typeChar, size) switch
		{
			('f', 4) => "float32",
			('f', 8) => "float64",
			('i', 1) => "int8",
			('i', 2) => "int16",
			('i', 4) => "int32",
			('i', 8) => "int64",
			('u', 1) => "uint8",
			('u', 2) => "uint16",
			('u', 4) => "uint32",
			('u', 8) => "uint64",
			('b', 1) => "bool",
			_ => throw new NotSupportedException($"Unsupported .npy dtype descriptor '{descr}'.")
		};

		return (dtype, size, byteOrder);
	}

	private static bool ParseFortranOrder(string header)
	{
		var match = FortranOrderPattern.Match(header);
		if (!match.Success)
		{
			throw new InvalidDataException("Malformed .npy header: missing 'fortran_order' field.");
		}

		return match.Groups[1].Value == "True";
	}

	private static int[] ParseShape(string header)
	{
		var match = ShapePattern.Match(header);
		if (!match.Success)
		{
			throw new InvalidDataException("Malformed .npy header: missing 'shape' field.");
		}

		var inner = match.Groups[1].Value.Trim();
		if (inner.Length == 0)
		{
			// 0-d scalar: 'shape': ()
			return Array.Empty<int>();
		}

		var parts = inner.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
		var shape = new int[parts.Length];
		for (var index = 0; index < parts.Length; index++)
		{
			shape[index] = int.Parse(parts[index], CultureInfo.InvariantCulture);
		}

		return shape;
	}

	private static void NormalizeByteOrder(byte[] data, int itemSize, char byteOrder)
	{
		if (itemSize <= 1)
		{
			return;
		}

		bool sourceIsLittleEndian = byteOrder switch
		{
			'<' => true,
			'>' => false,
			// '|' (not applicable) and '=' (native) are already in host order.
			_ => BitConverter.IsLittleEndian
		};

		if (sourceIsLittleEndian == BitConverter.IsLittleEndian)
		{
			return;
		}

		for (var offset = 0; offset < data.Length; offset += itemSize)
		{
			Array.Reverse(data, offset, itemSize);
		}
	}

	private static byte[] FortranToCOrder(byte[] source, IReadOnlyList<int> shape, int itemSize)
	{
		var rank = shape.Count;
		var elementCount = 1;
		foreach (var dimension in shape)
		{
			elementCount *= dimension;
		}

		var destination = new byte[source.Length];
		if (rank <= 1)
		{
			// A rank-0 or rank-1 array is identical in C and Fortran order.
			Array.Copy(source, destination, source.Length);
			return destination;
		}

		var fortranStrides = new int[rank];
		fortranStrides[0] = 1;
		for (var dimension = 1; dimension < rank; dimension++)
		{
			fortranStrides[dimension] = fortranStrides[dimension - 1] * shape[dimension - 1];
		}

		var multiIndex = new int[rank];
		for (var cIndex = 0; cIndex < elementCount; cIndex++)
		{
			var remainder = cIndex;
			for (var dimension = rank - 1; dimension >= 0; dimension--)
			{
				multiIndex[dimension] = remainder % shape[dimension];
				remainder /= shape[dimension];
			}

			var fortranIndex = 0;
			for (var dimension = 0; dimension < rank; dimension++)
			{
				fortranIndex += multiIndex[dimension] * fortranStrides[dimension];
			}

			Array.Copy(source, fortranIndex * itemSize, destination, cIndex * itemSize, itemSize);
		}

		return destination;
	}
}
