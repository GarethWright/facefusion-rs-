using System;
using System.Globalization;
using System.IO;

namespace FaceFusion.Core;

/// <summary>
/// Custom CRC32 implementation to match Python's zlib.crc32() exactly.
/// </summary>
internal static class Crc32Helper
{
	private static readonly uint[] CrcTable = InitCrcTable();

	private static uint[] InitCrcTable()
	{
		var table = new uint[256];
		for (uint n = 0; n < 256; n++)
		{
			uint crc = n;
			for (int k = 0; k < 8; k++)
			{
				if ((crc & 1) != 0)
				{
					crc = 0xedb88320U ^ (crc >> 1);
				}
				else
				{
					crc = crc >> 1;
				}
			}
			table[n] = crc;
		}
		return table;
	}

	internal static uint Calculate(byte[] data)
	{
		uint crc = 0xffffffffU;
		foreach (var byte_data in data)
		{
			crc = CrcTable[(byte)((crc ^ byte_data) & 0xff)] ^ (crc >> 8);
		}
		return crc ^ 0xffffffffU;
	}
}

/// <summary>
/// Hash helper functions for validating file integrity using CRC32.
/// Ported from facefusion/hash_helper.py.
///
/// NOTE: Uses custom CRC32 implementation compatible with Python's zlib.crc32().
/// </summary>
public static class HashHelper
{
	/// <summary>
	/// Create a CRC32 hash of the given content and return it as an 8-character
	/// lowercase hexadecimal string. This matches Python's zlib.crc32() output exactly.
	/// </summary>
	public static string CreateHash(byte[] content)
	{
		if (content == null)
		{
			throw new ArgumentNullException(nameof(content));
		}

		var crcValue = Crc32Helper.Calculate(content);
		return crcValue.ToString("x8", CultureInfo.InvariantCulture);
	}

	/// <summary>
	/// Validate that a file's hash matches the hash stored in a .hash sidecar file.
	/// Returns true if the hash file exists and matches, false otherwise.
	/// </summary>
	public static bool ValidateHash(string validatePath)
	{
		if (validatePath == null)
		{
			throw new ArgumentNullException(nameof(validatePath));
		}

		var hashPath = GetHashPath(validatePath);

		if (hashPath != null && File.Exists(hashPath))
		{
			try
			{
				var hashContent = File.ReadAllText(hashPath).Trim();
				var validateContent = File.ReadAllBytes(validatePath);
				return CreateHash(validateContent) == hashContent;
			}
			catch
			{
				return false;
			}
		}

		return false;
	}

	/// <summary>
	/// Get the .hash sidecar file path for a given file path.
	/// Returns null if the file does not exist.
	/// </summary>
	public static string? GetHashPath(string validatePath)
	{
		if (validatePath == null)
		{
			throw new ArgumentNullException(nameof(validatePath));
		}

		if (File.Exists(validatePath))
		{
			var directory = Path.GetDirectoryName(validatePath);
			var fileName = Path.GetFileNameWithoutExtension(validatePath);
			return Path.Combine(directory ?? ".", fileName + ".hash");
		}

		return null;
	}
}
