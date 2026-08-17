using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace FaceFusion.Core;

/// <summary>
/// JSON read/write helper functions using System.Text.Json.
/// Ported from facefusion/json.py.
/// </summary>
public static class Json
{
	private static readonly JsonSerializerOptions JsonOptions = new()
	{
		WriteIndented = true,
		DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
		PropertyNamingPolicy = null, // Use property names as-is
		PropertyNameCaseInsensitive = false
	};

	/// <summary>
	/// Read and parse a JSON file. Returns null if the file does not exist
	/// or if parsing fails.
	/// </summary>
	public static JsonElement? ReadJson(string jsonPath)
	{
		if (jsonPath == null)
		{
			throw new ArgumentNullException(nameof(jsonPath));
		}

		if (!File.Exists(jsonPath))
		{
			return null;
		}

		try
		{
			var content = File.ReadAllText(jsonPath);
			return JsonSerializer.Deserialize<JsonElement>(content);
		}
		catch (JsonException)
		{
			return null;
		}
		catch (IOException)
		{
			return null;
		}
	}

	/// <summary>
	/// Write content as JSON to a file with 4-space indentation.
	/// Returns true if the file exists after writing, false otherwise.
	/// </summary>
	public static bool WriteJson(string jsonPath, object content)
	{
		if (jsonPath == null)
		{
			throw new ArgumentNullException(nameof(jsonPath));
		}

		if (content == null)
		{
			throw new ArgumentNullException(nameof(content));
		}

		try
		{
			var jsonString = JsonSerializer.Serialize(content, JsonOptions);
			File.WriteAllText(jsonPath, jsonString);
			return File.Exists(jsonPath);
		}
		catch
		{
			return false;
		}
	}
}
