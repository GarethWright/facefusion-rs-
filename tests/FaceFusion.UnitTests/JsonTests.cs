using FaceFusion.Core;
using System.IO;
using System.Text.Json;

namespace FaceFusion.UnitTests;

public class JsonTests
{
	[Fact]
	public void TestReadJsonNonExistentFile()
	{
		var result = Json.ReadJson("/nonexistent/file.json");
		Assert.Null(result);
	}

	[Fact]
	public void TestReadJsonEmptyFile()
	{
		var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

		try
		{
			File.WriteAllText(tempPath, "");
			var result = Json.ReadJson(tempPath);
			Assert.Null(result);
		}
		finally
		{
			if (File.Exists(tempPath))
			{
				File.Delete(tempPath);
			}
		}
	}

	[Fact]
	public void TestReadJsonValid()
	{
		var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

		try
		{
			File.WriteAllText(tempPath, "{}");
			var result = Json.ReadJson(tempPath);
			Assert.NotNull(result);
			Assert.Equal(JsonValueKind.Object, result.Value.ValueKind);
		}
		finally
		{
			if (File.Exists(tempPath))
			{
				File.Delete(tempPath);
			}
		}
	}

	[Fact]
	public void TestReadJsonInvalid()
	{
		var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

		try
		{
			File.WriteAllText(tempPath, "{invalid json}");
			var result = Json.ReadJson(tempPath);
			Assert.Null(result);
		}
		finally
		{
			if (File.Exists(tempPath))
			{
				File.Delete(tempPath);
			}
		}
	}

	[Fact]
	public void TestWriteJson()
	{
		var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

		try
		{
			var data = new { key = "value", number = 42 };
			var result = Json.WriteJson(tempPath, data);

			Assert.True(result);
			Assert.True(File.Exists(tempPath));

			var content = File.ReadAllText(tempPath);
			Assert.Contains("key", content);
			Assert.Contains("value", content);
			Assert.Contains("number", content);
			Assert.Contains("42", content);
		}
		finally
		{
			if (File.Exists(tempPath))
			{
				File.Delete(tempPath);
			}
		}
	}

	[Fact]
	public void TestWriteAndReadJson()
	{
		var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

		try
		{
			var data = new { test = "data" };

			var writeResult = Json.WriteJson(tempPath, data);
			Assert.True(writeResult);

			var readResult = Json.ReadJson(tempPath);
			Assert.NotNull(readResult);
		}
		finally
		{
			if (File.Exists(tempPath))
			{
				File.Delete(tempPath);
			}
		}
	}

	[Fact]
	public void TestWriteJsonEmptyObject()
	{
		var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");

		try
		{
			var data = new { };
			var result = Json.WriteJson(tempPath, data);

			Assert.True(result);
			Assert.True(File.Exists(tempPath));

			var content = File.ReadAllText(tempPath);
			Assert.Contains("{", content);
		}
		finally
		{
			if (File.Exists(tempPath))
			{
				File.Delete(tempPath);
			}
		}
	}
}
