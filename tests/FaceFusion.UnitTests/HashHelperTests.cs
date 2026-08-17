using FaceFusion.Core;
using System.IO;
using System.Text;

namespace FaceFusion.UnitTests;

public class HashHelperTests
{
	[Theory]
	[InlineData(new byte[] { }, "00000000")]
	[InlineData(new byte[] { (byte)'t', (byte)'e', (byte)'s', (byte)'t' }, "d87f7e0c")]
	[InlineData(new byte[] { (byte)'h', (byte)'e', (byte)'l', (byte)'l', (byte)'o', (byte)' ', (byte)'w', (byte)'o', (byte)'r', (byte)'l', (byte)'d' }, "0d4a1185")]
	[InlineData(new byte[] { 0x00, 0x01, 0x02, 0x03 }, "8bb98613")]
	public void TestCreateHashMatchesPython(byte[] content, string expectedHash)
	{
		// These values are verified against Python's zlib.crc32() output
		var result = HashHelper.CreateHash(content);
		Assert.Equal(expectedHash, result);
	}

	[Fact]
	public void TestCreateHashReturnsLowercaseHex()
	{
		var result = HashHelper.CreateHash(Encoding.UTF8.GetBytes("test"));
		Assert.True(result.All(c => char.IsDigit(c) || (c >= 'a' && c <= 'f')));
		Assert.Equal(8, result.Length);
	}

	[Fact]
	public void TestValidateHashWithExistingHashFile()
	{
		var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
		Directory.CreateDirectory(tempDir);

		try
		{
			var filePath = Path.Combine(tempDir, "test.bin");
			var content = Encoding.UTF8.GetBytes("test");
			File.WriteAllBytes(filePath, content);

			var expectedHash = HashHelper.CreateHash(content);
			var hashPath = Path.Combine(tempDir, "test.hash");
			File.WriteAllText(hashPath, expectedHash);

			Assert.True(HashHelper.ValidateHash(filePath));
		}
		finally
		{
			Directory.Delete(tempDir, true);
		}
	}

	[Fact]
	public void TestValidateHashWithMismatchedHash()
	{
		var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
		Directory.CreateDirectory(tempDir);

		try
		{
			var filePath = Path.Combine(tempDir, "test.bin");
			var content = Encoding.UTF8.GetBytes("test");
			File.WriteAllBytes(filePath, content);

			var wrongHash = "wronghash";
			var hashPath = Path.Combine(tempDir, "test.hash");
			File.WriteAllText(hashPath, wrongHash);

			Assert.False(HashHelper.ValidateHash(filePath));
		}
		finally
		{
			Directory.Delete(tempDir, true);
		}
	}

	[Fact]
	public void TestValidateHashWithNoHashFile()
	{
		var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
		Directory.CreateDirectory(tempDir);

		try
		{
			var filePath = Path.Combine(tempDir, "test.bin");
			var content = Encoding.UTF8.GetBytes("test");
			File.WriteAllBytes(filePath, content);

			// No .hash file exists
			Assert.False(HashHelper.ValidateHash(filePath));
		}
		finally
		{
			Directory.Delete(tempDir, true);
		}
	}

	[Fact]
	public void TestGetHashPath()
	{
		var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
		Directory.CreateDirectory(tempDir);

		try
		{
			var filePath = Path.Combine(tempDir, "myfile.bin");
			File.WriteAllBytes(filePath, new byte[] { 1, 2, 3 });

			var hashPath = HashHelper.GetHashPath(filePath);
			Assert.NotNull(hashPath);
			Assert.Equal(Path.Combine(tempDir, "myfile.hash"), hashPath);
		}
		finally
		{
			Directory.Delete(tempDir, true);
		}
	}

	[Fact]
	public void TestGetHashPathNonExistentFile()
	{
		var hashPath = HashHelper.GetHashPath("/nonexistent/file.bin");
		Assert.Null(hashPath);
	}
}
