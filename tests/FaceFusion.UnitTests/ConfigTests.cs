using System;
using System.IO;
using FaceFusion.Core;

namespace FaceFusion.UnitTests;

/// <summary>
/// Ported from tests/test_config.py. Python builds one module-scoped ConfigParser (fixture
/// `before_all`) that reads facefusion.ini and then read_dict()s a handful of ad-hoc test
/// sections on top of it. Config is not a global singleton here (see Config.cs's doc comment
/// on why — tests must not depend on run order), so each test builds its own Config from the
/// equivalent ini text instead.
/// </summary>
public class ConfigTests
{
	private const string FixtureIni =
		"[str]\n" +
		"valid = a\n" +
		"unset =\n" +
		"\n" +
		"[int]\n" +
		"valid = 1\n" +
		"unset =\n" +
		"\n" +
		"[float]\n" +
		"valid = 1.0\n" +
		"unset =\n" +
		"\n" +
		"[bool]\n" +
		"valid = True\n" +
		"unset =\n" +
		"\n" +
		"[str_list]\n" +
		"valid = a b c\n" +
		"unset =\n" +
		"\n" +
		"[int_list]\n" +
		"valid = 1 2 3\n" +
		"unset =\n";

	private static Config BuildFixture()
	{
		return Config.FromText(FixtureIni);
	}

	[Fact]
	public void TestGetStrValue()
	{
		var config = BuildFixture();

		Assert.Equal("a", config.GetStrValue("str", "valid"));
		Assert.Equal("b", config.GetStrValue("str", "unset", "b"));
		Assert.Null(config.GetStrValue("str", "unset"));
		Assert.Null(config.GetStrValue("str", "invalid"));
	}

	[Fact]
	public void TestGetIntValue()
	{
		var config = BuildFixture();

		Assert.Equal(1, config.GetIntValue("int", "valid"));
		Assert.Equal(1, config.GetIntValue("int", "unset", "1"));
		Assert.Null(config.GetIntValue("int", "unset"));
		Assert.Null(config.GetIntValue("int", "invalid"));
	}

	[Fact]
	public void TestGetFloatValue()
	{
		var config = BuildFixture();

		Assert.Equal(1.0, config.GetFloatValue("float", "valid"));
		Assert.Equal(1.0, config.GetFloatValue("float", "unset", "1.0"));
		Assert.Null(config.GetFloatValue("float", "unset"));
		Assert.Null(config.GetFloatValue("float", "invalid"));
	}

	[Fact]
	public void TestGetBoolValue()
	{
		var config = BuildFixture();

		Assert.True(config.GetBoolValue("bool", "valid"));
		Assert.False(config.GetBoolValue("bool", "unset", "False"));
		Assert.Null(config.GetBoolValue("bool", "unset"));
		Assert.Null(config.GetBoolValue("bool", "invalid"));
	}

	[Fact]
	public void TestGetStrList()
	{
		var config = BuildFixture();

		Assert.Equal(new[] { "a", "b", "c" }, config.GetStrList("str_list", "valid"));
		Assert.Equal(new[] { "c", "b", "a" }, config.GetStrList("str_list", "unset", "c b a"));
		Assert.Null(config.GetStrList("str_list", "unset"));
		Assert.Null(config.GetStrList("str_list", "invalid"));
	}

	[Fact]
	public void TestGetIntList()
	{
		var config = BuildFixture();

		Assert.Equal(new[] { 1, 2, 3 }, config.GetIntList("int_list", "valid"));
		Assert.Equal(new[] { 3, 2, 1 }, config.GetIntList("int_list", "unset", "3 2 1"));
		Assert.Null(config.GetIntList("int_list", "unset"));
		Assert.Null(config.GetIntList("int_list", "invalid"));
	}

	// --- Additional coverage for the configparser semantics called out in the assignment ---
	// Each fact below documents the real Python behaviour it was checked against, via
	// `python3 -c "import configparser; ..."`.

	[Fact]
	public void TestGetStrListSplitsOnArbitraryWhitespace()
	{
		// Verified: "  a   b\tc\n".split() == ['a', 'b', 'c'] (collapses runs, strips ends).
		var config = Config.FromText("[s]\nvalue =   a   b\tc  \n");

		Assert.Equal(new[] { "a", "b", "c" }, config.GetStrList("s", "value"));
	}

	[Fact]
	public void TestGetIntListSplitsOnArbitraryWhitespaceInFallback()
	{
		// The fallback string is split the same way as a value read from the file.
		var config = Config.FromText("[s]\nother = 1\n");

		Assert.Equal(new[] { 1, 2, 3 }, config.GetIntList("s", "missing", "1\t2   3"));
	}

	[Fact]
	public void TestWhitespaceOnlyValueIsTreatedAsAbsent()
	{
		// Verified against Python: a key with an empty or whitespace-only value is treated the
		// same as a key that does not exist at all — every getter's `.strip()` check.
		var config = Config.FromText("[s]\nblank =    \n");

		Assert.Null(config.GetStrValue("s", "blank"));
		Assert.Equal("fallback", config.GetStrValue("s", "blank", "fallback"));
	}

	[Theory]
	[InlineData("1", true)]
	[InlineData("yes", true)]
	[InlineData("true", true)]
	[InlineData("on", true)]
	[InlineData("YES", true)]
	[InlineData("On", true)]
	[InlineData("TRUE", true)]
	[InlineData("0", false)]
	[InlineData("no", false)]
	[InlineData("false", false)]
	[InlineData("off", false)]
	[InlineData("NO", false)]
	[InlineData("Off", false)]
	public void TestGetBoolValueAcceptsConfigParserBooleanStates(string rawValue, bool expected)
	{
		// Verified against Python:
		//   configparser.ConfigParser().getboolean(...) for '1'/'yes'/'true'/'on' (any case)
		//   -> True, and '0'/'no'/'false'/'off' (any case) -> False.
		var config = Config.FromText($"[b]\nvalue = {rawValue}\n");

		Assert.Equal(expected, config.GetBoolValue("b", "value"));
	}

	[Fact]
	public void TestGetBoolValueRejectsUnrecognizedValue()
	{
		// Verified against Python: configparser.getboolean('nope') raises
		// ValueError("Not a boolean: nope") — config.py does not catch it, so it is not
		// silently coerced to null the way an *absent* value would be.
		var config = Config.FromText("[b]\nvalue = nope\n");

		Assert.Throws<FormatException>(() => config.GetBoolValue("b", "value"));
	}

	[Fact]
	public void TestOptionNamesAreLowerCasedButSectionsAreCaseSensitive()
	{
		// Verified against Python: ConfigParser's default optionxform lower-cases option
		// (key) names, but section names are matched case-sensitively.
		var config = Config.FromText("[Sec]\nMyKey = value\n");

		Assert.Equal("value", config.GetStrValue("Sec", "mykey"));
		Assert.Equal("value", config.GetStrValue("Sec", "MYKEY"));
		Assert.Null(config.GetStrValue("sec", "mykey"));
	}

	[Fact]
	public void TestCommentLinesAreIgnored()
	{
		var config = Config.FromText(
			"[s]\n" +
			"# a comment\n" +
			"; another comment\n" +
			"value = 1\n");

		Assert.Equal(1, config.GetIntValue("s", "value"));
	}

	[Fact]
	public void TestColonSeparatorIsAccepted()
	{
		var config = Config.FromText("[s]\nvalue: hello\n");

		Assert.Equal("hello", config.GetStrValue("s", "value"));
	}

	[Fact]
	public void TestMissingFileYieldsEmptyConfig()
	{
		// Mirrors ConfigParser().read(path): a nonexistent path is silently ignored rather
		// than raising, so every getter behaves as if every option were absent.
		var config = Config.FromFile(Path.Combine(Path.GetTempPath(), $"does-not-exist-{Guid.NewGuid():N}.ini"));

		Assert.Null(config.GetStrValue("any", "thing"));
		Assert.Equal("fallback", config.GetStrValue("any", "thing", "fallback"));
	}

	[Fact]
	public void TestRepositoryFacefusionIniLoadsAndAllOptionsAreBlank()
	{
		// facefusion.ini ships with every key present but every value blank — sanity check
		// that Config treats every one of them as absent (the whole point of the shipped file
		// is to be a documented, all-defaults template).
		var repoRoot = FindRepoRoot();
		var iniPath = Path.Combine(repoRoot, "facefusion.ini");
		var config = Config.FromFile(iniPath);

		Assert.Null(config.GetStrValue("paths", "temp_path"));
		Assert.Equal("/tmp", config.GetStrValue("paths", "temp_path", "/tmp"));
		Assert.Null(config.GetBoolValue("misc", "halt_on_error"));
	}

	private static string FindRepoRoot()
	{
		var directory = new DirectoryInfo(AppContext.BaseDirectory);

		while (directory != null)
		{
			if (File.Exists(Path.Combine(directory.FullName, "facefusion.ini")))
			{
				return directory.FullName;
			}
			directory = directory.Parent;
		}

		throw new InvalidOperationException("Could not locate facefusion.ini above " + AppContext.BaseDirectory);
	}
}
