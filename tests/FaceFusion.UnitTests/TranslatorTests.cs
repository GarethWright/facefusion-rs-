using System;
using System.Collections.Generic;
using FaceFusion.Core;
using Xunit;

namespace FaceFusion.UnitTests
{
	public class TranslatorTests
	{
		public TranslatorTests()
		{
			// Load the English locales for testing
			var localesDict = new Dictionary<string, object>
			{
				{ "en", Locales.En }
			};
			Translator.Load(localesDict, "facefusion");
		}

		[Fact]
		public void TestLoad()
		{
			// Test that we can load locales
			var testLocales = new Dictionary<string, object>
			{
				{
					"en", new Dictionary<string, object>
					{
						{ "test_key", "test_value" }
					}
				}
			};

			Translator.Load(testLocales, "test_module");

			// Verify that Get() can find the loaded module
			var result = Translator.Get("test_key", "test_module");
			Assert.Equal("test_value", result);
		}

		[Fact]
		public void TestGetSimpleKey()
		{
			// Test getting a simple string value
			var result = Translator.Get("processing_stopped");
			Assert.Equal("processing stopped", result);
		}

		[Fact]
		public void TestGetNestedKey()
		{
			// Test getting a nested value using dot notation
			var result = Translator.Get("help.run");
			Assert.Equal("run the program", result);
		}

		[Fact]
		public void TestGetInvalidKey()
		{
			// Test that invalid keys return null (matching Python behavior)
			var result = Translator.Get("invalid");
			Assert.Null(result);
		}

		[Fact]
		public void TestGetInvalidNestedKey()
		{
			// Test that invalid nested keys return null
			var result = Translator.Get("help.invalid");
			Assert.Null(result);
		}

		[Fact]
		public void TestFormat()
		{
			// Test the Format method with named placeholders
			var template = "processing step {step_current} of {step_total}";
			var result = Translator.Format(template, ("step_current", "1"), ("step_total", "5"));
			Assert.Equal("processing step 1 of 5", result);
		}

		[Fact]
		public void TestFormatMultiplePlaceholders()
		{
			// Test formatting with multiple different placeholders
			var template = "extracting frames with a resolution of {resolution} and {fps} frames per second";
			var result = Translator.Format(template, ("resolution", "1920x1080"), ("fps", "30"));
			Assert.Equal("extracting frames with a resolution of 1920x1080 and 30 frames per second", result);
		}

		[Fact]
		public void TestGetAndFormat()
		{
			// Test getting and formatting in one call
			var result = Translator.Get("processing_step", ("step_current", "1"), ("step_total", "5"));
			Assert.Equal("processing step 1 of 5", result);
		}

		[Fact]
		public void TestGetFromModule()
		{
			// Test getting from a specific module
			var result = Translator.Get("processing_stopped", "facefusion");
			Assert.Equal("processing stopped", result);
		}

		[Fact]
		public void TestGetAndFormatFromModule()
		{
			// Test getting from specific module and formatting
			var result = Translator.Get("processing_step", "facefusion", ("step_current", "2"), ("step_total", "10"));
			Assert.Equal("processing step 2 of 10", result);
		}

		[Fact]
		public void TestLocalesKeyCount()
		{
			// Verify that the English locales dictionary has the expected number of keys.
			// Python has 256 string values (not counting nested dict keys "help", "about", "uis").
			var stringKeyCount = CountStringValues(Locales.En);
			Assert.Equal(256, stringKeyCount);
		}

		[Fact]
		public void TestLocalesSampleValues()
		{
			// Test a sample of keys to ensure values are byte-identical to Python
			var sampleTests = new (string key, string expected)[]
			{
				("python_not_supported", "python version is not supported, upgrade to {version} or higher"),
				("dependency_not_installed", "{dependency} is not installed"),
				("creating_temp", "creating temporary resources"),
				("extracting_frames", "extracting frames with a resolution of {resolution} and {fps} frames per second"),
				("processing_stopped", "processing stopped"),
				("processing_step", "processing step {step_current} of {step_total}"),
				("no_source_face_detected", "no source face detected"),
				("help.run", "run the program"),
				("help.config_path", "choose the config file to override defaults"),
				("about.fund", "fund ai workstation"),
				("uis.apply_button", "APPLY"),
				("uis.webcam_resolution_dropdown", "WEBCAM RESOLUTION"),
				("point", "."),
				("comma", ","),
				("colon", ":"),
				("question_mark", "?"),
				("exclamation_mark", "!"),
				("time_ago_now", "just now"),
				("job_created", "job {job_id} created"),
				("loading_model_succeeded", "loading model {model_name} succeeded in {seconds} seconds")
			};

			foreach (var (key, expected) in sampleTests)
			{
				var result = Translator.Get(key);
				Assert.NotNull(result);
				Assert.Equal(expected, result);
			}
		}

		/// <summary>
		/// Helper method to count all string values in the locale dictionary,
		/// recursively descending into nested dictionaries.
		/// </summary>
		private static int CountStringValues(Dictionary<string, object> dict)
		{
			var count = 0;
			foreach (var value in dict.Values)
			{
				if (value is string)
				{
					count++;
				}
				else if (value is Dictionary<string, object> nestedDict)
				{
					count += CountStringValues(nestedDict);
				}
			}

			return count;
		}
	}
}
