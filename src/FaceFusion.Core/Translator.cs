using System;
using System.Collections.Generic;
using System.Linq;

namespace FaceFusion.Core
{
	/// <summary>
	/// Translator for localized messages.
	/// Ported from facefusion/translator.py
	/// </summary>
	public static class Translator
	{
		private static readonly Dictionary<string, Dictionary<string, object>> LocalePoolSet =
			new Dictionary<string, Dictionary<string, object>>();

		private static string CurrentLanguage = "en";

		/// <summary>
		/// Load a locales dictionary for a given module name.
		/// Ported from facefusion/translator.py load().
		/// </summary>
		/// <param name="locales">Dictionary mapping language codes to locale dictionaries</param>
		/// <param name="moduleName">Module identifier</param>
		public static void Load(Dictionary<string, object> locales, string moduleName)
		{
			LocalePoolSet[moduleName] = locales;
		}

		/// <summary>
		/// Get a localized message using dot notation for nested keys.
		/// Returns null if the key is not found, matching Python behavior.
		/// Ported from facefusion/translator.py get().
		/// </summary>
		/// <param name="notation">Dot-separated key notation (e.g., "help.run", "processing_stopped")</param>
		/// <param name="moduleName">Module name, defaults to "facefusion"</param>
		/// <returns>The localized message string, or null if not found</returns>
		public static string? Get(string notation, string moduleName = "facefusion")
		{
			if (!LocalePoolSet.ContainsKey(moduleName))
			{
				// Auto-load if needed (in C#, we just return null since we can't do dynamic imports)
				// The Python code would try to import the module, but in C# we rely on explicit Load()
			}

			if (!LocalePoolSet.TryGetValue(moduleName, out var moduleLocales))
			{
				return null;
			}

			if (!moduleLocales.TryGetValue(CurrentLanguage, out var currentLanguageLocales))
			{
				return null;
			}

			// Navigate through the nested dictionary using dot notation
			object? current = currentLanguageLocales;
			var fragments = notation.Split('.');

			foreach (var fragment in fragments)
			{
				if (current is Dictionary<string, object> dict)
				{
					if (dict.TryGetValue(fragment, out var value))
					{
						current = value;

						// If we found a string value, return it
						if (current is string str)
						{
							return str;
						}
					}
					else
					{
						return null;
					}
				}
				else
				{
					// Can't navigate further in a non-dictionary
					return null;
				}
			}

			return null;
		}

		/// <summary>
		/// Format a template string with named placeholders.
		/// Replaces {name} style placeholders with corresponding values.
		/// </summary>
		/// <param name="template">Template string with {name} placeholders</param>
		/// <param name="replacements">Variable-length arguments of (name, value) tuples</param>
		/// <returns>Formatted string</returns>
		public static string Format(string template, params (string name, object value)[] replacements)
		{
			var result = template;
			foreach (var (name, value) in replacements)
			{
				result = result.Replace($"{{{name}}}", value?.ToString() ?? "");
			}

			return result;
		}

		/// <summary>
		/// Get a localized message and format it with named parameters.
		/// Convenience method combining Get() and Format().
		/// </summary>
		/// <param name="notation">Dot-separated key notation</param>
		/// <param name="replacements">Variable-length arguments of (name, value) tuples</param>
		/// <returns>Formatted message, or null if the key is not found</returns>
		public static string? Get(string notation, params (string name, object value)[] replacements)
		{
			var message = Get(notation, "facefusion");
			if (message == null)
			{
				return null;
			}

			return Format(message, replacements);
		}

		/// <summary>
		/// Get a localized message from a specific module and format it with named parameters.
		/// </summary>
		/// <param name="notation">Dot-separated key notation</param>
		/// <param name="moduleName">Module name</param>
		/// <param name="replacements">Variable-length arguments of (name, value) tuples</param>
		/// <returns>Formatted message, or null if the key is not found</returns>
		public static string? Get(string notation, string moduleName, params (string name, object value)[] replacements)
		{
			var message = Get(notation, moduleName);
			if (message == null)
			{
				return null;
			}

			return Format(message, replacements);
		}
	}
}
