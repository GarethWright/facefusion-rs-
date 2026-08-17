using System;

namespace FaceFusion.Types;

/// <summary>
/// Declares the exact wire string an enum member serializes to/from — the literal string
/// FaceFusion uses on the CLI, in facefusion.ini, and in job JSON. Ported enums that stood
/// in for Python `Literal['a', 'b', ...]` string unions carry one of these per member so the
/// wire string round-trips exactly, including characters (like hyphens) that are not legal in
/// a C# identifier. See <see cref="EnumNames"/> for the conversion helpers that read this
/// attribute.
/// </summary>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
public sealed class WireNameAttribute : Attribute
{
	public string Name { get; }

	public WireNameAttribute(string name)
	{
		Name = name;
	}
}
