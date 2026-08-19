namespace FaceFusion.Types;

/// <summary>
/// Marks an enum member that deliberately has no Python wire-string counterpart, so
/// <see cref="EnumNames"/> excludes it from wire mapping instead of demanding a
/// <see cref="WireNameAttribute"/>.
///
/// The case this exists for is the zero member of a <c>[Flags]</c> enum — Python models
/// such sets as a list of literals, so "no flags set" has no string spelling. Inventing a
/// wire name for it would put a value on the CLI and in job JSON that the Python side
/// cannot produce or parse.
/// </summary>
[AttributeUsage(AttributeTargets.Field)]
public sealed class NonWireValueAttribute : Attribute
{
}
