using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>JobStep = TypedDict('JobStep', { 'args': Args, 'status': JobStepStatus })</c>.
/// <c>Args</c> is <c>Dict[str, Any]</c> — see the alias notes in TypeAliases.cs.
/// </summary>
public sealed record JobStep(IReadOnlyDictionary<string, object?> Args, JobStepStatus Status);
