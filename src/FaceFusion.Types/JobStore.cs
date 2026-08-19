using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>JobStore = TypedDict('JobStore', { 'job_keys': List[str], 'step_keys': List[str] })</c>.
/// </summary>
public sealed record JobStore(IReadOnlyList<string> JobKeys, IReadOnlyList<string> StepKeys);
