using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>Job</c> TypedDict.
/// </summary>
public sealed record Job(string Version, string DateCreated, string? DateUpdated, IReadOnlyList<JobStep> Steps);
