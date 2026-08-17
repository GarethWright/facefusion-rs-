using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ProcessStep : TypeAlias = Callable[[str, int, Args], bool]</c>.
/// <c>Args : TypeAlias = Dict[str, Any]</c>.
/// </summary>
public delegate bool ProcessStep(string stepName, int stepIndex, IReadOnlyDictionary<string, object?> args);
