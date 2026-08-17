using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceSet = TypedDict('FaceSet', { 'lock': Lock, 'faces': NotRequired[List[Face]] })</c>.
/// <c>Lock</c> stands in for Python's <c>threading.Lock</c> — FaceFusion.Types has no
/// synchronization-primitive dependency, so it is <c>object</c> here; FaceFusion.Core supplies
/// the concrete lock. <c>Faces</c> is nullable to mirror <c>NotRequired</c>.
/// </summary>
public sealed record FaceSet(object Lock, IReadOnlyList<Face>? Faces = null);
