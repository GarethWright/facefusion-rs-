using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>CameraPoolSet = TypedDict('CameraPoolSet', { 'capture': CameraCaptureSet })</c>, where
/// <c>CameraCaptureSet : TypeAlias = Dict[str, cv2.VideoCapture]</c>. The dictionary value is
/// <c>object</c> — FaceFusion.Types has no OpenCV dependency (see PORT_CONVENTIONS.md) — until
/// FaceFusion.Media supplies the concrete capture handle type.
/// </summary>
public sealed record CameraPoolSet(IReadOnlyDictionary<string, object> Capture);
