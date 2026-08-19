using System.Collections.Generic;

namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>EncoderSet = TypedDict('EncoderSet', { 'audio': List[AudioEncoder], 'video': List[VideoEncoder] })</c>.
/// </summary>
public sealed record EncoderSet(IReadOnlyList<AudioEncoder> Audio, IReadOnlyList<VideoEncoder> Video);
