namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceScoreSet = TypedDict('FaceScoreSet', { 'detector': Score, 'landmarker': Score })</c>.
/// <c>Score : TypeAlias = float</c>.
/// </summary>
public sealed record FaceScoreSet(double Detector, double Landmarker);
