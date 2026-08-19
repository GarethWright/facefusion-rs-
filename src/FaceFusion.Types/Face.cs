namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: the <c>Face</c> namedtuple. Several fields are
/// <c>NDArray[Any]</c> in Python (<c>bounding_box</c>, <c>embedding</c>, <c>embedding_norm</c>)
/// and are represented as <c>object</c> here — FaceFusion.Types has no tensor dependency; see
/// PORT_CONVENTIONS.md and the FaceFusion.Tensors project. <c>origin</c> is an untyped string
/// literal in Python (observed values include 'detect' and 'refill'), so it stays <c>string</c>
/// rather than becoming an enum. <c>Age : TypeAlias = range</c> maps to <see cref="System.Range"/>,
/// the closest BCL equivalent of a Python half-open integer range.
/// </summary>
public sealed record Face(
	string Origin,
	object BoundingBox,
	FaceScoreSet ScoreSet,
	FaceLandmarkSet LandmarkSet,
	int Angle,
	object Embedding,
	object EmbeddingNorm,
	System.Range Age,
	Gender Gender,
	Race Race);
