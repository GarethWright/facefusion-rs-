namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>WarpTemplate = Literal['arcface_112_v1', 'arcface_112_v2', 'arcface_128', 'dfl_whole_face', 'ffhq_512', 'mtcnn_512', 'styleganex_384']</c>.
/// </summary>
public enum WarpTemplate
{
	[WireName("arcface_112_v1")]
	Arcface112V1,

	[WireName("arcface_112_v2")]
	Arcface112V2,

	[WireName("arcface_128")]
	Arcface128,

	[WireName("dfl_whole_face")]
	DflWholeFace,

	[WireName("ffhq_512")]
	Ffhq512,

	[WireName("mtcnn_512")]
	Mtcnn512,

	[WireName("styleganex_384")]
	Styleganex384
}
