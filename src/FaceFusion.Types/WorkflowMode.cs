namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>WorkflowMode = Literal['auto', 'image-to-image', 'image-to-video']</c>.
/// </summary>
public enum WorkflowMode
{
	[WireName("auto")]
	Auto,

	[WireName("image-to-image")]
	ImageToImage,

	[WireName("image-to-video")]
	ImageToVideo
}
