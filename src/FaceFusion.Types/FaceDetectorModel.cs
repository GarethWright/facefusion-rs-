namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceDetectorModel = Literal['many', 'retinaface', 'scrfd', 'yolo_face', 'yunet']</c>.
/// </summary>
public enum FaceDetectorModel
{
	[WireName("many")]
	Many,

	[WireName("retinaface")]
	Retinaface,

	[WireName("scrfd")]
	Scrfd,

	[WireName("yolo_face")]
	YoloFace,

	[WireName("yunet")]
	Yunet
}
