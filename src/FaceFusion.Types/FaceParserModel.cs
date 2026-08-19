namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceParserModel = Literal['bisenet_resnet_18', 'bisenet_resnet_34']</c>.
/// </summary>
public enum FaceParserModel
{
	[WireName("bisenet_resnet_18")]
	BisenetResnet18,

	[WireName("bisenet_resnet_34")]
	BisenetResnet34
}
