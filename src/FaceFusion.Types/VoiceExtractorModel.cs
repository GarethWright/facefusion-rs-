namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VoiceExtractorModel = Literal['kim_vocal_1', 'kim_vocal_2', 'uvr_mdxnet']</c>.
/// </summary>
public enum VoiceExtractorModel
{
	[WireName("kim_vocal_1")]
	KimVocal1,

	[WireName("kim_vocal_2")]
	KimVocal2,

	[WireName("uvr_mdxnet")]
	UvrMdxnet
}
