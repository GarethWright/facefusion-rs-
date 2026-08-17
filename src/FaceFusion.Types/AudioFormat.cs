namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>AudioFormat = Literal['flac', 'm4a', 'mp3', 'ogg', 'opus', 'wav']</c>.
/// </summary>
public enum AudioFormat
{
	[WireName("flac")]
	Flac,

	[WireName("m4a")]
	M4a,

	[WireName("mp3")]
	Mp3,

	[WireName("ogg")]
	Ogg,

	[WireName("opus")]
	Opus,

	[WireName("wav")]
	Wav
}
