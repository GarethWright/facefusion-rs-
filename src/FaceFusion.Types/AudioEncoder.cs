namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>AudioEncoder = Literal['flac', 'aac', 'libmp3lame', 'libopus', 'libvorbis', 'pcm_s16le', 'pcm_s32le']</c>.
/// </summary>
public enum AudioEncoder
{
	[WireName("flac")]
	Flac,

	[WireName("aac")]
	Aac,

	[WireName("libmp3lame")]
	Libmp3lame,

	[WireName("libopus")]
	Libopus,

	[WireName("libvorbis")]
	Libvorbis,

	[WireName("pcm_s16le")]
	PcmS16le,

	[WireName("pcm_s32le")]
	PcmS32le
}
