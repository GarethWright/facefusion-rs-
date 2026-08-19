namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>VideoEncoder = Literal['libx264', 'libx264rgb', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'h264_qsv', 'hevc_qsv', 'h264_videotoolbox', 'hevc_videotoolbox', 'rawvideo']</c>.
/// </summary>
public enum VideoEncoder
{
	[WireName("libx264")]
	Libx264,

	[WireName("libx264rgb")]
	Libx264rgb,

	[WireName("libx265")]
	Libx265,

	[WireName("libvpx-vp9")]
	LibvpxVp9,

	[WireName("h264_nvenc")]
	H264Nvenc,

	[WireName("hevc_nvenc")]
	HevcNvenc,

	[WireName("h264_amf")]
	H264Amf,

	[WireName("hevc_amf")]
	HevcAmf,

	[WireName("h264_qsv")]
	H264Qsv,

	[WireName("hevc_qsv")]
	HevcQsv,

	[WireName("h264_videotoolbox")]
	H264Videotoolbox,

	[WireName("hevc_videotoolbox")]
	HevcVideotoolbox,

	[WireName("rawvideo")]
	Rawvideo
}
