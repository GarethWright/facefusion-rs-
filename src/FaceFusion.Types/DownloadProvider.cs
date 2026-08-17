namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>DownloadProvider = Literal['github', 'huggingface']</c>.
/// </summary>
public enum DownloadProvider
{
	[WireName("github")]
	Github,

	[WireName("huggingface")]
	Huggingface
}
