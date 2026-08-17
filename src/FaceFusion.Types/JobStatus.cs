namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>JobStatus = Literal['drafted', 'queued', 'completed', 'failed']</c>.
/// </summary>
public enum JobStatus
{
	[WireName("drafted")]
	Drafted,

	[WireName("queued")]
	Queued,

	[WireName("completed")]
	Completed,

	[WireName("failed")]
	Failed
}
