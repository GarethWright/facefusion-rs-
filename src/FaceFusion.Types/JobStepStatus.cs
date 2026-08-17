namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>JobStepStatus = Literal['drafted', 'queued', 'started', 'completed', 'failed']</c>.
/// </summary>
public enum JobStepStatus
{
	[WireName("drafted")]
	Drafted,

	[WireName("queued")]
	Queued,

	[WireName("started")]
	Started,

	[WireName("completed")]
	Completed,

	[WireName("failed")]
	Failed
}
