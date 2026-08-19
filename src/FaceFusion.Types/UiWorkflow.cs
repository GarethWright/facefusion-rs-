namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>UiWorkflow = Literal['instant_runner', 'job_runner', 'job_manager']</c>.
/// </summary>
public enum UiWorkflow
{
	[WireName("instant_runner")]
	InstantRunner,

	[WireName("job_runner")]
	JobRunner,

	[WireName("job_manager")]
	JobManager
}
