namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>WorkflowStrategy = Literal['disk', 'memory']</c>.
/// </summary>
public enum WorkflowStrategy
{
	[WireName("disk")]
	Disk,

	[WireName("memory")]
	Memory
}
