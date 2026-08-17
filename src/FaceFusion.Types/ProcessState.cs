namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>ProcessState = Literal['checking', 'processing', 'stopping', 'pending']</c>.
/// </summary>
public enum ProcessState
{
	[WireName("checking")]
	Checking,

	[WireName("processing")]
	Processing,

	[WireName("stopping")]
	Stopping,

	[WireName("pending")]
	Pending
}
