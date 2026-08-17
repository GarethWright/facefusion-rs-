namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>LogLevel = Literal['error', 'warn', 'info', 'debug']</c>.
/// </summary>
public enum LogLevel
{
	[WireName("error")]
	Error,

	[WireName("warn")]
	Warn,

	[WireName("info")]
	Info,

	[WireName("debug")]
	Debug
}
