namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py: <c>AppContext = Literal['cli', 'ui']</c>.
/// Named identically to the Python type for parity; note this shadows
/// <see cref="System.AppContext"/> by name within code that has both namespaces open —
/// qualify as <c>FaceFusion.Types.AppContext</c> or <c>System.AppContext</c> if ambiguous.
/// </summary>
public enum AppContext
{
	[WireName("cli")]
	Cli,

	[WireName("ui")]
	Ui
}
