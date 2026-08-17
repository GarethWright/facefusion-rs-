namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py <c>BenchmarkCycleSet</c> TypedDict.
/// </summary>
public sealed record BenchmarkCycleSet(
	string TargetPath,
	int CycleCount,
	double AverageRun,
	double FastestRun,
	double SlowestRun,
	double RelativeFps);
