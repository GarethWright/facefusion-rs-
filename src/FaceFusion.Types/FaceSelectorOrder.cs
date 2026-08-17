namespace FaceFusion.Types;

/// <summary>
/// Ported from Python facefusion/types.py:
/// <c>FaceSelectorOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']</c>.
/// </summary>
public enum FaceSelectorOrder
{
	[WireName("left-right")]
	LeftRight,

	[WireName("right-left")]
	RightLeft,

	[WireName("top-bottom")]
	TopBottom,

	[WireName("bottom-top")]
	BottomTop,

	[WireName("small-large")]
	SmallLarge,

	[WireName("large-small")]
	LargeSmall,

	[WireName("best-worst")]
	BestWorst,

	[WireName("worst-best")]
	WorstBest
}
