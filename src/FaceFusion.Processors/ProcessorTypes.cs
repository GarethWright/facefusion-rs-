using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/processors/types.py</c> — the type aliases shared by every processor
/// module.
///
/// <para>
/// <b><c>ProcessorOutputs</c>.</b> Python: <c>ProcessorOutputs : TypeAlias = Tuple[VisionFrame,
/// Mask]</c>. Represented as a real record rather than a bare <c>(Mat, Mat)</c> tuple so call
/// sites read <c>VisionFrame</c>/<c>Mask</c> instead of <c>Item1</c>/<c>Item2</c>. Follows the
/// <c>FaceFusion.Face.FaceHelper</c>/<c>FaceMasker</c> ownership convention: pixel data is a
/// <see cref="Mat"/>, native memory, and the caller owns (and must dispose) both fields of a
/// returned <see cref="ProcessorOutputs"/>.
/// </para>
///
/// <para>
/// <b><c>ProcessorState</c> / <c>ProcessorStateKey</c> / <c>ProcessorStateSet</c> — not
/// reproduced.</b> These three Python aliases (<c>Dict[str, Any]</c> and
/// <c>Dict[AppContext, ProcessorState]</c>) exist only to describe the shape of the module-level
/// slice of <c>state_manager</c> each processor reads. Per docs/DOTNET_PORT_PLAN.md §3 and
/// PORT_CONVENTIONS.md rule 5 ("no global mutable state — take settings as parameters"), this
/// port has no <c>state_manager</c> equivalent: every processor takes its settings as explicit
/// parameters (see <see cref="IProcessor"/>'s remarks and each processor's own
/// <c>*Inputs</c>/<c>*Settings</c> type), so there is nothing for these three aliases to name in
/// C#. Documented rather than silently dropped, matching how
/// <c>FaceFusion.Types.TypeAliases.cs</c> already treats aliases with no field use.
/// </para>
///
/// <para>
/// <b><c>LivePortraitPitch</c>/<c>Yaw</c>/<c>Roll</c>/<c>Expression</c>/<c>FeatureVolume</c>/
/// <c>MotionPoints</c>/<c>Rotation</c>/<c>Scale</c>/<c>Translation</c> — not reproduced here.</b>
/// These nine aliases are used only by <c>facefusion/processors/modules/live_portrait</c>, which
/// is not in this assignment (see the processor table in the assignment brief — it lands in a
/// later phase, in this same <c>FaceFusion.Processors</c> project). Adding unused type aliases
/// here would be speculative; the agent porting <c>live_portrait</c> should define them in that
/// module's own file, the same way <see cref="FaceSwapper"/> defines
/// <c>FaceSwapperModel</c>/<c>FaceSwapperWeight</c> in its own file rather than here — each
/// processor module's private types stay local to that module's file, and only the truly
/// shared/generic pieces (<see cref="ProcessorOutputs"/>, <see cref="IProcessor"/>,
/// <see cref="ProcessorRegistry"/>) live in this project's root.
/// </para>
/// </summary>
public sealed record ProcessorOutputs(Mat VisionFrame, Mat Mask);
