using FaceFusion.Types;

namespace FaceFusion.Processors;

/// <summary>
/// Marker for one processor's per-call inputs. Python: each processor module defines its own
/// <c>TypedDict</c> (e.g. <c>face_swapper/types.py</c>'s <c>FaceSwapperInputs</c> — <c>{
/// reference_vision_frame, source_vision_frames, target_vision_frames, temp_vision_frame,
/// temp_vision_mask }</c>) and <c>process_frame</c> only ever receives its own shape; there is
/// no shared inputs type in Python either. <see cref="IProcessor.ProcessFrame"/> is typed to
/// accept this marker (rather than one shared, lowest-common-denominator inputs record) so each
/// processor keeps its own concrete, strongly-typed inputs type — see
/// <c>FaceSwapper.FaceSwapperInputs</c> for the concrete example this assignment ports.
///
/// <para>
/// Per PORT_CONVENTIONS.md rule 5 ("no global mutable state — take settings as parameters"), a
/// concrete <c>*Inputs</c> record is also where every value the Python module would have pulled
/// out of <c>state_manager</c> for that call lives (model choice, weights, mask settings, the
/// resolved <c>InferenceSession</c>s, ...) — not just the per-frame vision data the Python
/// <c>TypedDict</c> itself lists. That is a deliberate widening of the Python shape, not a
/// divergence in behaviour: Python's <c>process_frame(inputs)</c> silently closes over
/// module-global state for all of that; the C# port makes the same information an explicit,
/// visible parameter instead.
/// </para>
/// </summary>
public interface IProcessorInputs
{
}

/// <summary>
/// Port of <c>facefusion/processors/core.py</c>'s <c>PROCESSORS_METHODS</c> contract — the set
/// of methods Python's <c>load_processor_module</c> demands (by <c>hasattr</c> duck-typing,
/// after an <c>importlib.import_module</c> directory scan) from every module under
/// <c>facefusion/processors/modules/&lt;name&gt;/core.py</c>. This interface is the compile-time
/// replacement for that runtime scan-and-duck-type check: implementing <see cref="IProcessor"/>
/// *is* satisfying <c>PROCESSORS_METHODS</c>, enforced by the C# compiler instead of a
/// <c>hasattr</c> loop that only fires at CLI startup. <see cref="ProcessorRegistry"/> is the
/// replacement for <c>load_processor_module</c>/<c>get_processors_modules</c> themselves (name
/// -&gt; instance resolution).
///
/// <para>
/// <b>Members intentionally not carried over from <c>PROCESSORS_METHODS</c>, and why:</b>
/// <list type="bullet">
/// <item><description><c>get_inference_pool</c> / <c>clear_inference_pool</c>. In Python these
/// resolve model URLs via <c>facefusion.download.resolve_download_url</c> and pool
/// <c>InferenceSession</c>s keyed by <c>state_manager.get_item('&lt;processor&gt;_model')</c>.
/// <c>facefusion/download.py</c> is not ported anywhere in this repo yet (out of scope for
/// every module that has hit this so far — see <c>FaceRecognizer</c>'s, <c>FaceMasker</c>'s and
/// <c>FaceCreator</c>'s class remarks for the same call). Per rule 5, the sessions a processor
/// needs are passed in in already-created, e.g. as fields on its own <c>IProcessorInputs</c>
/// implementation, the same way <c>FaceMasker.CreateOcclusionMask</c> takes an
/// <c>occluderInferencePool</c> parameter instead of resolving one itself. A later phase that
/// ports <c>download.py</c> can add pool-construction helpers next to each processor's model
/// catalog (see <c>FaceSwapper.FaceSwapperModelCatalog</c>) without touching this interface.
/// </description></item>
/// <item><description><c>register_args</c> / <c>apply_args</c>. These wire a Python
/// <c>argparse.ArgumentParser</c> group and copy parsed CLI args into <c>state_manager</c>. The
/// C# equivalent is a Phase-6 concern (<c>FaceFusion.Cli</c>, not built yet) — once it exists it
/// populates a processor's <c>*Settings</c>/<c>*Inputs</c> record directly (there is no
/// <c>state_manager</c> to "apply" into), so there is nothing for this interface to declare
/// today. Each processor's model-name/weight/etc. choice lists (the parts of
/// <c>register_args</c> a CLI layer actually needs — <c>choices=...</c>) are exposed as public
/// static data instead, e.g. <c>FaceSwapper.FaceSwapperModelCatalog.Keys</c> /
/// <c>FaceSwapper.FaceSwapperWeightRange</c>, matching how <c>FaceFusion.Types.Choices</c>
/// already exposes every other module's choice lists.</description></item>
/// </list>
/// Both omissions are the same shape of divergence PORT_CONVENTIONS.md rule 5 already commits
/// this whole port to, applied consistently at the processor layer rather than retrofitted.
/// </para>
/// </summary>
public interface IProcessor
{
    /// <summary>
    /// Python: the processor's module directory name under
    /// <c>facefusion/processors/modules/</c> (e.g. <c>"face_swapper"</c>) — the exact string
    /// that appears on the CLI (<c>--processors face_swapper</c>), in job step JSON, and as the
    /// key <c>importlib.import_module('facefusion.processors.modules.' + processor + '.core')</c>
    /// resolves against. <see cref="ProcessorRegistry"/> resolves by this same string in place
    /// of that import.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Python: <c>get_common_modules()</c> — the shared face-pipeline stages
    /// (<c>content_analyser</c>, <c>face_detector</c>, ...) this processor's <c>pre_check</c> /
    /// <c>post_process</c> delegate to. Represented as the modules' string names (matching the
    /// module names used throughout this repo's own class remarks — e.g. <c>"face_detector"</c>)
    /// rather than Python's <c>ModuleType</c>, since C# has no runtime module handle to hand
    /// back; a caller resolves each name's actual <c>PreCheck</c>/inference-pool-clear call
    /// itself (there is no shared common-module interface in this repo — each of
    /// <c>FaceDetector</c>/<c>FaceLandmarker</c>/.../<c>FaceMasker</c> is an unrelated static
    /// class with its own signature).
    /// </summary>
    IReadOnlyList<string> GetCommonModules();

    /// <summary>
    /// Python: <c>pre_check() -&gt; bool</c>. Verifies every model file this processor needs is
    /// present. Python's version also *downloads* missing files via
    /// <c>conditional_download_hashes</c>/<c>conditional_download_sources</c>; since
    /// <c>download.py</c> is not ported (see the interface remarks), this checks local presence
    /// only (<c>FileSystem.IsFile</c>-equivalent) and returns <see langword="false"/>, same as
    /// Python, when a required file is missing — it does not fetch it.
    /// </summary>
    bool PreCheck();

    /// <summary>
    /// Python: <c>pre_process(mode) -&gt; bool</c>. Validates the run's paths for the given
    /// <paramref name="mode"/> before processing starts (source present and an image, target/
    /// output present and of a matching kind, ...). <paramref name="paths"/> stands in for the
    /// several <c>state_manager.get_item('source_paths' | 'target_path' | 'output_path')</c>
    /// calls every processor's <c>pre_process</c> makes (rule 5).
    /// </summary>
    bool PreProcess(ProcessMode mode, ProcessorRunPaths paths);

    /// <summary>
    /// Python: <c>process_frame(inputs) -&gt; ProcessorOutputs</c>. <paramref name="inputs"/> is
    /// this processor's own concrete <see cref="IProcessorInputs"/> implementation (Python: its
    /// own <c>TypedDict</c>) — see that interface's remarks. Implementations should validate the
    /// runtime type and throw <see cref="ArgumentException"/> for the wrong shape (Python would
    /// raise <c>KeyError</c>/<c>AttributeError</c> reaching into the wrong dict; this is the
    /// statically-typed analogue of the same "wrong shape" failure, not a new behaviour).
    /// </summary>
    ProcessorOutputs ProcessFrame(IProcessorInputs inputs);

    /// <summary>
    /// Python: <c>post_process() -&gt; None</c>. Releases caches (inference pool, static-image
    /// cache, ...) per the configured <see cref="VideoMemoryStrategy"/>. Takes no parameters in
    /// Python either beyond <c>state_manager</c> reads; a concrete implementation takes whatever
    /// it needs (the strategy, the pools/caches to clear) as constructor or method parameters
    /// per rule 5 rather than this interface widening to describe every processor's cache set.
    /// </summary>
    void PostProcess();
}

/// <summary>
/// The subset of <c>state_manager</c> keys every processor's Python <c>pre_process(mode)</c>
/// reads (<c>source_paths</c>, <c>target_path</c>, <c>output_path</c>) — see
/// <see cref="IProcessor.PreProcess"/>'s remarks. <see cref="SourcePaths"/> is empty for a
/// processor that has no source concept (Python: not every processor's <c>pre_process</c> calls
/// <c>has_image(state_manager.get_item('source_paths'))</c> — <c>face_swapper</c> and a few
/// others do, most frame processors do not); it is still on this shared record rather than
/// split into a per-processor variant so <see cref="IProcessor.PreProcess"/> stays one signature
/// for all eleven processors.
/// </summary>
public sealed record ProcessorRunPaths(
    IReadOnlyList<string> SourcePaths,
    string? TargetPath,
    string? OutputPath);
