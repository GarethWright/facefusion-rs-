namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/processors/core.py</c>'s <c>load_processor_module</c> /
/// <c>get_processors_modules</c> — resolves a processor by the string name that appears on the
/// CLI and in job step JSON (e.g. <c>"face_swapper"</c>), in place of Python's
/// <c>importlib.import_module('facefusion.processors.modules.' + processor + '.core')</c>
/// directory scan.
///
/// <para>
/// <b>Why a registry class exists at all, given <see cref="IProcessor"/> already replaces the
/// <c>hasattr</c> contract check.</b> <c>docs/DOTNET_PORT_PLAN.md</c> §3 shows processors
/// resolved from DI (<c>services.AddKeyedSingleton&lt;IProcessor&gt;("face_swapper", …)</c>).
/// This class is the non-DI equivalent — usable directly from unit tests, tools, and any caller
/// that is not running inside the eventual <c>FaceFusion.Cli</c>/<c>FaceFusion.Ui</c> DI
/// container — built by handing it every known <see cref="IProcessor"/> instance once (typically
/// the same set a DI container would enumerate via <c>IEnumerable&lt;IProcessor&gt;</c>).
/// </para>
///
/// <para>
/// <b>Divergence: throws instead of <c>hard_exit(1)</c>.</b> Python's
/// <c>load_processor_module</c> logs and calls <c>exit_helper.hard_exit(1)</c> — terminates the
/// whole process — for both an unresolvable module name and a module missing a required method.
/// A class library must not call <c>Environment.Exit</c> (this is the same "the CLI decides what
/// a fatal error means" divergence <c>InferenceManager.CreateInferenceSession</c> already
/// documents for its own <c>hard_exit(1)</c> call): <see cref="Resolve"/> throws
/// <see cref="ProcessorNotFoundException"/> instead, leaving the decision of whether that is
/// fatal to the caller (a future CLI layer can catch it and exit(1) itself). The "missing a
/// required method" half of Python's check has no C# equivalent to fail at this point at all —
/// implementing <see cref="IProcessor"/> is enforced by the compiler, so there is no way to
/// register a processor object that is missing a member in the first place.
/// </para>
/// </summary>
public sealed class ProcessorRegistry
{
    private readonly Dictionary<string, IProcessor> _processorsByName;

    /// <summary>
    /// Builds a registry from every known <see cref="IProcessor"/> implementation. Duplicate
    /// <see cref="IProcessor.Name"/> values are not expected (Python's directory scan cannot
    /// produce two modules with the same directory name either); the last entry for a given
    /// name wins, matching <see cref="Dictionary{TKey,TValue}"/>'s own indexer-assignment
    /// semantics rather than throwing, since a caller re-registering the same processor (e.g. a
    /// DI container handing back a decorated wrapper) is a reasonable thing to want to support.
    /// </summary>
    public ProcessorRegistry(IEnumerable<IProcessor> processors)
    {
        ArgumentNullException.ThrowIfNull(processors);

        _processorsByName = new Dictionary<string, IProcessor>(StringComparer.Ordinal);
        foreach (var processor in processors)
        {
            _processorsByName[processor.Name] = processor;
        }
    }

    /// <summary>
    /// Python: <c>load_processor_module(processor)</c>. Resolves one processor by name.
    /// </summary>
    /// <exception cref="ProcessorNotFoundException">
    /// No registered processor has this name — Python: the module-not-found branch of
    /// <c>load_processor_module</c> (<c>ModuleNotFoundError</c> from the <c>importlib</c> call).
    /// </exception>
    public IProcessor Resolve(string processorName)
    {
        ArgumentNullException.ThrowIfNull(processorName);

        if (_processorsByName.TryGetValue(processorName, out var processor))
        {
            return processor;
        }

        throw new ProcessorNotFoundException(processorName);
    }

    /// <summary>
    /// Python: <c>get_processors_modules(processors)</c>. Resolves a list of processor names in
    /// order, e.g. the <c>--processors</c> CLI option's value.
    /// </summary>
    public IReadOnlyList<IProcessor> ResolveAll(IReadOnlyList<string> processorNames)
    {
        ArgumentNullException.ThrowIfNull(processorNames);

        var processors = new List<IProcessor>(processorNames.Count);
        foreach (var processorName in processorNames)
        {
            processors.Add(Resolve(processorName));
        }

        return processors;
    }

    /// <summary>Every registered processor name, in no particular order.</summary>
    public IReadOnlyCollection<string> Names => _processorsByName.Keys;
}

/// <summary>
/// Thrown by <see cref="ProcessorRegistry.Resolve"/> for a processor name with no registered
/// <see cref="IProcessor"/>. See <see cref="ProcessorRegistry"/>'s remarks for why this is a
/// thrown exception rather than Python's <c>hard_exit(1)</c>.
/// </summary>
public sealed class ProcessorNotFoundException : Exception
{
    public ProcessorNotFoundException(string processorName)
        : base($"processor '{processorName}' is not registered (Python: processor_not_loaded / ModuleNotFoundError for facefusion.processors.modules.{processorName}.core)")
    {
        ProcessorName = processorName;
    }

    public string ProcessorName { get; }
}
