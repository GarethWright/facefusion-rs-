using System.Collections.Generic;

namespace FaceFusion.Inference;

/// <summary>
/// C# shape for Python's <c>InferenceProvider</c> type alias
/// (<c>facefusion/types.py</c>: <c>InferenceProvider = Any</c>; see the "Cross-cutting"
/// note in <c>FaceFusion.Types/TypeAliases.cs</c>).
///
/// <c>onnxruntime.InferenceSession(providers = ...)</c> accepts a list whose elements are
/// each either a bare provider name string (no options) or a
/// <c>(provider_name, option_dict)</c> tuple. <see cref="Options"/> being <see langword="null"/>
/// stands in for the bare-string form; a non-null (possibly empty) dictionary stands in for
/// the tuple form. <c>Options</c> values are <see cref="object"/> to mirror Python's
/// <c>InferenceOptionSet = Dict[str, Any]</c> (values are a mix of <c>int</c>, <c>bool</c> and
/// <c>str</c> depending on the provider) — see <see cref="Execution.CreateInferenceProviders"/>
/// for the exact keys and value types per provider. Conversion to the string-only dictionary
/// ONNX Runtime's <c>SessionOptions.AppendExecutionProvider</c> actually accepts happens at the
/// point of use in <see cref="InferenceManager"/>, not here — this record stays a faithful,
/// directly-testable mirror of what <c>create_inference_providers</c> returns in Python.
/// </summary>
public sealed record InferenceProviderEntry(string ProviderName, IReadOnlyDictionary<string, object?>? Options);
