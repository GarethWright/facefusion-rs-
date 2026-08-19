using System.Collections.Concurrent;

namespace FaceFusion.Inference;

/// <summary>
/// Port of facefusion/model_helper.py.
/// </summary>
public static class ModelHelper
{
    // Python:
    //
    //     @lru_cache()
    //     def get_static_model_initializer(model_path : str) -> ModelInitializer:
    //         model = onnx.load(model_path)
    //         return onnx.numpy_helper.to_array(model.graph.initializer[-1])
    //
    // This is a pure memoization cache keyed by model_path (not shared mutable state read
    // from elsewhere), so per port convention rule 5 it is fine to keep as a module-level
    // cache the way Python's decorator does — the same pattern FaceFusion.Media.Ffprobe
    // uses for its own @lru_cache'd functions. Divergences from functools.lru_cache,
    // both accepted and documented:
    //   - No eviction bound (lru_cache() with no maxsize is actually unbounded too, so
    //     this is exact parity, not a divergence).
    //   - ConcurrentDictionary.GetOrAdd may invoke the factory more than once under
    //     concurrent first-access for the same key (the loser's result is discarded); a
    //     true single-flight-per-key cache would need extra locking that Ffprobe's cache
    //     also does not bother with, since the factory is pure and side-effect-free.
    // Tests avoid cross-test leakage the same way FfprobeTests does: each test that needs
    // an isolated cache entry uses its own uniquely-named fixture/temp file path, so
    // cache hits never cross tests.
    private static readonly ConcurrentDictionary<string, OnnxTensor> StaticModelInitializerCache = new(StringComparer.Ordinal);

    /// <summary>
    /// Python: <c>get_static_model_initializer</c>. Loads <paramref name="modelPath"/> as
    /// an ONNX <c>ModelProto</c> and returns the *last* tensor in
    /// <c>graph.initializer</c> — FaceFusion stores a per-model embedding / conversion
    /// matrix there (see <c>facefusion/processors/modules/face_swapper/core.py</c>).
    ///
    /// <para>
    /// Decoding approach (see port report): a minimal hand-written protobuf wire-format
    /// reader (<see cref="OnnxProtoReader"/>), not a generated onnx.proto message set, per
    /// docs/DOTNET_PORT_PLAN.md section 2: "A minimal TensorProto decode is enough — do
    /// not pull in a full ONNX graph library." Real <c>.onnx</c> weight files can be
    /// hundreds of megabytes; the reader streams past every uninteresting field —
    /// including every initializer except the last — with <see cref="Stream.Seek"/>
    /// instead of materialising it.
    /// </para>
    /// </summary>
    public static OnnxTensor GetStaticModelInitializer(string modelPath)
    {
        return StaticModelInitializerCache.GetOrAdd(modelPath, static path =>
        {
            using var stream = File.OpenRead(path);
            return OnnxProtoReader.ReadLastInitializer(stream);
        });
    }
}
