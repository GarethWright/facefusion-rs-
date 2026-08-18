namespace FaceFusion.ParityTests;

/// <summary>
/// Serialises every test class that creates ONNX Runtime sessions.
///
/// ORT's native layer is not robust under xunit's default parallel collections here: a
/// full-suite run aborted with "Test host process crashed" immediately after a test that
/// deliberately loads a malformed model, while that same test passes on its own. The
/// bindings also segfault rather than throw on use-after-dispose, so a native fault in one
/// collection takes down every other collection's tests with it — turning an isolated
/// failure into a whole-run abort that reports nothing useful.
///
/// These are also the slowest tests, so serialising them reduces CPU thrash on top of
/// making failures legible.
/// </summary>
[CollectionDefinition("NativeInference", DisableParallelization = true)]
public sealed class NativeInferenceCollection
{
}
