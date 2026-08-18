# Port conventions

Rules for anyone (human or agent) porting a Python module to C# in this repo. Read this
before writing code. See `docs/DOTNET_PORT_PLAN.md` for the overall plan.

## Toolchain

- .NET SDK 8 is installed at `/usr/lib/dotnet` — add it to `PATH`.
  `export PATH="$PATH:/usr/lib/dotnet"`
- The plan targets .NET 10; this container only has 8 available, and nothing in the
  foundation phases needs 9/10 features. `TargetFramework` is set once in
  `Directory.Build.props`.
- Build: `dotnet build` from the repo root. Test: `dotnet test`.
- `TreatWarningsAsErrors` is on. Nullable reference types are on. Code must build clean.

## Project layout

Projects already exist. **Only add `.cs` files** — do not edit `.csproj` files or the
`.sln`, and do not add NuGet packages without asking. SDK-style projects glob `**/*.cs`
automatically, so a new file is picked up with no project edit.

| Project | Holds |
| --- | --- |
| `src/FaceFusion.Types` | Enums, records, type aliases from `types.py` / `choices.py` |
| `src/FaceFusion.Tensors` | The bounded numpy-compat layer (plan §4) |
| `src/FaceFusion.Core` | Helpers, config, settings, logging, locales |
| `src/FaceFusion.Media` | ffmpeg / ffprobe / curl command builders and runners |
| `tests/FaceFusion.UnitTests` | xUnit tests for all of the above |

Namespaces match project names (`FaceFusion.Types`, `FaceFusion.Media`, …). One public
type per file; file name matches the type name.

## Porting rules

1. **Behaviour parity beats idiomatic C#.** When the Python does something odd, reproduce
   the oddity and add a comment saying it is deliberate. Do not "fix" behaviour during a
   port — divergence must be a separate, deliberate decision.
2. **Port the tests too.** Every `tests/test_<module>.py` that covers your module becomes
   `tests/FaceFusion.UnitTests/<Module>Tests.cs`. Keep the same cases and the same
   expected values. If a Python test is environment-dependent (needs ffmpeg, network, or
   model files), port it but mark it `[Fact(Skip = "requires <x>")]` — do not silently
   drop coverage.
3. **Python `List[str]` command lists → `IReadOnlyList<string>`** returned, `string[]`
   internally. Command builders are pure functions: no I/O, no global state.
4. **Static classes for Python modules.** A Python module of free functions becomes a
   `public static class` with the same method names in `PascalCase`. Keep the Python name
   in a `<summary>` doc comment when the C# name diverges, e.g. `set_video_fps` →
   `SetVideoFps`.
5. **No global mutable state** (plan §3). If your module reads `state_manager` in Python,
   take the value as a parameter instead and note it in your report.
6. **Nullability**: Python `Optional[X]` → `X?`. Do not use `!` to silence the compiler
   without a comment justifying it.
7. **Culture**: use `CultureInfo.InvariantCulture` for every number→string conversion.
   Command lines and file formats must not vary by locale. `InvariantGlobalization` is
   on, but be explicit anyway.

## Gotchas that have bitten several agents

- **`Assert.Equal(float, float, int)` does not compile.** xunit cannot choose between
  `Assert.Equal(double, double, int)` and `Assert.Equal(float, float, float)`, so a
  precision-based comparison of two floats is ambiguous. Cast both operands to `double`:
  `Assert.Equal((double)expected, (double)actual, precision: 4)`. Keep the `f` suffix on a
  float32 ground-truth literal before the cast — `(double)29.590002f` and
  `(double)29.590002` are different values.
- **Test classes that create ONNX Runtime sessions must be `[Collection("NativeInference")]`.**
  ORT's bindings segfault rather than throw on use-after-dispose, and a native fault in one
  xunit collection takes the whole test host down — reporting a partial run as passing.
- **Test classes that write to the shared output directory must be `[Collection("MediaOutput")]`,**
  or they delete each other's files mid-run and fail only when run together.
- **Do not run the full test suite** while other agents are working. It takes several
  minutes on 4 cores; concurrent full runs once drove load average to 71 and stalled
  everything. Use `--filter` scoped to what you are changing.

## What not to do

- Do not add NuGet packages without asking. The dependency set is settled: OpenCvSharp4
  (+ its native runtime) for imaging, Microsoft.ML.OnnxRuntime for inference,
  Google.Protobuf for reading .onnx initializers, xunit for tests. `FaceFusion.Types`,
  `.Core`, `.Tensors`, `.Jobs` and `.Parity` are deliberately pure BCL — keep them that
  way, and note that `FaceFusion.Types` cannot reference `FaceFusion.Core` (the dependency
  runs the other way).
- Do not port a module that is not in your assignment, even if it looks easy — parallel
  agents will collide. Report what you needed and stop.
- Do not create stub/placeholder implementations that throw `NotImplementedException`
  for work in your own scope. Either port it properly or report it as blocked.
