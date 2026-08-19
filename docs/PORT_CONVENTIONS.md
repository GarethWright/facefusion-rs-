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
| `src/FaceFusion.Ui` | Blazor Server UI (`facefusion/uis`) — Razor components, `UiState` |
| `tests/FaceFusion.UnitTests` | xUnit tests for all of the above |
| `tests/ui` | Browser-driven UI tests (Playwright); see its README |

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
   take the value as a parameter instead and note it in your report. The one exception is
   `FaceFusion.Ui`, where the user's control values genuinely are shared mutable state and
   `UiState` holds them — and even there the store stops at the UI boundary: it materialises a
   plain args bag for `HeadlessRunner`, so nothing below `FaceFusion.Ui` can see it.
6. **Nullability**: Python `Optional[X]` → `X?`. Do not use `!` to silence the compiler
   without a comment justifying it.
7. **Culture**: use `CultureInfo.InvariantCulture` for every number→string conversion.
   Command lines and file formats must not vary by locale. `InvariantGlobalization` is
   on, but be explicit anyway.
8. **Run it, do not just build it.** A processor, a UI panel or a workflow is finished when
   both implementations have been run on the same input and their outputs compared — not when
   it compiles and the suite is green. Every defect in `docs/IMPLEMENTATION_STATUS.md` was found
   that way and none was caught by the test suite: `face_swapper`'s assembled pipeline had never
   once executed while its parity tests were green, the CLI could not run any processor that
   reads source audio, and the UI showed stale markup on every conditional panel. Expect PSNR in
   the low 40s dB for a video comparison — that is two independent libx264 encodes of identical
   pixels — and byte-identical output when no encoder is involved.
9. **Diagnose before you adjust.** When a comparison disagrees, find out why before touching a
   tolerance or an expected value. A 33 dB "parity failure" in `face_swapper` turned out to be
   two different models being compared, because the port had guessed a default instead of
   reading `register_args`. The OOM defect was recorded as a large-object-heap problem on the
   strength of the plan predicting one; an instrumented run showed the managed heap never
   exceeded 260 MB and every extra byte was native. Four plausible fixes for it were then
   implemented and measured, and all four were wrong.

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
- **"Passed!" does not mean every test ran.** A native fault, or the OOM killer taking the test
  host, ends a collection early and the summary still reports `Passed!` with zero failures —
  once showing 886 of 924. When a run matters, compare its total against
  `dotnet test --list-tests | grep -c "^    "`. And never run a memory-heavy job (an
  `age_modifier` video, say) alongside the suite: that is exactly what killed that host.
- **A green local suite is not a green CI suite.** CI has neither the example media
  (`/tmp/facefusion-test-examples`, fetched by `tools/parity/fetch_examples.sh`) nor the
  gitignored models under `.assets/models`, so any test that reaches for them must skip, per
  rule 2 — never fail. `VisionParityTests` hardcoded those paths with no gate, passed here for
  weeks, and failed eleven cases on all three runners the first time CI got far enough to run
  tests at all. Before trusting a suite, run `tools/parity/ci_simulate.sh`, which moves both
  aside and restores them on exit.
- **You can emulate a Windows checkout on Linux.** Git checks files out with CRLF on Windows,
  which silently breaks anything that hashes or byte-compares source. Rewrite the file in place
  (`data.replace(b"\n", b"\r\n")`), run the affected tests, then restore — that is how the
  content-analyser gate defect and a follow-up bug in its own regression test were both confirmed
  fixed without waiting on CI. Same idea as `tools/parity/ci_simulate.sh`: reproduce the
  environment rather than guess at it.
- **The OpenCvSharp analyzer only runs in CI.** It targets Roslyn 4.14 and this container's SDK
  is 4.8, so the compiler skips it with a `CS9057` warning and its diagnostics never appear
  locally. With `TreatWarningsAsErrors` on they are build *errors* in CI — `OCVS002` ("`Rows` is
  a P/Invoke call on every iteration") failed the .NET build on all three OSes while the local
  build was clean. Cache `Mat.Rows`/`Mat.Cols` outside any loop as a matter of course, and treat
  a green local build as no evidence about this analyzer.
- **A Blazor component only re-renders itself.** An event handler re-renders the component that
  owns it, so a control that writes shared state does not re-render the sibling panel whose
  `@if` reads it. Every component that reads `UiState` derives from `UiComponentBase`, which
  subscribes to `UiState.Changed` — see its class remarks. Getting this wrong produces a UI that
  compiles, serves a correct page and runs jobs correctly while showing stale markup, which no
  C# test catches. `tests/ui` exists for exactly that.

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
