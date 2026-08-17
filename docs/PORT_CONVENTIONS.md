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

## What not to do

- Do not add dependencies on OpenCV, ONNX Runtime, or any NuGet package in these phases.
  The foundation layers are pure BCL.
- Do not port a module that is not in your assignment, even if it looks easy — parallel
  agents will collide. Report what you needed and stop.
- Do not create stub/placeholder implementations that throw `NotImplementedException`
  for work in your own scope. Either port it properly or report it as blocked.
