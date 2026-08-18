# .NET port — implementation status

Tracks what is actually built against `docs/DOTNET_PORT_PLAN.md`. Updated as phases land.

**Current state: Phases 0–2 complete.** Clean build with 0 warnings and 0 errors;
**433 tests passing** (375 unit + 58 parity), 9 skipped. OpenCV is in use via OpenCvSharp;
ONNX Runtime and the face pipeline are Phases 3–5 and remain unstarted.

The Phase 2 milestone is met in code: `ExtractFrames`, `MergeVideo`, `ConcatVideo`,
`RestoreAudio`, `ReplaceAudio`, `CopyImage`, `FinalizeImage`, `ReadAudioBuffer` and the
video reader/writer are ported. It is **not** verified end to end, because ffmpeg is not
installed here — command construction and graceful degradation are tested, an actual
encode round-trip is not.

## Library spike: OpenCvSharp and ONNX Runtime both work

The two riskiest bets in the plan were checked before Phase 2 committed to them, and both
hold in this environment (Ubuntu 24.04 x64, .NET 8):

- **OpenCvSharp4** (`OpenCvSharp4` + `OpenCvSharp4.official.runtime.linux-x64`) restores
  from NuGet, loads its native library, and `Cv2.Resize(..., InterpolationFlags.Area)`
  returns correct values. This is the parity-critical path — plan §9.1 chose OpenCvSharp
  precisely so `WarpAffine`/`Resize` semantics match cv2 exactly.
- **Microsoft.ML.OnnxRuntime** restores and initialises, reporting
  `CPUExecutionProvider`. No GPU is present here, so the CUDA/TensorRT provider packages
  remain unverified — that check belongs on real hardware and is still the open item from
  plan §3.

One wrinkle: OpenCvSharp4 4.13 ships a Roslyn 4.14 analyzer while SDK 8 runs Roslyn 4.8,
producing CS9057. With `TreatWarningsAsErrors` that fails the build, so CS9057 is listed
in `WarningsNotAsErrors` in `Directory.Build.props`. It disappears on a newer SDK.

## Environment deviation

The plan targets **.NET 10**; this repo currently targets **net8.0**. The SDK CDN
(`builds.dotnet.microsoft.com`) is blocked by the development environment's network
policy, so the SDK was installed from Ubuntu's archive, which carries 8.0 only. Nothing
in the foundation phases needs 9/10 features. Two places are affected and both are marked
in-code:

- `Json.ExpandIndentation` widens two-space indentation to four after serialisation,
  because `JsonSerializerOptions.IndentSize` arrived in .NET 9. Tagged `TODO(net9)`.
- `System.Numerics.Tensors.TensorPrimitives` is not available on net8.0 without an added
  NuGet package, so `NumPy`'s reductions use plain loops. Correctness first, per plan §4;
  SIMD is a Phase 9 concern.

Raising `TargetFramework` in `Directory.Build.props` is the whole migration.

## What is ported

| Python source | C# | Tests |
| --- | --- | --- |
| `types.py`, `choices.py` | `FaceFusion.Types` — 45 enums, value structs, records, `Choices` | wire round-trip for every enum |
| numpy subset | `FaceFusion.Tensors.NumPy` | verified against NumPy 2.4.6 |
| `common_helper.py` | `Core.CommonHelper` | ported |
| `normalizer.py` | `Core.Normalizer` | ported |
| `sanitizer.py` | `Core.Sanitizer` | ported |
| `hash_helper.py` | `Core.HashHelper` | verified against Python |
| `time_helper.py` | `Core.TimeHelper` | ported |
| `json.py` | `Core.Json` | byte-exact vs `json.dumps(indent=4)` |
| `filesystem.py` | `Core.FileSystem` | ported + `splitext` edge cases |
| `locales.py`, `translator.py` | `Core.Locales`, `Core.Translator` | 256/256 keys byte-identical |
| `ffmpeg_builder.py` | `Media.FfmpegBuilder` | 41 functions |
| `ffprobe_builder.py` | `Media.FfprobeBuilder` | ported |
| `curl_builder.py` | `Media.CurlBuilder` | ported |
| `shutil.which` (stdlib) | `Core.ProcessHelper.Which` | found/not-found, cross-checked |
| — (new) | `Parity.NpyReader` / `NpyArray` | 23 committed `.npy` fixtures |
| — (new) | `Parity.TensorComparison` | vs real `numpy.allclose` |
| — (new) | `Parity.ImageMetrics` | vs NumPy reference, cross-checked against skimage |

## Deliberate deviations from the Python

Recorded so they are choices rather than drift.

- **The `cli`/`ui` state split is dropped.** `app_context.py` inspects the Python call
  stack to decide whether the caller is under `jobs/` or `uis/`, and `state_manager.py`
  keys its global dict on the answer. That exists solely to work around Gradio's threading
  model (plan §3), so `app_context.py` is deliberately not ported.
- **Module globals become instance state.** `process_manager.py` and `logger.py` both keep
  module-level globals; the C# equivalents are instance classes with guarded fields, which
  makes them testable and thread-safe. Tests construct one per case instead of resetting a
  shared global.
- **`ProcessManager.Manage()` is not a port.** It is an `IDisposable` convenience added on
  top; `process_manager.py` has no context manager. Documented as such in the class so it
  is not mistaken for parity.
- **`Logger` does not use `Microsoft.Extensions.Logging`** as plan §2 maps it to, because
  that needs a NuGet package and the foundation layers were kept dependency-free. It is a
  minimal self-contained logger over the existing `LogLevel` enum. Revisit when the DI
  container lands — that needs `Microsoft.Extensions.DependencyInjection` anyway, at which
  point the abstractions package is no extra cost.

## Open divergences in the Vision port

Documented by the porting agent rather than left silent. Both are recorded here because
they are the kind of thing that quietly changes output.

- **Video metadata comes from OpenCV's demuxer, not ffprobe.** `vision.py` sources frame
  counts, fps and resolution from `ffprobe` via `video_manager.py`; `Vision.cs` uses
  `OpenCvSharp.VideoCapture` properties instead, because neither the ffprobe runner nor
  the ffmpeg binary existed when it was written. The two demuxers do not always agree on
  frame count for every container. `Ffprobe.cs` now exists, so this should be rewired —
  which requires breaking the `ffmpeg.py -> vision.py -> ffprobe.py` cycle, most cleanly
  with a metadata-provider interface in `FaceFusion.Core` implemented in `FaceFusion.Media`.
- ~~**`EqualizeFrameColor`'s final cast rounds where NumPy truncates.**~~ **Fixed.** See
  parity defect 7 below.

A third difference is deliberate and should stay: `ReadStaticImage`/`ReadStaticVideoFrame`
return a `Clone()` from their cache, where Python's `lru_cache` hands back the same array
object to every caller. Reproducing Python's aliasing is unsafe under `Mat` disposal — a
caller disposing "their" frame would corrupt the cache for everyone.

## Parity defects found and fixed

Each of these was caught by a test or by the de-duplication pass rather than by reading
the C#, which is the argument for porting tests alongside code (plan §7, conventions
rule 2).

1. **`numpy.round(a, decimals)` ≠ `Math.Round(value, decimals)`.** NumPy computes
   `round(a * 10**decimals) / 10**decimals` entirely in the array's dtype, so the scaling
   step introduces float32 error *before* the half-to-even step.
   `numpy.round(float32(2.675), 2)` is `2.68`; computing in double gives the
   mathematically-correct-but-divergent `2.67`. NumPy's arithmetic is replicated
   deliberately.
2. **`os.path.splitext` ≠ `Path.GetExtension`.** They disagree on leading-dot names such
   as `.gitignore`. File extensions drive format detection throughout the codebase, so
   Python's algorithm is reimplemented and its edge cases pinned against real Python
   output.
3. **`GetFirst`/`GetLast` lost the empty case for value types.** An unconstrained generic
   returning `default` yields `null` for a reference type but `0` for `int`, making an
   empty list indistinguishable from a leading `0`. Reachable in practice —
   `video_manager.py:92` calls `get_first`/`get_last` over ints. Struct-constrained
   `...OrNull` siblings added; both behaviours pinned by tests.
4. **All three `shutil.which` copies skipped the execute-permission check.** Python's
   `shutil.which` filters candidates with `os.access(path, X_OK)`; each of the three
   independently-written ports checked only for existence, so a non-executable file
   sharing a tool's name on `PATH` would have been returned as the tool. Fixed in the
   shared `ProcessHelper.Which`, which checks the Unix mode bits on POSIX and applies
   `PATHEXT` on Windows.
5. **JSON serialisation dropped null-valued keys and used the wrong indent width.**
   `JsonIgnoreCondition.WhenWritingNull` would have silently removed
   `"date_updated": null`, which `job_manager.create_job` writes. Job files must
   round-trip between the two implementations (plan §9.3), so output is now byte-exact
   against `json.dump(..., indent = 4)`.
7. **`EqualizeFrameColor` rounded where NumPy truncates.** `Mat.ConvertTo` to 8-bit uses
   `saturate_cast`, which rounds; NumPy's `.astype(uint8)` truncates toward zero. Confirmed
   reachable rather than theoretical: real fractional values captured out of the
   OpenCvSharp pipeline (e.g. `88.57573`) become `88` under NumPy and `89` under
   `ConvertTo`. Replaced with a bulk `GetArray`/`SetArray` round-trip using a plain
   `(byte)` cast, which truncates toward zero like NumPy, and pinned all 48 channel values
   in a test against the verified numbers.
6. **Non-nullable `State` fields conflated "unset" with zero — and silently turned an
   unset face-selector gender into `auto`.** 13 fields whose `program.py` default is
   `config.get_*_value(section, option)` with no fallback (therefore `None`) were declared
   non-nullable, so the builder filled them with zero values. The worst case was
   `FaceSelectorGender`/`FaceSelectorRace`: the placeholder resolved "unset" to the first
   enum member, which is `Auto` — but `auto` is a *real, distinct* value in
   `face_selector.py:88` that triggers gender inference from the source face. Unset and
   `auto` are different behaviours, and the port had merged them. The 13 fields are now
   nullable, with `trim_frame_start`/`trim_frame_end` (`Optional[int]` at
   `vision.py:149`) and `output_video_fps` among them.

## Known gaps

- **`TimeHelper` carries its own copies of a few locale strings**, written before
  `Locales`/`Translator` existed. Should be redirected through `Translator`.
- **`Choices` has private copies of `create_int_range`/`create_float_range`** because
  `FaceFusion.Types` cannot depend on `FaceFusion.Core` (the dependency runs the other
  way). Either move the helpers down into `Types` or accept the duplication deliberately.
- **`FaceFusion.Types` fields backed by `NDArray` in Python are typed `object`** —
  bounding boxes, embeddings, landmark sets — pending the real types from
  `FaceFusion.Tensors` and OpenCV in Phases 3–4.
- **`Media` builders still take `string` where an enum belongs**, marked `TODO(types)`
  throughout. The types now exist; tightening the signatures is mechanical.
- **The parity harness has no real pipeline to compare against yet.** The reader,
  comparison and metrics layers are built and tested (see `docs/PARITY_HARNESS.md`), and
  `tools/parity/parity_dump.py` is ready for Phase 4 to call — but nothing dumps from a
  real FaceFusion run, because the face pipeline is not ported and neither ffmpeg nor the
  models exist in this environment.
- ~~SSIM is not verified against skimage.~~ **Closed.** scikit-image is now installed and
  `generate_fixtures.py` cross-checks the reference against it, failing if they diverge by
  more than 1e-12; they agree to within 5e-14.

## Next steps, in order

1. Port `config.py` + `facefusion.ini` and the `Settings` record with DI (plan §3, Phase 1).
2. Tighten the `TODO(types)` signatures in `Media` now that `FaceFusion.Types` exists.
3. Port `ffmpeg.py` / `ffprobe.py` runners, `vision.py`, `temp_helper.py` (Phase 2),
   reaching the milestone of extracting, rewriting and remerging video frames with no
   models involved.
