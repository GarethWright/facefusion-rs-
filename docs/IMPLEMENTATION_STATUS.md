# .NET port — implementation status

Tracks what is actually built against `docs/DOTNET_PORT_PLAN.md`. Updated as phases land.

**Current state: Phase 0–2 foundations, partial.** Clean build with 0 warnings and
0 errors; **237 tests, all passing**. Nothing here touches ONNX Runtime, OpenCV, or the
face pipeline yet — those are Phases 3–5 and remain entirely unstarted.

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
- **No parity harness yet.** Plan §7's `.npy` interchange and `NpyReader` are not built.
  This is the most important outstanding Phase 0 item, and it gates Phase 4 —
  floating-point parity in the face pipeline cannot be verified without it.

## Next steps, in order

1. Build the parity harness (plan §7) — `.npy` dumping on the Python side, `NpyReader` on
   the .NET side, comparison driver. Everything in Phases 4–5 depends on it.
2. Port `config.py` + `facefusion.ini` and the `Settings` record with DI (plan §3, Phase 1).
3. Tighten the `TODO(types)` signatures in `Media` now that `FaceFusion.Types` exists.
4. Port `ffmpeg.py` / `ffprobe.py` runners, `vision.py`, `temp_helper.py` (Phase 2),
   reaching the milestone of extracting, rewriting and remerging video frames with no
   models involved.
