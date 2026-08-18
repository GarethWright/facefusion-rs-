# .NET port — implementation status

Tracks what is actually built against `docs/DOTNET_PORT_PLAN.md`. Updated as phases land.

**Current state: Phases 0–5 complete, Phase 6 substantially done.**

| Phase | State |
| --- | --- |
| 0 Foundations + parity harness | complete |
| 1 Config, settings, logging, i18n | complete |
| 2 Media plumbing | complete, verified end to end |
| 3 Inference layer | complete (CPU verified; GPU providers unverified — no GPU here) |
| 4 Face pipeline | complete, verified against real ONNX inference |
| 5 Processors | complete — 11/11 plus the audio layer |
| 6 Workflows, jobs, CLI | **complete.** All 11 processors wired; 10 verified pixel-for-pixel against the Python CLI, the 11th (`deep_swapper`) blocked by a proxy-inaccessible model host |
| 7 UI (Blazor) | **not started** |
| 8 Streaming / webcam | **not started** |
| 9 Performance tuning | **not started** (deliberately sequenced after parity) |

Phases 7 and 8 are the plan's own post-v1 items: the UI is 4,444 lines of Gradio with no
mechanical translation and was estimated at 4–8 weeks, streaming at 2–3.


## Environment: the real Python pipeline now runs here

Initially neither ffmpeg nor the example media nor most Python packages were available,
which forced several tests to be skipped and left the parity harness with nothing real to
compare against. Most of that is now resolved. What is reachable from this environment:

| Source | Status |
| --- | --- |
| `pypi.org` / `files.pythonhosted.org` | **reachable** (in the proxy's no-proxy list) |
| `archive.ubuntu.com` | **reachable** — this is where the .NET SDK and ffmpeg came from |
| `github.com` release assets | **reachable** — the example media download fine |
| `builds.dotnet.microsoft.com` | **blocked** by policy — hence net8.0 rather than net10.0 |

Installed as a result: ffmpeg/ffprobe 6.1.1, and for Python numpy 2.4.6,
opencv-python-headless 5.0.0, onnxruntime, onnx 1.22.0, scipy, scikit-image, tqdm.

**Consequence: `import facefusion.vision` works and the real Python implementation can be
executed for ground truth.** That is the piece the parity harness was missing — comparisons
can now be made against the actual pipeline rather than against reimplementations of it.
Example media live in `/tmp/facefusion-test-examples`, fetched by
`tools/parity/fetch_examples.sh`, the same path `tests/helper.py` uses so both suites share
one copy.

CI still has none of this, so tests that need the media must skip with a message pointing
at the fetch script rather than fail.

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

## Phase 4 parity results

The face pipeline was compared against real Python inference on `source.jpg` (10 detected
faces). The headline result: **every model-input tensor matched Python exactly**, which is
the check that matters — if inputs match, output differences are ONNX Runtime's own; if
inputs differ, there is a preprocessing bug.

| Stage | Result |
| --- | --- |
| `prepare_detect_frame` / `normalize_detect_frame` (all 3 variants) | exact, rtol = atol = 0 |
| Detector outputs, all 4 families | max rel. diff ~2e-7 (float32 noise) |
| Landmarker scale/translation | matched to 1e-9 |
| `conditional_optimize_contrast` | bit-for-bit |
| Recognizer / classifier input tensors | matched to 1e-6 |
| Content analyser tensors, scores, verdict | 1e-6; verdict exact |

One genuine, measured divergence remains, and it is **not** a port defect: OpenCvSharp's
native OpenCV build and `opencv-python-headless` resolve bilinear interpolation slightly
differently, so `WarpAffine` output differs by up to 2/255 on ~9% of pixels given
bit-identical affine matrices (~62 dB PSNR). It is isolated by dedicated tests and only
loosens the two assertions it cascades into — the landmarker input tensor (PSNR > 50 dB,
measured 56) and the recognizer embedding (cosine similarity > 0.999, measured 0.99996,
which is the property face matching actually depends on). Every discrete output — landmarks
after inverse transform, gender, age, race — is robust to it and asserts at 1e-3/1e-4.

## The content analyser gate

Plan §9.6 requires the NSFW gate to survive the port. It is ported completely — all three
models, the majority-vote rule, every threshold — with **no bypass flag anywhere**.

Python makes it tamper-evident by hashing `inspect.getsource(content_analyser)` in
`core.py`. C# has no `inspect.getsource`, so `ContentAnalyser.VerifyIntegrity` hashes the
module's own source file, located via a `[CallerFilePath]` constant fixed at compile time
so a caller cannot redirect it, using the same CRC32 hash. It fails closed. The expected
hash deliberately lives outside the file, mirroring Python's split (`3c6ce25e` lives in
`core.py`).

Honest limits, documented in the class: it detects on-disk edits to the file, but cannot
cover a deployment that ships only the compiled DLL without sources, nor IL tampering (IL
is not stable across build configurations, so a hard-coded IL hash would false-fail), nor
in-memory patching. Wiring it into a `common_pre_check` equivalent belongs to Phase 6.

## A note on running the test suite

Real ONNX inference makes the suite slow — a full `dotnet test` is several minutes on 4
cores. **Do not run it from several agents at once.** Doing so once drove load average to
71 on a 4-core box, at which point every run starved and the whole thing appeared to hang.
Iterate with `--filter` on the tests you are changing and leave full-suite verification to
one runner.

## The CLI runs, and matches Python

`headless-run` produces a real video end to end. Verified by running both CLIs with
identical arguments:

```
dotnet run --project src/FaceFusion.Cli -- headless-run \
    -t /tmp/facefusion-test-examples/target-240p.mp4 -o out.mp4 \
    --processors frame_colorizer --trim-frame-end 8
```

| | C# | Python |
| --- | --- | --- |
| Output | 426x226, 25fps, 8 frames | identical |
| Log lines | 3, line-for-line identical | — |
| Frame diff | PSNR 43.7 dB, max 16/255 | consistent with two independent libx264 encodes |
| Warm runtime | 25.3 s | 49.3 s |

The ~1.95x is one small CPU-only workload on a tiny model, where orchestration overhead
weighs heavier than it would on a long 1080p job — it is a data point, not a benchmark.
The first Python run took 105 s but included model downloads; re-running warm is what makes
the comparison fair.

Job files written by the C# CLI are validated and read back by the real Python
`job_manager`, and vice versa.

## All eleven processors are wired and compared against Python

Each was verified by running both CLIs with identical arguments on the same 8 frames of
`target-240p.mp4` (`--trim-frame-end 8 --execution-thread-count 1`) and diffing the decoded
pixels:

| processor | PSNR | max diff | pixels >30 |
| --- | --- | --- | --- |
| `frame_colorizer` | 43.7 dB | 16 | 0.000% |
| `background_remover` | verified | — | — |
| `face_debugger` | verified | — | — |
| `face_swapper` | 43.3 dB | — | 0.000% |
| `age_modifier` | 43.3 dB | 20 | 0.000% |
| `expression_restorer` | 43.36 dB | 21 | 0.000% |
| `face_editor` (smile 1.0, yaw 0.5) | 42.64 dB | 30 | 0.000% |
| `lip_syncer` (`source.mp3`) | 43.30 dB | 23 | 0.000% |
| `face_enhancer` | 43.02 dB | 25 | 0.000% |
| `frame_enhancer` | 42.82 dB | 20 | 0.000% |
| `deep_swapper` | **cannot be run here** — see below | | |

Two composed runs check that the chain works, not just each processor alone:

| run | PSNR | max diff | pixels >30 |
| --- | --- | --- | --- |
| video, `face_swapper face_enhancer frame_enhancer` | 42.20 dB | 86 | 0.001% |
| image, `face_swapper face_enhancer` (`image_to_image` path) | 47.13 dB | 21 | 0.000% |

The 42–43 dB band is what two independent libx264 encodes of *identical* pixels produce, so
these are encoder noise, not divergence — the image run, which skips video encoding entirely,
lands 4 dB higher for exactly that reason. The three-processor chain's single max-86 pixel is
0.001% of the frame (roughly one pixel in 100,000), a face-boundary pixel amplified through
three successive stages.

`deep_swapper` is wired but **unverified**, and deliberately not claimed otherwise: its
`.dfm` models are hosted on `huggingface.co`, which this environment's proxy refuses with 403.
Neither implementation can run it here, so there is nothing to compare. What was checked is
that both refuse rather than emit wrong output — Python fails hash validation after attempting
a download, this port fails its file-presence pre-check (`download.py` is deliberately not
ported).

Two defects surfaced only because the binary was run:

- **`face_swapper` had never executed end to end.** `PixelBoost.ExplodePixelBoost`
  hard-asserted `CV_8UC3` while `NormalizeCropFrame`'s own remarks are explicit that it
  deliberately returns float Mats. Its parity tests were green throughout, because they call
  `ForwardSwapFace` directly and never the assembled pipeline.
- **No processor reading source audio could run.** `HeadlessRunner.BuildRunContext` supplied
  an `ExtractVoice` delegate that threw unconditionally, so `lip_syncer` failed the moment it
  asked for a voice frame. It now opens the `voice_extractor` session lazily (`Lazy<T>`,
  thread-safe by default — `ToVideo` calls it from several worker threads), so a run that
  never touches audio does not pay for loading the model. Fixing it also put the whole audio
  chain under test for the first time: ffmpeg decode, voice extraction, spectrogram, mel-frame
  extraction — all of which the 43.30 dB `lip_syncer` figure now covers.

## Wiring a processor requires running it, not compiling it

An attempt to wire the remaining eight processors was **rejected and not merged**. It
compiled, the unit tests passed, and the report claimed ten of eleven processors wired —
but running the binary showed `face_swapper` exiting 1 with no output while the Python
equivalent succeeded, and, worse, `frame_colorizer` — previously wired, pixel-verified and
committed — had been regressed to failing too. Proved by stashing the changes and
rebuilding, which restored it.

The work is preserved in `git stash` rather than discarded, since the missing
`IProcessor` adapters in it may be salvageable. It is not applied, because a tree that
builds and passes a thousand tests while silently failing at runtime is worse than one
that does less and works — the suite stops being evidence.

The standing rule for this phase: **a processor is wired only when both CLIs have been run
on the same input and the outputs compared.** Anything else keeps its named
`NotSupportedException`. `face_swapper`'s own parity tests were green throughout while its
assembled pipeline had never once executed — which is how the `PixelBoost` dtype defect
survived to be found here.

## Open defect: memory scales badly with execution_thread_count

`age_modifier` on **8 frames of a 426x226 clip** was killed by the OOM killer at
**~11.7 GB RSS** (`dmesg`: `anon-rss:11669352kB`). The same run with
`--execution-thread-count 1` completes and produces output matching Python at 43.3 dB, so
this is not a correctness bug — the per-frame footprint is simply enormous and multiplies
by the thread count. Python survives the same run at the same default of 8 threads.

This is the failure plan §5a predicted, and it matters more than the test suite suggests:
~1.5 GB per in-flight frame on a tiny clip makes any real 1080p job impossible. Nothing in
the suite catches it because tests run few frames at low concurrency.

Worth investigating first: whether concurrent `OrtValue`/`Mat` allocations are being held
until GC rather than disposed promptly, and whether ONNX Runtime's arena allocator is being
defeated by per-run allocation. The plan's §5b IO-binding work is the likely remedy.

## Silent failures in the CLI were themselves a defect

The above took far too long to diagnose because the CLI exited 1 after
"creating temporary resources" with no further output. It now reports which processor's
pre-check failed, names the error code's meaning, and logs any exception with its stack
trace at debug level. Python lets a traceback reach the terminal; swallowing it was strictly
worse than a noisy failure.

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
