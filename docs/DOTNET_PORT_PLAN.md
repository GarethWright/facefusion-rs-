# FaceFusion → .NET Port Plan

Target of the port: FaceFusion 3.8.2 (this repository), ~18.7k lines of Python plus
~3.8k lines of tests.

Target platform: **.NET 10 (LTS), C# 14**, cross-platform (Linux / Windows / macOS).

This document is a plan, not an implementation. It states what gets ported, in what
order, onto which packages, how parity is proven, and which decisions need to be made
before code is written.

> **Note on repository naming.** This repo is named `facefusion-rs-` and the working
> branch is `claude/rust-application-port-*`, from an earlier Rust-targeted revision of
> this plan. The target is now .NET; the names are cosmetic and can be changed
> independently.

---

## 1. What the application actually is

Despite the "AI" framing, FaceFusion is mostly **plumbing around ONNX Runtime and
FFmpeg**. That is what makes it a realistic port target.

| Area | Python LOC | What it does |
| --- | ---: | --- |
| Core runtime (`facefusion/*.py`) | 8,364 | CLI, state, config, jobs, ffmpeg/ffprobe, vision, face pipeline, download, audio |
| Processors (`facefusion/processors/**`) | 5,875 | 11 processor modules + shared model registries |
| UI (`facefusion/uis/**`) | 4,444 | Gradio web UI: 50 components, 4 layouts |
| Tests (`tests/**`) | 3,821 | pytest, heavily CLI-integration-shaped |

Runtime dependencies split into three groups:

1. **Native tools shelled out to** — `ffmpeg`, `ffprobe`, `curl`, `nvidia-smi`. Invoked
   via `subprocess` with builder modules (`ffmpeg_builder.py`, `ffprobe_builder.py`,
   `curl_builder.py`). These port to C# almost verbatim; the builders are pure
   string/list construction and already have unit tests.
2. **Numeric/vision libraries** — `onnxruntime`, `opencv-python-headless`, `numpy`,
   `scipy`, `onnx`. These are the substance of the port.
3. **UI** — `gradio` + `gradio-rangeslider`. Rewritten as Blazor Server (§6).

Data flow, end to end:

```
CLI args ──▶ state_manager ──▶ job (JSON on disk) ──▶ job_runner
                                                        │
                                        ┌───────────────┴───────────────┐
                                   image-to-image                 image-to-video
                                        │                               │
                                        │                  ffmpeg extract frames
                                        │                  (or in-memory pipe)
                                        ▼                               ▼
                        content_analyser → face_detector → face_landmarker →
                        face_recognizer → face_classifier → face_selector →
                        face_masker → processor modules (ONNX) → paste_back
                                        │                               │
                                        ▼                               ▼
                                   write image                ffmpeg merge + audio
```

Every processor module implements the same informal interface (`pre_check`,
`pre_process`, `post_process`, `process_frame`, `create_static_model_set`,
`register_args`, `apply_args`, …) and is loaded by `importlib` from a directory scan.
In .NET this becomes an `IProcessor` interface resolved through dependency injection —
a natural fit, and the single largest structural change in the port (§3).

---

## 2. Package mapping

| Python | .NET | Notes / risk |
| --- | --- | --- |
| `onnxruntime` | **`Microsoft.ML.OnnxRuntime`** (+ `.Gpu`, `.Gpu.Linux`, `.Gpu.Windows`, `.DirectML`, OpenVINO/QNN packages) | **First-party, maintained by the ONNX Runtime team, with prebuilt natives on NuGet per execution provider.** This is the single biggest reason to choose .NET for this port — FaceFusion supports 8 EPs and `execution.py`'s provider matrix maps onto shipped packages instead of source builds. |
| `numpy` | **No direct analogue** — see §4. `OpenCvSharp.Mat` for image-shaped data, `System.Numerics.Tensors` (`TensorPrimitives`) for vectorised elementwise ops, `Span<T>`/`Memory<T>` for slicing, `OrtValue` for model I/O | The main ergonomic tax of choosing .NET. Mitigated by building a small `TensorOps` layer in Phase 0 (§4). `numpy.interp` (79 uses!) is the single most-used primitive and has no BCL equivalent — write and test it first. |
| `opencv-python-headless` | **`OpenCvSharp4`** + `OpenCvSharp4.runtime.{win,linux,osx}` | Only ~35 distinct cv2 functions are used. Same C++ implementations → **bit-exact parity** for `WarpAffine`, `Resize` (INTER_AREA/CUBIC), `GaussianBlur`, `EstimateAffinePartial2D`. Chosen over Emgu.CV on performance grounds: OpenCvSharp is a thinner P/Invoke layer over the native API with `Mat` as a direct handle, where Emgu interposes more managed abstraction per call — and this pipeline makes thousands of small cv2 calls per frame. |
| `cv2.dnn.blobFromImage` | `CvDnn.BlobFromImage` | Does HWC→NCHW + scale + mean subtraction in native code. Use it for every model preprocess rather than hand-written managed loops — faster and closer to Python's numerics. |
| `scipy.signal` | `MathNet.Numerics` (FFT, Hann window) + hand-written STFT/ISTFT/`lfilter`/`resample`; `triang` is a 3-line function | 6 call sites total (`audio.py`, `voice_extractor.py`). Contained but numerically fussy — `scipy.signal.stft` scaling and padding conventions must be reproduced exactly or lip-sync and voice extraction drift. |
| `scipy.spatial.transform.Rotation` | `System.Numerics.Matrix4x4` / `Quaternion` | 1 call site (`live_portrait.py`, Euler xyz → matrix). Trivial. |
| `onnx` (graph surgery) | `Google.Protobuf` + generated `onnx.proto` classes | `model_helper.py` only reads `graph.initializer[-1]` to get a model's embedding matrix. A minimal `TensorProto` decode is enough — do **not** pull in a full ONNX graph library. |
| `argparse` (`program.py`, 348 LOC) | `System.CommandLine` (fallback: `Spectre.Console.Cli`) | ~15 subcommands, ~100 flags, defaults sourced from `facefusion.ini` via `config.py`. Layered defaults (ini → env → CLI) are set programmatically after reading the ini. Watch for `System.CommandLine` API churn — pin the version. |
| `configparser` + `facefusion.ini` | `Microsoft.Extensions.Configuration.Ini` | Keep the same ini file and key names — user configs must not break. |
| `tqdm`, `cli_helper.render_table` | **`Spectre.Console`** | Covers progress bars, tables, and the terminal panel in one dependency. Match the `ascii = ' ='` bar style and postfix fields so logs stay diffable against Python. |
| `logging` | `Microsoft.Extensions.Logging` | Map `log_level` choices (error/warn/info/debug) directly. |
| `ThreadPoolExecutor` (frame fan-out) | **TPL Dataflow** (`TransformBlock` with `BoundedCapacity` + `EnsureOrdered`) | A better fit than the Python original: bounded backpressure and ordered output for free, which is exactly what the video pipeline needs. `execution_thread_count` → `MaxDegreeOfParallelism`. |
| `threading.Semaphore` (`thread_helper.py`) | `SemaphoreSlim` | Guards non-thread-safe EPs (CoreML/DirectML). Keep the same conditional logic. |
| `subprocess` | `System.Diagnostics.Process` with async stdout/stderr readers | ffmpeg progress parsing stays line-based. |
| `curl` subprocess + `curl_builder.py` | Keep shelling out to `curl` (v1); `HttpClient` later | Shelling out preserves proxy/mirror behaviour (`download_providers`, resume, `--retry`) with zero surprises. |
| `importlib` module registry | **`Microsoft.Extensions.DependencyInjection`** | See §3 — DI is the idiomatic replacement for both the processor scan and the global `state_manager`. |
| `gradio` | **Blazor Server** | See §6. |

Cross-cutting: `System.Text.Json` (job files, `json.py` — use source-generated contexts
for trim/AOT safety), `ArrayPool<byte>` (§5), and the existing hash scheme in
`hash_helper.py` (an 8-hex-digit CRC-style digest — **read it and reproduce exactly**,
model `.hash` sidecar files depend on it).

---

## 3. Proposed solution layout

A solution of class libraries, so the numeric core can be tested and reused without
dragging in the UI or the CLI.

```
FaceFusion.sln
├── src/
│   ├── FaceFusion.Types/          # types.py, choices.py — enums, records, state keys
│   ├── FaceFusion.Core/           # settings, config, logging, translator/locales,
│   │                              #   filesystem, temp, process state, hashing,
│   │                              #   time, normalizer, sanitizer
│   ├── FaceFusion.Tensors/        # the numpy-compat layer (§4) — no other deps
│   ├── FaceFusion.Media/          # ffmpeg/ffprobe/curl builders + runners, vision,
│   │                              #   audio, video/camera managers, download
│   ├── FaceFusion.Inference/      # session pool, execution providers, model registry
│   ├── FaceFusion.Face/           # detector, landmarker, recognizer, classifier,
│   │                              #   selector, masker, tracker, creator, store,
│   │                              #   face helpers (warp templates), content analyser
│   ├── FaceFusion.Processors/     # IProcessor + the 11 modules + pixel boost,
│   │                              #   live portrait, voice extractor
│   ├── FaceFusion.Workflows/      # to_image, to_video, image_to_image, image_to_video
│   ├── FaceFusion.Jobs/           # job manager, runner, store, list
│   ├── FaceFusion.Cli/            # System.CommandLine program + routing → executable
│   └── FaceFusion.Ui/             # Blazor Server app (§6) → executable
├── tests/
│   ├── FaceFusion.UnitTests/      # xUnit, ported from tests/test_*.py
│   └── FaceFusion.ParityTests/    # golden comparison against Python (§7)
└── build/                         # parity harness driver, model fixture fetch, packaging
```

Two hard rules for the layout:

**1. No global mutable state.** In Python, `state_manager.get_item('...')` is called from
200+ places. In .NET this becomes an immutable `Settings` record injected via DI:

```csharp
services.AddSingleton<Settings>(sp => SettingsBuilder.From(ini, env, args));
services.AddSingleton<IProcessor, FaceSwapper>();
services.AddSingleton<IProcessor, FaceEnhancer>();
services.AddKeyedSingleton<IProcessor>("face_swapper", …);
```

This is the change that makes the port testable and thread-safe, and it must be decided
up front rather than retrofitted. It also lets the `cli`/`ui` dual-context in
`state_manager.py` and `app_context.py` disappear entirely — that split exists purely to
work around Gradio's threading model, and Blazor's scoped DI replaces it properly.

**2. The processor interface is an interface**, resolved by name from DI:

```csharp
public interface IProcessor
{
    string Name { get; }
    ModelSet GetModelSet(DownloadScope scope);
    Task<bool> PreCheckAsync(CancellationToken ct);
    bool PreProcess(ProcessMode mode);
    ProcessorOutputs ProcessFrame(in ProcessorInputs inputs);
    void PostProcess();
}
```

Dynamic loading of third-party processors is lost by default, though .NET offers a
cleaner recovery path than most: `AssemblyLoadContext` can load a user-supplied DLL that
contributes `IProcessor` implementations, if it turns out anyone relies on it (§9.5).

---

## 4. The numpy problem (the main cost of choosing .NET)

.NET has no ndarray. This is the one place where the Rust alternative was clearly
stronger, and pretending otherwise would sink the schedule. The mitigation is to stop
treating it as "port numpy" and instead **route each category of array work to the right
existing tool**:

| Python pattern | .NET home |
| --- | --- |
| Image-shaped data (H×W×C uint8/float) | `OpenCvSharp.Mat` — native memory, real operators, identical semantics |
| Model input/output tensors | `OrtValue` created over pinned or native memory (§5) |
| Landmark/matrix math (5×2, 68×2, 3×3) | `System.Numerics` fixed-size types, or small `float[]` with helpers |
| Elementwise vector math over big buffers | `System.Numerics.Tensors.TensorPrimitives` (SIMD-accelerated) |
| Everything else | `FaceFusion.Tensors` — a small, hand-written, heavily-tested helper library |

`FaceFusion.Tensors` should stay **small and closed-ended**. An audit of the Python
source shows the genuinely-used numpy surface is about 20 operations:

`interp` (79 uses), `expand_dims`, `concatenate`, `stack`, `hstack`/`vstack`, `clip`,
`round`, `mean`, `min`/`max`/`amax`, `argmax`, `where`, `pad`, `squeeze`,
`ascontiguousarray`, `linalg.norm`, `dot`, `linspace`, `zeros`/`zeros_like`,
transpose/HWC↔CHW.

Build this in Phase 0, unit-test every function against `.npy` goldens generated from
NumPy, and freeze it. Do **not** let it grow into a general-purpose array library — that
is a multi-month project in its own right and is not what this port needs.

Honest accounting: this makes Phases 4–5 roughly 1–2 weeks more expensive than the same
phases would be with a real ndarray type, and it introduces a class of manual-indexing
bugs that the parity harness (§7) exists to catch.

---

## 5. Memory, GC, and performance engineering

### 5a. The GC risk

A single 1080p BGR frame is ~6 MB. Anything over 85 KB lands on the **Large Object
Heap**, which is not compacted by default. A naive port that allocates managed
`byte[]`/`float[]` per frame will produce LOH fragmentation and Gen2 pressure, and can
end up *slower than Python* — where cv2 and numpy keep pixel data in native memory the
whole time.

The port must therefore adopt these rules from Phase 2 onward, not as a later
optimisation:

1. **Pixel data lives in `Mat`, not in managed arrays.** `OpenCvSharp.Mat` allocates
   natively and is invisible to the GC. Frames should enter as `Mat`, flow through the
   pipeline as `Mat`, and leave as `Mat`. Managed copies only at genuine boundaries.
2. **Strict `IDisposable` discipline.** `Mat`, `OrtValue`, and `InferenceSession` are all
   disposable and all wrap native memory. Every `Mat` gets a `using`; enable the
   analyzers that enforce it (`CA2000`, `CA1063`). This is the most likely source of a
   slow leak.
3. **Zero-copy into ONNX Runtime.** Use the `OrtValue` API
   (`OrtValue.CreateTensorValueFromMemory` over pinned or native memory) with
   `InferenceSession.Run(RunOptions, inputNames, inputValues, outputNames, …)`, *not* the
   older `NamedOnnxValue`/`DenseTensor` path, which copies on every call.
4. **`ArrayPool<byte>.Shared`** for ffmpeg pipe buffers and any transient managed
   staging buffer.
5. **Server GC** (`<ServerGarbageCollection>true</ServerGarbageCollection>`) plus
   concurrent GC for the CLI; measure `<ConserveMemory>` if fragmentation shows up.

Add an allocation regression test to the parity harness: process N frames and assert
Gen2 collections and peak working set stay under a threshold. This is cheap to write and
catches the failure mode early, when it is still a one-line fix.

### 5b. Where the real speed is

The rules above are about *not being slower than Python*. These are about being
meaningfully faster. All of them are available in .NET and none are done by the Python
original — but each should be **measured before being adopted**, and none belong in v1
until CLI parity is green (§10, Phase 6).

**1. `OrtIoBinding` to cut host↔device copies.** Today every model invocation
round-trips: CPU tensor → device → inference → device → CPU tensor. The pipeline runs
detector → landmarker → recognizer → classifier → swapper → enhancer per face per frame,
so those copies add up. `InferenceSession.CreateIoBinding()` lets you bind pre-allocated
device memory to inputs and outputs; any layout-mismatch copy then happens **once at
binding time rather than on every `Run`**. Bind once per session at pool-creation time
and reuse across frames.

The caveat that stops this being a free win: the inter-model steps (`warp_face_by_*`,
`paste_back`) run on the CPU via OpenCV, so the pipeline cannot stay resident on the GPU
end to end. The win is per-stage — biggest for the largest models (`face_swapper`,
`frame_enhancer`) at high pixel-boost resolutions, negligible for the small detectors.
Target it there.

**2. Pinned (page-locked) host memory for transfers.** Where a copy is unavoidable, the
CUDA EP moves data substantially faster out of page-locked memory than pageable memory.
Allocate the staging buffers through ORT's CUDA pinned allocator (`OrtMemoryInfo` with
the pinned allocator) rather than as ordinary managed arrays.

**3. `System.IO.Pipelines` for the ffmpeg bridge.** `to_video`'s in-memory path pipes raw
frames to and from ffmpeg over stdio. `System.IO.Pipelines` handles the
partial-read/buffer-management problem with pooled memory and no per-read allocation —
strictly better than naive `Stream.Read` loops, and this is a genuine hot path at 1080p+.

**4. Default to the in-memory frame path.** FaceFusion already implements both a
disk-based (`process_disk_frames`) and an in-memory (`process_memory_frames`) video path.
The disk path writes every frame to a temp file and reads it back. Where
`video_memory_strategy` allows, the .NET port should prefer the memory path — this is an
application-level decision worth more than most micro-optimisation.

**5. `CvDnn.BlobFromImage` over managed loops** — already noted in §2, restated because
it matters: preprocessing must not be hand-written C# iterating pixels.

### 5c. JIT vs NativeAOT — pick JIT

An earlier revision of this plan treated NativeAOT as an attractive stretch goal. On
reflection that is backwards for this workload:

- **Steady-state throughput slightly favours the JIT.** Tiered compilation with Dynamic
  PGO specialises hot managed code using runtime profile data and detects the actual CPU
  ISA at runtime; NativeAOT commits to a baseline ISA at publish time.
- **It barely matters either way**, because essentially all the compute is inside native
  ORT and OpenCV. Managed codegen quality is not what determines this application's
  throughput.
- NativeAOT's genuine benefit is **startup time**, which matters for a CLI invoked once
  per job and not at all for a long video render.

**Recommendation: JIT, with `TieredPGO` enabled and ReadyToRun for startup** (§8). Treat
NativeAOT as unnecessary rather than aspirational; it also conflicts with reflection-based
DI and Blazor, so dropping it removes constraints from §3 and §6 for no measurable loss.

---

## 6. The UI: Blazor Server

`facefusion/uis` is 4,444 lines of Gradio: 50 components, 4 layouts (default, benchmark,
jobs, webcam), a global component registry (`UI_COMPONENTS`), and cross-component event
wiring.

**Blazor Server is the closest analogue to Gradio in any ecosystem**, and it is a genuine
reason to prefer .NET here:

| Gradio concept | Blazor Server equivalent |
| --- | --- |
| Server-side Python state, browser is a thin client | Server-side C# state, browser is a thin client over SignalR |
| `gradio.Component` with `.change()` / `.click()` handlers | Razor component with `@bind` / `EventCallback` |
| `UI_COMPONENTS` global registry for cross-wiring | Scoped DI service holding UI state, injected where needed |
| `gradio.Blocks` layouts | Razor layouts / component composition |
| Live preview updates pushed to the browser | `StateHasChanged()` over the existing SignalR circuit |
| `launch(server_name, server_port, share)` | Kestrel host configuration |

Practical notes:

- Keep "open localhost in a browser" as the deployment story — remote/headless-host use
  keeps working, which a desktop framework (Avalonia/MAUI) would lose.
- Live preview streams as JPEG frames over the circuit or a dedicated endpoint; do not
  round-trip large frames through component parameters.
- The webcam layout needs `getUserMedia` via JS interop plus a frame upload channel —
  this is why it is deferred to Phase 8.
- Blazor Server is **not** NativeAOT-friendly; see §8.

The 4 layouts are not equal in value. `default` is the product; `benchmark` and `jobs`
duplicate CLI functionality and can be dropped or deferred without user harm.

---

## 7. Proving parity

The port is only credible if it is measured, so the harness is a first-class deliverable.

1. **Fixture corpus** — a fixed set of source/target images and short videos, plus the
   pinned model set. Fetched via the existing download mechanism, not stored in git.
2. **`.npy` as the interchange format.** Instrument the Python pipeline to dump
   intermediate arrays (detection boxes, landmarks, warp matrices, masks, model I/O) to
   `.npy`, and write a ~100-line `NpyReader` in the test project. The `.npy` format is
   trivial to parse and this gives tensor-level diffing without any Python interop in the
   .NET build.
3. **Tensor-level checks** — assert `max |dotnet - python| < ε` with a per-stage ε
   recorded in the test. Where ORT is doing the arithmetic, ε should be ~0 (identical
   kernels); divergence there means a preprocessing bug, which is exactly what you want
   the test to catch.
4. **Image-level checks** — SSIM/PSNR thresholds on final frames rather than exact
   equality, since encoders differ.
5. **CLI-level checks** — exit codes, job JSON contents, and produced file hashes for the
   deterministic paths.
6. **Allocation checks** — per §5.
7. **Benchmark tracking** — port `benchmarker.py` early enough to answer "is .NET
   actually faster?" with numbers rather than assertion.

Honest expectation on performance: **the ONNX inference itself will not get faster** — it
is the same runtime, same kernels, called through a thinner binding. Realistic wins are
in frame I/O, the extract → process → merge orchestration, startup time (no Python import
cost, currently seconds), and eliminating the GIL from the frame fan-out. For a face-swap
on 1080p video, expect single-digit to ~30% wall-clock improvement, dominated by whether
the pipeline is GPU- or IO-bound — not an order of magnitude. The bigger practical win is
distribution: a self-contained publish instead of a conda environment.

---

## 8. Distribution

| Artifact | Approach |
| --- | --- |
| CLI | **Self-contained publish** per RID (`linux-x64`, `win-x64`, `osx-arm64`) with **ReadyToRun** for startup and **`TieredPGO`** for steady-state throughput. ORT and OpenCvSharp natives come from NuGet and land in the publish directory — no source builds, no conda. |
| CLI | **Not NativeAOT** — see §5c. It trades a small steady-state throughput loss for a startup gain that does not matter on a multi-minute render, while constraining DI and serialisation for no benefit. |
| UI | Framework-dependent or self-contained (Blazor Server relies on reflection). |
| Models | Unchanged — same `.assets/models` layout, same download/hash mechanism. |

Still required on the host: `ffmpeg`, `ffprobe`, `curl` (as today), plus vendor GPU
drivers/toolkits for the chosen EP (as today).

---

## 9. Decisions to make before Phase 0

1. **OpenCvSharp4 vs. Emgu.CV vs. a managed imaging stack.** Recommendation:
   **OpenCvSharp4**, on two technical grounds. Against a managed stack: parity is the
   dominant risk, and matching cv2's `WarpAffine` and `INTER_AREA` semantics by hand is
   weeks of work with no user-visible benefit. Against Emgu.CV: OpenCvSharp is the
   thinner binding, and this pipeline makes thousands of small native calls per frame, so
   per-call managed overhead is the axis that matters. Neither library's CUDA modules are
   relevant here — the Python original uses `opencv-python-headless`, which has no CUDA
   support, so every cv2 operation being ported is already a CPU operation.
2. **Scope of v1** — CLI-only parity, or CLI + UI? Recommendation: **CLI-only v1**. The
   Python Gradio UI keeps working against the Python core during the transition.
3. **Compatibility contract** — must .NET read/write the same `.jobs` JSON,
   `facefusion.ini`, and `.assets/models` layout as Python? Recommendation: **yes,
   non-negotiable.** It allows side-by-side operation and makes the parity harness
   trivial.
4. **`Settings` mutability.** Recommendation: immutable record + `with` expressions.
   `job_runner` mutates state per step in Python; in .NET each step should build a fresh
   `Settings` instead. Cleaner, and removes a whole class of cross-thread bugs.
5. **Third-party processor extensibility.** Python's `importlib` scan lets users drop in
   a processor. Recommendation: drop for v1; revisit via `AssemblyLoadContext` if anyone
   actually relies on it.
6. **Content analyser.** `core.py:common_pre_check()` hashes the source of
   `content_analyser.py` and refuses to run if it has been modified — the NSFW gate is
   deliberately tamper-evident. The .NET port must carry the equivalent safeguard
   (analyser present, models downloaded, and an integrity check over the analyser code
   path). Do not port the pipeline without it.

---

## 10. Port order

Bottom-up, because every layer above depends on the numeric primitives being right. Each
phase ends with a green test suite ported from the corresponding `tests/test_*.py`.

**Phase 0 — Foundations, tensor layer, parity harness** *(prerequisite for everything)*
- Solution skeleton, CI (build/analyzers/test on Linux + Windows + macOS).
- **`FaceFusion.Tensors`** (§4) with `.npy`-golden unit tests. `Interp` first.
- **Parity harness** (§7): Python-side `.npy` dumping, .NET-side `NpyReader`, comparison
  driver. *Build this first.* Without it, "did I port this correctly?" is unanswerable
  for the entire numeric core.
- Port `types.py`, `choices.py`, `common_helper.py`, `normalizer.py`, `sanitizer.py`,
  `hash_helper.py`, `time_helper.py`, `json.py`, `filesystem.py`.
- Direct test ports: `test_common_helper`, `test_normalizer`, `test_sanitizer`,
  `test_json`, `test_filesystem`, `test_time_helper`.

**Phase 1 — Config, settings, DI, logging, i18n**
- `config.py` + `facefusion.ini`, `state_manager.py` → immutable `Settings` + DI
  container, `logger.py` → `Microsoft.Extensions.Logging`, `locales.py`/`translator.py` →
  a resource table (273 lines of flat key→format-string; keep the same keys so log output
  is diffable against Python).
- Tests: `test_config`, `test_state_manager`, `test_translator`.

**Phase 2 — Media plumbing (no ML yet)**
- `ffmpeg_builder`, `ffprobe_builder`, `curl_builder` — pure functions; existing tests
  port nearly line for line and validate the test-porting approach early.
- `ffmpeg.py`, `ffprobe.py`, `download.py`, `temp_helper.py`, `vision.py`,
  `video_manager.py`, `audio.py`. Adopt the §5 memory rules here.
- Tests: `test_ffmpeg*`, `test_ffprobe*`, `test_curl_builder`, `test_download`,
  `test_vision`, `test_audio`, `test_temp_helper`, `test_video_manager`.
- **Milestone: the CLI can extract frames from a video, write them back, and merge with
  audio — no models involved.** Shippable and useful on its own.

**Phase 3 — Inference layer**
- `execution.py` (EP enumeration, provider option construction, `nvidia-smi` XML parse),
  `inference_manager.py` (session pool keyed by module+models+device+providers, including
  the CUDA arena-leak workaround), `model_helper.py`.
- Establish the `OrtValue` zero-copy calling convention (§5.3) here, once, and reuse it
  everywhere.
- Tests: `test_execution`, `test_inference_manager`.
- Lower-risk than on other runtimes thanks to first-party bindings, but still verify the
  exact option keys (`trt_engine_cache_path`, `cudnn_conv_algo_search`,
  `SpecializationStrategy`, …) round-trip correctly.

**Phase 4 — Face pipeline**
- `face_helper.py` (warp templates, `warp_face_by_*`, `paste_back`), `face_detector.py`
  (5 detector families: retinaface, scrfd, yolo_face, yunet, many), `face_landmarker.py`,
  `face_recognizer.py`, `face_classifier.py`, `face_selector.py`, `face_masker.py`
  (box/occlusion/region/area masks), `face_tracker.py`, `face_creator.py`,
  `face_store.py`, `content_analyser.py`.
- Tests: `test_face_detector`, `test_face_creator`, `test_face_tracker`, plus
  golden-tensor tests through the parity harness.
- **This is the phase where floating-point parity is won or lost.** Compare intermediate
  tensors — detection boxes, landmark coordinates, warp matrices — not just final pixels.

**Phase 5 — Processors**
- `IProcessor` + DI registry, then modules in order of payoff:
  `face_swapper` → `face_enhancer` → `frame_enhancer` → `face_debugger` →
  `expression_restorer` → `age_modifier` → `frame_colorizer` → `background_remover` →
  `deep_swapper` → `face_editor` → `lip_syncer`.
- `pixel_boost.py`, `live_portrait.py`, `voice_extractor.py` as shared support.
- `face_debugger` early is deliberate: it renders the pipeline's internal state and makes
  divergence from Python visible at a glance.
- `lip_syncer` last — it needs the audio mel-spectrogram path (§2, scipy row) and is the
  most numerically delicate.
- Tests: the `tests/test_cli_*.py` suite, which is already end-to-end and is the best
  parity oracle in the repo.

**Phase 6 — Workflows, jobs, CLI**
- `workflows/*`, `jobs/*` (JSON schema unchanged — .NET must read job files written by
  Python and vice versa), `program.py` → `System.CommandLine`, `core.py` routing,
  `args.py`, `program_helper.py`, `exit_helper.py`, `benchmarker.py`.
- Tests: `test_job_*`, `test_cli_job_*`, `test_cli_batch_runner`, `test_program_helper`.
- **Milestone: full CLI parity. `facefusion headless-run …` produces comparable output to
  `python facefusion.py headless-run …`.** Ship as v1.

**Phase 7 — UI** (§6, Blazor Server) and **Phase 8 — streaming/webcam** (`streamer.py`,
`camera_manager.py`, webcam layout) — both explicitly post-v1.

---

## 11. Explicitly out of scope for v1

- `conda.py` / `installer.py` / `install.py` — replaced by `dotnet publish` plus a
  documented FFmpeg install; a build target can fetch models.
- Dynamic third-party processor loading (§9.5).
- Webcam streaming — Phase 8.
- The `benchmark` and `jobs` UI layouts — CLI equivalents already exist.
- Non-English locales beyond the existing `en` table (`locales.py` ships `en` only; keep
  the key structure so translations remain addable).

---

## 12. Effort estimate

Rough, for one experienced C# developer with ML-adjacent familiarity, assuming the parity
harness is built first:

| Phase | Estimate |
| --- | --- |
| 0 Foundations + tensor layer + harness | 3–4 weeks |
| 1 Config/settings/DI/i18n | 1 week |
| 2 Media plumbing | 2 weeks |
| 3 Inference layer | ~1 week |
| 4 Face pipeline | 4–6 weeks — the hard part |
| 5 Processors (11 modules) | 7–9 weeks |
| 6 Workflows/jobs/CLI | 2–3 weeks |
| **CLI-parity v1 total** | **~5–6 months** |
| 7 UI (Blazor Server) | 4–8 weeks |
| 8 Streaming/webcam | 2–3 weeks |
| 9 Performance tuning (§5b) | 2–3 weeks, post-parity |

Phase 9 is deliberately sequenced *after* parity. IO binding, device-memory residency and
pinned transfers all change how tensors move through the pipeline, and doing that while
the numerics are still unverified means debugging two problems at once. Get it correct,
lock the goldens, then make it fast against a suite that can prove you did not break it.

The estimate is dominated by Phases 4–5, and those are dominated by **verification, not
typing**. Any schedule that assumes "it's just numpy → Span" will be wrong by a factor of
two.

Relative to the Rust alternative this plan replaces: Phase 0 is ~1 week longer (building
the tensor layer that `ndarray` would have given for free) and Phase 5 is ~1 week longer
for the same reason, while Phase 3 is shorter and carries far less tail risk, and Phase 7
is ~2 weeks cheaper. **Total time is roughly a wash; the variance is materially lower**,
because the largest unknown — execution-provider coverage — is answered by a supported
NuGet package rather than by a source build.

---

## 13. First concrete steps

1. Create the solution skeleton and CI.
2. Write `FaceFusion.Tensors.Interp` plus its `.npy` golden test — the single most-used
   primitive in the codebase, and a check that the harness approach works.
3. Port `ffmpeg_builder` / `ffprobe_builder` / `curl_builder` + their tests — pure
   functions, fast win, validates the test-porting approach.
4. Stand up the parity driver with a single fixture and one comparison (frame extraction).
5. Spike `Microsoft.ML.OnnxRuntime` against one real model (`yoloface_8n` from the
   detector set) on CPU and CUDA via the `OrtValue` zero-copy path, and confirm output
   tensors match the Python session bit-for-bit. Run this in week one — it is cheap, and
   it validates the calling convention every later phase depends on.
6. While in that spike, measure the same model with and without `OrtIoBinding` to get an
   early read on how much §5b's copy elimination is actually worth on this hardware. One
   afternoon's work, and it tells you whether Phase 9 deserves 3 weeks or 3 days.
