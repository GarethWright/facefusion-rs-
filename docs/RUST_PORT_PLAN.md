# FaceFusion → Rust Port Plan

Target of the port: FaceFusion 3.8.2 (this repository), ~18.7k lines of Python plus
~3.8k lines of tests.

This document is a plan, not an implementation. It states what gets ported, in what
order, onto which crates, how parity is proven, and which decisions need to be made
before code is written.

---

## 1. What the application actually is

Despite the "AI" framing, FaceFusion is mostly **plumbing around ONNX Runtime and
FFmpeg**. That is what makes it a realistic Rust port.

| Area | Python LOC | What it does |
| --- | ---: | --- |
| Core runtime (`facefusion/*.py`) | 8,364 | CLI, state, config, jobs, ffmpeg/ffprobe, vision, face pipeline, download, audio |
| Processors (`facefusion/processors/**`) | 5,875 | 11 processor modules + shared model registries |
| UI (`facefusion/uis/**`) | 4,444 | Gradio web UI: 50 components, 4 layouts |
| Tests (`tests/**`) | 3,821 | pytest, heavily CLI-integration-shaped |

Runtime dependencies split into three groups:

1. **Native tools shelled out to** — `ffmpeg`, `ffprobe`, `curl`, `nvidia-smi`.
   Invoked via `subprocess` with builder modules (`ffmpeg_builder.py`,
   `ffprobe_builder.py`, `curl_builder.py`). These port to Rust almost verbatim; the
   builders are pure string/list construction and already have unit tests.
2. **Numeric/vision libraries** — `onnxruntime`, `opencv-python-headless`, `numpy`,
   `scipy`, `onnx`. These are the substance of the port.
3. **UI** — `gradio` + `gradio-rangeslider`. No Rust equivalent exists; this is a
   rewrite, not a port (§6).

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
In Rust this becomes an explicit trait plus a static registry — the single largest
structural change in the port.

---

## 2. Crate mapping

| Python | Rust | Notes / risk |
| --- | --- | --- |
| `onnxruntime` | [`ort`](https://crates.io/crates/ort) v2 | Binds the same C++ ORT. Exposes CUDA, TensorRT, CoreML, DirectML, ROCm, OpenVINO, QNN EPs — a close match for `choices.execution_provider_set`. **Verify each EP's option keys** (`trt_engine_cache_path`, `cudnn_conv_algo_search`, `SpecializationStrategy`, …) map 1:1. |
| `numpy` | `ndarray` (+ `ndarray-stats`) | Direct analogue. `numpy.interp` (79 uses!) has no crate equivalent — write a small `interp()` helper first, it is used everywhere for range remapping. |
| `opencv-python-headless` | `opencv` crate (bindings) **recommended**, `image`/`imageproc`/`fast_image_resize` as the pure-Rust alternative | Only ~35 distinct cv2 functions are used. Bindings give **bit-exact parity** for `warpAffine`, `resize` (INTER_AREA/CUBIC), `GaussianBlur`, `estimateAffinePartial2D` — the ops that decide output quality. Pure Rust means chasing subtle interpolation differences through every golden test. Decision in §7. |
| `scipy.signal` | `rustfft` + hand-written STFT/ISTFT/`lfilter`/`resample`; `triang`/`hann` windows are 3-line functions | 6 call sites total (`audio.py`, `voice_extractor.py`). Contained but numerically fussy — `scipy.signal.stft` scaling and padding conventions must be reproduced exactly or lip-sync and voice extraction drift. |
| `scipy.spatial.transform.Rotation` | `nalgebra` | 1 call site (`live_portrait.py`, Euler xyz → matrix). Trivial. |
| `onnx` (graph surgery) | `prost` + ONNX protobuf schema, or `ort`'s initializer access | `model_helper.py` only reads `graph.initializer[-1]` to get a model's embedding matrix. A minimal protobuf decode of `TensorProto` is enough — do **not** pull in a full ONNX graph library. |
| `argparse` (`program.py`, 348 LOC) | `clap` v4 (derive) | ~15 subcommands, ~100 flags, defaults sourced from `facefusion.ini` via `config.py`. Layered defaults (ini → env → CLI) need `clap`'s `default_value` set programmatically after reading the ini. |
| `configparser` + `facefusion.ini` | `rust-ini` or `configparser` crate | Keep the same ini file and key names — user configs must not break. |
| `tqdm` | `indicatif` | Cosmetic; match the `ascii = ' ='` bar style and `set_postfix` fields so logs stay recognisable. |
| `logging` | `tracing` + `tracing-subscriber` | Map `log_level` choices (error/warn/info/debug) directly. |
| `ThreadPoolExecutor` | `rayon` for frame fan-out, `std::thread` + `crossbeam` channels for the streamer | `execution_thread_count` maps to a scoped `rayon` pool, not the global one. |
| `threading.Semaphore` (`thread_helper.py`) | `tokio::sync::Semaphore` (sync mode) or `std::sync::Condvar` wrapper | Guards non-thread-safe EPs (CoreML/DirectML). Keep the same conditional logic. |
| `subprocess` | `std::process::Command` / `tokio::process` | Progress parsing of ffmpeg stderr stays line-based. |
| `curl` subprocess + `curl_builder.py` | Keep shelling out to `curl` (v1), optionally `reqwest` later | Shelling out preserves proxy/mirror behaviour (`download_providers`, resume, `--retry`) with zero surprises. `reqwest` is a v2 nicety. |
| `gradio` | see §6 | Rewrite. |

Cross-cutting crates: `serde`/`serde_json` (job files, `json.py`), `thiserror`/`anyhow`,
`once_cell`/`std::sync::OnceLock` (replacing `@lru_cache` module-level caches), `blake3`
or the existing hash scheme in `hash_helper.py` (currently a CRC-style 8-hex digest —
**read it and reproduce exactly**, model `.hash` sidecar files depend on it).

---

## 3. Proposed workspace layout

A Cargo workspace, so the numeric core can be tested and reused without dragging in the
UI or the CLI.

```
facefusion-rs/
├── Cargo.toml                  # workspace
├── crates/
│   ├── ff-types/               # types.py, choices.py — enums, newtypes, state keys
│   ├── ff-core/                # state_manager, config, logger, translator/locales,
│   │                           #   filesystem, temp_helper, process_manager, hash_helper,
│   │                           #   time_helper, common_helper, normalizer, sanitizer
│   ├── ff-media/               # ffmpeg/ffprobe/curl builders + runners, vision.py,
│   │                           #   audio.py, video_manager, camera_manager, download.py
│   ├── ff-inference/           # inference_manager, execution, model_helper,
│   │                           #   model registry + download resolution
│   ├── ff-face/                # face_detector, _landmarker, _recognizer, _classifier,
│   │                           #   _selector, _masker, _tracker, _creator, _store,
│   │                           #   face_helper (warp templates), content_analyser
│   ├── ff-processors/          # Processor trait + the 11 modules + pixel_boost,
│   │                           #   live_portrait, voice_extractor
│   ├── ff-workflows/           # workflows/{core,to_image,to_video,image_to_*}
│   ├── ff-jobs/                # job_manager, job_runner, job_store, job_list, job_helper
│   ├── ff-cli/                 # clap program, core.py routing, benchmarker → binary
│   └── ff-ui/                  # web UI server (§6) → binary or feature of ff-cli
└── xtask/                      # parity harness, model fixture fetch, release packaging
```

Two hard rules for the layout:

- **`ff-face` and `ff-processors` never touch global state directly.** In Python,
  `state_manager.get_item('...')` is called from 200+ places. In Rust that becomes an
  explicit `&Settings` / per-processor config struct passed down. This is the change
  that makes the port testable and thread-safe; it is also the one that touches the most
  lines, so it must be decided up front, not retrofitted.
- **The processor interface is a trait**, e.g.

  ```rust
  pub trait Processor: Send + Sync {
      fn name(&self) -> &'static str;
      fn model_set(&self, scope: DownloadScope) -> &ModelSet;
      fn pre_check(&self, ctx: &Ctx) -> Result<()>;
      fn pre_process(&self, ctx: &Ctx, mode: ProcessMode) -> Result<()>;
      fn process_frame(&self, ctx: &Ctx, inputs: &ProcessorInputs) -> Result<ProcessorOutputs>;
      fn post_process(&self);
  }
  ```

  with an inventory-style registry (`linkme` or a plain `match` in a `registry.rs`)
  replacing `importlib` + directory scan. Dynamic loading of third-party processors is
  lost; that is an accepted trade (§8).

---

## 4. Port order

Bottom-up, because every layer above depends on the numeric primitives being right.
Each phase ends with a green test suite ported from the corresponding `tests/test_*.py`.

**Phase 0 — Foundations & parity harness (prerequisite for everything)**
- Workspace skeleton, CI (fmt/clippy/test on Linux + macOS + Windows).
- `xtask parity`: runs the Python and Rust implementations over the same fixtures and
  compares outputs — arrays with a tolerance, files by hash, CLI by exit code + stdout.
  *Build this first.* Without it, "did I port this correctly?" is unanswerable for the
  entire numeric core.
- Port `types.py`, `choices.py`, `common_helper.py`, `normalizer.py`, `sanitizer.py`,
  `hash_helper.py`, `time_helper.py`, `json.py`, `filesystem.py`.
- Direct test ports: `test_common_helper`, `test_normalizer`, `test_sanitizer`,
  `test_json`, `test_filesystem`, `test_time_helper`.

**Phase 1 — Config, state, logging, i18n**
- `config.py` + `facefusion.ini`, `state_manager.py` → typed `Settings` struct,
  `logger.py` → `tracing`, `locales.py`/`translator.py` → a compile-time string table
  (`locales.py` is 273 lines of flat key→format-string; keep the same keys so log output
  is diffable against Python).
- Tests: `test_config`, `test_state_manager`, `test_translator`.

**Phase 2 — Media plumbing (no ML yet)**
- `ffmpeg_builder`, `ffprobe_builder`, `curl_builder` — pure functions, existing tests
  port almost line for line and give early confidence.
- `ffmpeg.py`, `ffprobe.py`, `download.py`, `temp_helper.py`, `vision.py`,
  `video_manager.py`, `audio.py`.
- Tests: `test_ffmpeg*`, `test_ffprobe*`, `test_curl_builder`, `test_download`,
  `test_vision`, `test_audio`, `test_temp_helper`, `test_video_manager`.
- **Milestone: `facefusion-rs` can extract frames from a video, write them back, and
  merge with audio — no models involved.** This alone is a shippable, useful binary.

**Phase 3 — Inference layer**
- `execution.py` (EP enumeration, provider option construction, `nvidia-smi` XML parse),
  `inference_manager.py` (pool keyed by module+models+device+providers, with the
  cli/ui-context sharing and the CUDA arena-leak workaround), `model_helper.py`.
- Tests: `test_execution`, `test_inference_manager`.
- **Risk checkpoint:** confirm `ort` exposes every EP option FaceFusion sets. If an EP
  is unsupported, decide fallback vs. patch upstream *here*, not in Phase 5.

**Phase 4 — Face pipeline**
- `face_helper.py` (warp templates, `warp_face_by_*`, `paste_back`), `face_detector.py`
  (5 detector families: retinaface, scrfd, yolo_face, yunet, many), `face_landmarker.py`,
  `face_recognizer.py`, `face_classifier.py`, `face_selector.py`, `face_masker.py`
  (box/occlusion/region/area masks), `face_tracker.py`, `face_creator.py`,
  `face_store.py`, `content_analyser.py`.
- Tests: `test_face_detector`, `test_face_creator`, `test_face_tracker` + new
  golden-image tests through the parity harness.
- **This is the phase where floating-point parity is won or lost.** Compare
  intermediate tensors (detection boxes, landmark coordinates, warp matrices), not just
  final pixels.

**Phase 5 — Processors**
- Trait + registry, then modules in order of payoff:
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
- `workflows/*`, `jobs/*` (JSON schema unchanged — Rust must read job files written by
  Python and vice versa), `program.py` → clap, `core.py` routing, `args.py`,
  `program_helper.py`, `exit_helper.py`, `benchmarker.py`.
- Tests: `test_job_*`, `test_cli_job_*`, `test_cli_batch_runner`, `test_program_helper`.
- **Milestone: full CLI parity. `facefusion-rs headless-run …` produces byte-comparable
  output to `python facefusion.py headless-run …`.** Ship this as v1.

**Phase 7 — UI** (see §6) and **Phase 8 — streaming/webcam** (`streamer.py`,
`camera_manager.py`, `uis/components/webcam.py`) — both explicitly post-v1.

---

## 5. Proving parity

The port is only credible if it is measured, so the harness is a first-class deliverable:

1. **Fixture corpus** — a fixed set of source/target images and short videos, plus the
   pinned model set. Stored via the existing download mechanism, not in git.
2. **Tensor-level checks** — for detector/landmarker/masker outputs, assert
   `max |rust - python| < ε` with a per-stage ε recorded in the test. Where ORT is doing
   the arithmetic, ε should be ~0 (same kernels); divergence there means a preprocessing
   bug, which is exactly what you want the test to catch.
3. **Image-level checks** — SSIM/PSNR thresholds on final frames rather than exact
   equality, since interpolation and JPEG/PNG encoders differ.
4. **CLI-level checks** — exit codes, job JSON contents, and produced file hashes for the
   deterministic paths.
5. **Benchmark tracking** — `benchmarker.py` already measures fps; port it early enough
   to answer "is Rust actually faster?" with numbers rather than assertion.

Honest expectation on performance: **the ONNX inference itself will not get faster** —
it is the same runtime, same kernels. Realistic wins are in frame I/O, the
extract → process → merge orchestration, memory footprint, startup time (no Python
import cost, currently seconds), and eliminating the GIL from the frame fan-out. For a
face-swap on 1080p video, expect single-digit to ~30% wall-clock improvement, dominated
by whether the pipeline is GPU- or IO-bound — not an order of magnitude. Distribution is
the bigger win: a single static-ish binary instead of a conda environment.

---

## 6. The UI question (needs a decision — §7)

`facefusion/uis` is 4,444 lines of Gradio: 50 components, 4 layouts (default, benchmark,
jobs, webcam), with a global component registry (`UI_COMPONENTS`) and cross-component
event wiring. There is no Rust Gradio. Three viable directions:

| Option | Shape | Pros | Cons |
| --- | --- | --- | --- |
| **A. Axum + web frontend** (recommended) | Rust HTTP/WebSocket server, UI in HTML+HTMX or a Rust WASM framework (Leptos/Dioxus) | Keeps the "open localhost in a browser" workflow users know; server-side stays pure Rust; remote/headless-host use keeps working | Full UI rewrite; live preview needs a frame-streaming endpoint |
| **B. Tauri / egui desktop app** | Native window | Best responsiveness for preview and webcam; no port to bind | Loses remote access; another packaging story per OS |
| **C. Keep Python Gradio, drive Rust core** | Python UI calls a Rust binary/pyo3 module | Zero UI work; incremental | Keeps the Python dependency the port was meant to remove |

Option C is worth taking *during* the port regardless of the endpoint: exposing the Rust
core through `pyo3` lets the existing Gradio UI and the existing pytest suite exercise
Rust code long before Phase 7, which shortens the feedback loop dramatically.

---

## 7. Decisions to make before Phase 0

1. **OpenCV bindings vs. pure Rust imaging.** Recommendation: **use the `opencv` crate**
   for v1. Parity is the dominant risk in this port, and matching cv2's `warpAffine`
   and `INTER_AREA` semantics by hand is weeks of work with no user-visible benefit. It
   costs the "no native deps" story — but ONNX Runtime and FFmpeg are native deps
   already, so that story was never available.
2. **UI direction** — A, B, or C above. Affects Phase 7 sizing only, but affects
   `ff-core`'s state design immediately (the cli/ui dual-context in `state_manager.py`
   and `app_context.py` exists purely for Gradio's threading model and can be dropped
   under A or B).
3. **Scope of v1** — CLI-only parity, or CLI + UI? Recommendation: CLI-only v1, with the
   pyo3 bridge keeping the Python UI usable in the interim.
4. **Compatibility contract** — must Rust read/write the same `.jobs` JSON, `facefusion.ini`,
   and `.assets/models` layout as Python? Recommendation: **yes**, non-negotiable. It
   allows side-by-side operation and makes the parity harness trivial.
5. **Third-party processor extensibility** — Python's `importlib` scan lets users drop in
   a processor. Rust loses this unless a plugin ABI (`abi_stable`/`dylib`) or a WASM host
   is added. Recommendation: drop for v1, revisit if anyone actually relies on it.
6. **Content analyser.** `core.py:common_pre_check()` hashes the source of
   `content_analyser.py` and refuses to run if it has been modified — the NSFW gate is
   deliberately tamper-evident. The Rust port must carry the equivalent safeguard
   (analyser present, models downloaded, and an integrity check over the analyser code
   path). Do not port the pipeline without it.

---

## 8. Explicitly out of scope for v1

- `conda.py` / `installer.py` / `install.py` — replaced by `cargo build` plus a documented
  ONNX Runtime + FFmpeg install; a `cargo xtask setup` can fetch models.
- Dynamic third-party processor loading (§7.5).
- Webcam streaming (`streamer.py`, `camera_manager.py`) — Phase 8.
- The benchmark and jobs UI layouts — CLI equivalents already exist.
- Non-English locales beyond the existing `en` table (`locales.py` currently ships `en`
  only; keep the key structure so translations remain addable).

---

## 9. Effort estimate

Rough, for one experienced Rust developer with ML-adjacent familiarity, assuming the
parity harness is built first:

| Phase | Estimate |
| --- | --- |
| 0 Foundations + harness | 2–3 weeks |
| 1 Config/state/i18n | 1 week |
| 2 Media plumbing | 2 weeks |
| 3 Inference layer | 1–2 weeks (+ unknown if an EP is missing from `ort`) |
| 4 Face pipeline | 4–6 weeks — the hard part |
| 5 Processors (11 modules) | 6–8 weeks |
| 6 Workflows/jobs/CLI | 2–3 weeks |
| **CLI-parity v1 total** | **~4.5–6 months** |
| 7 UI (option A) | 6–10 weeks |
| 8 Streaming/webcam | 2–3 weeks |

The estimate is dominated by Phases 4–5, and those are dominated by verification, not by
typing. Any schedule that assumes "it's just numpy → ndarray" will be wrong by a factor
of two.

---

## 10. First concrete steps

1. Create the workspace skeleton and CI.
2. Port `ffmpeg_builder` / `ffprobe_builder` / `curl_builder` + their tests — pure
   functions, fast win, validates the test-porting approach.
3. Stand up `xtask parity` with a single fixture and one comparison (frame extraction).
4. Spike `ort` against one real model (`yoloface_8n` from the detector set) on CPU and
   CUDA, and confirm output tensors match the Python session bit-for-bit. **If this spike
   fails, the whole plan needs revisiting** — so run it in week one, before the
   foundations work is finished.
