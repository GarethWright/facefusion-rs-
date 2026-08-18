# Parity harness

How the C# port is verified against the Python original. Implements §7 of
`docs/DOTNET_PORT_PLAN.md`.

The harness exists because "did I port this correctly?" is otherwise unanswerable for the
numeric core. It is deliberately built *before* the face pipeline (Phases 4–5), not after.

## Why `.npy`

Ground truth crosses the boundary as NumPy `.npy` files:

- the format carries dtype, shape and memory order explicitly, so nothing is inferred;
- it is simple enough to parse in ~200 lines, so the .NET side needs no Python, no
  interop, and no extra dependency;
- CI machines can run the comparison with only the committed fixtures.

## Layout

| Path | Role |
| --- | --- |
| `tools/parity/parity_dump.py` | Instruments the **Python** pipeline: `dump(name, array)` |
| `tools/parity/generate_fixtures.py` | Regenerates the reader's fixture corpus |
| `tests/FaceFusion.ParityTests/fixtures/` | 23 committed `.npy` files + `manifest.json` |
| `src/FaceFusion.Parity/NpyReader.cs` | Reads `.npy` into `NpyArray`, always in C order |
| `src/FaceFusion.Parity/TensorComparison.cs` | `numpy.allclose` semantics + failure diagnostics |
| `src/FaceFusion.Parity/ImageMetrics.cs` | PSNR / SSIM for frame-level comparison |

## Capturing ground truth from Python

`dump()` is a no-op unless `FACEFUSION_PARITY_DIR` is set, so the instrumentation can stay
in the Python source permanently without affecting normal runs.

Add a dump at the stage you are porting:

```python
from tools.parity.parity_dump import dump

dump('face_detector/bounding_boxes', bounding_boxes)
dump('face_detector/face_scores', face_scores)
```

Then run the Python pipeline with the directory set:

```
FACEFUSION_PARITY_DIR=.parity/run-001 python facefusion.py headless-run \
    --source-paths source.jpg --target-path target.mp4 --output-path out.mp4
```

Nested names become directories, so the example above writes
`.parity/run-001/face_detector/bounding_boxes.npy`.

## Comparing from C#

```csharp
var expected = NpyReader.Load(".parity/run-001/face_detector/bounding_boxes.npy");
var actual   = DetectFaces(frame);

var result = TensorComparison.Compare(actual, expected.AsDoubles());

Assert.True(result.Passed, result.Describe());
```

`Compare` follows `numpy.allclose`: an element passes when
`|a - e| <= atol + rtol * |e|`, with `rtol = 1e-5` and `atol = 1e-8` by default. Note that
the tolerance scales with the **expected** value, so the comparison is asymmetric — pass
the Python value as `expected`.

When it fails, `Describe()` reports the max absolute and relative difference, the flat
index where it occurred, both values at that index, and the mismatch count. That detail is
the difference between a five-minute fix and an afternoon.

## Choosing a tolerance

Record a per-stage tolerance in the test rather than reaching for a global default:

- **Where ONNX Runtime does the arithmetic, expect ~0 difference.** Both sides call the
  same kernels, so a divergence means a *preprocessing* bug — the wrong normalisation, the
  wrong channel order, the wrong resize interpolation. Setting a loose tolerance here
  hides exactly the bug the test exists to find.
- **Where OpenCV does the arithmetic, expect ~0 difference** for the same reason, provided
  the port really is calling the same function with the same flags.
- **Managed float math** (the `FaceFusion.Tensors.NumPy` layer) is where a real epsilon
  belongs, because operation order and intermediate precision can legitimately differ.
- **Final frames** compare with PSNR/SSIM rather than element equality, since encoders
  differ. Assert a threshold, not equality.

## Regenerating fixtures

```
python3 tools/parity/generate_fixtures.py
```

Rewrites `tests/FaceFusion.ParityTests/fixtures/` and its manifest. The fixtures are
committed deliberately: the reader's tests must run on CI without Python. Regenerate only
when adding a genuinely new format case, and check the diff — a silent fixture change
would weaken the tests it feeds.

## What is not built yet

- **No end-to-end pipeline comparison.** The harness compares tensors and images; nothing
  yet dumps from a real FaceFusion run, because the face pipeline is not ported (Phases
  4–5) and neither ffmpeg nor the models are available in the current dev environment.
- **No CLI-level checks** (plan §7.5) — exit codes, job JSON, output file hashes. These
  arrive with Phase 6.
- **No allocation regression test** (plan §7.6, §5a). Worth adding as soon as frames flow
  through real code.
