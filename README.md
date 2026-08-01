# coreai-realesrgan-swift

Real-ESRGAN (SRVGGNetCompact) 4× super-resolution on the **Apple Neural Engine**, via Apple's
CoreAI framework. The first `coreai-*` package — sibling to
[`mlx-realesrgan-swift`](https://github.com/xocialize/mlx-realesrgan-swift): same capability, same
three checkpoints, different backend.

## Why an ANE backend exists (measured, not assumed)

On an M5 Max at the shared t128 tile geometry, ANE wall-clock **ties** the best MLX-GPU
configuration while drawing **≈4.5–4.9× less energy per frame** (~17 W vs ~83 W over idle, ~14.5 J
vs ~67 J/frame) — and the ANE does not thermally throttle where the GPU sags 1616→1366 MHz. For
long batch or video renders on a laptop, power *is* the product axis.

## Requirements

- **macOS 27+** (CoreAI.framework). Below 27, use the MLX sibling — that split is the intended
  consumption pattern, not a limitation to engineer around.
- Apple silicon. `Compute.gpu` is a legitimate second mode (CoreAI static executables measured
  2.2–2.4× faster than MLX dynamic GPU); `.cpu` is for testing.

## Usage

```swift
import RealESRGANCoreAI

let model = SRVGGNetCompact_CoreAI(variant: .general)   // .generalWDN / .anime
let upscaled = try await model.upscale(cgImage)         // 4×, any input size, tiled 128/8
```

⚠️ First use per variant per machine pays one E5RT specialization (~8 s, OS-cached afterwards).
Treat the first call as preparation; never profile it.

## Fixed geometry — deliberately

Tiles are **128², overlap 8, baked at export**. The ANE compiles one executable per static shape,
so tile size is a build-time property of the `.aimodel` asset, not a runtime knob — the deliberate
inversion of the MLX sibling's injectable geometry. 128 is the measured optimum on *both* backends.
Compositing is the normalized feathered blend (Σwᵢxᵢ/Σwᵢ in float32, quantized once).

## Parity (fp16 ANE vs fp32 PyTorch, per variant — none assumed)

7 real 128² tiles (5 signage stills + 2 video-master frames):

| variant | min | mean |
|---|---|---|
| `general` | 68.56 dB | 69.36 dB |
| `generalWDN` | 58.15 dB | 65.09 dB |
| `anime` | 64.11 dB | 69.43 dB |

Full pipeline (ANE + Swift compositor vs a torch-fp32 + numpy oracle with identical grid/feather,
300×200→1200×800): **66.85 dB, max |Δ| = 1 LSB, 1.3 % of pixels differ at all.**

## Provenance

Weights are the **original** released checkpoints from
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) —
`realesr-general-x4v3`, `realesr-general-wdn-x4v3`, `realesr-animevideov3` — exported
PyTorch → `coreai-torch` → `.aimodel` by `scripts/srvgg_export.py` in this repo. The export is
reproducible; the vendored assets (2.3–4.6 MB each) are not binaries of unknown origin.
`scripts/run_srvgg.py` reproduces the parity table.
