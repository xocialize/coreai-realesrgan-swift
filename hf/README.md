---
license: bsd-3-clause
tags:
- coreai
- neural-engine
- super-resolution
- image-to-image
- apple-silicon
- real-esrgan
library_name: coreai
---

# Real-ESRGAN-CoreAI

Real-ESRGAN (SRVGGNetCompact) 4× super-resolution as **CoreAI `.aimodel` assets** for the Apple
Neural Engine — fp16, static 128×128 input, exported from the original
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) checkpoints (BSD-3-Clause).

To our knowledge the first super-resolution model in `coreai-community`.

| asset | source checkpoint | num_conv | size |
|---|---|---|---|
| `realesr_general_x4v3_float16_static128.aimodel` | realesr-general-x4v3 | 32 | 4.6 MB |
| `realesr_general_wdn_x4v3_float16_static128.aimodel` | realesr-general-wdn-x4v3 (denoising) | 32 | 4.6 MB |
| `realesr_animevideov3_float16_static128.aimodel` | realesr-animevideov3 | 16 | 2.3 MB |

## Numbers (measured, M5 Max, macOS 27)

**Parity** — fp16 on ANE vs fp32 PyTorch, 7 real 128² tiles per variant:
general **min 68.56 / mean 69.36 dB** · general-wdn **min 58.15 / mean 65.09 dB** ·
anime **min 64.11 / mean 69.43 dB**. Full tiled pipeline vs an independent fp32 oracle:
**66.85 dB, max |Δ| = 1 LSB**.

**Why the ANE:** wall-clock ties a well-tuned GPU path at this tile size while drawing
**≈4.5–4.9× less energy per frame** (~17 W vs ~83 W over idle), with no thermal throttling.

## Usage

Static shape: input `x` = `[1, 3, 128, 128]` fp16 NCHW in `[0,1]`; output `[1, 3, 512, 512]`.
Tile larger images (overlap 8 recommended, feathered blend). A ready-made Swift package that does
exactly this — tiling, compositing, variant selection, tests — is
[`xocialize/coreai-realesrgan-swift`](https://github.com/xocialize/coreai-realesrgan-swift).

```swift
import CoreAI
let model = try await AIModel(contentsOf: url,
    options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
let fn = try model.loadFunction(named: "main")!
let out = try await fn.run(inputs: ["x": input])  // first load pays ~8 s E5RT specialization, OS-cached
```

## Reproducibility

`srvgg_export.py` (in this repo) re-creates every asset from the original `.pth` checkpoints:
PyTorch → `torch.export` → `coreai-torch` `TorchConverter` → `.aimodel`. No opaque binaries.

## Provenance & license

Architecture and weights: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
**BSD-3-Clause** (code and released checkpoints). This repo redistributes the same weights in a
converted container under the same license, with the conversion script included.
