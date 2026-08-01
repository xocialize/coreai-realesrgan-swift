# Stage 2 parity + latency: SRVGGNetCompact CoreAI fp16 (ANE/GPU) vs
# PyTorch fp32 reference on real corpus tiles.
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "torch",
#     "numpy",
#     "pillow",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_OS_COREAI", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def center_crop(img: Image.Image, size: int) -> np.ndarray:
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    img = img.crop((left, top, left + size, top + size))
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)


def save_img(x, path):
    arr = np.clip(x[0].transpose(1, 2, 0), 0, 1)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--num-conv", type=int, default=32)
    ap.add_argument("--unit", choices=["ane", "gpu", "cpu", "default"], default="ane")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--tag", default="srvgg")
    args = ap.parse_args()

    from coreai.runtime import AIModel, NDArray
    from coreai.runtime._specialization_options import SpecializationOptions

    opts = None
    if args.unit != "default":
        from coreai.runtime import ComputeUnitKind
        kind = {"ane": ComputeUnitKind.neural_engine,
                "gpu": ComputeUnitKind.gpu,
                "cpu": ComputeUnitKind.cpu}[args.unit]()
        opts = SpecializationOptions.from_preferred_compute_unit_kind(kind)

    t0 = time.perf_counter()
    model = await AIModel.load(args.model, opts)
    load_s = time.perf_counter() - t0
    fn = model.load_function("main")

    # torch fp32 reference
    ref_model = SRVGGNetCompact(num_conv=args.num_conv)
    sd = torch.load(args.pth, map_location="cpu", weights_only=True)
    if "params" in sd:
        sd = sd["params"]
    ref_model.load_state_dict(sd, strict=True)
    ref_model.eval()

    per_image = []
    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(args.images):
        x = center_crop(Image.open(img_path).convert("RGB"), args.size)
        out = await fn({"x": NDArray(np.ascontiguousarray(x).astype(np.float16))})
        y = out[next(iter(out.keys()))].numpy().astype(np.float32)
        with torch.no_grad():
            ref = ref_model(torch.from_numpy(x)).numpy()
        p = psnr(np.clip(y, 0, 1), np.clip(ref, 0, 1))
        per_image.append({"image": Path(img_path).name, "psnr_db": round(p, 2)})
        if dump and i < 3:
            stem = f"{args.tag}_{Path(img_path).stem}"
            save_img(x, dump / f"{stem}_input.png")
            save_img(y, dump / f"{stem}_coreai_{args.unit}.png")
            save_img(ref, dump / f"{stem}_torch_fp32.png")

    # latency on the last tile
    x16 = np.ascontiguousarray(x).astype(np.float16)
    inp = {"x": NDArray(x16)}
    for _ in range(args.warmup):
        await fn(inp)
    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        await fn(inp)
        lat.append(time.perf_counter() - t0)
    lat_ms = np.array(lat) * 1e3

    psnrs = [d["psnr_db"] for d in per_image]
    print(json.dumps({
        "tag": args.tag,
        "model": args.model,
        "unit": args.unit,
        "size": args.size,
        "load_s": round(load_s, 3),
        "psnr_min": min(psnrs),
        "psnr_mean": round(float(np.mean(psnrs)), 2),
        "per_image": per_image,
        "lat_ms_median": round(float(np.median(lat_ms)), 3),
        "lat_ms_p90": round(float(np.percentile(lat_ms, 90)), 3),
        "iters": args.iters,
    }, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
