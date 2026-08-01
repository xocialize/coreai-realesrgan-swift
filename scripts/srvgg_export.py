# Stage 2 export: Real-ESRGAN SRVGGNetCompact (original PyTorch weights) ->
# CoreAI .aimodel, static square input, fp16 or fp32.
# Architecture transcribed from realesrgan/archs/srvgg_arch.py (BSD-3-Clause).
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "torch",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch import TorchConverter, get_decomp_table


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _act(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def load_model(pth: str, num_conv: int) -> SRVGGNetCompact:
    model = SRVGGNetCompact(num_conv=num_conv)
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    if "params" in sd:
        sd = sd["params"]
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--num-conv", type=int, default=32)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = load_model(args.pth, args.num_conv).to(dtype)

    x = torch.rand(args.batch, 3, args.size, args.size, dtype=dtype)
    with torch.autocast(device_type="cpu", dtype=dtype):
        ep = torch.export.export(model, args=(x,))
    ep = ep.run_decompositions(get_decomp_table())

    program = (
        TorchConverter()
        .add_exported_program(ep, input_names=["x"], output_names=["output"])
        .to_coreai()
    )
    program.optimize()

    stem = Path(args.pth).stem.replace("-", "_")
    b = f"_b{args.batch}" if args.batch > 1 else ""
    out = Path(args.out_dir) / f"{stem}_{args.dtype}_static{args.size}{b}.aimodel"
    if out.exists():
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    program.save_asset(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
