#!/usr/bin/env python
"""
Batch-size and full-frame timing for one OWL variant.

RUNS INSIDE THE MegaDetector-Overhead .venv (Python 3.11), like the project's
other OWL helpers — animaloc and its torch build do not exist in the `waterfowl`
environment. Invoked as a subprocess by analysis/bench_extra.py.

Emits one JSON object on stdout so the caller need not parse logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pth", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-kwargs", default="{}")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--repeats", type=int, default=3,
                   help="timed passes per batch size; >1 gives a spread")
    p.add_argument("--frame", default=None, help="full UAS image for end-to-end timing")
    p.add_argument("--stride", type=int, default=None)
    args = p.parse_args()

    repo = os.environ.get("MDO_REPO_DIR")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    dino = os.environ.get("DINOV3_ROOT")
    if dino and dino not in sys.path:
        sys.path.insert(0, dino)

    from animaloc.models import HerdNet  # noqa: F401  (registers architectures)
    import animaloc.models as M

    kwargs = json.loads(args.model_kwargs or "{}")
    model = getattr(M, args.model_name)(**kwargs)
    state = torch.load(args.pth, map_location="cpu")
    state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def to_tensor(path_or_arr):
        if isinstance(path_or_arr, str):
            im = Image.open(path_or_arr).convert("RGB").resize((args.img_size, args.img_size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        else:
            arr = path_or_arr.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    out = {"batch_sweep": {}, "end_to_end": {}}

    # ── Batch-size sensitivity ───────────────────────────────────────────────
    tensors = [to_tensor(p_) for p_ in args.images]
    with torch.no_grad():
        for bs in args.batch_sizes:
            usable = (len(tensors) // bs) * bs
            if usable == 0:
                continue
            batch = torch.stack(tensors[:bs]).cuda()
            batch = (batch - mean) / std
            model(batch)  # warm-up at this shape; cuDNN retunes per shape
            sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            rates = []
            for _ in range(args.repeats):
                sync()
                t0 = time.perf_counter()
                for i in range(0, usable, bs):
                    b = torch.stack(tensors[i:i + bs]).cuda()
                    b = (b - mean) / std
                    model(b)
                sync()
                rates.append(usable / (time.perf_counter() - t0))

            mean_rate = sum(rates) / len(rates)
            var = sum((r - mean_rate) ** 2 for r in rates) / (len(rates) - 1) if len(rates) > 1 else 0.0
            peak = (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                    if torch.cuda.is_available() else None)
            out["batch_sweep"][str(bs)] = {
                "ms_per_crop": round(1000.0 / mean_rate, 3),
                "crops_per_second": round(mean_rate, 1),
                "crops_per_second_std": round(var ** 0.5, 1),
                "peak_gpu_mb": peak,
            }

    # ── End-to-end full frame ────────────────────────────────────────────────
    if args.frame and os.path.exists(args.frame):
        img = Image.open(args.frame).convert("RGB")
        W, H = img.size
        stride = args.stride or args.img_size
        sync()
        t0 = time.perf_counter()
        tiles = 0
        with torch.no_grad():
            for y in range(0, max(H - args.img_size, 0) + 1, stride):
                for x in range(0, max(W - args.img_size, 0) + 1, stride):
                    tile = np.asarray(img.crop((x, y, x + args.img_size, y + args.img_size)))
                    t = to_tensor(tile).unsqueeze(0).cuda()
                    t = (t - mean) / std
                    model(t)
                    tiles += 1
        sync()
        dt = time.perf_counter() - t0
        out["end_to_end"] = {
            "image": os.path.basename(args.frame), "resolution": f"{W}x{H}",
            "crops": tiles, "seconds": round(dt, 2),
            "seconds_per_megapixel": round(dt / (W * H / 1e6), 3),
        }

    print("__JSON__" + json.dumps(out))


if __name__ == "__main__":
    main()
