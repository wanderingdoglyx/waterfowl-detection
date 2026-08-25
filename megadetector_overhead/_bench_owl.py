#!/usr/bin/env python
"""
Efficiency benchmark for one OWL variant — runs INSIDE the MegaDetector-Overhead venv.

The animaloc stack cannot coexist with the detectron2/super-gradients/ultralytics
environment, so data_prep/efficiency.py shells into here exactly as evaluation shells into
_eval_owl.py.  Emits a single machine-readable line:

    BENCH_JSON {"ms_per_crop": [...], "peak_gpu_mb": ..., "params": ...}

Timing mirrors the box-model path so the numbers are comparable: warm-up discarded,
`--repeats` passes over the same crops, torch.cuda.synchronize() around every timed
region, peak allocation reset before the measured passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.environ.get("MDO_REPO_DIR")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from PIL import Image

import animaloc.models as animaloc_models
from animaloc.eval.lmds import HerdNet_Detection_Branch_LMDS
from animaloc.models.utils import LossWrapper, load_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pth", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-kwargs", default="{}")
    p.add_argument("--down-ratio", type=int, default=2)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--whole-process", action="store_true",
                   help="time the COMPLETE path — disk read, preprocess, forward and LMDS "
                        "point decoding — instead of the forward pass alone")
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--adapt-ts", type=float, default=0.3)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = animaloc_models.__dict__.get(a.model_name)
    if cls is None:
        print(f"BENCH_JSON {json.dumps({'error': f'model {a.model_name} not found'})}")
        return
    model = LossWrapper(cls(**json.loads(a.model_kwargs)), [])
    model = load_model(model, a.pth).to(device).eval()
    params = int(sum(x.numel() for x in model.parameters()))

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def load(path):
        im = Image.open(path).convert("RGB").resize((a.img_size, a.img_size))
        t = torch.from_numpy(np.asarray(im)).to(device).permute(2, 0, 1).float().div_(255)
        return (t.unsqueeze(0) - mean) / std

    lmds = HerdNet_Detection_Branch_LMDS(up=False, kernel_size=(a.kernel, a.kernel),
                                         adapt_ts=a.adapt_ts)

    def forward_only(paths, tensors):
        for t in tensors:
            model(t)

    def first_tensor(x):
        """Unwrap to the heatmap tensor.  OWL-C/T return it one level deep, OWL-D two —
        its DPT head wraps the output again — so unwrap until a tensor appears rather
        than assuming a fixed depth."""
        while isinstance(x, (list, tuple)):
            if not x:
                return None
            x = x[0]
        return x

    def whole_process(paths, _tensors):
        # Disk read -> preprocess -> forward -> LMDS point decoding, per crop, so the
        # number is comparable with a box model's load-to-final-detections figure.
        for path in paths:
            t = load(path)
            heat = first_tensor(model(t))
            if heat is not None:
                lmds(heat)

    if a.whole_process:
        tensors = []                      # loaded inside the timed region
        run, note = whole_process, "whole process: disk read, preprocess, forward, LMDS decode"
    else:
        tensors = [load(x) for x in a.images]
        run, note = forward_only, "forward pass only; LMDS point decoding excluded"

    with torch.no_grad():
        run(a.images[:a.warmup], tensors[:a.warmup])
        sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        per_pass = []
        for _ in range(a.repeats):
            sync(); t0 = time.perf_counter()
            run(a.images, tensors)
            sync()
            per_pass.append((time.perf_counter() - t0) / len(a.images) * 1000)

    out = {"ms_per_crop": per_pass, "params": params,
           "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                           if torch.cuda.is_available() else None),
           "note": note}
    print("BENCH_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
