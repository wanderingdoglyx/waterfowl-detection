"""
Training-step throughput against batch size.

`analysis/bench_extra.py` measures *inference*. Training is a different workload:
it adds a backward pass and a parameter update, so far more arithmetic per image
and much larger fixed per-step costs to amortise. Whether batching helps, and how
far, has to be measured separately rather than inferred from the inference sweep.

What is timed, per step:

    forward  ->  scalar loss  ->  backward  ->  optimiser step  ->  zero_grad

**What is excluded, deliberately:** data loading and augmentation (the input is a
resident tensor, so the number is a GPU ceiling, not what a real epoch achieves),
and the detector's real loss — target assignment, IoU matching and NMS-free
label assignment cost real time and vary by framework. The proxy loss is the mean
of the model's raw outputs, which still backpropagates gradients to every
parameter. So these figures answer "how does the *compute* scale with batch
size", not "how long will an epoch take".

Only models whose module takes a plain image tensor are covered — the three
Ultralytics generations and YOLO-NAS. Faster R-CNN's forward requires annotated
`Instances` in training mode, and the OWL variants live in another interpreter;
both would need a different harness, and are reported as skipped rather than
silently omitted.

    python -m analysis.bench_train                       # defaults
    python -m analysis.bench_train --batch-sizes 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config
from data_prep.efficiency import _sync, _reset_peak, _peak_mb
from analysis import manifest as manifest_mod
from analysis import predict, registry

OUT_DIR = os.path.join(config.OUTPUT_DIR, "analysis", "bench_train")
TRAINABLE = ("yolov5", "yolo11", "yolo26", "yolonas")
SKIPPED = {
    "fasterrcnn": "detectron2 requires annotated Instances in training mode",
    "mdo_owl_c": "runs in the MegaDetector-Overhead interpreter",
    "mdo_owl_t": "runs in the MegaDetector-Overhead interpreter",
    "mdo_owl_d": "runs in the MegaDetector-Overhead interpreter",
}


def _module(key: str, ckpt: str):
    """The bare nn.Module for a checkpoint, ready to train."""
    if key in ("yolov5", "yolo11", "yolo26"):
        from ultralytics import YOLO
        return YOLO(ckpt).model
    if key == "yolonas":
        from super_gradients.training import models
        return models.get(config.YOLONAS_MODEL, num_classes=1, checkpoint_path=ckpt)
    raise ValueError(key)


def _scalar_loss(out):
    """Reduce whatever the model returns to one differentiable scalar.

    Detector heads return deeply nested structures whose leaves are a mix of
    tensors, plain ints and None — YOLO-NAS returns a tuple of tuples, and
    Ultralytics a dict of {boxes, scores, feats}. Non-differentiable leaves are
    skipped rather than raising, since they are ordinary parts of the output;
    only a complete absence of gradient-carrying output is an error.
    """
    import torch

    parts: list = []

    def collect(node):
        if torch.is_tensor(node):
            if node.requires_grad and node.is_floating_point():
                parts.append(node.float().mean())
        elif isinstance(node, (list, tuple)):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            for item in node.values():
                collect(item)
        # ints, None and other leaves carry no gradient — ignore them

    collect(out)
    if not parts:
        raise RuntimeError("model produced no differentiable output")
    return sum(parts)


def sweep(key: str, ckpt: str, batch_sizes: list[int], steps: int,
          repeats: int) -> dict:
    import torch

    model = _module(key, ckpt).cuda().train()
    # Checkpoints are saved for inference: Ultralytics restores every parameter
    # with requires_grad=False, so a backward pass would have nothing to compute.
    model.requires_grad_(True)
    n_grad = sum(p.requires_grad for p in model.parameters())
    if n_grad == 0:
        raise RuntimeError("no trainable parameters after enabling gradients")
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr=0: time the step, don't move weights
    out: dict = {}

    for bs in batch_sizes:
        x = torch.randn(bs, 3, config.CROP_SIZE, config.CROP_SIZE, device="cuda")
        try:
            # Warm up at this shape: cuDNN picks algorithms per shape, and the
            # first backward also allocates its workspace.
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                _scalar_loss(model(x)).backward()
                opt.step()
            _sync(); _reset_peak()

            rates = []
            for _ in range(repeats):
                _sync()
                t0 = time.perf_counter()
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    _scalar_loss(model(x)).backward()
                    opt.step()
                _sync()
                rates.append(bs * steps / (time.perf_counter() - t0))
            out[str(bs)] = {
                "crops_per_second": round(statistics.mean(rates), 1),
                "crops_per_second_std": round(statistics.stdev(rates), 1) if len(rates) > 1 else 0.0,
                "ms_per_step": round(1000.0 * bs / statistics.mean(rates), 2),
                "peak_gpu_mb": _peak_mb(),
            }
        except torch.cuda.OutOfMemoryError:
            out[str(bs)] = {"error": "CUDA out of memory"}
            torch.cuda.empty_cache()
            break
        finally:
            del x
            torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Training-step throughput vs batch size")
    ap.add_argument("--models", nargs="*", default=list(TRAINABLE))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=12, help="timed steps per pass")
    ap.add_argument("--repeats", type=int, default=3, help="passes per batch size")
    ap.add_argument("-m", "--manifest", default=os.path.join(ROOT, "models_v1.docx"))
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    runs = {registry.resolve(e.key).key: e.run_id
            for e in manifest_mod.load(a.manifest)}

    results = {}
    for key in a.models:
        if key in SKIPPED:
            results[key] = {"skipped": SKIPPED[key]}
            print(f"  {key}: skipped — {SKIPPED[key]}", flush=True)
            continue
        print(f"  {key} ...", flush=True)
        try:
            ckpt = predict.checkpoint_for(registry.resolve(key), runs[key])
            results[key] = {"batch_sweep": sweep(key, ckpt, a.batch_sizes,
                                                 a.steps, a.repeats)}
            print(f"  {key}: ok", flush=True)
        except Exception as e:
            results[key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {key}: error — {e}", flush=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(a.out, stamp)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bench_train.json")
    with open(path, "w") as f:
        json.dump({"generated": stamp, "steps": a.steps, "repeats": a.repeats,
                   "batch_sizes": a.batch_sizes, "crop_size": config.CROP_SIZE,
                   "measures": "forward + backward + optimiser step; excludes data "
                               "loading and the detector's real loss",
                   "results": results}, f, indent=2)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
