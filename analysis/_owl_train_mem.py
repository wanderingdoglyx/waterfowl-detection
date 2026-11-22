#!/usr/bin/env python
"""
How large a training batch does an OWL variant fit in?

RUNS INSIDE THE MegaDetector-Overhead .venv, like the project's other OWL
helpers. Builds the model with the same constructor kwargs training uses — which
for OWL-D freezes the DINOv3 backbone — then runs a real optimiser step at
increasing batch sizes and records peak memory, stopping at the first failure.

Weights are randomly initialised rather than loaded: this measures the memory a
step needs, which depends on shapes and on which parameters carry gradients, not
on the values in them. Data loading and augmentation are excluded, so the figures
are a floor — a real epoch also holds prefetched batches.

Emits one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-kwargs", default="{}")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    for var in ("MDO_REPO_DIR", "DINOV3_ROOT"):
        path = os.environ.get(var)
        if path and path not in sys.path:
            sys.path.insert(0, path)

    import animaloc.models as M

    kwargs = json.loads(args.model_kwargs or "{}")
    model = getattr(M, args.model_name)(**kwargs).cuda().train()

    trainable = [q for q in model.parameters() if q.requires_grad]
    total = sum(q.numel() for q in model.parameters())
    out = {
        "params_total_m": round(total / 1e6, 1),
        "params_trainable_m": round(sum(q.numel() for q in trainable) / 1e6, 1),
        "results": {},
    }
    if not trainable:
        print("__JSON__" + json.dumps({**out, "error": "no trainable parameters"}))
        return

    opt = torch.optim.Adam(trainable, lr=0.0)   # lr=0: exercise the step, don't move weights

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(bs, 3, args.img_size, args.img_size, device="cuda")
            # one warm-up step, then timed steps: the first allocates workspace
            opt.zero_grad(set_to_none=True)
            _reduce(model(x)).backward()
            opt.step()
            torch.cuda.synchronize()

            rates = []
            for _ in range(args.steps):
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                _reduce(model(x)).backward()
                opt.step()
                torch.cuda.synchronize()
                rates.append(bs / (time.perf_counter() - t0))
            out["results"][str(bs)] = {
                "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                "crops_per_second": round(statistics.mean(rates), 2),
                "crops_per_second_std": round(statistics.stdev(rates), 2) if len(rates) > 1 else 0.0,
                "ok": True,
            }
        except torch.cuda.OutOfMemoryError:
            out["results"][str(bs)] = {"ok": False, "error": "CUDA out of memory"}
            torch.cuda.empty_cache()
            break
        except Exception as e:                      # a shape the model refuses, say
            out["results"][str(bs)] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            torch.cuda.empty_cache()
            break
        finally:
            for name in ("x", "y", "loss"):
                if name in dir():
                    pass
            torch.cuda.empty_cache()

    print("__JSON__" + json.dumps(out))


def _reduce(node, acc=None):
    """One differentiable scalar from a nested model output."""
    top = acc is None
    acc = [] if top else acc
    if torch.is_tensor(node):
        if node.requires_grad and node.is_floating_point():
            acc.append(node.float().mean())
    elif isinstance(node, (list, tuple)):
        for item in node:
            _reduce(item, acc)
    elif isinstance(node, dict):
        for item in node.values():
            _reduce(item, acc)
    if top:
        if not acc:
            raise RuntimeError("model produced no differentiable output")
        return sum(acc)
    return None


if __name__ == "__main__":
    main()
