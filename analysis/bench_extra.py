"""
Task 2.1, "recommended where practical" — the parts the existing benchmark
leaves to Ultralytics only.

``data_prep.efficiency`` measures every required metric for all eight models,
but its batch-size sweep and full-frame timing bail out for anything that is not
an Ultralytics model. This module covers the other five (Faster R-CNN, YOLO-NAS
and the three OWL variants) so batch sensitivity and end-to-end cost can be
compared across the whole benchmark.

Results are written beside the analysis, leaving data_prep/efficiency.py — the
pipeline's own measurement code — untouched.

    python -m analysis.bench_extra                  # all models
    python -m analysis.bench_extra --models owl_c   # one
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config
from data_prep.efficiency import (BOX_MODELS, OWL_MODELS, sample_crops,
                                  _sync, _reset_peak, _peak_mb)
from analysis import manifest as manifest_mod
from analysis import predict, registry

ALL_MODELS = tuple(BOX_MODELS) + tuple(OWL_MODELS)

# bench_extra keys -> registry keys
KEY_TO_REGISTRY = {"owl_c": "mdo_owl_c", "owl_t": "mdo_owl_t", "owl_d": "mdo_owl_d"}

OUT_DIR = os.path.join(config.OUTPUT_DIR, "analysis", "bench_extra")
BATCH_SIZES = (1, 2, 4, 8, 16)


def _representative_frame() -> str | None:
    """A real UAS frame to tile, for the end-to-end number."""
    for d in sorted(glob.glob(os.path.join(config.DATASET_ROOT, "Bird_*"))):
        hits = sorted(glob.glob(os.path.join(d, "*.JPG")) + glob.glob(os.path.join(d, "*.jpg")))
        if hits:
            return hits[0]
    return None


def _stride() -> int:
    return int(config.CROP_SIZE * (1 - config.CROP_OVERLAP / 2))


# ── Faster R-CNN (detectron2) ────────────────────────────────────────────────

def _fasterrcnn_runner(ckpt: str):
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from faster_rcnn.model import build_cfg

    cfg = build_cfg(use_pretrained=False)
    cfg.MODEL.WEIGHTS = ckpt
    model = build_model(cfg)
    DetectionCheckpointer(model).load(ckpt)
    model.eval()

    def run(images: list[np.ndarray]) -> None:
        # detectron2 takes a list of per-image dicts, which is its batch API.
        inputs = []
        for im in images:
            t = torch.as_tensor(im.transpose(2, 0, 1).astype("float32"))
            inputs.append({"image": t, "height": im.shape[0], "width": im.shape[1]})
        with torch.no_grad():
            model(inputs)

    return run


# ── YOLO-NAS (super-gradients) ───────────────────────────────────────────────

def _yolonas_runner(ckpt: str):
    from super_gradients.training import models

    model = models.get(config.YOLONAS_MODEL, num_classes=1, checkpoint_path=ckpt)
    model = model.cuda().eval()

    def run(images: list[np.ndarray]) -> None:
        model.predict(images, conf=config.CONF_THRESHOLD,
                      iou=config.YOLONAS_NMS_IOU,
                      max_predictions=config.YOLONAS_MAX_PRED, fuse_model=False)

    return run


# ── Ultralytics ──────────────────────────────────────────────────────────────

def _ultralytics_runner(ckpt: str):
    from ultralytics import YOLO

    model = YOLO(ckpt)

    def run(images: list[np.ndarray]) -> None:
        model.predict(images, imgsz=config.CROP_SIZE, conf=config.CONF_THRESHOLD,
                      max_det=config.MAX_DETECTIONS_PER_IMAGE, verbose=False)

    return run


def _sweep(runner, images: list[np.ndarray], repeats: int = 3) -> dict:
    """Time each batch size over the same images, warming up per shape.

    Repeated passes are not optional here. A single pass yields a plausible
    number whether or not another process is competing for the GPU, and the
    differences between adjacent batch sizes are often only a few percent — the
    same order as run-to-run variation. Without a spread there is no way to tell
    a real turn in the curve from noise, so mean and standard deviation are both
    reported and callers should compare them before believing a small gap.
    """
    out = {}
    for bs in BATCH_SIZES:
        usable = (len(images) // bs) * bs
        if usable == 0:
            continue
        runner(images[:bs])          # cuDNN retunes per input shape
        _sync(); _reset_peak()
        rates = []
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            for i in range(0, usable, bs):
                runner(images[i:i + bs])
            _sync()
            rates.append(usable / (time.perf_counter() - t0))
        mean = statistics.mean(rates)
        out[str(bs)] = {
            "ms_per_crop": round(1000.0 / mean, 3),
            "crops_per_second": round(mean, 1),
            "crops_per_second_std": round(statistics.stdev(rates), 1) if len(rates) > 1 else 0.0,
            "crops_per_second_runs": [round(r, 1) for r in rates],
            "peak_gpu_mb": _peak_mb(),
        }
    return out


def _end_to_end(runner, frame: str) -> dict:
    from PIL import Image

    img = Image.open(frame).convert("RGB")
    W, H = img.size
    stride = _stride()
    _sync()
    t0 = time.perf_counter()
    tiles = 0
    for y in range(0, max(H - config.CROP_SIZE, 0) + 1, stride):
        for x in range(0, max(W - config.CROP_SIZE, 0) + 1, stride):
            tile = np.asarray(img.crop((x, y, x + config.CROP_SIZE, y + config.CROP_SIZE)))
            runner([tile[:, :, ::-1].copy()])
            tiles += 1
    _sync()
    dt = time.perf_counter() - t0
    return {"image": os.path.basename(frame), "resolution": f"{W}x{H}",
            "crops": tiles, "seconds": round(dt, 2),
            "seconds_per_megapixel": round(dt / (W * H / 1e6), 3)}


# ── OWL, via its own interpreter ─────────────────────────────────────────────

def _owl(key: str, ckpt: str, paths: list[str], frame: str | None,
         repeats: int = 3) -> dict:
    model_spec = config.MDO_MODELS[{"owl_c": "OWLC", "owl_t": "OWLT",
                                    "owl_d": "OWLD_H"}[key]]
    env = os.environ.copy()
    env["MDO_REPO_DIR"] = config.MDO_REPO_DIR
    env["DINOV3_ROOT"] = config.MDO_DINOV3_ROOT
    cmd = [config.MDO_PYTHON, os.path.join(ROOT, "analysis", "_bench_owl_extra.py"),
           "--pth", ckpt, "--model-name", model_spec["name"],
           "--model-kwargs", json.dumps(model_spec["eval_kwargs"]),
           "--img-size", str(config.CROP_SIZE),
           "--batch-sizes", *[str(b) for b in BATCH_SIZES],
           "--repeats", str(repeats),
           "--images", *paths]
    if frame:
        cmd += ["--frame", frame, "--stride", str(_stride())]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__"):])
    return {"error": (proc.stderr or proc.stdout)[-400:]}


# ── Driver ───────────────────────────────────────────────────────────────────

def measure(key: str, n_crops: int, do_frame: bool, runs: dict[str, str],
            repeats: int = 3) -> dict:
    reg_key = KEY_TO_REGISTRY.get(key, key)
    run_id = runs.get(reg_key)
    if run_id is None:
        return {"error": f"{key} is not in the manifest"}
    try:
        ckpt = predict.checkpoint_for(registry.resolve(reg_key), run_id)
    except FileNotFoundError as e:
        return {"error": str(e)}
    paths = sample_crops(n_crops)
    frame = _representative_frame() if do_frame else None

    if key in ("owl_c", "owl_t", "owl_d"):
        return _owl(key, ckpt, paths, frame, repeats)

    import cv2
    images = [cv2.imread(p) for p in paths]
    images = [im for im in images if im is not None]

    if key in ("yolov5", "yolo11", "yolo26"):
        runner = _ultralytics_runner(ckpt)
    elif key == "yolonas":
        runner = _yolonas_runner(ckpt)
    elif key == "fasterrcnn":
        runner = _fasterrcnn_runner(ckpt)
    else:
        return {"error": f"no runner for {key}"}

    out = {"batch_sweep": _sweep(runner, images, repeats)}
    if frame:
        out["end_to_end"] = _end_to_end(runner, frame)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-size and end-to-end timing, all models")
    ap.add_argument("--models", nargs="*", default=list(ALL_MODELS))
    ap.add_argument("-m", "--manifest",
                    default=os.path.join(ROOT, "models_v1.docx"),
                    help="pins which run of each model is benchmarked")
    ap.add_argument("--crops", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed passes per batch size; >1 gives a spread")
    ap.add_argument("--no-end-to-end", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    runs = {registry.resolve(e.key).key: e.run_id
            for e in manifest_mod.load(args.manifest)}

    results = {}
    for key in args.models:
        print(f"  {key} ...", flush=True)
        try:
            results[key] = measure(key, args.crops, not args.no_end_to_end, runs,
                                   repeats=args.repeats)
        except Exception as e:  # one framework failing must not sink the rest
            results[key] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  {key}: {'error' if results[key].get('error') else 'ok'}", flush=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bench_extra.json")
    with open(path, "w") as f:
        json.dump({"generated": stamp, "crops": args.crops, "repeats": args.repeats,
                   "manifest": os.path.basename(args.manifest),
                   "runs": runs,
                   "batch_sizes": list(BATCH_SIZES), "results": results}, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
