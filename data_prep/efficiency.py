#!/usr/bin/env python
"""
Efficiency benchmark — all eight models, one hardware/software environment.

Accuracy alone does not decide which detector to deploy over a 17-gigapixel survey.  This
measures the other half: how large each model is, how fast it processes a 512x512 crop,
how much GPU memory it needs, and how long a full UAS frame takes end to end.

Measured for every model (Task 2.1, required):

    parameters              counted from the loaded model, or from the checkpoint's
                            state_dict for OWL variants whose framework lives elsewhere
    checkpoint size         bytes on disk
    peak GPU memory         torch peak allocation during the timed pass
    ms per 512x512 crop     mean +/- std over repeated passes, after warm-up
    throughput              crops per second

Also measured where the framework exposes it (Task 2.1, recommended):

    preprocess / postprocess split      Ultralytics reports these directly
    batch-size sensitivity              1, 2, 4, 8, 16
    training time and peak memory       recovered from each run's experiment.json
                                        or the training log, not re-measured
    end-to-end full UAS frame           tile a real 5472x3648 image at the project's
                                        stride, run every crop, merge

Method.  Timing is the mean of `--repeats` passes over the same `--crops` images, each
preceded by a warm-up that is discarded — a single measurement mostly reports cold cuDNN
autotuning.  `torch.cuda.synchronize()` brackets every timed region, without which GPU
work is queued asynchronously and the numbers are meaningless.

The three OWL variants run in their own Python 3.11 environment, so their timing is
delegated to megadetector_overhead/_bench_owl.py over a subprocess, exactly as evaluation
is.  Their parameter counts and checkpoint sizes are read directly here.

Usage (from the rebuild/ directory):

    python -m data_prep.efficiency                       # all models, 200 crops, 3 repeats
    python -m data_prep.efficiency --models yolo11 yolo26
    python -m data_prep.efficiency --crops 500 --repeats 5
    python -m data_prep.efficiency --batch-sweep         # add batch-size sensitivity
    python -m data_prep.efficiency --end-to-end          # add full-frame timing
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data_prep.config as config
from data_prep.experiment_record import capture_code, capture_hardware

BOX_MODELS = {
    "fasterrcnn": ("Faster R-CNN", "detectron2", config.CKPT_DIR, "model_best.pth"),
    "yolonas":    ("YOLO-NAS-m", "super-gradients", config.YOLONAS_CKPT_DIR,
                   os.path.join("RUN_*", "ckpt_best.pth")),
    "yolov5":     ("YOLOv5m", "ultralytics", config.YOLOV5_CKPT_DIR,
                   os.path.join("weights", "best.pt")),
    "yolo11":     ("YOLO11m", "ultralytics", config.YOLO11_CKPT_DIR,
                   os.path.join("weights", "best.pt")),
    "yolo26":     ("YOLO26m", "ultralytics", config.YOLO26_CKPT_DIR,
                   os.path.join("weights", "best.pt")),
}
OWL_MODELS = {
    "owl_c": ("OWL-C", "OWLC"), "owl_t": ("OWL-T", "OWLT"), "owl_d": ("OWL-D", "OWLD_H"),
}


def latest_run(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    runs = sorted((e.path for e in os.scandir(ckpt_dir) if e.is_dir()), reverse=True)
    return runs[0] if runs else None


def resolve_checkpoint(key: str) -> tuple[str | None, str | None]:
    """(checkpoint_path, run_dir) for a model, or (None, None)."""
    if key in BOX_MODELS:
        run = latest_run(BOX_MODELS[key][2])
        if not run:
            return None, None
        hits = sorted(glob.glob(os.path.join(run, BOX_MODELS[key][3])))
        return (hits[-1] if hits else None), run
    spec = config.mdo_model_spec(OWL_MODELS[key][1])
    run = latest_run(spec["ckpt_dir"])
    if not run:
        return None, None
    p = os.path.join(run, "weights", "best.pth")
    return (p if os.path.exists(p) else None), run


def count_parameters_from_checkpoint(path: str) -> int | None:
    """Sum tensor elements in a checkpoint's state_dict — works without the framework
    that produced it, which is how OWL variants are counted from this environment."""
    try:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    sd = None
    for k in ("model_state_dict", "state_dict", "model", "net"):
        if isinstance(obj.get(k), dict):
            sd = obj[k]
            break
    if sd is None:
        sd = obj
    try:
        return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    except Exception:
        return None


def sample_crops(n: int, seed: int = 42) -> list[str]:
    """A fixed, reproducible sample of test crops, so every model times the same pixels."""
    import random
    test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
    with open(test_json) as f:
        images = json.load(f)["images"]
    rng = random.Random(seed)
    picks = rng.sample(images, min(n, len(images)))
    return [os.path.join(config.CROPS_IMG_DIR, im["file_name"]) for im in picks]


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:
        pass
    return None


def time_ultralytics(ckpt: str, paths: list[str], repeats: int, warmup: int,
                     whole: bool = False) -> dict:
    from ultralytics import YOLO
    model = YOLO(ckpt)
    kw = dict(imgsz=config.CROP_SIZE, conf=config.CONF_THRESHOLD,
              max_det=config.MAX_DETECTIONS_PER_IMAGE, verbose=False)
    for p in paths[:warmup]:
        model.predict(p, **kw)
    _sync(); _reset_peak()
    per_pass, pre, inf, post = [], [], [], []
    for _ in range(repeats):
        _sync(); t0 = time.perf_counter()
        for p in paths:
            r = model.predict(p, **kw)
            s = r[0].speed
            pre.append(s["preprocess"]); inf.append(s["inference"]); post.append(s["postprocess"])
            if whole:
                # Materialise final coordinates, as a caller consuming detections would.
                b = r[0].boxes
                if b is not None and len(b):
                    b.xyxy.cpu().numpy(); b.conf.cpu().numpy()
        _sync(); per_pass.append((time.perf_counter() - t0) / len(paths) * 1000)
    return {"ms_per_crop": per_pass, "peak_gpu_mb": _peak_mb(),
            "preprocess_ms": round(statistics.mean(pre), 3),
            "inference_ms": round(statistics.mean(inf), 3),
            "postprocess_ms": round(statistics.mean(post), 3),
            "params": int(sum(x.numel() for x in model.model.parameters()))}


def time_yolonas(ckpt: str, paths: list[str], repeats: int, warmup: int,
                 whole: bool = False) -> dict:
    from super_gradients.training import models
    model = models.get(config.YOLONAS_MODEL, num_classes=1, checkpoint_path=ckpt)
    model = model.cuda().eval()
    kw = dict(conf=config.CONF_THRESHOLD, iou=config.YOLONAS_NMS_IOU,
              max_predictions=config.MAX_DETECTIONS_PER_IMAGE, fuse_model=False)
    for p in paths[:warmup]:
        model.predict(p, **kw)
    _sync(); _reset_peak()
    per_pass = []
    for _ in range(repeats):
        _sync(); t0 = time.perf_counter()
        for p in paths:
            r = model.predict(p, **kw)
            if whole:
                pr = (r if hasattr(r, "prediction") else list(r)[0]).prediction
                pr.bboxes_xyxy, pr.confidence
        _sync(); per_pass.append((time.perf_counter() - t0) / len(paths) * 1000)
    return {"ms_per_crop": per_pass, "peak_gpu_mb": _peak_mb(),
            "params": int(sum(x.numel() for x in model.parameters()))}


def time_fasterrcnn(ckpt: str, paths: list[str], repeats: int, warmup: int,
                    whole: bool = False) -> dict:
    import cv2
    from detectron2.engine import DefaultPredictor
    from faster_rcnn.model import build_cfg
    cfg = build_cfg(use_pretrained=False)
    cfg.MODEL.WEIGHTS = ckpt
    predictor = DefaultPredictor(cfg)
    imgs = None if whole else [cv2.imread(p) for p in paths]
    for x in (paths if whole else imgs)[:warmup]:
        predictor(cv2.imread(x) if whole else x)
    _sync(); _reset_peak()
    per_pass = []
    for _ in range(repeats):
        _sync(); t0 = time.perf_counter()
        if whole:
            for p in paths:                       # disk read inside the timed region
                inst = predictor(cv2.imread(p))["instances"].to("cpu")
                inst.pred_boxes.tensor.numpy(); inst.scores.numpy()
        else:
            for im in imgs:
                predictor(im)
        _sync(); per_pass.append((time.perf_counter() - t0) / len(paths) * 1000)
    return {"ms_per_crop": per_pass, "peak_gpu_mb": _peak_mb(),
            "params": int(sum(x.numel() for x in predictor.model.parameters()))}


def time_owl(key: str, ckpt: str, paths: list[str], repeats: int, warmup: int,
             whole: bool = False) -> dict:
    """Delegate to the MegaDetector-Overhead venv, which owns the animaloc stack."""
    helper = os.path.join(ROOT, "megadetector_overhead", "_bench_owl.py")
    if not os.path.exists(helper) or not os.path.exists(config.MDO_PYTHON):
        return {"error": "OWL benchmark helper or venv unavailable"}
    spec = config.mdo_model_spec(OWL_MODELS[key][1])
    env = os.environ.copy()
    env["MDO_REPO_DIR"] = config.MDO_REPO_DIR
    env["DINOV3_ROOT"] = config.MDO_DINOV3_ROOT
    cmd = [config.MDO_PYTHON, helper, "--pth", ckpt,
           "--model-name", spec["name"], "--model-kwargs", json.dumps(spec["eval_kwargs"]),
           "--down-ratio", str(config.MDO_DOWN_RATIO), "--img-size", str(config.CROP_SIZE),
           "--repeats", str(repeats), "--warmup", str(warmup),
           "--kernel", str(config.MDO_LMDS_KERNEL),
           "--adapt-ts", str(config.MDO_LMDS_ADAPT_TS),
           *(["--whole-process"] if whole else []),
           "--images", *paths]
    try:
        out = subprocess.run(cmd, cwd=config.MDO_REPO_DIR, env=env,
                             capture_output=True, text=True, timeout=1800)
        for line in out.stdout.splitlines():
            if line.startswith("BENCH_JSON "):
                return json.loads(line[len("BENCH_JSON "):])
        return {"error": (out.stderr or out.stdout)[-300:]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def batch_sweep(key: str, ckpt: str, paths: list[str], sizes=(1, 2, 4, 8, 16)) -> dict:
    """Throughput vs batch size.  Ultralytics only — evaluation deliberately runs at
    batch 1 for correctness, so this measures headroom that the eval path forgoes."""
    if key not in ("yolov5", "yolo11", "yolo26"):
        return {}
    from ultralytics import YOLO
    model = YOLO(ckpt)
    out = {}
    for bs in sizes:
        batch = paths[:max(bs * 4, bs)]
        model.predict(batch[:bs], imgsz=config.CROP_SIZE, verbose=False)
        _sync(); t0 = time.perf_counter()
        for i in range(0, len(batch), bs):
            model.predict(batch[i:i + bs], imgsz=config.CROP_SIZE,
                          conf=config.CONF_THRESHOLD,
                          max_det=config.MAX_DETECTIONS_PER_IMAGE, verbose=False)
        _sync(); dt = time.perf_counter() - t0
        out[str(bs)] = {"ms_per_crop": round(dt / len(batch) * 1000, 3),
                        "crops_per_second": round(len(batch) / dt, 1)}
    return out


def end_to_end(key: str, ckpt: str) -> dict:
    """Tile a real UAS frame at the project's stride and time every crop through the
    model — the number that matters for planning a survey, not a per-crop figure."""
    if key not in ("yolov5", "yolo11", "yolo26"):
        return {}
    from PIL import Image
    from ultralytics import YOLO
    src = None
    for d in sorted(glob.glob(os.path.join(config.DATASET_ROOT, "Bird_*"))):
        hits = sorted(glob.glob(os.path.join(d, "*.JPG")) + glob.glob(os.path.join(d, "*.jpg")))
        if hits:
            src = hits[0]; break
    if not src:
        return {}
    stride = int(config.CROP_SIZE * (1 - config.CROP_OVERLAP / 2))
    img = Image.open(src); W, H = img.size
    model = YOLO(ckpt)
    _sync(); t0 = time.perf_counter()
    tiles = 0
    import numpy as np
    for y in range(0, max(H - config.CROP_SIZE, 0) + 1, stride):
        for x in range(0, max(W - config.CROP_SIZE, 0) + 1, stride):
            tile = np.array(img.crop((x, y, x + config.CROP_SIZE, y + config.CROP_SIZE)))
            model.predict(tile[:, :, ::-1], imgsz=config.CROP_SIZE,
                          conf=config.CONF_THRESHOLD,
                          max_det=config.MAX_DETECTIONS_PER_IMAGE, verbose=False)
            tiles += 1
    _sync(); dt = time.perf_counter() - t0
    return {"image": os.path.basename(src), "resolution": f"{W}x{H}",
            "crops": tiles, "seconds": round(dt, 2),
            "seconds_per_megapixel": round(dt / (W * H / 1e6), 3)}


def training_cost(run_dir: str | None) -> dict:
    """Training time is recovered, not re-measured — rerunning 100 epochs to time it
    would cost more than the whole benchmark."""
    if not run_dir:
        return {}
    rec = os.path.join(run_dir, "experiment.json")
    if os.path.exists(rec):
        try:
            eff = json.load(open(rec)).get("metrics_efficiency", {})
            if eff.get("train_seconds"):
                return {"train_seconds": eff["train_seconds"],
                        "train_peak_gpu_mb": eff.get("train_peak_gpu_mb"),
                        "source": "experiment.json"}
        except Exception:
            pass
    csv = os.path.join(run_dir, "results.csv")          # Ultralytics logs cumulative time
    if os.path.exists(csv):
        try:
            rows = [l.strip().split(",") for l in open(csv) if l.strip()]
            hdr, last = rows[0], rows[-1]
            if "time" in hdr:
                return {"train_seconds": float(last[hdr.index("time")]),
                        "epochs": int(float(last[hdr.index("epoch")])),
                        "source": "results.csv"}
        except Exception:
            pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Efficiency benchmark for all models")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--crops", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-sweep", action="store_true")
    ap.add_argument("--end-to-end", action="store_true")
    ap.add_argument("--whole-process", action="store_true",
                    help="time the COMPLETE path per crop — disk read, preprocess, model, "
                         "postprocess (NMS or LMDS), and materialising final coordinates. "
                         "This is the deployment-relevant figure and the only one that is "
                         "strictly comparable across all eight models.")
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "efficiency"))
    a = ap.parse_args()

    keys = a.models or (list(BOX_MODELS) + list(OWL_MODELS))
    out_dir = os.path.join(a.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    paths = sample_crops(a.crops)

    print(f"Crops: {len(paths)}   repeats: {a.repeats}   warm-up: {a.warmup}")
    print(f"Output: {out_dir}\n")

    results = {}
    for key in keys:
        label = (BOX_MODELS.get(key) or OWL_MODELS.get(key) or (key,))[0]
        ckpt, run_dir = resolve_checkpoint(key)
        if not ckpt:
            print(f"  {label:14s} SKIP — no checkpoint"); continue
        print(f"  {label:14s} timing...", end="", flush=True)
        entry = {"label": label, "checkpoint": ckpt,
                 "checkpoint_bytes": os.path.getsize(ckpt),
                 "checkpoint_mb": round(os.path.getsize(ckpt) / 1024**2, 1),
                 "run_dir": run_dir}
        try:
            if key in ("yolov5", "yolo11", "yolo26"):
                entry.update(time_ultralytics(ckpt, paths, a.repeats, a.warmup, a.whole_process))
            elif key == "yolonas":
                entry.update(time_yolonas(ckpt, paths, a.repeats, a.warmup, a.whole_process))
            elif key == "fasterrcnn":
                entry.update(time_fasterrcnn(ckpt, paths, a.repeats, a.warmup, a.whole_process))
            else:
                entry.update(time_owl(key, ckpt, paths, a.repeats, a.warmup, a.whole_process))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"

        if entry.get("params") is None:
            entry["params"] = count_parameters_from_checkpoint(ckpt)
        ms = entry.get("ms_per_crop") or []
        if ms:
            entry["ms_per_crop_mean"] = round(statistics.mean(ms), 3)
            entry["ms_per_crop_std"] = round(statistics.stdev(ms), 3) if len(ms) > 1 else 0.0
            entry["crops_per_second"] = round(1000.0 / entry["ms_per_crop_mean"], 1)
        entry["training"] = training_cost(run_dir)
        if a.batch_sweep:
            entry["batch_sweep"] = batch_sweep(key, ckpt, paths)
        if a.end_to_end:
            entry["end_to_end"] = end_to_end(key, ckpt)
        results[key] = entry
        print(f"  {entry.get('ms_per_crop_mean', '?')} ms/crop"
              + (f"   ({entry['error'][:40]})" if "error" in entry else ""))

    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "protocol": {"mode": "whole_process" if a.whole_process else "model_only",
                            "crops": len(paths), "repeats": a.repeats,
                            "warmup": a.warmup, "crop_size": config.CROP_SIZE,
                            "conf": config.CONF_THRESHOLD,
                            "max_det": config.MAX_DETECTIONS_PER_IMAGE},
               "hardware": capture_hardware(), "software": capture_code(),
               "results": results}
    with open(os.path.join(out_dir, "efficiency.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = [f"Efficiency benchmark — {payload['generated']}", ""]
    hw = payload["hardware"]
    gpu = hw["gpus"][0]["name"] if hw["gpus"] else "CPU only"
    lines += [f"Hardware : {gpu}, {hw['cpu_count']} CPU, {hw['ram_gb']} GB RAM, "
              f"driver {hw['nvidia_driver']}, CUDA {hw['cuda']}",
              f"Software : Python {payload['software']['python']}, "
              f"torch {payload['software']['libraries'].get('torch')}",
              f"Mode     : {'WHOLE PROCESS (disk read -> final coordinates)' if a.whole_process else 'model only'}",
              f"Protocol : {len(paths)} crops x {a.repeats} repeats, {a.warmup} warm-up, "
              f"{config.CROP_SIZE}px, conf {config.CONF_THRESHOLD}, "
              f"max_det {config.MAX_DETECTIONS_PER_IMAGE}", ""]
    hdr = (f"{'Model':14s}{'params':>13s}{'ckpt MB':>11s}{'ms/crop':>15s}"
           f"{'crops/s':>10s}{'peak GPU MB':>13s}")
    lines += [hdr, "-" * len(hdr)]
    for k, e in results.items():
        p = f"{e['params']:,}" if e.get("params") else "-"
        ms = (f"{e['ms_per_crop_mean']:.2f}+/-{e['ms_per_crop_std']:.2f}"
              if e.get("ms_per_crop_mean") else "-")
        lines.append(f"{e['label']:14s}{p:>13s}{e['checkpoint_mb']:>11,.1f}{ms:>15s}"
                     f"{str(e.get('crops_per_second', '-')):>10s}"
                     f"{str(e.get('peak_gpu_mb', '-')):>13s}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "efficiency.txt"), "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
