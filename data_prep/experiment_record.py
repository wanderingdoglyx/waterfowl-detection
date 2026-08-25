#!/usr/bin/env python
"""
Standard experimental record — one JSON per run, written automatically.

Every training and evaluation run writes `experiment.json` into its own run directory,
capturing enough context to trace a number back to the code, data, weights and machine
that produced it.  The goal is that a result in a table can always be answered for: what
produced this, and could it be produced again?

The record accumulates across phases rather than being written once.  Training writes the
`training` section; a later `--eval` on the same run merges in `evaluation` without
disturbing what is already there, so one file describes the whole life of a run.

Sections (Task 1.3):

    experiment_id       <model>__<run timestamp>, unique per run directory
    date                created / last-updated UTC timestamps
    code                git commit, branch, dirty flag, library versions
    dataset             version, root, and SHA-256 of each split file
    split               train / val / test crop and image counts
    model               key, label, architecture family
    pretrained          starting weights and their SHA-256 where on disk
    training            hyperparameters + what actually happened
    inference           thresholds, detection limits, NMS settings
    hardware            GPU, driver, CPU, RAM, hostname
    metrics_accuracy    the reported numbers
    metrics_efficiency  wall-clock, throughput, peak GPU memory
    outputs             run directory, checkpoint, artifact paths

Everything degrades gracefully: a missing git binary, absent GPU or unimportable library
records `null` for that field rather than failing the run that produced real results.

Usage:

    from data_prep.experiment_record import ExperimentRecord, Timer

    rec = ExperimentRecord(run_dir, model="yolo26", label="YOLO26")
    rec.set_pretrained("yolo26m.pt")
    rec.set_training({"epochs": 100, "batch": 8, ...})
    with Timer() as t:
        ...train...
    rec.finish_training(checkpoint=ckpt, duration_s=t.seconds, epochs_completed=100)

    rec.set_inference({"conf": 0.05, "max_det": 500})
    rec.set_accuracy({"ap": 91.81, "mae": 0.449})
    rec.finish_evaluation(duration_s=t.seconds, n_images=25708)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data_prep.config as config

RECORD_NAME = "experiment.json"


class Timer:
    """Wall-clock stopwatch, and peak GPU memory if torch reports one."""

    def __enter__(self):
        self._t0 = time.time()
        self.peak_gpu_mb = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self._t0, 2)
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        except Exception:
            pass
        return False


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def sha256(path: str, limit_mb: int = 0) -> str | None:
    """SHA-256 of a file, or None.  `limit_mb` > 0 hashes only the first N MB, which is
    enough to identify a multi-GB checkpoint without reading all of it."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    cap = limit_mb * 1024**2 if limit_mb else None
    read = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            read += len(block)
            if cap and read >= cap:
                return h.hexdigest() + f"~first{limit_mb}MB"
    return h.hexdigest()


def capture_code() -> dict:
    """Git state plus the versions of every library that can change a result."""
    libs = {}
    for name in ("torch", "ultralytics", "super_gradients", "detectron2", "numpy",
                 "cv2", "pycocotools"):
        try:
            libs[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            libs[name] = None
    status = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_commit_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        # A dirty tree means the commit alone does not describe the code that ran.
        "git_dirty": bool(status) if status is not None else None,
        "git_dirty_files": len(status.splitlines()) if status else 0,
        "python": platform.python_version(),
        "libraries": libs,
    }


def capture_hardware() -> dict:
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({"name": p.name,
                             "memory_gb": round(p.total_memory / 1024**3, 1),
                             "capability": f"{p.major}.{p.minor}"})
        cuda = torch.version.cuda
    except Exception:
        cuda = None
    mem_gb = None
    try:
        mem_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": mem_gb,
        "gpus": gpus,
        "cuda": cuda,
        "nvidia_driver": _run(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"]),
    }


def capture_dataset() -> dict:
    """
    Dataset version plus a fingerprint of the split files.

    The SHA-256 of each `coco.json` is the part that matters: it pins the exact split a
    run consumed, so a later change to the data is detectable rather than merely
    documented.  Two runs sharing these hashes were scored on identical ground truth.
    """
    splits = {}
    for name in ("train", "val", "test"):
        p = os.path.join(config.CROPS_JSON_DIR, name, "coco.json")
        entry = {"path": p, "sha256": sha256(p), "images": None, "annotations": None}
        if os.path.exists(p):
            try:
                with open(p) as f:
                    j = json.load(f)
                entry["images"] = len(j["images"])
                entry["annotations"] = len(j["annotations"])
            except Exception:
                pass
        splits[name] = entry
    return {
        "version": getattr(config, "DATASET_VERSION", None),
        "root": config.DATASET_ROOT,
        "crop_size": config.CROP_SIZE,
        "crop_overlap": config.CROP_OVERLAP,
        "random_seed": config.RANDOM_SEED,
        "val_fraction": config.VAL_FRACTION,
        "splits": splits,
    }


def default_inference_config() -> dict:
    """The project-wide inference settings, so every record carries them even when a
    caller does not pass its own."""
    return {
        "score_threshold": config.CONF_THRESHOLD,
        "max_detections_per_image": config.MAX_DETECTIONS_PER_IMAGE,
        "map_iou_threshold": config.IOU_THRESHOLD_MAP,
        "point_match_tau_px": config.PAPER_TAU,
        "bootstrap_resamples": config.PAPER_BOOTSTRAP,
    }


class ExperimentRecord:
    """Accumulating record for one run directory."""

    def __init__(self, run_dir: str, model: str, label: str | None = None,
                 family: str | None = None):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, RECORD_NAME)
        os.makedirs(run_dir, exist_ok=True)
        self.data = self._load()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.data.setdefault("experiment_id", f"{model}__{os.path.basename(run_dir)}")
        self.data.setdefault("created_utc", now)
        self.data["updated_utc"] = now
        self.data["model"] = {"key": model, "label": label or model, "family": family}
        self.data["code"] = capture_code()
        self.data["hardware"] = capture_hardware()
        self.data["dataset"] = capture_dataset()
        self.data.setdefault("inference", default_inference_config())
        self.data.setdefault("outputs", {})["run_dir"] = run_dir
        self.save()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ── section setters ───────────────────────────────────────────────────────
    def set_pretrained(self, weights: str, **extra) -> "ExperimentRecord":
        self.data["pretrained"] = {"weights": weights,
                                   "sha256": sha256(weights, limit_mb=64), **extra}
        return self

    def set_training(self, cfg: dict) -> "ExperimentRecord":
        self.data.setdefault("training", {}).update({"config": cfg})
        return self

    def set_inference(self, cfg: dict) -> "ExperimentRecord":
        self.data["inference"] = {**default_inference_config(), **cfg}
        return self

    def set_accuracy(self, metrics: dict) -> "ExperimentRecord":
        self.data["metrics_accuracy"] = metrics
        return self

    def set_efficiency(self, metrics: dict) -> "ExperimentRecord":
        self.data.setdefault("metrics_efficiency", {}).update(metrics)
        return self

    def set_outputs(self, **paths) -> "ExperimentRecord":
        self.data.setdefault("outputs", {}).update(paths)
        return self

    # ── phase completion ──────────────────────────────────────────────────────
    def finish_training(self, checkpoint: str | None = None, duration_s: float | None = None,
                        epochs_completed: int | None = None, peak_gpu_mb: float | None = None):
        t = self.data.setdefault("training", {})
        t["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t["epochs_completed"] = epochs_completed
        if checkpoint:
            self.set_outputs(checkpoint=checkpoint)
            self.data.setdefault("training", {})["checkpoint_sha256"] = sha256(checkpoint, 64)
        eff = {"train_seconds": duration_s, "train_peak_gpu_mb": peak_gpu_mb}
        if duration_s and epochs_completed:
            eff["seconds_per_epoch"] = round(duration_s / max(epochs_completed, 1), 1)
        self.set_efficiency({k: v for k, v in eff.items() if v is not None})
        return self.save()

    def finish_evaluation(self, duration_s: float | None = None, n_images: int | None = None,
                          n_detections: int | None = None, peak_gpu_mb: float | None = None):
        self.data.setdefault("evaluation", {})["completed_utc"] = \
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        eff = {"eval_seconds": duration_s, "eval_images": n_images,
               "eval_detections": n_detections, "eval_peak_gpu_mb": peak_gpu_mb}
        if duration_s and n_images:
            eff["images_per_second"] = round(n_images / duration_s, 2)
            eff["ms_per_image"] = round(1000.0 * duration_s / n_images, 2)
        self.set_efficiency({k: v for k, v in eff.items() if v is not None})
        return self.save()

    def save(self) -> str:
        self.data["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        return self.path


def summarise(root: str | None = None) -> list[dict]:
    """Flatten every experiment.json under output/ into one row per run."""
    root = root or config.OUTPUT_DIR
    rows = []
    for dirpath, _, files in os.walk(root):
        if RECORD_NAME not in files:
            continue
        try:
            with open(os.path.join(dirpath, RECORD_NAME)) as f:
                d = json.load(f)
        except Exception:
            continue
        acc = d.get("metrics_accuracy") or {}
        eff = d.get("metrics_efficiency") or {}
        rows.append({
            "experiment_id": d.get("experiment_id"),
            "date": (d.get("created_utc") or "")[:10],
            "commit": (d.get("code") or {}).get("git_commit_short"),
            "dirty": (d.get("code") or {}).get("git_dirty"),
            "dataset": (d.get("dataset") or {}).get("version"),
            "test_sha": ((d.get("dataset") or {}).get("splits", {}).get("test") or {}).get("sha256", "")[:8],
            "ap": acc.get("ap"), "mae": acc.get("mae"),
            "train_s": eff.get("train_seconds"), "eval_s": eff.get("eval_seconds"),
            "run_dir": (d.get("outputs") or {}).get("run_dir"),
        })
    return sorted(rows, key=lambda r: r["experiment_id"] or "")


if __name__ == "__main__":
    rows = summarise()
    if not rows:
        print("No experiment.json records found under output/.")
    else:
        hdr = f"{'experiment_id':42s}{'date':12s}{'commit':10s}{'dataset':9s}{'test_sha':10s}{'AP':>7s}{'MAE':>8s}"
        print(hdr); print("-" * len(hdr))
        for r in rows:
            ap = f"{r['ap']:.2f}" if isinstance(r["ap"], (int, float)) else "-"
            mae = f"{r['mae']:.3f}" if isinstance(r["mae"], (int, float)) else "-"
            dirty = "*" if r["dirty"] else " "
            print(f"{(r['experiment_id'] or '?')[:41]:42s}{r['date']:12s}"
                  f"{(r['commit'] or '-')+dirty:10s}{str(r['dataset'] or '-'):9s}"
                  f"{r['test_sha'] or '-':10s}{ap:>7s}{mae:>8s}")
        print(f"\n{len(rows)} experiments.  * = uncommitted changes when the run started.")
