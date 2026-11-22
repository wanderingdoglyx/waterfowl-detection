"""
Task 2.1 — read the efficiency benchmark into the analysis.

The measurements themselves are produced by ``data_prep.efficiency`` (all eight
models, one machine, one protocol).  This module only *reads* them, picks a
coherent set, and attaches them to the accuracy records so Task 2.2 can trade
one against the other.

Choosing a benchmark run matters.  Numbers from different runs were measured in
different processes, so mixing them silently compares a model benchmarked while
the GPU was warm against one benchmarked cold.  ``load()`` therefore prefers the
most recent run in which *every* requested model succeeded, and reports which
run it used; it falls back to merging across runs only when no single run is
complete, and then records the provenance per model.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

from data_prep import config

EFF_DIR = os.path.join(config.OUTPUT_DIR, "efficiency")

# efficiency.py's model keys -> registry keys
EFF_TO_REGISTRY = {
    "fasterrcnn": "fasterrcnn", "yolonas": "yolonas", "yolov5": "yolov5",
    "yolo11": "yolo11", "yolo26": "yolo26",
    "owl_c": "mdo_owl_c", "owl_t": "mdo_owl_t", "owl_d": "mdo_owl_d",
}


@dataclass
class EfficiencyRecord:
    key: str
    label: str
    params: int | None = None
    checkpoint_mb: float | None = None
    peak_gpu_mb: float | None = None
    ms_per_crop: float | None = None
    ms_per_crop_std: float | None = None
    crops_per_second: float | None = None
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None
    inference_ms: float | None = None
    train_hours: float | None = None
    train_peak_gpu_mb: float | None = None
    batch_sweep: dict = field(default_factory=dict)
    end_to_end: dict = field(default_factory=dict)
    source_run: str | None = None


@dataclass
class Benchmark:
    """A set of efficiency records plus the environment they were measured in."""

    records: dict[str, EfficiencyRecord]
    protocol: dict
    hardware: dict
    software: dict
    run_id: str
    merged: bool = False


def _runs() -> list[str]:
    return sorted(glob.glob(os.path.join(EFF_DIR, "*", "efficiency.json")))


def _entry_ok(entry: dict) -> bool:
    return not entry.get("error") and entry.get("ms_per_crop_mean") is not None


def _to_record(eff_key: str, entry: dict, run_id: str) -> EfficiencyRecord:
    training = entry.get("training") or {}
    return EfficiencyRecord(
        key=EFF_TO_REGISTRY.get(eff_key, eff_key),
        label=entry.get("label", eff_key),
        params=entry.get("params"),
        checkpoint_mb=entry.get("checkpoint_mb"),
        peak_gpu_mb=entry.get("peak_gpu_mb"),
        ms_per_crop=entry.get("ms_per_crop_mean"),
        ms_per_crop_std=entry.get("ms_per_crop_std"),
        crops_per_second=entry.get("crops_per_second"),
        preprocess_ms=entry.get("preprocess_ms"),
        postprocess_ms=entry.get("postprocess_ms"),
        inference_ms=entry.get("inference_ms"),
        train_hours=training.get("hours"),
        train_peak_gpu_mb=training.get("peak_gpu_mb"),
        batch_sweep=entry.get("batch_sweep") or {},
        end_to_end=entry.get("end_to_end") or {},
        source_run=run_id,
    )


def load(want_keys: set[str] | None = None) -> Benchmark | None:
    """Load the best available benchmark for the requested registry keys."""
    runs = _runs()
    if not runs:
        return None

    parsed = []
    for path in runs:
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        parsed.append((os.path.basename(os.path.dirname(path)), payload))

    # Prefer the newest run where every wanted model succeeded.
    for run_id, payload in reversed(parsed):
        recs = {}
        for eff_key, entry in (payload.get("results") or {}).items():
            if _entry_ok(entry):
                r = _to_record(eff_key, entry, run_id)
                recs[r.key] = r
        if want_keys is None or want_keys <= set(recs):
            return Benchmark(recs, payload.get("protocol", {}),
                             payload.get("hardware", {}), payload.get("software", {}),
                             run_id)

    # Nothing complete — merge, newest wins, and remember where each came from.
    merged: dict[str, EfficiencyRecord] = {}
    newest = parsed[-1]
    for run_id, payload in parsed:
        for eff_key, entry in (payload.get("results") or {}).items():
            if _entry_ok(entry):
                r = _to_record(eff_key, entry, run_id)
                merged[r.key] = r
    if not merged:
        return None
    return Benchmark(merged, newest[1].get("protocol", {}), newest[1].get("hardware", {}),
                     newest[1].get("software", {}), newest[0], merged=True)


# ── Task 2.1 "recommended": batch sweep + end-to-end, from analysis.bench_extra ──

EXTRA_DIR = os.path.join(config.OUTPUT_DIR, "analysis", "bench_extra")


def load_extra() -> dict:
    """Newest bench_extra run, keyed by registry key. {} when never run."""
    runs = sorted(glob.glob(os.path.join(EXTRA_DIR, "*", "bench_extra.json")))
    if not runs:
        return {}
    with open(runs[-1]) as f:
        payload = json.load(f)
    out = {}
    for eff_key, entry in (payload.get("results") or {}).items():
        if entry.get("error"):
            continue
        out[EFF_TO_REGISTRY.get(eff_key, eff_key)] = entry
    out["_meta"] = {"run_id": os.path.basename(os.path.dirname(runs[-1])),
                    "crops": payload.get("crops"),
                    "batch_sizes": payload.get("batch_sizes", [])}
    return out
