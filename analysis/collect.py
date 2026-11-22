"""
Load one run's evaluation metrics off disk into a uniform record.

Every model writes the same OWL-paper point protocol (τ = 40 px, greedy
one-to-one NN matching), but in two different places:

    box detectors   <run>/paper_point_metrics.json
    OWL detectors   <run>/eval/metrics.json  →  ["paper_point_metrics"]

Those τ=40 numbers are the only figures directly comparable across all models,
so they are the backbone of the table.  mAP30 is read as well but kept apart:
real COCO mAP30 for box models, pseudo-box mAP30 for OWL.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field, asdict

from data_prep import config
from analysis.registry import ModelSpec

CKPT_ROOT = os.path.join(config.OUTPUT_DIR, "checkpoints")

# "  mAP30 :  75.59%"  /  "  mAR30 :  80.12%"
_MAP_RE = re.compile(r"^\s*mAP30\s*:\s*([\d.]+)%", re.M)
_MAR_RE = re.compile(r"^\s*mAR30\s*:\s*([\d.]+)%", re.M)


@dataclass
class RunMetrics:
    """One row of the comparison: a model, a run, and its evaluated metrics."""

    key: str
    label: str
    family: str
    kind: str
    run_id: str
    run_dir: str

    # OWL-paper point protocol (comparable across every model)
    ap: float | None = None
    auc_pr: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    mae: float | None = None
    rmse: float | None = None
    t_star: float | None = None
    signed_pct_error: float | None = None
    ap_ci: tuple[float, float] | None = None
    mae_ci: tuple[float, float] | None = None
    rmse_ci: tuple[float, float] | None = None

    # Box metrics — real mAP30 for box models, pseudo-box for OWL (see kind)
    map30: float | None = None
    mar30: float | None = None

    # Evaluation protocol, used to verify the runs are mutually comparable
    tau_px: float | None = None
    n_images: int | None = None
    n_gt_points: int | None = None
    n_detections: int | None = None

    # Provenance
    epochs: int | None = None
    n_train: int | None = None
    git_commit: str | None = None

    warnings: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("warnings")
        for k in ("ap_ci", "mae_ci", "rmse_ci"):
            lo_hi = d.pop(k)
            d[f"{k}_lo"], d[f"{k}_hi"] = lo_hi if lo_hi else (None, None)
        return d


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _apply_paper_metrics(rec: RunMetrics, ppm: dict) -> None:
    """Fill the τ=40 OWL-paper point metrics, shared by both layouts."""
    proto = ppm.get("protocol", {})
    det = ppm.get("detection", {})
    cnt = ppm.get("counting", {})
    ci = ppm.get("ci95", {})

    rec.tau_px = proto.get("tau_px")
    rec.n_images = proto.get("n_images")
    rec.n_gt_points = proto.get("n_gt_points")
    rec.n_detections = proto.get("n_detections")

    rec.ap = det.get("ap")
    rec.auc_pr = det.get("auc_pr")
    rec.precision = det.get("precision_at_t_star")
    rec.recall = det.get("recall_at_t_star")
    rec.f1 = det.get("f1_at_t_star")

    rec.t_star = cnt.get("t_star")
    rec.mae = cnt.get("mae")
    rec.rmse = cnt.get("rmse")
    rec.signed_pct_error = cnt.get("signed_pct_error")

    for name, src in (("ap_ci", "ap"), ("mae_ci", "mae"), ("rmse_ci", "rmse")):
        v = ci.get(src)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            setattr(rec, name, (float(v[0]), float(v[1])))


def _apply_box_metrics_from_text(rec: RunMetrics, path: str) -> None:
    """mAP30 / mAR30 live only in the human-readable eval_results.txt for box models."""
    if not os.path.exists(path):
        rec.warnings.append(f"no eval_results.txt at {path} — mAP30 unavailable")
        return
    with open(path) as f:
        text = f.read()
    m, r = _MAP_RE.search(text), _MAR_RE.search(text)
    if m:
        rec.map30 = float(m.group(1))
    if r:
        rec.mar30 = float(r.group(1))
    if not m:
        rec.warnings.append("mAP30 not found in eval_results.txt")


def _apply_provenance(rec: RunMetrics) -> None:
    info = _read_json(os.path.join(rec.run_dir, "run_info.json")) or {}
    rec.epochs = info.get("epochs")
    rec.n_train = info.get("n_train")

    exp = _read_json(os.path.join(rec.run_dir, "experiment.json")) or {}
    rec.git_commit = (exp.get("code") or {}).get("git_commit_short")


def load_run(spec: ModelSpec, run_id: str) -> RunMetrics:
    """Collect every metric available for one model's run.

    Missing metrics become None and are recorded in `warnings` rather than
    raising — one incomplete run should not sink the whole comparison.
    """
    run_dir = os.path.join(CKPT_ROOT, spec.key, run_id)
    rec = RunMetrics(
        key=spec.key, label=spec.label, family=spec.family, kind=spec.kind,
        run_id=run_id, run_dir=run_dir,
    )
    if not os.path.isdir(run_dir):
        rec.warnings.append(f"run directory does not exist: {run_dir}")
        return rec

    if spec.kind == "point":
        # OWL: everything sits in one eval/metrics.json.
        metrics = _read_json(os.path.join(run_dir, "eval", "metrics.json"))
        if metrics is None:
            rec.warnings.append("missing eval/metrics.json — run not evaluated?")
        else:
            ppm = metrics.get("paper_point_metrics")
            if ppm:
                _apply_paper_metrics(rec, ppm)
            else:
                rec.warnings.append("eval/metrics.json has no paper_point_metrics")
            pseudo = metrics.get("pseudo_box_map30") or {}
            rec.map30 = pseudo.get("ap30")
            rec.mar30 = pseudo.get("ar30")
            if rec.n_detections is None:
                rec.n_detections = metrics.get("n_detections")
    else:
        ppm = _read_json(os.path.join(run_dir, "paper_point_metrics.json"))
        if ppm is None:
            rec.warnings.append("missing paper_point_metrics.json — run not evaluated?")
        else:
            _apply_paper_metrics(rec, ppm)
        _apply_box_metrics_from_text(rec, os.path.join(run_dir, "eval_results.txt"))

    _apply_provenance(rec)
    return rec


def check_comparable(records: list[RunMetrics]) -> list[str]:
    """Verify the runs were evaluated under one protocol on one test set.

    Comparing AP across runs scored on different test sets or different τ is
    the easiest way to produce a confidently wrong table, so this is reported
    loudly rather than assumed.
    """
    problems: list[str] = []
    for attr, desc in (
        ("tau_px", "matching radius τ"),
        ("n_images", "test-set image count"),
        ("n_gt_points", "ground-truth point count"),
    ):
        seen: dict[object, list[str]] = {}
        for r in records:
            v = getattr(r, attr)
            if v is not None:
                seen.setdefault(v, []).append(r.label)
        if len(seen) > 1:
            detail = "; ".join(f"{v} ({', '.join(m)})" for v, m in sorted(seen.items(), key=lambda kv: str(kv[0])))
            problems.append(f"{desc} differs across runs: {detail}")
    return problems


def write_csv(records: list[RunMetrics], path: str) -> str:
    """One row per run, every stored metric as a column."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [r.as_row() for r in records]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
