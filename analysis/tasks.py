"""
Orchestrates Tasks 2.1-2.4 and writes their tables.

Everything here is computed on ONE test split from ONE prediction cache, so the
accuracy in the trade-off tables, the size curves and the density curves all
describe the same runs on the same images.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

from data_prep import config
from analysis import efficiency, pareto, predict, rescore, strata
from analysis.registry import ModelSpec


@dataclass
class ModelRow:
    """One model: accuracy on the current split, plus its measured cost."""

    key: str
    label: str
    kind: str
    run_id: str

    # Accuracy, recomputed from the cached predictions
    ap: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    mae: float | None = None
    rmse: float | None = None
    t_star: float | None = None
    signed_pct_error: float | None = None
    total_pred: int | None = None

    # Efficiency
    params: int | None = None
    params_m: float | None = None
    checkpoint_mb: float | None = None
    peak_gpu_mb: float | None = None
    ms_per_crop: float | None = None
    ms_per_crop_std: float | None = None
    crops_per_second: float | None = None
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None
    train_hours: float | None = None
    train_peak_gpu_mb: float | None = None
    eff_source_run: str | None = None
    train_gpu: str | None = None
    train_host: str | None = None

    size: list = field(default_factory=list)
    density: list = field(default_factory=list)

    @property
    def count_accuracy(self) -> float | None:
        """100 - |signed count error|, so 'closer to the true count' is higher."""
        if self.signed_pct_error is None:
            return None
        return 100.0 - abs(self.signed_pct_error)


def build(specs: list[tuple[ModelSpec, str]], out_dir: str,
          force: bool = False) -> tuple[list[ModelRow], dict]:
    """Predict, re-score, stratify and merge efficiency for every model."""
    gt = strata.load_ground_truth()
    counts = np.array([gt.counts[int(i)] for i in gt.image_ids])
    size_edges = strata.quantile_bins(gt.gt_size, 6)
    dens_edges, dens_labels = strata.density_bins(counts)
    size_labels = strata.size_bin_labels(size_edges)

    rows: list[ModelRow] = []
    for spec, run_id in specs:
        predict.ensure(spec, run_id, force=force)
        pm = rescore.ensure(spec, run_id, force=force)
        cache = predict.load(predict.cache_path(spec, run_id))

        det = pm["detection"]
        cnt = pm["counting"]
        row = ModelRow(
            key=spec.key, label=spec.label, kind=spec.kind, run_id=run_id,
            ap=det["ap"] * 100, f1=det["f1_at_t_star"] * 100,
            precision=det["precision_at_t_star"] * 100,
            recall=det["recall_at_t_star"] * 100,
            mae=cnt["mae"], rmse=cnt["rmse"], t_star=cnt["t_star"],
            signed_pct_error=cnt["signed_pct_error"],
            total_pred=cnt.get("total_pred_at_t_star"),
        )

        # Which machine trained this run. Once training is spread across hosts,
        # a wall-clock time or a peak-memory figure means nothing without it.
        exp = os.path.join(config.OUTPUT_DIR, "checkpoints", spec.key, run_id,
                           "experiment.json")
        if os.path.exists(exp):
            try:
                with open(exp) as f:
                    hw = json.load(f).get("hardware", {})
                gpus = hw.get("gpus") or []
                row.train_gpu = gpus[0]["name"] if gpus else None
                row.train_host = hw.get("hostname")
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        out = strata.outcomes_at_t_star(gt, cache, row.t_star)
        has_box = bool(np.isfinite(out.det_size).any())
        row.size = strata.size_analysis(gt, out, size_edges, has_box)
        row.density = strata.density_analysis(gt, out, dens_edges, dens_labels)
        rows.append(row)

    bench = efficiency.load({r.key for r in rows})
    if bench:
        for row in rows:
            e = bench.records.get(row.key)
            if not e:
                continue
            row.params = e.params
            row.params_m = e.params / 1e6 if e.params else None
            row.checkpoint_mb = e.checkpoint_mb
            row.peak_gpu_mb = e.peak_gpu_mb
            row.ms_per_crop = e.ms_per_crop
            row.ms_per_crop_std = e.ms_per_crop_std
            row.crops_per_second = e.crops_per_second
            row.preprocess_ms = e.preprocess_ms
            row.postprocess_ms = e.postprocess_ms
            row.train_hours = e.train_hours
            row.train_peak_gpu_mb = e.train_peak_gpu_mb
            row.eff_source_run = e.source_run

    gpus = {r.train_gpu for r in rows if r.train_gpu}
    hosts = {r.train_host for r in rows if r.train_host}
    unknown = [r.label for r in rows if not r.train_gpu]

    context = {
        "train_gpus": sorted(gpus), "train_hosts": sorted(hosts),
        "train_gpu_unknown": unknown,
        "bench": bench,
        "gt": gt,
        "size_edges": size_edges, "size_labels": size_labels,
        "dens_edges": dens_edges, "dens_labels": dens_labels,
        "n_images": len(gt.image_ids), "n_gt": len(gt.gt_size),
    }
    return rows, context


# ── Derived tables ───────────────────────────────────────────────────────────

# Operational budgets for "the best accuracy that fits": (name, cost axis, limit).
BUDGETS = [
    ("peak_gpu_le_512mb", pareto.Axis("peak_gpu_mb", "Peak GPU (MB)", False), 512.0),
    ("latency_le_5ms", pareto.Axis("ms_per_crop", "ms/crop", False), 5.0),
    ("checkpoint_le_100mb", pareto.Axis("checkpoint_mb", "Checkpoint (MB)", False), 100.0),
    ("params_le_25m", pareto.Axis("params_m", "Params (M)", False), 25.0),
]


def pareto_membership(rows: list[ModelRow]) -> list[dict]:
    """Every model x trade-off, flagged when no other model beats it on both axes."""
    out = []
    for x_axis, y_axis in pareto.TRADEOFFS:
        front = pareto.frontier_keys(rows, x_axis, y_axis)
        for r in rows:
            out.append({"x": x_axis.attr, "y": y_axis.attr, "model": r.key,
                        "label": r.label, "on_frontier": int(r.key in front)})
    return out


def budget_picks(rows: list[ModelRow]) -> list[dict]:
    """Highest-AP model under each budget in `BUDGETS`."""
    ap_axis = pareto.Axis("ap", "AP (%)", True)
    out = []
    for name, cost_axis, budget in BUDGETS:
        pick = pareto.best_under_budget(rows, ap_axis, cost_axis, budget)
        out.append({"constraint": name, "cost_metric": cost_axis.attr, "budget": budget,
                    "model": pick.key if pick else None,
                    "label": pick.label if pick else None,
                    "ap": pick.ap if pick else None,
                    "cost": getattr(pick, cost_axis.attr) if pick else None})
    return out


def size_degradation(rows: list[ModelRow], tol: float = 5.0) -> list[dict]:
    """Where recall starts to deteriorate with apparent size, per model.

    "Deteriorates" is read as: recall falls more than `tol` points below that
    model's own best bin — self-relative, so a weaker model is not judged against
    a stronger one's ceiling.

    Small and large birds are kept apart. Several models also lose recall on the
    *largest* birds (a 300 px bird spans over half the crop, and partially-cropped
    giants are their own failure), and folding that in with the small end would
    hide the threshold being asked for.
    """
    out = []
    for r in rows:
        vals = [(b, b.recall) for b in r.size if b.recall is not None]
        if not vals:
            continue
        best = max(v for _, v in vals)
        degraded = [v < best - tol for _, v in vals]

        # Contiguous degraded run starting at the smallest bin.
        small_edge, small_recall = None, None
        if degraded[0]:
            i = 0
            while i + 1 < len(degraded) and degraded[i + 1]:
                i += 1
            small_edge = vals[i][0].hi
            small_recall = min(v for _, v in vals[:i + 1])
        large = degraded[-1] and (small_edge is None or len(vals) > 1)

        first = r.size[0].recall
        out.append({"model": r.key, "label": r.label,
                    "recall_smallest_bin": first, "recall_best_bin": best,
                    "deficit": None if first is None else first - best,
                    "degrades_below_px": small_edge, "recall_there": small_recall,
                    "degrades_on_largest_bin": int(large)})
    return out


def density_trends(rows: list[ModelRow]) -> list[dict]:
    """Does the detector family change how recall responds to density?

    Trend is the slope of recall against density-band index over the bands that
    have ground truth, so it reads as points of recall gained or lost per band.
    """
    out = []
    for r in rows:
        pts = [(i, b.recall) for i, b in enumerate(r.density) if b.recall is not None]
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        family = ("two-stage box" if r.key == "fasterrcnn"
                  else "point" if r.kind == "point" else "single-stage box")
        out.append({"model": r.key, "label": r.label, "family": family,
                    "recall_slope_per_band": float(np.polyfit(xs, ys, 1)[0]),
                    "recall_sparsest": float(ys[0]), "recall_densest": float(ys[-1]),
                    "count_error_densest": r.density[-1].signed_pct_error})
    return out


def _write_dicts(path: str, rows: list[dict]) -> str:
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path


def write_task_csvs(rows: list[ModelRow], ctx: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []

    p = os.path.join(out_dir, "accuracy_efficiency.csv")
    # n_images/n_gt travel with the numbers: this CSV sits beside metrics.csv, which
    # was scored on a different, no-longer-reproducible split. A reader must be able
    # to tell them apart from the file alone.
    fields = ["key", "label", "kind", "run_id", "n_images", "n_gt", "ap", "f1", "precision", "recall",
              "mae", "rmse", "t_star", "signed_pct_error", "total_pred", "params", "params_m",
              "checkpoint_mb", "peak_gpu_mb", "ms_per_crop", "ms_per_crop_std",
              "crops_per_second", "preprocess_ms", "postprocess_ms", "train_hours",
              "train_peak_gpu_mb", "eff_source_run", "train_gpu", "train_host"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: getattr(r, k, None) for k in fields}
            row["n_images"] = ctx["n_images"]
            row["n_gt"] = ctx["n_gt"]
            w.writerow(row)
    written.append(p)

    p = os.path.join(out_dir, "by_size.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "test_images", "size_bin_px", "lo_px", "hi_px",
                    "n_gt", "recall", "fnr", "precision", "f1", "n_det", "count_ratio"])
        for r in rows:
            for b in r.size:
                w.writerow([r.label, r.kind, ctx["n_images"], b.label, b.lo, b.hi,
                            b.n_gt, b.recall, b.fnr, b.precision, b.f1, b.n_det,
                            b.count_ratio])
    written.append(p)

    p = os.path.join(out_dir, "by_density.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "test_images", "density_bin", "lo", "hi",
                    "n_crops", "n_gt", "recall", "precision", "f1", "mae", "rmse",
                    "signed_pct_error", "pred_total"])
        for r in rows:
            for b in r.density:
                w.writerow([r.label, r.kind, ctx["n_images"], b.label, b.lo, b.hi,
                            b.n_crops, b.n_gt,
                            b.recall, b.precision, b.f1, b.mae, b.rmse,
                            b.signed_pct_error, b.pred_total])
    written.append(p)

    for name, table in (("pareto_frontiers.csv", pareto_membership(rows)),
                        ("budget_picks.csv", budget_picks(rows)),
                        ("size_degradation.csv", size_degradation(rows)),
                        ("density_trend.csv", density_trends(rows))):
        written.append(_write_dicts(os.path.join(out_dir, name), table))
    return written
