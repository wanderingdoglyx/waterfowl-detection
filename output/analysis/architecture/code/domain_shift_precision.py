#!/usr/bin/env python
"""
How large an Arkansas test set has to be before a Missouri-to-Arkansas drop is measurable.

A domain-shift experiment reports one number per model on new imagery. That number carries
sampling error, and the sampling unit is the *photograph*, not the crop: crops cut from one
photograph share its habitat, altitude, flock and annotator, so treating them as independent
understates the error badly.

Method. The matching is computed once per crop on the Missouri test split from the cached
predictions, exactly as the published protocol does (tau = 40 px, greedy, score-ordered).
Photographs are then resampled with replacement (`--draws` per size), and for each draw:

    AP    threshold-free, over the crops of the drawn photographs
    MAE   at each model's OWN published t*, transferred rather than re-fitted, which is
          what a zero-shot Arkansas run must do (Section 5 of the report)

The spread across draws is the standard error of a survey of that many photographs, so the
95% interval half-width is 1.96 x SD. This says how precise an Arkansas result would be IF
Arkansas photographs resemble Missouri ones in birds per photograph and difficulty; it is a
planning figure, not a guarantee.

    python output/analysis/architecture/code/domain_shift_precision.py

Writes results/domain_shift_precision.csv.
"""

from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

from analysis import metadata, predict, registry, rescore
from data_prep import config
from data_prep.point_metrics import _ap_from_flags, _match_image, gt_points_from_coco

MODELS = [("yolov5", "2026-07-11_13-18-41"), ("fasterrcnn", "2026-06-27_16-56-51")]
SIZES = (2, 5, 10, 20, 40, 80, 160, 320)
DRAWS = 200
OUT_CSV = os.path.join(HERE, "..", "results", "domain_shift_precision.csv")


def per_crop_arrays(spec, run_id: str, gt: dict) -> tuple[list, list, np.ndarray]:
    """Per crop: detection scores (descending), true-positive flags, ground-truth count."""
    cache = predict.load(predict.cache_path(spec, run_id))
    dets = rescore.dets_by_image(cache)
    keys = list(gt.keys())
    scores, tp = [], []
    n_gt = np.zeros(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        gt_xy = np.asarray(gt[k], dtype=np.float64).reshape(-1, 2)
        n_gt[i] = len(gt_xy)
        d = np.asarray(dets.get(k, ()), dtype=np.float64).reshape(-1, 3)
        d = d[np.argsort(-d[:, 2], kind="stable")]
        scores.append(d[:, 2])
        tp.append(_match_image(gt_xy, d[:, :2], config.PAPER_TAU))
    return scores, tp, n_gt


def photograph_groups(keys: list[int]) -> list[np.ndarray]:
    """Crop indices grouped by the photograph they were cut from."""
    metas = metadata.load(predict.TEST_JSON)
    by_photo: dict[tuple[str, str], list[int]] = {}
    for i, k in enumerate(keys):
        m = metas.get(k)
        photo = (m.dataset, m.stem) if m else ("?", str(k))
        by_photo.setdefault(photo, []).append(i)
    return [np.asarray(v) for v in by_photo.values()]


def evaluate(idx: np.ndarray, scores, tp, n_gt, counts_at_t) -> tuple[float, float]:
    """(AP, MAE at the transferred threshold) over the given crops, with multiplicity."""
    ap, _, _, _ = _ap_from_flags(
        np.concatenate([scores[i] for i in idx]) if len(idx) else np.array([]),
        np.concatenate([tp[i] for i in idx]) if len(idx) else np.array([], dtype=bool),
        int(n_gt[idx].sum()),
    )
    return ap * 100.0, float(np.abs(counts_at_t[idx] - n_gt[idx]).mean())


def main() -> None:
    gt = gt_points_from_coco(predict.TEST_JSON)
    keys = list(gt.keys())
    groups = photograph_groups(keys)
    print(f"{len(keys):,} crops from {len(groups)} photographs "
          f"({sum(len(g) for g in groups) / len(groups):.0f} crops each)")

    rows = []
    for key, run_id in MODELS:
        spec = registry.resolve(key)
        scores, tp, n_gt = per_crop_arrays(spec, run_id, gt)

        with open(rescore.score_path(spec, run_id)) as f:
            t_star = float(json.load(f)["counting"]["t_star"])
        # predicted count per crop at the transferred threshold
        counts = np.array([int((s >= t_star).sum()) for s in scores], dtype=np.int64)

        full_ap, full_mae = evaluate(np.arange(len(keys)), scores, tp, n_gt, counts)
        print(f"\n{spec.label}: whole split AP {full_ap:.2f}, MAE at t*={t_star:.3f} {full_mae:.3f}")

        rng = np.random.default_rng(config.RANDOM_SEED)
        for n_photos in SIZES:
            aps, maes, birds = [], [], []
            for _ in range(DRAWS):
                picked = rng.integers(0, len(groups), size=n_photos)
                idx = np.concatenate([groups[p] for p in picked])
                a, m = evaluate(idx, scores, tp, n_gt, counts)
                aps.append(a)
                maes.append(m)
                birds.append(int(n_gt[idx].sum()))
            row = {
                "model": spec.label, "n_photographs": n_photos,
                "median_birds": int(np.median(birds)),
                "ap_halfwidth_pts": round(1.96 * float(np.std(aps, ddof=1)), 2),
                "mae_halfwidth": round(1.96 * float(np.std(maes, ddof=1)), 3),
                "whole_split_ap": round(full_ap, 2), "whole_split_mae": round(full_mae, 3),
            }
            rows.append(row)
            print(f"  {n_photos:4d} photographs (~{row['median_birds']:6,d} birds): "
                  f"AP +/- {row['ap_halfwidth_pts']:5.2f} pts, "
                  f"MAE +/- {row['mae_halfwidth']:.3f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {os.path.normpath(OUT_CSV)}")


if __name__ == "__main__":
    main()
