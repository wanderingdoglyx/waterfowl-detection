"""
Re-score every model from the cached predictions, on one test split.

The Baseline-v1 runs published their metrics against an older 25,708-image test
split that can no longer be reproduced (some of its crops are gone from disk).
Tasks 2.2-2.4 need accuracy and per-detection outcomes to come from the *same*
images, so accuracy is recomputed here from ``analysis.predict``'s cache using
the project's own protocol (``data_prep.point_metrics.paper_point_metrics``),
rather than read from each run's stored JSON.

Scores are cached per model, since the bootstrap CIs are the expensive part.
"""

from __future__ import annotations

import json
import os

import numpy as np

from data_prep import config
from data_prep.point_metrics import gt_points_from_coco, paper_point_metrics
from analysis import predict
from analysis.registry import ModelSpec

SCORE_DIR = os.path.join(config.OUTPUT_DIR, "analysis", "scores")


def score_path(spec: ModelSpec, run_id: str) -> str:
    return os.path.join(SCORE_DIR, f"{spec.key}__{run_id}.json")


def dets_by_image(cache: dict) -> dict[int, list[tuple[float, float, float]]]:
    """Cache arrays -> {image_id: [(x, y, score), ...]} as the protocol expects."""
    out: dict[int, list[tuple[float, float, float]]] = {}
    for image_id, x, y, s in zip(cache["image_id"], cache["x"], cache["y"], cache["score"]):
        out.setdefault(int(image_id), []).append((float(x), float(y), float(s)))
    return out


def ensure(spec: ModelSpec, run_id: str, force: bool = False) -> dict:
    """Return this run's re-scored metrics, computing them if not cached."""
    path = score_path(spec, run_id)
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)

    cache = predict.load(predict.cache_path(spec, run_id))
    gt = gt_points_from_coco(predict.TEST_JSON)
    pm = paper_point_metrics(
        gt, dets_by_image(cache),
        tau=config.PAPER_TAU, bootstrap=config.PAPER_BOOTSTRAP,
    )
    os.makedirs(SCORE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pm, f, indent=2)
    return pm
