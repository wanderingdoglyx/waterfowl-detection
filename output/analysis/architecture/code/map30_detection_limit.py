#!/usr/bin/env python
"""
mAP30 / mAR30 before and after raising COCOeval's detection budget from 100 to 500 per crop.

Re-scores the cached test-split predictions of all eight detectors (analysis/predict.py) with
the project's own mAP30 functions at both budgets, so no inference is re-run:

    box detectors   yolov5/evaluate.py:coco_map30 on the cached boxes
    OWL variants    megadetector_overhead/evaluate.py:points_to_coco + coco_map30 (pseudo-boxes)

Also reports the recall ceiling a 100-detection budget imposes on this split: in a crop with
n > 100 annotated birds at most 100 can be matched.

    python output/analysis/architecture/code/map30_detection_limit.py

Writes results/map30_detection_limit.csv beside this folder.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

from analysis import manifest, predict, registry
from data_prep import config
from megadetector_overhead.evaluate import coco_map30 as owl_map30, points_to_coco
from yolov5.evaluate import coco_map30 as box_map30

MANIFEST = os.path.join(REPO, "models_v1.docx")
OUT_CSV = os.path.join(HERE, "..", "results", "map30_detection_limit.csv")
LIMITS = (100, config.MAX_DETECTIONS_PER_IMAGE)


def box_results(cache: dict) -> list[dict]:
    """Cached centre/size arrays -> COCO result list, as the box evaluations produce it."""
    return [{"image_id": int(i), "category_id": 1,
             "bbox": [float(x - w / 2), float(y - h / 2), float(w), float(h)],
             "score": float(s)}
            for i, x, y, w, h, s in zip(cache["image_id"], cache["x"], cache["y"],
                                        cache["w"], cache["h"], cache["score"])]


def recall_ceiling(test_json: str, limit: int) -> tuple[int, int]:
    """(birds a `limit`-detection budget can never match, all birds) on this split."""
    with open(test_json) as f:
        per_crop = Counter(a["image_id"] for a in json.load(f)["annotations"])
    return sum(max(0, n - limit) for n in per_crop.values()), sum(per_crop.values())


def main() -> None:
    test_json = predict.TEST_JSON
    lost, total = recall_ceiling(test_json, 100)
    print(f"Test split: {total:,} birds; a 100-detection budget can never match {lost:,} "
          f"({lost / total:.1%}) -> AR30 ceiling {1 - lost / total:.2%}")

    rows = []
    for entry in manifest.load(MANIFEST):
        spec = registry.resolve(entry.key)
        if spec.kind == "point":
            owl_json = os.path.join(predict.PRED_DIR, f"{spec.key}__{entry.run_id}__owl_eval.json")
            with open(owl_json) as f:
                results = points_to_coco(json.load(f)["detections"], test_json)
            score = owl_map30
        else:
            results = box_results(predict.load(predict.cache_path(spec, entry.run_id)))
            score = box_map30

        row = {"model": spec.label, "kind": spec.kind, "run": entry.run_id,
               "n_detections": len(results)}
        for limit in LIMITS:
            ap, ar = score(test_json, results, max_dets=limit)
            row[f"map30_at_{limit}"] = round(ap, 2)
            row[f"mar30_at_{limit}"] = round(ar, 2)
        rows.append(row)
        print(f"  {spec.label:14s} mAP30 {row['map30_at_100']:6.2f} -> {row[f'map30_at_{LIMITS[1]}']:6.2f}"
              f"   mAR30 {row['mar30_at_100']:6.2f} -> {row[f'mar30_at_{LIMITS[1]}']:6.2f}", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {os.path.normpath(OUT_CSV)}")


if __name__ == "__main__":
    main()
