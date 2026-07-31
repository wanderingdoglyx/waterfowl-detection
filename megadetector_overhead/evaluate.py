"""
Turn OWL's point detections into the project's two comparison metrics.

OWL is a point detector, so it is scored two complementary ways:

  1. Native point metric (precision / recall / F1) — computed inside the MDO env by
     megadetector_overhead/_eval_owl.py and read back here from its JSON.  A prediction
     is a true positive when it falls within MDO_POINT_RADIUS of a ground-truth point.
     This is the honest metric for a point model.

  2. Pseudo-box mAP30 — each detected point is wrapped in an MDO_PSEUDO_BOX-px square and
     run through the *identical* pycocotools COCOeval@IoU=0.30 that Faster R-CNN, YOLO-NAS
     and YOLOv5 use (see yolov5/evaluate.py:coco_map30).  This lets OWL sit in the same
     mAP30 table as the other three, with the caveat that a point has no real extent — the
     absolute number depends on the (fixed) pseudo-box size, so read it as an approximate
     cross-model bridge, not a like-for-like detector score.
"""

from __future__ import annotations

import json
import os
from contextlib import redirect_stdout

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import data_prep.config as config


def coco_map30(coco_gt_path: str, coco_results: list) -> tuple[float, float]:
    """(AP30, AR30) as percentages — same extraction as yolov5/evaluate.py (IoU=0.30)."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        coco_gt = COCO(coco_gt_path)

    if not coco_results:
        return 0.0, 0.0

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array([config.IOU_THRESHOLD_MAP])
        coco_eval.evaluate()
        coco_eval.accumulate()

    precision = coco_eval.eval.get("precision")
    recall = coco_eval.eval.get("recall")

    if precision is not None and precision.size > 0:
        p = precision[0, :, :, 0, 2]
        valid = p[p >= 0]
        ap30 = float(np.mean(valid)) * 100.0 if valid.size > 0 else 0.0
    else:
        ap30 = 0.0

    if recall is not None and recall.size > 0:
        r = recall[0, :, 0, 2]
        valid = r[r >= 0]
        ar30 = float(np.mean(valid)) * 100.0 if valid.size > 0 else 0.0
    else:
        ar30 = 0.0

    return ap30, ar30


def points_to_coco(detections: list, coco_gt_path: str,
                   box_size: int = None, crop_size: int = None) -> list:
    """
    Convert OWL point detections (full-res crop pixels) to COCO-format pseudo-box results.

    `detections` is the list from _eval_owl.py's JSON: [{image, x, y, score}, ...] where
    `image` is the crop basename.  Each point becomes a `box_size`-px square centred on it,
    clipped to the crop, tagged with the GT image_id so COCOeval can match by image.
    """
    box_size = box_size if box_size is not None else config.MDO_PSEUDO_BOX
    crop_size = crop_size if crop_size is not None else config.CROP_SIZE

    with open(coco_gt_path) as f:
        coco = json.load(f)
    base_to_id = {os.path.basename(img["file_name"]): img["id"] for img in coco["images"]}

    half = box_size / 2.0
    results = []
    for det in detections:
        image_id = base_to_id.get(det["image"])
        if image_id is None:
            continue
        cx, cy = det["x"], det["y"]
        x1 = max(0.0, cx - half)
        y1 = max(0.0, cy - half)
        x2 = min(float(crop_size), cx + half)
        y2 = min(float(crop_size), cy + half)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(det["score"]),
        })
    return results


def evaluate_from_json(owl_eval_json: str, test_coco_path: str) -> dict:
    """
    Read _eval_owl.py's output and produce both metrics.

    Returns {point: {...}, pseudo_box_map30: {ap30, ar30, box_size}, n_detections}.
    """
    with open(owl_eval_json) as f:
        data = json.load(f)

    detections = data.get("detections", [])
    coco_results = points_to_coco(detections, test_coco_path)
    ap30, ar30 = coco_map30(test_coco_path, coco_results)

    return {
        "point": data.get("point_metrics", {}),
        "pseudo_box_map30": {
            "ap30": ap30, "ar30": ar30, "box_size": config.MDO_PSEUDO_BOX,
        },
        "n_detections": len(detections),
    }
