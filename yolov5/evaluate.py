"""
mAP30 evaluation for YOLOv5 predictions (IoU threshold = 0.30, per Section 4.4).

Ultralytics computes its own mAP@0.50 / mAP@0.50:0.95 internally, but the project
compares all detectors on a single, lower IoU threshold (0.30) for the tiny birds.
To stay apples-to-apples with Faster R-CNN's MAP30Evaluator and YOLO-NAS's
DetectionMetrics(iou_thres=0.30), we run pycocotools' COCOeval at iouThrs=[0.30]
and pull AP/AR straight from the accumulated arrays — exactly the extraction used
in faster_rcnn/evaluator.py.

`coco_results` must be a list of COCO-format detection dicts:
    {"image_id": int, "category_id": 1, "bbox": [x, y, w, h], "score": float}
with image_id / category_id matching the ground-truth coco.json.
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import data_prep.config as config


def coco_map30(coco_gt_path: str, coco_results: list) -> tuple[float, float]:
    """
    Returns (AP30, AR30) as percentages for the given GT json and detections.

    AP/AR are averaged over categories and (for the "all" area range, 100-detection
    budget) read directly from COCOeval's precision/recall arrays, bypassing
    summarize() which assumes the standard 0.50:0.95 IoU sweep.
    """
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

    # precision: [T=1, R=101, K, A=4, M=3];  recall: [T=1, K, A=4, M=3]
    # area index 0 = "all";  maxDets index 2 = 100 detections.
    precision = coco_eval.eval.get("precision")
    recall    = coco_eval.eval.get("recall")

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


def predictions_to_coco(model, coco_gt_path: str, conf: float) -> list:
    """
    Run the Ultralytics model over every image in `coco_gt_path` and return the
    detections as a COCO result list (category_id fixed to 1 = bird).

    Images are looked up on disk via the shared crops root so we score the exact
    same pixels as the ground truth, independent of the symlinked YOLO mirror.

    ONE IMAGE PER predict() CALL (bs=1) — this is deliberate, not a perf oversight.
    Ultralytics NMS uses a *cumulative per-batch* time budget (utils/nms.py:
    `time_limit = 2.0 + 0.05 * bs`); when a dense crop trips it, the loop `break`s
    and every remaining image in that batch is returned with ZERO detections. At the
    low eval conf (0.05) our dense flock crops (300+ birds) reliably trip it: an
    8-image batch of them returned 7 empties (all their birds silently became false
    negatives), while at bs=1 all 8 returned full detections. Passing a list — even
    with stream=True/batch=1 — still batches the NMS, so we must call predict per
    image. bs=1 also keeps GPU memory trivial. Slower, but correct.
    """
    import json

    with open(coco_gt_path) as f:
        coco = json.load(f)

    results: list = []
    for img in coco["images"]:
        path = os.path.join(config.CROPS_IMG_DIR, img["file_name"])
        preds = model.predict(
            source=path,
            conf=conf,
            iou=config.YOLOV5_NMS_IOU,
            imgsz=config.CROP_SIZE,
            max_det=config.YOLOV5_MAX_PRED,
            stream=False,
            verbose=False,
        )
        boxes = preds[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy   = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, scores):
            results.append({
                "image_id":    int(img["id"]),
                "category_id": 1,
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
            })
    return results
