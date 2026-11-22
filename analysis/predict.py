"""
Build a uniform per-detection cache, so every downstream analysis scores the
same predictions instead of re-running inference.

Tasks 2.3 and 2.4 need to know *which* ground-truth bird each model found, not
just the aggregate scores — and only the OWL variants persist their detections
(``eval/owl_eval.json``).  The five box models compute detections in memory
during evaluation and discard them.  This module re-runs each model once and
stores the result in one format:

    output/analysis/predictions/<key>__<run_id>.npz
        image_id int32   COCO image id from the test split
        x, y     float32 detection centre, crop pixel coordinates
        score    float32 confidence
        w, h     float32 box width/height; NaN for the point detectors

Detections are produced exactly as each model's own evaluation produces them
(same conf threshold, NMS IoU, max detections, image size), so the cached
predictions reproduce that model's published metrics rather than a variant of
them.

All models are scored on the CURRENT test split.  The Baseline-v1 runs were
scored on an older 25,708-image split whose crops are no longer all on disk
(`Bird_I_complete/` became `Bird_I/`), so that split cannot be re-run; mixing
the two would silently compare models on different images.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from data_prep import config
from analysis.registry import ModelSpec

PRED_DIR = os.path.join(config.OUTPUT_DIR, "analysis", "predictions")
TEST_JSON = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")


def checkpoint_for(spec: ModelSpec, run_id: str) -> str:
    """The checkpoint file this model's own evaluation would load for this run.

    Shared with analysis.bench_extra so timing and predictions always describe
    the same weights — data_prep.efficiency resolves the *latest* run instead,
    which silently drifts away from whatever the manifest pins.
    """
    run_dir = os.path.join(config.OUTPUT_DIR, "checkpoints", spec.key, run_id)
    if spec.key in ("yolov5", "yolo11", "yolo26"):
        for name in ("best.pt", "last.pt"):
            c = os.path.join(run_dir, "weights", name)
            if os.path.exists(c):
                return c
    elif spec.key == "yolonas":
        import glob as _glob
        cands = _glob.glob(os.path.join(run_dir, "*", "ckpt_best.pth"))
        if cands:
            return cands[0]
    elif spec.key == "fasterrcnn":
        for name in ("model_best.pth", "model_final.pth"):
            c = os.path.join(run_dir, name)
            if os.path.exists(c):
                return c
    elif spec.kind == "point":
        for c in (os.path.join(run_dir, "weights", "best.pth"),
                  os.path.join(run_dir, "best_model.pth")):
            if os.path.exists(c):
                return c
    raise FileNotFoundError(f"no checkpoint for {spec.key} run {run_id}")


def cache_path(spec: ModelSpec, run_id: str) -> str:
    return os.path.join(PRED_DIR, f"{spec.key}__{run_id}.npz")


def _save(path: str, rows: list[tuple], n_images: int) -> str:
    """rows: (image_id, x, y, score, w, h) with w/h possibly NaN."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rows:
        arr = np.asarray(rows, dtype=np.float64)
    else:
        arr = np.zeros((0, 6), dtype=np.float64)
    np.savez_compressed(
        path,
        image_id=arr[:, 0].astype(np.int32),
        x=arr[:, 1].astype(np.float32),
        y=arr[:, 2].astype(np.float32),
        score=arr[:, 3].astype(np.float32),
        w=arr[:, 4].astype(np.float32),
        h=arr[:, 5].astype(np.float32),
        n_images=np.int32(n_images),
    )
    return path


def load(path: str) -> dict:
    """Load a cache into {image_id: (x, y, score, w, h) arrays} plus n_images."""
    z = np.load(path)
    return {
        "image_id": z["image_id"], "x": z["x"], "y": z["y"],
        "score": z["score"], "w": z["w"], "h": z["h"],
        "n_images": int(z["n_images"]),
    }


def _test_images() -> tuple[list[str], list[int]]:
    with open(TEST_JSON) as f:
        coco = json.load(f)
    paths = [os.path.join(config.CROPS_IMG_DIR, im["file_name"]) for im in coco["images"]]
    ids = [im["id"] for im in coco["images"]]
    return paths, ids


# ── Per-family inference ─────────────────────────────────────────────────────

def _run_ultralytics(ckpt: str, nms_iou: float, max_det: int) -> list[tuple]:
    """Ultralytics (YOLOv5u / YOLO11 / YOLO26) via the project's own helper.

    predictions_to_coco runs one image per predict() call on purpose — batching
    trips Ultralytics' cumulative per-batch NMS time budget and silently returns
    empty detections for dense crops. Reusing it keeps that guarantee.
    """
    from ultralytics import YOLO
    from yolov5.evaluate import predictions_to_coco

    model = YOLO(ckpt)
    results = predictions_to_coco(
        model, TEST_JSON, conf=config.CONF_THRESHOLD,
        iou=nms_iou, max_det=max_det, imgsz=config.CROP_SIZE,
    )
    rows = []
    for r in results:
        x, y, w, h = r["bbox"]
        rows.append((r["image_id"], x + w / 2.0, y + h / 2.0, float(r["score"]), w, h))
    return rows


def _run_yolonas(ckpt: str) -> list[tuple]:
    from super_gradients.training import models

    model = models.get(config.YOLONAS_MODEL, num_classes=1, checkpoint_path=ckpt)
    model = model.cuda().eval()

    paths, ids = _test_images()
    rows = []
    chunk = 64  # bounded preprocessing memory, as in yolo_nas/main.py
    for i in range(0, len(paths), chunk):
        preds = model.predict(paths[i:i + chunk], conf=config.CONF_THRESHOLD,
                              iou=config.YOLONAS_NMS_IOU,
                              max_predictions=config.YOLONAS_MAX_PRED,
                              fuse_model=False)
        preds = preds if hasattr(preds, "__iter__") else [preds]
        for image_id, pred in zip(ids[i:i + chunk], preds):
            boxes = pred.prediction.bboxes_xyxy
            scores = pred.prediction.confidence
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                rows.append((image_id, float((x1 + x2) / 2), float((y1 + y2) / 2),
                             float(s), float(x2 - x1), float(y2 - y1)))
        if (i // chunk) % 20 == 0:
            print(f"    {min(i + chunk, len(paths))}/{len(paths)} crops", flush=True)
    return rows


def _run_fasterrcnn(ckpt: str) -> list[tuple]:
    """Detectron2 inference over the current test split.

    The run's own instances_predictions.pth is not reused: it was written against
    whichever split that run evaluated, which for the v1 runs is the older one.
    """
    from detectron2.engine import DefaultPredictor
    from faster_rcnn.model import build_cfg  # same cfg the training/eval path uses

    cfg = build_cfg(use_pretrained=False)
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.CONF_THRESHOLD
    predictor = DefaultPredictor(cfg)

    import cv2
    paths, ids = _test_images()
    rows = []
    for n, (p, image_id) in enumerate(zip(paths, ids)):
        img = cv2.imread(p)
        if img is None:
            continue
        inst = predictor(img)["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            rows.append((image_id, float((x1 + x2) / 2), float((y1 + y2) / 2),
                         float(s), float(x2 - x1), float(y2 - y1)))
        if n % 2000 == 0:
            print(f"    {n}/{len(paths)} crops", flush=True)
    return rows


def _run_owl(spec: ModelSpec, ckpt: str, out_json: str) -> list[tuple]:
    """OWL variants run in their own Python 3.11 venv, as evaluation does."""
    from megadetector_overhead.main import run_eval_helper

    model_spec = config.MDO_MODELS[
        {"mdo_owl_c": "OWLC", "mdo_owl_t": "OWLT", "mdo_owl_d": "OWLD_H"}[spec.key]
    ]
    if not os.path.exists(out_json):
        run_eval_helper(ckpt, out_json, model_spec)

    with open(out_json) as f:
        payload = json.load(f)
    with open(TEST_JSON) as f:
        coco = json.load(f)
    base_to_id = {os.path.basename(i["file_name"]): i["id"] for i in coco["images"]}

    rows = []
    dropped = 0
    for d in payload["detections"]:
        image_id = base_to_id.get(os.path.basename(d["image"]))
        if image_id is None:
            dropped += 1
            continue
        rows.append((image_id, float(d["x"]), float(d["y"]), float(d["score"]),
                     float("nan"), float("nan")))
    if dropped:
        print(f"    warning: {dropped} detections on images outside the test split")
    return rows


# ── Entry point ──────────────────────────────────────────────────────────────

def ensure(spec: ModelSpec, run_id: str, force: bool = False) -> str:
    """Return the cache path for this run, computing it if absent."""
    path = cache_path(spec, run_id)
    if os.path.exists(path) and not force:
        print(f"  {spec.label:14s} cached")
        return path

    run_dir = os.path.join(config.OUTPUT_DIR, "checkpoints", spec.key, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory missing: {run_dir}")

    print(f"  {spec.label:14s} running inference...", flush=True)
    ckpt = checkpoint_for(spec, run_id)
    if spec.key in ("yolov5", "yolo11", "yolo26"):
        um = config.ULTRALYTICS_MODELS.get(spec.key)
        nms_iou = um["nms_iou"] if um else config.YOLOV5_NMS_IOU
        max_det = um["max_pred"] if um else config.YOLOV5_MAX_PRED
        rows = _run_ultralytics(ckpt, nms_iou, max_det)
    elif spec.key == "yolonas":
        rows = _run_yolonas(ckpt)
    elif spec.key == "fasterrcnn":
        rows = _run_fasterrcnn(ckpt)
    elif spec.kind == "point":
        out_json = os.path.join(PRED_DIR, f"{spec.key}__{run_id}__owl_eval.json")
        rows = _run_owl(spec, ckpt, out_json)
    else:
        raise ValueError(f"no inference path for {spec.key}")

    with open(TEST_JSON) as f:
        n_images = len(json.load(f)["images"])
    _save(path, rows, n_images)
    print(f"  {spec.label:14s} {len(rows):,} detections -> {os.path.basename(path)}")
    return path
