"""
Per-family zero-shot detectors for the pretrained-baseline pipeline.

Each adapter takes a baseline spec from config.BASELINE_MODELS and returns detections in
ONE shared format — a COCO result list:

    {"image_id": int, "category_id": 1, "bbox": [x, y, w, h], "score": float}

which is exactly what yolov5/evaluate.py:coco_map30 and data_prep/point_metrics.py already
consume.  Producing that list is the *only* family-specific work in this package; scoring
is then byte-identical to what the fine-tuned runs do, which is the whole point — a
"before" number that differs from the "after" number only by the weights.

The COCO-pretrained box models (Ultralytics, YOLO-NAS, Faster R-CNN) predict over 80
classes, so each adapter resolves the index of the COCO class named
config.BASELINE_COCO_CLASS from the model's OWN label map rather than hardcoding 14, and
keeps only those detections (unless any_class=True, which keeps everything as a diagnostic
— see baseline/main.py --any-class).  OWL is handled separately in main.py because it runs
out-of-process in its own uv venv.

These adapters keep BOXES, not just point centres, because mAP30 needs them — COCOeval
requires boxes and the split's own image_ids.  They also keep the model's class labels,
so one category can be selected out of an 80-class COCO head; a fine-tuned single-class
checkpoint has no such need.  Both properties are what a zero-shot baseline requires.
"""

from __future__ import annotations

import json
import os

import data_prep.config as config


def _coco_class_index(names, wanted: str) -> int:
    """Index of `wanted` in a model's label map (dict or list).  Raises if absent."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for idx, name in items:
        if str(name).lower() == wanted.lower():
            return int(idx)
    raise KeyError(f"Pretrained model has no '{wanted}' class; labels = {names}")


def _load_test_images(test_json: str) -> list[dict]:
    with open(test_json) as f:
        return json.load(f)["images"]


# ── Ultralytics: YOLOv5m / YOLO11m / YOLO26m ──────────────────────────────────
def detect_ultralytics(spec: dict, test_json: str, any_class: bool = False) -> list:
    """
    Zero-shot COCO-pretrained Ultralytics detections.

    Delegates the actual loop to yolov5.evaluate.predictions_to_coco so the baseline
    inherits its bs=1 NMS-time-budget fix (batching silently drops all detections on
    dense flock crops) and its "read pixels from the shared crops root" behaviour.
    """
    from ultralytics import YOLO

    from yolov5.evaluate import predictions_to_coco

    model = YOLO(spec["weights"])
    classes = None if any_class else [_coco_class_index(model.names,
                                                        config.BASELINE_COCO_CLASS)]
    return predictions_to_coco(
        model, test_json, conf=config.CONF_THRESHOLD,
        iou=spec["nms_iou"], max_det=spec["max_pred"], imgsz=config.CROP_SIZE,
        classes=classes,
    )


# ── YOLO-NAS (super-gradients) ────────────────────────────────────────────────
def detect_yolonas(spec: dict, test_json: str, any_class: bool = False) -> list:
    """
    Zero-shot COCO-pretrained YOLO-NAS detections.

    Mirrors the chunked predict pass in yolo_nas/main.py --eval, including
    fuse_model=False (the default re-fuses a deep copy of the model on EVERY call).
    Unlike the fine-tuned run this keeps the class labels, so bird detections can be
    separated from the other 79 COCO categories.
    """
    from super_gradients.training import models

    model = models.get(config.YOLONAS_MODEL, pretrained_weights=spec["weights"])
    images = _load_test_images(test_json)
    paths = [os.path.join(config.CROPS_IMG_DIR, im["file_name"]) for im in images]
    ids = [im["id"] for im in images]

    results: list = []
    bird_idx: int | None = None
    chunk = 64
    for i in range(0, len(paths), chunk):
        preds = model.predict(paths[i:i + chunk], conf=config.CONF_THRESHOLD,
                              iou=spec["nms_iou"], max_predictions=spec["max_pred"],
                              fuse_model=False)
        preds = preds if hasattr(preds, "__iter__") else [preds]
        for image_id, pred in zip(ids[i:i + chunk], preds):
            if bird_idx is None and not any_class:
                bird_idx = _coco_class_index(pred.class_names,
                                             config.BASELINE_COCO_CLASS)
            p = pred.prediction
            for (x1, y1, x2, y2), score, label in zip(p.bboxes_xyxy, p.confidence,
                                                      p.labels):
                if not any_class and int(label) != bird_idx:
                    continue
                results.append({
                    "image_id": int(image_id), "category_id": 1,
                    "bbox": [float(x1), float(y1),
                             float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
    return results


# ── Faster R-CNN (detectron2) ─────────────────────────────────────────────────
def detect_fasterrcnn(spec: dict, test_json: str, any_class: bool = False) -> list:
    """
    Zero-shot COCO-pretrained Faster R-CNN R50-FPN detections.

    Deliberately uses the STOCK COCO config, not faster_rcnn/model.py:build_cfg.  That
    builder retunes the anchor generator to [8,16,32,64,128] for tiny birds and sets
    NUM_CLASSES=1 — correct for fine-tuning, but it would leave the pretrained RPN and
    box head reading weights that were learned against the default anchors and 80
    classes.  A baseline has to run the released model as released; only the test-time
    score threshold, input size and detection cap are aligned with our --eval.
    """
    import cv2
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(spec["weights"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(spec["weights"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.CONF_THRESHOLD
    cfg.TEST.DETECTIONS_PER_IMAGE = spec["max_pred"]
    cfg.INPUT.MIN_SIZE_TEST = config.CROP_SIZE
    cfg.INPUT.MAX_SIZE_TEST = config.CROP_SIZE

    predictor = DefaultPredictor(cfg)
    thing_classes = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
    bird_idx = None if any_class else _coco_class_index(thing_classes,
                                                        config.BASELINE_COCO_CLASS)

    results: list = []
    for img in _load_test_images(test_json):
        im = cv2.imread(os.path.join(config.CROPS_IMG_DIR, img["file_name"]))
        if im is None:
            continue
        inst = predictor(im)["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        labels = inst.pred_classes.numpy()
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            if bird_idx is not None and int(label) != bird_idx:
                continue
            results.append({
                "image_id": int(img["id"]), "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    return results


DETECTORS = {
    "ultralytics": detect_ultralytics,
    "yolonas":     detect_yolonas,
    "fasterrcnn":  detect_fasterrcnn,
}
