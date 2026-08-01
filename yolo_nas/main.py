#!/usr/bin/env python
"""
Waterfowl YOLO-NAS pipeline (super-gradients).

A drop-in sibling of faster_rcnn/main.py — same data, same mAP30 protocol,
same timestamped-run layout — but using YOLO-NAS instead of Faster R-CNN.

Usage (from the rebuild/ directory):

    # Step 1 — crop all datasets and build COCO JSON splits (one-time).
    # This is identical to the Faster R-CNN prep; if you already ran
    # ./faster_rcnn/main.py --prepare you can skip it.
    ./yolo_nas/main.py --prepare

    # Step 2 — train (results saved to output/checkpoints/yolonas/<timestamp>/)
    ./yolo_nas/main.py --train

    # Step 3 — evaluate best checkpoint on the test set (uses latest run by default)
    ./yolo_nas/main.py --eval
    ./yolo_nas/main.py --eval --run 2026-06-28_10-30-00

    # All-in-one
    ./yolo_nas/main.py --prepare --train --eval
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

# Project root = two levels up from this file (rebuild/yolo_nas/main.py → rebuild/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import data_prep.config as config
from data_prep.prepare_data import prepare


def make_run_dir() -> str:
    """Create and return a timestamped subdirectory under YOLONAS_CKPT_DIR."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.YOLONAS_CKPT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def latest_run_dir() -> str | None:
    """Return the most recently created timestamped run dir, or None."""
    if not os.path.exists(config.YOLONAS_CKPT_DIR):
        return None
    runs = sorted(
        (e.path for e in os.scandir(config.YOLONAS_CKPT_DIR) if e.is_dir()),
        reverse=True,
    )
    return runs[0] if runs else None


def count_crops(split: str) -> int:
    coco_path = os.path.join(config.CROPS_JSON_DIR, split, "coco.json")
    if not os.path.exists(coco_path):
        return 0
    with open(coco_path) as f:
        return len(json.load(f)["images"])


def _draw_boxes(img_bgr, boxes_xyxy, scores=None, color=(0, 0, 255)):
    """Draw xyxy boxes (and optional scores) on a BGR image in place; return it."""
    import cv2
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img_bgr, p1, p2, color, 1)
        if scores is not None:
            cv2.putText(img_bgr, f"{scores[i]:.2f}", (p1[0], max(p1[1] - 2, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return img_bgr


def _save_examples(model, eval_dir: str, n: int) -> None:
    """Save n side-by-side (ground truth | prediction) images to eval_dir/examples/."""
    import random
    import cv2
    import numpy as np

    test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
    with open(test_json) as f:
        coco = json.load(f)

    # image_id → list of GT xyxy boxes
    by_image = {img["id"]: [] for img in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image[a["image_id"]].append([x, y, x + w, y + h])

    images = random.sample(coco["images"], min(n, len(coco["images"])))
    out_dir = os.path.join(eval_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for img in images:
        path = os.path.join(config.CROPS_IMG_DIR, img["file_name"])
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        # Left panel: ground truth (green)
        gt_panel = _draw_boxes(img_bgr.copy(), by_image.get(img["id"], []),
                               color=(0, 255, 0))

        # Right panel: predictions (red) — only confident boxes, as for Faster R-CNN.
        # model.predict() on a single image returns one ImageDetectionPrediction
        # (not an iterable) in this super-gradients version.
        pred_panel = img_bgr.copy()
        pred = model.predict(path, conf=config.DISPLAY_CONF_THRESHOLD,
                             fuse_model=False)  # avoid re-fusing per example image
        _draw_boxes(pred_panel, pred.prediction.bboxes_xyxy,
                    pred.prediction.confidence, color=(0, 0, 255))

        combined = np.concatenate([gt_panel, pred_panel], axis=1)
        fname = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".jpg"
        cv2.imwrite(os.path.join(out_dir, fname), combined)
        saved += 1

    print(f"Saved {saved} examples to {out_dir}/  "
          f"(left = GT, right = predictions)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Waterfowl YOLO-NAS (super-gradients)")
    parser.add_argument("--prepare", action="store_true",
                        help="Pre-process all datasets into 512×512 COCO crops")
    parser.add_argument("--train",   action="store_true",
                        help="Train YOLO-NAS on the prepared crops")
    parser.add_argument("--eval",    action="store_true",
                        help="Evaluate the best checkpoint on the test set")
    parser.add_argument("--run",      default=None, metavar="TIMESTAMP",
                        help="Run timestamp to evaluate (default: latest)")
    parser.add_argument("--examples", type=int, default=40, metavar="N",
                        help="Number of example images to save during --eval (default: 40, 0 to skip)")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.eval):
        parser.print_help()
        return

    # ── 1. Data preparation (identical to the Faster R-CNN crops) ──────────────
    if args.prepare:
        print("\n=== Data Preparation ===")
        prepare()

    # ── 2. Training ────────────────────────────────────────────────────────────
    if args.train:
        from yolo_nas.dataset import build_dataloaders
        from yolo_nas.train import train as run_training

        print("\n=== Training ===")
        n_train = count_crops("train")
        if n_train == 0:
            print("No training crops found. Run --prepare first.")
            return

        run_dir = make_run_dir()
        print(f"Run directory: {run_dir}")

        run_info = {
            "timestamp": os.path.basename(run_dir),
            "model":     config.YOLONAS_MODEL,
            "n_train":   n_train,
            "n_val":     count_crops("val"),
            "n_test":    count_crops("test"),
        }
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        loaders = build_dataloaders()
        if "val" not in loaders:
            print("No validation crops found. Run --prepare first.")
            return
        print(f"Training crops: {n_train}  | Model: {config.YOLONAS_MODEL}  "
              f"| Max epochs: {config.NUM_EPOCHS}  "
              f"| Early-stop patience: {config.EARLY_STOP_PATIENCE}")

        ckpt = run_training(run_dir, loaders["train"], loaders["val"])
        print(f"Best checkpoint: {ckpt}")

    # ── 3. Evaluation on the test set ──────────────────────────────────────────
    if args.eval:
        from super_gradients.training import Trainer, models
        from yolo_nas.dataset import build_dataloaders
        from yolo_nas.model import build_metric, MAP_KEY, AR_KEY, NUM_CLASSES
        from yolo_nas.train import find_checkpoint

        print("\n=== Test-set Evaluation ===")

        eval_dir = os.path.join(config.YOLONAS_CKPT_DIR, args.run) if args.run \
            else latest_run_dir()
        if not eval_dir or not os.path.exists(eval_dir):
            print("No run directory found. Run --train first.")
            return

        ckpt = find_checkpoint(eval_dir, "ckpt_best.pth") \
            or find_checkpoint(eval_dir, "ckpt_latest.pth")
        if not ckpt:
            print(f"No checkpoint found in {eval_dir}. Run --train first.")
            return

        print(f"Evaluating run: {eval_dir}")
        print(f"Loading checkpoint: {ckpt}")

        loaders = build_dataloaders()
        if "test" not in loaders:
            print("No test crops found. Run --prepare first.")
            return

        model = models.get(
            config.YOLONAS_MODEL,
            num_classes=NUM_CLASSES,
            checkpoint_path=ckpt,
        )

        trainer = Trainer(
            experiment_name=os.path.basename(eval_dir),
            ckpt_root_dir=os.path.dirname(eval_dir),
        )
        results = trainer.test(
            model=model,
            test_loader=loaders["test"],
            test_metrics_list=[build_metric(NUM_CLASSES)],
        )
        results = dict(results)

        ap30 = float(results.get(MAP_KEY, 0.0)) * 100.0
        ar30 = float(results.get(AR_KEY, 0.0)) * 100.0
        print(f"\nTest mAP30 : {ap30:.2f}%")
        print(f"Test mAR30 : {ar30:.2f}%")

        # ── OWL-paper point metrics (shared protocol, Section 4.3) ────────────
        # trainer.test only returns aggregates, so run a chunked predict pass to
        # collect per-detection boxes, then score their centres against the GT
        # box centres (MAE/RMSE, AP/AUC-PR, P/R/F1 at t*, bootstrap CIs).
        from data_prep.point_metrics import (gt_points_from_coco, paper_point_metrics,
                                             format_report)

        test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
        with open(test_json) as f:
            test_coco = json.load(f)
        paths = [os.path.join(config.CROPS_IMG_DIR, im["file_name"])
                 for im in test_coco["images"]]
        ids = [im["id"] for im in test_coco["images"]]

        print(f"\nCollecting detections for point metrics ({len(paths)} crops)...")
        dets_by_image = {}
        chunk = 64  # keep preprocessing memory bounded (cf. yolov5/evaluate.py)
        for i in range(0, len(paths), chunk):
            # fuse_model=False: each predict() call builds a fresh pipeline, and the
            # default fuse_model=True deep-copies + layer-fuses the model EVERY call —
            # ~400 re-fuses over the test set (and a "Fusing some of the model's
            # layers" INFO line per chunk). Skipping fusion is far cheaper here.
            preds = model.predict(paths[i:i + chunk], conf=config.CONF_THRESHOLD,
                                  iou=config.YOLONAS_NMS_IOU,
                                  max_predictions=config.YOLONAS_MAX_PRED,
                                  fuse_model=False)
            preds = preds if hasattr(preds, "__iter__") else [preds]
            for image_id, pred in zip(ids[i:i + chunk], preds):
                boxes = pred.prediction.bboxes_xyxy
                scores = pred.prediction.confidence
                dets_by_image[image_id] = [
                    (float((x1 + x2) / 2), float((y1 + y2) / 2), float(s))
                    for (x1, y1, x2, y2), s in zip(boxes, scores)
                ]

        pm = paper_point_metrics(
            gt_points_from_coco(test_json), dets_by_image,
            tau=config.PAPER_TAU, bootstrap=config.PAPER_BOOTSTRAP,
        )
        print("\n" + format_report(pm, "YOLO-NAS"))
        with open(os.path.join(eval_dir, "paper_point_metrics.json"), "w") as f:
            json.dump(pm, f, indent=2)

        if args.examples > 0:
            print(f"\nSaving {args.examples} example images...")
            _save_examples(model, eval_dir=eval_dir, n=args.examples)


if __name__ == "__main__":
    main()
