#!/usr/bin/env python
"""
Waterfowl YOLOv5 pipeline (Ultralytics).

A drop-in sibling of yolo_nas/main.py and faster_rcnn/main.py — same data, same
mAP30 protocol (IoU=0.30), same timestamped-run layout — using Ultralytics YOLOv5.

Usage (from the rebuild/ directory):

    # Step 1 — crop all datasets, build COCO splits, and mirror them into the
    # Ultralytics YOLO layout (symlinks + txt labels).  Reuses the shared crops;
    # if you already ran another model's --prepare this only adds the YOLO mirror.
    ./yolov5/main.py --prepare

    # Step 2 — train (results saved to output/checkpoints/yolov5/<timestamp>/)
    ./yolov5/main.py --train

    # Step 3 — evaluate best checkpoint on the test set (uses latest run by default)
    ./yolov5/main.py --eval
    ./yolov5/main.py --eval --run 2026-06-28_10-30-00

    # All-in-one
    ./yolov5/main.py --prepare --train --eval
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

# Project root = two levels up from this file (rebuild/yolov5/main.py → rebuild/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import data_prep.config as config
from data_prep.prepare_data import prepare


def make_run_dir() -> str:
    """Create and return a timestamped subdirectory under YOLOV5_CKPT_DIR."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.YOLOV5_CKPT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def latest_run_dir() -> str | None:
    """Return the most recently created timestamped run dir, or None."""
    if not os.path.exists(config.YOLOV5_CKPT_DIR):
        return None
    runs = sorted(
        (e.path for e in os.scandir(config.YOLOV5_CKPT_DIR) if e.is_dir()),
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

        # Right panel: predictions (red) — only confident boxes, as for the others
        pred_panel = img_bgr.copy()
        res = model.predict(path, conf=config.DISPLAY_CONF_THRESHOLD,
                            iou=config.YOLOV5_NMS_IOU, imgsz=config.CROP_SIZE,
                            verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes  = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            _draw_boxes(pred_panel, boxes, scores, color=(0, 0, 255))

        combined = np.concatenate([gt_panel, pred_panel], axis=1)
        fname = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".jpg"
        cv2.imwrite(os.path.join(out_dir, fname), combined)
        saved += 1

    print(f"Saved {saved} examples to {out_dir}/  "
          f"(left = GT, right = predictions)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Waterfowl YOLOv5 (Ultralytics)")
    parser.add_argument("--prepare", action="store_true",
                        help="Pre-process datasets into COCO crops + YOLO mirror")
    parser.add_argument("--train",   action="store_true",
                        help="Train YOLOv5 on the prepared crops")
    parser.add_argument("--eval",    action="store_true",
                        help="Evaluate the best checkpoint on the test set (mAP30)")
    parser.add_argument("--run",      default=None, metavar="TIMESTAMP",
                        help="Run timestamp to evaluate (default: latest)")
    parser.add_argument("--examples", type=int, default=40, metavar="N",
                        help="Number of example images to save during --eval (default: 40, 0 to skip)")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.eval):
        parser.print_help()
        return

    # ── 1. Data preparation ────────────────────────────────────────────────────
    if args.prepare:
        from yolov5.dataset import build_yolo_dataset

        print("\n=== Data Preparation ===")
        prepare()
        print("Building Ultralytics YOLO dataset (symlinks + labels)...")
        _, counts = build_yolo_dataset()
        print(f"YOLO mirror: {counts}")

    # ── 2. Training ─────────────────────────────────────────────────────────────
    if args.train:
        from yolov5.dataset import build_yolo_dataset
        from yolov5.train import train as run_training

        print("\n=== Training ===")
        n_train = count_crops("train")
        if n_train == 0:
            print("No training crops found. Run --prepare first.")
            return

        run_dir = make_run_dir()
        print(f"Run directory: {run_dir}")

        print("Building Ultralytics YOLO dataset (symlinks + labels)...")
        data_yaml, counts = build_yolo_dataset()
        if counts.get("val", 0) == 0:
            print("No validation crops found. Run --prepare first.")
            return

        run_info = {
            "timestamp": os.path.basename(run_dir),
            "model":     config.YOLOV5_MODEL,
            "n_train":   counts.get("train", 0),
            "n_val":     counts.get("val", 0),
            "n_test":    counts.get("test", 0),
        }
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        print(f"Training crops: {n_train}  | Model: {config.YOLOV5_MODEL}  "
              f"| Max epochs: {config.NUM_EPOCHS}  "
              f"| Early-stop patience: {config.EARLY_STOP_PATIENCE}")

        ckpt = run_training(run_dir, data_yaml=data_yaml)
        print(f"Best checkpoint: {ckpt}")

    # ── 3. Evaluation on the test set (mAP30) ──────────────────────────────────
    if args.eval:
        from ultralytics import YOLO
        from yolov5.evaluate import coco_map30, predictions_to_coco

        print("\n=== Test-set Evaluation ===")

        eval_dir = os.path.join(config.YOLOV5_CKPT_DIR, args.run) if args.run \
            else latest_run_dir()
        if not eval_dir or not os.path.exists(eval_dir):
            print("No run directory found. Run --train first.")
            return

        best_ckpt = os.path.join(eval_dir, "weights", "best.pt")
        ckpt = best_ckpt if os.path.exists(best_ckpt) \
            else os.path.join(eval_dir, "weights", "last.pt")
        if not os.path.exists(ckpt):
            print(f"No checkpoint found in {eval_dir}. Run --train first.")
            return

        test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
        if not os.path.exists(test_json):
            print("No test crops found. Run --prepare first.")
            return

        print(f"Evaluating run: {eval_dir}")
        print(f"Loading checkpoint: {ckpt}")

        model = YOLO(ckpt)

        # Low score threshold so COCOeval sees the full PR curve (mirrors the
        # CONF_THRESHOLD used by Faster R-CNN / YOLO-NAS during mAP evaluation).
        coco_results = predictions_to_coco(model, test_json, conf=config.CONF_THRESHOLD)
        ap30, ar30 = coco_map30(test_json, coco_results)
        print(f"\nTest mAP30 : {ap30:.2f}%")
        print(f"Test mAR30 : {ar30:.2f}%")

        if args.examples > 0:
            print(f"\nSaving {args.examples} example images...")
            _save_examples(model, eval_dir=eval_dir, n=args.examples)


if __name__ == "__main__":
    main()
