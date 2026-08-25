#!/usr/bin/env python
"""
Waterfowl YOLO11 / YOLO26 pipeline (Ultralytics).

A drop-in sibling of yolov5/main.py — same data, same mAP30 protocol (IoU = 0.30), same
OWL-paper point metrics, same timestamped-run layout.  One entry point serves both
generations via --model, mirroring how megadetector_overhead/main.py serves OWL-C/T/D.

Usage (from the rebuild/ directory):

    # Step 1 — crops + COCO splits + the Ultralytics mirror.  This is the SAME mirror
    # yolov5 uses (yolov5_data/), so if you have already run any --prepare you can
    # skip this entirely.
    ./yolo_ultralytics/main.py --prepare

    # Step 2 — train (results saved to output/checkpoints/yolo11|yolo26/<timestamp>/)
    ./yolo_ultralytics/main.py --model yolo11 --train
    ./yolo_ultralytics/main.py --model yolo26 --train

    # Step 3 — evaluate best checkpoint on the test set (mAP30 + point metrics)
    ./yolo_ultralytics/main.py --model yolo11 --eval
    ./yolo_ultralytics/main.py --model yolo26 --eval --run 2026-08-14_09-00-00

    # All-in-one
    ./yolo_ultralytics/main.py --model yolo11 --prepare --train --eval

YOLO26 is end-to-end (NMS-free): its head reports end2end=True, so the NMS `iou`
argument is accepted but ignored at inference.  `max_det` still caps detections per
image and DOES bite on dense crops — both models keep config.YOLOV5_MAX_PRED so the
cap is identical across generations.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

# Project root = two levels up from this file (rebuild/yolo_ultralytics/main.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import data_prep.config as config
from data_prep.prepare_data import prepare


def make_run_dir(ckpt_dir: str) -> str:
    """Create and return a timestamped subdirectory under the model's ckpt dir."""
    run_dir = os.path.join(ckpt_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def latest_run_dir(ckpt_dir: str) -> str | None:
    """Return the most recently created timestamped run dir, or None."""
    if not os.path.exists(ckpt_dir):
        return None
    runs = sorted((e.path for e in os.scandir(ckpt_dir) if e.is_dir()), reverse=True)
    return runs[0] if runs else None


def count_crops(split: str) -> int:
    coco_path = os.path.join(config.CROPS_JSON_DIR, split, "coco.json")
    if not os.path.exists(coco_path):
        return 0
    with open(coco_path) as f:
        return len(json.load(f)["images"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Waterfowl YOLO11 / YOLO26 (Ultralytics)")
    parser.add_argument("--model", default="yolo11", choices=list(config.ULTRALYTICS_MODELS),
                        help="Ultralytics generation to train/eval (default: yolo11). "
                             "Each writes to its own checkpoint folder.")
    parser.add_argument("--prepare", action="store_true",
                        help="Pre-process datasets into COCO crops + Ultralytics mirror")
    parser.add_argument("--train", action="store_true",
                        help="Train the selected model on the prepared crops")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate the best checkpoint on the test set (mAP30)")
    parser.add_argument("--run", default=None, metavar="TIMESTAMP",
                        help="Run timestamp to evaluate (default: latest)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the epoch cap (default: config.NUM_EPOCHS)")
    parser.add_argument("--examples", type=int, default=40, metavar="N",
                        help="Example panels to save during --eval (default: 40, 0 to skip)")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.eval):
        parser.print_help()
        return

    spec = config.ultralytics_model_spec(args.model)
    ckpt_dir = spec["ckpt_dir"]
    label = spec["label"]

    # ── 1. Data preparation ───────────────────────────────────────────────────
    if args.prepare:
        from yolov5.dataset import build_yolo_dataset

        print("\n=== Data Preparation ===")
        prepare()
        data_yaml, counts = build_yolo_dataset()
        print(f"Ultralytics mirror: {data_yaml}")
        for split, n in counts.items():
            print(f"  {split}: {n} images")

    # ── 2. Training ───────────────────────────────────────────────────────────
    if args.train:
        from yolo_ultralytics.train import train as run_training

        print(f"\n=== Training {label} ({spec['weights']}) ===")
        n_train = count_crops("train")
        if n_train == 0:
            print("No training crops found. Run --prepare first.")
            return

        run_dir = make_run_dir(ckpt_dir)
        print(f"Run directory: {run_dir}")
        print(f"Train crops: {n_train}  | Val: {count_crops('val')}  "
              f"| Test: {count_crops('test')}")
        print(f"Epochs: {args.epochs or config.NUM_EPOCHS}  "
              f"| Batch: {spec['batch_size']}  | LR: {spec['lr']}  "
              f"| Early-stop patience: {config.EARLY_STOP_PATIENCE}")

        from data_prep.experiment_record import ExperimentRecord, Timer

        rec = ExperimentRecord(run_dir, model=args.model, label=label, family="ultralytics")
        rec.set_pretrained(spec["weights"])
        rec.set_training({"epochs": args.epochs or config.NUM_EPOCHS,
                          "patience": config.EARLY_STOP_PATIENCE,
                          "batch": spec["batch_size"], "lr0": spec["lr"],
                          "optimizer": "Adam", "imgsz": config.CROP_SIZE,
                          "seed": config.RANDOM_SEED, "end2end": spec["end2end"]})
        with Timer() as t:
            ckpt = run_training(args.model, run_dir, epochs=args.epochs)
        rec.finish_training(checkpoint=ckpt, duration_s=t.seconds,
                            epochs_completed=args.epochs or config.NUM_EPOCHS,
                            peak_gpu_mb=t.peak_gpu_mb)
        print(f"Best checkpoint: {ckpt}")

        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump({"timestamp": os.path.basename(run_dir), "model": args.model,
                       "weights": spec["weights"], "end2end": spec["end2end"],
                       "n_train": n_train, "n_val": count_crops("val"),
                       "n_test": count_crops("test"),
                       "epochs": args.epochs or config.NUM_EPOCHS}, f, indent=2)

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    if args.eval:
        from ultralytics import YOLO

        from yolov5.evaluate import coco_map30, predictions_to_coco
        from yolov5.main import _save_examples
        from data_prep.point_metrics import (gt_points_from_coco, dets_from_coco_results,
                                             paper_point_metrics, format_report,
                                             format_map30_block, write_eval_txt)

        print(f"\n=== Test-set Evaluation ({label}) ===")

        eval_dir = os.path.join(ckpt_dir, args.run) if args.run else latest_run_dir(ckpt_dir)
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

        # Low score threshold so COCOeval sees the full PR curve, and this model's own
        # NMS/max_det settings (iou is inert for the end-to-end YOLO26).
        from data_prep.experiment_record import ExperimentRecord, Timer

        rec = ExperimentRecord(eval_dir, model=args.model, label=label, family="ultralytics")
        rec.set_inference({"nms_iou": spec["nms_iou"], "max_det": spec["max_pred"],
                           "imgsz": config.CROP_SIZE, "end2end": spec["end2end"],
                           "checkpoint": ckpt})
        with Timer() as t:
            coco_results = predictions_to_coco(
                model, test_json, conf=config.CONF_THRESHOLD,
                iou=spec["nms_iou"], max_det=spec["max_pred"], imgsz=config.CROP_SIZE,
            )
        ap30, ar30 = coco_map30(test_json, coco_results)

        # ── OWL-paper point metrics (shared protocol, Section 4.3) ────────────
        pm = paper_point_metrics(
            gt_points_from_coco(test_json), dets_from_coco_results(coco_results),
            tau=config.PAPER_TAU, bootstrap=config.PAPER_BOOTSTRAP,
        )
        with open(os.path.join(eval_dir, "paper_point_metrics.json"), "w") as f:
            json.dump(pm, f, indent=2)

        rec.set_accuracy({"map30": ap30, "mar30": ar30,
                          "ap": pm["detection"]["ap"] * 100,
                          "f1": pm["detection"]["f1_at_t_star"] * 100,
                          "recall": pm["detection"]["recall_at_t_star"] * 100,
                          "precision": pm["detection"]["precision_at_t_star"] * 100,
                          "mae": pm["counting"]["mae"], "rmse": pm["counting"]["rmse"]})
        with open(test_json) as f:
            _n_test = len(json.load(f)["images"])
        rec.finish_evaluation(duration_s=t.seconds, n_images=_n_test,
                              n_detections=len(coco_results), peak_gpu_mb=t.peak_gpu_mb)
        rec.set_outputs(eval_results=os.path.join(eval_dir, "eval_results.txt"),
                        point_metrics=os.path.join(eval_dir, "paper_point_metrics.json")).save()

        text = write_eval_txt(os.path.join(eval_dir, "eval_results.txt"),
                              [format_map30_block(ap30, ar30), format_report(pm, label)])
        print("\n" + text)
        print(f"\nSaved results to {os.path.join(eval_dir, 'eval_results.txt')}")

        if args.examples > 0:
            print(f"\nSaving {args.examples} example images...")
            _save_examples(model, eval_dir=eval_dir, n=args.examples)


if __name__ == "__main__":
    main()
