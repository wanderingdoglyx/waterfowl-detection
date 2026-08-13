#!/usr/bin/env python
"""
Waterfowl Faster R-CNN pipeline (Detectron2).

Usage (from the rebuild/ directory):

    # Step 1 — crop all datasets and build COCO JSON splits (one-time)
    ./faster_rcnn/main.py --prepare

    # Step 2 — train (results saved to output/checkpoints/fasterrcnn/<timestamp>/)
    ./faster_rcnn/main.py --train

    # Step 3 — evaluate best checkpoint on the test set (uses latest run by default)
    ./faster_rcnn/main.py --eval
    ./faster_rcnn/main.py --eval --run 2026-06-23_14-30-00

    # All-in-one
    ./faster_rcnn/main.py --prepare --train --eval
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

# Project root = two levels up from this file (rebuild/faster_rcnn/main.py → rebuild/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Silence known harmless noise ─────────────────────────────────────────────
# iopath's telemetry system crashes on Python 3.14 (AttributeError on _evt).
# The call chain is:
#   PathManager.__log_tmetry_keys  →  EventLogger.log_event  →  del self._evt  ✗
# Fix 1: no-op EventLogger.log_event (the method that actually fails)
try:
    from iopath.common.event_logger import EventLogger
    EventLogger.log_event = lambda self, topic=None: None
except Exception:
    pass

# Fix 2: no-op PathManager.__log_tmetry_keys (the entry point, name-mangled)
try:
    from iopath.common.file_io import PathManager
    PathManager._PathManager__log_tmetry_keys = lambda self, *a, **k: None
except Exception:
    pass

# torch.meshgrid indexing warning (internal to detectron2 / older torch code)
warnings.filterwarnings("ignore", message="torch.meshgrid")
# setuptools pkg_resources deprecation from detectron2 model_zoo
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import data_prep.config as config
from data_prep.prepare_data import prepare


def make_run_dir() -> str:
    """Create and return a timestamped subdirectory under CKPT_DIR for this run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.CKPT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def latest_run_dir() -> str | None:
    """Return the most recently created timestamped run dir, or None if none exist."""
    if not os.path.exists(config.CKPT_DIR):
        return None
    runs = sorted(
        (e.path for e in os.scandir(config.CKPT_DIR) if e.is_dir()),
        reverse=True,
    )
    return runs[0] if runs else None


def count_train_crops() -> int:
    coco_path = os.path.join(config.CROPS_JSON_DIR, "train", "coco.json")
    if not os.path.exists(coco_path):
        return 0
    with open(coco_path) as f:
        return len(json.load(f)["images"])


def _save_examples(cfg, eval_dir: str, n: int) -> None:
    """Save n side-by-side (ground truth | prediction) images to eval_dir/examples/."""
    import random
    import cv2
    import numpy as np
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer, ColorMode

    dataset_name = cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    samples = random.sample(dataset_dicts, min(n, len(dataset_dicts)))
    out_dir = os.path.join(eval_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)

    # Draw only confident detections in the example panels.  The cfg used for
    # mAP keeps SCORE_THRESH_TEST low so COCOeval sees the full PR curve; that
    # same low threshold makes the qualitative panels look like a wall of false
    # alarms.  Clone the cfg and raise the threshold for visualisation only.
    vis_cfg = cfg.clone()
    vis_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.DISPLAY_CONF_THRESHOLD
    predictor = DefaultPredictor(vis_cfg)

    for d in samples:
        img_bgr = cv2.imread(d["file_name"])
        if img_bgr is None:
            continue
        img_rgb = img_bgr[:, :, ::-1]

        # Left panel: ground-truth boxes
        v_gt = Visualizer(img_rgb.copy(), metadata=metadata, scale=1.0)
        gt_panel = v_gt.draw_dataset_dict(d).get_image()

        # Right panel: predicted boxes with scores
        outputs = predictor(img_bgr)
        v_pred = Visualizer(img_rgb.copy(), metadata=metadata, scale=1.0,
                            instance_mode=ColorMode.IMAGE)
        pred_panel = v_pred.draw_instance_predictions(
            outputs["instances"].to("cpu")
        ).get_image()

        combined = np.concatenate([gt_panel, pred_panel], axis=1)
        fname = os.path.splitext(os.path.basename(d["file_name"]))[0] + ".jpg"
        cv2.imwrite(os.path.join(out_dir, fname), combined[:, :, ::-1])

    print(f"Saved {min(n, len(dataset_dicts))} examples to {out_dir}/  "
          f"(left = GT, right = predictions)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Waterfowl Faster R-CNN (Detectron2)")
    parser.add_argument("--prepare", action="store_true",
                        help="Pre-process all datasets into 512×512 COCO crops")
    parser.add_argument("--train",   action="store_true",
                        help="Train Faster R-CNN on the prepared crops")
    parser.add_argument("--eval",    action="store_true",
                        help="Evaluate the best checkpoint on the test set")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="FOLDER",
                        help="Dataset subfolders to train on (e.g. Bird_A Bird_B). "
                             "Default: all available datasets.")
    parser.add_argument("--run",      default=None, metavar="TIMESTAMP",
                        help="Run timestamp to evaluate (default: latest)")
    parser.add_argument("--examples", type=int, default=40, metavar="N",
                        help="Number of example images to save during --eval (default: 40, 0 to skip)")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.eval):
        parser.print_help()
        return

    # ── 1. Data preparation ───────────────────────────────────────────────────
    if args.prepare:
        print("\n=== Data Preparation ===")
        prepare()

    # ── 2. Training ───────────────────────────────────────────────────────────
    if args.train:
        # Detectron2 imports deferred so --prepare runs without loading the framework
        from detectron2.utils.logger import setup_logger
        from faster_rcnn.dataset import register_datasets
        from faster_rcnn.model import build_cfg
        from faster_rcnn.train import train as run_training

        setup_logger()
        print("\n=== Training ===")
        run_dir = make_run_dir()
        print(f"Run directory: {run_dir}")

        included = set(args.datasets) if args.datasets else None
        if included:
            print(f"Dataset filter: {sorted(included)}")

        counts = register_datasets(included=included)
        n_train = counts.get("train", 0)
        if n_train == 0:
            print("No training crops found. Run --prepare first.")
            return
        print(f"Training crops: {n_train}")

        run_info = {
            "timestamp":    os.path.basename(run_dir),
            "datasets":     sorted(included) if included else "all",
            "n_train":      counts.get("train", 0),
            "n_val":        counts.get("val", 0),
            "n_test":       counts.get("test", 0),
        }
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)
        print(f"Run info saved to {run_dir}/run_info.json")

        cfg = build_cfg(
            num_epochs=config.NUM_EPOCHS,
            num_train_imgs=n_train,
            output_dir=run_dir,
            use_pretrained=True,
        )
        print(f"Max iterations: {cfg.SOLVER.MAX_ITER}  "
              f"| Eval every: {cfg.TEST.EVAL_PERIOD} iters  "
              f"| Early-stop patience: {config.EARLY_STOP_PATIENCE} evals")
        run_training(cfg)

    # ── 3. Evaluation on test set ─────────────────────────────────────────────
    if args.eval:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.utils.logger import setup_logger
        from faster_rcnn.dataset import register_datasets, DATASET_NAMES
        from faster_rcnn.model import build_cfg
        from faster_rcnn.train import WaterfowlTrainer
        from faster_rcnn.evaluator import MAP30Evaluator

        setup_logger()
        print("\n=== Test-set Evaluation ===")
        register_datasets()

        if args.run:
            eval_dir = os.path.join(config.CKPT_DIR, args.run)
        else:
            eval_dir = latest_run_dir()

        if not eval_dir or not os.path.exists(eval_dir):
            print("No run directory found. Run --train first.")
            return

        print(f"Evaluating run: {eval_dir}")
        best_ckpt  = os.path.join(eval_dir, "model_best.pth")
        final_ckpt = os.path.join(eval_dir, "model_final.pth")
        ckpt = best_ckpt if os.path.exists(best_ckpt) else final_ckpt

        if not os.path.exists(ckpt):
            print(f"No checkpoint found in {eval_dir}. Run --train first.")
            return

        print(f"Loading checkpoint: {ckpt}")
        n_train = count_train_crops()
        cfg = build_cfg(
            num_epochs=config.NUM_EPOCHS,
            num_train_imgs=max(n_train, 1),
            output_dir=eval_dir,
        )
        cfg.MODEL.WEIGHTS = ckpt
        cfg.DATASETS.TEST = (DATASET_NAMES["test"],)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.CONF_THRESHOLD

        evaluator = MAP30Evaluator(
            DATASET_NAMES["test"],
            output_dir=os.path.join(eval_dir, "test_inference"),
        )
        # build_model() only constructs the architecture with random weights;
        # DefaultTrainer.test expects a model that already holds its weights, so
        # we must load the checkpoint explicitly (otherwise mAP is ~0).
        model = WaterfowlTrainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        results = WaterfowlTrainer.test(
            cfg, model, evaluators=[evaluator]
        )

        # ── OWL-paper point metrics (shared protocol, Section 4.3) ────────────
        # Box centres vs GT box centres: MAE/RMSE, AP/AUC-PR, P/R/F1 at t*,
        # bootstrap CIs.  Reads the per-detection dump COCOEvaluator just wrote.
        import torch as _torch
        from data_prep.point_metrics import (gt_points_from_coco, dets_from_coco_results,
                                             paper_point_metrics, format_report,
                                             format_map30_block, write_eval_txt)

        sections = []
        if results and "bbox" in results:
            ap30 = results["bbox"].get("AP30", 0.0); ar30 = results["bbox"].get("AR30", 0.0)
            sections.append(format_map30_block(ap30, ar30))
        else:
            print("No bbox results returned.")

        preds_pth = os.path.join(eval_dir, "test_inference", "instances_predictions.pth")
        test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
        if os.path.exists(preds_pth) and os.path.exists(test_json):
            flat = []
            for p in _torch.load(preds_pth):
                flat.extend(p["instances"])
            pm = paper_point_metrics(
                gt_points_from_coco(test_json), dets_from_coco_results(flat),
                tau=config.PAPER_TAU, bootstrap=config.PAPER_BOOTSTRAP,
            )
            sections.append(format_report(pm, "Faster R-CNN"))
            with open(os.path.join(eval_dir, "paper_point_metrics.json"), "w") as f:
                json.dump(pm, f, indent=2)
        else:
            sections.append(f"(point metrics skipped — {preds_pth} not found)")

        # Print the full summary and save it as a plain-text results file.
        text = write_eval_txt(os.path.join(eval_dir, "eval_results.txt"), sections)
        print("\n" + text)
        print(f"\nSaved results to {os.path.join(eval_dir, 'eval_results.txt')}")

        if args.examples > 0:
            print(f"\nSaving {args.examples} example images...")
            _save_examples(cfg, eval_dir=eval_dir, n=args.examples)


if __name__ == "__main__":
    main()
