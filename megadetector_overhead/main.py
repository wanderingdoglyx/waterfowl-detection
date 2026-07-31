#!/usr/bin/env python
"""
Waterfowl MegaDetector-Overhead (OWL) pipeline.

A sibling of faster_rcnn/main.py, yolo_nas/main.py and yolov5/main.py — same crops, same
timestamped-run layout — but for Microsoft's MegaDetector-Overhead, a *point* detector.
Because OWL needs its own Python 3.11 environment (vendored DINOv3, incompatible with the
detectron2/super-gradients/ultralytics stack), this entry point runs under the `waterfowl`
env like the others and shells into the OWL `.venv` for the actual train/eval.

OWL predicts points, not boxes, so it is scored two ways (see evaluate.py):
  • native point precision/recall/F1 at a distance threshold (the honest metric), and
  • pseudo-box mAP30 through the same COCOeval@IoU=0.30 as the other three (a bridge).

Usage (from the rebuild/ directory):

    # 1. crop datasets + build COCO splits (shared) + mirror to OWL point layout
    ./megadetector_overhead/main.py --prepare

    # 2. fine-tune from pretrained weights (output/checkpoints/mdo_owl_{c,t,d}/<timestamp>/)
    ./megadetector_overhead/main.py --train                 # OWL-C (default)
    ./megadetector_overhead/main.py --train --model OWLT    # OWL-T  -> mdo_owl_t/
    ./megadetector_overhead/main.py --train --model OWLD_S  # OWL-D  -> mdo_owl_d/ (needs DINOv3 wts)

    # 3. evaluate best checkpoint on the test set (point P/R/F1 + pseudo-box mAP30)
    ./megadetector_overhead/main.py --eval                  # latest OWL-C run
    ./megadetector_overhead/main.py --eval --model OWLT --run 2026-07-21_10-30-00

    # all-in-one
    ./megadetector_overhead/main.py --prepare --train --eval

Each --model writes to its own checkpoint folder, so OWL-C/T/D runs never mix.
OWL-T and OWL-D are selectable via config.MDO_MODELS (see data_prep/config.py).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data_prep.config as config
from data_prep.prepare_data import prepare


def make_run_dir(ckpt_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(ckpt_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def latest_run_dir(ckpt_dir: str) -> str | None:
    if not os.path.exists(ckpt_dir):
        return None
    runs = sorted(
        (e.path for e in os.scandir(ckpt_dir) if e.is_dir()),
        reverse=True,
    )
    return runs[0] if runs else None


def count_crops(split: str) -> int:
    coco_path = os.path.join(config.CROPS_JSON_DIR, split, "coco.json")
    if not os.path.exists(coco_path):
        return 0
    with open(coco_path) as f:
        return len(json.load(f)["images"])


def _check_env() -> bool:
    """Verify the OWL .venv interpreter exists before shelling into it."""
    if not os.path.exists(config.MDO_PYTHON):
        print(f"OWL environment not found at {config.MDO_PYTHON}")
        print("Set it up with:")
        print(f"  cd {config.MDO_REPO_DIR} && uv sync --no-default-groups --group gpu")
        return False
    return True


def fetch_zenodo_checkpoint(model_spec: dict) -> bool:
    """Download the model's released fine-tuning checkpoint (OWL-C/T/D.pth) from the Zenodo
    benchmark record into weights/ if missing.  Returns True if present afterwards."""
    dest = model_spec.get("requires_weights") or model_spec.get("load_from")
    zfile = model_spec.get("zenodo_file")
    if not dest or not zfile:
        print(f"No Zenodo checkpoint configured for {model_spec['name']}.")
        return False
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"Already present ({os.path.getsize(dest) / 1e6:.0f} MB): {dest}")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = config.mdo_zenodo_url(zfile)
    tmp = dest + ".part"
    print(f"Downloading {zfile} ({config.MDO_ZENODO_RECORD}) -> {dest}")

    def _hook(blocks: int, bs: int, total: int) -> None:
        if total > 0:
            done = blocks * bs
            print(f"\r  {min(100, 100 * done / total):5.1f}%  "
                  f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)", end="", flush=True)

    try:
        import urllib.request
        urllib.request.urlretrieve(url, tmp, reporthook=_hook)
        print()
        os.replace(tmp, dest)
    except Exception as e:
        print(f"\nDownload failed: {type(e).__name__}: {str(e)[:200]}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f"Done: {size / 1e6:.0f} MB  ({config.MDO_ZENODO_RECORD}, CC BY-NC-SA 4.0)")
    return size > 0


def _save_examples(owl_eval_json: str, eval_dir: str, n: int) -> None:
    """Save n side-by-side panels: GT boxes (green) | predicted points (red)."""
    import random
    import cv2

    test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
    with open(test_json) as f:
        coco = json.load(f)
    with open(owl_eval_json) as f:
        dets = json.load(f).get("detections", [])

    gt_by_img = {img["id"]: [] for img in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img[a["image_id"]].append([x, y, x + w, y + h])

    pts_by_base: dict[str, list] = {}
    for d in dets:
        pts_by_base.setdefault(d["image"], []).append((d["x"], d["y"], d["score"]))

    out_dir = os.path.join(eval_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)
    images = random.sample(coco["images"], min(n, len(coco["images"])))

    import numpy as np
    saved = 0
    for img in images:
        path = os.path.join(config.CROPS_IMG_DIR, img["file_name"])
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        gt_panel = img_bgr.copy()
        for x1, y1, x2, y2 in gt_by_img.get(img["id"], []):
            cv2.rectangle(gt_panel, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        pred_panel = img_bgr.copy()
        # No confidence filter: the OWL/HerdNet protocol treats every LMDS peak as a
        # detection (the adaptive threshold inside LMDS is the only filtering).
        for x, y, score in pts_by_base.get(os.path.basename(img["file_name"]), []):
            cv2.circle(pred_panel, (int(x), int(y)), 4, (0, 0, 255), 1)

        combined = np.concatenate([gt_panel, pred_panel], axis=1)
        fname = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".jpg"
        cv2.imwrite(os.path.join(out_dir, fname), combined)
        saved += 1

    print(f"Saved {saved} examples to {out_dir}/  (left = GT boxes, right = predicted points)")


def run_eval_helper(ckpt: str, out_json: str, model_spec: dict) -> None:
    """Invoke the OWL .venv to run inference + native point metric, writing out_json."""
    from megadetector_overhead.dataset import split_paths

    test_imgs, test_csv = split_paths("test")
    env = os.environ.copy()
    env["MDO_REPO_DIR"] = config.MDO_REPO_DIR
    env["DINOV3_ROOT"] = config.MDO_DINOV3_ROOT  # for OWL-D; harmless otherwise
    # animaloc rebuilds an albumentations Compose per __getitem__; with only a Normalize
    # transform it warns "no transform to process keypoints" once per crop, flooding the
    # log. Likewise sklearn >= 1.9 warns on every 1x1 confusion matrix — i.e. every image
    # in our single-class setup, despite animaloc passing labels=[1] correctly. Silence
    # just those two modules' UserWarnings in the .venv interpreter and every DataLoader
    # worker it spawns (PYTHONWARNINGS is read at each interpreter startup).
    env["PYTHONWARNINGS"] = ",".join(filter(None, [
        env.get("PYTHONWARNINGS"),
        "ignore::UserWarning:albumentations.core.composition",
        "ignore::UserWarning:sklearn.metrics._classification",
    ]))
    cmd = [
        config.MDO_PYTHON, os.path.join(ROOT, "megadetector_overhead", "_eval_owl.py"),
        "--pth", ckpt,
        "--images-dir", test_imgs,
        "--gt-csv", test_csv,
        "--out-json", out_json,
        "--device", "cuda",
        "--model-name", model_spec["name"],
        "--model-kwargs", json.dumps(model_spec["eval_kwargs"]),
        "--down-ratio", str(config.MDO_DOWN_RATIO),
        "--img-size", str(config.CROP_SIZE),
        "--radius", str(config.MDO_POINT_RADIUS),
        "--overlap", str(config.MDO_STITCH_OVERLAP),
        "--adapt-ts", str(config.MDO_LMDS_ADAPT_TS),
        "--kernel", str(config.MDO_LMDS_KERNEL),
    ]
    subprocess.run(cmd, cwd=config.MDO_REPO_DIR, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Waterfowl MegaDetector-Overhead (OWL)")
    parser.add_argument("--prepare", action="store_true",
                        help="Crop datasets + build COCO splits + OWL point mirror")
    parser.add_argument("--train", action="store_true",
                        help="Fine-tune the selected OWL variant on the prepared crops")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate best checkpoint (point P/R/F1 + pseudo-box mAP30)")
    parser.add_argument("--model", default=config.MDO_MODEL, choices=list(config.MDO_MODELS),
                        help=f"OWL variant to train/eval (default: {config.MDO_MODEL}). "
                             "Each writes to its own checkpoint folder.")
    parser.add_argument("--fetch-weights", action="store_true",
                        help="Download the selected --model's released fine-tuning checkpoint "
                             "from Zenodo (OWL-C/T/D.pth) into weights/ if not already present")
    parser.add_argument("--run", default=None, metavar="TIMESTAMP",
                        help="Run timestamp to evaluate (default: latest)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs (default: config.MDO_EPOCHS)")
    parser.add_argument("--examples", type=int, default=40, metavar="N",
                        help="Example panels to save during --eval (default: 40, 0 to skip)")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.eval or args.fetch_weights):
        parser.print_help()
        return

    model_spec = config.mdo_model_spec(args.model)
    ckpt_dir = model_spec["ckpt_dir"]

    # ── 0. Fetch the selected model's released Zenodo checkpoint ─────────────────
    if args.fetch_weights:
        print(f"\n=== Fetch {model_spec['name']} weights (Zenodo {config.MDO_ZENODO_RECORD}) ===")
        if not fetch_zenodo_checkpoint(model_spec):
            if args.train:
                print("Skipping --train because the checkpoint is not in place.")
            return

    # ── 1. Data preparation ─────────────────────────────────────────────────────
    if args.prepare:
        from megadetector_overhead.dataset import build_mdo_dataset

        print("\n=== Data Preparation ===")
        prepare()
        print("Building OWL point dataset (symlinks + gt.csv)...")
        _, counts = build_mdo_dataset()
        print(f"OWL point mirror crops per split: {counts}")

    # ── 2. Training ─────────────────────────────────────────────────────────────
    if args.train:
        from megadetector_overhead.dataset import build_mdo_dataset
        from megadetector_overhead.train import train as run_training

        print("\n=== Training ===")
        if not _check_env():
            return
        n_train = count_crops("train")
        if n_train == 0:
            print("No training crops found. Run --prepare first.")
            return

        # ensure the point mirror is present/fresh (cheap; just symlinks + csv)
        _, counts = build_mdo_dataset()
        if counts.get("val", 0) == 0:
            print("No validation crops found. Run --prepare first.")
            return

        # Each variant fine-tunes from its released Zenodo checkpoint; fail early (with the
        # exact fetch command) if it hasn't been downloaded yet.
        needed = model_spec.get("requires_weights")
        if needed and not os.path.exists(needed):
            print(f"Model {model_spec['name']} needs its start checkpoint, which is missing:")
            print(f"  {needed}")
            print(f"Fetch it with:  ./megadetector_overhead/main.py --fetch-weights "
                  f"--model {args.model}")
            return

        run_dir = make_run_dir(ckpt_dir)
        print(f"Run directory: {run_dir}")
        start_from = model_spec.get("load_from")
        run_info = {
            "timestamp": os.path.basename(run_dir),
            "model": model_spec["name"],
            "pretrained": start_from if start_from and os.path.exists(start_from) else None,
            "n_train": counts.get("train", 0),
            "n_val": counts.get("val", 0),
            "n_test": counts.get("test", 0),
            "epochs": args.epochs if args.epochs is not None else config.MDO_EPOCHS,
        }
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        print(f"Training crops: {n_train}  | Model: {model_spec['name']}  "
              f"| Epochs: {run_info['epochs']}  | Batch: {config.MDO_BATCH_SIZE}")
        ckpt = run_training(run_dir, epochs=args.epochs, model_key=args.model)
        print(f"Best checkpoint: {ckpt}")

    # ── 3. Evaluation on the test set ───────────────────────────────────────────
    if args.eval:
        from megadetector_overhead.evaluate import evaluate_from_json

        print("\n=== Test-set Evaluation ===")
        if not _check_env():
            return

        eval_dir = os.path.join(ckpt_dir, args.run) if args.run \
            else latest_run_dir(ckpt_dir)
        if not eval_dir or not os.path.exists(eval_dir):
            print(f"No run directory found under {ckpt_dir}. Run --train first "
                  f"(for --model {args.model}).")
            return

        ckpt = os.path.join(eval_dir, "weights", "best.pth")
        if not os.path.exists(ckpt):
            print(f"No checkpoint found in {eval_dir}. Run --train first.")
            return

        test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
        if not os.path.exists(test_json):
            print("No test crops found. Run --prepare first.")
            return

        print(f"Evaluating run: {eval_dir}")
        print(f"Loading checkpoint: {ckpt}")

        eval_out_dir = os.path.join(eval_dir, "eval")
        os.makedirs(eval_out_dir, exist_ok=True)
        owl_eval_json = os.path.join(eval_out_dir, "owl_eval.json")

        run_eval_helper(ckpt, owl_eval_json, model_spec)
        metrics = evaluate_from_json(owl_eval_json, test_json)

        pt = metrics["point"]
        pb = metrics["pseudo_box_map30"]
        print("\n── Point metric (native; TP within "
              f"{pt.get('radius_fullres_px', '?')} px of a GT point) ──")
        print(f"  Precision : {pt.get('precision', 0) * 100:.2f}%")
        print(f"  Recall    : {pt.get('recall', 0) * 100:.2f}%")
        print(f"  F1        : {pt.get('f1_score', 0) * 100:.2f}%")
        print(f"\n── Pseudo-box mAP30 (points → {pb['box_size']}px boxes, "
              "COCOeval@IoU=0.30) ──")
        print(f"  mAP30 : {pb['ap30']:.2f}%")
        print(f"  mAR30 : {pb['ar30']:.2f}%")
        print(f"\n  ({metrics['n_detections']} detections total)")

        with open(os.path.join(eval_out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if args.examples > 0:
            print(f"\nSaving {args.examples} example images...")
            _save_examples(owl_eval_json, eval_dir=eval_dir, n=args.examples)


if __name__ == "__main__":
    main()
