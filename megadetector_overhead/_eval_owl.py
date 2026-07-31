#!/usr/bin/env python
"""
OWL inference/eval helper — RUNS UNDER THE MegaDetector-Overhead .venv (Python 3.11),
never under the `waterfowl` env.  Invoked as a subprocess by megadetector_overhead/main.py.

It reuses animaloc's *own* validated detection-branch evaluator (the same code path as
the repo's tools/test.py) to produce, in a single pass over the test crops:

  1. the native point metric — precision / recall / F1 at a distance threshold, and
  2. per-point detections (location + heatmap-peak score),

then writes both to a JSON the (Python 3.10) orchestrator reads back to compute the
pseudo-box mAP30.  Detected points are emitted in FULL-RESOLUTION crop pixels (the
evaluator works in heatmap space = input / down_ratio, so we multiply by down_ratio).

Usage:
    python _eval_owl.py --pth best.pth --images-dir mdo_data/test/images \
        --gt-csv mdo_data/test/gt.csv --out-json eval/owl_eval.json [tuning flags]
"""

import argparse
import json
import os
import sys
import warnings

# animaloc rebuilds an albumentations Compose on every __getitem__; our transform list is
# just Normalize, which doesn't process keypoints, so albumentations warns once per crop.
# Silence that one warning here so a direct run of this helper stays quiet too (subprocess
# launches from main.py also set PYTHONWARNINGS, which covers spawned DataLoader workers).
warnings.filterwarnings(
    "ignore",
    message="Got processor for keypoints, but no transform to process it.",
    category=UserWarning,
)
# sklearn >= 1.9 warns on every 1x1 confusion matrix — i.e. on every image in our
# single-class (bird-only) setup, even though animaloc passes labels=[1] correctly.
warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'.",
    category=UserWarning,
)

# Resolve `import animaloc` even if the repo isn't pip-installed (mirrors tools/*.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The repo root is third_party/MegaDetector-Overhead; when this file is copied into the
# waterfowl project the editable install still resolves animaloc, but keep a fallback.
_REPO = os.environ.get("MDO_REPO_DIR")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pandas
import torch
import albumentations as A
from torch.utils.data import DataLoader, SequentialSampler

from animaloc.datasets import FolderDataset
from animaloc.data.transforms import DownSample
import animaloc.models as animaloc_models
from animaloc.models.utils import LossWrapper, load_model
from animaloc.eval.stitchers import HerdNet_Detection_Branch_Stitcher
from animaloc.eval.evaluators import HerdNet_Detection_Branch_Evaluator
from animaloc.eval.metrics import PointsMetrics

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OWL detection-branch eval helper (mdo env)")
    p.add_argument("--pth", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--device", default="cuda")
    # Model construction: a class name in animaloc.models + a JSON kwargs blob. Defaults
    # reproduce the historical OWL-C behaviour when --model-kwargs is omitted.
    p.add_argument("--model-name", default="OWLC")
    p.add_argument("--model-kwargs", default="", help="JSON dict of constructor kwargs")
    p.add_argument("--num-layers", type=int, default=34)
    p.add_argument("--head-conv", type=int, default=64)
    p.add_argument("--down-ratio", type=int, default=2)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--overlap", type=int, default=160)
    p.add_argument("--adapt-ts", type=float, default=0.3)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    mean = ckpt.get("mean", IMAGENET_MEAN) if isinstance(ckpt, dict) else IMAGENET_MEAN
    std = ckpt.get("std", IMAGENET_STD) if isinstance(ckpt, dict) else IMAGENET_STD

    # Dataset: FolderDataset so empty crops are included as background (honest FP count).
    gt_df = pandas.read_csv(args.gt_csv)
    dataset = FolderDataset(
        csv_file=gt_df,
        root_dir=args.images_dir,
        albu_transforms=[A.Normalize(mean=mean, std=std)],
        end_transforms=[DownSample(down_ratio=args.down_ratio, anno_type="point")],
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        sampler=SequentialSampler(dataset), num_workers=args.num_workers,
    )

    # Model: the selected OWL variant (single-class detection branch). We rebuild the bare
    # architecture and load the fine-tuned weights on top, so no pretrained backbone init is
    # needed here (in particular OWL-D's gated DINOv3 file is not required just to score).
    if args.model_kwargs:
        model_kwargs = json.loads(args.model_kwargs)
    else:
        model_kwargs = dict(
            num_layers=args.num_layers, pretrained=False,
            down_ratio=args.down_ratio, head_conv=args.head_conv,
        )
    model_cls = animaloc_models.__dict__.get(args.model_name)
    if model_cls is None:
        raise KeyError(f"Model '{args.model_name}' not found in animaloc.models")
    model = model_cls(**model_kwargs)
    model = LossWrapper(model, [])
    model = load_model(model, args.pth).to(device)

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)

    stitcher = HerdNet_Detection_Branch_Stitcher(
        model=model, size=(args.img_size, args.img_size), overlap=args.overlap,
        down_ratio=args.down_ratio, up=False, reduction="mean", device_name=device,
    )
    metrics = PointsMetrics(radius=args.radius, num_classes=2)
    evaluator = HerdNet_Detection_Branch_Evaluator(
        model=model, dataloader=dataloader, metrics=metrics,
        lmds_kwargs={"kernel_size": (args.kernel, args.kernel), "adapt_ts": args.adapt_ts},
        device_name=device, print_freq=args.print_freq, stitcher=stitcher,
        work_dir=out_dir, header="[MDO-EVAL]",
    )

    evaluator.evaluate(returns="f1_score", wandb_flag=False, viz=False, log_meters=True)

    # Native point metric — the aggregated binary row (object vs background).
    results = evaluator.results
    binary = results[results["class"] == "binary"].iloc[0]
    point_metrics = {
        "precision": float(binary["precision"]),
        "recall": float(binary["recall"]),
        "f1_score": float(binary["f1_score"]),
        "n_gt": int(binary["n"]),
        "radius_heatmap_px": args.radius,
        "radius_fullres_px": args.radius * args.down_ratio,
    }

    # Per-point detections, scaled from heatmap space back to full-resolution crop pixels.
    dets = evaluator.detections
    detections = []
    if "x" in dets.columns and "y" in dets.columns:
        dets = dets.dropna(subset=["x", "y"])
        score_col = "dscores" if "dscores" in dets.columns else (
            "scores" if "scores" in dets.columns else None
        )
        for _, r in dets.iterrows():
            detections.append({
                "image": str(r["images"]),
                "x": float(r["x"]) * args.down_ratio,
                "y": float(r["y"]) * args.down_ratio,
                "score": float(r[score_col]) if score_col else 1.0,
            })

    with open(args.out_json, "w") as f:
        json.dump({
            "point_metrics": point_metrics,
            "detections": detections,
            "params": {
                "down_ratio": args.down_ratio, "img_size": args.img_size,
                "overlap": args.overlap, "adapt_ts": args.adapt_ts,
                "kernel": args.kernel, "radius": args.radius,
            },
        }, f, indent=2)

    print(f"[MDO-EVAL] point P/R/F1 = "
          f"{point_metrics['precision']:.4f} / {point_metrics['recall']:.4f} / "
          f"{point_metrics['f1_score']:.4f}  | {len(detections)} detections "
          f"-> {args.out_json}")


if __name__ == "__main__":
    main()
