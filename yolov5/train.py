"""
Train YOLOv5 (Ultralytics) on the waterfowl crops.

A sibling of yolo_nas.train: same 100-epoch cap, same early-stopping patience, same
timestamped per-run layout.  Ultralytics owns its training loop and writes

    <project>/<name>/weights/best.pt     (best by Ultralytics' fitness metric)
    <project>/<name>/weights/last.pt

so we point project=<run_dir parent> and name=<run_dir basename>; the best
checkpoint then lands at run_dir/weights/best.pt.

Note on the watched metric: Ultralytics' early stopping tracks its internal fitness
(a blend of mAP@0.50 and mAP@0.50:0.95), not mAP30.  Unlike Faster R-CNN / YOLO-NAS
we cannot redirect its monitor to IoU=0.30 without patching the validator, so only
the *final* test report (main.py --eval) is computed at mAP30, via COCOeval.
"""

from __future__ import annotations

import os

from ultralytics import YOLO

import data_prep.config as config
from yolov5.dataset import build_yolo_dataset


def train(run_dir: str, data_yaml: str | None = None, pretrained: bool = True) -> str:
    """
    Build and run a YOLOv5 training session.

    Args:
        run_dir   : per-run output dir (…/checkpoints/yolov5/<timestamp>).  Its
                    parent is used as the Ultralytics `project` and its basename as
                    `name`, so weights land at run_dir/weights/best.pt.
        data_yaml : path to the Ultralytics data.yaml.  If None, the YOLO-format
                    mirror is (re)built from the shared COCO crops.
        pretrained: initialise from COCO-pretrained weights.

    Returns:
        Path to the best checkpoint (run_dir/weights/best.pt).
    """
    if data_yaml is None:
        data_yaml, _ = build_yolo_dataset()

    project = os.path.dirname(run_dir)
    name    = os.path.basename(run_dir)

    # Ultralytics auto-downloads "<model>.pt" (COCO-pretrained) from its GitHub
    # release assets; passing the bare ".yaml" arch instead trains from scratch.
    weights = f"{config.YOLOV5_MODEL}.pt" if pretrained else f"{config.YOLOV5_MODEL}.yaml"
    model = YOLO(weights)

    model.train(
        data=data_yaml,
        epochs=config.NUM_EPOCHS,
        patience=config.EARLY_STOP_PATIENCE,
        batch=config.YOLOV5_BATCH_SIZE,
        imgsz=config.CROP_SIZE,
        optimizer="Adam",
        lr0=config.YOLOV5_LR,
        project=project,
        name=name,
        exist_ok=True,
        workers=config.NUM_WORKERS,
        seed=config.RANDOM_SEED,
        device=0,
        val=True,
        verbose=True,
    )

    return os.path.join(run_dir, "weights", "best.pt")
