"""
Train YOLO11 / YOLO26 (Ultralytics) on the waterfowl crops.

A parameterised sibling of yolov5.train: same 100-epoch cap, same early-stopping
patience, same optimiser and LR, same timestamped per-run layout — only the model
weights and checkpoint directory change, both taken from
config.ULTRALYTICS_MODELS[<key>].

Ultralytics owns its training loop and writes

    <project>/<name>/weights/best.pt     (best by Ultralytics' fitness metric)
    <project>/<name>/weights/last.pt

so project=<run_dir parent> and name=<run_dir basename> put the best checkpoint at
run_dir/weights/best.pt, exactly as for YOLOv5.

Note on the watched metric (inherited from yolov5.train): Ultralytics' early stopping
tracks its internal fitness (a blend of mAP@0.50 and mAP@0.50:0.95), not mAP30, and we
do not patch the validator.  Only the final test report (main.py --eval) is mAP30.

Note on the optimiser: Adam at config.YOLOV5_LR is used for BOTH models, matching
YOLOv5 and YOLO-NAS.  YOLO26 ships a different default optimiser upstream; we override
it deliberately so the cross-generation comparison is like-for-like rather than
per-model tuned.  That is the project's standing convention, not an oversight.
"""

from __future__ import annotations

import os

from ultralytics import YOLO

import data_prep.config as config
from yolov5.dataset import build_yolo_dataset


def train(model_key: str, run_dir: str, data_yaml: str | None = None,
          pretrained: bool = True, epochs: int | None = None) -> str:
    """
    Build and run an Ultralytics training session for YOLO11 or YOLO26.

    Args:
        model_key : key into config.ULTRALYTICS_MODELS ("yolo11" / "yolo26").
        run_dir   : per-run output dir (…/checkpoints/<model>/<timestamp>).
        data_yaml : path to the Ultralytics data.yaml.  If None, the YOLO-format
                    mirror is (re)built from the shared COCO crops — the same
                    mirror YOLOv5 uses, so no data is duplicated.
        pretrained: initialise from COCO-pretrained weights.
        epochs    : override the epoch cap (default: config.NUM_EPOCHS).

    Returns:
        Path to the best checkpoint (run_dir/weights/best.pt).
    """
    spec = config.ultralytics_model_spec(model_key)

    if data_yaml is None:
        data_yaml, _ = build_yolo_dataset()

    project = os.path.dirname(run_dir)
    name = os.path.basename(run_dir)

    # Ultralytics auto-downloads "<model>.pt" (COCO-pretrained) from its GitHub
    # release assets; passing the bare ".yaml" arch instead trains from scratch.
    weights = f"{spec['weights']}.pt" if pretrained else f"{spec['weights']}.yaml"
    model = YOLO(weights)

    model.train(
        data=data_yaml,
        epochs=epochs if epochs is not None else config.NUM_EPOCHS,
        patience=config.EARLY_STOP_PATIENCE,
        batch=spec["batch_size"],
        imgsz=config.CROP_SIZE,
        optimizer="Adam",
        lr0=spec["lr"],
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
