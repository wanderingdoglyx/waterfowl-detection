"""
Train YOLO-NAS on the waterfowl crops with super-gradients.

Mirrors faster_rcnn.train: builds the model, plugs in the mAP30 metric, watches
mAP30 for early stopping, and writes checkpoints under a per-run directory.

super-gradients' Trainer manages checkpointing itself.  Newer versions nest each
train() call under a per-run id, so checkpoints land at
    <ckpt_root_dir>/<experiment_name>/RUN_<timestamp>/ckpt_best.pth   (best metric)
    <ckpt_root_dir>/<experiment_name>/RUN_<timestamp>/ckpt_latest.pth
(older versions wrote them directly under <experiment_name>/).  We give each run
its own experiment_name (a timestamp) so runs don't clobber each other — the same
convention as the Faster R-CNN timestamped run dirs — and resolve the actual file
location with find_checkpoint() rather than assuming either layout.
"""

from __future__ import annotations

import glob
import os

from super_gradients.training import Trainer

from yolo_nas.model import build_model, build_training_params, NUM_CLASSES
import data_prep.config as config


def find_checkpoint(run_dir: str, name: str = "ckpt_best.pth") -> str | None:
    """
    Resolve a super-gradients checkpoint inside `run_dir`, handling both layouts.

    Newer super-gradients nests checkpoints under run_dir/RUN_<timestamp>/; older
    versions wrote them directly under run_dir.  Returns the most recent matching
    path, or None if no such checkpoint exists.
    """
    direct = os.path.join(run_dir, name)
    if os.path.exists(direct):
        return direct
    for sub in sorted(glob.glob(os.path.join(run_dir, "RUN_*")), reverse=True):
        cand = os.path.join(sub, name)
        if os.path.exists(cand):
            return cand
    return None


def train(run_dir: str, train_loader, val_loader, pretrained: bool = True) -> str:
    """
    Build and run a YOLO-NAS training session.

    Args:
        run_dir      : per-run output directory (…/checkpoints/yolonas/<timestamp>).
                       Its basename is used as the super-gradients experiment_name
                       and its parent as ckpt_root_dir, so weights land directly in
                       run_dir/ckpt_best.pth.
        train_loader : DataLoader from yolo_nas.dataset.build_dataloaders()["train"]
        val_loader   : …["val"]
        pretrained   : initialise from COCO-pretrained weights.

    Returns:
        Path to the best checkpoint (run_dir/ckpt_best.pth).
    """
    ckpt_root   = os.path.dirname(run_dir)
    experiment  = os.path.basename(run_dir)

    trainer = Trainer(experiment_name=experiment, ckpt_root_dir=ckpt_root)
    model   = build_model(num_classes=NUM_CLASSES, pretrained=pretrained)
    params  = build_training_params(num_classes=NUM_CLASSES)

    trainer.train(
        model=model,
        training_params=params,
        train_loader=train_loader,
        valid_loader=val_loader,
    )

    return find_checkpoint(run_dir, "ckpt_best.pth") or os.path.join(run_dir, "ckpt_best.pth")
