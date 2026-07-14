"""
YOLO-NAS model, loss, metric and training-params builders (super-gradients).

mAP30 (IoU = 0.30, per Section 4.4) is obtained by passing iou_thres=0.30 to
super-gradients' DetectionMetrics — the same metric class used for the usual
mAP@0.50, just evaluated at a single, lower IoU threshold.
"""

from __future__ import annotations

from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.utils.callbacks import Phase
from super_gradients.training.utils.early_stopping import EarlyStop

import data_prep.config as config

NUM_CLASSES = 1  # bird

# super-gradients names DetectionMetrics components by IoU threshold: the map and
# recall components are "mAP@0.30" and "Recall@0.30" (NOT "AR@0.30").  Keep these
# derived from the configured threshold so a change to IOU_THRESHOLD_MAP stays
# consistent across train (metric_to_watch) and eval (result lookup).
_IOU = config.IOU_THRESHOLD_MAP
MAP_KEY = f"mAP@{_IOU:.2f}"
AR_KEY  = f"Recall@{_IOU:.2f}"


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """YOLO-NAS (variant from config) with a fresh head for `num_classes`."""
    return models.get(
        config.YOLONAS_MODEL,
        num_classes=num_classes,
        pretrained_weights="coco" if pretrained else None,
    )


def build_post_prediction_callback(score_threshold: float = config.CONF_THRESHOLD):
    """NMS / decoding callback shared by metric and inference.

    score_threshold defaults low (0.05) so COCO-style AP sees the full PR curve,
    mirroring Faster R-CNN's SCORE_THRESH_TEST during evaluation.
    """
    return PPYoloEPostPredictionCallback(
        score_threshold=score_threshold,
        nms_threshold=config.YOLONAS_NMS_IOU,
        nms_top_k=config.YOLONAS_NMS_TOPK,
        max_predictions=config.YOLONAS_MAX_PRED,
    )


def build_metric(num_classes: int = NUM_CLASSES) -> DetectionMetrics:
    """DetectionMetrics evaluated at IoU=0.30 (mAP30)."""
    return DetectionMetrics(
        num_cls=num_classes,
        post_prediction_callback=build_post_prediction_callback(),
        normalize_targets=True,
        iou_thres=_IOU,
    )


def build_training_params(num_classes: int = NUM_CLASSES) -> dict:
    """
    super-gradients training_params dict mirroring the Faster R-CNN recipe:
    100-epoch cap, early stopping on mAP30 with patience = EARLY_STOP_PATIENCE.
    """
    early_stop = EarlyStop(
        phase=Phase.VALIDATION_EPOCH_END,
        monitor=MAP_KEY,
        mode="max",
        patience=config.EARLY_STOP_PATIENCE,
        verbose=True,
    )

    return {
        "max_epochs": config.NUM_EPOCHS,
        "warmup_mode": "linear_epoch_step",
        "lr_warmup_epochs": 3,
        "initial_lr": config.YOLONAS_LR,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.01,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 1e-4},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=num_classes,
            reg_max=16,
        ),
        "valid_metrics_list": [build_metric(num_classes)],
        "metric_to_watch": MAP_KEY,
        "greater_metric_to_watch_is_better": True,
        "phase_callbacks": [early_stop],
        "save_ckpt_epoch_list": [],
    }
