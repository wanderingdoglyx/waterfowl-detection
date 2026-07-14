"""
Detectron2 trainer for waterfowl Faster R-CNN.

Features:
  • DefaultTrainer subclass that plugs in MAP30Evaluator.
  • EarlyStoppingHook: stops training when val mAP30 fails to improve for
    `patience` consecutive evaluations (Section 4.1: tolerance = 30 epochs).
  • Saves best checkpoint to OUTPUT_DIR/model_best.pth.
"""

import logging
import os

import torch

from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils import comm

from faster_rcnn.evaluator import MAP30Evaluator
import data_prep.config as config

logger = logging.getLogger(__name__)


# ── Early-stopping hook ───────────────────────────────────────────────────────

class EarlyStoppingHook(HookBase):
    """
    Monitors 'bbox/AP30' in the trainer's storage after each evaluation.
    Stops training (raises StopIteration) when the metric has not improved
    for `patience` evaluations.  Also copies the best checkpoint.
    """

    def __init__(
        self,
        patience:       int = config.EARLY_STOP_PATIENCE,
        metric_key:     str = "bbox/AP30",
        output_dir:     str = config.CKPT_DIR,
        eval_period:    int = 1,
    ):
        self.patience    = patience
        self.metric_key  = metric_key
        self.output_dir  = output_dir
        self.eval_period = eval_period
        self._best       = -1.0
        self._no_improve = 0
        self._best_iter  = -1

    def after_step(self):
        # Only check on evaluation steps
        if (self.trainer.iter + 1) % self.eval_period != 0:
            return

        storage = self.trainer.storage
        if self.metric_key not in storage.latest():
            return

        current = storage.latest()[self.metric_key][0]

        if current > self._best:
            self._best       = current
            self._no_improve = 0
            self._best_iter  = self.trainer.iter
            # Save current weights as model_best via the trainer's checkpointer
            self.trainer.checkpointer.save("model_best")
            logger.info(
                f"New best mAP30 = {self._best:.2f}% at iter {self._best_iter + 1}"
            )
        else:
            self._no_improve += 1
            logger.info(
                f"No improvement for {self._no_improve}/{self.patience} evaluations. "
                f"Best mAP30 = {self._best:.2f}%"
            )
            if self._no_improve >= self.patience:
                logger.info(
                    f"Early stopping triggered after {self.trainer.iter + 1} iterations."
                )
                # Suppress detectron2's catch-all ERROR log — StopIteration here is
                # an intentional signal, not a crash.
                logging.getLogger("detectron2.engine.train_loop").setLevel(logging.CRITICAL)
                raise StopIteration


# ── Custom DefaultTrainer ─────────────────────────────────────────────────────

class WaterfowlTrainer(DefaultTrainer):
    """DefaultTrainer that uses MAP30Evaluator and attaches EarlyStoppingHook."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([MAP30Evaluator(dataset_name, output_folder)])

    def build_hooks(self):
        hooks = super().build_hooks()
        # Insert EarlyStoppingHook just before the PeriodicWriter (last hook)
        hooks.insert(
            -1,
            EarlyStoppingHook(
                patience=config.EARLY_STOP_PATIENCE,
                metric_key="bbox/AP30",
                output_dir=self.cfg.OUTPUT_DIR,
                eval_period=self.cfg.TEST.EVAL_PERIOD,
            ),
        )
        return hooks


# ── Convenience train function ────────────────────────────────────────────────

def train(cfg) -> None:
    """
    Build and run the WaterfowlTrainer.

    Args:
        cfg : CfgNode produced by faster_rcnn.model.build_cfg()
    """
    trainer = WaterfowlTrainer(cfg)
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except StopIteration:
        pass   # early stopping raised StopIteration — this is expected
