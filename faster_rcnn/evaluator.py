"""
Custom mAP30 evaluator for Detectron2 (IoU threshold = 0.30, per Section 4.4).

Extends COCOEvaluator and overrides _eval_predictions to:
  - Run pycocotools COCOeval at a single IoU threshold (0.30)
  - Compute AP and AR directly from the accumulated eval arrays
    (avoids summarize() which assumes a range of IoU thresholds)
"""

import itertools
import os
from contextlib import redirect_stdout

import numpy as np

from pycocotools.cocoeval import COCOeval
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table


class MAP30Evaluator(COCOEvaluator):
    """Evaluates detections at IoU threshold = 0.30 (mAP30)."""

    def __init__(self, dataset_name: str, output_dir: str | None = None):
        super().__init__(
            dataset_name=dataset_name,
            tasks=("bbox",),
            distributed=False,
            output_dir=output_dir,
        )
        self._iou_thresh = 0.30

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Args:
            predictions: list of per-image dicts as Detectron2 provides them:
                         [{"image_id": int, "instances": [coco_result, ...]}, ...]
                         where each coco_result has image_id, category_id, bbox, score.
        """
        self._logger.info("Running mAP30 evaluation (IoU=0.30)...")

        # Detectron2 0.6 passes per-image dicts; flatten to a bare COCO result list
        coco_results = list(
            itertools.chain(*[x["instances"] for x in predictions])
        )

        # ── CRITICAL: map contiguous category ids back to dataset ids ─────────
        # Detectron2 emits predictions with *contiguous* class ids (bird → 0),
        # but the ground-truth COCO json uses the original dataset ids (bird → 1).
        # COCOeval matches detections to GT by category_id, so without this remap
        # every detection is in the wrong category and AP is identically 0.0.
        # COCOEvaluator does this in its own _eval_predictions; this override must
        # reproduce it.  (Falls back to identity if metadata mapping is absent.)
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            id_map = self._metadata.thing_dataset_id_to_contiguous_id
            reverse_id_mapping = {v: k for k, v in id_map.items()}
            num_classes = len(reverse_id_mapping)
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"Prediction has class={category_id} but dataset only "
                    f"registers {num_classes} classes."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        self._logger.info(
            f"  {len(predictions)} images evaluated, {len(coco_results)} detections total"
        )

        if not coco_results:
            self._logger.info("No detections — mAP30 = 0.0")
            self._results["bbox"] = {"AP30": 0.0, "AR30": 0.0}
            return

        coco_dt = self._coco_api.loadRes(coco_results)

        coco_eval = COCOeval(self._coco_api, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array([self._iou_thresh])
        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        # Suppress per-category stdout from pycocotools
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            coco_eval.evaluate()
            coco_eval.accumulate()

        # ── Extract AP and AR from raw eval arrays ────────────────────────────
        # precision shape: [T, R, K, A, M]
        #   T=1 (our single IoU thresh), R=101 recall pts,
        #   K=num_categories, A=4 area ranges, M=3 max-det levels
        # recall shape: [T, K, A, M]
        # area index 0 = "all";  maxDets index 2 = 100 detections
        precision = coco_eval.eval.get("precision")  # [1, 101, K, 4, 3]
        recall    = coco_eval.eval.get("recall")      # [1, K, 4, 3]

        if precision is not None and precision.size > 0:
            p = precision[0, :, :, 0, 2]   # [101, K]
            valid = p[p >= 0]
            ap30 = float(np.mean(valid)) * 100.0 if valid.size > 0 else 0.0
        else:
            ap30 = 0.0

        if recall is not None and recall.size > 0:
            r = recall[0, :, 0, 2]          # [K]
            valid = r[r >= 0]
            ar30 = float(np.mean(valid)) * 100.0 if valid.size > 0 else 0.0
        else:
            ar30 = 0.0

        results = {"AP30": ap30, "AR30": ar30}
        self._logger.info("mAP30 results:\n" + create_small_table(results))
        self._results["bbox"] = results

    def evaluate(self):
        results = super().evaluate()
        if results and "bbox" in results and "AP30" not in results["bbox"]:
            results["bbox"]["AP30"] = 0.0
        return results
