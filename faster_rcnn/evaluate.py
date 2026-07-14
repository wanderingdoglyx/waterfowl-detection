"""
Evaluation utilities — mAP30 (IoU threshold = 0.30, per Section 4.4).

11-point interpolated AP per class, averaged over all foreground classes.
"""

from __future__ import annotations
import torch
from typing import List, Dict


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for boxes in [x1,y1,x2,y2] format."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ix1 = torch.max(a[:, None, 0], b[None, :, 0])
    iy1 = torch.max(a[:, None, 1], b[None, :, 1])
    ix2 = torch.min(a[:, None, 2], b[None, :, 2])
    iy2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _compute_ap_single_class(
    pred_boxes:  List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes:    List[torch.Tensor],
    gt_labels:   List[torch.Tensor],
    class_id:    int,
    iou_thresh:  float,
) -> float:
    all_scores: list[float] = []
    all_tp:     list[int]   = []
    all_fp:     list[int]   = []
    n_gt_total = 0

    for i in range(len(pred_boxes)):
        pm = pred_labels[i] == class_id
        gm = gt_labels[i]   == class_id
        pb, ps, gb = pred_boxes[i][pm], pred_scores[i][pm], gt_boxes[i][gm]

        n_gt_total += len(gb)
        if len(pb) == 0:
            continue

        order      = torch.argsort(ps, descending=True)
        pb, ps     = pb[order], ps[order]
        gt_matched = torch.zeros(len(gb), dtype=torch.bool)

        for k in range(len(pb)):
            score = ps[k].item()
            if len(gb) == 0:
                all_scores.append(score); all_tp.append(0); all_fp.append(1)
                continue
            ious              = _box_iou(pb[k].unsqueeze(0), gb)[0]
            best_iou, best_j  = ious.max(0)
            if best_iou.item() >= iou_thresh and not gt_matched[best_j]:
                gt_matched[best_j] = True
                all_scores.append(score); all_tp.append(1); all_fp.append(0)
            else:
                all_scores.append(score); all_tp.append(0); all_fp.append(1)

    if n_gt_total == 0:
        return 0.0

    order = sorted(range(len(all_scores)), key=lambda i: -all_scores[i])
    tp_c, fp_c = 0, 0
    prec, rec  = [], []
    for i in order:
        tp_c += all_tp[i]; fp_c += all_fp[i]
        prec.append(tp_c / (tp_c + fp_c))
        rec.append(tp_c / n_gt_total)

    # 11-point interpolation
    ap = sum(max((p for p, r in zip(prec, rec) if r >= t), default=0.0)
             for t in [i / 10 for i in range(11)]) / 11.0
    return ap


def compute_map30(
    predictions: List[Dict[str, torch.Tensor]],
    targets:     List[Dict[str, torch.Tensor]],
    iou_thresh:  float = 0.30,
) -> Dict[str, float]:
    """
    Args:
        predictions : list of {'boxes': Tensor[N,4], 'scores': Tensor[N], 'labels': Tensor[N]}
        targets     : list of {'boxes': Tensor[M,4], 'labels': Tensor[M]}
        iou_thresh  : 0.30 per paper (mAP30)
    Returns:
        {'mAP': float, 'AP_per_class': {class_id: float}}
    """
    pb  = [p["boxes"].cpu()  for p in predictions]
    ps  = [p["scores"].cpu() for p in predictions]
    pl  = [p["labels"].cpu() for p in predictions]
    gb  = [t["boxes"].cpu()  for t in targets]
    gl  = [t["labels"].cpu() for t in targets]

    all_gt = torch.cat(gl) if any(len(g) for g in gl) else torch.tensor([])
    class_ids = [int(c) for c in all_gt.unique().tolist() if c > 0]

    if not class_ids:
        return {"mAP": 0.0, "AP_per_class": {}}

    ap_per_class = {
        cls: _compute_ap_single_class(pb, ps, pl, gb, gl, cls, iou_thresh)
        for cls in class_ids
    }
    return {
        "mAP": sum(ap_per_class.values()) / len(ap_per_class),
        "AP_per_class": ap_per_class,
    }


@torch.no_grad()
def run_inference(model, dataloader, device) -> tuple:
    """Collect all predictions and targets from a DataLoader."""
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        preds  = model(images)
        all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
        all_targets.extend(targets)
    return all_preds, all_targets
