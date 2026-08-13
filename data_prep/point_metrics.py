"""
Point-based evaluation protocol from the OWL paper (Chacón et al., Section 4.3),
shared by all four models so every --eval reports the same table:

  Counting metrics   MAE, RMSE (per-image counts), total predicted vs GT count and
                     signed percentage error — computed at the counting threshold t*,
                     which is selected to minimise MAE on the test images (the paper
                     does the same and flags the optimistic bias; it is uniform
                     across models so relative comparisons are unaffected).
  Detection metrics  A predicted point is a TP if within tau (=40 px) of an unmatched
                     ground-truth point, greedy one-to-one matching.  We report the
                     threshold-free Average Precision (AP, primary ranking metric),
                     the trapezoidal area under the PR curve (AUC-PR), and
                     precision / recall / F1 at t*.
  Confidence         Bootstrap 95% CIs (B image-level resamples) for MAE, RMSE and AP.

Box detectors participate through their box centres (score = box confidence); OWL
detects points natively (score = heatmap peak).  Ground truth for everyone is the box
centre of the shared test COCO annotations, so all four models are scored against the
identical targets.

Matching detail: detections are processed in descending score order, each matched to
its nearest unmatched GT point within tau (the COCO convention).  Score-ordered greedy
matching has the prefix property — the matching restricted to detections with score >= t
equals the matching computed from only those detections — so a single pass yields exact
per-threshold TP/FP/FN counts for the PR curve, t* selection and the bootstrap.
(The paper sorts candidate pairs by distance instead; the two orderings differ only in
rare tie-like configurations and the distance variant has no prefix property.)

Pure numpy; runs in the `waterfowl` env and is imported by all four pipelines'
--eval steps.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "gt_points_from_coco",
    "paper_point_metrics",
    "format_report",
    "format_map30_block",
    "write_eval_txt",
]


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth helpers
# ──────────────────────────────────────────────────────────────────────────────

def gt_points_from_coco(coco_gt_path: str) -> Dict[int, List[Tuple[float, float]]]:
    """image_id -> [(cx, cy), ...] box centres for EVERY image in the COCO file
    (images without annotations map to an empty list — background-only crops count
    toward MAE/RMSE exactly as in the paper)."""
    with open(coco_gt_path) as f:
        coco = json.load(f)
    gt: Dict[int, List[Tuple[float, float]]] = {img["id"]: [] for img in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((x + w / 2.0, y + h / 2.0))
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Matching
# ──────────────────────────────────────────────────────────────────────────────

def _match_image(
    gt_xy: np.ndarray,          # [G, 2]
    det_xy: np.ndarray,         # [D, 2] sorted by descending score
    tau: float,
) -> np.ndarray:
    """TP flag (bool) per detection, score-ordered greedy NN matching within tau."""
    D = len(det_xy)
    tp = np.zeros(D, dtype=bool)
    if D == 0 or len(gt_xy) == 0:
        return tp
    taken = np.zeros(len(gt_xy), dtype=bool)
    # pairwise distances [D, G]
    d = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    for i in range(D):
        di = np.where(taken, np.inf, d[i])
        j = int(np.argmin(di))
        if di[j] <= tau:
            tp[i] = True
            taken[j] = True
    return tp


# ──────────────────────────────────────────────────────────────────────────────
# Core protocol
# ──────────────────────────────────────────────────────────────────────────────

def _ap_from_flags(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Exact AP (all-point interpolation) + trapezoidal AUC-PR from per-detection
    TP flags.  Returns (ap, auc_pr, precision_curve, recall_curve)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    order = np.argsort(-scores, kind="stable")
    tp_c = np.cumsum(tp[order])
    fp_c = np.cumsum(~tp[order])
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    # all-point interpolated AP: precision envelope integrated over recall
    p_env = np.maximum.accumulate(precision[::-1])[::-1]
    r_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - r_prev) * p_env))
    # trapezoidal area under the raw PR curve (paper reports it alongside AP)
    auc = float(np.trapz(precision, recall))
    return ap, auc, precision, recall


def paper_point_metrics(
    gt_by_image: Dict[object, Sequence[Tuple[float, float]]],
    dets_by_image: Dict[object, Sequence[Tuple[float, float, float]]],
    tau: float = 40.0,
    bootstrap: int = 1000,
    n_thresholds: int = 256,
    seed: int = 42,
) -> dict:
    """
    Compute the OWL-paper metric suite.

    Args:
        gt_by_image:   image_key -> [(x, y), ...]; MUST contain every test image,
                       including background-only ones (empty list).
        dets_by_image: image_key -> [(x, y, score), ...]; keys not present in
                       gt_by_image are ignored.
        tau:           TP matching radius in pixels (paper: 40).
        bootstrap:     number of image-level bootstrap resamples (paper: 1000);
                       0 disables CIs.
        n_thresholds:  size of the score grid used for t* selection and the
                       bootstrap PR curves.
        seed:          RNG seed for the bootstrap.
    """
    keys = list(gt_by_image.keys())
    N = len(keys)
    rng = np.random.default_rng(seed)

    # ── per-image matching ────────────────────────────────────────────────────
    per_scores: List[np.ndarray] = []
    per_tp:     List[np.ndarray] = []
    n_gt_img = np.zeros(N, dtype=np.int64)
    for i, k in enumerate(keys):
        gt_xy = np.asarray(gt_by_image[k], dtype=np.float64).reshape(-1, 2)
        n_gt_img[i] = len(gt_xy)
        dets = np.asarray(list(dets_by_image.get(k, ())), dtype=np.float64).reshape(-1, 3)
        order = np.argsort(-dets[:, 2], kind="stable")
        dets = dets[order]
        per_scores.append(dets[:, 2])
        per_tp.append(_match_image(gt_xy, dets[:, :2], tau))

    n_gt_total = int(n_gt_img.sum())
    all_scores = np.concatenate(per_scores) if per_scores else np.array([])
    all_tp = np.concatenate(per_tp) if per_tp else np.array([], dtype=bool)

    # ── threshold-free detection metrics ──────────────────────────────────────
    ap, auc_pr, _, _ = _ap_from_flags(all_scores, all_tp, n_gt_total)

    # ── score grid (shared by t* selection and bootstrap) ─────────────────────
    if len(all_scores):
        qs = np.linspace(0.0, 1.0, n_thresholds)
        grid = np.unique(np.quantile(all_scores, qs))
    else:
        grid = np.array([0.0])
    T = len(grid)

    # per-image, per-threshold counts.  searchsorted on each image's ascending
    # scores gives #dets >= t; same on TP-only scores gives TP(t).
    cnt = np.zeros((N, T), dtype=np.int64)
    tpm = np.zeros((N, T), dtype=np.int64)
    for i in range(N):
        s = per_scores[i][::-1]                     # ascending
        cnt[i] = len(s) - np.searchsorted(s, grid, side="left")
        st = per_scores[i][per_tp[i]][::-1]         # ascending TP scores
        tpm[i] = len(st) - np.searchsorted(st, grid, side="left")

    # ── counting threshold t* = argmin MAE over the grid ──────────────────────
    abs_err = np.abs(cnt - n_gt_img[:, None])       # [N, T]
    mae_t = abs_err.mean(axis=0)
    t_idx = int(np.argmin(mae_t))
    t_star = float(grid[t_idx])

    err_star = cnt[:, t_idx] - n_gt_img             # signed per-image error at t*
    mae = float(np.abs(err_star).mean())
    rmse = float(np.sqrt(np.mean(err_star.astype(np.float64) ** 2)))
    total_pred = int(cnt[:, t_idx].sum())
    signed_pct = (100.0 * (total_pred - n_gt_total) / n_gt_total) if n_gt_total else 0.0

    tp_star = int(tpm[:, t_idx].sum())
    fp_star = int(cnt[:, t_idx].sum() - tp_star)
    fn_star = n_gt_total - tp_star
    prec_star = tp_star / max(tp_star + fp_star, 1)
    rec_star = tp_star / max(n_gt_total, 1)
    f1_star = 2 * prec_star * rec_star / max(prec_star + rec_star, 1e-12)

    out = {
        "protocol": {
            "tau_px": tau, "bootstrap_B": bootstrap, "n_images": N,
            "n_gt_points": n_gt_total, "n_detections": int(len(all_scores)),
            "score_grid_size": T, "matching": "score-ordered greedy NN, one-to-one",
        },
        "detection": {
            "ap": ap, "auc_pr": auc_pr,
            "precision_at_t_star": prec_star,
            "recall_at_t_star": rec_star,
            "f1_at_t_star": f1_star,
        },
        "counting": {
            "t_star": t_star, "mae": mae, "rmse": rmse,
            "total_gt": n_gt_total, "total_pred_at_t_star": total_pred,
            "signed_pct_error": signed_pct,
        },
    }

    # ── bootstrap 95% CIs (image-level resamples via multinomial weights) ─────
    if bootstrap and N > 0:
        B = bootstrap
        w = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)  # [B, N]

        abs_b = w @ np.abs(err_star).astype(np.float64) / N                     # [B]
        sq_b = w @ (err_star.astype(np.float64) ** 2) / N
        mae_ci = np.percentile(abs_b, [2.5, 97.5])
        rmse_ci = np.percentile(np.sqrt(sq_b), [2.5, 97.5])

        # PR/AP per resample from per-threshold sums (grid AP; t* held fixed,
        # exactly as the paper prescribes)
        tp_b = w @ tpm.astype(np.float64)            # [B, T]
        cnt_b = w @ cnt.astype(np.float64)
        gt_b = w @ n_gt_img.astype(np.float64)       # [B]
        ap_b = np.zeros(B)
        for b in range(B):
            if gt_b[b] <= 0:
                continue
            # grid is ascending => descending threshold order for the PR sweep
            tp_curve = tp_b[b][::-1]
            det_curve = cnt_b[b][::-1]
            rec = tp_curve / gt_b[b]
            prc = tp_curve / np.maximum(det_curve, 1e-12)
            p_env = np.maximum.accumulate(prc[::-1])[::-1]
            r_prev = np.concatenate([[0.0], rec[:-1]])
            ap_b[b] = np.sum((rec - r_prev) * p_env)
        ap_ci = np.percentile(ap_b, [2.5, 97.5])

        out["ci95"] = {
            "mae": [float(mae_ci[0]), float(mae_ci[1])],
            "rmse": [float(rmse_ci[0]), float(rmse_ci[1])],
            "ap": [float(ap_ci[0]), float(ap_ci[1])],
        }

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def format_report(m: dict, model_name: str = "") -> str:
    """Human-readable block for the --eval console output."""
    p, d, c = m["protocol"], m["detection"], m["counting"]
    ci = m.get("ci95", {})

    def _ci(key: str, scale: float = 1.0, fmt: str = ".3f") -> str:
        if key not in ci:
            return ""
        lo, hi = ci[key]
        return f"  [95% CI {lo * scale:{fmt}}–{hi * scale:{fmt}}]"

    lines = [
        f"── OWL-paper point metrics{f' — {model_name}' if model_name else ''} "
        f"(τ = {p['tau_px']:.0f} px, {p['n_images']} images, "
        f"{p['n_gt_points']} GT points) ──",
        f"  AP        : {d['ap'] * 100:6.2f}%{_ci('ap', 100, '.2f')}",
        f"  AUC-PR    : {d['auc_pr'] * 100:6.2f}%",
        f"  Precision : {d['precision_at_t_star'] * 100:6.2f}%   (at t* = {c['t_star']:.3f})",
        f"  Recall    : {d['recall_at_t_star'] * 100:6.2f}%",
        f"  F1        : {d['f1_at_t_star'] * 100:6.2f}%",
        f"  MAE       : {c['mae']:6.3f}{_ci('mae')}",
        f"  RMSE      : {c['rmse']:6.3f}{_ci('rmse')}",
        f"  Count     : {c['total_pred_at_t_star']} predicted vs {c['total_gt']} GT "
        f"({c['signed_pct_error']:+.1f}%)",
    ]
    return "\n".join(lines)


def format_map30_block(ap30: float, ar30: float) -> str:
    """Box-detection mAP30 block for the box models' results file."""
    return ("── mAP30 (box detection, COCOeval@IoU=0.30) ──\n"
            f"  mAP30 : {ap30:6.2f}%\n"
            f"  mAR30 : {ar30:6.2f}%")


def write_eval_txt(path: str, sections) -> str:
    """Join non-empty `sections` with blank lines and write them to `path`.
    Returns the full text (so callers can print it too)."""
    text = "\n\n".join(s for s in sections if s)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Adapters for the four pipelines
# ──────────────────────────────────────────────────────────────────────────────

def dets_from_coco_results(coco_results: Iterable[dict]) -> Dict[int, List[Tuple[float, float, float]]]:
    """COCO result list ({image_id, bbox [x,y,w,h], score}) -> box centres by image."""
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for r in coco_results:
        x, y, w, h = r["bbox"]
        out.setdefault(r["image_id"], []).append(
            (x + w / 2.0, y + h / 2.0, float(r["score"])))
    return out


def dets_from_owl_points(detections: Iterable[dict], coco_gt_path: str,
                         ) -> Dict[int, List[Tuple[float, float, float]]]:
    """OWL point detections ({image: basename, x, y, score}) -> keyed by image_id."""
    with open(coco_gt_path) as f:
        coco = json.load(f)
    base_to_id = {os.path.basename(i["file_name"]): i["id"] for i in coco["images"]}
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for det in detections:
        image_id = base_to_id.get(det["image"])
        if image_id is not None:
            out.setdefault(image_id, []).append(
                (float(det["x"]), float(det["y"]), float(det["score"])))
    return out
