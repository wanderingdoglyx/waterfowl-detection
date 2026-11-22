"""
Tasks 2.3 and 2.4 — how performance varies with apparent bird size and with
bird density, rather than as one number over the whole test set.

Both tasks need the same thing the aggregate metrics throw away: *which*
ground-truth bird each model found and which detection was spurious.  So each
model's cached detections are re-matched to the ground truth here, using the
same rule as the published protocol (score-ordered greedy nearest-neighbour,
one-to-one, within tau = 40 px), at that model's own operating threshold t* —
the threshold its precision/recall/F1 are already reported at.

Apparent size is defined as **sqrt(bounding-box area)**, i.e. the side of the
equivalent square, in crop pixels.  It is directly comparable across elongated
and compact boxes, is in pixel units a reader can reason about, and needs only
the box annotations that already exist.

Bin edges are taken from the observed distribution (quantiles), not from round
numbers, so every bin holds a comparable share of the data instead of a few bins
holding almost everything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from data_prep import config
from analysis import predict
from analysis.registry import ModelSpec

TAU = float(config.PAPER_TAU)


# ── Ground truth ─────────────────────────────────────────────────────────────

@dataclass
class GroundTruth:
    """Test-split ground truth, indexed for stratified scoring."""

    image_ids: np.ndarray            # [N] every test image, background included
    gt_image_id: np.ndarray          # [G] image id per GT bird
    gt_xy: np.ndarray                # [G, 2] box centres
    gt_size: np.ndarray              # [G] sqrt(box area), px
    gt_w: np.ndarray                 # [G]
    gt_h: np.ndarray                 # [G]
    counts: dict[int, int]           # image id -> number of GT birds


def load_ground_truth(coco_path: str | None = None) -> GroundTruth:
    with open(coco_path or predict.TEST_JSON) as f:
        coco = json.load(f)

    image_ids = np.array([im["id"] for im in coco["images"]], dtype=np.int64)
    counts = {int(i): 0 for i in image_ids}

    img_id, xs, ys, sizes, ws, hs = [], [], [], [], [], []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        img_id.append(a["image_id"])
        xs.append(x + w / 2.0)
        ys.append(y + h / 2.0)
        sizes.append(float(np.sqrt(max(w * h, 0.0))))
        ws.append(w)
        hs.append(h)
        counts[int(a["image_id"])] = counts.get(int(a["image_id"]), 0) + 1

    return GroundTruth(
        image_ids=image_ids,
        gt_image_id=np.asarray(img_id, dtype=np.int64),
        gt_xy=np.column_stack([xs, ys]).astype(np.float64) if xs else np.zeros((0, 2)),
        gt_size=np.asarray(sizes, dtype=np.float64),
        gt_w=np.asarray(ws, dtype=np.float64),
        gt_h=np.asarray(hs, dtype=np.float64),
        counts=counts,
    )


# ── Matching ─────────────────────────────────────────────────────────────────

def _match_one(gt_xy: np.ndarray, det_xy: np.ndarray, tau: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score-ordered greedy NN matching, one-to-one, within tau.

    `det_xy` must already be sorted by descending score. Returns
    (tp flag per detection, matched flag per GT, GT index per detection or -1).
    The published protocol keeps only the first; per-bird recall needs the
    second; the third says *which* detection was credited with each bird.

    That last one matters because the rule is greedy by confidence, not by
    distance. A detection 30 px from a bird can claim it ahead of one sitting on
    the bird if it scores higher, which leaves the well-placed detection scored
    as a false alarm. Nothing in the first two arrays reveals that; the pairing
    does.
    """
    d_n, g_n = len(det_xy), len(gt_xy)
    tp = np.zeros(d_n, dtype=bool)
    matched = np.zeros(g_n, dtype=bool)
    pair = np.full(d_n, -1, dtype=np.int64)
    if d_n == 0 or g_n == 0:
        return tp, matched, pair
    dist = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    for i in range(d_n):
        di = np.where(matched, np.inf, dist[i])
        j = int(np.argmin(di))
        if di[j] <= tau:
            tp[i] = True
            matched[j] = True
            pair[i] = j
    return tp, matched, pair


@dataclass
class Outcomes:
    """Per-bird and per-detection outcomes for one model at its operating point."""

    gt_matched: np.ndarray           # [G] bool, aligned with GroundTruth.gt_*
    det_image_id: np.ndarray         # [D]
    det_tp: np.ndarray               # [D] bool
    det_size: np.ndarray             # [D] sqrt(box area) or NaN for point models
    pred_count: dict[int, int]       # image id -> detections kept at t*
    t_star: float
    # [D] index into GroundTruth.gt_* of the bird each detection was credited
    # with, or -1 for a false alarm.
    det_match: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


def outcomes_at_t_star(gt: GroundTruth, cache: dict, t_star: float) -> Outcomes:
    """Match one model's detections (score >= t*) against the ground truth."""
    keep = cache["score"] >= t_star
    d_img = cache["image_id"][keep].astype(np.int64)
    d_x, d_y = cache["x"][keep].astype(np.float64), cache["y"][keep].astype(np.float64)
    d_s = cache["score"][keep].astype(np.float64)
    d_w, d_h = cache["w"][keep].astype(np.float64), cache["h"][keep].astype(np.float64)
    det_size = np.sqrt(np.maximum(d_w * d_h, 0.0))  # NaN propagates for point models

    gt_matched = np.zeros(len(gt.gt_size), dtype=bool)
    det_tp = np.zeros(len(d_img), dtype=bool)
    det_match = np.full(len(d_img), -1, dtype=np.int64)
    pred_count = {int(i): 0 for i in gt.image_ids}

    gt_order = np.argsort(gt.gt_image_id, kind="stable")
    gt_sorted = gt.gt_image_id[gt_order]
    det_order = np.argsort(d_img, kind="stable")
    det_sorted = d_img[det_order]

    for image_id in gt.image_ids:
        g_lo, g_hi = np.searchsorted(gt_sorted, [image_id, image_id + 1])
        d_lo, d_hi = np.searchsorted(det_sorted, [image_id, image_id + 1])
        g_idx = gt_order[g_lo:g_hi]
        d_idx = det_order[d_lo:d_hi]
        pred_count[int(image_id)] = len(d_idx)
        if len(d_idx) == 0 and len(g_idx) == 0:
            continue
        # Detections must enter the greedy match in descending score order.
        d_idx = d_idx[np.argsort(-d_s[d_idx], kind="stable")]
        tp, matched, pair = _match_one(gt.gt_xy[g_idx],
                                       np.column_stack([d_x[d_idx], d_y[d_idx]]), TAU)
        det_tp[d_idx] = tp
        gt_matched[g_idx] = matched
        hit = pair >= 0
        det_match[d_idx[hit]] = g_idx[pair[hit]]

    return Outcomes(gt_matched=gt_matched, det_image_id=d_img, det_tp=det_tp,
                    det_size=det_size, pred_count=pred_count, t_star=t_star,
                    det_match=det_match)


# ── Binning ──────────────────────────────────────────────────────────────────

def quantile_bins(values: np.ndarray, n_bins: int = 6) -> np.ndarray:
    """Edges at equal-count quantiles, deduplicated.

    Equal-count beats round numbers here: apparent size is heavily skewed, so
    fixed-width bins would put almost every bird in one or two of them and leave
    the rest too sparse to read.
    """
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, qs)
    edges = np.unique(np.round(edges, 2))
    edges[0] = np.floor(values.min() * 100) / 100
    edges[-1] = np.ceil(values.max() * 100) / 100
    return edges


def density_bins(counts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Density ranges from the observed per-crop counts.

    Empty crops are their own bin — a background crop can only ever contribute
    false positives, so folding it in with sparse crops would confound "no birds
    to find" with "few birds to find".
    """
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return np.array([0, 1]), ["0"]
    # Weighted towards the dense tail: quartile-width bands would put every crop
    # from 18 to 422 birds in one bucket, which is exactly the regime Task 2.4
    # asks about.
    qs = np.percentile(nonzero, [20, 40, 60, 75, 85, 93, 97, 99.5])
    edges = [0, 1]
    for q in qs:
        e = int(np.ceil(q))
        if e > edges[-1]:
            edges.append(e)
    top = int(nonzero.max()) + 1
    if top > edges[-1]:
        edges.append(top)
    edges = np.array(sorted(set(edges)))

    labels = ["0 (background)"]
    for i in range(1, len(edges) - 1):
        lo, hi = edges[i], edges[i + 1] - 1
        labels.append(f"{lo}" if lo == hi else f"{lo}–{hi}")
    return edges, labels


def size_bin_labels(edges: np.ndarray) -> list[str]:
    return [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]


# ── Task 2.3: performance vs apparent size ───────────────────────────────────

@dataclass
class SizeBinResult:
    label: str
    lo: float
    hi: float
    n_gt: int
    recall: float | None = None
    fnr: float | None = None
    precision: float | None = None       # box models only (needs detection size)
    f1: float | None = None
    n_det: int | None = None
    count_ratio: float | None = None     # detected / ground truth in this bin


def size_analysis(gt: GroundTruth, out: Outcomes, edges: np.ndarray,
                  has_box_size: bool) -> list[SizeBinResult]:
    """Recall/FNR per GT size bin; precision/F1 too when detections have a size.

    Precision is binned by the *detection's* own size, which only exists for box
    models. Point detectors emit no extent, so a false positive cannot be
    assigned a size bin and precision is left undefined rather than guessed.
    """
    idx = np.digitize(gt.gt_size, edges[1:-1], right=False)
    det_idx = (np.digitize(out.det_size, edges[1:-1], right=False)
               if has_box_size else None)

    results = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n_gt = int(sel.sum())
        r = SizeBinResult(label=f"{edges[b]:.0f}–{edges[b+1]:.0f}",
                          lo=float(edges[b]), hi=float(edges[b + 1]), n_gt=n_gt)
        if n_gt:
            recall = float(out.gt_matched[sel].mean())
            r.recall = recall * 100
            r.fnr = (1 - recall) * 100

        if has_box_size and det_idx is not None:
            dsel = (det_idx == b) & ~np.isnan(out.det_size)
            n_det = int(dsel.sum())
            r.n_det = n_det
            if n_det:
                prec = float(out.det_tp[dsel].mean())
                r.precision = prec * 100
                if r.recall is not None and (prec + recall) > 0:
                    r.f1 = 2 * prec * recall / (prec + recall) * 100
            if n_gt:
                r.count_ratio = n_det / n_gt
        results.append(r)
    return results


# ── Task 2.4: performance vs bird density ────────────────────────────────────

@dataclass
class DensityBinResult:
    label: str
    lo: int
    hi: int
    n_crops: int
    n_gt: int
    recall: float | None = None
    precision: float | None = None
    f1: float | None = None
    mae: float | None = None
    rmse: float | None = None
    signed_pct_error: float | None = None
    pred_total: int = 0


def density_analysis(gt: GroundTruth, out: Outcomes, edges: np.ndarray,
                     labels: list[str]) -> list[DensityBinResult]:
    """Detection and counting quality per crop-density band.

    Counting uses every kept detection, matched or not — an operator counting
    birds sees the raw count, so a false positive must be allowed to cancel a
    miss exactly as it would in practice.
    """
    image_ids = gt.image_ids
    gt_counts = np.array([gt.counts[int(i)] for i in image_ids])
    pred_counts = np.array([out.pred_count[int(i)] for i in image_ids])
    bin_of_image = np.digitize(gt_counts, edges[1:-1], right=False)

    # Per-image TP / detection totals, for recall and precision inside a band.
    tp_by_image: dict[int, int] = {}
    det_by_image: dict[int, int] = {}
    for image_id, tp in zip(out.det_image_id, out.det_tp):
        det_by_image[int(image_id)] = det_by_image.get(int(image_id), 0) + 1
        if tp:
            tp_by_image[int(image_id)] = tp_by_image.get(int(image_id), 0) + 1

    tp_arr = np.array([tp_by_image.get(int(i), 0) for i in image_ids])
    det_arr = np.array([det_by_image.get(int(i), 0) for i in image_ids])

    results = []
    for b in range(len(edges) - 1):
        sel = bin_of_image == b
        n_crops = int(sel.sum())
        n_gt = int(gt_counts[sel].sum())
        label = labels[b] if b < len(labels) else f"bin {b}"
        r = DensityBinResult(label=label, lo=int(edges[b]), hi=int(edges[b + 1] - 1),
                             n_crops=n_crops, n_gt=n_gt)
        if n_crops == 0:
            results.append(r)
            continue

        err = pred_counts[sel] - gt_counts[sel]
        r.mae = float(np.abs(err).mean())
        r.rmse = float(np.sqrt((err ** 2).mean()))
        r.pred_total = int(pred_counts[sel].sum())
        if n_gt:
            r.recall = float(tp_arr[sel].sum()) / n_gt * 100
            r.signed_pct_error = (r.pred_total - n_gt) / n_gt * 100
        n_det = int(det_arr[sel].sum())
        if n_det:
            r.precision = float(tp_arr[sel].sum()) / n_det * 100
        if r.recall is not None and r.precision is not None and (r.recall + r.precision) > 0:
            r.f1 = 2 * r.recall * r.precision / (r.recall + r.precision)
        results.append(r)
    return results
