"""
Scoring any subset of the test split with the project's own protocol.

Habitat, altitude and metadata analysis all ask the same question of different
partitions —
altitude, weather, site — so the metric code lives here once and the partitions
live in `survey_conditions`.

Two decisions make the resulting numbers comparable across subgroups, and both
differ from what a naive per-subgroup rescore would do:

**Detections are matched once, over the whole split, before any subsetting.**
The published protocol (`data_prep.point_metrics`) matches every detection
score-ordered and greedy, then reads metrics off that single matching at a
threshold.  Re-running the match inside a subgroup would let a detection that
lost its ground-truth bird to a higher-scoring rival elsewhere win it back here,
so a crop's outcome would depend on which subgroup it was being counted in.
Matching first and slicing second keeps every crop's outcome fixed.

**The operating threshold t\\* stays at the model's global value.**  t\\* is
chosen to maximise F1 over the whole split; re-selecting it per subgroup would
tune a different operating point for every habitat and turn a comparison of
habitats into a comparison of thresholds.  AP is threshold-free and unaffected
either way.

The consequence worth stating in any report: precision, recall and F1 here are
"this model, at its own single operating point, restricted to these crops" — not
"this model retuned for this habitat".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_prep import config
from data_prep.point_metrics import _ap_from_flags, _match_image

TAU = float(config.PAPER_TAU)


@dataclass
class MatchedRun:
    """One model's detections matched against ground truth, split-wide.

    Everything a subgroup needs is precomputed per image, so scoring a partition
    is a slice rather than a rematch.
    """

    key: str
    label: str
    kind: str
    t_star: float
    det_image_id: np.ndarray     # [D] image id per detection, all thresholds
    det_score: np.ndarray        # [D]
    det_tp: np.ndarray           # [D] bool, from the split-wide greedy match
    gt_count: dict[int, int]     # image id -> ground-truth birds
    pred_count: dict[int, int]   # image id -> detections kept at t*
    tp_count: dict[int, int]     # image id -> true positives at t*


def match_run(gt_counts: dict[int, int], gt_by_image: dict[int, list],
              cache: dict, t_star: float, key: str, label: str,
              kind: str) -> MatchedRun:
    """Match one model's cached detections against ground truth, split-wide."""
    order = np.argsort(cache["image_id"], kind="stable")
    img = cache["image_id"][order].astype(np.int64)
    xs = cache["x"][order].astype(np.float64)
    ys = cache["y"][order].astype(np.float64)
    sc = cache["score"][order].astype(np.float64)

    det_tp = np.zeros(len(img), dtype=bool)
    pred_count = {int(i): 0 for i in gt_counts}
    tp_count = {int(i): 0 for i in gt_counts}

    # Walk each image's contiguous detection block.
    uniq, starts = np.unique(img, return_index=True)
    starts = list(starts) + [len(img)]
    for n, image_id in enumerate(uniq):
        lo, hi = starts[n], starts[n + 1]
        idx = np.arange(lo, hi)
        idx = idx[np.argsort(-sc[idx], kind="stable")]   # descending score
        gt_xy = np.asarray(gt_by_image.get(int(image_id), []),
                           dtype=np.float64).reshape(-1, 2)
        det_xy = np.column_stack([xs[idx], ys[idx]])
        tp = _match_image(gt_xy, det_xy, TAU)
        det_tp[idx] = tp
        keep = sc[idx] >= t_star
        pred_count[int(image_id)] = int(keep.sum())
        tp_count[int(image_id)] = int((tp & keep).sum())

    return MatchedRun(key=key, label=label, kind=kind, t_star=t_star,
                      det_image_id=img, det_score=sc, det_tp=det_tp,
                      gt_count=dict(gt_counts), pred_count=pred_count,
                      tp_count=tp_count)


@dataclass
class SubgroupScore:
    """One model on one subgroup of crops."""

    group: object
    n_crops: int
    n_source_images: int
    n_birds: int
    ap: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    mae: float | None = None
    rmse: float | None = None
    count_bias: float | None = None      # signed % error on the total count
    pred_total: int = 0
    n_det: int = 0

    @property
    def sufficient(self) -> bool:
        """Whether this subgroup can carry a claim at all.

        A habitat represented by two source photographs is a statement about
        those photographs, not about the habitat, so the report marks it and
        keeps it out of ranking comparisons.
        """
        return self.n_birds >= MIN_BIRDS and self.n_source_images >= MIN_IMAGES


# Below either of these a subgroup is reported but never ranked or interpreted.
# Chosen so a group holds enough birds for a recall estimate to move less than a
# few points under resampling, and enough separate photographs that one unusual
# scene cannot define the group.
MIN_BIRDS = 200
MIN_IMAGES = 3


def score(run: MatchedRun, image_ids: list[int], group: object,
          n_source_images: int) -> SubgroupScore:
    """Score one model on one subgroup, at the model's global t*."""
    ids = np.asarray(sorted(image_ids), dtype=np.int64)
    id_set = set(int(i) for i in ids)

    n_birds = int(sum(run.gt_count.get(int(i), 0) for i in ids))
    s = SubgroupScore(group=group, n_crops=len(ids),
                      n_source_images=n_source_images, n_birds=n_birds)

    sel = np.isin(run.det_image_id, ids)
    sub_scores, sub_tp = run.det_score[sel], run.det_tp[sel]

    # AP over this subgroup's detections and its own bird total — threshold-free,
    # so it is the one number here that does not depend on t*.
    if n_birds and len(sub_scores):
        s.ap = float(_ap_from_flags(sub_scores, sub_tp, n_birds)[0]) * 100

    keep = sub_scores >= run.t_star
    s.n_det = int(keep.sum())
    tp_at_t = int(sub_tp[keep].sum())
    if s.n_det:
        s.precision = 100.0 * tp_at_t / s.n_det
    if n_birds:
        s.recall = 100.0 * tp_at_t / n_birds
    if s.precision is not None and s.recall is not None and (s.precision + s.recall) > 0:
        s.f1 = 2 * s.precision * s.recall / (s.precision + s.recall)

    # Counting uses every kept detection, matched or not: an operator reading a
    # total off the model sees false positives cancel misses exactly as here.
    gt_arr = np.array([run.gt_count.get(int(i), 0) for i in ids], dtype=np.float64)
    pr_arr = np.array([run.pred_count.get(int(i), 0) for i in ids], dtype=np.float64)
    err = pr_arr - gt_arr
    s.mae = float(np.abs(err).mean()) if len(err) else None
    s.rmse = float(np.sqrt((err ** 2).mean())) if len(err) else None
    s.pred_total = int(pr_arr.sum())
    if n_birds:
        s.count_bias = 100.0 * (s.pred_total - n_birds) / n_birds
    return s


def rank_changes(per_model: dict[str, list[SubgroupScore]], metric: str = "f1",
                 ) -> dict[object, list[tuple[str, float]]]:
    """Model ranking within each subgroup, best first.

    The habitat analysis asks whether rankings change between habitats; that is only
    meaningful where the subgroup is large enough to rank on, so insufficient
    groups are dropped here rather than silently ranked.
    """
    by_group: dict[object, list[tuple[str, float]]] = {}
    for key, scores in per_model.items():
        for s in scores:
            if not s.sufficient:
                continue
            v = getattr(s, metric, None)
            if v is None:
                continue
            by_group.setdefault(s.group, []).append((key, v))
    return {g: sorted(v, key=lambda kv: -kv[1]) for g, v in by_group.items()}


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Rank correlation between two orderings of the same models.

    Reported instead of eyeballing two ordered lists: it answers "did the
    ranking change" with one number, where +1 is identical and -1 reversed.
    """
    common = [k for k in a if k in b]
    if len(common) < 2:
        return None
    ia = {k: i for i, k in enumerate(a)}
    ib = {k: i for i, k in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s = (ia[x] - ia[y]) * (ib[x] - ib[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else None
