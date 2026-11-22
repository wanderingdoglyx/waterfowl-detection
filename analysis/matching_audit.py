"""
How much the matching rule itself moves the scores.

The published protocol matches detections to birds greedily in order of
confidence, accepting any bird within tau = 40 px. At the altitudes in this
corpus a bird is around 17 px across, so that radius spans more than one bird,
and a high-scoring detection on one object can claim a neighbouring bird ahead
of the detection that is actually sitting on it. Two things follow, and this
module measures both for every model:

* **Swaps.** False alarms that sit on a bird whose credit went to a detection
  further away. They change which detection is blamed, and they are what a
  failure panel shows as one bird "detected twice".
* **Lost true positives.** Greedy order can also leave a bird unclaimed that an
  optimal one-to-one assignment within the same radius would have matched.
  Recomputing with the Hungarian algorithm gives the number of true positives
  greedy matching forgoes, which is a direct understatement of recall.

The protocol is not changed. It is the published one, and every reported metric
stays comparable with it; this module states how far the rule itself moves the
numbers so the effect can be weighed rather than discovered in a figure.

Results are cached per run beside the rescored metrics, since the optimal
assignment is the slow part.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from data_prep import config
from analysis import predict, rescore, strata
from analysis.registry import ModelSpec

TAU = float(config.PAPER_TAU)


def cache_path(spec: ModelSpec, run_id: str) -> str:
    return os.path.join(rescore.SCORE_DIR, f"{spec.key}__{run_id}__matching.json")


def _optimal_tp(gxy: np.ndarray, dxy: np.ndarray) -> int:
    """Maximum one-to-one matches within tau, ties broken by total distance."""
    dist = np.linalg.norm(dxy[:, None, :] - gxy[None, :, :], axis=2)
    # A penalty far above any feasible total distance makes the solver maximise
    # the number of in-radius pairs first.
    cost = np.where(dist <= TAU, dist, 1e6)
    r, c = linear_sum_assignment(cost)
    return int((dist[r, c] <= TAU).sum())


def audit(gt: strata.GroundTruth, cache: dict, t_star: float) -> dict:
    out = strata.outcomes_at_t_star(gt, cache, t_star)
    keep = cache["score"] >= t_star
    x = cache["x"][keep].astype(np.float64)
    y = cache["y"][keep].astype(np.float64)

    g_order = np.argsort(gt.gt_image_id, kind="stable")
    g_sorted = gt.gt_image_id[g_order]
    d_order = np.argsort(out.det_image_id, kind="stable")
    d_sorted = out.det_image_id[d_order]

    n_fa = n_swap = tp_optimal = 0
    for image_id in np.unique(d_sorted):
        d_lo, d_hi = np.searchsorted(d_sorted, [image_id, image_id + 1])
        g_lo, g_hi = np.searchsorted(g_sorted, [image_id, image_id + 1])
        di = d_order[d_lo:d_hi]
        gi = g_order[g_lo:g_hi]
        fa = di[~out.det_tp[di]]
        n_fa += len(fa)
        if not len(gi):
            continue
        dxy = np.column_stack([x[di], y[di]])
        tp_optimal += _optimal_tp(gt.gt_xy[gi], dxy)

        credited = {int(out.det_match[i]): int(i) for i in di if out.det_match[i] >= 0}
        for i in fa:
            dist = np.hypot(gt.gt_xy[gi, 0] - x[i], gt.gt_xy[gi, 1] - y[i])
            k = int(np.argmin(dist)); j = int(gi[k])
            if dist[k] > 0.5 * gt.gt_size[j] or j not in credited:
                continue
            c = credited[j]
            if np.hypot(gt.gt_xy[j, 0] - x[c], gt.gt_xy[j, 1] - y[c]) > dist[k]:
                n_swap += 1

    tp_greedy = int(out.det_tp.sum())
    n_gt = int(len(gt.gt_size))
    return {
        "t_star": float(t_star), "tau": TAU, "n_gt": n_gt,
        "false_alarms": int(n_fa), "swaps": int(n_swap),
        "tp_greedy": tp_greedy, "tp_optimal": int(tp_optimal),
        "tp_lost": int(tp_optimal - tp_greedy),
        "recall_greedy": 100.0 * tp_greedy / n_gt,
        "recall_optimal": 100.0 * tp_optimal / n_gt,
    }


def ensure(spec: ModelSpec, run_id: str, gt: strata.GroundTruth | None = None,
           force: bool = False) -> dict:
    path = cache_path(spec, run_id)
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    gt = gt or strata.load_ground_truth()
    pm = rescore.ensure(spec, run_id)
    cache = predict.load(predict.cache_path(spec, run_id))
    result = audit(gt, cache, pm["counting"]["t_star"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result
