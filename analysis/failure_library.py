"""
Failure-case library — a structured collection of representative failures.

The aggregate tables say how often a model fails. They cannot say what the
failure looks like, and "AP = 92.3" gives an operator nothing to act on. This
module builds the other half: a fixed set of failure *types*, each illustrated
by a real crop, each carrying the five things needed to reason about it —

    1. the crop itself            (rendered panel, ground truth vs prediction)
    2. the ground truth           (bird count and positions)
    3. the model prediction       (what it found, missed, and invented)
    4. model and confidence       (which model, at what threshold, how sure)
    5. the survey metadata        (habitat, altitude, weather, site, dataset)

**Every case is selected by a stated rule applied to the entire test split**, not
chosen by eye. Each record carries the rule that selected it and how many crops
were eligible, so the library demonstrates the data's failure modes rather than
the author's expectations. Change the models or the split and the same rules pick
whatever now satisfies them.

Output:

    output/analysis/models_v1/failures/<slug>.png     rendered panels
    output/analysis/models_v1/failure_library.json    structured records

    python -m analysis.failure_library
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config
from analysis import failure_render, metadata, predict, registry, rescore, strata

# Habitats where the background itself is the difficulty: vegetation and
# structure that break a bird's outline, as opposed to open water or bare ground
# where a bird is a high-contrast blob. Task 3.1 measured these as the worst.
COMPLEX_HABITATS = ("Trees", "Wooded", "ShrubScrub", "WaterVegetation", "Lotus")

OUT_DIR = os.path.join(config.OUTPUT_DIR, "analysis", "models_v1")
FIG_DIR = os.path.join(OUT_DIR, "failures")


@dataclass
class ModelOutcome:
    """One model's behaviour on one crop, including how confident it was."""

    key: str
    label: str
    kind: str
    t_star: float
    found: int
    missed: int
    false_positives: int
    pred_count: int
    count_error: int
    recall_pct: float | None
    conf_min: float | None = None
    conf_max: float | None = None
    conf_mean: float | None = None
    top_false_positive_conf: float | None = None


@dataclass
class FailureCase:
    """One entry in the library."""

    slug: str
    failure_type: str
    heading: str
    criteria: str
    n_eligible: int
    image_id: int
    crop_file: str
    figure: str | None
    dataset: str
    habitat: str | None
    altitude_m: int | None
    weather: str | None
    site: str | None
    n_gt: int
    median_bird_px: float | None
    models: list[ModelOutcome] = field(default_factory=list)
    corroboration: dict | None = None
    duplicates: dict | None = None


# ── Per-crop measurements ────────────────────────────────────────────────────

def _outcome(key: str, m: dict, gt, image_id: int) -> ModelOutcome:
    out = m["out"]
    g = gt.gt_image_id == image_id
    matched = out.gt_matched[g]
    d = out.det_image_id == image_id
    tp = out.det_tp[d]

    keep = m["cache"]["score"] >= m["t_star"]
    conf = m["cache"]["score"][keep][d]
    fp_conf = conf[~tp] if len(conf) == len(tp) else np.array([])

    n_gt = int(g.sum())
    found = int(matched.sum())
    return ModelOutcome(
        key=key, label=m["spec"].label, kind=m["spec"].kind,
        t_star=float(m["t_star"]),
        found=found, missed=int((~matched).sum()),
        false_positives=int((~tp).sum()),
        pred_count=int(out.pred_count[image_id]),
        count_error=int(out.pred_count[image_id]) - n_gt,
        recall_pct=(100.0 * found / n_gt) if n_gt else None,
        conf_min=float(conf.min()) if len(conf) else None,
        conf_max=float(conf.max()) if len(conf) else None,
        conf_mean=float(conf.mean()) if len(conf) else None,
        top_false_positive_conf=float(fp_conf.max()) if len(fp_conf) else None,
    )


def _recall(models, key, gt, image_id) -> float | None:
    g = gt.gt_image_id == image_id
    n = int(g.sum())
    if not n:
        return None
    return float(models[key]["out"].gt_matched[g].sum()) / n


def _median_size(gt, image_id) -> float | None:
    g = gt.gt_image_id == image_id
    return float(np.median(gt.gt_size[g])) if g.any() else None


def corroboration(models: dict, image_id: int, primary: str,
                  tau: float = float(config.PAPER_TAU)) -> dict:
    """How many independent models fire at each of one model's false positives.

    A single model producing detections where no bird is annotated is a false
    positive. Every model in the set producing a detection at the *same* place,
    when those models differ in architecture, training regime and output type,
    is a different proposition: shared agreement of that kind is far more
    consistent with a bird the annotator missed than with eight independent
    hallucinations landing on the same coordinates.

    Returns the number of unmatched detections, how many were corroborated by
    every model, and the confidence the primary model assigned them.
    """
    m = models[primary]
    d = m["out"].det_image_id == image_id
    keep = m["cache"]["score"] >= m["t_star"]
    xy = np.column_stack([m["cache"]["x"][keep][d], m["cache"]["y"][keep][d]])
    tp = m["out"].det_tp[d]
    conf = m["cache"]["score"][keep][d]
    un, un_conf = xy[~tp], conf[~tp]
    if not len(un):
        return {"n_unmatched": 0, "n_corroborated": 0, "n_models": len(models),
                "median_conf": None}

    votes = np.zeros(len(un), dtype=int)
    for other in models.values():
        od = other["out"].det_image_id == image_id
        okeep = other["cache"]["score"] >= other["t_star"]
        oxy = np.column_stack([other["cache"]["x"][okeep][od],
                               other["cache"]["y"][okeep][od]])
        if not len(oxy):
            continue
        dist = np.linalg.norm(un[:, None, :] - oxy[None, :, :], axis=2).min(axis=1)
        votes += (dist <= tau)
    return {"n_unmatched": int(len(un)),
            "n_corroborated": int((votes >= len(models)).sum()),
            "n_models": len(models),
            "median_conf": float(np.median(un_conf))}


def false_alarm_breakdown(models: dict, gt, image_id: int, primary: str,
                          tau: float = float(config.PAPER_TAU)) -> dict:
    """Classify one model's false alarms by what actually produced them.

    A detection matching no annotated bird is scored the same way whatever
    caused it, but three different things cause it and they have three different
    remedies:

    * **Same-object duplicate.** Two detections on one bird; the protocol can
      match only one, so the second is an error. Removed by stricter
      non-maximum suppression at no cost in recall. Identified by box overlap
      with the detection that did match, not by proximity: at these altitudes a
      bird is around 17 px across while the matching radius is 40 px, so "close
      to a matched bird" spans a neighbourhood that comfortably holds a second
      whole bird.
    * **Matching swap.** The false alarm sits on a bird, but the protocol
      credited that bird to a different detection further away that scored
      higher. The model placed a detection correctly and the score-ordered
      greedy match assigned the credit elsewhere. This is a property of the
      evaluation, not of the model, and looks like a double detection in a
      panel.
    * **Corroborated.** Every other model fires in the same place. Consistent
      with a bird that is present and was not annotated, and not fixable in the
      model at all.
    * **Uncorroborated.** No other model agrees, and no box overlap. This is the
      residue that behaves like a genuine discrimination error.

    Point detectors emit no extent, so for those the duplicate test falls back to
    a separation of half the median annotated bird size in the crop.
    """
    m = models[primary]
    g = gt.gt_image_id == image_id
    d = m["out"].det_image_id == image_id
    keep = m["cache"]["score"] >= m["t_star"]
    x = m["cache"]["x"][keep][d]
    y = m["cache"]["y"][keep][d]
    w = m["cache"]["w"][keep][d]
    h = m["cache"]["h"][keep][d]
    tp = m["out"].det_tp[d]
    fp_i, tp_i = np.where(~tp)[0], np.where(tp)[0]
    n_fp = int(len(fp_i))
    out = {"n_fp": n_fp, "duplicate": 0, "swap": 0, "corroborated": 0,
           "uncorroborated": 0,
           "n_models": len(models), "tau": tau,
           "median_bird_px": float(np.median(gt.gt_size[g])) if g.any() else None}
    if not n_fp:
        return out

    has_box = bool(np.isfinite(w).any())
    if has_box:
        b = np.column_stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

    def overlaps_matched(i) -> bool:
        if not len(tp_i):
            return False
        if has_box:
            a, B = b[i], b[tp_i]
            xx1 = np.maximum(a[0], B[:, 0]); yy1 = np.maximum(a[1], B[:, 1])
            xx2 = np.minimum(a[2], B[:, 2]); yy2 = np.minimum(a[3], B[:, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            ar = (a[2] - a[0]) * (a[3] - a[1])
            br = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
            return bool((inter / np.maximum(ar + br - inter, 1e-9) > 0.1).any())
        near = 0.5 * (out["median_bird_px"] or tau)
        return bool((np.hypot(x[tp_i] - x[i], y[tp_i] - y[i]) <= near).any())

    xy = np.column_stack([x, y])[fp_i]
    votes = np.zeros(len(fp_i), dtype=int)
    for other in models.values():
        od = other["out"].det_image_id == image_id
        okeep = other["cache"]["score"] >= other["t_star"]
        oxy = np.column_stack([other["cache"]["x"][okeep][od],
                               other["cache"]["y"][okeep][od]])
        if len(oxy):
            votes += (np.linalg.norm(xy[:, None, :] - oxy[None, :, :],
                                     axis=2).min(axis=1) <= tau)

    # For the swap test: which detection (in crop-local indexing) was credited
    # with each annotated bird in this crop.
    g_idx = np.where(g)[0]
    local_match = m["out"].det_match[d] if len(m["out"].det_match) else np.full(len(x), -1)
    credited = {int(j): int(n) for n, j in enumerate(local_match) if j >= 0}

    def is_swap(i) -> bool:
        if not len(g_idx):
            return False
        dist = np.hypot(gt.gt_xy[g_idx, 0] - x[i], gt.gt_xy[g_idx, 1] - y[i])
        k = int(np.argmin(dist)); j = int(g_idx[k])
        if dist[k] > 0.5 * gt.gt_size[j] or j not in credited:
            return False
        c = credited[j]
        return bool(np.hypot(gt.gt_xy[j, 0] - x[c], gt.gt_xy[j, 1] - y[c]) > dist[k])

    for n, i in enumerate(fp_i):
        if overlaps_matched(i):
            out["duplicate"] += 1
        elif is_swap(i):
            out["swap"] += 1
        elif votes[n] >= len(models):
            out["corroborated"] += 1
        else:
            out["uncorroborated"] += 1
    return out


# ── Selection rules ──────────────────────────────────────────────────────────

def select(gt, models: dict, metas: dict[int, metadata.CropMeta]) -> list[dict]:
    """One case per failure type, by rules applied to the whole test split."""
    ids = [int(i) for i in gt.image_ids]
    counts = gt.counts
    med = {i: _median_size(gt, i) for i in ids}

    box_keys = [k for k, m in models.items() if m["spec"].kind == "box"]
    point_keys = [k for k, m in models.items() if m["spec"].kind == "point"]
    primary = box_keys[0] if box_keys else next(iter(models))

    def pred(key, i):
        return models[key]["out"].pred_count[i]

    cases: list[dict] = []

    # 1 — very small birds
    pool = [i for i in ids if counts[i] >= 8 and (med[i] or 99) < 13]
    if pool:
        i = min(pool, key=lambda x: _recall(models, primary, gt, x) or 1.0)
        cases.append(dict(
            slug="missed_small_birds", failure_type="Missed very small birds",
            heading="Very small birds — the dominant failure mode",
            criteria=f"crop with >= 8 birds and median apparent size < 13 px; "
                     f"lowest {models[primary]['spec'].label} recall",
            n_eligible=len(pool), image_id=i,
            keys=[primary] + point_keys[:1]))

    # 2 — dense groups
    pool = [i for i in ids if counts[i] >= 150]
    if pool:
        i = max(pool, key=lambda x: counts[x])
        cases.append(dict(
            slug="dense_group", failure_type="Dense groups",
            heading="Dense flock — overlapping birds in one crop",
            criteria=f"crop with >= 150 birds; the densest of them",
            n_eligible=len(pool), image_id=i,
            keys=box_keys[:2] or [primary]))

    # 3 — complex background, using the habitat labels from Task 3.1
    pool = [i for i in ids
            if metas.get(i) and metas[i].habitat in COMPLEX_HABITATS
            and counts[i] >= 5]
    if pool:
        i = min(pool, key=lambda x: _recall(models, primary, gt, x) or 1.0)
        cases.append(dict(
            slug="complex_background", failure_type="Complex background",
            heading="Birds against vegetation and structure",
            criteria=f"crop in a structurally complex habitat "
                     f"({', '.join(COMPLEX_HABITATS)}) with >= 5 birds; "
                     f"lowest {models[primary]['spec'].label} recall",
            n_eligible=len(pool), image_id=i,
            keys=[primary] + point_keys[:1]))

    # 4 — false positives with nothing to find
    pool = [i for i in ids if counts[i] == 0]
    if pool:
        i = max(pool, key=lambda x: pred(primary, x))
        cases.append(dict(
            slug="false_positives_background", failure_type="False positives",
            heading="Bird-free crop that still draws detections",
            criteria=f"crop with zero birds; most {models[primary]['spec'].label} "
                     f"detections",
            n_eligible=len(pool), image_id=i, keys=box_keys[:2] or [primary]))

    # 5 — systematic undercounting
    pool = [i for i in ids if counts[i] >= 20]
    if pool:
        i = min(pool, key=lambda x: pred(primary, x) - counts[x])
        cases.append(dict(
            slug="undercount", failure_type="Systematic undercounting",
            heading="Largest undercount on a populated crop",
            criteria=f"crop with >= 20 birds; most negative "
                     f"{models[primary]['spec'].label} count error",
            n_eligible=len(pool), image_id=i, keys=box_keys[:2] or [primary]))

    # 6 — systematic overcounting
    if pool:
        i = max(pool, key=lambda x: pred(primary, x) - counts[x])
        cases.append(dict(
            slug="overcount", failure_type="Systematic overcounting",
            heading="Largest overcount on a populated crop",
            criteria=f"crop with >= 20 birds; most positive "
                     f"{models[primary]['spec'].label} count error",
            n_eligible=len(pool), image_id=i, keys=box_keys[:2] or [primary],
            duplicates=True))

    # 7 — high-confidence mistakes
    best_i, best_conf, n_elig = None, -1.0, 0
    for i in ids:
        m = models[primary]
        d = m["out"].det_image_id == i
        if not d.any():
            continue
        keep = m["cache"]["score"] >= m["t_star"]
        conf = m["cache"]["score"][keep][d]
        tp = m["out"].det_tp[d]
        if (~tp).any():
            n_elig += 1
            c = float(conf[~tp].max())
            if c > best_conf:
                best_conf, best_i = c, i
    if best_i is not None:
        cases.append(dict(
            slug="high_confidence_mistake", failure_type="High-confidence mistake",
            heading="The most confident false positive in the test set",
            criteria=f"highest-scoring false positive produced by "
                     f"{models[primary]['spec'].label} anywhere in the split "
                     f"(confidence {best_conf:.3f})",
            n_eligible=n_elig, image_id=best_i, keys=[primary] + point_keys[:1]))

    # 9 — every annotated bird found, and the count still far too high.
    # This is the signature of an incomplete annotation rather than of a model
    # error, and the corroboration test in the record distinguishes the two.
    pool = [i for i in ids
            if counts[i] >= 5
            and (_recall(models, primary, gt, i) or 0) >= 0.999
            and pred(primary, i) - counts[i] > 0]
    if pool:
        i = max(pool, key=lambda x: pred(primary, x) - counts[x])
        cases.append(dict(
            slug="every_bird_found_count_wrong",
            failure_type="Probable annotation error",
            heading="Every annotated bird found, and the count still far too high",
            criteria=f"crop with >= 5 birds where {models[primary]['spec'].label} "
                     f"found every annotated bird; largest count overshoot",
            n_eligible=len(pool), image_id=i,
            keys=[primary] + point_keys[:1], corroborate=True))

    # 8 — model families disagree
    if box_keys and point_keys:
        bk, pk = box_keys[0], point_keys[0]
        pool = [i for i in ids if counts[i] >= 10]
        if pool:
            def gap(x):
                a, b = _recall(models, bk, gt, x), _recall(models, pk, gt, x)
                return (a - b) if None not in (a, b) else 0.0
            i = max(pool, key=lambda x: abs(gap(x)))
            cases.append(dict(
                slug="family_disagreement",
                failure_type="Model families behave differently",
                heading="Box and point detectors disagree on the same crop",
                criteria=f"crop with >= 10 birds; largest recall gap between "
                         f"{models[bk]['spec'].label} (box) and "
                         f"{models[pk]['spec'].label} (point)",
                n_eligible=len(pool), image_id=i, keys=[bk, pk]))

    return cases


# ── Build ────────────────────────────────────────────────────────────────────

def build(manifest_path: str | None = None, render: bool = True) -> list[FailureCase]:
    from analysis import manifest as manifest_mod

    manifest_path = manifest_path or os.path.join(ROOT, "models_v1.docx")
    runs = {registry.resolve(e.key).key: e.run_id
            for e in manifest_mod.load(manifest_path)}
    gt = strata.load_ground_truth()
    metas = metadata.load(predict.TEST_JSON)

    models = {}
    for key, run in runs.items():
        spec = registry.resolve(key)
        pm = rescore.ensure(spec, run)
        cache = predict.load(predict.cache_path(spec, run))
        out = strata.outcomes_at_t_star(gt, cache, pm["counting"]["t_star"])
        models[key] = {"spec": spec, "out": out, "cache": cache,
                       "t_star": pm["counting"]["t_star"]}

    library: list[FailureCase] = []
    for c in select(gt, models, metas):
        iid = c["image_id"]
        meta = metas.get(iid)
        outcomes = [_outcome(k, models[k], gt, iid) for k in c["keys"]]
        n_gt = gt.counts[iid]
        med = _median_size(gt, iid)

        corr = (corroboration(models, iid, c["keys"][0])
                if c.get("corroborate") else None)
        dup = (false_alarm_breakdown(models, gt, iid, c["keys"][0])
               if c.get("duplicates") else None)

        fig = None
        if render:
            fig = os.path.join(FIG_DIR, f"{c['slug']}.png")
            failure_render.render_case(gt, models, c["keys"], iid, c["heading"], fig)
            fig = os.path.relpath(fig, OUT_DIR)

        library.append(FailureCase(
            slug=c["slug"], failure_type=c["failure_type"], heading=c["heading"],
            criteria=c["criteria"], n_eligible=c["n_eligible"], image_id=iid,
            crop_file=meta.file_name if meta else "",
            figure=fig,
            dataset=meta.dataset if meta else "",
            habitat=meta.habitat if meta else None,
            altitude_m=meta.altitude_m if meta else None,
            weather=meta.weather if meta else None,
            site=meta.site if meta else None,
            n_gt=n_gt, median_bird_px=med, models=outcomes,
            corroboration=corr, duplicates=dup,
        ))
    return library


def write_json(library: list[FailureCase], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(c) for c in library], f, indent=2)
    return path


def main() -> int:
    library = build()
    if not library:
        print("No failure cases selected.", file=sys.stderr)
        return 1
    j = write_json(library, os.path.join(OUT_DIR, "failure_library.json"))
    for c in library:
        print(f"  {c.slug:28s} image {c.image_id:6d}  {c.n_gt:4d} birds  "
              f"{c.habitat or '—'}")
    print(f"Wrote {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
