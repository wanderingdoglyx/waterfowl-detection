"""
Habitat, altitude and survey-metadata analysis — how the conditions a flight was
flown under change detection and counting accuracy.

Tasks 2.3 and 2.4 asked what the *image* contains (how big the birds are, how
many).  This module asks what the *survey* was: which habitat was flown, from what
height, in what weather.  The metrics are the project's own; only the partition
changes, so `subgroups` does the scoring and this module decides what to
partition on and how to report it honestly.

Three things shape every table here:

**Coverage is measured, never assumed.**  Altitude covers all ten datasets;
habitat covers three; weather covers two; decoy-vs-real covers none.
`coverage` records what each field actually covers on the split being scored.

**Subgroups too small to carry a claim are scored but not ranked.**  A habitat
with two source photographs is a fact about those photographs.  It stays in the
table, flagged `sufficient = 0`, so it is visibly measured rather than dropped.

**Between-dataset confounding is measured alongside every pooled finding.**
Habitat, weather and site vary far more *between* sub-datasets than within them,
so a habitat difference may really be a difference between Bird_G and Bird_J —
different sites, dates, cameras and annotators.  Sub-dataset is scored as a
factor of its own, and altitude, the one field that varies cleanly *inside* six
datasets, additionally gets a within-dataset analysis that no between-dataset
difference can explain.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

from data_prep import config
from data_prep.point_metrics import gt_points_from_coco
from analysis import metadata, predict, registry, rescore, strata, subgroups
from analysis.subgroups import MIN_BIRDS, MIN_IMAGES, SubgroupScore

# The shared survey ladder. Every other recorded height belongs to Bird_A's or
# Bird_B's own ladder and carries too few birds to read as its own band.
LADDER = (15, 30, 60, 90, 120)

# The proposal names Arkansas environments that MDC's data does not contain
# verbatim. Mapping is stated rather than assumed so nobody reads "StandingCorn"
# as evidence about rice.
ARKANSAS_ANALOGUES: dict[str, tuple[str, ...]] = {
    "Forested / bottomland hardwood": ("Trees", "Wooded"),
    "Row-crop agriculture (rice / soybean analogue)": (
        "StandingCorn", "HarvestedCrop", "WaterCorn"),
    "Moist-soil / managed wetland": ("MoistSoil",),
    "Open water": ("OpenWater", "Ice"),
    "Emergent / submerged vegetation": (
        "WaterVegetation", "WaterSubVeg", "Lotus", "ShrubScrub"),
    "Dry land / levee": ("Land", "Rocks"),
}


@dataclass
class ConditionsContext:
    metas: dict[int, metadata.CropMeta]
    gt: strata.GroundTruth
    runs: list[subgroups.MatchedRun]
    n_crops: int
    n_birds: int
    n_source_images: int


def build(specs: list[tuple[registry.ModelSpec, str]],
          force: bool = False) -> ConditionsContext:
    """Match every model once, split-wide, and load the survey metadata."""
    gt = strata.load_ground_truth()
    gt_by_image = gt_points_from_coco(predict.TEST_JSON)
    metas = metadata.load(predict.TEST_JSON)

    runs = []
    for spec, run_id in specs:
        predict.ensure(spec, run_id, force=force)
        pm = rescore.ensure(spec, run_id, force=force)
        cache = predict.load(predict.cache_path(spec, run_id))
        runs.append(subgroups.match_run(
            gt.counts, gt_by_image, cache, pm["counting"]["t_star"],
            spec.key, spec.label, spec.kind))

    return ConditionsContext(
        metas=metas, gt=gt, runs=runs,
        n_crops=len(gt.image_ids), n_birds=len(gt.gt_size),
        n_source_images=metadata.source_image_count(metas),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_groups(ctx: ConditionsContext, groups: dict[object, list[int]],
                  order: list | None = None
                  ) -> dict[str, list[SubgroupScore]]:
    """Score every model on every group, in a stable group order."""
    keys = order if order is not None else sorted(groups, key=str)
    per_model: dict[str, list[SubgroupScore]] = {}
    for run in ctx.runs:
        per_model[run.key] = [
            subgroups.score(run, groups[g], g,
                            metadata.source_image_count(ctx.metas, groups[g]))
            for g in keys if g in groups
        ]
    return per_model


def _mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def _group_mean(per_model: dict[str, list[SubgroupScore]], group, attr: str
                ) -> float | None:
    """One subgroup's metric averaged over every model that has a value for it."""
    return _mean([getattr(s, attr) for ss in per_model.values()
                  for s in ss if s.group == group and getattr(s, attr) is not None])


def coverage(ctx: ConditionsContext,
             fields: tuple[str, ...] = ("altitude_nominal", "habitat", "weather",
                                        "site", "date")) -> list[dict]:
    """How much of the split each field covers, in crops and in birds."""
    rows = []
    for field in fields:
        a = metadata.availability(ctx.metas, ctx.gt.counts, field)
        rows.append({**a, "datasets": ";".join(a["datasets"])})
    return rows


# ── Habitat ───────────────────────────────────────────────────────

def habitat_table(ctx: ConditionsContext) -> dict[str, list[SubgroupScore]]:
    """Every model scored on every habitat, largest habitat first.

    Habitat is recorded three different ways and only for three datasets: a curated
    `background_info` column in Bird_H, and a filename token in Bird_G and Bird_J.
    `metadata` merges Bird_G's `WaterSubVeg`/`WaterSubVege` and excludes Bird_J's
    `Dense`, which describes bird density rather than habitat.
    """
    groups = metadata.group_by(ctx.metas, "habitat")
    keys = sorted(groups, key=lambda g: -sum(ctx.gt.counts[i] for i in groups[g]))
    return _score_groups(ctx, groups, keys)


def habitat_rankings(ctx: ConditionsContext,
                     per_model: dict[str, list[SubgroupScore]]) -> list[dict]:
    """Model order by F1 inside each usable habitat, against the whole-split order.

    Kendall tau near +1 means habitat changes the level of performance without
    reordering the models; lower values mean the overall ranking hides reversals.
    """
    ranks = subgroups.rank_changes(per_model, "f1")
    usable = [g for g in ranks if len(ranks[g]) >= 2]
    if len(usable) < 2:
        return []
    overall = [r.key for r in sorted(
        ctx.runs, key=lambda r: -(subgroups.score(
            r, list(ctx.gt.counts), "ALL", ctx.n_source_images).f1 or 0))]
    label = {r.key: r.label for r in ctx.runs}
    rows = []
    for g in usable:
        order = [k for k, _ in ranks[g]]
        tau = subgroups.kendall_tau(overall, order)
        for n, (key, f1) in enumerate(ranks[g], 1):
            rows.append({"habitat": g, "rank": n, "model": key, "label": label[key],
                         "f1": f1, "kendall_tau_vs_whole_split": tau})
    return rows


def habitat_datasets(ctx: ConditionsContext) -> list[dict]:
    """Which sub-datasets each habitat is drawn from.

    A habitat found in one sub-dataset cannot be separated from that dataset's
    site, date, camera and annotators; only multi-dataset habitats are even
    partly clean comparisons.
    """
    groups = metadata.group_by(ctx.metas, "habitat")
    rows = []
    for g in sorted(groups, key=lambda g: -sum(ctx.gt.counts[i] for i in groups[g])):
        for ds, n in sorted(metadata.counts_by_dataset(ctx.metas, groups[g]).items()):
            rows.append({"habitat": g, "dataset": ds, "n_crops": n})
    return rows


def arkansas_analogues(ctx: ConditionsContext) -> list[dict]:
    """Score the MDC habitats grouped into the environments the proposal names.

    None of bottomland hardwood, rice or soybean appear verbatim in the MDC data,
    so each row is an analogue (see `ARKANSAS_ANALOGUES`), not a match.
    """
    groups = metadata.group_by(ctx.metas, "habitat")
    label = {r.key: r.label for r in ctx.runs}
    rows = []
    for env, members in ARKANSAS_ANALOGUES.items():
        present = [g for g in sorted(groups) if g in members]
        if not present:
            continue
        ids = [i for g in present for i in groups[g]]
        imgs = metadata.source_image_count(ctx.metas, ids)
        birds = sum(ctx.gt.counts[i] for i in ids)
        for run in ctx.runs:
            s = subgroups.score(run, ids, env, imgs)
            rows.append({"environment": env, "habitats": ";".join(present),
                         "n_crops": len(ids), "n_birds": birds, "model": run.key,
                         "label": label[run.key], "f1": s.f1})
    return rows


# ── Altitude ──────────────────────────────────────────────────────

def altitude_table(ctx: ConditionsContext) -> dict[str, list[SubgroupScore]]:
    """Every model scored on every nominal altitude.

    Heights are snapped to the survey ladder within 2 m by `metadata`, which merges
    recording jitter without inventing bands; Bird_A's distinct heights stay as
    they are.
    """
    groups = metadata.group_by(ctx.metas, "altitude_nominal")
    return _score_groups(ctx, groups, sorted(groups))


def _train_altitude_share() -> dict[int, float] | None:
    """Share of training birds flown at each nominal altitude.

    The obvious reading of the altitude tables — "higher is easier" — is almost
    certainly not about height at all, so the competing explanation is measured
    rather than left as a caveat.
    """
    path = os.path.join(config.CROPS_JSON_DIR, "train", "coco.json")
    if not os.path.exists(path):
        return None
    metas = metadata.load(path)
    gt = strata.load_ground_truth(path)
    groups = metadata.group_by(metas, "altitude_nominal")
    total = sum(gt.counts.values())
    if not total:
        return None
    return {g: 100.0 * sum(gt.counts[i] for i in ids) / total
            for g, ids in groups.items()}


def _median_size(ctx: ConditionsContext, image_ids: list[int]) -> float | None:
    """Median apparent bird size (sqrt box area, px) over a set of crops."""
    ids = np.asarray(sorted(image_ids), dtype=np.int64)
    sel = np.isin(ctx.gt.gt_image_id, ids)
    return float(np.median(ctx.gt.gt_size[sel])) if sel.any() else None


def altitude_chain(ctx: ConditionsContext,
                   per_model: dict[str, list[SubgroupScore]]) -> list[dict]:
    """altitude -> apparent size -> recall -> counting error, on the ladder heights.

    Altitude does not degrade a detector by itself; it shrinks the birds, and small
    birds are what the detector misses. Each row follows that chain one link at a
    time. The training split's share of birds at each height sits beside it: recall
    peaks mid-range rather than falling with height, which size alone cannot explain,
    and the competing explanation is how much training data each height had.

    Association only — the pooled heights differ in which datasets contribute them
    (90 m is the only height Bird_D, Bird_E and Bird_F fly). `altitude_within_dataset`
    is the comparison that survives that objection.
    """
    groups = metadata.group_by(ctx.metas, "altitude_nominal")
    share = _train_altitude_share() or {}
    rows = []
    for g in sorted(groups):
        if g not in LADDER:
            continue
        rows.append({
            "altitude_m": g,
            "median_size_px": _median_size(ctx, groups[g]),
            "n_birds": sum(ctx.gt.counts[i] for i in groups[g]),
            "mean_recall": _group_mean(per_model, g, "recall"),
            "mean_mae": _group_mean(per_model, g, "mae"),
            "mean_count_bias": _group_mean(per_model, g, "count_bias"),
            "train_bird_share_pct": share.get(g),
        })
    return rows


def _curve_shape(vals: list[float]) -> str:
    """Endpoints alone are misleading on a curve that turns: name the shape."""
    if all(b >= a for a, b in zip(vals, vals[1:])):
        return "rising"
    if all(b <= a for a, b in zip(vals, vals[1:])):
        return "falling"
    if vals.index(min(vals)) not in (0, len(vals) - 1):
        return "U-shaped"
    if vals.index(max(vals)) not in (0, len(vals) - 1):
        return "peaked"
    return "non-monotone"


def altitude_within_dataset(ctx: ConditionsContext) -> list[dict]:
    """Altitude effect measured inside each dataset that varies it.

    This is the analysis that between-dataset confounding cannot explain: the
    site, camera, annotator and season are held fixed, and only the height changes.
    Recall is averaged over all models, at each model's own global operating point.
    """
    rows = []
    for ds in metadata.datasets_varying(ctx.metas, "altitude_nominal"):
        groups = metadata.group_by(ctx.metas, "altitude_nominal", datasets={ds})
        curve = []
        for g in sorted(groups):
            ids = groups[g]
            imgs = metadata.source_image_count(ctx.metas, ids)
            if sum(ctx.gt.counts[i] for i in ids) < MIN_BIRDS or imgs < MIN_IMAGES:
                continue
            rec = _mean([s.recall for s in (subgroups.score(r, ids, g, imgs)
                                            for r in ctx.runs) if s.recall is not None])
            if rec is not None:
                curve.append((g, rec, _median_size(ctx, ids)))
        if len(curve) < 2:
            continue
        shape = _curve_shape([v for _, v, _ in curve])
        for g, rec, size in curve:
            rows.append({"dataset": ds, "altitude_m": g, "mean_recall": rec,
                         "median_size_px": size, "shape": shape})
    return rows


# ── Other survey metadata ────────────────────────────────────────────────

def other_tables(ctx: ConditionsContext) -> dict[str, dict[str, list[SubgroupScore]]]:
    """Weather, site, date and sub-dataset — the variables the data can support.

    Site and date are Bird_H only: its named sites are the only ones paired with
    curated habitat, a survey date and a known altitude for every image. Season is
    confounded with site there, since not every site was flown on every date.
    Sub-dataset is not a survey condition but the reference every other factor has
    to be read against. Variables the data cannot support at all are listed in
    `metadata.UNAVAILABLE_FIELDS`.
    """
    tables: dict[str, dict] = {}
    groups = metadata.group_by(ctx.metas, "weather")
    if groups:
        tables["weather"] = _score_groups(ctx, groups, sorted(groups))
    for field in ("site", "date"):
        groups = metadata.group_by(ctx.metas, field, datasets={"Bird_H"})
        if len(groups) > 1:
            tables[field] = _score_groups(ctx, groups, sorted(groups))
    groups = metadata.group_by(ctx.metas, "dataset")
    tables["dataset"] = _score_groups(ctx, groups, sorted(groups))
    return tables


def weather_by_dataset(ctx: ConditionsContext) -> list[dict]:
    """Sun vs cloud holding the dataset fixed (Bird_G and Bird_J both have both)."""
    rows = []
    for ds in ("Bird_G", "Bird_J"):
        groups = metadata.group_by(ctx.metas, "weather", datasets={ds})
        for cond in sorted(groups):
            ids = groups[cond]
            imgs = metadata.source_image_count(ctx.metas, ids)
            f1 = _mean([s.f1 for s in (subgroups.score(r, ids, cond, imgs)
                                       for r in ctx.runs) if s.f1 is not None])
            rows.append({"dataset": ds, "weather": cond, "n_crops": len(ids),
                         "n_birds": sum(ctx.gt.counts[i] for i in ids), "mean_f1": f1})
    return rows


# ── Where the error actually comes from ──────────────────────

FACTOR_NOTES = {
    "dataset": "Which sub-dataset the crop came from — site, camera, annotator and season together",
    "habitat": "Habitat class (Bird_G/H/J only)",
    "altitude": "Nominal flight altitude",
    "weather": "Sun vs cloud (Bird_G/J only)",
    "site": "Survey site within Bird_H",
    "date": "Survey date within Bird_H",
    "density": "Birds per crop (density band)",
}


def _spread(per_model) -> float | None:
    """Mean F1 range across a factor's usable groups, averaged over models."""
    vals = []
    for scores in per_model.values():
        v = [s.f1 for s in scores if s.sufficient and s.f1 is not None]
        if len(v) >= 2:
            vals.append(max(v) - min(v))
    return float(np.mean(vals)) if vals else None


def spread_detail(per_model) -> dict | None:
    """The spread statistic with the numbers needed to judge it.

    The headline figure is a max-minus-min range, which widens as a factor gains
    groups — a factor split eleven ways has more chances to produce an extreme
    than one split two ways. Reporting the group count and the standard
    deviation alongside lets a reader see whether an ordering is real or an
    artefact of how finely each factor happens to be divided.
    """
    ranges, sds, n_groups = [], [], 0
    for scores in per_model.values():
        v = [s.f1 for s in scores if s.sufficient and s.f1 is not None]
        n_groups = max(n_groups, len(v))
        if len(v) >= 2:
            arr = np.asarray(v, dtype=float)
            ranges.append(float(arr.max() - arr.min()))
            sds.append(float(arr.std(ddof=1)))
    if not ranges:
        return None
    return {"range": float(np.mean(ranges)), "sd": float(np.mean(sds)),
            "n_groups": n_groups}


def spread_table(ctx: ConditionsContext, tables: dict[str, dict]) -> list[dict]:
    """Per-factor spread detail, in the same order as `summary_factors`."""
    rows = []
    for name, val, note in summary_factors(ctx, tables):
        pm = tables.get(name) if name != "density" else _score_groups(
            ctx, density_groups(ctx), list(density_groups(ctx)))
        d = spread_detail(pm) if pm else None
        rows.append({"factor": name, "note": note, "range": val,
                     "sd": d["sd"] if d else None,
                     "n_groups": d["n_groups"] if d else None})
    return rows


def density_groups(ctx: ConditionsContext) -> dict[object, list[int]]:
    """Task 2.4's density bands, so density sits on the same scale as habitat."""
    counts = np.array([ctx.gt.counts[int(i)] for i in ctx.gt.image_ids])
    edges, labels = strata.density_bins(counts)
    bin_of = np.digitize(counts, edges[1:-1], right=False)
    groups: dict[object, list[int]] = {}
    for b, label in enumerate(labels):
        ids = [int(i) for i, k in zip(ctx.gt.image_ids, bin_of) if k == b]
        if ids:
            groups[label] = ids
    return groups


def summary_factors(ctx: ConditionsContext,
                    tables: dict[str, dict]) -> list[tuple[str, float, str]]:
    """Every factor's F1 spread, largest first.

    Shared by `spread_table` and the error-sources figure so the two cannot
    disagree about what was measured or which factors were included.
    """
    factors: list[tuple[str, float, str]] = []
    for key in FACTOR_NOTES:
        if key == "density":
            continue
        pm = tables.get(key)
        if not pm:
            continue
        v = _spread(pm)
        if v is not None:
            factors.append((key, v, FACTOR_NOTES[key]))

    groups = density_groups(ctx)
    if groups:
        v = _spread(_score_groups(ctx, groups, list(groups)))
        if v is not None:
            factors.append(("density", v, FACTOR_NOTES["density"]))

    factors.sort(key=lambda f: -f[1])
    return factors


# ── CSV output ───────────────────────────────────────────────────────────────

def write_csvs(ctx: ConditionsContext, tables: dict[str, dict], out_dir: str) -> list[str]:
    """One tidy CSV per partition — long format, one row per model x subgroup."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    label = {r.key: r.label for r in ctx.runs}
    for name, per_model in tables.items():
        path = os.path.join(out_dir, f"by_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "label", "group", "n_crops", "n_source_images",
                        "n_birds", "sufficient", "ap", "precision", "recall", "f1",
                        "mae", "rmse", "count_bias_pct", "pred_total", "n_det"])
            for key, scores in per_model.items():
                for s in scores:
                    w.writerow([key, label.get(key, key), s.group, s.n_crops,
                                s.n_source_images, s.n_birds, int(s.sufficient),
                                _csv(s.ap), _csv(s.precision), _csv(s.recall),
                                _csv(s.f1), _csv(s.mae), _csv(s.rmse),
                                _csv(s.count_bias), s.pred_total, s.n_det])
        written.append(path)

    for name, rows in (("coverage.csv", coverage(ctx)),
                       ("habitat_rankings.csv",
                        habitat_rankings(ctx, tables["habitat"]) if "habitat" in tables else []),
                       ("habitat_datasets.csv", habitat_datasets(ctx)),
                       ("arkansas_analogues.csv", arkansas_analogues(ctx)),
                       ("altitude_chain.csv",
                        altitude_chain(ctx, tables["altitude"]) if "altitude" in tables else []),
                       ("altitude_within_dataset.csv", altitude_within_dataset(ctx)),
                       ("weather_by_dataset.csv", weather_by_dataset(ctx)),
                       ("error_sources.csv", spread_table(ctx, tables))):
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows({k: _csv(v) for k, v in r.items()} for r in rows)
        written.append(path)
    return written


def _csv(v):
    if v is None:
        return ""
    return round(v, 4) if isinstance(v, float) else v
