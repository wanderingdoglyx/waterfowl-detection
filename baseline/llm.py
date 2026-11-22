#!/usr/bin/env python
"""
LLM zero-shot baseline — can a vision-language model do the detection task at all?

A multimodal LLM has no detection head and no score calibration, and it is far too slow
and too costly to run over 28,588 crops, so it cannot enter the full-split table.  This
module scores it on a small, fixed sample instead, using the SAME protocol functions as
every other model, and scores the eight fine-tuned models on the same crops next to it
so the small-sample number has a reference.

Two steps, because the detections come from outside this process (an LLM session or an
API call) and the model must not see the ground truth before it answers:

    python -m baseline.llm select [--n 10]
        Creates output/baselines/llm_claude/<timestamp>/ with
            test_subset_coco.json   the sampled crops' ground truth (do NOT show the LLM)
            llm_inputs/             what the LLM is shown: each crop upscaled 2x, plain
                                    and with a labelled grid in crop-pixel coordinates
        and prints only the crop names, never their bird counts.

    python -m baseline.llm score <run_dir>
        Reads <run_dir>/llm_detections.json
            {"model": ..., "crops": {"<file_name>": [{"bbox": [x, y, w, h],
                                                      "score": s}, ...]}}
        (bbox in 512x512 crop pixels) and writes the usual baseline layout:
            eval_results.txt, paper_point_metrics.json, metrics.json, examples/
        plus comparison.txt: every cached fine-tuned model on the same crops.

The sample is stratified by bird count so it spans sparse to dense crops, with one crop
per sub-dataset where possible so it also spans habitats.  The densest stratum is capped
at 100 birds: hand-placing several hundred boxes is not a test an LLM session can
complete, and the cap is recorded in metrics.json.  Only crops with birds are drawn,
as in baseline/main.py --limit, so false alarms on empty background go unmeasured.

Every metric on 10 crops is statistically thin (see the bootstrap CIs); like a --limit
run it is excluded from ./baseline/main.py --summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data_prep.config as config

MODEL_KEY = "llm_claude"
MODEL_ID = "claude-opus-5-5"
LABEL = "Claude Opus 5.5 (LLM, zero-shot)"

TEST_JSON = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")

# (name, min birds, max birds, crops drawn).  Filled rarest-first, so the few datasets
# that have dense crops are not used up by the sparse strata.
STRATA = [("41-100", 41, 100, 1), ("16-40", 16, 40, 2), ("6-15", 6, 15, 3),
          ("1-5", 1, 5, 4)]

UPSCALE = 2          # LLM input magnification; the median bird is ~20 px at 1x
GRID_STEP = 32       # grid spacing in crop pixels; labelled every 2 lines


# ── select ────────────────────────────────────────────────────────────────────
def select_crops(coco: dict, seed: int) -> list[tuple[dict, str]]:
    """Draw one crop per stratum slot, preferring a sub-dataset not used yet."""
    counts: dict[int, int] = {}
    for a in coco["annotations"]:
        counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1

    rng = random.Random(seed)
    used_ids: set[int] = set()
    used_sets: set[str] = set()
    picks = []
    for name, lo, hi, n in STRATA:
        pool = [im for im in coco["images"] if lo <= counts.get(im["id"], 0) <= hi]
        by_set: dict[str, list[dict]] = {}
        for im in pool:
            by_set.setdefault(im["file_name"].split("/")[0], []).append(im)
        for _ in range(n):
            fresh = sorted(s for s in by_set if s not in used_sets)
            choices = fresh or sorted(by_set)
            ds = rng.choice(choices)
            im = rng.choice([c for c in by_set[ds] if c["id"] not in used_ids])
            picks.append((im, name))
            used_ids.add(im["id"])
            used_sets.add(ds)
    # Shuffle so the viewing order does not reveal the stratum.
    rng.shuffle(picks)
    return picks


def render_inputs(img_path: str, out_dir: str) -> tuple[str, str]:
    """Write the two views the LLM is shown: plain 2x, and 2x with a coordinate grid.

    Grid labels are in ORIGINAL crop pixels, so the coordinates the LLM reads off the
    image are already the ones the protocol scores.
    """
    import cv2

    im = cv2.imread(img_path)
    big = cv2.resize(im, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    plain = os.path.join(out_dir, f"{stem}_2x.jpg")
    cv2.imwrite(plain, big, [cv2.IMWRITE_JPEG_QUALITY, 95])

    grid = big.copy()
    overlay = big.copy()
    size = big.shape[0]
    for i, v in enumerate(range(0, im.shape[0] + 1, GRID_STEP)):
        p = min(v * UPSCALE, size - 1)
        colour = (0, 255, 255) if i % 2 == 0 else (200, 200, 200)
        cv2.line(overlay, (p, 0), (p, size - 1), colour, 1)
        cv2.line(overlay, (0, p), (size - 1, p), colour, 1)
    grid = cv2.addWeighted(overlay, 0.45, grid, 0.55, 0)
    for v in range(0, im.shape[0], GRID_STEP * 2):
        p = v * UPSCALE
        for (x, y) in ((p + 3, 16), (3, p + 16)):
            cv2.putText(grid, str(v), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(grid, str(v), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
    gridded = os.path.join(out_dir, f"{stem}_grid.jpg")
    cv2.imwrite(gridded, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return plain, gridded


def cmd_select(n: int) -> None:
    if n != sum(s[3] for s in STRATA):
        sys.exit(f"--n must equal the strata total ({sum(s[3] for s in STRATA)}); "
                 "edit STRATA to change the sample size")
    with open(TEST_JSON) as f:
        coco = json.load(f)
    picks = select_crops(coco, config.RANDOM_SEED)

    run_dir = os.path.join(config.BASELINE_DIR, MODEL_KEY,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    in_dir = os.path.join(run_dir, "llm_inputs")
    os.makedirs(in_dir, exist_ok=True)

    keep = {im["id"] for im, _ in picks}
    subset = dict(coco)
    subset["images"] = [im for im, _ in picks]
    subset["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep]
    with open(os.path.join(run_dir, "test_subset_coco.json"), "w") as f:
        json.dump(subset, f)
    with open(os.path.join(run_dir, "selection.json"), "w") as f:
        json.dump({"seed": config.RANDOM_SEED, "strata": STRATA,
                   "crops": [{"image_id": im["id"], "file_name": im["file_name"],
                              "stratum": s} for im, s in picks]}, f, indent=2)

    print(f"Run directory: {run_dir}")
    print("Crops (ground truth withheld):")
    for im, _ in picks:
        plain, gridded = render_inputs(
            os.path.join(config.CROPS_IMG_DIR, im["file_name"]), in_dir)
        print(f"  {im['id']:>6}  {im['file_name']}\n          {plain}\n          {gridded}")


# ── score ─────────────────────────────────────────────────────────────────────
def _subset_metrics(subset_json: str, coco_results: list, box: bool) -> dict:
    """mAP30 (box models only) and the paper point metrics on the subset."""
    from data_prep.point_metrics import (dets_from_coco_results, gt_points_from_coco,
                                         paper_point_metrics)
    from yolov5.evaluate import coco_map30

    ap30, ar30 = coco_map30(subset_json, coco_results) if box else (None, None)
    pm = paper_point_metrics(gt_points_from_coco(subset_json),
                             dets_from_coco_results(coco_results),
                             tau=config.PAPER_TAU, bootstrap=config.PAPER_BOOTSTRAP)
    return {"map30": ap30, "mar30": ar30, "pm": pm}


def _counts_at(coco_results: list, t: float) -> dict[int, int]:
    """Detections per crop at score >= t — the count paper_point_metrics reads at t*."""
    out: dict[int, int] = {}
    for r in coco_results:
        if r["score"] >= t:
            out[r["image_id"]] = out.get(r["image_id"], 0) + 1
    return out


def _cached_finetuned(ids: set[int]) -> list[tuple[str, str, list]]:
    """(label, kind, COCO result list) for every cached fine-tuned run, on `ids` only.

    Point detectors carry no box size in the cache, so they get a nominal 1 px box:
    only their point metrics are read, never their mAP30.
    """
    from analysis import predict
    from analysis.registry import MODELS

    rows = []
    for spec in MODELS:
        paths = sorted(glob.glob(os.path.join(predict.PRED_DIR, f"{spec.key}__*.npz")))
        if not paths:
            continue
        c = predict.load(paths[-1])
        res = []
        for i, x, y, s, w, h in zip(c["image_id"], c["x"], c["y"], c["score"],
                                    c["w"], c["h"]):
            if int(i) not in ids:
                continue
            w, h = (float(w), float(h)) if spec.kind == "box" else (1.0, 1.0)
            res.append({"image_id": int(i), "category_id": 1,
                        "bbox": [float(x) - w / 2, float(y) - h / 2, w, h],
                        "score": float(s)})
        run_id = os.path.basename(paths[-1]).split("__")[1].removesuffix(".npz")
        rows.append((f"{spec.label} (fine-tuned {run_id[:10]})", spec.kind, res))
    return rows


def cmd_score(run_dir: str) -> None:
    from baseline.main import _save_examples
    from data_prep.point_metrics import format_map30_block, format_report, write_eval_txt

    subset_json = os.path.join(run_dir, "test_subset_coco.json")
    with open(subset_json) as f:
        subset = json.load(f)
    with open(os.path.join(run_dir, "llm_detections.json")) as f:
        llm = json.load(f)

    id_of = {im["file_name"]: im["id"] for im in subset["images"]}
    missing = set(id_of) - set(llm["crops"])
    if missing:
        sys.exit(f"llm_detections.json has no entry for: {sorted(missing)}")
    coco_results = [{"image_id": id_of[fn], "category_id": 1,
                     "bbox": [float(v) for v in d["bbox"]], "score": float(d["score"])}
                    for fn, dets in llm["crops"].items() for d in dets]

    m = _subset_metrics(subset_json, coco_results, box=True)
    pm, n = m["pm"], len(subset["images"])
    tag = f"{LABEL} [SAMPLE {n} crops]"
    text = write_eval_txt(os.path.join(run_dir, "eval_results.txt"),
                          [format_map30_block(m["map30"], m["mar30"]),
                           format_report(pm, tag)])
    with open(os.path.join(run_dir, "paper_point_metrics.json"), "w") as f:
        json.dump(pm, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({
            "model": MODEL_KEY, "label": LABEL, "family": "llm",
            "weights": llm.get("model", MODEL_ID), "zero_shot": True,
            "class_filter": None,
            "timestamp": os.path.basename(run_dir),
            "limit": n, "n_test": n, "n_detections": len(coco_results),
            "selection": "stratified by bird count (1-5 x4, 6-15 x3, 16-40 x2, "
                         "41-100 x1), one sub-dataset per crop where possible, "
                         f"seed {config.RANDOM_SEED}; crops with birds only",
            "protocol": llm.get("protocol"),
            "map30": m["map30"], "mar30": m["mar30"],
            "paper_point_metrics": pm,
        }, f, indent=2)
    print(text)

    _save_examples(coco_results, run_dir, n, subset_json)

    # Same crops, same functions, every cached fine-tuned model.
    ids = set(id_of.values())
    rows = [(LABEL, "box", coco_results, m)]
    for label, kind, res in _cached_finetuned(ids):
        rows.append((label, kind, res, _subset_metrics(subset_json, res, kind == "box")))
    gt_n: dict[int, int] = {}
    for a in subset["annotations"]:
        gt_n[a["image_id"]] = gt_n.get(a["image_id"], 0) + 1

    def pct(v):
        return "     —" if v is None else f"{v:6.2f}"

    lines = [f"Same {n} test crops, same protocol (tau = {config.PAPER_TAU} px, "
             f"B = {config.PAPER_BOOTSTRAP}); {sum(gt_n.values())} GT birds",
             "",
             f"{'Model':<42}{'mAP30':>7}{'AP':>8}{'P':>8}{'R':>8}{'F1':>8}"
             f"{'MAE':>8}{'RMSE':>8}{'Pred':>6}",
             "─" * 103]
    for label, kind, _, mm in rows:
        d, c = mm["pm"]["detection"], mm["pm"]["counting"]
        lines.append(f"{label:<42}{pct(mm['map30']):>7}{d['ap'] * 100:8.2f}"
                     f"{d['precision_at_t_star'] * 100:8.2f}"
                     f"{d['recall_at_t_star'] * 100:8.2f}{d['f1_at_t_star'] * 100:8.2f}"
                     f"{c['mae']:8.2f}{c['rmse']:8.2f}{c['total_pred_at_t_star']:6d}")
    lines += ["─" * 103, "",
              "Per-crop count at each model's t* (GT first):", ""]
    short = ["GT", "LLM"] + [r[0].split(" (")[0] for r in rows[1:]]
    name_w = max(len(im["file_name"]) for im in subset["images"]) + 2
    col_w = max(len(s) for s in short) + 2
    lines.append(f"{'crop':<{name_w}}" + "".join(f"{s:>{col_w}}" for s in short))
    for im in subset["images"]:
        cells = [gt_n.get(im["id"], 0)]
        for _, _, res, mm in rows:
            cells.append(_counts_at(res, mm["pm"]["counting"]["t_star"]).get(im["id"], 0))
        lines.append(f"{im['file_name']:<{name_w}}" + "".join(f"{v:>{col_w}}" for v in cells))
    lines += ["",
              "t* is chosen per model ON THESE CROPS (argmin MAE), for every row alike.",
              "mAP30 is omitted for the OWL point models (no boxes in the cache).",
              "The fine-tuned rows come from output/analysis/predictions/, the cache",
              "analysis/ scores the current split with.  Treat every number here as",
              "a sample of 10: compare rows, not against the full-split table."]
    comp = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write(comp)
    print("\n" + comp)
    print(f"Saved results to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM zero-shot detection on a sample")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="Draw the sample and render the LLM's inputs")
    s.add_argument("--n", type=int, default=10)
    sc = sub.add_parser("score", help="Score <run_dir>/llm_detections.json")
    sc.add_argument("run_dir")
    args = parser.parse_args()
    if args.cmd == "select":
        cmd_select(args.n)
    else:
        cmd_score(args.run_dir)


if __name__ == "__main__":
    main()
