#!/usr/bin/env python
"""
Waterfowl zero-shot baseline pipeline — how the pretrained checkpoints score BEFORE
any fine-tuning.

Every model in this project is fine-tuned from a public checkpoint.  The training
pipelines report only the "after" number, so there is nothing to attribute the result
to: how much of a model's mAP30 came from the waterfowl crops, and how much was already
in the released weights?  This pipeline answers that by scoring each *starting*
checkpoint on the same test split, with the same protocols, changing nothing but the
weights.

    output/baselines/<model>/<timestamp>/
        eval_results.txt          same layout as every --eval in this repo
        paper_point_metrics.json  OWL-paper point protocol (Section 4.3)
        metrics.json              + mAP30 / mAR30 and the run's provenance
        examples/                 GT | prediction panels

Two kinds of baseline, and the table must be read with the difference in mind:

  • COCO-pretrained box models (YOLOv5/11/26, YOLO-NAS, Faster R-CNN) have never seen an
    overhead crop, and COCO's only relevant label is "bird" — a category built from
    large, side-on, ground-level birds.  Detections are filtered to that class and
    remapped to category 1.  Near-zero scores are the expected, honest result.  Use
    --any-class to check the alternative hypothesis (birds found under a *different*
    COCO label, e.g. "kite" or "boat"); it keeps all 80 classes, so it is a diagnostic,
    not a comparable number.

  • Overhead-pretrained point models (OWL-C/T/D) were trained by Microsoft AI for Good
    on aerial wildlife and are already single-class animal detectors — no class remap,
    no category mismatch.  This is a real domain-transfer baseline, and the one that
    should be read as "how much did fine-tuning actually add".

Usage (from the rebuild/ directory):

    ./baseline/main.py --model yolo26                 # one model
    ./baseline/main.py --all                          # every registered baseline
    ./baseline/main.py --model yolo26 --any-class     # diagnostic, all 80 COCO classes
    ./baseline/main.py --summary                      # before/after comparison tables
    ./baseline/main.py --save                         # ...and write them to a text file

--summary prints two blocks, DETECTION (mAP30/mAR30/AP/AUC-PR/precision/recall/F1) and
COUNTING (MAE/RMSE/signed count error/t*), because one row carrying every metric as a
"before → after" pair is far too wide to read.

Requires the shared crops (any model's --prepare); it never re-prepares data itself,
because a baseline that rebuilt the splits would no longer be comparing like with like.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

# Project root = two levels up from this file (rebuild/baseline/main.py → rebuild/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import data_prep.config as config

# Default target for --save: alongside the per-model run directories it summarises, so the
# whole baseline story lives under output/ with the rest of the project's results rather
# than in the source tree.  Pass --save <path> to override; a directory is also accepted.
DEFAULT_SUMMARY_PATH = os.path.join(config.BASELINE_DIR, "summary.txt")


def make_run_dir(model_key: str) -> str:
    """Create and return output/baselines/<model>/<timestamp>/."""
    run_dir = os.path.join(config.BASELINE_DIR, model_key,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_dirs(parent: str) -> list[str]:
    """All timestamped run dirs under `parent`, newest first."""
    if not os.path.exists(parent):
        return []
    return sorted((e.path for e in os.scandir(parent) if e.is_dir()), reverse=True)


def latest_run_dir(parent: str) -> str | None:
    """Most recently created timestamped run dir under `parent`, or None."""
    runs = run_dirs(parent)
    return runs[0] if runs else None


def _save_examples(coco_results: list, eval_dir: str, n: int, test_json: str) -> None:
    """
    Save n side-by-side (ground truth | prediction) panels to eval_dir/examples/.

    Family-agnostic by construction: it draws from the shared COCO result list rather
    than from a live model, so one implementation covers box models and OWL's
    pseudo-boxes alike (contrast the per-family savers in yolov5/ and
    megadetector_overhead/, which each hold their own model object).
    """
    import random

    import cv2

    with open(test_json) as f:
        coco = json.load(f)

    gt_by_image: dict = {img["id"]: [] for img in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_image[a["image_id"]].append((x, y, x + w, y + h))

    pred_by_image: dict = {}
    for d in coco_results:
        if d["score"] < config.DISPLAY_CONF_THRESHOLD:
            continue
        x, y, w, h = d["bbox"]
        pred_by_image.setdefault(d["image_id"], []).append((x, y, x + w, y + h))

    # Prefer crops that actually contain birds — an all-background panel shows nothing.
    candidates = [img for img in coco["images"] if gt_by_image.get(img["id"])]
    if not candidates:
        candidates = coco["images"]
    random.seed(config.RANDOM_SEED)
    picks = random.sample(candidates, min(n, len(candidates)))

    out_dir = os.path.join(eval_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)
    for img in picks:
        im = cv2.imread(os.path.join(config.CROPS_IMG_DIR, img["file_name"]))
        if im is None:
            continue
        left, right = im.copy(), im.copy()
        for x1, y1, x2, y2 in gt_by_image.get(img["id"], []):
            cv2.rectangle(left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for x1, y1, x2, y2 in pred_by_image.get(img["id"], []):
            cv2.rectangle(right, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        panel = cv2.hconcat([left, right])
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img["file_name"])), panel)


# ── OWL baselines run out-of-process in the MDO uv venv ───────────────────────
def _run_owl_baseline(spec: dict, run_dir: str, test_json: str) -> tuple[dict, list]:
    """
    Score a released OWL checkpoint as-is.

    The fine-tuned path already isolates OWL inference behind
    megadetector_overhead.main.run_eval_helper(ckpt, out_json, model_spec), which shells
    into the MDO .venv.  A zero-shot run is the same call with the *pretrained* .pth in
    place of the run's best.pth — the fine-tuning config loads these checkpoints with
    partial_load=False, so they fit the eval architecture exactly as they are.

    Returns (metrics dict, COCO result list built from the detected points).
    """
    from megadetector_overhead.evaluate import evaluate_from_json, points_to_coco
    from megadetector_overhead.main import run_eval_helper

    mdo_spec = config.mdo_model_spec(spec["mdo_key"])
    owl_eval_json = os.path.join(run_dir, "owl_eval.json")
    run_eval_helper(spec["weights"], owl_eval_json, mdo_spec)

    metrics = evaluate_from_json(owl_eval_json, test_json)
    with open(owl_eval_json) as f:
        detections = json.load(f).get("detections", [])
    return metrics, points_to_coco(detections, test_json)


def _subset_test_json(test_json: str, limit: int, run_dir: str) -> str:
    """
    Write a `limit`-image subset of the test COCO json into run_dir and return its path.

    Subsetting the GROUND TRUTH (rather than just stopping inference early) keeps every
    downstream metric self-consistent: COCOeval and the point protocol both see the same
    image set, so a smoke run produces real — if statistically thin — numbers instead of
    a recall artificially crushed by images that were never scored.

    Images WITH annotations are preferred.  The split's leading crops are largely empty
    background, and a subset with zero GT points exercises none of the matching code —
    every metric is trivially 0.00% whether or not the model works.  This biases the
    subset, which is exactly why --limit runs are barred from --summary.
    """
    with open(test_json) as f:
        coco = json.load(f)

    annotated = {a["image_id"] for a in coco["annotations"]}
    with_gt = [img for img in coco["images"] if img["id"] in annotated]
    pool = with_gt if len(with_gt) >= limit else coco["images"]
    keep = pool[:limit]
    keep_ids = {img["id"] for img in keep}
    subset = dict(coco)
    subset["images"] = keep
    subset["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]

    path = os.path.join(run_dir, "test_subset_coco.json")
    with open(path, "w") as f:
        json.dump(subset, f)
    return path


def evaluate_baseline(model_key: str, any_class: bool, n_examples: int,
                      limit: int | None = None) -> str | None:
    """Run one zero-shot baseline end to end.  Returns its run dir, or None on failure."""
    from data_prep.point_metrics import (dets_from_coco_results, format_map30_block,
                                         format_report, gt_points_from_coco,
                                         paper_point_metrics, write_eval_txt)
    from yolov5.evaluate import coco_map30

    spec = config.baseline_model_spec(model_key)
    label = spec["label"]

    test_json = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
    if not os.path.exists(test_json):
        print(f"No test crops found at {test_json}. Run any model's --prepare first.")
        return None

    weights = spec["weights"]
    is_file_weight = spec["family"] in ("ultralytics", "owl")
    if is_file_weight and not os.path.exists(weights):
        hint = ("  Fetch it with:  ./megadetector_overhead/main.py "
                f"--model {spec.get('mdo_key')} --fetch-weights"
                if spec["family"] == "owl" else
                "  Ultralytics downloads it on first use; or run the model's --train once.")
        print(f"[{model_key}] Pretrained weights not found: {weights}\n{hint}")
        return None

    run_dir = make_run_dir(model_key)
    print(f"\n=== Zero-shot baseline: {label} ===")
    print(f"Run directory : {run_dir}")
    print(f"Weights       : {weights}")
    if spec["family"] != "owl":
        print(f"Class filter  : {'ALL 80 COCO classes (diagnostic)' if any_class else config.BASELINE_COCO_CLASS}")

    if limit and spec["family"] == "owl":
        # OWL scores a directory + gt.csv inside its own venv, not this COCO json, so a
        # subset here would silently be ignored.  Fail loudly rather than mislabel a
        # full-split run as a smoke run.
        print(f"[{model_key}] --limit is not supported for OWL baselines "
              "(they score the prepared point mirror, not the COCO json). Skipping.")
        return None
    if limit:
        test_json = _subset_test_json(test_json, limit, run_dir)
        print(f"LIMIT         : {limit} crops — smoke run, NOT a comparable number")

    sections = []
    if spec["family"] == "owl":
        metrics, coco_results = _run_owl_baseline(spec, run_dir, test_json)
        pm = metrics["paper_point_metrics"]
        pb = metrics["pseudo_box_map30"]
        ap30, ar30 = pb["ap30"], pb["ar30"]
        pt = metrics["point"]
        sections.append(
            f"── Point metric (native; TP within {pt.get('radius_fullres_px', '?')} px "
            "of a GT point) ──\n"
            f"  Precision : {pt.get('precision', 0) * 100:.2f}%\n"
            f"  Recall    : {pt.get('recall', 0) * 100:.2f}%\n"
            f"  F1        : {pt.get('f1_score', 0) * 100:.2f}%"
        )
        sections.append(
            f"── Pseudo-box mAP30 (points → {pb['box_size']}px boxes, "
            "COCOeval@IoU=0.30) ──\n"
            f"  mAP30 : {ap30:.2f}%\n  mAR30 : {ar30:.2f}%"
        )
    else:
        from baseline.detect import DETECTORS

        with open(test_json) as f:
            n_crops = len(json.load(f)["images"])
        print(f"Running inference over the test split ({n_crops} crops)...")
        coco_results = DETECTORS[spec["family"]](spec, test_json, any_class=any_class)
        ap30, ar30 = coco_map30(test_json, coco_results)
        pm = paper_point_metrics(
            gt_points_from_coco(test_json), dets_from_coco_results(coco_results),
            tau=config.PAPER_TAU, bootstrap=config.PAPER_BOOTSTRAP,
        )
        sections.append(format_map30_block(ap30, ar30))

    tag = f"{label} — zero-shot" + (f" [LIMIT {limit}]" if limit else "")
    sections.append(format_report(pm, tag))
    if not coco_results:
        sections.append("  (no detections above the eval threshold — the pretrained "
                        "model found nothing it would call a bird in these crops)")

    with open(os.path.join(run_dir, "paper_point_metrics.json"), "w") as f:
        json.dump(pm, f, indent=2)
    with open(os.path.join(test_json)) as f:
        n_test = len(json.load(f)["images"])
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({
            "model": model_key, "label": label, "family": spec["family"],
            "weights": weights, "zero_shot": True,
            "class_filter": None if any_class else config.BASELINE_COCO_CLASS,
            "timestamp": os.path.basename(run_dir),
            "limit": limit,
            "n_test": n_test,
            "n_detections": len(coco_results),
            "map30": ap30, "mar30": ar30,
            "paper_point_metrics": pm,
        }, f, indent=2)

    text = write_eval_txt(os.path.join(run_dir, "eval_results.txt"), sections)
    print("\n" + text)
    print(f"\nSaved results to {os.path.join(run_dir, 'eval_results.txt')}")

    if n_examples > 0:
        print(f"\nSaving {n_examples} example images...")
        _save_examples(coco_results, run_dir, n_examples, test_json)
    return run_dir


# ── Before/after summary ──────────────────────────────────────────────────────
def _read_metric(eval_txt: str, name: str) -> float | None:
    """
    Pull a named percentage metric out of an eval_results.txt.

    Covers both the box block ("mAP30 : 76.79%") and the OWL pseudo-box block, which
    uses the same labels.  These two numbers live only in the text report — no JSON in
    the run dir carries them — so the table has to parse them back out.
    """
    if not os.path.exists(eval_txt):
        return None
    with open(eval_txt) as f:
        for line in f:
            if line.strip().startswith(name):
                try:
                    return float(line.split(":")[1].strip().rstrip("%"))
                except (IndexError, ValueError):
                    return None
    return None


def _unpack_pm(pm: dict) -> dict:
    """Flatten a paper_point_metrics payload into the fields the table prints."""
    det, cnt = pm["detection"], pm["counting"]
    return {
        "ap":    det["ap"] * 100,
        "aucpr": det["auc_pr"] * 100,
        "prec":  det["precision_at_t_star"] * 100,
        "rec":   det["recall_at_t_star"] * 100,
        "f1":    det["f1_at_t_star"] * 100,
        "mae":   cnt["mae"],
        "rmse":  cnt["rmse"],
        "err":   cnt["signed_pct_error"],
        "pred":  cnt["total_pred_at_t_star"],
        "gt":    cnt["total_gt"],
        "tstar": cnt["t_star"],
    }


def _finetuned_metrics(model_key: str) -> dict:
    """
    Every table metric for the newest *evaluated* fine-tuned run of `model_key`.

    Walks the run dirs newest-first rather than taking the newest outright: a training
    run that is still in flight (or was killed before --eval) has a run dir but no
    metrics, and reporting "—" for a model that has perfectly good older results would
    be misleading.
    """
    for run in run_dirs(config.BASELINE_FINETUNED_DIRS.get(model_key, "")):
        # Box models write straight into the run dir; OWL nests its results under eval/.
        for base in (run, os.path.join(run, "eval")):
            pm = None
            pm_path = os.path.join(base, "paper_point_metrics.json")
            metrics_path = os.path.join(base, "metrics.json")
            if os.path.exists(pm_path):
                with open(pm_path) as f:
                    pm = json.load(f)
            elif os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    pm = json.load(f).get("paper_point_metrics")
            if pm:
                eval_txt = os.path.join(base, "eval_results.txt")
                out = _unpack_pm(pm)
                out.update(map30=_read_metric(eval_txt, "mAP30"),
                           mar30=_read_metric(eval_txt, "mAR30"),
                           run=os.path.basename(run))
                return out
    return {}


def _baseline_metrics(model_key: str) -> dict:
    """Same newest-with-results walk over this model's zero-shot runs."""
    for run in run_dirs(os.path.join(config.BASELINE_DIR, model_key)):
        path = os.path.join(run, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        # A --limit smoke run is not a comparable number; never let it into the table.
        if m.get("limit"):
            continue
        out = _unpack_pm(m["paper_point_metrics"])
        out.update(map30=m.get("map30"), mar30=m.get("mar30"),
                   run=os.path.basename(run))
        return out
    return {}


# Column spec: (heading, field, format).  Split across two tables because one row of
# eleven "before → after" pairs is ~250 characters — unreadable in a terminal and worse
# in a text file.  Detection answers "did it find the bird", counting answers "how many
# were there"; those are the two questions the project actually reports on.
def _pred_cell(d: dict) -> str | None:
    """`Pred. (Err %)` exactly as Table 2 of the OWL paper prints it: the total predicted
    count over the split, with its signed percentage error against the GT total."""
    if d.get("pred") is None:
        return None
    return f"{d['pred']:,} ({d['err']:+.1f}%)"


# Column spec: (heading, key, format, width).  `format` is either a str.format template
# applied to row[key], or a callable taking the whole row — the latter lets one column
# combine several fields, as the paper's count column does.
_DETECTION_COLS = [("mAP30", "map30", "{:.2f}", 17), ("mAR30", "mar30", "{:.2f}", 17),
                   ("point AP", "ap", "{:.2f}", 17), ("AUC-PR", "aucpr", "{:.2f}", 17),
                   ("precision", "prec", "{:.2f}", 17), ("recall", "rec", "{:.2f}", 17),
                   ("F1", "f1", "{:.2f}", 17)]
_COUNTING_COLS = [("MAE", "mae", "{:.3f}", 17), ("RMSE", "rmse", "{:.3f}", 17),
                  ("t*", "tstar", "{:.3f}", 17),
                  ("Pred. (Err %)", None, _pred_cell, 42)]


def _table(title: str, cols: list, rows: list) -> list[str]:
    """Render one before → after block; `rows` is [(label, baseline, finetuned), ...]."""
    label_w = 28

    def cell(d, key, fmt):
        if callable(fmt):
            return fmt(d) or "  —  "
        v = d.get(key)
        return "  —  " if v is None else fmt.format(v)

    header = f"{'Model':<{label_w}}" + "".join(f"{h:>{w}}" for h, _, _, w in cols)
    lines = ["═" * len(header), title, "═" * len(header), header, "─" * len(header)]
    for label, b, f in rows:
        cells = "".join(
            f"{cell(b, key, fmt) + ' → ' + cell(f, key, fmt):>{w}}"
            for _, key, fmt, w in cols)
        lines.append(f"{label:<{label_w}}" + cells)
    lines.append("─" * len(header))
    return lines


def build_summary() -> str:
    """
    Render the zero-shot vs fine-tuned comparison as text.

    Returned rather than printed so the same bytes can go to the terminal and to
    --save, which keeps a saved snapshot from drifting out of sync with the run
    directories it was built from.
    """
    rows, provenance = [], []
    for key, spec in config.BASELINE_MODELS.items():
        b, f = _baseline_metrics(key), _finetuned_metrics(key)
        if not b and not f:
            continue
        rows.append((spec["label"], b, f))
        provenance.append(f"  {spec['label']:<28} zero-shot {b.get('run', '—'):<22} "
                          f"fine-tuned {f.get('run', '—')}")

    if not rows:
        return "No results yet — run ./baseline/main.py --all first.\n"

    lines = _table("Zero-shot pretrained  →  fine-tuned   ·  DETECTION"
                   "   (test split, identical protocols)", _DETECTION_COLS, rows)
    lines.append("")
    gt = next((r[2].get("gt") or r[1].get("gt") for r in rows
               if r[2].get("gt") or r[1].get("gt")), None)
    gt_note = f"; GT total {gt:,}" if gt else ""
    lines += _table("Zero-shot pretrained  →  fine-tuned   ·  COUNTING"
                    f"   (per-image error at t*{gt_note})", _COUNTING_COLS, rows)
    lines += [
        "",
        "mAP30/mAR30 for OWL rows is the pseudo-box bridge metric, not a true box mAP.",
        "AP/AUC-PR rank the full score sweep; precision/recall/F1 and every counting",
        "column are read at each model's own optimal threshold t*, so a model can win",
        "one block and lose the other.  Pred. (Err %) follows Table 2 of the OWL paper:",
        "the total predicted count over the split, signed against GT — - under, + over.",
        "'—' means that side has not been run yet.",
        "",
        # A bare table cannot be audited later: two runs of the same model give
        # different numbers, so record which run dir each side was read from.
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from:",
    ]
    lines += provenance
    return "\n".join(lines) + "\n"


def print_summary(save_path: str | None = None) -> None:
    """
    Print the summary table, and write it to `save_path` when given.

    `save_path` may name a directory (existing, or written with a trailing separator),
    in which case the table lands in it as summary.txt — passing the folder is the
    obvious thing to type, and silently creating a *file* named after the folder would
    be a confusing way to punish it.
    """
    text = build_summary()
    print("\n" + text)
    if not save_path:
        return

    path = os.path.abspath(save_path)
    if os.path.isdir(path) or save_path.endswith(os.sep):
        path = os.path.join(path, "summary.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved summary to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Waterfowl zero-shot baselines (pretrained, before fine-tuning)")
    parser.add_argument("--model", default=None, choices=list(config.BASELINE_MODELS),
                        help="Which pretrained checkpoint to score")
    parser.add_argument("--all", action="store_true",
                        help="Score every registered baseline in turn")
    parser.add_argument("--any-class", action="store_true",
                        help="Diagnostic: keep all 80 COCO classes instead of just "
                             "'bird' (box models only; not a comparable number)")
    parser.add_argument("--summary", action="store_true",
                        help="Print the zero-shot vs fine-tuned comparison table")
    parser.add_argument("--examples", type=int, default=40, metavar="N",
                        help="Example panels to save per model (default: 40, 0 to skip)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Smoke run: score only the first N test crops. Verifies the "
                             "pipeline end to end in seconds; the numbers are NOT "
                             "comparable and are excluded from --summary. Box models only.")
    parser.add_argument("--save", nargs="?", const=DEFAULT_SUMMARY_PATH, default=None,
                        metavar="PATH",
                        help=f"Write the summary table to a text file "
                             f"(default: {os.path.relpath(DEFAULT_SUMMARY_PATH, ROOT)}). "
                             f"Implies --summary.")
    args = parser.parse_args()

    if not (args.model or args.all or args.summary or args.save):
        parser.print_help()
        return

    keys = list(config.BASELINE_MODELS) if args.all else ([args.model] if args.model
                                                          else [])
    failed = []
    for key in keys:
        try:
            if evaluate_baseline(key, args.any_class, args.examples,
                                 limit=args.limit) is None:
                failed.append(key)
        except Exception as exc:                       # keep --all going past one bad model
            print(f"[{key}] FAILED: {type(exc).__name__}: {exc}")
            failed.append(key)

    if keys:
        done = [k for k in keys if k not in failed]
        print(f"\nCompleted {len(done)}/{len(keys)} baselines"
              + (f"; failed: {', '.join(failed)}" if failed else ""))

    if args.summary or args.all or args.save:
        print_summary(save_path=args.save)


if __name__ == "__main__":
    main()
