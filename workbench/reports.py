"""
Exports: what leaves the workbench, written under reports/ at the repository root.

    survey_report(result)      counts per photograph, before and after review, as
                               Markdown + HTML + CSV, with an annotated overview of each
                               photograph and — for a labelled survey — accuracy against
                               the labels
    corrected_labels(result)   the reviewed birds as training data: one bird,x1,y1,x2,y2
                               file per photograph (the dataset_full/ format), an
                               image_info.csv, and a COCO file
    leaderboard_report(runs)   the model comparison, with its caveats
    run_report(run)            one run's scores and provenance

Accuracy against a labelled survey is computed with the project's own protocol
(data_prep/point_metrics.py: greedy score-ordered matching within τ = 40 px), on whole
photographs rather than 512 px crops.
"""

from __future__ import annotations

import csv
import html
import os
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from data_prep import config
from data_prep.point_metrics import _match_image, paper_point_metrics
from workbench import catalog, detections, paths
from workbench.paths import rel
from workbench.util import num, pct, slug

OVERVIEW_WIDTH = 1600
POINT_BOX_PX = config.MDO_PSEUDO_BOX   # box drawn round a point when a label needs one


def _out_dir(name: str) -> str:
    base = os.path.join(paths.REPORTS_DIR, f"{time.strftime('%Y-%m-%d')}_{slug(name)}")
    out, n = base, 1
    while os.path.exists(out):
        n += 1
        out = f"{base}-{n}"
    os.makedirs(out)
    return out


def _md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" * len(header)
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join("—" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


_CSS = """
:root { color-scheme: light dark; --ink:#1f2328; --muted:#59636e; --line:#d1d9e0; --bg:#fff;
        --head:#f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e6edf3; --muted:#9198a1; --line:#3d444d; --bg:#0d1117; --head:#151b23; } }
body { font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
       background: var(--bg); max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.7rem; margin-bottom: .2rem; } h2 { margin-top: 2.2rem; border-bottom:
     1px solid var(--line); padding-bottom: .3rem; }
table { border-collapse: collapse; margin: 1rem 0; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid var(--line); padding: .35rem .7rem; }
th { background: var(--head); text-align: left; }
td:not(:first-child) { text-align: right; }
img { max-width: 100%; border: 1px solid var(--line); border-radius: 4px; }
code { font-size: .9em; } p.muted, .muted { color: var(--muted); }
"""


def _write_html(md_text: str, path: str, title: str) -> str:
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    with open(path, "w") as f:
        f.write(f"<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
                f"<body>\n{body}\n</body></html>\n")
    return path


# ── Survey count report ───────────────────────────────────────────────────────

def validation(result: detections.Result, threshold: float) -> dict | None:
    """Accuracy of the model — and of the reviewed counts — against the survey's labels."""
    survey = result.survey()
    if survey is None or not survey.labelled:
        return None
    review = result.review()
    gt_by, dets_by, rows = {}, {}, []
    tp = fp = n_gt = 0
    for p in result.photos():
        boxes = survey.boxes(survey.photo(p.file))
        if boxes is None:
            continue
        gt = np.asarray([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]).reshape(-1, 2)
        gt_by[p.file] = [tuple(g) for g in gt]
        dets_by[p.file] = [(float(x), float(y), float(s)) for (x, y), s in zip(p.xy, p.score)]
        keep = p.score >= threshold
        order = np.argsort(-p.score[keep], kind="stable")
        flags = _match_image(gt, p.xy[keep][order], float(config.PAPER_TAU))
        tp, fp, n_gt = tp + int(flags.sum()), fp + int((~flags).sum()), n_gt + len(gt)
        counts = review.counts(p, threshold)
        rows.append((p.file, len(gt), counts.model, counts.final))
    if not rows:
        return None
    protocol = paper_point_metrics(gt_by, dets_by, tau=float(config.PAPER_TAU), bootstrap=0)
    gts = np.asarray([r[1] for r in rows], float)
    model = np.asarray([r[2] for r in rows], float)
    final = np.asarray([r[3] for r in rows], float)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    return {
        "photos": len(rows), "gt": int(gts.sum()), "tp": tp, "fp": fp, "fn": n_gt - tp,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "ap": protocol["detection"]["ap"],
        "mae_model": float(np.mean(np.abs(model - gts))),
        "mae_reviewed": float(np.mean(np.abs(final - gts))),
        "total_model": int(model.sum()), "total_reviewed": int(final.sum()),
        "per_photo": {r[0]: r[1] for r in rows},
    }


def _overview(result: detections.Result, photo: detections.PhotoResult, out_path: str,
              threshold: float) -> str:
    survey, review = result.survey(), result.review()
    with Image.open(survey.path(survey.photo(photo.file))) as im:
        img = im.convert("RGB")
    scale = min(1.0, OVERVIEW_WIDTH / img.width)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    r = max(3, int(9 * scale * 2))
    marks = review.marks(photo.file)
    for i, ((x, y), s) in enumerate(zip(photo.xy, photo.score)):
        mark = marks.get(i)
        px, py = x * scale, y * scale
        if mark == detections.NOT_BIRD:
            draw.line([px - r, py - r, px + r, py + r], fill=(235, 64, 52), width=2)
            draw.line([px - r, py + r, px + r, py - r], fill=(235, 64, 52), width=2)
        elif mark == detections.BIRD or (mark is None and s >= threshold):
            colour = (46, 204, 113) if mark == detections.BIRD else (255, 176, 0)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=colour, width=2)
    for x, y in review.added(photo.file):
        px, py = x * scale, y * scale
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 210, 255), width=2)
    img.save(out_path, quality=88)
    return out_path


def survey_report(result: detections.Result, threshold: float | None = None,
                  images: bool = True) -> str:
    review = result.review()
    t = review.threshold if threshold is None else threshold
    photos = result.photos()
    survey = result.survey()
    meta = result.meta
    out = _out_dir(f"{meta['survey']}_{meta['model']}")
    stem = f"counts_{slug(meta['survey'])}"

    per_photo = [(p, review.counts(p, t)) for p in photos]
    check = validation(result, t)
    total_model = sum(c.model for _, c in per_photo)
    total_final = sum(c.final for _, c in per_photo)
    cells_done = sum(c.cells_done for _, c in per_photo)
    cells_total = sum(c.cells_total for _, c in per_photo)

    # CSVs
    with open(os.path.join(out, f"{stem}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["photo", "width", "height", "cells", "cells_reviewed", "model_count",
                  "rejected", "promoted", "added", "reviewed_count"]
        if check:
            header.append("annotated_count")
        w.writerow(header)
        for p, c in per_photo:
            row = [os.path.basename(p.file), p.width, p.height, c.cells_total, c.cells_done,
                   c.model, c.rejected, c.promoted, c.added, c.final]
            if check:
                row.append(check["per_photo"].get(p.file, ""))
            w.writerow(row)
    with open(os.path.join(out, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "x", "y", "w", "h", "score", "source", "decision", "counted"])
        for p in photos:
            marks = review.marks(p.file)
            for i, ((x, y), (bw, bh), s) in enumerate(zip(p.xy, p.wh, p.score)):
                decision = marks.get(i, "")
                counted = decision == detections.BIRD or (not decision and s >= t)
                w.writerow([os.path.basename(p.file), round(x, 1), round(y, 1),
                            "" if np.isnan(bw) else round(bw, 1),
                            "" if np.isnan(bh) else round(bh, 1), round(float(s), 4),
                            "model", decision, int(counted)])
            for x, y in review.added(p.file):
                w.writerow([os.path.basename(p.file), x, y, "", "", "", "reviewer", "bird", 1])

    # overview images
    if images:
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        for p, _ in per_photo:
            _overview(result, p, os.path.join(out, "images",
                                              os.path.splitext(os.path.basename(p.file))[0] + ".jpg"), t)

    # the report
    rm = meta.get("run_metrics") or {}
    model_line = f"{meta['model_label']}, run `{meta['run']}`"
    if meta.get("checkpoint_kind") == "last":
        model_line += ", last checkpoint"
    if rm.get("ap") is not None:
        model_line += (f" — on the test split: point AP {pct(rm['ap'])}, F1 {pct(rm.get('f1'))}, "
                       f"counting MAE {num(rm.get('mae'))} birds per 512 px crop")
    t_note = ("the run's t*, the score cut that minimised counting error on the test split"
              if abs(t - result.threshold) < 1e-9 and "t*" in meta.get("threshold_source", "")
              else "chosen by the reviewer")
    lines = [f"# Bird counts — {meta['survey']}", "",
             f"<p class='muted'>Waterfowl Workbench · generated {time.strftime('%Y-%m-%d %H:%M')}</p>",
             "", "## Summary", "",
             _md_table(["", ""], [
                 ["Photographs", f"{len(photos):,}" + (f" of {len(survey.photos):,} in the survey"
                                                       if survey and len(photos) < len(survey.photos) else "")],
                 ["**Birds counted**", f"**{total_final:,}**"],
                 ["Model count before review", f"{total_model:,}"],
                 ["Changed by review", f"{total_final - total_model:+,}"],
                 ["Reviewed by a person", f"{cells_done:,} of {cells_total:,} cells "
                                          f"({100 * cells_done / max(cells_total, 1):.0f}%)"],
                 ["Model", model_line],
                 ["Threshold", f"{t:.3f} — {t_note}"],
             ] + [[k.replace("_", " ").capitalize(), v] for k, v in (survey.meta if survey else {}).items()])]
    if check:
        lines += ["", "## Accuracy against the survey's labels", "",
                  f"This survey came with label files for {check['photos']} photograph(s), "
                  f"holding {check['gt']:,} annotated birds.  Detections were matched to "
                  f"annotated birds one-to-one within {config.PAPER_TAU} px, highest score first "
                  "(the project's point protocol).", "",
                  _md_table(["", "Model", "After review"], [
                      ["Birds counted", f"{check['total_model']:,}", f"{check['total_reviewed']:,}"],
                      ["Annotated birds", f"{check['gt']:,}", f"{check['gt']:,}"],
                      ["Mean absolute error per photograph", num(check["mae_model"], 2),
                       num(check["mae_reviewed"], 2)],
                      ["Precision at the threshold", pct(check["precision"]), ""],
                      ["Recall at the threshold", pct(check["recall"]), ""],
                      ["F1 at the threshold", pct(check["f1"]), ""],
                      ["Point AP (threshold-free)", pct(check["ap"]), ""],
                  ], "lrr"), "",
                  "Annotations can be incomplete: the project found 1,639 detections that all "
                  "eight models agree on but no label covers.  A \"false positive\" here may be "
                  "an unlabelled bird — the reviewed column is the one to trust."]
    rows = []
    for p, c in per_photo:
        row = [os.path.basename(p.file), f"{c.model:,}", f"{c.final:,}", f"{c.final - c.model:+,}",
               f"{c.cells_done}/{c.cells_total}"]
        if check:
            row.append(check["per_photo"].get(p.file, "—"))
        rows.append(row)
    header = ["Photograph", "Model", "Reviewed", "Change", "Cells reviewed"]
    if check:
        header.append("Annotated")
    lines += ["", "## Per photograph", "", _md_table(header, rows, "l" + "r" * (len(header) - 1)),
              "", f"Full table: `{stem}.csv`; every detection with its score and decision: "
                  "`detections.csv`."]
    if images:
        lines += ["", "## Photographs", "",
                  "Amber: counted by the model, not reviewed · green: confirmed · red cross: "
                  "rejected · cyan: added by the reviewer.", ""]
        for p, c in per_photo:
            name = os.path.splitext(os.path.basename(p.file))[0] + ".jpg"
            lines += [f"**{os.path.basename(p.file)}** — {c.final:,} birds", "",
                      f"![{name}](images/{name})", ""]
    lines += ["", "## How these numbers were made", "",
              f"- Each photograph was cut into {config.CROP_SIZE} × {config.CROP_SIZE} px tiles at "
              f"a {detections.stride()} px stride — the same tiling as the training and test "
              "crops — and every tile was scored the way the model's own evaluation scores a crop.",
              "- Neighbouring tiles overlap, so each bird was counted once: a tile keeps only "
              "the detections in the part of the photograph it owns (bounded by the middle of "
              "each overlap), and a bird seen by two tiles near that boundary is merged.",
              f"- A detection counts when its score is at least the threshold ({t:.3f}), unless "
              "a reviewer marked it otherwise.  Birds a reviewer added count too.  Cells no "
              "one reviewed keep the model's verdict.",
              "- The model's accuracy figures above were measured on 512 px test crops, and "
              "are a guide to — not a guarantee of — its accuracy on these photographs."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(md)
    _write_html(md, os.path.join(out, "report.html"), f"Bird counts — {meta['survey']}")
    return out


# ── Corrected labels ──────────────────────────────────────────────────────────

def corrected_labels(result: detections.Result, only_reviewed: bool = True,
                     collection: str | None = None) -> tuple[str, int, int]:
    """Reviewed birds as training labels.  Returns (folder, photographs, birds)."""
    review, survey = result.review(), result.survey()
    photos = [p for p in result.photos()
              if not only_reviewed or review.counts(p).reviewed]
    sizes = [p.wh[~np.isnan(p.wh[:, 0])] for p in result.photos()]
    sizes = np.concatenate(sizes) if sizes else np.zeros((0, 2))
    default_w, default_h = ((float(np.median(sizes[:, 0])), float(np.median(sizes[:, 1])))
                            if len(sizes) else (POINT_BOX_PX, POINT_BOX_PX))
    name = collection or slug(f"{result.meta['survey']}_reviewed")
    out = _out_dir(f"{result.meta['survey']}_labels")
    folder = os.path.join(out, name)
    os.makedirs(folder)
    images, annotations, index_rows = [], [], []
    birds = 0
    for n, p in enumerate(photos):
        src = survey.path(survey.photo(p.file))
        base = os.path.basename(p.file)
        stem = os.path.splitext(base)[0]
        os.symlink(os.path.realpath(src), os.path.join(folder, base))
        boxes = []
        for i in range(p.n):
            if not review.is_bird(p, i):
                continue
            w, h = p.wh[i]
            if np.isnan(w):
                w, h = default_w, default_h
            boxes.append((p.xy[i, 0] - w / 2, p.xy[i, 1] - h / 2, p.xy[i, 0] + w / 2, p.xy[i, 1] + h / 2))
        for x, y in review.added(p.file):
            boxes.append((x - default_w / 2, y - default_h / 2, x + default_w / 2, y + default_h / 2))
        clipped = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(p.width, int(round(x2))), min(p.height, int(round(y2)))
            if x2 > x1 and y2 > y1:
                clipped.append((x1, y1, x2, y2))
        with open(os.path.join(folder, f"{stem}.txt"), "w") as f:
            for x1, y1, x2, y2 in clipped:
                f.write(f"bird,{x1},{y1},{x2},{y2}\n")
        birds += len(clipped)
        altitude = (survey.meta or {}).get("altitude_m", "")
        index_rows.append([n, base, f"{stem}.txt", altitude, "train"])
        images.append({"id": n + 1, "file_name": base, "width": p.width, "height": p.height})
        for x1, y1, x2, y2 in clipped:
            annotations.append({"id": len(annotations) + 1, "image_id": n + 1, "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0})
    with open(os.path.join(folder, "image_info.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "image_name", "annotation_name", "height", "bbox_split_Robert"])
        w.writerows(index_rows)
    from workbench.util import write_json
    write_json(os.path.join(out, "coco.json"), {
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "bird", "supercategory": "none"}]}, indent=None)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# Reviewed labels — {result.meta['survey']}\n\n"
                f"{len(photos)} photograph(s), {birds:,} birds, from detection result "
                f"`{result.id}` ({result.meta['model_label']} run {result.meta['run']}) after "
                f"review, at threshold {review.threshold:.3f}.\n\n"
                f"- `{name}/` is a collection in the dataset_full/ format: the photographs "
                "(symlinks), one `bird,x1,y1,x2,y2` file each, and an image_info.csv marking "
                "every photograph `train`.  Import it with *Import labelled photos as a "
                "training collection*.\n"
                f"- `coco.json` holds the same boxes in COCO form.\n"
                + (f"- Birds without a box (added by hand, or found by a point detector) were "
                   f"given a {default_w:.0f} × {default_h:.0f} px box centred on the point"
                   + (" — the median detection box of this result." if len(sizes) else
                      " — the project's pseudo-box size.") + "\n")
                + ("- Only photographs whose every cell was reviewed are included.\n"
                   if only_reviewed else
                   "- Photographs not fully reviewed are included with the model's verdict "
                   "in unreviewed cells — check before training on them.\n"))
    return out, len(photos), birds


# ── Model comparison ──────────────────────────────────────────────────────────

def _run_rows(runs: list[catalog.Run]) -> list[list]:
    rows = []
    for r in runs:
        m = r.metrics
        rows.append([r.model.label, r.run_id + (" ↳" if r.parent else ""), pct(m.ap),
                     pct(m.precision), pct(m.recall),
                     pct(m.f1), num(m.mae), num(m.rmse), num(m.t_star),
                     "—" if m.map30 is None else f"{m.map30:.2f}%" + ("*" if m.map30_pseudo else ""),
                     f"{m.n_images:,}" if m.n_images else "—"])
    return rows


LEADER_HEADER = ["Model", "Run", "Point AP", "Precision", "Recall", "F1", "MAE", "RMSE", "t*",
                 "mAP30", "Test crops"]


def leaderboard_report(runs: list[catalog.Run], sort_key: str = "ap") -> str:
    out = _out_dir("leaderboard")
    current = [r for r in runs if r.metrics and r.metrics.split == "current"]
    legacy = [r for r in runs if r.metrics and r.metrics.split == "v1"]
    subset = [r for r in runs if r.metrics and r.metrics.split == "subset"]
    lines = ["# Model comparison", "",
             f"<p class='muted'>Waterfowl Workbench · generated {time.strftime('%Y-%m-%d %H:%M')} "
             f"· sorted by {sort_key}</p>", ""]
    for title, group in ((f"Current test split ({catalog.CURRENT_TEST_CROPS:,} crops)", current),
                         (f"Earlier test split ({catalog.LEGACY_TEST_CROPS:,} crops, Baseline v1)",
                          legacy),
                         ("Data-folder test splits (compare only equal test crops)", subset)):
        if group:
            lines += [f"## {title}", "", _md_table(LEADER_HEADER, _run_rows(group),
                                                   "ll" + "r" * 9), ""]
    lines += ["## Reading this table", "",
              f"- Point metrics follow the OWL paper's protocol (τ = {config.PAPER_TAU} px); "
              "precision, recall, F1 and the counting errors are read at each run's own t*.",
              "- MAE and RMSE are per-crop counting errors in birds.",
              "- \\* OWL rows' mAP30 is the pseudo-box bridge (points wrapped in 28 px boxes), "
              "not comparable with the box models' mAP30.",
              "- Runs scored on different test splits are in different tables and must not be "
              "ranked against each other.",
              "- ↳ marks a version fine-tuned from an earlier one; \"(last)\" marks a run's "
              "last checkpoint, scored with `--eval --checkpoint last`, beside its best."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out, "leaderboard.md"), "w") as f:
        f.write(md)
    _write_html(md, os.path.join(out, "leaderboard.html"), "Model comparison")
    with open(os.path.join(out, "leaderboard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "run", "ap", "precision", "recall", "f1", "mae", "rmse", "t_star",
                    "map30", "map30_pseudo", "test_crops", "started_from", "folder"])
        for r in current + legacy + subset:
            m = r.metrics
            w.writerow([r.model.key, r.run_id, m.ap, m.precision, m.recall, m.f1, m.mae, m.rmse,
                        m.t_star, m.map30, int(m.map30_pseudo), m.n_images, r.started_from(),
                        rel(r.path)])
    return out


def run_report(run: catalog.Run) -> str:
    out = _out_dir(f"{run.model.key}_{run.run_id}")
    lines = [f"# {run.model.label} — run {run.run_id}", "",
             f"<p class='muted'>Waterfowl Workbench · generated {time.strftime('%Y-%m-%d %H:%M')}</p>",
             "", _md_table(["", ""], [[k, v] for k, v in run.details()])]
    text = run.eval_text()
    if text:
        lines += ["", "## Test-set scores — best checkpoint", "", "```", text, "```"]
    last = run.eval_text("last")
    if last:
        lines += ["", "## Test-set scores — last checkpoint", "", "```", last, "```"]
    examples = run.examples()[:6]
    if examples:
        os.makedirs(os.path.join(out, "examples"))
        lines += ["", "## Example panels", "", "Left: ground truth · right: predictions.", ""]
        for e in examples:
            shutil.copy2(e, os.path.join(out, "examples", os.path.basename(e)))
            lines += [f"![{os.path.basename(e)}](examples/{os.path.basename(e)})", ""]
    md = "\n".join(lines) + "\n"
    path = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write(md)
    _write_html(md, os.path.join(out, "report.html"), run.name)
    return path

