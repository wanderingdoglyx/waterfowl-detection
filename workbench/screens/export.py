"""
Export reports: survey counts, corrected labels, the model comparison, one run.

Everything is written under reports/<date>_<name>/ at the repository root.
"""

from __future__ import annotations

import os

from workbench import catalog, detections, imaging, paths, reports, ui
from workbench.paths import rel
from workbench.screens import common
from workbench.util import ago


def run() -> None:
    while True:
        ui.title("Export reports")
        ui.dim(f"Reports are written to {rel(paths.REPORTS_DIR)}/<date>_<name>/.")
        ui.console.print()
        options = [
            ("survey", "Bird count report for a survey", "Markdown + HTML + CSV + annotated photos"),
            ("labels", "Corrected labels for training", "reviewed birds, in the dataset_full format"),
            ("board", "Model comparison", "every evaluated run, with its caveats"),
            ("run", "One version's report", "lineage, scores of each checkpoint, panels"),
        ]
        if os.path.isdir(paths.REPORTS_DIR) and imaging.viewer_available():
            options.append(("open", "Open the reports folder", rel(paths.REPORTS_DIR)))
        picked = ui.choose("What do you want to export?", options)
        if picked is None:
            return
        if picked == "survey":
            survey_report()
        elif picked == "labels":
            labels()
        elif picked == "board":
            out = reports.leaderboard_report(catalog.evaluated())
            _done(out, "leaderboard")
        elif picked == "run":
            ui.title("Export reports", "One version")
            chosen = common.pick_version("report on", include_active=True,
                                         with_checkpoint=False)
            if chosen:
                path = reports.run_report(chosen[0])
                _done(os.path.dirname(path), "report")
        elif picked == "open":
            ui.note(imaging.open_external(paths.REPORTS_DIR))
            ui.pause()


def _done(out: str, stem: str) -> None:
    ui.ok(f"Wrote {rel(out)}/")
    for name in sorted(os.listdir(out)):
        full = os.path.join(out, name)
        ui.dim(f"    {name}{'/' if os.path.isdir(full) else ''}")
    tool = imaging.viewer_available()
    if tool:
        # VS Code shows Markdown with a formatted preview (Ctrl+Shift+V); a desktop viewer
        # opens the HTML in a browser.
        target = os.path.join(out, f"{stem}.md" if tool == "code" else f"{stem}.html")
        if os.path.exists(target) and ui.confirm(f"Open {os.path.basename(target)}?", default=True):
            ui.note(imaging.open_external(target)
                    + (" — Ctrl+Shift+V shows it formatted" if tool == "code" else ""))
    ui.pause()


def _pick_result(message: str) -> detections.Result | None:
    found = [r for r in detections.all_results() if r.photos_done() or r.state() == "done"]
    if not found:
        ui.warn("No detection results yet — count the birds in a survey first.")
        ui.pause()
        return None
    options = []
    for r in found:
        done, total = r.review_progress()
        options.append((r, r.title, f"{done}/{total} cells reviewed · {ago(r.meta.get('created'))}"))
    return ui.choose(message, options)


def survey_report(result: detections.Result | None = None) -> None:
    ui.title("Export reports", "Bird count report")
    result = result or _pick_result("Which detection result?")
    if result is None:
        return
    review = result.review()
    photos = result.photos()
    ui.note(f"Counts use the threshold {review.threshold:.3f}"
            + (" (the run's t*)" if abs(review.threshold - result.threshold) < 1e-9 else
               " (set in the reviewer)")
            + ", plus every decision made in review.")
    t = ui.ask_float("Threshold for the report:", default=round(review.threshold, 3), lo=0.05,
                     hi=0.99)
    if t is None:
        return
    if abs(t - round(review.threshold, 3)) < 1e-9:
        t = review.threshold
    images = ui.confirm(f"Include an annotated overview of each photograph ({len(photos)})?",
                        default=len(photos) <= 200)
    with ui.console.status("Writing the report…"):
        out = reports.survey_report(result, threshold=t, images=images)
    _done(out, "report")


def labels(result: detections.Result | None = None) -> None:
    ui.title("Export reports", "Corrected labels")
    result = result or _pick_result("Which detection result?")
    if result is None:
        return
    review = result.review()
    photos = result.photos()
    complete = sum(1 for p in photos if review.counts(p).reviewed)
    ui.dim("Every photograph gets a bird,x1,y1,x2,y2 file holding the birds the review "
           "settled on — the format of dataset_full/, ready to import as a training "
           "collection.")
    ui.console.print()
    options = []
    if complete:
        options.append((True, f"Only fully reviewed photographs ({complete})",
                        "every cell checked by a person — safest for training"))
    options.append((False, f"All photographs ({len(photos)})",
                    "unreviewed cells keep the model's verdict"))
    only = ui.choose("Which photographs?", options)
    if only is None:
        return
    with ui.console.status("Writing labels…"):
        out, n, birds = reports.corrected_labels(result, only_reviewed=only)
    ui.ok(f"{n} photograph(s), {birds:,} birds → {rel(out)}/")
    collection = next((e.path for e in os.scandir(out) if e.is_dir()), None)
    if collection and ui.confirm("Import them as a training collection now?", default=False):
        from workbench.screens import data
        data.import_labelled(collection)
    else:
        ui.pause()
