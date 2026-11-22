"""
Review & correct: pick a detection result, choose which cells to visit, and open the
full-screen reviewer (workbench/reviewer.py).
"""

from __future__ import annotations

from workbench import detections, reviewer, ui
from workbench.paths import rel
from workbench.util import ago

MIN_COLS, MIN_ROWS = 110, 32


def run() -> None:
    while True:
        ui.title("Review & correct detections")
        found = [r for r in detections.all_results() if r.photos_done() or r.state() == "done"]
        if not found:
            ui.dim("Nothing to review yet — count the birds in a survey first "
                   "(Detect birds in new imagery).")
            ui.pause()
            return
        options = []
        for r in found:
            done, total = r.review_progress()
            progress = f"{done}/{total} cells reviewed" if done else "not reviewed"
            options.append((r, r.title, f"{progress} · {r.state()} · {ago(r.meta.get('created'))}"))
        picked = ui.choose("Which detection result?", options)
        if picked is None:
            return
        open_session(picked.id)


def open_session(result_id: str) -> None:
    result = detections.load(result_id)
    if result is None or not result.photos():
        ui.warn("That result has no photographs finished yet.")
        ui.pause()
        return
    ui.title("Review & correct detections", result.title)
    review = result.review()
    photos = result.photos()
    before = sum(review.counts(p).model for p in photos)
    after = sum(review.counts(p).final for p in photos)
    done, total = result.review_progress()
    survey = result.survey()
    ui.show(ui.kv([
        ("Photographs", f"{len(photos):,}" + (" (the count is still running)"
                                              if result.state() == "running" else "")),
        ("Birds", f"{before:,} counted by the model at threshold {review.threshold:.3f}"
                  + (f" → {after:,} after review so far" if review.touched else "")),
        ("Reviewed", f"{done:,} of {total:,} cells"),
        ("Labels", f"{survey.labelled} photograph(s) have them — shown in magenta in the "
                   "reviewer (g hides them)" if survey and survey.labelled
         else "none (an unlabelled survey)"),
    ]))
    ui.console.print()
    ui.dim("In the reviewer: ↑↓ pick a detection · y / n bird or not · click (or a) to add a "
           "missed bird · Enter when a cell is right · q to leave.  Everything is saved as "
           "you go; ? lists every key.")
    ui.console.print()

    sizes = {m: len(reviewer.Reviewer(result, m).queue)
             for m in ("detections", "uncertain", "unreviewed", "every")}
    options = [("detections", "Cells with detections", f"{sizes['detections']:,} cells — the usual pass"),
               ("uncertain", "Borderline cells first",
                f"{sizes['uncertain']:,} cells with a score within ±{reviewer.BAND} of the threshold")]
    if done:
        options.append(("unreviewed", "Cells not reviewed yet", f"{sizes['unreviewed']:,} cells"))
    options.append(("every", "Every cell, empty ones too",
                    f"{sizes['every']:,} cells — an audit that also finds birds the model missed"))
    mode = ui.choose("Which cells?", options, default="unreviewed" if done else "detections")
    if mode is None:
        return
    if sizes[mode] == 0:
        ui.ok("No cells to visit in that selection.")
        ui.pause()
        return
    cols, rows = ui.console.size
    if cols < MIN_COLS or rows < MIN_ROWS:
        ui.warn(f"This terminal is {cols}×{rows}; the reviewer is much easier to use at "
                f"{MIN_COLS}×{MIN_ROWS} or more.  In VS Code, maximise the terminal panel "
                "(or move the terminal into the editor area).")
        if not ui.confirm("Open it anyway?", default=True):
            return

    rv = reviewer.run(result, mode)
    ui.title("Review & correct detections", result.title)
    result = detections.load(result_id)
    review = result.review()
    photos = result.photos()
    after = sum(review.counts(p).final for p in photos)
    done, total = result.review_progress()
    ui.show(ui.kv([
        ("Birds", f"{before:,} by the model → [bold]{after:,}[/] after review ({after - before:+,})"),
        ("Reviewed", f"{done:,} of {total:,} cells ({100 * done / max(total, 1):.0f}%)"),
        ("Saved in", rel(review.path)),
    ], title="Review saved"))
    if rv is not None and not rv.queue:
        ui.dim("  (there were no cells to visit)")
    picked = ui.choose("Next:", [("report", "Export the count report", "Markdown, HTML, CSV"),
                                 ("labels", "Export corrected labels for training", "")],
                       back_label="Done")
    from workbench.screens import export
    if picked == "report":
        export.survey_report(result)
    elif picked == "labels":
        export.labels(result)
