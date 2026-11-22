"""
Detect birds in new imagery: choose a survey and a trained run, and count.

The run's own t* is the default counting threshold — the score cut that minimised the
counting error (MAE) on the test split, i.e. the operating point its published counting
numbers describe.  Every detection above the 0.05 floor is kept anyway, so the threshold
can be changed later without running the model again.
"""

from __future__ import annotations


from data_prep import config
from workbench import catalog, detections, surveys, ui
from workbench.jobs import python_cmd
from workbench.paths import rel
from workbench.screens import common
from workbench.util import ago, human_duration, num, pct


def run() -> None:
    while True:
        ui.title("Detect birds in new imagery")
        found = detections.all_results()
        if found:
            rows = []
            for r in found[:15]:
                done, total = r.review_progress()
                rows.append([r.meta.get("survey"), f"{r.meta.get('model_label')} {r.meta.get('run')}",
                             f"{r.photos_done()}/{r.meta.get('photos_total', '?')}",
                             f"{r.meta.get('birds_at_threshold', '—'):,}"
                             if isinstance(r.meta.get("birds_at_threshold"), int) else "—",
                             common.styled(r.state()), f"{done}/{total}" if done else "—",
                             ago(r.meta.get("created"))])
            ui.show(ui.table(["Survey", "Model run", ("Photos", "right"), ("Birds", "right"),
                              "State", ("Cells reviewed", "right"), "Started"], rows,
                             title="Detection results"))
        else:
            ui.dim("No detection results yet.  Import a survey, then count the birds in it.")
            ui.console.print()
        options = [("new", "Count birds in a survey", "choose survey, model and threshold"),
                   ("import", "Import a new survey first", "photographs from a folder")]
        options += [(r, f"Open: {r.title}", r.state()) for r in found[:15]]
        picked = ui.choose("What do you want to do?", options)
        if picked is None:
            return
        if picked == "new":
            start()
        elif picked == "import":
            from workbench.screens import data
            data.import_survey()
        else:
            result_menu(picked)


def estimate_seconds(model: catalog.Model, survey: surveys.Survey) -> float:
    tiles = sum(detections.n_cells(p.width, p.height) for p in survey.photos)
    return tiles * model.ms_per_tile / 1000.0 + 0.4 * len(survey.photos)


def pick_detector(survey: surveys.Survey | None
                  ) -> tuple[catalog.Run, catalog.Checkpoint | None] | None:
    """Which weights count the birds: the scored versions, sorted by counting error, or
    — through the last entry — any trained version and either of its checkpoints."""
    usable = [r for r in catalog.evaluated() if r.checkpoint and not r.active]
    others = any(r.checkpoints() and not r.active for r in catalog.runs())
    if not usable and not others:
        ui.error("No trained version with a checkpoint is available yet.")
        ui.pause()
        return None
    current = [r for r in usable if r.metrics.split == "current"]
    recommended = catalog.best(current or usable, "mae", lower_is_better=True) if usable else None
    rank = {"current": 0, "v1": 1, "subset": 2}
    usable.sort(key=lambda r: (rank[r.metrics.split],
                               r.metrics.mae if r.metrics.mae is not None else 9))
    ui.dim("Scored versions, sorted by counting error on the test split (MAE per 512 px crop; "
           "lower is better), each with its best checkpoint.  Times are rough, from each "
           "model's measured speed on an RTX 4080 SUPER.")
    ui.console.print()
    options = []
    for r in usable:
        eta = f" · ≈ {human_duration(estimate_seconds(r.model, survey))}" if survey else ""
        options.append((r, f"{r.model.label:13s} {r.run_id}",
                        f"MAE {num(r.metrics.mae)} · AP {pct(r.metrics.ap)} · "
                        f"recall {pct(r.metrics.recall)}{eta}"
                        + ("  ★ lowest counting error" if r is recommended else "")
                        + ("  ↳ fine-tuned" if r.parent else "")
                        + {"current": "", "v1": "  (earlier test split)",
                           "subset": f"  ({r.metrics.n_images:,}-crop data-folder split)"}[
                            r.metrics.split]))
    if others:
        options.append(("other", "Another version or checkpoint…",
                        "any trained version, tested or not, and its best or last weights"))
    picked = ui.choose("Which trained version should count the birds?", options,
                       default=recommended)
    if picked is None:
        return None
    if picked == "other":
        return common.pick_version("count with")
    best = next((c for c in picked.checkpoints() if c.kind == "best"), None)
    return picked, best


def start(survey: surveys.Survey | None = None, run: catalog.Run | None = None,
          checkpoint: catalog.Checkpoint | None = None) -> None:
    ui.title("Detect birds in new imagery", "New count")
    if survey is None:
        found = surveys.all_surveys()
        if not found:
            ui.warn("No surveys yet — import the photographs first.")
            if ui.confirm("Import a survey now?", default=True):
                from workbench.screens import data
                data.import_survey()
            return
        survey = ui.choose("Which survey?", [(s, s.name, f"{len(s.photos)} photos · "
                                             + (f"{s.labelled} labelled · " if s.labelled else "")
                                             + ago(s.created)) for s in found])
        if survey is None:
            return
    if run is None:
        ui.title("Detect birds in new imagery", survey.name)
        got = pick_detector(survey)
        if got is None:
            return
        run, checkpoint = got
    if checkpoint is None:
        checkpoint = next((c for c in run.checkpoints() if c.kind == "best"), None)
    if checkpoint is None:
        ui.error(f"{run.name} has no checkpoint to load.")
        ui.pause()
        return
    model = run.model
    if model.owl and not catalog.owl_env_ready():
        ui.error(f"{model.label} needs the OWL environment ({rel(config.MDO_PYTHON)}).")
        ui.pause()
        return

    ui.title("Detect birds in new imagery", survey.name, run.name)
    # t* belongs to the weights it was measured on: the tested checkpoint's own, else the
    # best checkpoint's as a starting point, else a plain default.
    own = run.last_metrics if checkpoint.kind == "last" else run.metrics
    t_star = own.t_star if own else None
    if t_star is not None:
        ui.note(f"Counting threshold: t* = {t_star:.3f} of the {checkpoint.kind} checkpoint "
                "(it minimised counting error on the test split).  Detections down to "
                f"{config.CONF_THRESHOLD} are kept, so it can be changed later.")
        default = round(t_star, 3)
    elif checkpoint.kind == "last" and run.metrics and run.metrics.t_star is not None:
        ui.warn("The last checkpoint has not been tested, so it has no t* of its own; the "
                f"best checkpoint's t* ({run.metrics.t_star:.3f}) is offered as a start.")
        default = round(run.metrics.t_star, 3)
    else:
        ui.warn("These weights have not been tested, so there is no t*; choose a threshold.")
        default = 0.25
    threshold = ui.ask_float("Threshold:", default=default, lo=config.CONF_THRESHOLD, hi=0.99)
    if threshold is None:
        return
    if t_star is not None and abs(threshold - round(t_star, 3)) < 1e-9:
        threshold, source = t_star, f"t* of the {checkpoint.kind} checkpoint (minimises test-set MAE)"
    else:
        source = "chosen when the count was started"
    gpu = common.pick_gpu("detection", allow_cpu=True)
    if gpu is None:
        return
    ui.console.print()
    ui.show(ui.kv([("Survey", f"{survey.name} — {len(survey.photos):,} photographs"),
                   ("Model", f"{model.label} version {run.run_id}"),
                   ("Weights", f"{checkpoint.kind}: {checkpoint.what}"),
                   ("Threshold", f"{threshold:.3f} ({source})"),
                   ("Estimated time", "≈ " + human_duration(estimate_seconds(model, survey))
                    + (" (much longer on the CPU)" if gpu == "cpu" else ""))]))
    if not ui.confirm("Start counting?", default=True):
        return
    result = detections.new(survey, model, run, threshold, source, checkpoint=checkpoint)
    job = common.launch(f"Count birds: {survey.name} with {model.label}", "detect",
                        python_cmd("-m", "workbench.detect", result.id), gpu=gpu,
                        links={"detection": result.id, "survey": survey.name,
                               "model": model.key, "run": run.run_id,
                               "checkpoint": checkpoint.kind},
                        offer_follow=False)
    result.meta["job"] = job.id
    result.save_meta()
    if ui.confirm("Follow its progress now?", default=True):
        common.follow(job)
    fresh = detections.load(result.id)
    if fresh and fresh.state() == "done" and ui.confirm("Review the detections now?",
                                                        default=True):
        from workbench.screens import review
        review.open_session(result.id)


def result_menu(result: detections.Result) -> None:
    while True:
        result = detections.load(result.id) or result
        ui.title("Detect birds in new imagery", result.title)
        state = result.state()
        done, total = result.review_progress()
        photos = result.photos(reload=True)
        review = result.review()
        final = sum(review.counts(p).final for p in photos)
        model_count = sum(review.counts(p).model for p in photos)
        ui.show(ui.kv([
            ("State", common.styled(state)),
            ("Photographs", f"{len(photos)}/{result.meta.get('photos_total')} done"),
            ("Birds", f"{model_count:,} by the model at threshold {review.threshold:.3f}"
                      + (f" → {final:,} after review" if review.touched else "")),
            ("Reviewed", f"{done:,}/{total:,} cells"),
            ("Folder", rel(result.dir)),
        ]))
        options = []
        job = result.job()
        if job and state == "running":
            options.append(("follow", "Follow the job", ""))
        if photos:
            options.append(("review", "Review & correct", f"{done}/{total} cells reviewed"))
            options.append(("report", "Export the count report", "Markdown, HTML, CSV"))
        if state in ("failed", "stopped", "cancelled"):
            options.append(("resume", "Continue counting", "skips photographs already done"))
        if state != "running":
            options.append(("delete", "Delete this result", "detections and review"))
        picked = ui.choose("Next:", options)
        if picked is None:
            return
        if picked == "follow":
            common.follow(job)
        elif picked == "review":
            from workbench.screens import review as review_screen
            review_screen.open_session(result.id)
        elif picked == "report":
            from workbench.screens import export
            export.survey_report(result)
        elif picked == "resume":
            gpu = common.pick_gpu("detection", allow_cpu=True)
            if gpu:
                new_job = common.launch(f"Count birds: {result.meta['survey']} (continued)", "detect",
                                        python_cmd("-m", "workbench.detect", result.id), gpu=gpu,
                                        links={"detection": result.id})
                result.meta["job"] = new_job.id
                result.save_meta()
        elif picked == "delete":
            if ui.confirm("Delete these detections and any review of them?", default=False):
                import shutil
                shutil.rmtree(result.dir, ignore_errors=True)
                return
