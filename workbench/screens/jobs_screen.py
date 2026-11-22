"""Jobs & logs: everything started from the workbench, and pipeline processes started
elsewhere (a keep-alive training script, a command typed in another terminal)."""

from __future__ import annotations

import os
import subprocess

from rich.panel import Panel
from rich.text import Text

from workbench import catalog, imaging, jobs, ui
from workbench.paths import rel
from workbench.screens import common
from workbench.util import ago, human_duration


def created_run(job: jobs.Job) -> catalog.Run | None:
    """The run folder a training job made (the model's first run started after it)."""
    key = job.links.get("model")
    if job.kind != "train" or not key:
        return None
    if job.links.get("run"):
        return catalog.load_run(catalog.model(key), job.links["run"])
    model = catalog.model(key)
    if not os.path.isdir(model.ckpt_dir):
        return None
    for entry in sorted(os.scandir(model.ckpt_dir), key=lambda e: e.name):
        if catalog.RUN_ID_RE.match(entry.name) and \
                catalog.run_started(entry.name) >= job.created - 5:
            return catalog.load_run(model, entry.name)
    return None


def run() -> None:
    while True:
        ui.title("Jobs & logs")
        everything = jobs.all_jobs()
        external = [p for p in catalog.pipeline_processes() if not p.workbench_job]

        if everything:
            rows = []
            for i, j in enumerate(everything[:25], 1):
                rows.append([str(i), j.title, common.styled(j.status()),
                             human_duration(j.duration()), ago(j.created),
                             (j.last_line() or "")[:60]])
            ui.show(ui.table(["#", "Job", "Status", ("Time", "right"), "Started",
                              "Latest output"], rows))
        else:
            ui.dim("No jobs started from the workbench yet.")
            ui.console.print()

        if external:
            rows = [[str(p.pid), p.describe(), human_duration(p.elapsed),
                     p.command[:70] + ("…" if len(p.command) > 70 else "")]
                    for p in external]
            ui.show(ui.table([("PID", "right"), "Started outside the workbench",
                              ("Running for", "right"), "Command"], rows))

        options = [(j, j.title, f"{j.status()} · {ago(j.created)}") for j in everything[:25]]
        finished = [j for j in everything if j.status() != "running"]
        if finished:
            options.append(("clean", f"Remove {len(finished)} finished job record(s)",
                            "deletes their logs"))
        options.append(("refresh", "Refresh", ""))
        picked = ui.choose("Choose a job:", options)
        if picked is None:
            return
        if picked == "refresh":
            continue
        if picked == "clean":
            if ui.confirm(f"Delete the records and logs of {len(finished)} finished job(s)?",
                          default=False):
                for j in finished:
                    jobs.forget(j)
            continue
        job_menu(picked)


def job_menu(job: jobs.Job) -> None:
    while True:
        ui.title("Jobs & logs", job.title)
        status = job.status()
        pairs = [("Status", common.job_summary_line(job)),
                 ("Command", job.command_line()),
                 ("Log", rel(job.log_path))]
        if job.gpu is not None:
            pairs.insert(1, ("Runs on", "CPU" if job.gpu == "cpu" else f"GPU {job.gpu}"))
        run_made = created_run(job)
        if run_made:
            pairs.append(("Run folder", rel(run_made.path)))
        if status in ("failed", "stopped"):
            pairs.append(("Exit code", job.exit_code() if status == "failed"
                          else "none — the process ended without reporting "
                               "(killed, or the machine restarted)"))
        ui.show(ui.kv(pairs))
        ui.console.print()
        body = Text(no_wrap=True, overflow="ellipsis")
        for line in job.tail(max(5, ui.console.size.height - 22)):
            body.append_text(Text.from_ansi(line))
            body.append("\n")
        ui.show(Panel(body, title="[bold]latest output[/]", title_align="left",
                      border_style="grey50"))

        options = []
        if status == "running":
            options.append(("follow", "Follow the log live", "refreshes every second"))
        options.append(("page", "Page through the whole log", "less; q to return"))
        if imaging.viewer_available():
            options.append(("open", "Open the log in VS Code / a viewer", ""))
        if run_made and status == "succeeded":
            options.append(("inspect", "Inspect the run this job made", rel(run_made.path)))
        if job.links.get("detection") and status == "succeeded":
            options.append(("review", "Review these detections", job.links["detection"]))
        if status == "running":
            options.append(("cancel", "Cancel this job", "stops it and everything it started"))
        else:
            options.append(("again", "Run the same command again", ""))
            options.append(("forget", "Remove this job's record and log", ""))

        picked = ui.choose("What next?", options)
        if picked is None:
            return
        if picked == "follow":
            common.follow(job)
        elif picked == "page":
            subprocess.run(["less", "-R", "+G", job.log_path])
        elif picked == "open":
            ui.note(imaging.open_external(job.log_path))
            ui.pause()
        elif picked == "inspect":
            from workbench.screens import evaluate
            evaluate.inspect(run_made)
        elif picked == "review":
            from workbench.screens import review
            review.open_session(job.links["detection"])
        elif picked == "cancel":
            if ui.confirm("Cancel this job? Work since its last checkpoint is lost.",
                          default=False):
                with ui.console.status("Stopping…"):
                    jobs.cancel(job)
                ui.ok("Cancelled.")
                ui.pause()
        elif picked == "again":
            if ui.confirm(f"Start again: {job.command_line()}?", default=True):
                common.launch(job.title, job.kind, job.cmd, gpu=job.gpu, links=job.links,
                              env=job.env)
                return
        elif picked == "forget":
            if ui.confirm("Delete this job's record and log?", default=False):
                jobs.forget(job)
                return
