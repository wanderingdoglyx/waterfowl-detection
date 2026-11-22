"""Pieces several screens share: run tables, the GPU picker, and following a job's log."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from contextlib import contextmanager

from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from workbench import catalog, jobs, settings, ui
from workbench.paths import rel
from workbench.util import ago, human_duration, num, pct

STATUS_STYLE = {
    "evaluated": "green", "trained": "cyan", "running": "yellow", "no checkpoint": "red",
    "succeeded": "green", "failed": "red", "cancelled": "grey50", "stopped": "red",
    "done": "green", "queued": "grey50",
}


def styled(status: str) -> str:
    return f"[{STATUS_STYLE.get(status, 'white')}]{status}[/]"


def split_label(m: catalog.Metrics | None) -> str:
    if not m or not m.n_images:
        return "—"
    return f"{m.n_images:,}" + {"current": "", "v1": " (v1)", "subset": " (subset)"}[m.split]


def runs_table(runs: list[catalog.Run], title: str | None = None, numbered: bool = False,
               caption: str | None = None):
    cols = (["#"] if numbered else []) + [
        "Model", "Run", "Status", ("Point AP", "right"), ("F1", "right"), ("MAE", "right"),
        ("RMSE", "right"), ("mAP30", "right"), ("Test crops", "right"), "Data"]
    rows = []
    for i, r in enumerate(runs, 1):
        m = r.metrics
        map30 = "—"
        if m and m.map30 is not None:
            map30 = f"{m.map30:.2f}%" + ("*" if m.map30_pseudo else "")
        rows.append(([str(i)] if numbered else []) + [
            r.model.label, r.run_id + (" ↳" if r.parent else ""), styled(r.status),
            pct(m.ap if m else None), pct(m.f1 if m else None),
            num(m.mae if m else None), num(m.rmse if m else None), map30,
            split_label(m), r.data_folder or "—"])
    return ui.table(cols, rows, title=title, caption=caption)


RUNS_CAPTION = ("Point metrics: OWL-paper protocol (τ = 40 px); P/R/F1 and counts at each "
                "run's own t*.  * pseudo-box mAP30 (points in 28 px boxes) — not comparable "
                "with box mAP30.  (v1) = scored on the earlier 25,708-crop test split, not "
                "comparable with the current 28,588-crop split; (subset) = scored on a data "
                "folder's own test photographs.  ↳ fine-tuned from an earlier version.  "
                "(last) = the run's last checkpoint rather than its best.")


def scored_entries(runs: list[catalog.Run]) -> list[catalog.Run]:
    """Every scored set of weights: each run with scores, plus a '(last)' entry when its
    last checkpoint has been tested too (--checkpoint last)."""
    from dataclasses import replace
    out = []
    for r in runs:
        if r.metrics:
            out.append(r)
        if r.last_metrics:
            out.append(replace(r, run_id=f"{r.run_id} (last)", metrics=r.last_metrics,
                               last_metrics=None))
    return out


# ── Choosing a trained version ────────────────────────────────────────────────

def version_line(r: catalog.Run) -> str:
    """One line describing a trained version, for menus."""
    bits = [r.status]
    if r.metrics:
        bits.append(f"AP {pct(r.metrics.ap)} · MAE {num(r.metrics.mae)}")
        if r.metrics.split == "v1":
            bits.append("earlier test split")
        elif r.metrics.split == "subset":
            bits.append(f"tested on a {r.metrics.n_images:,}-crop data-folder split")
    if r.parent:
        bits.append(f"↳ fine-tuned from {r.parent[1]}")
    if r.data_folder and r.data_folder != settings.configured_data_folder():
        bits.append(f"data {r.data_folder}")
    return " · ".join(bits)


def versions_table(runs: list[catalog.Run]):
    rows = []
    for r in runs:
        m = r.metrics
        rows.append([r.run_id, r.started_from(), styled(r.status), pct(m.ap if m else None),
                     num(m.mae if m else None), split_label(m), r.data_folder or "—",
                     " + ".join(c.kind for c in r.checkpoints())])
    return ui.table(["Version (run)", "Started from", "Status", ("Point AP", "right"),
                     ("MAE", "right"), ("Test crops", "right"), "Data", "Checkpoints"], rows)


def pick_version(verb: str, model: catalog.Model | None = None, *,
                 include_active: bool = False, with_checkpoint: bool = True):
    """Model → trained version (a run) → which of its checkpoints.

    Returns (run, checkpoint) — checkpoint None when `with_checkpoint` is False — or None
    when the user backs out.  Versions still training are offered only with
    `include_active`, and then only their last checkpoint: their best one is still
    changing.
    """
    procs = catalog.pipeline_processes()
    usable = [r for r in catalog.runs(procs=procs)
              if r.checkpoints() and (include_active or not r.active)]
    if model is None:
        options = []
        for m in catalog.MODELS:
            mine = [r for r in usable if r.model is m]
            if mine:
                options.append((m, m.label, f"{len(mine)} trained version(s) · newest "
                                            f"{mine[0].run_id}"))
        if not options:
            ui.warn("No trained versions yet — train a model first.")
            ui.pause()
            return None
        model = ui.choose(f"Which model do you want to {verb}?", options)
        if model is None:
            return None
    runs = [r for r in usable if r.model is model]
    if not runs:
        ui.warn(f"{model.label} has no trained version with a checkpoint yet.")
        ui.pause()
        return None
    if ui.console.size.height >= 2 * len(runs) + 18:
        ui.show(versions_table(runs))
    run = ui.choose(f"Which {model.label} version do you want to {verb}?",
                    [(r, r.run_id, version_line(r)) for r in runs])
    if run is None:
        return None
    if not with_checkpoint:
        return run, None
    ckpts = run.checkpoints()
    if run.active:
        ckpts = [c for c in ckpts if c.kind == "last"] or ckpts
    if verb == "test" and not catalog.supports(run.model, "--checkpoint"):
        # an entry point without --checkpoint can only score its best weights
        ckpts = [c for c in ckpts if c.kind == "best"]
        if not ckpts:
            ui.warn(f"This checkout's {run.model.script} has no --checkpoint flag, so it can "
                    "only test a version's best checkpoint — and this one has none yet.")
            ui.pause()
            return None
    if len(ckpts) == 1:
        ui.dim(f"  Checkpoint: {ckpts[0].kind} — {ckpts[0].what}")
        return run, ckpts[0]
    ckpt = ui.choose("Which of its checkpoints?", [(c, c.kind, c.what) for c in ckpts],
                     default=ckpts[0])
    return (run, ckpt) if ckpt is not None else None


def pick_gpu(purpose: str, allow_cpu: bool = False) -> str | None:
    """Choose where a job runs.  Returns a GPU index as a string, "cpu", or None (back).

    Busy cards are shown as busy and never pre-selected: a second job on a card that is
    already training slows both and can run either out of memory.
    """
    found = catalog.gpus()
    if not found:
        ui.warn("No NVIDIA GPU found (nvidia-smi did not answer).")
        if allow_cpu and ui.confirm("Run on the CPU instead? (much slower)", default=False):
            return "cpu"
        return None

    preference = settings.get("gpu")
    suggested = catalog.suggest_gpu(found)
    default = next((g for g in found if str(g.index) == str(preference)), suggested)
    options = []
    for g in found:
        bar = "█" * round(g.util / 10) + "░" * (10 - round(g.util / 10))
        state = "busy" if g.busy else "idle"
        tag = "  (recommended)" if g is suggested and not g.busy else ""
        options.append((g, f"GPU {g.index}  {bar} {g.util:3d}%",
                        f"{g.used_mb / 1024:.1f}/{g.total_mb / 1024:.0f} GB used · {state}{tag}"))
    if allow_cpu:
        options.append(("cpu", "CPU only", "slow; only sensible for a handful of photos"))
    picked = ui.choose(f"Where should the {purpose} run?", options, default=default)
    if picked is None:
        return None
    if picked == "cpu":
        return "cpu"
    if picked.busy:
        ui.warn(f"GPU {picked.index} is already busy ({picked.util}% utilised, "
                f"{picked.used_mb / 1024:.1f} GB in use). Both jobs will slow down, and "
                "either may run out of memory.")
        if not ui.confirm("Use it anyway?", default=False):
            return None
    return str(picked.index)


# ── Launching and following jobs ──────────────────────────────────────────────

def launch(title: str, kind: str, cmd: list[str], gpu: str | None = None,
           links: dict | None = None, env: dict | None = None,
           offer_follow: bool = True) -> jobs.Job:
    job = jobs.launch(title, kind, cmd, env={**settings.job_env(), **(env or {})},
                      gpu=gpu, links=links)
    ui.console.print()
    ui.ok(f"Started: {title}")
    ui.dim(f"  job {job.id} · log {rel(job.log_path)}")
    ui.dim("  It runs in the background: you can leave this screen or quit the workbench, "
           "and it keeps going. Find it again under Jobs & logs.")
    if offer_follow and ui.confirm("Follow its log now?", default=True):
        follow(job)
    return job


@contextmanager
def _keys_nonblocking():
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _key_pressed(timeout: float) -> bool:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        os.read(sys.stdin.fileno(), 64)
        return True
    return False


def follow(job: jobs.Job) -> None:
    """Show the log's newest lines, refreshed every second, until a key is pressed.

    Leaving this view never stops the job.
    """
    if not ui.interactive():
        print("\n".join(job.tail(40)))
        return

    def frame() -> Panel:
        height = max(8, ui.console.size.height - 6)
        body = Text(no_wrap=True, overflow="ellipsis")
        for line in job.tail(height):
            body.append_text(Text.from_ansi(line))
            body.append("\n")
        status = job.status()
        return Panel(body, title=f"[bold]{job.title}[/]  {styled(status)}  "
                                 f"{human_duration(job.duration())}",
                     subtitle="[dim]any key: back to the menu — the job keeps running[/]",
                     title_align="left", border_style="grey50")

    ui.console.clear()
    try:
        with _keys_nonblocking(), Live(frame(), console=ui.console, auto_refresh=False,
                                       screen=True) as live:
            while True:
                if _key_pressed(1.0):
                    break
                live.update(frame(), refresh=True)
    except KeyboardInterrupt:
        pass


def job_summary_line(job: jobs.Job) -> str:
    status = job.status()
    when = f"started {ago(job.created)}"
    if status != "running":
        when = f"ended {ago(job.finished_at())}"
    return f"{styled(status)} · {human_duration(job.duration())} · {when}"
