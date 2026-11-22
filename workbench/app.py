"""
The home screen and the loop around it.

The home screen answers "where am I" before asking "what next": which data the jobs will
read, whether the crops exist, how many runs are trained and evaluated, what the GPUs are
doing and which jobs are running.  Everything under it is one of eight menus, each a
module in workbench/screens/.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
import traceback

from rich.panel import Panel
from rich.table import Table

from workbench import __version__, catalog, jobs, paths, settings, ui
from workbench.util import human_duration, pct

MENU = (
    # key        label                               hint
    ("data",     "Import imagery & prepare data",   "survey photos, training collections, crops"),
    ("train",    "Train a model",                   "any of the 8 detectors, in the background"),
    ("evaluate", "Evaluate & compare",              "leaderboard, test-set scoring, baselines, analysis"),
    ("detect",   "Detect birds in new imagery",     "count birds in imported survey photos"),
    ("review",   "Review & correct detections",     "confirm, reject and add birds; fix the counts"),
    ("export",   "Export reports",                  "count reports, corrected labels, comparisons"),
    ("jobs",     "Jobs & logs",                     ""),
    ("settings", "Settings & health check",         "data folder, GPU, viewer, environment check"),
)


def _screen(key: str):
    from workbench.screens import (data, detect, evaluate, export, jobs_screen, review,
                                   settings_screen, train)
    return {"data": data.run, "train": train.run, "evaluate": evaluate.run,
            "detect": detect.run, "review": review.run, "export": export.run,
            "jobs": jobs_screen.run, "settings": settings_screen.run}[key]


def guarded(fn, *args) -> None:
    """Run one screen; an error is shown and logged instead of ending the program."""
    try:
        fn(*args)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 — the whole point is to catch anything
        paths.ensure_state_dir()
        with open(paths.ERROR_LOG, "a") as f:
            f.write(f"\n=== {time.strftime('%F %T')}\n{traceback.format_exc()}")
        ui.console.print()
        ui.error(f"{type(exc).__name__}: {exc}")
        ui.dim(f"  The full traceback is in {paths.rel(paths.ERROR_LOG)}.")
        ui.pause()


def _gpu_cell(g: catalog.GPU) -> str:
    filled = round(g.util / 10)
    colour = "yellow" if g.busy else "green"
    return (f"GPU {g.index} [{colour}]{'█' * filled}[/][grey30]{'█' * (10 - filled)}[/] "
            f"{g.util:3d}% · {g.used_mb / 1024:.1f}/{g.total_mb / 1024:.0f} GB")


def status_panel() -> Panel:
    from workbench import detections, surveys

    procs = catalog.pipeline_processes()
    all_runs = catalog.runs(procs=procs)
    scored = catalog.evaluated(all_runs)
    current = [r for r in scored if r.metrics.split == "current"]
    top = catalog.best(current or scored, "ap")
    folder = settings.data_folder()
    cols = catalog.collections(settings.data_root())
    split = catalog.splits_status()
    active_jobs = jobs.running()
    external = [p for p in procs if not p.workbench_job]

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()

    data_line = (f"{folder} · {len(cols)} collections · "
                 f"{sum(c.photos for c in cols):,} photographs")
    if folder != settings.configured_data_folder():
        data_line += "  [yellow](chosen in the workbench)[/]"
    grid.add_row("Data", data_line)

    crops = ("[green]ready[/] · train/val/test COCO splits" if split.ready
             else "[red]not prepared[/] — Import imagery & prepare data → Prepare crops")
    if split.experiments:
        crops += f" · {len(split.experiments)} experiment selection(s)"
    grid.add_row("Crops", crops)

    running_runs = [r for r in all_runs if r.active]
    models = f"{len(all_runs)} runs · {len(scored)} evaluated"
    if running_runs:
        models += f" · [yellow]{len(running_runs)} training[/]"
    if top:
        models += f" · best point AP {pct(top.metrics.ap)} ({top.model.label} {top.run_id[:10]})"
    grid.add_row("Models", models)

    found = catalog.gpus()
    grid.add_row("GPUs", "   ".join(_gpu_cell(g) for g in found) if found
                 else "[dim]none detected[/]")

    parts = [f"[yellow]{len(active_jobs)} running[/]" if active_jobs else "none running"]
    if external:
        parts.append(f"{len(external)} started elsewhere: " + ", ".join(
            f"{p.describe()} ({human_duration(p.elapsed)})" for p in external[:2]))
    grid.add_row("Jobs", " · ".join(parts))

    n_surveys = len(surveys.all_surveys())
    results = detections.all_results()
    reviewed = sum(1 for r in results if r.review_progress()[0])
    grid.add_row("Surveys", f"{n_surveys} imported · {len(results)} detection result(s)"
                            + (f" · {reviewed} under review" if reviewed else ""))

    host = socket.gethostname().split(".")[0]
    return Panel(grid, title=f"[bold]Waterfowl Workbench[/] [dim]v{__version__}[/]",
                 subtitle=f"[dim]{host} · {time.strftime('%Y-%m-%d %H:%M')}[/]",
                 title_align="left", subtitle_align="right", border_style="cyan",
                 padding=(0, 1))


def home() -> None:
    while True:
        ui.console.clear()
        with ui.console.status("Reading the project…"):
            panel = status_panel()
        ui.show(panel)
        ui.console.print()
        n_running = len(jobs.running())
        options = []
        for key, label, hint in MENU:
            if key == "jobs":
                hint = f"{n_running} running" if n_running else "logs of every job"
            options.append((key, label, hint))
        picked = ui.choose("What would you like to do?", options, back_label="Quit")
        if picked is None:
            ui.console.clear()
            if jobs.running():
                ui.note("Jobs keep running in the background; reopen the workbench to "
                        "check on them.")
            return
        guarded(_screen(picked))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="waterfowl_detection.sh",
        description="Terminal interface for the waterfowl detection pipeline.")
    parser.add_argument("screen", nargs="?", choices=[k for k, _, _ in MENU],
                        help="open this screen directly, then return to the main menu")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    os.chdir(paths.ROOT)
    paths.ensure_state_dir()
    if not ui.interactive():
        print("(not a terminal: menus fall back to numbered prompts)")
    if args.screen:
        guarded(_screen(args.screen))
    try:
        home()
    except KeyboardInterrupt:
        ui.console.clear()
    return 0
