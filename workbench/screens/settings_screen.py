"""
Settings & health check: the data folder, default GPU and image viewer; a check of
everything the pipeline needs; and a desktop launcher for machines with a desktop.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

from data_prep import config
from workbench import __version__, catalog, imaging, paths, settings, ui
from workbench.paths import ROOT, rel
from workbench.util import human_bytes

DESKTOP_FILE = os.path.expanduser("~/.local/share/applications/waterfowl-workbench.desktop")


def run() -> None:
    while True:
        ui.title("Settings & health check")
        viewer = imaging.viewer_available()
        options = [
            ("folder", "Active data folder", settings.data_folder()),
            ("gpu", "Preferred GPU", "idlest at launch" if settings.get("gpu") == "auto"
             else f"GPU {settings.get('gpu')}"),
            ("viewer", "Image viewer", f"{settings.get('viewer')} → "
             + {"code": "VS Code", "xdg-open": "desktop viewer", None: "none available"}[viewer]),
            ("doctor", "Health check", "environments, GPUs, weights, data, disk"),
            ("selftest", "Self-test", "checks the workbench's own logic; seconds, no GPU"),
            ("desktop", "Create a desktop launcher", "opens the workbench in a terminal window"),
            ("about", "About", f"version {__version__}"),
        ]
        picked = ui.choose("Change or check:", options)
        if picked is None:
            return
        if picked == "folder":
            from workbench.screens import data
            data.switch_folder()
        elif picked == "gpu":
            found = catalog.gpus()
            choice = ui.choose("Which GPU should be pre-selected for new jobs?",
                               [("auto", "The idlest one at launch", "recommended")]
                               + [(str(g.index), f"GPU {g.index}", g.describe()) for g in found],
                               default=settings.get("gpu"))
            if choice:
                settings.put("gpu", choice)
        elif picked == "viewer":
            choice = ui.choose("How should images and reports open?", [
                ("auto", "Automatic", "VS Code when inside it, else the desktop viewer"),
                ("code", "VS Code", "`code <file>` — works over Remote-SSH"),
                ("xdg-open", "Desktop viewer", "needs a graphical session"),
                ("none", "Never open", "just print the path"),
            ], default=settings.get("viewer"))
            if choice:
                settings.put("viewer", choice)
        elif picked == "doctor":
            doctor()
        elif picked == "selftest":
            ui.title("Settings & health check", "Self-test")
            with ui.console.status("Running python -m workbench.selftest…"):
                out = subprocess.run([sys.executable, "-m", "workbench.selftest"], cwd=ROOT,
                                     capture_output=True, text=True)
            ui.console.print((out.stdout + out.stderr).rstrip(), markup=False)
            ui.pause()
        elif picked == "desktop":
            desktop_launcher()
        elif picked == "about":
            about()


def _row(ok: bool | None, what: str, detail: str) -> list[str]:
    mark = {True: "[green]✓[/]", False: "[red]✗[/]", None: "[yellow]![/]"}[ok]
    return [mark, what, detail]


def doctor() -> None:
    ui.title("Settings & health check", "Health check")
    rows = []
    with ui.console.status("Checking…"):
        rows.append(_row(True, "Python", f"{sys.version.split()[0]} · {sys.executable}"))
        for module, used_by in (("torch", "all box models"), ("ultralytics", "YOLOv5/11/26"),
                                ("super_gradients", "YOLO-NAS"), ("detectron2", "Faster R-CNN"),
                                ("pycocotools", "mAP30"), ("cv2", "image I/O")):
            found = importlib.util.find_spec(module) is not None
            rows.append(_row(found, module, used_by if found else f"missing — needed by {used_by}"))
        gpus = catalog.gpus()
        rows.append(_row(bool(gpus), "GPUs", ", ".join(g.describe() for g in gpus)
                         if gpus else "none found (nvidia-smi)"))

        if os.path.exists(config.MDO_PYTHON):
            probe = subprocess.run(
                [config.MDO_PYTHON, "-c",
                 "import torch, animaloc; print(torch.__version__, torch.cuda.is_available())"],
                cwd=config.MDO_REPO_DIR, capture_output=True, text=True, timeout=180,
                env={**os.environ, "MDO_REPO_DIR": config.MDO_REPO_DIR})
            ok = probe.returncode == 0
            rows.append(_row(ok, "OWL environment",
                             f"torch {probe.stdout.split()[0]} · CUDA "
                             f"{probe.stdout.split()[1]}" if ok else
                             (probe.stderr.strip().splitlines() or ["failed"])[-1][:90]))
        else:
            rows.append(_row(False, "OWL environment", f"missing: {rel(config.MDO_PYTHON)}"))
        for key, spec in config.MDO_MODELS.items():
            w = spec.get("requires_weights")
            present = bool(w and os.path.exists(w))
            rows.append(_row(present or None, f"{key} start weights",
                             f"{os.path.basename(w)} ({human_bytes(os.path.getsize(w))})" if present
                             else f"{os.path.basename(w)} not downloaded (Train offers to fetch it)"))
        for key in ("yolov5", "yolo11", "yolo26"):
            w = config.BASELINE_MODELS[key]["weights"]
            rows.append(_row(os.path.exists(w) or None, f"{os.path.basename(w)}",
                             "COCO weights for baselines" if os.path.exists(w)
                             else "missing; Ultralytics downloads it on first use"))

        root = settings.data_root()
        cols = catalog.collections(root)
        rows.append(_row(bool(cols), "Data folder", f"{settings.data_folder()} · {len(cols)} "
                         f"collections · {sum(c.photos for c in cols):,} photos"
                         if cols else f"{rel(root)} has no collections"))
        split = catalog.splits_status()
        rows.append(_row(split.ready, "COCO splits", rel(config.COMMITTED_CROPS_JSON_DIR)
                         if split.ready else "not prepared (Import imagery & prepare data)"))
        rows.append(_row(os.path.isdir(config.CROPS_IMG_DIR) or None, "Crop images",
                         rel(config.CROPS_IMG_DIR)))
        usage = shutil.disk_usage(config.OUTPUT_DIR)
        rows.append(_row(usage.free > 20e9 or None, "Free disk",
                         f"{human_bytes(usage.free)} free of {human_bytes(usage.total)} "
                         f"under {rel(config.OUTPUT_DIR)}/"))
        paths.ensure_state_dir()
        rows.append(_row(os.access(paths.STATE_DIR, os.W_OK), "Workbench state",
                         rel(paths.STATE_DIR)))
        cols_, rows_ = ui.console.size
        colour = os.environ.get("COLORTERM", "")
        rows.append(_row(cols_ >= 110 and rows_ >= 32 or None, "Terminal",
                         f"{cols_}×{rows_}, colour: {colour or 'not truecolor (256 colours)'}"))
        viewer = imaging.viewer_available()
        rows.append(_row(viewer is not None or None, "Image viewer",
                         {"code": "VS Code (code)", "xdg-open": "desktop (xdg-open)"}.get(
                             viewer, "none — images are saved and their path printed")))
    ui.show(ui.table(["", "Check", "Result"], rows))
    ui.pause()


def desktop_launcher() -> None:
    ui.title("Settings & health check", "Desktop launcher")
    ui.dim("On a machine with a desktop (the lab workstation), this adds 'Waterfowl "
           "Workbench' to the applications menu: clicking it opens a terminal window with "
           "the workbench already running.")
    ui.console.print()
    launcher = os.path.join(ROOT, "waterfowl_detection.sh")
    entry = ("[Desktop Entry]\nType=Application\nName=Waterfowl Workbench\n"
             "Comment=Import imagery, train, evaluate, review and report — waterfowl detection\n"
             f"Exec={launcher}\nPath={ROOT}\nTerminal=true\nIcon=utilities-terminal\n"
             "Categories=Science;Education;\n")
    ui.console.print(entry, markup=False)
    if not ui.confirm(f"Write {DESKTOP_FILE}?", default=True):
        return
    os.makedirs(os.path.dirname(DESKTOP_FILE), exist_ok=True)
    with open(DESKTOP_FILE, "w") as f:
        f.write(entry)
    os.chmod(DESKTOP_FILE, 0o755)
    ui.ok(f"Wrote {DESKTOP_FILE}")
    desktop = os.path.expanduser("~/Desktop")
    if os.path.isdir(desktop) and ui.confirm("Put a copy on the Desktop too?", default=False):
        target = os.path.join(desktop, os.path.basename(DESKTOP_FILE))
        shutil.copy2(DESKTOP_FILE, target)
        ui.ok(f"Wrote {target} (some desktops ask you to 'Allow launching' it once)")
    ui.pause()


def about() -> None:
    ui.title("Settings & health check", "About")
    ui.show(ui.kv([
        ("Version", __version__),
        ("Launch", "./waterfowl_detection.sh   (or: python -m workbench)"),
        ("Jump to a screen", "./waterfowl_detection.sh data | train | evaluate | detect | review | export | "
                             "jobs | settings"),
        ("State", f"{rel(paths.STATE_DIR)}/ — jobs, logs, surveys, detections, reviews "
                  "(ignored by git)"),
        ("Reports", f"{rel(paths.REPORTS_DIR)}/"),
        ("Manual", "WATERFOWL_DETECTION_MANUAL.md"),
        ("Design notes", "workbench/README.md"),
    ]))
    ui.pause()
