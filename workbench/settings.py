"""
The workbench's own preferences, kept in output/workbench/settings.json.

The one that matters is `data_folder`.  The pipeline chooses its data by folder
(`DATASET_FOLDER` in data_prep/config.py, or `WATERFOWL_DATASET` for one command).  The
workbench never edits config.py: it passes `WATERFOWL_DATASET=<folder>` to every job it
launches when the chosen folder differs from the configured default, so a choice made
here applies to all eight models without touching a code file — and a command copied from
the jobs screen still says exactly which data it read.
"""

from __future__ import annotations

import os

from workbench import paths
from workbench.util import read_json, write_json

DEFAULTS = {
    "data_folder": None,   # None = whatever data_prep/config.py names (DATASET_FOLDER)
    "gpu": "auto",         # "auto" (least busy at launch), a GPU index, or "cpu"
    "viewer": "auto",      # how to open images: "auto", "code", "xdg-open" or "none"
}


def load() -> dict:
    stored = read_json(paths.SETTINGS_FILE, {}) or {}
    return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}


def get(key: str):
    return load()[key]


def put(key: str, value) -> None:
    current = load()
    current[key] = value
    paths.ensure_state_dir()
    write_json(paths.SETTINGS_FILE, current)


def configured_data_folder() -> str:
    """The folder data_prep/config.py names, e.g. "dataset_full"."""
    from data_prep import config
    return config.DATASET_FOLDER


def data_folder() -> str:
    """The data folder every job launched from the workbench reads.

    A folder chosen in the workbench wins; otherwise a `WATERFOWL_DATASET` the workbench
    itself was started under (jobs inherit it); otherwise config.py's default.
    """
    return (get("data_folder") or os.environ.get("WATERFOWL_DATASET", "").strip()
            or configured_data_folder())


def data_root() -> str:
    folder = data_folder()
    return folder if os.path.isabs(folder) else os.path.join(paths.ROOT, folder)


def job_env() -> dict:
    """Environment overrides every launched job receives (the data folder, if chosen)."""
    folder = get("data_folder")
    if folder and folder != configured_data_folder():
        return {"WATERFOWL_DATASET": folder}
    return {}
