"""
Where the workbench keeps what it writes.

Everything the workbench records for itself — job records and logs, imported surveys,
detection results, review decisions, settings — lives under `output/workbench/`.  That
folder ignores itself in git (a `*` .gitignore inside it), because `output/` is the
versioned results tree and none of this is a result: a job log or a half-finished review
committed by accident would be noise at best.

Reports are the exception.  They are deliverables, so they go to `reports/` at the
repository root, where committing one is a deliberate act rather than a side effect.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE_DIR = os.path.join(ROOT, "output", "workbench")
JOBS_DIR = os.path.join(STATE_DIR, "jobs")
SURVEYS_DIR = os.path.join(STATE_DIR, "surveys")
DETECTIONS_DIR = os.path.join(STATE_DIR, "detections")
MANIFESTS_DIR = os.path.join(STATE_DIR, "manifests")
TMP_DIR = os.path.join(STATE_DIR, "tmp")
SETTINGS_FILE = os.path.join(STATE_DIR, "settings.json")
ERROR_LOG = os.path.join(STATE_DIR, "errors.log")

REPORTS_DIR = os.path.join(ROOT, "reports")

_GITIGNORE = ("# Workbench state (jobs, logs, surveys, detections, reviews) is local to\n"
              "# this machine and never part of the versioned results tree.\n*\n")


def ensure_state_dir() -> str:
    """Create output/workbench/ (and its self-ignoring .gitignore) if missing."""
    os.makedirs(STATE_DIR, exist_ok=True)
    marker = os.path.join(STATE_DIR, ".gitignore")
    if not os.path.exists(marker):
        with open(marker, "w") as f:
            f.write(_GITIGNORE)
    return STATE_DIR


def rel(path: str | None) -> str:
    """A path as shown to the user: relative to the repository when it is inside it."""
    if not path:
        return ""
    path = os.path.abspath(path)
    if path == ROOT or path.startswith(ROOT + os.sep):
        return os.path.relpath(path, ROOT)
    return path
