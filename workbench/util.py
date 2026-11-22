"""Small helpers shared by every workbench module: JSON on disk, names, and units."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime


def timestamp(t: float | None = None) -> str:
    """The project's run-folder timestamp format: 2026-09-26_23-45-12."""
    return datetime.fromtimestamp(t if t is not None else time.time()).strftime(
        "%Y-%m-%d_%H-%M-%S")


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: str, obj, indent: int | None = 2) -> str:
    """Write JSON atomically: a crash mid-write leaves the previous file, never half of one.

    Review decisions are saved on every keystroke, so a torn file here would lose a
    reviewer's work rather than one line of it.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def slug(text: str, fallback: str = "untitled") -> str:
    """A file-system-safe name: letters, digits, '-', '_' and '.' only."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    return s or fallback


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def ago(t: float | None) -> str:
    if not t:
        return "—"
    return human_duration(time.time() - t) + " ago"


def pct(x: float | None, digits: int = 1) -> str:
    """A 0-1 fraction as a percentage, or a dash when the number does not exist."""
    return "—" if x is None else f"{100.0 * x:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"
