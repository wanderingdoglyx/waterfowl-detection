"""
Read the model manifest that decides which runs an analysis covers.

The manifest is a two-column list of ``model, run_timestamp``:

    Fasterrcnn, 2026-06-27_16-56-51
    owl_c, 2026-07-21_22-59-50
    ...

``.docx`` is parsed with the standard library only (a .docx is a zip holding
``word/document.xml``), so no python-docx / pandoc dependency is needed — the
analysis runs in the same `waterfowl` env as everything else.  Plain ``.txt``,
``.csv`` and ``.md`` manifests are accepted too, which makes a manifest easy to
hand-write or generate.
"""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# A run id is the checkpoint folder name written by every trainer: 2026-06-27_16-56-51
RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ManifestEntry:
    """One line of the manifest: which model, and which of its runs."""

    key: str        # model name as written by the user, e.g. "owl_c"
    run_id: str     # checkpoint folder name, e.g. "2026-07-21_22-59-50"
    line_no: int    # 1-based line number, for error messages


def _docx_lines(path: str) -> list[str]:
    """Extract paragraph text from a .docx without third-party libraries."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")

    root = ElementTree.fromstring(xml)
    lines: list[str] = []
    for para in root.iter(f"{_W}p"):
        # A paragraph's visible text is the concatenation of its <w:t> runs; a
        # <w:tab> or <w:br> between runs reads as whitespace.
        parts: list[str] = []
        for node in para.iter():
            if node.tag == f"{_W}t":
                parts.append(node.text or "")
            elif node.tag in (f"{_W}tab", f"{_W}br"):
                parts.append(" ")
        lines.append("".join(parts))
    return lines


def _plain_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def load(path: str) -> list[ManifestEntry]:
    """Parse a manifest into entries, preserving document order.

    Blank lines and ``#`` comments are skipped, as is a ``model,run`` header row.
    Raises ValueError with the offending line number on anything unparseable —
    a silently dropped model would quietly shrink the comparison table.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    raw = _docx_lines(path) if ext == ".docx" else _plain_lines(path)

    entries: list[ManifestEntry] = []
    seen: dict[str, int] = {}
    for i, line in enumerate(raw, start=1):
        text = line.strip().strip("|").strip()
        if not text or text.startswith("#"):
            continue
        # Markdown table separators (|---|---|) and similar rules.
        if set(text) <= set("-: |"):
            continue

        fields = [c.strip() for c in next(csv.reader(io.StringIO(text)))]
        fields = [c for c in fields if c]
        if len(fields) == 1:
            # Tolerate whitespace instead of a comma: "owl_c  2026-07-21_22-59-50".
            fields = fields[0].split()
        if len(fields) < 2:
            raise ValueError(
                f"{path}:{i}: expected 'model, run_timestamp', got {text!r}"
            )

        key, run_id = fields[0], fields[1]
        if key.lower() in ("model", "name") and not RUN_ID_RE.match(run_id):
            continue  # header row
        if not RUN_ID_RE.match(run_id):
            raise ValueError(
                f"{path}:{i}: {run_id!r} is not a run timestamp "
                f"(expected YYYY-MM-DD_HH-MM-SS)"
            )

        norm = key.lower()
        if norm in seen:
            raise ValueError(
                f"{path}:{i}: model {key!r} already listed on line {seen[norm]}; "
                "a manifest must name each model at most once"
            )
        seen[norm] = i
        entries.append(ManifestEntry(key=key, run_id=run_id, line_no=i))

    if not entries:
        raise ValueError(f"{path}: no model entries found")
    return entries
