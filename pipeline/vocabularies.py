"""
Canonical vocabularies, and the per-programme crosswalks that feed them.

A crosswalk is a small CSV shipped beside this module, one row per label a programme
actually writes:

    raw,canonical,notes

Nothing is normalised in code.  `WaterSubVeg` and `WaterSubVege` are two rows with the
same canonical term, not a special case in a function; `Dense` maps to an empty canonical
term with a note saying it describes bird density rather than habitat.  When AGFC's
labels arrive they become rows in `arkansas_habitat.csv` and nothing above this layer
changes — which is the whole point of Task 4.3.

The canonical habitat terms are the six environments the Arkansas proposal names.  They
are a starting proposal: the vocabulary is versioned, and a change means re-running the
condition tables, not editing the analysis.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

CROSSWALK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crosswalks")

# ── Canonical terms ───────────────────────────────────────────────────────────
HABITATS = (
    "forest",                 # bottomland hardwood, trees, wooded
    "cropland",               # rice, soybean, corn, harvested crop
    "moist_soil",             # managed moist-soil units
    "open_water",             # open water, reservoirs, ice
    "emergent_vegetation",    # emergent / submerged marsh vegetation
    "bare_ground",            # levees, bare fields, rock
)

WEATHER = ("sun", "cloud", "overcast", "rain", "fog")

SEXES = ("male", "female", "unknown")


@dataclass(frozen=True)
class Mapping:
    raw: str
    canonical: str | None
    notes: str | None = None


class Crosswalk:
    """raw label -> canonical term for one programme and one vocabulary."""

    def __init__(self, name: str, allowed: tuple[str, ...], mappings: list[Mapping]):
        self.name = name
        self.allowed = allowed
        self.mappings = mappings
        self._index = {m.raw.strip().lower(): m for m in mappings}
        for m in mappings:
            if m.canonical and m.canonical not in allowed:
                raise ValueError(f"{name}: '{m.canonical}' is not in the canonical vocabulary")

    def __contains__(self, raw: str) -> bool:
        return raw.strip().lower() in self._index

    def get(self, raw: str | None) -> tuple[str | None, str | None]:
        """(canonical term, note).  An unknown label returns (None, 'not in crosswalk')."""
        if raw is None or not str(raw).strip():
            return None, None
        m = self._index.get(str(raw).strip().lower())
        if m is None:
            return None, "not in crosswalk"
        return m.canonical or None, m.notes

    def unmapped(self, raws) -> list[str]:
        """Labels seen in the data that the crosswalk does not cover — the thing to review."""
        return sorted({str(r) for r in raws if r and str(r).strip() and str(r) not in self})


def load(source: str, vocabulary: str, allowed: tuple[str, ...]) -> Crosswalk:
    """Read `<source>_<vocabulary>.csv`; an absent file is an empty crosswalk, not an error."""
    path = os.path.join(CROSSWALK_DIR, f"{source}_{vocabulary}.csv")
    mappings: list[Mapping] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("raw", "").strip() or row["raw"].lstrip().startswith("#"):
                    continue
                mappings.append(Mapping(row["raw"].strip(),
                                        (row.get("canonical") or "").strip() or None,
                                        (row.get("notes") or "").strip() or None))
    return Crosswalk(f"{source}/{vocabulary}", allowed, mappings)


def habitat(source: str) -> Crosswalk:
    return load(source, "habitat", HABITATS)


def weather(source: str) -> Crosswalk:
    return load(source, "weather", WEATHER)
