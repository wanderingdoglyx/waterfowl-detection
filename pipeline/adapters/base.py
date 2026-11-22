"""
The adapter contract: what a survey programme must provide, and nothing more.

An adapter yields canonical records.  It never writes crops, splits or COCO files — the
dataset-agnostic core does that, identically for every programme, which is what keeps
Missouri and Arkansas results comparable.

Six rules, from the Task 4.3 design:

1. The adapter never writes crops, splits or COCO files.
2. The adapter never drops data silently.  A photograph whose annotation file is missing
   is yielded with `usable=False` and a reason; it is not skipped.  The two loader bugs
   in this project's history — 140 photographs lost to an unexpected annotation file
   name, 2,646 boxes lost to an unexpected line format — were both silent skips.
3. Raw labels are always retained beside canonical ones.
4. Units and references are declared, never assumed.
5. Identity is stable: `<source>/<collection>/<stem>`.
6. Everything the programme knows is offered once, `notes` included.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from pipeline import vocabularies
from pipeline.records import (Annotation, Collection, CoverageReport, FieldCoverage,
                              Photograph, SourceCard)

# Fields whose coverage is reported for every source.  Adapters may extend the list.
COVERAGE_FIELDS = ("altitude_m", "camera", "habitat", "weather", "site", "date", "decoys")


class DatasetAdapter(ABC):
    """One survey programme's knowledge, and the only place it lives."""

    source: str = "unknown"

    # ── What the programme contains ───────────────────────────────────────────
    @abstractmethod
    def collections(self) -> list[Collection]:
        ...

    @abstractmethod
    def photographs(self, collection: Collection) -> Iterator[Photograph]:
        ...

    @abstractmethod
    def annotations(self, photo: Photograph) -> list[Annotation]:
        ...

    # ── What the programme is ─────────────────────────────────────────────────
    @abstractmethod
    def describe(self) -> SourceCard:
        ...

    def crosswalks(self) -> dict[str, vocabularies.Crosswalk]:
        """raw label -> canonical term, one per vocabulary."""
        return {"habitat": vocabularies.habitat(self.source),
                "weather": vocabularies.weather(self.source)}

    # Fields this programme cannot support at all, with the reason, so a report can say
    # so rather than quietly omitting the question that was asked.
    unavailable: dict[str, str] = {}

    # How a field is recorded, for fields whose provenance is the same for the whole
    # programme (a per-photograph `<field>_source` wins over this).
    field_how: dict[str, str] = {}

    # ── Helpers every adapter gets for free ───────────────────────────────────
    def all_photographs(self) -> Iterator[Photograph]:
        for collection in self.collections():
            yield from self.photographs(collection)

    def usable_photographs(self) -> Iterator[Photograph]:
        for photo in self.all_photographs():
            if photo.usable:
                yield photo

    def coverage(self, photos: Iterable[Photograph] | None = None) -> CoverageReport:
        """Measured coverage of each field, over photographs and over birds.

        Computed by iterating, never declared, so it cannot drift from the data.
        """
        photos = list(photos if photos is not None else self.usable_photographs())
        counts = {f: [0, 0] for f in COVERAGE_FIELDS}
        how: dict[str, set[str]] = {f: set() for f in COVERAGE_FIELDS}
        total_birds = 0

        for photo in photos:
            n = len(self.annotations(photo))
            total_birds += n
            for f in COVERAGE_FIELDS:
                if f == "camera":
                    present = photo.camera.complete
                    src = photo.camera.source
                else:
                    present = getattr(photo, f, None) is not None
                    src = getattr(photo, f"{f}_source", None)
                if present:
                    counts[f][0] += 1
                    counts[f][1] += n
                    if src:
                        how[f].add(src)
                    elif self.field_how.get(f):
                        how[f].add(self.field_how[f])

        n_photos = len(photos) or 1
        fields = [
            FieldCoverage(
                field=f,
                photographs=counts[f][0],
                photographs_pct=100.0 * counts[f][0] / n_photos,
                birds=counts[f][1],
                birds_pct=100.0 * counts[f][1] / (total_birds or 1),
                how=", ".join(sorted(how[f])) or None,
            )
            for f in COVERAGE_FIELDS
        ]
        return CoverageReport(source=self.source, photographs=len(photos),
                              birds=total_birds, fields=fields,
                              unavailable=dict(self.unavailable))
