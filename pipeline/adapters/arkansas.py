"""
The Arkansas adapter: reads a delivery in the format the specification asks AGFC for.

It is written before any data exists, against the specification itself (Task 4.4), which
is the point of writing the specification first: when the first delivery lands, acceptance
is a command rather than a project.  The synthetic delivery in `pipeline.verify` exercises
every path here.

Expected layout:

    <root>/photographs.csv                one row per photograph (the metadata table)
    <root>/<site>/<date>/<frame>.jpg      original frames, unaltered
    <root>/<site>/<date>/<frame>.csv      one annotation file per photograph
      or
    <root>/labels/<site>/<date>/<frame>.csv

    annotation rows:  x1,y1,x2,y2,species,sex,decoy,notes     (boxes)
                 or:  x,y,species,sex,decoy,notes             (points)

A header-only annotation file means "no birds in this photograph", which is different
from a missing file ("not yet annotated") — and only the first counts as evidence about
false alarms.

Nothing here is Missouri-specific, and nothing in the rest of the pipeline is Arkansas-
specific: adding this programme changes no file above the interface.
"""

from __future__ import annotations

import csv
import os
from typing import Iterator

from pipeline import cameras, vocabularies
from pipeline.adapters.base import DatasetAdapter
from pipeline.records import (Annotation, Camera, Collection, Photograph, SourceCard)

SOURCE = "arkansas"

REQUIRED_COLUMNS = ("file", "site", "date", "flight_altitude_m", "altitude_reference",
                    "drone_platform", "sensor", "focal_length_mm", "sensor_width_mm",
                    "image_width_px", "image_height_px", "habitat")


def _float(row: dict, key: str) -> float | None:
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(row: dict, key: str) -> int | None:
    value = _float(row, key)
    return int(value) if value is not None else None


def _text(row: dict, key: str) -> str | None:
    value = (row.get(key) or "").strip()
    return value or None


class ArkansasAdapter(DatasetAdapter):
    """AGFC surveys, delivered in the agreed format."""

    source = SOURCE

    unavailable: dict[str, str] = {}

    def __init__(self, root: str, table: str = "photographs.csv",
                 provider: str = "Arkansas Game and Fish Commission",
                 licence: str | None = None, contact: str | None = None,
                 received: str | None = None):
        self.root = root
        self.table_path = os.path.join(root, table)
        self.provider = provider
        self.licence = licence
        self.contact = contact
        self.received = received
        self._habitat = vocabularies.habitat(SOURCE)
        self._weather = vocabularies.weather(SOURCE)
        self._rows: list[dict] | None = None
        self._cache: dict[str, list[Photograph]] = {}

    # ── The delivery's own metadata table ─────────────────────────────────────
    def rows(self) -> list[dict]:
        if self._rows is None:
            if not os.path.exists(self.table_path):
                raise FileNotFoundError(
                    f"{self.table_path} is missing: a delivery must carry its photograph "
                    "table (see the Arkansas specification, Table 1)")
            with open(self.table_path, newline="") as f:
                self._rows = [r for r in csv.DictReader(f) if (r.get("file") or "").strip()]
        return self._rows

    def missing_columns(self) -> list[str]:
        """Required columns the delivery does not provide — reported, not worked around."""
        rows = self.rows()
        present = set(rows[0].keys()) if rows else set()
        return [c for c in REQUIRED_COLUMNS if c not in present]

    # ── Structure: a collection is a survey site ──────────────────────────────
    def collections(self) -> list[Collection]:
        sites = sorted({(r.get("site") or "").strip() for r in self.rows()} - {""})
        return [Collection(SOURCE, site, os.path.join(self.root, site)) for site in sites]

    def photographs(self, collection: Collection) -> Iterator[Photograph]:
        if collection.name in self._cache:
            yield from self._cache[collection.name]
            return
        photos = [self._photograph(r) for r in self.rows()
                  if (r.get("site") or "").strip() == collection.name]
        self._cache[collection.name] = photos
        yield from photos

    def _photograph(self, row: dict) -> Photograph:
        rel = (row["file"] or "").strip()
        path = os.path.join(self.root, rel)
        stem = os.path.splitext(os.path.basename(rel))[0]
        site = _text(row, "site") or "unknown_site"

        ann_path = self._annotation_path(rel)
        reasons = []
        if not os.path.exists(path):
            reasons.append("image file not found")
        if ann_path is None:
            reasons.append("annotation file not found")

        habitat_raw = _text(row, "habitat")
        habitat, note = self._habitat.get(habitat_raw)
        weather_raw = _text(row, "weather")
        weather, _ = self._weather.get(weather_raw)

        camera = Camera(
            model=_text(row, "sensor") or _text(row, "drone_platform"),
            focal_length_mm=_float(row, "focal_length_mm"),
            sensor_width_mm=_float(row, "sensor_width_mm"),
            image_width_px=_int(row, "image_width_px"),
            source="delivery",
        )
        if not camera.complete and os.path.exists(path):
            # The table is authoritative; EXIF is the fallback, and says so.
            exif = cameras.from_exif(path)
            if exif.complete:
                camera = exif

        width, height = _int(row, "image_width_px"), _int(row, "image_height_px")
        if (width is None or height is None) and os.path.exists(path):
            width, height = cameras.image_size(path)

        split = (_text(row, "split") or "unassigned").lower()
        if split not in ("train", "test", "unassigned"):
            split = "unassigned"

        return Photograph(
            photo_id=f"{SOURCE}/{site}/{stem}",
            source=SOURCE,
            collection=site,
            path=path,
            width_px=width,
            height_px=height,
            split_role=split,
            pre_tiled=(_text(row, "pre_tiled") or "").lower() in ("1", "true", "yes"),
            altitude_m=_float(row, "flight_altitude_m"),
            altitude_reference=_text(row, "altitude_reference"),
            camera=camera,
            habitat=habitat,
            habitat_raw=habitat_raw,
            habitat_source="column" if habitat_raw else None,
            weather=weather,
            weather_raw=weather_raw,
            weather_source="column" if weather_raw else None,
            site=site,
            date=_text(row, "date"),
            decoys=_text(row, "decoys"),
            decoys_source="column" if _text(row, "decoys") else None,
            annotation_path=ann_path,
            species_path=None,
            usable=not reasons,
            skip_reason="; ".join(reasons) or None,
            notes=note or _text(row, "notes"),
        )

    def _annotation_path(self, rel: str) -> str | None:
        """Beside the photograph, or in a mirrored labels/ tree."""
        stem = os.path.splitext(rel)[0]
        for candidate in (os.path.join(self.root, stem + ".csv"),
                          os.path.join(self.root, "labels", stem + ".csv")):
            if os.path.exists(candidate):
                return candidate
        return None

    # ── Annotations ───────────────────────────────────────────────────────────
    def annotations(self, photo: Photograph) -> list[Annotation]:
        if not photo.annotation_path:
            return []
        out: list[Annotation] = []
        with open(photo.annotation_path, newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or ())
            is_box = {"x1", "y1", "x2", "y2"} <= columns
            is_point = {"x", "y"} <= columns
            if not (is_box or is_point):
                return []
            for row in reader:
                try:
                    if is_box:
                        coords = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                        if coords[2] <= coords[0] or coords[3] <= coords[1]:
                            continue
                    else:
                        coords = tuple(float(row[k]) for k in ("x", "y"))
                except (TypeError, ValueError):
                    continue
                out.append(Annotation(
                    photo_id=photo.photo_id,
                    geometry="box" if is_box else "point",
                    coords=coords,
                    species=_text(row, "species"),
                    sex=_text(row, "sex"),
                    decoy=_text(row, "decoy") or photo.decoys,
                    batch=_text(row, "batch") or photo.site,
                ))
        return out

    # ── Provenance ────────────────────────────────────────────────────────────
    def describe(self) -> SourceCard:
        photos = list(self.usable_photographs())
        return SourceCard(
            source=SOURCE,
            provider=self.provider,
            licence=self.licence,
            contact=self.contact,
            received=self.received,
            root=self.root,
            collections=len(self.collections()),
            photographs=len(photos),
            annotations=sum(len(self.annotations(p)) for p in photos),
            notes=None if not self.missing_columns()
            else f"delivery is missing required columns: {', '.join(self.missing_columns())}",
        )
