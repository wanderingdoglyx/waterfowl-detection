"""
The MDC adapter: everything specific to the Missouri corpus, in one file.

The ten collections under `dataset_full/` do not agree on much — seven spreadsheet
layouts, two annotation-file naming conventions, two annotation line formats, three image
extensions, three habitat vocabularies, one collection distributed already tiled.  All of
that is absorbed here, so that nothing above the interface needs to know a collection's
name.

File parsing is *reused*, not reimplemented: `data_prep.prepare_data` remains the one
place that knows how to find an image, find its annotation file and read its boxes, so
this adapter cannot drift from the loader that produced every published result.

Conventions this corpus uses, declared rather than assumed:

* **Altitude** is the `height` column: DJI barometric relative altitude above take-off,
  truncated to a whole metre by whoever compiled the spreadsheets.
* **Split role** is the `bbox_split_Robert` column — the annotators' own decision, which
  the project honours and never reshuffles.  Anything other than "train" is test.
* **Dates are not ISO-8601 here**, because the corpus's own convention is ambiguous:
  Bird_H writes `03082022`, which could be 8 March or 3 August 2022.  The raw token is
  carried through rather than guessed at, and the Arkansas specification asks AGFC for
  ISO dates so that new data does not inherit the problem.
* **Bird_H is decoys.**  No file in the corpus records this; it is stated in the group's
  own 2024 paper (Table 1, "Decoys"), and it matters because decoy boxes run 8-26% larger
  than real birds of the same species.  It is recorded with `decoys_source="collection"`
  so an analysis can treat it as documentation rather than data.
* **Optics** come from EXIF where it survives (9% of the corpus) and are otherwise absent,
  so most photographs report no ground sample distance rather than a guessed one.
"""

from __future__ import annotations

import csv
import os
from typing import Iterator

from data_prep import config
from data_prep.prepare_data import find_annotation, find_image, read_annotations
from pipeline import cameras, vocabularies
from pipeline.adapters.base import DatasetAdapter
from pipeline.records import (ABOVE_TAKEOFF, Annotation, Camera, Collection, Photograph,
                              SourceCard)

SOURCE = "mdc"

# Habitat is a curated column in one collection and a file-name token in two others.
HABITAT_COLUMN = {"Bird_H": "background_info"}
HABITAT_TOKEN = ("Bird_G", "Bird_J")          # second underscore token
WEATHER_TOKEN = ("Bird_G", "Bird_J")          # first underscore token
PRE_TILED = {"Bird_E"}                        # distributed as 512 x 512 tiles

# Collections whose birds are decoys rather than wild (paper Table 1).
DECOY_COLLECTIONS = {"Bird_H": "Zhang et al., Information 2024, Table 1"}


def _site_and_date(collection: str, stem: str) -> tuple[str | None, str | None]:
    """Site and survey date, for the three collections documented to carry them."""
    parts = stem.split("_")
    if collection == "Bird_H" and len(parts) >= 2:
        return parts[0], parts[1]             # EagleBluffs_03082022_Trees_Spot1_...
    if collection == "Bird_D" and len(parts) >= 2:
        return parts[0], parts[1]             # G135_1Dec2016_0530
    if collection == "Bird_E":
        # mar2019_clipped_MODOC1214_0015GSD_LINE03B0405
        return next((p for p in parts if p.startswith("MODOC")), None), parts[0]
    return None, None


def _species_and_sex(label: str) -> tuple[str | None, str | None]:
    """"Pintail Male" -> ("Pintail", "male");  "Unknown" -> (None, None)."""
    text = label.strip()
    if not text or text.lower() == "unknown":
        return None, None
    for suffix, sex in ((" male", "male"), (" female", "female")):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)].strip(), sex
    return text, None


class MdcAdapter(DatasetAdapter):
    """Missouri Department of Conservation surveys, 2016-2022."""

    source = SOURCE

    # Bird counts the corpus itself publishes (dataset_full/DATASET_INVENTORY.md, compiled
    # 24 August 2026).  Checking against them is what turns a silent skip into a failure:
    # the two loader bugs in this project's history were both invisible without it.
    declared_birds = {
        "Bird_A": 8952, "Bird_B": 2175, "Bird_C": 963, "Bird_D": 28800, "Bird_E": 79424,
        "Bird_F": 1936, "Bird_G": 62742, "Bird_H": 16560, "Bird_I": 7038, "Bird_J": 108797,
    }

    field_how = {
        "altitude_m": "`height` column of each collection's image_info.csv",
        "site": "file-name conventions in Bird_D, Bird_E and Bird_H",
        "date": "file-name conventions in Bird_D, Bird_E and Bird_H",
    }

    unavailable = {
        "environment_beyond_sun_cloud":
            "No wind, temperature, water level or light-angle data exists in any collection.",
        "observer":
            "Annotator identity is not recorded, so annotation-style differences between "
            "collections cannot be attributed.",
    }

    def __init__(self, root: str | None = None, read_exif: bool = True):
        self.root = root or config.DATASET_ROOT
        self.read_exif = read_exif
        self._habitat = vocabularies.habitat(SOURCE)
        self._weather = vocabularies.weather(SOURCE)
        self._cache: dict[str, list[Photograph]] = {}

    # ── Structure ─────────────────────────────────────────────────────────────
    IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".png", ".PNG")

    def collections(self) -> list[Collection]:
        """A directory is a collection when it carries the corpus's index file — or, for a
        folder someone assembled by hand to choose the data for a run, simply images."""
        out = []
        for name in sorted(os.listdir(self.root)):
            if name in config.EXCLUDED_DATASETS:
                continue
            path = os.path.join(self.root, name)
            if not os.path.isdir(path):
                continue
            if os.path.exists(os.path.join(path, "image_info.csv")) or self._has_images(path):
                out.append(Collection(SOURCE, name, path, pre_tiled=name in PRE_TILED))
        return out

    def _has_images(self, path: str) -> bool:
        return any(e.name.endswith(self.IMAGE_EXTENSIONS) for e in os.scandir(path)
                   if e.is_file())

    def photographs(self, collection: Collection) -> Iterator[Photograph]:
        if collection.name in self._cache:
            yield from self._cache[collection.name]
            return

        photos: list[Photograph] = []
        index = os.path.join(collection.path, "image_info.csv")
        if os.path.exists(index):
            with open(index, newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            # No index: the folder is a hand-made selection.  Every image in it takes
            # part, with no recorded altitude, habitat or train/test mark — those are
            # reported as not recorded rather than invented.
            rows = [{"image_name": e.name} for e in sorted(os.scandir(collection.path),
                                                           key=lambda e: e.name)
                    if e.is_file() and e.name.endswith(self.IMAGE_EXTENSIONS)]
        for row in rows:
            photo = self._photograph(collection, row)
            if photo is not None:
                photos.append(photo)
        self._cache[collection.name] = photos
        yield from photos

    def _photograph(self, collection: Collection, row: dict) -> Photograph | None:
        name = (row.get("image_name") or "").strip()
        if not name:
            return None
        stem = os.path.splitext(name)[0]
        image_path = find_image(collection.path, stem)
        ann_path = find_annotation(collection.path, stem, row.get("annotation_name"))

        # Rule 2: never drop silently.  A photograph the loader cannot use is still
        # yielded, with the reason attached.
        reasons = []
        if image_path is None:
            reasons.append("image file not found")
        if ann_path is None:
            reasons.append("annotation file not found")

        altitude = None
        raw_height = (row.get("height") or "").strip()
        if raw_height:
            try:
                altitude = int(float(raw_height))     # as recorded: truncated to a metre
            except ValueError:
                altitude = None

        habitat_raw = habitat_source = None
        column = HABITAT_COLUMN.get(collection.name)
        if column and (row.get(column) or "").strip():
            habitat_raw, habitat_source = row[column].strip(), "column"
        elif collection.name in HABITAT_TOKEN:
            parts = stem.split("_")
            if len(parts) >= 2:
                habitat_raw, habitat_source = parts[1], "filename"

        weather_raw = weather_source = None
        if collection.name in WEATHER_TOKEN:
            token = stem.split("_")[0]
            if token in self._weather:
                weather_raw, weather_source = token, "filename"

        habitat, habitat_note = self._habitat.get(habitat_raw)
        weather, _ = self._weather.get(weather_raw)
        site, date = _site_and_date(collection.name, stem)

        width = height = None
        camera = Camera()
        if image_path:
            width, height = cameras.image_size(image_path)
            if self.read_exif:
                camera = cameras.from_exif(image_path)

        species_path = os.path.join(collection.path, stem + "_class.txt")
        if "bbox_split_Robert" in row:
            split = "train" if (row["bbox_split_Robert"] or "train").strip() == "train" else "test"
        else:
            split = "unassigned"          # no index file: the core decides (see splits.py)

        return Photograph(
            photo_id=f"{SOURCE}/{collection.name}/{stem}",
            source=SOURCE,
            collection=collection.name,
            path=image_path or os.path.join(collection.path, name),
            width_px=width,
            height_px=height,
            split_role=split,
            pre_tiled=collection.pre_tiled,
            altitude_m=altitude,
            altitude_reference=ABOVE_TAKEOFF if altitude is not None else None,
            camera=camera,
            habitat=habitat,
            habitat_raw=habitat_raw,
            habitat_source=habitat_source,
            weather=weather,
            weather_raw=weather_raw,
            weather_source=weather_source,
            site=site,
            date=date,
            decoys="decoy" if collection.name in DECOY_COLLECTIONS else None,
            decoys_source="collection" if collection.name in DECOY_COLLECTIONS else None,
            annotation_path=ann_path,
            species_path=species_path if os.path.exists(species_path) else None,
            usable=not reasons,
            skip_reason="; ".join(reasons) or None,
            notes=habitat_note,
        )

    # ── Annotations ───────────────────────────────────────────────────────────
    def annotations(self, photo: Photograph) -> list[Annotation]:
        """Boxes from the corpus's own loader, with species and sex merged in by position.

        The species file lists the same boxes with a species in place of the `bird`
        label, but the two files do not always hold the same number of rows, so they are
        matched on coordinates rather than zipped by line.
        """
        if not photo.annotation_path:
            return []
        boxes = read_annotations(photo.annotation_path)
        labels = self._species_by_box(photo)
        decoy = photo.decoys if photo.decoys in ("decoy", "real") else None

        out = []
        for x1, y1, x2, y2 in boxes:
            species, sex = labels.get((x1, y1, x2, y2), (None, None))
            out.append(Annotation(photo_id=photo.photo_id, geometry="box",
                                  coords=(float(x1), float(y1), float(x2), float(y2)),
                                  species=species, sex=sex, decoy=decoy,
                                  batch=photo.collection))
        return out

    def _species_by_box(self, photo: Photograph) -> dict[tuple, tuple]:
        if not photo.species_path:
            return {}
        out: dict[tuple, tuple] = {}
        with open(photo.species_path) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
                if len(parts) < 5:
                    continue
                try:
                    box = tuple(int(float(v)) for v in parts[-4:])
                except ValueError:
                    continue
                out[box] = _species_and_sex(",".join(parts[:-4]))
        return out

    # ── Provenance ────────────────────────────────────────────────────────────
    def describe(self) -> SourceCard:
        collections = self.collections()
        photos = list(self.usable_photographs())
        return SourceCard(
            source=SOURCE,
            provider="Missouri Department of Conservation",
            licence="project-internal",
            contact=None,
            received="2016-2022 surveys; corpus compiled 2026",
            root=self.root,
            collections=len(collections),
            photographs=len(photos),
            annotations=sum(len(self.annotations(p)) for p in photos),
            notes="Altitudes are barometric above take-off, truncated to a metre. "
                  "Dates are the corpus's own tokens, not ISO-8601 (see module docstring).",
        )
