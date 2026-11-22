"""
The link between a crop and the photograph it came from.

Tiling is one-way today: `data_prep.prepare_data` writes crops named
`{letter}_{stem}_{row}_{col}.JPG` under a per-collection folder, and the COCO record for
a crop keeps only an id, a file name and a size.  Everything about the survey is dropped,
and the analysis rebuilds it afterwards from the file name.

This module is the interface-based replacement, and it works on the *existing* COCO files
without regenerating them: crop -> photograph -> the adapter's canonical record.  So the
survey conditions of any split can be read through the interface today, for any source,
while the committed splits stay exactly as they are.

When the splits are next regenerated, `photograph_fields()` is what the tiler should write
into each COCO image record — at which point nothing needs to parse a file name at all.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from pipeline.adapters.base import DatasetAdapter
from pipeline.records import Photograph

# Written by the tiler: "<collection>/<letter>_<stem>_<row>_<col>.JPG".  The stem itself
# contains underscores, so the two tile offsets are stripped from the right.
_CROP_RE = re.compile(r"^([A-Za-z0-9]+)_(.+)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class CropRecord:
    """One crop, with the photograph record it inherits its conditions from."""

    image_id: int
    file_name: str
    collection: str
    stem: str
    row: int                      # tile origin in source pixels
    col: int
    photo: Photograph | None      # None when the crop cannot be traced to a photograph

    @property
    def traced(self) -> bool:
        return self.photo is not None

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Crop coordinates -> coordinates in the source photograph."""
        return x + self.col, y + self.row


def parse_crop_name(file_name: str) -> tuple[str, str, int, int] | None:
    """"Bird_G/G_Cloud_..._0_460.JPG" -> ("Bird_G", "Cloud_...", 0, 460)."""
    if "/" not in file_name:
        return None
    collection, base = file_name.split("/", 1)
    m = _CROP_RE.match(os.path.splitext(base)[0])
    if not m:
        return None
    return collection, m.group(2), int(m.group(3)), int(m.group(4))


def photograph_index(adapter: DatasetAdapter) -> dict[tuple[str, str], Photograph]:
    """{(collection, stem): photograph} for one source."""
    return {(p.collection, p.stem): p for p in adapter.all_photographs()}


def load_split(coco_path: str, adapter: DatasetAdapter) -> dict[int, CropRecord]:
    """Every crop in a COCO split, joined to its photograph through the adapter.

    Crops that cannot be traced keep their collection and stem and carry `photo=None`, so
    they stay in the denominator as unattributed rather than vanishing from it.
    """
    with open(coco_path) as f:
        coco = json.load(f)

    index = photograph_index(adapter)
    out: dict[int, CropRecord] = {}
    for image in coco["images"]:
        parsed = parse_crop_name(image["file_name"])
        if parsed is None:
            continue
        collection, stem, row, col = parsed
        out[int(image["id"])] = CropRecord(
            image_id=int(image["id"]), file_name=image["file_name"],
            collection=collection, stem=stem, row=row, col=col,
            photo=index.get((collection, stem)),
        )
    return out


def photograph_fields(photo: Photograph) -> dict:
    """The fields the tiler should copy onto every crop of this photograph.

    This is the payload that makes the analysis dataset-agnostic: with these in the COCO
    record, no downstream module needs the file name, the spreadsheet, or the name of a
    collection.
    """
    return {
        "photo_id": photo.photo_id,
        "source": photo.source,
        "collection": photo.collection,
        "site": photo.site,
        "date": photo.date,
        "altitude_m": photo.altitude_m,
        "altitude_reference": photo.altitude_reference,
        "gsd_cm_px": photo.gsd_cm_px(),
        "camera_model": photo.camera.model,
        "habitat": photo.habitat,
        "habitat_raw": photo.habitat_raw,
        "habitat_source": photo.habitat_source,
        "weather": photo.weather,
        "weather_raw": photo.weather_raw,
        "decoys": photo.decoys,
        "pre_tiled": photo.pre_tiled,
    }


def group_by(crops: dict[int, CropRecord], field: str) -> dict[object, list[int]]:
    """{value -> image ids}, skipping crops where the field is not recorded.

    `field` is any attribute of the photograph record ("habitat", "altitude_m",
    "weather", "site", "date", "source", "collection") or "gsd_cm_px".
    """
    groups: dict[object, list[int]] = {}
    for image_id, crop in crops.items():
        if crop.photo is None:
            continue
        value = crop.photo.gsd_cm_px() if field == "gsd_cm_px" else getattr(crop.photo, field, None)
        if value is None:
            continue
        groups.setdefault(value, []).append(image_id)
    return groups
