"""
Per-crop survey metadata: the join from a test crop back to the conditions its
source image was flown under.

The evaluation pipeline deliberately forgets where a crop came from — every
model sees `Bird_G/G_Cloud_WaterSubVeg_15m_test2_0_0.JPG` as an opaque 512x512
tile.  The survey-condition analysis asks the opposite question ("does habitat
change the ranking?"),
so this module rebuilds the link:

    crop file_name  ->  (dataset, source stem, tile offset)
                    ->  dataset_full/<ds>/image_info.csv  +  filename tokens
                    ->  altitude / habitat / weather / site / date

Nothing here is inferred.  A field is either read from a file or left None, and
every field records *where it came from* (`FIELD_SOURCES`) so a reader can weigh
a filename token differently from a curated CSV column.

**Coverage is not uniform and must not be treated as such.**  Altitude is the
only field present for all ten datasets; habitat exists for three, weather for
two, and decoy-vs-real for none.  `availability()` reports the real coverage of
each field on the split actually being scored, so a table can state its own n
instead of implying corpus-wide support.

The vocabularies are normalised in exactly two places, both documented in
`DATASET_INVENTORY.md` and both applied here rather than left to the caller:

* Bird_G spells one habitat two ways (`WaterSubVeg` / `WaterSubVege`); they are
  merged.
* Bird_J's habitat token includes `Dense`, which describes bird density, not
  habitat.  It is dropped from habitat rather than analysed as one, because
  pooling it would mix two different variables in one column.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


# Crops are named "{letter}_{source stem}_{row}_{col}.JPG" by data_prep.prepare_data.
# The stem itself contains underscores ("Cloud_WaterSubVeg_15m_test2"), so the two
# tile offsets are stripped from the right and everything between is the stem.
_CROP_RE = re.compile(r"^([A-J])_(.+)_(\d+)_(\d+)$")

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_full"
)

# Altitudes are recorded as flown, so the same nominal height appears as 14/15,
# 29/30, 59/60, 89/90 and 119/120.  Snapping to the nominal ladder within a small
# tolerance merges instrument jitter without inventing bands: 40/50/70/80 in
# Bird_A are real, separate survey heights and are left alone.
_NOMINAL_ALTITUDES = (15, 30, 45, 60, 90, 120)
_NOMINAL_TOLERANCE = 2

# Which datasets carry which field, and how it was recorded.  Used for the
# provenance line under every survey-condition table.
FIELD_SOURCES: dict[str, str] = {
    "altitude_m": "`height` column of each dataset's image_info.csv (all 10 datasets)",
    "habitat": (
        "`background_info` column (Bird_H) and filename token 2 (Bird_G, Bird_J); "
        "no other dataset records habitat"
    ),
    "weather": "filename token 1, Bird_G and Bird_J only (Sun/Cloud; no other conditions recorded)",
    "site": "filename site name (Bird_H) and plot code (Bird_D); MODOC for Bird_E",
    "date": "filename date (Bird_D, Bird_E, Bird_H); no EXIF survives for these datasets",
    "dataset": "sub-dataset folder name (always known)",
}

# Fields the data cannot support at all, kept explicit so a report can say so
# rather than quietly omitting the question that was asked.
UNAVAILABLE_FIELDS: dict[str, str] = {
    "decoy_vs_real": (
        "No marker exists in any filename, CSV column or annotation file in any of "
        "the ten datasets. Distinguishing decoys would require new annotation or a "
        "field record from MDC."
    ),
    "camera_sensor": (
        "EXIF survives for only 191 of 2,095 images (9%), covering two camera models "
        "in Bird_B/Bird_G. Too thin, and fully confounded with dataset identity."
    ),
    "environment_beyond_sun_cloud": (
        "No wind, temperature, water level or light-angle data exists in any dataset."
    ),
}


@dataclass(frozen=True)
class CropMeta:
    """Survey conditions for one test crop. None means 'not recorded', never 'unknown-but-guessed'."""

    image_id: int
    file_name: str
    dataset: str                 # "Bird_G"
    stem: str                    # source image stem, "Cloud_WaterSubVeg_15m_test2"
    row: int                     # tile origin, source pixels
    col: int
    altitude_m: int | None = None        # as recorded
    altitude_nominal: int | None = None  # snapped to the survey ladder
    habitat: str | None = None
    habitat_source: str | None = None    # "column" (curated) or "filename" (parsed)
    weather: str | None = None
    site: str | None = None
    date: str | None = None


def parse_crop_name(file_name: str) -> tuple[str, str, int, int] | None:
    """"Bird_G/G_Cloud_..._0_460.JPG" -> ("Bird_G", "Cloud_...", 0, 460)."""
    if "/" not in file_name:
        return None
    dataset, base = file_name.split("/", 1)
    m = _CROP_RE.match(os.path.splitext(base)[0])
    if not m:
        return None
    return dataset, m.group(2), int(m.group(3)), int(m.group(4))


def _nominal(alt: int | None) -> int | None:
    """Snap a flown altitude to the nominal survey ladder, within tolerance."""
    if alt is None:
        return None
    for nom in _NOMINAL_ALTITUDES:
        if abs(alt - nom) <= _NOMINAL_TOLERANCE:
            return nom
    return alt


def _habitat_from_filename(stem: str) -> str | None:
    """Bird_G / Bird_J encode habitat as the second underscore token."""
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    token = parts[1]
    # One class, two spellings — merge before anything counts them separately.
    if token == "WaterSubVege":
        return "WaterSubVeg"
    # Bird_J's vocabulary mixes in a density word; it is not a habitat.
    if token == "Dense":
        return None
    return token


def _weather_from_filename(stem: str) -> str | None:
    token = stem.split("_")[0]
    return token if token in ("Sun", "Cloud") else None


def _site_and_date(dataset: str, stem: str) -> tuple[str | None, str | None]:
    """Site and survey date, only for the datasets documented to carry them."""
    parts = stem.split("_")
    if dataset == "Bird_H" and len(parts) >= 2:
        # EagleBluffs_03082022_Trees_Spot1_30meters_DJI_0070
        return parts[0], parts[1]
    if dataset == "Bird_D" and len(parts) >= 2:
        # G135_1Dec2016_0530 — plot code and day
        return parts[0], parts[1]
    if dataset == "Bird_E":
        # mar2019_clipped_MODOC1214_0015GSD_LINE03B0405 — month, and a MODOC site id
        site = next((p for p in parts if p.startswith("MODOC")), None)
        return site, parts[0]
    return None, None


def _load_source_rows(dataset: str) -> dict[str, dict]:
    """image_info.csv for one dataset, keyed by image stem."""
    path = os.path.join(DATASET_DIR, dataset, "image_info.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {
            os.path.splitext(r["image_name"])[0]: r
            for r in csv.DictReader(f)
            if r.get("image_name")
        }


def load(coco_path: str) -> dict[int, CropMeta]:
    """Metadata for every crop in a COCO split, keyed by COCO image id.

    Crops whose source row cannot be found keep their dataset and stem and leave
    every recorded field None, so they still appear in totals as unattributed
    rather than vanishing from the denominator.
    """
    with open(coco_path) as f:
        coco = json.load(f)

    sources: dict[str, dict[str, dict]] = {}
    metas: dict[int, CropMeta] = {}

    for im in coco["images"]:
        parsed = parse_crop_name(im["file_name"])
        if parsed is None:
            continue
        dataset, stem, row, col = parsed
        if dataset not in sources:
            sources[dataset] = _load_source_rows(dataset)
        src = sources[dataset].get(stem)

        altitude = None
        if src and src.get("height") not in (None, ""):
            try:
                altitude = int(float(src["height"]))
            except ValueError:
                altitude = None

        # Habitat: a curated column beats a filename token, and only three
        # datasets have either.
        habitat = habitat_source = None
        if dataset == "Bird_H" and src and src.get("background_info"):
            habitat, habitat_source = src["background_info"], "column"
        elif dataset in ("Bird_G", "Bird_J"):
            habitat = _habitat_from_filename(stem)
            habitat_source = "filename" if habitat else None

        site, date = _site_and_date(dataset, stem)
        metas[int(im["id"])] = CropMeta(
            image_id=int(im["id"]),
            file_name=im["file_name"],
            dataset=dataset,
            stem=stem,
            row=row,
            col=col,
            altitude_m=altitude,
            altitude_nominal=_nominal(altitude),
            habitat=habitat,
            habitat_source=habitat_source,
            weather=(_weather_from_filename(stem)
                     if dataset in ("Bird_G", "Bird_J") else None),
            site=site,
            date=date,
        )
    return metas


# ── Grouping ─────────────────────────────────────────────────────────────────

def group_by(metas: dict[int, CropMeta], field: str,
             datasets: set[str] | None = None) -> dict[object, list[int]]:
    """{field value -> image ids}, skipping crops where the field is not recorded.

    `datasets` restricts the grouping to named sub-datasets, which is how a
    within-dataset comparison avoids pooling across sites that differ in far
    more than the field being studied.
    """
    groups: dict[object, list[int]] = defaultdict(list)
    for image_id, m in metas.items():
        if datasets is not None and m.dataset not in datasets:
            continue
        value = getattr(m, field, None)
        if value is None:
            continue
        groups[value].append(image_id)
    return dict(groups)


def availability(metas: dict[int, CropMeta], gt_counts: dict[int, int],
                 field: str) -> dict:
    """How much of the split a field actually covers, in crops and in birds.

    Reported next to every subgroup table: a habitat breakdown covering 27% of
    images is a subset finding, and the table should say so itself rather than
    rely on the reader remembering it.
    """
    n_crops = n_birds = 0
    cov_crops = cov_birds = 0
    datasets: set[str] = set()
    for image_id, m in metas.items():
        birds = gt_counts.get(image_id, 0)
        n_crops += 1
        n_birds += birds
        if getattr(m, field, None) is not None:
            cov_crops += 1
            cov_birds += birds
            datasets.add(m.dataset)
    return {
        "field": field,
        "n_crops": n_crops,
        "n_birds": n_birds,
        "covered_crops": cov_crops,
        "covered_birds": cov_birds,
        "crop_pct": 100.0 * cov_crops / n_crops if n_crops else 0.0,
        "bird_pct": 100.0 * cov_birds / n_birds if n_birds else 0.0,
        "datasets": sorted(datasets),
        "source": FIELD_SOURCES.get(field, "—"),
    }


def datasets_varying(metas: dict[int, CropMeta], field: str) -> list[str]:
    """Sub-datasets that vary `field` internally.

    These are the only ones where a within-dataset comparison is possible: if a
    dataset flew everything at 90 m, its numbers say nothing about altitude and
    everything about that dataset.
    """
    seen: dict[str, set] = defaultdict(set)
    for m in metas.values():
        v = getattr(m, field, None)
        if v is not None:
            seen[m.dataset].add(v)
    return sorted(ds for ds, vals in seen.items() if len(vals) > 1)


def source_image_count(metas: dict[int, CropMeta],
                       image_ids: list[int] | None = None) -> int:
    """Distinct source images behind a set of crops.

    Crops from one photograph are not independent samples, so a table reporting
    "n = 3,168 crops" over 33 photographs should say both numbers.
    """
    ids = image_ids if image_ids is not None else list(metas)
    return len({(metas[i].dataset, metas[i].stem) for i in ids if i in metas})


def counts_by_dataset(metas: dict[int, CropMeta],
                      image_ids: list[int]) -> dict[str, int]:
    """Crops per sub-dataset within a subgroup — the confound check for any
    pooled finding."""
    return dict(Counter(metas[i].dataset for i in image_ids if i in metas))
