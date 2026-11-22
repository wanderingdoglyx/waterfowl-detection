"""
The canonical records every dataset adapter produces — the interface itself.

Three rules hold for every field here, and they are what make two survey programmes
comparable rather than merely concatenated:

* **"Not recorded" is a value.**  A field no source file contains is None, never a
  guess, and `CoverageReport` states how much of the data actually carries it.
* **Raw labels are retained.**  `habitat` is a canonical term; `habitat_raw` is what the
  programme itself wrote.  A harmonisation mistake stays visible and reversible.
* **Units and references are declared.**  Altitude carries what it is measured from;
  geometry carries its coordinate space.  Missouri's altitudes are barometric above
  take-off, and silently mixing them with above-ground-level values would shift every
  apparent-size comparison.

Identity: `photo_id` is "<source>/<collection>/<stem>", stable across regenerations and
unique across programmes, so Missouri and Arkansas can be pooled without collisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

# Altitude reference values (what the number is measured from).
ABOVE_TAKEOFF = "above_takeoff"
ABOVE_GROUND = "above_ground"
ABOVE_SEA_LEVEL = "above_sea_level"
ALTITUDE_REFERENCES = (ABOVE_TAKEOFF, ABOVE_GROUND, ABOVE_SEA_LEVEL)

# Who recorded a field: a curated column, a file name, image EXIF, or a documented
# property of the collection.  Used to weigh a parsed token differently from a column.
SOURCES = ("column", "filename", "exif", "collection", "delivery")

DECOY_VALUES = ("real", "decoy", "mixed")


@dataclass(frozen=True)
class Camera:
    """Optics, enough to turn an altitude into a ground sample distance."""

    model: str | None = None
    focal_length_mm: float | None = None
    sensor_width_mm: float | None = None
    image_width_px: int | None = None
    source: str | None = None          # one of SOURCES

    @property
    def complete(self) -> bool:
        return None not in (self.focal_length_mm, self.sensor_width_mm, self.image_width_px)


@dataclass(frozen=True)
class Collection:
    """A sub-collection within a programme: an MDC folder, or an AGFC site."""

    source: str
    name: str
    path: str
    pre_tiled: bool = False
    notes: str | None = None


@dataclass(frozen=True)
class Photograph:
    """One frame as it came off the aircraft, with the conditions it was flown under."""

    photo_id: str
    source: str
    collection: str
    path: str
    width_px: int | None = None
    height_px: int | None = None
    split_role: str = "unassigned"          # train / test / unassigned, as the provider marked it
    pre_tiled: bool = False

    altitude_m: float | None = None
    altitude_reference: str | None = None
    camera: Camera = field(default_factory=Camera)

    habitat: str | None = None              # canonical term
    habitat_raw: str | None = None          # what the programme wrote
    habitat_source: str | None = None
    weather: str | None = None
    weather_raw: str | None = None
    weather_source: str | None = None

    site: str | None = None
    date: str | None = None                 # ISO-8601 where the programme records one
    decoys: str | None = None               # real / decoy / mixed
    decoys_source: str | None = None

    annotation_path: str | None = None
    species_path: str | None = None
    usable: bool = True                     # False = no image or no annotations on disk
    skip_reason: str | None = None
    notes: str | None = None

    @property
    def stem(self) -> str:
        return self.photo_id.rsplit("/", 1)[-1]

    def gsd_cm_px(self) -> float | None:
        """Centimetres of ground per pixel, or None when the optics or height are absent.

        GSD = altitude x (sensor width / image width) / focal length, all in consistent
        units.  This is the number that makes two programmes comparable: the same bird at
        the same altitude covers a different number of pixels on a different sensor.
        """
        c = self.camera
        if self.altitude_m is None or not c.complete:
            return None
        pixel_pitch_mm = c.sensor_width_mm / c.image_width_px
        return float(self.altitude_m * 1000.0 * pixel_pitch_mm / c.focal_length_mm / 10.0)

    def as_row(self) -> dict:
        d = asdict(self)
        cam = d.pop("camera")
        d.update({f"camera_{k}": v for k, v in cam.items()})
        d["gsd_cm_px"] = self.gsd_cm_px()
        return d


@dataclass(frozen=True)
class Annotation:
    """One annotated bird, in the pixel space of its (full-resolution) photograph."""

    photo_id: str
    geometry: str                            # "box" or "point"
    coords: tuple[float, ...]                # box: (x1, y1, x2, y2); point: (x, y)
    category: str = "bird"
    species: str | None = None
    sex: str | None = None
    decoy: str | None = None
    batch: str | None = None

    @property
    def centre(self) -> tuple[float, float]:
        if self.geometry == "point":
            return float(self.coords[0]), float(self.coords[1])
        x1, y1, x2, y2 = self.coords
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size_px(self) -> float | None:
        """Square-root of box area — the project's "apparent size". None for points."""
        if self.geometry != "box":
            return None
        x1, y1, x2, y2 = self.coords
        return float(((x2 - x1) * (y2 - y1)) ** 0.5)


@dataclass
class SourceCard:
    """Provenance for a whole delivery, archived with the data version."""

    source: str
    provider: str | None = None
    licence: str | None = None
    contact: str | None = None
    received: str | None = None
    root: str | None = None
    collections: int = 0
    photographs: int = 0
    annotations: int = 0
    notes: str | None = None


@dataclass
class FieldCoverage:
    field: str
    photographs: int
    photographs_pct: float
    birds: int
    birds_pct: float
    how: str | None = None                  # how it was recorded, where known


@dataclass
class CoverageReport:
    """What this source actually records, measured rather than claimed.

    Every condition table in the analysis states its own basis from this, so a habitat
    result never implies corpus-wide support it does not have.
    """

    source: str
    photographs: int
    birds: int
    fields: list[FieldCoverage] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    def rows(self) -> Iterable[dict]:
        for f in self.fields:
            yield {"source": self.source, **asdict(f)}

    def format(self) -> str:
        head = (f"{self.source}: {self.photographs:,} photographs, {self.birds:,} birds\n"
                f"  {'field':22s}{'photographs':>14s}{'birds':>16s}  how")
        lines = [head]
        for f in self.fields:
            lines.append(f"  {f.field:22s}{f.photographs:8,d} ({f.photographs_pct:4.0f}%)"
                         f"{f.birds:10,d} ({f.birds_pct:4.0f}%)  {f.how or ''}")
        for name, why in self.unavailable.items():
            lines.append(f"  {name:22s}{'not recorded':>14s}{'':16s}  {why}")
        return "\n".join(lines)
