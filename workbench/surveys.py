"""
Survey imagery: photographs to count birds in, brought in without touching training data.

A survey is a named batch of photographs — one flight, one site, one day — registered
under output/workbench/surveys/<name>/:

    survey.json    where the photos came from, each photo's size, optional metadata
    images/        the photos: symlinks to the originals (default) or copies
    labels/        copies of any label files that came with them (bird,x1,y1,x2,y2)

Surveys are kept apart from dataset_full/ and the data-selection folders on purpose.
Adding photographs to the training corpus reorders the pool the seeded train/validation
split is drawn from, which reshuffles membership for every collection (README, "Are the
splits fixed?").  Counting birds in a new flight must never have that side effect;
photographs meant to become *training* data go through "Import labelled photos as a
training collection" instead, which builds a data-selection folder.

Label files are optional.  When a survey arrives with them — a held-out collection, or a
flight someone has already counted by hand — the count report scores the detections
against them, which is the quickest way to check a model on data it has never seen.
Photographs are read the way the training crops were cut (no EXIF rotation), so label
coordinates and pixels agree.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field

from PIL import Image

from data_prep.prepare_data import find_annotation, read_annotations
from workbench import paths
from workbench.catalog import IMAGE_EXTS
from workbench.util import read_json, slug, write_json


@dataclass
class Photo:
    file: str                 # relative to the survey folder, e.g. "images/DJI_0001.JPG"
    width: int
    height: int
    label: str | None = None  # relative, e.g. "labels/DJI_0001.txt"
    source: str = ""          # the original file

    @property
    def name(self) -> str:
        return os.path.basename(self.file)


@dataclass
class Survey:
    name: str
    created: float
    source: str
    mode: str                              # "link" or "copy"
    photos: list[Photo] = field(default_factory=list)
    meta: dict = field(default_factory=dict)   # site, date, altitude_m, notes

    @property
    def dir(self) -> str:
        return os.path.join(paths.SURVEYS_DIR, self.name)

    def path(self, photo: Photo) -> str:
        return os.path.join(self.dir, photo.file)

    def label_path(self, photo: Photo) -> str | None:
        return os.path.join(self.dir, photo.label) if photo.label else None

    def boxes(self, photo: Photo) -> list[list[int]] | None:
        """Annotated boxes for a photo, or None when it came without a label file."""
        path = self.label_path(photo)
        return read_annotations(path) if path and os.path.exists(path) else None

    @property
    def labelled(self) -> int:
        return sum(1 for p in self.photos if p.label)

    def photo(self, file: str) -> Photo | None:
        return next((p for p in self.photos if p.file == file), None)

    def save(self) -> None:
        payload = asdict(self)
        write_json(os.path.join(self.dir, "survey.json"), payload)


def load(name: str) -> Survey | None:
    payload = read_json(os.path.join(paths.SURVEYS_DIR, name, "survey.json"))
    if not payload:
        return None
    photos = [Photo(**p) for p in payload.pop("photos", [])]
    return Survey(photos=photos, **payload)


def all_surveys() -> list[Survey]:
    if not os.path.isdir(paths.SURVEYS_DIR):
        return []
    found = [load(e.name) for e in os.scandir(paths.SURVEYS_DIR) if e.is_dir()]
    return sorted((s for s in found if s), key=lambda s: s.created, reverse=True)


# ── Import ────────────────────────────────────────────────────────────────────

@dataclass
class Scan:
    folder: str
    images: list[str]
    labels: dict[str, str]        # image path -> label file
    has_index: bool
    total_bytes: int
    subfolders: int


def scan(folder: str, recursive: bool = False) -> Scan:
    """Find the photographs (and any label files beside them) in a folder."""
    images, labels, index, total, subdirs = [], {}, False, 0, 0
    walker = os.walk(folder, followlinks=True) if recursive else [
        (folder, [], sorted(os.listdir(folder)))]
    for directory, dirs, files in walker:
        dirs.sort()
        if directory != folder:
            subdirs += 1
        for name in sorted(files):
            full = os.path.join(directory, name)
            if name == "image_info.csv":
                index = True
            if not name.lower().endswith(IMAGE_EXTS) or not os.path.isfile(full):
                continue
            images.append(full)
            total += os.path.getsize(full)
            label = find_annotation(directory, os.path.splitext(name)[0])
            if label:
                labels[full] = label
    return Scan(folder, images, labels, index, total, subdirs)


def valid_name(name: str) -> str | None:
    """Why `name` cannot be a survey name, or None when it can."""
    if not name:
        return "enter a name"
    if slug(name) != name:
        return "use letters, digits, '-', '_' and '.' only"
    if os.path.exists(os.path.join(paths.SURVEYS_DIR, name)):
        return f"a survey called {name} already exists"
    return None


def sample_held_out(n: int, collections: list[str] | None = None, seed: int = 0) -> Scan:
    """`n` photographs the annotators held out for testing, with their labels.

    None of them was ever trained on, so a survey made from them measures a model on whole
    photographs the way it would meet a new flight.  Drawn round-robin across the chosen
    collections, so a small sample still covers them all.  Read through the dataset
    adapter, which knows every collection's spreadsheet layout and label naming.
    """
    import random
    from data_prep import config
    from pipeline import adapters
    adapter = adapters.get("mdc", root=config.DEFAULT_DATASET_ROOT, read_exif=False)
    pool: dict[str, list] = {}
    for photo in adapter.usable_photographs():
        if photo.split_role == "test" and photo.annotation_path and \
                (not collections or photo.collection in collections):
            pool.setdefault(photo.collection, []).append(photo)
    rng = random.Random(seed)
    for photos in pool.values():
        rng.shuffle(photos)
    picked = []
    while len(picked) < n and any(pool.values()):
        for name in sorted(pool):
            if pool[name] and len(picked) < n:
                picked.append(pool[name].pop())
    return Scan(folder=f"held-out test photographs of {os.path.basename(config.DEFAULT_DATASET_ROOT)}",
                images=[p.path for p in picked],
                labels={p.path: p.annotation_path for p in picked},
                has_index=False, total_bytes=sum(os.path.getsize(p.path) for p in picked),
                subfolders=len({p.collection for p in picked}))


def create(name: str, found: Scan, mode: str = "link", meta: dict | None = None,
           progress=None, prefix_folder: bool = False) -> tuple[Survey, list[str]]:
    """Register the scanned photographs as a survey.  Returns (survey, skipped files).

    `prefix_folder` names every photo after its folder too (Bird_J_DJI_0001.JPG), for
    surveys gathered from several collections whose file names repeat.
    """
    survey = Survey(name=name, created=time.time(), source=found.folder, mode=mode,
                    meta=dict(meta or {}))
    img_dir = os.path.join(survey.dir, "images")
    lbl_dir = os.path.join(survey.dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    used: set[str] = set()
    skipped: list[str] = []
    for i, src in enumerate(found.images):
        if progress:
            progress(i, len(found.images))
        try:
            with Image.open(src) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001 — any unreadable file is reported, not fatal
            skipped.append(src)
            continue
        # Flattening sub-folders can collide (two flights both have DJI_0001.JPG), so a
        # clash takes its folder name as a prefix.
        base = os.path.basename(src)
        parent = slug(os.path.basename(os.path.dirname(src)))
        if prefix_folder:
            base = f"{parent}_{base}"
        elif base in used:
            base = f"{parent}__{base}"
        used.add(base)
        dest = os.path.join(img_dir, base)
        if mode == "copy":
            shutil.copy2(src, dest)
        else:
            # to the file itself, so removing an intermediate symlinked folder (a data
            # selection, say) cannot break the survey
            os.symlink(os.path.realpath(src), dest)
        label_rel = None
        if src in found.labels:
            os.makedirs(lbl_dir, exist_ok=True)
            label_name = os.path.splitext(base)[0] + ".txt"
            shutil.copy2(found.labels[src], os.path.join(lbl_dir, label_name))
            label_rel = f"labels/{label_name}"
        survey.photos.append(Photo(file=f"images/{base}", width=width, height=height,
                                   label=label_rel, source=os.path.realpath(src)))
    survey.save()
    return survey, skipped


def delete(survey: Survey) -> None:
    """Remove a survey's folder (the original photographs are never touched)."""
    shutil.rmtree(survey.dir, ignore_errors=True)
