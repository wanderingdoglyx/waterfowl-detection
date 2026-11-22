"""
Choose the data for an experiment once, for every model.

Today each model package prepares its own data: `--prepare` re-crops everything and each
framework builds its own mirror, and only Faster R-CNN can restrict the run to a subset of
collections (`--datasets Bird_A Bird_B`).  So "train all eight models on Bird_G + Bird_J"
is not expressible — the other seven would silently train on the whole corpus, and the
comparison would be meaningless.

A `DataSelection` is that choice, written once — and the choice is simply **a folder**:
put the collections (or the images) you want in one, and the pipeline reads it.

    selection = DataSelection(name="g_and_j", root="dataset_g_and_j")
    spec, photos = resolve(selection, adapters={"mdc": MdcAdapter(root="dataset_g_and_j")})
    materialise(selection, spec, photos, mirrors=True)

`materialise` writes, under `output/experiments/<name>/`:

    selection.json              the choice, its checksum, and what it resolved to
    crops_json/{split}/coco.json  the COCO splits, filtered from the committed ones
    yolo/                       the Ultralytics mirror (if mirrors=True)
    mdo/                        the OWL point-CSV mirror (if mirrors=True)

Every model then reads that selection with no code change, because `data_prep.config`
redirects its data paths when an experiment is active:

    WATERFOWL_EXPERIMENT=g_and_j ./yolov5/main.py --train
    WATERFOWL_EXPERIMENT=g_and_j ./faster_rcnn/main.py --train

Two properties matter for comparability, and both are enforced here:

* **Crops are filtered, never re-cut.**  The committed crops and their COCO records are
  the published ones; a selection takes a subset of them, so a subset experiment is
  comparable with the full-corpus results rather than being a different tiling.
* **Train/validation membership is inherited by default.**  The project's split is a
  seeded shuffle over the assembled pool, so filtering the pool first would reshuffle
  every remaining photograph.  `inherit_roles=True` keeps each photograph in the split it
  is already in; set it to False to derive a fresh split over the subset instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable

from data_prep import config
from pipeline import crops as crop_tools
from pipeline import splits
from pipeline.adapters import get as get_adapter
from pipeline.records import Photograph

SPLITS = ("train", "val", "test")


@dataclass
class DataSelection:
    """Which photographs take part in an experiment."""

    name: str
    root: str | None = None                           # the folder the data was read from
    sources: list[str] = field(default_factory=lambda: ["mdc"])
    inherit_roles: bool = True
    val_fraction: float = config.VAL_FRACTION
    seed: int = config.RANDOM_SEED
    notes: str | None = None

    def directory(self) -> str:
        return os.path.join(config.EXPERIMENTS_DIR, self.name)


def resolve(selection: DataSelection, adapters: dict | None = None
            ) -> tuple[splits.SplitSpec, list[Photograph]]:
    """Turn a selection into a split specification and the photographs it keeps.

    With `inherit_roles`, roles come from the project's own derivation over the *whole*
    source and are then restricted, so a photograph never changes split because its
    neighbours were filtered out.
    """
    kept: list[Photograph] = []
    assignment: dict[str, str] = {}

    for source in selection.sources:
        adapter = adapters[source] if adapters else get_adapter(source)
        # The folder is the filter: whatever the adapter finds there takes part.
        chosen = [p for p in adapter.usable_photographs()]
        kept.extend(chosen)
        fresh = splits.provider_marked(chosen, selection.val_fraction, selection.seed)

        if not selection.inherit_roles:
            assignment.update(fresh.assignment)
            continue

        # Inherited roles must come from the split over the WHOLE corpus, not over the
        # folder: the split is a seeded shuffle of the pool, so deriving it from a subset
        # would move photographs between train and validation.  A photograph the full
        # corpus does not contain (new data dropped into a folder) falls back to the
        # subset's own split.
        baseline = splits.provider_marked(_reference_photographs(source, adapter),
                                          selection.val_fraction, selection.seed)
        for p in chosen:
            assignment[p.photo_id] = baseline.assignment.get(p.photo_id,
                                                             fresh.assignment[p.photo_id])

    spec = splits.SplitSpec(
        name=selection.name, sources=list(selection.sources), assignment=assignment,
        val_fraction=selection.val_fraction, seed=selection.seed,
        rule=("inherited from the full-corpus split" if selection.inherit_roles
              else "derived over the selected photographs"),
    )
    return spec, kept


def _reference_photographs(source: str, adapter) -> list[Photograph]:
    """Every photograph of a source at its default location, for inheriting split roles."""
    default_root = getattr(config, "DEFAULT_DATASET_ROOT", config.DATASET_ROOT)
    if getattr(adapter, "root", None) == default_root:
        return list(adapter.usable_photographs())
    try:
        # Optics are irrelevant to a split, so skip EXIF: this is the fast path.
        reference = get_adapter(source, root=default_root, read_exif=False)
    except TypeError:
        try:
            reference = get_adapter(source, root=default_root)
        except Exception:
            return list(adapter.usable_photographs())
    except Exception:
        return list(adapter.usable_photographs())
    return list(reference.usable_photographs())


def _source_crops(crops_json_dir: str) -> dict[str, list[dict]]:
    """{split: [image records]} from the committed COCO files, with their annotations."""
    out = {}
    for split in SPLITS:
        path = os.path.join(crops_json_dir, split, "coco.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            out[split] = json.load(f)
    return out


def materialise(selection: DataSelection, spec: splits.SplitSpec,
                photos: Iterable[Photograph], crops_json_dir: str | None = None,
                out_dir: str | None = None, mirrors: bool = False) -> dict:
    """Write the experiment's COCO splits (and optionally the framework mirrors).

    Crops are taken from the committed COCO files rather than re-cut, so the pixels and
    the boxes are exactly those every published result was computed on.
    """
    crops_json_dir = crops_json_dir or config.CROPS_JSON_DIR
    out_dir = out_dir or selection.directory()
    by_id = {p.photo_id: p for p in photos}
    source = _source_crops(crops_json_dir)

    # Which output split does each crop belong to?  A crop follows its photograph.
    per_split: dict[str, dict] = {s: {"images": [], "annotations": []} for s in SPLITS}
    moved = 0
    for split, coco in source.items():
        anns_by_image: dict[int, list] = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        for image in coco["images"]:
            parsed = crop_tools.parse_crop_name(image["file_name"])
            if parsed is None:
                continue
            collection, stem = parsed[0], parsed[1]
            photo_id = None
            for src in selection.sources:
                candidate = f"{src}/{collection}/{stem}"
                if candidate in by_id:
                    photo_id = candidate
                    break
            if photo_id is None:
                continue
            target = spec.assignment.get(photo_id)
            if target is None:
                continue
            if target != split:
                moved += 1
            record = dict(image)
            record["photo_id"] = photo_id
            # The metadata travels with the crop, so nothing downstream has to parse a
            # file name or re-read a spreadsheet (Task 4.3, step 3).
            record.update(crop_tools.photograph_fields(by_id[photo_id]))
            per_split[target]["images"].append(record)
            per_split[target]["annotations"].extend(anns_by_image.get(image["id"], []))

    # Ids must stay unique inside each output file; they only collide if photographs
    # moved between splits, which happens with inherit_roles=False.
    summary = {"name": selection.name, "splits": {}, "collections": {}, "moved_crops": moved}
    os.makedirs(out_dir, exist_ok=True)

    for split in SPLITS:
        images = per_split[split]["images"]
        annotations = per_split[split]["annotations"]
        ids = [im["id"] for im in images]
        if len(ids) != len(set(ids)):
            remap = {}
            for new_id, image in enumerate(images, start=1):
                remap[image["id"]] = new_id
                image["source_image_id"] = image["id"]
                image["id"] = new_id
            for a in annotations:
                a["image_id"] = remap[a["image_id"]]

        split_dir = os.path.join(out_dir, "crops_json", split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "coco.json"), "w") as f:
            json.dump({"type": "instances", "images": images, "annotations": annotations,
                       "categories": [{"id": 1, "name": "bird", "supercategory": "none"}]}, f)

        photographs = {im["photo_id"] for im in images}
        summary["splits"][split] = {"crops": len(images), "birds": len(annotations),
                                    "photographs": len(photographs)}
        for im in images:
            key = im["collection"]
            entry = summary["collections"].setdefault(key, {s: 0 for s in SPLITS})
            entry[split] += 1

    if mirrors:
        summary["mirrors"] = build_mirrors(out_dir)

    payload = {"selection": asdict(selection), "checksum": spec.checksum,
               "rule": spec.rule, "summary": summary}
    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return summary


def build_mirrors(out_dir: str) -> dict:
    """Build the Ultralytics and OWL mirrors for an experiment's COCO splits.

    The framework builders read their paths from `data_prep.config` at call time, so
    pointing config at the experiment is enough — the builders themselves are untouched.
    """
    from megadetector_overhead.dataset import build_mdo_dataset
    from yolov5.dataset import build_yolo_dataset

    saved = (config.CROPS_JSON_DIR, config.YOLOV5_DATA_DIR, config.MDO_DATA_DIR)
    try:
        config.CROPS_JSON_DIR = os.path.join(out_dir, "crops_json")
        config.YOLOV5_DATA_DIR = os.path.join(out_dir, "yolo")
        config.MDO_DATA_DIR = os.path.join(out_dir, "mdo")
        _, yolo_counts = build_yolo_dataset()
        _, mdo_counts = build_mdo_dataset(crops_json_dir=config.CROPS_JSON_DIR,
                                          out_dir=config.MDO_DATA_DIR)
        return {"yolo": yolo_counts, "mdo": mdo_counts}
    finally:
        config.CROPS_JSON_DIR, config.YOLOV5_DATA_DIR, config.MDO_DATA_DIR = saved


def load(name: str) -> dict:
    """Read a materialised experiment's selection.json."""
    with open(os.path.join(config.EXPERIMENTS_DIR, name, "selection.json")) as f:
        return json.load(f)


def available() -> list[str]:
    if not os.path.isdir(config.EXPERIMENTS_DIR):
        return []
    return sorted(d for d in os.listdir(config.EXPERIMENTS_DIR)
                  if os.path.exists(os.path.join(config.EXPERIMENTS_DIR, d, "selection.json")))


def format_summary(summary: dict) -> str:
    lines = [f"{summary['name']}:"]
    for split in SPLITS:
        s = summary["splits"].get(split, {})
        lines.append(f"  {split:6s} {s.get('crops', 0):7,d} crops  "
                     f"{s.get('birds', 0):8,d} birds  "
                     f"{s.get('photographs', 0):5,d} photographs")
    if summary["collections"]:
        lines.append("  collections: " + ", ".join(
            f"{k} ({sum(v.values()):,})" for k, v in sorted(summary["collections"].items())))
    if summary.get("moved_crops"):
        lines.append(f"  {summary['moved_crops']:,} crops changed split "
                     "(roles were derived over the subset, not inherited)")
    return "\n".join(lines)
