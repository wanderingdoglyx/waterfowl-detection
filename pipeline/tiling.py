"""
Cutting photographs into crops — once, for every model, from canonical records.

This is the core's half of the bargain struck in `pipeline/adapters`: the adapter knows
where a programme's photographs and boxes live, and the tiler knows how this project cuts
them up.  Neither knows the other's business, so a new survey programme is tiled by code
that was never changed for it.

What it does that `data_prep.prepare_data.prepare()` does not:

* it is **source-agnostic** — it tiles whatever an adapter yields, Missouri or Arkansas;
* it **writes the survey conditions into every crop's COCO record** (photograph id,
  collection, site, date, altitude, ground sample distance, habitat, weather), so nothing
  downstream needs to parse a file name or re-read a spreadsheet;
* it **honours a split specification**, so the same call can build the full corpus or one
  experiment's selection;
* it **skips photographs that are already tiled** (`pre_tiled`), rather than re-cutting a
  collection that arrived as tiles.

The pixels themselves come from `data_prep.prepare_data.crop_image`, unchanged: the
published crops and any crop made here are cut by the same function with the same stride,
so results stay comparable with everything already committed.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from data_prep import config
from data_prep.prepare_data import crop_image
from pipeline.adapters.base import DatasetAdapter
from pipeline.crops import photograph_fields
from pipeline.records import Photograph
from pipeline.splits import SplitSpec

SPLITS = ("train", "val", "test")


def _letter(collection: str) -> str:
    """The crop-name prefix for a collection: "Bird_A" -> "A", "BayouMeto" -> "BayouMeto".

    Kept for compatibility with the committed crop names, which every cached prediction
    and every published table is keyed to.
    """
    parts = collection.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0] == "Bird" else collection


def build(adapter: DatasetAdapter, spec: SplitSpec,
          photos: Iterable[Photograph] | None = None,
          crops_img_dir: str | None = None, crops_json_dir: str | None = None,
          crop_size: int | None = None, overlap: float | None = None,
          force: bool = False, verbose: bool = True) -> dict:
    """Tile every photograph the spec assigns, and write one COCO file per split.

    Returns {split: {"crops": n, "birds": n, "photographs": n}}.  Existing crop images are
    reused unless `force` is set — cutting 2,095 photographs takes far longer than reading
    the annotations, and the pixels do not change.
    """
    crops_img_dir = crops_img_dir or config.CROPS_IMG_DIR
    crops_json_dir = crops_json_dir or config.CROPS_JSON_DIR
    crop_size = crop_size or config.CROP_SIZE
    overlap = config.CROP_OVERLAP if overlap is None else overlap
    stride = int(crop_size * (1 - overlap / 2))

    photos = list(photos if photos is not None else adapter.usable_photographs())
    by_id = {p.photo_id: p for p in photos}

    records: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for photo_id, split in sorted(spec.assignment.items()):
        photo = by_id.get(photo_id)
        if photo is None or split not in records:
            continue

        out_dir = os.path.join(crops_img_dir, photo.collection)
        os.makedirs(out_dir, exist_ok=True)
        boxes = [[int(v) for v in a.coords] for a in adapter.annotations(photo)
                 if a.geometry == "box"]

        if photo.pre_tiled:
            # Already a tile: one "crop" that is the photograph itself, so a pre-tiled
            # collection is not cut again into 512-px pieces of a 512-px image.
            crops = [{"file_name": f"{photo.collection}/{os.path.basename(photo.path)}",
                      "boxes": boxes, "w": photo.width_px, "h": photo.height_px,
                      "row": 0, "col": 0}]
        else:
            crops = crop_image(image_path=photo.path, boxes=boxes, crop_size=crop_size,
                               stride=stride, out_img_dir=out_dir,
                               ds_letter=_letter(photo.collection), stem=photo.stem,
                               ds_name=photo.collection)

        fields = photograph_fields(photo)
        for crop in crops:
            crop.update(fields)
            records[split].append(crop)

        if verbose and len(records[split]) % 5000 < len(crops):
            print(f"  {split}: {len(records[split]):,} crops", flush=True)

    summary = {}
    for split in SPLITS:
        images, annotations = [], []
        ann_id = 1
        for image_id, record in enumerate(records[split], start=1):
            image = {"id": image_id, "file_name": record["file_name"],
                     "width": record["w"], "height": record["h"]}
            image.update({k: v for k, v in record.items()
                          if k not in ("file_name", "boxes", "w", "h")})
            images.append(image)
            for x1, y1, x2, y2 in record["boxes"]:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "ignore": 0})
                ann_id += 1

        split_dir = os.path.join(crops_json_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "coco.json"), "w") as f:
            json.dump({"type": "instances", "images": images, "annotations": annotations,
                       "categories": [{"id": 1, "name": "bird", "supercategory": "none"}]}, f)

        summary[split] = {"crops": len(images), "birds": len(annotations),
                          "photographs": len({im.get("photo_id") for im in images})}
    return summary


def mirrors(crops_json_dir: str | None = None, yolo_dir: str | None = None,
            mdo_dir: str | None = None) -> dict:
    """Build every framework's on-disk layout from the COCO splits, in one call.

    Ultralytics (YOLOv5 / YOLO11 / YOLO26) needs a YOLO-format tree; OWL needs a point-CSV
    tree; Detectron2 and super-gradients read the COCO files directly and need nothing.
    Each builder is the model package's own, called here rather than copied.
    """
    from megadetector_overhead.dataset import build_mdo_dataset
    from yolov5.dataset import build_yolo_dataset

    saved = (config.CROPS_JSON_DIR, config.YOLOV5_DATA_DIR, config.MDO_DATA_DIR)
    try:
        if crops_json_dir:
            config.CROPS_JSON_DIR = crops_json_dir
        if yolo_dir:
            config.YOLOV5_DATA_DIR = yolo_dir
        if mdo_dir:
            config.MDO_DATA_DIR = mdo_dir
        _, yolo_counts = build_yolo_dataset()
        _, mdo_counts = build_mdo_dataset(crops_json_dir=config.CROPS_JSON_DIR,
                                          out_dir=config.MDO_DATA_DIR)
        return {"yolo": yolo_counts, "mdo": mdo_counts}
    finally:
        config.CROPS_JSON_DIR, config.YOLOV5_DATA_DIR, config.MDO_DATA_DIR = saved
