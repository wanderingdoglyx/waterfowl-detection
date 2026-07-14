"""
Build an Ultralytics-format YOLOv5 dataset from the shared COCO crops.

The rest of the project stores crops + COCO json (the same data Faster R-CNN and
YOLO-NAS train on):

    crops/Bird_A/A_DJI_0001_0_0.JPG               (images, one folder per dataset)
    output/crops_json/{train,val,test}/coco.json  (annotations; file_name relative
                                                    to the crops root: "Bird_A/..JPG")

Ultralytics instead wants its own on-disk layout — an ``images/<split>/`` tree and a
parallel ``labels/<split>/`` tree of YOLO txt files (one ``class cx cy w h`` line per
box, all normalised 0-1) — discovered through a ``data.yaml``.  Rather than copy
~125k images, we mirror them with **symlinks** and write fresh label files:

    output/yolov5_data/
        data.yaml
        images/{train,val,test}/Bird_A__A_DJI_0001_0_0.JPG  -> symlink to the crop
        labels/{train,val,test}/Bird_A__A_DJI_0001_0_0.txt

The "/" in each COCO file_name is flattened to "__" so the per-dataset folders
collapse into one split directory without name collisions.  Building is idempotent:
existing symlinks are left in place and labels are rewritten (cheap).

Single foreground class: bird -> class index 0 (COCO category_id 1).
"""

from __future__ import annotations

import json
import os

import yaml

import data_prep.config as config

# Single foreground class (matches faster_rcnn.dataset.THING_CLASSES and
# yolo_nas.dataset.CLASS_NAMES).  COCO category_id 1 -> YOLO class index 0.
CLASS_NAMES = ["bird"]

SPLITS = ("train", "val", "test")


def _flat_name(file_name: str) -> str:
    """"Bird_A/foo.JPG" -> "Bird_A__foo.JPG" (collapse the per-dataset folder)."""
    return file_name.replace("/", "__")


def _coco_json(split: str) -> str:
    return os.path.join(config.CROPS_JSON_DIR, split, "coco.json")


def data_yaml_path() -> str:
    return os.path.join(config.YOLOV5_DATA_DIR, "data.yaml")


def _convert_split(split: str) -> int:
    """
    Symlink images and write YOLO label files for one split.

    Returns the number of images in the split (0 if its coco.json is missing).
    """
    coco_path = _coco_json(split)
    if not os.path.exists(coco_path):
        return 0

    with open(coco_path) as f:
        coco = json.load(f)

    img_dir = os.path.join(config.YOLOV5_DATA_DIR, "images", split)
    lbl_dir = os.path.join(config.YOLOV5_DATA_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # image_id -> (file_name, flat_name, width, height)
    images = {
        img["id"]: (img["file_name"], _flat_name(img["file_name"]),
                    img["width"], img["height"])
        for img in coco["images"]
    }

    # image_id -> list of normalised "0 cx cy w h" lines
    lines: dict[int, list[str]] = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        _fn, _flat, w, h = images[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]                 # COCO: top-left x,y + width,height (px)
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        lines[ann["image_id"]].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    for img_id, (file_name, flat, _w, _h) in images.items():
        # Symlink the real crop into images/<split>/ (skip if already linked).
        src = os.path.join(config.CROPS_IMG_DIR, file_name)
        dst = os.path.join(img_dir, flat)
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

        # Write the label file.  Images with no boxes get an empty .txt — that is
        # how Ultralytics encodes a legitimate background/empty crop.
        stem = os.path.splitext(flat)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines[img_id]))

    return len(images)


def build_yolo_dataset() -> tuple[str, dict]:
    """
    Mirror every available split into Ultralytics layout and write data.yaml.

    Returns (data_yaml_path, {split: n_images}).  Splits whose coco.json is
    missing are skipped, so this works for train-only or test-only situations.
    """
    os.makedirs(config.YOLOV5_DATA_DIR, exist_ok=True)

    counts = {split: _convert_split(split) for split in SPLITS}

    # data.yaml: paths are relative to `path`; Ultralytics derives label paths by
    # swapping the `images` component for `labels`.
    data = {
        "path":  os.path.abspath(config.YOLOV5_DATA_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    with open(data_yaml_path(), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return data_yaml_path(), counts
