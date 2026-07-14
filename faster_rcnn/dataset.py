"""
Register the pre-cropped waterfowl splits with Detectron2's DatasetCatalog.

Call register_datasets() once before building a trainer or running inference.
It registers three datasets: waterfowl_train, waterfowl_val, waterfowl_test,
each backed by the COCO JSON produced by prepare_data.py.

Pass `included` to restrict training/val to specific dataset subfolders:
    counts = register_datasets(included={"Bird_A", "Bird_B"})
A filtered COCO JSON is written alongside the original and registered in its place.
The test split is never filtered — it always uses the full test data.
"""

import json
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

import data_prep.config as config

DATASET_NAMES = {
    "train": "waterfowl_train",
    "val":   "waterfowl_val",
    "test":  "waterfowl_test",
}

# Single foreground class
THING_CLASSES = ["bird"]


def _filter_coco(coco: dict, included: set) -> dict:
    """Return a copy of coco restricted to images from the given dataset folders."""
    kept_images = [
        img for img in coco["images"]
        if img["file_name"].split("/")[0] in included
    ]
    kept_ids = {img["id"] for img in kept_images}
    kept_anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
    return {**coco, "images": kept_images, "annotations": kept_anns}


def _filtered_json_path(split: str, included: set, crops_json_dir: str) -> str:
    tag = "_".join(sorted(included))
    return os.path.join(crops_json_dir, split, f"coco_{tag}.json")


def register_datasets(
    crops_img_dir:  str        = config.CROPS_IMG_DIR,
    crops_json_dir: str        = config.CROPS_JSON_DIR,
    included:       set | None = None,
) -> dict:
    """
    Register waterfowl_train / waterfowl_val / waterfowl_test and return
    a dict of {split: n_images} for the registered (possibly filtered) splits.

    included : set of subfolder names to include in train and val
               (e.g. {"Bird_A", "Bird_B"}). None or empty = use all datasets.
               The test split is never filtered.

    When included is set, a filtered COCO JSON is written next to the original
    (e.g. coco_Bird_A_Bird_B.json) and used for registration instead.
    The original coco.json is never modified.
    """
    counts = {}
    for split, name in DATASET_NAMES.items():
        if name in DatasetCatalog.list():
            continue

        base_json = os.path.join(crops_json_dir, split, "coco.json")
        if not os.path.exists(base_json):
            raise FileNotFoundError(
                f"COCO JSON not found: {base_json}\n"
                "Run './faster_rcnn/main.py --prepare' first."
            )

        if included and split in ("train", "val"):
            json_path = _filtered_json_path(split, included, crops_json_dir)
            with open(base_json) as f:
                coco = _filter_coco(json.load(f), included)
            with open(json_path, "w") as f:
                json.dump(coco, f)
            n = len(coco["images"])
            print(f"  [{split}] {sorted(included)}: {n} crops, "
                  f"{len(coco['annotations'])} annotations → {json_path}")
        else:
            json_path = base_json
            with open(json_path) as f:
                n = len(json.load(f)["images"])

        counts[split] = n
        register_coco_instances(name, {}, json_path, crops_img_dir)
        MetadataCatalog.get(name).thing_classes = THING_CLASSES

    return counts
