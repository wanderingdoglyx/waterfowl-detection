"""
Mirror the shared COCO crops into the point-CSV layout MegaDetector-Overhead / OWL
expects, without touching the originals.

OWL is a *point* detector trained from `animaloc`'s CSVDataset / FolderDataset, which
read a flat folder of images plus a `gt.csv`:

    mdo_data/
      train/
        images/          ← symlinks to crops/<Bird_X>/<crop>.JPG (flat, basenames)
        gt.csv           ← columns: images,x,y,labels   (one row per bird)
      val/   (same)
      test/  (same)

Each box in the shared COCO crops becomes a single point at the box centre
(x = x + w/2, y = y + h/2), label 1 (bird).  Crops with no birds are still symlinked
but get no rows in gt.csv — FolderDataset then treats them as background images, so
false positives on empty crops are counted honestly during evaluation.

Crop filenames carry a per-dataset letter prefix ("A_...", "B_...") so basenames are
globally unique; flattening into one folder per split is collision-free.
"""

from __future__ import annotations

import csv
import json
import os
import shutil

from data_prep import config

SPLITS = ("train", "val", "test")


def _relink(src_abs: str, dst_abs: str) -> None:
    """Create/refresh a symlink dst_abs -> src_abs."""
    if os.path.islink(dst_abs) or os.path.exists(dst_abs):
        os.remove(dst_abs)
    os.symlink(src_abs, dst_abs)


def build_mdo_dataset(
    crops_img_dir: str = config.CROPS_IMG_DIR,
    crops_json_dir: str = config.CROPS_JSON_DIR,
    out_dir: str = config.MDO_DATA_DIR,
) -> tuple[str, dict]:
    """
    Build the OWL point-CSV mirror for every split that has a coco.json.

    Returns (out_dir, counts) where counts maps split -> number of images (crops).
    """
    counts: dict[str, int] = {}

    for split in SPLITS:
        coco_path = os.path.join(crops_json_dir, split, "coco.json")
        if not os.path.exists(coco_path):
            counts[split] = 0
            continue

        with open(coco_path) as f:
            coco = json.load(f)

        # Rebuild the flat image folder fresh — FolderDataset scans it wholesale, so a
        # stale symlink from a previous prepare would silently enter the dataset.
        img_dir = os.path.join(out_dir, split, "images")
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)

        # image_id -> (basename, width, height); symlink each crop into the flat folder
        id_to_meta: dict[int, tuple[str, int, int]] = {}
        for img in coco["images"]:
            rel = img["file_name"]                       # e.g. "Bird_A/A_DJI_0001_0_0.JPG"
            base = os.path.basename(rel)
            id_to_meta[img["id"]] = (base, img["width"], img["height"])
            _relink(os.path.join(crops_img_dir, rel), os.path.join(img_dir, base))

        # one point per box (box centre).  Clamp to [0, w-1] × [0, h-1]: a box flush
        # against the crop edge can round to exactly w/h, which albumentations rejects
        # as an out-of-range keypoint (valid range is the half-open [0, size)).
        rows: list[tuple[str, int, int, int]] = []
        for ann in coco["annotations"]:
            meta = id_to_meta.get(ann["image_id"])
            if meta is None:
                continue
            base, w_img, h_img = meta
            x, y, bw, bh = ann["bbox"]
            cx = min(w_img - 1, max(0, int(round(x + bw / 2.0))))
            cy = min(h_img - 1, max(0, int(round(y + bh / 2.0))))
            rows.append((base, cx, cy, 1))

        gt_csv = os.path.join(out_dir, split, "gt.csv")
        with open(gt_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["images", "x", "y", "labels"])
            writer.writerows(rows)

        counts[split] = len(coco["images"])

    return out_dir, counts


def split_paths(split: str, out_dir: str = config.MDO_DATA_DIR) -> tuple[str, str]:
    """Return (images_dir, gt_csv) for a split of the mirrored dataset."""
    base = os.path.join(out_dir, split)
    return os.path.join(base, "images"), os.path.join(base, "gt.csv")


if __name__ == "__main__":
    d, c = build_mdo_dataset()
    print(f"MDO point dataset at {d}")
    print(f"Crops per split: {c}")
