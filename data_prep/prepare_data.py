"""
Pre-process all datasets under DATASET_ROOT and output COCO-format JSON files
that Detectron2 can consume directly via register_coco_instances.

Output layout:
  crops/
    Bird_A/          ← one sub-folder per dataset (images only)
      A_DJI_0001_0_0.JPG
      A_DJI_0001_0_460.JPG
      ...
    Bird_B/
    Bird_G/
    ...

  output/crops_json/
    train/coco.json  ← file_name in JSON is relative: "Bird_A/A_DJI_0001_0_0.JPG"
    val/coco.json
    test/coco.json

Crop filename convention (matches BirdI_faster reference dataset):
  {DatasetLetter}_{ImageStem}_{row_offset}_{col_offset}.JPG
  - DatasetLetter: second segment of folder name split on "_"
    (Bird_A → A, Bird_I_complete → I)
  - row_offset / col_offset: pixel coordinates of the crop's top-left corner
    (y first, then x, no zero-padding)
  - stride = int(crop_size × (1 − overlap/2)); with overlap=0.20 → stride=460 px

Annotation format (all datasets): bird,x1,y1,x2,y2  (xyxy, integer pixels)
COCO bbox format: [x, y, width, height]  (converted from xyxy)
"""

import csv
import json
import os
import random

from PIL import Image

from data_prep import config


def find_image(dataset_dir: str, stem: str) -> str | None:
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        p = os.path.join(dataset_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_annotations(txt_path: str) -> list[list[int]]:
    """Parse 'bird,x1,y1,x2,y2' → [[x1,y1,x2,y2], ...]."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
            except ValueError:
                continue
    return boxes


def _tile_origins(dim: int, stride: int, crop_size: int) -> list[int]:
    """Return tile start positions matching the reference dataset layout."""
    starts = list(range(0, max(dim - crop_size, 0) + 1, stride))
    edge = max(dim - crop_size, 0)
    if not starts:
        return [0]
    if starts[-1] != edge:
        starts.append(edge)
    return starts


def crop_image(
    image_path: str,
    boxes: list[list[int]],
    crop_size: int,
    stride: int,
    out_img_dir: str,   # dataset-specific folder, e.g. crops/Bird_A/
    ds_letter: str,     # e.g. "A" from "Bird_A", "I" from "Bird_I_complete"
    stem: str,          # original image filename stem, e.g. "DJI_0001"
    ds_name: str,       # folder name, e.g. "Bird_A" — used for the relative file_name
) -> list[dict]:
    """
    Slide a 512×512 window with the given stride and produce crop patches.
    Includes any box that has any intersection with the crop (matches original code).

    Filename format: {ds_letter}_{stem}_{row_offset}_{col_offset}.JPG
    Returns list of:
      {
        'file_name': str,   # relative path from crops root, e.g. "Bird_A/A_DJI_0001_0_0.JPG"
        'boxes':     [[x1,y1,x2,y2], ...],
        'w': int, 'h': int
      }
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    records = []
    for cy in _tile_origins(H, stride, crop_size):
        for cx in _tile_origins(W, stride, crop_size):
            cx2, cy2 = cx + crop_size, cy + crop_size
            crop_boxes = []
            for bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(bx1, cx), max(by1, cy)
                ix2, iy2 = min(bx2, cx2), min(by2, cy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                crop_boxes.append([ix1 - cx, iy1 - cy, ix2 - cx, iy2 - cy])

            # {Letter}_{Stem}_{row}_{col}.JPG  (row=y, col=x, no zero-padding)
            fname    = f"{ds_letter}_{stem}_{cy}_{cx}.JPG"
            rel_path = f"{ds_name}/{fname}"
            img.crop((cx, cy, cx2, cy2)).save(
                os.path.join(out_img_dir, fname), quality=95
            )
            records.append({
                "file_name": rel_path,
                "boxes":     crop_boxes,
                "w":         crop_size,
                "h":         crop_size,
            })
    return records


def build_coco_json(records: list[dict]) -> dict:
    """Convert crop records to COCO-format annotation dict."""
    categories = [{"id": 1, "name": "bird", "supercategory": "none"}]
    images, annotations = [], []
    ann_id = 1

    for img_id, rec in enumerate(records, start=1):
        images.append({
            "id":        img_id,
            "file_name": rec["file_name"],
            "width":     rec["w"],
            "height":    rec["h"],
        })
        for bx1, by1, bx2, by2 in rec["boxes"]:
            bw, bh = bx2 - bx1, by2 - by1
            annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": 1,
                "bbox":        [bx1, by1, bw, bh],
                "area":        bw * bh,
                "iscrowd":     0,
                "ignore":      0,
            })
            ann_id += 1

    return {
        "type":        "instances",
        "images":      images,
        "annotations": annotations,
        "categories":  categories,
    }


def discover_datasets(root: str, excluded: set) -> list[str]:
    dirs = []
    for name in sorted(os.listdir(root)):
        if name in excluded:
            continue
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "image_info.csv")):
            dirs.append(d)
    return dirs


def read_split(dataset_dir: str) -> tuple[list, list]:
    """Returns (train_items, test_items) where each item is (dataset_dir, image_name)."""
    train, test = [], []
    with open(os.path.join(dataset_dir, "image_info.csv")) as f:
        for row in csv.DictReader(f):
            name = row["image_name"]
            stem = os.path.splitext(name)[0]
            if find_image(dataset_dir, stem) is None:
                continue
            if not os.path.exists(os.path.join(dataset_dir, stem + ".txt")):
                continue
            bucket = train if row.get("bbox_split_Robert", "train") == "train" else test
            bucket.append((dataset_dir, name))
    return train, test


def prepare(
    dataset_root:  str = config.DATASET_ROOT,
    excluded:      set = config.EXCLUDED_DATASETS,
    crops_img_dir: str = config.CROPS_IMG_DIR,
    crops_json_dir: str = config.CROPS_JSON_DIR,
    crop_size:     int = config.CROP_SIZE,
    overlap:       float = config.CROP_OVERLAP,
    val_fraction:  float = config.VAL_FRACTION,
    seed:          int = config.RANDOM_SEED,
) -> None:

    # Each middle crop overlaps both neighbours, so each side = overlap/2
    stride = int(crop_size * (1 - overlap / 2))  # 0.20 → 460 px

    os.makedirs(crops_img_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(crops_json_dir, split), exist_ok=True)

    # Collect images from all datasets
    all_train, all_test = [], []
    datasets = discover_datasets(dataset_root, excluded)
    print(f"Datasets: {[os.path.basename(d) for d in datasets]}")
    for ddir in datasets:
        tr, te = read_split(ddir)
        all_train.extend(tr)
        all_test.extend(te)

    # 75/25 split of CSV-train → train / val
    random.seed(seed)
    random.shuffle(all_train)
    n_val     = max(1, int(len(all_train) * val_fraction))
    all_val   = all_train[:n_val]
    all_train = all_train[n_val:]

    splits = {"train": all_train, "val": all_val, "test": all_test}
    print(f"Images — train: {len(all_train)}, val: {len(all_val)}, test: {len(all_test)}")

    for split, img_list in splits.items():
        all_records: list[dict] = []
        for ddir, img_name in img_list:
            stem    = os.path.splitext(img_name)[0]
            ds_name = os.path.basename(ddir)
            # "Bird_A" → "A",  "Bird_I_complete" → "I"
            ds_letter = ds_name.split("_")[1]

            # Per-dataset image sub-folder: crops/Bird_A/
            ds_img_dir = os.path.join(crops_img_dir, ds_name)
            os.makedirs(ds_img_dir, exist_ok=True)

            records = crop_image(
                image_path=find_image(ddir, stem),
                boxes=read_annotations(os.path.join(ddir, stem + ".txt")),
                crop_size=crop_size,
                stride=stride,
                out_img_dir=ds_img_dir,
                ds_letter=ds_letter,
                stem=stem,
                ds_name=ds_name,
            )
            all_records.extend(records)

        coco      = build_coco_json(all_records)
        coco_path = os.path.join(crops_json_dir, split, "coco.json")
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        total_boxes = sum(len(r["boxes"]) for r in all_records)
        print(f"  {split}: {len(all_records)} crops, {total_boxes} annotations → {coco_path}")

    print("Data preparation complete.")
    print(f"Crop images : {crops_img_dir}/{{Bird_A,Bird_B,...}}/")
    print(f"COCO JSON   : {crops_json_dir}/{{train,val,test}}/coco.json")


if __name__ == "__main__":
    prepare()
