"""
Build super-gradients detection datasets / dataloaders for the waterfowl crops.

Reuses the *exact same* COCO crops produced by data_prep.prepare() — the ones the
Faster R-CNN pipeline trains on:

    crops/Bird_A/A_DJI_0001_0_0.JPG          (images, one folder per dataset)
    output/crops_json/{train,val,test}/coco.json   (annotations, file_name relative
                                                     to the crops root: "Bird_A/....JPG")

super-gradients consumes this through COCOFormatDetectionDataset, which reads the
COCO json with pycocotools and remaps the single category (bird, id=1) to the
contiguous class index 0.  num_classes = 1.

The three public splits map to:
    train → augmented, shuffled, drop_last, empty crops dropped
    val   → deterministic, keeps empty crops + crowd targets (for honest mAP)
    test  → same as val
"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader

from super_gradients.training.datasets.detection_datasets.coco_format_detection import (
    COCOFormatDetectionDataset,
)
from super_gradients.training.transforms.transforms import (
    DetectionMosaic,
    DetectionRandomAffine,
    DetectionHSV,
    DetectionHorizontalFlip,
    DetectionPaddedRescale,
    DetectionStandardize,
    DetectionTargetsFormatTransform,
)
from super_gradients.training.utils.collate_fn import (
    DetectionCollateFN,
    CrowdDetectionCollateFN,
)

import data_prep.config as config

# Single foreground class (matches faster_rcnn.dataset.THING_CLASSES).
CLASS_NAMES = ["bird"]

# file_name in coco.json is "Bird_A/....JPG", relative to the crops root.
# COCOFormatDetectionDataset joins data_dir / images_dir / file_name, so:
#   data_dir   = project root
#   images_dir = "crops"
_DATA_DIR     = os.path.dirname(config.CROPS_IMG_DIR)          # project root
_IMAGES_DIR   = os.path.basename(config.CROPS_IMG_DIR)         # "crops"


def _json_rel(split: str) -> str:
    """coco.json path relative to _DATA_DIR (COCOFormatDetectionDataset joins them)."""
    abs_path = os.path.join(config.CROPS_JSON_DIR, split, "coco.json")
    return os.path.relpath(abs_path, _DATA_DIR)


def _train_transforms(input_dim):
    return [
        DetectionMosaic(input_dim=input_dim, prob=0.5),
        DetectionRandomAffine(degrees=0.0, scales=(0.5, 1.5), shear=0.0,
                              target_size=input_dim, filter_box_candidates=False),
        DetectionHSV(prob=0.5, hgain=5, sgain=30, vgain=30),
        DetectionHorizontalFlip(prob=0.5),
        DetectionPaddedRescale(input_dim=input_dim),
        DetectionStandardize(max_value=255.0),
        DetectionTargetsFormatTransform(input_dim=input_dim, output_format="LABEL_CXCYWH"),
    ]


def _eval_transforms(input_dim):
    return [
        DetectionPaddedRescale(input_dim=input_dim),
        DetectionStandardize(max_value=255.0),
        DetectionTargetsFormatTransform(input_dim=input_dim, output_format="LABEL_CXCYWH"),
    ]


def _build_dataset(split: str, transforms, ignore_empty: bool, with_crowd: bool):
    input_dim = (config.CROP_SIZE, config.CROP_SIZE)
    return COCOFormatDetectionDataset(
        data_dir=_DATA_DIR,
        images_dir=_IMAGES_DIR,
        json_annotation_file=_json_rel(split),
        input_dim=input_dim,
        transforms=transforms,
        ignore_empty_annotations=ignore_empty,
        with_crowd=with_crowd,
    )


def build_dataloaders(
    batch_size:  int = config.YOLONAS_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict:
    """
    Return {"train": DataLoader, "val": DataLoader, "test": DataLoader}.

    Any split whose coco.json is missing is skipped (e.g. you can ask for only
    the test loader during --eval).
    """
    input_dim = (config.CROP_SIZE, config.CROP_SIZE)
    loaders: dict = {}

    specs = {
        # split : (transforms, shuffle, drop_last, ignore_empty, with_crowd, collate)
        "train": (_train_transforms(input_dim), True,  True,  True,  False, DetectionCollateFN()),
        "val":   (_eval_transforms(input_dim),  False, False, False, True,  CrowdDetectionCollateFN()),
        "test":  (_eval_transforms(input_dim),  False, False, False, True,  CrowdDetectionCollateFN()),
    }

    for split, (tfms, shuffle, drop_last, ignore_empty, with_crowd, collate) in specs.items():
        json_path = os.path.join(config.CROPS_JSON_DIR, split, "coco.json")
        if not os.path.exists(json_path):
            continue
        ds = _build_dataset(split, tfms, ignore_empty=ignore_empty, with_crowd=with_crowd)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
        )
    return loaders
