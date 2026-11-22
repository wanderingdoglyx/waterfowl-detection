"""
Map the model names people write in a manifest onto checkpoint folders.

Manifest names are human-written ("Fasterrcnn", "owl_c"), while the checkpoint
folders under output/checkpoints/ use the pipeline's own keys ("fasterrcnn",
"mdo_owl_c").  Lookup is case-insensitive and accepts a few natural aliases.

`kind` matters for the report: box detectors produce a real COCO mAP30, while
the OWL point detectors only produce a *pseudo*-box mAP30 (points inflated to
28 px boxes).  Those two numbers are not comparable, so they are kept in
separate columns rather than ranked against each other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str            # canonical key == checkpoint folder under output/checkpoints/
    label: str          # display name for tables and figures
    family: str
    kind: str           # "box" (real mAP30) or "point" (pseudo-box mAP30 only)
    aliases: tuple[str, ...] = ()


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("fasterrcnn", "Faster R-CNN", "detectron2", "box",
              aliases=("faster_rcnn", "faster-rcnn", "frcnn")),
    ModelSpec("yolov5", "YOLOv5", "ultralytics", "box"),
    ModelSpec("yolo11", "YOLO11", "ultralytics", "box"),
    ModelSpec("yolo26", "YOLO26", "ultralytics", "box"),
    ModelSpec("yolonas", "YOLO-NAS", "super_gradients", "box",
              aliases=("yolo_nas", "yolo-nas")),
    ModelSpec("mdo_owl_c", "OWL-C", "megadetector-overhead", "point",
              aliases=("owl_c", "owlc", "owl-c")),
    ModelSpec("mdo_owl_t", "OWL-T", "megadetector-overhead", "point",
              aliases=("owl_t", "owlt", "owl-t")),
    ModelSpec("mdo_owl_d", "OWL-D", "megadetector-overhead", "point",
              aliases=("owl_d", "owld", "owl-d", "owld_s", "owld_h")),
)

_INDEX: dict[str, ModelSpec] = {}
for _spec in MODELS:
    for _name in (_spec.key, *_spec.aliases):
        _INDEX[_name.lower()] = _spec


def resolve(name: str) -> ModelSpec:
    """Look up a manifest model name. Raises KeyError listing the valid names."""
    spec = _INDEX.get(name.strip().lower())
    if spec is None:
        known = ", ".join(sorted({s.key for s in MODELS}))
        raise KeyError(f"unknown model {name!r} (known: {known})")
    return spec
