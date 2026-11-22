#!/usr/bin/env python
"""
Command line for the common dataset interface.

    python -m pipeline.cli describe  --source mdc
    python -m pipeline.cli coverage  --source mdc
    python -m pipeline.cli conform   --source mdc [--sample 5]
    python -m pipeline.cli split     --source mdc [--out spec.json]
    python -m pipeline.cli metadata  --source mdc --coco output/crops_json/test/coco.json

    python -m pipeline.cli conform   --source arkansas --root <delivery>

Preparing the data once, for all eight models:

    python -m pipeline.cli prepare --source mdc --mirrors      # crops + COCO + mirrors

Choosing the data for an experiment, once, for all eight models — the choice is a folder:

    python -m pipeline.cli folder --name g_and_j --collections Bird_G Bird_J
    python -m pipeline.cli select --from dataset_g_and_j --mirrors
    python -m pipeline.cli experiments
    WATERFOWL_EXPERIMENT=g_and_j ./yolov5/main.py --train

Every command works the same way for every programme, which is the point.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config
from pipeline import adapters, conformance, crops, selection as selection_mod, splits, tiling


def _dataset_folder(args) -> str | None:
    """The folder the data is read from: --from, else the configured default."""
    folder = getattr(args, "from_folder", None)
    if not folder:
        return None
    return folder if os.path.isabs(folder) else os.path.join(ROOT, folder)


def _adapter(args):
    kwargs = {}
    folder = _dataset_folder(args)
    if folder:
        kwargs["root"] = folder
    if args.root:
        kwargs["root"] = args.root
    if args.source == "mdc" and args.no_exif:
        kwargs["read_exif"] = False
    return adapters.get(args.source, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Common dataset interface")
    ap.add_argument("command",
                    choices=["describe", "coverage", "conform", "split", "metadata",
                             "prepare", "folder", "select", "experiments"])
    ap.add_argument("--source", default="mdc", help="dataset adapter (default: mdc)")
    ap.add_argument("--root", default=None, help="delivery root, for sources that need one")
    ap.add_argument("--sample", type=int, default=None,
                    help="conform: check only the first N photographs per collection")
    ap.add_argument("--coco", default=None, help="metadata: a COCO split to describe")
    ap.add_argument("--out", default=None, help="split: write the specification here")
    ap.add_argument("--no-exif", action="store_true", help="mdc: skip reading EXIF (faster)")
    ap.add_argument("--name", default=None,
                    help="select/folder: name of the experiment (default: the folder's)")
    ap.add_argument("--from", dest="from_folder", default=None,
                    help="select/prepare: read the data from this folder "
                         f"(default: the WATERFOWL_DATASET folder, else dataset_full)")
    ap.add_argument("--collections", nargs="*", default=None,
                    help="folder: which collections to link into the new folder")
    ap.add_argument("--fresh-split", action="store_true",
                    help="select: derive train/val over the subset instead of inheriting "
                         "the full-corpus split")
    ap.add_argument("--mirrors", action="store_true",
                    help="select/prepare: also build the Ultralytics and OWL mirrors")
    ap.add_argument("--force", action="store_true",
                    help="prepare: re-cut the crops and REWRITE the committed COCO splits")
    args = ap.parse_args()

    if args.command == "folder":
        if not args.name or not args.collections:
            print("usage: folder --name <name> --collections Bird_G Bird_J", file=sys.stderr)
            return 2
        source_root = _dataset_folder(args) or config.DATASET_ROOT
        target = os.path.join(ROOT, args.name if args.name.startswith("dataset_")
                              else f"dataset_{args.name}")
        os.makedirs(target, exist_ok=True)
        made = []
        for collection in args.collections:
            src = os.path.join(source_root, collection)
            if not os.path.isdir(src):
                print(f"  {collection}: not in {os.path.relpath(source_root, ROOT)}",
                      file=sys.stderr)
                return 2
            link = os.path.join(target, collection)
            if not os.path.exists(link):
                os.symlink(os.path.relpath(src, target), link)
            made.append(collection)
        print(f"{os.path.relpath(target, ROOT)}/ -> {', '.join(made)} "
              f"(symlinks; no images copied)")
        print(f"  next:  python -m pipeline.cli select --from {os.path.relpath(target, ROOT)} --mirrors")
        return 0

    if args.command == "prepare":
        adapter = _adapter(args)
        existing = all(os.path.exists(os.path.join(config.CROPS_JSON_DIR, s, "coco.json"))
                       for s in tiling.SPLITS)
        if existing and not args.force:
            print(f"crops and COCO splits already exist in "
                  f"{os.path.relpath(config.CROPS_JSON_DIR, ROOT)} — leaving them alone.")
            print("  (they are the published split; pass --force to re-cut everything)")
        else:
            photos = list(adapter.usable_photographs())
            spec = splits.provider_marked(photos, config.VAL_FRACTION, config.RANDOM_SEED,
                                          name=f"{args.source}-provider")
            print(f"Tiling {len(photos):,} photographs from {args.source} "
                  f"into {config.CROP_SIZE}px crops...")
            summary = tiling.build(adapter, spec, photos)
            for split, s in summary.items():
                print(f"  {split:6s} {s['crops']:7,d} crops  {s['birds']:8,d} birds  "
                      f"{s['photographs']:5,d} photographs")
            print(f"  split checksum {spec.checksum[:16]}…")
        if args.mirrors:
            counts = tiling.mirrors()
            print(f"  Ultralytics mirror: {counts['yolo']}")
            print(f"  OWL mirror:         {counts['mdo']}")
        print("Every model reads this; no per-model --prepare is needed.")
        return 0

    if args.command == "experiments":
        names = selection_mod.available()
        if not names:
            print("no experiments materialised yet "
                  "(python -m pipeline.cli select --name <name> ...)")
        for name in names:
            payload = selection_mod.load(name)
            print(selection_mod.format_summary(payload["summary"]))
            print(f"  checksum {payload['checksum'][:16]}…  ({payload['rule']})")
        return 0

    if args.command == "select":
        folder = _dataset_folder(args) or config.DATASET_ROOT
        name = args.name or os.path.basename(folder.rstrip("/")).replace("dataset_", "")
        if not name:
            print("--name is required when the folder has no usable name", file=sys.stderr)
            return 2
        sel = selection_mod.DataSelection(
            name=name, root=os.path.relpath(folder, ROOT), sources=[args.source],
            inherit_roles=not args.fresh_split,
        )
        adapter = _adapter(args)
        spec, photos = selection_mod.resolve(sel, adapters={args.source: adapter})
        summary = selection_mod.materialise(sel, spec, photos, mirrors=args.mirrors)
        print(f"read {os.path.relpath(folder, ROOT)}/ "
              f"({len({p.collection for p in photos})} collections, {len(photos):,} photographs)")
        print(selection_mod.format_summary(summary))
        print(f"  checksum {spec.checksum[:16]}…  ({spec.rule})")
        print(f"  wrote {os.path.relpath(sel.directory(), ROOT)}/")
        print(f"  use it with:  WATERFOWL_EXPERIMENT={name} ./yolov5/main.py --train")
        return 0

    if args.source == "arkansas" and not args.root:
        print("--root is required for the arkansas adapter", file=sys.stderr)
        return 2

    adapter = _adapter(args)

    if args.command == "describe":
        card = adapter.describe()
        for key, value in vars(card).items():
            print(f"{key:14s} {value}")
        for name in sorted(adapter.crosswalks()):
            print(f"{'crosswalk':14s} {name}")

    elif args.command == "coverage":
        print(adapter.coverage().format())

    elif args.command == "conform":
        report = conformance.run(adapter, sample=args.sample)
        print(report.format())
        return 0 if report.passed else 1

    elif args.command == "split":
        photos = list(adapter.usable_photographs())
        spec = splits.provider_marked(photos, config.VAL_FRACTION, config.RANDOM_SEED,
                                      name=f"{args.source}-provider")
        counts = spec.counts()
        print(f"{spec.name}: train {counts['train']:,}  val {counts['val']:,}  "
              f"test {counts['test']:,} photographs")
        print(f"rule     {spec.rule}")
        print(f"checksum {spec.checksum}")
        if args.out:
            print("wrote", spec.to_json(args.out))

    elif args.command == "metadata":
        coco = args.coco or os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
        records = crops.load_split(coco, adapter)
        traced = [c for c in records.values() if c.traced]
        print(f"{os.path.relpath(coco, ROOT)}: {len(records):,} crops, "
              f"{len(traced):,} traced to {len({c.photo.photo_id for c in traced})} photographs")
        for f in ("habitat", "altitude_m", "weather", "site", "source"):
            groups = crops.group_by(records, f)
            covered = sum(len(v) for v in groups.values())
            top = Counter({k: len(v) for k, v in groups.items()}).most_common(4)
            print(f"  {f:12s} {covered * 100.0 / max(len(records), 1):5.1f}% of crops, "
                  f"{len(groups)} value(s): "
                  + ", ".join(f"{k}={n:,}" for k, n in top)
                  + (" ..." if len(groups) > 4 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
