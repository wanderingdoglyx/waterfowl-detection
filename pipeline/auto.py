"""
Point the pipeline at a folder of data and run — no data commands.

The data folder is named once, in `data_prep/config.py` (`DATASET_FOLDER`, default
`dataset_full`), or overridden for a single command with `WATERFOWL_DATASET`.  Every model
then calls `ensure_data()` when it starts, and whatever the folder holds is what it trains
and evaluates on:

    # data_prep/config.py
    DATASET_FOLDER = "dataset_g_and_j"

    ./yolov5/main.py --train --eval        # trains on Bird_G + Bird_J, nothing else

What `ensure_data()` does, in order:

1. Reads the folder through its dataset adapter and fingerprints the photographs it holds.
2. If that fingerprint matches the committed splits, those are used unchanged — the normal
   full-corpus case costs one adapter pass and writes nothing.
3. Otherwise it materialises an experiment named after the folder: the committed crops are
   *filtered* to the folder's photographs, and any photograph with no crops yet is tiled.
   The result is cached, so the second run of any model is instant.
4. It repoints `config.CROPS_JSON_DIR` (and the framework mirror directories) at that
   experiment, and builds the mirror the calling model needs.

The fingerprint is what makes this safe: change the folder's contents and the next run
notices and rebuilds; leave it alone and nothing is recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os

from data_prep import config
from pipeline import crops as crop_tools
from pipeline import selection as selection_mod
from pipeline import splits, tiling
from pipeline.adapters import get as get_adapter

FINGERPRINT_FILE = "data_fingerprint.json"


def _fingerprint(photo_ids) -> str:
    return hashlib.sha256("\n".join(sorted(photo_ids)).encode()).hexdigest()


def folder_fingerprint(adapter) -> tuple[str, list]:
    photos = list(adapter.usable_photographs())
    return _fingerprint(p.photo_id for p in photos), photos


def committed_fingerprint(source: str, crops_json_dir: str | None = None) -> str | None:
    """Fingerprint of the photographs behind the committed COCO splits, cached on disk.

    Computing it means reading the three split files once; the answer is stored beside
    them so later runs cost nothing.
    """
    crops_json_dir = crops_json_dir or config.COMMITTED_CROPS_JSON_DIR
    cache = os.path.join(crops_json_dir, FINGERPRINT_FILE)
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                return json.load(f)["fingerprint"]
        except Exception:
            pass

    photo_ids = set()
    for split in tiling.SPLITS:
        path = os.path.join(crops_json_dir, split, "coco.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            for image in json.load(f)["images"]:
                if image.get("photo_id"):
                    photo_ids.add(image["photo_id"])
                    continue
                parsed = crop_tools.parse_crop_name(image["file_name"])
                if parsed:
                    photo_ids.add(f"{source}/{parsed[0]}/{parsed[1]}")
    if not photo_ids:
        return None

    value = _fingerprint(photo_ids)
    try:
        with open(cache, "w") as f:
            json.dump({"fingerprint": value, "photographs": len(photo_ids)}, f, indent=2)
    except OSError:
        pass
    return value


def experiment_name(folder: str) -> str:
    base = os.path.basename(folder.rstrip("/"))
    return base[len("dataset_"):] if base.startswith("dataset_") else base


def ensure_data(mirror: str | None = None, source: str = "mdc", verbose: bool = True) -> str | None:
    """Make the configured data folder ready to train on.  Returns the experiment name.

    `mirror` is the on-disk layout the calling model needs: "yolo" for Ultralytics, "mdo"
    for OWL, None for frameworks that read the COCO files directly.
    """
    folder = config.DATASET_ROOT
    name = experiment_name(folder)

    try:
        adapter = get_adapter(source, root=folder, read_exif=False)
    except TypeError:
        adapter = get_adapter(source, root=folder)
    fingerprint, photos = folder_fingerprint(adapter)

    # 1. The folder is the corpus the committed splits were built from: use them as they are.
    if fingerprint == committed_fingerprint(source):
        config.use_committed_data()
        if mirror:
            _build_mirror(mirror, config.CROPS_JSON_DIR, config.YOLOV5_DATA_DIR,
                          config.MDO_DATA_DIR, verbose)
        if verbose:
            print(f"Data: {os.path.basename(folder)} ({len(photos):,} photographs, "
                  "the committed splits)")
        return None

    # 2. A different set of data: an experiment named after the folder, cached.
    out_dir = os.path.join(config.EXPERIMENTS_DIR, name)
    record = os.path.join(out_dir, "selection.json")
    cached = None
    if os.path.exists(record):
        try:
            with open(record) as f:
                cached = json.load(f).get("fingerprint")
        except Exception:
            cached = None

    if cached != fingerprint:
        if verbose:
            print(f"Data: {os.path.basename(folder)} ({len(photos):,} photographs) — "
                  f"{'rebuilding' if cached else 'building'} {name}...", flush=True)
        _materialise(adapter, name, folder, photos, fingerprint, out_dir, verbose)
    elif verbose:
        print(f"Data: {os.path.basename(folder)} ({len(photos):,} photographs, "
              f"experiment {name})")

    config.use_experiment_data(name)
    if mirror:
        _build_mirror(mirror, config.CROPS_JSON_DIR, config.YOLOV5_DATA_DIR,
                      config.MDO_DATA_DIR, verbose)
    return name


def _materialise(adapter, name, folder, photos, fingerprint, out_dir, verbose) -> None:
    """Filter the committed crops to this folder, tiling anything that has none yet."""
    selection = selection_mod.DataSelection(
        name=name, root=os.path.relpath(folder, os.path.dirname(config.OUTPUT_DIR)))
    spec, kept = selection_mod.resolve(selection, adapters={adapter.source: adapter})

    summary = selection_mod.materialise(selection, spec, kept,
                                        crops_json_dir=config.COMMITTED_CROPS_JSON_DIR,
                                        out_dir=out_dir, mirrors=False)

    # Photographs the committed crops do not cover have to be cut before they can be used.
    covered = set()
    for split in tiling.SPLITS:
        path = os.path.join(out_dir, "crops_json", split, "coco.json")
        if os.path.exists(path):
            with open(path) as f:
                covered.update(image.get("photo_id") for image in json.load(f)["images"])
    missing = [p for p in kept if p.photo_id not in covered]

    if missing:
        if verbose:
            print(f"  {len(missing):,} photographs have no crops yet — tiling them "
                  "(this happens once)", flush=True)
        sub = splits.SplitSpec(name=name, sources=spec.sources,
                               assignment={p.photo_id: spec.assignment[p.photo_id]
                                           for p in missing},
                               val_fraction=spec.val_fraction, seed=spec.seed,
                               rule=spec.rule)
        extra_dir = os.path.join(out_dir, "new_crops_json")
        tiling.build(adapter, sub, missing, crops_json_dir=extra_dir, verbose=False)
        _merge(out_dir, extra_dir)
        summary = _summarise(out_dir)

    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump({"selection": {"name": name, "root": os.path.basename(folder)},
                   "fingerprint": fingerprint, "checksum": spec.checksum,
                   "rule": spec.rule, "summary": summary}, f, indent=2)


def _merge(out_dir: str, extra_dir: str) -> None:
    """Fold newly tiled crops into the experiment's COCO splits, renumbering ids."""
    for split in tiling.SPLITS:
        base_path = os.path.join(out_dir, "crops_json", split, "coco.json")
        extra_path = os.path.join(extra_dir, split, "coco.json")
        if not os.path.exists(extra_path):
            continue
        with open(base_path) as f:
            base = json.load(f)
        with open(extra_path) as f:
            extra = json.load(f)

        next_image = max([i["id"] for i in base["images"]], default=0) + 1
        next_ann = max([a["id"] for a in base["annotations"]], default=0) + 1
        remap = {}
        for image in extra["images"]:
            remap[image["id"]] = next_image
            image["id"] = next_image
            next_image += 1
        for a in extra["annotations"]:
            a["image_id"] = remap[a["image_id"]]
            a["id"] = next_ann
            next_ann += 1
        base["images"].extend(extra["images"])
        base["annotations"].extend(extra["annotations"])
        with open(base_path, "w") as f:
            json.dump(base, f)
    import shutil
    shutil.rmtree(extra_dir, ignore_errors=True)


def _summarise(out_dir: str) -> dict:
    summary = {"splits": {}, "collections": {}, "moved_crops": 0}
    for split in tiling.SPLITS:
        path = os.path.join(out_dir, "crops_json", split, "coco.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            coco = json.load(f)
        summary["splits"][split] = {
            "crops": len(coco["images"]), "birds": len(coco["annotations"]),
            "photographs": len({i.get("photo_id") for i in coco["images"]})}
        for image in coco["images"]:
            entry = summary["collections"].setdefault(image.get("collection", "?"),
                                                      {s: 0 for s in tiling.SPLITS})
            entry[split] += 1
    return summary


def _build_mirror(mirror: str, crops_json_dir: str, yolo_dir: str, mdo_dir: str,
                  verbose: bool) -> None:
    """Build only the layout the calling model needs, and only when it is out of date."""
    if mirror == "yolo":
        marker = os.path.join(yolo_dir, "data.yaml")
    elif mirror == "mdo":
        marker = os.path.join(mdo_dir, "test", "gt.csv")
    else:
        return

    fresh = os.path.join(crops_json_dir, FINGERPRINT_FILE)
    if os.path.exists(marker) and os.path.exists(fresh) \
            and os.path.getmtime(marker) >= os.path.getmtime(fresh):
        return

    if verbose:
        print(f"  building the {mirror} layout...", flush=True)
    counts = tiling.mirrors(crops_json_dir=crops_json_dir, yolo_dir=yolo_dir, mdo_dir=mdo_dir)
    if verbose:
        print(f"  {mirror}: {counts[mirror]}")
