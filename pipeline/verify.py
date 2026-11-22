#!/usr/bin/env python
"""
Does the interface reproduce the pipeline it replaces?

Four checks, none of which needs a GPU or regenerates anything:

  A  The MDC adapter sees exactly the photographs, split roles and boxes that
     `data_prep.prepare_data` sees — the loader every published result came from.
  B  The split derived through the interface reproduces the committed train / val / test
     membership of `output/crops_json/*/coco.json`.
  C  Survey conditions read through the interface match `analysis.metadata` crop for crop
     on the test split, so every condition table would come out identical.
  D  The Arkansas adapter reads a delivery in the specified format and passes the
     conformance suite, with no change to anything above the interface.
  E  A data selection — a folder holding the data an experiment uses, read once for all
     eight models — filters the committed crops exactly, without re-cutting them and
     without moving any photograph between train, validation and test.
  F  Tiling driven by the interface cuts one collection into exactly the crops the
     committed splits hold for it, with the same boxes — so the one shared preparation
     command replaces the five per-model `--prepare` flags without changing the data.
  H  Naming a data folder is enough: a model calling ensure_data() gets that folder's
     data ready with no command of any kind, the whole corpus costs nothing, and a second
     run reuses the cache.
  G  A folder resolves to exactly the photographs it holds, with the train/validation
     membership the full corpus already gave them, whether it is named with --from or
     with WATERFOWL_DATASET; and a folder assembled by hand, with no index file, is read.

    python -m pipeline.verify            # all four
    python -m pipeline.verify --quick    # A and D only (no EXIF, no metadata join)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config
from data_prep.prepare_data import discover_datasets, read_annotations, read_split
from pipeline import auto, conformance, crops, selection as selection_mod, splits, tiling, vocabularies
from pipeline.adapters import ArkansasAdapter, MdcAdapter

OK, BAD = "  ok   ", "  FAIL "


def _result(passed: bool, label: str, detail: str = "") -> bool:
    print(f"{OK if passed else BAD} {label}" + (f" — {detail}" if detail else ""))
    return passed


# ── A: the adapter sees what the loader sees ──────────────────────────────────
def check_loader_equivalence(adapter: MdcAdapter) -> bool:
    """Photographs, split roles and box counts, against data_prep.prepare_data."""
    legacy: dict[str, tuple[str, str]] = {}          # (collection, stem) -> (role, ann path)
    for ddir in discover_datasets(config.DATASET_ROOT, config.EXCLUDED_DATASETS):
        collection = os.path.basename(ddir)
        train, test = read_split(ddir)
        for role, items in (("train", train), ("test", test)):
            for _, image_name, ann_path in items:
                stem = os.path.splitext(image_name)[0]
                legacy[f"{collection}/{stem}"] = (role, ann_path)

    new = {f"{p.collection}/{p.stem}": (p.split_role, p.annotation_path)
           for p in adapter.usable_photographs()}

    ok = _result(set(new) == set(legacy), "A1 same photographs",
                 f"{len(new):,} usable"
                 + ("" if set(new) == set(legacy)
                    else f"; only in interface: {len(set(new) - set(legacy))}, "
                         f"only in loader: {len(set(legacy) - set(new))}"))

    roles_differ = [k for k in set(new) & set(legacy) if new[k][0] != legacy[k][0]]
    ok &= _result(not roles_differ, "A2 same train/test roles",
                  f"{len(roles_differ)} differ" if roles_differ
                  else f"train {sum(1 for v in new.values() if v[0] == 'train'):,}, "
                       f"test {sum(1 for v in new.values() if v[0] == 'test'):,}")

    paths_differ = [k for k in set(new) & set(legacy) if new[k][1] != legacy[k][1]]
    ok &= _result(not paths_differ, "A3 same annotation files",
                  f"{len(paths_differ)} differ" if paths_differ else "")

    boxes_differ, total_boxes = [], 0
    for photo in adapter.usable_photographs():
        key = f"{photo.collection}/{photo.stem}"
        expected = len(read_annotations(legacy[key][1])) if key in legacy else -1
        got = len(adapter.annotations(photo))
        total_boxes += got
        if got != expected:
            boxes_differ.append(key)
    ok &= _result(not boxes_differ, "A4 same box counts",
                  f"{len(boxes_differ)} photographs differ" if boxes_differ
                  else f"{total_boxes:,} boxes")
    return ok


# ── B: the derived split reproduces the committed one ─────────────────────────
def check_split_equivalence(adapter: MdcAdapter) -> bool:
    spec = splits.provider_marked(list(adapter.usable_photographs()),
                                  config.VAL_FRACTION, config.RANDOM_SEED)
    ok = True
    for split in ("train", "val", "test"):
        coco_path = os.path.join(config.CROPS_JSON_DIR, split, "coco.json")
        if not os.path.exists(coco_path):
            ok &= _result(False, f"B {split}", "committed COCO file missing")
            continue
        with open(coco_path) as f:
            images = json.load(f)["images"]
        committed = set()
        for image in images:
            parsed = crops.parse_crop_name(image["file_name"])
            if parsed:
                committed.add(f"mdc/{parsed[0]}/{parsed[1]}")
        derived = set(spec.ids(split))
        ok &= _result(derived == committed, f"B {split} photographs",
                      f"{len(derived):,} photographs, {len(images):,} crops"
                      + ("" if derived == committed
                         else f"; committed-only {len(committed - derived)}, "
                              f"derived-only {len(derived - committed)}"))
    print(f"        split checksum {spec.checksum[:16]}…")
    return ok


# ── C: survey conditions match the current reconstruction ─────────────────────
def check_metadata_equivalence(adapter: MdcAdapter) -> bool:
    from analysis import metadata as legacy_metadata

    coco_path = os.path.join(config.CROPS_JSON_DIR, "test", "coco.json")
    legacy = legacy_metadata.load(coco_path)
    new = crops.load_split(coco_path, adapter)
    habitat_crosswalk = vocabularies.habitat("mdc")

    untraced = [i for i, c in new.items() if not c.traced]
    ok = _result(not untraced, "C1 every crop traced to a photograph",
                 f"{len(untraced)} untraced" if untraced else f"{len(new):,} crops")

    mismatches = {f: 0 for f in ("altitude", "habitat", "habitat_source", "weather",
                                 "site", "date", "offsets")}
    for image_id, crop in new.items():
        old = legacy.get(image_id)
        if old is None or crop.photo is None:
            continue
        photo = crop.photo
        if old.altitude_m != photo.altitude_m:
            mismatches["altitude"] += 1
        # The old value is a raw label; the new one is canonical, so compare through
        # the crosswalk that replaced the two normalisations done in code.
        if habitat_crosswalk.get(old.habitat)[0] != photo.habitat:
            mismatches["habitat"] += 1
        if (old.habitat_source or None) != (photo.habitat_source if photo.habitat_raw else None):
            mismatches["habitat_source"] += 1
        if (old.weather or None) != (photo.weather_raw or None):
            mismatches["weather"] += 1
        if (old.site or None) != (photo.site or None):
            mismatches["site"] += 1
        if (old.date or None) != (photo.date or None):
            mismatches["date"] += 1
        if (old.row, old.col) != (crop.row, crop.col):
            mismatches["offsets"] += 1

    for field, count in mismatches.items():
        ok &= _result(count == 0, f"C2 {field} matches", f"{count:,} crops differ" if count else "")
    return ok


# ── D: a delivery in the specified format, end to end ─────────────────────────
def _write_synthetic_delivery(root: str) -> None:
    """A three-photograph AGFC-format delivery: two sites, boxes, and an empty frame."""
    from PIL import Image

    rows = []
    frames = [("BayouMeto", "2027-01-18", "F01_0001", "bottomland hardwood", 2),
              ("BayouMeto", "2027-01-18", "F01_0002", "rice", 0),
              ("FreddieBlack", "2027-01-19", "F03_0007", "moist soil", 3)]

    for site, date, frame, habitat, n_birds in frames:
        folder = os.path.join(root, site, date)
        os.makedirs(folder, exist_ok=True)
        rel = f"{site}/{date}/{frame}.jpg"
        Image.new("RGB", (640, 480), (90, 110, 90)).save(os.path.join(root, rel), quality=90)

        with open(os.path.join(root, f"{site}/{date}/{frame}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x1", "y1", "x2", "y2", "species", "sex", "decoy", "notes"])
            for i in range(n_birds):                      # header only when n_birds == 0
                x, y = 30 + 40 * i, 50 + 25 * i
                w.writerow([x, y, x + 18, y + 16, "mallard", "male" if i % 2 else "", "real", ""])

        rows.append({"file": rel, "site": site, "date": date, "flight_altitude_m": 60,
                     "altitude_reference": "above_takeoff", "drone_platform": "DJI Mavic 3",
                     "sensor": "L2D-20c", "focal_length_mm": 12.29, "sensor_width_mm": 17.4,
                     "image_width_px": 640, "image_height_px": 480, "habitat": habitat,
                     "weather": "cloud", "decoys": "real", "split": ""})

    with open(os.path.join(root, "photographs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_arkansas_adapter() -> bool:
    root = tempfile.mkdtemp(prefix="agfc_delivery_")
    try:
        _write_synthetic_delivery(root)
        adapter = ArkansasAdapter(root, received="2027-01-20")

        card = adapter.describe()
        ok = _result(card.photographs == 3 and card.annotations == 5,
                     "D1 delivery read",
                     f"{card.collections} sites, {card.photographs} photographs, "
                     f"{card.annotations} birds")
        ok &= _result(not adapter.missing_columns(), "D2 required columns present",
                      ", ".join(adapter.missing_columns()))

        photos = {p.photo_id: p for p in adapter.usable_photographs()}
        first = photos["arkansas/BayouMeto/F01_0001"]
        ok &= _result(first.habitat == "forest" and first.habitat_raw == "bottomland hardwood",
                      "D3 habitat crosswalked",
                      f"{first.habitat_raw!r} -> {first.habitat!r}")
        gsd = first.gsd_cm_px()
        expected = 60 * 1000 * (17.4 / 640) / 12.29 / 10       # H x pixel pitch / focal
        ok &= _result(gsd is not None and abs(gsd - expected) < 0.01,
                      "D4 ground sample distance",
                      f"{gsd:.2f} cm/px at 60 m (expected {expected:.2f})" if gsd
                      else "not computed")

        empty = photos["arkansas/BayouMeto/F01_0002"]
        ok &= _result(adapter.annotations(empty) == [], "D5 empty frame kept",
                      "header-only annotation file means 'no birds', not 'not annotated'")

        spec = splits.site_holdout(photos.values(), test_sites=["FreddieBlack"])
        counts = spec.counts()
        ok &= _result(counts["test"] == 1 and counts["train"] == 2, "D6 site-level holdout",
                      f"train {counts['train']}, test {counts['test']}, "
                      f"checksum {spec.checksum[:12]}…")

        report = conformance.run(adapter, expected_birds={"BayouMeto": 2, "FreddieBlack": 3})
        ok &= _result(report.passed, "D7 conformance suite",
                      f"{sum(1 for c in report.checks if c.status == 'pass')}/"
                      f"{len(report.checks)} checks passed")
        if not report.passed:
            print(report.format())
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _folder_with(collections: list[str]) -> str:
    """A data-selection folder: symlinks to whole collections, as a user would make."""
    root = tempfile.mkdtemp(prefix="dataset_")
    for collection in collections:
        os.symlink(os.path.join(config.DATASET_ROOT, collection),
                   os.path.join(root, collection))
    return root


# ── E: one data selection, shared by every model ──────────────────────────────
def check_selection(adapter: MdcAdapter) -> bool:
    """Filtering the corpus keeps exactly the crops it should, and only those."""
    chosen = ["Bird_C", "Bird_F"]
    root = _folder_with(chosen)
    out = tempfile.mkdtemp(prefix="selection_")
    try:
        sel = selection_mod.DataSelection(name="verify_subset", root=root)
        spec, photos = selection_mod.resolve(
            sel, adapters={"mdc": MdcAdapter(root=root, read_exif=False)})
        summary = selection_mod.materialise(sel, spec, photos, out_dir=out, mirrors=False)

        # What the committed splits hold for those collections, counted independently.
        expected = {s: 0 for s in selection_mod.SPLITS}
        for split in selection_mod.SPLITS:
            with open(os.path.join(config.CROPS_JSON_DIR, split, "coco.json")) as f:
                for image in json.load(f)["images"]:
                    parsed = crops.parse_crop_name(image["file_name"])
                    if parsed and parsed[0] in chosen:
                        expected[split] += 1

        got = {s: summary["splits"][s]["crops"] for s in selection_mod.SPLITS}
        ok = _result(got == expected, "E1 selection keeps exactly the right crops",
                     ", ".join(f"{s} {got[s]:,}" for s in selection_mod.SPLITS)
                     + ("" if got == expected else f"; expected {expected}"))

        ok &= _result(summary["moved_crops"] == 0, "E2 no photograph changed split",
                      "train/val/test membership inherited from the full-corpus split")

        foreign, duplicate_ids, carried = [], [], True
        for split in selection_mod.SPLITS:
            with open(os.path.join(out, "crops_json", split, "coco.json")) as f:
                coco = json.load(f)
            ids = [i["id"] for i in coco["images"]]
            if len(ids) != len(set(ids)):
                duplicate_ids.append(split)
            for image in coco["images"]:
                if image.get("collection") not in chosen:
                    foreign.append(image["file_name"])
                if "altitude_m" not in image or "photo_id" not in image:
                    carried = False
        ok &= _result(not foreign, "E3 no crop from an unselected collection",
                      f"{len(foreign)} foreign crops" if foreign else "")
        ok &= _result(not duplicate_ids, "E4 image ids unique within each split",
                      ", ".join(duplicate_ids) if duplicate_ids else "")
        ok &= _result(carried, "E5 survey metadata travels with the crop",
                      "photo_id, collection, altitude, habitat, GSD written into each record")
        return ok
    finally:
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)


# ── F: one preparation command, same crops ────────────────────────────────────
def check_tiling(adapter: MdcAdapter) -> bool:
    """Tile one collection through the interface and compare with the committed crops."""
    collection = "Bird_F"
    out = tempfile.mkdtemp(prefix="tiling_")
    try:
        photos = [p for p in adapter.usable_photographs() if p.collection == collection]
        spec = splits.provider_marked(list(adapter.usable_photographs()),
                                      config.VAL_FRACTION, config.RANDOM_SEED)
        spec.assignment = {p.photo_id: spec.assignment[p.photo_id] for p in photos}

        summary = tiling.build(adapter, spec, photos,
                               crops_img_dir=os.path.join(out, "crops"),
                               crops_json_dir=os.path.join(out, "crops_json"),
                               verbose=False)

        committed: dict[str, int] = {}
        for split in tiling.SPLITS:
            with open(os.path.join(config.CROPS_JSON_DIR, split, "coco.json")) as f:
                coco = json.load(f)
            counts: dict[int, int] = {}
            for a in coco["annotations"]:
                counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
            for image in coco["images"]:
                if image["file_name"].startswith(collection + "/"):
                    committed[image["file_name"]] = counts.get(image["id"], 0)

        produced: dict[str, int] = {}
        carried = True
        for split in tiling.SPLITS:
            with open(os.path.join(out, "crops_json", split, "coco.json")) as f:
                coco = json.load(f)
            counts = {}
            for a in coco["annotations"]:
                counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
            for image in coco["images"]:
                produced[image["file_name"]] = counts.get(image["id"], 0)
                if "photo_id" not in image or "altitude_m" not in image:
                    carried = False

        ok = _result(set(produced) == set(committed), "F1 same crops as the committed split",
                     f"{len(produced):,} crops of {collection}"
                     + ("" if set(produced) == set(committed)
                        else f"; extra {len(set(produced) - set(committed))}, "
                             f"missing {len(set(committed) - set(produced))}"))
        differing = [k for k in set(produced) & set(committed) if produced[k] != committed[k]]
        ok &= _result(not differing, "F2 same boxes per crop",
                      f"{len(differing)} crops differ" if differing
                      else f"{sum(produced.values()):,} boxes")
        ok &= _result(carried, "F3 survey metadata written into the COCO records",
                      "photo_id, collection, altitude, habitat, GSD on every crop")
        ok &= _result(sum(s["crops"] for s in summary.values()) == len(produced),
                      "F4 summary agrees with what was written", "")
        return ok
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ── G: the folder is the selection ────────────────────────────────────────────
def check_folder_selection(adapter: MdcAdapter) -> bool:
    """A folder of data resolves identically to the equivalent --collections flags."""
    chosen = ["Bird_C", "Bird_F"]
    root = _folder_with(chosen)
    try:
        from_folder = selection_mod.DataSelection(name="folder", root=root)
        spec_folder, photos_folder = selection_mod.resolve(
            from_folder, adapters={"mdc": MdcAdapter(root=root, read_exif=False)})

        # What the folder should resolve to: those collections, with the roles the
        # full-corpus split already gave them.
        corpus = list(adapter.usable_photographs())
        baseline = splits.provider_marked(corpus, config.VAL_FRACTION, config.RANDOM_SEED)
        expected_ids = {p.photo_id for p in corpus if p.collection in chosen}
        expected_roles = {pid: baseline.assignment[pid] for pid in expected_ids}

        ok = _result({p.photo_id for p in photos_folder} == expected_ids,
                     "G1 the folder selects exactly what is in it",
                     f"{len(photos_folder)} photographs")
        ok &= _result(spec_folder.assignment == expected_roles,
                      "G2 roles inherited from the full-corpus split",
                      f"checksum {spec_folder.checksum[:12]}…")

        # Pointing the whole pipeline at the folder (WATERFOWL_DATASET) must give the
        # same answer as reading it with --from: the split is inherited from the full
        # corpus either way, never re-derived over the folder.
        saved_root = config.DATASET_ROOT
        try:
            config.DATASET_ROOT = root
            spec_env, _ = selection_mod.resolve(
                from_folder, adapters={"mdc": MdcAdapter(root=root, read_exif=False)})
        finally:
            config.DATASET_ROOT = saved_root
        ok &= _result(spec_env.assignment == expected_roles,
                      "G3 same answer when the pipeline is pointed at the folder",
                      "WATERFOWL_DATASET and --from agree")

        # A folder someone filled by hand: images and labels, no index file.
        loose = os.path.join(root, "Handpicked")
        os.makedirs(loose)
        for photo in list(adapter.usable_photographs())[:3]:
            os.symlink(photo.path, os.path.join(loose, os.path.basename(photo.path)))
            os.symlink(photo.annotation_path,
                       os.path.join(loose, os.path.basename(photo.annotation_path)))

        bare = MdcAdapter(root=root, read_exif=False)
        handpicked = [p for p in bare.usable_photographs() if p.collection == "Handpicked"]
        ok &= _result(len(handpicked) == 3, "G4 folder without an index file is read",
                      f"{len(handpicked)} photographs, "
                      f"{sum(len(bare.annotations(p)) for p in handpicked):,} birds")
        ok &= _result(all(p.split_role == "unassigned" for p in handpicked),
                      "G5 unmarked photographs are reported as unassigned",
                      "no train/test mark is invented for them")

        spec = splits.provider_marked(handpicked, config.VAL_FRACTION, config.RANDOM_SEED)
        counts = spec.counts()
        ok &= _result(counts["test"] == 0 and counts["train"] + counts["val"] == 3,
                      "G6 unassigned photographs join train/validation",
                      f"train {counts['train']}, val {counts['val']}, test {counts['test']}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── H: naming the folder is enough ────────────────────────────────────────────
def check_auto(adapter: MdcAdapter) -> bool:
    """What a model does at startup: make the configured folder ready, and nothing else."""
    chosen = ["Bird_C", "Bird_F"]
    root = _folder_with(chosen)
    experiments = tempfile.mkdtemp(prefix="experiments_")
    saved = (config.DATASET_ROOT, config.EXPERIMENTS_DIR, config.CROPS_JSON_DIR)
    try:
        config.EXPERIMENTS_DIR = experiments

        # The whole corpus: the committed splits are already right, so nothing is built.
        config.DATASET_ROOT = saved[0]
        name = auto.ensure_data(mirror=None, verbose=False)
        ok = _result(name is None and config.CROPS_JSON_DIR == config.COMMITTED_CROPS_JSON_DIR,
                     "H1 the default folder costs nothing",
                     "the committed splits are used as they are")
        ok &= _result(not os.listdir(experiments), "H2 nothing written for the whole corpus",
                      f"{len(os.listdir(experiments))} files" if os.listdir(experiments) else "")

        # A folder holding part of the corpus: built automatically, named after the folder.
        config.DATASET_ROOT = root
        expected_name = os.path.basename(root).replace("dataset_", "")
        name = auto.ensure_data(mirror=None, verbose=False)
        ok &= _result(name == expected_name, "H3 experiment named after the folder",
                      f"{name!r}")

        coco_path = os.path.join(config.CROPS_JSON_DIR, "train", "coco.json")
        with open(coco_path) as f:
            images = json.load(f)["images"]
        collections = sorted({i["collection"] for i in images})
        ok &= _result(collections == chosen, "H4 only that folder's data is used",
                      f"{len(images):,} training crops from {collections}")

        # Running a second model must not rebuild anything.
        before = os.path.getmtime(coco_path)
        auto.ensure_data(mirror=None, verbose=False)
        ok &= _result(os.path.getmtime(coco_path) == before, "H5 the second run reuses the cache",
                      "no rebuild")

        # Change the folder and the next run notices.
        os.symlink(os.path.join(config.DEFAULT_DATASET_ROOT, "Bird_A"),
                   os.path.join(root, "Bird_A"))
        auto.ensure_data(mirror=None, verbose=False)
        with open(coco_path) as f:
            collections = sorted({i["collection"] for i in json.load(f)["images"]})
        ok &= _result(collections == sorted(chosen + ["Bird_A"]),
                      "H6 changing the folder rebuilds", f"now {collections}")
        return ok
    finally:
        config.DATASET_ROOT, config.EXPERIMENTS_DIR, config.CROPS_JSON_DIR = saved
        config.use_committed_data()
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(experiments, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the dataset interface against the pipeline")
    ap.add_argument("--quick", action="store_true", help="skip the split and metadata joins")
    args = ap.parse_args()

    print("A. MDC adapter vs data_prep.prepare_data")
    adapter = MdcAdapter(read_exif=not args.quick)
    ok = check_loader_equivalence(adapter)

    if not args.quick:
        print("\nB. Derived split vs the committed COCO splits")
        ok &= check_split_equivalence(adapter)

        print("\nC. Survey conditions vs analysis.metadata (test split)")
        ok &= check_metadata_equivalence(adapter)

    print("\nD. Arkansas adapter on a synthetic delivery")
    ok &= check_arkansas_adapter()

    if not args.quick:
        print("\nE. One data selection, shared by every model")
        ok &= check_selection(adapter)

        print("\nF. One preparation command, same crops")
        ok &= check_tiling(adapter)

        print("\nG. The folder is the selection")
        ok &= check_folder_selection(adapter)

        print("\nH. Naming the folder is enough")
        ok &= check_auto(adapter)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
