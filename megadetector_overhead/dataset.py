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

import contextlib
import csv
import errno
import json
import os
import time

from data_prep import config

SPLITS = ("train", "val", "test")

# How long to wait for another process's rebuild of the same split before giving up.
LOCK_TIMEOUT_S = 900
LOCK_POLL_S = 0.5


def _relink(src_abs: str, dst_abs: str) -> None:
    """Create/refresh a symlink dst_abs -> src_abs."""
    if os.path.islink(dst_abs) or os.path.exists(dst_abs):
        os.remove(dst_abs)
    os.symlink(src_abs, dst_abs)


@contextlib.contextmanager
def _split_lock(split_dir: str, timeout: float = LOCK_TIMEOUT_S):
    """Serialise rebuilds of one split across processes.

    Two trainers starting at once (a second OWL variant, say) would otherwise
    interleave their rebuilds of the same shared directory. O_CREAT|O_EXCL is
    atomic on a local filesystem, which is all this needs.
    """
    os.makedirs(split_dir, exist_ok=True)
    lock_path = os.path.join(split_dir, ".rebuild.lock")
    deadline = time.time() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # Reclaim a lock left behind by a killed process.
            try:
                if time.time() - os.path.getmtime(lock_path) > timeout:
                    os.unlink(lock_path)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(
                    f"timed out waiting for another process to finish rebuilding "
                    f"{split_dir} (stale lock? remove {lock_path})")
            time.sleep(LOCK_POLL_S)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(lock_path)


def _reconcile_links(img_dir: str, wanted: dict[str, str]) -> dict[str, int]:
    """Make img_dir hold exactly `wanted` (basename -> absolute source), in place.

    Deliberately NOT a wipe-and-rebuild. This directory is shared by every OWL
    variant, and FolderDataset scans it wholesale, so removing it out from under
    a running trainer empties its dataset mid-epoch — and a rebuild that is
    interrupted leaves a silently truncated training set with no error at all.
    Reconciling means a concurrent reader only ever sees the old state, the new
    state, or a superset of both; never an empty or half-built directory.
    """
    os.makedirs(img_dir, exist_ok=True)
    stats = {"added": 0, "retargeted": 0, "removed": 0, "kept": 0}

    existing = {}
    with os.scandir(img_dir) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            existing[entry.name] = entry.path

    for base, src in wanted.items():
        dst = os.path.join(img_dir, base)
        if base in existing:
            try:
                if os.readlink(dst) == src:
                    stats["kept"] += 1
                    continue
            except OSError:
                pass  # a real file, or an unreadable link: replace it
            _relink(src, dst)
            stats["retargeted"] += 1
        else:
            try:
                os.symlink(src, dst)
                stats["added"] += 1
            except FileExistsError:
                _relink(src, dst)
                stats["retargeted"] += 1

    for base, path in existing.items():
        if base not in wanted:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
            stats["removed"] += 1

    return stats


def build_mdo_dataset(
    crops_img_dir: str = config.CROPS_IMG_DIR,
    crops_json_dir: str = config.CROPS_JSON_DIR,
    out_dir: str = config.MDO_DATA_DIR,
) -> tuple[str, dict]:
    """
    Build the OWL point-CSV mirror for every split that has a coco.json.

    Returns (out_dir, counts) where counts maps split -> number of images (crops).

    Safe to call while another process is training from the same directory: the
    image folder is reconciled in place under a per-split lock rather than
    deleted and rebuilt, and gt.csv is swapped in atomically.
    """
    if not os.path.isdir(crops_img_dir):
        # os.symlink happily creates links to nothing, so a wrong crops root
        # yields a full set of dangling links and a training run that fails one
        # crop at a time. Refuse up front instead.
        raise FileNotFoundError(
            f"crops directory does not exist: {crops_img_dir} — refusing to build "
            "a mirror of dangling symlinks")

    counts: dict[str, int] = {}

    for split in SPLITS:
        coco_path = os.path.join(crops_json_dir, split, "coco.json")
        if not os.path.exists(coco_path):
            counts[split] = 0
            continue

        with open(coco_path) as f:
            coco = json.load(f)

        split_dir = os.path.join(out_dir, split)
        img_dir = os.path.join(split_dir, "images")

        # image_id -> (basename, width, height), and the links the split should hold
        id_to_meta: dict[int, tuple[str, int, int]] = {}
        wanted: dict[str, str] = {}
        for img in coco["images"]:
            rel = img["file_name"]                       # e.g. "Bird_A/A_DJI_0001_0_0.JPG"
            base = os.path.basename(rel)
            id_to_meta[img["id"]] = (base, img["width"], img["height"])
            wanted[base] = os.path.join(crops_img_dir, rel)

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

        gt_csv = os.path.join(split_dir, "gt.csv")
        with _split_lock(split_dir):
            _reconcile_links(img_dir, wanted)

            # Write then rename: open(gt_csv, "w") truncates in place, so a reader
            # mid-write sees an empty or half-written CSV. os.replace is atomic.
            tmp_csv = gt_csv + f".tmp.{os.getpid()}"
            try:
                with open(tmp_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["images", "x", "y", "labels"])
                    writer.writerows(rows)
                os.replace(tmp_csv, gt_csv)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp_csv)

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
