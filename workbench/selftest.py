#!/usr/bin/env python
"""
Checks that the workbench's own logic does what it claims — no GPU, nothing written
outside a temporary folder, a few seconds:

    python -m workbench.selftest

  A  Tiling      survey photographs are cut at exactly the offsets the training crops
                 were (read back from the crop file names under crops/)
  B  Cells       every point of a photograph belongs to exactly one cell
  C  Merging     a bird seen by two overlapping tiles is counted once; two birds either
                 side of a boundary are both kept; a truncated duplicate is dropped
  D  Counting    review decisions change a count exactly as documented
  E  Labels      exported labels read back through the project's own parser
  F  Jobs        a detached job reports success, failure and cancellation
  G  Versions    each run's "best" checkpoint is the file analysis.predict loads, and its
                 "last" one is a different, existing file
  H  Lineage     a run started with --init-from names the version it came from
  I  Flags       this checkout's entry points declare --init-from and --checkpoint
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from data_prep import config

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def test_tiling() -> None:
    from PIL import Image
    from workbench import detections
    crops_dir = os.path.join(config.CROPS_IMG_DIR, "Bird_A")
    photo_dir = os.path.join(config.DEFAULT_DATASET_ROOT, "Bird_A")
    if not (os.path.isdir(crops_dir) and os.path.isdir(photo_dir)):
        check("A tiling", True, "skipped (no crops/Bird_A to compare against)")
        return
    names = os.listdir(crops_dir)
    photos = sorted(f for f in os.listdir(photo_dir) if f.lower().endswith(".jpg"))[:3]
    compared = 0
    for photo in photos:
        stem = os.path.splitext(photo)[0]
        pattern = re.compile(rf"^A_{re.escape(stem)}_(\d+)_(\d+)\.JPG$")
        offsets = {(int(m.group(1)), int(m.group(2))) for n in names if (m := pattern.match(n))}
        if not offsets:
            continue
        with Image.open(os.path.join(photo_dir, photo)) as im:
            w, h = im.size
        ours = {(y, x) for y in detections.origins(h) for x in detections.origins(w)}
        check(f"A tiling {photo}", ours == offsets,
              f"{len(ours)} tiles, {'same' if ours == offsets else 'DIFFERENT'} offsets as crops/")
        compared += 1
    if not compared:
        check("A tiling", True, "skipped (no crops found for the sample photographs)")


def test_cells() -> None:
    from workbench import detections
    for w, h in ((5472, 3648), (5120, 3584), (512, 512), (700, 400)):
        cells = detections.cells(w, h)
        rng = np.random.default_rng(0)
        pts = rng.uniform([0, 0], [w, h], size=(2000, 2))
        owners = [sum(c.contains(x, y) for c in cells) for x, y in pts]
        agree = all(cells[detections.cell_index(x, y, w, h)].contains(x, y) for x, y in pts)
        check(f"B cells {w}x{h}", all(o == 1 for o in owners) and agree,
              f"{len(cells)} cells; every point owned once")


def _box(x, y, w=17.0, h=17.0, s=0.9):
    return [x, y, w, h, s]


def test_merge() -> None:
    from workbench import detect, detections
    W, H = 1400, 600                        # three tiles across (0, 460, 888), two down
    xs, ys = detections.origins(W), detections.origins(H)
    cut = detections.cuts(W)[0]             # boundary between the first two columns
    t0x, t1x = xs[0], xs[1]

    def tiles(**per_tile):
        out = []
        for ty in ys:
            for tx in xs:
                out.append((tx, ty, np.asarray(per_tile.get(f"{tx},{ty}", []), float).reshape(-1, 5)))
        return out

    # one bird on the boundary, both tiles see it, centres 1 px either side of the cut
    bird = [_box(cut - 1 - t0x, 100), ]
    same = [_box(cut + 1 - t1x, 100)]
    merged, _ = detect.merge(tiles(**{f"{t0x},0": bird, f"{t1x},0": same}), W, H)
    check("C same bird, two tiles", len(merged) == 1, f"{len(merged)} kept (want 1)")

    # two different birds 12 px apart across the boundary
    a, b = [_box(cut - 6 - t0x, 200)], [_box(cut + 6 - t1x, 200)]
    merged, _ = detect.merge(tiles(**{f"{t0x},0": a, f"{t1x},0": b}), W, H)
    check("C two birds across a boundary", len(merged) == 2, f"{len(merged)} kept (want 2)")

    # a bird cut in half at tile 0's right edge (seen whole by tile 1) — the half is dropped
    edge = t0x + config.CROP_SIZE
    half = [_box(edge - 4 - t0x, 300, w=8.0)]
    whole = [_box(edge - t1x, 300)]
    merged, _ = detect.merge(tiles(**{f"{t0x},0": half, f"{t1x},0": whole}), W, H)
    check("C truncated duplicate", len(merged) == 1, f"{len(merged)} kept (want 1)")

    # point detector: same bird ±2 px from two tiles; two birds 10 px apart
    nan = float("nan")
    p1 = [[cut - 1 - t0x, 150, nan, nan, 0.8]]
    p2 = [[cut + 1 - t1x, 151, nan, nan, 0.7], [cut + 9 - t1x, 150, nan, nan, 0.7]]
    merged, _ = detect.merge(tiles(**{f"{t0x},0": p1, f"{t1x},0": p2}), W, H)
    check("C points", len(merged) == 2, f"{len(merged)} kept (want 2: one merged pair + one)")

    # a photograph smaller than one tile passes straight through
    small = detect.merge([(0, 0, np.asarray([_box(100, 100), _box(300, 200)]))], 400, 300)[0]
    check("C single-tile photograph", len(small) == 2, f"{len(small)} kept (want 2)")


def test_counting() -> None:
    from workbench import detections
    tmp = tempfile.mkdtemp(prefix="wb-selftest-")

    class FakeResult:
        dir = tmp
        threshold = 0.5

    review = detections.Review(FakeResult())
    photo = detections.PhotoResult(file="images/a.jpg", width=1000, height=1000,
                                   xy=np.array([[10, 10], [20, 20], [30, 30], [40, 40]], float),
                                   wh=np.full((4, 2), 17.0),
                                   score=np.array([0.9, 0.6, 0.3, 0.2]))
    c = review.counts(photo)
    check("D model count", c.model == 2 and c.final == 2, f"{c.model} → {c.final} (want 2 → 2)")
    review.set_mark(photo.file, 0, detections.NOT_BIRD)   # -1
    review.set_mark(photo.file, 2, detections.BIRD)       # +1 (promoted)
    review.set_mark(photo.file, 1, detections.BIRD)       # confirmed: no change
    review.add(photo.file, 500, 500)                      # +1
    c = review.counts(photo)
    check("D reviewed count", c.final == 3 and c.rejected == 1 and c.promoted == 1 and c.added == 1,
          f"final {c.final} (want 3 = 2 − 1 + 1 + 1)")
    review.save()
    again = detections.Review(FakeResult()).counts(photo)
    check("D saved and reloaded", again.final == 3, "review.json round trip")


def test_labels() -> None:
    from data_prep.prepare_data import find_annotation, read_annotations
    tmp = tempfile.mkdtemp(prefix="wb-selftest-")
    boxes = [(10, 12, 27, 29), (100, 100, 140, 131)]
    with open(os.path.join(tmp, "p.txt"), "w") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"bird,{x1},{y1},{x2},{y2}\n")
    back = read_annotations(find_annotation(tmp, "p"))
    check("E labels round trip", [tuple(b) for b in back] == boxes, f"{len(back)} boxes read back")


def test_jobs() -> None:
    from workbench import jobs
    good = jobs.launch("selftest: ok", "selftest", ["true"])
    bad = jobs.launch("selftest: fail", "selftest", ["bash", "-c", "exit 3"])
    slow = jobs.launch("selftest: cancel", "selftest", ["sleep", "60"])
    deadline = time.time() + 10
    while time.time() < deadline and (good.status() == "running" or bad.status() == "running"):
        time.sleep(0.1)
    cancelled = jobs.cancel(slow)
    check("F job succeeded", good.status() == "succeeded", good.status())
    check("F job failed", bad.status() == "failed" and bad.exit_code() == 3,
          f"{bad.status()}, exit {bad.exit_code()}")
    check("F job cancelled", cancelled and slow.status() == "cancelled", slow.status())
    for j in (good, bad, slow):
        jobs.forget(j)


def test_versions() -> None:
    from workbench import catalog
    runs = [r for r in catalog.runs() if r.checkpoint]
    if not runs:
        check("G versions", True, "skipped (no trained runs)")
        return
    agree = sum(1 for r in runs if r.checkpoints() and r.checkpoints()[0].path == r.checkpoint)
    check("G best = analysis checkpoint", agree == len(runs), f"{agree}/{len(runs)} runs")
    lasts = [c for r in runs for c in r.checkpoints() if c.kind == "last"]
    ok = all(os.path.exists(c.path) for c in lasts)
    check("G last checkpoints exist", ok, f"{len(lasts)} runs have a separate last checkpoint")


def test_lineage() -> None:
    from workbench import catalog
    model = catalog.model("yolo11")
    parent = os.path.join(config.OUTPUT_DIR, "checkpoints", "yolo11", "2026-08-27_19-48-46",
                          "weights", "best.pt")
    run = catalog.Run(model=model, run_id="2099-12-31_23-59-59", path="/nonexistent",
                      checkpoint=None, metrics=None, info={"init_from": parent},
                      experiment={}, last_activity=0.0)
    got = run.parent
    check("H lineage from run_info", got is not None and got[1] == "2026-08-27_19-48-46",
          run.started_from())
    run2 = catalog.Run(model=model, run_id="2099-12-31_23-59-59", path="/nonexistent",
                       checkpoint=None, metrics=None, info={},
                       experiment={"pretrained": {"weights": parent}}, last_activity=0.0)
    check("H lineage from experiment.json", run2.parent is not None, run2.started_from())
    run3 = catalog.Run(model=model, run_id="2099-12-31_23-59-59", path="/nonexistent",
                       checkpoint=None, metrics=None, info={},
                       experiment={"pretrained": {"weights": "yolo11m"}}, last_activity=0.0)
    check("H public start", run3.parent is None and run3.init_from is None, run3.started_from())


def test_flags() -> None:
    from workbench import catalog
    missing = [f"{m.script} {flag}" for m in catalog.MODELS
               for flag in ("--init-from", "--checkpoint") if not catalog.supports(m, flag)]
    check("I entry-point flags", not missing,
          "all entry points declare --init-from and --checkpoint" if not missing
          else "missing: " + ", ".join(sorted(set(missing))))


def main() -> int:
    for test in (test_tiling, test_cells, test_merge, test_counting, test_labels, test_jobs,
                 test_versions, test_lineage, test_flags):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 — report, keep going
            check(test.__name__, False, f"{type(exc).__name__}: {exc}")
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
