#!/usr/bin/env python
"""
Visual check that the annotations line up with the birds, across every dataset.

Reads each dataset through the SAME loader the training pipeline uses
(find_image / find_annotation / read_annotations), so what you see here is exactly what
`--prepare` would consume.  If a dataset's labels are misread, mis-scaled or off by a
transform, it shows up in these panels rather than three steps later in a metric.

For each sampled image it writes three files:

    <ds>__<stem>__full.jpg     whole frame, downscaled, every box drawn
    <ds>__<stem>__zoom.jpg     a grid of birds at NATIVE resolution, coordinates printed
    <ds>__<stem>__coords.txt   every box as "idx  x1 y1 x2 y2  w h  [species]"

The zoom grid is the point of the tool.  These birds are 20-30 px in a 5472x3648 frame,
so on a downscaled full view a box that is off by 50 px still looks correct; at native
resolution it obviously is not.

Usage (from the rebuild/ directory):

    python -m data_prep.verify_annotations                     # 2 images per dataset
    python -m data_prep.verify_annotations --per-dataset 4
    python -m data_prep.verify_annotations --datasets Bird_I Bird_B
    python -m data_prep.verify_annotations --image DJI_0036    # one specific image
    python -m data_prep.verify_annotations --densest           # busiest image per dataset
"""

from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2

import data_prep.config as config
from data_prep.prepare_data import (discover_datasets, find_annotation, find_image,
                                    read_annotations)

GREEN = (0, 255, 0)      # the bird this panel is centred on
CYAN = (255, 200, 0)     # other labelled birds overlapping the patch
YELLOW = (0, 255, 255)


def species_labels(dataset_dir: str, stem: str) -> list[str]:
    """
    Per-bird species strings from a `<stem>_class.txt`, or [] when the dataset has none.

    Only Bird_H and Bird_I carry these.  The file lists the same boxes with a species in
    place of the `bird` label, so row order matches read_annotations() and the two can be
    zipped — but only when the counts agree, which they do not always do, so callers must
    tolerate a short list.
    """
    path = os.path.join(dataset_dir, stem + "_class.txt")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(line.rsplit(",", 4)[0].strip())
    return out


def draw_full(img, boxes, max_width: int = 1800):
    """Whole frame with every box drawn, downscaled to something a viewer can open."""
    view = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(view, (x1, y1), (x2, y2), GREEN, 3)
    h, w = view.shape[:2]
    if w > max_width:
        scale = max_width / w
        view = cv2.resize(view, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return view


def draw_zoom(img, boxes, labels, n: int, pad: int = 60, cell: int = 260):
    """
    Grid of birds at native resolution, each captioned with its own coordinates.

    Native resolution is deliberate: downscaling is what hides a misaligned box.
    """
    if not boxes:
        return None
    picks = list(range(len(boxes)))
    if len(picks) > n:                      # spread across the frame, not the first n
        step = len(picks) / n
        picks = [picks[int(i * step)] for i in range(n)]

    cols = min(3, len(picks))
    rows = (len(picks) + cols - 1) // cols
    canvas_h, canvas_w = rows * (cell + 34), cols * cell
    canvas = cv2.copyMakeBorder(cv2.resize(img[:1, :1], (canvas_w, canvas_h)),
                                0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    canvas[:] = (30, 30, 30)

    H, W = img.shape[:2]
    for slot, idx in enumerate(picks):
        x1, y1, x2, y2 = boxes[idx]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = max(pad, (x2 - x1) // 2 + pad, (y2 - y1) // 2 + pad)
        sx1, sy1 = max(0, cx - half), max(0, cy - half)
        sx2, sy2 = min(W, cx + half), min(H, cy + half)
        patch = img[sy1:sy2, sx1:sx2].copy()
        if patch.size == 0:
            continue
        # Draw EVERY box overlapping this patch, not just the focus one — otherwise a
        # labelled neighbour looks unlabelled and the panel reads as a missing annotation.
        for bx1, by1, bx2, by2 in boxes:
            if bx2 < sx1 or bx1 > sx2 or by2 < sy1 or by1 > sy2:
                continue
            colour = GREEN if (bx1, by1, bx2, by2) == (x1, y1, x2, y2) else CYAN
            cv2.rectangle(patch, (bx1 - sx1, by1 - sy1), (bx2 - sx1, by2 - sy1), colour, 2)
        ph, pw = patch.shape[:2]
        s = min(cell / pw, cell / ph)
        patch = cv2.resize(patch, (int(pw * s), int(ph * s)), interpolation=cv2.INTER_NEAREST)

        r, c = divmod(slot, cols)
        oy, ox = r * (cell + 34), c * cell
        canvas[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
        cap = f"#{idx}  ({x1},{y1})-({x2},{y2})  {x2-x1}x{y2-y1}px"
        cv2.putText(canvas, cap, (ox + 4, oy + cell + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1, cv2.LINE_AA)
        if idx < len(labels):
            cv2.putText(canvas, labels[idx][:34], (ox + 4, oy + cell + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1, cv2.LINE_AA)
    return canvas


def process(dataset_dir: str, stem: str, out_dir: str, n_zoom: int) -> dict | None:
    ds = os.path.basename(dataset_dir)
    img_path = find_image(dataset_dir, stem)
    ann_path = find_annotation(dataset_dir, stem)
    if img_path is None or ann_path is None:
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None

    boxes = read_annotations(ann_path)
    labels = species_labels(dataset_dir, stem)
    H, W = img.shape[:2]
    tag = f"{ds}__{stem}"

    cv2.imwrite(os.path.join(out_dir, f"{tag}__full.jpg"), draw_full(img, boxes),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    zoom = draw_zoom(img, boxes, labels, n_zoom)
    if zoom is not None:
        cv2.imwrite(os.path.join(out_dir, f"{tag}__zoom.jpg"), zoom,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(os.path.join(out_dir, f"{tag}__coords.txt"), "w") as f:
        f.write(f"# {ds} / {stem}\n# image   : {img_path}  ({W}x{H})\n")
        f.write(f"# labels  : {ann_path}\n# boxes   : {len(boxes)}"
                f"   species labels: {len(labels) or 'none'}\n")
        f.write(f"#\n# {'idx':>5s}  {'x1':>6s} {'y1':>6s} {'x2':>6s} {'y2':>6s}"
                f"  {'w':>4s} {'h':>4s}  species\n")
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            sp = labels[i] if i < len(labels) else ""
            f.write(f"  {i:5d}  {x1:6d} {y1:6d} {x2:6d} {y2:6d}"
                    f"  {x2-x1:4d} {y2-y1:4d}  {sp}\n")

    # Out-of-frame boxes are the failure this tool exists to catch.
    bad = [b for b in boxes if b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H]
    sizes = [max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
    return {"ds": ds, "stem": stem, "wh": (W, H), "n": len(boxes),
            "labels": len(labels), "bad": len(bad),
            "px": (min(sizes), max(sizes)) if sizes else (0, 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw annotations for visual verification")
    ap.add_argument("--datasets", nargs="*", default=None, help="default: all")
    ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--image", default=None, help="one specific image stem")
    ap.add_argument("--densest", action="store_true",
                    help="pick each dataset's busiest image instead of sampling")
    ap.add_argument("--zoom", type=int, default=6, help="birds per zoom grid")
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "annotation_check"))
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dirs = discover_datasets(config.DATASET_ROOT, config.EXCLUDED_DATASETS)
    if a.datasets:
        want = set(a.datasets)
        dirs = [d for d in dirs if os.path.basename(d) in want]
    if not dirs:
        print("No datasets matched."); return

    print(f"Dataset root : {config.DATASET_ROOT}")
    print(f"Output       : {a.out}\n")
    rng = random.Random(a.seed)
    rows = []
    for d in dirs:
        stems = sorted({os.path.splitext(f)[0] for f in os.listdir(d)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))})
        if a.image:
            stems = [s for s in stems if s == a.image]
        elif a.densest:
            stems = [max(stems, key=lambda s: len(read_annotations(find_annotation(d, s)))
                         if find_annotation(d, s) else 0)]
        else:
            # An empty frame verifies nothing, so sample from annotated images when
            # there are enough of them.
            annotated = [s for s in stems
                         if find_annotation(d, s) and read_annotations(find_annotation(d, s))]
            pool = annotated if len(annotated) >= a.per_dataset else stems
            stems = rng.sample(pool, min(a.per_dataset, len(pool)))
        for s in stems:
            r = process(d, s, a.out, a.zoom)
            if r:
                rows.append(r)
                flag = f"  <-- {r['bad']} BOX(ES) OUTSIDE FRAME" if r["bad"] else ""
                print(f"  {r['ds']:8s} {r['stem'][:34]:34s} {r['wh'][0]}x{r['wh'][1]}"
                      f"  boxes={r['n']:4d}  size={r['px'][0]}-{r['px'][1]}px"
                      f"  species={r['labels'] or '-'}{flag}")

    with open(os.path.join(a.out, "SUMMARY.txt"), "w") as f:
        f.write(f"Annotation verification — {config.DATASET_ROOT}\n\n")
        f.write(f"{'dataset':10s}{'image':34s}{'WxH':>13s}{'boxes':>7s}"
                f"{'px min-max':>12s}{'species':>9s}{'off-frame':>11s}\n")
        f.write("-" * 96 + "\n")
        for r in rows:
            wh = f"{r['wh'][0]}x{r['wh'][1]}"
            px = f"{r['px'][0]}-{r['px'][1]}"
            f.write(f"{r['ds']:10s}{r['stem'][:33]:34s}{wh:>13s}{r['n']:7d}"
                    f"{px:>12s}{r['labels'] or 0:9d}{r['bad']:11d}\n")
    print(f"\n{len(rows)} images written to {a.out}")
    bad = sum(r["bad"] for r in rows)
    print("No boxes fall outside their image." if bad == 0
          else f"WARNING: {bad} boxes fall outside their image bounds.")


if __name__ == "__main__":
    main()
