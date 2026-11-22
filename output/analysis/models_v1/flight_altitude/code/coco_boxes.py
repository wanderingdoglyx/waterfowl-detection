"""
Bird boxes on the source photographs, rebuilt from the project's COCO crop annotations.

The study measures birds on the original images, but the dataset every detector in this
project is trained and scored on is the 512 px crop set that data_prep.prepare_data writes
to output/crops_json/{train,val,test}/coco.json.  Reading the original label files instead
would measure a *different* corpus from the one the models saw: the crop set drops images
whose label file cannot be resolved, clips boxes that run past the edge of the photograph
(1,235 of them here, annotated as extending beyond the image), and carries the train / val /
test split that the benchmark actually used.  Every size in this study therefore comes from
the COCO files, mapped back to source-image pixels by this module.

Reconstruction is exact, not approximate.  prepare_data slides a 512 px window at a 460 px
stride and writes, for every tile, the intersection of each box with that tile; the tile
origin is encoded in the crop's file name.  So one bird appears once per tile it touches,
in tile coordinates, truncated wherever the tile cut it.  Translating by the tile origin
undoes the offset; what is left is to group the fragments of one bird *without* merging two
birds whose boxes overlap, which is common in these flocks.

The grouping rule is a verification, not a heuristic.  A candidate box is accepted only if,
for **every** tile it meets, the intersection of the candidate with that tile is a fragment
the COCO file actually contains.  Two overlapping birds fail that test — their union does
not slice into the fragments on record — while the pieces of one bird pass it by
construction.  Candidates come from the fragments themselves: a fragment whose left and top
edges lie inside its tile (or on the border of the crop grid) pins the box's top-left
corner, one whose right and bottom edges do pins the bottom-right, and whole fragments pin
all four at once.  Smaller candidates are settled first, so a bird is never absorbed into a
larger neighbour's box.

Checked against the label files the crops were built from: 317,386 of the corpus's 317,387
boxes come back, no box is invented, and no fragment is left unexplained.  The one box that
does not come back is 1 px wide and lies off the right edge of its photograph, so no crop
ever contained it.  A naive alternative — merging fragments that overlap — loses 62,544
boxes and invents 22,512 by fusing neighbours in the dense datasets, which is why the
verification is worth its cost (about 6 s for the whole corpus).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from common import REPO

CROPS_JSON = os.path.join(REPO, "output", "crops_json")
SPLITS = ("train", "val", "test")
CROP = 512                      # data_prep.config.CROP_SIZE
MAX_SPAN = 3 * CROP             # no bird in this corpus spans more than a few crops


@dataclass(frozen=True)
class Box:
    """One bird on its source photograph, in source pixels."""
    x1: int
    y1: int
    x2: int
    y2: int
    whole: bool     # some crop held the bird entire — the size a detector was shown
    edge: bool      # touches the border of the crop grid, so the photograph cut the bird
    crops: int      # crops the bird appears in

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class SourceImage:
    dataset: str
    stem: str
    split: str
    grid_w: int     # extent of the crop grid, the bound boxes were clipped against
    grid_h: int
    boxes: list[Box]


def _intersect(b, tile):
    x1, y1 = max(b[0], tile[0]), max(b[1], tile[1])
    x2, y2 = min(b[2], tile[0] + CROP), min(b[3], tile[1] + CROP)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _rebuild(tiles: set, fragments: list) -> tuple[list[Box], int]:
    """Source boxes for one photograph, from its (tile, fragment) pairs. Returns the boxes and
    the number of fragments no accepted box explains (0 everywhere in this corpus)."""
    grid_w = max(t[0] for t in tiles) + CROP
    grid_h = max(t[1] for t in tiles) + CROP
    pool = set(fragments)

    pinned = []                                   # (fragment, tile, top-left pinned, bottom-right pinned)
    for tile, f in fragments:
        tl = (f[0] > tile[0] or f[0] == 0) and (f[1] > tile[1] or f[1] == 0)
        br = (f[2] < tile[0] + CROP or f[2] == grid_w) and (f[3] < tile[1] + CROP or f[3] == grid_h)
        pinned.append((f, tile, tl, br))

    def accept(b) -> Box | None:
        sl = [(t, _intersect(b, t)) for t in tiles if _intersect(b, t)]
        if not all(s in pool for s in sl):
            return None
        pool.difference_update(sl)
        return Box(*b, whole=any(s[1] == b for s in sl),
                   edge=b[0] == 0 or b[1] == 0 or b[2] == grid_w or b[3] == grid_h,
                   crops=len(sl))

    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    boxes = [x for x in (accept(b) for b in sorted({f for f, _, tl, br in pinned if tl and br}, key=area))
             if x is not None]

    # Fragments still unspoken for belong to birds no single crop held whole: pair a pinned
    # top-left corner with a pinned bottom-right corner and put the same verification to it.
    rest = [(f, t, tl, br) for f, t, tl, br in pinned if (t, f) in pool]
    corners = {(f[0], f[1], g[2], g[3])
               for f, _, tl, _ in rest if tl
               for g, _, _, br in rest if br
               if 0 < g[2] - f[0] <= MAX_SPAN and 0 < g[3] - f[1] <= MAX_SPAN}
    boxes += [x for x in (accept(b) for b in sorted(corners, key=area)) if x is not None]
    return boxes, len(pool)


def load(splits=SPLITS, crops_json: str = None) -> dict[tuple[str, str], SourceImage]:
    """{(dataset, stem) -> SourceImage} for every photograph behind the given COCO splits."""
    from analysis.metadata import parse_crop_name

    root = crops_json or CROPS_JSON
    tiles: dict[tuple[str, str], set] = {}
    frags: dict[tuple[str, str], list] = {}
    split_of: dict[tuple[str, str], str] = {}

    for split in splits:
        with open(os.path.join(root, split, "coco.json")) as f:
            coco = json.load(f)
        meta = {}
        for im in coco["images"]:
            parsed = parse_crop_name(im["file_name"])
            if parsed is None:
                continue
            dataset, stem, row, col = parsed
            key = (dataset, stem)
            meta[im["id"]] = (key, col, row)
            tiles.setdefault(key, set()).add((col, row))
            frags.setdefault(key, [])
            split_of[key] = split                 # a photograph is tiled into one split only
        for a in coco["annotations"]:
            if a["image_id"] not in meta:
                continue
            key, col, row = meta[a["image_id"]]
            x, y, w, h = a["bbox"]
            frags[key].append(((col, row), (x + col, y + row, x + w + col, y + h + row)))

    out, orphans = {}, 0
    for key, tl in tiles.items():
        boxes, left = _rebuild(tl, frags[key])
        orphans += left
        out[key] = SourceImage(dataset=key[0], stem=key[1], split=split_of[key],
                               grid_w=max(t[0] for t in tl) + CROP,
                               grid_h=max(t[1] for t in tl) + CROP,
                               boxes=sorted(boxes, key=lambda b: (b.y1, b.x1)))
    if orphans:
        print(f"WARNING: {orphans} crop fragments could not be assigned to a source box")
    return out


if __name__ == "__main__":
    import collections

    src = load()
    n = sum(len(s.boxes) for s in src.values())
    per = collections.Counter(s.dataset for s in src.values())
    print(f"{len(src):,} photographs, {n:,} birds")
    print(f"  cut by the photograph's edge : {sum(b.edge for s in src.values() for b in s.boxes):,}")
    print(f"  never whole inside one crop  : {sum(not b.whole for s in src.values() for b in s.boxes):,}")
    for ds in sorted(per):
        bx = [b for s in src.values() if s.dataset == ds for b in s.boxes]
        print(f"  {ds}: {per[ds]:4d} images, {len(bx):7,} birds, "
              f"{sum(b.edge for b in bx):5,} edge, {sum(not b.whole for b in bx):5,} split across crops")
