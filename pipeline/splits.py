"""
Split specifications: which photographs train, which validate, which test — as data.

Two rules survive from the current pipeline and are kept exactly:

* the provider's own train/test marking is honoured and never reshuffled;
* train and validation come from a seeded shuffle of the training pool, so the split is
  reproducible as long as the pool is assembled in the same order.

Two rules are added, because a second programme needs them:

* **splits are made at the collection or site level** for any source that does not mark
  its own, so no survey site appears in both training and test.  A random split over
  photographs would put frames of the same flock, water body and day on both sides;
* a spec carries a **checksum** of its own membership, so a prediction set can be refused
  when it was scored against a different split.  Cross-programme experiments — "train on
  Missouri, test on Arkansas" — are then a spec rather than a code change.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from pipeline.records import Photograph

TRAIN, VAL, TEST = "train", "val", "test"


@dataclass
class SplitSpec:
    """A named, checksummed assignment of photographs to train / val / test."""

    name: str
    sources: list[str]
    assignment: dict[str, str] = field(default_factory=dict)   # photo_id -> split
    val_fraction: float | None = None
    seed: int | None = None
    rule: str = ""

    @property
    def checksum(self) -> str:
        payload = "\n".join(f"{pid},{split}" for pid, split in sorted(self.assignment.items()))
        return hashlib.sha256(payload.encode()).hexdigest()

    def counts(self) -> dict[str, int]:
        out = {TRAIN: 0, VAL: 0, TEST: 0}
        for split in self.assignment.values():
            out[split] = out.get(split, 0) + 1
        return out

    def ids(self, split: str) -> list[str]:
        return sorted(pid for pid, s in self.assignment.items() if s == split)

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump({**asdict(self), "checksum": self.checksum,
                       "counts": self.counts()}, f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "SplitSpec":
        with open(path) as f:
            d = json.load(f)
        d.pop("checksum", None)
        d.pop("counts", None)
        return cls(**d)


def provider_marked(photos: Iterable[Photograph], val_fraction: float, seed: int,
                    name: str = "provider") -> SplitSpec:
    """The current pipeline's rule, reproduced exactly.

    The training pool is shuffled with `random.seed(seed)` in the order the photographs
    are yielded — collections in sorted order, rows in spreadsheet order — and the first
    `max(1, int(n * val_fraction))` become validation.  Photographs the provider marked
    for testing are never touched.
    """
    photos = [p for p in photos if p.usable]
    # A photograph the provider did not mark joins the training pool, and is then split
    # into train/validation like the rest.  This is what makes a hand-assembled folder
    # with no index file usable without inventing a test set for it.
    train_pool = [p.photo_id for p in photos if p.split_role in (TRAIN, "unassigned")]
    test_pool = [p.photo_id for p in photos if p.split_role == TEST]

    random.seed(seed)
    random.shuffle(train_pool)
    n_val = max(1, int(len(train_pool) * val_fraction))

    assignment = {pid: VAL for pid in train_pool[:n_val]}
    assignment.update({pid: TRAIN for pid in train_pool[n_val:]})
    assignment.update({pid: TEST for pid in test_pool})

    return SplitSpec(name=name, sources=sorted({p.source for p in photos}),
                     assignment=assignment, val_fraction=val_fraction, seed=seed,
                     rule="provider train/test marks; seeded shuffle for validation")


def site_holdout(photos: Iterable[Photograph], test_sites: Sequence[str],
                 val_sites: Sequence[str] = (), name: str = "site-holdout") -> SplitSpec:
    """Hold out whole sites, for a source that does not mark its own splits.

    This is the rule the Arkansas domain-shift protocol requires: no site contributes to
    more than one split, so no flock, water body or day appears on both sides.
    """
    photos = [p for p in photos if p.usable]
    test_sites, val_sites = set(test_sites), set(val_sites)
    assignment = {}
    for p in photos:
        key = p.site or p.collection
        assignment[p.photo_id] = TEST if key in test_sites else VAL if key in val_sites else TRAIN
    return SplitSpec(name=name, sources=sorted({p.source for p in photos}),
                     assignment=assignment,
                     rule=f"held-out sites: {sorted(test_sites)}; validation sites: {sorted(val_sites)}")


def cross_source(train_photos: Iterable[Photograph], test_photos: Iterable[Photograph],
                 name: str = "cross-source") -> SplitSpec:
    """Train on one programme, test on another — the Missouri-to-Arkansas experiment.

    No model, metric or analysis module changes to support this: it is a spec.
    """
    assignment = {p.photo_id: TRAIN for p in train_photos if p.usable}
    assignment.update({p.photo_id: TEST for p in test_photos if p.usable})
    sources = sorted({p.source for p in list(train_photos) + list(test_photos)}) \
        if not isinstance(train_photos, (list, tuple)) else \
        sorted({p.source for p in train_photos} | {p.source for p in test_photos})
    return SplitSpec(name=name, sources=sources, assignment=assignment,
                     rule="train on one source, test on another")
