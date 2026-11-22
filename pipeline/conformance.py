"""
The conformance suite every adapter must pass before its data enters a result.

The suite is shared, not written per programme: that is what makes "the Arkansas adapter
works" a statement with the same meaning as "the MDC adapter works".

Two of these checks exist because of specific, expensive mistakes in this project's
history.  A loader that guessed annotation file names silently skipped 140 photographs
and 3,138 birds; a parser that required a leading label silently discarded 2,646 boxes.
Both were found months later, and both would have been caught by `bird_totals` (which
compares parsed totals against what the source declares) or by `no_silent_skips` (which
demands a reason for every unusable photograph).

    from pipeline import conformance
    report = conformance.run(adapter)
    print(report.format())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pipeline.adapters.base import DatasetAdapter
from pipeline.records import ALTITUDE_REFERENCES, DECOY_VALUES

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


@dataclass
class Report:
    source: str
    checks: list[Check] = field(default_factory=list)
    photographs: int = 0
    birds: int = 0

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def format(self) -> str:
        head = (f"Conformance — {self.source}: {self.photographs:,} photographs, "
                f"{self.birds:,} birds")
        lines = [head, "-" * len(head)]
        for c in self.checks:
            mark = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL"}[c.status]
            lines.append(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        lines.append(f"  => {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def run(adapter: DatasetAdapter, sample: int | None = None,
        expected_birds: dict[str, int] | None = None,
        check_pixels: bool = True) -> Report:
    """Run every check against an adapter.

    `sample` limits the per-photograph checks to the first N photographs of each
    collection, for a quick pass over a large corpus; `expected_birds` is the source's
    own declared totals per collection, when it provides them.
    """
    report = Report(source=adapter.source)

    all_photos = list(adapter.all_photographs())
    usable = [p for p in all_photos if p.usable]
    report.photographs = len(usable)

    if sample:
        by_collection: dict[str, list] = {}
        for p in usable:
            by_collection.setdefault(p.collection, []).append(p)
        checked = [p for photos in by_collection.values() for p in photos[:sample]]
    else:
        checked = usable

    # 1. Unusable photographs are reported, never dropped in silence.
    unusable = [p for p in all_photos if not p.usable]
    without_reason = [p for p in unusable if not p.skip_reason]
    report.add("no_silent_skips",
               FAIL if without_reason else (WARN if unusable else PASS),
               f"{len(unusable)} unusable photograph(s)"
               + (f", {len(without_reason)} without a reason" if without_reason else "")
               + (f"; first: {unusable[0].photo_id} ({unusable[0].skip_reason})" if unusable else ""))

    # 2. Identity is unique and stable.
    ids = [p.photo_id for p in all_photos]
    duplicates = {i for i in ids if ids.count(i) > 1} if len(ids) != len(set(ids)) else set()
    report.add("unique_ids", FAIL if duplicates else PASS,
               f"{len(duplicates)} duplicate id(s)" if duplicates else f"{len(ids):,} ids")

    # 3. Files exist and match their recorded size.
    missing, mismatched = [], []
    for p in checked:
        if not os.path.exists(p.path):
            missing.append(p.photo_id)
            continue
        if check_pixels and p.width_px and p.camera.image_width_px:
            if p.width_px != p.camera.image_width_px and p.camera.source == "delivery":
                mismatched.append(p.photo_id)
    report.add("files_present", FAIL if missing else PASS,
               f"{len(missing)} missing" if missing else f"{len(checked):,} checked")
    report.add("recorded_size_matches", FAIL if mismatched else PASS,
               f"{len(mismatched)} disagree with the delivery table" if mismatched
               else "declared widths agree with the files")

    # 4. Annotations parse, lie inside the frame, and have positive area.
    #    A box overhanging the edge by a pixel or two is an annotator touching the border,
    #    and the tiler clips it; a box hundreds of pixels out means the annotations belong
    #    to a different image size, which is the mistake this check exists to catch.  The
    #    two are separated by a tolerance relative to the frame.
    tolerance_fraction = 0.05
    gross, slight, degenerate, total, worst = [], 0, [], 0, 0.0
    worst_id = None
    for p in checked:
        for a in adapter.annotations(p):
            total += 1
            xs, ys = a.coords[0::2], a.coords[1::2]
            if p.width_px and p.height_px:
                over = max(-min(xs), -min(ys), max(xs) - p.width_px, max(ys) - p.height_px)
                if over > 0:
                    if over > worst:
                        worst, worst_id = over, p.photo_id
                    if over > tolerance_fraction * min(p.width_px, p.height_px):
                        gross.append(p.photo_id)
                    else:
                        slight += 1
            if a.geometry == "box" and not (a.coords[2] > a.coords[0] and a.coords[3] > a.coords[1]):
                degenerate.append(p.photo_id)
    report.birds = total
    report.add("annotations_in_frame",
               FAIL if gross else (WARN if slight else PASS),
               (f"{len(gross)} annotation(s) far outside their photograph "
                f"(worst {worst:.0f} px, {worst_id})" if gross else
                f"{slight} of {total:,} annotations overhang the edge "
                f"(max {worst:.0f} px, clipped by the tiler)" if slight else
                f"{total:,} annotations checked"))
    report.add("annotations_non_degenerate", FAIL if degenerate else PASS,
               f"{len(degenerate)} zero-area box(es)" if degenerate else "all boxes have area")

    # 5. Declared totals, where the source states them.
    expected_birds = expected_birds or getattr(adapter, 'declared_birds', None)
    if expected_birds:
        wrong = []
        for collection in adapter.collections():
            if collection.name not in expected_birds:
                continue
            got = sum(len(adapter.annotations(p)) for p in adapter.photographs(collection))
            if got != expected_birds[collection.name]:
                wrong.append(f"{collection.name}: {got:,} parsed vs {expected_birds[collection.name]:,} declared")
        report.add("bird_totals", FAIL if wrong else PASS,
                   "; ".join(wrong) if wrong else "parsed totals match the source's own")
    else:
        report.add("bird_totals", WARN, "the source declares no totals to check against")

    # 6. Units and references are declared.
    bad_reference = [p.photo_id for p in usable
                     if p.altitude_m is not None and p.altitude_reference not in ALTITUDE_REFERENCES]
    report.add("altitude_reference_declared", FAIL if bad_reference else PASS,
               f"{len(bad_reference)} altitude(s) without a declared reference"
               if bad_reference else "every altitude declares what it measures from")

    bad_decoy = [p.photo_id for p in usable if p.decoys and p.decoys not in DECOY_VALUES]
    report.add("decoy_values_valid", FAIL if bad_decoy else PASS,
               f"{len(bad_decoy)} unexpected value(s)" if bad_decoy else "")

    # 7. Vocabularies: canonical terms are valid, raw labels are retained, and anything
    #    the crosswalk does not cover is named rather than silently dropped.
    crosswalks = adapter.crosswalks()
    for vocabulary, crosswalk in crosswalks.items():
        raws = {getattr(p, f"{vocabulary}_raw", None) for p in usable}
        unmapped = crosswalk.unmapped(raws)
        lost = [p.photo_id for p in usable
                if getattr(p, vocabulary, None) and not getattr(p, f"{vocabulary}_raw", None)]
        status = WARN if unmapped else PASS
        report.add(f"{vocabulary}_crosswalk", FAIL if lost else status,
                   (f"{len(lost)} canonical value(s) with no raw label; " if lost else "")
                   + (f"not in crosswalk: {', '.join(unmapped)}" if unmapped
                      else f"{len([r for r in raws if r])} label(s) all mapped"))

    # 8. Split roles are valid and singular.
    bad_roles = [p.photo_id for p in usable
                 if p.split_role not in ("train", "test", "unassigned")]
    report.add("split_roles_valid", FAIL if bad_roles else PASS,
               f"{len(bad_roles)} unexpected role(s)" if bad_roles else "")

    # 9. Determinism: the same call twice gives the same records.
    again = [p.photo_id for p in adapter.all_photographs()]
    report.add("deterministic", PASS if again == ids else FAIL,
               "two passes agree" if again == ids else "two passes differ")

    return report
