"""
A detection result — one trained model run over one survey — and its review.

    output/workbench/detections/<id>/
        detect.json        what was run: survey, model, run, checkpoint, threshold, tiling
        detections.jsonl   one line per photograph, appended as the job finishes each one
        review.json        the reviewer's decisions (appears on the first review)

Every detection above the score floor (0.05, the floor every model's --eval uses) is kept
with its score, so the counting threshold is a setting of the result, not of the run:
moving it in the reviewer or a report never means running the model again.

How a photograph is counted
---------------------------
The default threshold is the run's t*, the score cut that minimised counting error (MAE)
on the test split — the operating point the published P/R/F1/MAE are read at.  A review
then changes the count in exactly three ways:

    count = detections ≥ threshold                (the model's count)
          − those the reviewer marked "not a bird"
          + those below the threshold marked "bird"
          + birds the reviewer added by hand

A detection nobody looked at keeps the model's verdict, so a partly reviewed survey still
has a sensible count, and every report says how much of it a person has checked.

Cells
-----
Review happens one cell at a time.  The cells are the tiles the model saw, trimmed to the
part each tile *owns*: neighbouring tiles overlap by 52 px (more at the right and bottom
edges, where the last tile is aligned to the edge), and the boundary between two tiles runs
through the middle of their overlap.  Every point of a photograph belongs to exactly one
cell, so every detection — and every bird added by hand — is reviewed exactly once.  The
same boundaries decide which tile's detection of a bird is kept when two tiles both see
it (workbench/detect.py).
"""

from __future__ import annotations

import bisect
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from data_prep import config
from data_prep.prepare_data import _tile_origins
from workbench import paths
from workbench.util import read_json, slug, timestamp, write_json

BIRD, NOT_BIRD = "bird", "not_bird"


# ── Geometry ──────────────────────────────────────────────────────────────────

def stride() -> int:
    """The training crops' stride (pipeline/tiling.py): 512 * (1 - 0.20 / 2) = 460 px."""
    return int(config.CROP_SIZE * (1 - config.CROP_OVERLAP / 2))


def origins(dim: int) -> list[int]:
    return _tile_origins(dim, stride(), config.CROP_SIZE)


def cuts(dim: int) -> list[float]:
    """Cell boundaries along one axis: the middle of each pair of neighbouring tiles'
    overlap."""
    o = origins(dim)
    return [(o[i + 1] + o[i] + config.CROP_SIZE) / 2.0 for i in range(len(o) - 1)]


def bounds(dim: int) -> list[tuple[float, float]]:
    edges = [0.0, *cuts(dim), float(dim)]
    return list(zip(edges[:-1], edges[1:]))


@dataclass(frozen=True)
class Cell:
    index: int
    row: int
    col: int
    x0: float     # owned rectangle, photograph pixels
    y0: float
    x1: float
    y1: float
    tile_x: int   # origin of the 512 px tile the model saw
    tile_y: int

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def cells(width: int, height: int) -> list[Cell]:
    xs, ys = origins(width), origins(height)
    out = []
    for r, (y0, y1) in enumerate(bounds(height)):
        for c, (x0, x1) in enumerate(bounds(width)):
            out.append(Cell(len(out), r, c, x0, y0, x1, y1, xs[c], ys[r]))
    return out


def n_cells(width: int, height: int) -> int:
    return len(origins(width)) * len(origins(height))


def cell_index(x: float, y: float, width: int, height: int) -> int:
    col = bisect.bisect_right(cuts(width), x)
    row = bisect.bisect_right(cuts(height), y)
    return row * len(origins(width)) + col


# ── One photograph's detections ───────────────────────────────────────────────

@dataclass
class PhotoResult:
    file: str                  # the survey's relative photo path — the key everywhere
    width: int
    height: int
    xy: np.ndarray             # [N, 2] detection centres, photograph pixels
    wh: np.ndarray             # [N, 2] box sizes; NaN for point detectors
    score: np.ndarray          # [N]
    n_raw: int = 0             # tile detections before merging overlaps
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.score)

    @classmethod
    def from_line(cls, payload: dict) -> "PhotoResult":
        dets = np.asarray(payload.get("dets") or np.zeros((0, 5)), dtype=np.float64)
        if dets.size == 0:
            dets = np.zeros((0, 5))
        return cls(file=payload["photo"], width=int(payload["width"]),
                   height=int(payload["height"]), xy=dets[:, 0:2], wh=dets[:, 2:4],
                   score=dets[:, 4], n_raw=int(payload.get("n_raw", 0)),
                   seconds=float(payload.get("seconds", 0.0)))

    def to_line(self) -> dict:
        dets = []
        for (x, y), (w, h), s in zip(self.xy, self.wh, self.score):
            dets.append([round(float(x), 1), round(float(y), 1),
                         None if np.isnan(w) else round(float(w), 1),
                         None if np.isnan(h) else round(float(h), 1),
                         round(float(s), 4)])
        return {"photo": self.file, "width": self.width, "height": self.height,
                "n_raw": self.n_raw, "seconds": round(self.seconds, 2), "dets": dets}


# ── Review decisions ──────────────────────────────────────────────────────────

@dataclass
class Counts:
    model: int          # detections at or above the threshold
    rejected: int       # ... that a reviewer marked "not a bird"
    promoted: int       # detections below the threshold a reviewer marked "bird"
    confirmed: int      # detections at/above the threshold a reviewer marked "bird"
    added: int          # birds added by hand
    cells_done: int
    cells_total: int

    @property
    def final(self) -> int:
        return self.model - self.rejected + self.promoted + self.added

    @property
    def changed(self) -> int:
        return self.final - self.model

    @property
    def reviewed(self) -> bool:
        return self.cells_total > 0 and self.cells_done >= self.cells_total


class Review:
    """The reviewer's decisions for one detection result, saved after every change."""

    def __init__(self, result: "Result"):
        self.result = result
        self.path = os.path.join(result.dir, "review.json")
        data = read_json(self.path, {}) or {}
        self.threshold: float = float(data.get("threshold", result.threshold))
        self.photos: dict[str, dict] = data.get("photos", {})
        self.started: str | None = data.get("started")
        self.updated: str | None = data.get("updated")

    def _p(self, key: str) -> dict:
        entry = self.photos.setdefault(key, {})
        entry.setdefault("marks", {})
        entry.setdefault("added", [])
        entry.setdefault("cells_done", [])
        return entry

    # detections
    def mark(self, key: str, index: int) -> str | None:
        return self.photos.get(key, {}).get("marks", {}).get(str(index))

    def marks(self, key: str) -> dict[int, str]:
        return {int(k): v for k, v in self.photos.get(key, {}).get("marks", {}).items()}

    def set_mark(self, key: str, index: int, value: str | None) -> None:
        marks = self._p(key)["marks"]
        if value is None:
            marks.pop(str(index), None)
        else:
            marks[str(index)] = value

    # birds added by hand
    def added(self, key: str) -> list[list[float]]:
        return self.photos.get(key, {}).get("added", [])

    def add(self, key: str, x: float, y: float) -> int:
        self._p(key)["added"].append([round(float(x), 1), round(float(y), 1)])
        return len(self._p(key)["added"]) - 1

    def remove_added(self, key: str, index: int) -> list[float]:
        return self._p(key)["added"].pop(index)

    def insert_added(self, key: str, index: int, point: list[float]) -> None:
        self._p(key)["added"].insert(index, point)

    # progress
    def cells_done(self, key: str) -> set[int]:
        return set(self.photos.get(key, {}).get("cells_done", []))

    def set_cell_done(self, key: str, cell: int, done: bool = True) -> None:
        current = set(self._p(key)["cells_done"])
        current = current | {cell} if done else current - {cell}
        self._p(key)["cells_done"] = sorted(current)

    @property
    def touched(self) -> bool:
        return any(v.get("marks") or v.get("added") or v.get("cells_done")
                   for v in self.photos.values())

    def save(self) -> None:
        now = timestamp()
        self.started = self.started or now
        self.updated = now
        write_json(self.path, {"threshold": self.threshold, "photos": self.photos,
                               "started": self.started, "updated": self.updated}, indent=None)

    def counts(self, photo: PhotoResult, threshold: float | None = None) -> Counts:
        t = self.threshold if threshold is None else threshold
        above = photo.score >= t
        marks = self.marks(photo.file)
        rejected = sum(1 for i, m in marks.items() if m == NOT_BIRD and i < photo.n and above[i])
        promoted = sum(1 for i, m in marks.items() if m == BIRD and i < photo.n and not above[i])
        confirmed = sum(1 for i, m in marks.items() if m == BIRD and i < photo.n and above[i])
        return Counts(model=int(above.sum()), rejected=rejected, promoted=promoted,
                      confirmed=confirmed, added=len(self.added(photo.file)),
                      cells_done=len(self.cells_done(photo.file)),
                      cells_total=n_cells(photo.width, photo.height))

    def is_bird(self, photo: PhotoResult, index: int) -> bool:
        """The final verdict on one detection: the reviewer's mark, else the model's."""
        mark = self.mark(photo.file, index)
        if mark is not None:
            return mark == BIRD
        return bool(photo.score[index] >= self.threshold)


# ── A detection result ────────────────────────────────────────────────────────

class Result:
    def __init__(self, result_id: str, meta: dict):
        self.id = result_id
        self.meta = meta
        self._photos: list[PhotoResult] | None = None
        self._review: Review | None = None

    @property
    def dir(self) -> str:
        return os.path.join(paths.DETECTIONS_DIR, self.id)

    @property
    def lines_path(self) -> str:
        return os.path.join(self.dir, "detections.jsonl")

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.25))

    @property
    def title(self) -> str:
        last = " (last checkpoint)" if self.meta.get("checkpoint_kind") == "last" else ""
        return (f"{self.meta.get('survey')} · {self.meta.get('model_label')} "
                f"{self.meta.get('run')}{last}")

    def save_meta(self) -> None:
        write_json(os.path.join(self.dir, "detect.json"), self.meta)

    def photos(self, reload: bool = False) -> list[PhotoResult]:
        if self._photos is None or reload:
            found: dict[str, PhotoResult] = {}
            if os.path.exists(self.lines_path):
                with open(self.lines_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pr = PhotoResult.from_line(json.loads(line))
                        except (ValueError, KeyError):
                            continue          # a line cut short by a crash mid-write
                        found[pr.file] = pr   # a re-run photo replaces its old line
            order = {p.file: i for i, p in enumerate(self.survey().photos)} if self.survey() else {}
            self._photos = sorted(found.values(), key=lambda p: order.get(p.file, 1 << 30))
        return self._photos

    def append(self, photo: PhotoResult) -> None:
        with open(self.lines_path, "a") as f:
            f.write(json.dumps(photo.to_line(), separators=(",", ":")) + "\n")
        if self._photos is not None:
            self._photos.append(photo)

    def survey(self):
        from workbench import surveys
        if not hasattr(self, "_survey"):
            self._survey = surveys.load(self.meta.get("survey", ""))
        return self._survey

    def review(self) -> Review:
        if self._review is None:
            self._review = Review(self)
        return self._review

    def job(self):
        from workbench import jobs
        return jobs.load(self.meta["job"]) if self.meta.get("job") else None

    def state(self) -> str:
        """queued / running / done / failed / cancelled / stopped (from the job)."""
        if self.meta.get("status") == "done":
            return "done"
        job = self.job()
        if job is None:
            return "queued"
        status = job.status()
        return {"succeeded": "done"}.get(status, status)

    def photos_done(self) -> int:
        """Photographs finished so far — the job records it after each one."""
        fresh = read_json(os.path.join(self.dir, "detect.json"), {}) or {}
        return int(fresh.get("photos_done", self.meta.get("photos_done", 0)))

    def review_progress(self) -> tuple[int, int]:
        """(cells reviewed, cells in the survey) without reading the detections."""
        survey = self.survey()
        total = sum(n_cells(p.width, p.height) for p in survey.photos) if survey else 0
        data = read_json(os.path.join(self.dir, "review.json"), {}) or {}
        done = sum(len(v.get("cells_done", [])) for v in data.get("photos", {}).values())
        return done, total


def new(survey, model, run, threshold: float, threshold_source: str,
        checkpoint=None) -> Result:
    """Create the folder and detect.json for a result the detect job will fill in.

    `checkpoint` is a catalog.Checkpoint (None: the run's best weights); the scores kept
    with the result are those of that checkpoint, when it has been tested.
    """
    result_id = f"{timestamp()}_{slug(survey.name)}_{model.key}"
    kind = checkpoint.kind if checkpoint else "best"
    metrics = run.last_metrics if kind == "last" else run.metrics
    meta = {
        "id": result_id, "created": time.time(), "status": "queued",
        "survey": survey.name, "photos_total": len(survey.photos),
        "model": model.key, "model_label": model.label, "kind": model.kind,
        "run": run.run_id, "checkpoint": checkpoint.path if checkpoint else run.checkpoint,
        "checkpoint_kind": kind,
        "score_floor": config.CONF_THRESHOLD,
        "threshold": round(float(threshold), 4), "threshold_source": threshold_source,
        "tile": {"size": config.CROP_SIZE, "stride": stride()},
        "run_metrics": None if metrics is None else {
            "ap": metrics.ap, "f1": metrics.f1, "precision": metrics.precision,
            "recall": metrics.recall, "mae": metrics.mae, "rmse": metrics.rmse,
            "t_star": metrics.t_star, "n_images": metrics.n_images,
            "count_error_pct": metrics.count_error_pct},
    }
    result = Result(result_id, meta)
    os.makedirs(result.dir, exist_ok=True)
    result.save_meta()
    return result


def load(result_id: str) -> Result | None:
    meta = read_json(os.path.join(paths.DETECTIONS_DIR, result_id, "detect.json"))
    return Result(result_id, meta) if meta else None


def all_results() -> list[Result]:
    if not os.path.isdir(paths.DETECTIONS_DIR):
        return []
    found = [load(e.name) for e in os.scandir(paths.DETECTIONS_DIR) if e.is_dir()]
    return sorted((r for r in found if r), key=lambda r: r.meta.get("created", 0), reverse=True)
