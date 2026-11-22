#!/usr/bin/env python
"""
Count birds in survey photographs with a trained checkpoint.

    python -m workbench.detect <result-id>

Started as a background job by the Detect screen, which first writes the result's
detect.json (survey, model, run, checkpoint, threshold).  Safe to re-run: photographs
already in detections.jsonl are skipped, so a cancelled or crashed job picks up where it
stopped.

Seeing a photograph the way the model saw its training data
------------------------------------------------------------
Each photograph is cut into the same 512 x 512 tiles, at the same stride (460 px), with the
same edge rule as the training and test crops (`data_prep.prepare_data._tile_origins`).
Each tile is scored with the settings that model's own --eval uses — score floor 0.05, the
family's NMS IoU, up to 500 detections per tile, one tile per Ultralytics predict() call
(yolov5/evaluate.py explains why batching is unsafe there).  OWL variants run in their own
Python 3.11 interpreter through workbench/_owl_detect.py, with the stitcher and LMDS
settings of megadetector_overhead/_eval_owl.py.

Counting each bird once
-----------------------
Neighbouring tiles overlap, so a bird in an overlap is seen twice.  Each tile keeps only
the detections whose centre falls in the part of the photograph it *owns* — its cell,
bounded by the middle of each overlap (workbench/detections.py) — widened by SLACK px.
The cell boundary sits at least 26 px inside every tile, so a bird owned by a tile is seen
whole by it; a bird cut in half at a tile's edge lies in the neighbour's cell and is
dropped from this tile.  The slack guards against the one remaining failure: two tiles
placing the same bird's centre a pixel either side of the boundary, which without slack
could drop it from both.  Pairs that the slack lets through twice — same bird, two tiles,
near a boundary — are merged: boxes overlapping by IoU >= 0.3 (or one mostly inside the
other), or points within POINT_MERGE_PX, keep the detection from the tile that saw it
furthest from its edge.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

from data_prep import config
from workbench import catalog, detections, surveys
from workbench.detections import PhotoResult

SLACK = 8.0            # px a tile may keep beyond its cell boundary
BOX_MERGE_IOU = 0.3
BOX_MERGE_IOMIN = 0.6  # intersection over the smaller box: a truncated duplicate
POINT_MERGE_PX = 6.0


# ── Merging tiles into one photograph ─────────────────────────────────────────

def merge(tile_dets: list[tuple[int, int, np.ndarray]], width: int, height: int
          ) -> tuple[np.ndarray, int]:
    """Photograph-level detections from per-tile ones, each bird once.

    tile_dets: (tile_x, tile_y, [N, 5] array of x, y, w, h, score in tile pixels; w/h NaN
    for points).  Returns ([M, 5] array in photograph pixels, raw detection count).
    """
    xcuts, ycuts = detections.cuts(width), detections.cuts(height)
    xs, ys = detections.origins(width), detections.origins(height)
    kept, tile_id, depth = [], [], []
    raw = 0
    for tx, ty, dets in tile_dets:
        raw += len(dets)
        if not len(dets):
            continue
        c, r = xs.index(tx), ys.index(ty)
        lo_x = xcuts[c - 1] - SLACK if c > 0 else -np.inf
        hi_x = xcuts[c] + SLACK if c < len(xcuts) else np.inf
        lo_y = ycuts[r - 1] - SLACK if r > 0 else -np.inf
        hi_y = ycuts[r] + SLACK if r < len(ycuts) else np.inf
        px, py = dets[:, 0] + tx, dets[:, 1] + ty
        inside = (px >= lo_x) & (px < hi_x) & (py >= lo_y) & (py < hi_y)
        if not inside.any():
            continue
        d = dets[inside].copy()
        d[:, 0] += tx
        d[:, 1] += ty
        # distance from the centre to the nearest tile edge that has a neighbour behind it
        edges = []
        if c > 0:
            edges.append(dets[inside, 0])
        if c < len(xcuts):
            edges.append(config.CROP_SIZE - dets[inside, 0])
        if r > 0:
            edges.append(dets[inside, 1])
        if r < len(ycuts):
            edges.append(config.CROP_SIZE - dets[inside, 1])
        dep = np.min(np.stack(edges), axis=0) if edges else np.full(inside.sum(), np.inf)
        kept.append(d)
        tile_id.append(np.full(len(d), r * len(xs) + c))
        depth.append(dep)
    if not kept:
        return np.zeros((0, 5)), raw
    dets = np.concatenate(kept)
    tiles = np.concatenate(tile_id)
    dep = np.concatenate(depth)
    keep = _dedupe(dets, tiles, dep, xcuts, ycuts)
    return dets[keep], raw


def _near_cut(v: np.ndarray, cut_list: list[float], margin: float) -> np.ndarray:
    if not cut_list:
        return np.zeros(len(v), dtype=bool)
    dist = np.min(np.abs(v[:, None] - np.asarray(cut_list)[None, :]), axis=1)
    return dist <= margin


def _duplicate(a: np.ndarray, b: np.ndarray) -> bool:
    if np.isnan(a[2]) or np.isnan(b[2]):
        return float(np.hypot(a[0] - b[0], a[1] - b[1])) <= POINT_MERGE_PX
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return False
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    iou = inter / (area_a + area_b - inter)
    return iou >= BOX_MERGE_IOU or inter / max(min(area_a, area_b), 1e-9) >= BOX_MERGE_IOMIN


def _dedupe(dets: np.ndarray, tiles: np.ndarray, depth: np.ndarray,
            xcuts: list[float], ycuts: list[float]) -> np.ndarray:
    keep = np.ones(len(dets), dtype=bool)
    margin = SLACK + 32.0
    cand = np.where(_near_cut(dets[:, 0], xcuts, margin) | _near_cut(dets[:, 1], ycuts, margin))[0]
    if len(cand) < 2:
        return keep
    # Strongest evidence first: deepest inside its tile, then highest score.
    order = cand[np.lexsort((-dets[cand, 4], -np.round(depth[cand], 0)))]
    bucket = 32.0
    grid: dict[tuple[int, int], list[int]] = {}
    for i in order:
        gx, gy = int(dets[i, 0] // bucket), int(dets[i, 1] // bucket)
        clash = False
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for j in grid.get((nx, ny), ()):
                    if tiles[j] != tiles[i] and _duplicate(dets[i], dets[j]):
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            keep[i] = False
        else:
            grid.setdefault((gx, gy), []).append(i)
    return keep


# ── Per-family tile inference (the `waterfowl` environment) ───────────────────

def _cut_tiles(img: Image.Image) -> list[tuple[int, int, np.ndarray]]:
    """Tiles exactly as data_prep.prepare_data.crop_image cuts them (PIL crop, black
    padding past the edge), as RGB arrays."""
    size = config.CROP_SIZE
    out = []
    for ty in detections.origins(img.height):
        for tx in detections.origins(img.width):
            out.append((tx, ty, np.asarray(img.crop((tx, ty, tx + size, ty + size)))))
    return out


def _rows(xyxy: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if len(xyxy) == 0:
        return np.zeros((0, 5))
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, scores], axis=1)


class UltralyticsRunner:
    def __init__(self, model: catalog.Model, ckpt: str):
        from ultralytics import YOLO
        spec = config.ULTRALYTICS_MODELS.get(model.key, {})
        self.iou = spec.get("nms_iou", config.YOLOV5_NMS_IOU)
        self.max_det = spec.get("max_pred", config.YOLOV5_MAX_PRED)
        self.model = YOLO(ckpt)

    def __call__(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for rgb in tiles:           # one tile per call — see yolov5/evaluate.py
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            pred = self.model.predict(source=bgr, conf=config.CONF_THRESHOLD, iou=self.iou,
                                      imgsz=config.CROP_SIZE, max_det=self.max_det,
                                      verbose=False)[0]
            boxes = pred.boxes
            if boxes is None or len(boxes) == 0:
                out.append(np.zeros((0, 5)))
                continue
            out.append(_rows(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()))
        return out


class YoloNasRunner:
    def __init__(self, model: catalog.Model, ckpt: str):
        import torch
        from super_gradients.training import models
        net = models.get(config.YOLONAS_MODEL, num_classes=1, checkpoint_path=ckpt)
        self.model = (net.cuda() if torch.cuda.is_available() else net).eval()

    def __call__(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for i in range(0, len(tiles), 32):
            preds = self.model.predict(list(tiles[i:i + 32]), conf=config.CONF_THRESHOLD,
                                       iou=config.YOLONAS_NMS_IOU,
                                       max_predictions=config.YOLONAS_MAX_PRED,
                                       fuse_model=False)
            preds = preds if hasattr(preds, "__iter__") else [preds]
            for pred in preds:
                out.append(_rows(np.asarray(pred.prediction.bboxes_xyxy).reshape(-1, 4),
                                 np.asarray(pred.prediction.confidence).reshape(-1)))
        return out


class FasterRcnnRunner:
    def __init__(self, model: catalog.Model, ckpt: str):
        import torch
        from detectron2.engine import DefaultPredictor
        from faster_rcnn.model import build_cfg     # the cfg training and --eval use
        cfg = build_cfg(use_pretrained=False)
        cfg.MODEL.WEIGHTS = ckpt
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.CONF_THRESHOLD
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for rgb in tiles:           # detectron2 expects BGR, as cv2.imread gives
            inst = self.predictor(np.ascontiguousarray(rgb[:, :, ::-1]))["instances"].to("cpu")
            out.append(_rows(inst.pred_boxes.tensor.numpy(), inst.scores.numpy()))
        return out


def make_runner(model: catalog.Model, ckpt: str):
    if model.key in ("yolov5", "yolo11", "yolo26"):
        return UltralyticsRunner(model, ckpt)
    if model.key == "yolonas":
        return YoloNasRunner(model, ckpt)
    if model.key == "fasterrcnn":
        return FasterRcnnRunner(model, ckpt)
    raise ValueError(f"{model.key} has no in-process runner")


# ── The job ───────────────────────────────────────────────────────────────────

def _progress(result, k: int, total: int, photo: PhotoResult, t0: float) -> None:
    n_above = int((photo.score >= result.threshold).sum())
    elapsed = time.time() - t0
    eta = elapsed / (k + 1) * (total - k - 1)
    print(f"[{k + 1}/{total}] {os.path.basename(photo.file)}: {n_above} birds at "
          f"threshold {result.threshold:.2f} ({photo.n} detections >= "
          f"{config.CONF_THRESHOLD}, {photo.n_raw} before merging overlaps) · "
          f"{photo.seconds:.1f}s · about {eta / 60:.0f} min left", flush=True)
    result.meta["photos_done"] = len(result.photos())
    result.save_meta()


def _run_box(result, model, survey, todo) -> None:
    print(f"Loading {model.label} from {os.path.relpath(result.meta['checkpoint'], ROOT)}",
          flush=True)
    runner = make_runner(model, result.meta["checkpoint"])
    t0 = time.time()
    for k, photo in enumerate(todo):
        start = time.time()
        with Image.open(survey.path(photo)) as im:
            img = im.convert("RGB")
        tiles = _cut_tiles(img)
        per_tile = runner([t for _, _, t in tiles])
        merged, raw = merge([(tx, ty, d) for (tx, ty, _), d in zip(tiles, per_tile)],
                            img.width, img.height)
        pr = PhotoResult(file=photo.file, width=img.width, height=img.height,
                         xy=merged[:, 0:2], wh=merged[:, 2:4], score=merged[:, 4],
                         n_raw=raw, seconds=time.time() - start)
        result.append(pr)
        _progress(result, k, len(todo), pr, t0)


def _run_owl(result, model, survey, todo) -> None:
    """One interpreter hop for the whole survey: the OWL helper loads the model once and
    writes one line of per-tile points per photograph, which are merged here."""
    if not os.path.exists(config.MDO_PYTHON):
        raise SystemExit(f"OWL environment not found at {config.MDO_PYTHON} "
                         f"(cd {config.MDO_REPO_DIR} && uv sync --no-default-groups --group gpu)")
    spec = config.MDO_MODELS[model.mdo_key]
    work_path = os.path.join(result.dir, "owl_work.json")
    tiles_path = os.path.join(result.dir, "owl_tiles.jsonl")
    work = [{"photo": os.path.realpath(survey.path(p)), "key": p.file,
             "xs": detections.origins(p.width), "ys": detections.origins(p.height)}
            for p in todo]
    with open(work_path, "w") as f:
        json.dump(work, f)

    env = os.environ.copy()
    env["MDO_REPO_DIR"] = config.MDO_REPO_DIR
    env["DINOV3_ROOT"] = config.MDO_DINOV3_ROOT
    env["PYTHONWARNINGS"] = ",".join(filter(None, [
        env.get("PYTHONWARNINGS"),
        "ignore::UserWarning:albumentations.core.composition"]))
    cmd = [config.MDO_PYTHON, os.path.join(ROOT, "workbench", "_owl_detect.py"),
           "--pth", result.meta["checkpoint"], "--work", work_path, "--out", tiles_path,
           "--model-name", spec["name"], "--model-kwargs", json.dumps(spec["eval_kwargs"]),
           "--down-ratio", str(config.MDO_DOWN_RATIO), "--img-size", str(config.CROP_SIZE),
           "--overlap", str(config.MDO_STITCH_OVERLAP),
           "--adapt-ts", str(config.MDO_LMDS_ADAPT_TS), "--kernel", str(config.MDO_LMDS_KERNEL)]
    print(f"Running {model.label} in its own environment ({config.MDO_PYTHON})", flush=True)
    t0 = time.time()
    done = {p.file for p in result.photos()}
    by_key = {p.file: p for p in todo}
    proc = subprocess.Popen(cmd, cwd=config.MDO_REPO_DIR, env=env)
    # Merge each photograph as soon as the helper finishes it, so progress (and a partial
    # result) is visible while the helper is still working.
    offset, finished = 0, 0

    def drain() -> None:
        nonlocal offset, finished
        if not os.path.exists(tiles_path):
            return
        with open(tiles_path) as f:
            f.seek(offset)
            while True:
                line = f.readline()
                if not line.endswith("\n"):
                    break
                offset = f.tell()
                payload = json.loads(line)
                photo = by_key.get(payload["key"])
                if photo is None or photo.file in done:
                    continue
                tile_dets = []
                for tx, ty, pts in payload["tiles"]:
                    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                    rows = np.full((len(arr), 5), np.nan)
                    rows[:, 0:2], rows[:, 4] = arr[:, 0:2], arr[:, 2]
                    tile_dets.append((tx, ty, rows))
                merged, raw = merge(tile_dets, photo.width, photo.height)
                pr = PhotoResult(file=photo.file, width=photo.width, height=photo.height,
                                 xy=merged[:, 0:2], wh=merged[:, 2:4], score=merged[:, 4],
                                 n_raw=raw, seconds=float(payload.get("seconds", 0.0)))
                result.append(pr)
                done.add(photo.file)
                _progress(result, finished, len(todo), pr, t0)
                finished += 1

    while proc.poll() is None:
        drain()
        time.sleep(2.0)
    drain()
    if proc.returncode != 0:
        raise SystemExit(f"OWL helper failed with exit code {proc.returncode}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    result = detections.load(argv[0])
    if result is None:
        print(f"no detection result {argv[0]!r}")
        return 2
    survey = surveys.load(result.meta["survey"])
    if survey is None:
        print(f"survey {result.meta['survey']!r} no longer exists")
        return 2
    model = catalog.model(result.meta["model"])
    done = {p.file for p in result.photos()}
    todo = [p for p in survey.photos if p.file not in done]
    print(f"{model.label} run {result.meta['run']} on survey {survey.name}: "
          f"{len(todo)} of {len(survey.photos)} photographs to do "
          f"(threshold {result.threshold:.3f}, {result.meta['threshold_source']})", flush=True)

    result.meta.update(status="running", started=result.meta.get("started") or time.time())
    result.save_meta()
    if todo:
        (_run_owl if model.owl else _run_box)(result, model, survey, todo)

    photos = result.photos(reload=True)
    total = sum(int((p.score >= result.threshold).sum()) for p in photos)
    result.meta.update(status="done", finished=time.time(), photos_done=len(photos),
                       birds_at_threshold=total)
    result.save_meta()
    print(f"\nDone: {total:,} birds counted at threshold {result.threshold:.3f} in "
          f"{len(photos)} photographs.  Next: Review & correct detections.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
