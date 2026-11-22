"""
Shared paths, constants and model functions for the flight-altitude study.

Pipeline (run from the repository root with the `waterfowl` env):

    coco_boxes.py                 the dataset's COCO crop boxes, rebuilt on the source photographs
    step1_sizes.py                those box sizes per photo, camera EXIF/XMP, species boxes
    step2_outcomes.py             per-bird detection outcome of every benchmark detector on the test split
    step3_calibrate.py            size model, detector size response, species constants, validation
    step3b_optimum.py             optimum pixel size (first peak of the detection rate), Task 2.3 comparison
    step4_bootstrap.py            image-cluster bootstrap of the size thresholds  (slow, ~25 min)
    step4b_optimum_bootstrap.py   image-cluster bootstrap of the optimum pixel size  (~12 min)
    step5_figures.py              report figures
    step6_equations.py            LaTeX-typeset equation images
    build_report.py               the Word report
    flight_altitude.py            the operator calculator

Large intermediates go to $FA_WORK (default: <tmp>/flight_altitude_work), never into the
repository. Small result tables go to ../results/.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
RESULTS = os.path.join(REPORT_DIR, "results")
REPO = os.path.abspath(os.path.join(REPORT_DIR, "..", "..", "..", ".."))
DATASET = os.path.join(REPO, "dataset_full")
WORK = os.environ.get("FA_WORK", os.path.join(tempfile.gettempdir(), "flight_altitude_work"))
os.makedirs(WORK, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Cameras verified from EXIF in this corpus: focal length (mm), sensor width (mm), image width (px)
CAMERAS = {
    "L1D-20c": dict(label="DJI Mavic 2 Pro (Hasselblad L1D-20c)", focal_mm=10.26, sensor_w_mm=13.2, width_px=5472),
    "FC3170": dict(label="DJI Mavic Air 2 (FC3170, 12 MP)", focal_mm=4.5, sensor_w_mm=6.4, width_px=4000),
}
F_PX = {k: v["focal_mm"] * v["width_px"] / v["sensor_w_mm"] for k, v in CAMERAS.items()}

# Blur/annotation-margin term of the size model  s = sqrt((S f_px / H)^2 + B^2), fitted in step3
B_PX = 7.29

# Optimum bird size in the photo for detection: the apparent size, sqrt(box area) in px (the unit of the earlier
# Task 2.3 analysis), at the first peak of YOLOv5's dataset-standardised detection rate (18.1 px; step3b, step4b).
P_OPTIMUM = 18.0
# The first peak is searched below 30 px: every detector's rate rises to a peak at 17-23 px, dips at 30-40 px
# and rises again only for very large birds (photographed from under half the optimum height).
PEAK_RANGE = (10.0, 30.0)


def blur_factor(p_box: float = None, b: float = None) -> float:
    """k = P_box / sqrt(P_box^2 - b^2): how much larger a bird's box is than the bird itself at size P_box."""
    p_box = P_OPTIMUM if p_box is None else p_box
    b = B_PX if b is None else b
    return p_box / np.sqrt(p_box ** 2 - b ** 2)


def peak(curve, lo: float = None, hi: float = None, tol_points: float = 1.0):
    """Box size (px) where the detection-rate curve is highest inside PEAK_RANGE, its rate, and the size range
    within tol_points percentage points of that rate."""
    lo, hi = PEAK_RANGE if lo is None else (lo, hi)
    grid = np.arange(lo, hi + 1e-9, 0.1)
    r = curve(grid) * 100
    i = int(r.argmax())
    flat = grid[r >= r[i] - tol_points]
    return round(float(grid[i]), 1), float(r[i]), round(float(flat.min()), 1), round(float(flat.max()), 1)

PLATEAU_RANGE = (18.0, 40.0)   # px, where the plateau recall of the size-response curve is read
SIZE_CV = 0.12                 # within-species size spread used for n* (real birds 8-13%, decoys 14-17%)


def find_image(ds: str, stem: str) -> str | None:
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        p = os.path.join(DATASET, ds, stem + ext)
        if os.path.exists(p):
            return p
    return None


def norm_label(label: str) -> str:
    l = label.strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", l)


def class_file(ds: str, row: dict) -> str | None:
    """The *_class.txt species file for one image_info.csv row (Bird_H, Bird_I)."""
    p = os.path.join(DATASET, ds, row.get("classification_name") or "")
    if os.path.isfile(p):
        return p
    p = os.path.join(DATASET, ds, os.path.splitext(row["image_name"])[0] + "_class.txt")
    return p if os.path.isfile(p) else None


def read_class_boxes(path: str) -> list[tuple[str, int, int, int, int]]:
    out = []
    for line in open(path):
        p = [x.strip() for x in line.strip().split(",")]
        if len(p) < 5:
            continue
        try:
            x1, y1, x2, y2 = (int(float(v)) for v in p[-4:])
        except ValueError:
            continue
        if x2 > x1 and y2 > y1:
            out.append((norm_label(",".join(p[:-4])), x1, y1, x2, y2))
    return out


def image_info(ds: str) -> list[dict]:
    return list(csv.DictReader(open(os.path.join(DATASET, ds, "image_info.csv"))))


def dji_xmp(path: str, key: str) -> str | None:
    raw = open(path, "rb").read(300000)
    m = re.search(rb'drone-dji:' + key.encode() + rb'="([^"]+)"', raw)
    return m.group(1).decode() if m else None


# ── model functions ──────────────────────────────────────────────────────────

def apparent_size(S_m, f_px, H, b=B_PX):
    """Annotated sqrt(box area) in px of a bird of size S_m (m) seen from H (m) with focal length f_px."""
    return np.sqrt((np.asarray(S_m) * f_px / np.asarray(H)) ** 2 + b ** 2)


def fit_curve(d, col: str):
    """Detection probability vs apparent size, with dataset fixed effects.

    logit P(hit) = beta_dataset + spline(log size); returned as a function of size giving the
    marginal recall over the dataset mix of `d` (direct standardisation)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer
    import pandas as pd

    spl = SplineTransformer(n_knots=7, degree=3, knots="quantile", extrapolation="linear")
    Z = spl.fit_transform(np.log(d[["size"]].values))
    ds = pd.get_dummies(d.dataset).astype(float)
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(np.hstack([ds.values, Z]), d[col].values)
    beta_d = lr.coef_[0][: ds.shape[1]] + lr.intercept_[0]
    gam = lr.coef_[0][ds.shape[1]:]
    w = ds.mean().values
    grid = np.exp(np.linspace(np.log(4), np.log(200), 400))
    g = spl.transform(np.log(grid[:, None])) @ gam
    R = (w[None, :] / (1 + np.exp(-(beta_d[None, :] + g[:, None])))).sum(1)
    return lambda s: np.interp(np.log(np.clip(s, 4, 200)), np.log(grid), R)


def plateau(curve) -> float:
    return float(curve(np.linspace(*PLATEAU_RANGE, 50)).max())


def n_star(curve, loss: float, b=B_PX, cv=SIZE_CV, seed=0) -> float:
    """Geometric median size (px) at which expected recall over a species' size spread
    first reaches (1 - loss) x plateau. NaN when it is not reached below 25 px."""
    z = np.random.default_rng(seed).normal(0, 1, 4000)
    target = (1 - loss) * plateau(curve)
    for n in np.arange(8, 25, 0.05):
        if np.mean(curve(np.sqrt((n * np.exp(cv * z)) ** 2 + b ** 2))) >= target:
            return round(float(n), 2)
    return float("nan")
