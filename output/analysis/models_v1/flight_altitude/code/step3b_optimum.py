"""
Step 3b — the optimum pixel size for detection, in the units of the earlier Task 2.3 analysis.

P_optimum is the apparent bird size, sqrt(box area) in px, at which the dataset-standardised detection rate is
highest over the sizes survey flights produce (PEAK_RANGE). The flight height that puts a species' median bird
at that box size is H_rec = k * S * C / P_optimum with k = P_optimum / sqrt(P_optimum^2 - b^2).

Writes to results/:
  optimum_point.csv          per detector: peak size, rate at the peak, sizes within 1 point, Task 2.3 best bins
  optimum_task23_bins.csv    recall in the Task 2.3 bins, all boxes (as published) and uncut boxes
  optimum_tradeoff.csv       YOLOv5: detection rate, flight height and ground area per photo against box size
  optimum_species_check.csv  mean detection over each species' own sizes at H_rec vs the best height
"""
import os
import warnings

import numpy as np
import pandas as pd

from common import B_PX, F_PX, P_OPTIMUM, PEAK_RANGE, REPO, RESULTS, WORK, blur_factor, fit_curve, peak
from step3_calibrate import read_boxes, species_constants

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
MODELS = {"Faster R-CNN": "fasterrcnn", "OWL-C": "mdo_owl_c", "OWL-D": "mdo_owl_d", "OWL-T": "mdo_owl_t",
          "YOLO11": "yolo11", "YOLO26": "yolo26", "YOLO-NAS": "yolonas", "YOLOv5": "yolov5"}


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(RESULTS, name), index=False, float_format="%.4g")


def main() -> None:
    out = pd.read_pickle(os.path.join(WORK, "outcomes.pkl"))
    u = out[~out.clipped]
    t23 = pd.read_csv(os.path.join(REPO, "output", "analysis", "models_v1", "by_size.csv"))

    curves, rows = {}, []
    for label, key in MODELS.items():
        c = curves[label] = fit_curve(u, "hit_" + key)
        p, rate, lo, hi = peak(c)
        g = t23[t23.model == label]
        gf = g.dropna(subset=["f1"])
        rows.append(dict(detector=label, peak_px=p, peak_rate_pct=rate, within_1pt_lo_px=lo, within_1pt_hi_px=hi,
                         task23_best_recall_bin=g.loc[g.recall.idxmax(), "size_bin_px"],
                         task23_best_f1_bin=gf.loc[gf.f1.idxmax(), "size_bin_px"] if len(gf) else ""))
    t = pd.DataFrame(rows)
    save(t, "optimum_point.csv")
    print(t.round(1).to_string(index=False))

    edges = sorted(set(t23.lo_px) | set(t23.hi_px))
    bins = pd.cut(u["size"], edges, right=False)
    uncut = u.groupby(bins, observed=True)[["hit_" + k for k in MODELS.values()]].mean().mul(100)
    uncut.columns = list(MODELS)
    uncut = uncut.reset_index().melt(id_vars="size", var_name="detector", value_name="recall_uncut")
    uncut["bin"] = uncut["size"].map(lambda iv: f"{iv.left:.2f}-{iv.right:.2f}")
    pub = t23[["model", "lo_px", "hi_px", "recall"]].rename(columns={"model": "detector", "recall": "recall_task23"})
    pub["bin"] = pub.apply(lambda r: f"{r.lo_px:.2f}-{r.hi_px:.2f}", axis=1)
    save(pub.merge(uncut[["detector", "bin", "recall_uncut"]], on=["detector", "bin"]), "optimum_task23_bins.csv")

    c = curves["YOLOv5"]
    geo_opt = np.sqrt(P_OPTIMUM ** 2 - B_PX ** 2)
    trade = []
    for p in (14, 16, 18, 20, 25, 30, 40, 60, 100):
        h = geo_opt / np.sqrt(p ** 2 - B_PX ** 2)                   # flight height relative to the optimum
        trade.append(dict(box_px=p, detection_pct=c(p) * 100, height_x=h, area_x=h * h))
    trade = pd.DataFrame(trade)
    save(trade, "optimum_tradeoff.csv")
    print(trade.round(2).to_string(index=False))

    sp = read_boxes("species.csv")
    sp["s"] = np.sqrt(sp.w * sp.h)
    real, _ = species_constants(sp)
    k = blur_factor()
    checks = []
    for name, g in real.groupby("sp"):
        if len(g) < 30 or name in ("unknown", "not a bird"):
            continue
        S = g.S_cm.values / 100
        for cam, C in F_PX.items():
            h_rec = k * np.median(S) * C / P_OPTIMUM
            Hs = np.linspace(0.3, 2.5, 441) * h_rec
            med_box = np.sqrt((np.median(S) * C / Hs) ** 2 + B_PX ** 2)
            Hs = Hs[(med_box >= PEAK_RANGE[0]) & (med_box <= PEAK_RANGE[1])]
            rbar = np.array([np.mean(c(np.sqrt((S * C / H) ** 2 + B_PX ** 2))) for H in Hs]) * 100
            at_rec = np.mean(c(np.sqrt((S * C / h_rec) ** 2 + B_PX ** 2))) * 100
            ok = Hs[rbar >= rbar.max() - 1.0]
            checks.append(dict(species=name, camera=cam, n=len(g), H_rec_m=h_rec, mean_detection_at_H_rec=at_rec,
                               best_mean_detection=rbar.max(), best_height_over_H_rec=Hs[rbar.argmax()] / h_rec,
                               within_1pt_height_lo=ok.min() / h_rec, within_1pt_height_hi=ok.max() / h_rec))
    ch = pd.DataFrame(checks)
    save(ch, "optimum_species_check.csv")
    print(ch.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
