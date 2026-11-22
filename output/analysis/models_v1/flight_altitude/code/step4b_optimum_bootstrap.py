"""
Step 4b — uncertainty of the optimum pixel size: cluster bootstrap over source photographs.

Each replicate refits the dataset-standardised detection-rate curve and records its first peak (common.peak over
PEAK_RANGE) and, for transparency, the highest point over 10-40 px. Uses the same resampling stream as step 4.

Output: results/optimum_bootstrap.csv (summary), results/optimum_bootstrap_reps.csv (every replicate)
"""
import os
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, SIZE_CV, WORK, fit_curve, peak

warnings.filterwarnings("ignore")
MODELS = ("yolov5", "yolo11", "mdo_owl_t")
REPS = 100


def main() -> None:
    out = pd.read_pickle(os.path.join(WORK, "outcomes.pkl"))
    u = out[~out.clipped].copy()
    u["src"] = u.dataset + "/" + u.stem
    rng = np.random.default_rng(0)
    rng.normal(0, SIZE_CV, 4000)          # same resampling stream as step 4
    srcs = u.src.unique()
    idx = u.groupby("src").indices
    rows = []
    for b in range(REPS + 1):
        d = u if b == 0 else u.iloc[np.concatenate([idx[s] for s in rng.choice(srcs, size=len(srcs), replace=True)])]
        for m in MODELS:
            c = fit_curve(d, "hit_" + m)
            first = peak(c)
            wide = peak(c, 10.0, 40.0)
            rows.append(dict(rep=b, model=m, first_peak_px=first[0], first_peak_rate=first[1],
                             peak_10_40_px=wide[0], rate_at_40px=float(c(40.0) * 100)))
    reps = pd.DataFrame(rows)
    reps.to_csv(os.path.join(RESULTS, "optimum_bootstrap_reps.csv"), index=False, float_format="%.4g")
    summ = []
    for m, g in reps.groupby("model", sort=False):
        full, bs = g[g.rep == 0].iloc[0], g[g.rep > 0]
        summ.append(dict(model=m, first_peak_px=full.first_peak_px,
                         ci_low=np.percentile(bs.first_peak_px, 2.5), ci_high=np.percentile(bs.first_peak_px, 97.5),
                         reps_highest_at_40px=int((bs.peak_10_40_px >= 39.95).sum()), reps=REPS))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(RESULTS, "optimum_bootstrap.csv"), index=False, float_format="%.4g")
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
