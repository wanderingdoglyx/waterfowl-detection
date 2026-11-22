"""
Step 4 — uncertainty of the pixel threshold n*: cluster bootstrap over source photographs.

Birds in one photograph share a flock, light, habitat and annotator, so photographs (not birds)
are resampled. Each replicate refits the dataset-standardised size-response curve and re-solves
n* for 2 / 5 / 10 % tolerated recall loss. 100 replicates, ~25 min on one CPU.

Output: results/nstar_bootstrap.csv
"""
import os
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, SIZE_CV, WORK, fit_curve, n_star

warnings.filterwarnings("ignore")
MODELS = ("yolov5", "yolo11", "fasterrcnn", "mdo_owl_t")
LOSSES = (0.02, 0.05, 0.10)
REPS = 100


def main() -> None:
    out = pd.read_pickle(os.path.join(WORK, "outcomes.pkl"))
    u = out[~out.clipped].copy()
    u["src"] = u.dataset + "/" + u.stem
    rng = np.random.default_rng(0)
    rng.normal(0, SIZE_CV, 4000)          # keeps the resampling stream identical to the original study run
    srcs = u.src.unique()
    idx = u.groupby("src").indices
    res = {m: [] for m in MODELS}
    for b in range(REPS + 1):
        d = u if b == 0 else u.iloc[np.concatenate([idx[s] for s in rng.choice(srcs, size=len(srcs), replace=True)])]
        for m in MODELS:
            c = fit_curve(d, "hit_" + m)
            res[m].append([n_star(c, l) for l in LOSSES])
    rows = []
    for m, v in res.items():
        v = np.array(v, float)
        for k, l in enumerate(LOSSES):
            bs = v[1:, k]
            rows.append(dict(model=m, loss_pct=int(l * 100), n_star_point=v[0, k],
                             ci_low=np.nanpercentile(bs, 2.5), ci_high=np.nanpercentile(bs, 97.5),
                             reps_not_reached=int(np.isnan(bs).sum()), reps=REPS))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, "nstar_bootstrap.csv"), index=False, float_format="%.4g")
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
