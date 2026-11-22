"""
Step 3 — calibrate the size model and the constants behind the flight-height equation.

Writes to results/:
  blur_margin_checks.csv         how well the blur margin b is determined (splits, per type, other ladders)
  size_by_dataset_altitude.csv   median sqrt(box area) per dataset x camera x resolution x altitude
  decoy_size_fit.csv             per decoy type: fitted a = S*f_px (px*m) and S (cm)
  decoy_size_cells.csv           decoy medians per type x altitude, with geometry-only and fitted predictions
  camera_ratio.csv               same-species size ratio between the two cameras (Bird_I, 15 m)
  recall_by_size_raw.csv         recall per size bin, all boxes vs unclipped boxes, 8 models
  recall_curve_standardised.csv  dataset-standardised recall vs size, 8 models
  precision_by_size.csv          precision per predicted-box size bin, raw and standardised
  nstar_point.csv                n* per model for 2 / 5 / 10 % tolerated recall loss
  species_size.csv               species size constants S (real birds and decoys)
  hstar_exact_vs_closed.csv      full expected-recall solution vs the closed form
  validation_decoys.csv          out-of-sample recall and size prediction for Bird_H test decoys
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import (B_PX, DATASET, F_PX, RESULTS, WORK, REPO, apparent_size, dji_xmp, fit_curve,
                    n_star, plateau)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
LOSSES = (0.02, 0.05, 0.10)
MODELS = ["fasterrcnn", "mdo_owl_c", "mdo_owl_d", "mdo_owl_t", "yolo11", "yolo26", "yolonas", "yolov5"]


def save(df: pd.DataFrame, name: str, index=False) -> None:
    df.to_csv(os.path.join(RESULTS, name), index=index, float_format="%.4g")


def read_boxes(name: str) -> pd.DataFrame:
    """One of step1's box tables, less the birds the photograph's own edge cut short.

    Those boxes stop at the image border, so their annotated size is not the bird's size and
    the size model would read them as small birds. step1 flags them; nothing here guesses."""
    d = pd.read_csv(os.path.join(WORK, name), keep_default_na=False)
    kept = d[~d.edge].reset_index(drop=True)
    print(f"{name}: {len(d):,} boxes, {len(d) - len(kept):,} dropped as cut by the image edge")
    return kept


def size_tables(birds: pd.DataFrame) -> None:
    birds["s"] = np.sqrt(birds.w * birds.h)
    birds["res"] = birds.W.astype(str) + "x" + birds.H.astype(str)
    g = birds.groupby(["ds", "camera", "res", "alt"]).agg(
        images=("image", "nunique"), birds=("s", "size"), median_px=("s", "median")).reset_index()
    g["k_px_m"] = g.median_px * g.alt
    save(g, "size_by_dataset_altitude.csv")

    # cross-camera: Bird_J 4000x3000 vs 5472x3648 at 15 m (no species control)
    j = birds[(birds.ds == "Bird_J") & (birds.alt == 15)].groupby("res").s.agg(["size", "median"])
    print("Bird_J 15 m by resolution:\n", j.round(2).to_string())


def decoy_model(sp: pd.DataFrame, birds: pd.DataFrame) -> None:
    h = sp[(sp.ds == "Bird_H") & (sp.sp != "unknown")]
    med = h.groupby(["sp", "alt"]).s.agg(n="size", s="median").reset_index()
    med = med[med.n >= 10].reset_index(drop=True)
    types = sorted(med.sp.unique())

    def resid(p, quad):
        a = dict(zip(types, p[:-1]))
        geo = np.array([a[t] for t in med.sp]) / med.alt.values
        pred = np.sqrt(geo ** 2 + p[-1] ** 2) if quad else geo + p[-1]
        return (pred - med.s.values) * np.sqrt(med.n.values)

    p0 = [med[med.sp == t].eval("s*alt").iloc[0] for t in types] + [2.0]
    fits = {q: least_squares(resid, p0, kwargs=dict(quad=q)) for q in (False, True)}
    for q, fit in fits.items():
        r = fit.fun / np.sqrt(med.n.values)
        print(f"decoy size model ({'sqrt(geo^2+b^2)' if q else 'geo + c'}): term = {fit.x[-1]:.2f} px, "
              f"RMS = {np.sqrt(np.mean(r ** 2)):.2f} px, max |res| = {np.abs(r).max():.2f} px")
    fq = fits[True]
    b = fq.x[-1]
    if abs(b - B_PX) > 0.05:
        print(f"  NOTE: fitted b = {b:.2f} differs from common.B_PX = {B_PX}")
    a = dict(zip(types, fq.x[:-1]))
    save(pd.DataFrame({"decoy_type": types, "a_px_m": [a[t] for t in types],
                       "S_cm_L1D": [a[t] / F_PX["L1D-20c"] * 100 for t in types]}), "decoy_size_fit.csv")
    blur_checks(sp, med, birds)
    med["k_ratio_vs_15m"] = med.s * med.alt / med.groupby("sp").apply(
        lambda g: float(g.loc[g.alt == 15, "s"].iloc[0] * 15)).reindex(med.sp).values
    med["pred_geometry_px"] = [a[t] / H for t, H in zip(med.sp, med.alt)]
    med["pred_fit_px"] = np.sqrt(med.pred_geometry_px ** 2 + b ** 2)
    save(med, "decoy_size_cells.csv")


def _fit_blur(med, key, quad=True, per_type=False):
    """Weighted least squares of the size model: one size constant per group + blur margin(s)."""
    types = sorted(med[key].unique())
    nb = len(types) if per_type else 1

    def resid(p):
        a = dict(zip(types, p[:len(types)]))
        geo = np.array([a[t] for t in med[key]]) / med.alt.values
        if per_type:
            bmap = dict(zip(types, p[len(types):]))
            b = np.array([bmap[t] for t in med[key]])
        else:
            b = p[-1]
        pred = np.sqrt(geo ** 2 + b ** 2) if quad else geo + b
        return (pred - med.s.values) * np.sqrt(med.n.values)

    p0 = [med[med[key] == t].eval("s*alt").iloc[0] for t in types] + [2.0] * nb
    f = least_squares(resid, p0)
    r = f.fun / np.sqrt(med.n.values)
    return f.x, float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max()), types


def _cells(d, key):
    g = d.groupby([key, "alt"]).s.agg(n="size", s="median").reset_index()
    return g[g.n >= 10].reset_index(drop=True)


def blur_checks(sp: pd.DataFrame, med: pd.DataFrame, birds: pd.DataFrame) -> None:
    """How well determined is the blur margin b? Refit it without the test split, per decoy type, with an
    additive margin instead, and on the other height ladders (Bird_A, Bird_C), where the same scene was
    photographed from every height. Writes results/blur_margin_checks.csv."""
    rows = []
    h = sp[(sp.ds == "Bird_H") & (sp.sp != "unknown")]
    for label, m, quad in (("Bird_H decoys, per type, all splits", med, True),
                           ("Bird_H decoys, per type, train+val only", _cells(h[h.split != "test"], "sp"), True),
                           ("Bird_H decoys, per type, additive margin", med, False)):
        x, rms, mx, _ = _fit_blur(m, "sp", quad=quad)
        rows.append(dict(fit=label, b_px=x[-1], rms_px=rms, max_px=mx, groups=len(m)))
    x, rms, mx, t = _fit_blur(med, "sp", per_type=True)
    for name, b in zip(t, x[len(t):]):
        rows.append(dict(fit=f"Bird_H decoy type: {name}", b_px=b, rms_px=rms, max_px=mx, groups=4))
    birds = birds.assign(s=np.sqrt(birds.w * birds.h))
    for ds in ("Bird_A", "Bird_C", "Bird_H"):
        d = birds[birds.ds == ds].copy()
        d["grp"] = ds
        m = _cells(d, "grp")
        x, rms, mx, _ = _fit_blur(m, "grp")
        rows.append(dict(fit=f"{ds} ladder, all birds pooled", b_px=x[-1], rms_px=rms, max_px=mx, groups=len(m)))
    t = pd.DataFrame(rows)
    save(t, "blur_margin_checks.csv")
    print(t.round(2).to_string(index=False))


def camera_ratio(sp: pd.DataFrame) -> None:
    i = sp[(sp.ds == "Bird_I") & (sp.cam != "")]
    t = i.pivot_table(index="sp", columns="cam", values="s", aggfunc=["median", "size"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.dropna()
    t = t[(t["size_FC3170"] >= 5) & (t["size_L1D-20c"] >= 5)]
    t["ratio_L1D_over_FC3170"] = t["median_L1D-20c"] / t["median_FC3170"]
    t["predicted_ratio"] = F_PX["L1D-20c"] / F_PX["FC3170"]
    save(t.reset_index(), "camera_ratio.csv")
    print(f"Bird_I same-species ratio L1D/FC3170: {t.ratio_L1D_over_FC3170.min():.2f}-{t.ratio_L1D_over_FC3170.max():.2f} "
          f"(predicted {F_PX['L1D-20c'] / F_PX['FC3170']:.3f})")


def detector_response(out: pd.DataFrame) -> dict:
    hits = ["hit_" + m for m in MODELS]
    edges = [0, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 30, 40, 60, 90, 400]
    frames = []
    for label, d in (("all", out), ("unclipped", out[~out.clipped])):
        d = d.assign(bin=pd.cut(d["size"], edges, right=False))
        t = d.groupby("bin", observed=True)[hits].mean().mul(100)
        t.insert(0, "n", d.groupby("bin", observed=True).size())
        t.insert(0, "boxes", label)
        frames.append(t.reset_index())
    save(pd.concat(frames), "recall_by_size_raw.csv")

    u = out[~out.clipped]
    curves = {m: fit_curve(u, "hit_" + m) for m in MODELS}
    grid = np.round(np.exp(np.linspace(np.log(6), np.log(150), 120)), 3)
    save(pd.DataFrame({"size_px": grid, **{m: curves[m](grid) * 100 for m in MODELS}}), "recall_curve_standardised.csv")

    rows = []
    for m in MODELS:
        c = curves[m]
        rows.append(dict(model=m, plateau_pct=plateau(c) * 100, R12=c(12) * 100, R14=c(14) * 100, R16=c(16) * 100,
                         R18=c(18) * 100, R80=c(80) * 100, R100=c(100) * 100,
                         **{f"nstar_{int(l * 100)}pct": n_star(c, l) for l in LOSSES}))
    t = pd.DataFrame(rows)
    save(t, "nstar_point.csv")
    print(t.round(2).to_string(index=False))
    return curves


def precision_by_size() -> None:
    from analysis import manifest, metadata, predict, registry, rescore, strata
    gt = strata.load_ground_truth()
    metas = metadata.load(predict.TEST_JSON)
    specs = {registry.resolve(e.key).key: (registry.resolve(e.key), e.run_id)
             for e in manifest.load(os.path.join(REPO, "models_v1.docx"))}
    edges = [0, 10, 12, 14, 16, 18, 20, 25, 30, 40, 60, 400]
    frames = []
    for key in ("yolov5", "yolo11", "fasterrcnn"):
        spec, run = specs[key]
        pm = rescore.ensure(spec, run)
        o = strata.outcomes_at_t_star(gt, predict.load(predict.cache_path(spec, run)), pm["counting"]["t_star"])
        d = pd.DataFrame(dict(size=o.det_size, tp=o.det_tp, ds=[metas[int(i)].dataset for i in o.det_image_id]))
        d["bin"] = pd.cut(d["size"], edges, right=False)
        share = d.ds.value_counts(normalize=True)
        per = d.groupby(["bin", "ds"], observed=True).tp.mean().unstack()
        std = (per * share).sum(1) / per.notna().mul(share).sum(1)
        frames.append(pd.DataFrame(dict(model=key, n=d.groupby("bin", observed=True).size(),
                                        precision_raw=d.groupby("bin", observed=True).tp.mean() * 100,
                                        precision_std=std * 100)).reset_index())
    save(pd.concat(frames), "precision_by_size.csv")


def species_constants(sp: pd.DataFrame):
    real = sp[(sp.ds == "Bird_I") & (sp.cam == "L1D-20c")].copy()
    xmp = {img: float(dji_xmp(os.path.join(DATASET, "Bird_I", img), "RelativeAltitude")) for img in real.image.unique()}
    real["H"] = real.image.map(xmp)
    real["S_cm"] = np.sqrt(np.maximum(real.s ** 2 - B_PX ** 2, 1)) * real.H / F_PX["L1D-20c"] * 100
    dec = sp[(sp.ds == "Bird_H") & (sp.alt == 15)].copy()
    dec["S_cm"] = np.sqrt(np.maximum(dec.s ** 2 - B_PX ** 2, 1)) * 15 / F_PX["L1D-20c"] * 100
    frames = []
    for src, d in (("real (Bird_I)", real), ("decoy (Bird_H)", dec)):
        g = d.groupby("sp").S_cm.agg(n="size", S_median_cm="median", S_p25_cm=lambda x: x.quantile(.25),
                                     S_p75_cm=lambda x: x.quantile(.75), cv=lambda x: x.std() / x.mean())
        frames.append(g.reset_index().assign(source=src))
    save(pd.concat(frames), "species_size.csv")
    print(f"real-bird frames: {real.image.nunique()} images, XMP altitude {min(xmp.values()):.1f}-{max(xmp.values()):.1f} m")
    return real, dec


def exact_vs_closed(real, dec, curve) -> None:
    nst = {l: n_star(curve, l) for l in LOSSES}
    p = plateau(curve)
    Hs = np.arange(5, 200.1, 0.5)
    rows = []
    for src, d in (("real", real), ("decoy", dec)):
        for name, g in d.groupby("sp"):
            if len(g) < 30 or name in ("unknown", "not a bird"):
                continue
            S = g.S_cm.values / 100
            for cam, f in F_PX.items():
                for l in LOSSES:
                    ok = np.array([np.mean(curve(apparent_size(S, f, H))) >= (1 - l) * p for H in Hs])
                    H_exact = Hs[ok].max() if ok.any() else np.nan
                    H_closed = np.median(S) * f / nst[l]
                    rows.append(dict(source=src, species=name, n=len(g), S_median_cm=np.median(S) * 100, camera=cam,
                                     loss_pct=int(l * 100), n_star=nst[l], H_exact_m=H_exact, H_closed_m=H_closed,
                                     closed_over_exact=H_closed / H_exact))
    t = pd.DataFrame(rows)
    save(t, "hstar_exact_vs_closed.csv")
    print(f"closed/exact: {t.closed_over_exact.min():.3f}-{t.closed_over_exact.max():.3f}")


def validate_decoys(out: pd.DataFrame, sp: pd.DataFrame) -> None:
    u = out[~out.clipped]
    # The size prior comes from train+val decoys only; `out` is the test split, so nothing below
    # is fitted on the birds it is checked against.
    dec = sp[(sp.ds == "Bird_H") & (sp.alt == 15) & (sp.split != "test")].copy()
    dec["S_m"] = np.sqrt(np.maximum(dec.s ** 2 - B_PX ** 2, 1)) * 15 / F_PX["L1D-20c"]
    S_by = {k: g.S_m.values for k, g in dec.groupby("sp")}
    test = u[(u.dataset == "Bird_H") & (u.species != "") & (u.species != "unknown")]
    noH = u[u.dataset != "Bird_H"]
    rows = []
    for m in ("yolov5", "yolo11", "fasterrcnn", "mdo_owl_t"):
        curve = fit_curve(noH, "hit_" + m)            # fitted without any Bird_H bird
        low = test[test.alt_nom <= 30]                  # one level offset for Bird_H, set where size is not limiting
        base_pred = np.mean([np.mean(curve(apparent_size(S_by[s], F_PX["L1D-20c"], H)))
                             for s, H in zip(low.species, low.alt_nom)])
        k = low["hit_" + m].mean() / base_pred
        for (s, H), g in test.groupby(["species", "alt_nom"]):
            if len(g) < 15 or len(S_by.get(s, [])) < 30:
                continue
            sizes = apparent_size(S_by[s], F_PX["L1D-20c"], H)
            rows.append(dict(model=m, decoy_type=s, altitude_m=H, n=len(g), recall_obs=g["hit_" + m].mean() * 100,
                             recall_pred=min(k * np.mean(curve(sizes)), 1) * 100,
                             size_pred_px=np.median(sizes), size_obs_px=g["size"].median()))
    v = pd.DataFrame(rows)
    save(v, "validation_decoys.csv")
    y5 = v[v.model == "yolov5"]
    print(f"validation: size MAE {np.mean(np.abs(y5.size_pred_px - y5.size_obs_px)):.2f} px over {len(y5)} cells")
    for (m, H), g in v.groupby(["model", "altitude_m"]):
        w = g.n / g.n.sum()
        print(f"  {m:10s} {H:3d} m  obs {np.sum(w * g.recall_obs):5.1f}  pred {np.sum(w * g.recall_pred):5.1f}"
              f"  cell MAE {np.mean(np.abs(g.recall_pred - g.recall_obs)):.1f}")
    print(y5[y5.altitude_m == 90][["decoy_type", "n", "recall_obs", "recall_pred", "size_obs_px"]].round(1).to_string(index=False))


def main() -> None:
    birds = read_boxes("birds.csv")
    sp = read_boxes("species.csv")
    sp["s"] = np.sqrt(sp.w * sp.h)
    out = pd.read_pickle(os.path.join(WORK, "outcomes.pkl"))

    size_tables(birds)
    decoy_model(sp, birds)
    camera_ratio(sp)
    curves = detector_response(out)
    precision_by_size()
    real, dec = species_constants(sp)
    exact_vs_closed(real, dec, curves["yolov5"])
    validate_decoys(out, sp)


if __name__ == "__main__":
    main()
