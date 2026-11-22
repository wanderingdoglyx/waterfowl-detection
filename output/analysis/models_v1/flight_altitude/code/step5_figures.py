"""
Step 5 — report figures (PNG, 200 dpi, 6.5 in wide) from results/ and the source images.

  fig1_geometry.png           imaging geometry behind the equation
  fig2_operator_chart.png     recommended flight height vs bird size for the two cameras
  fig3_decoy_size.png         decoy box size vs height, with the fitted size model
  fig4_camera_annotation.png  the same species from both cameras at one ground scale
  fig5_size_response.png      dataset-standardised recall and precision vs box size
  fig6_validation.png         out-of-sample size and recall prediction on Bird_H test decoys
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from PIL import Image

from common import (B_PX, DATASET, F_PX, P_OPTIMUM, REPORT_DIR, RESULTS, WORK, blur_factor, dji_xmp, image_info, read_class_boxes,
                    class_file)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
QUIET = "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 8,
    "savefig.dpi": 200, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})
MARK = dict(markeredgecolor="white", markeredgewidth=1.0)


def out(name: str) -> str:
    return os.path.join(REPORT_DIR, name)


def res(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def log_ticks(ax, axis, ticks):
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([f"{t:g}" for t in ticks])
    getattr(ax, f"{axis}axis").set_minor_locator(matplotlib.ticker.NullLocator())


# ── Figure 1: imaging geometry (fig1_geometry.png) ───────────────────────────
def fig1():
    """Side view of the imaging geometry. Symbols only; the equations themselves are typeset in the text."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_axis_off()
    ax.set_xlim(-8.6, 9.6)
    ax.set_ylim(-1.6, 10.4)
    ys, yl = 9.4, 8.0
    gx = 1.0 * yl / (ys - yl)                                           # ground half-width of the footprint
    ax.plot([-1, 1], [ys, ys], color=INK, lw=3.4)
    ax.plot([0.55, 0.75], [ys, ys], color=BLUE, lw=3.6)
    ax.add_patch(Ellipse((0, yl), 1.0, 0.3, fill=False, edgecolor=INK, lw=1.2))
    for sx in (-1, 1):
        ax.plot([sx, -sx * gx], [ys, 0], color=MUTED, lw=0.8)
    for px in (0.55, 0.75):
        ax.plot([px, -px * gx], [ys, 0], color=BLUE, lw=0.8)
    ax.plot([-6.6, 6.6], [0, 0], color=AXIS, lw=1.4)
    ax.plot([-0.75 * gx, -0.55 * gx], [0, 0], color=BLUE, lw=4.5)
    ax.add_patch(Ellipse((2.3, 0.24), 1.5, 0.46, facecolor=ORANGE, edgecolor="white", lw=1.0))

    big, small = 11, 10

    def dim(x0, y0, x1, y1, text, tx, ty, ha="left"):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        ax.text(tx, ty, text, color=INK, ha=ha, va="center", fontsize=big)

    dim(1.5, yl, 1.5, ys, "$f$  focal length", 1.75, (yl + ys) / 2)
    dim(6.9, 0, 6.9, yl, "$H$  flight height\nabove the birds", 7.15, yl / 2)
    dim(-gx, -0.8, gx, -0.8, "ground footprint", 0, -1.25, ha="center")
    dim(1.55, 0.8, 3.05, 0.8, "$S$  bird size", 3.25, 0.8)
    ax.text(-1.3, ys, "sensor: width $w$, $N$ pixels", ha="right", va="center", color=INK, fontsize=big)
    ax.text(0.65, ys + 0.55, "one pixel", color=INK2, fontsize=small, ha="center")
    ax.text(-0.75, yl, "lens", ha="right", va="center", color=INK2, fontsize=small)
    ax.text(-0.65 * gx, 0.5, "GSD", ha="center", color=INK, fontsize=big)
    ax.text(9.5, -1.55, "not to scale", ha="right", color=MUTED, fontsize=8.5)
    fig.savefig(out("fig1_geometry.png"), bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: decoy size vs height (fig3_decoy_size.png) ─────────────────────
def fig2():
    cells = res("decoy_size_cells.csv")
    fit = res("decoy_size_fit.csv").set_index("decoy_type").a_px_m
    hl = {"green winged teal": ("teal", BLUE), "gadwall": ("gadwall", ORANGE), "mallard male": ("mallard ♂", AQUA)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.32))
    H = np.linspace(13, 100, 200)
    for t, g in cells.groupby("sp"):
        if t not in hl:
            a.plot(g.alt, g.s, "o", color=QUIET, ms=3.5, zorder=1)
    series = []
    for t, (lab, col) in hl.items():
        g = cells[cells.sp == t]
        a.plot(H, fit[t] / H, "--", color=col, lw=1.0, zorder=2)
        a.plot(H, np.sqrt((fit[t] / H) ** 2 + B_PX ** 2), "-", color=col, lw=1.8, zorder=3)
        series.append(a.plot(g.alt, g.s, "o-", color=col, lw=1.8, ms=6, zorder=4, **MARK, label=lab)[0])
        series[-1].set_linestyle("none")
    series.append(a.plot([], [], "o", color=QUIET, ms=3.5, label="other 7 types")[0])
    log_ticks(a, "x", [15, 30, 60, 90])
    log_ticks(a, "y", [10, 20, 50, 100])
    a.set_xlim(12, 110)
    a.set_ylim(9, 130)
    a.set_xlabel("Flight height (m)")
    a.set_ylabel("Median box size $P_\\mathrm{box}$ (px)")
    a.set_title("(a) Decoy size in pixels vs height")
    leg = a.legend(handles=series, loc="upper right", handletextpad=0.3)
    a.add_artist(leg)
    styles = [a.plot([], [], "-", color=INK2, lw=1.8, label="size model")[0],
              a.plot([], [], "--", color=INK2, lw=1.0, label="geometry only")[0]]
    a.legend(handles=styles, loc="lower left", handlelength=2.2)

    # (b) ratio to the 15 m prediction of pure geometry
    rng = np.random.default_rng(0)
    for alt, g in cells.groupby("alt"):
        b.plot(alt * np.exp(rng.uniform(-0.05, 0.05, len(g))), g.k_ratio_vs_15m, "o", color=QUIET, ms=4, zorder=2)
    Hb = np.linspace(15, 95, 100)
    ratio = lambda A: np.sqrt((A / Hb) ** 2 + B_PX ** 2) * Hb / (np.sqrt((A / 15) ** 2 + B_PX ** 2) * 15)
    lo, hi = ratio(fit.max()), ratio(fit.min())
    b.fill_between(Hb, lo, hi, color=BLUE, alpha=0.12, lw=0, zorder=1)
    b.plot(Hb, lo, color=BLUE, lw=1.0)
    b.plot(Hb, hi, color=BLUE, lw=1.0)
    mean = cells.groupby("alt").k_ratio_vs_15m.mean()
    b.plot(mean.index, mean.values, "-o", color=INK, lw=1.4, ms=5, zorder=3, **MARK)
    b.axhline(1.0, color=INK2, lw=0.8, zorder=1)
    log_ticks(b, "x", [15, 30, 60, 90])
    b.set_xlim(13, 105)
    b.set_ylim(0.95, 1.25)
    b.set_xlabel("Flight height (m)")
    b.set_ylabel("Box size × height, relative to 15 m")
    b.set_title("(b) Departure from size ∝ 1/height")
    handles = [b.plot([], [], "-o", color=INK, lw=1.4, ms=5, **MARK, label="mean of 10 decoy types")[0],
               b.plot([], [], "o", color=QUIET, ms=4, label="one decoy type")[0],
               b.fill_between([], [], [], color=BLUE, alpha=0.25, lw=0, label="size model, teal to mallard"),
               b.plot([], [], "-", color=INK2, lw=0.8, label="pure geometry")[0]]
    b.legend(handles=handles, loc="upper left", handletextpad=0.5)
    fig.savefig(out("fig3_decoy_size.png"), bbox_inches="tight")
    plt.close(fig)


# ── Figure 5: detector size response (fig5_size_response.png) ────────────────
def fig3():
    cur = res("recall_curve_standardised.csv")
    prec = res("precision_by_size.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.3))
    others = [c for c in cur.columns if c not in ("size_px", "yolov5")]
    for c in others:
        a.plot(cur.size_px, cur[c], color=QUIET, lw=0.9, zorder=1)
    a.plot(cur.size_px, cur.yolov5, color=BLUE, lw=1.9, zorder=3)
    pl = cur.loc[(cur.size_px >= 18) & (cur.size_px <= 40), "yolov5"].max()
    a.axhline(pl, color=INK2, lw=0.7, zorder=2)
    a.text(18.5, pl + 1.3, f"plateau {pl:.1f}%", ha="left", color=INK2, fontsize=8)
    a.axvspan(6, 14, color=MUTED, alpha=0.08, lw=0)
    a.text(9.2, 97, "below 14 px", ha="center", va="center", color=INK2, fontsize=8)
    a.axvline(P_OPTIMUM, color=INK, lw=0.9, zorder=2)
    a.text(P_OPTIMUM * 1.05, 62, f"optimum\n{P_OPTIMUM:.0f} px", color=INK, fontsize=8, va="center")
    a.plot([], [], color=BLUE, lw=1.9, label="YOLOv5")
    a.plot([], [], color=QUIET, lw=0.9, label="other 7 detectors")
    a.legend(loc="lower right")
    log_ticks(a, "x", [8, 10, 15, 20, 30, 50, 100])
    a.set_xlim(6, 150)
    a.set_ylim(30, 100)
    a.set_xlabel("Box size of the bird, $P_\\mathrm{box}$ (px)")
    a.set_ylabel("Recall (%), dataset-standardised")
    a.set_title("(a) Share of birds found vs bird size")

    edges = [0, 10, 12, 14, 16, 18, 20, 25, 30, 40, 60, 400]
    mids = [8, 11, 13, 15, 17, 19, 22.4, 27.4, 34.6, 49, 90]
    y5 = prec[prec.model == "yolov5"].reset_index(drop=True)
    b.plot(mids[:len(y5)], y5.precision_std, "-o", color=BLUE, lw=1.9, ms=5.5, **MARK)
    b.axvspan(6, 14, color=MUTED, alpha=0.08, lw=0)
    log_ticks(b, "x", [8, 10, 15, 20, 30, 50, 100])
    b.set_xlim(6, 150)
    b.set_ylim(60, 100)
    b.set_xlabel("Size of the detected box (px)")
    b.set_ylabel("Precision (%), dataset-standardised")
    b.set_title("(b) YOLOv5 precision vs box size")
    fig.savefig(out("fig5_size_response.png"), bbox_inches="tight")
    plt.close(fig)


# ── Figure 6: validation (fig6_validation.png) ───────────────────────────────
def fig4():
    v = res("validation_decoys.csv")
    y5 = v[v.model == "yolov5"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.32))
    lim = [10, 130]
    a.plot(lim, lim, color=INK2, lw=0.8)
    a.plot(y5.size_obs_px, y5.size_pred_px, "o", color=BLUE, ms=5.5, **MARK)
    for H, g in y5.groupby("altitude_m"):
        a.text(g.size_obs_px.max() * 1.1, g.size_pred_px.median(), f"{H} m", color=INK2, fontsize=8, va="center")
    mae = np.mean(np.abs(y5.size_pred_px - y5.size_obs_px))
    a.text(11, 110, f"{len(y5)} decoy type × altitude cells\nmean abs. error {mae:.1f} px", color=INK, fontsize=8, va="top")
    log_ticks(a, "x", [10, 20, 50, 100])
    log_ticks(a, "y", [10, 20, 50, 100])
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel("Observed median box size (px)")
    a.set_ylabel("Predicted from 15 m train decoys (px)")
    a.set_title("(a) Apparent size")

    small = (y5.size_pred_px < 15).values
    n, p = y5.n.values, y5.recall_obs.values / 100
    z = 1.96                                               # Wilson score interval
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    err = np.clip(np.vstack([p - (centre - half), (centre + half) - p]) * 100, 0, None)
    b.plot([70, 100], [70, 100], color=INK2, lw=0.8)
    for m, col, lab in ((~small, QUIET, "size ≥ 15 px (plateau)"), (small, ORANGE, "teal at 90 m (13 px)")):
        b.errorbar(p[m] * 100, y5.recall_pred.values[m], xerr=err[:, m], fmt="o", color=col, ms=5.5,
                   ecolor=col, elinewidth=0.8, capsize=0, mec="white", mew=1.0, label=lab, zorder=3)
    b.set_xlim(70, 101)
    b.set_ylim(70, 101)
    b.set_xlabel("Observed recall (%), with 95% CI")
    b.set_ylabel("Predicted recall (%)")
    b.set_title("(b) Recall, YOLOv5")
    b.legend(loc="lower right")
    fig.savefig(out("fig6_validation.png"), bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: operator chart (fig2_operator_chart.png) ───────────────────────
def fig5():
    opt = res("optimum_point.csv").set_index("detector").loc["YOLOv5"]
    geo = np.sqrt(P_OPTIMUM ** 2 - B_PX ** 2)
    f_hi = geo / np.sqrt(opt.within_1pt_lo_px ** 2 - B_PX ** 2)    # smallest good box size -> highest height
    f_lo = geo / np.sqrt(opt.within_1pt_hi_px ** 2 - B_PX ** 2)
    k = blur_factor()
    sp = res("species_size.csv")
    real = sp[sp.source.str.startswith("real")].set_index("sp").S_median_cm
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    S = np.linspace(15, 40, 100)
    for cam, col, lab in (("L1D-20c", BLUE, "Mavic 2 Pro"), ("FC3170", ORANGE, "Mavic Air 2 (12 MP)")):
        f = F_PX[cam]
        h_rec = k * S / 100 * f / P_OPTIMUM
        ax.plot(S, h_rec, color=col, lw=1.9, label=f"{lab}, camera factor $C$ = {f:,.0f}")
        ax.text(40.4, h_rec[-1], lab, color=INK, va="center", fontsize=8.5)
        for name, lbl in (("green winged teal", "teal"), ("gadwall", "gadwall"), ("mallard male", "mallard ♂")):
            s = real[name]
            h = k * s / 100 * f / P_OPTIMUM
            ax.plot(s, h, "o", color=col, ms=6, zorder=4, **MARK)
            if cam == "L1D-20c":
                ax.annotate(f"{lbl} {h:.0f} m", (s, h), (-7, 5), textcoords="offset points",
                            ha="right", va="bottom", fontsize=7.5, color=INK)
            else:
                ax.annotate(f"{lbl} {h:.0f} m", (s, h), (7, -5), textcoords="offset points",
                            ha="left", va="top", fontsize=7.5, color=INK)
    ax.axhline(121.9, color=MUTED, lw=0.8)
    ax.text(15.3, 124, "FAA Part 107 ceiling, 122 m (400 ft)", color=INK2, fontsize=8)
    ax.text(15.3, 112, f"Flying {f_lo:.2f}–{f_hi:.2f} times the recommended height keeps detection within 1 point of the best",
            color=INK2, fontsize=8)
    ax.set_xlim(15, 47)
    ax.set_ylim(0, 135)
    ax.set_xticks([15, 20, 25, 30, 35, 40])
    ax.set_xlabel("Bird size $S$ (cm)")
    ax.set_ylabel("Recommended flight height $H_\\mathrm{rec}$ (m)")
    ax.set_title(f"Recommended flight height at the optimum pixel size (YOLOv5, $P_\\mathrm{{optimum}}$ = {P_OPTIMUM:.0f} px)")
    ax.legend(loc="lower right")
    fig.savefig(out("fig2_operator_chart.png"), bbox_inches="tight")
    plt.close(fig)


# ── Figure 4: same species, both cameras (fig4_camera_annotation.png) ────────
def fig6():
    target_gsd = 0.35  # cm/px
    rows = {r["image_name"]: r for r in image_info("Bird_I")}
    picks = []
    rng = np.random.default_rng(3)
    for sp, lab in (("mallard male", "Mallard ♂"), ("pintail male", "Pintail ♂")):
        for cam, cname in (("FC3170", "Mavic Air 2"), ("L1D-20c", "Mavic 2 Pro")):
            cands = []
            for name, r in rows.items():
                p = os.path.join(DATASET, "Bird_I", name)
                if not name.lower().endswith(".jpg") or not os.path.exists(p):
                    continue
                if (Image.open(p).getexif().get(0x0110) or "") != cam:
                    continue
                cf = class_file("Bird_I", r)
                cands += [(name, b) for b in (read_class_boxes(cf) if cf else []) if b[0] == sp]
            sel = [cands[i] for i in rng.choice(len(cands), size=4, replace=False)]
            picks.append((f"{lab}\n{cname}", cam, sel))
    fig, axes = plt.subplots(len(picks), 4, figsize=(6.5, 6.9), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    win = 250
    for ri, (label, cam, sel) in enumerate(picks):
        for ci, (name, (_, x1, y1, x2, y2)) in enumerate(sel):
            p = os.path.join(DATASET, "Bird_I", name)
            H = float(dji_xmp(p, "RelativeAltitude"))
            gsd = H / F_PX[cam] * 100
            k = gsd / target_gsd                          # source px -> display px
            half = int(np.ceil(win / 2 / k))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop = Image.open(p).convert("RGB").crop((cx - half, cy - half, cx + half, cy + half))
            crop = crop.resize((int(round(2 * half * k)),) * 2, Image.LANCZOS)
            ax = axes[ri, ci]
            ax.imshow(crop)
            ax.add_patch(plt.Rectangle(((x1 - cx + half) * k, (y1 - cy + half) * k), (x2 - x1) * k, (y2 - y1) * k,
                                       fill=False, edgecolor=RED, lw=1.1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if ci == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", color=INK, fontsize=8.5, labelpad=6)
            if ri == 0 and ci == 0:                       # 50 cm scale bar at the common ground scale
                bar = 50 / target_gsd
                ax.plot([12, 12 + bar], [win - 14, win - 14], color="white", lw=2.2, solid_capstyle="butt")
                ax.text(12 + bar / 2, win - 22, "50 cm", color="white", ha="center", va="bottom", fontsize=8)
    fig.savefig(out("fig4_camera_annotation.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print("wrote", f.__name__)
