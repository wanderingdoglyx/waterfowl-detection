#!/usr/bin/env python
"""
Figures for the Arkansas specification and domain-shift protocol report (Tasks 4.4 and 4.5).

    fig5_experiment_matrix.png   the five experiments: what each trains on, tests on, establishes
    fig6_test_set_precision.png  how precise an Arkansas result is, against test-set size

    python output/analysis/architecture/code/make_arkansas_figures.py

The precision figure reads results/domain_shift_precision.csv (domain_shift_precision.py).
Series colours are slots 1 and 2 of the validated categorical palette; identity is carried
by a legend and by direct labels, never by colour alone.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
CSV_PATH = os.path.join(HERE, "..", "results", "domain_shift_precision.csv")

INK = "#1f2933"
MUTED = "#52606d"
GRID = "#d8dee5"
GREY_FILL = "#f5f7fa"
GREY_EDGE = "#7b8794"
MO_FILL, MO_EDGE = "#dcebf5", "#2c6e9b"          # Missouri
AR_FILL, AR_EDGE = "#fbecd0", "#a8641a"          # Arkansas
SERIES = ("#2a78d6", "#eb6834")                   # categorical slots 1 and 2

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, fill, edge, lw=1.0, radius=0.05, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle="--" if dashed else "-"))


def experiment_matrix(path: str) -> str:
    rows = [
        ("1", "Missouri", "mo", "Missouri", "mo",
         "The reference point: today's published results", "ready now"),
        ("2", "Missouri", "mo", "Arkansas (all of it)", "ar",
         "True Missouri-to-Arkansas generalisation",
         "first experiment \u2014 no Arkansas in training"),
        ("3", "Arkansas", "ar", "Arkansas (held-out sites)", "ar",
         "What Arkansas data alone achieves", "needs Arkansas data"),
        ("4", "Missouri, then Arkansas", "both", "Arkansas (held-out sites)", "ar",
         "What fine-tuning buys, and how little is enough", "needs Arkansas data"),
        ("5", "Missouri + Arkansas", "both", "Arkansas (held-out sites)", "ar",
         "Whether pooling both states beats either", "needs Arkansas data"),
    ]

    W, row_h, gap = 6.5, 0.62, 0.14
    H = len(rows) * (row_h + gap) + 0.46
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    xn, xt, xa, xs, xw = 0.05, 0.38, 1.92, 2.30, 3.85
    ax.text(xt, H - 0.16, "Train / fine-tune on", fontsize=7.4, weight="bold", color=MUTED)
    ax.text(xs, H - 0.16, "Test on", fontsize=7.4, weight="bold", color=MUTED)
    ax.text(xw, H - 0.16, "What it establishes", fontsize=7.4, weight="bold", color=MUTED)

    palette = {"mo": (MO_FILL, MO_EDGE), "ar": (AR_FILL, AR_EDGE), "both": (MO_FILL, MO_EDGE)}
    top = H - 0.34
    for i, (num, train, tkind, test, skind, what, status) in enumerate(rows):
        y = top - (i + 1) * row_h - i * gap
        first = num == "2"

        ax.text(xn, y + row_h / 2, num, fontsize=10, weight="bold",
                color=AR_EDGE if first else MUTED, va="center")

        tf, te = palette[tkind]
        box(ax, xt, y + 0.12, 1.44, row_h - 0.24, tf, te,
            lw=1.3 if first else 1.0, dashed=tkind == "ar")
        ax.text(xt + 0.72, y + row_h / 2, train, fontsize=6.8, color=INK,
                ha="center", va="center")
        if tkind == "both":
            ax.text(xt + 0.72, y + 0.05, "Missouri first, then Arkansas" if num == "4"
                    else "one pooled training set",
                    fontsize=5.8, color=MUTED, ha="center", va="center")

        ax.add_patch(FancyArrowPatch((xa, y + row_h / 2), (xs - 0.06, y + row_h / 2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=AR_EDGE if first else GREY_EDGE,
                                     lw=1.3 if first else 1.0))

        sf, se = palette[skind]
        box(ax, xs, y + 0.12, 1.44, row_h - 0.24, sf, se,
            lw=1.3 if first else 1.0, dashed=skind == "ar")
        ax.text(xs + 0.72, y + row_h / 2, test, fontsize=6.8, color=INK,
                ha="center", va="center")

        ax.text(xw, y + row_h * 0.62, what, fontsize=6.8, color=INK, va="center")
        ax.text(xw, y + row_h * 0.26, status, fontsize=6.4, style="italic",
                weight="bold" if first else "normal",
                color=AR_EDGE if first else MUTED, va="center")

    ax.text(0.05, 0.06, "Dashed outline: data that does not exist yet.  "
                        "Held-out sites: no survey site appears in both training and test.",
            fontsize=6.4, color=MUTED)

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def test_set_precision(path: str) -> str:
    rows = list(csv.DictReader(open(CSV_PATH)))
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), dpi=300)
    panels = [
        (axes[0], "ap_halfwidth_pts", "Detection AP", "95% interval half-width (AP points)",
         5.0, "a 5-point drop", "right", 80, (9, -12)),
        (axes[1], "mae_halfwidth", "Counting error", "95% interval half-width (birds per crop)",
         0.2, "a 0.2-bird difference", "left", 20, (-12, 9)),
    ]

    yticks = {"ap_halfwidth_pts": [2, 5, 10, 20, 35],
              "mae_halfwidth": [0.1, 0.2, 0.5, 1, 2, 5, 10]}
    for ax, field, title, ylabel, guide, guide_label, guide_side, label_x, dys in panels:
        for (model, colour, dy) in zip(models, SERIES, dys):
            pts = sorted((int(r["n_photographs"]), float(r[field]))
                         for r in rows if r["model"] == model)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=colour, lw=1.8, marker="o", markersize=3.4,
                    markeredgecolor="white", markeredgewidth=0.6, label=model, zorder=3)
            i = xs.index(label_x)
            ax.annotate(model, (xs[i], ys[i]), textcoords="offset points",
                        xytext=(0, dy), fontsize=6.4, color=INK, ha="center")

        ax.axhline(guide, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        gx, gha = (330, "right") if guide_side == "right" else (2.05, "left")
        ax.text(gx, guide * 1.12, guide_label, fontsize=6.2, color=MUTED,
                va="bottom", ha=gha)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 10, 20, 40, 80, 160, 320])
        ax.set_xticklabels(["2", "5", "10", "20", "40", "80", "160", "320"], fontsize=6.4)
        ax.set_yticks(yticks[field])
        ax.set_yticklabels([f"{v:g}" for v in yticks[field]], fontsize=6.4)
        ax.minorticks_off()
        ax.set_xlabel("Arkansas photographs in the test set", fontsize=7, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8, weight="bold", color=INK, pad=4)
        ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY_EDGE)
            ax.spines[side].set_linewidth(0.8)

    axes[0].legend(fontsize=6.4, frameon=False, loc="lower left", handlelength=1.6)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for p in (experiment_matrix(os.path.join(FIG_DIR, "fig5_experiment_matrix.png")),
              test_set_precision(os.path.join(FIG_DIR, "fig6_test_set_precision.png"))):
        print("wrote", os.path.normpath(p))


if __name__ == "__main__":
    main()
