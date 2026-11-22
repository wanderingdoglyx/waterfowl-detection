#!/usr/bin/env python
"""
Figures for the common-architecture report.

    fig1_workflow_status.png   the nine workflow stages, where each lives today, and its state
    fig2_call_structure.png    who loads and runs models: today vs. through one interface

    python output/analysis/architecture/code/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")

INK = "#1f2933"
MUTED = "#52606d"
LINE = "#9aa5b1"
STAGE_FILL = "#f5f7fa"

# State colours carry a text label too, so nothing is read from colour alone.
STATES = {
    "In place":   ("#dcebf5", "#2c6e9b"),
    "Partial":    ("#fbecd0", "#a8641a"),
    "Duplicated": ("#f7dcd6", "#a63d32"),
    "Missing":    ("#f7dcd6", "#a63d32"),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, fill, edge, lw=1.0, radius=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw))


def workflow_status(path: str) -> str:
    stages = [
        ("Dataset", "In place",
         "One adapter per survey programme (pipeline/adapters/)",
         "Canonical records; coverage measured from the files, not claimed"),
        ("Preprocessing / tiling", "In place",
         "512 px crops, stride 460; one shared, adapter-driven tiler",
         "Crops carry their survey conditions and their photograph offsets"),
        ("Model interface", "In place",
         "The command line: each package's main.py, the same four steps",
         "Prepare, train, evaluate, benchmark \u2014 for all eight models"),
        ("Training / checkpoints", "In place",
         "Five train.py wrappers; experiment.json written at training",
         "All eight models save a checkpoint and resolve it again"),
        ("Inference", "In place",
         "Every model runs on crops and writes detections",
         "Six callers: evaluation, analysis, baselines, three benchmarks"),
        ("Common prediction format", "In place",
         "Analysis cache (.npz: crop, x, y, score, w, h), all 8 models",
         "One format behind every cross-model result"),
        ("Evaluation", "In place",
         "Point protocol: one implementation for all 8 models",
         "AP, precision, recall, F1, MAE, RMSE, bootstrap intervals"),
        ("Condition-specific analysis", "In place",
         "Size, density, habitat, altitude, failures, matching audit",
         "All read the analysis cache, so every model is treated alike"),
        ("Tables / figures / reports", "In place",
         "analysis/: CSV tables and figures; Markdown \u2192 Word reports",
         "A model list in a document decides which runs are compared"),
    ]

    W, row_h, gap = 6.5, 0.54, 0.12
    H = len(stages) * (row_h + gap) + 0.36
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    sx, sw, tx, px, pw = 0.04, 2.02, 2.18, 5.66, 0.80
    ax.text(sx, H - 0.14, "Stage", fontsize=7.6, weight="bold", color=MUTED, va="center")
    ax.text(tx, H - 0.14, "Where it lives today", fontsize=7.6, weight="bold", color=MUTED, va="center")
    ax.text(px, H - 0.14, "State", fontsize=7.6, weight="bold", color=MUTED, va="center")

    top = H - 0.30
    for i, (name, state, line1, line2) in enumerate(stages):
        y = top - (i + 1) * row_h - i * gap
        fill, edge = STATES[state]

        box(ax, sx, y, sw, row_h, STAGE_FILL, "#7b8794")
        ax.text(sx + 0.12, y + row_h / 2, f"{i + 1}", fontsize=10, weight="bold", color=MUTED, va="center")
        ax.text(sx + 0.36, y + row_h / 2, name, fontsize=7.8, weight="bold", color=INK, va="center")

        ax.text(tx, y + row_h * 0.68, line1, fontsize=6.7, color=INK, va="center")
        ax.text(tx, y + row_h * 0.30, line2, fontsize=6.7, color=MUTED, va="center")

        box(ax, px, y + 0.11, pw, row_h - 0.22, fill, edge, lw=1.1, radius=0.09)
        ax.text(px + pw / 2, y + row_h / 2, state, fontsize=7.2, weight="bold", color=edge,
                ha="center", va="center")

        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((sx + sw / 2, y - 0.005), (sx + sw / 2, y - gap + 0.005),
                                         arrowstyle="-|>", mutation_scale=6, color="#7b8794", lw=1))

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def call_structure(path: str) -> str:
    callers = ["Per-model evaluation", "Analysis prediction cache", "Zero-shot baselines",
               "Efficiency benchmark", "Batch / frame benchmark", "Training benchmark"]
    frameworks = ["Ultralytics\nYOLOv5 · YOLO11 · YOLO26", "super-gradients\nYOLO-NAS",
                  "Detectron2\nFaster R-CNN", "animaloc (own Python)\nOWL-C · OWL-T · OWL-D"]
    # callers x frameworks -> number of separate implementations (0 = not covered)
    impl = [[2, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0]]

    W, H = 6.5, 3.55
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    cw, ch = 1.28, 0.36
    fw, fh = 1.30, 0.52
    ys_c = [2.93 - k * 0.5 for k in range(6)]
    ys_f = [2.78 - k * 0.72 for k in range(4)]

    # ── Left: today ───────────────────────────────────────────────────────────
    ax.text(0.05, 3.40, "Today: 23 separate implementations", fontsize=9, weight="bold", color=INK)
    cx0, fx0 = 0.05, 1.84
    for r, y in enumerate(ys_c):
        for c, yf in enumerate(ys_f):
            n = impl[r][c]
            for k in range(n):
                off = (k - (n - 1) / 2) * 0.05
                ax.plot([cx0 + cw, fx0], [y + ch / 2 + off, yf + fh / 2 + off],
                        color="#b0413e", lw=0.8, alpha=0.75, zorder=1)
    for label, y in zip(callers, ys_c):
        box(ax, cx0, y, cw, ch, STAGE_FILL, "#7b8794")
        ax.text(cx0 + cw / 2, y + ch / 2, label, fontsize=6.6, color=INK, ha="center", va="center")
    for label, y in zip(frameworks, ys_f):
        box(ax, fx0, y, fw, fh, "#ffffff", "#7b8794")
        ax.text(fx0 + fw / 2, y + fh / 2, label, fontsize=6.3, color=INK, ha="center", va="center",
                linespacing=1.3)

    ax.plot([3.27, 3.27], [0.1, 3.3], color="#cbd2d9", lw=0.8)

    # ── Right: proposed ───────────────────────────────────────────────────────
    ax.text(3.42, 3.40, "Proposed: one interface, four adapters", fontsize=9, weight="bold", color=INK)
    cx1, ix, iw = 3.42, 4.92, 0.34
    ax1 = 5.40
    aw = 1.06
    for y in ys_c:
        ax.plot([cx1 + cw, ix], [y + ch / 2, y + ch / 2], color="#2c6e9b", lw=0.9, zorder=1)
    for yf in ys_f:
        ax.plot([ix + iw, ax1], [yf + fh / 2, yf + fh / 2], color="#2c6e9b", lw=0.9, zorder=1)
    for label, y in zip(callers, ys_c):
        box(ax, cx1, y, cw, ch, STAGE_FILL, "#7b8794")
        ax.text(cx1 + cw / 2, y + ch / 2, label, fontsize=6.6, color=INK, ha="center", va="center")
    y_lo, y_hi = ys_c[-1] - 0.05, ys_c[0] + ch + 0.05
    box(ax, ix, y_lo, iw, y_hi - y_lo, "#dcebf5", "#2c6e9b", lw=1.2, radius=0.08)
    ax.text(ix + iw / 2, (y_lo + y_hi) / 2, "Detector interface", rotation=90, fontsize=7.8,
            weight="bold", color="#2c6e9b", ha="center", va="center")
    adapters = ["Ultralytics\nadapter", "YOLO-NAS\nadapter", "Faster R-CNN\nadapter",
                "OWL adapter\n(calls its own Python)"]
    for label, y in zip(adapters, ys_f):
        box(ax, ax1, y, aw, fh, "#ffffff", "#2c6e9b")
        ax.text(ax1 + aw / 2, y + fh / 2, label, fontsize=6.3, color=INK, ha="center", va="center",
                linespacing=1.3)

    ax.text(0.05, 0.08, "One line = one implementation", fontsize=6.8, color=MUTED)
    ax.text(3.42, 0.08, "Framework-specific code is written once, in its adapter", fontsize=6.8, color=MUTED)

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for p in (workflow_status(os.path.join(FIG_DIR, "fig1_workflow_status.png")),
              call_structure(os.path.join(FIG_DIR, "fig2_call_structure.png"))):
        print("wrote", os.path.normpath(p))


if __name__ == "__main__":
    main()
