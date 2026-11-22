#!/usr/bin/env python
"""
Figures for the common-dataset-interface report (Task 4.3).

    fig3_dataset_layers.png   where the dataset boundary sits, and what each side may know
    fig4_metadata_flow.png    how a crop's survey conditions reach a table: today vs proposed

    python output/analysis/architecture/code/make_dataset_figures.py
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
GREY_FILL = "#f5f7fa"
GREY_EDGE = "#7b8794"
BLUE_FILL = "#dcebf5"
BLUE_EDGE = "#2c6e9b"
SAND_FILL = "#fbecd0"
SAND_EDGE = "#a8641a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, fill, edge, lw=1.0, radius=0.06, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle="--" if dashed else "-"))


def arrow(ax, x, y0, y1, color=GREY_EDGE):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=8, color=color, lw=1.1))


def dataset_layers(path: str) -> str:
    W, H = 6.5, 5.25
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    left, wide = 0.28, 5.0

    # Sources
    box(ax, left, 0.22, 2.35, 0.62, GREY_FILL, GREY_EDGE)
    ax.text(left + 1.17, 0.66, "MDC surveys (Missouri)", fontsize=7.4, weight="bold",
            color=INK, ha="center", va="center")
    ax.text(left + 1.17, 0.41, "10 collections, 2,095 photographs\n7 spreadsheet layouts",
            fontsize=6.4, color=MUTED, ha="center", va="center", linespacing=1.35)

    box(ax, left + 2.65, 0.22, 2.35, 0.62, "#ffffff", GREY_EDGE, dashed=True)
    ax.text(left + 3.82, 0.66, "Arkansas surveys (future)", fontsize=7.4, weight="bold",
            color=MUTED, ha="center", va="center")
    ax.text(left + 3.82, 0.41, "bottomland hardwood, rice, soybean\nits own layout and vocabulary",
            fontsize=6.4, color=MUTED, ha="center", va="center", linespacing=1.35)

    arrow(ax, left + 1.17, 0.88, 1.12)
    arrow(ax, left + 3.82, 0.88, 1.12)

    # Adapters
    box(ax, left, 1.16, 2.35, 0.66, SAND_FILL, SAND_EDGE, lw=1.2)
    ax.text(left + 1.17, 1.63, "MDC dataset adapter", fontsize=7.4, weight="bold",
            color=SAND_EDGE, ha="center", va="center")
    ax.text(left + 1.17, 1.36, "file naming · CSV columns · label\nformats · vocabulary · cameras",
            fontsize=6.4, color=INK, ha="center", va="center", linespacing=1.35)

    box(ax, left + 2.65, 1.16, 2.35, 0.66, "#ffffff", SAND_EDGE, lw=1.2, dashed=True)
    ax.text(left + 3.82, 1.63, "Arkansas dataset adapter", fontsize=7.4, weight="bold",
            color=SAND_EDGE, ha="center", va="center")
    ax.text(left + 3.82, 1.36, "one new file; nothing above\nthis line changes",
            fontsize=6.4, color=MUTED, ha="center", va="center", linespacing=1.35)

    arrow(ax, left + 1.17, 1.86, 2.10)
    arrow(ax, left + 3.82, 1.86, 2.10)

    # Common dataset interface
    box(ax, left, 2.14, wide, 0.72, BLUE_FILL, BLUE_EDGE, lw=1.4)
    ax.text(left + wide / 2, 2.67, "Common dataset interface", fontsize=8.4, weight="bold",
            color=BLUE_EDGE, ha="center", va="center")
    ax.text(left + wide / 2, 2.37,
            "canonical records: photograph (camera, altitude, GSD, habitat, weather, site, date,\n"
            "split role, collection) · annotation (box or point, species, decoy) · coverage report",
            fontsize=6.4, color=INK, ha="center", va="center", linespacing=1.35)

    arrow(ax, left + wide / 2, 2.90, 3.14)

    # Dataset-agnostic core
    box(ax, left, 3.18, wide, 0.56, GREY_FILL, GREY_EDGE)
    ax.text(left + wide / 2, 3.58, "Dataset-agnostic core", fontsize=7.8, weight="bold",
            color=INK, ha="center", va="center")
    ax.text(left + wide / 2, 3.34, "tiling · split assembly · COCO files carrying the metadata",
            fontsize=6.4, color=MUTED, ha="center", va="center")

    arrow(ax, left + wide / 2, 3.78, 4.02)

    # Model adapters (Task 4.2)
    box(ax, left, 4.06, wide, 0.46, GREY_FILL, GREY_EDGE)
    ax.text(left + wide / 2, 4.29,
            "Model adapters (Task 4.2):  Ultralytics · YOLO-NAS · Detectron2 · OWL",
            fontsize=7.0, color=INK, ha="center", va="center")

    arrow(ax, left + wide / 2, 4.56, 4.80)

    # Evaluation
    box(ax, left, 4.84, wide, 0.30, GREY_FILL, GREY_EDGE)
    ax.text(left + wide / 2, 4.99, "Evaluation, condition analysis, tables and figures",
            fontsize=7.0, color=INK, ha="center", va="center")

    # The rule, drawn as a boundary line with a caption
    ax.plot([0.08, W - 0.08], [2.98, 2.98], color=BLUE_EDGE, lw=0.9, ls=(0, (4, 3)))
    ax.text(W - 0.10, 3.03, "above this line nothing parses a file name or names a collection",
            fontsize=6.4, style="italic", color=BLUE_EDGE, ha="right", va="bottom")

    ax.plot([0.08, W - 0.08], [2.06, 2.06], color=SAND_EDGE, lw=0.9, ls=(0, (4, 3)))
    ax.text(W - 0.10, 1.99, "specific to one survey programme",
            fontsize=6.4, style="italic", color=SAND_EDGE, ha="right", va="top")

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def metadata_flow(path: str) -> str:
    W, H = 6.5, 4.15
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    col_w = 2.85
    x0, x1 = 0.12, 3.53

    def stack(x, title, title_colour, rows, note):
        ax.text(x + col_w / 2, H - 0.18, title, fontsize=8.6, weight="bold",
                color=title_colour, ha="center", va="center")
        y = H - 0.52
        for i, (text, sub, fill, edge) in enumerate(rows):
            h = 0.46 if sub else 0.32
            box(ax, x, y - h, col_w, h, fill, edge)
            if sub:
                ax.text(x + col_w / 2, y - 0.16, text, fontsize=6.8, color=INK,
                        ha="center", va="center")
                ax.text(x + col_w / 2, y - 0.33, sub, fontsize=6.2, color=MUTED,
                        ha="center", va="center", linespacing=1.3)
            else:
                ax.text(x + col_w / 2, y - h / 2, text, fontsize=6.8, color=INK,
                        ha="center", va="center")
            if i < len(rows) - 1:
                arrow(ax, x + col_w / 2, y - h - 0.02, y - h - 0.20)
            y -= h + 0.22
        ax.text(x + col_w / 2, y + 0.02, note, fontsize=6.6, weight="bold",
                color=title_colour, ha="center", va="center")

    stack(x0, "Before", "#a63d32", [
        ("Crop file name", "G_Cloud_WaterSubVeg_15m_test2_0_460.JPG", GREY_FILL, GREY_EDGE),
        ("Regular expression in the analysis", "collection, source stem, tile offset", GREY_FILL, GREY_EDGE),
        ("Per-collection rules", "habitat: column in one, file-name token in two;\nsite and date: three more rules", "#f7dcd6", "#a63d32"),
        ("Spreadsheet read again", "altitude, habitat column", GREY_FILL, GREY_EDGE),
        ("Condition tables", None, GREY_FILL, GREY_EDGE),
    ], "A new survey means editing the analysis")

    stack(x1, "Now (built)", BLUE_EDGE, [
        ("Dataset adapter", "reads the survey's own layout once", SAND_FILL, SAND_EDGE),
        ("Canonical photograph record", "habitat, altitude, GSD, weather, site, date,\nsplit role, collection, coverage", BLUE_FILL, BLUE_EDGE),
        ("Tiling copies it onto every crop", "stored in the COCO file", GREY_FILL, GREY_EDGE),
        ("Analysis reads fields", "no parsing, no collection names", GREY_FILL, GREY_EDGE),
        ("Condition tables", None, GREY_FILL, GREY_EDGE),
    ], "A new survey means one new adapter")

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for p in (dataset_layers(os.path.join(FIG_DIR, "fig3_dataset_layers.png")),
              metadata_flow(os.path.join(FIG_DIR, "fig4_metadata_flow.png"))):
        print("wrote", os.path.normpath(p))


if __name__ == "__main__":
    main()
