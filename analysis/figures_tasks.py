"""
Figures for Tasks 2.2-2.4.

Eight models is past the point where colour can carry identity, so these charts
use **composite encoding**: hue marks the detector *family* (box vs point) — the
comparison Tasks 2.3 and 2.4 actually ask about — and each line is labelled
directly at its end. No reader ever has to match eight swatches to a legend.

Trade-off charts additionally draw the Pareto frontier as a connected path, with
dominated models shown hollow: the shape of the frontier is the answer to "which
model should I deploy", and the hollow marks show what it costs to be off it.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures import (SURFACE, INK, INK_MUTED, GRID, KIND_COLOR,
                              KIND_LABEL, _style, _legend, _place_labels)
from analysis.pareto import Axis, frontier


def _series_style(kind: str) -> dict:
    return {"color": KIND_COLOR[kind], "linewidth": 2, "marker": "o",
            "markersize": 6, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def _frontier_legend(ax, kinds: list[str]) -> None:
    """Legend showing every marker style that actually appears in the plot.

    The chart encodes two things at once: hue is the detector family, fill is
    whether the model is on the Pareto frontier. Describing those as separate
    legend rows needs a neutral swatch colour for the fill entries — and a grey
    circle that appears nowhere in the data gives the reader nothing to match.
    So the legend enumerates the real combinations instead: every swatch below is
    a marker style that exists on the axes.
    """
    handles = []
    for k in ("box", "point"):
        if k not in kinds:
            continue
        family = KIND_LABEL[k].split()[0]          # "Box" / "Point"
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="", markersize=9,
            markerfacecolor=KIND_COLOR[k], markeredgecolor=KIND_COLOR[k],
            markeredgewidth=2, label=f"{family} · on the frontier"))
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="", markersize=9,
            markerfacecolor=SURFACE, markeredgecolor=KIND_COLOR[k],
            markeredgewidth=2, label=f"{family} · dominated"))

    ncol = 2 if len(handles) > 2 else max(len(handles), 1)
    ax.legend(handles=handles, frameon=False, ncol=ncol,
              loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=8.5,
              labelcolor=INK_MUTED, handletextpad=0.4, columnspacing=1.6,
              labelspacing=0.35)
    return -(-len(handles) // ncol) if handles else 0   # rows used


def _line_obstacles(ax, fig, xs, ys, step: int = 14) -> list:
    """Small display-space boxes sampled along a polyline, for label avoidance."""
    boxes = []
    pts = ax.transData.transform(list(zip(xs, ys)))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dist = max(abs(x1 - x0), abs(y1 - y0))
        n = max(int(dist // step), 1)
        for i in range(n + 1):
            t = i / n
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            boxes.append(matplotlib.transforms.Bbox.from_bounds(cx - 5, cy - 5, 10, 10))
    return boxes


def tradeoff(items: list, x_axis: Axis, y_axis: Axis, path: str) -> str | None:
    """Scatter of one accuracy/cost pair with its Pareto frontier."""
    usable = [it for it in items
              if getattr(it, x_axis.attr, None) is not None
              and getattr(it, y_axis.attr, None) is not None]
    if len(usable) < 2:
        return None

    front = frontier(usable, x_axis, y_axis)
    front_keys = {it.key for it in front}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    fig.patch.set_facecolor(SURFACE)

    fx = [getattr(it, x_axis.attr) for it in front]
    fy = [getattr(it, y_axis.attr) for it in front]
    ax.plot(fx, fy, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2,
            label="Pareto frontier")

    for it in usable:
        on_front = it.key in front_keys
        ax.plot(getattr(it, x_axis.attr), getattr(it, y_axis.attr),
                marker="o", markersize=12 if on_front else 10, linestyle="",
                markerfacecolor=KIND_COLOR[it.kind] if on_front else SURFACE,
                markeredgecolor=KIND_COLOR[it.kind],
                markeredgewidth=2, zorder=4)

    xs = [getattr(it, x_axis.attr) for it in usable]
    ys = [getattr(it, y_axis.attr) for it in usable]
    # Cost axes span orders of magnitude (20 M vs 855 M parameters); a linear
    # axis would pile seven models into one corner.
    if max(xs) / max(min(xs), 1e-9) > 20:
        ax.set_xscale("log")
    if max(ys) / max(min(ys), 1e-9) > 20:
        ax.set_yscale("log")

    # Labels sit outside the marks, so the data must not reach the frame edge.
    if ax.get_xscale() == "log":
        ax.set_xlim(min(xs) / 1.35, max(xs) * 1.35)
    else:
        pad = (max(xs) - min(xs)) * 0.22 or abs(max(xs)) * 0.1 or 1.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    if ax.get_yscale() == "log":
        ax.set_ylim(min(ys) / 1.8, max(ys) * 1.8)
    else:
        pad = (max(ys) - min(ys)) * 0.22 or abs(max(ys)) * 0.1 or 1.0
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_xlabel(x_axis.label, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(y_axis.label, fontsize=9, color=INK_MUTED)
    arrow = "higher" if x_axis.higher_is_better else "lower"
    arrow_y = "higher" if y_axis.higher_is_better else "lower"
    title = f"{x_axis.label} vs {y_axis.label}"
    ax.annotate(f"{arrow} {x_axis.label.split('(')[0].strip()} and "
                f"{arrow_y} {y_axis.label.split('(')[0].strip()} is better",
                xy=(0, -0.14), xycoords="axes fraction", fontsize=8, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    _style(ax)
    # Title pad has to clear the legend, which sits in the same strip above the
    # axes and grows a row whenever both detector families are present.
    rows = _frontier_legend(ax, [it.kind for it in usable])
    ax.set_title(title, fontsize=12, color=INK, loc="left",
                 pad=24 + 15 * max(rows - 1, 0))

    fig.tight_layout()
    fig.canvas.draw()

    # The frontier line is an obstacle: a label struck through by it is unreadable.
    obstacles = _line_obstacles(ax, fig, fx, fy)
    _place_labels(ax, fig, [(getattr(it, x_axis.attr), getattr(it, y_axis.attr), it.label)
                            for it in usable], obstacles=obstacles)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def _binned_lines(series: list[tuple[str, str, list[float | None]]],
                  bin_labels: list[str], title: str, ylabel: str, path: str,
                  *, xlabel: str, footnote: str | None = None,
                  baseline: float | None = None) -> str | None:
    """One line per model across ordered bins; hue = family, label = identity."""
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.patch.set_facecolor(SURFACE)

    xs = np.arange(len(bin_labels))
    if baseline is not None:
        ax.axhline(baseline, color=GRID, linewidth=1.5, zorder=1)

    end_points = []
    for label, kind, values in series:
        ys = [np.nan if v is None else v for v in values]
        ax.plot(xs, ys, **_series_style(kind), zorder=3)
        finite = [i for i, v in enumerate(ys) if np.isfinite(v)]
        if finite:
            last = finite[-1]
            end_points.append((xs[last], ys[last], label))

    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, fontsize=9, color=INK, rotation=0)
    ax.set_xlim(-0.4, len(bin_labels) - 1 + 2.6)  # room for end labels
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=26)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    _style(ax)
    _legend(ax, [k for _, k, _ in series])
    if footnote:
        ax.annotate(footnote, xy=(0, -0.16), xycoords="axes fraction",
                    fontsize=8, color=INK_MUTED)

    fig.tight_layout()
    fig.canvas.draw()
    _place_labels(ax, fig, end_points)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def size_curves(per_model: dict, bin_labels: list[str], out_dir: str) -> list[str]:
    """Task 2.3: recall and false-negative rate against apparent bird size."""
    made = []
    recall = [(m["label"], m["kind"], [b.recall for b in m["size"]])
              for m in per_model.values()]
    made.append(_binned_lines(
        recall, bin_labels,
        "Recall vs apparent bird size", "Recall (%)",
        os.path.join(out_dir, "size_recall.png"),
        xlabel="Apparent size — √(box area), px (equal-count bins)",
        footnote="Every model at its own operating point t*, τ = 40 px matching."))

    f1 = [(m["label"], m["kind"], [b.f1 for b in m["size"]])
          for m in per_model.values() if any(b.f1 is not None for b in m["size"])]
    if f1:
        made.append(_binned_lines(
            f1, bin_labels, "F1 vs apparent bird size (box detectors)", "F1 (%)",
            os.path.join(out_dir, "size_f1.png"),
            xlabel="Apparent size — √(box area), px",
            footnote="Point detectors emit no box extent, so a false positive cannot "
                     "be assigned a size bin and F1 is undefined for them."))
    return [p for p in made if p]


def density_curves(per_model: dict, bin_labels: list[str], out_dir: str) -> list[str]:
    """Task 2.4: detection and counting behaviour as density rises."""
    made = []
    made.append(_binned_lines(
        [(m["label"], m["kind"], [b.recall for b in m["density"]]) for m in per_model.values()],
        bin_labels, "Recall vs bird density", "Recall (%)",
        os.path.join(out_dir, "density_recall.png"),
        xlabel="Ground-truth birds per crop"))

    made.append(_binned_lines(
        [(m["label"], m["kind"], [b.signed_pct_error for b in m["density"]])
         for m in per_model.values()],
        bin_labels, "Systematic count error vs bird density",
        "Count error (%)  ·  negative = undercount",
        os.path.join(out_dir, "density_count_error.png"),
        xlabel="Ground-truth birds per crop", baseline=0.0,
        footnote="Zero line = counted exactly right. Below it the model misses birds."))

    made.append(_binned_lines(
        [(m["label"], m["kind"], [b.mae for b in m["density"]]) for m in per_model.values()],
        bin_labels, "Counting MAE vs bird density", "MAE (birds per crop)",
        os.path.join(out_dir, "density_mae.png"),
        xlabel="Ground-truth birds per crop"))
    return [p for p in made if p]
