"""
Comparison figures for the analysed runs.

Colour carries one thing only: whether a model is a **box** detector or a
**point** detector — two categorical hues, so a reader never has to tell eight
colours apart.  Rank is carried by position (rows are sorted), never by hue.
Everything is drawn on the light surface: these PNGs are made to be dropped into
a report document, so a single committed appearance beats a theme-aware one.

Ranked metrics use a **dot plot with its 95% CI**, not bars.  Every score here
sits in a narrow band far from zero (AP 84-93%), where bars either start at zero
and flatten the differences, or start elsewhere and overstate them.  A dot has
no baseline to lie about, and it leaves the interval — the thing that says
whether two models actually differ — legible.

Palette slots 1-2 of the reference categorical theme (blue / orange), documented
as clearing the all-pairs CVD and normal-vision floors in both modes.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.collect import RunMetrics

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"
GUIDE = "#ebeae6"
KIND_COLOR = {"box": "#2a78d6", "point": "#eb6834"}
KIND_LABEL = {"box": "Box detector", "point": "Point detector"}


def _style(ax) -> None:
    """Recessive axes: no chartjunk frame, one soft grid direction."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    # Log axes otherwise draw a ladder of unlabelled minor ticks that reads as noise.
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)


def _legend(ax, kinds: list[str]) -> None:
    """Identity is never colour-alone: two series always carry a legend.

    Anchored above the axes so it can never sit on top of a mark.
    """
    if len(set(kinds)) < 2:
        return
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=KIND_COLOR[k], markeredgecolor=KIND_COLOR[k],
                   label=KIND_LABEL[k])
        for k in ("box", "point") if k in kinds
    ]
    ax.legend(handles=handles, frameon=False, ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.01),
              fontsize=9, labelcolor=INK_MUTED, handletextpad=0.4,
              columnspacing=1.6)


def _ranked_dots(records: list[RunMetrics], attr: str, ci_attr: str | None,
                 title: str, xlabel: str, path: str, *,
                 scale: float = 1.0, higher_is_better: bool = True,
                 fmt: str = "{:.2f}") -> str | None:
    rows = [r for r in records if getattr(r, attr) is not None]
    if not rows:
        return None
    # Best first, so the winner sits at the top of the chart in both directions.
    rows.sort(key=lambda r: getattr(r, attr), reverse=higher_is_better)

    fig, ax = plt.subplots(figsize=(8, 0.48 * len(rows) + 2.0))
    fig.patch.set_facecolor(SURFACE)

    ypos = list(range(len(rows)))
    values = [getattr(r, attr) * scale for r in rows]

    # Span covers dots and intervals, so nothing is clipped.
    lo_all, hi_all = min(values), max(values)
    for r in rows:
        ci = getattr(r, ci_attr) if ci_attr else None
        if ci:
            lo_all = min(lo_all, ci[0] * scale)
            hi_all = max(hi_all, ci[1] * scale)
    pad = (hi_all - lo_all) * 0.18 or 1.0
    x0, x1 = lo_all - pad, hi_all + pad * 1.9

    for y, r, v in zip(ypos, rows, values):
        # Faint guide from the label to the dot: traces row -> mark without ink.
        ax.plot([x0, v], [y, y], color=GUIDE, linewidth=1, zorder=1)
        ci = getattr(r, ci_attr) if ci_attr else None
        if ci:
            c_lo, c_hi = ci[0] * scale, ci[1] * scale
            ax.plot([c_lo, c_hi], [y, y], color=INK_MUTED, linewidth=2,
                    solid_capstyle="butt", zorder=3)
            for x in (c_lo, c_hi):
                ax.plot([x, x], [y - 0.16, y + 0.16], color=INK_MUTED,
                        linewidth=2, zorder=3)
        # 2px surface ring keeps a dot separable where it meets its interval.
        ax.plot(v, y, marker="o", markersize=10, linestyle="",
                markerfacecolor=KIND_COLOR[r.kind], markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)

    for y, v in zip(ypos, values):
        ax.text(x1, y, fmt.format(v), va="center", ha="right",
                fontsize=9.5, color=INK, zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r.label for r in rows], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(x0, x1)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=26)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    _style(ax)
    _legend(ax, [r.kind for r in rows])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# Candidate label placements, tried in order: (dx, dy, ha, va) in points.
_LABEL_SLOTS = (
    (12, 0, "left", "center"),
    (-12, 0, "right", "center"),
    (0, 12, "center", "bottom"),
    (0, -12, "center", "top"),
    (10, 10, "left", "bottom"),
    (10, -10, "left", "top"),
    (-10, 10, "right", "bottom"),
    (-10, -10, "right", "top"),
    (14, 20, "left", "bottom"),
    (14, -20, "left", "top"),
    (-14, 20, "right", "bottom"),
    (-14, -20, "right", "top"),
    (0, 24, "center", "bottom"),
    (0, -24, "center", "top"),
    (26, 0, "left", "center"),
    (-26, 0, "right", "center"),
)


def _place_labels(ax, fig, points: list[tuple[float, float, str]],
                  obstacles: list | None = None) -> None:
    """Annotate each point, choosing the first slot that collides with nothing.

    Marks often cluster tightly, so a fixed offset guarantees overlapping text.
    Real rendered text boxes are measured rather than guessed, and a candidate
    that would spill outside the axes is rejected too — a label clipped by the
    figure edge is as unreadable as one sitting on its neighbour.

    `obstacles` are extra display-space boxes to route around.
    """
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer).expanded(1.02, 1.06)
    # Anything a label must not sit on: other labels, the marks themselves, and
    # whatever the caller adds (a frontier line, say — text struck through by a
    # line is as unreadable as text on text).
    placed = list(obstacles or [])

    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        placed.append(matplotlib.transforms.Bbox.from_bounds(px - 7, py - 7, 14, 14))

    for x, y, label in points:
        chosen = None
        for dx, dy, ha, va in _LABEL_SLOTS:
            ann = ax.annotate(label, (x, y), textcoords="offset points",
                              xytext=(dx, dy), ha=ha, va=va, fontsize=9,
                              color=INK, zorder=5)
            bbox = ann.get_window_extent(renderer).expanded(1.06, 1.20)
            inside = (bbox.x0 >= frame.x0 and bbox.x1 <= frame.x1
                      and bbox.y0 >= frame.y0 and bbox.y1 <= frame.y1)
            if inside and not any(bbox.overlaps(b) for b in placed):
                chosen = (ann, bbox)
                break
            ann.remove()
        if chosen is None:
            # Every slot was taken; keep the default so no model goes unlabelled.
            ann = ax.annotate(label, (x, y), textcoords="offset points",
                              xytext=(12, 0), ha="left", va="center",
                              fontsize=9, color=INK, zorder=5)
            chosen = (ann, ann.get_window_extent(renderer))
        placed.append(chosen[1])


def precision_recall(records: list[RunMetrics], path: str) -> str | None:
    """Where each model sits in the precision/recall trade-off at its own t*."""
    rows = [r for r in records if r.precision is not None and r.recall is not None]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    fig.patch.set_facecolor(SURFACE)

    xs = [r.recall * 100 for r in rows]
    ys = [r.precision * 100 for r in rows]
    xpad = (max(xs) - min(xs)) * 0.22 or 1.0
    ypad = (max(ys) - min(ys)) * 0.22 or 1.0
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    for r in rows:
        ax.plot(r.recall * 100, r.precision * 100, marker="o", markersize=11,
                markerfacecolor=KIND_COLOR[r.kind], markeredgecolor=SURFACE,
                markeredgewidth=2, linestyle="", zorder=4)

    ax.set_xlabel("Recall (%)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("Precision (%)", fontsize=9, color=INK_MUTED)
    ax.set_title("Precision vs recall at each model's operating point t*",
                 fontsize=12, color=INK, loc="left", pad=26)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    _style(ax)
    _legend(ax, [r.kind for r in rows])

    fig.tight_layout()
    fig.canvas.draw()  # labels need a renderer with final axes geometry
    _place_labels(ax, fig, [(r.recall * 100, r.precision * 100, r.label) for r in rows])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def render_all(records: list[RunMetrics], out_dir: str) -> list[str]:
    """Write every comparison figure; returns the paths actually produced."""
    made = [
        _ranked_dots(records, "ap", "ap_ci",
                     "Detection AP (τ = 40 px) with 95% CI", "AP (%)",
                     os.path.join(out_dir, "ap.png"), scale=100),
        _ranked_dots(records, "f1", None,
                     "F1 at each model's operating point t*", "F1 (%)",
                     os.path.join(out_dir, "f1.png"), scale=100),
        _ranked_dots(records, "mae", "mae_ci",
                     "Counting MAE with 95% CI (lower is better)",
                     "MAE (birds per image)",
                     os.path.join(out_dir, "mae.png"),
                     higher_is_better=False, fmt="{:.3f}"),
        precision_recall(records, os.path.join(out_dir, "precision_recall.png")),
    ]
    return [p for p in made if p]
