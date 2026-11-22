"""
Figures for the survey-condition analysis.

Style is inherited wholesale from `analysis.figures` — same surface, same ink,
same two categorical hues carrying the one thing colour is allowed to carry here
(box detector vs point detector). A reader moving between the benchmark and
survey-condition
figures should not have to relearn the encoding.

Form follows the question, and two of these questions are about a *relationship*
rather than a ranking, so they are not dot plots:

* **Habitat** is a ranking with a spread — habitats ordered by difficulty, every
  model drawn, so both "which habitat is hard" and "do the models agree" are
  visible in one pass.
* **The altitude chain** is three measures on one x axis. They have different
  units, so they are three stacked panels sharing that axis rather than one
  chart with two y scales — a dual axis would invite the reader to see a
  crossing point that means nothing.
* **The training-share confound** is a relationship between two continuous
  quantities, so it is a scatter with the altitudes labelled directly.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import metadata, subgroups
from analysis.figures import (GRID, INK, INK_MUTED, KIND_COLOR, KIND_LABEL,
                              SURFACE, _style)
from analysis.survey_conditions import LADDER, _median_size, _train_altitude_share


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _kind_legend(ax, kinds, loc="lower left", anchor=(0, 1.01)):
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=KIND_COLOR[k], markeredgecolor=SURFACE,
                   markeredgewidth=1.2, label=KIND_LABEL[k])
        for k in ("box", "point") if k in kinds
    ]
    if len(handles) >= 2:
        ax.legend(handles=handles, frameon=False, ncol=2, loc=loc,
                  bbox_to_anchor=anchor, fontsize=9, labelcolor=INK_MUTED,
                  handletextpad=0.4, columnspacing=1.6)


def _model_order(ctx) -> list[str]:
    """Fixed model order, so a colour always denotes the same model."""
    return [r.key for r in ctx.runs]


def _line_colours(ctx) -> dict[str, str]:
    """One hue per model, for line charts only.

    The reference palette's eight-slot order is validated on the *adjacent*
    pairlist, which covers stacks, bars and lines. It is explicitly NOT
    validated for scatter and dot plots, where every pair of series can end up
    side by side: past the third slot yellow and orange appear together and fail
    both the CVD and the normal-vision floors. Per-model identity is therefore
    carried by hue on the line charts here and by faceting on the dot plots.
    """
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    return {k: slots[i % len(slots)] for i, k in enumerate(_model_order(ctx))}


def _model_legend(fig, ctx, colours: dict[str, str], ncol: int = 4) -> None:
    """Figure-level legend naming every model, placed under the plot."""
    label = {r.key: r.label for r in ctx.runs}
    handles = [plt.Line2D([0], [0], color=colours[k], linewidth=2.4,
                          marker="o", markersize=6, markeredgecolor=SURFACE,
                          markeredgewidth=0.8, label=label[k])
               for k in _model_order(ctx)]
    fig.legend(handles=handles, frameon=False, ncol=ncol, loc="lower center",
               fontsize=9, labelcolor=INK_MUTED, handletextpad=0.5,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.015))


def habitat_by_model(ctx, per_model: dict, path: str) -> str | None:
    """One panel per model, so every mark is attributable to a named model.

    The single-panel overview shows which habitats are hard but cannot say which
    dot belongs to which model: eight categorical hues do not survive the
    all-pairs colour-separation floors on a dot plot. Faceting is the remedy the
    palette documentation prescribes, and it makes identity exact rather than
    merely distinguishable.
    """
    kinds = {r.key: r.kind for r in ctx.runs}
    label = {r.key: r.label for r in ctx.runs}
    order = _model_order(ctx)

    per_group: dict[str, dict[str, float]] = {}
    for key, scores in per_model.items():
        for s in scores:
            if s.sufficient and s.f1 is not None:
                per_group.setdefault(s.group, {})[key] = s.f1
    if not per_group:
        return None

    # Shared row order across every panel: the mean over models, hardest last,
    # matching the overview figure so the two can be read together.
    groups = sorted(per_group, key=lambda g: np.mean(list(per_group[g].values())))
    lo = min(v for d in per_group.values() for v in d.values())
    hi = max(v for d in per_group.values() for v in d.values())
    pad = max(1.0, 0.05 * (hi - lo))

    cols = 4
    rows_n = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, squeeze=False, sharex=True, sharey=True,
                             figsize=(3.3 * cols, 0.30 * len(groups) * rows_n + 1.6))
    fig.patch.set_facecolor(SURFACE)

    for i, key in enumerate(order):
        ax = axes[i // cols][i % cols]
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ys, xs = [], []
        for y, g in enumerate(groups):
            if key in per_group[g]:
                ys.append(y)
                xs.append(per_group[g][key])
        ax.plot(xs, ys, marker="o", markersize=6, linestyle="",
                markerfacecolor=KIND_COLOR[kinds[key]], markeredgecolor=SURFACE,
                markeredgewidth=1.0)
        ax.set_title(label[key], fontsize=10, color=INK, loc="left", pad=4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(range(len(groups)))
        if i % cols == 0:
            ax.set_yticklabels(groups, fontsize=8, color=INK)
        if i // cols == rows_n - 1:
            ax.set_xlabel("F1 (%)", fontsize=8.5, color=INK_MUTED)
    for j in range(len(order), rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Habitat difficulty by model. Panels share one scale and one "
                 "row order, hardest habitat at the bottom.",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def habitat_ranking(ctx, per_model: dict, path: str) -> str | None:
    """Habitats ordered by difficulty, every model shown.

    A dot per model per habitat: the row position answers "how hard is this
    habitat", the spread within a row answers "do the models agree about it".
    """
    kinds = {r.key: r.kind for r in ctx.runs}
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    for key, scores in per_model.items():
        for s in scores:
            if not s.sufficient or s.f1 is None:
                continue
            found = next((r for r in rows if r[0] == s.group), None)
            if found is None:
                rows.append((s.group, [(key, s.f1)]))
            else:
                found[1].append((key, s.f1))
    if not rows:
        return None

    rows.sort(key=lambda r: np.mean([v for _, v in r[1]]))
    labels = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(rows) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7)

    for y, (_, vals) in enumerate(rows):
        v = [x for _, x in vals]
        # A connecting rule makes the spread readable without implying an order
        # among the models sitting on it.
        ax.plot([min(v), max(v)], [y, y], color=GRID, linewidth=2,
                solid_capstyle="round", zorder=1)
        for key, val in vals:
            ax.plot(val, y, marker="o", markersize=8, linestyle="",
                    markerfacecolor=KIND_COLOR[kinds[key]],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlabel("F1 (%) at each model's global operating threshold",
                  fontsize=9.5, color=INK_MUTED)
    ax.set_title("Habitat difficulty — every model, hardest at the bottom",
                 fontsize=12, color=INK, loc="left", pad=26)
    _kind_legend(ax, set(kinds.values()))
    return _save(fig, path)


def altitude_chain(ctx, per_model: dict, path: str) -> str | None:
    """The requested chain, three panels on one shared altitude axis.

    Apparent size, recall and count bias have three different units. Stacking
    them keeps the x alignment that makes the chain legible while refusing the
    dual y axis that would fake a crossing point.
    """
    groups = metadata.group_by(ctx.metas, "altitude_nominal")
    ladder = [g for g in sorted(groups) if g in LADDER]
    if len(ladder) < 3:
        return None

    colours = _line_colours(ctx)
    by_model = {r.key: {s.group: s for s in per_model[r.key]} for r in ctx.runs}
    size = [_median_size(ctx, groups[g]) for g in ladder]

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.8), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(ladder))

    # Panel 1 is a property of the imagery rather than of any model, so it
    # carries one neutral series and its axis label says so.
    _style(axes[0])
    axes[0].grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].plot(x, size, marker="o", markersize=8, linewidth=2,
                 color=INK_MUTED, markeredgecolor=SURFACE, markeredgewidth=1.2)
    axes[0].set_ylabel("Median apparent bird size\n(px, identical for all models)",
                       fontsize=9, color=INK_MUTED)
    for xi, v in zip(x, size):
        if v is not None:
            axes[0].annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=8.5,
                             color=INK_MUTED)

    # Panels 2 and 3 carry one line per model, which is what lets the figure be
    # read for an individual model instead of only for the average.
    for ax, attr, ylabel in ((axes[1], "recall", "Recall (%)"),
                             (axes[2], "count_bias", "Count bias (%)")):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        for key in _model_order(ctx):
            ys = [getattr(by_model[key].get(g), attr, None) for g in ladder]
            ax.plot(x, [np.nan if v is None else v for v in ys], marker="o",
                    markersize=5.5, linewidth=1.9, color=colours[key],
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK_MUTED)
    axes[2].axhline(0, color=GRID, linewidth=1, zorder=0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{g} m" for g in ladder], fontsize=9.5, color=INK)
    axes[2].set_xlabel("Nominal flight altitude", fontsize=9.5, color=INK_MUTED)
    axes[0].set_title("Altitude, apparent size, recall and counting error",
                      fontsize=12, color=INK, loc="left", pad=12)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    _model_legend(fig, ctx, colours)
    return _save(fig, path)


def altitude_training_confound(ctx, per_model: dict, path: str) -> str | None:
    """Recall against how much training data each altitude actually had.

    The finding this figure exists to make unavoidable: recall tracks the
    training split's altitude mix, so the apparent "higher is easier" result is
    about sampling, not about height.
    """
    share = _train_altitude_share()
    if not share:
        return None
    groups = metadata.group_by(ctx.metas, "altitude_nominal")
    pts = []
    for g in sorted(groups):
        if g not in LADDER or g not in share:
            continue
        rs = [s.recall for ss in per_model.values()
              for s in ss if s.group == g and s.recall is not None]
        if rs:
            pts.append((g, share[g], float(np.mean(rs))))
    if len(pts) < 3:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(color=GRID, linewidth=0.7)

    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.plot(xs, ys, marker="o", markersize=10, linestyle="",
            markerfacecolor=KIND_COLOR["box"], markeredgecolor=SURFACE,
            markeredgewidth=1.4, zorder=3)
    for g, sh, rec in pts:
        ax.annotate(f"{g} m", (sh, rec), textcoords="offset points",
                    xytext=(9, -4), fontsize=9.5, color=INK)

    if len(pts) >= 3:
        m, b = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 50)
        ax.plot(gx, m * gx + b, color=INK_MUTED, linewidth=1.4,
                linestyle="--", zorder=1)
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.annotate(f"r = {r:+.2f}", (0.97, 0.06), xycoords="axes fraction",
                    ha="right", fontsize=10, color=INK_MUTED)

    ax.set_xlabel("Share of training birds flown at this altitude (%)",
                  fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("Mean test recall (%), averaged over all 8 models",
                  fontsize=9, color=INK_MUTED)
    ax.set_title("Recall follows the training mix, not the flight height",
                 fontsize=12, color=INK, loc="left", pad=12)
    return _save(fig, path)


def altitude_within_dataset(ctx, path: str) -> str | None:
    """Altitude effect inside each dataset that varies it.

    Small multiples rather than one crowded axis: the datasets sit at very
    different recall levels, and overlaying them would read as a ranking of
    datasets instead of six independent within-dataset trends.
    """
    varying = metadata.datasets_varying(ctx.metas, "altitude_nominal")
    series = []
    for ds in varying:
        groups = metadata.group_by(ctx.metas, "altitude_nominal", datasets={ds})
        pts = []
        for g in sorted(groups):
            ids = groups[g]
            if (sum(ctx.gt.counts[i] for i in ids) < subgroups.MIN_BIRDS
                    or metadata.source_image_count(ctx.metas, ids) < subgroups.MIN_IMAGES):
                continue
            vals = [s.recall for s in
                    (subgroups.score(r, ids, g,
                                     metadata.source_image_count(ctx.metas, ids))
                     for r in ctx.runs) if s.recall is not None]
            if vals:
                pts.append((g, float(np.mean(vals))))
        if len(pts) >= 2:
            series.append((ds, pts))
    if not series:
        return None

    n = len(series)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    colours = _line_colours(ctx)
    for i, (ds, pts) in enumerate(series):
        ax = axes[i // cols][i % cols]
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        heights = [p[0] for p in pts]
        groups = metadata.group_by(ctx.metas, "altitude_nominal", datasets={ds})
        for key in _model_order(ctx):
            run = next(r for r in ctx.runs if r.key == key)
            ys = []
            for g in heights:
                ids = groups[g]
                sc = subgroups.score(run, ids, g,
                                     metadata.source_image_count(ctx.metas, ids))
                ys.append(sc.recall if sc.recall is not None else np.nan)
            ax.plot(heights, ys, marker="o", markersize=5, linewidth=1.7,
                    color=colours[key], markeredgecolor=SURFACE,
                    markeredgewidth=0.7)
        ax.set_title(ds, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Altitude (m)", fontsize=8.5, color=INK_MUTED)
        ax.set_ylabel("Recall (%)", fontsize=8.5, color=INK_MUTED)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Altitude effect measured inside each dataset, with site, camera "
                 "and season held fixed",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    _model_legend(fig, ctx, colours)
    return _save(fig, path)


def error_sources(ctx, tables: dict, path: str) -> str | None:
    """How much F1 each survey factor actually moves — the headline result.

    Uses `survey_conditions.summary_factors` rather than recomputing, so this figure and
    error_sources.csv are guaranteed to show the same factors and the same numbers.
    """
    from analysis.survey_conditions import summary_factors

    ranked = summary_factors(ctx, tables)
    if len(ranked) < 2:
        return None
    factors = [(name, val) for name, val, _ in ranked][::-1]  # smallest at bottom

    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(factors) + 1.8))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    for y, (name, val) in enumerate(factors):
        ax.plot([0, val], [y, y], color=GRID, linewidth=2,
                solid_capstyle="round", zorder=1)
        ax.plot(val, y, marker="o", markersize=9, linestyle="",
                markerfacecolor=KIND_COLOR["box"], markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=2)
        ax.annotate(f"{val:.1f}", (val, y), textcoords="offset points",
                    xytext=(10, -3), fontsize=9, color=INK_MUTED)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([f[0] for f in factors], fontsize=10, color=INK)
    ax.set_xlabel("Mean F1 spread across that factor's subgroups (points), "
                  "averaged over all 8 models", fontsize=9, color=INK_MUTED)
    ax.set_title("What actually moves accuracy", fontsize=12, color=INK,
                 loc="left", pad=12)
    ax.set_xlim(left=0)
    return _save(fig, path)


def render_all(ctx, tables: dict, out_dir: str) -> list[str]:
    made = []
    if "habitat" in tables:
        for fn, name in ((habitat_ranking, "habitat_f1.png"),
                         (habitat_by_model, "habitat_f1_by_model.png")):
            p = fn(ctx, tables["habitat"], os.path.join(out_dir, name))
            if p:
                made.append(p)
    if "altitude" in tables:
        for fn, name in ((altitude_chain, "altitude_chain.png"),
                         (altitude_training_confound, "altitude_training_confound.png")):
            p = fn(ctx, tables["altitude"], os.path.join(out_dir, name))
            if p:
                made.append(p)
    p = altitude_within_dataset(ctx, os.path.join(out_dir, "altitude_within_dataset.png"))
    if p:
        made.append(p)
    p = error_sources(ctx, tables, os.path.join(out_dir, "error_sources.png"))
    if p:
        made.append(p)
    return made
