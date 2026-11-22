"""
Rendering for the failure-case library: one crop, one or more models, ground
truth drawn against predictions.

Only the drawing lives here. Which crops are chosen, and why, is decided by
`analysis.failure_library`, which owns the selection rules and the written
record. Splitting them this way leaves exactly one definition of each failure
case and one output directory, rather than two galleries that drift apart.

A circle is an annotated bird, green when the model was credited with it and red
when it was not. A cross is a false alarm, a detection credited with no bird.
When the detection credited with a bird is not on that bird, a dashed line joins
the two and a small dot marks the credited detection.

That link is drawn because a cross inside a green circle is usually not one
bird detected twice. The protocol matches greedily by confidence within a 40 px
radius, and birds here are around 17 px across, so a higher-scoring detection on
something else nearby can claim the bird first. The detection that is actually
on the bird is then scored as the false alarm.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config
from analysis import predict
from analysis.figures import SURFACE, INK, INK_MUTED

FOUND = "#1baf7a"     # ground-truth bird the model matched
MISSED = "#e34948"    # ground-truth bird the model did not find
FALSE_POS = "#eda100" # detection matching no annotated bird (false alarm)

def _crop_path(image_id: int) -> str:
    with open(predict.TEST_JSON) as f:
        coco = json.load(f)
    for im in coco["images"]:
        if im["id"] == image_id:
            return os.path.join(config.CROPS_IMG_DIR, im["file_name"])
    raise KeyError(image_id)


def _panel(ax, img, gt, model, image_id: int, title: str) -> tuple[int, int, int]:
    """Draw one model's outcome on one crop. Returns (found, missed, false_pos)."""
    out = model["out"]
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK_MUTED); s.set_linewidth(0.8)

    # Marks overlap often, so each carries a surface-coloured halo and false
    # alarms are drawn on top.
    halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]

    g_idx = np.where(gt.gt_image_id == image_id)[0]
    xy, matched = gt.gt_xy[g_idx], out.gt_matched[g_idx]
    for (x, y), ok in zip(xy, matched):
        ax.plot(x, y, marker="o", markersize=9, linestyle="",
                markerfacecolor="none",
                markeredgecolor=FOUND if ok else MISSED, markeredgewidth=1.6,
                path_effects=halo, zorder=3)

    d = out.det_image_id == image_id
    dx, dy, tp = out_x(model, d), out_y(model, d), out.det_tp[d]

    # Where the protocol credited a bird to a detection that is not on it, draw
    # the link. Matching is greedy by confidence within the matching radius, so
    # a higher-scoring detection up to 40 px away can claim a bird before the
    # detection sitting on it; the well-placed one is then scored as a false
    # alarm. Without the link, that reads as one bird detected twice.
    pair = out.det_match[d] if len(out.det_match) else np.full(len(dx), -1)
    for x, y, j in zip(dx, dy, pair):
        if j < 0:
            continue
        bx, by = gt.gt_xy[j]
        if np.hypot(bx - x, by - y) > 0.5 * gt.gt_size[j]:
            ax.plot([bx, x], [by, y], color=FOUND, linewidth=1.3, linestyle=(0, (2, 1.5)),
                    path_effects=halo, zorder=2.5)
            ax.plot(x, y, marker="o", markersize=4.5, linestyle="",
                    markerfacecolor=FOUND, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, zorder=3.5)

    for x, y, is_tp in zip(dx, dy, tp):
        if not is_tp:
            ax.plot(x, y, marker="x", markersize=7.5, linestyle="",
                    markeredgecolor=FALSE_POS, markeredgewidth=2.0,
                    path_effects=halo, zorder=4)

    found, missed, fp = int(matched.sum()), int((~matched).sum()), int((~tp).sum())
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    ax.set_xlabel(f"{found} found · {missed} missed · {fp} false alarms",
                  fontsize=8.5, color=INK_MUTED)
    return found, missed, fp


def out_x(model, mask):
    o = model["out"]
    keep = model["cache"]["score"] >= model["t_star"]
    return model["cache"]["x"][keep][mask]


def out_y(model, mask):
    keep = model["cache"]["score"] >= model["t_star"]
    return model["cache"]["y"][keep][mask]


def render_case(gt, models: dict, keys: list[str], image_id: int,
                heading: str, path: str) -> str:
    img = Image.open(_crop_path(image_id)).convert("RGB")
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 5.9))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        _panel(ax, img, gt, models[key], image_id, models[key]["spec"].label)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor="none", markeredgecolor=FOUND,
                   markeredgewidth=1.6, label="bird found"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor="none", markeredgecolor=MISSED,
                   markeredgewidth=1.6, label="bird missed"),
        plt.Line2D([0], [0], marker="x", linestyle="", markersize=8,
                   markeredgecolor=FALSE_POS, markeredgewidth=2.0,
                   label="false alarm"),
        plt.Line2D([0], [0], marker="o", color=FOUND, linewidth=1.3,
                   linestyle=(0, (2, 1.5)), markersize=4.5, markerfacecolor=FOUND,
                   markeredgecolor=SURFACE, label="bird credited to a detection off the bird"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center",
               fontsize=9, labelcolor=INK_MUTED, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(heading, fontsize=12, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path
