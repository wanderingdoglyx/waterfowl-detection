"""
Task 2.2 — accuracy against cost, instead of a single accuracy ranking.

A ranking by AP answers "which model scores highest", which is rarely the
deployment question.  This module computes, for a set of (accuracy, cost) pairs,
which models are **non-dominated**: no other model is at least as good on both
axes and strictly better on one.  Those are the only models that can ever be the
right choice; everything else is beaten outright on both counts.

It also answers the operational form of the question — given a budget on memory,
latency or size, which model has the best accuracy that fits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One objective: where to read it from, and which direction is better."""

    attr: str
    label: str
    higher_is_better: bool
    fmt: str = "{:.3g}"


# The trade-offs Task 2.2 names, plus the headline accuracy-vs-latency pair.
TRADEOFFS: tuple[tuple[Axis, Axis], ...] = (
    (Axis("ap", "AP (%)", True, "{:.2f}"),
     Axis("params_m", "Parameters (M)", False, "{:.1f}")),
    (Axis("f1", "F1 (%)", True, "{:.2f}"),
     Axis("checkpoint_mb", "Checkpoint size (MB)", False, "{:.0f}")),
    (Axis("mae", "MAE (birds/crop)", False, "{:.3f}"),
     Axis("ms_per_crop", "Inference (ms/crop)", False, "{:.2f}")),
    (Axis("rmse", "RMSE (birds/crop)", False, "{:.3f}"),
     Axis("ms_per_crop", "Inference (ms/crop)", False, "{:.2f}")),
    (Axis("recall", "Recall (%)", True, "{:.2f}"),
     Axis("peak_gpu_mb", "Peak GPU memory (MB)", False, "{:.0f}")),
    (Axis("count_accuracy", "Counting accuracy (%)", True, "{:.2f}"),
     Axis("crops_per_second", "Throughput (crops/s)", True, "{:.0f}")),
)


def _oriented(value: float, axis: Axis) -> float:
    """Flip minimise-axes so dominance is always 'larger is better'."""
    return value if axis.higher_is_better else -value


def frontier(items: list, x_axis: Axis, y_axis: Axis) -> list:
    """Non-dominated items for the (x, y) objective pair.

    `items` need only expose the two attributes; anything missing either value is
    excluded rather than assumed, since a None would otherwise dominate or be
    dominated for the wrong reason.
    """
    usable = [
        it for it in items
        if getattr(it, x_axis.attr, None) is not None
        and getattr(it, y_axis.attr, None) is not None
    ]
    front = []
    for a in usable:
        ax, ay = _oriented(getattr(a, x_axis.attr), x_axis), _oriented(getattr(a, y_axis.attr), y_axis)
        dominated = False
        for b in usable:
            if b is a:
                continue
            bx, by = _oriented(getattr(b, x_axis.attr), x_axis), _oriented(getattr(b, y_axis.attr), y_axis)
            if bx >= ax and by >= ay and (bx > ax or by > ay):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda it: getattr(it, x_axis.attr))


def frontier_keys(items: list, x_axis: Axis, y_axis: Axis) -> set[str]:
    return {it.key for it in frontier(items, x_axis, y_axis)}


def best_under_budget(items: list, accuracy: Axis, cost: Axis,
                      budget: float) -> object | None:
    """Highest-accuracy model whose cost fits the budget."""
    fits = [
        it for it in items
        if getattr(it, cost.attr, None) is not None
        and getattr(it, accuracy.attr, None) is not None
        and getattr(it, cost.attr) <= budget
    ]
    if not fits:
        return None
    return max(fits, key=lambda it: _oriented(getattr(it, accuracy.attr), accuracy))
