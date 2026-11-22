"""
The reviewer: a full-screen, keyboard-first way to check a detection result cell by cell.

    ┌ title: survey · model run · photo n/N · cell k/K ───────────────── saved ┐
    │                                           │  loupe: the selected bird,    │
    │   the cell (the part of one 512 px tile   │  magnified                    │
    │   this tile owns), with a 16 px margin    ├───────────────────────────────┤
    │   of its neighbours for context           │  counts: cell / photo / survey│
    │                                           │  model → reviewed             │
    │   amber ring  counted by the model        ├───────────────────────────────┤
    │   green ring  confirmed bird              │  this cell's detections,      │
    │   red ring    rejected                    │  least confident first        │
    │   grey ring   below the threshold         │                               │
    │   cyan ring   added by hand               │                               │
    │   magenta     annotated (labelled surveys)│                               │
    └ keys ───────────────────────────────────────────────────────────────────────┘

Least-confident detections come first because that is where a model's mistakes are;
Enter marks the cell reviewed and moves on, so a cell the model got right costs one key.
A missed bird is added by clicking on it, or with `a` and the arrow keys for a precise
position.  Every decision is written to review.json immediately — closing the terminal
loses nothing — and `u` undoes.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (ConditionalContainer, Float, FloatContainer, HSplit,
                                   Layout, VSplit, Window)
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from workbench import detections, imaging, paths
from workbench.detections import BIRD, NOT_BIRD

MARGIN = 16                                   # px of neighbouring context around a cell
LOUPE_SPANS = (24, 32, 48, 64, 96, 128, 192)  # loupe width in photograph pixels
RIGHT_WIDTH = 46
BAND = 0.15                                   # "borderline": within this of the threshold
DEFAULT_BIRD_PX = 17.0                        # typical bird size, for point detectors

COLORS = {
    "model": (255, 176, 0), BIRD: (46, 204, 113), NOT_BIRD: (235, 64, 52),
    "below": (150, 150, 150), "added": (0, 210, 255), "gt": (255, 102, 255),
    "selected": (255, 255, 255), "cursor": (255, 255, 0), "neighbour": (110, 110, 110),
    "cell": (90, 90, 90),
}
GLYPH = {"model": ("●", "#ffb000"), BIRD: ("✓", "#2ecc71"), NOT_BIRD: ("✗", "#eb4034"),
         "below": ("○", "#969696"), "added": ("+", "#00d2ff")}
WORDS = {"model": "counted, not reviewed", BIRD: "bird", NOT_BIRD: "not a bird",
         "below": "below threshold", "added": "added by hand"}

STYLE = Style.from_dict({
    "title": "bg:#1f2937 #e5e7eb", "title.accent": "bg:#1f2937 #7dd3fc bold",
    "bar": "bg:#111827 #9ca3af", "bar.key": "bg:#111827 #f9fafb bold",
    "bar.msg": "bg:#111827 #fde68a", "dim": "#8a8a8a", "head": "bold",
    "sel": "reverse", "good": "#2ecc71", "warn": "#fde68a",
})

def _help_text() -> str:
    left = [("MOVING", ""), ("↑ ↓  k j", "select a detection"),
            ("Enter", "cell done, go to the next"), ("→ ←  PgDn PgUp", "next / previous cell"),
            ("] [", "next / previous photo"), ("", ""), ("SEEING", ""),
            ("+ -  wheel", "zoom the loupe"), ("b", "detections below threshold"),
            ("g", "annotations (labelled surveys)"), ("o", "live full-resolution preview"),
            ("< >  T", "threshold −/+ 0.05, T resets")]
    right = [("DECIDING", ""), ("y", "it is a bird"), ("n", "it is not a bird"),
             ("c", "clear the decision"), ("x  Del", "delete a bird you added"),
             ("a", "add a missed bird: arrows move"), ("", "the cross (Shift: faster),"),
             ("", "Enter places, Esc cancels"), ("click", "select a marker, or add a"),
             ("", "bird where there is none"), ("u", "undo"), ("q  Esc", "save and leave")]

    def column(key: str, text: str) -> str:
        if key and not text:
            return f" {key:<43}"
        return f"   {key:<15}{text:<27}"

    lines = [column(*a) + column(*b) for a, b in zip(left, right)]
    lines += ["",
              " A detection nobody decides keeps the model's verdict: it counts when its score",
              " is at or above the threshold.  Every decision is saved the moment it is made.",
              "",
              f"{'? closes this help':>88}"]
    return "\n".join(lines)


HELP = _help_text()


@dataclass
class Item:
    kind: str          # "det" or "added"
    index: int         # detection index in the photo, or index in the added list
    x: float
    y: float
    w: float
    h: float
    score: float | None


class ImageControl(UIControl):
    """Draws what `render(width, height)` returns; reports clicks and wheel turns."""

    def __init__(self, render, on_mouse):
        self.render = render
        self.on_mouse = on_mouse

    def create_content(self, width: int, height: int) -> UIContent:
        rows = self.render(width, height)
        return UIContent(get_line=lambda i: rows[i] if i < len(rows) else [],
                         line_count=len(rows))

    def mouse_handler(self, mouse_event):
        kind = mouse_event.event_type
        if kind in (MouseEventType.MOUSE_UP, MouseEventType.SCROLL_UP,
                    MouseEventType.SCROLL_DOWN):
            self.on_mouse(kind, mouse_event.position.x, mouse_event.position.y)
            return None
        return NotImplemented

    def is_focusable(self) -> bool:
        return False


class Reviewer:
    def __init__(self, result: detections.Result, mode: str = "detections"):
        self.result = result
        self.review = result.review()
        self.survey = result.survey()
        self.photos = result.photos()
        self.t_star = result.threshold
        self.mode = mode
        self.show_below = False
        self.show_gt = bool(self.survey and self.survey.labelled)
        self.span_i = 2
        self.adding = False
        self.cursor = (0.0, 0.0)
        self.sel = 0
        self.message = ""
        self.help = False
        self.live = False
        self.history: list[tuple] = []
        self._images: OrderedDict = OrderedDict()
        self._regions: OrderedDict = OrderedDict()
        self._assign: dict[int, np.ndarray] = {}
        self._cells: dict[int, list] = {}
        self._gt: dict[int, np.ndarray | None] = {}
        self._totals = None
        self.geometry: dict[str, tuple] = {}
        self.queue = self._build_queue()
        self.qpos = self._first_open()

    # ── Where we are ─────────────────────────────────────────────────────────
    @property
    def pi(self) -> int:
        return self.queue[self.qpos][0]

    @property
    def photo(self) -> detections.PhotoResult:
        return self.photos[self.pi]

    @property
    def cell(self) -> detections.Cell:
        return self.cells_of(self.pi)[self.queue[self.qpos][1]]

    def cells_of(self, pi: int) -> list:
        if pi not in self._cells:
            p = self.photos[pi]
            self._cells[pi] = detections.cells(p.width, p.height)
        return self._cells[pi]

    def assignment(self, pi: int) -> np.ndarray:
        """The cell each detection of photo `pi` belongs to."""
        if pi not in self._assign:
            p = self.photos[pi]
            xc, yc = np.asarray(detections.cuts(p.width)), np.asarray(detections.cuts(p.height))
            col = np.searchsorted(xc, p.xy[:, 0], side="right")
            row = np.searchsorted(yc, p.xy[:, 1], side="right")
            self._assign[pi] = row * (len(xc) + 1) + col
        return self._assign[pi]

    def gt(self, pi: int) -> np.ndarray | None:
        """Annotated boxes [G, 4] for photo `pi`, or None for an unlabelled survey."""
        if pi not in self._gt:
            boxes = None
            if self.survey:
                ph = self.survey.photo(self.photos[pi].file)
                found = self.survey.boxes(ph) if ph else None
                boxes = None if found is None else np.asarray(found, float).reshape(-1, 4)
            self._gt[pi] = boxes
        return self._gt[pi]

    # ── The queue of cells to visit ──────────────────────────────────────────
    def _build_queue(self) -> list[tuple[int, int]]:
        t = self.review.threshold
        queue, weight = [], []
        for pi, p in enumerate(self.photos):
            cell_of = self.assignment(pi)
            n = len(self.cells_of(pi))
            above = np.bincount(cell_of[p.score >= t], minlength=n)
            near = np.bincount(cell_of[np.abs(p.score - t) <= BAND], minlength=n)
            marked = [i for i in self.review.marks(p.file) if i < len(cell_of)]
            touched = set(cell_of[marked].tolist()) if marked else set()
            for a in self.review.added(p.file):
                touched.add(detections.cell_index(a[0], a[1], p.width, p.height))
            done = self.review.cells_done(p.file)
            for c in range(n):
                has = above[c] > 0 or c in touched
                keep = {"every": True, "detections": has, "uncertain": near[c] > 0,
                        "unreviewed": has and c not in done}.get(self.mode, has)
                if keep:
                    queue.append((pi, c))
                    weight.append(near[c])
        if self.mode == "uncertain":
            order = np.argsort(-np.asarray(weight), kind="stable")
            queue = [queue[i] for i in order]
        return queue

    def _first_open(self) -> int:
        for q, (pi, c) in enumerate(self.queue):
            if c not in self.review.cells_done(self.photos[pi].file):
                return q
        return 0

    # ── Items in the current cell ────────────────────────────────────────────
    def items(self) -> list[Item]:
        p, ci, t = self.photo, self.cell.index, self.review.threshold
        marks = self.review.marks(p.file)
        out = []
        for i in np.where(self.assignment(self.pi) == ci)[0]:
            s = float(p.score[i])
            if s >= t or self.show_below or int(i) in marks:
                out.append(Item("det", int(i), float(p.xy[i, 0]), float(p.xy[i, 1]),
                                float(p.wh[i, 0]), float(p.wh[i, 1]), s))
        out.sort(key=lambda it: it.score)
        for k, (x, y) in enumerate(self.review.added(p.file)):
            if detections.cell_index(x, y, p.width, p.height) == ci:
                out.append(Item("added", k, x, y, float("nan"), float("nan"), None))
        return out

    def status(self, it: Item) -> str:
        if it.kind == "added":
            return "added"
        mark = self.review.mark(self.photo.file, it.index)
        if mark:
            return mark
        return "model" if it.score >= self.review.threshold else "below"

    def selected(self) -> Item | None:
        items = self.items()
        if not items:
            return None
        self.sel = max(0, min(self.sel, len(items) - 1))
        return items[self.sel]

    # ── Changing things ──────────────────────────────────────────────────────
    def _changed(self, message: str) -> None:
        self.review.save()
        self._totals = None
        self.message = message
        self._write_live()

    def decide(self, value: str | None) -> None:
        it = self.selected()
        if it is None:
            self.message = "no detection selected"
            return
        if it.kind == "added":
            if value == NOT_BIRD:
                self.delete_added()
            else:
                self.message = "added birds are birds — x deletes one"
            return
        key = self.photo.file
        before = self.review.mark(key, it.index)
        if before == value:
            self._advance()
            return
        self.review.set_mark(key, it.index, value)
        self.history.append(("mark", key, it.index, before))
        self._changed({BIRD: "bird", NOT_BIRD: "not a bird", None: "decision cleared"}[value])
        if value is not None:
            self._advance()

    def _advance(self) -> None:
        """Move to the next detection nobody has decided yet (wrapping round the cell)."""
        items = self.items()
        for step in range(1, len(items) + 1):
            j = (self.sel + step) % len(items)
            if self.status(items[j]) in ("model", "below"):
                self.sel = j
                return
        self.message += " · every detection here is decided — Enter for the next cell"

    def add_at(self, x: float, y: float) -> None:
        p = self.photo
        x, y = float(np.clip(x, 0, p.width - 1)), float(np.clip(y, 0, p.height - 1))
        k = self.review.add(p.file, x, y)
        self.history.append(("add", p.file, k))
        note = ""
        if detections.cell_index(x, y, p.width, p.height) != self.cell.index:
            note = " (in the neighbouring cell — it is counted there)"
        self._changed("added a bird" + note)
        items = self.items()
        for j, it in enumerate(items):
            if it.kind == "added" and it.index == k:
                self.sel = j

    def delete_added(self) -> None:
        it = self.selected()
        if it is None or it.kind != "added":
            self.message = "select a bird you added (cyan) to delete it"
            return
        point = self.review.remove_added(self.photo.file, it.index)
        self.history.append(("remove", self.photo.file, it.index, point))
        self._changed("deleted an added bird")

    def cell_done(self) -> None:
        key, ci = self.photo.file, self.cell.index
        before = ci in self.review.cells_done(key)
        if not before:
            self.review.set_cell_done(key, ci, True)
            self.history.append(("done", key, ci, before))
            self._changed("cell reviewed")
        if self.qpos + 1 < len(self.queue):
            self.goto(self.qpos + 1)
        else:
            left = sum(1 for pi, c in self.queue
                       if c not in self.review.cells_done(self.photos[pi].file))
            self.message = ("end of the queue — every cell reviewed" if not left
                            else f"end of the queue — {left} cell(s) still open")

    def undo(self) -> None:
        if not self.history:
            self.message = "nothing to undo"
            return
        op = self.history.pop()
        if op[0] == "mark":
            self.review.set_mark(op[1], op[2], op[3])
        elif op[0] == "add":
            self.review.remove_added(op[1], op[2])
        elif op[0] == "remove":
            self.review.insert_added(op[1], op[2], op[3])
        elif op[0] == "done":
            self.review.set_cell_done(op[1], op[2], op[3])
        elif op[0] == "threshold":
            self.review.threshold = op[1]
        self._changed("undone")

    def set_threshold(self, value: float) -> None:
        value = round(float(np.clip(value, 0.05, 0.95)), 3)
        self.history.append(("threshold", self.review.threshold))
        self.review.threshold = value
        self._changed(f"threshold {value:.2f} — counts follow it")

    def goto(self, qpos: int) -> None:
        self.qpos = qpos % len(self.queue)
        self.sel = 0
        self.adding = False
        self._write_live()

    def jump_photo(self, step: int) -> None:
        here = self.pi
        q = self.qpos
        for _ in range(len(self.queue)):
            q = (q + step) % len(self.queue)
            if self.queue[q][0] != here:
                if step < 0:     # land on the first cell of that photograph
                    while q > 0 and self.queue[q - 1][0] == self.queue[q][0]:
                        q -= 1
                self.goto(q)
                return
        self.message = "this is the only photograph"

    # ── Counts ───────────────────────────────────────────────────────────────
    def totals(self) -> tuple[int, int, int, int]:
        if self._totals is None:
            model = final = done = cells = 0
            for p in self.photos:
                c = self.review.counts(p)
                model, final = model + c.model, final + c.final
                done, cells = done + c.cells_done, cells + c.cells_total
            self._totals = (model, final, done, cells)
        return self._totals

    def cell_counts(self) -> tuple[int, int]:
        p, ci, t = self.photo, self.cell.index, self.review.threshold
        idx = np.where(self.assignment(self.pi) == ci)[0]
        model = int((p.score[idx] >= t).sum())
        final = sum(1 for i in idx if self.review.is_bird(p, int(i)))
        final += sum(1 for a in self.review.added(p.file)
                     if detections.cell_index(a[0], a[1], p.width, p.height) == ci)
        return model, final

    # ── Pixels ───────────────────────────────────────────────────────────────
    def image(self, pi: int) -> Image.Image:
        if pi not in self._images:
            path = self.survey.path(self.survey.photo(self.photos[pi].file))
            with Image.open(path) as im:
                self._images[pi] = im.convert("RGB")
            while len(self._images) > 2:
                self._images.popitem(last=False)
        self._images.move_to_end(pi)
        return self._images[pi]

    def _region(self, box: tuple, size: tuple) -> Image.Image:
        key = (self.pi, box, size)
        if key not in self._regions:
            crop = self.image(self.pi).crop(tuple(int(round(v)) for v in box))
            self._regions[key] = crop.resize(size, Image.LANCZOS)
            while len(self._regions) > 12:
                self._regions.popitem(last=False)
        return self._regions[key]

    def _radius(self, it: Item, scale: float, minimum: float) -> float:
        size = it.w if not np.isnan(it.w) else DEFAULT_BIRD_PX
        return max(minimum, 0.5 * size * scale + 1)

    def _draw_items(self, draw: ImageDraw.ImageDraw, to_px, scale: float, box: tuple,
                    minimum: float) -> None:
        p = self.photo
        x0, y0, x1, y1 = box
        t = self.review.threshold
        # neighbours' counted detections, for context
        own = self.assignment(self.pi) == self.cell.index
        inside = ((p.xy[:, 0] >= x0) & (p.xy[:, 0] < x1) & (p.xy[:, 1] >= y0) & (p.xy[:, 1] < y1))
        for i in np.where(inside & ~own & (p.score >= t))[0]:
            px, py = to_px(p.xy[i, 0], p.xy[i, 1])
            r = max(minimum - 1, 1)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=COLORS["neighbour"])
        gt = self.gt(self.pi) if self.show_gt else None
        if gt is not None and len(gt):
            for bx1, by1, bx2, by2 in gt:
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if x0 <= cx < x1 and y0 <= cy < y1:
                    px, py = to_px(cx, cy)
                    draw.point([(px, py)], fill=COLORS["gt"])
                    if scale >= 0.5:
                        a, b = to_px(bx1, by1), to_px(bx2, by2)
                        draw.rectangle([a, b], outline=COLORS["gt"])
        chosen = self.selected()
        for it in self.items():
            px, py = to_px(it.x, it.y)
            r = self._radius(it, scale, minimum)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=COLORS[self.status(it)],
                         width=2 if scale >= 1 else 1)
            if chosen is not None and it.kind == chosen.kind and it.index == chosen.index:
                rr = r + (3 if scale >= 1 else 2)
                draw.ellipse([px - rr, py - rr, px + rr, py + rr], outline=COLORS["selected"])
        if self.adding:
            cx, cy = to_px(*self.cursor)
            arm = max(3, 6 * min(scale, 1.5))
            draw.line([cx - arm, cy, cx + arm, cy], fill=COLORS["cursor"])
            draw.line([cx, cy - arm, cx, cy + arm], fill=COLORS["cursor"])

    def _frame(self, img: Image.Image, width: int, height: int, name: str,
               origin: tuple, scale: float) -> list:
        rows = imaging.to_rows(img)
        pad_x = max(0, (width - img.width) // 2)
        pad_y = max(0, (height - len(rows)) // 2)
        self.geometry[name] = (origin[0], origin[1], scale, pad_x, pad_y)
        blank = [("", " " * width)]
        lines = [blank] * pad_y
        lines += [[("", " " * pad_x)] + r + [("", " " * max(0, width - pad_x - img.width))]
                  for r in rows]
        lines += [blank] * max(0, height - len(lines))
        return lines[:height]

    def render_main(self, width: int, height: int) -> list:
        if width < 20 or height < 6:
            return [[("class:dim", " enlarge the terminal")]]
        p, c = self.photo, self.cell
        box = (max(0.0, c.x0 - MARGIN), max(0.0, c.y0 - MARGIN),
               min(float(p.width), c.x1 + MARGIN), min(float(p.height), c.y1 + MARGIN))
        pw, ph, scale = imaging.fit(box[2] - box[0], box[3] - box[1], width, height)
        img = self._region(box, (pw, ph)).copy()
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            return (x - box[0]) * scale, (y - box[1]) * scale

        a, b = to_px(c.x0, c.y0), to_px(c.x1, c.y1)
        draw.rectangle([a[0], a[1], b[0] - 1, b[1] - 1], outline=COLORS["cell"])
        self._draw_items(draw, to_px, scale, box, minimum=2.0)
        return self._frame(img, width, height, "main", box[:2], scale)

    def loupe_centre(self) -> tuple[float, float]:
        if self.adding:
            return self.cursor
        it = self.selected()
        if it is not None:
            return it.x, it.y
        c = self.cell
        return (c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2

    def render_loupe(self, width: int, height: int) -> list:
        span = LOUPE_SPANS[self.span_i]
        cx, cy = self.loupe_centre()
        box = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)
        pw, ph, scale = imaging.fit(span, span, width, height)
        crop = self.image(self.pi).crop(tuple(int(round(v)) for v in box))
        img = crop.resize((pw, ph), Image.NEAREST if scale >= 1.5 else Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            return (x - box[0]) * scale, (y - box[1]) * scale

        self._draw_items(draw, to_px, scale, box, minimum=3.0)
        return self._frame(img, width, height, "loupe", box[:2], scale)

    def on_mouse(self, pane: str, kind, x: int, y: int) -> None:
        if kind in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
            step = -1 if kind == MouseEventType.SCROLL_UP else 1
            self.span_i = int(np.clip(self.span_i + step, 0, len(LOUPE_SPANS) - 1))
            return
        if pane not in self.geometry:
            return
        ox, oy, scale, pad_x, pad_y = self.geometry[pane]
        X = ox + (x - pad_x + 0.5) / scale
        Y = oy + ((y - pad_y) * 2 + 1) / scale
        p = self.photo
        if not (0 <= X < p.width and 0 <= Y < p.height):
            return
        if self.adding:
            self.adding = False
            self.add_at(X, Y)
            return
        items = self.items()
        if items:
            d = [np.hypot(it.x - X, it.y - Y) for it in items]
            j = int(np.argmin(d))
            if d[j] <= max(10.0, 3.0 / scale):
                self.sel = j
                self.message = ""
                return
        self.add_at(X, Y)

    # ── Live full-resolution preview (a PNG a viewer can show beside the terminal) ──
    @property
    def live_path(self) -> str:
        return os.path.join(paths.TMP_DIR, "review_preview.png")

    def _write_live(self) -> None:
        if not self.live:
            return
        p, c = self.photo, self.cell
        box = (max(0.0, c.x0 - MARGIN), max(0.0, c.y0 - MARGIN),
               min(float(p.width), c.x1 + MARGIN), min(float(p.height), c.y1 + MARGIN))
        scale = 2.0
        img = self.image(self.pi).crop(tuple(int(v) for v in box))
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            return (x - box[0]) * scale, (y - box[1]) * scale

        self._draw_items(draw, to_px, scale, box, minimum=6.0)
        os.makedirs(paths.TMP_DIR, exist_ok=True)
        tmp = self.live_path + ".tmp.png"
        img.save(tmp)
        os.replace(tmp, self.live_path)

    def toggle_live(self) -> None:
        self.live = not self.live
        if self.live:
            self._write_live()
            self.message = imaging.open_external(self.live_path) + " — it follows the reviewer"
        else:
            self.message = "live preview off"

    # ── Text panels ──────────────────────────────────────────────────────────
    def title(self) -> list:
        p = self.photo
        name = os.path.basename(p.file)
        left = [("class:title.accent", " Review "),
                ("class:title", f" {self.result.meta.get('survey')} · "
                                f"{self.result.meta.get('model_label')} {self.result.meta.get('run')} "
                                f"· photo {self.pi + 1}/{len(self.photos)} {name} "
                                f"· cell {self.cell.index + 1}/{len(self.cells_of(self.pi))} "
                                f"· queue {self.qpos + 1}/{len(self.queue)} ")]
        return left

    def info(self) -> list:
        p, t = self.photo, self.review.threshold
        cm, cf = self.cell_counts()
        pc = self.review.counts(p)
        sm, sf, done, cells = self.totals()
        mark = " (t*)" if abs(t - self.t_star) < 1e-6 else f" (t* {self.t_star:.2f})"
        out = [("class:head", f" Threshold {t:.2f}"), ("class:dim", mark + "\n"),
               ("class:dim", "              model → reviewed\n"),
               ("", f" this cell   {cm:8,d} → {cf:,}\n"),
               ("", f" this photo  {pc.model:8,d} → {pc.final:,}\n"),
               ("", f" survey      {sm:8,d} → {sf:,}\n"),
               ("class:dim", f" reviewed {pc.cells_done}/{pc.cells_total} cells here, "
                             f"{done:,}/{cells:,} overall\n")]
        gt = self.gt(self.pi)
        if gt is not None:
            c = self.cell
            cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
            here = int(((cx >= c.x0) & (cx < c.x1) & (cy >= c.y0) & (cy < c.y1)).sum())
            out.append(("class:dim", f" annotated   {here:8,d} here, {len(gt):,} in photo\n"))
        if self.adding:
            out.append(("class:warn", " ADDING: arrows move, Enter places, Esc cancels\n"))
        return out

    def listing(self) -> list:
        items = self.items()
        if not items:
            return [("class:dim", " no detections in this cell" +
                     ("" if self.show_below else " (b shows those below threshold)"))]
        out = []
        for j, it in enumerate(items):
            st = self.status(it)
            glyph, colour = GLYPH[st]
            score = "  +  " if it.score is None else f"{it.score:5.2f}"
            style = "class:sel" if j == self.sel else ""
            out.append((style, " " + score + " "))
            out.append((f"{style} fg:{colour}", glyph))
            out.append((style, f" {WORDS[st]:<24}\n"))
        return out

    def bar(self) -> list:
        keys = [("↑↓", "select"), ("y/n", "bird/not"), ("Enter", "cell done"), ("←→", "cells"),
                ("[ ]", "photos"), ("a", "add"), ("u", "undo"), ("?", "help"), ("q", "quit")]
        out = []
        for k, what in keys:
            out += [("class:bar.key", f" {k}"), ("class:bar", f" {what} ")]
        if self.message:
            out.append(("class:bar.msg", "  " + self.message))
        return out


def run(result: detections.Result, mode: str = "detections") -> Reviewer | None:
    """Open the reviewer full-screen.  Returns the Reviewer (for a summary) when done."""
    if not result.photos():
        return None
    rv = Reviewer(result, mode)
    if not rv.queue:
        return rv
    kb = KeyBindings()
    adding = Condition(lambda: rv.adding)
    browsing = ~adding

    def step_cursor(dx: float, dy: float) -> None:
        scale = rv.geometry.get("loupe", (0, 0, 1.0))[2]
        x, y = rv.cursor
        rv.cursor = (x + dx / max(scale, 0.25), y + dy / max(scale, 0.25))

    @kb.add("up", filter=browsing)
    @kb.add("k", filter=browsing)
    def _(e):
        rv.sel = max(0, rv.sel - 1)

    @kb.add("down", filter=browsing)
    @kb.add("j", filter=browsing)
    def _(e):
        rv.sel = min(max(0, len(rv.items()) - 1), rv.sel + 1)

    @kb.add("y", filter=browsing)
    def _(e):
        rv.decide(BIRD)

    @kb.add("n", filter=browsing)
    def _(e):
        rv.decide(NOT_BIRD)

    @kb.add("c", filter=browsing)
    def _(e):
        rv.decide(None)

    @kb.add("x", filter=browsing)
    @kb.add("delete", filter=browsing)
    def _(e):
        rv.delete_added()

    @kb.add("enter", filter=browsing)
    def _(e):
        rv.cell_done()

    @kb.add("right", filter=browsing)
    @kb.add("pagedown", filter=browsing)
    @kb.add("l", filter=browsing)
    def _(e):
        rv.goto(rv.qpos + 1)

    @kb.add("left", filter=browsing)
    @kb.add("pageup", filter=browsing)
    @kb.add("h", filter=browsing)
    def _(e):
        rv.goto(rv.qpos - 1)

    @kb.add("]", filter=browsing)
    def _(e):
        rv.jump_photo(+1)

    @kb.add("[", filter=browsing)
    def _(e):
        rv.jump_photo(-1)

    @kb.add("a", filter=browsing)
    def _(e):
        rv.cursor = rv.loupe_centre()    # start on the selected detection, before adding=True
        rv.adding = True
        rv.message = "move the cross onto the bird, Enter to add it"

    for key, dx, dy in (("up", 0, -2), ("down", 0, 2), ("left", -2, 0), ("right", 2, 0),
                        ("s-up", 0, -10), ("s-down", 0, 10), ("s-left", -10, 0),
                        ("s-right", 10, 0)):
        kb.add(key, filter=adding)(lambda e, dx=dx, dy=dy: step_cursor(dx, dy))

    @kb.add("enter", filter=adding)
    def _(e):
        rv.adding = False
        rv.add_at(*rv.cursor)

    @kb.add("escape", filter=adding, eager=True)
    def _(e):
        rv.adding = False
        rv.message = "add cancelled"

    @kb.add("+")
    @kb.add("=")
    def _(e):
        rv.span_i = max(0, rv.span_i - 1)

    @kb.add("-")
    def _(e):
        rv.span_i = min(len(LOUPE_SPANS) - 1, rv.span_i + 1)

    @kb.add("b", filter=browsing)
    def _(e):
        rv.show_below = not rv.show_below
        rv.message = ("showing" if rv.show_below else "hiding") + " detections below threshold"

    @kb.add("g", filter=browsing)
    def _(e):
        if rv.gt(rv.pi) is None:
            rv.message = "this survey has no labels"
            return
        rv.show_gt = not rv.show_gt
        rv.message = "annotations " + ("shown (magenta)" if rv.show_gt else "hidden")

    @kb.add(">", filter=browsing)
    def _(e):
        rv.set_threshold(rv.review.threshold + 0.05)

    @kb.add("<", filter=browsing)
    def _(e):
        rv.set_threshold(rv.review.threshold - 0.05)

    @kb.add("T", filter=browsing)
    def _(e):
        rv.set_threshold(rv.t_star)

    @kb.add("o", filter=browsing)
    def _(e):
        rv.toggle_live()

    @kb.add("u", filter=browsing)
    def _(e):
        rv.undo()

    @kb.add("?")
    def _(e):
        rv.help = not rv.help

    @kb.add("q", filter=browsing)
    @kb.add("escape", filter=browsing, eager=True)
    @kb.add("c-c")
    def _(e):
        if rv.help:
            rv.help = False
            return
        rv.review.save()
        e.app.exit(result=rv)

    main = Window(ImageControl(rv.render_main, lambda k, x, y: rv.on_mouse("main", k, x, y)))
    loupe_rows = 21
    loupe = Window(ImageControl(rv.render_loupe, lambda k, x, y: rv.on_mouse("loupe", k, x, y)),
                   height=D(min=8, preferred=loupe_rows, max=loupe_rows))
    listing = FormattedTextControl(rv.listing, focusable=False,
                                   get_cursor_position=lambda: Point(0, rv.sel))
    right = HSplit([
        loupe,
        Window(height=1, char="─", style="class:dim"),
        Window(FormattedTextControl(rv.info), height=D(min=6, preferred=9)),
        Window(height=1, char="─", style="class:dim"),
        Window(listing, wrap_lines=False),
    ], width=RIGHT_WIDTH)
    body = VSplit([main, Window(width=1, char="│", style="class:dim"), right])
    root = FloatContainer(
        HSplit([Window(FormattedTextControl(rv.title), height=1, style="class:title"),
                body,
                Window(FormattedTextControl(rv.bar), height=1, style="class:bar")]),
        floats=[Float(ConditionalContainer(
            Frame(Window(FormattedTextControl(HELP), width=90,
                         height=HELP.count("\n") + 1), title="Keys"),
            filter=Condition(lambda: rv.help)))])

    depth = (ColorDepth.TRUE_COLOR if os.environ.get("COLORTERM") in ("truecolor", "24bit")
             else None)
    app = Application(layout=Layout(root), key_bindings=kb, full_screen=True,
                      mouse_support=True, style=STYLE, color_depth=depth)
    rv.message = "? shows every key"
    app.run()
    return rv
