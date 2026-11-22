"""
Pictures in a terminal, and pictures outside it.

In the terminal an image is drawn with the upper-half-block character "▀": its
foreground colour paints the top pixel of a cell and its background the bottom one, so a
W x H character area holds a W x 2H pixel image.  Terminal cells are about twice as tall as
they are wide, which makes those pixels roughly square.  It needs nothing from the
terminal but 24-bit colour (VS Code's has it; others fall back to 256 colours), and it
works over plain SSH.

It is coarse — a 512 px crop drawn 100 columns wide shows about one terminal pixel per
five image pixels, and birds here are ~17 px across — so the reviewer pairs it with a
magnified loupe, and anything that needs full resolution can be opened in an image viewer:
inside VS Code's terminal that is a VS Code editor tab (`code <file>`), which also works
over Remote-SSH; on a desktop session it is the system viewer (`xdg-open`).
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
from PIL import Image
from rich.style import Style
from rich.text import Text

_HEX = [f"{i:02x}" for i in range(256)]
UPPER = "▀"


def fit(width: int, height: int, cols: int, rows: int) -> tuple[int, int, float]:
    """Largest (pixel_w, pixel_h) that fits `cols` x `rows` cells, and the scale used."""
    scale = min(cols / max(width, 1), (2 * rows) / max(height, 1))
    return max(1, int(width * scale)), max(1, int(height * scale)), scale


def to_rows(img: Image.Image) -> list[list[tuple[str, str]]]:
    """prompt_toolkit fragments for an RGB image, one list per character row.

    The image height is padded to an even number of pixels.  Runs of identical cells are
    merged, which keeps a large flat background cheap to draw.
    """
    arr = np.asarray(img.convert("RGB"))
    if arr.shape[0] % 2:
        arr = np.concatenate([arr, np.zeros((1, arr.shape[1], 3), arr.dtype)], axis=0)
    rows = []
    for r in range(0, arr.shape[0], 2):
        top, bottom = arr[r], arr[r + 1]
        frags: list[tuple[str, str]] = []
        last, run = None, 0
        for c in range(arr.shape[1]):
            t, b = top[c], bottom[c]
            style = (f"fg:#{_HEX[t[0]]}{_HEX[t[1]]}{_HEX[t[2]]} "
                     f"bg:#{_HEX[b[0]]}{_HEX[b[1]]}{_HEX[b[2]]}")
            if style == last:
                run += 1
            else:
                if last is not None:
                    frags.append((last, UPPER * run))
                last, run = style, 1
        if last is not None:
            frags.append((last, UPPER * run))
        rows.append(frags)
    return rows


def rich_image(source: str | Image.Image, cols: int, rows: int) -> Text:
    """The same half-block drawing as a rich renderable, for printing to the console."""
    img = Image.open(source) if isinstance(source, str) else source
    w, h, _ = fit(img.width, img.height, cols, rows)
    small = img.convert("RGB").resize((w, h + (h % 2)), Image.LANCZOS)
    arr = np.asarray(small)
    text = Text(no_wrap=True, overflow="crop")
    for r in range(0, arr.shape[0], 2):
        for c in range(arr.shape[1]):
            t, b = arr[r, c], arr[r + 1, c]
            text.append(UPPER, Style(color=f"#{_HEX[t[0]]}{_HEX[t[1]]}{_HEX[t[2]]}",
                                     bgcolor=f"#{_HEX[b[0]]}{_HEX[b[1]]}{_HEX[b[2]]}"))
        text.append("\n")
    return text


def viewer_available() -> str | None:
    """Which external viewer `open_external` would use, or None."""
    from workbench import settings
    choice = settings.get("viewer")
    if choice == "none":
        return None
    in_vscode = bool(os.environ.get("VSCODE_IPC_HOOK_CLI"))
    if choice in ("auto", "code") and in_vscode and shutil.which("code"):
        return "code"
    desktop = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if choice in ("auto", "xdg-open") and desktop and shutil.which("xdg-open"):
        return "xdg-open"
    return None


def open_external(path: str) -> str:
    """Open a file (image, report, folder) outside the terminal.  Returns what happened."""
    tool = viewer_available()
    if tool is None:
        return f"saved at {path} (no viewer available from this terminal)"
    subprocess.Popen([tool, path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return f"opened {os.path.basename(path)} in {'VS Code' if tool == 'code' else 'the viewer'}"
