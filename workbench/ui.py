"""
Terminal primitives.  Every screen is built from these, so the whole program behaves the
same way everywhere: arrow keys (or the option's number) to move, Enter to choose, Esc to
go back, Ctrl-C to cancel whatever is being asked.

Menus and questions are prompt_toolkit; everything printed (tables, panels, messages) is
rich.  Both were already in the `waterfowl` environment, so the workbench installs
nothing.  When stdin is not a terminal (piped input, a test harness) every question falls
back to a plain numbered prompt, so scripted use still works.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Sequence

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console(highlight=False)

STYLE = Style.from_dict({
    "question": "bold",
    "hint": "#8a8a8a",
    "rprompt": "#8a8a8a",
    "pointer": "#58a6ff bold",
    "checked": "#3fb950 bold",
    "unchecked": "#8a8a8a",
    "current": "bold",
})

_BACK = object()
_MENU_KEYS = " ↑/↓ move · 1-9 choose · Enter select · Esc back"
_MULTI_KEYS = " ↑/↓ move · Space tick/untick · a all/none · Enter done · Esc back"


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


# ── Output ────────────────────────────────────────────────────────────────────

def title(*crumbs: str) -> None:
    """Clear the screen and draw the breadcrumb header every screen starts with."""
    console.clear()
    trail = "  ›  ".join(("Waterfowl Workbench",) + tuple(c for c in crumbs if c))
    console.rule(f"[bold cyan]{trail}[/]", align="left", style="cyan")
    console.print()


def ok(message: str) -> None:
    console.print(f"[green]✓[/] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]![/] {message}")


def error(message: str) -> None:
    console.print(f"[red]✗[/] {message}")


def note(message: str) -> None:
    console.print(f"[cyan]›[/] {message}")


def dim(message: str) -> None:
    console.print(f"[dim]{message}[/]")


def table(columns: Sequence[str | tuple[str, str]], rows: Sequence[Sequence[Any]],
          title: str | None = None, caption: str | None = None) -> Table:
    """A table in the house style.  A column given as (name, "right") is right-aligned."""
    t = Table(box=box.SIMPLE_HEAD, title=title, title_justify="left", caption=caption,
              caption_justify="left", header_style="bold", pad_edge=False,
              title_style="bold")
    for col in columns:
        name, justify = (col, "left") if isinstance(col, str) else col
        t.add_column(name, justify=justify, no_wrap=justify == "right")
    for row in rows:
        t.add_row(*[str(c) if c is not None else "—" for c in row])
    return t


def kv(pairs: Sequence[tuple[str, Any]], title: str | None = None) -> Panel | Table:
    """Key/value lines: the summary block most screens open with."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()
    for key, value in pairs:
        grid.add_row(key, "—" if value is None else str(value))
    if title:
        return Panel(grid, title=f"[bold]{title}[/]", title_align="left",
                     border_style="grey50", padding=(0, 1))
    return grid


def show(*renderables) -> None:
    for r in renderables:
        console.print(r)


# ── Questions ─────────────────────────────────────────────────────────────────

def _normalise(option) -> tuple[Any, str, str]:
    if len(option) == 2:
        return option[0], str(option[1]), ""
    return option[0], str(option[1]), str(option[2] or "")


def choose(message: str, options: Sequence[tuple], default: Any = None,
           back: bool = True, back_label: str = "Back") -> Any:
    """Pick one option.  Options are (value, label) or (value, label, hint).

    Returns the chosen value, or None when the user backs out (Esc, q, Ctrl-C or the
    trailing "Back" entry) — so option values must never themselves be None.  Number
    keys 1-9 choose at once, in every menu; entries past the ninth are reached with the
    arrows.
    """
    items = [_normalise(o) for o in options]
    if back:
        items.append((_BACK, back_label, ""))
    if not items:
        return None
    if not interactive():
        return _choose_plain(message, items)

    start = next((i for i, (v, _, _) in enumerate(items)
                  if default is not None and (v is default or v == default)), 0)
    state = {"index": start}
    width = max(len(label) for _, label, _ in items)
    digits = len(str(len(items)))

    def lines() -> StyleAndTextTuples:
        out: StyleAndTextTuples = []
        for i, (_, label, hint) in enumerate(items):
            here = i == state["index"]
            out.append(("class:pointer", " > " if here else "   "))
            out.append(("class:hint", f"{i + 1:>{digits}}. "))
            out.append(("class:current" if here else "", label.ljust(width)))
            if hint:
                out.append(("class:hint", "   " + hint))
            out.append(("", "\n"))
        return out[:-1]

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _(event):
        state["index"] = (state["index"] - 1) % len(items)

    @kb.add("down")
    @kb.add("j")
    @kb.add("tab")
    def _(event):
        state["index"] = (state["index"] + 1) % len(items)

    @kb.add("home")
    def _(event):
        state["index"] = 0

    @kb.add("end")
    def _(event):
        state["index"] = len(items) - 1

    @kb.add("enter")
    def _(event):
        event.app.exit(result=items[state["index"]][0])

    @kb.add("escape", eager=True)
    @kb.add("q")
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=_BACK)

    for d in range(1, 10):
        @kb.add(str(d))
        def _(event, d=d):
            if d <= len(items):
                state["index"] = d - 1
                event.app.exit(result=items[d - 1][0])

    rows = max(3, min(len(items), console.size.height - 4))
    parts = []
    if message:
        parts.append(Window(FormattedTextControl([("class:question", message)]), height=1))
    parts += [
        Window(FormattedTextControl(lines, focusable=True, show_cursor=False,
                                    get_cursor_position=lambda: Point(0, state["index"])),
               height=Dimension(min=1, max=rows, preferred=rows)),
        Window(FormattedTextControl([("class:hint", _MENU_KEYS)]), height=1),
    ]
    app = Application(layout=Layout(HSplit(parts)), key_bindings=kb, style=STYLE,
                      full_screen=False, erase_when_done=True)
    try:
        result = app.run()
    except (KeyboardInterrupt, EOFError):
        return None
    if result is _BACK or result is None:
        return None
    if message:
        chosen = next(label for v, label, _ in items if v is result)
        console.print(f"[bold]{escape(message)}[/] {escape(chosen)}")
    return result


def _choose_plain(message: str, items: list[tuple[Any, str, str]]) -> Any:
    print(message)
    for i, (_, label, hint) in enumerate(items, 1):
        print(f"  {i}. {label}" + (f"   ({hint})" if hint else ""))
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            value = items[int(raw) - 1][0]
            return None if value is _BACK else value
        if raw.lower() in ("q", "b", "back", ""):
            return None


def choose_many(message: str, options: Sequence[tuple], selected: Sequence[Any] = (),
                min_count: int = 1) -> list | None:
    """Tick any number of options.  Returns the ticked values in option order, or None."""
    items = [_normalise(o) for o in options]
    if not items:
        return None
    if not interactive():
        print(message)
        for i, (_, label, hint) in enumerate(items, 1):
            print(f"  {i}. {label}" + (f"   ({hint})" if hint else ""))
        raw = input("numbers separated by spaces (blank = back): ").split()
        picked = [items[int(r) - 1][0] for r in raw if r.isdigit() and 1 <= int(r) <= len(items)]
        return picked or None

    state = {"index": 0, "ticked": {i for i, (v, _, _) in enumerate(items) if v in selected},
             "message": ""}
    width = max(len(label) for _, label, _ in items)

    def lines() -> StyleAndTextTuples:
        out: StyleAndTextTuples = []
        for i, (_, label, hint) in enumerate(items):
            here = i == state["index"]
            out.append(("class:pointer", " > " if here else "   "))
            out.append(("class:checked", "[x] ") if i in state["ticked"]
                       else ("class:unchecked", "[ ] "))
            out.append(("class:current" if here else "", label.ljust(width)))
            if hint:
                out.append(("class:hint", "   " + hint))
            out.append(("", "\n"))
        return out[:-1]

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _(event):
        state["index"] = (state["index"] - 1) % len(items)

    @kb.add("down")
    @kb.add("j")
    def _(event):
        state["index"] = (state["index"] + 1) % len(items)

    @kb.add(" ")
    def _(event):
        state["ticked"] ^= {state["index"]}
        state["message"] = ""

    @kb.add("a")
    def _(event):
        state["ticked"] = set() if len(state["ticked"]) == len(items) else set(range(len(items)))

    @kb.add("enter")
    def _(event):
        if len(state["ticked"]) < min_count:
            state["message"] = f"  tick at least {min_count} (Space)"
            return
        event.app.exit(result=[items[i][0] for i in sorted(state["ticked"])])

    @kb.add("escape", eager=True)
    @kb.add("q")
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    body = FormattedTextControl(lines, focusable=True, show_cursor=False,
                                get_cursor_position=lambda: Point(0, state["index"]))
    rows = max(3, min(len(items), console.size.height - 6))
    layout = Layout(HSplit([
        Window(FormattedTextControl([("class:question", message)]), height=1),
        Window(body, height=Dimension(min=1, max=rows, preferred=rows)),
        Window(FormattedTextControl(lambda: [("class:hint", _MULTI_KEYS),
                                             ("ansiyellow", state["message"])]), height=1),
    ]))
    app = Application(layout=layout, key_bindings=kb, style=STYLE, full_screen=False,
                      erase_when_done=True)
    result = app.run()
    if result is not None:
        labels = [items[i][1] for i in sorted(state["ticked"])]
        console.print(f"[bold]{escape(message)}[/] {escape(', '.join(labels))}")
    return result


class _FnValidator(Validator):
    def __init__(self, fn: Callable[[str], str | None]):
        self.fn = fn

    def validate(self, document) -> None:
        problem = self.fn(document.text)
        if problem:
            raise ValidationError(message=problem, cursor_position=len(document.text))


def ask(message: str, default: str = "", validate: Callable[[str], str | None] | None = None,
        completer=None, hint: str | None = None) -> str | None:
    """Free-text answer.  `validate` returns an error message or None.  None = cancelled."""
    if not interactive():
        try:
            raw = input(f"{message} [{default}]: ").strip()
        except EOFError:
            return None
        return raw or default

    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event):
        event.app.exit(result=None)

    keys = [("class:hint", (hint + " · " if hint else "") + "Enter accept · Esc back ")]
    session = PromptSession(erase_when_done=True)
    try:
        answer = session.prompt(
            [("class:question", message + " ")], default=default, completer=completer,
            complete_while_typing=completer is not None,
            validator=_FnValidator(validate) if validate else None,
            validate_while_typing=False, key_bindings=kb, style=STYLE, rprompt=keys,
        )
    except (KeyboardInterrupt, EOFError):
        return None
    if answer is None:
        return None
    console.print(f"[bold]{escape(message)}[/] {escape(answer.strip())}")
    return answer.strip()


def ask_path(message: str, default: str = "", dirs_only: bool = True,
             must_exist: bool = True) -> str | None:
    """A file-system path, with Tab completion.  Returned absolute, with ~ expanded."""
    def check(text: str) -> str | None:
        path = os.path.expanduser(text.strip())
        if not path:
            return "enter a path"
        if must_exist and not os.path.exists(path):
            return f"{path} does not exist"
        if must_exist and dirs_only and not os.path.isdir(path):
            return f"{path} is not a folder"
        return None

    answer = ask(message, default=default, validate=check,
                 completer=PathCompleter(only_directories=dirs_only, expanduser=True),
                 hint="Tab completes")
    return os.path.abspath(os.path.expanduser(answer)) if answer else None


def ask_int(message: str, default: int | None = None, lo: int | None = None,
            hi: int | None = None) -> int | None:
    def check(text: str) -> str | None:
        try:
            value = int(text)
        except ValueError:
            return "enter a whole number"
        if lo is not None and value < lo:
            return f"must be at least {lo}"
        if hi is not None and value > hi:
            return f"must be at most {hi}"
        return None

    answer = ask(message, default="" if default is None else str(default), validate=check)
    return int(answer) if answer not in (None, "") else None


def ask_float(message: str, default: float | None = None, lo: float | None = None,
              hi: float | None = None) -> float | None:
    def check(text: str) -> str | None:
        try:
            value = float(text)
        except ValueError:
            return "enter a number"
        if lo is not None and value < lo:
            return f"must be at least {lo}"
        if hi is not None and value > hi:
            return f"must be at most {hi}"
        return None

    answer = ask(message, default="" if default is None else f"{default:g}", validate=check)
    return float(answer) if answer not in (None, "") else None


def _one_key_app(fragments: StyleAndTextTuples, kb: KeyBindings) -> Any:
    app = Application(layout=Layout(Window(FormattedTextControl(fragments), height=1)),
                      key_bindings=kb, style=STYLE, full_screen=False, erase_when_done=True)
    return app.run()


def confirm(message: str, default: bool = True) -> bool:
    """Yes/no with one key: y, n, or Enter for the default.  Esc and Ctrl-C mean no."""
    if not interactive():
        try:
            raw = input(f"{message} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        except EOFError:
            return False
        return default if not raw else raw.startswith("y")

    kb = KeyBindings()

    @kb.add("y")
    @kb.add("Y")
    def _(event):
        event.app.exit(result=True)

    @kb.add("n")
    @kb.add("N")
    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=False)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=default)

    answer = _one_key_app([("class:question", message),
                           ("class:hint", " [Y/n] " if default else " [y/N] ")], kb)
    console.print(f"[bold]{escape(message)}[/] {'yes' if answer else 'no'}")
    return bool(answer)


def pause(message: str = "Press any key to go back") -> None:
    if not interactive():
        try:
            input(f"{message}… ")
        except EOFError:
            pass
        return
    kb = KeyBindings()

    @kb.add(Keys.Any)
    @kb.add("enter")
    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    console.print()
    _one_key_app([("class:hint", message + "…")], kb)
