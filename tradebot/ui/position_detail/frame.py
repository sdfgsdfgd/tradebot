"""Shared geometry for the position-detail panes."""

from rich.text import Text
from textual.widgets import Static


BAR_FILL = "█"
BAR_EMPTY = "░"


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) > width:
        return "…" if width <= 1 else text[: width - 1] + "…"
    return text.ljust(width)


def pane_width(widget: Static, floor: int) -> int:
    size = int(getattr(widget.size, "width", 0) or 0)
    # Detail panes are padded by one char on each side in app CSS.
    return floor if size <= 0 else max(size - 2, 24)


def meter(ratio: float | None, width: int) -> str:
    width = max(width, 1)
    bounded = min(max(float(ratio), 0.0), 1.0) if ratio is not None else 0.0
    filled = max(0, min(width, int(round(bounded * width))))
    return (BAR_FILL * filled) + (BAR_EMPTY * (width - filled))


def box_top(title: str, inner_width: int, *, style: str) -> Text:
    label = f" {title} "
    if len(label) > inner_width:
        label = clip(label, inner_width)
    line = Text("┌", style=style)
    line.append(label, style="bold")
    line.append("─" * max(inner_width - len(label), 0), style=style)
    line.append("┐", style=style)
    return line


def box_rule(title: str, inner_width: int, *, style: str) -> Text:
    label = f" {title} "
    if len(label) > inner_width:
        label = clip(label, inner_width)
    line = Text("├", style=style)
    line.append(label, style="bold")
    line.append("─" * max(inner_width - len(label), 0), style=style)
    line.append("┤", style=style)
    return line


def box_row(content: Text | str, inner_width: int, *, style: str) -> Text:
    row = content.copy() if isinstance(content, Text) else Text(str(content))
    if len(row.plain) > inner_width:
        row = Text(clip(row.plain, inner_width))
    line = Text("│", style=style)
    line.append_text(row)
    line.append(" " * max(inner_width - len(row.plain), 0))
    line.append("│", style=style)
    return line


def box_bottom(inner_width: int, *, style: str) -> Text:
    return Text("└" + ("─" * inner_width) + "┘", style=style)
