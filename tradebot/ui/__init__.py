"""UI package (TUI + bot helpers)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import PositionsApp as PositionsApp

__all__ = ["PositionsApp"]


def __getattr__(name: str):
    if name == "PositionsApp":
        from .app import PositionsApp

        return PositionsApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
