"""Entrypoint for the minimal portfolio TUI."""
from __future__ import annotations

from .ui import PositionsApp


def main() -> None:
    PositionsApp().run()


if __name__ == "__main__":
    main()
