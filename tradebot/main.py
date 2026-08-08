"""Entrypoint for the interactive portfolio TUI."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("tradebot TUI requires an interactive terminal; use tradebot --status for automation")
    from .ui import PositionsApp

    PositionsApp().run()


if __name__ == "__main__":
    main()
