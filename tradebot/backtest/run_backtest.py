"""Unified CLI entrypoint for backtests.

This keeps the workflow streamlined while preserving existing module CLIs:
- `python -m tradebot.backtest --config ...` (single run)
- `python -m tradebot.backtest evolve_spot ...` (sweeps)
- `python -m tradebot.backtest kingmaker ...` (multi-window stability eval)
"""

from __future__ import annotations

import sys
from contextlib import contextmanager


@contextmanager
def _argv(argv: list[str]):
    old = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def _print_help() -> None:
    print("tradebot.backtest")
    print("")
    print("Usage:")
    print("  python -m tradebot.backtest --config <path> [--no-write] [--calibrate]")
    print("  python -m tradebot.backtest evolve_spot <args...>")
    print("  python -m tradebot.backtest kingmaker <args...>")
    print("")
    print("Shortcuts:")
    print("  python -m tradebot.backtest run ...       # alias for --config runner")
    print("  python -m tradebot.backtest evolve ...    # alias for evolve_spot")
    print("")


def main() -> None:
    argv = list(sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return

    cmd = argv[0]
    rest = argv[1:]

    # Backwards compatibility: treat flag-first invocations as the old single-run CLI.
    if cmd.startswith("-"):
        from . import cli

        with _argv([sys.argv[0]] + argv):
            cli.main()
        return

    if cmd in ("single", "run", "config"):
        from . import cli

        with _argv([sys.argv[0]] + rest):
            cli.main()
        return

    if cmd in ("evolve_spot", "evolve", "spot"):
        from . import evolve_spot

        with _argv([sys.argv[0]] + rest):
            evolve_spot.main()
        return

    if cmd in ("kingmaker",):
        from . import kingmaker

        with _argv([sys.argv[0]] + rest):
            kingmaker.main()
        return

    print(f"Unknown command: {cmd!r}")
    _print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()

