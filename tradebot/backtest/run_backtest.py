"""Unified CLI entrypoint for backtests.

This keeps the workflow streamlined while preserving the key workflows:
- `python -m tradebot.backtest --config ...` (single run)
- `python -m tradebot.backtest spot ...` (spot sweeps / evolution)
- `python -m tradebot.backtest spot_multitimeframe ...` (spot multi-window stability eval)
- `python -m tradebot.backtest options_leaderboard ...` (options sweeps → leaderboard.json)
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
    print("  python -m tradebot.backtest spot <args...>")
    print("  python -m tradebot.backtest spot_multitimeframe <args...>")
    print("  python -m tradebot.backtest options_leaderboard <args...>")
    print("")
    print("Aliases:")
    print("  python -m tradebot.backtest run ...       # alias for --config runner")
    print("  python -m tradebot.backtest evolve ...    # alias for spot")
    print("  python -m tradebot.backtest multitimeframe ...   # alias for spot_multitimeframe")
    print("  python -m tradebot.backtest leaderboard ...      # alias for options_leaderboard")
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

    if cmd in ("spot", "evolve_spot", "evolve"):
        from . import run_backtest_spot

        with _argv([sys.argv[0]] + rest):
            run_backtest_spot.main()
        return

    if cmd in ("spot_multitimeframe", "multitimeframe", "multiwindow", "kingmaker"):
        from . import run_backtest_spot

        with _argv([sys.argv[0]] + rest):
            run_backtest_spot.spot_multitimeframe_main()
        return

    if cmd in ("options_leaderboard", "leaderboard", "generate_leaderboard", "lb", "options"):
        from . import run_backtest_options

        with _argv([sys.argv[0]] + rest):
            run_backtest_options.options_leaderboard_main()
        return

    print(f"Unknown command: {cmd!r}")
    _print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
