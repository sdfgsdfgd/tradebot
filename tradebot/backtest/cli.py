"""CLI entrypoint for synthetic backtests."""
from __future__ import annotations

import argparse
from dataclasses import replace

from .config import load_config
from .engine import run_backtest
from .report import print_summary, write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic options backtest")
    parser.add_argument("--config", required=True, help="Path to backtest JSON config")
    parser.add_argument("--no-write", action="store_true", help="Skip writing CSV outputs")
    parser.add_argument("--calibrate", action="store_true", help="Refresh calibration before run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.calibrate and not cfg.backtest.calibrate:
        cfg = replace(cfg, backtest=replace(cfg.backtest, calibrate=True))
    result = run_backtest(cfg)
    print_summary(result)
    if not args.no_write:
        write_reports(result, cfg.backtest.output_dir)


if __name__ == "__main__":
    main()
