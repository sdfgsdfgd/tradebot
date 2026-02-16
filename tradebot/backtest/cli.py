"""CLI entrypoint for synthetic backtests."""
from __future__ import annotations

import argparse
import threading
import time
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
    started_at = float(time.perf_counter())
    done_evt = threading.Event()
    result_holder: dict[str, object] = {}
    error_holder: dict[str, BaseException] = {}

    def _run_single() -> None:
        try:
            result_holder["result"] = run_backtest(cfg)
        except BaseException as exc:  # surface exact original failure after heartbeat loop stops
            error_holder["error"] = exc
        finally:
            done_evt.set()

    t = threading.Thread(target=_run_single, daemon=True)
    t.start()
    while not done_evt.wait(30.0):
        elapsed = max(0.0, float(time.perf_counter()) - float(started_at))
        print(f"single backtest heartbeat inflight={elapsed:0.1f}s", flush=True)
    t.join()
    if "error" in error_holder:
        raise error_holder["error"]
    result = result_holder.get("result")
    if result is None:
        raise RuntimeError("single backtest run returned no result")
    print_summary(result)
    if not args.no_write:
        write_reports(result, cfg.backtest.output_dir)


if __name__ == "__main__":
    main()
