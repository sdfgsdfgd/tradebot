"""Command-line adapter for the non-submitting XSP shadow."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from ..backtest.quotes import iter_snapshots
from ..config import auxiliary_client_config, load_config
from ..engines.market import xsp_trading_date
from ..news.contract import NewsError, load_news_history
from .live_calibration import LiveCalibrationLedger
from .xsp_benchmarks import (
    xsp_fundamental_defensive_benchmark,
    xsp_option_parity_participation_benchmark,
    xsp_profitability_policy_from_selected_run,
)
from .xsp_shadow import (
    XSP_DIRECTIONAL_HISTORY_DURATION,
    advance_xsp_shadow_from_ibkr,
)
from .xsp_opening_edge_v2 import (
    XSP_OPENING_EDGE_V2_HISTORY_DURATION,
    advance_xsp_opening_edge_v2_from_ibkr,
    load_xsp_opening_edge_v2_spec,
    xsp_opening_edge_v2_run_start,
)


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advance one explicit non-submitting XSP observer."
    )
    parser.add_argument(
        "--mode",
        choices=("directional-v1", "opening-edge-v2"),
        default="directional-v1",
    )
    parser.add_argument(
        "--ledger",
        default="db/calibration/xsp_live_calibration.jsonl",
    )
    parser.add_argument("--duration")
    parser.add_argument("--option-tape")
    parser.add_argument(
        "--news-signal",
        default="~/.local/state/tradebot/news/latest.json",
    )
    parser.add_argument(
        "--selected-run",
        default="db/calibration/xsp_selected_shadow.json",
    )
    args = parser.parse_args(argv)

    selected_path = Path(args.selected_run).expanduser()
    selected_run = None
    selected_policy = None
    if args.mode == "directional-v1" and selected_path.exists():
        loaded_selection = json.loads(selected_path.read_text())
        if not isinstance(loaded_selection, dict):
            raise ValueError("selected XSP shadow run must be an object")
        selected_policy = xsp_profitability_policy_from_selected_run(
            loaded_selection
        )
        selected_run = loaded_selection

    from ..client import IBKRClient

    config = auxiliary_client_config(load_config(), 80)
    client = IBKRClient(config)
    observed_at = datetime.now(tz=timezone.utc)
    option_day = xsp_trading_date(observed_at) or observed_at.date()
    option_path = (
        Path(args.option_tape)
        if args.option_tape
        else Path("db/quotes/XSP") / f"{option_day}.jsonl"
    )
    options = tuple(iter_snapshots(option_path)) if option_path.exists() else ()
    news_path = Path(args.news_signal).expanduser()
    previous_month = option_day.replace(day=1) - timedelta(days=1)
    history_paths = tuple(
        news_path.parent
        / "history"
        / f"{month.year:04d}-{month.month:02d}.jsonl"
        for month in (previous_month, option_day)
    )
    news = []
    for history_path in history_paths:
        try:
            news.extend(load_news_history(history_path))
        except NewsError as exc:
            news.append({"load_error": str(exc)})
    try:
        loaded = json.loads(news_path.read_text())
        if isinstance(loaded, dict):
            news.append(loaded)
        else:
            news.append({"load_error": "latest news snapshot is not an object"})
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        news.append({"load_error": str(exc)})
    ledger = LiveCalibrationLedger(args.ledger)
    v2_run_start = None
    try:
        if args.mode == "opening-edge-v2":
            v2_spec = load_xsp_opening_edge_v2_spec()
            v2_run_start = xsp_opening_edge_v2_run_start(
                tuple(ledger.records()),
                observed_at=observed_at,
            )
            receipt = await advance_xsp_opening_edge_v2_from_ibkr(
                ledger,
                client=client,
                observed_at=observed_at,
                run_started_at=v2_run_start,
                duration_str=str(
                    args.duration or XSP_OPENING_EDGE_V2_HISTORY_DURATION
                ),
                news_snapshot=tuple(news),
                spec=v2_spec,
            )
        else:
            receipt = await advance_xsp_shadow_from_ibkr(
                ledger,
                client=client,
                observed_at=observed_at,
                duration_str=str(
                    args.duration or XSP_DIRECTIONAL_HISTORY_DURATION
                ),
                option_snapshots=options,
                news_snapshot=tuple(news),
                selected_run=selected_run,
            )
    finally:
        await client.disconnect()
    completed_at = datetime.now(tz=timezone.utc)
    print(
        json.dumps(
            {
                **receipt,
                "mode": str(args.mode),
                "fundamental_defensive_benchmark": (
                    xsp_fundamental_defensive_benchmark(ledger)
                    if args.mode == "directional-v1"
                    else None
                ),
                "option_parity_participation_benchmark": (
                    xsp_option_parity_participation_benchmark(ledger)
                    if args.mode == "directional-v1"
                    else None
                ),
                "profitability": (
                    ledger.xsp_profitability_receipt(
                        policy=selected_policy,
                        as_of=completed_at,
                    )
                    if selected_policy is not None
                    else None
                ),
                "client_ids": {
                    "main": config.client_id,
                    "proxy": config.proxy_client_id,
                    "index": config.proxy_client_id + 1,
                },
                "broker_readonly": config.readonly,
                "order_authority": "none",
                "option_tape": str(option_path),
                "news_signal": str(news_path),
                "news_history": [str(path) for path in history_paths],
                "news_publications": len(news),
                "selected_run": str(selected_path),
                "selected_run_id": (
                    selected_policy.run_id
                    if selected_policy is not None
                    else None
                ),
                "v2_run_started_at_utc": (
                    v2_run_start.astimezone(timezone.utc).isoformat()
                    if v2_run_start is not None
                    else None
                ),
                "completed_at_utc": completed_at.isoformat(),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    successful_preflight = (
        args.mode == "opening-edge-v2"
        and receipt.get("broker_request_skipped") == "run_not_started"
    )
    return (
        0
        if receipt.get("evaluation_status") == "EVALUATED"
        or successful_preflight
        else 2
    )


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))
