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
)
from .xsp_shadow import advance_xsp_shadow_from_ibkr


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advance the non-submitting XSP directional shadow."
    )
    parser.add_argument(
        "--ledger",
        default="db/calibration/xsp_live_calibration.jsonl",
    )
    parser.add_argument("--duration", default="2 D")
    parser.add_argument("--option-tape")
    parser.add_argument(
        "--news-signal",
        default="~/.local/state/tradebot/news/latest.json",
    )
    args = parser.parse_args(argv)

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
    try:
        receipt = await advance_xsp_shadow_from_ibkr(
            ledger,
            client=client,
            observed_at=observed_at,
            duration_str=str(args.duration),
            option_snapshots=options,
            news_snapshot=tuple(news),
        )
    finally:
        await client.disconnect()
    print(
        json.dumps(
            {
                **receipt,
                "fundamental_defensive_benchmark": (
                    xsp_fundamental_defensive_benchmark(ledger)
                ),
                "option_parity_participation_benchmark": (
                    xsp_option_parity_participation_benchmark(ledger)
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
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt.get("evaluation_status") == "EVALUATED" else 2


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))
