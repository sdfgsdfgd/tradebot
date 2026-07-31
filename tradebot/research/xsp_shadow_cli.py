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
from .xsp_opening_edge_v3 import (
    XSP_OPENING_EDGE_V3_HISTORY_DURATION,
    XSP_OPENING_EDGE_V3_VERSION,
    advance_xsp_opening_edge_v3_from_ibkr,
    load_xsp_opening_edge_v3_spec,
    xsp_opening_edge_v3_fundamental_pairs,
    xsp_opening_edge_v3_run_start,
)
from .xsp_execution_observer import (
    advance_xsp_v2_etf_execution_observer,
)
from .xsp_live_transport import (
    XSP_V3_TRANSPORT_SELECTION_SCHEMA,
    load_xsp_v2_transport_selection,
    select_xsp_v2_transport,
    write_xsp_v2_transport_selection,
)
from .xsp_live_transport_state import (
    latest_xsp_v2_source_receipt,
    latest_xsp_v3_source_receipt,
    xsp_v2_broker_snapshot,
)
from .xsp_live_transport_handoff import (
    handoff_xsp_v3_immediate_proceeds,
    rebase_xsp_v3_immediate_proceeds,
)
from .xsp_live_transport_v3 import (
    load_xsp_v3_transport_selection,
    select_xsp_v3_transport,
    write_xsp_v3_transport_selection as write_xsp_transport_selection,
    xsp_v3_transport_profitability_policy,
)
from .xsp_live_transport_runtime import (
    advance_xsp_live_transport,
    advance_xsp_v2_live_transport,
)


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advance one explicit non-submitting XSP observer."
    )
    parser.add_argument(
        "--mode",
        choices=("directional-v1", "opening-edge-v2", "opening-edge-v3"),
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
    parser.add_argument(
        "--selected-transport",
        default="db/calibration/xsp_selected_live_transport.json",
    )
    parser.add_argument("--freeze-selected-transport", action="store_true")
    parser.add_argument("--handoff-immediate-proceeds", action="store_true")
    parser.add_argument("--transport-ranking")
    parser.add_argument("--transport-dwell")
    parser.add_argument("--transport-preview")
    parser.add_argument(
        "--transport-cash-receipt",
        default="backtests/xsp/opening_edge_v3_regime_harmony_cash_receipt.json",
    )
    parser.add_argument(
        "--transport-immediate-proceeds-receipt",
        default="backtests/xsp/opening_edge_v3_immediate_proceeds_receipt.json",
    )
    parser.add_argument("--accept-rth-only-cash-scope", action="store_true")
    args = parser.parse_args(argv)

    selected_path = Path(args.selected_run).expanduser()
    selected_transport_path = Path(args.selected_transport).expanduser()
    if args.freeze_selected_transport and args.handoff_immediate_proceeds:
        raise ValueError("transport freeze and handoff are mutually exclusive")
    if args.handoff_immediate_proceeds:
        if args.mode != "opening-edge-v3" or not selected_transport_path.exists():
            raise ValueError("immediate-proceeds handoff requires a selected v3 run")
        from ..client import IBKRClient

        config = auxiliary_client_config(load_config(), 84)
        if not config.readonly:
            raise ValueError("transport handoff requires a read-only broker client")
        ledger = LiveCalibrationLedger(args.ledger)
        predecessor = load_xsp_v3_transport_selection(selected_transport_path)
        if (
            predecessor["schema"] != XSP_V3_TRANSPORT_SELECTION_SCHEMA
            and not args.transport_preview
        ):
            raise ValueError("clean immediate-proceeds rebase requires a fresh preview")
        client = IBKRClient(config)
        try:
            records = tuple(ledger.records())
            common = {
                "predecessor": predecessor,
                "records": records,
                "source_receipt": latest_xsp_v3_source_receipt(records),
                "broker_snapshot": await xsp_v2_broker_snapshot(
                    client,
                    symbols=("UPRO", "SPXU"),
                ),
                "immediate_proceeds_receipt_path": Path(
                    args.transport_immediate_proceeds_receipt
                ).expanduser(),
                "selected_at": datetime.now(timezone.utc),
            }
            successor = (
                handoff_xsp_v3_immediate_proceeds(**common)
                if predecessor["schema"] == XSP_V3_TRANSPORT_SELECTION_SCHEMA
                else rebase_xsp_v3_immediate_proceeds(
                    **common,
                    cash_receipt_path=Path(
                        args.transport_cash_receipt
                    ).expanduser(),
                    preview_path=Path(args.transport_preview).expanduser(),
                    rth_scope_accepted=args.accept_rth_only_cash_scope,
                )
            )
            archive = selected_transport_path.with_name(
                f"{selected_transport_path.stem}."
                f"{predecessor['selection_id']}.json"
            )
            if archive.exists():
                if load_xsp_v3_transport_selection(archive) != predecessor:
                    raise ValueError("selected predecessor archive changed")
            else:
                write_xsp_transport_selection(archive, predecessor)
            write_xsp_transport_selection(selected_transport_path, successor)
        finally:
            await client.disconnect()
        print(json.dumps(successor, allow_nan=False, indent=2, sort_keys=True))
        return 0
    if args.freeze_selected_transport:
        evidence_paths = tuple(
            Path(value).expanduser()
            for value in (
                args.transport_ranking,
                args.transport_dwell,
                args.transport_preview,
            )
            if value
        )
        if args.mode == "opening-edge-v2" and len(evidence_paths) != 3:
            raise ValueError(
                "transport selection requires v2 ranking, dwell, and preview"
            )
        if args.mode == "opening-edge-v3" and (
            not args.transport_preview
            or args.transport_ranking
            or args.transport_dwell
        ):
            raise ValueError("v3 transport selection requires only its exact preview")
        if args.mode not in {"opening-edge-v2", "opening-edge-v3"}:
            raise ValueError("transport selection requires Opening Edge v2 or v3")
        if selected_transport_path.exists():
            existing = (
                load_xsp_v2_transport_selection(selected_transport_path)
                if args.mode == "opening-edge-v2"
                else load_xsp_v3_transport_selection(selected_transport_path)
            )
            print(json.dumps(existing, allow_nan=False, indent=2, sort_keys=True))
            return 0
        from ..client import IBKRClient

        config = auxiliary_client_config(load_config(), 84)
        if not config.readonly:
            raise ValueError("transport selection requires a read-only broker client")
        client = IBKRClient(config)
        ledger = LiveCalibrationLedger(args.ledger)
        try:
            if args.mode == "opening-edge-v2":
                source = latest_xsp_v2_source_receipt(tuple(ledger.records()))
                broker = await xsp_v2_broker_snapshot(client)
                selection = select_xsp_v2_transport(
                    ranking_path=evidence_paths[0],
                    dwell_path=evidence_paths[1],
                    preview_path=evidence_paths[2],
                    source_receipt=source,
                    broker_snapshot=broker,
                    selected_at=datetime.now(timezone.utc),
                )
            else:
                source = latest_xsp_v3_source_receipt(tuple(ledger.records()))
                broker = await xsp_v2_broker_snapshot(
                    client,
                    symbols=("UPRO", "SPXU"),
                )
                selection = select_xsp_v3_transport(
                    cash_receipt_path=Path(args.transport_cash_receipt).expanduser(),
                    preview_path=Path(args.transport_preview).expanduser(),
                    source_receipt=source,
                    broker_snapshot=broker,
                    selected_at=datetime.now(timezone.utc),
                    rth_scope_accepted=args.accept_rth_only_cash_scope,
                )
            if args.mode == "opening-edge-v2":
                write_xsp_v2_transport_selection(selected_transport_path, selection)
            else:
                write_xsp_transport_selection(selected_transport_path, selection)
        finally:
            await client.disconnect()
        print(json.dumps(selection, allow_nan=False, indent=2, sort_keys=True))
        return 0
    selected_run = None
    selected_policy = None
    if args.mode == "directional-v1" and selected_path.exists():
        loaded_selection = json.loads(selected_path.read_text())
        if not isinstance(loaded_selection, dict):
            raise ValueError("selected XSP shadow run must be an object")
        selected_policy = xsp_profitability_policy_from_selected_run(loaded_selection)
        selected_run = loaded_selection
    selected_transport = None
    if selected_transport_path.exists():
        if args.mode == "opening-edge-v2":
            selected_transport = load_xsp_v2_transport_selection(
                selected_transport_path
            )
        elif args.mode == "opening-edge-v3":
            selected_transport = load_xsp_v3_transport_selection(
                selected_transport_path
            )
    if (
        args.mode == "opening-edge-v3"
        and selected_transport is not None
        and selected_transport.get("strategy_version") != XSP_OPENING_EDGE_V3_VERSION
    ):
        raise ValueError("selected XSP transport does not match observer mode")
    if args.mode == "opening-edge-v3" and selected_transport is not None:
        selected_policy = xsp_v3_transport_profitability_policy(
            selected_transport
        )

    from ..client import IBKRClient

    config = auxiliary_client_config(load_config(), 80)
    if selected_transport is not None and config.readonly:
        raise ValueError(
            "selected XSP transport requires an explicitly writable broker connection"
        )
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
        news_path.parent / "history" / f"{month.year:04d}-{month.month:02d}.jsonl"
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
    v3_run_start = None
    execution_observation = None
    transport_execution = None
    try:
        if args.mode == "opening-edge-v3":
            v3_spec = load_xsp_opening_edge_v3_spec()
            v3_run_start = xsp_opening_edge_v3_run_start(
                tuple(ledger.records()),
                observed_at=observed_at,
            )
            receipt = await advance_xsp_opening_edge_v3_from_ibkr(
                ledger,
                client=client,
                observed_at=observed_at,
                run_started_at=v3_run_start,
                duration_str=str(args.duration or XSP_OPENING_EDGE_V3_HISTORY_DURATION),
                news_snapshot=tuple(news),
                spec=v3_spec,
            )
            if selected_transport is not None:
                transport_execution = await advance_xsp_live_transport(
                    ledger,
                    client=client,
                    selection=selected_transport,
                    source_receipt=receipt,
                    observed_at=observed_at,
                )
        elif args.mode == "opening-edge-v2":
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
                duration_str=str(args.duration or XSP_OPENING_EDGE_V2_HISTORY_DURATION),
                news_snapshot=tuple(news),
                spec=v2_spec,
            )
            execution_observation = await advance_xsp_v2_etf_execution_observer(
                ledger,
                client=client,
                source_receipt=receipt,
                observed_at=observed_at,
            )
            if selected_transport is not None:
                transport_execution = await advance_xsp_v2_live_transport(
                    ledger,
                    client=client,
                    selection=selected_transport,
                    source_receipt=receipt,
                    observed_at=observed_at,
                )
        else:
            receipt = await advance_xsp_shadow_from_ibkr(
                ledger,
                client=client,
                observed_at=observed_at,
                duration_str=str(args.duration or XSP_DIRECTIONAL_HISTORY_DURATION),
                option_snapshots=options,
                news_snapshot=tuple(news),
                selected_run=selected_run,
            )
    finally:
        await client.disconnect()
    completed_at = datetime.now(tz=timezone.utc)
    fundamental_benchmark = (
        xsp_fundamental_defensive_benchmark(ledger)
        if args.mode == "directional-v1"
        else xsp_fundamental_defensive_benchmark(
            ledger,
            settled_pairs=xsp_opening_edge_v3_fundamental_pairs(
                tuple(ledger.records())
            ),
            prospective_evidence_mode="forward_v3_checkpoint",
        )
        if args.mode == "opening-edge-v3"
        else None
    )
    print(
        json.dumps(
            {
                **receipt,
                "mode": str(args.mode),
                "fundamental_defensive_benchmark": fundamental_benchmark,
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
                "order_authority": (
                    selected_transport["order_authority"]
                    if selected_transport is not None
                    else "none"
                ),
                "option_tape": str(option_path),
                "news_signal": str(news_path),
                "news_history": [str(path) for path in history_paths],
                "news_publications": len(news),
                "selected_run": str(selected_path),
                "selected_run_id": (
                    selected_policy.run_id if selected_policy is not None else None
                ),
                "v2_run_started_at_utc": (
                    v2_run_start.astimezone(timezone.utc).isoformat()
                    if v2_run_start is not None
                    else None
                ),
                "v3_run_started_at_utc": (
                    v3_run_start.astimezone(timezone.utc).isoformat()
                    if v3_run_start is not None
                    else None
                ),
                "execution_observation": execution_observation,
                "selected_transport": str(selected_transport_path),
                "selected_transport_id": (
                    selected_transport["selection_id"]
                    if selected_transport is not None
                    else None
                ),
                "transport_execution": transport_execution,
                "completed_at_utc": completed_at.isoformat(),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    successful_noop = (
        args.mode in {"opening-edge-v2", "opening-edge-v3"}
        and receipt.get("evaluation_status") == "CLOSED"
        and receipt.get("broker_request_skipped")
        in {"run_not_started", "closed_calendar"}
    )
    terminal_plan = (
        transport_execution.get("plan")
        if isinstance(transport_execution, dict)
        else None
    )
    successful_terminal_noop = (
        args.mode == "opening-edge-v3"
        and receipt.get("evaluation_status") == "STALE_DATA"
        and isinstance(terminal_plan, dict)
        and terminal_plan.get("source_session") in {"RTH", "CURB"}
        and terminal_plan.get("entry_window_open") is False
        and terminal_plan.get("reason") == "source_not_executable"
        and transport_execution.get("status") == "UNCHANGED"
        and transport_execution.get("submitted_orders") == 0
    )
    return (
        0
        if receipt.get("evaluation_status") == "EVALUATED"
        or successful_noop
        or successful_terminal_noop
        else 2
    )


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))
