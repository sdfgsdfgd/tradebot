"""Prospective standard-L1 pressure tape for SPY and the XSP cash transports."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from ib_insync import Stock

from ..client import IBKRClient
from ..config import load_config
from ..engines.market import xsp_session_label_et, xsp_trading_date


XSP_PRESSURE_TAPE_SCHEMA = "xsp.pressure-microstructure-second.v1"
XSP_PRESSURE_TAPE_LATE_SCHEMA = "xsp.pressure-microstructure-late.v1"
XSP_PRESSURE_TAPE_GENERATION_SCHEMA = "xsp.pressure-microstructure-generation.v1"
XSP_PRESSURE_TAPE_AUTHORITY = (
    "prospective_microstructure_observation_only_no_signal_no_outcomes_no_orders_no_capital"
)
XSP_PRESSURE_TAPE_TIMESTAMP_SEMANTICS = (
    "local_ib_insync_update_callback_receipt_utc_not_exchange_or_broker_event_time"
)
XSP_PRESSURE_TAPE_GENERATION_PATH = Path(
    "backtests/xsp/opening_edge_v3_pressure_tape_generation.json"
)
XSP_PRESSURE_TAPE_STATE_DIR = (
    Path.home() / ".local/state/tradebot/research/xsp_pressure_tape"
)
XSP_PRESSURE_TAPE_SYMBOLS = ("SPY", "UPRO", "SPXU")
_FLUSH_LAG_SECONDS = 3
_MAX_SNAPSHOT_AGE_SECONDS = 5.0


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) and abs(number) < 1e300 else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("XSP pressure-tape timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _repair_tail(handle) -> None:
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end <= 0:
        return
    handle.seek(end - 1)
    if handle.read(1) == b"\n":
        return
    cursor = end
    while cursor > 0:
        size = min(8192, cursor)
        cursor -= size
        handle.seek(cursor)
        newline = handle.read(size).rfind(b"\n")
        if newline >= 0:
            cursor += newline + 1
            break
    handle.truncate(cursor)


def append_xsp_pressure_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            _repair_tail(handle)
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_xsp_pressure_tape_generation(
    path: Path = XSP_PRESSURE_TAPE_GENERATION_PATH,
    *,
    root: Path | None = None,
    recorder_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("XSP pressure-tape generation must be an object")
    if payload.get("schema") != XSP_PRESSURE_TAPE_GENERATION_SCHEMA:
        raise ValueError("XSP pressure-tape generation schema drifted")
    if payload.get("authority") != XSP_PRESSURE_TAPE_AUTHORITY:
        raise ValueError("XSP pressure-tape authority drifted")
    if payload.get("timestamp_semantics") != XSP_PRESSURE_TAPE_TIMESTAMP_SEMANTICS:
        raise ValueError("XSP pressure-tape timestamp semantics drifted")
    if payload.get("order_authority") != "none" or payload.get("submitted_orders") != 0:
        raise ValueError("XSP pressure-tape generation has order authority")
    if payload.get("standard_l1_only") is not True:
        raise ValueError("XSP pressure tape requires standard L1")
    if payload.get("tick_by_tick_subscriptions") != 0:
        raise ValueError("XSP pressure tape may not consume tick-by-tick capacity")
    if int(payload.get("flush_lag_seconds") or 0) != _FLUSH_LAG_SECONDS:
        raise ValueError("XSP pressure-tape flush law drifted")
    if float(payload.get("max_snapshot_age_seconds") or 0.0) != _MAX_SNAPSHOT_AGE_SECONDS:
        raise ValueError("XSP pressure-tape freshness law drifted")
    if tuple(payload.get("symbols") or ()) != XSP_PRESSURE_TAPE_SYMBOLS:
        raise ValueError("XSP pressure-tape symbol order drifted")
    try:
        eligible_start = datetime.fromisoformat(
            str(payload.get("eligible_start_utc") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("XSP pressure-tape eligible start is invalid") from exc
    if eligible_start.tzinfo is None:
        raise ValueError("XSP pressure-tape eligible start must be timezone-aware")

    repo = root or Path(__file__).resolve().parents[2]
    preregistration = repo / str(payload.get("preregistration_path") or "")
    if not preregistration.is_file():
        raise ValueError("XSP pressure-tape preregistration is missing")
    if _sha256(preregistration) != payload.get("preregistration_sha256"):
        raise ValueError("XSP pressure-tape preregistration drifted")
    recorder = recorder_path or Path(__file__).resolve()
    if _sha256(recorder) != payload.get("recorder_sha256"):
        raise ValueError("XSP pressure-tape recorder drifted")

    contracts = payload.get("contracts")
    if not isinstance(contracts, dict) or tuple(contracts) != XSP_PRESSURE_TAPE_SYMBOLS:
        raise ValueError("XSP pressure-tape contracts drifted")
    for symbol in XSP_PRESSURE_TAPE_SYMBOLS:
        identity = contracts.get(symbol)
        if (
            not isinstance(identity, dict)
            or str(identity.get("symbol") or "").upper() != symbol
            or str(identity.get("sec_type") or "").upper() != "STK"
            or int(identity.get("con_id") or 0) <= 0
            or str(identity.get("currency") or "").upper() != "USD"
            or float(identity.get("tick_size") or 0.0) <= 0
        ):
            raise ValueError(f"XSP pressure-tape {symbol} identity is invalid")
    return payload, _sha256(source)


def _contract(identity: Mapping[str, object]) -> Stock:
    return Stock(
        conId=int(identity["con_id"]),
        symbol=str(identity["symbol"]),
        exchange=str(identity.get("exchange") or "SMART"),
        primaryExchange=str(identity.get("primary_exchange") or "ARCA"),
        currency=str(identity.get("currency") or "USD"),
        localSymbol=str(identity.get("local_symbol") or identity["symbol"]),
    )


def _identity_matches(contract: object, identity: Mapping[str, object]) -> bool:
    return bool(
        int(getattr(contract, "conId", 0) or 0) == int(identity["con_id"])
        and str(getattr(contract, "symbol", "") or "").upper()
        == str(identity["symbol"]).upper()
        and str(getattr(contract, "secType", "") or "").upper() == "STK"
        and str(getattr(contract, "currency", "") or "").upper() == "USD"
    )


def _snapshot(ticker: object) -> list[float | None]:
    return [
        _finite(getattr(ticker, "bid", None)),
        _finite(getattr(ticker, "ask", None)),
        _finite(getattr(ticker, "bidSize", None)),
        _finite(getattr(ticker, "askSize", None)),
        _finite(getattr(ticker, "last", None)),
        _finite(getattr(ticker, "lastSize", None)),
        _finite(getattr(ticker, "volume", None)),
    ]


def _valid_book(snapshot: Sequence[float | None] | None) -> bool:
    if snapshot is None or len(snapshot) < 4:
        return False
    bid, ask, bid_size, ask_size = snapshot[:4]
    return bool(
        bid is not None
        and ask is not None
        and bid_size is not None
        and ask_size is not None
        and bid > 0
        and ask >= bid
        and bid_size >= 0
        and ask_size >= 0
    )


def _book_metrics(snapshot: Sequence[float | None]) -> dict[str, float]:
    bid, ask, bid_size, ask_size = (float(value) for value in snapshot[:4])
    total = bid_size + ask_size
    mid = (bid + ask) / 2.0
    return {
        "mid": mid,
        "microprice": (
            ((ask * bid_size) + (bid * ask_size)) / total if total > 0 else mid
        ),
        "spread_bps": ((ask - bid) / mid) * 10_000.0,
        "imbalance": (bid_size - ask_size) / total if total > 0 else 0.0,
    }


def _ohlc(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    return [float(values[0]), max(values), min(values), float(values[-1])]


def _book_summary(
    events: Sequence[Sequence[object]],
    opening: Sequence[float | None] | None,
) -> tuple[dict[str, object], list[float | None] | None]:
    current = list(opening) if opening is not None else None
    opening_mid = _book_metrics(current)["mid"] if _valid_book(current) else None
    mids: list[float] = []
    microprices: list[float] = []
    spreads: list[float] = []
    imbalances: list[float] = []
    last_prices: list[float] = []
    volumes: list[float] = []
    bid_add = bid_remove = ask_add = ask_remove = 0.0
    first_mid_move: int | None = None
    quote_changes = 0

    if _valid_book(current):
        metrics = _book_metrics(current)
        mids.append(metrics["mid"])
        microprices.append(metrics["microprice"])
        spreads.append(metrics["spread_bps"])
        imbalances.append(metrics["imbalance"])
    if current is not None and len(current) >= 7:
        if current[4] is not None and current[4] > 0:
            last_prices.append(float(current[4]))
        if current[6] is not None and current[6] >= 0:
            volumes.append(float(current[6]))

    for row in sorted(events, key=lambda value: (int(value[0]), int(value[1]))):
        next_snapshot = list(row[2:9])
        if current is not None and _valid_book(current) and _valid_book(next_snapshot):
            if next_snapshot[0] == current[0]:
                delta = float(next_snapshot[2]) - float(current[2])
                bid_add += max(delta, 0.0)
                bid_remove += max(-delta, 0.0)
            if next_snapshot[1] == current[1]:
                delta = float(next_snapshot[3]) - float(current[3])
                ask_add += max(delta, 0.0)
                ask_remove += max(-delta, 0.0)
        current = next_snapshot
        if _valid_book(current):
            metrics = _book_metrics(current)
            quote_changes += 1
            mids.append(metrics["mid"])
            microprices.append(metrics["microprice"])
            spreads.append(metrics["spread_bps"])
            imbalances.append(metrics["imbalance"])
            if (
                opening_mid is not None
                and first_mid_move is None
                and metrics["mid"] != opening_mid
            ):
                first_mid_move = int(row[0])
        if current[4] is not None and current[4] > 0:
            last_prices.append(float(current[4]))
        if current[6] is not None and current[6] >= 0:
            volumes.append(float(current[6]))

    volume_delta = (
        max(0.0, volumes[-1] - volumes[0]) if len(volumes) >= 2 else 0.0
    )
    return (
        {
            "update_events": len(events),
            "valid_book_updates": quote_changes,
            "mid_ohlc": _ohlc(mids),
            "microprice_ohlc": _ohlc(microprices),
            "spread_bps_min_max_last": (
                [min(spreads), max(spreads), spreads[-1]] if spreads else None
            ),
            "imbalance_open_min_max_close": (
                [imbalances[0], min(imbalances), max(imbalances), imbalances[-1]]
                if imbalances
                else None
            ),
            "same_price_size_proxy": {
                "bid_add": bid_add,
                "bid_remove": bid_remove,
                "ask_add": ask_add,
                "ask_remove": ask_remove,
            },
            "last_price_ohlc": _ohlc(last_prices),
            "cumulative_volume_delta": volume_delta,
            "first_mid_move_offset_us": first_mid_move,
        },
        current,
    )


def _displacement_bps(summary: Mapping[str, object]) -> float | None:
    values = summary.get("mid_ohlc")
    if not isinstance(values, list) or len(values) != 4:
        return None
    opening, closing = float(values[0]), float(values[3])
    return ((closing / opening) - 1.0) * 10_000.0 if opening > 0 else None


class XspPressureTapeRecorder:
    """Aggregate changed standard-L1 snapshots into restart-safe UTC seconds."""

    def __init__(
        self,
        *,
        generation_sha256: str,
        contracts: Mapping[str, Mapping[str, object]],
        output_dir: Path,
        eligible_start_utc: datetime | None = None,
        max_snapshot_age_seconds: float = _MAX_SNAPSHOT_AGE_SECONDS,
    ) -> None:
        self.generation_sha256 = str(generation_sha256)
        self.contracts = {
            symbol: dict(contracts[symbol]) for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        self.output_dir = Path(output_dir)
        self.eligible_start_utc = _utc(
            eligible_start_utc
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        self.max_snapshot_age_seconds = float(max_snapshot_age_seconds)
        if self.max_snapshot_age_seconds <= 0:
            raise ValueError("XSP pressure-tape maximum snapshot age must be positive")
        self.buckets: dict[int, dict[str, list[list[object]]]] = {}
        self.previous_snapshots: dict[str, list[float | None] | None] = {
            symbol: None for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        self.last_signatures: dict[str, tuple[float | None, ...] | None] = {
            symbol: None for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        self.last_update_times: dict[str, float | None] = {
            symbol: None for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        self.sequences = {symbol: 0 for symbol in XSP_PRESSURE_TAPE_SYMBOLS}
        self.last_flushed_second = -1
        self.late_events: list[dict[str, object]] = []
        self.records = 0
        self.update_events = 0

    def heartbeat(self, observed_at: datetime) -> None:
        second = int(_utc(observed_at).timestamp())
        if second > self.last_flushed_second:
            self.buckets.setdefault(
                second,
                {symbol: [] for symbol in XSP_PRESSURE_TAPE_SYMBOLS},
            )

    def ingest_ticker(
        self,
        symbol: str,
        ticker: object,
        *,
        received_at: datetime | None = None,
    ) -> None:
        self.ingest(
            symbol,
            _snapshot(ticker),
            received_at=received_at or datetime.now(timezone.utc),
        )

    def ingest(
        self,
        symbol: str,
        snapshot: Sequence[float | None],
        *,
        received_at: datetime,
    ) -> None:
        symbol = str(symbol).strip().upper()
        if symbol not in XSP_PRESSURE_TAPE_SYMBOLS:
            raise ValueError(f"unsupported XSP pressure-tape symbol: {symbol!r}")
        normalized = tuple(_finite(value) for value in snapshot[:7])
        if len(normalized) != 7 or normalized == self.last_signatures[symbol]:
            return
        self.last_signatures[symbol] = normalized
        observed = _utc(received_at)
        second = int(observed.timestamp())
        offset = observed.microsecond
        sequence = self.sequences[symbol]
        self.sequences[symbol] += 1
        row: list[object] = [offset, sequence, *normalized]
        if second <= self.last_flushed_second:
            self.late_events.append(
                {
                    "symbol": symbol,
                    "receipt_time_utc": observed.isoformat(),
                    "snapshot": list(normalized),
                }
            )
            return
        self.heartbeat(observed)
        self.buckets[second][symbol].append(row)
        self.update_events += 1

    def drain(
        self,
        *,
        now: datetime,
        market_data_types: Mapping[str, int | None],
        force: bool = False,
    ) -> list[dict[str, object]]:
        now_utc = _utc(now)
        cutoff = (
            int(now_utc.timestamp())
            if force
            else int(now_utc.timestamp()) - _FLUSH_LAG_SECONDS
        )
        flushed: list[dict[str, object]] = []
        for second in sorted(value for value in self.buckets if value <= cutoff):
            bucket = self.buckets.pop(second)
            record = self._record(second, bucket, market_data_types, now_utc)
            append_xsp_pressure_record(
                self.output_dir
                / f"{datetime.fromtimestamp(second, timezone.utc).date()}.jsonl",
                record,
            )
            self.last_flushed_second = max(self.last_flushed_second, second)
            self.records += 1
            flushed.append(record)
        if self.late_events:
            late: dict[str, object] = {
                "schema": XSP_PRESSURE_TAPE_LATE_SCHEMA,
                "kind": "late_events",
                "authority": XSP_PRESSURE_TAPE_AUTHORITY,
                "timestamp_semantics": XSP_PRESSURE_TAPE_TIMESTAMP_SEMANTICS,
                "generation_sha256": self.generation_sha256,
                "recorded_at_utc": now_utc.isoformat(),
                "valid_evidence": False,
                "events": self.late_events,
                "submitted_orders": 0,
            }
            late["record_id"] = hashlib.sha256(_canonical(late)).hexdigest()
            append_xsp_pressure_record(
                self.output_dir / f"{now_utc.date()}.jsonl",
                late,
            )
            self.late_events = []
            flushed.append(late)
        return flushed

    def _record(
        self,
        second: int,
        bucket: Mapping[str, Sequence[Sequence[object]]],
        market_data_types: Mapping[str, int | None],
        recorded_at: datetime,
    ) -> dict[str, object]:
        openings = {
            symbol: list(value) if value is not None else None
            for symbol, value in self.previous_snapshots.items()
        }
        books: dict[str, object] = {}
        summaries: dict[str, dict[str, object]] = {}
        for symbol in XSP_PRESSURE_TAPE_SYMBOLS:
            events = sorted(
                bucket.get(symbol, ()),
                key=lambda row: (int(row[0]), int(row[1])),
            )
            summary, closing = _book_summary(events, openings[symbol])
            if events:
                self.last_update_times[symbol] = second + (
                    int(events[-1][0]) / 1_000_000.0
                )
            last_update = self.last_update_times[symbol]
            snapshot_age = (
                max(0.0, (second + 1.0) - last_update)
                if last_update is not None
                else None
            )
            summary["snapshot_age_seconds_at_close"] = snapshot_age
            self.previous_snapshots[symbol] = closing
            summaries[symbol] = summary
            books[symbol] = {
                "contract": self.contracts[symbol],
                "opening_snapshot": openings[symbol],
                "updates": events,
                "summary": summary,
            }

        displacements = {
            symbol: _displacement_bps(summaries[symbol])
            for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        directional = {
            "SPY": displacements["SPY"],
            "UPRO": displacements["UPRO"],
            "INVERSE_SPXU": (
                -float(displacements["SPXU"])
                if displacements["SPXU"] is not None
                else None
            ),
        }
        votes = {"up": 0, "down": 0, "flat": 0}
        for value in directional.values():
            if value is None or value == 0:
                votes["flat"] += 1
            elif value > 0:
                votes["up"] += 1
            else:
                votes["down"] += 1
        full_alignment = (
            "up"
            if votes["up"] == 3
            else "down"
            if votes["down"] == 3
            else None
        )
        transport_response = (
            (
                float(displacements["UPRO"])
                - float(displacements["SPXU"])
            )
            / 2.0
            if displacements["UPRO"] is not None
            and displacements["SPXU"] is not None
            else None
        )
        first_moves = {
            symbol: summaries[symbol]["first_mid_move_offset_us"]
            for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        resolved_offsets = [
            int(value) for value in first_moves.values() if value is not None
        ]
        first_leaders = (
            sorted(
                symbol
                for symbol, value in first_moves.items()
                if value is not None and int(value) == min(resolved_offsets)
            )
            if resolved_offsets
            else []
        )
        book_pressure_close: dict[str, float | None] = {}
        for symbol in XSP_PRESSURE_TAPE_SYMBOLS:
            values = summaries[symbol]["imbalance_open_min_max_close"]
            close = float(values[3]) if isinstance(values, list) else None
            book_pressure_close[
                "INVERSE_SPXU" if symbol == "SPXU" else symbol
            ] = -close if symbol == "SPXU" and close is not None else close

        bucket_start = datetime.fromtimestamp(second, timezone.utc)
        md_types = {
            symbol: market_data_types.get(symbol)
            for symbol in XSP_PRESSURE_TAPE_SYMBOLS
        }
        valid = bool(
            bucket_start >= self.eligible_start_utc
            and all(md_types[symbol] == 1 for symbol in XSP_PRESSURE_TAPE_SYMBOLS)
            and all(
                _valid_book(self.previous_snapshots[symbol])
                for symbol in XSP_PRESSURE_TAPE_SYMBOLS
            )
            and all(
                summaries[symbol]["snapshot_age_seconds_at_close"] is not None
                and float(summaries[symbol]["snapshot_age_seconds_at_close"])
                <= self.max_snapshot_age_seconds
                for symbol in XSP_PRESSURE_TAPE_SYMBOLS
            )
        )
        record: dict[str, object] = {
            "schema": XSP_PRESSURE_TAPE_SCHEMA,
            "kind": "second",
            "authority": XSP_PRESSURE_TAPE_AUTHORITY,
            "timestamp_semantics": XSP_PRESSURE_TAPE_TIMESTAMP_SEMANTICS,
            "generation_sha256": self.generation_sha256,
            "eligible_start_utc": self.eligible_start_utc.isoformat(),
            "eligible_treatment": bucket_start >= self.eligible_start_utc,
            "bucket_start_utc": bucket_start.isoformat(),
            "recorded_at_utc": recorded_at.isoformat(),
            "session": xsp_session_label_et(bucket_start),
            "trading_date": (
                xsp_trading_date(bucket_start).isoformat()
                if xsp_trading_date(bucket_start) is not None
                else None
            ),
            "market_data_types": md_types,
            "valid_evidence": valid,
            "books": books,
            "cross_book": {
                "displacement_bps": directional,
                "alignment_votes": votes,
                "full_alignment_direction": full_alignment,
                "transport_response_bps": transport_response,
                "cash_minus_transport_bps": (
                    float(displacements["SPY"]) - transport_response
                    if displacements["SPY"] is not None
                    and transport_response is not None
                    else None
                ),
                "first_mid_move_offsets_us": first_moves,
                "first_mid_move_leaders": first_leaders,
                "direction_normalized_book_pressure_close": book_pressure_close,
            },
            "submitted_orders": 0,
        }
        record["record_id"] = hashlib.sha256(_canonical(record)).hexdigest()
        return record


async def run_xsp_pressure_tape(
    *,
    generation_path: Path,
    output_dir: Path,
    duration_sec: float = 0.0,
    status_sec: float = 60.0,
) -> None:
    generation, generation_sha = load_xsp_pressure_tape_generation(generation_path)
    identities = generation["contracts"]
    assert isinstance(identities, dict)
    client = IBKRClient(load_config())
    await client.connect()
    tickers = {
        symbol: await client.ensure_ticker(
            _contract(identities[symbol]),
            owner="xsp-pressure-tape",
            generic_ticks="233",
        )
        for symbol in XSP_PRESSURE_TAPE_SYMBOLS
    }
    for symbol, ticker in tickers.items():
        if not _identity_matches(ticker.contract, identities[symbol]):
            raise RuntimeError(f"{symbol} XSP pressure-tape identity drifted")

    recorder = XspPressureTapeRecorder(
        generation_sha256=generation_sha,
        contracts=identities,
        output_dir=output_dir,
        eligible_start_utc=datetime.fromisoformat(
            str(generation["eligible_start_utc"]).replace("Z", "+00:00")
        ),
        max_snapshot_age_seconds=float(
            generation.get("max_snapshot_age_seconds")
            or _MAX_SNAPSHOT_AGE_SECONDS
        ),
    )
    callbacks = {}
    for symbol, ticker in tickers.items():
        def observe(value: object, key: str = symbol) -> None:
            recorder.ingest_ticker(key, value)

        callbacks[symbol] = observe
        ticker.updateEvent += observe
        observe(ticker)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    started = time.monotonic()
    next_status = started
    try:
        while not stop.is_set():
            await asyncio.sleep(0.25)
            if not client.is_connected:
                raise ConnectionError("XSP pressure-tape IBKR stream disconnected")
            now = datetime.now(timezone.utc)
            recorder.heartbeat(now)
            market_data_types = {
                symbol: (
                    int(ticker.marketDataType)
                    if getattr(ticker, "marketDataType", None) is not None
                    else None
                )
                for symbol, ticker in tickers.items()
            }
            recorder.drain(now=now, market_data_types=market_data_types)
            elapsed = time.monotonic() - started
            if elapsed >= next_status:
                print(
                    json.dumps(
                        {
                            "schema": "xsp.pressure-microstructure-status.v1",
                            "authority": XSP_PRESSURE_TAPE_AUTHORITY,
                            "generation_sha256": generation_sha,
                            "records": recorder.records,
                            "update_events": recorder.update_events,
                            "market_data_types": market_data_types,
                            "submitted_orders": 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_status = elapsed + max(1.0, float(status_sec))
            if duration_sec > 0 and elapsed >= duration_sec:
                break
    finally:
        for symbol, ticker in tickers.items():
            ticker.updateEvent -= callbacks[symbol]
            recorder.ingest_ticker(symbol, ticker)
        now = datetime.now(timezone.utc)
        recorder.heartbeat(now)
        recorder.drain(
            now=now,
            market_data_types={
                symbol: (
                    int(ticker.marketDataType)
                    if getattr(ticker, "marketDataType", None) is not None
                    else None
                )
                for symbol, ticker in tickers.items()
            },
            force=True,
        )
        for ticker in tickers.values():
            client.release_ticker(
                int(ticker.contract.conId or 0),
                owner="xsp-pressure-tape",
            )
        await client.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation",
        type=Path,
        default=XSP_PRESSURE_TAPE_GENERATION_PATH,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get("XSP_PRESSURE_TAPE_DIR", XSP_PRESSURE_TAPE_STATE_DIR)
        ),
    )
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--status-sec", type=float, default=60.0)
    args = parser.parse_args(argv)
    asyncio.run(
        run_xsp_pressure_tape(
            generation_path=args.generation,
            output_dir=args.out_dir,
            duration_sec=max(0.0, args.duration_sec),
            status_sec=max(1.0, args.status_sec),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
