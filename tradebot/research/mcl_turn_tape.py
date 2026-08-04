"""Prospective, non-submitting CL/MCL turn-microstructure evidence tape."""

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

from ib_insync import Contract, Future, TickByTickAllLast, TickByTickBidAsk

from ..client import IBKRClient
from ..config import load_config


MCL_TURN_TAPE_SCHEMA = "mcl.turn-authenticity-microstructure-second.v1"
MCL_TURN_TAPE_LATE_SCHEMA = "mcl.turn-authenticity-microstructure-late.v1"
MCL_TURN_TAPE_GENERATION_SCHEMA = "mcl.turn-authenticity-microstructure-generation.v1"
MCL_TURN_TAPE_AUTHORITY = "prospective_observation_only_no_signal_no_orders_no_capital"
MCL_TURN_TAPE_TIMESTAMP_SEMANTICS = (
    "ib_insync_tcp_packet_receipt_utc_not_exchange_or_broker_event_time"
)
MCL_TURN_TAPE_GENERATION_PATH = Path(
    "backtests/mcl/mcl_turn_authenticity_microstructure_generation.json"
)
MCL_TURN_TAPE_STATE_DIR = Path.home() / ".local/state/tradebot/research/mcl_turn_tape"
_SYMBOLS = ("CL", "MCL")
_FLUSH_LAG_SECONDS = 3


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
    return float(number) if math.isfinite(number) else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("tick-by-tick event timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


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


def append_turn_tape_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical(record) + b"\n"
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            _repair_tail(handle)
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_turn_tape_generation(
    path: Path = MCL_TURN_TAPE_GENERATION_PATH,
    *,
    root: Path | None = None,
    recorder_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("MCL turn-tape generation must be an object")
    if payload.get("schema") != MCL_TURN_TAPE_GENERATION_SCHEMA:
        raise ValueError("MCL turn-tape generation schema drifted")
    if payload.get("authority") != MCL_TURN_TAPE_AUTHORITY:
        raise ValueError("MCL turn-tape generation authority drifted")
    if payload.get("order_authority") != "none" or payload.get("submitted_orders") != 0:
        raise ValueError("MCL turn-tape generation has order authority")
    if payload.get("timestamp_semantics") != MCL_TURN_TAPE_TIMESTAMP_SEMANTICS:
        raise ValueError("MCL turn-tape timestamp semantics drifted")
    repo = root or Path(__file__).resolve().parents[2]
    preregistration = repo / str(payload.get("preregistration_path") or "")
    if not preregistration.is_file():
        raise ValueError("MCL turn-tape preregistration is missing")
    if _sha256(preregistration) != payload.get("preregistration_sha256"):
        raise ValueError("MCL turn-tape preregistration drifted")
    recorder = recorder_path or Path(__file__).resolve()
    if _sha256(recorder) != payload.get("recorder_sha256"):
        raise ValueError("MCL turn-tape recorder drifted")
    contracts = payload.get("contracts")
    if not isinstance(contracts, dict) or set(contracts) != set(_SYMBOLS):
        raise ValueError("MCL turn-tape generation requires exact CL/MCL contracts")
    months = set()
    for symbol in _SYMBOLS:
        identity = contracts.get(symbol)
        if not isinstance(identity, dict):
            raise ValueError(f"MCL turn-tape {symbol} contract is invalid")
        if int(identity.get("con_id") or 0) <= 0:
            raise ValueError(f"MCL turn-tape {symbol} conId is invalid")
        expiry = str(identity.get("expiry") or "")
        if len(expiry) < 6 or not expiry[:6].isdigit():
            raise ValueError(f"MCL turn-tape {symbol} expiry is invalid")
        months.add(expiry[:6])
    if len(months) != 1:
        raise ValueError("MCL turn-tape contract months do not match")
    if int(payload.get("flush_lag_seconds") or 0) != _FLUSH_LAG_SECONDS:
        raise ValueError("MCL turn-tape flush law drifted")
    return payload, _sha256(source)


def _contract(identity: Mapping[str, object]) -> Future:
    return Future(
        conId=int(identity["con_id"]),
        symbol=str(identity["symbol"]),
        lastTradeDateOrContractMonth=str(identity["expiry"]),
        exchange=str(identity.get("exchange") or "NYMEX"),
        currency=str(identity.get("currency") or "USD"),
        localSymbol=str(identity["local_symbol"]),
        multiplier=str(identity.get("multiplier") or ""),
    )


def _book_values(event: Sequence[object]) -> tuple[float, float, float, float] | None:
    bid, ask, bid_size, ask_size = (_finite(value) for value in event[2:6])
    if (
        bid is None
        or ask is None
        or bid_size is None
        or ask_size is None
        or bid <= 0
        or ask < bid
        or bid_size < 0
        or ask_size < 0
    ):
        return None
    return bid, ask, bid_size, ask_size


def _book_metrics(book: Sequence[float], tick_size: float) -> dict[str, float]:
    bid, ask, bid_size, ask_size = (float(value) for value in book)
    total = bid_size + ask_size
    mid = (bid + ask) / 2.0
    microprice = (
        ((ask * bid_size) + (bid * ask_size)) / total if total > 0 else mid
    )
    imbalance = (bid_size - ask_size) / total if total > 0 else 0.0
    return {
        "mid": mid,
        "microprice": microprice,
        "spread_ticks": (ask - bid) / tick_size,
        "imbalance": imbalance,
    }


def _ohlc(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    return [float(values[0]), max(values), min(values), float(values[-1])]


def _summary(
    bid_ask: Sequence[Sequence[object]],
    trades: Sequence[Sequence[object]],
    opening: Sequence[float] | None,
    *,
    tick_size: float,
) -> tuple[dict[str, object], list[float] | None]:
    current = list(opening) if opening is not None else None
    mids: list[float] = []
    microprices: list[float] = []
    spreads: list[float] = []
    imbalances: list[float] = []
    bid_add = bid_remove = ask_add = ask_remove = 0.0
    first_mid_move: int | None = None
    prior_mid = _book_metrics(current, tick_size)["mid"] if current is not None else None
    books_by_offset: list[tuple[int, int, list[float]]] = []
    for event in sorted(bid_ask, key=lambda row: (int(row[0]), int(row[1]))):
        values = _book_values(event)
        if values is None:
            continue
        next_book = list(values)
        if current is not None:
            if next_book[0] == current[0]:
                delta = next_book[2] - current[2]
                bid_add += max(delta, 0.0)
                bid_remove += max(-delta, 0.0)
            if next_book[1] == current[1]:
                delta = next_book[3] - current[3]
                ask_add += max(delta, 0.0)
                ask_remove += max(-delta, 0.0)
        current = next_book
        metrics = _book_metrics(current, tick_size)
        for target, key in (
            (mids, "mid"),
            (microprices, "microprice"),
            (spreads, "spread_ticks"),
            (imbalances, "imbalance"),
        ):
            target.append(metrics[key])
        if prior_mid is not None and metrics["mid"] != prior_mid and first_mid_move is None:
            first_mid_move = int(event[0])
        prior_mid = metrics["mid"]
        books_by_offset.append((int(event[0]), int(event[1]), list(current)))

    signed_volume = 0.0
    trade_volume = 0.0
    trade_prices: list[float] = []
    prior_trade: float | None = None
    book_cursor = 0
    trade_book = list(opening) if opening is not None else None
    for event in sorted(trades, key=lambda row: (int(row[0]), int(row[1]))):
        offset = int(event[0])
        sequence = int(event[1])
        while (
            book_cursor < len(books_by_offset)
            and books_by_offset[book_cursor][:2] <= (offset, sequence)
        ):
            trade_book = books_by_offset[book_cursor][2]
            book_cursor += 1
        price = _finite(event[3])
        size = _finite(event[4])
        if price is None or size is None or price <= 0 or size < 0:
            continue
        side = 0
        if trade_book is not None:
            if price >= trade_book[1]:
                side = 1
            elif price <= trade_book[0]:
                side = -1
        if side == 0 and prior_trade is not None:
            side = 1 if price > prior_trade else -1 if price < prior_trade else 0
        prior_trade = price
        trade_volume += size
        signed_volume += side * size
        trade_prices.append(price)

    summary: dict[str, object] = {
        "bid_ask_events": len(bid_ask),
        "trade_events": len(trades),
        "mid_ohlc": _ohlc(mids),
        "microprice_ohlc": _ohlc(microprices),
        "spread_ticks_min_max_last": (
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
        "trade_price_ohlc": _ohlc(trade_prices),
        "trade_volume": trade_volume,
        "signed_trade_volume_proxy": signed_volume,
        "first_mid_move_offset_us": first_mid_move,
    }
    return summary, current


def _basis_summary(
    books: Mapping[str, Sequence[Sequence[object]]],
    openings: Mapping[str, Sequence[float] | None],
    *,
    tick_size: float,
) -> list[float] | None:
    current = {
        symbol: list(openings[symbol]) if openings.get(symbol) is not None else None
        for symbol in _SYMBOLS
    }
    events: dict[int, list[tuple[str, Sequence[object]]]] = {}
    for symbol in _SYMBOLS:
        for row in books.get(symbol, ()):
            events.setdefault(int(row[0]), []).append((symbol, row))
    basis: list[float] = []
    if all(current.values()):
        basis.append(
            (
                _book_metrics(current["MCL"], tick_size)["mid"]
                - _book_metrics(current["CL"], tick_size)["mid"]
            )
            / tick_size
        )
    for offset in sorted(events):
        for symbol, row in sorted(events[offset], key=lambda value: int(value[1][1])):
            values = _book_values(row)
            if values is not None:
                current[symbol] = list(values)
        if not all(current.values()):
            continue
        basis.append(
            (
                _book_metrics(current["MCL"], tick_size)["mid"]
                - _book_metrics(current["CL"], tick_size)["mid"]
            )
            / tick_size
        )
    return _ohlc(basis)


class MclTurnTapeRecorder:
    """Aggregate raw IBKR tick-by-tick rows into content-addressed seconds."""

    def __init__(
        self,
        *,
        generation_sha256: str,
        contracts: Mapping[str, Mapping[str, object]],
        output_dir: Path,
        tick_size: float = 0.01,
    ) -> None:
        self.generation_sha256 = str(generation_sha256)
        self.contracts = {symbol: dict(contracts[symbol]) for symbol in _SYMBOLS}
        self.output_dir = Path(output_dir)
        self.tick_size = float(tick_size)
        self.buckets: dict[int, dict[str, dict[str, list[list[object]]]]] = {}
        self.previous_books: dict[str, list[float] | None] = {symbol: None for symbol in _SYMBOLS}
        self.last_flushed_second = -1
        self.late_events: list[dict[str, object]] = []
        self.records = 0
        self.bid_ask_events = 0
        self.trade_events = 0
        self.event_sequences = {symbol: 0 for symbol in _SYMBOLS}

    def ingest_tickers(self, tickers: Sequence[object]) -> None:
        seen: set[int] = set()
        for ticker in tickers:
            if id(ticker) in seen:
                continue
            seen.add(id(ticker))
            contract = getattr(ticker, "contract", None)
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            if symbol not in _SYMBOLS:
                continue
            for event in tuple(getattr(ticker, "tickByTicks", ()) or ()):
                self.ingest(symbol, event)

    def ingest(self, symbol: str, event: object) -> None:
        symbol = str(symbol).strip().upper()
        if symbol not in _SYMBOLS:
            raise ValueError(f"unsupported turn-tape symbol: {symbol!r}")
        event_time = _utc(getattr(event, "time"))
        second = int(event_time.timestamp())
        offset = int(event_time.microsecond)
        sequence = self.event_sequences[symbol]
        self.event_sequences[symbol] += 1
        if isinstance(event, TickByTickBidAsk):
            row = [
                offset,
                sequence,
                _finite(event.bidPrice),
                _finite(event.askPrice),
                _finite(event.bidSize),
                _finite(event.askSize),
                bool(event.tickAttribBidAsk.bidPastLow),
                bool(event.tickAttribBidAsk.askPastHigh),
            ]
            kind = "bid_ask"
            self.bid_ask_events += 1
        elif isinstance(event, TickByTickAllLast):
            row = [
                offset,
                sequence,
                int(event.tickType),
                _finite(event.price),
                _finite(event.size),
                bool(event.tickAttribLast.pastLimit),
                bool(event.tickAttribLast.unreported),
            ]
            kind = "trades"
            self.trade_events += 1
        else:
            return
        if second <= self.last_flushed_second:
            self.late_events.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "receipt_time_utc": _iso(event_time),
                    "row": row[2:],
                }
            )
            return
        bucket = self.buckets.setdefault(
            second,
            {
                value: {"bid_ask": [], "trades": []}
                for value in _SYMBOLS
            },
        )
        bucket[symbol][kind].append(row)

    def drain(
        self,
        *,
        now: datetime,
        market_data_types: Mapping[str, int | None],
        force: bool = False,
    ) -> list[dict[str, object]]:
        now_utc = _utc(now)
        cutoff = int(now_utc.timestamp()) if force else int(now_utc.timestamp()) - _FLUSH_LAG_SECONDS
        flushed: list[dict[str, object]] = []
        for second in sorted(value for value in self.buckets if value <= cutoff):
            bucket = self.buckets.pop(second)
            record = self._record(second, bucket, market_data_types, now_utc)
            append_turn_tape_record(
                self.output_dir / f"{datetime.fromtimestamp(second, timezone.utc).date()}.jsonl",
                record,
            )
            self.last_flushed_second = max(self.last_flushed_second, second)
            self.records += 1
            flushed.append(record)
        if self.late_events:
            record = {
                "schema": MCL_TURN_TAPE_LATE_SCHEMA,
                "kind": "late_events",
                "authority": MCL_TURN_TAPE_AUTHORITY,
                "timestamp_semantics": MCL_TURN_TAPE_TIMESTAMP_SEMANTICS,
                "generation_sha256": self.generation_sha256,
                "recorded_at_utc": _iso(now_utc),
                "valid_evidence": False,
                "events": self.late_events,
                "submitted_orders": 0,
            }
            record["record_id"] = hashlib.sha256(_canonical(record)).hexdigest()
            append_turn_tape_record(
                self.output_dir / f"{now_utc.date()}.jsonl",
                record,
            )
            self.late_events = []
            flushed.append(record)
        return flushed

    def _record(
        self,
        second: int,
        bucket: Mapping[str, Mapping[str, list[list[object]]]],
        market_data_types: Mapping[str, int | None],
        recorded_at: datetime,
    ) -> dict[str, object]:
        openings = {
            symbol: list(book) if book is not None else None
            for symbol, book in self.previous_books.items()
        }
        books: dict[str, object] = {}
        summaries: dict[str, dict[str, object]] = {}
        raw_books: dict[str, Sequence[Sequence[object]]] = {}
        for symbol in _SYMBOLS:
            rows = bucket[symbol]
            bid_ask = sorted(rows["bid_ask"], key=lambda row: (int(row[0]), int(row[1])))
            trades = sorted(rows["trades"], key=lambda row: (int(row[0]), int(row[1])))
            summary, closing = _summary(
                bid_ask,
                trades,
                openings[symbol],
                tick_size=self.tick_size,
            )
            self.previous_books[symbol] = closing
            summaries[symbol] = summary
            raw_books[symbol] = bid_ask
            books[symbol] = {
                "contract": self.contracts[symbol],
                "opening_book": openings[symbol],
                "bid_ask": bid_ask,
                "trades": trades,
                "summary": summary,
            }
        cl_first = summaries["CL"]["first_mid_move_offset_us"]
        mcl_first = summaries["MCL"]["first_mid_move_offset_us"]
        lead_us = (
            int(mcl_first) - int(cl_first)
            if cl_first is not None and mcl_first is not None
            else None
        )
        leader = (
            "CL"
            if lead_us is not None and lead_us > 0
            else "MCL"
            if lead_us is not None and lead_us < 0
            else "SIMULTANEOUS"
            if lead_us == 0
            else None
        )
        md_types = {symbol: market_data_types.get(symbol) for symbol in _SYMBOLS}
        record: dict[str, object] = {
            "schema": MCL_TURN_TAPE_SCHEMA,
            "kind": "second",
            "authority": MCL_TURN_TAPE_AUTHORITY,
            "timestamp_semantics": MCL_TURN_TAPE_TIMESTAMP_SEMANTICS,
            "generation_sha256": self.generation_sha256,
            "bucket_start_utc": datetime.fromtimestamp(second, timezone.utc).isoformat(),
            "recorded_at_utc": _iso(recorded_at),
            "market_data_types": md_types,
            "valid_evidence": bool(
                all(md_types[symbol] == 1 for symbol in _SYMBOLS)
                and all(self.previous_books[symbol] is not None for symbol in _SYMBOLS)
                and any(
                    bucket[symbol][kind]
                    for symbol in _SYMBOLS
                    for kind in ("bid_ask", "trades")
                )
            ),
            "books": books,
            "cross_book": {
                "basis_ticks_ohlc": _basis_summary(
                    raw_books,
                    openings,
                    tick_size=self.tick_size,
                ),
                "first_mid_move_leader": leader,
                "mcl_minus_cl_first_mid_move_us": lead_us,
            },
            "submitted_orders": 0,
        }
        record["record_id"] = hashlib.sha256(_canonical(record)).hexdigest()
        return record


def _identity_matches(contract: Contract, identity: Mapping[str, object]) -> bool:
    return bool(
        int(getattr(contract, "conId", 0) or 0) == int(identity["con_id"])
        and str(getattr(contract, "symbol", "") or "").strip().upper()
        == str(identity["symbol"]).strip().upper()
        and str(getattr(contract, "localSymbol", "") or "").strip().upper()
        == str(identity["local_symbol"]).strip().upper()
        and str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
        == str(identity["expiry"])
    )


async def run_turn_tape(
    *,
    generation_path: Path,
    output_dir: Path,
    duration_sec: float = 0.0,
    status_sec: float = 60.0,
) -> None:
    generation, generation_sha = load_turn_tape_generation(generation_path)
    identities = generation["contracts"]
    assert isinstance(identities, dict)
    contracts = {
        symbol: _contract(identities[symbol])
        for symbol in _SYMBOLS
    }
    client = IBKRClient(load_config())
    await client.connect()
    market_tickers = {
        symbol: await client.ensure_ticker(
            contract,
            owner="mcl-turn-tape",
            generic_ticks="233",
        )
        for symbol, contract in contracts.items()
    }
    for symbol, ticker in market_tickers.items():
        if not _identity_matches(ticker.contract, identities[symbol]):
            raise RuntimeError(f"{symbol} turn-tape contract identity drifted")
    subscriptions: list[tuple[Contract, str, object]] = []
    for symbol, contract in contracts.items():
        for kind in ("BidAsk", "AllLast"):
            ticker = await client.subscribe_tick_by_tick(contract, kind)
            if not _identity_matches(ticker.contract, identities[symbol]):
                raise RuntimeError(f"{symbol} {kind} subscription identity drifted")
            subscriptions.append((ticker.contract, kind, ticker))
    recorder = MclTurnTapeRecorder(
        generation_sha256=generation_sha,
        contracts=identities,
        output_dir=output_dir,
        tick_size=float(generation.get("tick_size") or 0.01),
    )
    tickers = list({id(value[2]): value[2] for value in subscriptions}.values())

    def observe(ticker: object) -> None:
        recorder.ingest_tickers((ticker,))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    for ticker in tickers:
        ticker.updateEvent += observe
        observe(ticker)
    started = time.monotonic()
    next_status = started
    try:
        while not stop.is_set():
            await asyncio.sleep(0.25)
            if not client.is_connected:
                raise ConnectionError("MCL turn-tape IBKR stream disconnected")
            now = datetime.now(timezone.utc)
            recorder.drain(
                now=now,
                market_data_types={
                    symbol: (
                        int(ticker.marketDataType)
                        if getattr(ticker, "marketDataType", None) is not None
                        else None
                    )
                    for symbol, ticker in market_tickers.items()
                },
            )
            elapsed = time.monotonic() - started
            if elapsed >= next_status:
                print(
                    json.dumps(
                        {
                            "schema": "mcl.turn-authenticity-microstructure-status.v1",
                            "authority": MCL_TURN_TAPE_AUTHORITY,
                            "generation_sha256": generation_sha,
                            "records": recorder.records,
                            "bid_ask_events": recorder.bid_ask_events,
                            "trade_events": recorder.trade_events,
                            "market_data_types": {
                                symbol: getattr(ticker, "marketDataType", None)
                                for symbol, ticker in market_tickers.items()
                            },
                            "submitted_orders": 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_status = elapsed + max(1.0, float(status_sec))
            if duration_sec > 0 and elapsed >= float(duration_sec):
                break
    finally:
        for ticker in tickers:
            ticker.updateEvent -= observe
            observe(ticker)
        recorder.drain(
            now=datetime.now(timezone.utc),
            market_data_types={
                symbol: (
                    int(ticker.marketDataType)
                    if getattr(ticker, "marketDataType", None) is not None
                    else None
                )
                for symbol, ticker in market_tickers.items()
            },
            force=True,
        )
        for contract, kind, _ticker in subscriptions:
            try:
                client.unsubscribe_tick_by_tick(contract, kind)
            except Exception:
                pass
        for ticker in market_tickers.values():
            client.release_ticker(int(ticker.contract.conId or 0), owner="mcl-turn-tape")
        await client.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=MCL_TURN_TAPE_GENERATION_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("MCL_TURN_TAPE_DIR", MCL_TURN_TAPE_STATE_DIR)),
    )
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--status-sec", type=float, default=60.0)
    args = parser.parse_args(argv)
    asyncio.run(
        run_turn_tape(
            generation_path=args.generation,
            output_dir=args.out_dir,
            duration_sec=max(0.0, args.duration_sec),
            status_sec=max(1.0, args.status_sec),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
