"""Historical data access with IBKR + on-disk cache."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Iterable

from ib_insync import IB, ContFuture, Stock, util

from .models import Bar
from ..config import load_config

_FUTURE_EXCHANGES = {
    "MNQ": "CME",
    "MBT": "CME",
}


@dataclass(frozen=True)
class ContractMeta:
    symbol: str
    exchange: str
    multiplier: float
    min_tick: float


class IBKRHistoricalData:
    def __init__(self) -> None:
        self._config = load_config()
        self._ib = IB()

    def connect(self) -> None:
        if self._ib.isConnected():
            return
        self._ib.connect(
            self._config.host,
            self._config.port,
            clientId=self._config.client_id + 50,
            timeout=10,
        )

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def resolve_contract(self, symbol: str, exchange: str | None) -> tuple[object, ContractMeta]:
        if exchange is None:
            exchange = _FUTURE_EXCHANGES.get(symbol, "SMART")
        if exchange != "SMART" and symbol in _FUTURE_EXCHANGES:
            contract = ContFuture(symbol=symbol, exchange=exchange, currency="USD")
        else:
            contract = Stock(symbol=symbol, exchange="SMART", currency="USD")
        self.connect()
        qualified = self._ib.qualifyContracts(contract)
        resolved = qualified[0] if qualified else contract
        multiplier = _parse_float(getattr(resolved, "multiplier", None)) or 1.0
        min_tick = _parse_float(getattr(resolved, "minTick", None)) or 0.01
        meta = ContractMeta(symbol=symbol, exchange=exchange, multiplier=multiplier, min_tick=min_tick)
        return resolved, meta

    def load_or_fetch_bars(
        self,
        symbol: str,
        exchange: str | None,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
        cache_dir: Path,
    ) -> list[Bar]:
        cache_path = _cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return _read_cache(cache_path)
        contract, _ = self.resolve_contract(symbol, exchange)
        bars = self._fetch_bars(contract, start, end, bar_size, use_rth)
        _write_cache(cache_path, bars)
        return bars

    def _fetch_bars(
        self,
        contract: object,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
    ) -> list[Bar]:
        self.connect()
        duration = _duration_for_bar_size(bar_size)
        cursor = end
        collected: list[Bar] = []
        while cursor >= start:
            chunk = self._ib.reqHistoricalData(
                contract,
                endDateTime=cursor,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
            )
            if not chunk:
                break
            bars = [_convert_bar(bar) for bar in chunk]
            collected = bars + collected
            earliest = bars[0].ts
            cursor = earliest - timedelta(seconds=1)
            if earliest <= start:
                break
        return [bar for bar in collected if start <= bar.ts <= end]


def _duration_for_bar_size(bar_size: str) -> str:
    if bar_size.lower().startswith("1 hour"):
        return "1 M"
    if bar_size.lower().startswith("30 mins"):
        return "1 M"
    if bar_size.lower().startswith("15 mins"):
        return "1 M"
    if bar_size.lower().startswith("1 day"):
        return "1 Y"
    return "1 M"


def _cache_path(cache_dir: Path, symbol: str, start: datetime, end: datetime, bar: str, use_rth: bool) -> Path:
    tag = "rth" if use_rth else "full"
    safe_bar = bar.replace(" ", "")
    return cache_dir / symbol / f"{symbol}_{start.date()}_{end.date()}_{safe_bar}_{tag}.csv"


def _read_cache(path: Path) -> list[Bar]:
    rows: list[Bar] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                Bar(
                    ts=datetime.fromisoformat(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return rows


def _write_cache(path: Path, bars: Iterable[Bar]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "ts": bar.ts.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )


def _convert_bar(bar) -> Bar:
    dt = bar.date
    if isinstance(dt, str):
        dt = util.parseIBDatetime(dt)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.replace(tzinfo=None)
    return Bar(
        ts=dt,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume or 0),
    )


def _parse_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
