"""Shared bar-series contract used by backtest and runtime paths."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Iterable, Literal, Mapping, Sequence, TypeVar

BarT = TypeVar("BarT")

BarSeriesTzMode = Literal[
    "utc_naive",
    "et_naive",
    "utc_aware",
    "et_aware",
    "mixed",
    "unknown",
]
BarSeriesSessionMode = Literal["rth", "full24", "mixed", "unknown"]
BarSeriesSignature = tuple[int, str | None, str | None, str | None]


@dataclass(frozen=True)
class BarSeriesMeta:
    symbol: str | None = None
    bar_size: str | None = None
    tz_mode: BarSeriesTzMode = "unknown"
    session_mode: BarSeriesSessionMode = "unknown"
    source: str = "unknown"
    source_path: str | None = None
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BarSeries(Generic[BarT]):
    bars: tuple[BarT, ...]
    meta: BarSeriesMeta = BarSeriesMeta()

    def __iter__(self):
        return iter(self.bars)

    def __len__(self) -> int:
        return len(self.bars)

    def as_list(self) -> list[BarT]:
        return list(self.bars)


def is_bar_series(value: object) -> bool:
    return isinstance(value, BarSeries)


def to_bar_series(
    bars: BarSeries[BarT] | Sequence[BarT] | Iterable[BarT],
    *,
    meta: BarSeriesMeta | None = None,
) -> BarSeries[BarT]:
    if isinstance(bars, BarSeries):
        if meta is None:
            return bars
        return BarSeries(bars=bars.bars, meta=meta)
    if isinstance(bars, tuple):
        values = bars
    elif isinstance(bars, list):
        values = tuple(bars)
    else:
        values = tuple(bars)
    return BarSeries(bars=values, meta=meta or BarSeriesMeta())


def bars_list(bars: BarSeries[BarT] | Sequence[BarT] | Iterable[BarT]) -> list[BarT]:
    if isinstance(bars, BarSeries):
        return bars.as_list()
    if isinstance(bars, list):
        return bars
    return list(bars)


def bar_series_signature(
    bars: BarSeries[BarT] | Sequence[BarT],
) -> BarSeriesSignature:
    """Return an exact, stable identity for an immutable OHLCV tape."""
    values = bars.bars if isinstance(bars, BarSeries) else bars
    if not values:
        return (0, None, None, None)

    digest = hashlib.blake2b(digest_size=16)
    for bar in values:
        ts = getattr(bar, "ts", None)
        parts = [
            ts.isoformat() if isinstance(ts, datetime) else str(ts),
            *(
                float(getattr(bar, field_name)).hex()
                for field_name in ("open", "high", "low", "close", "volume")
            ),
        ]
        digest.update("\x1f".join(parts).encode("utf-8"))
        digest.update(b"\x1e")

    first_ts = getattr(values[0], "ts", None)
    last_ts = getattr(values[-1], "ts", None)
    return (
        len(values),
        first_ts.isoformat() if isinstance(first_ts, datetime) else str(first_ts),
        last_ts.isoformat() if isinstance(last_ts, datetime) else str(last_ts),
        digest.hexdigest(),
    )
