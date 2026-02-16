"""Shared bar-series contract used by backtest and runtime paths."""
from __future__ import annotations

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
