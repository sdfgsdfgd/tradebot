"""Optional native-minute history transport for the XSP opening bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from ..backtest.models import Bar
from ..chart_data.history import normalize_bars_to_close
from ..time_utils import to_utc_naive
from ..utils.bar_utils import trim_incomplete_last_bar


@dataclass(frozen=True, slots=True)
class XspRthOneMinuteContext:
    raw_spy: tuple[Bar, ...] = ()
    spy: tuple[Bar, ...] = ()
    raw_xsp: tuple[Bar, ...] = ()
    xsp: tuple[Bar, ...] = ()

    def ready(
        self,
        *,
        required: bool,
        observed_at: datetime,
        freshness_seconds: float,
    ) -> bool:
        if not required:
            return True
        latest_spy_utc = (
            to_utc_naive(self.spy[-1].ts, naive_ts_mode="et").replace(
                tzinfo=timezone.utc
            )
            if self.spy
            else None
        )
        age_seconds = (
            (observed_at - latest_spy_utc).total_seconds()
            if latest_spy_utc is not None
            else None
        )
        return bool(
            self.spy
            and self.xsp
            and self.spy[-1].ts == self.xsp[-1].ts
            and age_seconds is not None
            and 0.0 <= age_seconds <= freshness_seconds
        )

    def builder_kwargs(self, *, enabled: bool) -> dict[str, object]:
        return (
            {
                "spy_rth_one_minute_bars": self.spy,
                "xsp_rth_one_minute_bars": self.xsp,
            }
            if enabled
            else {}
        )

    def evidence(self, *, enabled: bool) -> Mapping[str, object]:
        return (
            {
                "spy_one_minute_raw_bars": len(self.raw_spy),
                "spy_one_minute_complete_bars": len(self.spy),
                "xsp_one_minute_raw_bars": len(self.raw_xsp),
                "xsp_one_minute_complete_bars": len(self.xsp),
            }
            if enabled
            else {}
        )


async def load_xsp_rth_one_minute_context(
    client,
    *,
    spy_contract,
    xsp_contract,
    duration_str: str,
    observed_et_naive: datetime,
) -> XspRthOneMinuteContext:
    """Fetch the same completed RTH minute window for SPY and XSP."""

    async def load(contract, symbol: str) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
        raw: Sequence[Bar] = await client.historical_bars_ohlcv(
            contract,
            duration_str=duration_str,
            bar_size="1 min",
            use_rth=True,
            what_to_show="TRADES",
            cache_ttl_sec=0.0,
        )
        raw_rows = tuple(raw)
        complete = trim_incomplete_last_bar(
            list(raw_rows),
            bar_size="1 min",
            now_ref=observed_et_naive,
        )
        normalized = normalize_bars_to_close(
            complete,
            symbol=symbol,
            bar_size="1 min",
            use_rth=True,
            naive_ts_mode="et",
        )
        return raw_rows, tuple(normalized)

    raw_spy, spy = await load(spy_contract, "SPY")
    raw_xsp, xsp = await load(xsp_contract, "XSP")
    return XspRthOneMinuteContext(raw_spy, spy, raw_xsp, xsp)
