"""Read-only broker/cache adapter for the prospective 1OZ onset tape."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ib_insync import Contract, Future, IB, Stock

from ..chart_data.history import normalize_bars_to_close, read_cache
from ..chart_data.series import OhlcvBar
from ..news.contract import load_news_history
from .gold_onset import (
    advance_gold_onset_tape,
    build_gold_onset_context,
    gold_signal_context,
    select_gold_contract_pair,
)
from .live_calibration import LiveCalibrationLedger


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATHS = {
    "xau_h1": (
        ROOT / "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_1hour_full24.csv",
        ROOT / "db/XAUUSD/XAUUSD_2016-07-01_2026-08-02_1hour_full24.csv",
    ),
    "xau_h4": (
        ROOT / "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_4hours_full24.csv",
        ROOT / "db/XAUUSD/XAUUSD_2016-01-01_2026-08-02_4hours_full24.csv",
    ),
    "xau_d1": (
        ROOT / "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_1day_full24.csv",
        ROOT / "db/XAUUSD/XAUUSD_2016-01-01_2026-08-02_1day_full24.csv",
    ),
    "uup_d1": (ROOT / "db/UUP/UUP_2015-07-01_2026-08-02_1day_rth.csv",),
    "tip_d1": (ROOT / "db/TIP/TIP_2015-07-01_2026-08-02_1day_rth.csv",),
}
_MONTH_CODES = set("FGHJKMNQUVXZ")


def _aware(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bars_from_cache(paths: Iterable[Path]) -> list[OhlcvBar]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"gold onset cache missing: {missing}")
    return [
        bar
        for _stamp, bar in sorted(
            {bar.ts: bar for path in paths for bar in read_cache(path)}.items()
        )
    ]


def _raw_bars(raw: Iterable[object]) -> list[OhlcvBar]:
    output = []
    for source in raw:
        stamp = getattr(source, "date", None)
        if isinstance(stamp, datetime):
            ts = _aware(stamp).replace(tzinfo=None)
        elif stamp is not None:
            try:
                ts = datetime.fromisoformat(str(stamp)).replace(tzinfo=None)
            except ValueError:
                continue
        else:
            continue
        values = [_finite(getattr(source, name, None)) for name in ("open", "high", "low", "close", "volume")]
        if any(value is None for value in values):
            continue
        output.append(OhlcvBar(ts, *[float(value) for value in values]))
    return output


def _merge(*groups: Iterable[OhlcvBar]) -> list[OhlcvBar]:
    return [
        bar
        for _stamp, bar in sorted(
            {bar.ts: bar for group in groups for bar in group}.items()
        )
    ]


def _request_bars(
    ib: IB,
    contract: Contract,
    *,
    duration: str,
    bar_size: str,
    what: str,
    use_rth: bool,
    symbol: str,
) -> list[OhlcvBar]:
    raw = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what,
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
        timeout=60,
    )
    return normalize_bars_to_close(
        _raw_bars(raw),
        symbol=symbol,
        bar_size=bar_size,
        use_rth=use_rth,
    )


def _qualify_one(ib: IB, contract: Contract) -> Contract:
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"contract did not qualify: {contract}")
    return qualified[0]


def _expiry(value: object) -> datetime | None:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) < 6:
        return None
    day = digits[6:8] if len(digits) >= 8 else "28"
    try:
        return datetime.strptime(digits[:6] + day, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _chain(ib: IB, symbol: str, *, now: datetime) -> list[Contract]:
    details = ib.reqContractDetails(Future(symbol=symbol, exchange="COMEX", currency="USD"))
    output = []
    for detail in details:
        contract = detail.contract
        expiry = _expiry(
            getattr(detail, "realExpirationDate", "")
            or getattr(contract, "lastTradeDateOrContractMonth", "")
        )
        if expiry is None or not now.date() <= expiry.date() <= (now + timedelta(days=400)).date():
            continue
        local = str(getattr(contract, "localSymbol", "") or "").upper()
        if not re.search(rf"[{''.join(sorted(_MONTH_CODES))}]\d$", local):
            continue
        output.append(contract)
    output.sort(key=lambda contract: str(getattr(contract, "lastTradeDateOrContractMonth", "") or ""))
    return output[:10]


def _chain_quotes(
    ib: IB, *, now: datetime
) -> tuple[list[dict[str, object]], dict[int, Contract]]:
    contracts = [contract for symbol in ("GC", "1OZ") for contract in _chain(ib, symbol, now=now)]
    qualified = []
    for contract in contracts:
        rows = ib.qualifyContracts(contract)
        if rows:
            qualified.append(rows[0])
    ib.reqMarketDataType(1)
    tickers = [ib.reqMktData(contract, "233", False, False) for contract in qualified]
    ib.sleep(5.0)
    quotes = []
    by_id = {}
    for contract, ticker in zip(qualified, tickers, strict=True):
        by_id[int(contract.conId)] = contract
        quotes.append(
            {
                "symbol": contract.symbol,
                "local_symbol": contract.localSymbol,
                "con_id": contract.conId,
                "expiry": contract.lastTradeDateOrContractMonth,
                "market_data_type": ticker.marketDataType,
                "bid": _finite(ticker.bid),
                "bid_size": _finite(ticker.bidSize),
                "ask": _finite(ticker.ask),
                "ask_size": _finite(ticker.askSize),
                "last": _finite(ticker.last),
                "volume": _finite(ticker.volume),
                "time": _aware(ticker.time or now).isoformat(),
            }
        )
        ib.cancelMktData(contract)
    return quotes, by_id


def _point_at(rows: Sequence[OhlcvBar], at: datetime) -> dict[str, object]:
    aware = sorted(((_aware(row.ts), row) for row in rows), key=lambda item: item[0])
    candidates = [(stamp, row) for stamp, row in aware if stamp <= at]
    if not candidates:
        return {"close": None, "bar_end_utc": None, "age_seconds": None}
    stamp, row = candidates[-1]
    return {
        "close": float(row.close),
        "bar_end_utc": stamp.isoformat(),
        "age_seconds": (_aware(at) - stamp).total_seconds(),
    }


def _news_history(path: Path) -> list[dict[str, object]]:
    return [
        row
        for source in sorted(path.glob("*.jsonl"))
        for row in load_news_history(source)
    ]


def main() -> None:
    request_started_at = datetime.now(timezone.utc)
    ib = IB()
    ib.connect(
        os.environ.get("IBKR_HOST", "127.0.0.1"),
        int(os.environ.get("IBKR_PORT", "4001")),
        clientId=int(os.environ.get("IBKR_CLIENT_ID", "3199")),
        readonly=True,
        timeout=12,
    )
    try:
        xau = _qualify_one(
            ib,
            Contract(secType="CMDTY", symbol="XAUUSD", exchange="SMART", currency="USD"),
        )
        uup = _qualify_one(ib, Stock("UUP", "SMART", "USD"))
        tip = _qualify_one(ib, Stock("TIP", "SMART", "USD"))
        xau_h1 = _merge(
            _bars_from_cache(CACHE_PATHS["xau_h1"]),
            _request_bars(ib, xau, duration="10 D", bar_size="1 hour", what="MIDPOINT", use_rth=False, symbol="XAUUSD"),
        )
        xau_h4 = _merge(
            _bars_from_cache(CACHE_PATHS["xau_h4"]),
            _request_bars(ib, xau, duration="1 M", bar_size="4 hours", what="MIDPOINT", use_rth=False, symbol="XAUUSD"),
        )
        xau_d1 = _merge(
            _bars_from_cache(CACHE_PATHS["xau_d1"]),
            _request_bars(ib, xau, duration="1 Y", bar_size="1 day", what="MIDPOINT", use_rth=False, symbol="XAUUSD"),
        )
        uup_d1 = _merge(
            _bars_from_cache(CACHE_PATHS["uup_d1"]),
            _request_bars(ib, uup, duration="1 M", bar_size="1 day", what="TRADES", use_rth=True, symbol="UUP"),
        )
        tip_d1 = _merge(
            _bars_from_cache(CACHE_PATHS["tip_d1"]),
            _request_bars(ib, tip, duration="1 M", bar_size="1 day", what="TRADES", use_rth=True, symbol="TIP"),
        )
        quotes, contracts = _chain_quotes(ib, now=request_started_at)
        quote_as_of = datetime.now(timezone.utc)
        pair = select_gold_contract_pair(quotes, observed_at=quote_as_of)
        if not pair.get("usable"):
            raise RuntimeError(
                "gold shared-month book unavailable: "
                + json.dumps(pair, sort_keys=True, allow_nan=False)
            )
        gc_contract = contracts[int(dict(pair["gc"])["con_id"])]
        one_contract = contracts[int(dict(pair["one_oz"])["con_id"])]
        gc_h1 = _request_bars(ib, gc_contract, duration="10 D", bar_size="1 hour", what="TRADES", use_rth=False, symbol="GC")
        one_m30 = _request_bars(ib, one_contract, duration="10 D", bar_size="30 mins", what="TRADES", use_rth=False, symbol="1OZ")
    finally:
        ib.disconnect()

    observed_at = datetime.now(timezone.utc)
    signal = gold_signal_context(
        xau_h4, xau_d1, uup_d1, tip_d1, as_of=observed_at
    )
    decision_at = (
        datetime.fromisoformat(str(signal["decision_bar_end_utc"]).replace("Z", "+00:00"))
        if signal.get("usable")
        else observed_at
    )
    source_points = {
        "XAUUSD": _point_at(xau_h1, decision_at),
        "GC": _point_at(gc_h1, decision_at),
        "1OZ": _point_at(one_m30, decision_at),
    }
    news_dir = Path(
        os.environ.get(
            "TRADEBOT_NEWS_HISTORY",
            str(Path.home() / ".local/state/tradebot/news/history"),
        )
    ).expanduser()
    context = build_gold_onset_context(
        xau_h4=xau_h4,
        xau_daily=xau_d1,
        uup_daily=uup_d1,
        tip_daily=tip_d1,
        quotes=quotes,
        news_history=_news_history(news_dir),
        source_points=source_points,
        observed_at=observed_at,
    )
    ledger = LiveCalibrationLedger(
        Path(
            os.environ.get(
                "GOLD_ONSET_LEDGER",
                str(Path.home() / ".local/state/tradebot/research/gold_onset.jsonl"),
            )
        ).expanduser()
    )
    output = advance_gold_onset_tape(
        ledger,
        context=context,
        outcome_bars={"XAUUSD": xau_h1, "GC": gc_h1, "1OZ": one_m30},
        observed_at=observed_at,
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
