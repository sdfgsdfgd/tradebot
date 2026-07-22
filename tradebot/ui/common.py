"""Shared UI helpers.

This module contains pure formatting/quoting helpers used by multiple UI screens.
Keep it dependency-light and free of IBKR side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from time import monotonic

from ib_insync import Contract, PnL, PortfolioItem, Ticker, Trade
from rich.text import Text

from tradebot.engines.execution import (
    _quote_num_actionable,
    _quote_num_display,
    _sanitize_nbbo,
    _sanitize_nbbo_extremes,
)
from tradebot.ui.time_compat import now_et as _now_et

_QUOTE_INFO_ERROR_CODES = {10167, 354, 10089, 10090, 10091, 10168}


@dataclass
class _SyntheticPortfolioItem:
    contract: Contract
    position: float = 0.0
    averageCost: float = 0.0
    marketPrice: float = 0.0
    marketValue: float = 0.0
    unrealizedPNL: float = 0.0
    realizedPNL: float = 0.0

# region Formatting Helpers
def _fmt_expiry(raw: str) -> str:
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    if len(raw) == 6 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}"
    return raw


def _fmt_qty(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"
# endregion


def _price_direction_glyph(pct24: float | None, pct72: float | None) -> Text:
    ref = pct24 if pct24 is not None else pct72
    if ref is None:
        return Text("•", style="dim")
    if ref > 0:
        return Text("▲", style="bold green")
    if ref < 0:
        return Text("▼", style="bold red")
    return Text("•", style="dim")


def _quote_age_ribbon(age_sec: float | None) -> Text:
    if age_sec is None:
        return Text("")
    if age_sec < 1.5:
        return Text("▰▰▰▰", style="bold #73d89e")
    if age_sec < 3.5:
        return Text("▰▰▰▱", style="#6fc18f")
    if age_sec < 6.0:
        return Text("▰▰▱▱", style="#8fa2b3")
    if age_sec < 10.0:
        return Text("▰▱▱▱", style="#778797")
    return Text("▱▱▱▱", style="#5e6a74")


def _safe_float(value: object, *, abs_cap: float = 1e307) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    # Guard against stream sentinels/invalid values (nan, inf, max-float style sentinels).
    if math.isnan(parsed) or not math.isfinite(parsed):
        return None
    if abs(parsed) >= float(abs_cap):
        return None
    return float(parsed)


def _portfolio_sort_key(item: PortfolioItem) -> float:
    unreal = float(item.unrealizedPNL or 0.0)
    realized = float(item.realizedPNL or 0.0)
    return unreal + realized


def _portfolio_row(
    item: PortfolioItem,
    contract_change: Text,
    *,
    unreal_text: Text | None = None,
    unreal_pct_text: Text | None = None,
) -> list[Text | str]:
    contract = item.contract
    symbol = Text(str(contract.symbol or ""), style="bold")
    if contract.secType in ("OPT", "FOP"):
        sep_style = "grey35"
        expiry_style = "grey58"
        expiry = _fmt_expiry(contract.lastTradeDateOrContractMonth or "")
        if expiry:
            symbol.append(" · ", style=sep_style)
            symbol.append(expiry, style=expiry_style)
        right = (contract.right or "").strip().upper()[:1]
        strike = _fmt_money(contract.strike) if contract.strike else ""
        if right and strike:
            symbol.append(" · ", style=sep_style)
            right_style = "bold green" if right == "C" else "bold red" if right == "P" else "bold"
            symbol.append(right, style=right_style)
            symbol.append(strike, style="bold")
        elif strike:
            symbol.append(" · ", style=sep_style)
            symbol.append(strike, style="bold")
    qty = _fmt_qty(float(item.position))
    avg_cost = _fmt_money(float(item.averageCost)) if item.averageCost else ""
    live_unreal, live_pct = _unrealized_pnl_values(item)
    unreal = unreal_text or _pnl_text(live_unreal)
    unreal_pct = unreal_pct_text or _pnl_pct_value(live_pct)
    unreal_combined = _combined_value_pct(unreal, unreal_pct)
    realized = _pnl_text(item.realizedPNL)
    return [
        symbol,
        qty,
        avg_cost,
        contract_change,
        unreal_combined,
        realized,
    ]


def _trade_sort_key(trade: Trade) -> int:
    order = trade.order
    order_id = getattr(order, "orderId", 0) or 0
    perm_id = getattr(order, "permId", 0) or 0
    return int(order_id or perm_id or 0)


# endregion



def _pnl_text(value: float | None, *, prefix: str = "") -> Text:
    if value is None:
        return Text("")
    text = f"{prefix}{_fmt_money(value)}"
    if value > 0:
        return Text(text, style="green")
    if value < 0:
        return Text(text, style="red")
    return Text(text)


def _unrealized_pnl_values(
    item: PortfolioItem,
    *,
    mark_price: float | None = None,
) -> tuple[float | None, float | None]:
    fallback_unreal = _safe_float(getattr(item, "unrealizedPNL", None))

    try:
        position = float(getattr(item, "position", 0.0) or 0.0)
    except (TypeError, ValueError):
        position = 0.0

    cost_basis = _cost_basis(item)

    def _fallback_pct() -> float | None:
        market_value = getattr(item, "marketValue", None)
        market_value_f = _safe_float(market_value)
        if market_value_f is None:
            market_value_f = 0.0
        denom_local = abs(cost_basis) if cost_basis else abs(market_value_f)
        return (fallback_unreal / denom_local * 100.0) if denom_local > 0 and fallback_unreal is not None else None

    # Truth-first PnL: prefer broker-reported unrealized whenever available.
    # Modeled mark-based unrealized is only a fallback when broker unrealized is missing.
    if fallback_unreal is not None:
        return fallback_unreal, _fallback_pct()

    if mark_price is None:
        mark_price = _mark_price(item)
    if mark_price is not None and position:
        multiplier = _infer_multiplier(item)
        mark_value = float(mark_price) * float(position) * float(multiplier)
        unreal = float(mark_value - cost_basis)
        denom = abs(cost_basis) if cost_basis else abs(mark_value)
        pct = (unreal / denom * 100.0) if denom > 0 else None
        return unreal, pct

    if fallback_unreal is None:
        return None, None
    return fallback_unreal, _fallback_pct()


def _pnl_pct_text(item: PortfolioItem, *, mark_price: float | None = None) -> Text:
    value, pct = _unrealized_pnl_values(item, mark_price=mark_price)
    if value is None:
        return Text("")
    return _pnl_pct_value(pct)


def _pnl_pct_value(pct: float | None) -> Text:
    if pct is None:
        return Text("")
    text = f"{pct:.2f}%"
    if pct > 0:
        return Text(text, style="green")
    if pct < 0:
        return Text(text, style="red")
    return Text(text)


def _combined_value_pct(value: Text, pct: Text) -> Text:
    text = Text("")
    if value.plain:
        text.append_text(value)
    if pct.plain:
        if text.plain:
            text.append("  ")
        text.append("(")
        text.append_text(pct)
        text.append(")")
    return text


def _pct_change(price: float | None, baseline: float | None) -> float | None:
    if price is None or baseline is None:
        return None
    if baseline <= 0:
        return None
    return ((price - baseline) / baseline) * 100.0


def _pct24_72_from_price(
    *,
    price: float | None,
    ticker: Ticker | None,
    session_prev_close: float | None,
    session_close_3ago: float | None,
) -> tuple[float | None, float | None]:
    """Return (24h_pct, 72h_pct) using stable close baselines.

    24h always compares against the latest close (ticker.close/prevLast when present,
    otherwise the cached session close anchor). This avoids oscillation when
    actionable quotes (NBBO/last) appear/disappear between refresh ticks.
    """
    ticker_close = _ticker_close(ticker) if ticker is not None else None
    baseline_24 = ticker_close if ticker_close is not None else session_prev_close
    pct24 = _pct_change(price, baseline_24)
    pct72 = _pct_change(price, session_close_3ago)
    return pct24, pct72


def _pct_dual_text(
    pct24: float | None,
    pct72: float | None,
    *,
    separator: str = "-",
) -> Text:
    text = Text("")
    if pct24 is not None:
        text.append(f"{pct24:.2f}%", style=_pct_style(pct24))
    if pct72 is not None:
        if pct24 is not None:
            sep_style = "red" if separator == "-" else "dim"
            text.append(separator, style=sep_style)
        text.append(f"{pct72:.2f}%", style=_pct_style(pct72))
    return text


def _price_pct_dual_text(
    price: float | None,
    pct24: float | None,
    pct72: float | None,
    *,
    separator: str = "·",
    center_width: int | None = None,
) -> Text:
    price_width = 9
    pct_width = 7

    def _pct_field(value: float | None) -> str:
        if value is None:
            return " " * pct_width
        # Reserve a sign slot so positives and negatives align vertically.
        return f"{value: .2f}%".rjust(pct_width)

    text = Text("")
    if price is not None:
        style = _pct_style(pct24) if pct24 is not None else ""
        text.append(f"{price:,.2f}".rjust(price_width), style=style)
    elif pct24 is not None or pct72 is not None:
        text.append(" " * price_width, style="dim")

    if pct24 is not None or pct72 is not None:
        if price is not None:
            text.append(" ¦ ", style="grey35")
        text.append(_pct_field(pct24), style=_pct_style(pct24) if pct24 is not None else "dim")
        sep_style = "red" if separator == "-" else "dim"
        text.append(f" {separator} ", style=sep_style)
        text.append(_pct_field(pct72), style=_pct_style(pct72) if pct72 is not None else "dim")

    if center_width and center_width > 0:
        pad = int(center_width) - len(text.plain)
        if pad > 0:
            left = pad // 2
            right = pad - left
            centered = Text(" " * left)
            centered.append_text(text)
            centered.append(" " * right)
            return centered
    return text


def _pct_style(pct: float) -> str:
    if pct > 0:
        return "green"
    if pct < 0:
        return "red"
    return ""


def _pnl_value(pnl: PnL | None) -> float | None:
    if not pnl:
        return None
    return _safe_float(getattr(pnl, "dailyPnL", None))


def _estimate_buying_power(
    buying_power: float, pnl: PnL | None, anchor: float | None
) -> float | None:
    daily = _pnl_value(pnl)
    if daily is None or anchor is None:
        return None
    return buying_power + (daily - anchor)


_SECTION_ORDER = (
    ("OPTIONS", "OPT"),
    ("STOCKS", "STK"),
    ("FUTURES", "FUT"),
    ("FUTURES OPT", "FOP"),
)
_SECTION_TYPES = {sec_type for _, sec_type in _SECTION_ORDER}

_INDEX_FUT_ORDER = ("NQ", "ES")
_INDEX_FUT_LABELS = {
    "NQ": "NQ",
    "ES": "ES",
}
_INDEX_ORDER = ("QQQ", "SPY", "DIA")
_INDEX_LABELS = {
    "QQQ": "NASDAQ",
    "SPY": "S&P",
    "DIA": "DOW",
}
_PROXY_ORDER = ("TQQQ",)
_PROXY_LABELS = {
    "TQQQ": "TQQQ",
}
_TICKER_WIDTHS = {
    "label": 10,
    "price": 9,
    "change": 9,
    "pct": 10,
}


def _ticker_price(ticker: Ticker) -> float | None:
    bid, ask, last = _sanitize_nbbo(
        getattr(ticker, "bid", None),
        getattr(ticker, "ask", None),
        getattr(ticker, "last", None),
    )
    if bid is not None and ask is not None and bid <= ask:
        return (bid + ask) / 2.0
    if last is not None:
        return last
    try:
        value = float(ticker.marketPrice())
    except Exception:
        value = None
    value = _quote_num_display(value)
    if value is not None:
        return value
    return _ticker_close(ticker)


def _ticker_actionable_price(ticker: Ticker) -> float | None:
    bid, ask, last = _sanitize_nbbo(
        getattr(ticker, "bid", None),
        getattr(ticker, "ask", None),
        getattr(ticker, "last", None),
    )
    bid, ask = _sanitize_nbbo_extremes(bid, ask, ref_price=last)
    if bid is not None and ask is not None and bid <= ask:
        return (bid + ask) / 2.0
    if last is not None:
        return float(last)
    return None


def _option_display_price(item: PortfolioItem, ticker: Ticker | None) -> float | None:
    sec_type = str(getattr(getattr(item, "contract", None), "secType", "") or "").strip().upper()
    portfolio_mark = _quote_num_display(getattr(item, "marketPrice", None))
    if ticker:
        bid, ask, last = _sanitize_nbbo(
            getattr(ticker, "bid", None),
            getattr(ticker, "ask", None),
            getattr(ticker, "last", None),
        )
        model = getattr(ticker, "modelGreeks", None)
        model_price = _quote_num_display(getattr(model, "optPrice", None)) if model else None
        ref_price = model_price if model_price is not None else (portfolio_mark if portfolio_mark is not None else last)
        bid, ask = _sanitize_nbbo_extremes(bid, ask, ref_price=ref_price)
        if bid is not None and ask is not None and bid <= ask:
            return (bid + ask) / 2.0
        if sec_type == "FOP":
            if model_price is not None:
                return float(model_price)
            if portfolio_mark is not None:
                return float(portfolio_mark)
            if last is not None:
                return float(last)
        else:
            if last is not None:
                return float(last)
        if model_price is not None:
            return float(model_price)
        close = _ticker_close(ticker)
        if close is not None:
            close_value = float(close)
        else:
            close_value = None
    else:
        close_value = None

    if portfolio_mark is not None and (sec_type != "FOP" or ticker is not None):
        return float(portfolio_mark)
    return close_value


def _ticker_close(ticker: Ticker) -> float | None:
    for attr in ("close", "prevLast"):
        value = getattr(ticker, attr, None)
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(num) or num <= 0:
            continue
        return num
    return None


def _market_data_tag(ticker: Ticker) -> str:
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return " [L]"
    if md_type in (3, 4):
        return " [D]"
    return ""


def _market_data_label(ticker: Ticker) -> str:
    md_raw = getattr(ticker, "marketDataType", None)
    try:
        md_type = int(md_raw) if md_raw is not None else None
    except (TypeError, ValueError):
        md_type = None
    if md_type == 1:
        return "Live"
    if md_type == 2:
        return "Live-Frozen"
    if md_type == 3:
        return "Delayed"
    if md_type == 4:
        return "Delayed-Frozen"
    return "n/a"


def _quote_source_label(
    ticker: Ticker,
    *,
    source: str,
    has_nbbo: bool,
    has_last: bool,
) -> str:
    clean_source = str(source or "").strip()
    if not clean_source:
        return ""
    src = clean_source.lower()
    has_close = _ticker_close(ticker) is not None
    if src in ("stream-close-only", "delayed-snapshot", "delayed-frozen-snapshot"):
        if (not has_nbbo) and (not has_last) and has_close:
            return "close-only"
    return clean_source


def _should_show_quote_error_code(ticker: Ticker, *, has_nbbo: bool, has_last: bool) -> int | None:
    raw = getattr(ticker, "tbQuoteErrorCode", None)
    if raw is None:
        return None
    try:
        code = int(raw)
    except (TypeError, ValueError):
        return None
    has_close = _ticker_close(ticker) is not None
    if code in _QUOTE_INFO_ERROR_CODES and (has_nbbo or has_last or has_close):
        return None
    return code


def _quote_status_line(ticker: Ticker) -> Text:
    health = _quote_health(
        bid=getattr(ticker, "bid", None),
        ask=getattr(ticker, "ask", None),
        last=getattr(ticker, "last", None),
    )
    if bool(health.get("has_nbbo")):
        bid_ask = "ok"
    elif bool(health.get("has_one_sided")):
        bid_ask = "1-sided"
    else:
        bid_ask = "n/a"
    has_nbbo = bool(health.get("has_nbbo"))
    has_last = bool(health.get("has_last"))
    last_label = "ok" if has_last else "n/a"
    parts = [f"MD Quotes: bid/ask {bid_ask}", f"last {last_label}"]
    source = str(getattr(ticker, "tbQuoteSource", "") or "").strip()
    source_label = _quote_source_label(
        ticker,
        source=source,
        has_nbbo=has_nbbo,
        has_last=has_last,
    )
    if source_label:
        parts.append(f"src {source_label}")
    as_of_raw = str(getattr(ticker, "tbQuoteAsOf", "") or "").strip()
    if as_of_raw:
        try:
            as_of_dt = datetime.fromisoformat(as_of_raw.replace("Z", "+00:00"))
            parts.append(f"asof {as_of_dt.strftime('%H:%M:%S')}")
        except ValueError:
            parts.append(f"asof {as_of_raw[:19]}")
    updated_mono = getattr(ticker, "tbQuoteUpdatedMono", None)
    try:
        age_sec = max(0.0, monotonic() - float(updated_mono)) if updated_mono is not None else None
    except (TypeError, ValueError):
        age_sec = None
    if age_sec is not None:
        parts.append(f"age {age_sec:.0f}s")
    top_updated_mono = getattr(ticker, "tbTopQuoteUpdatedMono", None)
    try:
        top_age_sec = (
            max(0.0, monotonic() - float(top_updated_mono))
            if top_updated_mono is not None
            else None
        )
    except (TypeError, ValueError):
        top_age_sec = None
    if top_age_sec is not None:
        parts.append(f"topchg {top_age_sec:.0f}s")
    top_moves = getattr(ticker, "tbTopQuoteMoveCount", None)
    if top_moves is not None:
        try:
            parts.append(f"moves {max(0, int(top_moves))}")
        except (TypeError, ValueError):
            pass
    error_code = _should_show_quote_error_code(
        ticker,
        has_nbbo=has_nbbo,
        has_last=has_last,
    )
    if error_code is not None:
        parts.append(f"code {error_code}")
    return Text(" · ".join(parts), style="dim")


def _market_session_bucket(ts_et: datetime | time) -> str:
    current = ts_et.time() if isinstance(ts_et, datetime) else ts_et
    if time(4, 0) <= current < time(9, 30):
        return "PRE"
    if time(9, 30) <= current < time(16, 0):
        return "RTH"
    if time(16, 0) <= current < time(20, 0):
        return "POST"
    return "OVERNIGHT"


def _market_session_label() -> str:
    bucket = _market_session_bucket(_now_et())
    if bucket == "RTH":
        return "MRKT"
    if bucket == "OVERNIGHT":
        return "OVRNGHT"
    return bucket


def _ticker_line(
    order: tuple[str, ...],
    labels: dict[str, str],
    tickers: dict[str, Ticker],
    error: str | None,
    prefix: str,
    *,
    allow_display_fallback: bool = False,
) -> Text:
    if error:
        return Text(f"{prefix}Data error: {error}", style="red")
    text = Text()
    if prefix:
        text.append(prefix)
    for idx, symbol in enumerate(order):
        if idx:
            text.append(" | ", style="dim")
        label = labels[symbol]
        ticker = tickers.get(symbol)
        if not ticker:
            text.append_text(_ticker_missing(label))
            continue
        tag = _market_data_tag(ticker)
        price = _ticker_actionable_price(ticker)
        if (price is None or price <= 0) and allow_display_fallback:
            fallback = _ticker_price(ticker)
            if fallback is not None and fallback > 0:
                price = float(fallback)
        close = _ticker_close(ticker)
        if price is None or price <= 0:
            if close and close > 0:
                text.append_text(_ticker_closed(label + tag, close))
            else:
                text.append_text(_ticker_missing(label + tag))
            continue
        if close is None or close <= 0:
            text.append_text(_ticker_price_only(label + tag, price))
            continue
        change = price - close
        pct = (change / close) * 100.0
        style = "green" if change > 0 else "red" if change < 0 else ""
        text.append_text(_ticker_block(label + tag, price, change, pct, style))
    return text


def _ticker_block(label: str, price: float, change: float, pct: float, style: str) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = f"{price:,.2f}".rjust(_TICKER_WIDTHS["price"])
    change_text = f"{change:+.2f}".rjust(_TICKER_WIDTHS["change"])
    pct_text = f"({pct:+.2f}%)".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text)
    text.append(" ")
    text.append(change_text, style=style)
    text.append(" ")
    text.append(pct_text, style=style)
    return text


def _ticker_missing(label: str) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = "n/a".rjust(_TICKER_WIDTHS["price"])
    blank_change = "".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text, style="dim")
    text.append(" ", style="dim")
    text.append(price_text, style="dim")
    text.append(" ", style="dim")
    text.append(blank_change, style="dim")
    text.append(" ", style="dim")
    text.append(blank_pct, style="dim")
    return text


def _ticker_price_only(label: str, price: float) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = f"{price:,.2f}".rjust(_TICKER_WIDTHS["price"])
    blank_change = "n/a".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text)
    text.append(" ")
    text.append(blank_change, style="dim")
    text.append(" ")
    text.append(blank_pct, style="dim")
    return text


def _ticker_closed(label: str, last_price: float) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = "Closed".rjust(_TICKER_WIDTHS["price"])
    last_text = f"({last_price:,.2f})".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text, style="red")
    text.append(" ")
    text.append(last_text, style="red")
    text.append(" ")
    text.append(blank_pct, style="red")
    return text


def _safe_num(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or num == 0:
        return None
    return num


def _quote_health(
    *,
    bid: float | None,
    ask: float | None,
    last: float | None,
    close: float | None = None,
) -> dict[str, bool]:
    clean_bid, clean_ask, clean_last = _sanitize_nbbo(bid, ask, last)
    clean_close = _quote_num_display(close)
    has_bid = bool(clean_bid is not None)
    has_ask = bool(clean_ask is not None)
    has_nbbo = bool(clean_bid is not None and clean_ask is not None and clean_bid <= clean_ask)
    has_one_sided = bool((has_bid or has_ask) and not has_nbbo)
    has_last = bool(clean_last is not None)
    has_close_only = bool((not has_nbbo) and (not has_last) and clean_close is not None)
    return {
        "has_bid": has_bid,
        "has_ask": has_ask,
        "has_nbbo": has_nbbo,
        "has_one_sided": has_one_sided,
        "has_last": has_last,
        "has_close_only": has_close_only,
        "has_actionable": bool(has_nbbo or has_last),
    }


def _mark_price(item: PortfolioItem) -> float | None:
    value = _quote_num_display(getattr(item, "marketPrice", None))
    if value is not None:
        return value
    if item.marketValue is not None and item.position:
        try:
            return float(item.marketValue) / float(item.position)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _infer_multiplier(item: PortfolioItem) -> float:
    position = float(item.position or 0.0)
    market_value = _safe_num(getattr(item, "marketValue", None))
    market_price = _safe_num(getattr(item, "marketPrice", None))
    if position and market_value is not None and market_price is not None:
        denom = market_price * position
        if denom:
            mult = market_value / denom
            if mult and not math.isnan(mult):
                mult = abs(float(mult))
                if mult > 0:
                    return mult
    raw = getattr(item.contract, "multiplier", None)
    try:
        mult = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        mult = 1.0
    if math.isnan(mult) or mult <= 0:
        return 1.0
    return float(mult)


def _cost_basis(item: PortfolioItem) -> float:
    market_value = item.marketValue
    unreal = item.unrealizedPNL
    if market_value is not None and unreal is not None:
        try:
            return float(market_value) - float(unreal)
        except (TypeError, ValueError):
            pass
    avg_cost = item.averageCost
    position = item.position
    if avg_cost is not None and position is not None:
        try:
            return float(avg_cost) * float(position)
        except (TypeError, ValueError):
            pass
    return 0.0


def _fmt_quote(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _default_order_qty(item: PortfolioItem) -> int:
    return 1


def _append_digit(value: str, char: str, allow_decimal: bool) -> str:
    if char == "." and not allow_decimal:
        return value
    if char == "." and "." in value:
        return value
    if char == "." and not value:
        return "0."
    return value + char


def _parse_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: str) -> int | None:
    if not value:
        return None
    if not value.isdigit():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


# endregion
