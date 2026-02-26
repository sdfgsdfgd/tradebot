"""Shared UI helpers.

This module contains pure formatting/quoting helpers used by multiple UI screens.
Keep it dependency-light and free of IBKR side effects.
"""

from __future__ import annotations

import math
from datetime import datetime, time
from time import monotonic

from ib_insync import PnL, PortfolioItem, Ticker, Trade
from rich.text import Text

from tradebot.ui.time_compat import now_et as _now_et

_QUOTE_INFO_ERROR_CODES = {10167, 354, 10089, 10090, 10091, 10168}

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
    session_prev_close_1ago: float | None,
    session_close_3ago: float | None,
    has_actionable_quote: bool | None = None,
) -> tuple[float | None, float | None]:
    """Return (24h_pct, 72h_pct) with truthful fallback baselines.

    If an actionable quote is present (NBBO/last), 24h compares against the latest close.
    If not, 24h falls back to previous-session close so close-only rows don't pin to 0.00%.
    """
    actionable = bool(has_actionable_quote)
    if has_actionable_quote is None:
        actionable = bool(_ticker_actionable_price(ticker)) if ticker is not None else False

    if actionable:
        ticker_close = _ticker_close(ticker) if ticker is not None else None
        baseline_24 = ticker_close if ticker_close is not None else session_prev_close
    else:
        baseline_24 = session_prev_close_1ago

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

_INDEX_FUT_ORDER = ("NQ", "ES", "MYM")
_INDEX_FUT_LABELS = {
    "NQ": "NQ",
    "ES": "ES",
    "MYM": "MYM",
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
    if bid is not None and ask is not None and bid <= ask:
        return (bid + ask) / 2.0
    if last is not None:
        return float(last)
    return None


def _option_display_price(item: PortfolioItem, ticker: Ticker | None) -> float | None:
    if ticker:
        bid, ask, last = _sanitize_nbbo(
            getattr(ticker, "bid", None),
            getattr(ticker, "ask", None),
            getattr(ticker, "last", None),
        )
        if bid is not None and ask is not None and bid <= ask:
            return (bid + ask) / 2.0
        if last is not None:
            return float(last)
        model = getattr(ticker, "modelGreeks", None)
        model_price = _quote_num_display(getattr(model, "optPrice", None)) if model else None
        if model_price is not None:
            return float(model_price)
        close = _ticker_close(ticker)
        if close is not None:
            close_value = float(close)
        else:
            close_value = None
    else:
        close_value = None

    portfolio_mark = _quote_num_display(getattr(item, "marketPrice", None))
    if portfolio_mark is not None:
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


def _quote_num_actionable(value: float | None) -> float | None:
    num = _safe_num(value)
    if num is None or num <= 0:
        return None
    return float(num)


def _quote_num_display(value: float | None) -> float | None:
    num = _safe_num(value)
    if num is None or num <= 0:
        return None
    return float(num)


def _sanitize_nbbo(
    bid: float | None,
    ask: float | None,
    last: float | None,
) -> tuple[float | None, float | None, float | None]:
    return (
        _quote_num_actionable(bid),
        _quote_num_actionable(ask),
        _quote_num_actionable(last),
    )


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


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0 or bid > ask:
        return None
    return (bid + ask) / 2.0


def _fmt_quote(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _default_order_qty(item: PortfolioItem) -> int:
    return 1


def _price_increment_from_ladder(
    ladder: object,
    *,
    ref_price: float | None,
) -> float | None:
    if not isinstance(ladder, (list, tuple)):
        return None
    rows: list[tuple[float, float]] = []
    for row in ladder:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            low_edge = float(row[0])
            increment = float(row[1])
        except (TypeError, ValueError):
            continue
        if increment <= 0:
            continue
        rows.append((max(0.0, low_edge), increment))
    if not rows:
        return None
    rows.sort(key=lambda entry: entry[0])
    try:
        ref = float(ref_price) if ref_price is not None else 0.0
    except (TypeError, ValueError):
        ref = 0.0
    ref = max(0.0, ref)
    selected = rows[0][1]
    for low_edge, increment in rows:
        if ref >= low_edge:
            selected = increment
        else:
            break
    return selected if selected > 0 else None


def _tick_size(contract, ticker: Ticker | None, ref_price: float | None) -> float:
    for source in (ticker, contract):
        if source is None:
            continue
        ladder_tick = _price_increment_from_ladder(
            getattr(source, "tbPriceIncrements", None),
            ref_price=ref_price,
        )
        if ladder_tick is not None:
            return ladder_tick
    if ticker is not None:
        value = getattr(ticker, "minTick", None)
        if value:
            try:
                tick = float(value)
                if tick > 0:
                    return tick
            except (TypeError, ValueError):
                pass
    value = getattr(contract, "minTick", None)
    if value:
        try:
            tick = float(value)
            if tick > 0:
                return tick
        except (TypeError, ValueError):
            pass
    if contract.secType in ("OPT", "FOP"):
        if ref_price is not None and ref_price >= 3:
            return 0.05
        return 0.01
    return 0.01


def _round_to_tick(value: float | None, tick: float) -> float | None:
    if value is None:
        return None
    if not tick:
        return value
    return round(value / tick) * tick


def _tick_decimals(tick: float) -> int:
    text = f"{tick:.10f}".rstrip("0").rstrip(".")
    if "." in text:
        return len(text.split(".")[1])
    return 0


def _optimistic_price(
    bid: float | None, ask: float | None, mid: float | None, action: str
) -> float | None:
    if mid is None:
        return bid if action == "BUY" else ask
    if action == "BUY":
        if bid is None:
            return mid
        return (mid + bid) / 2.0
    if ask is None:
        return mid
    return (mid + ask) / 2.0


def _aggressive_price(
    bid: float | None, ask: float | None, mid: float | None, action: str
) -> float | None:
    # "Aggressive" is between midpoint and "cross" (ask for BUY, bid for SELL).
    if action == "BUY":
        if ask is None:
            return mid or bid
        if mid is None:
            return ask
        return (mid + ask) / 2.0
    if bid is None:
        return mid or ask
    if mid is None:
        return bid
    return (mid + bid) / 2.0


def _cross_price(bid: float | None, ask: float | None, action: str) -> float | None:
    if action == "BUY":
        return ask
    return bid


_EXEC_LADDER_OPTIMISTIC_SEC = 15.0
_EXEC_LADDER_MID_SEC = 15.0
_EXEC_LADDER_AGGRESSIVE_SEC = 15.0
_EXEC_LADDER_CROSS_SEC = 5 * 60.0
_EXEC_RELENTLESS_TIMEOUT_SEC = 15 * 60.0
_EXEC_LADDER_TIMEOUT_SEC = (
    _EXEC_LADDER_OPTIMISTIC_SEC
    + _EXEC_LADDER_MID_SEC
    + _EXEC_LADDER_AGGRESSIVE_SEC
    + _EXEC_LADDER_CROSS_SEC
)


def _exec_ladder_mode(elapsed_sec: float) -> str | None:
    """Return current execution ladder phase for an order.

    Phases:
      OPTIMISTIC (15s) -> MID (15s) -> AGGRESSIVE (15s) -> CROSS (5m) -> timeout.
    """
    try:
        t = float(elapsed_sec)
    except (TypeError, ValueError):
        t = 0.0
    if t < 0:
        t = 0.0

    if t < _EXEC_LADDER_OPTIMISTIC_SEC:
        return "OPTIMISTIC"
    t -= _EXEC_LADDER_OPTIMISTIC_SEC

    if t < _EXEC_LADDER_MID_SEC:
        return "MID"
    t -= _EXEC_LADDER_MID_SEC

    if t < _EXEC_LADDER_AGGRESSIVE_SEC:
        return "AGGRESSIVE"
    t -= _EXEC_LADDER_AGGRESSIVE_SEC

    if t < _EXEC_LADDER_CROSS_SEC:
        return "CROSS"
    return None


def _exec_chase_mode(elapsed_sec: float, *, selected_mode: str | None = "AUTO") -> str | None:
    cleaned = str(selected_mode or "AUTO").strip().upper()
    try:
        elapsed = float(elapsed_sec)
    except (TypeError, ValueError):
        elapsed = 0.0
    if cleaned in ("RELENTLESS", "RELENTLESS_DELAY"):
        return cleaned if elapsed <= _EXEC_RELENTLESS_TIMEOUT_SEC else None
    if cleaned and cleaned not in ("AUTO", "LADDER"):
        return cleaned if elapsed <= _EXEC_LADDER_TIMEOUT_SEC else None
    return _exec_ladder_mode(elapsed)


def _exec_chase_quote_signature(
    bid: float | None,
    ask: float | None,
    last: float | None,
) -> tuple[float | None, float | None, float | None]:
    return _sanitize_nbbo(bid, ask, last)


def _exec_chase_should_reprice(
    *,
    now_sec: float,
    last_reprice_sec: float | None,
    mode_now: str | None,
    prev_mode: str | None,
    quote_signature: tuple[float | None, float | None, float | None],
    prev_quote_signature: tuple[float | None, float | None, float | None] | None,
    min_interval_sec: float = 5.0,
) -> bool:
    if mode_now is None:
        return False
    if prev_mode is None:
        return True
    if str(mode_now) != str(prev_mode):
        return True
    if prev_quote_signature is None:
        return True
    if last_reprice_sec is None:
        return True
    elapsed = max(0.0, float(now_sec) - float(last_reprice_sec))
    if quote_signature != prev_quote_signature:
        return elapsed >= float(min_interval_sec)
    return elapsed >= float(min_interval_sec)


def _limit_price_for_mode(
    bid: float | None,
    ask: float | None,
    last: float | None,
    *,
    action: str,
    mode: str,
) -> float | None:
    bid = bid if bid is not None and bid > 0 else None
    ask = ask if ask is not None and ask > 0 else None
    last = last if last is not None and last > 0 else None
    mid = _midpoint(bid, ask)
    cleaned = str(mode or "").strip().upper()
    if cleaned == "CROSS":
        value = _cross_price(bid, ask, action)
    elif cleaned == "MID":
        value = mid
    elif cleaned == "OPTIMISTIC":
        value = _optimistic_price(bid, ask, mid, action)
    elif cleaned == "AGGRESSIVE":
        value = _aggressive_price(bid, ask, mid, action)
    else:
        value = mid
    if value is None:
        value = mid if mid is not None else last
    if value is None or value <= 0:
        return None
    return value


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
