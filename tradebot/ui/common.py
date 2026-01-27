"""Shared UI helpers.

This module contains pure formatting/quoting helpers used by multiple UI screens.
Keep it dependency-light and free of IBKR side effects.
"""

from __future__ import annotations

import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

from ib_insync import PnL, PortfolioItem, Ticker, Trade
from rich.text import Text

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

def _portfolio_sort_key(item: PortfolioItem) -> float:
    unreal = float(item.unrealizedPNL or 0.0)
    realized = float(item.realizedPNL or 0.0)
    return unreal + realized


def _portfolio_row(
    item: PortfolioItem,
    contract_change: Text,
    underlying_change: Text,
    *,
    unreal_text: Text | None = None,
    unreal_pct_text: Text | None = None,
) -> list[Text | str]:
    contract = item.contract
    expiry = _fmt_expiry(contract.lastTradeDateOrContractMonth or "")
    right = contract.right or ""
    strike = _fmt_money(contract.strike) if contract.strike else ""
    qty = _fmt_qty(float(item.position))
    avg_cost = _fmt_money(float(item.averageCost)) if item.averageCost else ""
    unreal = unreal_text or _pnl_text(item.unrealizedPNL)
    unreal_pct = unreal_pct_text or _pnl_pct_text(item)
    unreal_combined = _combined_value_pct(unreal, unreal_pct)
    realized = _pnl_text(item.realizedPNL)
    return [
        contract.symbol,
        expiry,
        right,
        strike,
        qty,
        avg_cost,
        contract_change,
        unreal_combined,
        realized,
        underlying_change,
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


def _pnl_pct_text(item: PortfolioItem) -> Text:
    value = item.unrealizedPNL
    if value is None:
        return Text("")
    cost_basis = 0.0
    if item.averageCost:
        cost_basis = float(item.averageCost) * float(item.position)
    denom = abs(cost_basis) if cost_basis else abs(float(item.marketValue or 0.0))
    if denom <= 0:
        return Text("")
    pct = (float(value) / denom) * 100.0
    text = f"{pct:.2f}%"
    if pct > 0:
        return Text(text, style="green")
    if pct < 0:
        return Text(text, style="red")
    return Text(text)


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


def _pct_dual_text(pct24: float | None, pct72: float | None) -> Text:
    text = Text("")
    if pct24 is not None:
        text.append(f"{pct24:.2f}%", style=_pct_style(pct24))
    if pct72 is not None:
        if pct24 is not None:
            text.append("-", style="red")
        text.append(f"{pct72:.2f}%", style=_pct_style(pct72))
    return text


def _price_pct_dual_text(price: float | None, pct24: float | None, pct72: float | None) -> Text:
    text = Text("")
    if price is not None:
        style = _pct_style(pct24) if pct24 is not None else ""
        text.append(f"{price:,.2f}", style=style)
        if pct24 is not None or pct72 is not None:
            text.append(" ")
    text.append_text(_pct_dual_text(pct24, pct72))
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
    value = pnl.dailyPnL
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _estimate_net_liq(
    net_liq: float, pnl: PnL | None, anchor: float | None
) -> float | None:
    daily = _pnl_value(pnl)
    if daily is None or anchor is None:
        return None
    return net_liq + (daily - anchor)


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

_INDEX_ORDER = ("NQ", "ES", "YM")
_INDEX_LABELS = {
    "NQ": "NASDAQ",
    "ES": "S&P",
    "YM": "DOW",
}
_PROXY_ORDER = ("QQQ", "TQQQ")
_PROXY_LABELS = {
    "QQQ": "QQQ",
    "TQQQ": "TQQQ",
}
_TICKER_WIDTHS = {
    "label": 10,
    "price": 9,
    "change": 9,
    "pct": 10,
}


def _ticker_price(ticker: Ticker) -> float | None:
    bid = _safe_num(getattr(ticker, "bid", None))
    ask = _safe_num(getattr(ticker, "ask", None))
    if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
        return (bid + ask) / 2.0
    last = _safe_num(getattr(ticker, "last", None))
    if last is not None and last > 0:
        return last
    try:
        value = float(ticker.marketPrice())
    except Exception:
        value = None
    if value is not None and value > 0 and not math.isnan(value):
        return value
    return _ticker_close(ticker)


def _ticker_close(ticker: Ticker) -> float | None:
    for attr in ("close", "prevLast"):
        value = getattr(ticker, attr, None)
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(num) or num == 0:
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
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return "Live"
    if md_type in (3, 4):
        return "Delayed"
    return "n/a"


def _quote_status_line(ticker: Ticker) -> Text:
    bid = _safe_num(getattr(ticker, "bid", None))
    ask = _safe_num(getattr(ticker, "ask", None))
    last = _safe_num(getattr(ticker, "last", None))
    bid_ask = "ok" if bid is not None and ask is not None else "n/a"
    last_label = "ok" if last is not None else "n/a"
    return Text(f"MD Quotes: bid/ask {bid_ask} · last {last_label}", style="dim")


def _market_session_label() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York")).time()
    if time(4, 0) <= now_et < time(9, 30):
        return "PRE"
    if time(9, 30) <= now_et < time(16, 0):
        return "MRKT"
    if time(16, 0) <= now_et < time(20, 0):
        return "POST"
    return "OVRNGHT"


def _ticker_line(
    order: tuple[str, ...],
    labels: dict[str, str],
    tickers: dict[str, Ticker],
    error: str | None,
    prefix: str,
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
        price = _ticker_price(ticker)
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


def _mark_price(item: PortfolioItem) -> float | None:
    value = _safe_num(getattr(item, "marketPrice", None))
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
    return (bid + ask) / 2.0


def _fmt_quote(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _default_order_qty(item: PortfolioItem) -> int:
    return 1


def _tick_size(contract, ticker: Ticker | None, ref_price: float | None) -> float:
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
    if contract.secType == "OPT":
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
    if action == "BUY":
        return ask or mid or bid
    return bid or mid or ask


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
