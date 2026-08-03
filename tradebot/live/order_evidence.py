"""Shared live-order evidence and execution-price projection."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from ..engines.execution import EXECUTION_POLICY, execution_price
from .execution import order_ids


def broker_trade_snapshot(trade: object) -> dict[str, object]:
    """Freeze the broker-visible order and exact fill economics."""

    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    contract = getattr(trade, "contract", None)
    order_id, perm_id = order_ids(trade)
    symbol = str(getattr(contract, "symbol", "") or "")
    fills = []
    for fill in getattr(trade, "fills", ()) or ():
        execution = getattr(fill, "execution", None)
        commission = getattr(fill, "commissionReport", None)
        fill_time = getattr(fill, "time", None)
        fills.append(
            {
                "exec_id": str(getattr(execution, "execId", "") or ""),
                "time_utc": (
                    fill_time.astimezone(timezone.utc).isoformat()
                    if isinstance(fill_time, datetime)
                    and fill_time.tzinfo is not None
                    else None
                ),
                "side": str(getattr(execution, "side", "") or "").upper(),
                "symbol": symbol,
                "shares": getattr(execution, "shares", None),
                "price": getattr(execution, "price", None),
                "commission": getattr(commission, "commission", None),
                "commission_currency": getattr(commission, "currency", None),
            }
        )
    return {
        "order_id": order_id,
        "perm_id": perm_id,
        "order_ref": str(getattr(order, "orderRef", "") or ""),
        "symbol": symbol,
        "con_id": int(getattr(contract, "conId", 0) or 0),
        "action": str(getattr(order, "action", "") or "").upper(),
        "quantity": getattr(order, "totalQuantity", None),
        "limit_price": getattr(order, "lmtPrice", None),
        "status": str(getattr(status, "status", "") or ""),
        "filled": getattr(status, "filled", None),
        "remaining": getattr(status, "remaining", None),
        "average_fill_price": getattr(status, "avgFillPrice", None),
        "done": bool(getattr(trade, "isDone", lambda: False)()),
        "fills": fills,
    }


def terminal_broker_snapshot_complete(snapshot: Mapping[str, object]) -> bool:
    """Require terminal broker state plus complete, USD-costed fills."""

    if snapshot.get("done") is not True:
        return False
    try:
        filled = float(snapshot.get("filled") or 0.0)
        fills = snapshot["fills"]
        fill_total = sum(float(fill["shares"]) for fill in fills)
        fills_complete = all(
            str(fill.get("exec_id") or "")
            and str(fill.get("time_utc") or "")
            and float(fill["price"]) > 0
            and float(fill["commission"]) >= 0
            and str(fill.get("commission_currency") or "").upper() == "USD"
            for fill in fills
        )
    except (KeyError, TypeError, ValueError):
        return False
    return filled >= 0 and abs(fill_total - filled) <= 1e-9 and fills_complete


def execution_price_for_ticker(contract: object, ticker: object):
    """Bind the shared execution ladder to one live ticker."""

    def price(
        mode: str,
        action: str,
        *,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
        ticker: object | None = None,
        elapsed_sec: float = 0.0,
        quote_stale: bool = False,
        open_shock: bool = False,
        no_progress_reprices: int = 0,
        arrival_ref: float | None = None,
        delay_recoveries: int = 0,
        delay_anchor_price: float | None = None,
        delay_sweep_anchor_price: float | None = None,
        delay_locked_price_dir: float | None = None,
    ) -> float | None:
        active_ticker = ticker or ticker_ref
        return execution_price(
            contract,
            active_ticker,
            mode,
            action,
            bid=bid if bid is not None else getattr(active_ticker, "bid", None),
            ask=ask if ask is not None else getattr(active_ticker, "ask", None),
            last=last if last is not None else getattr(active_ticker, "last", None),
            fallback_price=getattr(active_ticker, "close", None),
            custom_price=None,
            policy=EXECUTION_POLICY,
            elapsed_sec=elapsed_sec,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=no_progress_reprices,
            arrival_ref=arrival_ref,
            delay_recoveries=delay_recoveries,
            delay_anchor_price=delay_anchor_price,
            delay_sweep_anchor_price=delay_sweep_anchor_price,
            delay_locked_price_dir=delay_locked_price_dir,
        )

    ticker_ref = ticker
    return price
