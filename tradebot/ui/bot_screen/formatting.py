"""Pure bot-screen labels, rows, and payload path helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ib_insync import PortfolioItem
from rich.text import Text

from ...engine import normalize_spot_entry_signal
from ...time_utils import to_et as _to_et_shared
from ..bot_models import _BotInstance, _BotOrder, _BotPreset
from ..common import _fmt_quote, _pnl_text, _safe_num, _unrealized_pnl_values

@dataclass
class _InstancePnlState:
    realized_total: float = 0.0
    realized_seen_by_conid: dict[int, float] = field(default_factory=dict)


def _fmt_pct(value: float) -> str:
    return f"{value:.0f}%"


def _version_tag(raw: str | None) -> str | None:
    value = str(raw or "").strip()
    if not value:
        return None
    match = re.search(r"\d+(?:\.\d+)?", value)
    if match is not None:
        return f"v{match.group(0)}"
    return value if value.lower().startswith("v") else f"v{value}"


def _missing_signal_transport_keys(strategy: dict | None) -> tuple[str, ...]:
    if not isinstance(strategy, dict):
        return ("signal_bar_size", "signal_use_rth")
    missing: list[str] = []
    for key in ("signal_bar_size", "signal_use_rth"):
        if strategy.get(key) is None:
            missing.append(key)
    return tuple(missing)


def _clean_group_label(raw: str) -> str:
    """Shorten leaderboard group names for table display.

    Many generated groups include full metrics in the name. Keep the identifier part so the table
    stays readable, and rely on the numeric columns + the Selected preset panel for details.
    """
    value = str(raw or "").strip()
    if not value:
        return value
    cut = min(
        (value.index(token) for token in (" floor=", " roi/dd=", " pnl/dd=", " roi=", " pnl=", " dd%=", " win=", " tr=", " 2020=", " 2022=", " 2025=") if token in value),
        default=-1,
    )
    if cut >= 0:
        return value[:cut].strip()
    return value


def _legs_label(legs: list[dict]) -> str:
    if not legs:
        return "-"
    parts = []
    for leg in legs:
        action = str(leg.get("action", "?"))[:1].upper()
        right = str(leg.get("right", "?"))[:1].upper()
        m = leg.get("moneyness_pct", 0.0)
        try:
            m = float(m)
        except (TypeError, ValueError):
            m = 0.0
        parts.append(f"{action}{right} {m:+.1f}%")
    return " + ".join(parts)


def _legs_direction_hint(legs: list[dict], ema_directional: bool) -> str:
    if ema_directional:
        return "EMA (up/down)"
    if not legs:
        return "ANY"
    first = legs[0] if isinstance(legs, list) else None
    if not isinstance(first, dict):
        return "ANY"
    action = str(first.get("action", "")).upper()
    right = str(first.get("right", "")).upper()
    if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
        return "UP"
    if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
        return "DOWN"
    return "ANY"


def _preset_lines(preset: _BotPreset) -> list[Text]:
    entry = preset.entry
    metrics = entry.get("metrics", {})
    strat = entry.get("strategy", {})
    instrument = str(strat.get("instrument", "options") or "options").strip().lower()

    legs_label = _legs_label(strat.get("legs", []))
    if instrument == "spot":
        mapping = strat.get("directional_spot") if isinstance(strat.get("directional_spot"), dict) else None
        if mapping and mapping.get("up") and mapping.get("down"):
            legs_label = "SPOT (up/down)"
        else:
            legs_label = "SPOT"

        signal_bar = str(strat.get("signal_bar_size") or "").strip()
        entry_signal = normalize_spot_entry_signal(strat.get("entry_signal"))
        confirm = strat.get("entry_confirm_bars", 0)
        regime_mode = str(strat.get("regime_mode") or "ema").strip().lower()
        regime = str(strat.get("regime_ema_preset") or "").strip()
        regime_bar = str(strat.get("regime_bar_size") or "").strip()
        regime2_mode = str(strat.get("regime2_mode") or "off").strip().lower()
        regime2 = str(strat.get("regime2_ema_preset") or "").strip()
        regime2_bar = str(strat.get("regime2_bar_size") or "").strip()
        mode_parts: list[str] = []
        if entry_signal == "orb":
            window = strat.get("orb_window_mins", "?")
            rr = strat.get("orb_risk_reward", "?")
            tgt_mode = str(strat.get("orb_target_mode", "rr") or "rr").strip().lower()
            if tgt_mode not in ("rr", "or_range"):
                tgt_mode = "rr"
            mode_parts.append(f"ORB: {window}m {tgt_mode} rr={rr}")
        else:
            mode_parts.append(f"EMA: {strat.get('ema_preset', '')} cross c{confirm}")
        if regime_mode == "supertrend":
            atr_p = strat.get("supertrend_atr_period", "?")
            mult = strat.get("supertrend_multiplier", "?")
            src = str(strat.get("supertrend_source", "hl2") or "hl2").strip()
            mode_parts.append(f"Regime: ST({atr_p},{mult},{src}) @ {regime_bar or '?'}")
        elif regime:
            mode_parts.append(f"Regime: {regime} @ {regime_bar or '?'}")
        if regime2_mode == "supertrend":
            atr_p = strat.get("regime2_supertrend_atr_period", "?")
            mult = strat.get("regime2_supertrend_multiplier", "?")
            src = str(strat.get("regime2_supertrend_source", "hl2") or "hl2").strip()
            mode_parts.append(f"Regime2: ST({atr_p},{mult},{src}) @ {regime2_bar or '?'}")
        elif regime2_mode == "ema" and regime2:
            mode_parts.append(f"Regime2: {regime2} @ {regime2_bar or '?'}")
        exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
        if exit_mode == "atr":
            atr_p = strat.get("spot_atr_period", "?")
            pt_mult = strat.get("spot_pt_atr_mult", "?")
            sl_mult = strat.get("spot_sl_atr_mult", "?")
            mode_parts.append(f"Exit: ATR({atr_p}) PTx{pt_mult} SLx{sl_mult}")
        if signal_bar:
            mode_parts.append(f"Bar: {signal_bar}")
        mode = "  ".join(mode_parts)
    else:
        pt = float(strat.get("profit_target", 0.0)) * 100.0
        sl = float(strat.get("stop_loss", 0.0)) * 100.0
        mode = (
            f"DTE: {strat.get('dte', '?')}  "
            f"PT: {_fmt_pct(pt)}  SL: {_fmt_pct(sl)}  "
            f"EMA: {strat.get('ema_preset', '')}"
        )

    pnl = float(metrics.get("pnl", 0.0))
    try:
        dd = float(metrics.get("max_drawdown")) if metrics.get("max_drawdown") is not None else None
    except (TypeError, ValueError):
        dd = None
    pnl_over_dd = None
    try:
        pnl_over_dd = (
            float(metrics.get("pnl_over_dd"))
            if metrics.get("pnl_over_dd") is not None
            else (pnl / dd if dd and dd > 0 else None)
        )
    except (TypeError, ValueError, ZeroDivisionError):
        pnl_over_dd = None
    lines = [
        Text(preset.group),
        Text(f"Legs: {legs_label}", style="dim"),
        Text(mode, style="dim"),
        Text(
            f"PnL: {pnl:.2f}  "
            f"Win: {float(metrics.get('win_rate', 0.0)) * 100.0:.1f}%  "
            f"Trades: {int(metrics.get('trades', 0))}",
            style="dim",
        ),
    ]
    if dd is not None:
        extra = f"DD: {dd:.2f}"
        if pnl_over_dd is not None:
            extra += f"  PnL/DD: {pnl_over_dd:.2f}"
        lines.append(Text(extra, style="dim"))
    return lines


def _order_lines(order: _BotOrder) -> list[Text]:
    legs = order.legs or []
    legs_line: str | None = None
    if len(legs) == 1 and order.order_contract.secType != "BAG":
        contract = legs[0].contract
        local = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
        sec_type = getattr(contract, "secType", "") or ""
        if sec_type == "STK":
            header = f"{local} STK".strip()
        elif sec_type == "FUT":
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            header = f"{local} {expiry} FUT".strip()
        else:
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "") or "?"
            right = getattr(contract, "right", "") or "?"
            strike = getattr(contract, "strike", None)
            header = f"{local} {expiry}{right} {strike}"
    else:
        symbol = getattr(order.order_contract, "symbol", "") or getattr(
            order.underlying, "symbol", ""
        )
        header = f"{symbol} BAG ({len(legs)} legs)"
        legs_desc: list[str] = []
        for leg in legs[:4]:
            contract = leg.contract
            local = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
            legs_desc.append(f"{leg.action[:1]}{local}".strip())
        if legs_desc:
            legs_line = "  ".join(legs_desc) + ("  …" if len(legs) > 4 else "")

    parts: list[Text] = [Text(header, style="dim")]
    if legs_line:
        parts.append(Text(f"Legs: {legs_line}", style="dim"))
    parts.append(Text(f"Side: {order.action}  Qty: {order.quantity}", style="dim"))
    parts.append(Text(f"Limit: {_fmt_quote(order.limit_price)}", style="dim"))
    parts.append(
        Text(
            f"Bid: {_fmt_quote(order.bid)}  Ask: {_fmt_quote(order.ask)}  "
            f"Last: {_fmt_quote(order.last)}",
            style="dim",
        )
    )
    return parts


def _parse_entry_days(raw: str) -> list[str]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return []
    normalized = (
        cleaned.replace(";", ",")
        .replace("|", ",")
        .replace("/", ",")
        .replace("\\", ",")
        .replace(" ", ",")
    )
    out: list[str] = []
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        key = token.upper()[:3]
        if key in ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"):
            out.append(key.title())
    return out


def _filters_for_group(payload: dict | None, group_name: str) -> dict | None:
    if not payload:
        return None
    for group in payload.get("groups", []):
        if str(group.get("name")) == group_name:
            return group.get("filters") or None
    return None


def _contract_expiry_date(raw: object) -> date | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) >= 8 and text[:8].isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        except ValueError:
            return None
    if len(text) >= 6 and text[:6].isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), 1)
        except ValueError:
            return None
    return None


def _get_path(root: object, path: str) -> object:
    current: object = root
    for part in str(path).split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _set_path(root: object, path: str, value: object) -> None:
    if not isinstance(path, str) or not path:
        return
    parts = path.split(".")
    current = root
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        next_part = parts[idx + 1] if not last else ""

        if isinstance(current, dict):
            if last:
                current[part] = value
                return
            nxt = current.get(part)
            if not isinstance(nxt, (dict, list)):
                nxt = [] if next_part.isdigit() else {}
                current[part] = nxt
            current = nxt
            continue

        if isinstance(current, list):
            try:
                list_idx = int(part)
            except (TypeError, ValueError):
                return
            while len(current) <= list_idx:
                current.append({})
            if last:
                current[list_idx] = value
                return
            nxt = current[list_idx]
            if not isinstance(nxt, (dict, list)):
                nxt = [] if next_part.isdigit() else {}
                current[list_idx] = nxt
            current = nxt
            continue
        return


def _center_table_cell(value: object) -> Text:
    if isinstance(value, Text):
        centered = value.copy()
    else:
        centered = Text("" if value is None else str(value))
    centered.justify = "center"
    return centered


def _center_table_row(*cells: object) -> tuple[Text, ...]:
    return tuple(_center_table_cell(cell) for cell in cells)


def _order_row(order: _BotOrder) -> tuple[str, str, str, str, str, str, str, str, str]:
    ts = _to_et_shared(order.created_at, naive_ts_mode="et", default_naive_ts_mode="et").strftime("%H:%M:%S")
    inst = str(order.instance_id)
    contract = order.order_contract
    if contract.secType == "BAG" or len(order.legs) > 1:
        symbol = getattr(contract, "symbol", "") or getattr(order.underlying, "symbol", "") or "?"
        local = f"{symbol} BAG {len(order.legs)}L"
    else:
        leg_contract = order.legs[0].contract if order.legs else contract
        local = getattr(leg_contract, "localSymbol", "") or getattr(leg_contract, "symbol", "") or "?"
    local = str(local)[:12]
    side = order.action[:1]
    qty = str(int(order.quantity))
    limit = _fmt_quote(order.limit_price)
    bid = _fmt_quote(order.bid)
    ask = _fmt_quote(order.ask)
    bid_ask = f"{bid}/{ask}"
    return (ts, inst, side, qty, local, limit, bid_ask, "", "")


def _position_as_order_row(
    item: PortfolioItem,
    *,
    scope: int | None,
    daily: float | None = None,
    unreal: float | None = None,
    realized: float | None = None,
    market_price: float | None = None,
    entry_now_text: Text | None = None,
    px_change_text: Text | None = None,
) -> tuple[str, str, str, str, str, str, str, Text, Text]:
    contract = getattr(item, "contract", None)
    sec_type = str(getattr(contract, "secType", "") or "") if contract is not None else ""
    symbol = str(getattr(contract, "symbol", "") or "") if contract is not None else ""

    local = ""
    if contract is not None:
        if sec_type == "STK":
            local = symbol
        elif sec_type == "FUT":
            local = str(getattr(contract, "localSymbol", "") or symbol or "?")
        elif sec_type in ("OPT", "FOP"):
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
            right = str(getattr(contract, "right", "") or "")
            strike = getattr(contract, "strike", None)
            strike_s = ""
            if strike is not None:
                try:
                    strike_s = f"{float(strike):.1f}"
                except (TypeError, ValueError):
                    strike_s = ""
            local = f"{symbol} {expiry}{right[:1]} {strike_s}".strip()
        else:
            local = str(getattr(contract, "localSymbol", "") or symbol or "?")
    local = str(local or "?")[:12]

    try:
        pos = float(getattr(item, "position", 0.0) or 0.0)
    except (TypeError, ValueError):
        pos = 0.0
    side = "L" if pos > 0 else "S" if pos < 0 else ""
    abs_pos = abs(pos)
    qty = f"{abs_pos:.2f}"
    if abs_pos.is_integer():
        qty = str(int(abs_pos))

    avg = _safe_num(getattr(item, "averageCost", None))
    mkt = _safe_num(market_price)
    if mkt is None:
        mkt = _safe_num(getattr(item, "marketPrice", None))
    if unreal is None:
        unreal, _ = _unrealized_pnl_values(item)
    if realized is None:
        realized = _safe_num(getattr(item, "realizedPNL", None))

    def _pnl_style(value: float | None) -> str:
        if value is None:
            return "dim"
        if value > 0:
            return "green"
        if value < 0:
            return "red"
        return ""

    if daily is None and unreal is None:
        unreal_cell = Text("warming...", style="dim")
    else:
        unreal_cell = Text("")
        unreal_cell.append("D ", style="dim")
        if daily is None:
            unreal_cell.append("warm", style="dim")
        else:
            unreal_cell.append(f"{float(daily):+,.2f}", style=_pnl_style(float(daily)))
        unreal_cell.append(" · ", style="dim")
        unreal_cell.append("U ", style="dim")
        if unreal is None:
            unreal_cell.append("warm", style="dim")
        else:
            unreal_cell.append(f"{float(unreal):+,.2f}", style=_pnl_style(float(unreal)))
    realized_cell = _pnl_text(realized) if realized is not None else Text("")

    return (
        "POS",
        str(int(scope)) if scope is not None else "",
        side,
        qty,
        local,
        entry_now_text if entry_now_text is not None else _fmt_quote(avg),
        px_change_text if px_change_text is not None else _fmt_quote(mkt),
        unreal_cell,
        realized_cell,
    )


def _positions_subheader_row() -> tuple[Text | str, Text | str, Text | str, Text | str, Text | str, Text | str, Text | str, Text | str, Text | str]:
    style = "bold #7f8fa0"
    return (
        "",
        "",
        "",
        Text("Qty", style=style),
        Text("Contract", style=style),
        Text("Entry¦Now", style=style),
        Text("Px 24-72", style=style),
        Text("PnL D|U (Pos)", style=style),
        Text("Realized", style=style),
    )


def _instance_pnl_cells(
    instance: _BotInstance,
    positions: list[PortfolioItem],
    *,
    pnl_value_fn=None,
) -> tuple[Text | str, Text | str]:
    con_ids = set(instance.touched_conids)
    if not con_ids:
        return "", ""
    has_match = False
    unreal_total = 0.0
    realized_total = 0.0
    for item in positions:
        try:
            con_id = int(getattr(item.contract, "conId", 0) or 0)
        except (TypeError, ValueError):
            con_id = 0
        if con_id not in con_ids:
            continue
        has_match = True
        if callable(pnl_value_fn):
            try:
                unreal, realized = pnl_value_fn(item)
            except Exception:
                unreal, _ = _unrealized_pnl_values(item)
                realized = _safe_num(getattr(item, "realizedPNL", None))
        else:
            unreal, _ = _unrealized_pnl_values(item)
            realized = _safe_num(getattr(item, "realizedPNL", None))
        if unreal is not None:
            unreal_total += float(unreal)
        if realized is not None:
            realized_total += float(realized)
    if not has_match:
        return "", ""
    return _pnl_text(unreal_total), _pnl_text(realized_total)
