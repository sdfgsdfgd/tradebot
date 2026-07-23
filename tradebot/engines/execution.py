"""Canonical quote normalization and deterministic limit-order pricing."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time
from time import monotonic


def _quote_num_actionable(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _quote_num_display(value: float | None) -> float | None:
    return _quote_num_actionable(value)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and abs(number) <= 1e307 else None


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


def _sanitize_nbbo_extremes(
    bid: float | None,
    ask: float | None,
    *,
    ref_price: float | None,
    max_ref_mult: float = 2.5,
    abs_floor: float = 5.0,
) -> tuple[float | None, float | None]:
    """Drop crossed or implausibly distant NBBO sides."""
    ref = _quote_num_display(ref_price)
    if ref is None:
        return bid, ask
    try:
        mult = max(1.25, float(max_ref_mult))
    except (TypeError, ValueError):
        mult = 2.5
    try:
        floor = max(0.0, float(abs_floor))
    except (TypeError, ValueError):
        floor = 5.0
    if ask is not None and ask > (ref * mult) and (ask - ref) > floor:
        ask = None
    if bid is not None and bid < (ref / mult) and (ref - bid) > floor:
        bid = None
    if bid is not None and ask is not None and bid > ask:
        return None, None
    return bid, ask


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0 or bid > ask:
        return None
    return (bid + ask) / 2.0


def _ticker_close(ticker: object) -> float | None:
    for attr in ("close", "prevLast"):
        value = _quote_num_display(getattr(ticker, attr, None))
        if value is not None:
            return value
    return None


def _ticker_price(ticker: object) -> float | None:
    """Resolve the shared display/reference price from one live ticker."""
    bid, ask, last = _sanitize_nbbo(
        getattr(ticker, "bid", None),
        getattr(ticker, "ask", None),
        getattr(ticker, "last", None),
    )
    midpoint = _midpoint(bid, ask)
    if midpoint is not None:
        return midpoint
    if last is not None:
        return last
    try:
        market_price = ticker.marketPrice()
    except Exception:
        market_price = None
    return _quote_num_display(market_price) or _ticker_close(ticker)


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
            low_edge, increment = float(row[0]), float(row[1])
        except (TypeError, ValueError):
            continue
        if increment > 0:
            rows.append((max(0.0, low_edge), increment))
    if not rows:
        return None
    rows.sort(key=lambda entry: entry[0])
    try:
        ref = max(0.0, float(ref_price) if ref_price is not None else 0.0)
    except (TypeError, ValueError):
        ref = 0.0
    selected = rows[0][1]
    for low_edge, increment in rows:
        if ref < low_edge:
            break
        selected = increment
    return selected if selected > 0 else None


def _tick_size(contract: object, ticker: object | None, ref_price: float | None) -> float:
    for source in (ticker, contract):
        if source is None:
            continue
        ladder_tick = _price_increment_from_ladder(
            getattr(source, "tbPriceIncrements", None),
            ref_price=ref_price,
        )
        if ladder_tick is not None:
            return ladder_tick
    for source in (ticker, contract):
        if source is None:
            continue
        try:
            tick = float(getattr(source, "minTick", None))
        except (TypeError, ValueError):
            continue
        if tick > 0:
            return tick
    if getattr(contract, "secType", None) in ("OPT", "FOP"):
        return 0.05 if ref_price is not None and ref_price >= 3 else 0.01
    return 0.01


def _round_to_tick(value: float | None, tick: float) -> float | None:
    if value is None:
        return None
    return round(value / tick) * tick if tick else value


def _tick_decimals(tick: float) -> int:
    text = f"{tick:.10f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def _optimistic_price(
    bid: float | None,
    ask: float | None,
    mid: float | None,
    action: str,
) -> float | None:
    if mid is None:
        return bid if action == "BUY" else ask
    if action == "BUY":
        return mid if bid is None else (mid + bid) / 2.0
    return mid if ask is None else (mid + ask) / 2.0


def _aggressive_price(
    bid: float | None,
    ask: float | None,
    mid: float | None,
    action: str,
) -> float | None:
    if action == "BUY":
        if ask is None:
            return mid or bid
        return ask if mid is None else (mid + ask) / 2.0
    if bid is None:
        return mid or ask
    return bid if mid is None else (mid + bid) / 2.0


def _cross_price(bid: float | None, ask: float | None, action: str) -> float | None:
    return ask if action == "BUY" else bid


_EXEC_LADDER_OPTIMISTIC_SEC = 6.0
_EXEC_LADDER_MID_SEC = 6.0
_EXEC_LADDER_AGGRESSIVE_SEC = 6.0
_EXEC_LADDER_CROSS_SEC = 6.0
_EXEC_RELENTLESS_TIMEOUT_SEC = 5 * 60.0
_EXEC_LADDER_TIMEOUT_SEC = 5 * 60.0 + 45.0
_EXEC_LADDER_PHASES_SEC = (
    _EXEC_LADDER_OPTIMISTIC_SEC
    + _EXEC_LADDER_MID_SEC
    + _EXEC_LADDER_AGGRESSIVE_SEC
    + _EXEC_LADDER_CROSS_SEC
)
_EXEC_AUTO_TIMEOUT_SEC = _EXEC_LADDER_PHASES_SEC + _EXEC_RELENTLESS_TIMEOUT_SEC


def _exec_ladder_mode(elapsed_sec: float) -> str | None:
    """Advance OPTIMISTIC -> MID -> AGGRESSIVE -> CROSS in six-second phases."""
    try:
        elapsed = max(0.0, float(elapsed_sec))
    except (TypeError, ValueError):
        elapsed = 0.0
    for mode, duration in (
        ("OPTIMISTIC", _EXEC_LADDER_OPTIMISTIC_SEC),
        ("MID", _EXEC_LADDER_MID_SEC),
        ("AGGRESSIVE", _EXEC_LADDER_AGGRESSIVE_SEC),
        ("CROSS", _EXEC_LADDER_CROSS_SEC),
    ):
        if elapsed < duration:
            return mode
        elapsed -= duration
    return None


def _exec_chase_mode(elapsed_sec: float, *, selected_mode: str | None = "AUTO") -> str | None:
    cleaned = str(selected_mode or "AUTO").strip().upper()
    try:
        elapsed = max(0.0, float(elapsed_sec))
    except (TypeError, ValueError):
        elapsed = 0.0
    if cleaned in ("RELENTLESS", "RELENTLESS_DELAY"):
        return cleaned if elapsed <= _EXEC_RELENTLESS_TIMEOUT_SEC else None
    if cleaned and cleaned not in ("AUTO", "LADDER"):
        return cleaned if elapsed <= _EXEC_LADDER_TIMEOUT_SEC else None
    ladder = _exec_ladder_mode(elapsed)
    if ladder is not None:
        return ladder
    relentless_elapsed = max(0.0, elapsed - _EXEC_LADDER_PHASES_SEC)
    return "RELENTLESS" if relentless_elapsed <= _EXEC_RELENTLESS_TIMEOUT_SEC else None


def _exec_chase_quote_signature(
    bid: float | None,
    ask: float | None,
    last: float | None,
) -> tuple[float | None, float | None, float | None]:
    clean_bid, clean_ask, clean_last = _sanitize_nbbo(bid, ask, last)
    clean_bid, clean_ask = _sanitize_nbbo_extremes(
        clean_bid,
        clean_ask,
        ref_price=clean_last,
    )
    return clean_bid, clean_ask, clean_last


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
    if prev_mode is None or str(mode_now) != str(prev_mode):
        return True
    if prev_quote_signature is None or last_reprice_sec is None:
        return True
    elapsed = max(0.0, float(now_sec) - float(last_reprice_sec))
    return elapsed >= float(min_interval_sec)


def _limit_price_for_mode(
    bid: float | None,
    ask: float | None,
    last: float | None,
    *,
    action: str,
    mode: str,
    ref_price: float | None = None,
) -> float | None:
    bid, ask, last = _sanitize_nbbo(bid, ask, last)
    ref = _quote_num_actionable(ref_price) or last
    bid, ask = _sanitize_nbbo_extremes(bid, ask, ref_price=ref)
    mid = _midpoint(bid, ask)
    cleaned = str(mode or "").strip().upper()
    if cleaned in ("CROSS", "RELENTLESS", "RELENTLESS_DELAY"):
        value = _cross_price(bid, ask, action)
    elif cleaned == "MID":
        value = mid
    elif cleaned == "OPTIMISTIC":
        value = _optimistic_price(bid, ask, mid, action)
    elif cleaned == "AGGRESSIVE":
        value = _aggressive_price(bid, ask, mid, action)
    else:
        value = mid
    value = value if value is not None else (mid if mid is not None else last)
    return value if value is not None and value > 0 else None


_PRICE_HINT_RE = re.compile(r"(?<!\d)(\d[\d,]*\.\d+)")


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """One deterministic policy for relentless pricing and recovery."""

    base_cross_sec: float = 2.0
    step_sec: float = 3.0
    shock_step_sec: float = 1.5
    min_reprice_sec: float = 0.75
    shock_min_reprice_sec: float = 0.35
    hyper_min_reprice_sec: float = 0.25
    open_shock_window_sec: float = 120.0
    stale_top_age_sec: float = 2.0
    spread_pressure_trigger: float = 2.0
    spread_pressure_hyper: float = 3.5
    max_edge_ticks: int = 40
    delay_recover_attempts: int = 24
    delay_recover_cooldown_sec: float = 0.35
    delay_recover_window_sec: float = 25.0
    delay_recover_settle_sec: float = 2.0
    delay_favorable_pct: float = 0.06
    delay_adverse_pct: float = 0.12
    delay_favorable_spread_mult: float = 0.8
    delay_adverse_spread_mult: float = 2.0
    delay_favorable_max_ticks: int = 20
    delay_adverse_max_ticks: int = 60
    delay_shrink_per_reject: float = 0.75

    def in_open_shock(self, now: time) -> bool:
        if now < time(9, 30) or now >= time(16, 0):
            return False
        elapsed = (now.hour * 3600) + (now.minute * 60) + now.second - (9 * 3600 + 30 * 60)
        return max(0, elapsed) <= int(self.open_shock_window_sec)

    def quote_is_stale(
        self,
        *,
        ticker: object | None,
        bid: float | None,
        ask: float | None,
        last: float | None,
        now_sec: float | None = None,
    ) -> bool:
        if not ((bid is not None and ask is not None and bid <= ask) or last is not None):
            return True
        if ticker is None:
            return False
        updated = getattr(ticker, "tbTopQuoteUpdatedMono", None)
        try:
            age = max(0.0, (monotonic() if now_sec is None else now_sec) - float(updated))
        except (TypeError, ValueError):
            return False
        return age >= self.stale_top_age_sec

    def spread_pressure(
        self,
        *,
        spread: float | None,
        tick: float,
        recent_spreads: Iterable[float] = (),
    ) -> float:
        if spread is None or spread <= 0 or tick <= 0:
            return 1.0
        recent = [float(value) for value in list(recent_spreads)[-24:] if float(value) > 0]
        recent.sort()
        baseline = recent[len(recent) // 2] if recent else float(spread)
        return max(0.5, float(spread) / max(float(tick), baseline))

    def reprice_interval(
        self,
        *,
        quote_stale: bool,
        open_shock: bool,
        no_progress_reprices: int,
        spread_pressure: float,
    ) -> float:
        if open_shock and (
            no_progress_reprices >= 2 or spread_pressure >= self.spread_pressure_trigger
        ):
            return self.hyper_min_reprice_sec
        if quote_stale or open_shock or no_progress_reprices >= 4:
            return self.shock_min_reprice_sec
        return self.min_reprice_sec

    def relentless_price(
        self,
        *,
        action: str,
        bid: float | None,
        ask: float | None,
        last_ref: float | None,
        tick: float,
        elapsed_sec: float,
        quote_stale: bool,
        open_shock: bool,
        no_progress_reprices: int,
        arrival_ref: float | None,
        recent_spreads: Iterable[float] = (),
        direction_sign_override: float | None = None,
    ) -> float | None:
        cross = _round_to_tick(ask if action == "BUY" else bid, tick)
        if cross is None:
            cross = _round_to_tick(last_ref, tick)
        if cross is None:
            return None
        spread = float(ask) - float(bid) if bid is not None and ask is not None and ask >= bid else None
        if spread is None or spread <= 0:
            spread = tick * 2.0
        pressure = self.spread_pressure(
            spread=spread,
            tick=tick,
            recent_spreads=recent_spreads,
        )
        cap_mult = 24.0 if open_shock else 12.0
        if pressure >= self.spread_pressure_trigger:
            cap_mult = max(cap_mult, 18.0)
        step = max(tick, min(spread / 3.0, tick * cap_mult))
        if pressure >= self.spread_pressure_hyper:
            step = max(step, tick * 2.0)

        elapsed = max(0.0, float(elapsed_sec))
        if elapsed < self.base_cross_sec:
            edge_ticks = 0
        else:
            interval = self.shock_step_sec if open_shock else self.step_sec
            if quote_stale:
                interval *= 0.5
            if no_progress_reprices >= 2:
                interval *= 0.5
            edge_ticks = 1 + int((elapsed - self.base_cross_sec) // max(interval, 0.5))
        if quote_stale:
            edge_ticks += 2
        if open_shock:
            edge_ticks += 1
        if no_progress_reprices >= 1:
            edge_ticks += min(8, int(no_progress_reprices))
        if pressure >= self.spread_pressure_trigger:
            edge_ticks += min(4, int(pressure))
        if pressure >= self.spread_pressure_hyper:
            edge_ticks = max(edge_ticks, 6)

        current_ref = _midpoint(bid, ask) or last_ref
        if arrival_ref is not None and current_ref is not None:
            adverse = (
                current_ref - arrival_ref if action == "BUY" else arrival_ref - current_ref
            )
            if adverse >= spread * 0.75:
                edge_ticks += 1
            if adverse >= spread * 1.5:
                edge_ticks += 2
            if adverse >= spread * 2.5:
                edge_ticks = max(edge_ticks, 8)
        if open_shock and no_progress_reprices >= 3:
            edge_ticks = max(edge_ticks, 6)
        edge_ticks = max(0, min(self.max_edge_ticks, edge_ticks))
        side_sign = 1.0 if action == "BUY" else -1.0
        direction = _finite_float(direction_sign_override)
        if direction is not None:
            side_sign = 1.0 if direction >= 0 else -1.0
        return _round_to_tick(cross + side_sign * step * edge_ticks, tick)

    def delay_cap_ticks(
        self,
        *,
        anchor_price: float,
        spread: float | None,
        tick: float,
        favorable: bool,
        recoveries: int,
    ) -> int:
        if tick <= 0 or anchor_price <= 0:
            return 1
        pct = self.delay_favorable_pct if favorable else self.delay_adverse_pct
        spread_mult = (
            self.delay_favorable_spread_mult if favorable else self.delay_adverse_spread_mult
        )
        hard_cap = (
            max(1, self.delay_favorable_max_ticks)
            if favorable
            else max(1, self.delay_adverse_max_ticks)
        )
        spread_cap = spread * spread_mult if spread is not None and spread > 0 else hard_cap * tick
        cap_ticks = min(anchor_price * pct / tick, spread_cap / tick, float(hard_cap))
        shrink = min(0.95, max(0.25, self.delay_shrink_per_reject)) ** max(
            0, int(recoveries) - 1
        )
        return max(1, min(hard_cap, int(math.floor(max(1.0, cap_ticks * shrink)))))

    def delay_price(
        self,
        *,
        action: str,
        bid: float | None,
        ask: float | None,
        last_ref: float | None,
        tick: float,
        recoveries: int,
        cap_price: float | None = None,
        sweep_anchor_price: float | None = None,
    ) -> float | None:
        cross = _round_to_tick(ask if action == "BUY" else bid, tick)
        if cross is None:
            cross = _round_to_tick(last_ref, tick)
        if cross is None:
            return None
        mid = _round_to_tick(_midpoint(bid, ask), tick)
        fallback = _round_to_tick(last_ref, tick)
        anchor = _round_to_tick(_finite_float(sweep_anchor_price), tick)
        if anchor is None:
            anchor = mid if mid is not None else (fallback if fallback is not None else cross)
        if anchor is None or anchor <= 0:
            return None
        recoveries = max(1, int(recoveries))
        spread = float(ask) - float(bid) if ask is not None and bid is not None and ask >= bid else None
        favorable, leg_sign = self.delay_leg(action, recoveries)
        cap_ticks = self.delay_cap_ticks(
            anchor_price=anchor,
            spread=spread,
            tick=tick,
            favorable=favorable,
            recoveries=recoveries,
        )
        target = anchor + leg_sign * tick * max(1, min(cap_ticks, 1 + (recoveries - 1) // 2))
        cap = _finite_float(cap_price)
        if cap is not None and cap > 0:
            target = min(target, cap) if action == "BUY" else max(target, cap)
        rounded = _round_to_tick(target, tick)
        return float(rounded) if rounded is not None and rounded > 0 else None

    def delay_sweep_span(self) -> int:
        span = max(2, int(self.delay_recover_attempts))
        return span if span % 2 == 0 else span + 1

    def delay_next_step(self, prior_recoveries: int) -> int:
        return 1 + max(0, int(prior_recoveries)) % self.delay_sweep_span()

    @staticmethod
    def delay_leg(action: str, recoveries: int) -> tuple[bool, float]:
        favorable = max(1, int(recoveries)) % 2 == 1
        side_sign = 1.0 if str(action or "").strip().upper() == "BUY" else -1.0
        return favorable, (-side_sign if favorable else side_sign)

    @staticmethod
    def price_hint_from_error(message: str) -> float | None:
        values: list[float] = []
        for match in _PRICE_HINT_RE.finditer(str(message or "").strip()):
            try:
                value = float(match.group(1).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if value > 0 and math.isfinite(value):
                values.append(value)
        return values[-1] if values else None


EXECUTION_POLICY = ExecutionPolicy()


def execution_mode_label(mode: str) -> str:
    """Return the compact operator label for an execution mode."""

    return {
        "RELENTLESS": "RLT",
        "RELENTLESS_DELAY": "RLT⚔Delay",
        "OPTIMISTIC": "OPT",
        "AGGRESSIVE": "AGG",
    }.get(mode, mode)


def execution_price(
    contract: object,
    ticker: object | None,
    mode: str,
    action: str,
    *,
    bid: float | None,
    ask: float | None,
    last: float | None,
    fallback_price: float | None,
    custom_price: float | None,
    policy: ExecutionPolicy = EXECUTION_POLICY,
    elapsed_sec: float = 0.0,
    quote_stale: bool = False,
    open_shock: bool = False,
    no_progress_reprices: int = 0,
    arrival_ref: float | None = None,
    recent_spreads: Iterable[float] = (),
    delay_recoveries: int = 0,
    delay_anchor_price: float | None = None,
    delay_sweep_anchor_price: float | None = None,
    delay_locked_price_dir: float | None = None,
) -> float | None:
    """Price one limit-order mode from normalized market and policy inputs."""

    last_ref = last if last is not None else (
        bid if bid is not None else (ask if ask is not None else fallback_price)
    )
    tick = _tick_size(contract, ticker, last_ref)
    if mode == "CUSTOM":
        custom = _round_to_tick(custom_price, tick)
        return custom if custom is not None else _round_to_tick(last_ref, tick)
    if mode in ("RELENTLESS", "RELENTLESS_DELAY") and (
        mode == "RELENTLESS" or int(delay_recoveries) <= 0
    ):
        direction = _finite_float(delay_locked_price_dir)
        if direction is not None:
            direction = 1.0 if direction >= 0 else -1.0
        return policy.relentless_price(
            action=action,
            bid=bid,
            ask=ask,
            last_ref=last_ref,
            tick=tick,
            elapsed_sec=elapsed_sec,
            quote_stale=bool(quote_stale),
            open_shock=bool(open_shock),
            no_progress_reprices=int(no_progress_reprices),
            arrival_ref=arrival_ref,
            recent_spreads=recent_spreads,
            direction_sign_override=direction if mode == "RELENTLESS_DELAY" else None,
        )
    if mode == "RELENTLESS_DELAY":
        return policy.delay_price(
            action=action,
            bid=bid,
            ask=ask,
            last_ref=last_ref,
            tick=tick,
            recoveries=int(delay_recoveries),
            cap_price=delay_anchor_price,
            sweep_anchor_price=delay_sweep_anchor_price,
        )
    value = _limit_price_for_mode(bid, ask, last_ref, action=action, mode=mode)
    return _round_to_tick(value if value is not None else last_ref, tick)
