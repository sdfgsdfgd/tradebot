"""Canonical scalar, enum, and leg parsers for backtest configuration."""

from __future__ import annotations

from datetime import date

from ..knobs.models import LegConfig, SpotLegConfig
from ..signals import parse_bar_size
from ..spot.fill_modes import SPOT_FILL_MODE_CLOSE, normalize_spot_fill_mode
from .cli_utils import parse_date as _parse_date_impl


_WEEKDAYS = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}


def _parse_date(value: str | None) -> date:
    if not value:
        raise ValueError("Config requires start/end dates in YYYY-MM-DD format")
    return _parse_date_impl(value)


def _parse_weekdays(days: list[str]) -> tuple[int, ...]:
    if not days:
        return (0, 1, 2, 3, 4)
    parsed: list[int] = []
    for day in days:
        if not isinstance(day, str):
            continue
        key = day.strip().upper()[:3]
        if key not in _WEEKDAYS:
            raise ValueError(f"Unknown weekday: {day}")
        parsed.append(_WEEKDAYS[key])
    if not parsed:
        return (0, 1, 2, 3, 4)
    return tuple(parsed)


def _parse_ema_preset(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in ("none", "null"):
            return None
        return cleaned
    return None


def _parse_bar_size(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid bar size: {value!r}")
    cleaned = value.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in ("none", "null", "same", "default"):
        return None
    if parse_bar_size(cleaned) is None:
        raise ValueError(f"Invalid bar size: {value!r}")
    return cleaned


def _parse_ema_entry_mode(value) -> str:
    if value is None:
        return "trend"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("trend", "state"):
            return "trend"
        if cleaned in ("cross", "crossover"):
            return "cross"
    raise ValueError(f"Invalid ema_entry_mode: {value!r} (expected 'trend' or 'cross')")


def _parse_flip_exit_mode(value) -> str:
    if value is None:
        return "entry"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("entry", "default", "same", "auto"):
            return "entry"
        if cleaned in ("state", "trend"):
            return "state"
        if cleaned in ("cross", "crossover"):
            return "cross"
    raise ValueError(
        f"Invalid flip_exit_mode: {value!r} (expected 'entry', 'state', or 'cross')"
    )


def _parse_flip_exit_gate_mode(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("off", "none", "disabled", "false", "0", "", "default"):
            return "off"
        if cleaned in ("regime", "bias"):
            return "regime"
        if cleaned in ("permission", "perm", "filters", "filter"):
            return "permission"
        if cleaned in ("regime_or_permission", "regime_or_perm", "bias_or_permission", "bias_or_perm"):
            return "regime_or_permission"
        if cleaned in ("regime_and_permission", "regime_and_perm", "bias_and_permission", "bias_and_perm"):
            return "regime_and_permission"
    raise ValueError(
        f"Invalid flip_exit_gate_mode: {value!r} "
        "(expected 'off', 'regime', 'permission', 'regime_or_permission', or 'regime_and_permission')"
    )


def _parse_direction_source(value) -> str:
    if value is None:
        return "ema"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("ema",):
            return "ema"
    raise ValueError(f"Invalid direction_source: {value!r} (expected 'ema')")


def _parse_spot_dual_branch_priority(value) -> str:
    if value is None:
        return "b_then_a"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("a_then_b", "a", "a_first", "a-first"):
            return "a_then_b"
        if cleaned in ("b_then_a", "b", "b_first", "b-first", "default", ""):
            return "b_then_a"
    raise ValueError(f"Invalid spot_dual_branch_priority: {value!r} (expected 'a_then_b' or 'b_then_a')")


def _parse_non_negative_int(value, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected integer, got: {value!r}") from None
    if parsed < 0:
        raise ValueError(f"Expected non-negative integer, got: {value!r}")
    return parsed


def _parse_directional_legs(raw) -> dict[str, tuple[LegConfig, ...]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("directional_legs must be an object")
    parsed: dict[str, tuple[LegConfig, ...]] = {}
    for key in ("up", "down"):
        legs_raw = raw.get(key)
        if not legs_raw:
            continue
        parsed[key] = _parse_legs(legs_raw) or ()
    return parsed or None


def _parse_legs(raw) -> tuple[LegConfig, ...] | None:
    if not raw:
        return None
    if not isinstance(raw, list):
        raise ValueError("legs must be a list")
    legs: list[LegConfig] = []
    for idx, leg in enumerate(raw, start=1):
        if not isinstance(leg, dict):
            raise ValueError(f"legs[{idx}] must be an object")
        action = str(leg.get("action", "")).strip().upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"legs[{idx}].action must be BUY or SELL")
        right = str(leg.get("right", "")).strip().upper()
        if right not in ("PUT", "CALL"):
            raise ValueError(f"legs[{idx}].right must be PUT or CALL")
        if "moneyness_pct" not in leg:
            raise ValueError(f"legs[{idx}] requires moneyness_pct")
        moneyness = float(leg["moneyness_pct"])
        qty = int(leg.get("qty", 1))
        if qty <= 0:
            raise ValueError(f"legs[{idx}].qty must be positive")
        legs.append(LegConfig(action=action, right=right, moneyness_pct=moneyness, qty=qty))
    return tuple(legs)


def _parse_entry_signal(value) -> str:
    if value is None:
        return "ema"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("ema", "default", ""):
            return "ema"
        if cleaned in ("orb", "opening_range", "opening_range_breakout", "opening-range"):
            return "orb"
    return "ema"


def _parse_orb_target_mode(value) -> str:
    if value is None:
        return "rr"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("rr", "risk_reward", "risk-reward", "riskreward", "default", ""):
            return "rr"
        if cleaned in ("or_range", "or-range", "range", "opening_range", "opening-range"):
            return "or_range"
    return "rr"


def _parse_spot_exit_mode(value) -> str:
    if value is None:
        return "pct"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("pct", "percent", "percentage", "fixed", "fixed_pct", "fixed-pct", "default"):
            return "pct"
        if cleaned in ("atr", "atr_pct", "atr-pct"):
            return "atr"
    return "pct"


def _parse_regime_mode(value) -> str:
    if value is None:
        return "ema"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("ema", "default", ""):
            return "ema"
        if cleaned in ("supertrend", "st"):
            return "supertrend"
    return "ema"


def _parse_regime_gate_mode(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("off", "none", "disabled", "false", "0"):
            return "off"
        if cleaned in ("ema", "default"):
            return "ema"
        if cleaned in ("supertrend", "st"):
            return "supertrend"
    return "off"


def _parse_regime2_apply_to(value) -> str:
    if value is None:
        return "both"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("off", "none", "disabled", "false", "0", "soft"):
            return "off"
        if cleaned in ("both", "all", "default", ""):
            return "both"
        if cleaned in ("long", "longs", "up", "buy"):
            return "longs"
        if cleaned in ("short", "shorts", "down", "sell"):
            return "shorts"
    return "both"


def _parse_regime2_bear_entry_mode(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("", "off", "none", "disabled", "false", "0"):
            return "off"
        if cleaned in ("supertrend", "st", "supertrend_confirmed", "st_confirmed"):
            return "supertrend"
    return "off"


def _parse_regime2_bear_takeover_mode(value) -> str:
    if value is None:
        return "always"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("", "always", "on", "true", "1"):
            return "always"
        if cleaned in ("hostile", "risk", "risk_or_panic"):
            return "hostile"
        if cleaned in ("riskoff", "off_risk"):
            return "riskoff"
        if cleaned in ("riskpanic", "panic"):
            return "riskpanic"
        if cleaned in ("shockdown", "shock_down", "shock-down"):
            return "shockdown"
        if cleaned in ("hostile_or_shockdown", "hostile+shockdown", "hostile_shockdown"):
            return "hostile_or_shockdown"
    return "always"


def _parse_regime2_crash_prearm_apply_to(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("", "off", "none", "disabled", "false", "0"):
            return "off"
        if cleaned in ("branch_b_longs", "b_longs", "branchb", "b"):
            return "branch_b_longs"
        if cleaned in ("all_longs", "longs", "all"):
            return "all_longs"
    return "off"


def _parse_regime2_clean_host_takeover_state(value) -> str:
    if value is None:
        return "trend_up_clean"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("", "trend_up_clean", "clean_up", "trendup", "clean"):
            return "trend_up_clean"
        if cleaned in ("crash_down", "crash", "crashdown"):
            return "crash_down"
        if cleaned in ("transition_up_hot", "transition", "repair_up", "hot_transition"):
            return "transition_up_hot"
        if cleaned in (
            "crash_or_transition_up_hot",
            "crash_or_transition",
            "crash+transition",
            "crash_transition",
        ):
            return "crash_or_transition_up_hot"
    return "trend_up_clean"


def _parse_tick_gate_mode(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("off", "none", "disabled", "false", "0"):
            return "off"
        if cleaned in ("raschke", "tick", "tick_width", "tickwidth"):
            return "raschke"
    return "off"


def _parse_tick_neutral_policy(value) -> str:
    if value is None:
        return "allow"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("allow", "pass", "permit"):
            return "allow"
        if cleaned in ("block", "deny", "none"):
            return "block"
    return "allow"


def _parse_tick_direction_policy(value) -> str:
    if value is None:
        return "both"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("both", "two_sided", "twosided", "long_short", "longshort"):
            return "both"
        if cleaned in ("wide_only", "wideonly", "long_only", "longonly", "wide_long", "widelong"):
            return "wide_only"
    return "both"


def _parse_spot_fill_mode(value, *, default: str) -> str:
    fallback = normalize_spot_fill_mode(default, default=SPOT_FILL_MODE_CLOSE)
    return normalize_spot_fill_mode(value, default=fallback)


def _parse_spot_next_open_session_mode(value) -> str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        aliases = {
            "full24": "tradable_24x5",
            "tradable": "tradable_24x5",
            "overnight_plus_extended": "tradable_24x5",
        }
        cleaned = aliases.get(cleaned, cleaned)
        if cleaned in ("auto", "rth", "extended", "always", "tradable_24x5"):
            return cleaned
    return "auto"


def _parse_spot_mark_to_market(value) -> str:
    if value is None:
        return "close"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("close", "mid", "bar_close"):
            return "close"
        if cleaned in ("liquidation", "liq", "bidask", "bid_ask", "bid-ask"):
            return "liquidation"
    return "close"


def _parse_spot_drawdown_mode(value) -> str:
    if value is None:
        return "close"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("close", "bar_close"):
            return "close"
        if cleaned in ("intrabar", "intra", "ohlc"):
            return "intrabar"
    return "close"


def _parse_spot_sizing_mode(value) -> str:
    if value is None:
        return "fixed"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("fixed", "qty", "quantity", "shares"):
            return "fixed"
        if cleaned in ("notional_pct", "notional", "pct_notional", "equity_pct"):
            return "notional_pct"
        if cleaned in ("risk_pct", "risk", "risk_percent", "risk_per_trade"):
            return "risk_pct"
    return "fixed"


def _parse_spot_resize_mode(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("off", "none", "disabled", "false", "0"):
            return "off"
        if cleaned in ("target", "adaptive", "on", "enabled", "1"):
            return "target"
    return "off"


def _parse_spot_resize_adaptive_mode(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("off", "none", "disabled", "false", "0"):
            return "off"
        if cleaned in ("atr",):
            return "atr"
        if cleaned in ("slope", "velocity", "vel"):
            return "slope"
        if cleaned in ("hybrid", "atr_slope", "slope_atr", "both"):
            return "hybrid"
    return "off"


def _parse_non_negative_float(value, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected float, got: {value!r}") from None
    if parsed < 0:
        raise ValueError(f"Expected non-negative float, got: {value!r}")
    return parsed


def _parse_positive_int(value, *, default: int) -> int:
    parsed = _parse_non_negative_int(value, default=default)
    return max(1, int(parsed))


def _parse_positive_float(value, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _parse_non_negative_float_or_default(value, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed >= 0 else float(default)


def _parse_supertrend_source(value) -> str:
    if value is None:
        return "hl2"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("hl2",):
            return "hl2"
        if cleaned in ("close", "c"):
            return "close"
    return "hl2"


def _parse_instrument(value) -> str:
    if value is None:
        return "options"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("options", "option", "opts"):
            return "options"
        if cleaned in ("spot", "stock", "equity", "shares", "futures"):
            return "spot"
    raise ValueError(f"Invalid instrument: {value!r} (expected 'options' or 'spot')")


def _parse_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected float, got: {value!r}") from None


def _parse_directional_spot(raw) -> dict[str, SpotLegConfig] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("directional_spot must be an object")
    parsed: dict[str, SpotLegConfig] = {}
    for key in ("up", "down"):
        leg_raw = raw.get(key)
        if not leg_raw:
            continue
        if not isinstance(leg_raw, dict):
            raise ValueError(f"directional_spot.{key} must be an object")
        action = str(leg_raw.get("action", "")).strip().upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"directional_spot.{key}.action must be BUY or SELL")
        qty = int(leg_raw.get("qty", 1))
        if qty <= 0:
            raise ValueError(f"directional_spot.{key}.qty must be positive")
        parsed[key] = SpotLegConfig(action=action, qty=qty)
    return parsed or None
