"""Config loading for synthetic backtests."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..signals import parse_bar_size
from .cli_utils import parse_date as _parse_date_impl
from ..knobs.models import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    SpotLegConfig,
    StrategyConfig,
    SyntheticConfig,
)

# region Constants
_WEEKDAYS = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}
# endregion


_MISSING = object()


@dataclass(frozen=True)
class _FieldSpec:
    parser: Callable[[object], object]
    default: object = _MISSING


def _field(parser: Callable[[object], object], default: object = _MISSING) -> _FieldSpec:
    return _FieldSpec(parser=parser, default=default)


def _identity(value):
    return value


# region Public API
def load_config(path: str | Path) -> ConfigBundle:
    raw = json.loads(Path(path).read_text())
    backtest_raw = raw.get("backtest", {}) if isinstance(raw, dict) else {}
    strategy_raw = raw.get("strategy", {}) if isinstance(raw, dict) else {}
    synthetic_raw = raw.get("synthetic", {}) if isinstance(raw, dict) else {}
    if not isinstance(backtest_raw, dict):
        backtest_raw = {}
    if not isinstance(strategy_raw, dict):
        strategy_raw = {}
    if not isinstance(synthetic_raw, dict):
        synthetic_raw = {}

    backtest = BacktestConfig(**_parse_from_schema(backtest_raw, _backtest_schema()))
    strategy = StrategyConfig(**_parse_from_schema(strategy_raw, _strategy_schema()))
    synthetic = SyntheticConfig(**_parse_from_schema(synthetic_raw, _synthetic_schema()))

    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def _parse_from_schema(
    raw: dict,
    schema: dict[str, Callable[[dict], object] | _FieldSpec],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for field, parser in schema.items():
        if isinstance(parser, _FieldSpec):
            if parser.default is _MISSING:
                value = raw.get(field)
            else:
                value = raw.get(field, parser.default)
            out[field] = parser.parser(value)
        else:
            out[field] = parser(raw)
    return out


def _backtest_schema() -> dict[str, _FieldSpec]:
    return {
        "start": _field(_parse_date),
        "end": _field(_parse_date),
        "bar_size": _field(_identity, "1 hour"),
        "use_rth": _field(bool, False),
        "starting_cash": _field(float, 100_000.0),
        "risk_free_rate": _field(float, 0.02),
        "cache_dir": _field(Path, "db"),
        "calibration_dir": _field(Path, "db/calibration"),
        "output_dir": _field(Path, "backtests/out"),
        "calibrate": _field(bool, False),
        "offline": _field(bool, False),
    }


def _strategy_schema() -> dict[str, _FieldSpec]:
    return {
        "name": _field(_identity, "credit_spread"),
        "instrument": _field(_parse_instrument, "options"),
        "symbol": _field(_identity, "MNQ"),
        "exchange": _field(_identity, None),
        "right": _field(lambda value: str(value or "PUT").upper(), "PUT"),
        "entry_days": _field(_parse_weekdays, []),
        "max_entries_per_day": _field(lambda value: _parse_non_negative_int(value, default=1), 1),
        "max_open_trades": _field(lambda value: _parse_non_negative_int(value, default=1), 1),
        "dte": _field(int, 0),
        "otm_pct": _field(float, 2.5),
        "width_pct": _field(float, 1.0),
        "profit_target": _field(float, 0.5),
        "stop_loss": _field(float, 0.35),
        "exit_dte": _field(int, 0),
        "quantity": _field(int, 1),
        "stop_loss_basis": _field(_identity, "max_loss"),
        "min_credit": _field(_parse_optional_float, None),
        "ema_preset": _field(_parse_ema_preset, None),
        "ema_entry_mode": _field(_parse_ema_entry_mode, None),
        "entry_confirm_bars": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "regime_ema_preset": _field(_parse_ema_preset, None),
        "regime_bar_size": _field(_parse_bar_size, None),
        "ema_directional": _field(bool, False),
        "exit_on_signal_flip": _field(bool, False),
        "flip_exit_mode": _field(_parse_flip_exit_mode, None),
        "flip_exit_gate_mode": _field(_parse_flip_exit_gate_mode, None),
        "flip_exit_min_hold_bars": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "flip_exit_only_if_profit": _field(bool, False),
        "direction_source": _field(_parse_direction_source, None),
        "directional_legs": _field(_parse_directional_legs, None),
        "directional_spot": _field(_parse_directional_spot, None),
        "legs": _field(_parse_legs, None),
        "filters": _field(_parse_filters, None),
        "spot_profit_target_pct": _field(_parse_optional_float, None),
        "spot_stop_loss_pct": _field(_parse_optional_float, None),
        "spot_close_eod": _field(bool, False),
        "entry_signal": _field(_parse_entry_signal, None),
        "orb_window_mins": _field(lambda value: _parse_positive_int(value, default=15), 15),
        "orb_risk_reward": _field(lambda value: _parse_positive_float(value, default=2.0), 2.0),
        "orb_target_mode": _field(_parse_orb_target_mode, None),
        "orb_open_time_et": _field(_identity, None),
        "spot_exit_mode": _field(_parse_spot_exit_mode, None),
        "spot_atr_period": _field(lambda value: _parse_positive_int(value, default=14), 14),
        "spot_pt_atr_mult": _field(lambda value: _parse_non_negative_float_or_default(value, default=1.5), 1.5),
        "spot_sl_atr_mult": _field(lambda value: _parse_non_negative_float_or_default(value, default=1.0), 1.0),
        "spot_exit_time_et": _field(_identity, None),
        "spot_exec_bar_size": _field(_parse_bar_size, None),
        "regime_mode": _field(_parse_regime_mode, None),
        "regime2_mode": _field(_parse_regime_gate_mode, None),
        "regime2_apply_to": _field(_parse_regime2_apply_to, None),
        "regime2_ema_preset": _field(_parse_ema_preset, None),
        "regime2_bar_size": _field(_parse_bar_size, None),
        "regime2_supertrend_atr_period": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "regime2_supertrend_multiplier": _field(lambda value: _parse_positive_float(value, default=3.0), 3.0),
        "regime2_supertrend_source": _field(_parse_supertrend_source, None),
        "supertrend_atr_period": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "supertrend_multiplier": _field(lambda value: _parse_positive_float(value, default=3.0), 3.0),
        "supertrend_source": _field(_parse_supertrend_source, None),
        "tick_gate_mode": _field(_parse_tick_gate_mode, None),
        "tick_gate_symbol": _field(lambda value: str(value or "TICK-NYSE").strip(), "TICK-NYSE"),
        "tick_gate_exchange": _field(lambda value: str(value or "NYSE").strip(), "NYSE"),
        "tick_band_ma_period": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "tick_width_z_lookback": _field(lambda value: _parse_positive_int(value, default=252), 252),
        "tick_width_z_enter": _field(lambda value: _parse_positive_float(value, default=1.0), 1.0),
        "tick_width_z_exit": _field(lambda value: _parse_positive_float(value, default=0.5), 0.5),
        "tick_width_slope_lookback": _field(lambda value: _parse_positive_int(value, default=3), 3),
        "tick_neutral_policy": _field(_parse_tick_neutral_policy, None),
        "tick_direction_policy": _field(_parse_tick_direction_policy, None),
        "spot_entry_fill_mode": _field(lambda value: _parse_spot_fill_mode(value, default="close"), "close"),
        "spot_flip_exit_fill_mode": _field(lambda value: _parse_spot_fill_mode(value, default="close"), "close"),
        "spot_intrabar_exits": _field(bool, False),
        "spot_spread": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_commission_per_share": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_commission_min": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_slippage_per_share": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_mark_to_market": _field(_parse_spot_mark_to_market, None),
        "spot_drawdown_mode": _field(_parse_spot_drawdown_mode, None),
        "spot_sizing_mode": _field(_parse_spot_sizing_mode, None),
        "spot_notional_pct": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_risk_pct": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_short_risk_mult": _field(lambda value: _parse_non_negative_float(value, default=1.0), 1.0),
        "spot_max_notional_pct": _field(lambda value: _parse_non_negative_float(value, default=1.0), 1.0),
        "spot_min_qty": _field(lambda value: _parse_positive_int(value, default=1), 1),
        "spot_max_qty": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
    }


def _synthetic_schema() -> dict[str, _FieldSpec]:
    return {
        "rv_lookback": _field(int, 60),
        "rv_ewma_lambda": _field(float, 0.94),
        "iv_risk_premium": _field(float, 1.2),
        "iv_floor": _field(float, 0.05),
        "term_slope": _field(float, 0.02),
        "skew": _field(float, -0.25),
        "min_spread_pct": _field(float, 0.1),
    }


# endregion


# region Basic Parsing
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


# endregion


# region Filters Parsing
def _parse_filters(raw) -> FiltersConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object")
    def _f(val):
        return None if val is None else float(val)
    def _i(val):
        return None if val is None else int(val)
    volume_period = _i(raw.get("volume_ema_period"))
    if volume_period is not None and volume_period <= 0:
        volume_period = None
    start_et = _i(raw.get("entry_start_hour_et"))
    end_et = _i(raw.get("entry_end_hour_et"))
    if start_et is not None and not (0 <= int(start_et) <= 23):
        start_et = None
    if end_et is not None and not (0 <= int(end_et) <= 23):
        end_et = None

    slope_signed_up = _f(raw.get("ema_slope_signed_min_pct_up"))
    if slope_signed_up is not None and slope_signed_up <= 0:
        slope_signed_up = None
    slope_signed_down = _f(raw.get("ema_slope_signed_min_pct_down"))
    if slope_signed_down is not None and slope_signed_down <= 0:
        slope_signed_down = None

    risk_cutoff_et = _i(raw.get("risk_entry_cutoff_hour_et"))
    if risk_cutoff_et is not None and not (0 <= int(risk_cutoff_et) <= 23):
        risk_cutoff_et = None

    shock_gate_mode = raw.get("shock_gate_mode")
    if shock_gate_mode is None:
        shock_gate_mode = raw.get("shock_mode")
    if isinstance(shock_gate_mode, bool):
        shock_gate_mode = "block" if shock_gate_mode else "off"
    shock_gate_mode = str(shock_gate_mode or "off").strip().lower()
    if shock_gate_mode in ("", "0", "false", "none", "null"):
        shock_gate_mode = "off"
    if shock_gate_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
        shock_gate_mode = "off"

    shock_detector = str(raw.get("shock_detector") or "atr_ratio").strip().lower()
    if shock_detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
        shock_detector = "daily_atr_pct"
    elif shock_detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
        shock_detector = "daily_drawdown"
    elif shock_detector in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
        shock_detector = "tr_ratio"
    elif shock_detector in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
        shock_detector = "atr_ratio"
    else:
        shock_detector = "atr_ratio"

    shock_scale_detector = raw.get("shock_scale_detector")
    if shock_scale_detector is not None:
        shock_scale_detector = str(shock_scale_detector).strip().lower()
    if shock_scale_detector in ("", "0", "false", "none", "null", "off"):
        shock_scale_detector = None
    elif shock_scale_detector is not None:
        if shock_scale_detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
            shock_scale_detector = "daily_atr_pct"
        elif shock_scale_detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
            shock_scale_detector = "daily_drawdown"
        elif shock_scale_detector in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
            shock_scale_detector = "tr_ratio"
        elif shock_scale_detector in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
            shock_scale_detector = "atr_ratio"
        else:
            shock_scale_detector = None

    shock_fast = _i(raw.get("shock_atr_fast_period"))
    if shock_fast is None or shock_fast <= 0:
        shock_fast = 7
    shock_slow = _i(raw.get("shock_atr_slow_period"))
    if shock_slow is None or shock_slow <= 0:
        shock_slow = 50
    shock_on = _f(raw.get("shock_on_ratio"))
    shock_off = _f(raw.get("shock_off_ratio"))
    shock_min_atr = _f(raw.get("shock_min_atr_pct"))
    shock_short_mult = _f(raw.get("shock_short_risk_mult_factor"))
    if shock_short_mult is None or shock_short_mult < 0:
        shock_short_mult = 1.0
    shock_long_mult = _f(raw.get("shock_long_risk_mult_factor"))
    if shock_long_mult is None or shock_long_mult < 0:
        shock_long_mult = 1.0
    shock_long_mult_down = _f(raw.get("shock_long_risk_mult_factor_down"))
    if shock_long_mult_down is None or shock_long_mult_down < 0:
        shock_long_mult_down = 1.0
    shock_sl_mult = _f(raw.get("shock_stop_loss_pct_mult"))
    if shock_sl_mult is None or shock_sl_mult <= 0:
        shock_sl_mult = 1.0
    shock_pt_mult = _f(raw.get("shock_profit_target_pct_mult"))
    if shock_pt_mult is None or shock_pt_mult <= 0:
        shock_pt_mult = 1.0

    daily_atr_period = _i(raw.get("shock_daily_atr_period"))
    if daily_atr_period is None or daily_atr_period <= 0:
        daily_atr_period = 14
    daily_on = _f(raw.get("shock_daily_on_atr_pct"))
    daily_off = _f(raw.get("shock_daily_off_atr_pct"))
    daily_tr_on = _f(raw.get("shock_daily_on_tr_pct"))
    if daily_tr_on is not None and daily_tr_on <= 0:
        daily_tr_on = None
    if daily_on is None:
        daily_on = 13.0
    if daily_off is None:
        daily_off = 11.0
    if float(daily_off) > float(daily_on):
        daily_off = float(daily_on)

    dd_lb = _i(raw.get("shock_drawdown_lookback_days"))
    if dd_lb is None or dd_lb <= 1:
        dd_lb = 20
    dd_on = _f(raw.get("shock_on_drawdown_pct"))
    if dd_on is None:
        dd_on = -20.0
    dd_off = _f(raw.get("shock_off_drawdown_pct"))
    if dd_off is None:
        dd_off = -10.0
    if float(dd_off) < float(dd_on):
        dd_off = float(dd_on)
    shock_dir_lb = _i(raw.get("shock_direction_lookback"))
    if shock_dir_lb is None or shock_dir_lb <= 0:
        shock_dir_lb = 2
    shock_dir_source = str(raw.get("shock_direction_source") or "regime").strip().lower()
    if shock_dir_source not in ("regime", "signal"):
        shock_dir_source = "regime"
    shock_regime_override_dir = bool(raw.get("shock_regime_override_dir"))

    shock_regime_st_mult = _f(raw.get("shock_regime_supertrend_multiplier"))
    if shock_regime_st_mult is not None and shock_regime_st_mult <= 0:
        shock_regime_st_mult = None
    shock_cool_st_mult = _f(raw.get("shock_cooling_regime_supertrend_multiplier"))
    if shock_cool_st_mult is not None and shock_cool_st_mult <= 0:
        shock_cool_st_mult = None
    shock_daily_cool_atr = _f(raw.get("shock_daily_cooling_atr_pct"))
    if shock_daily_cool_atr is not None and shock_daily_cool_atr <= 0:
        shock_daily_cool_atr = None

    shock_scale_target = _f(raw.get("shock_risk_scale_target_atr_pct"))
    if shock_scale_target is not None and shock_scale_target <= 0:
        shock_scale_target = None
    shock_scale_min = _f(raw.get("shock_risk_scale_min_mult"))
    if shock_scale_min is None:
        shock_scale_min = 0.2
    shock_scale_min = float(max(0.0, min(1.0, shock_scale_min)))

    shock_scale_apply_to = raw.get("shock_risk_scale_apply_to")
    if shock_scale_apply_to is not None:
        shock_scale_apply_to = str(shock_scale_apply_to).strip().lower()
    if shock_scale_apply_to in ("", "0", "false", "none", "null"):
        shock_scale_apply_to = None
    if shock_scale_apply_to in ("cap", "notional_cap", "max_notional", "cap_only"):
        shock_scale_apply_to = "cap"
    elif shock_scale_apply_to in ("both", "cap_and_risk", "risk_and_cap", "all"):
        shock_scale_apply_to = "both"
    else:
        shock_scale_apply_to = "risk"
    riskoff_tr5_med = _f(raw.get("riskoff_tr5_med_pct"))
    if riskoff_tr5_med is not None and riskoff_tr5_med <= 0:
        riskoff_tr5_med = None
    riskoff_tr5_lb = _i(raw.get("riskoff_tr5_lookback_days"))
    if riskoff_tr5_lb is None or riskoff_tr5_lb <= 0:
        riskoff_tr5_lb = 5

    riskoff_mode = raw.get("riskoff_mode")
    if isinstance(riskoff_mode, str):
        riskoff_mode = riskoff_mode.strip().lower()
    else:
        riskoff_mode = "hygiene"
    if riskoff_mode not in ("hygiene", "directional"):
        riskoff_mode = "hygiene"
    riskoff_short_factor = _f(raw.get("riskoff_short_risk_mult_factor"))
    if riskoff_short_factor is None:
        riskoff_short_factor = 1.0
    if riskoff_short_factor < 0:
        riskoff_short_factor = 1.0
    riskoff_long_factor = _f(raw.get("riskoff_long_risk_mult_factor"))
    if riskoff_long_factor is None:
        riskoff_long_factor = 1.0
    if riskoff_long_factor < 0:
        riskoff_long_factor = 1.0

    riskpanic_tr5_med = _f(raw.get("riskpanic_tr5_med_pct"))
    if riskpanic_tr5_med is not None and riskpanic_tr5_med <= 0:
        riskpanic_tr5_med = None
    riskpanic_neg_gap = _f(raw.get("riskpanic_neg_gap_ratio_min"))
    if riskpanic_neg_gap is not None:
        riskpanic_neg_gap = float(max(0.0, min(1.0, riskpanic_neg_gap)))
    riskpanic_neg_gap_abs = _f(raw.get("riskpanic_neg_gap_abs_pct_min"))
    if riskpanic_neg_gap_abs is not None:
        riskpanic_neg_gap_abs = float(max(0.0, min(1.0, riskpanic_neg_gap_abs)))
        if riskpanic_neg_gap_abs <= 0:
            riskpanic_neg_gap_abs = None
    riskpanic_lb = _i(raw.get("riskpanic_lookback_days"))
    if riskpanic_lb is None or riskpanic_lb <= 0:
        riskpanic_lb = 5
    riskpanic_tr5_delta_min = _f(raw.get("riskpanic_tr5_med_delta_min_pct"))
    riskpanic_tr5_delta_lb = _i(raw.get("riskpanic_tr5_med_delta_lookback_days"))
    if riskpanic_tr5_delta_lb is None or riskpanic_tr5_delta_lb <= 0:
        riskpanic_tr5_delta_lb = 1
    riskpanic_long_factor = _f(raw.get("riskpanic_long_risk_mult_factor"))
    if riskpanic_long_factor is None:
        riskpanic_long_factor = 1.0
    if riskpanic_long_factor < 0:
        riskpanic_long_factor = 1.0
    riskpanic_long_scale_mode = raw.get("riskpanic_long_scale_mode")
    if isinstance(riskpanic_long_scale_mode, str):
        riskpanic_long_scale_mode = riskpanic_long_scale_mode.strip().lower()
    else:
        riskpanic_long_scale_mode = "off"
    if riskpanic_long_scale_mode in ("linear", "lin", "delta", "linear_delta", "linear_tr_delta"):
        riskpanic_long_scale_mode = "linear"
    elif riskpanic_long_scale_mode in ("", "0", "false", "none", "null", "off"):
        riskpanic_long_scale_mode = "off"
    else:
        riskpanic_long_scale_mode = "off"
    riskpanic_long_scale_delta_max = _f(raw.get("riskpanic_long_scale_tr_delta_max_pct"))
    if riskpanic_long_scale_delta_max is not None and riskpanic_long_scale_delta_max <= 0:
        riskpanic_long_scale_delta_max = None
    riskpanic_short_factor = _f(raw.get("riskpanic_short_risk_mult_factor"))
    if riskpanic_short_factor is None:
        riskpanic_short_factor = 1.0
    if riskpanic_short_factor < 0:
        riskpanic_short_factor = 1.0

    riskpop_tr5_med = _f(raw.get("riskpop_tr5_med_pct"))
    if riskpop_tr5_med is not None and riskpop_tr5_med <= 0:
        riskpop_tr5_med = None
    riskpop_pos_gap = _f(raw.get("riskpop_pos_gap_ratio_min"))
    if riskpop_pos_gap is not None:
        riskpop_pos_gap = float(max(0.0, min(1.0, riskpop_pos_gap)))
    riskpop_pos_gap_abs = _f(raw.get("riskpop_pos_gap_abs_pct_min"))
    if riskpop_pos_gap_abs is not None:
        riskpop_pos_gap_abs = float(max(0.0, min(1.0, riskpop_pos_gap_abs)))
        if riskpop_pos_gap_abs <= 0:
            riskpop_pos_gap_abs = None
    riskpop_lb = _i(raw.get("riskpop_lookback_days"))
    if riskpop_lb is None or riskpop_lb <= 0:
        riskpop_lb = 5
    riskpop_tr5_delta_min = _f(raw.get("riskpop_tr5_med_delta_min_pct"))
    riskpop_tr5_delta_lb = _i(raw.get("riskpop_tr5_med_delta_lookback_days"))
    if riskpop_tr5_delta_lb is None or riskpop_tr5_delta_lb <= 0:
        riskpop_tr5_delta_lb = 1
    riskpop_long_factor = _f(raw.get("riskpop_long_risk_mult_factor"))
    if riskpop_long_factor is None:
        riskpop_long_factor = 1.0
    if riskpop_long_factor < 0:
        riskpop_long_factor = 1.0
    riskpop_short_factor = _f(raw.get("riskpop_short_risk_mult_factor"))
    if riskpop_short_factor is None:
        riskpop_short_factor = 1.0
    if riskpop_short_factor < 0:
        riskpop_short_factor = 1.0
    return FiltersConfig(
        rv_min=_f(raw.get("rv_min")),
        rv_max=_f(raw.get("rv_max")),
        ema_spread_min_pct=_f(raw.get("ema_spread_min_pct")),
        ema_spread_min_pct_down=_f(raw.get("ema_spread_min_pct_down")),
        ema_slope_min_pct=_f(raw.get("ema_slope_min_pct")),
        ema_slope_signed_min_pct_up=slope_signed_up,
        ema_slope_signed_min_pct_down=slope_signed_down,
        entry_start_hour=_i(raw.get("entry_start_hour")),
        entry_end_hour=_i(raw.get("entry_end_hour")),
        entry_start_hour_et=start_et,
        entry_end_hour_et=end_et,
        skip_first_bars=int(raw.get("skip_first_bars", 0) or 0),
        cooldown_bars=int(raw.get("cooldown_bars", 0) or 0),
        volume_ema_period=volume_period,
        volume_ratio_min=_f(raw.get("volume_ratio_min")),
        shock_gate_mode=shock_gate_mode,
        shock_detector=shock_detector,
        shock_scale_detector=str(shock_scale_detector) if shock_scale_detector is not None else None,
        shock_atr_fast_period=int(shock_fast),
        shock_atr_slow_period=int(shock_slow),
        shock_on_ratio=float(shock_on) if shock_on is not None else 1.55,
        shock_off_ratio=float(shock_off) if shock_off is not None else 1.30,
        shock_min_atr_pct=float(shock_min_atr) if shock_min_atr is not None else 7.0,
        shock_daily_atr_period=int(daily_atr_period),
        shock_daily_on_atr_pct=float(daily_on),
        shock_daily_off_atr_pct=float(daily_off),
        shock_daily_on_tr_pct=float(daily_tr_on) if daily_tr_on is not None else None,
        shock_drawdown_lookback_days=int(dd_lb),
        shock_on_drawdown_pct=float(dd_on),
        shock_off_drawdown_pct=float(dd_off),
        shock_short_risk_mult_factor=float(shock_short_mult),
        shock_long_risk_mult_factor=float(shock_long_mult),
        shock_long_risk_mult_factor_down=float(shock_long_mult_down),
        shock_stop_loss_pct_mult=float(shock_sl_mult),
        shock_profit_target_pct_mult=float(shock_pt_mult),
        shock_direction_lookback=int(shock_dir_lb),
        shock_direction_source=shock_dir_source,
        shock_regime_override_dir=shock_regime_override_dir,
        shock_regime_supertrend_multiplier=shock_regime_st_mult,
        shock_cooling_regime_supertrend_multiplier=shock_cool_st_mult,
        shock_daily_cooling_atr_pct=shock_daily_cool_atr,
        shock_risk_scale_target_atr_pct=shock_scale_target,
        shock_risk_scale_min_mult=shock_scale_min,
        shock_risk_scale_apply_to=str(shock_scale_apply_to),
        risk_entry_cutoff_hour_et=risk_cutoff_et,
        riskoff_tr5_med_pct=riskoff_tr5_med,
        riskoff_tr5_lookback_days=int(riskoff_tr5_lb),
        riskoff_mode=str(riskoff_mode),
        riskoff_short_risk_mult_factor=float(riskoff_short_factor),
        riskoff_long_risk_mult_factor=float(riskoff_long_factor),
        riskpanic_tr5_med_pct=riskpanic_tr5_med,
        riskpanic_neg_gap_ratio_min=riskpanic_neg_gap,
        riskpanic_neg_gap_abs_pct_min=riskpanic_neg_gap_abs,
        riskpanic_lookback_days=int(riskpanic_lb),
        riskpanic_tr5_med_delta_min_pct=riskpanic_tr5_delta_min,
        riskpanic_tr5_med_delta_lookback_days=int(riskpanic_tr5_delta_lb),
        riskpanic_long_risk_mult_factor=float(riskpanic_long_factor),
        riskpanic_long_scale_mode=str(riskpanic_long_scale_mode),
        riskpanic_long_scale_tr_delta_max_pct=riskpanic_long_scale_delta_max,
        riskpanic_short_risk_mult_factor=float(riskpanic_short_factor),
        riskpop_tr5_med_pct=riskpop_tr5_med,
        riskpop_pos_gap_ratio_min=riskpop_pos_gap,
        riskpop_pos_gap_abs_pct_min=riskpop_pos_gap_abs,
        riskpop_lookback_days=int(riskpop_lb),
        riskpop_tr5_med_delta_min_pct=riskpop_tr5_delta_min,
        riskpop_tr5_med_delta_lookback_days=int(riskpop_tr5_delta_lb),
        riskpop_long_risk_mult_factor=float(riskpop_long_factor),
        riskpop_short_risk_mult_factor=float(riskpop_short_factor),
    )


# endregion


# region Misc Normalizers
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
        if cleaned in ("both", "all", "default", ""):
            return "both"
        if cleaned in ("long", "longs", "up", "buy"):
            return "longs"
        if cleaned in ("short", "shorts", "down", "sell"):
            return "shorts"
    return "both"


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
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("close", "bar_close", "at_close"):
            return "close"
        if cleaned in ("next_open", "nextopen", "open", "next_bar_open", "nextbaropen"):
            return "next_open"
    return default


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


# endregion
