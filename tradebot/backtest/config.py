"""Config loading for synthetic backtests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_WEEKDAYS = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    bar_size: str
    use_rth: bool
    starting_cash: float
    risk_free_rate: float
    cache_dir: Path
    calibration_dir: Path
    output_dir: Path
    calibrate: bool
    offline: bool


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    symbol: str
    exchange: str | None
    right: str
    entry_days: tuple[int, ...]
    max_entries_per_day: int
    max_open_trades: int
    dte: int
    otm_pct: float
    width_pct: float
    profit_target: float
    stop_loss: float
    exit_dte: int
    quantity: int
    stop_loss_basis: str
    min_credit: float | None
    ema_preset: str | None
    ema_entry_mode: str
    ema_directional: bool
    exit_on_signal_flip: bool
    flip_exit_mode: str
    flip_exit_min_hold_bars: int
    flip_exit_only_if_profit: bool
    direction_source: str
    directional_legs: dict[str, tuple["LegConfig", ...]] | None
    legs: tuple["LegConfig", ...] | None
    filters: "FiltersConfig" | None


@dataclass(frozen=True)
class SyntheticConfig:
    rv_lookback: int
    rv_ewma_lambda: float
    iv_risk_premium: float
    iv_floor: float
    term_slope: float
    skew: float
    min_spread_pct: float


@dataclass(frozen=True)
class LegConfig:
    action: str
    right: str
    moneyness_pct: float
    qty: int


@dataclass(frozen=True)
class FiltersConfig:
    rv_min: float | None
    rv_max: float | None
    ema_spread_min_pct: float | None
    ema_slope_min_pct: float | None
    entry_start_hour: int | None
    entry_end_hour: int | None
    skip_first_bars: int
    cooldown_bars: int


@dataclass(frozen=True)
class ConfigBundle:
    backtest: BacktestConfig
    strategy: StrategyConfig
    synthetic: SyntheticConfig


def load_config(path: str | Path) -> ConfigBundle:
    raw = json.loads(Path(path).read_text())
    backtest_raw = raw.get("backtest", {})
    strategy_raw = raw.get("strategy", {})
    synthetic_raw = raw.get("synthetic", {})

    # TODO: replace with historical rates source for higher fidelity.
    backtest = BacktestConfig(
        start=_parse_date(backtest_raw.get("start")),
        end=_parse_date(backtest_raw.get("end")),
        bar_size=backtest_raw.get("bar_size", "1 hour"),
        use_rth=bool(backtest_raw.get("use_rth", False)),
        starting_cash=float(backtest_raw.get("starting_cash", 100_000.0)),
        risk_free_rate=float(backtest_raw.get("risk_free_rate", 0.02)),
        cache_dir=Path(backtest_raw.get("cache_dir", "db")),
        calibration_dir=Path(backtest_raw.get("calibration_dir", "db/calibration")),
        output_dir=Path(backtest_raw.get("output_dir", "backtests/out")),
        calibrate=bool(backtest_raw.get("calibrate", False)),
        offline=bool(backtest_raw.get("offline", False)),
    )

    entry_days = _parse_weekdays(strategy_raw.get("entry_days", []))
    strategy = StrategyConfig(
        name=strategy_raw.get("name", "credit_spread"),
        symbol=strategy_raw.get("symbol", "MNQ"),
        exchange=strategy_raw.get("exchange"),
        right=str(strategy_raw.get("right", "PUT")).upper(),
        entry_days=entry_days,
        max_entries_per_day=_parse_non_negative_int(
            strategy_raw.get("max_entries_per_day"), default=1
        ),
        max_open_trades=_parse_non_negative_int(strategy_raw.get("max_open_trades"), default=1),
        dte=int(strategy_raw.get("dte", 0)),
        otm_pct=float(strategy_raw.get("otm_pct", 2.5)),
        width_pct=float(strategy_raw.get("width_pct", 1.0)),
        profit_target=float(strategy_raw.get("profit_target", 0.5)),
        stop_loss=float(strategy_raw.get("stop_loss", 0.35)),
        exit_dte=int(strategy_raw.get("exit_dte", 0)),
        quantity=int(strategy_raw.get("quantity", 1)),
        stop_loss_basis=strategy_raw.get("stop_loss_basis", "max_loss"),
        min_credit=(
            float(strategy_raw["min_credit"]) if "min_credit" in strategy_raw else None
        ),
        ema_preset=_parse_ema_preset(strategy_raw.get("ema_preset")),
        ema_entry_mode=_parse_ema_entry_mode(strategy_raw.get("ema_entry_mode")),
        ema_directional=bool(strategy_raw.get("ema_directional", False)),
        exit_on_signal_flip=bool(strategy_raw.get("exit_on_signal_flip", False)),
        flip_exit_mode=_parse_flip_exit_mode(strategy_raw.get("flip_exit_mode")),
        flip_exit_min_hold_bars=_parse_non_negative_int(
            strategy_raw.get("flip_exit_min_hold_bars"), default=0
        ),
        flip_exit_only_if_profit=bool(strategy_raw.get("flip_exit_only_if_profit", False)),
        direction_source=_parse_direction_source(strategy_raw.get("direction_source")),
        directional_legs=_parse_directional_legs(strategy_raw.get("directional_legs")),
        legs=_parse_legs(strategy_raw.get("legs")),
        filters=_parse_filters(strategy_raw.get("filters")),
    )

    synthetic = SyntheticConfig(
        rv_lookback=int(synthetic_raw.get("rv_lookback", 60)),
        rv_ewma_lambda=float(synthetic_raw.get("rv_ewma_lambda", 0.94)),
        iv_risk_premium=float(synthetic_raw.get("iv_risk_premium", 1.2)),
        iv_floor=float(synthetic_raw.get("iv_floor", 0.05)),
        term_slope=float(synthetic_raw.get("term_slope", 0.02)),
        skew=float(synthetic_raw.get("skew", -0.25)),
        min_spread_pct=float(synthetic_raw.get("min_spread_pct", 0.1)),
    )

    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def _parse_date(value: str | None) -> date:
    if not value:
        raise ValueError("Config requires start/end dates in YYYY-MM-DD format")
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


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


def _parse_filters(raw) -> FiltersConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object")
    def _f(val):
        return None if val is None else float(val)
    def _i(val):
        return None if val is None else int(val)
    return FiltersConfig(
        rv_min=_f(raw.get("rv_min")),
        rv_max=_f(raw.get("rv_max")),
        ema_spread_min_pct=_f(raw.get("ema_spread_min_pct")),
        ema_slope_min_pct=_f(raw.get("ema_slope_min_pct")),
        entry_start_hour=_i(raw.get("entry_start_hour")),
        entry_end_hour=_i(raw.get("entry_end_hour")),
        skip_first_bars=int(raw.get("skip_first_bars", 0) or 0),
        cooldown_bars=int(raw.get("cooldown_bars", 0) or 0),
    )
