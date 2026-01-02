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


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    symbol: str
    exchange: str | None
    right: str
    entry_days: tuple[int, ...]
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
    )

    entry_days = _parse_weekdays(strategy_raw.get("entry_days", []))
    strategy = StrategyConfig(
        name=strategy_raw.get("name", "credit_spread"),
        symbol=strategy_raw.get("symbol", "MNQ"),
        exchange=strategy_raw.get("exchange"),
        right=str(strategy_raw.get("right", "PUT")).upper(),
        entry_days=entry_days,
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
