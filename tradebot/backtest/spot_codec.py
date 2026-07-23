from __future__ import annotations

import math
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path

from ..contract_identity import future_exchange_for_symbol, is_future_symbol
from .config import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    SpotLegConfig,
    SpotStrategyConfig,
    SyntheticConfig,
)
from .config_filters import _parse_filters
from .config_values import _parse_spot_exchange, _parse_spot_sec_type
from .data import ContractMeta
from ..spot.codec import bool_from_payload

from ..option_package import normalize_option_leg, normalize_option_legs

_WDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_WDAY_TO_IDX = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def entry_days_labels(days: tuple[int, ...]) -> list[str]:
    out: list[str] = []
    for d in days:
        try:
            idx = int(d)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(_WDAYS):
            out.append(_WDAYS[idx])
    return out


def weekdays_from_payload(value) -> tuple[int, ...]:
    if not value:
        return (0, 1, 2, 3, 4)
    out: list[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
            continue
        key = str(item).strip().upper()[:3]
        if key in _WDAY_TO_IDX:
            out.append(_WDAY_TO_IDX[key])
    return tuple(out) if out else (0, 1, 2, 3, 4)


def spot_leg_from_payload(raw) -> SpotLegConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"directional_spot leg must be an object, got: {raw!r}")
    action = str(raw.get("action") or "").strip().upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(f"directional_spot.action must be BUY/SELL, got: {action!r}")
    qty = int(raw.get("qty") or 1)
    if qty <= 0:
        raise ValueError(f"directional_spot.qty must be positive, got: {qty!r}")
    return SpotLegConfig(action=action, qty=qty)


def leg_from_payload(raw) -> LegConfig:
    return normalize_option_leg(raw, path="leg")


def filters_from_payload(raw) -> FiltersConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"filters must be an object, got: {raw!r}")
    return _parse_filters(raw)


def strategy_from_payload(strategy: dict, *, filters: FiltersConfig | None) -> SpotStrategyConfig:
    if not isinstance(strategy, dict):
        raise ValueError(f"strategy must be an object, got: {strategy!r}")

    raw = dict(strategy)

    # Match live UI hydration semantics: if a spot-STK strategy is explicitly
    # full24 (signal_use_rth=false) but does not set spot_next_open_session,
    # force tradable_24x5 so deferred fill modes (next_open/next_tradable_bar)
    # schedule against the intended session window.
    instrument = str(raw.get("instrument", "spot") or "spot").strip().lower()
    spot_sec_type = _parse_spot_sec_type(raw.get("spot_sec_type"))
    spot_exchange = _parse_spot_exchange(raw.get("spot_exchange"))
    sec_type = spot_sec_type or "STK"
    mode_raw = str(raw.get("spot_next_open_session") or "").strip()
    use_rth = bool_from_payload(raw.get("signal_use_rth"))
    if instrument == "spot" and sec_type == "STK" and not use_rth and not mode_raw:
        raw["spot_next_open_session"] = "tradable_24x5"

    raw.pop("signal_bar_size", None)
    raw.pop("signal_use_rth", None)
    raw["spot_sec_type"] = spot_sec_type
    raw["spot_exchange"] = spot_exchange

    raw["entry_days"] = weekdays_from_payload(raw.get("entry_days") or [])
    raw.setdefault("flip_exit_gate_mode", "off")
    raw["filters"] = filters

    dspot = raw.get("directional_spot")
    if dspot is not None:
        if not isinstance(dspot, dict):
            raise ValueError(f"directional_spot must be an object, got: {dspot!r}")
        parsed_spot: dict[str, SpotLegConfig] = {}
        for key, value in dspot.items():
            k = str(key).strip()
            if not k:
                continue
            parsed_spot[k] = spot_leg_from_payload(value)
        raw["directional_spot"] = parsed_spot or None

    dlegs = raw.get("directional_legs")
    if dlegs is not None:
        if not isinstance(dlegs, dict):
            raise ValueError(f"directional_legs must be an object, got: {dlegs!r}")
        parsed_legs: dict[str, tuple[LegConfig, ...]] = {}
        for key, legs in dlegs.items():
            k = str(key).strip()
            if not k or not legs:
                continue
            parsed_legs[k] = normalize_option_legs(
                legs,
                path=f"directional_legs.{k}",
            )
        raw["directional_legs"] = parsed_legs or None

    legs = raw.get("legs")
    if legs is not None:
        raw["legs"] = normalize_option_legs(legs, path="legs")

    allowed_keys = {field.name for field in fields(SpotStrategyConfig)}
    raw = {key: value for key, value in raw.items() if key in allowed_keys}
    return SpotStrategyConfig(**raw)


def make_bundle(
    *,
    strategy: SpotStrategyConfig,
    start: date,
    end: date,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
) -> ConfigBundle:
    backtest = BacktestConfig(
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        starting_cash=100_000.0,
        risk_free_rate=0.02,
        cache_dir=Path(cache_dir),
        calibration_dir=Path(cache_dir) / "calibration",
        output_dir=Path("backtests/out"),
        calibrate=False,
        offline=bool(offline),
    )
    synthetic = SyntheticConfig(
        rv_lookback=60,
        rv_ewma_lambda=0.94,
        iv_risk_premium=1.2,
        iv_floor=0.05,
        term_slope=0.02,
        skew=-0.25,
        min_spread_pct=0.1,
    )
    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def metrics_from_summary(summary) -> dict:
    pnl = float(getattr(summary, "total_pnl", 0.0) or 0.0)
    dd = float(getattr(summary, "max_drawdown", 0.0) or 0.0)
    trades = int(getattr(summary, "trades", 0) or 0)
    win_rate = float(getattr(summary, "win_rate", 0.0) or 0.0)
    roi = float(getattr(summary, "roi", 0.0) or 0.0)
    dd_pct = float(getattr(summary, "max_drawdown_pct", 0.0) or 0.0)
    pnl_dd = pnl / dd if dd > 0 else (math.inf if pnl > 0 else -math.inf if pnl < 0 else 0.0)
    roi_dd = (
        roi / dd_pct
        if dd_pct > 0
        else (math.inf if roi > 0 else -math.inf if roi < 0 else 0.0)
    )
    return {
        "trades": trades,
        "win_rate": win_rate,
        "pnl": pnl,
        "dd": dd,
        "pnl_over_dd": pnl_dd,
        "roi": roi,
        "dd_pct": dd_pct,
        "roi_over_dd_pct": roi_dd,
    }


def filters_payload(filters: FiltersConfig | None) -> dict | None:
    if filters is None:
        return None
    raw = asdict(filters)
    defaults_cfg = _parse_filters({})
    defaults = asdict(defaults_cfg) if defaults_cfg is not None else {}
    out: dict[str, object] = {}
    for key, value in raw.items():
        if key in defaults and value == defaults.get(key):
            continue
        out[key] = value
    return out or None


def spot_strategy_payload(cfg: ConfigBundle, *, meta: ContractMeta) -> dict:
    strategy = asdict(cfg.strategy)
    strategy["entry_days"] = entry_days_labels(cfg.strategy.entry_days)
    strategy["signal_bar_size"] = str(cfg.backtest.bar_size)
    strategy["signal_use_rth"] = bool(cfg.backtest.use_rth)
    strategy["filters"] = filters_payload(cfg.strategy.filters)

    sym = str(cfg.strategy.symbol or "").strip().upper()
    sec_type = _parse_spot_sec_type(cfg.strategy.spot_sec_type)
    exchange = _parse_spot_exchange(cfg.strategy.spot_exchange)
    if sec_type is None:
        sec_type = "FUT" if is_future_symbol(sym) else "STK"
    if exchange is None:
        exchange = (
            str(future_exchange_for_symbol(sym) or meta.exchange or "CME").strip().upper()
            if sec_type == "FUT"
            else "SMART"
        )
    strategy["spot_sec_type"] = sec_type
    strategy["spot_exchange"] = exchange
    return strategy
