from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..engine import (
    normalize_shock_gate_mode,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
)


@dataclass(frozen=True)
class SpotBarRequirement:
    kind: str
    symbol: str
    exchange: str | None
    bar_size: str
    use_rth: bool
    warmup_days: int = 0


@dataclass(frozen=True)
class SpotContextBars:
    regime_bars: object | None = None
    regime2_bars: object | None = None
    tick_bars: object | None = None
    exec_bars: object | None = None


LoadRequirementFn = Callable[[SpotBarRequirement, datetime, datetime], object]
MissingRequirementFn = Callable[[SpotBarRequirement, datetime, datetime], None]


def _get(source: Mapping[str, object] | object | None, key: str, default: object = None) -> object:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _parse_int(value: object, *, default: int, min_value: int | None = None) -> int:
    try:
        out = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        out = int(default)
    if min_value is not None and out < int(min_value):
        return int(min_value)
    return int(out)


def _normalize_exchange(value: object, *, default: str | None = None) -> str | None:
    raw = str(value if value is not None else (default or "")).strip()
    if not raw:
        return None if default is None else str(default)
    return raw


def spot_bar_requirements_from_strategy(
    *,
    strategy: Mapping[str, object] | object,
    default_symbol: str,
    default_exchange: str | None,
    default_signal_bar_size: str,
    default_signal_use_rth: bool,
    include_signal: bool = True,
) -> tuple[SpotBarRequirement, ...]:
    symbol = str(_get(strategy, "symbol", default_symbol) or default_symbol).strip().upper()
    exchange = _normalize_exchange(_get(strategy, "exchange", default_exchange), default=default_exchange)

    signal_bar_size = str(_get(strategy, "signal_bar_size", default_signal_bar_size) or default_signal_bar_size).strip()
    if not signal_bar_size:
        signal_bar_size = str(default_signal_bar_size)
    signal_use_rth_raw = _get(strategy, "signal_use_rth", None)
    signal_use_rth = bool(default_signal_use_rth if signal_use_rth_raw is None else signal_use_rth_raw)

    out: list[SpotBarRequirement] = []
    if include_signal:
        out.append(
            SpotBarRequirement(
                kind="signal",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(signal_bar_size),
                use_rth=bool(signal_use_rth),
                warmup_days=0,
            )
        )

    filters = _get(strategy, "filters", None)

    _regime_mode, _regime_preset, regime_bar, use_mtf_regime = resolve_spot_regime_spec(
        bar_size=str(signal_bar_size),
        regime_mode_raw=_get(strategy, "regime_mode", "ema"),
        regime_ema_preset_raw=_get(strategy, "regime_ema_preset", None),
        regime_bar_size_raw=_get(strategy, "regime_bar_size", None),
    )
    if bool(use_mtf_regime):
        regime_warm_days = 0
        if normalize_shock_gate_mode(filters) != "off":
            slow_period = _parse_int(_get(filters, "shock_atr_slow_period", 50), default=50, min_value=1)
            regime_warm_days = max(30, int(slow_period))
        out.append(
            SpotBarRequirement(
                kind="regime",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(regime_bar),
                use_rth=bool(signal_use_rth),
                warmup_days=int(regime_warm_days),
            )
        )

    _r2_mode, _r2_preset, regime2_bar, use_mtf_regime2 = resolve_spot_regime2_spec(
        bar_size=str(signal_bar_size),
        regime2_mode_raw=_get(strategy, "regime2_mode", "off"),
        regime2_ema_preset_raw=_get(strategy, "regime2_ema_preset", None),
        regime2_bar_size_raw=_get(strategy, "regime2_bar_size", None),
    )
    if bool(use_mtf_regime2):
        out.append(
            SpotBarRequirement(
                kind="regime2",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(regime2_bar),
                use_rth=bool(signal_use_rth),
                warmup_days=0,
            )
        )

    tick_mode = str(_get(strategy, "tick_gate_mode", "off") or "off").strip().lower()
    if tick_mode not in ("off", "raschke"):
        tick_mode = "off"
    if tick_mode != "off":
        z_lookback = _parse_int(_get(strategy, "tick_width_z_lookback", 252), default=252, min_value=1)
        ma_period = _parse_int(_get(strategy, "tick_band_ma_period", 10), default=10, min_value=1)
        slope_lb = _parse_int(_get(strategy, "tick_width_slope_lookback", 3), default=3, min_value=1)
        tick_warm_days = max(60, int(z_lookback) + int(ma_period) + int(slope_lb) + 5)
        tick_symbol = str(_get(strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip().upper()
        tick_exchange = str(_get(strategy, "tick_gate_exchange", "NYSE") or "NYSE").strip().upper()
        out.append(
            SpotBarRequirement(
                kind="tick",
                symbol=tick_symbol,
                exchange=tick_exchange,
                bar_size="1 day",
                use_rth=True,
                warmup_days=int(tick_warm_days),
            )
        )

    exec_bar_size = str(_get(strategy, "spot_exec_bar_size", "") or "").strip()
    if exec_bar_size and str(exec_bar_size) != str(signal_bar_size):
        out.append(
            SpotBarRequirement(
                kind="exec",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(exec_bar_size),
                use_rth=bool(signal_use_rth),
                warmup_days=0,
            )
        )

    deduped: list[SpotBarRequirement] = []
    seen: set[tuple[object, ...]] = set()
    for req in out:
        key = (
            str(req.kind),
            str(req.symbol),
            str(req.exchange),
            str(req.bar_size),
            bool(req.use_rth),
            int(req.warmup_days),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(req)
    return tuple(deduped)


def load_spot_context_bars(
    *,
    strategy: Mapping[str, object] | object,
    default_symbol: str,
    default_exchange: str | None,
    default_signal_bar_size: str,
    default_signal_use_rth: bool,
    start_dt: datetime,
    end_dt: datetime,
    load_requirement: LoadRequirementFn,
    on_missing: MissingRequirementFn | None = None,
) -> SpotContextBars:
    reqs = spot_bar_requirements_from_strategy(
        strategy=strategy,
        default_symbol=default_symbol,
        default_exchange=default_exchange,
        default_signal_bar_size=default_signal_bar_size,
        default_signal_use_rth=default_signal_use_rth,
        include_signal=False,
    )

    by_kind: dict[str, object] = {}
    for req in reqs:
        req_start = start_dt - timedelta(days=max(0, int(req.warmup_days)))
        loaded = load_requirement(req, req_start, end_dt)
        has_rows = bool(loaded)
        if not has_rows:
            if on_missing is not None:
                on_missing(req, req_start, end_dt)
            raise ValueError(
                f"Missing {req.kind} bars: {req.symbol} {req.bar_size} "
                f"{req_start.isoformat()}..{end_dt.isoformat()} rth={int(req.use_rth)}"
            )
        by_kind[str(req.kind)] = loaded

    return SpotContextBars(
        regime_bars=by_kind.get("regime"),
        regime2_bars=by_kind.get("regime2"),
        tick_bars=by_kind.get("tick"),
        exec_bars=by_kind.get("exec"),
    )
