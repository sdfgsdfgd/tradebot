from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..engine import (
    normalize_shock_gate_mode,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
)
from ..signals import ema_periods, parse_bar_size


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
    regime2_bear_hard_bars: object | None = None
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


def _supertrend_warmup_days(period: object, *, default: int = 10) -> int:
    atr_period = _parse_int(period, default=default, min_value=1)
    return max(60, int(atr_period) * 6)


def _ema_warmup_days(preset_raw: object) -> int:
    preset = str(preset_raw or "").strip()
    if not preset:
        return 0
    periods = ema_periods(preset)
    if periods is None:
        return 0
    return max(0, max(int(periods[0]), int(periods[1])))


def _bars_to_warmup_days(*, bar_size: str, use_rth: bool, bars: int) -> int:
    needed_bars = max(0, int(bars))
    if needed_bars <= 0:
        return 0
    parsed = parse_bar_size(str(bar_size))
    if parsed is None:
        return max(1, int(needed_bars))
    duration_minutes = max(1.0, float(parsed.duration.total_seconds()) / 60.0)
    trading_minutes_per_day = 390.0 if bool(use_rth) else (24.0 * 60.0)
    bars_per_day = max(1.0, trading_minutes_per_day / duration_minutes)
    return max(1, int(math.ceil(float(needed_bars) / float(bars_per_day))) + 1)


def spot_signal_warmup_days_from_strategy(
    *,
    strategy: Mapping[str, object] | object,
    default_signal_bar_size: str,
    default_signal_use_rth: bool,
) -> int:
    signal_bar_size = str(_get(strategy, "signal_bar_size", default_signal_bar_size) or default_signal_bar_size).strip()
    if not signal_bar_size:
        signal_bar_size = str(default_signal_bar_size)
    signal_use_rth_raw = _get(strategy, "signal_use_rth", None)
    signal_use_rth = bool(default_signal_use_rth if signal_use_rth_raw is None else signal_use_rth_raw)
    filters = _get(strategy, "filters", None)

    bars_needed = 0

    entry_signal = str(_get(strategy, "entry_signal", "ema") or "ema").strip().lower()
    if entry_signal == "ema":
        bars_needed = max(bars_needed, _ema_warmup_days(_get(strategy, "ema_preset", None)))

    if _get(strategy, "spot_exit_mode", "pct") == "atr":
        bars_needed = max(bars_needed, _parse_int(_get(strategy, "spot_atr_period", 14), default=14, min_value=1))

    if _get(filters, "volume_ratio_min", None) is not None:
        bars_needed = max(bars_needed, _parse_int(_get(filters, "volume_ema_period", 20), default=20, min_value=1))

    if normalize_shock_gate_mode(filters) != "off":
        bars_needed = max(bars_needed, _parse_int(_get(filters, "shock_atr_slow_period", 50), default=50, min_value=1))

    regime_mode, regime_preset, _regime_bar, use_mtf_regime = resolve_spot_regime_spec(
        bar_size=str(signal_bar_size),
        regime_mode_raw=_get(strategy, "regime_mode", "ema"),
        regime_ema_preset_raw=_get(strategy, "regime_ema_preset", None),
        regime_bar_size_raw=_get(strategy, "regime_bar_size", None),
    )
    if not bool(use_mtf_regime):
        if str(regime_mode) == "supertrend":
            bars_needed = max(
                bars_needed,
                _supertrend_warmup_days(_get(strategy, "supertrend_atr_period", 10), default=10),
            )
        elif str(regime_mode) == "ema":
            bars_needed = max(bars_needed, _ema_warmup_days(regime_preset))

    regime2_mode, regime2_preset, _regime2_bar, use_mtf_regime2 = resolve_spot_regime2_spec(
        bar_size=str(signal_bar_size),
        regime2_mode_raw=_get(strategy, "regime2_mode", "off"),
        regime2_ema_preset_raw=_get(strategy, "regime2_ema_preset", None),
        regime2_bar_size_raw=_get(strategy, "regime2_bar_size", None),
    )
    if not bool(use_mtf_regime2):
        if str(regime2_mode) == "supertrend":
            bars_needed = max(
                bars_needed,
                _supertrend_warmup_days(_get(strategy, "regime2_supertrend_atr_period", 10), default=10),
            )
        elif str(regime2_mode) == "ema":
            bars_needed = max(bars_needed, _ema_warmup_days(regime2_preset))

    warmup_days = _bars_to_warmup_days(
        bar_size=str(signal_bar_size),
        use_rth=bool(signal_use_rth),
        bars=int(bars_needed),
    )
    parsed_signal = parse_bar_size(str(signal_bar_size))
    if (
        entry_signal == "ema"
        and bool(signal_use_rth)
        and parsed_signal is not None
        and parsed_signal.duration < timedelta(days=1)
    ):
        warmup_days = max(
            7,
            _bars_to_warmup_days(
            bar_size=str(signal_bar_size),
            use_rth=bool(signal_use_rth),
            bars=int(max(bars_needed, _ema_warmup_days(_get(strategy, "ema_preset", None)))),
            ),
        )
    if bool(_get(strategy, "regime_router", False)):
        # Regime router needs enough *completed days* to become ready at the scoring start.
        # Without this, per-year backtests spend weeks/months in the default hf_host path,
        # skewing both performance and stability metrics.
        router_slow = _parse_int(_get(strategy, "regime_router_slow_window_days", 0), default=0, min_value=0)
        if router_slow > 0:
            # Router windows are in *trading days*; warmup is in *calendar days*.
            # Approximate trading→calendar expansion when running RTH-only data.
            router_warmup_days = int(router_slow) + 5
            if bool(signal_use_rth):
                router_warmup_days = int(math.ceil(float(router_slow) * (7.0 / 5.0))) + 7
            warmup_days = max(int(warmup_days), int(router_warmup_days))
    return int(warmup_days)


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
    signal_warmup_days = spot_signal_warmup_days_from_strategy(
        strategy=strategy,
        default_signal_bar_size=str(signal_bar_size),
        default_signal_use_rth=bool(signal_use_rth),
    )

    out: list[SpotBarRequirement] = []
    if include_signal:
        out.append(
            SpotBarRequirement(
                kind="signal",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(signal_bar_size),
                use_rth=bool(signal_use_rth),
                warmup_days=int(signal_warmup_days),
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
        if str(_regime_mode) == "supertrend":
            regime_warm_days = max(
                int(regime_warm_days),
                _supertrend_warmup_days(_get(strategy, "supertrend_atr_period", 10), default=10),
            )
        elif str(_regime_mode) == "ema":
            regime_warm_days = max(int(regime_warm_days), _ema_warmup_days(_regime_preset))
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
    regime2_warm_days = 0
    if str(_r2_mode) == "supertrend":
        regime2_warm_days = _supertrend_warmup_days(_get(strategy, "regime2_supertrend_atr_period", 10), default=10)
    elif str(_r2_mode) == "ema":
        regime2_warm_days = _ema_warmup_days(_r2_preset)
    if bool(use_mtf_regime2):
        out.append(
            SpotBarRequirement(
                kind="regime2",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(regime2_bar),
                use_rth=bool(signal_use_rth),
                warmup_days=int(regime2_warm_days),
            )
        )
    regime2_bear_hard_mode = str(_get(strategy, "regime2_bear_hard_mode", "off") or "off").strip().lower()
    regime2_bear_hard_bar = str(_get(strategy, "regime2_bear_hard_bar_size", "") or "").strip()
    if not regime2_bear_hard_bar or regime2_bear_hard_bar.lower() in ("same", "default"):
        regime2_bear_hard_bar = str(regime2_bar)
    if regime2_bear_hard_mode == "supertrend" and str(regime2_bear_hard_bar) != str(signal_bar_size):
        hard_warm_days = _supertrend_warmup_days(
            _get(
                strategy,
                "regime2_bear_hard_supertrend_atr_period",
                _get(strategy, "regime2_supertrend_atr_period", 10),
            ),
            default=_parse_int(_get(strategy, "regime2_supertrend_atr_period", 10), default=10, min_value=1),
        )
        out.append(
            SpotBarRequirement(
                kind="regime2_bear_hard",
                symbol=symbol,
                exchange=exchange,
                bar_size=str(regime2_bear_hard_bar),
                use_rth=bool(signal_use_rth),
                warmup_days=int(hard_warm_days),
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
                warmup_days=int(signal_warmup_days),
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
        regime2_bear_hard_bars=by_kind.get("regime2_bear_hard"),
        tick_bars=by_kind.get("tick"),
        exec_bars=by_kind.get("exec"),
    )
