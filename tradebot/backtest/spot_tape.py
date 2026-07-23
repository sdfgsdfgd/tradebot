"""Immutable spot-evaluator evidence shared across backtest combinations."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

from ..chart_data.cache import series_cache_service
from ..engines.risk import RiskOverlaySnapshot
from ..spot.evaluator_common import SpotSignalSnapshot
from ..spot_engine import SpotSignalEvaluator
from .config import ConfigBundle, SpotStrategyConfig
from .models import Bar


_SPOT_EVALUATOR_TAPE_NAMESPACE = "spot.evaluator_tape.v1"
_SPOT_EVALUATOR_TAPE_CACHE = series_cache_service()
# These affect fills/economics only; SpotSignalEvaluator must never consume them.
_EXECUTION_ONLY_STRATEGY_FIELDS = frozenset(
    {
        "spot_close_eod",
        "spot_profit_target_pct",
        "spot_pt_atr_mult",
        "spot_short_risk_mult",
        "spot_sl_atr_mult",
        "spot_stop_loss_pct",
    }
)
ShockView = tuple[bool | None, str | None, float | None]


@dataclass(frozen=True)
class PreparedSpotEvaluatorTape:
    """Position-independent evidence aligned one-to-one with execution bars."""

    prior_shocks: tuple[ShockView, ...]
    risks: tuple[RiskOverlaySnapshot | None, ...]
    signals: tuple[SpotSignalSnapshot | None, ...]
    risk_overlay_enabled: bool
    shock_enabled: bool


def _strategy_key(cfg: ConfigBundle) -> str:
    payload = asdict(cfg.strategy)
    for field_name in _EXECUTION_ONLY_STRATEGY_FIELDS:
        payload.pop(field_name, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def prepare_spot_evaluator_tape(
    *,
    cfg: ConfigBundle,
    signal_bars: Sequence[Bar],
    exec_bars: Sequence[Bar],
    sig_idx_by_exec_idx: Sequence[int],
    exec_dates: Sequence[date],
    regime_bars: Sequence[Bar] | None = None,
    regime2_bars: Sequence[Bar] | None = None,
    regime2_bear_hard_bars: Sequence[Bar] | None = None,
) -> PreparedSpotEvaluatorTape:
    """Replay position-independent signal, regime, shock, and risk evidence once."""

    if not isinstance(cfg.strategy, SpotStrategyConfig):
        raise ValueError("prepare_spot_evaluator_tape requires a spot strategy config")
    if len(sig_idx_by_exec_idx) != len(exec_bars) or len(exec_dates) != len(exec_bars):
        raise ValueError("spot evaluator alignment does not match execution bars")

    revisions = tuple(
        _SPOT_EVALUATOR_TAPE_CACHE.revision(rows or ())
        for rows in (
            signal_bars,
            exec_bars,
            regime_bars,
            regime2_bars,
            regime2_bear_hard_bars,
        )
    )
    key = (
        revisions,
        _strategy_key(cfg),
        str(cfg.backtest.bar_size),
        bool(cfg.backtest.use_rth),
        int(cfg.synthetic.rv_lookback),
        float(cfg.synthetic.rv_ewma_lambda),
    )
    cached = _SPOT_EVALUATOR_TAPE_CACHE.get(
        namespace=_SPOT_EVALUATOR_TAPE_NAMESPACE,
        key=key,
    )
    if isinstance(cached, PreparedSpotEvaluatorTape):
        return cached

    evaluator = SpotSignalEvaluator(
        strategy=cfg.strategy,
        filters=cfg.strategy.filters,
        bar_size=str(cfg.backtest.bar_size),
        use_rth=bool(cfg.backtest.use_rth),
        naive_ts_mode="utc",
        rv_lookback=int(cfg.synthetic.rv_lookback),
        rv_ewma_lambda=float(cfg.synthetic.rv_ewma_lambda),
        regime_bars=list(regime_bars) if regime_bars else None,
        regime2_bars=list(regime2_bars) if regime2_bars else None,
        regime2_bear_hard_bars=(
            list(regime2_bear_hard_bars) if regime2_bear_hard_bars else None
        ),
    )
    prior_shocks: list[ShockView] = []
    risks: list[RiskOverlaySnapshot | None] = []
    signals: list[SpotSignalSnapshot | None] = []
    for exec_idx, bar in enumerate(exec_bars):
        prior_shocks.append(evaluator.shock_view)
        evaluator.update_exec_bar(
            bar,
            is_last_bar=(
                exec_idx + 1 == len(exec_bars)
                or exec_dates[exec_idx + 1] != exec_dates[exec_idx]
            ),
        )
        sig_idx = int(sig_idx_by_exec_idx[exec_idx])
        signals.append(
            evaluator.update_signal_bar(signal_bars[sig_idx])
            if 0 <= sig_idx < len(signal_bars)
            else None
        )
        risks.append(evaluator.last_risk)

    prepared = PreparedSpotEvaluatorTape(
        prior_shocks=tuple(prior_shocks),
        risks=tuple(risks),
        signals=tuple(signals),
        risk_overlay_enabled=bool(evaluator.risk_overlay_enabled),
        shock_enabled=bool(evaluator.shock_enabled),
    )
    return _SPOT_EVALUATOR_TAPE_CACHE.set(
        namespace=_SPOT_EVALUATOR_TAPE_NAMESPACE,
        key=key,
        value=prepared,
        max_entries=8,
    )
