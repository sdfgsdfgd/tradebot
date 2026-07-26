"""Canonical spot entry source and permission control plane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..engine import (
    normalize_spot_entry_signal,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
)
from ..engines.directional_impulse import DirectionalImpulseAdmissionPolicy
from .evaluator_common import SpotGateBand, SpotRegimeGatePolicy
from .graph import SpotPolicyGraph
from .policy_contract import normalize_shock_gate_mode
from .policy_contract import source_value as _get
from .gates import _active_signal_filter_names


def normalize_tick_gate_mode(raw: object | None) -> str:
    mode = str(raw or "off").strip().lower()
    return mode if mode in ("off", "raschke") else "off"


def spot_allowed_entry_directions(
    strategy: Mapping[str, object] | object | None,
) -> tuple[str, ...]:
    """Decode the spot direction/action map once for live and backtests."""
    mapping = _get(strategy, "directional_spot", None)
    if not isinstance(mapping, Mapping):
        return ("up",)
    allowed: list[str] = []
    for direction in ("up", "down"):
        leg = mapping.get(direction)
        action = (
            leg.get("action")
            if isinstance(leg, Mapping)
            else getattr(leg, "action", None)
        )
        if str(action or "").strip().upper() in ("BUY", "SELL"):
            allowed.append(direction)
    return tuple(allowed)


def entry_day_allowed(
    *,
    weekday: int,
    entry_days: Sequence[object],
    default: Sequence[int] = (0, 1, 2, 3, 4),
) -> bool:
    """Normalize configured weekdays once for live and backtest permission."""
    names = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }
    allowed: set[int] = set()
    for raw in entry_days or default:
        if isinstance(raw, int) and not isinstance(raw, bool):
            if 0 <= int(raw) <= 6:
                allowed.add(int(raw))
            continue
        value = names.get(str(raw or "").strip().upper()[:3])
        if value is not None:
            allowed.add(value)
    if not allowed:
        allowed.update(int(day) for day in default)
    return int(weekday) in allowed


def _optional_number(
    source: object,
    key: str,
    *,
    integer: bool = False,
    nonnegative: bool = False,
    maximum: int | None = None,
) -> float | int | None:
    raw = _get(source, key, None)
    try:
        value = int(raw) if integer and raw is not None else (
            float(raw) if raw is not None else None
        )
    except (TypeError, ValueError):
        return None
    if value is None or (nonnegative and value < 0):
        return None
    return min(int(value), maximum) if integer and maximum is not None else value


def _gate_band(
    source: object,
    minimum_key: str,
    maximum_key: str,
    *,
    integer: bool = False,
    nonnegative: bool = False,
) -> SpotGateBand:
    minimum = _optional_number(
        source,
        minimum_key,
        integer=integer,
        nonnegative=nonnegative,
    )
    maximum = _optional_number(
        source,
        maximum_key,
        integer=integer,
        nonnegative=nonnegative,
    )
    if minimum is not None and maximum is not None and maximum < minimum:
        maximum = minimum
    return SpotGateBand(minimum, maximum)


def spot_regime_gate_policy(strategy: object) -> SpotRegimeGatePolicy:
    """Decode historical `regime2_*` vetoes once into the entry control plane."""
    prearm_scope = str(
        _get(strategy, "regime2_crash_prearm_apply_to", "off") or "off"
    ).strip().lower()
    if prearm_scope not in ("off", "branch_b_longs", "all_longs"):
        prearm_scope = "off"

    def number(key, **kwargs):
        return _optional_number(strategy, key, **kwargs)

    def band(low, high, **kwargs):
        return _gate_band(strategy, low, high, **kwargs)

    return SpotRegimeGatePolicy(
        crash_atr_min=number(
            "regime2_crash_atr_pct_min",
            nonnegative=True,
        ),
        crash_block_longs=bool(
            _get(strategy, "regime2_crash_block_longs", False)
        ),
        transition_hot_atr_min=number(
            "regime2_transition_hot_shock_atr_pct_min",
            nonnegative=True,
        ),
        transition_hot_release_age_max=number(
            "regime2_transition_hot_release_max_bars",
            integer=True,
            nonnegative=True,
        ),
        crash_prearm_scope=prearm_scope,
        crash_prearm_atr_min=number(
            "regime2_crash_prearm_shock_atr_pct_min",
            nonnegative=True,
        ),
        crash_prearm_ret_max=number(
            "regime2_crash_prearm_shock_dir_ret_sum_pct_max"
        ),
        crash_prearm_branch_a_atr_min=number(
            "regime2_crash_prearm_branch_a_shock_atr_pct_min",
            nonnegative=True,
        ),
        crash_prearm_branch_a_ret_max=number(
            "regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max"
        ),
        repair_branch_b_block=bool(
            _get(strategy, "regime2_repair_block_branch_b_longs", False)
        ),
        repair_branch_b_atr_max=number(
            "regime2_repair_branch_b_long_max_shock_atr_pct",
            nonnegative=True,
        ),
        repair_branch_b_after_hour=number(
            "regime2_repair_branch_b_long_block_after_hour_et",
            integer=True,
            nonnegative=True,
            maximum=23,
        ),
        upcorridor_branch_a_mid_atr=band(
            "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min",
            "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max",
            nonnegative=True,
        ),
        upcorridor_branch_a_extreme_atr_min=number(
            "regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min",
            nonnegative=True,
        ),
        upcorridor_branch_a_fresh_age_max=number(
            "regime2_upcorridor_branch_a_long_fresh_release_age_max_bars",
            integer=True,
            nonnegative=True,
        ),
        upcorridor_branch_a_stale_age_min=number(
            "regime2_upcorridor_branch_a_long_stale_release_age_min_bars",
            integer=True,
            nonnegative=True,
        ),
        upcorridor_branch_b_stale_age_min=number(
            "regime2_upcorridor_branch_b_long_stale_release_age_min_bars",
            integer=True,
            nonnegative=True,
        ),
        upcorridor_branch_b_flat_low_atr_max=number(
            "regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max",
            nonnegative=True,
        ),
        upcorridor_branch_b_flat_low_stale_age_min=number(
            "regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars",
            integer=True,
            nonnegative=True,
        ),
        upcorridor_branch_b_flat_atr_max=number(
            "regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max",
            nonnegative=True,
        ),
        upcorridor_branch_b_flat_ddv_abs_max=number(
            "regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp",
            nonnegative=True,
        ),
        trenddown_branch_b_release_age=band(
            "regime2_trenddown_branch_b_long_hard_up_release_age_min_bars",
            "regime2_trenddown_branch_b_long_hard_up_release_age_max_bars",
            integer=True,
            nonnegative=True,
        ),
        trenddown_branch_b_atr=band(
            "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min",
            "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max",
            nonnegative=True,
        ),
        trenddown_branch_b_ddv=band(
            "regime2_trenddown_branch_b_long_hard_up_ddv_min_pp",
            "regime2_trenddown_branch_b_long_hard_up_ddv_max_pp",
        ),
        trenddown_branch_b_recovery_atr=band(
            "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min",
            "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max",
            nonnegative=True,
        ),
        trenddown_branch_b_recovery_ddv=band(
            "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp",
            "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp",
        ),
        continuation_branch_b_release_age=band(
            "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars",
            "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars",
            integer=True,
            nonnegative=True,
        ),
        continuation_branch_a_release_age_max=number(
            "regime2_continuation_confidence_branch_a_transition_release_age_max_bars",
            integer=True,
            nonnegative=True,
        ),
        continuation_branch_a_atr=band(
            "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min",
            "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max",
            nonnegative=True,
        ),
        continuation_branch_a_ddv_max=number(
            "regime2_continuation_confidence_branch_a_transition_ddv_max_pp"
        ),
    )


@dataclass(frozen=True)
class SpotEntryControlPlan:
    """One normalized control plane for spot entry direction and permission."""

    source: str
    source_gates: tuple[str, ...]
    primary_regime: str
    primary_regime_preset: str | None
    primary_regime_bar_size: str
    primary_regime_mtf: bool
    confirmation_regime: str
    confirmation_regime_preset: str | None
    confirmation_regime_bar_size: str
    confirmation_regime_mtf: bool
    confirmation_scope: str
    bear_takeover: str
    bear_takeover_scope: str
    shock_gate: str
    signal_filters: tuple[str, ...]
    tick_gate: str
    allowed_directions: tuple[str, ...]
    graph_entry_policy: str
    directional_impulse: str
    directional_impulse_admission: DirectionalImpulseAdmissionPolicy | None
    fundamental_pressure: str
    regime_gates: SpotRegimeGatePolicy
    observations: tuple[str, ...] = ("directional_impulse",)
    lifecycle_checks: tuple[str, ...] = (
        "freshness",
        "data_gap",
        "preflight",
        "pending_order",
        "weekday",
        "entry_day",
        "entry_capacity",
        "signal_ready",
        "allowed_direction",
        "signal_filters",
        "exit_atr_ready",
        "next_open",
    )

    @classmethod
    def from_sources(
        cls,
        *,
        strategy: Mapping[str, object] | object | None,
        filters: Mapping[str, object] | object | None,
        bar_size: str,
    ) -> SpotEntryControlPlan:
        (
            primary_mode,
            primary_preset,
            primary_bar_size,
            primary_mtf,
        ) = resolve_spot_regime_spec(
            bar_size=str(bar_size),
            regime_mode_raw=_get(strategy, "regime_mode", "ema"),
            regime_ema_preset_raw=_get(strategy, "regime_ema_preset", None),
            regime_bar_size_raw=_get(strategy, "regime_bar_size", None),
        )
        primary = (
            str(primary_mode)
            if primary_mode == "supertrend"
            or (primary_mode == "ema" and primary_preset)
            else "off"
        )
        (
            confirmation,
            confirmation_preset,
            confirmation_bar_size,
            confirmation_mtf,
        ) = resolve_spot_regime2_spec(
            bar_size=str(bar_size),
            regime2_mode_raw=_get(strategy, "regime2_mode", "off"),
            regime2_ema_preset_raw=_get(strategy, "regime2_ema_preset", None),
            regime2_bar_size_raw=_get(strategy, "regime2_bar_size", None),
        )
        bear_takeover = str(
            _get(strategy, "regime2_bear_entry_mode", "off") or "off"
        ).strip().lower()
        if bear_takeover not in ("off", "supertrend"):
            bear_takeover = "off"
        confirmation_scope = str(
            _get(strategy, "regime2_apply_to", "both") or "both"
        ).strip().lower()
        if confirmation_scope in ("off", "none", "disabled", "false", "0", "soft"):
            confirmation_scope = "off"
        elif confirmation_scope not in ("longs", "shorts"):
            confirmation_scope = "both"
        bear_takeover_scope = str(
            _get(strategy, "regime2_bear_takeover_mode", "always") or "always"
        ).strip().lower()
        if bear_takeover_scope not in (
            "always",
            "hostile",
            "riskoff",
            "riskpanic",
            "shockdown",
            "hostile_or_shockdown",
        ):
            bear_takeover_scope = "always"
        # Entry graph policy is strategy-owned in both live and backtest;
        # filter-owned sizing remains a separate graph.
        graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
        regime_gates = spot_regime_gate_policy(strategy)
        source = normalize_spot_entry_signal(
            _get(strategy, "entry_signal", "ema")
        )
        impulse_admission = DirectionalImpulseAdmissionPolicy.from_mapping(
            _get(strategy, "directional_impulse_admission", None)
        )
        dual_branch = bool(
            source == "ema"
            and _get(strategy, "spot_dual_branch_enabled", False)
        )
        branch_slope = bool(
            dual_branch
            and any(
                (
                    _optional_number(strategy, key, nonnegative=True) or 0.0
                )
                > 0.0
                for key in (
                    "spot_branch_a_min_signed_slope_pct",
                    "spot_branch_a_max_signed_slope_pct",
                    "spot_branch_b_min_signed_slope_pct",
                    "spot_branch_b_max_signed_slope_pct",
                )
            )
        )
        source_gates = tuple(
            name
            for name, enabled in (
                ("dual_branch", dual_branch),
                ("branch_slope", branch_slope),
                (
                    "directional_impulse_admission",
                    source == "directional_impulse"
                    and impulse_admission is not None,
                ),
                (
                    "ratsv",
                    source == "ema"
                    and bool(_get(filters, "ratsv_enabled", False)),
                ),
            )
            if enabled
        )
        impulse_mode = str(
            _get(strategy, "directional_impulse_mode", "observe") or "observe"
        ).strip().lower()
        if impulse_mode not in ("off", "observe"):
            impulse_mode = "observe"
        news_mode = str(
            _get(strategy, "fundamental_pressure_mode", "off") or "off"
        ).strip().lower()
        if news_mode not in ("off", "observe"):
            news_mode = "off"
        return cls(
            source=source,
            source_gates=source_gates,
            primary_regime=primary,
            primary_regime_preset=primary_preset,
            primary_regime_bar_size=primary_bar_size,
            primary_regime_mtf=bool(primary_mtf),
            confirmation_regime=str(confirmation),
            confirmation_regime_preset=confirmation_preset,
            confirmation_regime_bar_size=confirmation_bar_size,
            confirmation_regime_mtf=bool(confirmation_mtf),
            confirmation_scope=confirmation_scope,
            bear_takeover=bear_takeover,
            bear_takeover_scope=bear_takeover_scope,
            shock_gate=normalize_shock_gate_mode(filters),
            signal_filters=_active_signal_filter_names(filters),
            tick_gate=normalize_tick_gate_mode(
                _get(strategy, "tick_gate_mode", "off")
            ),
            allowed_directions=spot_allowed_entry_directions(strategy),
            graph_entry_policy=str(graph.entry_policy),
            directional_impulse=impulse_mode,
            directional_impulse_admission=impulse_admission,
            fundamental_pressure=news_mode,
            regime_gates=regime_gates,
            observations=tuple(
                name
                for name, enabled in (
                    (
                        "directional_impulse",
                        impulse_mode == "observe"
                        and source != "directional_impulse",
                    ),
                    ("fundamental_pressure", news_mode == "observe"),
                )
                if enabled
            ),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_gates": list(self.source_gates),
            "observations": list(self.observations),
            "observation_modes": {
                "directional_impulse": self.directional_impulse,
                "fundamental_pressure": self.fundamental_pressure,
            },
            "directional_impulse_admission": (
                self.directional_impulse_admission.as_payload()
                if self.directional_impulse_admission is not None
                else None
            ),
            "confirmations": {
                "primary_regime": self.primary_regime,
                "regime2": self.confirmation_regime,
                "bear_takeover": self.bear_takeover,
            },
            "confirmation_scopes": {
                "regime2": self.confirmation_scope,
                "bear_takeover": self.bear_takeover_scope,
            },
            "confirmation_inputs": {
                "primary_regime": {
                    "preset": self.primary_regime_preset,
                    "bar_size": self.primary_regime_bar_size,
                    "multi_timeframe": self.primary_regime_mtf,
                },
                "regime2": {
                    "preset": self.confirmation_regime_preset,
                    "bar_size": self.confirmation_regime_bar_size,
                    "multi_timeframe": self.confirmation_regime_mtf,
                },
            },
            "filters": list(self.signal_filters),
            "regime_entry_gates": list(self.regime_gates.active_gates()),
            "tick_gate": self.tick_gate,
            "allowed_directions": list(self.allowed_directions),
            "lifecycle_checks": list(self.lifecycle_checks),
            "graph_entry_policy": self.graph_entry_policy,
            "order": [
                "source",
                "source_gates",
                "primary_regime",
                "regime2",
                "bear_takeover",
                "regime_entry_gates",
                "signal_filters",
                "tick_gate",
                "direction_mapping",
                "lifecycle",
                "graph_entry_policy",
            ],
        }
