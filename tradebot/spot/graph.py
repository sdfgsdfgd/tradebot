"""Composable spot policy graph.

This module provides a central registry for spot lifecycle/sizing policy nodes:
- entry gate policies
- exit arbitration policies
- resize target policies
- risk overlay policies

Profiles map to a coherent set of node names and can be overridden per-node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime

_DEFAULT_EXIT_PRIORITY: tuple[str, ...] = (
    "quote_starvation_kill",
    "stop_loss",
    "stop_loss_pct",
    "profit_target",
    "profit_target_pct",
    "ratsv_probe_cancel",
    "ratsv_adverse_release",
    "flip",
    "exit_time",
    "close_eod",
    "dte",
    "stop_loss_credit",
    "stop_loss_max_loss",
)

_EXIT_REASON_ALIASES: dict[str, str] = {
    "profit": "profit_target_pct",
    "stop": "stop_loss_pct",
    "time": "exit_time",
    "eod": "close_eod",
}


def _get(source: Mapping[str, object] | object | None, key: str, default: object = None) -> object:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _has(source: Mapping[str, object] | object | None, key: str) -> bool:
    if source is None:
        return False
    if isinstance(source, Mapping):
        return key in source
    return hasattr(source, key)


def _parse_float(value: object, *, default: float) -> float:
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: object, *, default: int) -> int:
    try:
        return int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _pick_value(
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    keys: Sequence[str],
) -> object | None:
    for key in keys:
        if _has(strategy, str(key)):
            return _get(strategy, str(key))
    for key in keys:
        if _has(filters, str(key)):
            return _get(filters, str(key))
    return None


def _normalize_dynamic_scale_mode(raw: object | None) -> str:
    mode = str(raw or "off").strip().lower()
    if mode in ("", "0", "false", "none", "null", "off"):
        return "off"
    aliases = {
        "hybrid": "atrtr",
        "tr_atr": "atrtr",
        "atr_tr": "atrtr",
        "atr+tr": "atrtr",
        "atr-tr": "atrtr",
        "atr_tr_med": "atrtrmed",
        "atr+tr+med": "atrtrmed",
        "atrtrm": "atrtrmed",
        "trmed": "tr_median",
        "tr_med": "tr_median",
        "trmedian": "tr_median",
        "median_tr": "tr_median",
        "atrvel": "atr_vel",
        "atrvelocity": "atr_vel",
        "trd": "tr_direct",
        "tr_direct": "tr_direct",
        "atrveld": "atr_vel_direct",
        "atr_vel_direct": "atr_vel_direct",
        "trmedd": "tr_median_direct",
        "tr_median_direct": "tr_median_direct",
        "atrtrd": "atrtr_direct",
        "atrtr_direct": "atrtr_direct",
        "atrtrmedd": "atrtrmed_direct",
        "atrtrmed_direct": "atrtrmed_direct",
        "atr+tr+direct": "atrtr_direct",
        "atr+tr+med+direct": "atrtrmed_direct",
    }
    mode = aliases.get(mode, mode)
    if mode not in (
        "off",
        "tr",
        "atr_vel",
        "atrtr",
        "tr_median",
        "atrtrmed",
        "tr_direct",
        "atr_vel_direct",
        "atrtr_direct",
        "tr_median_direct",
        "atrtrmed_direct",
    ):
        return "off"
    return mode


def _dynamic_scale_from_mode(
    *,
    strategy: Mapping[str, object] | object | None,
    mode_key: str,
    min_mult_key: str,
    max_mult_key: str,
    tr_ref_key: str,
    atr_vel_ref_key: str,
    tr_median_ref_key: str | None,
    tr_ratio: float | None,
    shock_atr_vel_pct: float | None,
    tr_median_pct: float | None,
    default_min_mult: float,
    default_max_mult: float,
    default_tr_ref: float,
    default_atr_vel_ref: float,
    default_tr_median_ref: float,
) -> tuple[float, dict[str, object]]:
    mode = _normalize_dynamic_scale_mode(_get(strategy, mode_key))
    direct = bool(str(mode).endswith("_direct"))
    base_mode = str(mode)[:-7] if direct else str(mode)
    min_mult = max(0.05, _parse_float(_get(strategy, min_mult_key), default=float(default_min_mult)))
    max_mult = max(float(min_mult), _parse_float(_get(strategy, max_mult_key), default=float(default_max_mult)))
    tr_ref = max(1e-9, _parse_float(_get(strategy, tr_ref_key), default=float(default_tr_ref)))
    atr_vel_ref = max(1e-9, _parse_float(_get(strategy, atr_vel_ref_key), default=float(default_atr_vel_ref)))
    tr_median_ref = max(
        1e-9,
        _parse_float(
            _get(strategy, str(tr_median_ref_key)) if tr_median_ref_key else None,
            default=float(default_tr_median_ref),
        ),
    )

    components: list[float] = []
    if base_mode in ("tr", "atrtr", "atrtrmed") and tr_ratio is not None and float(tr_ratio) > 0.0:
        tr_now = max(1e-9, float(tr_ratio))
        components.append((tr_now / float(tr_ref)) if direct else (float(tr_ref) / tr_now))
    if base_mode in ("atr_vel", "atrtr", "atrtrmed") and shock_atr_vel_pct is not None:
        atr_vel_up = max(0.0, float(shock_atr_vel_pct))
        if atr_vel_up <= 0.0:
            components.append(float(min_mult) if direct else float(max_mult))
        else:
            atr_now = max(1e-9, float(atr_vel_up))
            components.append((atr_now / float(atr_vel_ref)) if direct else (float(atr_vel_ref) / atr_now))
    if base_mode in ("tr_median", "atrtrmed") and tr_median_pct is not None:
        tr_median_up = max(0.0, float(tr_median_pct))
        if tr_median_up <= 0.0:
            components.append(float(min_mult) if direct else float(max_mult))
        else:
            tr_median_now = max(1e-9, float(tr_median_up))
            components.append(
                (tr_median_now / float(tr_median_ref)) if direct else (float(tr_median_ref) / tr_median_now)
            )

    if mode == "off" or not components:
        scale = 1.0
    else:
        scale = float(sum(components) / float(len(components)))
    scale = float(max(float(min_mult), min(float(max_mult), float(scale))))
    return float(scale), {
        "mode": str(mode),
        "min_mult": float(min_mult),
        "max_mult": float(max_mult),
        "tr_ref": float(tr_ref),
        "tr_median_ref_pct": float(tr_median_ref),
        "atr_vel_ref_pct": float(atr_vel_ref),
        "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
        "tr_median_pct": float(tr_median_pct) if tr_median_pct is not None else None,
        "shock_atr_vel_pct": float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
        "components": [float(x) for x in components],
        "scale": float(scale),
    }


def spot_guard_threshold_scale(
    *,
    strategy: Mapping[str, object] | object | None,
    tr_ratio: float | None,
    shock_atr_vel_pct: float | None,
    tr_median_pct: float | None = None,
) -> tuple[float, dict[str, object]]:
    tr_ref_raw = _pick_value(
        strategy=strategy,
        filters=None,
        keys=("spot_graph_overlay_tr_ratio_ref", "spot_resize_adaptive_tr_ratio_ref"),
    )
    atr_ref_raw = _pick_value(
        strategy=strategy,
        filters=None,
        keys=("spot_graph_overlay_atr_vel_ref_pct", "spot_resize_adaptive_atr_vel_ref_pct"),
    )
    default_tr_ref = max(1e-9, _parse_float(tr_ref_raw, default=1.0))
    default_atr_ref = max(1e-9, _parse_float(atr_ref_raw, default=0.40))
    return _dynamic_scale_from_mode(
        strategy=strategy,
        mode_key="spot_guard_threshold_scale_mode",
        min_mult_key="spot_guard_threshold_scale_min_mult",
        max_mult_key="spot_guard_threshold_scale_max_mult",
        tr_ref_key="spot_guard_threshold_scale_tr_ref",
        atr_vel_ref_key="spot_guard_threshold_scale_atr_vel_ref_pct",
        tr_median_ref_key="spot_guard_threshold_scale_tr_median_ref_pct",
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
        default_min_mult=0.70,
        default_max_mult=1.80,
        default_tr_ref=float(default_tr_ref),
        default_atr_vel_ref=float(default_atr_ref),
        default_tr_median_ref=1.0,
    )


def spot_dynamic_flip_hold_bars(
    *,
    strategy: Mapping[str, object] | object | None,
    tr_ratio: float | None,
    shock_atr_vel_pct: float | None,
    tr_median_pct: float | None = None,
) -> tuple[int, dict[str, object]]:
    base_hold = max(0, _parse_int(_get(strategy, "flip_exit_min_hold_bars"), default=0))
    tr_ref_raw = _pick_value(
        strategy=strategy,
        filters=None,
        keys=("spot_graph_overlay_tr_ratio_ref", "spot_resize_adaptive_tr_ratio_ref"),
    )
    atr_ref_raw = _pick_value(
        strategy=strategy,
        filters=None,
        keys=("spot_graph_overlay_atr_vel_ref_pct", "spot_resize_adaptive_atr_vel_ref_pct"),
    )
    default_tr_ref = max(1e-9, _parse_float(tr_ref_raw, default=1.0))
    default_atr_ref = max(1e-9, _parse_float(atr_ref_raw, default=0.40))
    scale, scale_trace = _dynamic_scale_from_mode(
        strategy=strategy,
        mode_key="spot_flip_hold_dynamic_mode",
        min_mult_key="spot_flip_hold_dynamic_min_mult",
        max_mult_key="spot_flip_hold_dynamic_max_mult",
        tr_ref_key="spot_flip_hold_dynamic_tr_ref",
        atr_vel_ref_key="spot_flip_hold_dynamic_atr_vel_ref_pct",
        tr_median_ref_key="spot_flip_hold_dynamic_tr_median_ref_pct",
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
        default_min_mult=0.50,
        default_max_mult=2.50,
        default_tr_ref=float(default_tr_ref),
        default_atr_vel_ref=float(default_atr_ref),
        default_tr_median_ref=1.0,
    )
    hold_bars = max(0, int(round(float(base_hold) * float(scale))))
    return int(hold_bars), {
        "base_hold_bars": int(base_hold),
        "dynamic_scale": float(scale),
        "dynamic": dict(scale_trace),
        "hold_bars": int(hold_bars),
    }


def canonical_exit_reason(reason: str | None) -> str:
    key = str(reason or "").strip().lower()
    if not key:
        return ""
    return _EXIT_REASON_ALIASES.get(key, key)


def pick_exit_reason(
    exit_candidates: Mapping[str, bool] | None,
    *,
    priority: Sequence[str] | None = None,
) -> str | None:
    if not isinstance(exit_candidates, Mapping) or not exit_candidates:
        return None
    ordered = tuple(str(p) for p in (priority or _DEFAULT_EXIT_PRIORITY))
    for reason in ordered:
        if bool(exit_candidates.get(reason)):
            return canonical_exit_reason(reason)
    for reason, active in exit_candidates.items():
        if bool(active):
            return canonical_exit_reason(str(reason))
    return None


@dataclass(frozen=True)
class SpotGraphProfile:
    name: str
    entry_policy: str
    exit_policy: str
    resize_policy: str
    risk_overlay_policy: str
    notes: str = ""


@dataclass(frozen=True)
class SpotEntryGateResult:
    allow: bool
    reason: str = "entry_allowed"
    gate: str = "TRIGGER_ENTRY"
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpotExitReasonResult:
    reason: str | None
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpotResizeTargetResult:
    target_qty: int
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpotRiskOverlayAdjustments:
    risk_mult: float = 1.0
    long_risk_mult: float = 1.0
    short_risk_mult: float = 1.0
    cap_mult: float = 1.0
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


_GRAPH_PROFILES: dict[str, SpotGraphProfile] = {
    "neutral": SpotGraphProfile(
        name="neutral",
        entry_policy="default",
        exit_policy="priority",
        resize_policy="adaptive",
        risk_overlay_policy="legacy",
        notes="Conservative default graph; behavior driven by direct knobs.",
    ),
    "defensive": SpotGraphProfile(
        name="defensive",
        entry_policy="slope_tr_guard",
        exit_policy="slope_flip_guard",
        resize_policy="adaptive_atr_defensive",
        risk_overlay_policy="atr_compress",
        notes="More selective entries and tighter ATR-driven risk compression.",
    ),
    "aggressive": SpotGraphProfile(
        name="aggressive",
        entry_policy="default",
        exit_policy="priority",
        resize_policy="adaptive_hybrid_aggressive",
        risk_overlay_policy="trend_bias",
        notes="Allows stronger trend-following expansion when momentum confirms.",
    ),
    "hf_probe": SpotGraphProfile(
        name="hf_probe",
        entry_policy="slope_tr_guard",
        exit_policy="slope_flip_guard",
        resize_policy="adaptive_slope_probe",
        risk_overlay_policy="trend_bias",
        notes="Faster probe loops with slope-aware entry/resize controls.",
    ),
}


def normalize_profile_name(raw: object | None) -> str | None:
    name = str(raw or "").strip().lower()
    if not name:
        return None
    if name not in _GRAPH_PROFILES:
        return None
    return name


def all_graph_profiles() -> dict[str, SpotGraphProfile]:
    return dict(_GRAPH_PROFILES)


def _resolve_graph_profile_name(
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
) -> str:
    raw = _pick_value(
        strategy=strategy,
        filters=filters,
        keys=("spot_policy_graph", "spot_graph_profile"),
    )
    if raw is None:
        raw = _pick_value(
            strategy=strategy,
            filters=filters,
            keys=("spot_policy_pack",),
        )
    name = normalize_profile_name(raw)
    return str(name or "neutral")


def _node_name(
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    keys: Sequence[str],
    default: str,
) -> str:
    raw = _pick_value(strategy=strategy, filters=filters, keys=keys)
    if raw is None:
        return str(default)
    name = str(raw).strip().lower()
    return str(name or default)


def _adaptive_target(
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None = None,
    shock_atr_vel_pct: float | None = None,
    shock_atr_accel_pct: float | None = None,
    tr_ratio: float | None = None,
    slope_med_pct: float | None = None,
    slope_vel_pct: float | None = None,
    slope_med_slow_pct: float | None = None,
    slope_vel_slow_pct: float | None = None,
    default_mode: str = "off",
    default_min_mult: float = 0.5,
    default_max_mult: float = 1.75,
) -> SpotResizeTargetResult:
    base = int(base_target_qty)
    current = int(current_qty)

    raw_mode = _get(strategy, "spot_resize_adaptive_mode")
    mode = str(raw_mode if raw_mode is not None else default_mode).strip().lower()
    if mode not in ("off", "atr", "slope", "hybrid"):
        mode = "off"
    if mode == "off" or base == 0:
        return SpotResizeTargetResult(
            target_qty=int(base),
            trace={"policy_mode": mode, "scale": 1.0, "base_target_qty": int(base), "target_qty": int(base)},
        )

    min_mult = max(
        0.05,
        _parse_float(_get(strategy, "spot_resize_adaptive_min_mult"), default=float(default_min_mult)),
    )
    max_mult = max(
        float(min_mult),
        _parse_float(_get(strategy, "spot_resize_adaptive_max_mult"), default=float(default_max_mult)),
    )

    atr_target_raw = _get(strategy, "spot_resize_adaptive_atr_target_pct")
    atr_target = None if atr_target_raw is None else _parse_float(atr_target_raw, default=0.0)
    atr_scale = 1.0
    if mode in ("atr", "hybrid") and atr_target is not None and atr_target > 0 and shock_atr_pct is not None:
        atr_now = _parse_float(shock_atr_pct, default=0.0)
        if atr_now > 0:
            atr_scale = float(atr_target) / float(atr_now)

    slope_ref = max(1e-9, _parse_float(_get(strategy, "spot_resize_adaptive_slope_ref_pct"), default=0.10))
    vel_ref = max(1e-9, _parse_float(_get(strategy, "spot_resize_adaptive_vel_ref_pct"), default=0.08))
    tr_ref = max(1e-9, _parse_float(_get(strategy, "spot_resize_adaptive_tr_ratio_ref"), default=1.0))
    atr_vel_ref = max(1e-9, _parse_float(_get(strategy, "spot_resize_adaptive_atr_vel_ref_pct"), default=0.40))

    slope_scale = 1.0
    if mode in ("slope", "hybrid"):
        slope_strength = 0.0
        if slope_med_pct is not None:
            slope_strength += abs(float(slope_med_pct)) / float(slope_ref)
        if slope_vel_pct is not None:
            slope_strength += 0.75 * (abs(float(slope_vel_pct)) / float(vel_ref))
        if slope_med_slow_pct is not None:
            slope_strength += 0.50 * (abs(float(slope_med_slow_pct)) / float(slope_ref))
        if slope_vel_slow_pct is not None:
            slope_strength += 0.35 * (abs(float(slope_vel_slow_pct)) / float(vel_ref))
        if tr_ratio is not None:
            slope_strength += max(0.0, float(tr_ratio) - float(tr_ref))
        if shock_atr_vel_pct is not None:
            slope_strength += 0.30 * max(0.0, abs(float(shock_atr_vel_pct)) / float(atr_vel_ref))
        if shock_atr_accel_pct is not None:
            slope_strength += 0.15 * max(0.0, abs(float(shock_atr_accel_pct)) / float(atr_vel_ref))
        slope_scale = 0.6 + (0.4 * float(max(0.0, slope_strength)))
        if current != 0:
            adverse_med = (
                slope_med_pct is not None
                and ((current > 0 and float(slope_med_pct) < 0) or (current < 0 and float(slope_med_pct) > 0))
            )
            adverse_med_slow = (
                slope_med_slow_pct is not None
                and ((current > 0 and float(slope_med_slow_pct) < 0) or (current < 0 and float(slope_med_slow_pct) > 0))
            )
            adverse_vel = (
                slope_vel_pct is not None
                and ((current > 0 and float(slope_vel_pct) < 0) or (current < 0 and float(slope_vel_pct) > 0))
            )
            adverse_vel_slow = (
                slope_vel_slow_pct is not None
                and (
                    (current > 0 and float(slope_vel_slow_pct) < 0)
                    or (current < 0 and float(slope_vel_slow_pct) > 0)
                )
            )
            if bool(adverse_med or adverse_med_slow or adverse_vel or adverse_vel_slow):
                slope_scale *= 0.55

    if mode == "atr":
        scale = float(atr_scale)
    elif mode == "slope":
        scale = float(slope_scale)
    else:
        scale = 0.5 * float(atr_scale) + 0.5 * float(slope_scale)

    scale = float(max(min_mult, min(max_mult, float(scale))))
    target_abs = max(1, int(abs(int(base)) * float(scale)))
    target_qty = target_abs if int(base) > 0 else -target_abs
    return SpotResizeTargetResult(
        target_qty=int(target_qty),
        trace={
            "policy_mode": str(mode),
            "scale": float(scale),
            "min_mult": float(min_mult),
            "max_mult": float(max_mult),
            "atr_scale": float(atr_scale),
            "slope_scale": float(slope_scale),
            "base_target_qty": int(base),
            "target_qty": int(target_qty),
            "shock_atr_pct": float(shock_atr_pct) if shock_atr_pct is not None else None,
            "shock_atr_vel_pct": float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
            "shock_atr_accel_pct": float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "slope_med_pct": float(slope_med_pct) if slope_med_pct is not None else None,
            "slope_vel_pct": float(slope_vel_pct) if slope_vel_pct is not None else None,
            "slope_med_slow_pct": float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
            "slope_vel_slow_pct": float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
        },
    )


def _entry_policy_default(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    entry_dir: str | None,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotEntryGateResult:
    _ = (
        graph,
        strategy,
        bar_ts,
        entry_dir,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        tr_median_pct,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
    )
    return SpotEntryGateResult(
        allow=True,
        reason="entry_allowed",
        gate="TRIGGER_ENTRY",
        trace={"policy": "default"},
    )


def _entry_policy_slope_tr_guard(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    entry_dir: str | None,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotEntryGateResult:
    _ = (graph, bar_ts)
    tr_min = max(0.0, _parse_float(_get(strategy, "spot_entry_tr_ratio_min"), default=0.0))
    slope_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slope_med_abs_min_pct"), default=0.0))
    slope_vel_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slope_vel_abs_min_pct"), default=0.0))
    slope_slow_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slow_slope_med_abs_min_pct"), default=0.0))
    slope_vel_slow_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slow_slope_vel_abs_min_pct"), default=0.0))
    threshold_scale, threshold_trace = spot_guard_threshold_scale(
        strategy=strategy,
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
    )
    tr_min *= float(threshold_scale)
    slope_min *= float(threshold_scale)
    slope_vel_min *= float(threshold_scale)
    slope_slow_min *= float(threshold_scale)
    slope_vel_slow_min *= float(threshold_scale)
    atr_max = max(0.0, _parse_float(_get(strategy, "spot_entry_shock_atr_max_pct"), default=0.0))
    atr_vel_min = max(0.0, _parse_float(_get(strategy, "spot_entry_atr_vel_min_pct"), default=0.0))
    atr_accel_min = max(0.0, _parse_float(_get(strategy, "spot_entry_atr_accel_min_pct"), default=0.0))
    if atr_max > 0.0 and shock_atr_pct is not None and float(shock_atr_pct) > float(atr_max):
        return SpotEntryGateResult(
            allow=False,
            reason="graph_entry_shock_atr",
            gate="BLOCKED_GRAPH_ENTRY_SHOCK",
            trace={"policy": "slope_tr_guard", "shock_atr_pct": float(shock_atr_pct), "max": float(atr_max)},
        )
    if atr_vel_min > 0.0:
        if shock_atr_vel_pct is None or float(shock_atr_vel_pct) < float(atr_vel_min):
            return SpotEntryGateResult(
                allow=False,
                reason="graph_entry_atr_vel",
                gate="BLOCKED_GRAPH_ENTRY_ATR_VEL",
                trace={"policy": "slope_tr_guard", "shock_atr_vel_pct": shock_atr_vel_pct, "min": float(atr_vel_min)},
            )
    if atr_accel_min > 0.0:
        if shock_atr_accel_pct is None or float(shock_atr_accel_pct) < float(atr_accel_min):
            return SpotEntryGateResult(
                allow=False,
                reason="graph_entry_atr_accel",
                gate="BLOCKED_GRAPH_ENTRY_ATR_ACCEL",
                trace={
                    "policy": "slope_tr_guard",
                    "shock_atr_accel_pct": shock_atr_accel_pct,
                    "min": float(atr_accel_min),
                },
            )
    if tr_min > 0.0 and (tr_ratio is None or float(tr_ratio) < float(tr_min)):
        return SpotEntryGateResult(
            allow=False,
            reason="graph_entry_tr_ratio",
            gate="BLOCKED_GRAPH_ENTRY_TR_RATIO",
            trace={"policy": "slope_tr_guard", "tr_ratio": tr_ratio, "min": float(tr_min)},
        )
    if entry_dir in ("up", "down"):
        direction = str(entry_dir)
        if slope_min > 0.0:
            if slope_med_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE",
                    trace={"policy": "slope_tr_guard", "slope_med_pct": None, "min_abs": float(slope_min)},
                )
            slope_ok = (direction == "up" and float(slope_med_pct) >= float(slope_min)) or (
                direction == "down" and float(slope_med_pct) <= -float(slope_min)
            )
            if not bool(slope_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_med_pct": float(slope_med_pct),
                        "min_abs": float(slope_min),
                    },
                )
        if slope_vel_min > 0.0:
            if slope_vel_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope_vel_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE_VEL",
                    trace={"policy": "slope_tr_guard", "slope_vel_pct": None, "min_abs": float(slope_vel_min)},
                )
            slope_vel_ok = (direction == "up" and float(slope_vel_pct) >= float(slope_vel_min)) or (
                direction == "down" and float(slope_vel_pct) <= -float(slope_vel_min)
            )
            if not bool(slope_vel_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope_vel",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE_VEL",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_vel_pct": float(slope_vel_pct),
                        "min_abs": float(slope_vel_min),
                    },
                )
        if slope_slow_min > 0.0:
            if slope_med_slow_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE",
                    trace={"policy": "slope_tr_guard", "slope_med_slow_pct": None, "min_abs": float(slope_slow_min)},
                )
            slope_slow_ok = (direction == "up" and float(slope_med_slow_pct) >= float(slope_slow_min)) or (
                direction == "down" and float(slope_med_slow_pct) <= -float(slope_slow_min)
            )
            if not bool(slope_slow_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_med_slow_pct": float(slope_med_slow_pct),
                        "min_abs": float(slope_slow_min),
                    },
                )
        if slope_vel_slow_min > 0.0:
            if slope_vel_slow_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope_vel_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE_VEL",
                    trace={
                        "policy": "slope_tr_guard",
                        "slope_vel_slow_pct": None,
                        "min_abs": float(slope_vel_slow_min),
                    },
                )
            slope_vel_slow_ok = (direction == "up" and float(slope_vel_slow_pct) >= float(slope_vel_slow_min)) or (
                direction == "down" and float(slope_vel_slow_pct) <= -float(slope_vel_slow_min)
            )
            if not bool(slope_vel_slow_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope_vel",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE_VEL",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_vel_slow_pct": float(slope_vel_slow_pct),
                        "min_abs": float(slope_vel_slow_min),
                    },
                )
    return SpotEntryGateResult(
        allow=True,
        reason="entry_allowed",
        gate="TRIGGER_ENTRY",
        trace={
            "policy": "slope_tr_guard",
            "guard_threshold_scale": float(threshold_scale),
            "guard_threshold_trace": dict(threshold_trace),
        },
    )


def _exit_policy_priority(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    open_dir: str | None,
    signal_entry_dir: str | None,
    exit_candidates: Mapping[str, bool] | None,
    exit_priority: Sequence[str] | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
) -> SpotExitReasonResult:
    _ = (
        graph,
        strategy,
        open_dir,
        signal_entry_dir,
        tr_ratio,
        tr_median_pct,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
    )
    reason = pick_exit_reason(exit_candidates, priority=exit_priority)
    return SpotExitReasonResult(reason=reason, trace={"policy": "priority"})


def _exit_policy_slope_flip_guard(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    open_dir: str | None,
    signal_entry_dir: str | None,
    exit_candidates: Mapping[str, bool] | None,
    exit_priority: Sequence[str] | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
) -> SpotExitReasonResult:
    _ = (graph, signal_entry_dir, shock_atr_vel_pct, shock_atr_accel_pct)
    candidates = {str(k): bool(v) for k, v in (exit_candidates or {}).items()}
    flip_suppressed = False
    slope_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slope_min_pct"), default=0.0))
    tr_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_tr_ratio_min"), default=0.0))
    slope_slow_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slow_slope_min_pct"), default=0.0))
    slope_vel_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slope_vel_min_pct"), default=0.0))
    slope_vel_slow_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slow_slope_vel_min_pct"), default=0.0))
    threshold_scale, threshold_trace = spot_guard_threshold_scale(
        strategy=strategy,
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
    )
    slope_min *= float(threshold_scale)
    tr_min *= float(threshold_scale)
    slope_slow_min *= float(threshold_scale)
    slope_vel_min *= float(threshold_scale)
    slope_vel_slow_min *= float(threshold_scale)
    if bool(candidates.get("flip")) and open_dir in ("up", "down"):
        directional_ok = slope_min <= 0.0 or (
            slope_med_pct is not None
            and ((open_dir == "up" and float(slope_med_pct) >= float(slope_min)) or (open_dir == "down" and float(slope_med_pct) <= -float(slope_min)))
        )
        directional_slow_ok = slope_slow_min <= 0.0 or (
            slope_med_slow_pct is not None
            and (
                (open_dir == "up" and float(slope_med_slow_pct) >= float(slope_slow_min))
                or (open_dir == "down" and float(slope_med_slow_pct) <= -float(slope_slow_min))
            )
        )
        vel_ok = slope_vel_min <= 0.0 or (
            slope_vel_pct is not None
            and ((open_dir == "up" and float(slope_vel_pct) >= float(slope_vel_min)) or (open_dir == "down" and float(slope_vel_pct) <= -float(slope_vel_min)))
        )
        vel_slow_ok = slope_vel_slow_min <= 0.0 or (
            slope_vel_slow_pct is not None
            and (
                (open_dir == "up" and float(slope_vel_slow_pct) >= float(slope_vel_slow_min))
                or (open_dir == "down" and float(slope_vel_slow_pct) <= -float(slope_vel_slow_min))
            )
        )
        tr_ok = tr_min <= 0.0 or (tr_ratio is not None and float(tr_ratio) >= float(tr_min))
        if bool(directional_ok) and bool(directional_slow_ok) and bool(vel_ok) and bool(vel_slow_ok) and bool(tr_ok):
            candidates["flip"] = False
            flip_suppressed = True
    reason = pick_exit_reason(candidates, priority=exit_priority)
    return SpotExitReasonResult(
        reason=reason,
        trace={
            "policy": "slope_flip_guard",
            "flip_suppressed": bool(flip_suppressed),
            "guard_threshold_scale": float(threshold_scale),
            "guard_threshold_trace": dict(threshold_trace),
            "slope_med_pct": float(slope_med_pct) if slope_med_pct is not None else None,
            "slope_vel_pct": float(slope_vel_pct) if slope_vel_pct is not None else None,
            "slope_med_slow_pct": float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
            "slope_vel_slow_pct": float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "tr_median_pct": float(tr_median_pct) if tr_median_pct is not None else None,
        },
    )


def _resize_policy_adaptive(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="off",
        default_min_mult=0.5,
        default_max_mult=1.75,
    )
    return SpotResizeTargetResult(target_qty=int(out.target_qty), trace={"policy": "adaptive", **out.trace})


def _resize_policy_adaptive_atr_defensive(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="atr",
        default_min_mult=0.35,
        default_max_mult=1.0,
    )
    return SpotResizeTargetResult(target_qty=int(out.target_qty), trace={"policy": "adaptive_atr_defensive", **out.trace})


def _resize_policy_adaptive_hybrid_aggressive(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="hybrid",
        default_min_mult=0.5,
        default_max_mult=2.25,
    )
    return SpotResizeTargetResult(
        target_qty=int(out.target_qty),
        trace={"policy": "adaptive_hybrid_aggressive", **out.trace},
    )


def _resize_policy_adaptive_slope_probe(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="slope",
        default_min_mult=0.4,
        default_max_mult=1.5,
    )
    return SpotResizeTargetResult(target_qty=int(out.target_qty), trace={"policy": "adaptive_slope_probe", **out.trace})


def _risk_overlay_policy_legacy(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    _ = (
        graph,
        strategy,
        filters,
        action,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
        shock_dir,
    )
    return SpotRiskOverlayAdjustments(trace={"policy": "legacy"})


def _risk_overlay_policy_atr_compress(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    _ = (
        graph,
        action,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
        shock_dir,
    )
    hi_raw = _pick_value(
        strategy=strategy,
        filters=filters,
        keys=("spot_graph_overlay_atr_hi_pct", "spot_overlay_atr_hi_pct"),
    )
    min_raw = _pick_value(
        strategy=strategy,
        filters=filters,
        keys=("spot_graph_overlay_atr_hi_min_mult", "spot_overlay_atr_hi_min_mult"),
    )
    atr_hi = max(0.0, _parse_float(hi_raw, default=2.5))
    floor_mult = max(0.05, _parse_float(min_raw, default=0.5))
    scale = 1.0
    if atr_hi > 0.0 and shock_atr_pct is not None and float(shock_atr_pct) > float(atr_hi):
        scale = float(atr_hi) / float(shock_atr_pct)
        scale = float(max(floor_mult, min(1.0, scale)))
    return SpotRiskOverlayAdjustments(
        risk_mult=float(scale),
        cap_mult=float(scale),
        trace={
            "policy": "atr_compress",
            "shock_atr_pct": float(shock_atr_pct) if shock_atr_pct is not None else None,
            "atr_hi_pct": float(atr_hi),
            "min_mult": float(floor_mult),
            "scale": float(scale),
        },
    )


def _risk_overlay_policy_trend_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    _ = (graph, filters, shock_atr_pct, riskoff, riskpanic, riskpop, shock, shock_dir)
    action_u = str(action or "BUY").strip().upper()
    tr_ref = max(1e-9, _parse_float(_get(strategy, "spot_graph_overlay_tr_ratio_ref"), default=1.0))
    slope_ref = max(1e-9, _parse_float(_get(strategy, "spot_graph_overlay_slope_ref_pct"), default=0.08))
    boost_max = max(1.0, _parse_float(_get(strategy, "spot_graph_overlay_trend_boost_max"), default=1.35))
    floor_mult = max(0.05, _parse_float(_get(strategy, "spot_graph_overlay_trend_floor_mult"), default=0.65))
    atr_vel_ref = max(1e-9, _parse_float(_get(strategy, "spot_graph_overlay_atr_vel_ref_pct"), default=0.40))

    slope = float(slope_med_pct) if slope_med_pct is not None else 0.0
    vel = float(slope_vel_pct) if slope_vel_pct is not None else 0.0
    slope_slow = float(slope_med_slow_pct) if slope_med_slow_pct is not None else 0.0
    vel_slow = float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else 0.0
    atr_vel = float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else 0.0
    atr_accel = float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else 0.0
    tr_bonus = max(0.0, (float(tr_ratio) - float(tr_ref))) if tr_ratio is not None else 0.0

    if action_u == "SELL":
        slope = -float(slope)
        vel = -float(vel)
        slope_slow = -float(slope_slow)
        vel_slow = -float(vel_slow)

    score = (
        (float(slope) / float(slope_ref))
        + 0.5 * (float(vel) / float(slope_ref))
        + 0.35 * (float(slope_slow) / float(slope_ref))
        + 0.25 * (float(vel_slow) / float(slope_ref))
        + 0.20 * (float(atr_vel) / float(atr_vel_ref))
        + 0.10 * (float(atr_accel) / float(atr_vel_ref))
        + float(tr_bonus)
    )
    if score >= 0:
        mult = min(float(boost_max), 1.0 + 0.20 * float(score))
    else:
        mult = max(float(floor_mult), 1.0 + 0.20 * float(score))

    long_mult = float(mult) if action_u == "BUY" else 1.0
    short_mult = float(mult) if action_u == "SELL" else 1.0
    return SpotRiskOverlayAdjustments(
        risk_mult=1.0,
        long_risk_mult=float(long_mult),
        short_risk_mult=float(short_mult),
        cap_mult=1.0,
        trace={
            "policy": "trend_bias",
            "action": str(action_u),
            "score": float(score),
            "mult": float(mult),
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "slope_med_pct": float(slope_med_pct) if slope_med_pct is not None else None,
            "slope_vel_pct": float(slope_vel_pct) if slope_vel_pct is not None else None,
            "slope_med_slow_pct": float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
            "slope_vel_slow_pct": float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
            "shock_atr_vel_pct": float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
            "shock_atr_accel_pct": float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
            "tr_ref": float(tr_ref),
            "slope_ref_pct": float(slope_ref),
        },
    )


def _risk_overlay_policy_shock_dir_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    # Direction-only overlay: bias long/short risk based on the smoothed shock direction
    # (sign of recent returns). This is explicitly not tied to the volatility ramp.
    _ = (
        graph,
        filters,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
    )
    action_u = str(action or "BUY").strip().upper()
    shock_dir_u = str(shock_dir) if shock_dir in ("up", "down") else None

    boost_max = max(1.0, _parse_float(_get(strategy, "spot_graph_overlay_trend_boost_max"), default=1.15))
    floor_mult = max(0.05, _parse_float(_get(strategy, "spot_graph_overlay_trend_floor_mult"), default=0.65))

    align = None
    if shock_dir_u is not None:
        if action_u == "BUY":
            align = shock_dir_u == "up"
        elif action_u == "SELL":
            align = shock_dir_u == "down"

    if align is None:
        mult = 1.0
        reason = "dir_missing"
    elif align:
        mult = float(boost_max)
        reason = "align"
    else:
        mult = float(floor_mult)
        reason = "misalign"

    long_mult = float(mult) if action_u == "BUY" else 1.0
    short_mult = float(mult) if action_u == "SELL" else 1.0
    return SpotRiskOverlayAdjustments(
        risk_mult=1.0,
        long_risk_mult=float(long_mult),
        short_risk_mult=float(short_mult),
        cap_mult=1.0,
        trace={
            "policy": "shock_dir_bias",
            "action": str(action_u),
            "shock_dir": shock_dir_u,
            "align": align,
            "reason": str(reason),
            "mult": float(mult),
            "floor_mult": float(floor_mult),
            "boost_max": float(boost_max),
        },
    )


def _risk_overlay_policy_atr_compress_trend_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    # Keep ATR compression (crash hygiene) while also biasing long/short risk
    # based on signed trend strength (drift-down detection, chop defense).
    atr = _risk_overlay_policy_atr_compress(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    trend = _risk_overlay_policy_trend_bias(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    return SpotRiskOverlayAdjustments(
        risk_mult=float(atr.risk_mult) * float(trend.risk_mult),
        cap_mult=float(atr.cap_mult) * float(trend.cap_mult),
        long_risk_mult=float(trend.long_risk_mult),
        short_risk_mult=float(trend.short_risk_mult),
        trace={
            "policy": "atr_compress_trend_bias",
            "atr_compress": dict(atr.trace or {}),
            "trend_bias": dict(trend.trace or {}),
        },
    )


def _risk_overlay_policy_atr_compress_shock_dir_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    atr = _risk_overlay_policy_atr_compress(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    bias = _risk_overlay_policy_shock_dir_bias(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    return SpotRiskOverlayAdjustments(
        risk_mult=float(atr.risk_mult) * float(bias.risk_mult),
        cap_mult=float(atr.cap_mult) * float(bias.cap_mult),
        long_risk_mult=float(bias.long_risk_mult),
        short_risk_mult=float(bias.short_risk_mult),
        trace={
            "policy": "atr_compress_shock_dir_bias",
            "atr_compress": dict(atr.trace or {}),
            "shock_dir_bias": dict(bias.trace or {}),
        },
    )


class SpotPolicyGraph:
    def __init__(
        self,
        *,
        profile_name: str,
        entry_policy: str,
        exit_policy: str,
        resize_policy: str,
        risk_overlay_policy: str,
    ) -> None:
        self.profile_name = str(profile_name)
        self.entry_policy = str(entry_policy)
        self.exit_policy = str(exit_policy)
        self.resize_policy = str(resize_policy)
        self.risk_overlay_policy = str(risk_overlay_policy)

    @classmethod
    def from_sources(
        cls,
        *,
        strategy: Mapping[str, object] | object | None,
        filters: Mapping[str, object] | object | None,
    ) -> SpotPolicyGraph:
        profile_name = _resolve_graph_profile_name(strategy=strategy, filters=filters)
        profile = _GRAPH_PROFILES.get(str(profile_name), _GRAPH_PROFILES["neutral"])
        entry_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_entry_policy", "spot_graph_entry_policy"),
            default=str(profile.entry_policy),
        )
        exit_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_exit_policy", "spot_graph_exit_policy"),
            default=str(profile.exit_policy),
        )
        resize_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_resize_policy", "spot_graph_resize_policy"),
            default=str(profile.resize_policy),
        )
        risk_overlay_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_risk_overlay_policy", "spot_graph_risk_overlay_policy"),
            default=str(profile.risk_overlay_policy),
        )
        return cls(
            profile_name=str(profile.name),
            entry_policy=str(entry_policy),
            exit_policy=str(exit_policy),
            resize_policy=str(resize_policy),
            risk_overlay_policy=str(risk_overlay_policy),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "profile": str(self.profile_name),
            "entry_policy": str(self.entry_policy),
            "exit_policy": str(self.exit_policy),
            "resize_policy": str(self.resize_policy),
            "risk_overlay_policy": str(self.risk_overlay_policy),
        }

    def evaluate_entry_gate(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        bar_ts: datetime,
        entry_dir: str | None,
        shock_atr_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
        tr_ratio: float | None = None,
        tr_median_pct: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
    ) -> SpotEntryGateResult:
        fn = _ENTRY_POLICY_REGISTRY.get(str(self.entry_policy), _entry_policy_default)
        out = fn(
            self,
            strategy=strategy,
            bar_ts=bar_ts,
            entry_dir=entry_dir,
            shock_atr_pct=shock_atr_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
            tr_ratio=tr_ratio,
            tr_median_pct=tr_median_pct,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
        )
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotEntryGateResult(allow=bool(out.allow), reason=str(out.reason), gate=str(out.gate), trace=trace)

    def resolve_exit_reason(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        open_dir: str | None,
        signal_entry_dir: str | None,
        exit_candidates: Mapping[str, bool] | None,
        exit_priority: Sequence[str] | None = None,
        tr_ratio: float | None = None,
        tr_median_pct: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
    ) -> SpotExitReasonResult:
        fn = _EXIT_POLICY_REGISTRY.get(str(self.exit_policy), _exit_policy_priority)
        out = fn(
            self,
            strategy=strategy,
            open_dir=open_dir,
            signal_entry_dir=signal_entry_dir,
            exit_candidates=exit_candidates,
            exit_priority=exit_priority,
            tr_ratio=tr_ratio,
            tr_median_pct=tr_median_pct,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
        )
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotExitReasonResult(reason=out.reason, trace=trace)

    def resolve_resize_target(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        current_qty: int,
        base_target_qty: int,
        shock_atr_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
        tr_ratio: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
    ) -> SpotResizeTargetResult:
        fn = _RESIZE_POLICY_REGISTRY.get(str(self.resize_policy), _resize_policy_adaptive)
        out = fn(
            self,
            strategy=strategy,
            current_qty=int(current_qty),
            base_target_qty=int(base_target_qty),
            shock_atr_pct=shock_atr_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
            tr_ratio=tr_ratio,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
        )
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotResizeTargetResult(target_qty=int(out.target_qty), trace=trace)

    def resolve_risk_overlay_adjustments(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        filters: Mapping[str, object] | object | None,
        action: str,
        shock_atr_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
        tr_ratio: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
        riskoff: bool = False,
        riskpanic: bool = False,
        riskpop: bool = False,
        shock: bool = False,
        shock_dir: str | None = None,
    ) -> SpotRiskOverlayAdjustments:
        fn = _RISK_OVERLAY_POLICY_REGISTRY.get(str(self.risk_overlay_policy), _risk_overlay_policy_legacy)
        out = fn(
            self,
            strategy=strategy,
            filters=filters,
            action=str(action),
            shock_atr_pct=shock_atr_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
            tr_ratio=tr_ratio,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            shock=bool(shock),
            shock_dir=shock_dir,
        )
        risk_mult = max(0.0, float(out.risk_mult))
        long_mult = max(0.0, float(out.long_risk_mult))
        short_mult = max(0.0, float(out.short_risk_mult))
        cap_mult = max(0.0, float(out.cap_mult))
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotRiskOverlayAdjustments(
            risk_mult=float(risk_mult),
            long_risk_mult=float(long_mult),
            short_risk_mult=float(short_mult),
            cap_mult=float(cap_mult),
            trace=trace,
        )


_ENTRY_POLICY_REGISTRY: dict[str, Callable[..., SpotEntryGateResult]] = {
    "default": _entry_policy_default,
    "slope_tr_guard": _entry_policy_slope_tr_guard,
}

_EXIT_POLICY_REGISTRY: dict[str, Callable[..., SpotExitReasonResult]] = {
    "priority": _exit_policy_priority,
    "slope_flip_guard": _exit_policy_slope_flip_guard,
}

_RESIZE_POLICY_REGISTRY: dict[str, Callable[..., SpotResizeTargetResult]] = {
    "adaptive": _resize_policy_adaptive,
    "adaptive_atr_defensive": _resize_policy_adaptive_atr_defensive,
    "adaptive_hybrid_aggressive": _resize_policy_adaptive_hybrid_aggressive,
    "adaptive_slope_probe": _resize_policy_adaptive_slope_probe,
}

_RISK_OVERLAY_POLICY_REGISTRY: dict[str, Callable[..., SpotRiskOverlayAdjustments]] = {
    "legacy": _risk_overlay_policy_legacy,
    "atr_compress": _risk_overlay_policy_atr_compress,
    "atr_compress_shock_dir_bias": _risk_overlay_policy_atr_compress_shock_dir_bias,
    "atr_compress_trend_bias": _risk_overlay_policy_atr_compress_trend_bias,
    "shock_dir_bias": _risk_overlay_policy_shock_dir_bias,
    "trend_bias": _risk_overlay_policy_trend_bias,
}
