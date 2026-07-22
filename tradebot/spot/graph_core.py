"""Spot policy-graph contracts, profiles, and adaptive scaling primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from .policy_contract import parse_float as _parse_float
from .policy_contract import parse_int as _parse_int
from .policy_contract import source_value as _get


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


def _has(source: Mapping[str, object] | object | None, key: str) -> bool:
    if source is None:
        return False
    if isinstance(source, Mapping):
        return key in source
    return hasattr(source, key)


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
