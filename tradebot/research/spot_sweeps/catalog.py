"""Canonical spot-sweep axes, coverage tiers, and CLI catalog."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ...time_utils import now_et as _now_et
from .dimensions import _AXIS_DIMENSION_REGISTRY


@dataclass(frozen=True)
class AxisSurfaceSpec:
    name: str
    include_in_axis_all: bool = False
    help_text: str = "See source."
    sharding_class: str = "none"
    coverage_note: str = ""
    parallel_profile_axis_all: str = "single"
    dimensional_cost_source: str = ""
    total_hint_mode: str = ""
    total_hint_static: int | None = None
    total_hint_dims: tuple[str, ...] = ()


_AXIS_SURFACE_SPECS: tuple[AxisSurfaceSpec, ...] = (
    AxisSurfaceSpec("ema", True, "EMA preset timing sweep (direction speed).", total_hint_static=6),
    AxisSurfaceSpec("entry_mode", False, "Entry semantics sweep (cross vs trend) with directional rules.", total_hint_static=6),
    AxisSurfaceSpec(
        "combo_full",
        False,
        "Unified tight Cartesian sweep over centralized combo dimensions.",
        sharding_class="stage",
        coverage_note="unified Cartesian core (mixed-radix shard ranges)",
        dimensional_cost_source="combo_full_cartesian_tight",
    ),
    AxisSurfaceSpec("volume", True, "Volume gate sweep (ratio threshold x EMA period).", total_hint_static=13),
    AxisSurfaceSpec("rv", True, "Realized volatility gate sweep.", total_hint_static=30),
    AxisSurfaceSpec("tod", True, "Time-of-day gate sweep (ET entry windows).", total_hint_static=29),
    AxisSurfaceSpec("weekday", False, "Weekday entry gating sweep."),
    AxisSurfaceSpec("exit_time", False, "Fixed ET flatten-time sweep.", total_hint_static=7),
    AxisSurfaceSpec(
        "atr",
        True,
        "Core ATR exits sweep.",
        total_hint_mode="atr_profile",
    ),
    AxisSurfaceSpec(
        "atr_fine",
        False,
        "Fine ATR PT/SL pocket sweep.",
        total_hint_mode="atr_profile",
    ),
    AxisSurfaceSpec(
        "atr_ultra",
        False,
        "Ultra-fine ATR PT/SL micro-grid.",
        total_hint_mode="atr_profile",
    ),
    AxisSurfaceSpec("chop_joint", False, "Chop-killer joint sweep (slope x cooldown x skip-open)."),
    AxisSurfaceSpec(
        "ptsl",
        True,
        "Fixed-percent PT/SL exits sweep with flip/close_eod semantics (non-ATR).",
    ),
    AxisSurfaceSpec("hold", True, "Flip-exit minimum-hold-bars sweep.", total_hint_static=7),
    AxisSurfaceSpec(
        "spot_short_risk_mult",
        True,
        "Short-side risk multiplier sweep.",
        total_hint_static=13,
    ),
    AxisSurfaceSpec("orb", True, "ORB sweep (open-time, window, target semantics)."),
    AxisSurfaceSpec("orb_joint", False, "ORB x regime x TICK joint sweep."),
    AxisSurfaceSpec(
        "frontier",
        False,
        "Frontier sweep over shortlist dimensions.",
    ),
    AxisSurfaceSpec(
        "regime",
        True,
        "Primary Supertrend regime params/timeframe sweep.",
        total_hint_mode="regime_profile",
    ),
    AxisSurfaceSpec(
        "regime2",
        True,
        "Secondary (regime2) Supertrend params/timeframe sweep.",
        total_hint_mode="regime2_profile",
    ),
    AxisSurfaceSpec("regime2_ema", False, "Regime2 EMA confirm sweep.", total_hint_static=13),
    AxisSurfaceSpec("joint", True, "Targeted regime x regime2 interaction hunt."),
    AxisSurfaceSpec("micro_st", False, "Micro sweep around current ST/ST2 neighborhood."),
    AxisSurfaceSpec(
        "flip_exit",
        True,
        "Flip-exit semantics/gating sweep.",
        total_hint_static=48,
    ),
    AxisSurfaceSpec("confirm", True, "Entry confirmation bars sweep.", total_hint_static=4),
    AxisSurfaceSpec(
        "spread",
        True,
        "EMA spread quality gate sweep.",
        total_hint_mode="spread_profile",
    ),
    AxisSurfaceSpec(
        "spread_fine",
        False,
        "Fine EMA spread threshold sweep.",
        total_hint_mode="spread_profile",
    ),
    AxisSurfaceSpec(
        "spread_down",
        False,
        "Directional down-side EMA spread gate sweep.",
        total_hint_mode="spread_profile",
    ),
    AxisSurfaceSpec("slope", True, "EMA slope quality gate sweep.", total_hint_static=6),
    AxisSurfaceSpec("slope_signed", True, "Signed directional EMA slope gate sweep."),
    AxisSurfaceSpec("cooldown", True, "Entry cooldown bars sweep.", total_hint_static=7),
    AxisSurfaceSpec("skip_open", True, "Skip-first-bars-after-open sweep.", total_hint_static=6),
    AxisSurfaceSpec(
        "shock",
        True,
        "Shock detector/mode/threshold sweep (+ monetization and throttle pocket).",
        dimensional_cost_source="shock",
        total_hint_mode="shock_profile",
    ),
    AxisSurfaceSpec("loosen", True, "Loosenings sweep (single-position parity + EOD behavior)."),
    AxisSurfaceSpec("tick", True, "Raschke-style $TICK width gate sweep."),
)
_AXIS_CHOICES = tuple(spec.name for spec in _AXIS_SURFACE_SPECS)
_AXIS_ALL_PLAN = tuple(spec.name for spec in _AXIS_SURFACE_SPECS if bool(spec.include_in_axis_all))
_AXIS_INCLUDE_IN_AXIS_ALL_DEFAULTS = frozenset(_AXIS_ALL_PLAN)
_COMBO_FULL_CARTESIAN_DIM_ORDER: tuple[str, ...] = (
    "timing_profile",
    "direction",
    "confirm",
    "perm",
    "tod",
    "vol",
    "cadence",
    "regime",
    "regime2",
    "exit",
    "tick",
    "shock",
    "slope",
    "risk",
    "short_mult",
)
_COMBO_FULL_PAIR_DIM_VARIANT_SPECS: tuple[tuple[str, str], ...] = (
    ("direction", "direction_variants"),
    ("perm", "perm_variants"),
    ("tod", "tod_variants"),
    ("vol", "vol_variants"),
    ("cadence", "cadence_variants"),
    ("regime", "regime_variants"),
    ("regime2", "regime2_variants"),
    ("exit", "exit_variants"),
    ("tick", "tick_variants"),
    ("shock", "shock_variants"),
    ("slope", "slope_variants"),
    ("risk", "risk_variants"),
)
_COMBO_FULL_NOTE_PAIR_DIM_ORDER: tuple[str, ...] = (
    "perm",
    "tod",
    "vol",
    "cadence",
    "regime",
    "regime2",
    "exit",
    "tick",
    "shock",
    "slope",
    "risk",
)
_COMBO_FULL_COVERAGE_TIER_REGISTRY: dict[str, dict[str, object]] = {
    "full": {"freeze_dims": ()},
    "profile": {
        "freeze_dims": (
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
        "customizer": "hf_timing_sniper",
    },
    "gate": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "ema": {
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "tick": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "regime": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime2",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "risk": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "short_mult",
        ),
        "customizer": "risk_overlays",
    },
}
_COMBO_FULL_PRESET_ALIAS_REGISTRY: dict[str, dict[str, object]] = {
    "squeeze": {
        "tier": "regime",
        "customizer": "squeeze",
        "freeze_dims": (
            "timing_profile",
            "direction",
            "perm",
            "cadence",
            "regime",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "tod_interaction": {
        "tier": "gate",
        "customizer": "tod_interaction",
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "vol",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "perm_joint": {"tier": "gate"},
    "ema_perm_joint": {"tier": "ema"},
    "tick_perm_joint": {"tier": "tick"},
    "regime_atr": {"tier": "regime"},
    "ema_regime": {
        "tier": "ema",
        "customizer": "ema_regime",
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "tick_ema": {
        "tier": "tick",
        "customizer": "tick_ema",
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "ema_atr": {
        "tier": "ema",
        "customizer": "ema_atr",
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "r2_atr": {
        "tier": "regime",
        "customizer": "r2_atr",
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "r2_tod": {
        "tier": "regime",
        "customizer": "r2_tod",
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "vol",
            "cadence",
            "regime",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "loosen_atr": {
        "tier": "risk",
        "customizer": "loosen_atr",
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "risk_overlays": {"tier": "risk", "customizer": "risk_overlays"},
    "gate_matrix": {
        "tier": "gate",
        "customizer": "gate_matrix",
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "regime",
            "exit",
            "slope",
            "vol",
            "cadence",
        ),
    },
    "lf_shock_sniper": {
        "tier": "risk",
        "customizer": "lf_shock_sniper",
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "slope",
            "risk",
            "short_mult",
        ),
    },
    "hf_timing_sniper": {
        "tier": "profile",
        "customizer": "hf_timing_sniper",
    },
}
_COMBO_FULL_PRESET_TIER_NAMES: tuple[str, ...] = tuple(_COMBO_FULL_COVERAGE_TIER_REGISTRY.keys())
_COMBO_FULL_PRESET_ALIAS_NAMES: tuple[str, ...] = tuple(_COMBO_FULL_PRESET_ALIAS_REGISTRY.keys())
_COMBO_FULL_PRESET_AXES: tuple[str, ...] = tuple(_COMBO_FULL_PRESET_TIER_NAMES + _COMBO_FULL_PRESET_ALIAS_NAMES)
_AXIS_HANDLER_NAME_OVERRIDES: dict[str, str] = {
    "weekday": "_sweep_weekdays",
    "atr": "_sweep_atr_exits",
    "atr_fine": "_sweep_atr_exits_fine",
    "atr_ultra": "_sweep_atr_exits_ultra",
}
_UNKNOWN_AXIS_HANDLER_OVERRIDE_KEYS = sorted(set(_AXIS_HANDLER_NAME_OVERRIDES) - set(_AXIS_CHOICES))
if _UNKNOWN_AXIS_HANDLER_OVERRIDE_KEYS:
    raise ValueError(f"Unknown axis handler override keys: {_UNKNOWN_AXIS_HANDLER_OVERRIDE_KEYS}")


def _dedupe_axis_names(names) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        key = str(raw).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _axis_handler_name(axis_name: str) -> str:
    key = str(axis_name).strip().lower()
    if not key:
        raise ValueError("axis_name must be non-empty")
    override = _AXIS_HANDLER_NAME_OVERRIDES.get(key)
    if override:
        return str(override)
    return f"_sweep_{key}"


def build_axis_registry(owner: object) -> dict[str, object]:
    registry: dict[str, object] = {}
    missing: list[str] = []
    for axis_name in _AXIS_CHOICES:
        handler_name = _axis_handler_name(str(axis_name))
        fn_obj = getattr(owner, handler_name, None)
        if callable(fn_obj):
            registry[str(axis_name)] = fn_obj
        else:
            missing.append(f"{axis_name}->{handler_name}")
    if missing:
        raise RuntimeError(f"Axis handlers missing: {', '.join(missing)}")
    return registry


def _axis_mode_plan(*, mode: str) -> tuple[tuple[str, str, bool], ...]:
    mode_key = str(mode).strip().lower()
    if mode_key != "axis_all":
        raise ValueError(f"Unknown axis mode: {mode!r}")

    def _profile_for_axis(axis_name: str) -> str:
        spec_map = globals().get("_AXIS_EXECUTION_SPEC_BY_NAME")
        if not isinstance(spec_map, dict):
            return "single"
        spec = spec_map.get(str(axis_name))
        if spec is None:
            return "single"
        return str(getattr(spec, "parallel_profile_axis_all", "single") or "single")

    spec_map = globals().get("_AXIS_EXECUTION_SPEC_BY_NAME")
    if not isinstance(spec_map, dict):
        return tuple((axis_name, "single", True) for axis_name in _AXIS_ALL_PLAN)
    out: list[tuple[str, str, bool]] = []
    for axis_name in _AXIS_CHOICES:
        spec = spec_map.get(str(axis_name))
        if not isinstance(spec, AxisExecutionSpec) or not bool(spec.include_in_axis_all):
            continue
        out.append((str(axis_name), _profile_for_axis(str(axis_name)), True))
    return tuple(out)


# region CLI
class _SpotSweepsHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if action.help is argparse.SUPPRESS:
            return help_text
        if "%(default)" in help_text:
            return help_text
        if action.default is argparse.SUPPRESS or action.default is None:
            return help_text
        return f"{help_text} (default: %(default)s)"


_BASE_HELP_NOTES: dict[str, str] = {
    "default": "Minimal baseline profile (EMA 2/4, fixed pct exits, no extra gates).",
    "champion": "Loads the promoted HF/LF crown matching symbol/bar/session, ranked by pnl/dd.",
    "champion_pnl": "Loads the promoted HF/LF crown matching symbol/bar/session, ranked by pnl.",
    "dual_regime": "Regime + regime2 baseline profile used for dual-regime refinement flows.",
}


@dataclass(frozen=True)
class AxisExecutionSpec:
    include_in_axis_all: bool
    help_text: str
    sharding_class: str = "none"
    coverage_note: str = ""
    parallel_profile_axis_all: str = "single"
    dimensional_cost_priors: tuple[tuple[str, float], ...] = ()
    total_hint_mode: str = ""
    total_hint_static: int | None = None
    total_hint_dims: tuple[str, ...] = ()


_AXIS_SURFACE_SPEC_BY_NAME: dict[str, AxisSurfaceSpec] = {}
_AXIS_SURFACE_SPEC_DUPLICATES: list[str] = []
for _spec in _AXIS_SURFACE_SPECS:
    _key = str(_spec.name).strip()
    if not _key:
        raise ValueError("AxisSurfaceSpec name must be non-empty.")
    if _key in _AXIS_SURFACE_SPEC_BY_NAME:
        _AXIS_SURFACE_SPEC_DUPLICATES.append(_key)
        continue
    _AXIS_SURFACE_SPEC_BY_NAME[_key] = _spec
if _AXIS_SURFACE_SPEC_DUPLICATES:
    raise ValueError(f"Duplicate AxisSurfaceSpec keys: {sorted(set(_AXIS_SURFACE_SPEC_DUPLICATES))}")

_AXIS_SURFACE_SPEC_MISSING_KEYS = sorted(set(_AXIS_CHOICES) - set(_AXIS_SURFACE_SPEC_BY_NAME))
if _AXIS_SURFACE_SPEC_MISSING_KEYS:
    raise ValueError(f"Missing AxisSurfaceSpec keys: {_AXIS_SURFACE_SPEC_MISSING_KEYS}")


def _build_axis_surface_maps() -> tuple[dict[str, str], dict[str, int], dict[str, str], dict[str, tuple[str, ...]]]:
    help_map: dict[str, str] = {}
    hint_static: dict[str, int] = {}
    hint_mode: dict[str, str] = {}
    hint_dims: dict[str, tuple[str, ...]] = {}
    for spec in _AXIS_SURFACE_SPECS:
        key = str(spec.name).strip()
        if not key:
            continue
        help_map[key] = str(spec.help_text or "See source.")
        if spec.total_hint_static is not None:
            try:
                hint_static[key] = int(spec.total_hint_static)
            except (TypeError, ValueError):
                pass
        mode = str(spec.total_hint_mode or "").strip()
        if mode:
            hint_mode[key] = str(mode)
        dims = tuple(str(dim).strip() for dim in tuple(spec.total_hint_dims or ()) if str(dim).strip())
        if dims:
            hint_dims[key] = tuple(dims)
    return help_map, hint_static, hint_mode, hint_dims


(
    _AXIS_HELP_TEXT_BY_NAME,
    _AXIS_TOTAL_HINT_STATIC_BY_NAME,
    _AXIS_TOTAL_HINT_MODE_BY_NAME,
    _AXIS_TOTAL_HINT_DIMS_BY_NAME,
) = _build_axis_surface_maps()


def _axis_dimensional_cost_priors(axis_name: str, *, source: str | None = None) -> tuple[tuple[str, float], ...]:
    source_key = str(source or axis_name).strip()
    if not source_key:
        return ()
    dims = _AXIS_DIMENSION_REGISTRY.get(source_key, {})
    if not isinstance(dims, dict):
        return ()
    priors_raw = dims.get("cost_hints")
    if not isinstance(priors_raw, dict):
        return ()
    priors: list[tuple[str, float]] = []
    for key, val in priors_raw.items():
        key_s = str(key).strip()
        if not key_s:
            continue
        try:
            priors.append((key_s, float(val)))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(priors, key=lambda row: row[0]))


def _build_axis_execution_specs() -> dict[str, AxisExecutionSpec]:
    out: dict[str, AxisExecutionSpec] = {}
    for axis_name in _AXIS_CHOICES:
        axis_key = str(axis_name)
        surface_spec = _AXIS_SURFACE_SPEC_BY_NAME.get(str(axis_key))
        in_all = bool(surface_spec.include_in_axis_all) if isinstance(surface_spec, AxisSurfaceSpec) else (str(axis_key) in _AXIS_INCLUDE_IN_AXIS_ALL_DEFAULTS)
        dim_source = str(surface_spec.dimensional_cost_source).strip() if isinstance(surface_spec, AxisSurfaceSpec) and str(surface_spec.dimensional_cost_source).strip() else str(axis_name)
        hint_dims = tuple(
            str(dim).strip()
            for dim in (
                tuple(surface_spec.total_hint_dims)
                if isinstance(surface_spec, AxisSurfaceSpec) and isinstance(surface_spec.total_hint_dims, tuple)
                else tuple(_AXIS_TOTAL_HINT_DIMS_BY_NAME.get(str(axis_name), ()))
            )
            if str(dim).strip()
        )
        hint_mode = (
            str(surface_spec.total_hint_mode or "").strip()
            if isinstance(surface_spec, AxisSurfaceSpec) and str(surface_spec.total_hint_mode or "").strip()
            else str(_AXIS_TOTAL_HINT_MODE_BY_NAME.get(str(axis_name), "")).strip()
        )
        hint_static = (
            int(surface_spec.total_hint_static)
            if isinstance(surface_spec, AxisSurfaceSpec) and isinstance(surface_spec.total_hint_static, int)
            else (int(_AXIS_TOTAL_HINT_STATIC_BY_NAME.get(str(axis_name))) if isinstance(_AXIS_TOTAL_HINT_STATIC_BY_NAME.get(str(axis_name)), int) else None)
        )
        out[str(axis_name)] = AxisExecutionSpec(
            include_in_axis_all=bool(in_all),
            help_text=(str(surface_spec.help_text or "See source.") if isinstance(surface_spec, AxisSurfaceSpec) else str(_AXIS_HELP_TEXT_BY_NAME.get(str(axis_name)) or "See source.")),
            sharding_class=(str(surface_spec.sharding_class or "none") if isinstance(surface_spec, AxisSurfaceSpec) else "none"),
            coverage_note=(str(surface_spec.coverage_note or "") if isinstance(surface_spec, AxisSurfaceSpec) else ""),
            parallel_profile_axis_all=(str(surface_spec.parallel_profile_axis_all or "single") if isinstance(surface_spec, AxisSurfaceSpec) else "single"),
            dimensional_cost_priors=_axis_dimensional_cost_priors(str(axis_name), source=dim_source),
            total_hint_mode=str(hint_mode),
            total_hint_static=(int(hint_static) if isinstance(hint_static, int) else None),
            total_hint_dims=tuple(hint_dims),
        )
    return out


_AXIS_EXECUTION_SPEC_BY_NAME: dict[str, AxisExecutionSpec] = _build_axis_execution_specs()


def _combo_full_preset_key(name: str) -> str:
    key = str(name or "").strip().lower()
    if key == "none":
        return ""
    return str(key)


def _combo_full_preset_axes(
    *,
    include_tiers: bool = True,
    include_aliases: bool = True,
) -> tuple[str, ...]:
    out: list[str] = []
    if bool(include_tiers):
        out.extend(str(name) for name in _COMBO_FULL_PRESET_TIER_NAMES)
    if bool(include_aliases):
        out.extend(str(name) for name in _COMBO_FULL_PRESET_ALIAS_NAMES)
    return tuple(out)


def _combo_full_preset_spec(name: str) -> dict[str, object]:
    key = _combo_full_preset_key(str(name))
    if not key:
        return {}
    if key in _COMBO_FULL_COVERAGE_TIER_REGISTRY:
        tier_key = str(key)
        tier_spec = _COMBO_FULL_COVERAGE_TIER_REGISTRY.get(tier_key) or {}
        raw_dims = tier_spec.get("freeze_dims")
        freeze_dims = tuple(str(dim).strip() for dim in tuple(raw_dims or ()) if str(dim).strip())
        customizer = str(tier_spec.get("customizer") or "").strip().lower()
        return {
            "name": str(tier_key),
            "tier": str(tier_key),
            "freeze_dims": tuple(freeze_dims),
            "customizer": str(customizer),
            "is_alias": False,
        }
    alias_spec = _COMBO_FULL_PRESET_ALIAS_REGISTRY.get(str(key))
    if not isinstance(alias_spec, dict):
        return {}
    tier_key = str(alias_spec.get("tier") or "").strip().lower()
    tier_spec = _COMBO_FULL_COVERAGE_TIER_REGISTRY.get(str(tier_key)) or {}
    raw_dims = alias_spec.get("freeze_dims")
    if not isinstance(raw_dims, (tuple, list)):
        raw_dims = tier_spec.get("freeze_dims")
    freeze_dims = tuple(str(dim).strip() for dim in tuple(raw_dims or ()) if str(dim).strip())
    customizer = str(alias_spec.get("customizer") or tier_spec.get("customizer") or "").strip().lower()
    return {
        "name": str(key),
        "tier": str(tier_key or "custom"),
        "freeze_dims": tuple(freeze_dims),
        "customizer": str(customizer),
        "is_alias": True,
    }


def _combo_full_preset_tier(name: str) -> str:
    spec = _combo_full_preset_spec(str(name))
    tier = str(spec.get("tier") or "").strip().lower()
    return tier or "custom"


def _combo_full_preset_customizer(name: str) -> str:
    spec = _combo_full_preset_spec(str(name))
    return str(spec.get("customizer") or "").strip().lower()


def _combo_full_preset_freeze_dims(name: str) -> tuple[str, ...]:
    spec = _combo_full_preset_spec(str(name))
    raw = spec.get("freeze_dims")
    if not isinstance(raw, (tuple, list)):
        return ()
    return tuple(str(dim).strip() for dim in tuple(raw) if str(dim).strip())


def _combo_full_preset_choices(
    *,
    include_empty: bool = True,
    include_none: bool = True,
    include_aliases: bool = True,
) -> tuple[str, ...]:
    out: list[str] = []
    if bool(include_empty):
        out.append("")
    if bool(include_none):
        out.append("none")
    out.extend(str(axis_name) for axis_name in _combo_full_preset_axes(include_tiers=True, include_aliases=bool(include_aliases)))
    return tuple(out)


def _axis_catalog_lines() -> list[str]:
    lines: list[str] = [
        "Axis Catalog:",
        "  all                        Serial axis_all plan (or parallel when --jobs > 1 with --offline).",
    ]
    for axis_name in _AXIS_CHOICES:
        spec = _AXIS_EXECUTION_SPEC_BY_NAME.get(str(axis_name))
        axis_desc = str(spec.help_text if isinstance(spec, AxisExecutionSpec) else "See source.")
        lines.append(f"  {axis_name:<26} {axis_desc}")
    return lines


def _combo_full_preset_catalog_lines() -> list[str]:
    lines: list[str] = ["combo_full Coverage Tiers:"]
    for tier in _combo_full_preset_axes(include_tiers=True, include_aliases=False):
        freeze_dims = _combo_full_preset_freeze_dims(str(tier))
        alias_count = sum(1 for alias in _combo_full_preset_axes(include_tiers=False, include_aliases=True) if _combo_full_preset_tier(str(alias)) == str(tier))
        lines.append(f"  {tier:<10} frozen_dims={len(freeze_dims):<3} aliases={int(alias_count)}")
    aliases = _combo_full_preset_axes(include_tiers=False, include_aliases=True)
    if aliases:
        lines.append(f"  preset_aliases(hidden): {', '.join(aliases)}")
    return lines


def _axis_coverage_row(axis_name: str) -> tuple[str, str, str]:
    axis_key = str(axis_name).strip().lower()
    spec = _AXIS_EXECUTION_SPEC_BY_NAME.get(axis_key)
    sharded = "yes" if isinstance(spec, AxisExecutionSpec) and str(spec.sharding_class) != "none" else "no"
    cached = "yes"
    notes = str(spec.coverage_note if isinstance(spec, AxisExecutionSpec) else "")
    return sharded, cached, notes


def _spot_sweep_coverage_map_markdown(*, generated_on: date | None = None) -> str:
    generated = generated_on if generated_on is not None else _now_et().date()
    lines: list[str] = [
        "# Spot Sweep Coverage Map",
        "",
        f"Generated: {generated.isoformat()}",
        "",
        "Legend: `sharded` = stage-level worker sharding inside the axis; `cached` = run_cfg+context cache applies.",
        "",
        "| axis | sharded | cached | notes |",
        "|---|---|---|---|",
    ]
    for axis_name in _AXIS_CHOICES:
        sharded, cached, notes = _axis_coverage_row(axis_name)
        lines.append(f"| `{axis_name}` | {sharded} | {cached} | {notes} |")
    lines.extend(
        (
            "",
            "## Notes",
            "- `combo_full` and `--axis all` additionally support axis-level subprocess orchestration.",
            "- Persistent cross-process run_cfg cache is enabled via sqlite (`spot_sweeps_run_cfg_cache.sqlite3`).",
            "- Engine labels describe canonical summary execution; all axes share one lifecycle implementation.",
        )
    )
    return "\n".join(lines) + "\n"


def _write_spot_sweep_coverage_map(path: Path, *, generated_on: date | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_spot_sweep_coverage_map_markdown(generated_on=generated_on))


_INTERNAL_FLAG_HELP_LINES: tuple[str, ...] = (
    "--combo-full-cartesian-stage VALUE: combo_full Cartesian worker stage marker.",
    "--combo-full-cartesian-worker INT / --combo-full-cartesian-workers INT / --combo-full-cartesian-out PATH: combo_full Cartesian shard controls.",
    "--combo-full-cartesian-run-min-trades INT: stage-specific min trades override for combo_full Cartesian.",
    f"--combo-full-preset {{{','.join(_combo_full_preset_choices(include_empty=False, include_none=True, include_aliases=False))}}}: optional combo_full range preset selector.",
)


def _spot_sweeps_help_epilog() -> str:
    axis_lines = _axis_catalog_lines()

    base_lines = ["Base Profiles:"]
    for key in ("default", "champion", "champion_pnl", "dual_regime"):
        base_lines.append(f"  {key:<14} {_BASE_HELP_NOTES.get(key, '')}")

    internal_lines = ["Internal Orchestration Flags (auto-managed by parent stages):"]
    for raw in _INTERNAL_FLAG_HELP_LINES:
        internal_lines.append(f"  {raw}")

    return "\n".join(
        (
            "Examples:",
            "  python -m tradebot.backtest spot --offline --axis all --start 2025-01-08 --end 2026-01-08",
            "  python -m tradebot.backtest spot --axis regime --symbol MNQ --bar-size '1 hour'",
            "  python -m tradebot.backtest spot --axis combo_full --offline --jobs 8",
            "  python -m tradebot.backtest spot --axis combo_full --combo-full-preset profile --offline",
            "",
            "Execution Notes:",
            "  - --start/--end are inclusive dates (YYYY-MM-DD).",
            "  - --bar-size is signal timeframe; ORB sweeps always use 15m signal bars internally.",
            "  - --spot-exec-bar-size overrides only execution bars; signals still use --bar-size.",
            "  - --jobs <= 0 (or omitted) means auto CPU count; effective jobs are clamped to available CPUs.",
            "  - Parallel worker modes require --offline to avoid concurrent IBKR sessions.",
            "  - --cache-dir stores bar cache plus run_cfg cache DB (spot_sweeps_run_cfg_cache.sqlite3).",
            "  - --write-milestones emits UI preset candidates filtered by milestone thresholds.",
            "  - --track keeps promoted HF/LF lanes distinct; ambiguous auto-selection fails closed.",
            "  - --seed-milestones explicitly overrides the promoted champion source.",
            "",
            *base_lines,
            "",
            *axis_lines,
            "",
            *_combo_full_preset_catalog_lines(),
            "",
            *internal_lines,
        )
    )


# endregion
