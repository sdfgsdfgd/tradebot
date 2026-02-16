"""Hot-swappable spot policy packs.

Packs provide baseline strategy/filter defaults for rapid experimentation.
User-specified fields always override pack defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SpotPolicyPack:
    name: str
    strategy_defaults: Mapping[str, object]
    filter_defaults: Mapping[str, object]
    notes: str = ""


_PACKS: dict[str, SpotPolicyPack] = {
    "neutral": SpotPolicyPack(
        name="neutral",
        strategy_defaults={},
        filter_defaults={},
        notes="No-op defaults; explicit strategy/filter fields drive behavior.",
    ),
    "defensive": SpotPolicyPack(
        name="defensive",
        strategy_defaults={
            "spot_resize_mode": "target",
            "spot_resize_allow_scale_in": False,
            "spot_resize_allow_scale_out": True,
            "spot_resize_min_delta_qty": 2,
            "spot_resize_adaptive_mode": "atr",
            "spot_resize_adaptive_min_mult": 0.35,
            "spot_resize_adaptive_max_mult": 1.0,
        },
        filter_defaults={
            "riskoff_mode": "directional",
            "shock_risk_scale_apply_to": "both",
            "shock_risk_scale_min_mult": 0.25,
        },
        notes="Prioritizes drawdown control and downshifts aggressively in stress.",
    ),
    "aggressive": SpotPolicyPack(
        name="aggressive",
        strategy_defaults={
            "spot_resize_mode": "target",
            "spot_resize_allow_scale_in": True,
            "spot_resize_allow_scale_out": True,
            "spot_resize_min_delta_qty": 1,
            "spot_resize_adaptive_mode": "hybrid",
            "spot_resize_adaptive_min_mult": 0.5,
            "spot_resize_adaptive_max_mult": 2.25,
            "spot_resize_adaptive_slope_ref_pct": 0.08,
            "spot_resize_adaptive_vel_ref_pct": 0.06,
        },
        filter_defaults={
            "riskoff_mode": "directional",
            "shock_risk_scale_apply_to": "risk",
        },
        notes="Favors fast scale-up in favorable trend/velocity conditions.",
    ),
    "hf_probe": SpotPolicyPack(
        name="hf_probe",
        strategy_defaults={
            "spot_resize_mode": "target",
            "spot_resize_allow_scale_in": True,
            "spot_resize_allow_scale_out": True,
            "spot_resize_min_delta_qty": 1,
            "spot_resize_max_step_qty": 2,
            "spot_resize_cooldown_bars": 1,
            "spot_resize_adaptive_mode": "slope",
            "spot_resize_adaptive_min_mult": 0.4,
            "spot_resize_adaptive_max_mult": 1.5,
            "spot_resize_adaptive_slope_ref_pct": 0.06,
            "spot_resize_adaptive_vel_ref_pct": 0.05,
            "spot_resize_adaptive_tr_ratio_ref": 1.1,
        },
        filter_defaults={},
        notes="Probe-friendly profile with tighter step sizing and quick feedback loops.",
    ),
}


def normalize_pack_name(raw: object | None) -> str | None:
    name = str(raw or "").strip().lower()
    if not name:
        return None
    if name not in _PACKS:
        return None
    return name


def resolve_pack(
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
) -> SpotPolicyPack | None:
    if isinstance(strategy, Mapping):
        raw = strategy.get("spot_policy_pack")
    else:
        raw = getattr(strategy, "spot_policy_pack", None)
    if raw is None:
        if isinstance(filters, Mapping):
            raw = filters.get("spot_policy_pack")
        else:
            raw = getattr(filters, "spot_policy_pack", None)
    name = normalize_pack_name(raw)
    if name is None:
        return None
    return _PACKS.get(name)


def all_packs() -> dict[str, SpotPolicyPack]:
    return dict(_PACKS)

