"""Named spot-sweep search profiles."""

from __future__ import annotations

from .dimensions import _AXIS_DIMENSION_REGISTRY

_ATR_EXIT_PROFILE_REGISTRY: dict[str, dict[str, object]] = {
    "atr": {
        "atr_periods": (7, 10, 14, 21),
        "pt_mults": (0.6, 0.8, 0.9, 1.0, 1.5, 2.0),
        "sl_mults": (1.0, 1.5, 2.0),
        "title": "C) ATR exits sweep (1h timing + 1d Supertrend)",
        "decimals": None,
    },
    "atr_fine": {
        "atr_periods": (7, 10, 14, 21),
        "pt_mults": (0.8, 0.9, 1.0, 1.1, 1.2),
        "sl_mults": (1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8),
        "title": "ATR exits fine sweep (PT/SL multipliers)",
        "decimals": 2,
    },
    "atr_ultra": {
        "atr_periods": (7,),
        "pt_mults": (1.05, 1.08, 1.10, 1.12, 1.15),
        "sl_mults": (1.35, 1.40, 1.45, 1.50, 1.55),
        "title": "ATR exits ultra-fine sweep (PT/SL micro-grid)",
        "decimals": 2,
    },
}
_SPREAD_PROFILE_REGISTRY: dict[str, dict[str, object]] = {
    "spread": {
        "field": "ema_spread_min_pct",
        "values": (None, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1),
        "note_prefix": "spread",
        "title": "EMA spread sweep (quality gate)",
        "decimals": None,
    },
    "spread_fine": {
        "field": "ema_spread_min_pct",
        "values": (
            None,
            0.002,
            0.0025,
            0.003,
            0.0035,
            0.004,
            0.0045,
            0.005,
            0.0055,
            0.006,
            0.0065,
            0.007,
            0.0075,
            0.008,
        ),
        "note_prefix": "spread",
        "title": "EMA spread fine sweep (quality gate)",
        "decimals": 4,
    },
    "spread_down": {
        "field": "ema_spread_min_pct_down",
        "values": (
            None,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.010,
            0.012,
            0.015,
            0.02,
            0.03,
            0.05,
        ),
        "note_prefix": "spread_down",
        "title": "EMA spread DOWN sweep (directional permission)",
        "decimals": 4,
    },
}
_PERM_JOINT_PROFILE: dict[str, tuple] = {
    "tod_windows": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("tod_windows") or ()),
    "perm_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("perm_variants") or ()),
    "vol_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("vol_variants") or ()),
    "cadence_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("cadence_variants") or ()),
}
_REGIME_ST_PROFILE: dict[str, tuple] = {
    "bars": ("4 hours", "1 day"),
    "atr_periods": (2, 3, 4, 5, 6, 7, 10, 11, 14, 21),
    "multipliers": (
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
    ),
    "sources": ("close", "hl2"),
}
_REGIME2_ST_PROFILE: dict[str, tuple] = {
    "atr_periods": (2, 3, 4, 5, 6, 7, 10, 11, 14, 21),
    "multipliers": (
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
    ),
    "sources": ("close", "hl2"),
}
_SHOCK_SWEEP_PROFILE: dict[str, tuple] = {
    "modes": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("modes") or ()),
    "dir_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("dir_variants") or ()),
    "sl_mults": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("sl_mults") or ()),
    "pt_mults": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("pt_mults") or ()),
    "short_risk_factors": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("short_risk_factors") or ()),
    "ratio_rows": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("ratio_rows") or ()),
    "daily_atr_rows": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("daily_atr_rows") or ()),
    "drawdown_rows": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("drawdown_rows") or ()),
}
