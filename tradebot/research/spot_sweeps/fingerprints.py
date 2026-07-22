"""Stable identities for spot-sweep evaluation and Cartesian spaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ...backtest.config import ConfigBundle
from ...backtest.spot_codec import filters_payload as _filters_payload
from ...backtest.sweep_fingerprint import _canonicalize_fingerprint_value
from .catalog import _COMBO_FULL_CARTESIAN_DIM_ORDER
from .dimensions import _AXIS_DIMENSION_REGISTRY

_RUN_CFG_CACHE_ENGINE_VERSION = "spot_stage_v9"
_RANK_BIN_SIZE = 2048
_AXIS_DIMENSION_FINGERPRINT_KEYS: tuple[str, ...] = tuple(
    str(k) for k in (tuple(_AXIS_DIMENSION_REGISTRY.get("cache", {}).get("dimension_keys") or ())) if str(k).strip()
)


def _axis_dimension_fingerprint(cfg: ConfigBundle) -> str:
    # Use a full strategy+filters fingerprint to avoid false cache collisions
    # when new knobs are introduced but not yet mirrored in narrow dimension key lists.
    strategy = asdict(cfg.strategy)
    filters_payload = _filters_payload(cfg.strategy.filters) or {}
    strategy.pop("filters", None)
    dims = {
        "strategy": _canonicalize_fingerprint_value(strategy),
        "filters": _canonicalize_fingerprint_value(filters_payload),
    }
    return json.dumps(dims, sort_keys=True, default=str)


def _window_signature(
    *,
    bars_sig: tuple[int, object | None, object | None],
    regime_sig: tuple[int, object | None, object | None],
    regime2_sig: tuple[int, object | None, object | None],
) -> str:
    raw = {
        "bars": tuple(bars_sig),
        "regime": tuple(regime_sig),
        "regime2": tuple(regime2_sig),
    }
    return json.dumps(_canonicalize_fingerprint_value(raw), sort_keys=True, default=str)


def _combo_full_dimension_space_signature(
    *,
    ordered_dims: tuple[str, ...] | list[str],
    size_by_dim: dict[str, int],
    timing_profile_variants: list[tuple[str, dict[str, object], dict[str, object]]],
    confirm_bars: list[int],
    pair_variants_by_dim: dict[str, list[tuple[str, dict[str, object]]]],
    short_mults: list[float],
) -> str:
    """Stable fingerprint for combo_full Cartesian variant space.

    Rank-manifest cache keys must change when variant payload values change, even
    if dimension cardinalities stay constant.
    """
    dim_rows: list[tuple[str, tuple[object, ...]]] = []
    for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER:
        key = str(dim_name)
        if key == "timing_profile":
            rows = tuple(
                (
                    str(label),
                    _canonicalize_fingerprint_value(dict(strat_over) if isinstance(strat_over, dict) else {}),
                    _canonicalize_fingerprint_value(dict(filt_over) if isinstance(filt_over, dict) else {}),
                )
                for label, strat_over, filt_over in tuple(timing_profile_variants or ())
            )
        elif key == "confirm":
            rows = tuple(int(v) for v in tuple(confirm_bars or ()))
        elif key == "short_mult":
            rows = tuple(float(v) for v in tuple(short_mults or ()))
        else:
            rows = tuple(
                (
                    str(label),
                    _canonicalize_fingerprint_value(dict(payload) if isinstance(payload, dict) else {}),
                )
                for label, payload in tuple(pair_variants_by_dim.get(str(key)) or ())
            )
        dim_rows.append((str(key), tuple(rows)))
    raw = {
        "ordered_dims": tuple(str(v) for v in tuple(ordered_dims or ())),
        "size_by_dim": tuple((str(dim_name), int(size_by_dim.get(str(dim_name), 0) or 0)) for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER),
        "dim_rows": tuple(dim_rows),
    }
    return hashlib.sha1(json.dumps(_canonicalize_fingerprint_value(raw), sort_keys=True, default=str).encode("utf-8")).hexdigest()
