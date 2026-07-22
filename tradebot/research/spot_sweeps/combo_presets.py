"""Preset-driven shaping of the canonical combo-full search space."""

from __future__ import annotations

import itertools
import os
from dataclasses import replace
from functools import cached_property

from ...backtest.config import ConfigBundle
from .catalog import (
    _COMBO_FULL_CARTESIAN_DIM_ORDER,
    _COMBO_FULL_NOTE_PAIR_DIM_ORDER,
    _COMBO_FULL_PAIR_DIM_VARIANT_SPECS,
    _combo_full_preset_axes,
    _combo_full_preset_customizer,
    _combo_full_preset_spec,
)
from .dimensions import _AXIS_DIMENSION_REGISTRY, _ema_signal_variants
from .fingerprints import _combo_full_dimension_space_signature
from .milestones import _filters_payload
from .profiles import _PERM_JOINT_PROFILE
from .support import _mk_filters


class ComboPresetContext:
    """Own dimension mutations; runtime execution remains outside the search-space model."""

    def __init__(
        self,
        runtime,
        dims: dict[str, object],
        rows: dict[str, list[object]],
        timing_profiles_loader,
    ) -> None:
        self.runtime = runtime
        self.dims = dims
        self.rows = rows
        self.timing_profiles_loader = timing_profiles_loader

    @cached_property
    def size_by_dim(self) -> dict[str, int]:
        return {
            dim_name: len(self.rows.get(dim_name) or ())
            for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
        }

    @cached_property
    def total(self) -> int:
        total = 1
        for size in self.size_by_dim.values():
            total *= size
        return total

    @cached_property
    def ordered_dims(self) -> tuple[str, ...]:
        dominant = tuple(self.dims.get("dominant_dims") or ())
        ordered = [str(name).strip() for name in dominant if str(name).strip() in self.size_by_dim]
        ordered.extend(name for name in self.size_by_dim if name not in ordered)
        return tuple(ordered)

    @cached_property
    def dimension_signature(self) -> str:
        return _combo_full_dimension_space_signature(
            ordered_dims=self.ordered_dims,
            size_by_dim=self.size_by_dim,
            timing_profile_variants=list(self.rows["timing_profile"]),
            confirm_bars=[int(value) for value in self.rows["confirm"]],
            pair_variants_by_dim={
                dim_name: list(self.rows[dim_name])
                for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
            },
            short_mults=[float(value) for value in self.rows["short_mult"]],
        )

    def rank(self, indices: dict[str, int]) -> int:
        rank = 0
        for dim_name in self.ordered_dims:
            rank = rank * self.size_by_dim[dim_name] + int(indices.get(dim_name, 0))
        return rank

    def indices(self, rank: int) -> dict[str, int]:
        if rank < 0 or rank >= self.total:
            raise ValueError(f"rank out of range: {rank} not in [0,{self.total - 1}]")
        indices: dict[str, int] = {}
        remainder = rank
        for dim_name in reversed(self.ordered_dims):
            size = self.size_by_dim[dim_name]
            if size <= 0:
                raise ValueError(f"invalid dimension cardinality: {dim_name}={size}")
            indices[dim_name] = remainder % size
            remainder //= size
        return indices

    def plan_item(
        self,
        indices: dict[str, int],
        *,
        rank: int | None = None,
    ) -> tuple[ConfigBundle, str, dict]:
        selected = {
            dim_name: self.rows[dim_name][int(indices.get(dim_name, 0))]
            for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
        }
        timing_label, timing_strategy, timing_filters = selected["timing_profile"]
        pair_rows = {
            dim_name: selected[dim_name]
            for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }
        pair_payloads = {dim_name: dict(row[1]) for dim_name, row in pair_rows.items()}

        filter_overrides: dict[str, object] = {}
        for dim_name in ("perm", "tod", "vol", "cadence", "shock", "slope", "risk"):
            filter_overrides.update(pair_payloads[dim_name])
        filter_overrides.update(timing_filters)

        strategy_overrides: dict[str, object] = {}
        for dim_name in ("direction", "regime", "regime2", "exit", "tick"):
            strategy_overrides.update(pair_payloads[dim_name])
        strategy_overrides.update(timing_strategy)
        strategy_overrides.pop("entry_confirm_bars", None)
        strategy_overrides.pop("spot_short_risk_mult", None)
        strategy_overrides.pop("filters", None)

        confirm = int(selected["confirm"])
        short_mult = float(selected["short_mult"])
        base = self.base
        cfg = replace(
            base,
            strategy=replace(
                base.strategy,
                filters=_mk_filters(overrides=filter_overrides) if filter_overrides else None,
                entry_confirm_bars=confirm,
                spot_short_risk_mult=short_mult,
                **strategy_overrides,
            ),
        )
        normalized_indices = {
            dim_name: int(indices.get(dim_name, 0))
            for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
        }
        normalized_indices["_mr_rank"] = self.rank(indices) if rank is None else int(rank)
        note = " | ".join(
            (
                str(timing_label),
                str(pair_rows["direction"][0]),
                f"c={confirm}",
                *(str(pair_rows[dim_name][0]) for dim_name in _COMBO_FULL_NOTE_PAIR_DIM_ORDER),
                f"short_mult={short_mult:g}",
            )
        )
        return cfg, note, normalized_indices

    def plan_item_from_rank(self, rank: int) -> tuple[ConfigBundle, str, dict]:
        return self.plan_item(self.indices(rank), rank=rank)

    def iter_plan(self):
        ranges = tuple(range(self.size_by_dim[dim_name]) for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER)
        for raw_indices in itertools.product(*ranges):
            indices = dict(zip(_COMBO_FULL_CARTESIAN_DIM_ORDER, raw_indices))
            yield self.plan_item(indices)

    def apply(self, preset: str) -> None:
        customizers = {
            "squeeze": self._preset_squeeze,
            "tod_interaction": self._preset_tod_interaction,
            "risk_overlays": self._preset_risk_overlays,
            "gate_matrix": self._preset_gate_matrix,
            "lf_shock_sniper": self._preset_lf_shock_sniper,
            "hf_timing_sniper": self._preset_hf_timing_sniper,
            "ema_regime": self._preset_ema_regime,
            "tick_ema": self._preset_tick_ema,
            "ema_atr": self._preset_ema_atr,
            "r2_atr": self._preset_r2_atr,
            "r2_tod": self._preset_r2_tod,
            "loosen_atr": self._preset_loosen_atr,
        }
        required = {
            str(_combo_full_preset_customizer(name))
            for name in _combo_full_preset_axes(include_tiers=True, include_aliases=True)
            if str(_combo_full_preset_customizer(name)).strip()
        }
        unknown = sorted(required - set(customizers))
        if unknown:
            raise SystemExit(f"Unknown combo_full preset customizers: {', '.join(unknown)}")
        if not preset:
            return
        spec = _combo_full_preset_spec(str(preset))
        if not spec:
            raise SystemExit(f"Unknown combo_full preset: {preset!r}")
        freeze_dims = tuple(spec.get("freeze_dims") or ())
        if freeze_dims:
            self._freeze_dims(*freeze_dims)
        customizer = customizers.get(str(spec.get("customizer") or "").strip().lower())
        if callable(customizer):
            customizer()

    def _freeze_dims(self, *dim_names: str) -> None:
        for dim_name in dim_names:
            rows = list(self.rows.get(str(dim_name)) or ())
            if not rows:
                raise SystemExit(f"combo_full preset requires non-empty {dim_name} variants.")
            self.rows[str(dim_name)] = [rows[0]]

    def _set_dim_rows(self, dim_name: str, rows: list[object]) -> None:
        values = list(rows or ())
        if not values:
            raise SystemExit(f"combo_full preset generated empty {dim_name} variants.")
        self.rows[str(dim_name)] = list(values)

    @cached_property
    def base(self) -> ConfigBundle:
        root = self.runtime._base_bundle(bar_size=self.runtime.signal_bar_size, filters=None)
        return replace(
            root,
            strategy=replace(
                root.strategy,
                filters=None,
                tick_gate_mode="off",
                regime2_mode="off",
                regime2_bar_size=None,
                spot_exit_time_et=None,
            ),
        )

    def _ema_direction_rows(self) -> list[tuple[str, dict[str, object]]]:
        return list(_ema_signal_variants("combo"))

    def _atr_exit_rows(self, *, with_close_eod: bool = False) -> list[tuple[str, dict[str, object]]]:
        rows: list[tuple[str, dict[str, object]]] = []
        atr_periods = (10, 14, 21)
        if with_close_eod:
            pt_mults = (0.6, 0.65, 0.7, 0.75, 0.8)
            sl_mults = (1.2, 1.4, 1.6, 1.8, 2.0)
            close_eod_vals = (False, True)
        else:
            pt_mults = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
            sl_mults = (1.2, 1.4, 1.5, 1.6, 1.8, 2.0)
            close_eod_vals = (False,)
        for close_eod in close_eod_vals:
            for atr_p in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        label = f"exit=atr({int(atr_p)},{float(pt_m):.2f},{float(sl_m):.2f})" + (f" close_eod={int(bool(close_eod))}" if with_close_eod else "")
                        payload = {
                            "spot_exit_mode": "atr",
                            "spot_atr_period": int(atr_p),
                            "spot_pt_atr_mult": float(pt_m),
                            "spot_sl_atr_mult": float(sl_m),
                            "spot_profit_target_pct": None,
                            "spot_stop_loss_pct": None,
                        }
                        if with_close_eod:
                            payload["spot_close_eod"] = bool(close_eod)
                        rows.append((label, payload))
        return rows

    def _preset_squeeze(self) -> None:
        regime2_rows: list[tuple[str, dict[str, object]]] = [("r2=off", {"regime2_mode": "off", "regime2_bar_size": None})]
        atr_periods = (2, 3, 4, 5, 6, 7, 10, 11)
        multipliers = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3)
        for r2_bar in ("4 hours", "1 day"):
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in ("close", "hl2"):
                        regime2_rows.append(
                            (
                                f"r2=ST({int(atr_p)},{float(mult):g},{str(src)})@{str(r2_bar)}",
                                {
                                    "regime2_mode": "supertrend",
                                    "regime2_bar_size": str(r2_bar),
                                    "regime2_supertrend_atr_period": int(atr_p),
                                    "regime2_supertrend_multiplier": float(mult),
                                    "regime2_supertrend_source": str(src),
                                },
                            )
                        )
        vol_rows: list[tuple[str, dict[str, object]]] = [
            ("vol=-", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.0@20", {"volume_ratio_min": 1.0, "volume_ema_period": 20}),
            ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
            ("vol>=1.5@10", {"volume_ratio_min": 1.5, "volume_ema_period": 10}),
            ("vol>=1.5@20", {"volume_ratio_min": 1.5, "volume_ema_period": 20}),
        ]
        tod_rows: list[tuple[str, dict[str, object]]] = [
            ("tod=base", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=18-03 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 3}),
            ("tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
            ("tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
            ("tod=11-16 ET", {"entry_start_hour_et": 11, "entry_end_hour_et": 16}),
        ]
        self._set_dim_rows("regime2", regime2_rows)
        self._set_dim_rows("vol", vol_rows)
        self._set_dim_rows("tod", tod_rows)
        self._set_dim_rows("confirm", [0, 1, 2])
        self._set_dim_rows("short_mult", [1.0])

    def _preset_tod_interaction(self) -> None:
        tod_rows: list[tuple[str, dict[str, object]]] = []
        for start_h in (17, 18, 19):
            for end_h in (3, 4, 5):
                tod_rows.append(
                    (
                        f"tod={int(start_h):02d}-{int(end_h):02d} ET",
                        {
                            "entry_start_hour_et": int(start_h),
                            "entry_end_hour_et": int(end_h),
                        },
                    )
                )
        cadence_rows: list[tuple[str, dict[str, object]]] = []
        for skip in (0, 1, 2):
            for cooldown in (0, 1, 2):
                cadence_rows.append(
                    (
                        f"cad=skip{int(skip)} cd{int(cooldown)}",
                        {"skip_first_bars": int(skip), "cooldown_bars": int(cooldown)},
                    )
                )
        self._set_dim_rows("tod", tod_rows)
        self._set_dim_rows("cadence", cadence_rows)
        self._set_dim_rows("short_mult", [1.0])

    def _preset_risk_overlays(self) -> None:
        if bool(getattr(self.runtime.args, "risk_overlays_skip_pop", False)):
            filtered = [row for row in list(self.rows.get("risk") or ()) if "riskpop" not in str(row[0]).lower()]
            self._set_dim_rows("risk", filtered)
        self._set_dim_rows("short_mult", [1.0])

    def _preset_gate_matrix(self) -> None:
        gate_dims = _AXIS_DIMENSION_REGISTRY.get("gate_matrix", {})
        if isinstance(gate_dims, dict):
            perm_raw = tuple(gate_dims.get("perm_variants") or ())
            if perm_raw:
                self._set_dim_rows(
                    "perm",
                    [(str(label), dict(payload)) for label, payload in perm_raw if isinstance(label, str) and isinstance(payload, dict)],
                )
            tod_raw = tuple(gate_dims.get("tod_variants") or ())
            if tod_raw:
                self._set_dim_rows(
                    "tod",
                    [
                        (
                            str(label),
                            {
                                "entry_start_hour_et": (int(start_h) if start_h is not None else None),
                                "entry_end_hour_et": (int(end_h) if end_h is not None else None),
                            },
                        )
                        for label, start_h, end_h in tod_raw
                        if isinstance(label, str)
                    ],
                )
            short_raw = tuple(gate_dims.get("short_mults") or ())
            if short_raw:
                self._set_dim_rows("short_mult", [float(v) for v in short_raw])

    def _preset_ema_regime(self) -> None:
        self._set_dim_rows("direction", self._ema_direction_rows())
        self._set_dim_rows("short_mult", [1.0])

    def _preset_tick_ema(self) -> None:
        self._set_dim_rows("direction", self._ema_direction_rows())
        tick_rows: list[tuple[str, dict[str, object]]] = []
        for policy in ("allow", "block"):
            for z_enter in (0.8, 1.0, 1.2):
                for z_exit in (0.4, 0.5, 0.6):
                    for slope_lb in (3, 5):
                        for lookback in (126, 252):
                            tick_rows.append(
                                (
                                    f"tick=wide policy={policy} z_in={float(z_enter):g} z_out={float(z_exit):g} slope={int(slope_lb)} lb={int(lookback)}",
                                    {
                                        "tick_gate_mode": "raschke",
                                        "tick_gate_symbol": "TICK-AMEX",
                                        "tick_gate_exchange": "AMEX",
                                        "tick_neutral_policy": str(policy),
                                        "tick_direction_policy": "wide_only",
                                        "tick_band_ma_period": 10,
                                        "tick_width_z_lookback": int(lookback),
                                        "tick_width_z_enter": float(z_enter),
                                        "tick_width_z_exit": float(z_exit),
                                        "tick_width_slope_lookback": int(slope_lb),
                                    },
                                )
                            )
        if bool(getattr(self.runtime.args, "combo_full_include_tick", False)):
            self._set_dim_rows("tick", tick_rows)
        else:
            self._set_dim_rows("tick", [("tick=off", {"tick_gate_mode": "off"})])
        self._set_dim_rows("short_mult", [1.0])

    def _preset_ema_atr(self) -> None:
        self._set_dim_rows("direction", self._ema_direction_rows())
        self._set_dim_rows("exit", self._atr_exit_rows(with_close_eod=False))
        self._set_dim_rows("short_mult", [1.0])

    def _preset_r2_atr(self) -> None:
        self._set_dim_rows("exit", self._atr_exit_rows(with_close_eod=False))
        self._set_dim_rows("short_mult", [1.0])

    def _preset_r2_tod(self) -> None:
        tod_rows = [
            (str(note), dict(over))
            for _start_h, _end_h, note, over in tuple(_PERM_JOINT_PROFILE.get("tod_windows") or ())
            if isinstance(note, str) and isinstance(over, dict)
        ]
        if tod_rows:
            self._set_dim_rows("tod", tod_rows)
        self._set_dim_rows("short_mult", [1.0])

    def _preset_loosen_atr(self) -> None:
        self._set_dim_rows("exit", self._atr_exit_rows(with_close_eod=True))
        self._set_dim_rows("short_mult", [1.0])

    def _preset_lf_shock_sniper(self) -> None:
        self._set_dim_rows(
            "regime",
            [
                (
                    "regime=ST(1d:14,1.0,hl2)",
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 14,
                        "supertrend_multiplier": 1.0,
                        "supertrend_source": "hl2",
                    },
                )
            ],
        )
        self._set_dim_rows("regime2", [("r2=off", {"regime2_mode": "off", "regime2_bar_size": None})])
        self._set_dim_rows(
            "perm",
            [("perm=off", {"ema_spread_min_pct": None, "ema_slope_min_pct": None})],
        )
        self._set_dim_rows(
            "tod",
            [("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None})],
        )
        self._set_dim_rows("vol", [("vol=off", {"volume_ratio_min": None, "volume_ema_period": None})])
        self._set_dim_rows("cadence", [("cad=base", {})])
        self._set_dim_rows("tick", [("tick=off", {"tick_gate_mode": "off"})])
        self._set_dim_rows("risk", [("risk=off", {})])
        self._set_dim_rows(
            "shock",
            [
                ("shock=off", {"shock_gate_mode": "off"}),
                (
                    "shock=tr_ratio on=1.300 off=1.200",
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "tr_ratio",
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.300,
                        "shock_off_ratio": 1.200,
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                        "shock_profit_target_pct_mult": 1.0,
                    },
                ),
                (
                    "shock=tr_ratio on=1.325 off=1.225",
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "tr_ratio",
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.325,
                        "shock_off_ratio": 1.225,
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                        "shock_profit_target_pct_mult": 1.0,
                    },
                ),
                (
                    "shock=tr_ratio on=1.350 off=1.250",
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "tr_ratio",
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.350,
                        "shock_off_ratio": 1.250,
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                        "shock_profit_target_pct_mult": 1.0,
                    },
                ),
                (
                    "shock=tr_ratio on=1.355 off=1.255",
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "tr_ratio",
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.355,
                        "shock_off_ratio": 1.255,
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                        "shock_profit_target_pct_mult": 1.0,
                    },
                ),
            ],
        )
        self._set_dim_rows("short_mult", [1.0])

    def _preset_hf_timing_sniper(self) -> None:
        base = self.base
        hf_profiles = self.timing_profiles_loader(variants_key="hf_profile_variants")
        if not hf_profiles:
            self._set_dim_rows("short_mult", [1.0])
            return
        base_filters_payload = _filters_payload(getattr(base.strategy, "filters", None)) or {}
        base_filter_row = ("base_filters", dict(base_filters_payload))
        self._set_dim_rows(
            "direction",
            [
                (
                    "direction=base",
                    {
                        "entry_signal": str(getattr(base.strategy, "entry_signal", "ema") or "ema"),
                        "ema_preset": str(getattr(base.strategy, "ema_preset", "3/7") or "3/7"),
                        "ema_entry_mode": str(getattr(base.strategy, "ema_entry_mode", "trend") or "trend"),
                    },
                )
            ],
        )
        self._set_dim_rows("confirm", [int(getattr(base.strategy, "entry_confirm_bars", 0) or 0)])
        self._set_dim_rows("perm", [base_filter_row])
        self._set_dim_rows("tod", [base_filter_row])
        self._set_dim_rows("vol", [base_filter_row])
        self._set_dim_rows("cadence", [base_filter_row])
        self._set_dim_rows(
            "regime",
            [
                (
                    "regime=base",
                    {
                        "regime_mode": str(getattr(base.strategy, "regime_mode", "supertrend") or "supertrend"),
                        "regime_bar_size": str(getattr(base.strategy, "regime_bar_size", "1 day") or "1 day"),
                        "supertrend_atr_period": int(getattr(base.strategy, "supertrend_atr_period", 7) or 7),
                        "supertrend_multiplier": float(getattr(base.strategy, "supertrend_multiplier", 0.4) or 0.4),
                        "supertrend_source": str(getattr(base.strategy, "supertrend_source", "close") or "close"),
                    },
                )
            ],
        )
        self._set_dim_rows(
            "regime2",
            [
                (
                    "regime2=base",
                    {
                        "regime2_mode": str(getattr(base.strategy, "regime2_mode", "off") or "off"),
                        "regime2_bar_size": getattr(base.strategy, "regime2_bar_size", None),
                        "regime2_ema_preset": getattr(base.strategy, "regime2_ema_preset", None),
                        "regime2_supertrend_atr_period": int(getattr(base.strategy, "regime2_supertrend_atr_period", 10) or 10),
                        "regime2_supertrend_multiplier": float(getattr(base.strategy, "regime2_supertrend_multiplier", 3.0) or 3.0),
                        "regime2_supertrend_source": str(getattr(base.strategy, "regime2_supertrend_source", "hl2") or "hl2"),
                    },
                )
            ],
        )
        self._set_dim_rows(
            "exit",
            [
                (
                    "exit=base",
                    {
                        "spot_exit_mode": str(getattr(base.strategy, "spot_exit_mode", "pct") or "pct"),
                        "spot_profit_target_pct": getattr(base.strategy, "spot_profit_target_pct", None),
                        "spot_stop_loss_pct": getattr(base.strategy, "spot_stop_loss_pct", None),
                        "spot_atr_period": int(getattr(base.strategy, "spot_atr_period", 14) or 14),
                        "spot_pt_atr_mult": float(getattr(base.strategy, "spot_pt_atr_mult", 1.5) or 1.5),
                        "spot_sl_atr_mult": float(getattr(base.strategy, "spot_sl_atr_mult", 1.0) or 1.0),
                        "spot_close_eod": bool(getattr(base.strategy, "spot_close_eod", False)),
                    },
                )
            ],
        )
        self._set_dim_rows(
            "tick",
            [
                (
                    "tick=base",
                    {
                        "tick_gate_mode": str(getattr(base.strategy, "tick_gate_mode", "off") or "off"),
                        "tick_gate_symbol": str(getattr(base.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE"),
                        "tick_gate_exchange": str(getattr(base.strategy, "tick_gate_exchange", "NYSE") or "NYSE"),
                        "tick_neutral_policy": str(getattr(base.strategy, "tick_neutral_policy", "allow") or "allow"),
                        "tick_direction_policy": str(getattr(base.strategy, "tick_direction_policy", "both") or "both"),
                        "tick_band_ma_period": int(getattr(base.strategy, "tick_band_ma_period", 10) or 10),
                        "tick_width_z_lookback": int(getattr(base.strategy, "tick_width_z_lookback", 252) or 252),
                        "tick_width_z_enter": float(getattr(base.strategy, "tick_width_z_enter", 1.0) or 1.0),
                        "tick_width_z_exit": float(getattr(base.strategy, "tick_width_z_exit", 0.5) or 0.5),
                        "tick_width_slope_lookback": int(getattr(base.strategy, "tick_width_slope_lookback", 3) or 3),
                    },
                )
            ],
        )
        self._set_dim_rows("shock", [base_filter_row])
        self._set_dim_rows("slope", [base_filter_row])
        self._set_dim_rows("risk", [base_filter_row])
        self._set_dim_rows(
            "short_mult",
            [float(getattr(base.strategy, "spot_short_risk_mult", 1.0) or 1.0)],
        )
        base_rows = [row for row in hf_profiles if "hf_symm" in str(row[0]).lower()]
        # Keep this preset intentionally tight: one HF symm anchor, then mutate
        # only the highest-value timing/slope/velocity/branch-size pockets.
        if base_rows:
            base_rows = [base_rows[0]]
        if not base_rows:
            base_rows = list(hf_profiles[:1])
        if not base_rows:
            base_rows = list(hf_profiles[:1])
        timing_rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
        seen_labels: set[str] = set()
        base_mode_profiles: tuple[tuple[str, dict[str, object], dict[str, object]], ...] = (
            (
                "overlay_only_hybrid_baseline",
                {
                    "ratsv_enabled": True,
                    "ratsv_slope_slow_window_bars": 11,
                },
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 0.6,
                    "supertrend_source": "hl2",
                    "regime2_mode": "off",
                    "regime2_bar_size": None,
                    "spot_policy_graph": "aggressive",
                    "spot_risk_overlay_policy": "trend_bias",
                    "spot_resize_mode": "target",
                    "spot_resize_min_delta_qty": 3,
                    "spot_resize_max_step_qty": 2,
                    "spot_resize_cooldown_bars": 6,
                    "spot_resize_adaptive_mode": "hybrid",
                    "spot_resize_adaptive_min_mult": 0.90,
                    "spot_resize_adaptive_max_mult": 1.40,
                    "spot_resize_adaptive_slope_ref_pct": 0.06,
                    "spot_resize_adaptive_vel_ref_pct": 0.04,
                    "spot_resize_adaptive_tr_ratio_ref": 1.00,
                    "spot_graph_overlay_atr_hi_pct": 8.0,
                    "spot_graph_overlay_atr_hi_min_mult": 0.85,
                    "spot_graph_overlay_trend_boost_max": 1.35,
                    "spot_graph_overlay_slope_ref_pct": 0.06,
                    "spot_graph_overlay_tr_ratio_ref": 1.05,
                    "spot_graph_overlay_trend_floor_mult": 0.90,
                    "exit_on_signal_flip": True,
                    "flip_exit_mode": "cross",
                    "flip_exit_gate_mode": "regime_or_permission",
                    "flip_exit_min_hold_bars": 0,
                    "flip_exit_only_if_profit": False,
                },
            ),
            (
                "overlay_only_hybrid_predref",
                {
                    "ratsv_enabled": True,
                    "ratsv_slope_slow_window_bars": 11,
                },
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 0.6,
                    "supertrend_source": "hl2",
                    "regime2_mode": "off",
                    "regime2_bar_size": None,
                    "spot_policy_graph": "aggressive",
                    "spot_risk_overlay_policy": "trend_bias",
                    "spot_resize_mode": "target",
                    "spot_resize_min_delta_qty": 3,
                    "spot_resize_max_step_qty": 2,
                    "spot_resize_cooldown_bars": 6,
                    "spot_resize_adaptive_mode": "hybrid",
                    "spot_resize_adaptive_min_mult": 0.90,
                    "spot_resize_adaptive_max_mult": 1.60,
                    "spot_resize_adaptive_atr_target_pct": 8.0,
                    "spot_resize_adaptive_atr_vel_ref_pct": 0.25,
                    "spot_resize_adaptive_slope_ref_pct": 0.055,
                    "spot_resize_adaptive_vel_ref_pct": 0.032,
                    "spot_resize_adaptive_tr_ratio_ref": 0.98,
                    "spot_exit_flip_hold_tr_ratio_min": 1.00,
                    "spot_exit_flip_hold_slow_slope_min_pct": 0.000002,
                    "spot_exit_flip_hold_slow_slope_vel_min_pct": 0.000001,
                    "spot_graph_overlay_atr_hi_pct": 8.0,
                    "spot_graph_overlay_atr_hi_min_mult": 0.84,
                    "spot_graph_overlay_atr_vel_ref_pct": 0.25,
                    "spot_graph_overlay_trend_boost_max": 1.75,
                    "spot_graph_overlay_slope_ref_pct": 0.055,
                    "spot_graph_overlay_tr_ratio_ref": 1.00,
                    "spot_graph_overlay_trend_floor_mult": 0.88,
                    "exit_on_signal_flip": True,
                    "flip_exit_mode": "cross",
                    "flip_exit_gate_mode": "regime_or_permission",
                    "flip_exit_min_hold_bars": 0,
                    "flip_exit_only_if_profit": False,
                },
            ),
        )
        bridge_only = str(os.environ.get("TB_HF_TIMING_SNIPER_BRIDGE", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        mode_profiles: tuple[tuple[str, dict[str, object], dict[str, object]], ...]
        if bridge_only:
            predref_filter_over = dict(base_mode_profiles[1][1])
            predref_strategy_over = dict(base_mode_profiles[1][2])
            bridge_rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
            for flip_hold_tr_ratio in (0.97, 0.99):
                for trend_floor_mult in (0.84, 0.86):
                    strat_over = dict(predref_strategy_over)
                    strat_over["spot_exit_flip_hold_tr_ratio_min"] = float(flip_hold_tr_ratio)
                    strat_over["spot_graph_overlay_trend_floor_mult"] = float(trend_floor_mult)
                    tag = f"overlay_only_hybrid_predref_bridge_tr{float(flip_hold_tr_ratio):0.2f}_floor{float(trend_floor_mult):0.2f}"
                    bridge_rows.append((str(tag), dict(predref_filter_over), strat_over))
            mode_profiles = tuple(bridge_rows)
        else:
            mode_profiles = tuple(base_mode_profiles)
        rank_values = (0.0035,) if bridge_only else (0.0035, 0.0185)
        cross_age_values = (4, 6) if bridge_only else (6, 10)
        slope_pairs = (
            (0.000002, 0.000001),
            (0.000006, 0.000002),
        )
        branch_b_mult_values = (1.20, 1.60)
        variants: list[
            tuple[
                float,
                int,
                float,
                float,
                float,
                str,
                dict[str, object],
                dict[str, object],
            ]
        ] = []
        for tag, filt_extra, strat_extra in mode_profiles:
            for rank_min in rank_values:
                for cross_age in cross_age_values:
                    for slope_med, slope_vel in slope_pairs:
                        for branch_b_mult in branch_b_mult_values:
                            variants.append(
                                (
                                    float(rank_min),
                                    int(cross_age),
                                    float(slope_med),
                                    float(slope_vel),
                                    float(branch_b_mult),
                                    str(tag),
                                    dict(filt_extra),
                                    dict(strat_extra),
                                )
                            )
        for label, strat_over, filt_over in tuple(base_rows):
            for (
                rank_min,
                cross_age,
                slope_med,
                slope_vel,
                branch_b_mult,
                tag,
                filt_extra,
                strat_extra,
            ) in variants:
                filt = dict(filt_over)
                filt["ratsv_branch_a_rank_min"] = float(rank_min)
                filt["ratsv_branch_a_cross_age_max_bars"] = int(cross_age)
                filt["ratsv_branch_a_slope_med_min_pct"] = float(slope_med)
                filt["ratsv_branch_a_slope_vel_min_pct"] = float(slope_vel)
                filt.update(dict(filt_extra))
                strat = dict(strat_over)
                strat["spot_branch_b_size_mult"] = float(branch_b_mult)
                strat.update(dict(strat_extra))
                custom_label = (
                    f"{str(label)} | sniper rank={float(rank_min):0.4f} "
                    f"cross={int(cross_age)} "
                    f"slope={float(slope_med):0.6f}/{float(slope_vel):0.6f} "
                    f"b_mult={float(branch_b_mult):0.2f} tag={tag}"
                )
                if custom_label in seen_labels:
                    continue
                seen_labels.add(custom_label)
                timing_rows.append((str(custom_label), strat, filt))
        if timing_rows:
            self._set_dim_rows("timing_profile", list(timing_rows))
        self._set_dim_rows("short_mult", [1.0])
