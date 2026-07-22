"""SweepRegimeAxes capability slice for the canonical spot research runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from ...backtest.config import (
    ConfigBundle,
)
from .dimensions import _AXIS_DIMENSION_REGISTRY
from .profiles import (
    _SHOCK_SWEEP_PROFILE,
    _SPREAD_PROFILE_REGISTRY,
    _SUPERTREND_SEARCH_PROFILE,
    SupertrendVariant,
)
from .milestones import (
    _print_leaderboards,
)
from .support import (
    _mk_filters,
)


def _with_supertrend(
    base: ConfigBundle,
    variant: SupertrendVariant,
    *,
    layer: Literal["primary", "confirmation"],
) -> ConfigBundle:
    if layer == "primary":
        overrides = {
            "regime_mode": "supertrend",
            "regime_bar_size": variant.bar_size,
            "supertrend_atr_period": variant.atr_period,
            "supertrend_multiplier": variant.multiplier,
            "supertrend_source": variant.source,
        }
    else:
        overrides = {
            "regime2_mode": "supertrend",
            "regime2_bar_size": variant.bar_size,
            "regime2_supertrend_atr_period": variant.atr_period,
            "regime2_supertrend_multiplier": variant.multiplier,
            "regime2_supertrend_source": variant.source,
        }
    return replace(base, strategy=replace(base.strategy, **overrides))


class SweepRegimeAxes:
    def _sweep_regime(self) -> None:
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for gate in _SUPERTREND_SEARCH_PROFILE.variants("primary"):
            cfg = _with_supertrend(base, gate, layer="primary")
            note = f"ST({gate.atr_period},{gate.multiplier},{gate.source}) @{gate.bar_size}"
            cfg_pairs.append((cfg, note))
        tested_total = self._run_cfg_pairs_grid(
            axis_tag="regime",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=100,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(
            rows,
            title="Regime sweep (Supertrend params + timeframe)",
            top_n=int(self.args.top),
        )

    def _sweep_regime2(self) -> None:
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = [(base, "base")]
        for gate in _SUPERTREND_SEARCH_PROFILE.variants("confirmation"):
            cfg = _with_supertrend(base, gate, layer="confirmation")
            note = f"ST2(4h:{gate.atr_period},{gate.multiplier},{gate.source})"
            cfg_pairs.append((cfg, note))
        tested_total = self._run_cfg_pairs_grid(
            axis_tag="regime2",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=100,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(
            rows,
            title="Dual regime sweep (regime2 Supertrend @ 4h)",
            top_n=int(self.args.top),
        )

    def _sweep_regime2_ema(self) -> None:
        """Confirm layer: EMA trend gate on a higher timeframe (4h/1d)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        presets = ["3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]
        cfg_pairs = [(base, "base")]
        for r2_bar in ("4 hours", "1 day"):
            for preset in presets:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        regime2_mode="ema",
                        regime2_bar_size=str(r2_bar),
                        regime2_ema_preset=str(preset),
                    ),
                )
                note = f"r2=EMA({preset})@{r2_bar}"
                cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="regime2_ema", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Regime2 EMA sweep (trend confirm)", top_n=int(self.args.top))

    def _sweep_joint(self) -> None:
        """Targeted interaction hunt: sweep regime + regime2 together (keeps base filters)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        # Keep this tight and focused; the point is to cover interaction edges that compact preset funnels can miss.
        cfg_pairs = [(base, "base")]
        primary_gates = _SUPERTREND_SEARCH_PROFILE.variants(
            "primary",
            bars=("4 hours",),
            atr_periods=(10, 14, 21),
            multipliers=(0.4, 0.5, 0.6),
        )
        confirmation_gates = tuple(
            _SUPERTREND_SEARCH_PROFILE.variants(
                "confirmation",
                bars=("4 hours", "1 day"),
                atr_periods=(3, 4, 5, 6, 7, 10, 14),
                multipliers=(0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
            )
        )
        for primary in primary_gates:
            for confirmation in confirmation_gates:
                cfg = _with_supertrend(base, primary, layer="primary")
                cfg = _with_supertrend(cfg, confirmation, layer="confirmation")
                note = (
                    f"ST({primary.atr_period},{primary.multiplier},{primary.source})@{primary.bar_size} + "
                    f"ST2({confirmation.bar_size}:{confirmation.atr_period},"
                    f"{confirmation.multiplier},{confirmation.source})"
                )
                cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(
            axis_tag="joint", cfg_pairs=cfg_pairs, rows=rows, report_every=100, heartbeat_sec=20.0
        ) < 0:
            return
        _print_leaderboards(rows, title="Joint sweep (regime × regime2)", top_n=int(self.args.top))

    def _sweep_micro_st(self) -> None:
        """Micro sweep around the current ST + ST2 neighborhood (tighter, more granular)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        cfg_pairs = [(base, "base")]
        primary_gates = _SUPERTREND_SEARCH_PROFILE.variants(
            "primary",
            bars=("4 hours",),
            atr_periods=(14, 21),
            multipliers=(0.4, 0.45, 0.5, 0.55, 0.6),
            sources=("close",),
        )
        confirmation_gates = tuple(
            _SUPERTREND_SEARCH_PROFILE.variants(
                "confirmation",
                atr_periods=(4, 5, 6),
                multipliers=(0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4),
                sources=("close",),
            )
        )
        for primary in primary_gates:
            for confirmation in confirmation_gates:
                cfg = _with_supertrend(base, primary, layer="primary")
                cfg = _with_supertrend(cfg, confirmation, layer="confirmation")
                note = (
                    f"ST({primary.atr_period},{primary.multiplier},close) + "
                    f"ST2(4h:{confirmation.atr_period},{confirmation.multiplier},close)"
                )
                cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(
            axis_tag="micro_st", cfg_pairs=cfg_pairs, rows=rows, report_every=50, heartbeat_sec=15.0
        ) < 0:
            return
        _print_leaderboards(rows, title="Micro ST sweep (granular mults)", top_n=int(self.args.top))

    def _sweep_flip_exit(self) -> None:
        """Targeted exit semantics: flip-exit mode + profit-only gating."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        cfg_pairs = [(base, "base")]
        for exit_on_flip in (True, False):
            for mode in ("entry", "state", "cross"):
                for only_profit in (False, True):
                    for hold in (0, 2, 4, 6):
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                exit_on_signal_flip=bool(exit_on_flip),
                                flip_exit_mode=str(mode),
                                flip_exit_only_if_profit=bool(only_profit),
                                flip_exit_min_hold_bars=int(hold),
                            ),
                        )
                        note = f"flip={'on' if exit_on_flip else 'off'} mode={mode} hold={hold} only_profit={int(only_profit)}"
                        cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="flip_exit", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Flip-exit semantics sweep", top_n=int(self.args.top))

    def _sweep_confirm(self) -> None:
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for confirm in (0, 1, 2, 3):
            cfg = replace(base, strategy=replace(base.strategy, entry_confirm_bars=int(confirm)))
            note = f"confirm={confirm}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="confirm", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Confirm-bars sweep (quality gate)", top_n=int(self.args.top))

    def _run_spread_profile(self, profile_name: str) -> None:
        profile = _SPREAD_PROFILE_REGISTRY.get(str(profile_name))
        if not isinstance(profile, dict):
            raise SystemExit(f"Unknown spread sweep profile: {profile_name!r}")
        axis_tag = str(profile_name).strip().lower()
        field_name = str(profile.get("field") or "")
        note_prefix = str(profile.get("note_prefix") or "spread")
        decimals_raw = profile.get("decimals")
        decimals = int(decimals_raw) if isinstance(decimals_raw, int) else None
        values = tuple(profile.get("values") or ())
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for raw in values:
            spread_val = float(raw) if raw is not None else None
            overrides: dict[str, object] = {field_name: spread_val}
            f = _mk_filters(overrides=overrides)
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
            note = "-" if spread_val is None else f"{note_prefix}>={self._fmt_sweep_float(spread_val, decimals)}"
            cfg_pairs.append((cfg, note))
        tested_total = self._run_cfg_pairs_grid(
            axis_tag=str(axis_tag),
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=20,
            heartbeat_sec=8.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(
            rows,
            title=str(profile.get("title") or "EMA spread sweep"),
            top_n=int(self.args.top),
        )

    def _sweep_spread(self) -> None:
        self._run_spread_profile("spread")

    def _sweep_spread_fine(self) -> None:
        """Fine-grained sweep around the current champion spread gate."""
        self._run_spread_profile("spread_fine")

    def _sweep_spread_down(self) -> None:
        """Directional permission: sweep stricter EMA spread gate for down entries only."""
        self._run_spread_profile("spread_down")

    def _sweep_slope(self) -> None:
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for slope in (None, 0.005, 0.01, 0.02, 0.03, 0.05):
            f = _mk_filters(ema_slope_min_pct=float(slope) if slope is not None else None)
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
            note = "-" if slope is None else f"slope>={slope}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="slope", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="EMA slope sweep (quality gate)", top_n=int(self.args.top))

    def _sweep_slope_signed(self) -> None:
        """Directional slope gate: require EMA fast slope to be positive/negative by direction."""
        thr_vals = [None, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
        variants: list[tuple[float | None, float | None, str]] = [(None, None, "signed_slope=off")]
        for up_thr in thr_vals:
            if up_thr is None:
                continue
            variants.append((float(up_thr), None, f"slope_up>={up_thr:g}"))
        for down_thr in thr_vals:
            if down_thr is None:
                continue
            variants.append((None, float(down_thr), f"slope_down>={down_thr:g}"))
        for both_thr in (0.005, 0.01, 0.02, 0.03):
            variants.append((float(both_thr), float(both_thr), f"slope_signed>={both_thr:g} (both)"))

        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for up_thr, down_thr, note in variants:
            f = _mk_filters(
                overrides={
                    "ema_slope_signed_min_pct_up": up_thr,
                    "ema_slope_signed_min_pct_down": down_thr,
                }
            )
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="slope_signed", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(
            rows,
            title="EMA signed-slope sweep (directional permission)",
            top_n=int(self.args.top),
        )

    def _sweep_cooldown(self) -> None:
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for cooldown in (0, 1, 2, 3, 4, 6, 8):
            f = _mk_filters(cooldown_bars=int(cooldown))
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
            note = f"cooldown={cooldown}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="cooldown", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Cooldown sweep (quality gate)", top_n=int(self.args.top))

    def _sweep_skip_open(self) -> None:
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for skip in (0, 1, 2, 3, 4, 6):
            f = _mk_filters(skip_first_bars=int(skip))
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
            note = f"skip_first={skip}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="skip_open", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Skip-open sweep (quality gate)", top_n=int(self.args.top))

    def _sweep_shock(self) -> None:
        """Shock overlay sweep (detectors, modes, and a few core threshold grids)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        profile = _SHOCK_SWEEP_PROFILE
        modes = tuple(profile.get("modes") or ())
        dir_variants = tuple(profile.get("dir_variants") or ())
        sl_mults = tuple(profile.get("sl_mults") or ())
        pt_mults = tuple(profile.get("pt_mults") or ())
        short_risk_factors = tuple(profile.get("short_risk_factors") or ())

        presets: list[tuple[str, dict[str, object], str]] = []
        for detector, fast, slow, on, off, min_pct in tuple(profile.get("ratio_rows") or ()):
            presets.append(
                (
                    str(detector),
                    {
                        "shock_atr_fast_period": int(fast),
                        "shock_atr_slow_period": int(slow),
                        "shock_on_ratio": float(on),
                        "shock_off_ratio": float(off),
                        "shock_min_atr_pct": float(min_pct),
                    },
                    f"{detector} fast={fast} slow={slow} on={on:g} off={off:g} min={min_pct:g}",
                )
            )
        for period, on_atr, off_atr, tr_on in tuple(profile.get("daily_atr_rows") or ()):
            presets.append(
                (
                    "daily_atr_pct",
                    {
                        "shock_daily_atr_period": int(period),
                        "shock_daily_on_atr_pct": float(on_atr),
                        "shock_daily_off_atr_pct": float(off_atr),
                        "shock_daily_on_tr_pct": float(tr_on) if tr_on is not None else None,
                    },
                    f"daily_atr_pct p={period} on={on_atr:g} off={off_atr:g} tr_on={tr_on if tr_on is not None else '-'}",
                )
            )
        for lb, dd_on, dd_off in tuple(profile.get("drawdown_rows") or ()):
            presets.append(
                (
                    "daily_drawdown",
                    {
                        "shock_drawdown_lookback_days": int(lb),
                        "shock_on_drawdown_pct": float(dd_on),
                        "shock_off_drawdown_pct": float(dd_off),
                    },
                    f"daily_drawdown lb={lb} on={dd_on:g} off={dd_off:g}",
                )
            )

        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = [(base, "base")]
        for detector, params, det_note in presets:
            for mode in modes:
                for dir_src, dir_lb, dir_note in dir_variants:
                    for sl_mult in sl_mults:
                        for pt_mult in pt_mults:
                            for short_factor in short_risk_factors:
                                overrides = {
                                    "shock_gate_mode": str(mode),
                                    "shock_detector": str(detector),
                                    "shock_direction_source": str(dir_src),
                                    "shock_direction_lookback": int(dir_lb),
                                    "shock_stop_loss_pct_mult": float(sl_mult),
                                    "shock_profit_target_pct_mult": float(pt_mult),
                                    "shock_short_risk_mult_factor": float(short_factor),
                                }
                                overrides.update(params)
                                f = _mk_filters(overrides=overrides)
                                cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
                                note = f"shock={mode} {det_note} | {dir_note} | sl_mult={sl_mult:g} pt_mult={pt_mult:g} short_factor={short_factor:g}"
                                cfg_pairs.append((cfg, note))

        # Unified advanced pocket from centralized axis dimensions.
        shock_dims = _AXIS_DIMENSION_REGISTRY.get("shock", {})
        advanced_modes = tuple(shock_dims.get("advanced_modes") or ())
        advanced_detectors: tuple[tuple[dict[str, object], str], ...] = tuple(shock_dims.get("advanced_detectors") or ())
        advanced_short_factors = tuple(shock_dims.get("advanced_short_risk_factors") or ())
        advanced_long_down_factors = tuple(shock_dims.get("advanced_long_down_factors") or ())
        advanced_scales: tuple[tuple[dict[str, object], str], ...] = tuple(shock_dims.get("advanced_scales") or ())
        for mode in advanced_modes:
            for det_over, det_note in advanced_detectors:
                for dir_src, dir_lb, dir_note in dir_variants:
                    for sl_mult in sl_mults:
                        for short_factor in advanced_short_factors:
                            for long_down in advanced_long_down_factors:
                                for scale_over, scale_note in advanced_scales:
                                    overrides = {
                                        "shock_gate_mode": str(mode),
                                        "shock_direction_source": str(dir_src),
                                        "shock_direction_lookback": int(dir_lb),
                                        "shock_stop_loss_pct_mult": float(sl_mult),
                                        "shock_profit_target_pct_mult": 1.0,
                                        "shock_short_risk_mult_factor": float(short_factor),
                                        "shock_long_risk_mult_factor_down": float(long_down),
                                    }
                                    overrides.update(det_over)
                                    overrides.update(scale_over)
                                    f = _mk_filters(overrides=overrides)
                                    cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
                                    note = (
                                        f"shock_adv={mode} {det_note} | {dir_note} | "
                                        f"sl_mult={sl_mult:g} short_factor={short_factor:g} "
                                        f"long_down={long_down:g} | {scale_note}"
                                    )
                                    cfg_pairs.append((cfg, note))
        tested_total = self._run_cfg_pairs_grid(
            axis_tag="shock",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=50,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(
            rows,
            title="Shock sweep (modes × detectors × thresholds)",
            top_n=int(self.args.top),
        )

    def _sweep_loosen(self) -> None:
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for close_eod in (False, True):
            cfg = replace(
                base,
                strategy=replace(
                    base.strategy,
                    spot_close_eod=bool(close_eod),
                ),
            )
            note = f"close_eod={int(close_eod)}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="loosen", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(
            rows,
            title="Loosenings sweep (single-position + EOD exit)",
            top_n=int(self.args.top),
        )

    def _sweep_tick(self) -> None:
        """Permission layer: Raschke-style $TICK width gate (daily, RTH only)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]
        policies = ["allow", "block"]
        dir_policies = ["both", "wide_only"]
        regime2_variants: list[tuple[dict, str]] = []
        base_r2_mode = str(getattr(base.strategy, "regime2_mode", "off") or "off").strip().lower()
        if base_r2_mode != "off":
            regime2_variants.append(
                (
                    {
                        "regime2_mode": str(getattr(base.strategy, "regime2_mode") or "off"),
                        "regime2_bar_size": getattr(base.strategy, "regime2_bar_size", None),
                        "regime2_supertrend_atr_period": getattr(base.strategy, "regime2_supertrend_atr_period", None),
                        "regime2_supertrend_multiplier": getattr(base.strategy, "regime2_supertrend_multiplier", None),
                        "regime2_supertrend_source": getattr(base.strategy, "regime2_supertrend_source", None),
                    },
                    "r2=base",
                )
            )
        regime2_variants += [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off"),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(4h:3,0.25,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 5,
                    "regime2_supertrend_multiplier": 0.2,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(4h:5,0.2,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "1 day",
                    "regime2_supertrend_atr_period": 7,
                    "regime2_supertrend_multiplier": 0.4,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(1d:7,0.4,close)",
            ),
        ]

        cfg_pairs: list[tuple[ConfigBundle, str]] = [(base, "tick=off (base)")]
        for dir_policy in dir_policies:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                for r2_over, r2_note in regime2_variants:
                                    strat = base.strategy
                                    cfg = replace(
                                        base,
                                        strategy=replace(
                                            strat,
                                            tick_gate_mode="raschke",
                                            tick_gate_symbol="TICK-AMEX",
                                            tick_gate_exchange="AMEX",
                                            tick_neutral_policy=str(policy),
                                            tick_direction_policy=str(dir_policy),
                                            tick_band_ma_period=10,
                                            tick_width_z_lookback=int(lookback),
                                            tick_width_z_enter=float(z_enter),
                                            tick_width_z_exit=float(z_exit),
                                            tick_width_slope_lookback=int(slope_lb),
                                            regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                                            regime2_bar_size=r2_over.get("regime2_bar_size"),
                                            regime2_supertrend_atr_period=int(r2_over.get("regime2_supertrend_atr_period") or 10),
                                            regime2_supertrend_multiplier=float(r2_over.get("regime2_supertrend_multiplier") or 3.0),
                                            regime2_supertrend_source=str(r2_over.get("regime2_supertrend_source") or "hl2"),
                                        ),
                                    )
                                    note = (
                                        f"tick=raschke dir={dir_policy} policy={policy} z_in={z_enter} z_out={z_exit} slope={slope_lb} lb={lookback} {r2_note}"
                                    )
                                    cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(
            axis_tag="tick", cfg_pairs=cfg_pairs, rows=rows, report_every=50, heartbeat_sec=20.0
        ) < 0:
            return
        _print_leaderboards(rows, title="Tick gate sweep ($TICK width)", top_n=int(self.args.top))
