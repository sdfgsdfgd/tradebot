"""SweepHighFrequencyAxes capability slice for the canonical spot research runtime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace

from ...backtest.config import ConfigBundle
from .milestones import _print_leaderboards
from .support import _mk_filters


_OrbKey = tuple[str, int, str, float, float | None]


def _orb_candidates(
    base: ConfigBundle,
    *,
    risk_rewards: Sequence[float],
) -> Iterator[tuple[_OrbKey, ConfigBundle, str]]:
    """Yield the canonical ORB mechanics space shared by standalone and joint sweeps."""
    for open_time, start_h, end_h in (("09:30", 9, 16), ("18:00", 18, 4)):
        for window_mins in (15, 30, 60):
            for target_mode in ("rr", "or_range"):
                for rr in risk_rewards:
                    for vol_min in (None, 1.2):
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                # ORB owns its entry window; do not merge EMA-only gates.
                                filters=_mk_filters(
                                    entry_start_hour_et=start_h,
                                    entry_end_hour_et=end_h,
                                    volume_ratio_min=vol_min,
                                    volume_ema_period=20 if vol_min is not None else None,
                                ),
                                orb_open_time_et=open_time,
                                orb_window_mins=window_mins,
                                orb_risk_reward=float(rr),
                                orb_target_mode=target_mode,
                            ),
                        )
                        key = (open_time, window_mins, target_mode, float(rr), vol_min)
                        vol_note = "-" if vol_min is None else f"vol>={vol_min}@20"
                        note = (
                            f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} "
                            f"tod={start_h:02d}-{end_h:02d} ET {vol_note}"
                        )
                        yield key, cfg, note


class SweepHighFrequencyAxes:
    def _sweep_hold(self) -> None:
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for hold in (0, 1, 2, 3, 4, 6, 8):
            cfg = replace(base, strategy=replace(base.strategy, flip_exit_min_hold_bars=int(hold)))
            note = f"hold={hold}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="hold", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Flip-exit min hold sweep", top_n=int(self.args.top))

    def _sweep_spot_short_risk_mult(self) -> None:
        """Sweep the short sizing multiplier (only affects spot_sizing_mode=risk_pct)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        vals = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0]
        cfg_pairs = [(base, "base")]
        for mult in vals:
            cfg = replace(base, strategy=replace(base.strategy, spot_short_risk_mult=float(mult)))
            note = f"spot_short_risk_mult={mult:g}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="spot_short_risk_mult", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Spot short risk multiplier sweep", top_n=int(self.args.top))

    def _sweep_orb(self) -> None:
        base = self._base_bundle(bar_size="15 mins", filters=None)
        cfg_pairs: list[tuple[ConfigBundle, str]] = [(base, "base")]
        orb_base = replace(
            base,
            strategy=replace(
                base.strategy,
                entry_signal="orb",
                ema_preset=None,
                entry_confirm_bars=0,
                spot_profit_target_pct=None,
                spot_stop_loss_pct=None,
            ),
        )
        cfg_pairs.extend(
            (cfg, note)
            for _key, cfg, note in _orb_candidates(
                orb_base,
                risk_rewards=(0.618, 0.707, 0.786, 1.0, 1.272, 1.618, 2.0),
            )
        )
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(
            axis_tag="orb", cfg_pairs=cfg_pairs, rows=rows, report_every=50, heartbeat_sec=20.0
        ) < 0:
            return
        _print_leaderboards(rows, title="D) ORB sweep (open-time + window)", top_n=int(self.args.top))

    def _sweep_orb_joint(self) -> None:
        """Joint ORB exploration: ORB params × (regime bias) × (optional tick bias).

        Note: ORB uses its own stop/target derived from the opening range, so EMA-based
        quality gates (spread/slope) aren't applicable here unless we compute EMA in
        parallel. We stick to regime/tick/volume/TOD gates that remain well-defined.
        """
        bars_15m = self._bars_cached("15 mins")

        # Start from the selected base shape, but neutralize regime/tick so stage1 can
        # shortlist ORB mechanics without hidden gating.
        base = self._base_bundle(bar_size="15 mins", filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                entry_signal="orb",
                ema_preset=None,
                entry_confirm_bars=0,
                regime_mode="ema",
                regime_bar_size="15 mins",
                regime_ema_preset=None,
                regime2_mode="off",
                regime2_bar_size=None,
                tick_gate_mode="off",
            ),
        )
        base_row = self._run_cfg(
            cfg=base,
            bars=bars_15m,
        )
        if base_row:
            base_row["note"] = "base (orb, no regime/tick)"
            self._record_milestone(base, base_row, str(base_row["note"]))

        # Stage 1: find the best ORB mechanics without regime/tick overlays.
        best_by_orb: dict[tuple, dict] = {}
        for key, cfg, note in _orb_candidates(
            base,
            risk_rewards=(0.618, 0.707, 0.786, 0.8, 1.0, 1.272, 1.618, 2.0),
        ):
            row = self._run_cfg(cfg=cfg, bars=bars_15m)
            if row:
                best_by_orb[key] = {"row": row, "cfg": cfg, "note": note}

        shortlisted = self._ranked_keys_by_row_scores(best_by_orb, top_pnl=8, top_pnl_dd=8)
        if not shortlisted:
            print("No eligible ORB candidates (try lowering --min-trades).")
            return
        print("")
        print(f"ORB×(regime/tick): stage1 shortlisted orb={len(shortlisted)} (from {len(best_by_orb)})")

        # Stage 2: apply a small curated set of regime overlays + tick "wide-only" bias.
        regime_variants: list[tuple[str, dict[str, object]]] = [
            (
                "regime=off",
                {
                    "regime_mode": "ema",
                    "regime_bar_size": "15 mins",
                    "regime_ema_preset": None,
                },
            ),
        ]
        for atr_p, mult, src in (
            (3, 0.4, "hl2"),
            (6, 0.6, "hl2"),
            (7, 0.6, "hl2"),
            (14, 0.6, "hl2"),
            (21, 0.5, "close"),
            (21, 0.6, "hl2"),
        ):
            regime_variants.append(
                (
                    f"ST({atr_p},{mult:g},{src})@4h",
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "4 hours",
                        "supertrend_atr_period": int(atr_p),
                        "supertrend_multiplier": float(mult),
                        "supertrend_source": str(src),
                    },
                )
            )

        tick_variants: list[tuple[str, dict[str, object]]] = [
            ("tick=off", {"tick_gate_mode": "off"}),
            (
                "tick=wide_only allow (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "allow",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
            (
                "tick=wide_only block (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "block",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
        ]

        rows: list[dict] = []
        for key in shortlisted:
            seed = best_by_orb[key]
            seed_cfg = seed["cfg"]
            for regime_note, reg_over in regime_variants:
                for tick_note, tick_over in tick_variants:
                    cfg = replace(
                        seed_cfg,
                        strategy=replace(
                            seed_cfg.strategy,
                            regime_mode=str(reg_over.get("regime_mode") or "ema"),
                            regime_bar_size=str(reg_over.get("regime_bar_size") or "15 mins"),
                            regime_ema_preset=reg_over.get("regime_ema_preset"),
                            supertrend_atr_period=int(reg_over.get("supertrend_atr_period") or 10),
                            supertrend_multiplier=float(reg_over.get("supertrend_multiplier") or 3.0),
                            supertrend_source=str(reg_over.get("supertrend_source") or "hl2"),
                            tick_gate_mode=str(tick_over.get("tick_gate_mode") or "off"),
                            tick_gate_symbol=str(tick_over.get("tick_gate_symbol") or "TICK-NYSE"),
                            tick_gate_exchange=str(tick_over.get("tick_gate_exchange") or "NYSE"),
                            tick_neutral_policy=str(tick_over.get("tick_neutral_policy") or "allow"),
                            tick_direction_policy=str(tick_over.get("tick_direction_policy") or "both"),
                            tick_band_ma_period=int(tick_over.get("tick_band_ma_period") or 10),
                            tick_width_z_lookback=int(tick_over.get("tick_width_z_lookback") or 252),
                            tick_width_z_enter=float(tick_over.get("tick_width_z_enter") or 1.0),
                            tick_width_z_exit=float(tick_over.get("tick_width_z_exit") or 0.5),
                            tick_width_slope_lookback=int(tick_over.get("tick_width_slope_lookback") or 3),
                        ),
                    )
                    row = self._run_cfg(
                        cfg=cfg,
                        bars=bars_15m,
                    )
                    if not row:
                        continue
                    note = f"{seed['note']} | {regime_note} | {tick_note}"
                    row["note"] = note
                    self._record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(
            rows,
            title="ORB joint sweep (ORB × regime × tick)",
            top_n=int(self.args.top),
        )
