"""Canonical live signal evaluation and snapshot caching."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime

from ib_insync import Contract

from ...chart_data.cache import series_cache_service
from ...engine import (
    normalize_spot_entry_signal,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
)
from ...time_utils import now_et_naive as _now_et_naive
from ..bot_models import _SignalSnapshot


_SERIES_CACHE = series_cache_service()
_UI_SIGNAL_SNAPSHOT_NAMESPACE = "ui.signal.snapshot.v1"


class BotSignalSnapshotMixin:
    def _signal_regime_spec(
        self,
        *,
        regime_mode_raw: str | None,
        regime_ema_preset_raw: str | None,
        regime_bar_size_raw: str | None,
        bar_size: str,
    ) -> tuple[str, str | None, str, bool]:
        return resolve_spot_regime_spec(
            bar_size=bar_size,
            regime_mode_raw=regime_mode_raw,
            regime_ema_preset_raw=regime_ema_preset_raw,
            regime_bar_size_raw=regime_bar_size_raw,
        )

    def _signal_regime2_spec(
        self,
        *,
        regime2_mode_raw: str | None,
        regime2_ema_preset_raw: str | None,
        regime2_bar_size_raw: str | None,
        bar_size: str,
    ) -> tuple[str, str | None, str, bool]:
        return resolve_spot_regime2_spec(
            bar_size=bar_size,
            regime2_mode_raw=regime2_mode_raw,
            regime2_ema_preset_raw=regime2_ema_preset_raw,
            regime2_bar_size_raw=regime2_bar_size_raw,
        )

    def _signal_regime_duration(
        self,
        *,
        regime_duration: str,
        regime_bar_size: str,
        filters: dict | None,
    ) -> str:
        if not isinstance(filters, dict) or "hour" not in str(regime_bar_size).strip().lower():
            return regime_duration
        from ...engines.shock import normalize_shock_gate_mode

        shock_gate_mode = normalize_shock_gate_mode(filters)
        if shock_gate_mode == "off":
            return regime_duration
        try:
            atr_slow = int(filters.get("shock_atr_slow_period", 50))
        except (TypeError, ValueError):
            atr_slow = 50
        if atr_slow <= 0:
            return regime_duration
        alt = "1 M" if atr_slow <= 60 else "2 M"
        order = ("1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
        try:
            return alt if order.index(str(alt)) > order.index(str(regime_duration)) else regime_duration
        except ValueError:
            return regime_duration

    def _signal_strategy_payload(
        self,
        *,
        base_strategy_raw: dict | None = None,
        entry_signal: str,
        ema_preset_raw: str | None,
        entry_mode_raw: str | None,
        entry_confirm_bars: int,
        spot_dual_branch_enabled_raw: object | None,
        spot_dual_branch_priority_raw: object | None,
        spot_branch_a_ema_preset_raw: object | None,
        spot_branch_a_entry_confirm_bars_raw: object | None,
        spot_branch_a_min_signed_slope_pct_raw: object | None,
        spot_branch_a_max_signed_slope_pct_raw: object | None,
        spot_branch_a_size_mult_raw: object | None,
        spot_branch_b_ema_preset_raw: object | None,
        spot_branch_b_entry_confirm_bars_raw: object | None,
        spot_branch_b_min_signed_slope_pct_raw: object | None,
        spot_branch_b_max_signed_slope_pct_raw: object | None,
        spot_branch_b_size_mult_raw: object | None,
        orb_window_mins_raw: int | None,
        orb_open_time_et_raw: str | None,
        spot_exit_mode_raw: str | None,
        spot_atr_period_raw: int | None,
        regime_mode: str,
        regime_preset: str | None,
        supertrend_atr_period_raw: int | None,
        supertrend_multiplier_raw: float | None,
        supertrend_source_raw: str | None,
        regime2_mode: str,
        regime2_preset: str | None,
        regime2_supertrend_atr_period_raw: int | None,
        regime2_supertrend_multiplier_raw: float | None,
        regime2_supertrend_source_raw: str | None,
    ) -> dict:
        strategy = copy.deepcopy(base_strategy_raw) if isinstance(base_strategy_raw, dict) else {}
        strategy["entry_signal"] = entry_signal
        strategy["regime_mode"] = regime_mode
        strategy["regime2_mode"] = regime2_mode
        try:
            strategy["entry_confirm_bars"] = int(entry_confirm_bars)
        except (TypeError, ValueError):
            strategy["entry_confirm_bars"] = 0
        if regime_mode == "ema" or regime_preset is not None:
            strategy["regime_ema_preset"] = regime_preset
        if regime2_mode == "ema" or regime2_preset is not None:
            strategy["regime2_ema_preset"] = regime2_preset

        optional_values: dict[str, object] = {
            "ema_preset": ema_preset_raw,
            "ema_entry_mode": entry_mode_raw,
            "spot_dual_branch_enabled": spot_dual_branch_enabled_raw,
            "spot_dual_branch_priority": spot_dual_branch_priority_raw,
            "spot_branch_a_ema_preset": spot_branch_a_ema_preset_raw,
            "spot_branch_a_entry_confirm_bars": spot_branch_a_entry_confirm_bars_raw,
            "spot_branch_a_min_signed_slope_pct": spot_branch_a_min_signed_slope_pct_raw,
            "spot_branch_a_max_signed_slope_pct": spot_branch_a_max_signed_slope_pct_raw,
            "spot_branch_a_size_mult": spot_branch_a_size_mult_raw,
            "spot_branch_b_ema_preset": spot_branch_b_ema_preset_raw,
            "spot_branch_b_entry_confirm_bars": spot_branch_b_entry_confirm_bars_raw,
            "spot_branch_b_min_signed_slope_pct": spot_branch_b_min_signed_slope_pct_raw,
            "spot_branch_b_max_signed_slope_pct": spot_branch_b_max_signed_slope_pct_raw,
            "spot_branch_b_size_mult": spot_branch_b_size_mult_raw,
            "orb_window_mins": orb_window_mins_raw,
            "orb_open_time_et": orb_open_time_et_raw,
            "spot_exit_mode": spot_exit_mode_raw,
            "spot_atr_period": spot_atr_period_raw,
            "supertrend_atr_period": supertrend_atr_period_raw,
            "supertrend_multiplier": supertrend_multiplier_raw,
            "supertrend_source": supertrend_source_raw,
            "regime2_supertrend_atr_period": regime2_supertrend_atr_period_raw,
            "regime2_supertrend_multiplier": regime2_supertrend_multiplier_raw,
            "regime2_supertrend_source": regime2_supertrend_source_raw,
        }
        for key, value in optional_values.items():
            if value is not None:
                strategy[key] = value
        return strategy

    def _signal_eval_last_snapshot(self, *, evaluator: object, bars: list) -> object | None:
        last_snap = None
        trade_day_fn = getattr(evaluator, "_trade_date", None)
        for idx, bar in enumerate(bars):
            next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
            if next_bar is None:
                is_last_bar = True
            elif callable(trade_day_fn):
                is_last_bar = bool(trade_day_fn(next_bar.ts) != trade_day_fn(bar.ts))
            else:
                is_last_bar = next_bar.ts.date() != bar.ts.date()
            evaluator.update_exec_bar(bar, is_last_bar=bool(is_last_bar))
            snap = evaluator.update_signal_bar(bar)
            if snap is not None:
                last_snap = snap
        return last_snap

    def _signal_snapshot_from_eval(
        self,
        snap: object,
        *,
        bar_health: dict | None = None,
        regime_bar_health: dict | None = None,
        regime2_bar_health: dict | None = None,
    ) -> _SignalSnapshot:
        return _SignalSnapshot(
            bar_ts=snap.bar_ts,
            close=float(snap.close),
            signal=snap.signal,
            bars_in_day=int(snap.bars_in_day),
            rv=float(snap.rv) if snap.rv is not None else None,
            volume=float(snap.volume) if snap.volume is not None else None,
            volume_ema=float(snap.volume_ema) if snap.volume_ema is not None else None,
            volume_ema_ready=bool(snap.volume_ema_ready),
            shock=snap.shock,
            shock_dir=snap.shock_dir,
            shock_detector=str(getattr(snap, "shock_detector", "") or "") or None,
            shock_direction_source_effective=(
                str(getattr(snap, "shock_direction_source_effective", "") or "") or None
            ),
            shock_scale_detector=str(getattr(snap, "shock_scale_detector", "") or "") or None,
            shock_dir_ret_sum_pct=(
                float(getattr(snap, "shock_dir_ret_sum_pct", 0.0))
                if getattr(snap, "shock_dir_ret_sum_pct", None) is not None
                else None
            ),
            shock_atr_pct=float(snap.shock_atr_pct) if snap.shock_atr_pct is not None else None,
            shock_drawdown_pct=(
                float(getattr(snap, "shock_drawdown_pct", 0.0))
                if getattr(snap, "shock_drawdown_pct", None) is not None
                else None
            ),
            shock_drawdown_on_pct=(
                float(getattr(snap, "shock_drawdown_on_pct", 0.0))
                if getattr(snap, "shock_drawdown_on_pct", None) is not None
                else None
            ),
            shock_drawdown_off_pct=(
                float(getattr(snap, "shock_drawdown_off_pct", 0.0))
                if getattr(snap, "shock_drawdown_off_pct", None) is not None
                else None
            ),
            shock_drawdown_dist_on_pct=(
                float(getattr(snap, "shock_drawdown_dist_on_pct", 0.0))
                if getattr(snap, "shock_drawdown_dist_on_pct", None) is not None
                else None
            ),
            shock_drawdown_dist_on_vel_pp=(
                float(getattr(snap, "shock_drawdown_dist_on_vel_pp", 0.0))
                if getattr(snap, "shock_drawdown_dist_on_vel_pp", None) is not None
                else None
            ),
            shock_drawdown_dist_on_accel_pp=(
                float(getattr(snap, "shock_drawdown_dist_on_accel_pp", 0.0))
                if getattr(snap, "shock_drawdown_dist_on_accel_pp", None) is not None
                else None
            ),
            shock_prearm_down_streak_bars=(
                int(getattr(snap, "shock_prearm_down_streak_bars", 0))
                if getattr(snap, "shock_prearm_down_streak_bars", None) is not None
                else None
            ),
            shock_drawdown_dist_off_pct=(
                float(getattr(snap, "shock_drawdown_dist_off_pct", 0.0))
                if getattr(snap, "shock_drawdown_dist_off_pct", None) is not None
                else None
            ),
            shock_scale_drawdown_pct=(
                float(getattr(snap, "shock_scale_drawdown_pct", 0.0))
                if getattr(snap, "shock_scale_drawdown_pct", None) is not None
                else None
            ),
            shock_peak_close=(
                float(getattr(snap, "shock_peak_close", 0.0))
                if getattr(snap, "shock_peak_close", None) is not None
                else None
            ),
            shock_dir_down_streak_bars=(
                int(getattr(snap, "shock_dir_down_streak_bars", 0))
                if getattr(snap, "shock_dir_down_streak_bars", None) is not None
                else None
            ),
            shock_dir_up_streak_bars=(
                int(getattr(snap, "shock_dir_up_streak_bars", 0))
                if getattr(snap, "shock_dir_up_streak_bars", None) is not None
                else None
            ),
            risk=snap.risk,
            atr=float(snap.atr) if snap.atr is not None else None,
            or_high=float(snap.or_high) if snap.or_high is not None else None,
            or_low=float(snap.or_low) if snap.or_low is not None else None,
            or_ready=bool(snap.or_ready),
            entry_dir=str(snap.entry_dir) if getattr(snap, "entry_dir", None) in ("up", "down") else None,
            entry_branch=str(snap.entry_branch) if getattr(snap, "entry_branch", None) in ("a", "b") else None,
            ratsv_side_rank=float(snap.ratsv_side_rank) if getattr(snap, "ratsv_side_rank", None) is not None else None,
            ratsv_tr_ratio=float(snap.ratsv_tr_ratio) if getattr(snap, "ratsv_tr_ratio", None) is not None else None,
            ratsv_fast_slope_pct=(
                float(snap.ratsv_fast_slope_pct) if getattr(snap, "ratsv_fast_slope_pct", None) is not None else None
            ),
            ratsv_fast_slope_med_pct=(
                float(snap.ratsv_fast_slope_med_pct)
                if getattr(snap, "ratsv_fast_slope_med_pct", None) is not None
                else None
            ),
            ratsv_fast_slope_vel_pct=(
                float(snap.ratsv_fast_slope_vel_pct)
                if getattr(snap, "ratsv_fast_slope_vel_pct", None) is not None
                else None
            ),
            ratsv_slow_slope_med_pct=(
                float(snap.ratsv_slow_slope_med_pct)
                if getattr(snap, "ratsv_slow_slope_med_pct", None) is not None
                else None
            ),
            ratsv_slow_slope_vel_pct=(
                float(snap.ratsv_slow_slope_vel_pct)
                if getattr(snap, "ratsv_slow_slope_vel_pct", None) is not None
                else None
            ),
            ratsv_slope_vel_consistency=(
                float(snap.ratsv_slope_vel_consistency)
                if getattr(snap, "ratsv_slope_vel_consistency", None) is not None
                else None
            ),
            ratsv_cross_age_bars=(
                int(snap.ratsv_cross_age_bars) if getattr(snap, "ratsv_cross_age_bars", None) is not None else None
            ),
            shock_atr_vel_pct=(
                float(snap.shock_atr_vel_pct) if getattr(snap, "shock_atr_vel_pct", None) is not None else None
            ),
            shock_atr_accel_pct=(
                float(snap.shock_atr_accel_pct) if getattr(snap, "shock_atr_accel_pct", None) is not None else None
            ),
            shock_ramp=(
                dict(getattr(snap, "shock_ramp"))
                if isinstance(getattr(snap, "shock_ramp", None), dict)
                else None
            ),
            regime_router_ready=bool(getattr(snap, "regime_router_ready", False)),
            regime_router_climate=(
                str(getattr(snap, "regime_router_climate", "") or "") or None
            ),
            regime_router_host=(
                str(getattr(snap, "regime_router_host", "") or "") or None
            ),
            regime_router_entry_dir=(
                str(getattr(snap, "regime_router_entry_dir", None))
                if getattr(snap, "regime_router_entry_dir", None) in ("up", "down")
                else None
            ),
            regime_router_host_managed=bool(getattr(snap, "regime_router_host_managed", False)),
            regime_router_bull_sovereign_ok=bool(getattr(snap, "regime_router_bull_sovereign_ok", False)),
            regime2_dir=(
                str(getattr(snap, "regime2_dir"))
                if getattr(snap, "regime2_dir", None) in ("up", "down")
                else None
            ),
            regime2_bear_hard_dir=(
                str(getattr(snap, "regime2_bear_hard_dir"))
                if getattr(snap, "regime2_bear_hard_dir", None) in ("up", "down")
                else None
            ),
            regime4_state=str(getattr(snap, "regime4_state", "") or "") or None,
            regime4_owner=str(getattr(snap, "regime4_owner", "") or "") or None,
            regime_router_dwell_days=(
                int(getattr(snap, "regime_router_dwell_days", 0))
                if getattr(snap, "regime_router_dwell_days", None) is not None
                else None
            ),
            regime_router_crash_ret=(
                float(getattr(snap, "regime_router_crash_ret"))
                if getattr(snap, "regime_router_crash_ret", None) is not None
                else None
            ),
            regime_router_crash_maxdd=(
                float(getattr(snap, "regime_router_crash_maxdd"))
                if getattr(snap, "regime_router_crash_maxdd", None) is not None
                else None
            ),
            regime_router_crash_rv=(
                float(getattr(snap, "regime_router_crash_rv"))
                if getattr(snap, "regime_router_crash_rv", None) is not None
                else None
            ),
            regime_router_fast_ret=(
                float(getattr(snap, "regime_router_fast_ret"))
                if getattr(snap, "regime_router_fast_ret", None) is not None
                else None
            ),
            regime_router_slow_ret=(
                float(getattr(snap, "regime_router_slow_ret"))
                if getattr(snap, "regime_router_slow_ret", None) is not None
                else None
            ),
            bar_health=bar_health,
            regime_bar_health=regime_bar_health,
            regime2_bar_health=regime2_bar_health,
        )

    @staticmethod
    def _signal_health_payload(bar_health: dict | None) -> dict | None:
        if not isinstance(bar_health, dict):
            return None
        payload: dict[str, object] = {}
        for key, value in bar_health.items():
            if isinstance(value, datetime):
                payload[str(key)] = value.isoformat()
            elif isinstance(value, list):
                payload[str(key)] = list(value)
            else:
                payload[str(key)] = value
        return payload

    @staticmethod
    def _stable_json_key(payload: object) -> str:
        try:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return repr(payload)

    async def _signal_snapshot_for_contract(
        self,
        *,
        contract: Contract,
        ema_preset_raw: str | None,
        bar_size: str,
        use_rth: bool,
        entry_signal_raw: str | None = None,
        orb_window_mins_raw: int | None = None,
        orb_open_time_et_raw: str | None = None,
        entry_mode_raw: str | None = None,
        entry_confirm_bars: int = 0,
        spot_dual_branch_enabled_raw: object | None = None,
        spot_dual_branch_priority_raw: object | None = None,
        spot_branch_a_ema_preset_raw: object | None = None,
        spot_branch_a_entry_confirm_bars_raw: object | None = None,
        spot_branch_a_min_signed_slope_pct_raw: object | None = None,
        spot_branch_a_max_signed_slope_pct_raw: object | None = None,
        spot_branch_a_size_mult_raw: object | None = None,
        spot_branch_b_ema_preset_raw: object | None = None,
        spot_branch_b_entry_confirm_bars_raw: object | None = None,
        spot_branch_b_min_signed_slope_pct_raw: object | None = None,
        spot_branch_b_max_signed_slope_pct_raw: object | None = None,
        spot_branch_b_size_mult_raw: object | None = None,
        spot_exit_mode_raw: str | None = None,
        spot_atr_period_raw: int | None = None,
        regime_ema_preset_raw: str | None = None,
        regime_bar_size_raw: str | None = None,
        regime_mode_raw: str | None = None,
        supertrend_atr_period_raw: int | None = None,
        supertrend_multiplier_raw: float | None = None,
        supertrend_source_raw: str | None = None,
        regime2_ema_preset_raw: str | None = None,
        regime2_bar_size_raw: str | None = None,
        regime2_mode_raw: str | None = None,
        regime2_supertrend_atr_period_raw: int | None = None,
        regime2_supertrend_multiplier_raw: float | None = None,
        regime2_supertrend_source_raw: str | None = None,
        base_strategy_raw: dict | None = None,
        filters: dict | None = None,
    ) -> _SignalSnapshot | None:
        from ...spot_engine import SpotSignalEvaluator

        entry_signal = normalize_spot_entry_signal(entry_signal_raw)
        strict_zero_gap = self._signal_zero_gap_enabled(filters)
        diag: dict[str, object] = {
            "stage": "init",
            "bar_size": str(bar_size),
            "use_rth": bool(use_rth),
            "entry_signal": str(entry_signal),
            "strict_zero_gap": bool(strict_zero_gap),
            "proxy_error": self._client.proxy_error(),
            "historical_request": self._client.last_historical_request(contract),
        }

        def _set_diag(stage: str, **extra: object) -> None:
            payload = dict(diag)
            payload["stage"] = str(stage)
            payload.update(extra)
            payload["proxy_error"] = self._client.proxy_error()
            payload["historical_request"] = self._client.last_historical_request(contract)
            self._last_signal_snapshot_diag = payload

        # IB intraday bars are timestamped in ET wall-clock for this flow; use ET here so
        # trim_incomplete_last_bar drops the in-progress bar instead of treating it as complete.
        now_ref = _now_et_naive()
        min_duration_str = self._signal_min_duration_str(
            bar_size,
            filters=filters,
            strategy=base_strategy_raw,
            use_rth=use_rth,
        )
        _set_diag(
            "signal_fetch",
            min_duration_str=str(min_duration_str) if min_duration_str is not None else None,
        )
        bars, bar_health = await self._signal_fetch_bars(
            contract=contract,
            duration_str=self._signal_duration_str(
                bar_size,
                filters=filters,
                strategy=base_strategy_raw,
                use_rth=use_rth,
            ),
            min_duration_str=min_duration_str,
            bar_size=bar_size,
            use_rth=use_rth,
            now_ref=now_ref,
            strict_zero_gap=bool(strict_zero_gap),
            heal_if_stale=True,
        )
        if bars is None or len(bars) == 0:
            _set_diag(
                "signal_bars",
                bar_health=self._signal_health_payload(bar_health),
            )
            return None

        regime_mode, regime_preset, regime_bar_size, use_mtf_regime = self._signal_regime_spec(
            regime_mode_raw=regime_mode_raw,
            regime_ema_preset_raw=regime_ema_preset_raw,
            regime_bar_size_raw=regime_bar_size_raw,
            bar_size=bar_size,
        )

        regime_bars = None
        regime_health = None
        if use_mtf_regime:
            regime_duration = self._signal_regime_duration(
                regime_duration=self._signal_duration_str(regime_bar_size, filters=filters),
                regime_bar_size=regime_bar_size,
                filters=filters,
            )
            _set_diag(
                "regime_fetch",
                regime_bar_size=str(regime_bar_size),
                regime_duration=str(regime_duration),
                bar_health=self._signal_health_payload(bar_health),
            )
            regime_bars, regime_health = await self._signal_fetch_bars(
                contract=contract,
                duration_str=regime_duration,
                bar_size=regime_bar_size,
                use_rth=use_rth,
                now_ref=now_ref,
                strict_zero_gap=bool(strict_zero_gap),
                heal_if_stale=True,
            )
            if regime_bars is None or len(regime_bars) == 0:
                _set_diag(
                    "regime_bars",
                    regime_bar_size=str(regime_bar_size),
                    regime_duration=str(regime_duration),
                    bar_health=self._signal_health_payload(bar_health),
                    regime_bar_health=self._signal_health_payload(regime_health),
                )
                return None

        regime2_mode, regime2_preset, regime2_bar_size, use_mtf_regime2 = self._signal_regime2_spec(
            regime2_mode_raw=regime2_mode_raw,
            regime2_ema_preset_raw=regime2_ema_preset_raw,
            regime2_bar_size_raw=regime2_bar_size_raw,
            bar_size=bar_size,
        )

        regime2_bars = None
        regime2_health = None
        if regime2_mode != "off" and use_mtf_regime2:
            regime2_duration = self._signal_duration_str(regime2_bar_size, filters=filters)
            _set_diag(
                "regime2_fetch",
                regime2_mode=str(regime2_mode),
                regime2_bar_size=str(regime2_bar_size),
                regime2_duration=str(regime2_duration),
                bar_health=self._signal_health_payload(bar_health),
                regime_bar_health=self._signal_health_payload(regime_health),
            )
            regime2_bars, regime2_health = await self._signal_fetch_bars(
                contract=contract,
                duration_str=regime2_duration,
                bar_size=regime2_bar_size,
                use_rth=use_rth,
                now_ref=now_ref,
                strict_zero_gap=bool(strict_zero_gap),
                heal_if_stale=True,
            )
            if regime2_bars is None or len(regime2_bars) == 0:
                _set_diag(
                    "regime2_bars",
                    regime2_mode=str(regime2_mode),
                    regime2_bar_size=str(regime2_bar_size),
                    bar_health=self._signal_health_payload(bar_health),
                    regime_bar_health=self._signal_health_payload(regime_health),
                    regime2_bar_health=self._signal_health_payload(regime2_health),
                )
                return None
        strategy = self._signal_strategy_payload(
            base_strategy_raw=base_strategy_raw,
            entry_signal=entry_signal,
            ema_preset_raw=ema_preset_raw,
            entry_mode_raw=entry_mode_raw,
            entry_confirm_bars=entry_confirm_bars,
            spot_dual_branch_enabled_raw=spot_dual_branch_enabled_raw,
            spot_dual_branch_priority_raw=spot_dual_branch_priority_raw,
            spot_branch_a_ema_preset_raw=spot_branch_a_ema_preset_raw,
            spot_branch_a_entry_confirm_bars_raw=spot_branch_a_entry_confirm_bars_raw,
            spot_branch_a_min_signed_slope_pct_raw=spot_branch_a_min_signed_slope_pct_raw,
            spot_branch_a_max_signed_slope_pct_raw=spot_branch_a_max_signed_slope_pct_raw,
            spot_branch_a_size_mult_raw=spot_branch_a_size_mult_raw,
            spot_branch_b_ema_preset_raw=spot_branch_b_ema_preset_raw,
            spot_branch_b_entry_confirm_bars_raw=spot_branch_b_entry_confirm_bars_raw,
            spot_branch_b_min_signed_slope_pct_raw=spot_branch_b_min_signed_slope_pct_raw,
            spot_branch_b_max_signed_slope_pct_raw=spot_branch_b_max_signed_slope_pct_raw,
            spot_branch_b_size_mult_raw=spot_branch_b_size_mult_raw,
            orb_window_mins_raw=orb_window_mins_raw,
            orb_open_time_et_raw=orb_open_time_et_raw,
            spot_exit_mode_raw=spot_exit_mode_raw,
            spot_atr_period_raw=spot_atr_period_raw,
            regime_mode=regime_mode,
            regime_preset=regime_preset,
            supertrend_atr_period_raw=supertrend_atr_period_raw,
            supertrend_multiplier_raw=supertrend_multiplier_raw,
            supertrend_source_raw=supertrend_source_raw,
            regime2_mode=regime2_mode,
            regime2_preset=regime2_preset,
            regime2_supertrend_atr_period_raw=regime2_supertrend_atr_period_raw,
            regime2_supertrend_multiplier_raw=regime2_supertrend_multiplier_raw,
            regime2_supertrend_source_raw=regime2_supertrend_source_raw,
        )

        regime_router_seed = None
        regime_router_seed_health = None
        if bool(strategy.get("regime_router")):
            slow_days = self._int_from(
                strategy.get("regime_router_slow_window_days"),
                default=0,
                min_value=0,
            )

            warmup_days = int(slow_days) + 5
            if bool(use_rth) and slow_days > 0:
                warmup_days = int(math.ceil(float(slow_days) * (7.0 / 5.0))) + 7
            daily_duration = self._duration_for_days(max(2, int(warmup_days)))
            _set_diag(
                "regime_router_seed_fetch",
                regime_router_seed_bar_size="1 day",
                regime_router_seed_duration=str(daily_duration),
                bar_health=self._signal_health_payload(bar_health),
            )
            regime_router_seed, regime_router_seed_health = await self._signal_fetch_bars(
                contract=contract,
                duration_str=str(daily_duration),
                bar_size="1 day",
                use_rth=use_rth,
                now_ref=now_ref,
                strict_zero_gap=False,
                heal_if_stale=False,
            )
            if regime_router_seed is None or len(regime_router_seed) == 0:
                _set_diag(
                    "regime_router_seed_bars",
                    regime_router_seed_bar_size="1 day",
                    regime_router_seed_duration=str(daily_duration),
                    bar_health=self._signal_health_payload(bar_health),
                    regime_router_seed_health=self._signal_health_payload(regime_router_seed_health),
                )

        bars_list = self._signal_series_list(bars)
        regime_bars_list = self._signal_series_list(regime_bars)
        regime2_bars_list = self._signal_series_list(regime2_bars)
        regime_router_seed_list = self._signal_series_list(regime_router_seed)
        snapshot_key = (
            int(getattr(contract, "conId", 0) or 0),
            str(getattr(contract, "symbol", "") or "").strip().upper(),
            str(bar_size),
            bool(use_rth),
            self._signal_series_signature(bars),
            self._signal_series_signature(regime_bars),
            self._signal_series_signature(regime2_bars),
            self._signal_series_signature(regime_router_seed),
            self._stable_json_key(strategy),
            self._stable_json_key(filters or {}),
        )
        cached_snapshot = _SERIES_CACHE.get(
            namespace=_UI_SIGNAL_SNAPSHOT_NAMESPACE,
            key=snapshot_key,
        )
        if isinstance(cached_snapshot, _SignalSnapshot):
            _set_diag(
                "snapshot_cache_hit",
                bars_count=int(len(bars_list)),
                regime_bars_count=int(len(regime_bars_list)),
                regime2_bars_count=int(len(regime2_bars_list)),
                regime_router_seed_count=int(len(regime_router_seed_list)),
                bar_health=self._signal_health_payload(bar_health),
            )
            return cached_snapshot

        evaluator = SpotSignalEvaluator(
            strategy=strategy,
            filters=filters,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            naive_ts_mode="et",
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            regime_router_seed_days=regime_router_seed,
        )

        _set_diag(
            "eval",
            bars_count=int(len(bars_list)),
            regime_bars_count=int(len(regime_bars_list)),
            regime2_bars_count=int(len(regime2_bars_list)),
            regime_router_seed_count=int(len(regime_router_seed_list)),
            bar_health=self._signal_health_payload(bar_health),
            regime_bar_health=self._signal_health_payload(regime_health),
            regime2_bar_health=self._signal_health_payload(regime2_health),
            regime_router_seed_health=self._signal_health_payload(regime_router_seed_health),
        )
        last_snap = self._signal_eval_last_snapshot(evaluator=evaluator, bars=bars_list)
        if last_snap is None:
            _set_diag(
                "eval_no_snapshot",
                bars_count=int(len(bars_list)),
                regime_bars_count=int(len(regime_bars_list)),
                regime2_bars_count=int(len(regime2_bars_list)),
                bar_health=self._signal_health_payload(bar_health),
            )
            return None
        if not bool(last_snap.signal.ema_ready):
            _set_diag(
                "eval_ema_not_ready",
                bars_count=int(len(bars_list)),
                last_bar_ts=(last_snap.bar_ts.isoformat() if isinstance(last_snap.bar_ts, datetime) else None),
                signal_state=str(last_snap.signal.state or ""),
                entry_dir=str(last_snap.signal.entry_dir or ""),
                regime_dir=str(last_snap.signal.regime_dir or ""),
                bar_health=self._signal_health_payload(bar_health),
            )
            return None

        if bool(strategy.get("regime_router")) and not bool(getattr(last_snap, "regime_router_ready", False)):
            _set_diag(
                "regime_router_not_ready",
                bars_count=int(len(bars_list)),
                min_duration_str=str(min_duration_str) if min_duration_str is not None else None,
                bar_health=self._signal_health_payload(bar_health),
            )
            return None

        # Non-negotiable readiness: if a daily shock detector is configured, do not emit a
        # snapshot until the daily engine is actually ready. Silent degradation here breaks
        # live-vs-backtest parity because shock gating/overlays won't engage.
        from ...engines.shock import normalize_shock_detector, normalize_shock_gate_mode

        shock_mode = normalize_shock_gate_mode(filters)
        if shock_mode != "off":
            detector = normalize_shock_detector(filters)
            if detector == "daily_drawdown" and getattr(last_snap, "shock_drawdown_pct", None) is None:
                _set_diag(
                    "daily_shock_not_ready",
                    shock_detector=str(detector),
                    bars_count=int(len(bars_list)),
                    min_duration_str=str(min_duration_str) if min_duration_str is not None else None,
                    bar_health=self._signal_health_payload(bar_health),
                )
                return None
            if detector == "daily_atr_pct" and getattr(last_snap, "shock_atr_pct", None) is None:
                _set_diag(
                    "daily_shock_not_ready",
                    shock_detector=str(detector),
                    bars_count=int(len(bars_list)),
                    min_duration_str=str(min_duration_str) if min_duration_str is not None else None,
                    bar_health=self._signal_health_payload(bar_health),
                )
                return None

        _set_diag(
            "ok",
            bars_count=int(len(bars_list)),
            regime_bars_count=int(len(regime_bars_list)),
            regime2_bars_count=int(len(regime2_bars_list)),
            regime_router_seed_count=int(len(regime_router_seed_list)),
            bar_health=self._signal_health_payload(bar_health),
            regime_bar_health=self._signal_health_payload(regime_health),
            regime2_bar_health=self._signal_health_payload(regime2_health),
            regime_router_seed_health=self._signal_health_payload(regime_router_seed_health),
            bar_ts=last_snap.bar_ts.isoformat() if isinstance(last_snap.bar_ts, datetime) else None,
        )
        snapshot = self._signal_snapshot_from_eval(
            last_snap,
            bar_health=bar_health,
            regime_bar_health=regime_health,
            regime2_bar_health=regime2_health,
        )
        _SERIES_CACHE.set(namespace=_UI_SIGNAL_SNAPSHOT_NAMESPACE, key=snapshot_key, value=snapshot)
        return snapshot
