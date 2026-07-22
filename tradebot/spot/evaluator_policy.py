"""Entry selection and risk-policy decisions for spot signals."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import date, datetime
from statistics import median

from ..engine import _trade_date as _trade_date_shared, _trade_hour_et as _trade_hour_et_shared
from ..engines.risk import RiskOverlaySnapshot
from ..engines.signals import EmaDecisionSnapshot, OrbDecisionEngine
from .evaluator_common import BarLike, SpotSignalSnapshot


class SpotSignalPolicyMixin:
    @property
    def last_snapshot(self) -> SpotSignalSnapshot | None:
        return self._last_snapshot

    @property
    def shock_enabled(self) -> bool:
        return self._shock_engine is not None

    @property
    def shock_view(self) -> tuple[bool | None, str | None, float | None]:
        return self._shock_view()

    @property
    def risk_overlay_enabled(self) -> bool:
        return self._risk_overlay is not None

    @property
    def last_risk(self) -> RiskOverlaySnapshot | None:
        return self._last_risk

    @property
    def orb_engine(self) -> OrbDecisionEngine | None:
        return self._orb_engine

    @staticmethod
    def _signed_fast_slope_pct(signal: EmaDecisionSnapshot, close: float) -> float | None:
        if signal.ema_fast is None or signal.prev_ema_fast is None:
            return None
        if close <= 0:
            return None
        return float(signal.ema_fast - signal.prev_ema_fast) / float(close)

    @staticmethod
    def _median(values: deque[float] | list[float]) -> float | None:
        if not values:
            return None
        try:
            return float(median(values))
        except Exception:
            return None

    def _trade_date(self, ts: datetime) -> date:
        return _trade_date_shared(ts, naive_ts_mode=self._naive_ts_mode)

    def _ratsv_tr_fast_slow(self) -> tuple[float | None, float | None]:
        tr_hist = list(self._ratsv_tr_pct_hist)
        if not tr_hist:
            return None, None
        fast = self._median(tr_hist[-int(self._ratsv_tr_fast) :])
        slow = self._median(tr_hist[-int(self._ratsv_tr_slow) :])
        return fast, slow

    def _ratsv_update_bar_metrics(self, *, high: float, low: float, close: float) -> None:
        prev_close = self._ratsv_prev_tr_close
        self._ratsv_prev_tr_close = float(close)
        if close <= 0:
            return
        if prev_close is None or prev_close <= 0:
            tr = max(0.0, float(high) - float(low))
        else:
            tr = max(
                0.0,
                float(high) - float(low),
                abs(float(high) - float(prev_close)),
                abs(float(low) - float(prev_close)),
            )
        tr_pct = float(tr) / max(float(close), 1e-9)
        self._ratsv_tr_pct_hist.append(float(tr_pct))

    def _ratsv_side_rank(self, *, signal: EmaDecisionSnapshot, entry_dir: str, close: float) -> float | None:
        if close <= 0:
            return None
        if signal.ema_fast is None or signal.ema_slow is None:
            return None
        tr_fast, _tr_slow = self._ratsv_tr_fast_slow()
        if tr_fast is None or tr_fast <= 0:
            return None
        spread = (float(signal.ema_fast) - float(signal.ema_slow)) / float(close)
        aligned = float(spread) if str(entry_dir) == "up" else -float(spread)
        if aligned <= 0:
            return 0.0
        rank = float(aligned) / (float(aligned) + float(tr_fast))
        return float(max(0.0, min(1.0, rank)))

    def _ratsv_thresholds_for_branch(
        self,
        *,
        branch_key: str,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        int,
        float | None,
        int | None,
    ]:
        rank_min = self._ratsv_rank_min
        tr_ratio_min = self._ratsv_tr_ratio_min
        slope_med_min = self._ratsv_slope_med_min_pct
        slope_vel_min = self._ratsv_slope_vel_min_pct
        slope_med_slow_min = self._ratsv_slope_med_slow_min_pct
        slope_vel_slow_min = self._ratsv_slope_vel_slow_min_pct
        slope_vel_consistency_bars = int(self._ratsv_slope_vel_consistency_bars)
        slope_vel_consistency_min = self._ratsv_slope_vel_consistency_min
        cross_age_max = self._ratsv_cross_age_max
        if branch_key == "a":
            rank_min = self._ratsv_branch_a_rank_min if self._ratsv_branch_a_rank_min is not None else rank_min
            tr_ratio_min = (
                self._ratsv_branch_a_tr_ratio_min if self._ratsv_branch_a_tr_ratio_min is not None else tr_ratio_min
            )
            slope_med_min = (
                self._ratsv_branch_a_slope_med_min_pct
                if self._ratsv_branch_a_slope_med_min_pct is not None
                else slope_med_min
            )
            slope_vel_min = (
                self._ratsv_branch_a_slope_vel_min_pct
                if self._ratsv_branch_a_slope_vel_min_pct is not None
                else slope_vel_min
            )
            slope_med_slow_min = (
                self._ratsv_branch_a_slope_med_slow_min_pct
                if self._ratsv_branch_a_slope_med_slow_min_pct is not None
                else slope_med_slow_min
            )
            slope_vel_slow_min = (
                self._ratsv_branch_a_slope_vel_slow_min_pct
                if self._ratsv_branch_a_slope_vel_slow_min_pct is not None
                else slope_vel_slow_min
            )
            if self._active_regime2_ready and self._active_regime2_dir == "down":
                if self._regime2_soft_bear_branch_a_slope_med_slow_min_pct is not None:
                    slope_med_slow_min = max(
                        float(slope_med_slow_min or 0.0),
                        float(self._regime2_soft_bear_branch_a_slope_med_slow_min_pct),
                    )
                if self._regime2_soft_bear_branch_a_slope_vel_slow_min_pct is not None:
                    slope_vel_slow_min = max(
                        float(slope_vel_slow_min or 0.0),
                        float(self._regime2_soft_bear_branch_a_slope_vel_slow_min_pct),
                    )
            if self._ratsv_branch_a_slope_vel_consistency_bars is not None:
                slope_vel_consistency_bars = max(0, int(self._ratsv_branch_a_slope_vel_consistency_bars))
            if self._ratsv_branch_a_slope_vel_consistency_min is not None:
                slope_vel_consistency_min = self._ratsv_branch_a_slope_vel_consistency_min
            cross_age_max = (
                self._ratsv_branch_a_cross_age_max
                if self._ratsv_branch_a_cross_age_max is not None
                else cross_age_max
            )
        elif branch_key == "b":
            rank_min = self._ratsv_branch_b_rank_min if self._ratsv_branch_b_rank_min is not None else rank_min
            tr_ratio_min = (
                self._ratsv_branch_b_tr_ratio_min if self._ratsv_branch_b_tr_ratio_min is not None else tr_ratio_min
            )
            slope_med_min = (
                self._ratsv_branch_b_slope_med_min_pct
                if self._ratsv_branch_b_slope_med_min_pct is not None
                else slope_med_min
            )
            slope_vel_min = (
                self._ratsv_branch_b_slope_vel_min_pct
                if self._ratsv_branch_b_slope_vel_min_pct is not None
                else slope_vel_min
            )
            slope_med_slow_min = (
                self._ratsv_branch_b_slope_med_slow_min_pct
                if self._ratsv_branch_b_slope_med_slow_min_pct is not None
                else slope_med_slow_min
            )
            slope_vel_slow_min = (
                self._ratsv_branch_b_slope_vel_slow_min_pct
                if self._ratsv_branch_b_slope_vel_slow_min_pct is not None
                else slope_vel_slow_min
            )
            if self._ratsv_branch_b_slope_vel_consistency_bars is not None:
                slope_vel_consistency_bars = max(0, int(self._ratsv_branch_b_slope_vel_consistency_bars))
            if self._ratsv_branch_b_slope_vel_consistency_min is not None:
                slope_vel_consistency_min = self._ratsv_branch_b_slope_vel_consistency_min
            cross_age_max = (
                self._ratsv_branch_b_cross_age_max
                if self._ratsv_branch_b_cross_age_max is not None
                else cross_age_max
            )
        return (
            rank_min,
            tr_ratio_min,
            slope_med_min,
            slope_vel_min,
            slope_med_slow_min,
            slope_vel_slow_min,
            int(slope_vel_consistency_bars),
            slope_vel_consistency_min,
            cross_age_max,
        )

    def _ratsv_branch_metrics(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        entry_dir: str | None,
    ) -> dict[str, float | int | None] | None:
        if signal is None:
            self._ratsv_last_candidate_metrics[branch_key] = None
            return None

        cross_age = self._ratsv_branch_cross_age.get(branch_key)
        if bool(signal.cross_up) or bool(signal.cross_down):
            cross_age = 0
        elif cross_age is None:
            cross_age = 1
        else:
            cross_age = int(cross_age) + 1
        self._ratsv_branch_cross_age[branch_key] = int(cross_age)

        slope_now = self._signed_fast_slope_pct(signal, float(close))
        if slope_now is not None:
            self._ratsv_branch_slope_hist[branch_key].append(float(slope_now))
        window = list(self._ratsv_branch_slope_hist[branch_key])[-int(self._ratsv_slope_window) :]
        slope_med = self._median(window)
        window_slow = list(self._ratsv_branch_slope_hist[branch_key])[-int(self._ratsv_slope_slow_window) :]
        slope_med_slow = self._median(window_slow)
        prev_med = self._ratsv_branch_last_slope_med.get(branch_key)
        prev_med_slow = self._ratsv_branch_last_slope_med_slow.get(branch_key)
        slope_vel = None
        if slope_med is not None and prev_med is not None:
            slope_vel = float(slope_med) - float(prev_med)
        slope_vel_slow = None
        if slope_med_slow is not None and prev_med_slow is not None:
            slope_vel_slow = float(slope_med_slow) - float(prev_med_slow)
        if slope_med is not None:
            self._ratsv_branch_last_slope_med[branch_key] = float(slope_med)
        if slope_med_slow is not None:
            self._ratsv_branch_last_slope_med_slow[branch_key] = float(slope_med_slow)
        if slope_vel is not None:
            self._ratsv_branch_slope_vel_hist[branch_key].append(float(slope_vel))

        tr_fast, tr_slow = self._ratsv_tr_fast_slow()
        tr_ratio = None
        if tr_fast is not None and tr_slow is not None and tr_slow > 0:
            tr_ratio = float(tr_fast) / float(tr_slow)

        side_rank = None
        if entry_dir in ("up", "down") and signal.ema_ready:
            side_rank = self._ratsv_side_rank(signal=signal, entry_dir=str(entry_dir), close=float(close))

        (
            _rank_min,
            _tr_ratio_min,
            _slope_med_min,
            _slope_vel_min,
            _slope_med_slow_min,
            _slope_vel_slow_min,
            slope_vel_consistency_bars,
            _slope_vel_consistency_min,
            _cross_age_max,
        ) = self._ratsv_thresholds_for_branch(branch_key=branch_key)
        slope_vel_consistency = None
        if entry_dir in ("up", "down") and int(slope_vel_consistency_bars) > 0:
            vel_hist = list(self._ratsv_branch_slope_vel_hist[branch_key])
            if vel_hist:
                n = min(len(vel_hist), int(slope_vel_consistency_bars))
                tail = vel_hist[-int(n) :]
                if tail:
                    if str(entry_dir) == "up":
                        aligned = sum(1 for vel in tail if float(vel) >= 0.0)
                    else:
                        aligned = sum(1 for vel in tail if float(vel) <= 0.0)
                    slope_vel_consistency = float(aligned) / float(len(tail))

        metrics = {
            "side_rank": float(side_rank) if side_rank is not None else None,
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "slope_now": float(slope_now) if slope_now is not None else None,
            "slope_med": float(slope_med) if slope_med is not None else None,
            "slope_vel": float(slope_vel) if slope_vel is not None else None,
            "slope_med_slow": float(slope_med_slow) if slope_med_slow is not None else None,
            "slope_vel_slow": float(slope_vel_slow) if slope_vel_slow is not None else None,
            "slope_vel_consistency": float(slope_vel_consistency) if slope_vel_consistency is not None else None,
            "cross_age": int(cross_age) if cross_age is not None else None,
        }
        self._ratsv_last_candidate_metrics[branch_key] = metrics
        return metrics

    def _ratsv_entry_ok(
        self,
        *,
        branch_key: str,
        entry_dir: str | None,
        metrics: dict[str, float | int | None] | None,
    ) -> bool:
        if not bool(self._ratsv_enabled):
            return True
        if entry_dir not in ("up", "down"):
            return False
        if not isinstance(metrics, dict):
            return False

        (
            rank_min,
            tr_ratio_min,
            slope_med_min,
            slope_vel_min,
            slope_med_slow_min,
            slope_vel_slow_min,
            _slope_vel_consistency_bars,
            slope_vel_consistency_min,
            cross_age_max,
        ) = self._ratsv_thresholds_for_branch(branch_key=branch_key)

        side_rank = metrics.get("side_rank")
        tr_ratio = metrics.get("tr_ratio")
        slope_med = metrics.get("slope_med")
        slope_vel = metrics.get("slope_vel")
        slope_med_slow = metrics.get("slope_med_slow")
        slope_vel_slow = metrics.get("slope_vel_slow")
        slope_vel_consistency = metrics.get("slope_vel_consistency")
        cross_age = metrics.get("cross_age")

        if rank_min is not None:
            if side_rank is None or float(side_rank) < float(rank_min):
                return False
        if tr_ratio_min is not None:
            if tr_ratio is None or float(tr_ratio) < float(tr_ratio_min):
                return False

        signed_med = None
        if slope_med is not None:
            signed_med = float(slope_med) if str(entry_dir) == "up" else -float(slope_med)
        if slope_med_min is not None:
            if signed_med is None or float(signed_med) < float(slope_med_min):
                return False

        signed_vel = None
        if slope_vel is not None:
            signed_vel = float(slope_vel) if str(entry_dir) == "up" else -float(slope_vel)
        if slope_vel_min is not None:
            if signed_vel is None or float(signed_vel) < float(slope_vel_min):
                return False

        signed_med_slow = None
        if slope_med_slow is not None:
            signed_med_slow = float(slope_med_slow) if str(entry_dir) == "up" else -float(slope_med_slow)
        if slope_med_slow_min is not None:
            if signed_med_slow is None or float(signed_med_slow) < float(slope_med_slow_min):
                return False

        signed_vel_slow = None
        if slope_vel_slow is not None:
            signed_vel_slow = float(slope_vel_slow) if str(entry_dir) == "up" else -float(slope_vel_slow)
        if slope_vel_slow_min is not None:
            if signed_vel_slow is None or float(signed_vel_slow) < float(slope_vel_slow_min):
                return False

        if slope_vel_consistency_min is not None:
            if slope_vel_consistency is None or float(slope_vel_consistency) < float(slope_vel_consistency_min):
                return False

        if cross_age_max is not None:
            if cross_age is None or int(cross_age) > int(cross_age_max):
                return False
        return True

    def _branch_signed_slope_thresholds(self, *, branch_key: str) -> tuple[float | None, float | None]:
        if branch_key == "a":
            return self._branch_a_min_signed_slope_pct, self._branch_a_max_signed_slope_pct
        if branch_key == "b":
            return self._branch_b_min_signed_slope_pct, self._branch_b_max_signed_slope_pct
        return None, None

    def _candidate_entry_dir(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        candidate_dir: str | None,
        min_signed_slope_pct: float | None,
        max_signed_slope_pct: float | None,
    ) -> str | None:
        if signal is None or not bool(signal.ema_ready):
            self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=None)
            return None
        entry_dir = candidate_dir
        if entry_dir not in ("up", "down"):
            self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=None)
            return None

        slope = self._signed_fast_slope_pct(signal, float(close))
        if min_signed_slope_pct is not None or max_signed_slope_pct is not None:
            if slope is None:
                self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=str(entry_dir))
                return None
            signed = float(slope) if entry_dir == "up" else -float(slope)
            if min_signed_slope_pct is not None and signed < float(min_signed_slope_pct):
                self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=str(entry_dir))
                return None
            if max_signed_slope_pct is not None and signed > float(max_signed_slope_pct):
                self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=str(entry_dir))
                return None

        metrics = self._ratsv_branch_metrics(
            branch_key=branch_key,
            signal=signal,
            close=float(close),
            entry_dir=str(entry_dir),
        )
        if not self._ratsv_entry_ok(branch_key=branch_key, entry_dir=str(entry_dir), metrics=metrics):
            return None
        return str(entry_dir)

    def _branch_entry_dir(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        min_signed_slope_pct: float | None,
        max_signed_slope_pct: float | None,
    ) -> str | None:
        return self._candidate_entry_dir(
            branch_key=branch_key,
            signal=signal,
            close=float(close),
            candidate_dir=getattr(signal, "entry_dir", None),
            min_signed_slope_pct=min_signed_slope_pct,
            max_signed_slope_pct=max_signed_slope_pct,
        )

    def _apply_regime2_bear_primary(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        bar: BarLike,
        close: float,
    ) -> tuple[EmaDecisionSnapshot | None, str | None]:
        if (
            signal is None
            or not bool(signal.ema_ready)
            or self._bear_supertrend_engine is None
            or not self._active_regime2_ready
            or self._active_regime2_dir != "down"
        ):
            return signal, None
        if self._regime2_bear_hard_mode != "off":
            if not self._active_regime2_bear_hard_ready or self._active_regime2_bear_hard_dir != "down":
                return signal, None
        if not self._regime2_bear_takeover_allowed():
            return signal, None

        use_clean_host = bool(self._regime4_owner == "clean_host" and self._clean_bear_supertrend_engine is not None)
        bear_engine = self._clean_bear_supertrend_engine if use_clean_host else self._bear_supertrend_engine
        if bear_engine is None:
            return signal, None

        last_bear_supertrend = self._clean_bear_supertrend_engine.update(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        ) if use_clean_host else self._bear_supertrend_engine.update(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )
        if use_clean_host:
            self._last_clean_bear_supertrend = last_bear_supertrend
        else:
            self._last_bear_supertrend = last_bear_supertrend

        bear_ready = bool(last_bear_supertrend and last_bear_supertrend.ready)
        bear_dir = last_bear_supertrend.direction if last_bear_supertrend is not None else None
        bear_dir = str(bear_dir) if bear_dir in ("up", "down") else None
        prev_dir = self._clean_bear_prev_dir if use_clean_host else self._bear_prev_dir
        cross_up = bool(bear_ready and bear_dir == "up" and prev_dir == "down")
        cross_down = bool(bear_ready and bear_dir == "down" and prev_dir == "up")
        if bear_ready:
            if use_clean_host:
                self._clean_bear_prev_dir = bear_dir
            else:
                self._bear_prev_dir = bear_dir

        min_signed_slope_pct, max_signed_slope_pct = self._branch_signed_slope_thresholds(branch_key=branch_key)
        candidate_dir: str | None = None
        if bear_dir == "down":
            candidate_dir = self._candidate_entry_dir(
                branch_key=branch_key,
                signal=signal,
                close=float(close),
                candidate_dir="down",
                min_signed_slope_pct=min_signed_slope_pct,
                max_signed_slope_pct=max_signed_slope_pct,
            )
        elif bear_dir == "up" and bool(self._regime2_bear_allow_long_recovery) and bool(cross_up):
            candidate_dir = self._candidate_entry_dir(
                branch_key=branch_key,
                signal=signal,
                close=float(close),
                candidate_dir="up",
                min_signed_slope_pct=min_signed_slope_pct,
                max_signed_slope_pct=max_signed_slope_pct,
            )

        return (
            replace(
                signal,
                cross_up=bool(cross_up),
                cross_down=bool(cross_down),
                state=bear_dir,
                entry_dir=candidate_dir,
                regime_dir=bear_dir,
                regime_ready=bool(bear_ready),
            ),
            candidate_dir,
        )

    def _regime2_bear_takeover_allowed(self) -> bool:
        mode = str(self._regime2_bear_takeover_mode or "always").strip().lower()
        if mode == "always":
            return True
        risk = self._last_risk
        riskoff = bool(getattr(risk, "riskoff", False))
        riskpanic = bool(getattr(risk, "riskpanic", False))
        hostile = bool(riskoff or riskpanic)
        shock, shock_dir, _shock_atr_pct = self._shock_view()
        shockdown = bool(shock) and shock_dir == "down"
        if mode == "hostile":
            return hostile
        if mode == "riskoff":
            return riskoff
        if mode == "riskpanic":
            return riskpanic
        if mode == "shockdown":
            return shockdown
        if mode == "hostile_or_shockdown":
            return bool(hostile or shockdown)
        return True

    def _classify_regime4_state(
        self,
        *,
        shock_atr_pct: float | None,
        fast_dir: str | None,
        fast_ready: bool,
        hard_dir: str | None,
        hard_ready: bool,
        hard_release_age_bars: int | None,
    ) -> tuple[str | None, bool]:
        fast_dir = str(fast_dir) if bool(fast_ready) and fast_dir in ("up", "down") else None
        hard_dir = str(hard_dir) if bool(hard_ready) and hard_dir in ("up", "down") else None
        if hard_dir == "down":
            if fast_dir == "up":
                return "transition_up_hot", True
            if fast_dir == "down" or fast_dir is None:
                if (
                    self._regime2_crash_atr_pct_min is not None
                    and shock_atr_pct is not None
                    and float(shock_atr_pct) >= float(self._regime2_crash_atr_pct_min)
                ):
                    return "crash_down", False
                return "trend_down", False
        if fast_dir == "up":
            transition_hot = False
            if (
                self._regime2_transition_hot_shock_atr_pct_min is not None
                and shock_atr_pct is not None
                and float(shock_atr_pct) >= float(self._regime2_transition_hot_shock_atr_pct_min)
            ):
                transition_hot = True
            if (
                not transition_hot
                and self._regime2_transition_hot_release_max_bars is not None
                and hard_release_age_bars is not None
                and int(hard_release_age_bars) <= int(self._regime2_transition_hot_release_max_bars)
            ):
                transition_hot = True
            return ("transition_up_hot" if transition_hot else "trend_up_clean"), bool(transition_hot)
        if fast_dir == "down":
            return "trend_down", False
        return None, False

    def _regime2_crash_blocks_long(self, *, regime4_state: str | None, entry_dir: str | None) -> bool:
        return bool(self._regime2_crash_block_longs and regime4_state == "crash_down" and entry_dir == "up")

    def _regime2_crash_prearm_blocks_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_dir_ret_sum_pct: float | None,
    ) -> bool:
        apply_to = str(self._regime2_crash_prearm_apply_to or "off")
        if apply_to == "off":
            return False
        if regime4_state != "trend_down" or entry_dir != "up" or shock_dir != "down":
            return False
        branch_key = str(entry_branch or "")
        atr_pct_min = self._regime2_crash_prearm_shock_atr_pct_min
        ret_sum_pct_max = self._regime2_crash_prearm_shock_dir_ret_sum_pct_max
        if branch_key == "a":
            if self._regime2_crash_prearm_branch_a_shock_atr_pct_min is not None:
                atr_pct_min = self._regime2_crash_prearm_branch_a_shock_atr_pct_min
            if self._regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max is not None:
                ret_sum_pct_max = self._regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max
        if (
            atr_pct_min is not None
            and (shock_atr_pct is None or float(shock_atr_pct) < float(atr_pct_min))
        ):
            return False
        if (
            ret_sum_pct_max is not None
            and (
                shock_dir_ret_sum_pct is None
                or float(shock_dir_ret_sum_pct) > float(ret_sum_pct_max)
            )
        ):
            return False
        if apply_to == "branch_b_longs":
            return branch_key == "b"
        return True

    def _regime2_blocks_branch_b_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_drawdown_dist_on_vel_pp: float | None,
        bar_ts: datetime,
    ) -> bool:
        if not (self._dual_branch_enabled and entry_branch == "b" and entry_dir == "up"):
            return False
        if regime4_state == "transition_up_hot":
            if self._regime2_repair_block_branch_b_longs:
                return True
            if (
                self._regime2_repair_branch_b_long_max_shock_atr_pct is not None
                and shock_atr_pct is not None
                and float(shock_atr_pct) >= float(self._regime2_repair_branch_b_long_max_shock_atr_pct)
            ):
                return True
            if self._regime2_repair_branch_b_long_block_after_hour_et is not None:
                hour_et = _trade_hour_et_shared(bar_ts, naive_ts_mode=self._naive_ts_mode)
                return int(hour_et) >= int(self._regime2_repair_branch_b_long_block_after_hour_et)
            return False

        if shock_atr_pct is None or shock_drawdown_dist_on_vel_pp is None:
            return False
        atr_value = float(shock_atr_pct)
        ddv_value = float(shock_drawdown_dist_on_vel_pp)
        release_age = self._active_regime2_bear_hard_release_age_bars
        release_age_value = int(release_age) if release_age is not None else None
        if regime4_state == "trend_down":
            age_min = self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars
            age_max = self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars
            atr_min = self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min
            atr_max = self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max
            ddv_min = self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp
            ddv_max = self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp
            in_primary_band = bool(
                shock_dir == "down"
                and self._active_regime2_bear_hard_dir == "up"
                and release_age_value is not None
                and age_min is not None
                and age_max is not None
                and int(age_min) <= release_age_value < int(age_max)
                and atr_min is not None
                and atr_max is not None
                and float(atr_min) <= atr_value < float(atr_max)
                and ddv_min is not None
                and ddv_max is not None
                and float(ddv_min) <= ddv_value < float(ddv_max)
            )
            recovery_atr_min = self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min
            recovery_atr_max = self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max
            recovery_ddv_min = self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp
            recovery_ddv_max = self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp
            in_recovery_band = bool(
                shock_dir == "up"
                and self._active_regime2_bear_hard_dir == "up"
                and release_age_value is not None
                and age_min is not None
                and age_max is not None
                and int(age_min) <= release_age_value < int(age_max)
                and recovery_atr_min is not None
                and recovery_atr_max is not None
                and float(recovery_atr_min) <= atr_value < float(recovery_atr_max)
                and recovery_ddv_min is not None
                and recovery_ddv_max is not None
                and float(recovery_ddv_min) <= ddv_value < float(recovery_ddv_max)
            )
            return bool(in_primary_band or in_recovery_band)
        if regime4_state != "trend_up_clean":
            return False
        stale_min = self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars
        if not (
            shock_dir == "up"
            and self._active_regime2_bear_hard_dir == "up"
            and release_age_value is not None
        ):
            return False
        flat_ddv_abs_max = self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp
        flat_low_atr_max = self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max
        flat_low_stale_min = self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars
        flat_high_atr_max = self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max
        return bool(
            flat_ddv_abs_max is not None
            and abs(ddv_value) < float(flat_ddv_abs_max)
            and (
                (
                    flat_low_atr_max is not None
                    and flat_low_stale_min is not None
                    and atr_value < float(flat_low_atr_max)
                    and release_age_value >= int(flat_low_stale_min)
                )
                or (
                    flat_high_atr_max is not None
                    and stale_min is not None
                    and atr_value < float(flat_high_atr_max)
                    and (flat_low_atr_max is None or atr_value >= float(flat_low_atr_max))
                    and release_age_value >= int(stale_min)
                )
            )
        )

    def _regime2_upcorridor_blocks_branch_a_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_atr_pct: float | None,
    ) -> bool:
        if not (
            self._dual_branch_enabled
            and regime4_state in ("transition_up_hot", "trend_up_clean")
            and entry_branch == "a"
            and entry_dir == "up"
        ):
            return False
        if shock_atr_pct is None:
            return False
        atr_value = float(shock_atr_pct)
        in_mid_band = False
        if (
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min is not None
            and self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max is not None
        ):
            in_mid_band = (
                float(self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min)
                <= atr_value
                < float(self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max)
            )
        in_extreme_band = (
            self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min is not None
            and atr_value >= float(self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min)
        )
        if not (in_mid_band or in_extreme_band):
            return False
        release_age = self._active_regime2_bear_hard_release_age_bars
        if release_age is None:
            return False
        if regime4_state == "transition_up_hot":
            fresh_max = self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars
            return fresh_max is not None and int(release_age) <= int(fresh_max)
        stale_min = self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars
        return stale_min is not None and int(release_age) >= int(stale_min)

    def _continuation_confidence_blocks_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_drawdown_dist_on_vel_pp: float | None,
    ) -> bool:
        release_age = self._active_regime2_bear_hard_release_age_bars
        if (
            entry_dir == "up"
            and entry_branch == "b"
            and regime4_state == "trend_up_clean"
            and shock_dir == "up"
            and self._active_regime2_bear_hard_dir == "up"
        ):
            age_min = self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars
            age_max = self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars
            return bool(
                release_age is not None
                and age_min is not None
                and age_max is not None
                and int(age_min) <= int(release_age) < int(age_max)
            )
        return bool(
            entry_dir == "up"
            and entry_branch == "a"
            and regime4_state == "transition_up_hot"
            and shock_dir == "up"
            and self._active_regime2_bear_hard_dir == "down"
            and release_age is not None
            and self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars is not None
            and int(release_age) <= int(self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars)
            and shock_atr_pct is not None
            and self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min is not None
            and self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max is not None
            and float(self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min)
            <= float(shock_atr_pct)
            < float(self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max)
            and shock_drawdown_dist_on_vel_pp is not None
            and self._regime2_continuation_confidence_branch_a_transition_ddv_max_pp is not None
            and float(shock_drawdown_dist_on_vel_pp) <= float(self._regime2_continuation_confidence_branch_a_transition_ddv_max_pp)
        )

    def _select_dual_signal(
        self,
        *,
        close: float,
        signal_a: EmaDecisionSnapshot,
        signal_b: EmaDecisionSnapshot,
    ) -> tuple[EmaDecisionSnapshot, str | None, str | None]:
        """Return (signal_for_gate/flip, entry_dir_for_entries, entry_branch)."""
        dir_a = self._branch_entry_dir(
            branch_key="a",
            signal=signal_a,
            close=float(close),
            min_signed_slope_pct=self._branch_a_min_signed_slope_pct,
            max_signed_slope_pct=self._branch_a_max_signed_slope_pct,
        )
        dir_b = self._branch_entry_dir(
            branch_key="b",
            signal=signal_b,
            close=float(close),
            min_signed_slope_pct=self._branch_b_min_signed_slope_pct,
            max_signed_slope_pct=self._branch_b_max_signed_slope_pct,
        )

        if self._dual_branch_priority == "a_then_b":
            if dir_a in ("up", "down"):
                return signal_a, str(dir_a), "a"
            if dir_b in ("up", "down"):
                return signal_b, str(dir_b), "b"
            return signal_a, None, None

        # default: b_then_a
        if dir_b in ("up", "down"):
            return signal_b, str(dir_b), "b"
        if dir_a in ("up", "down"):
            return signal_a, str(dir_a), "a"
        return signal_b, None, None
