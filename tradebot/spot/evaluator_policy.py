"""Entry selection and risk-policy decisions for spot signals."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import date, datetime
from statistics import median

from ..engine import _trade_date as _trade_date_shared, _trade_hour_et as _trade_hour_et_shared
from ..engines.risk import RiskOverlaySnapshot
from ..engines.signals import EmaDecisionSnapshot, OrbDecisionEngine
from .evaluator_common import (
    BarLike,
    SpotEntryCandidate,
    SpotEntryGateContext,
    SpotRegimeState,
    SpotSignalSelection,
    SpotSignalSnapshot,
)


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

    def _advance_entry_signal(self, *, bar: BarLike, close: float) -> SpotSignalSelection:
        """Normalize EMA, dual-EMA, and ORB into one entry-selection contract."""
        if self._signal_engine is not None:
            signal = self._signal_engine.update(close)
            return SpotSignalSelection(
                signal=signal,
                candidate=SpotEntryCandidate(
                    self._branch_entry_dir(
                        branch_key="single",
                        signal=signal,
                        close=float(close),
                        min_signed_slope_pct=None,
                        max_signed_slope_pct=None,
                    )
                ),
                branch_key="single",
            )

        if (
            self._signal_engine_a is not None
            and self._signal_engine_b is not None
            and bool(self._dual_branch_enabled)
        ):
            signal_a = self._signal_engine_a.update(close)
            signal_b = self._signal_engine_b.update(close)
            signal, direction, branch = self._select_dual_signal(
                close=float(close),
                signal_a=signal_a,
                signal_b=signal_b,
            )
            branch_key = "a" if signal is signal_a else "b" if signal is signal_b else None
            return SpotSignalSelection(
                signal=signal,
                candidate=SpotEntryCandidate(direction, branch),
                branch_key=branch_key,
            )

        if self._orb_engine is not None:
            signal = self._orb_engine.update(
                ts=bar.ts,
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )
            return SpotSignalSelection(
                signal=signal,
                candidate=SpotEntryCandidate(signal.entry_dir if signal is not None else None),
            )

        return SpotSignalSelection(signal=None, candidate=SpotEntryCandidate(None))

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
        return (
            self._median(self._ratsv_tr_fast_hist),
            self._median(self._ratsv_tr_slow_hist),
        )

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
        self._ratsv_tr_fast_hist.append(float(tr_pct))
        self._ratsv_tr_slow_hist.append(float(tr_pct))

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
            if self._fast_regime_ready and self._fast_regime_dir == "down":
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
            self._ratsv_branch_slope_fast_hist[branch_key].append(float(slope_now))
            self._ratsv_branch_slope_slow_hist[branch_key].append(float(slope_now))
        slope_med = self._median(self._ratsv_branch_slope_fast_hist[branch_key])
        slope_med_slow = self._median(
            self._ratsv_branch_slope_slow_hist[branch_key]
        )
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
        regime: SpotRegimeState,
    ) -> tuple[EmaDecisionSnapshot | None, str | None]:
        if (
            signal is None
            or not bool(signal.ema_ready)
            or self._bear_supertrend_engine is None
            or not regime.fast_ready
            or regime.fast_dir != "down"
        ):
            return signal, None
        if self._regime2_bear_hard_mode != "off":
            if not regime.hard_ready or regime.hard_dir != "down":
                return signal, None
        if not self._regime2_bear_takeover_allowed():
            return signal, None

        use_clean_host = bool(
            regime.owner == "clean_host"
            and self._clean_bear_supertrend_engine is not None
        )
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

    def _apply_entry_gates(
        self,
        candidate: SpotEntryCandidate,
        context: SpotEntryGateContext,
    ) -> SpotEntryCandidate:
        """Apply the ordered, independently configured entry-policy gates."""
        for name, blocks in (
            ("crash", self._crash_gate_blocks),
            ("crash_prearm", self._crash_prearm_gate_blocks),
            ("branch_b_regime", self._branch_b_regime_gate_blocks),
            ("branch_a_upcorridor", self._branch_a_upcorridor_gate_blocks),
            ("continuation_confidence", self._continuation_confidence_gate_blocks),
        ):
            if blocks(candidate, context):
                return candidate.block(name)
        return candidate

    def _crash_gate_blocks(
        self,
        candidate: SpotEntryCandidate,
        context: SpotEntryGateContext,
    ) -> bool:
        return bool(
            self._regime_gates.crash_block_longs
            and context.regime.label == "crash_down"
            and candidate.direction == "up"
        )

    def _crash_prearm_gate_blocks(
        self,
        candidate: SpotEntryCandidate,
        context: SpotEntryGateContext,
    ) -> bool:
        policy = self._regime_gates
        apply_to = policy.crash_prearm_scope
        if apply_to == "off":
            return False
        if (
            context.regime.label != "trend_down"
            or candidate.direction != "up"
            or context.shock_dir != "down"
        ):
            return False
        branch_key = str(candidate.branch or "")
        atr_pct_min = policy.crash_prearm_atr_min
        ret_sum_pct_max = policy.crash_prearm_ret_max
        if branch_key == "a":
            if policy.crash_prearm_branch_a_atr_min is not None:
                atr_pct_min = policy.crash_prearm_branch_a_atr_min
            if policy.crash_prearm_branch_a_ret_max is not None:
                ret_sum_pct_max = policy.crash_prearm_branch_a_ret_max
        if (
            atr_pct_min is not None
            and (
                context.shock_atr_pct is None
                or float(context.shock_atr_pct) < float(atr_pct_min)
            )
        ):
            return False
        if (
            ret_sum_pct_max is not None
            and (
                context.shock_dir_ret_sum_pct is None
                or float(context.shock_dir_ret_sum_pct) > float(ret_sum_pct_max)
            )
        ):
            return False
        if apply_to == "branch_b_longs":
            return branch_key == "b"
        return True

    def _branch_b_regime_gate_blocks(
        self,
        candidate: SpotEntryCandidate,
        context: SpotEntryGateContext,
    ) -> bool:
        policy = self._regime_gates
        if not (
            self._dual_branch_enabled
            and candidate.branch == "b"
            and candidate.direction == "up"
        ):
            return False
        if context.regime.label == "transition_up_hot":
            if policy.repair_branch_b_block:
                return True
            if (
                policy.repair_branch_b_atr_max is not None
                and context.shock_atr_pct is not None
                and float(context.shock_atr_pct)
                >= policy.repair_branch_b_atr_max
            ):
                return True
            if policy.repair_branch_b_after_hour is not None:
                hour_et = _trade_hour_et_shared(
                    context.bar_ts,
                    naive_ts_mode=self._naive_ts_mode,
                )
                return int(hour_et) >= policy.repair_branch_b_after_hour
            return False

        if (
            context.shock_atr_pct is None
            or context.shock_drawdown_dist_on_vel_pp is None
        ):
            return False
        atr_value = float(context.shock_atr_pct)
        ddv_value = float(context.shock_drawdown_dist_on_vel_pp)
        release_age = context.regime.hard_release_age_bars
        release_age_value = int(release_age) if release_age is not None else None
        if context.regime.label == "trend_down":
            in_primary_band = bool(
                context.shock_dir == "down"
                and context.regime.hard_dir == "up"
                and policy.trenddown_branch_b_release_age.contains(release_age_value)
                and policy.trenddown_branch_b_atr.contains(atr_value)
                and policy.trenddown_branch_b_ddv.contains(ddv_value)
            )
            in_recovery_band = bool(
                context.shock_dir == "up"
                and context.regime.hard_dir == "up"
                and policy.trenddown_branch_b_release_age.contains(release_age_value)
                and policy.trenddown_branch_b_recovery_atr.contains(atr_value)
                and policy.trenddown_branch_b_recovery_ddv.contains(ddv_value)
            )
            return bool(in_primary_band or in_recovery_band)
        if context.regime.label != "trend_up_clean":
            return False
        stale_min = policy.upcorridor_branch_b_stale_age_min
        if not (
            context.shock_dir == "up"
            and context.regime.hard_dir == "up"
            and release_age_value is not None
        ):
            return False
        flat_ddv_abs_max = policy.upcorridor_branch_b_flat_ddv_abs_max
        flat_low_atr_max = policy.upcorridor_branch_b_flat_low_atr_max
        flat_low_stale_min = policy.upcorridor_branch_b_flat_low_stale_age_min
        flat_high_atr_max = policy.upcorridor_branch_b_flat_atr_max
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

    def _branch_a_upcorridor_gate_blocks(
        self,
        candidate: SpotEntryCandidate,
        context: SpotEntryGateContext,
    ) -> bool:
        policy = self._regime_gates
        if not (
            self._dual_branch_enabled
            and context.regime.label in ("transition_up_hot", "trend_up_clean")
            and candidate.branch == "a"
            and candidate.direction == "up"
        ):
            return False
        if context.shock_atr_pct is None:
            return False
        atr_value = float(context.shock_atr_pct)
        in_mid_band = policy.upcorridor_branch_a_mid_atr.contains(atr_value)
        in_extreme_band = (
            policy.upcorridor_branch_a_extreme_atr_min is not None
            and atr_value >= policy.upcorridor_branch_a_extreme_atr_min
        )
        if not (in_mid_band or in_extreme_band):
            return False
        release_age = context.regime.hard_release_age_bars
        if release_age is None:
            return False
        if context.regime.label == "transition_up_hot":
            fresh_max = policy.upcorridor_branch_a_fresh_age_max
            return fresh_max is not None and int(release_age) <= int(fresh_max)
        stale_min = policy.upcorridor_branch_a_stale_age_min
        return stale_min is not None and int(release_age) >= int(stale_min)

    def _continuation_confidence_gate_blocks(
        self,
        candidate: SpotEntryCandidate,
        context: SpotEntryGateContext,
    ) -> bool:
        policy = self._regime_gates
        release_age = context.regime.hard_release_age_bars
        if (
            candidate.direction == "up"
            and candidate.branch == "b"
            and context.regime.label == "trend_up_clean"
            and context.shock_dir == "up"
            and context.regime.hard_dir == "up"
        ):
            return policy.continuation_branch_b_release_age.contains(release_age)
        return bool(
            candidate.direction == "up"
            and candidate.branch == "a"
            and context.regime.label == "transition_up_hot"
            and context.shock_dir == "up"
            and context.regime.hard_dir == "down"
            and release_age is not None
            and policy.continuation_branch_a_release_age_max is not None
            and int(release_age) <= policy.continuation_branch_a_release_age_max
            and policy.continuation_branch_a_atr.contains(context.shock_atr_pct)
            and context.shock_drawdown_dist_on_vel_pp is not None
            and policy.continuation_branch_a_ddv_max is not None
            and float(context.shock_drawdown_dist_on_vel_pp)
            <= policy.continuation_branch_a_ddv_max
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
