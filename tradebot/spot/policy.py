"""Canonical spot decision policy shared by backtests and live execution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime

from ..time_utils import NaiveTsModeInput
from ..time_utils import trade_date as _trade_date_shared
from ..time_utils import trade_hour_et as _trade_hour_et_shared
from .policy_contract import SpotDecisionTrace, SpotIntentDecision, SpotPolicyConfigView, SpotRuntimeSpec
from .sizing import SpotSizingPolicy


class SpotPolicy(SpotSizingPolicy):
    @staticmethod
    def _get(source: Mapping[str, object] | object | None, key: str, default: object = None) -> object:
        return SpotPolicyConfigView._get(source, key, default)

    @classmethod
    def _parse_int(cls, value: object, *, default: int, min_value: int | None = None) -> int:
        return SpotPolicyConfigView._parse_int(value, default=default, min_value=min_value)

    @classmethod
    def _parse_float(cls, value: object, *, default: float, min_value: float | None = None) -> float:
        return SpotPolicyConfigView._parse_float(value, default=default, min_value=min_value)

    @staticmethod
    def _trade_date(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None) -> date:
        return _trade_date_shared(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc")

    @staticmethod
    def _trade_hour_et(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None) -> int:
        return int(_trade_hour_et_shared(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc"))

    @classmethod
    def policy_config(
        cls,
        *,
        strategy: Mapping[str, object] | object | None = None,
        filters: Mapping[str, object] | object | None = None,
    ) -> SpotPolicyConfigView:
        return SpotPolicyConfigView.from_sources(strategy=strategy, filters=filters)

    @classmethod
    def runtime_spec(
        cls,
        *,
        strategy: Mapping[str, object] | object | None = None,
        filters: Mapping[str, object] | object | None = None,
    ) -> SpotRuntimeSpec:
        return SpotRuntimeSpec.from_sources(strategy=strategy, filters=filters)

    @classmethod
    def risk_overlay_policy(
        cls,
        filters: Mapping[str, object] | object | None,
    ) -> tuple[str, float, float, float, float, float, float]:
        """Return sanitized overlay factors.

        Returns:
            (riskoff_mode, riskoff_long_factor, riskoff_short_factor,
             riskpanic_long_factor, riskpanic_short_factor,
             riskpop_long_factor, riskpop_short_factor)
        """
        cfg = cls.policy_config(filters=filters)
        return (
            str(cfg.riskoff_mode),
            float(cfg.riskoff_long_factor),
            float(cfg.riskoff_short_factor),
            float(cfg.riskpanic_long_factor),
            float(cfg.riskpanic_short_factor),
            float(cfg.riskpop_long_factor),
            float(cfg.riskpop_short_factor),
        )

    @classmethod
    def riskoff_mode(cls, filters: Mapping[str, object] | object | None) -> str:
        mode, *_ = cls.risk_overlay_policy(filters)
        return str(mode)

    @classmethod
    def resolve_entry_action_qty(
        cls,
        *,
        strategy: Mapping[str, object] | object,
        entry_dir: str | None,
        needs_direction: bool,
        fallback_short_sell: bool,
    ) -> tuple[str, int] | None:
        if entry_dir not in ("up", "down"):
            return None

        raw_map = cls._get(strategy, "directional_spot")
        leg = raw_map.get(entry_dir) if isinstance(raw_map, Mapping) else None
        if leg is not None:
            if isinstance(leg, Mapping):
                action_raw = leg.get("action")
                qty_raw = leg.get("qty")
            else:
                action_raw = getattr(leg, "action", None)
                qty_raw = getattr(leg, "qty", None)
            action = str(action_raw or "").strip().upper()
            if action in ("BUY", "SELL"):
                qty = cls._parse_int(qty_raw, default=1, min_value=1)
                return action, max(1, abs(int(qty)))
            if bool(needs_direction):
                return None

        if bool(needs_direction):
            return None
        if entry_dir == "up":
            return "BUY", 1
        if bool(fallback_short_sell):
            return "SELL", 1
        return None

    @classmethod
    def risk_entry_cutoff_hour_et(cls, filters: Mapping[str, object] | object | None) -> int | None:
        return cls.policy_config(filters=filters).risk_entry_cutoff_hour_et

    @classmethod
    def pending_entry_should_cancel(
        cls,
        *,
        pending_dir: str,
        pending_set_date: date | None,
        exec_ts: datetime,
        risk_overlay_enabled: bool,
        riskoff_today: bool,
        riskpanic_today: bool,
        riskpop_today: bool,
        riskoff_mode: str,
        shock_dir_now: str | None,
        riskoff_end_hour: int | None,
        naive_ts_mode: str | None = None,
    ) -> bool:
        if not bool(risk_overlay_enabled):
            return False
        if not (bool(riskoff_today) or bool(riskpanic_today) or bool(riskpop_today)):
            return False
        if str(riskoff_mode) == "directional" and shock_dir_now in ("up", "down"):
            return str(pending_dir) != str(shock_dir_now)
        if pending_set_date is not None and pending_set_date != cls._trade_date(exec_ts, naive_ts_mode=naive_ts_mode):
            return True
        if riskoff_end_hour is not None and cls._trade_hour_et(exec_ts, naive_ts_mode=naive_ts_mode) >= int(riskoff_end_hour):
            return True
        return False

    @classmethod
    def branch_size_mult(
        cls,
        *,
        strategy: Mapping[str, object] | object,
        entry_branch: str | None,
    ) -> float:
        cfg = cls.policy_config(strategy=strategy, filters=None)
        if not bool(cfg.spot_dual_branch_enabled):
            return 1.0
        if entry_branch == "a":
            return float(cfg.spot_branch_a_size_mult)
        if entry_branch == "b":
            return float(cfg.spot_branch_b_size_mult)
        return 1.0

    @classmethod
    def apply_branch_size_mult(
        cls,
        *,
        signed_qty: int,
        size_mult: float,
        spot_min_qty: object,
        spot_max_qty: object,
    ) -> int:
        qty = int(signed_qty)
        if qty == 0:
            return 0
        try:
            mult = float(size_mult)
        except (TypeError, ValueError):
            return int(qty)
        if mult <= 0 or abs(mult - 1.0) <= 1e-12:
            return int(qty)

        q_sign = 1 if qty > 0 else -1
        scaled_abs = int(abs(qty) * float(mult))
        if scaled_abs <= 0:
            scaled_abs = 1

        max_qty = cls._parse_int(spot_max_qty, default=0, min_value=0)
        if max_qty > 0:
            scaled_abs = min(int(scaled_abs), int(max_qty))

        min_qty = cls._parse_int(spot_min_qty, default=1, min_value=1)
        scaled_abs = max(max(1, int(min_qty)), int(scaled_abs))
        return int(q_sign) * int(scaled_abs)

    @classmethod
    def shock_exit_pct_multipliers(
        cls,
        filters: Mapping[str, object] | object | None,
        *,
        shock: bool | None,
    ) -> tuple[float, float]:
        if not bool(shock) or filters is None:
            return 1.0, 1.0
        cfg = cls.policy_config(filters=filters)
        return float(cfg.shock_stop_loss_pct_mult), float(cfg.shock_profit_target_pct_mult)

    @classmethod
    def scale_exit_pcts(
        cls,
        *,
        stop_loss_pct: float | None,
        profit_target_pct: float | None,
        stop_mult: float = 1.0,
        profit_mult: float = 1.0,
    ) -> tuple[float | None, float | None]:
        try:
            stop_mult_f = float(stop_mult)
        except (TypeError, ValueError):
            stop_mult_f = 1.0
        try:
            profit_mult_f = float(profit_mult)
        except (TypeError, ValueError):
            profit_mult_f = 1.0
        if stop_mult_f <= 0:
            stop_mult_f = 1.0
        if profit_mult_f <= 0:
            profit_mult_f = 1.0

        stop_pct: float | None
        try:
            stop_pct = float(stop_loss_pct) if stop_loss_pct is not None else None
        except (TypeError, ValueError):
            stop_pct = None
        if stop_pct is not None:
            stop_pct = min(stop_pct * stop_mult_f, 0.99) if stop_pct > 0 else None

        profit_pct: float | None
        try:
            profit_pct = float(profit_target_pct) if profit_target_pct is not None else None
        except (TypeError, ValueError):
            profit_pct = None
        if profit_pct is not None:
            profit_pct = min(profit_pct * profit_mult_f, 0.99) if profit_pct > 0 else None

        return stop_pct, profit_pct

    @classmethod
    def resolve_position_intent(
        cls,
        *,
        strategy: Mapping[str, object] | object | None,
        current_qty: int,
        target_qty: int,
        policy_config: SpotPolicyConfigView | None = None,
    ) -> SpotIntentDecision:
        cfg = policy_config or cls.policy_config(strategy=strategy, filters=None)
        mode = str(cfg.spot_resize_mode or "off")

        try:
            current = int(current_qty)
        except (TypeError, ValueError):
            current = 0
        try:
            target = int(target_qty)
        except (TypeError, ValueError):
            target = 0

        delta = int(target) - int(current)
        if delta == 0:
            return SpotIntentDecision(
                intent="hold",
                current_qty=int(current),
                target_qty=int(target),
                delta_qty=0,
                order_action=None,
                order_qty=0,
                reason="target_met",
                blocked=False,
                clamped=False,
                resize_mode=mode,
                resize_kind=None,
            )

        def _decision(
            *,
            intent: str,
            reason: str,
            blocked: bool = False,
            clamped: bool = False,
            delta_qty: int = delta,
            resize_kind: str | None = None,
        ) -> SpotIntentDecision:
            dq = int(delta_qty)
            qty = abs(int(dq))
            action = "BUY" if dq > 0 else "SELL" if dq < 0 else None
            return SpotIntentDecision(
                intent=str(intent),
                current_qty=int(current),
                target_qty=int(target),
                delta_qty=int(dq),
                order_action=action,
                order_qty=int(qty),
                reason=str(reason),
                blocked=bool(blocked),
                clamped=bool(clamped),
                resize_mode=mode,
                resize_kind=str(resize_kind) if resize_kind in ("scale_in", "scale_out", "flip") else None,
            )

        if current == 0:
            return _decision(intent="enter", reason="from_flat")
        if target == 0:
            return _decision(intent="exit", reason="target_zero")

        current_sign = 1 if current > 0 else -1
        target_sign = 1 if target > 0 else -1
        if current_sign != target_sign:
            return _decision(
                intent="hold",
                reason="flip_requires_exit_then_entry",
                blocked=True,
                resize_kind="flip",
                delta_qty=0,
            )

        if mode != "target":
            return _decision(
                intent="hold",
                reason="resize_mode_off",
                blocked=True,
                delta_qty=0,
            )

        resize_kind = "scale_in" if ((current > 0 and delta > 0) or (current < 0 and delta < 0)) else "scale_out"
        if resize_kind == "scale_in" and not bool(cfg.spot_resize_allow_scale_in):
            return _decision(
                intent="hold",
                reason="scale_in_disabled",
                blocked=True,
                resize_kind=resize_kind,
                delta_qty=0,
            )
        if resize_kind == "scale_out" and not bool(cfg.spot_resize_allow_scale_out):
            return _decision(
                intent="hold",
                reason="scale_out_disabled",
                blocked=True,
                resize_kind=resize_kind,
                delta_qty=0,
            )

        min_delta = max(1, int(cfg.spot_resize_min_delta_qty))
        if abs(int(delta)) < int(min_delta):
            return _decision(
                intent="hold",
                reason="min_delta_gate",
                blocked=True,
                resize_kind=resize_kind,
                delta_qty=0,
            )

        eff_delta = int(delta)
        clamped = False
        max_step = max(0, int(cfg.spot_resize_max_step_qty))
        if max_step > 0 and abs(int(eff_delta)) > int(max_step):
            eff_delta = int(max_step) if eff_delta > 0 else -int(max_step)
            clamped = True

        if eff_delta == 0:
            return _decision(
                intent="hold",
                reason="max_step_gate",
                blocked=True,
                resize_kind=resize_kind,
                delta_qty=0,
            )
        return _decision(
            intent="resize",
            reason="target_delta",
            blocked=False,
            clamped=bool(clamped),
            delta_qty=int(eff_delta),
            resize_kind=resize_kind,
        )

    @classmethod
    def stop_level(
        cls,
        *,
        entry_price: float,
        qty: int,
        stop_loss_price: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> float | None:
        if stop_loss_price is not None:
            stop = float(stop_loss_price)
            return stop if stop > 0 else None
        if stop_loss_pct is None:
            return None
        entry = float(entry_price)
        if entry <= 0:
            return None
        pct = float(stop_loss_pct)
        if pct <= 0:
            return None
        if int(qty) > 0:
            return entry * (1.0 - pct)
        if int(qty) < 0:
            return entry * (1.0 + pct)
        return None
