"""Shared spot decision policy used by both backtests and live UI.

Pipeline (entry sizing path):
1) Parse/normalize risk overlay factors from filters.
2) Resolve optional shock-aware stop/profit pct multipliers.
3) Compute base qty by sizing mode (fixed / notional_pct / risk_pct).
4) Apply overlay multipliers (riskoff/riskpanic/riskpop/shock).
5) Apply optional dynamic shock ATR throttle.
6) Apply caps/flooring (notional cap, buying power, min/max qty, lot rounding).
7) Apply optional branch size multiplier for branch-aware entries.

This module keeps strategy-evolution knobs in one place so live and backtest
paths remain behaviorally aligned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime

from ..time_utils import NaiveTsModeInput
from ..time_utils import trade_date as _trade_date_shared
from ..time_utils import trade_hour_et as _trade_hour_et_shared
from .fill_modes import SPOT_FILL_MODE_CLOSE, normalize_spot_fill_mode
from .graph import SpotPolicyGraph
from .packs import resolve_pack


@dataclass(frozen=True)
class SpotPolicyConfigView:
    """Sanitized/defaulted view of strategy+filter knobs used by spot policy."""

    spot_policy_pack: str | None = None
    quantity_mult: int = 1
    sizing_mode: str = "fixed"
    spot_notional_pct: float = 0.0
    spot_risk_pct: float = 0.0
    spot_short_risk_mult: float = 1.0
    spot_max_notional_pct: float = 1.0
    spot_min_qty: int = 1
    spot_max_qty: int = 0
    spot_resize_mode: str = "off"
    spot_resize_min_delta_qty: int = 1
    spot_resize_max_step_qty: int = 0
    spot_resize_allow_scale_in: bool = True
    spot_resize_allow_scale_out: bool = True
    spot_resize_cooldown_bars: int = 0

    spot_dual_branch_enabled: bool = False
    spot_branch_a_size_mult: float = 1.0
    spot_branch_b_size_mult: float = 1.0

    riskoff_mode: str = "hygiene"
    riskoff_long_factor: float = 1.0
    riskoff_short_factor: float = 1.0
    riskpanic_long_factor: float = 1.0
    riskpanic_short_factor: float = 1.0
    riskpop_long_factor: float = 1.0
    riskpop_short_factor: float = 1.0

    riskpanic_long_scale_mode: str = "off"
    riskpanic_neg_gap_ratio_min: float = 0.0
    riskpanic_long_scale_tr_delta_max_pct: float = 1.0

    shock_short_risk_mult_factor: float = 1.0
    shock_short_boost_min_down_streak_bars: int = 1
    shock_short_boost_require_regime_down: bool = False
    shock_short_boost_require_entry_down: bool = False
    shock_short_boost_min_dist_on_pp: float = 0.0
    shock_short_boost_max_dist_on_pp: float = 0.0
    shock_short_entry_max_dist_on_pp: float = 0.0
    shock_prearm_dist_on_max_pp: float = 0.0
    shock_prearm_min_dist_on_vel_pp: float = 0.0
    shock_prearm_min_dist_on_accel_pp: float = 0.0
    shock_prearm_min_streak_bars: int = 0
    shock_prearm_short_risk_mult_factor: float = 1.0
    shock_prearm_require_regime_down: bool = True
    shock_prearm_require_entry_down: bool = True
    shock_on_drawdown_pct: float = -20.0
    shock_off_drawdown_pct: float = -10.0
    shock_long_risk_mult_factor: float = 1.0
    shock_long_risk_mult_factor_down: float = 1.0
    shock_long_boost_require_regime_up: bool = False
    shock_long_boost_require_entry_up: bool = False
    shock_long_boost_max_dist_off_pp: float = 0.0
    shock_stop_loss_pct_mult: float = 1.0
    shock_profit_target_pct_mult: float = 1.0
    shock_risk_scale_target_atr_pct: float | None = None
    shock_risk_scale_min_mult: float = 0.2
    shock_risk_scale_apply_to: str = "risk"
    liq_boost_enable: bool = False
    liq_boost_score_min: float = 2.0
    liq_boost_score_span: float = 2.0
    liq_boost_max_risk_mult: float = 1.0
    liq_boost_cap_floor_frac: float = 0.0
    liq_boost_require_alignment: bool = True
    liq_boost_require_shock: bool = False

    risk_entry_cutoff_hour_et: int | None = None

    @staticmethod
    def _get(source: Mapping[str, object] | object | None, key: str, default: object = None) -> object:
        if source is None:
            return default
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    @classmethod
    def _parse_int(cls, value: object, *, default: int, min_value: int | None = None) -> int:
        try:
            parsed = int(value) if value is not None else int(default)
        except (TypeError, ValueError):
            parsed = int(default)
        if min_value is not None and parsed < int(min_value):
            return int(min_value)
        return int(parsed)

    @classmethod
    def _parse_float(cls, value: object, *, default: float, min_value: float | None = None) -> float:
        try:
            parsed = float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            parsed = float(default)
        if min_value is not None and parsed < float(min_value):
            return float(min_value)
        return float(parsed)

    @classmethod
    def _parse_optional_float(cls, value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_factor(cls, value: object) -> float:
        factor = cls._parse_float(value, default=1.0)
        if factor < 0:
            return 1.0
        return float(factor)

    @staticmethod
    def _parse_bool(value: object, *, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return bool(value)
        raw = str(value).strip().lower()
        if raw in ("1", "true", "yes", "on", "y"):
            return True
        if raw in ("0", "false", "no", "off", "n"):
            return False
        return bool(default)

    @staticmethod
    def _normalize_sizing_mode(raw: object | None) -> str:
        mode = str(raw or "fixed").strip().lower()
        if mode not in ("fixed", "notional_pct", "risk_pct"):
            return "fixed"
        return mode

    @staticmethod
    def _normalize_riskoff_mode(raw: object | None) -> str:
        mode = str(raw or "hygiene").strip().lower()
        if mode not in ("hygiene", "directional"):
            return "hygiene"
        return mode

    @staticmethod
    def _normalize_riskpanic_scale_mode(raw: object | None) -> str:
        mode = str(raw or "off").strip().lower()
        if mode in ("linear", "lin", "delta", "linear_delta", "linear_tr_delta"):
            return "linear"
        return "off"

    @staticmethod
    def _normalize_shock_scale_apply_to(raw: object | None) -> str:
        apply_to = str(raw or "risk").strip().lower()
        if apply_to in (
            "cap",
            "cap_only",
            "cap-only",
            "notional_cap",
            "max_notional",
            "notional_pct_cap",
        ):
            return "cap"
        if apply_to in ("both", "all", "risk_and_cap", "cap_and_risk", "cap+risk"):
            return "both"
        return "risk"

    @staticmethod
    def _normalize_resize_mode(raw: object | None) -> str:
        mode = str(raw or "off").strip().lower()
        if mode in ("target", "target_qty", "target-qty", "targetqty"):
            return "target"
        return "off"

    @classmethod
    def _risk_entry_cutoff_hour_et(cls, filters: Mapping[str, object] | object | None) -> int | None:
        if filters is None:
            return None
        raw_cutoff_et = cls._get(filters, "risk_entry_cutoff_hour_et")
        if raw_cutoff_et is not None:
            try:
                return int(raw_cutoff_et)
            except (TypeError, ValueError):
                return None

        raw_start_et = cls._get(filters, "entry_start_hour_et")
        raw_end_et = cls._get(filters, "entry_end_hour_et")
        raw_start = cls._get(filters, "entry_start_hour")
        raw_end = cls._get(filters, "entry_end_hour")
        if raw_start_et is None and raw_end_et is not None:
            try:
                return int(raw_end_et)
            except (TypeError, ValueError):
                return None
        if raw_start is None and raw_end is not None:
            try:
                return int(raw_end)
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def from_sources(
        cls,
        *,
        strategy: Mapping[str, object] | object | None = None,
        filters: Mapping[str, object] | object | None = None,
    ) -> SpotPolicyConfigView:
        pack = resolve_pack(strategy=strategy, filters=filters)
        missing = object()

        def _sget(key: str, default: object = None) -> object:
            raw = cls._get(strategy, key, missing)
            if raw is not missing:
                return raw
            if pack is not None and key in pack.strategy_defaults:
                return pack.strategy_defaults[key]
            return default

        def _fget(key: str, default: object = None) -> object:
            raw = cls._get(filters, key, missing)
            if raw is not missing:
                return raw
            if pack is not None and key in pack.filter_defaults:
                return pack.filter_defaults[key]
            return default

        quantity_mult = max(1, cls._parse_int(_sget("quantity", 1), default=1, min_value=1))
        sizing_mode = cls._normalize_sizing_mode(_sget("spot_sizing_mode", "fixed"))

        spot_notional_pct = max(0.0, cls._parse_float(_sget("spot_notional_pct"), default=0.0))
        spot_risk_pct = max(0.0, cls._parse_float(_sget("spot_risk_pct"), default=0.0))
        spot_short_risk_mult = max(0.0, cls._parse_float(_sget("spot_short_risk_mult"), default=1.0))
        spot_max_notional_pct = max(0.0, cls._parse_float(_sget("spot_max_notional_pct"), default=1.0))
        spot_min_qty = cls._parse_int(_sget("spot_min_qty"), default=1, min_value=1)
        spot_max_qty = cls._parse_int(_sget("spot_max_qty"), default=0, min_value=0)
        spot_resize_mode = cls._normalize_resize_mode(_sget("spot_resize_mode", "off"))
        spot_resize_min_delta_qty = cls._parse_int(
            _sget("spot_resize_min_delta_qty"),
            default=1,
            min_value=1,
        )
        spot_resize_max_step_qty = cls._parse_int(
            _sget("spot_resize_max_step_qty"),
            default=0,
            min_value=0,
        )
        spot_resize_allow_scale_in = cls._parse_bool(
            _sget("spot_resize_allow_scale_in"),
            default=True,
        )
        spot_resize_allow_scale_out = cls._parse_bool(
            _sget("spot_resize_allow_scale_out"),
            default=True,
        )
        spot_resize_cooldown_bars = cls._parse_int(
            _sget("spot_resize_cooldown_bars"),
            default=0,
            min_value=0,
        )

        spot_dual_branch_enabled = bool(_sget("spot_dual_branch_enabled", False))
        spot_branch_a_size_mult = cls._parse_float(_sget("spot_branch_a_size_mult"), default=1.0)
        if spot_branch_a_size_mult <= 0:
            spot_branch_a_size_mult = 1.0
        spot_branch_b_size_mult = cls._parse_float(_sget("spot_branch_b_size_mult"), default=1.0)
        if spot_branch_b_size_mult <= 0:
            spot_branch_b_size_mult = 1.0

        riskoff_mode = cls._normalize_riskoff_mode(_fget("riskoff_mode"))
        riskoff_short_factor = cls._parse_factor(_fget("riskoff_short_risk_mult_factor"))
        riskoff_long_factor = cls._parse_factor(_fget("riskoff_long_risk_mult_factor"))
        riskpanic_long_factor = cls._parse_factor(_fget("riskpanic_long_risk_mult_factor"))
        riskpanic_short_factor = cls._parse_factor(_fget("riskpanic_short_risk_mult_factor"))
        riskpop_long_factor = cls._parse_factor(_fget("riskpop_long_risk_mult_factor"))
        riskpop_short_factor = cls._parse_factor(_fget("riskpop_short_risk_mult_factor"))

        riskpanic_long_scale_mode = cls._normalize_riskpanic_scale_mode(
            _fget("riskpanic_long_scale_mode", "off")
        )
        riskpanic_neg_gap_ratio_min = cls._parse_float(
            _fget("riskpanic_neg_gap_ratio_min"),
            default=0.0,
        )
        riskpanic_neg_gap_ratio_min = float(max(0.0, min(1.0, riskpanic_neg_gap_ratio_min)))

        raw_delta_max = _fget("riskpanic_long_scale_tr_delta_max_pct")
        if raw_delta_max is None:
            raw_delta_max = _fget("riskpanic_tr5_med_delta_min_pct")
        delta_max = cls._parse_optional_float(raw_delta_max)
        if delta_max is None or delta_max <= 0:
            delta_max = 1.0

        shock_short_risk_mult_factor = cls._parse_factor(_fget("shock_short_risk_mult_factor"))
        shock_short_boost_min_down_streak_bars = cls._parse_int(
            _fget("shock_short_boost_min_down_streak_bars"),
            default=1,
            min_value=1,
        )
        shock_short_boost_require_regime_down = cls._parse_bool(
            _fget("shock_short_boost_require_regime_down"),
            default=False,
        )
        shock_short_boost_require_entry_down = cls._parse_bool(
            _fget("shock_short_boost_require_entry_down"),
            default=False,
        )
        shock_short_boost_min_dist_on_pp = cls._parse_float(
            _fget("shock_short_boost_min_dist_on_pp"),
            default=0.0,
        )
        if shock_short_boost_min_dist_on_pp < 0:
            shock_short_boost_min_dist_on_pp = 0.0
        shock_short_boost_max_dist_on_pp = cls._parse_float(
            _fget("shock_short_boost_max_dist_on_pp"),
            default=0.0,
        )
        if shock_short_boost_max_dist_on_pp < 0:
            shock_short_boost_max_dist_on_pp = 0.0
        shock_short_entry_max_dist_on_pp = cls._parse_float(
            _fget("shock_short_entry_max_dist_on_pp"),
            default=0.0,
        )
        if shock_short_entry_max_dist_on_pp < 0:
            shock_short_entry_max_dist_on_pp = 0.0
        shock_prearm_dist_on_max_pp = cls._parse_float(
            _fget("shock_prearm_dist_on_max_pp"),
            default=0.0,
        )
        if shock_prearm_dist_on_max_pp < 0:
            shock_prearm_dist_on_max_pp = 0.0
        shock_prearm_min_dist_on_vel_pp = cls._parse_float(
            _fget("shock_prearm_min_dist_on_vel_pp"),
            default=0.0,
        )
        if shock_prearm_min_dist_on_vel_pp < 0:
            shock_prearm_min_dist_on_vel_pp = 0.0
        shock_prearm_min_dist_on_accel_pp = cls._parse_float(
            _fget("shock_prearm_min_dist_on_accel_pp"),
            default=0.0,
        )
        if shock_prearm_min_dist_on_accel_pp < 0:
            shock_prearm_min_dist_on_accel_pp = 0.0
        shock_prearm_min_streak_bars = cls._parse_int(
            _fget("shock_prearm_min_streak_bars"),
            default=0,
            min_value=0,
        )
        shock_prearm_short_risk_mult_factor = cls._parse_factor(_fget("shock_prearm_short_risk_mult_factor"))
        shock_prearm_require_regime_down = cls._parse_bool(
            _fget("shock_prearm_require_regime_down"),
            default=True,
        )
        shock_prearm_require_entry_down = cls._parse_bool(
            _fget("shock_prearm_require_entry_down"),
            default=True,
        )
        shock_on_drawdown_pct = cls._parse_float(
            _fget("shock_on_drawdown_pct"),
            default=-20.0,
        )
        shock_off_drawdown_pct = cls._parse_float(
            _fget("shock_off_drawdown_pct"),
            default=-10.0,
        )
        if float(shock_off_drawdown_pct) < float(shock_on_drawdown_pct):
            shock_off_drawdown_pct = float(shock_on_drawdown_pct)
        shock_long_risk_mult_factor = cls._parse_factor(_fget("shock_long_risk_mult_factor"))
        shock_long_risk_mult_factor_down = cls._parse_factor(_fget("shock_long_risk_mult_factor_down"))
        shock_long_boost_require_regime_up = cls._parse_bool(
            _fget("shock_long_boost_require_regime_up"),
            default=False,
        )
        shock_long_boost_require_entry_up = cls._parse_bool(
            _fget("shock_long_boost_require_entry_up"),
            default=False,
        )
        shock_long_boost_max_dist_off_pp = cls._parse_float(
            _fget("shock_long_boost_max_dist_off_pp"),
            default=0.0,
        )
        if shock_long_boost_max_dist_off_pp < 0:
            shock_long_boost_max_dist_off_pp = 0.0

        shock_stop_loss_pct_mult = cls._parse_float(_fget("shock_stop_loss_pct_mult"), default=1.0)
        if shock_stop_loss_pct_mult <= 0:
            shock_stop_loss_pct_mult = 1.0
        shock_profit_target_pct_mult = cls._parse_float(
            _fget("shock_profit_target_pct_mult"),
            default=1.0,
        )
        if shock_profit_target_pct_mult <= 0:
            shock_profit_target_pct_mult = 1.0

        target_atr = cls._parse_optional_float(_fget("shock_risk_scale_target_atr_pct"))
        if target_atr is not None and target_atr <= 0:
            target_atr = None

        shock_risk_scale_min_mult = cls._parse_float(_fget("shock_risk_scale_min_mult"), default=0.2)
        shock_risk_scale_min_mult = float(max(0.0, min(1.0, shock_risk_scale_min_mult)))
        shock_risk_scale_apply_to = cls._normalize_shock_scale_apply_to(_fget("shock_risk_scale_apply_to"))
        liq_boost_enable = cls._parse_bool(_fget("liq_boost_enable"), default=False)
        liq_boost_score_min = cls._parse_float(_fget("liq_boost_score_min"), default=2.0)
        liq_boost_score_span = cls._parse_float(_fget("liq_boost_score_span"), default=2.0)
        if liq_boost_score_span <= 0:
            liq_boost_score_span = 2.0
        liq_boost_max_risk_mult = cls._parse_float(_fget("liq_boost_max_risk_mult"), default=1.0)
        if liq_boost_max_risk_mult < 1.0:
            liq_boost_max_risk_mult = 1.0
        liq_boost_cap_floor_frac = cls._parse_float(_fget("liq_boost_cap_floor_frac"), default=0.0)
        liq_boost_cap_floor_frac = float(max(0.0, min(1.0, liq_boost_cap_floor_frac)))
        liq_boost_require_alignment = cls._parse_bool(_fget("liq_boost_require_alignment"), default=True)
        liq_boost_require_shock = cls._parse_bool(_fget("liq_boost_require_shock"), default=False)

        risk_entry_cutoff_hour_et = cls._risk_entry_cutoff_hour_et(filters)
        if risk_entry_cutoff_hour_et is None:
            raw_cutoff = _fget("risk_entry_cutoff_hour_et")
            if raw_cutoff is not None:
                try:
                    risk_entry_cutoff_hour_et = int(raw_cutoff)
                except (TypeError, ValueError):
                    risk_entry_cutoff_hour_et = None

        return cls(
            spot_policy_pack=str(pack.name) if pack is not None else None,
            quantity_mult=int(quantity_mult),
            sizing_mode=str(sizing_mode),
            spot_notional_pct=float(spot_notional_pct),
            spot_risk_pct=float(spot_risk_pct),
            spot_short_risk_mult=float(spot_short_risk_mult),
            spot_max_notional_pct=float(spot_max_notional_pct),
            spot_min_qty=int(spot_min_qty),
            spot_max_qty=int(spot_max_qty),
            spot_resize_mode=str(spot_resize_mode),
            spot_resize_min_delta_qty=int(spot_resize_min_delta_qty),
            spot_resize_max_step_qty=int(spot_resize_max_step_qty),
            spot_resize_allow_scale_in=bool(spot_resize_allow_scale_in),
            spot_resize_allow_scale_out=bool(spot_resize_allow_scale_out),
            spot_resize_cooldown_bars=int(spot_resize_cooldown_bars),
            spot_dual_branch_enabled=bool(spot_dual_branch_enabled),
            spot_branch_a_size_mult=float(spot_branch_a_size_mult),
            spot_branch_b_size_mult=float(spot_branch_b_size_mult),
            riskoff_mode=str(riskoff_mode),
            riskoff_long_factor=float(riskoff_long_factor),
            riskoff_short_factor=float(riskoff_short_factor),
            riskpanic_long_factor=float(riskpanic_long_factor),
            riskpanic_short_factor=float(riskpanic_short_factor),
            riskpop_long_factor=float(riskpop_long_factor),
            riskpop_short_factor=float(riskpop_short_factor),
            riskpanic_long_scale_mode=str(riskpanic_long_scale_mode),
            riskpanic_neg_gap_ratio_min=float(riskpanic_neg_gap_ratio_min),
            riskpanic_long_scale_tr_delta_max_pct=float(delta_max),
            shock_short_risk_mult_factor=float(shock_short_risk_mult_factor),
            shock_short_boost_min_down_streak_bars=int(shock_short_boost_min_down_streak_bars),
            shock_short_boost_require_regime_down=bool(shock_short_boost_require_regime_down),
            shock_short_boost_require_entry_down=bool(shock_short_boost_require_entry_down),
            shock_short_boost_min_dist_on_pp=float(shock_short_boost_min_dist_on_pp),
            shock_short_boost_max_dist_on_pp=float(shock_short_boost_max_dist_on_pp),
            shock_short_entry_max_dist_on_pp=float(shock_short_entry_max_dist_on_pp),
            shock_prearm_dist_on_max_pp=float(shock_prearm_dist_on_max_pp),
            shock_prearm_min_dist_on_vel_pp=float(shock_prearm_min_dist_on_vel_pp),
            shock_prearm_min_dist_on_accel_pp=float(shock_prearm_min_dist_on_accel_pp),
            shock_prearm_min_streak_bars=int(shock_prearm_min_streak_bars),
            shock_prearm_short_risk_mult_factor=float(shock_prearm_short_risk_mult_factor),
            shock_prearm_require_regime_down=bool(shock_prearm_require_regime_down),
            shock_prearm_require_entry_down=bool(shock_prearm_require_entry_down),
            shock_on_drawdown_pct=float(shock_on_drawdown_pct),
            shock_off_drawdown_pct=float(shock_off_drawdown_pct),
            shock_long_risk_mult_factor=float(shock_long_risk_mult_factor),
            shock_long_risk_mult_factor_down=float(shock_long_risk_mult_factor_down),
            shock_long_boost_require_regime_up=bool(shock_long_boost_require_regime_up),
            shock_long_boost_require_entry_up=bool(shock_long_boost_require_entry_up),
            shock_long_boost_max_dist_off_pp=float(shock_long_boost_max_dist_off_pp),
            shock_stop_loss_pct_mult=float(shock_stop_loss_pct_mult),
            shock_profit_target_pct_mult=float(shock_profit_target_pct_mult),
            shock_risk_scale_target_atr_pct=float(target_atr) if target_atr is not None else None,
            shock_risk_scale_min_mult=float(shock_risk_scale_min_mult),
            shock_risk_scale_apply_to=str(shock_risk_scale_apply_to),
            liq_boost_enable=bool(liq_boost_enable),
            liq_boost_score_min=float(liq_boost_score_min),
            liq_boost_score_span=float(liq_boost_score_span),
            liq_boost_max_risk_mult=float(liq_boost_max_risk_mult),
            liq_boost_cap_floor_frac=float(liq_boost_cap_floor_frac),
            liq_boost_require_alignment=bool(liq_boost_require_alignment),
            liq_boost_require_shock=bool(liq_boost_require_shock),
            risk_entry_cutoff_hour_et=risk_entry_cutoff_hour_et,
        )


@dataclass(frozen=True)
class SpotRuntimeSpec:
    """Sanitized/defaulted runtime execution knobs shared across backtest + live."""

    spot_policy_pack: str | None = None
    entry_fill_mode: str = "close"
    flip_exit_fill_mode: str = "close"
    exit_mode: str = "pct"
    intrabar_exits: bool = False
    close_eod: bool = False
    spread: float = 0.0
    commission_per_share: float = 0.0
    commission_min: float = 0.0
    slippage_per_share: float = 0.0
    mark_to_market: str = "close"
    drawdown_mode: str = "close"

    @staticmethod
    def _normalize_fill_mode(raw: object | None) -> str:
        return normalize_spot_fill_mode(raw, default=SPOT_FILL_MODE_CLOSE)

    @staticmethod
    def _normalize_exit_mode(raw: object | None) -> str:
        mode = str(raw or "pct").strip().lower()
        if mode not in ("pct", "atr"):
            return "pct"
        return mode

    @staticmethod
    def _normalize_mark_to_market(raw: object | None) -> str:
        mode = str(raw or "close").strip().lower()
        if mode not in ("close", "liquidation"):
            return "close"
        return mode

    @staticmethod
    def _normalize_drawdown_mode(raw: object | None) -> str:
        mode = str(raw or "close").strip().lower()
        if mode not in ("close", "intrabar"):
            return "close"
        return mode

    @classmethod
    def from_sources(
        cls,
        *,
        strategy: Mapping[str, object] | object | None = None,
        filters: Mapping[str, object] | object | None = None,
    ) -> SpotRuntimeSpec:
        pack = resolve_pack(strategy=strategy, filters=filters)
        missing = object()

        def _sget(key: str, default: object = None) -> object:
            raw = SpotPolicyConfigView._get(strategy, key, missing)
            if raw is not missing:
                return raw
            if pack is not None and key in pack.strategy_defaults:
                return pack.strategy_defaults[key]
            return default

        spread = max(0.0, SpotPolicyConfigView._parse_float(_sget("spot_spread"), default=0.0))
        commission_per_share = max(
            0.0,
            SpotPolicyConfigView._parse_float(_sget("spot_commission_per_share"), default=0.0),
        )
        commission_min = max(0.0, SpotPolicyConfigView._parse_float(_sget("spot_commission_min"), default=0.0))
        slippage_per_share = max(
            0.0,
            SpotPolicyConfigView._parse_float(_sget("spot_slippage_per_share"), default=0.0),
        )

        return cls(
            spot_policy_pack=str(pack.name) if pack is not None else None,
            entry_fill_mode=str(cls._normalize_fill_mode(_sget("spot_entry_fill_mode", "close"))),
            flip_exit_fill_mode=str(cls._normalize_fill_mode(_sget("spot_flip_exit_fill_mode", "close"))),
            exit_mode=str(cls._normalize_exit_mode(_sget("spot_exit_mode", "pct"))),
            intrabar_exits=bool(SpotPolicyConfigView._parse_bool(_sget("spot_intrabar_exits"), default=False)),
            close_eod=bool(SpotPolicyConfigView._parse_bool(_sget("spot_close_eod"), default=False)),
            spread=float(spread),
            commission_per_share=float(commission_per_share),
            commission_min=float(commission_min),
            slippage_per_share=float(slippage_per_share),
            mark_to_market=str(cls._normalize_mark_to_market(_sget("spot_mark_to_market", "close"))),
            drawdown_mode=str(cls._normalize_drawdown_mode(_sget("spot_drawdown_mode", "close"))),
        )


@dataclass(frozen=True)
class SpotDecisionTrace:
    """Typed decision trace for spot sizing decisions."""

    action: str
    sizing_mode: str
    lot: int
    quantity_mult: int
    base_signed_qty: int

    entry_price: float
    stop_price: float | None
    stop_loss_pct: float | None
    equity_ref: float
    cash_ref: float | None

    riskoff: bool
    riskpanic: bool
    riskpop: bool
    shock: bool
    shock_dir: str | None
    shock_dir_down_streak_bars: int | None
    shock_atr_pct: float | None
    shock_drawdown_dist_on_pct: float | None
    shock_drawdown_dist_on_vel_pp: float | None
    shock_drawdown_dist_on_accel_pp: float | None
    shock_drawdown_dist_off_pct: float | None
    shock_prearm_down_streak_bars: int | None
    riskoff_mode: str
    risk_dir: str | None
    signal_entry_dir: str | None
    signal_regime_dir: str | None

    riskoff_long_factor: float
    riskoff_short_factor: float
    riskpanic_long_factor: float
    riskpanic_short_factor: float
    riskpop_long_factor: float
    riskpop_short_factor: float
    graph_profile: str | None = None
    graph_entry_policy: str | None = None
    graph_exit_policy: str | None = None
    graph_resize_policy: str | None = None
    graph_risk_overlay_policy: str | None = None
    graph_overlay_trace: dict[str, object] | None = None

    risk_dollars_base: float | None = None
    risk_dollars_final: float | None = None
    per_share_risk: float | None = None

    short_mult_base: float | None = None
    short_mult_final: float | None = None
    shock_long_factor: float | None = None
    shock_long_boost_applied: bool | None = None
    shock_long_boost_gate_reason: str | None = None
    shock_short_factor: float | None = None
    shock_short_boost_applied: bool | None = None
    shock_short_boost_gate_reason: str | None = None
    shock_short_entry_blocked: bool | None = None
    shock_short_entry_gate_reason: str | None = None
    shock_prearm_applied: bool | None = None
    shock_prearm_factor: float | None = None
    shock_prearm_reason: str | None = None
    shock_ramp_applied: bool | None = None
    shock_ramp_dir: str | None = None
    shock_ramp_phase: str | None = None
    shock_ramp_intensity: float | None = None
    shock_ramp_risk_mult: float | None = None
    shock_ramp_cap_floor_frac: float | None = None
    shock_ramp_reason: str | None = None
    shock_ramp_cap_floor_qty: int | None = None

    cap_pct_base: float | None = None
    cap_pct_final: float | None = None
    liq_boost_applied: bool | None = None
    liq_boost_score: float | None = None
    liq_boost_mult: float | None = None
    liq_boost_reason: str | None = None
    liq_boost_cap_floor_frac: float | None = None
    liq_boost_cap_floor_qty: int | None = None

    shock_scale_target_atr_pct: float | None = None
    shock_scale_mult: float | None = None
    shock_scale_apply_to: str | None = None

    desired_qty_pre_caps: int | None = None
    desired_qty_post_caps: int | None = None
    cap_qty: int | None = None
    afford_qty: int | None = None
    fallback_to_lot: bool = False

    lot_rounded_qty: int | None = None
    min_effective_qty: int | None = None
    min_qty_blocked: bool = False

    signed_qty_final: int = 0
    zero_reason: str | None = None

    entry_branch: str | None = None
    branch_size_mult: float | None = None
    signed_qty_after_branch: int | None = None

    def with_branch_scaling(
        self,
        *,
        entry_branch: str | None,
        size_mult: float,
        signed_qty_after_branch: int,
    ) -> SpotDecisionTrace:
        return replace(
            self,
            entry_branch=str(entry_branch) if entry_branch in ("a", "b") else None,
            branch_size_mult=float(size_mult),
            signed_qty_after_branch=int(signed_qty_after_branch),
        )

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpotIntentDecision:
    """Typed intent decision for transitioning from current -> target spot qty."""

    intent: str
    current_qty: int
    target_qty: int
    delta_qty: int
    order_action: str | None = None
    order_qty: int = 0
    reason: str | None = None
    blocked: bool = False
    clamped: bool = False
    resize_mode: str = "off"
    resize_kind: str | None = None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


class SpotPolicy:
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
    ) -> SpotIntentDecision:
        cfg = cls.policy_config(strategy=strategy, filters=None)
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

    @classmethod
    def calc_signed_qty_with_trace(
        cls,
        *,
        strategy: Mapping[str, object] | object,
        filters: Mapping[str, object] | object | None,
        action: str,
        lot: int,
        entry_price: float,
        stop_price: float | None,
        stop_loss_pct: float | None,
        shock: bool | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_dir_down_streak_bars: int | None = None,
        shock_drawdown_dist_on_pct: float | None = None,
        shock_drawdown_dist_on_vel_pp: float | None = None,
        shock_drawdown_dist_on_accel_pp: float | None = None,
        shock_prearm_down_streak_bars: int | None = None,
        shock_ramp: Mapping[str, object] | object | None = None,
        riskoff: bool = False,
        risk_dir: str | None = None,
        riskpanic: bool = False,
        riskpop: bool = False,
        risk: object | None = None,
        signal_entry_dir: str | None = None,
        signal_regime_dir: str | None = None,
        equity_ref: float = 0.0,
        cash_ref: float | None = None,
    ) -> tuple[int, SpotDecisionTrace]:
        cfg = cls.policy_config(strategy=strategy, filters=filters)

        lot_i = max(1, int(lot or 1))
        raw_action = str(action or "BUY").strip().upper()
        if raw_action not in ("BUY", "SELL"):
            raw_action = "BUY"
        base_signed_qty = int(lot_i) * int(cfg.quantity_mult)
        if raw_action != "BUY":
            base_signed_qty = -int(base_signed_qty)

        try:
            entry_price_f = float(entry_price)
        except (TypeError, ValueError):
            entry_price_f = 0.0

        stop_price_clean = None
        if stop_price is not None:
            try:
                stop_price_f = float(stop_price)
            except (TypeError, ValueError):
                stop_price_f = 0.0
            if stop_price_f > 0:
                stop_price_clean = float(stop_price_f)

        stop_loss_pct_clean = None
        if stop_loss_pct is not None:
            try:
                sl_pct = float(stop_loss_pct)
            except (TypeError, ValueError):
                sl_pct = 0.0
            if sl_pct > 0:
                stop_loss_pct_clean = float(sl_pct)

        try:
            equity_ref_f = float(equity_ref)
        except (TypeError, ValueError):
            equity_ref_f = 0.0

        cash_ref_f = None
        if cash_ref is not None:
            try:
                cash_ref_f = float(cash_ref)
            except (TypeError, ValueError):
                cash_ref_f = None

        shock_dir_clean = str(shock_dir) if shock_dir in ("up", "down") else None
        risk_dir_clean = str(risk_dir) if risk_dir in ("up", "down") else None
        signal_entry_dir_clean = str(signal_entry_dir) if signal_entry_dir in ("up", "down") else None
        signal_regime_dir_clean = str(signal_regime_dir) if signal_regime_dir in ("up", "down") else None
        shock_dir_down_streak_clean: int | None = None
        if shock_dir_down_streak_bars is not None:
            try:
                shock_dir_down_streak_clean = max(0, int(shock_dir_down_streak_bars))
            except (TypeError, ValueError):
                shock_dir_down_streak_clean = None
        shock_drawdown_dist_on_clean: float | None = None
        if shock_drawdown_dist_on_pct is not None:
            try:
                shock_drawdown_dist_on_clean = float(shock_drawdown_dist_on_pct)
            except (TypeError, ValueError):
                shock_drawdown_dist_on_clean = None
        shock_drawdown_dist_on_vel_clean: float | None = None
        if shock_drawdown_dist_on_vel_pp is not None:
            try:
                shock_drawdown_dist_on_vel_clean = float(shock_drawdown_dist_on_vel_pp)
            except (TypeError, ValueError):
                shock_drawdown_dist_on_vel_clean = None
        shock_drawdown_dist_on_accel_clean: float | None = None
        if shock_drawdown_dist_on_accel_pp is not None:
            try:
                shock_drawdown_dist_on_accel_clean = float(shock_drawdown_dist_on_accel_pp)
            except (TypeError, ValueError):
                shock_drawdown_dist_on_accel_clean = None
        shock_prearm_down_streak_clean: int | None = None
        if shock_prearm_down_streak_bars is not None:
            try:
                shock_prearm_down_streak_clean = max(0, int(shock_prearm_down_streak_bars))
            except (TypeError, ValueError):
                shock_prearm_down_streak_clean = None

        shock_drawdown_dist_off_clean: float | None = None
        if shock_drawdown_dist_on_clean is not None:
            try:
                dd_on = float(cfg.shock_on_drawdown_pct)
                dd_off = float(cfg.shock_off_drawdown_pct)
                dd_dist_on = float(shock_drawdown_dist_on_clean)
                dd_pct = float(dd_on) - float(dd_dist_on)
                shock_drawdown_dist_off_clean = float(dd_pct) - float(dd_off)
            except (TypeError, ValueError):
                shock_drawdown_dist_off_clean = None

        shock_atr_pct_clean = None
        if shock_atr_pct is not None:
            try:
                shock_atr_pct_f = float(shock_atr_pct)
            except (TypeError, ValueError):
                shock_atr_pct_f = 0.0
            if shock_atr_pct_f > 0:
                shock_atr_pct_clean = float(shock_atr_pct_f)

        graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=filters)
        risk_tr_ratio = None
        risk_slope_med = None
        risk_slope_vel = None
        if risk is not None:
            raw_tr_ratio = getattr(risk, "tr_ratio", None)
            if raw_tr_ratio is not None:
                try:
                    risk_tr_ratio = float(raw_tr_ratio)
                except (TypeError, ValueError):
                    risk_tr_ratio = None
            raw_slope_med = getattr(risk, "tr_median_delta_pct", None)
            if raw_slope_med is not None:
                try:
                    risk_slope_med = float(raw_slope_med)
                except (TypeError, ValueError):
                    risk_slope_med = None
            raw_slope_vel = getattr(risk, "tr_slope_vel_pct", None)
            if raw_slope_vel is not None:
                try:
                    risk_slope_vel = float(raw_slope_vel)
                except (TypeError, ValueError):
                    risk_slope_vel = None
        graph_overlay = graph.resolve_risk_overlay_adjustments(
            strategy=strategy,
            filters=filters,
            action=str(raw_action),
            shock_atr_pct=shock_atr_pct_clean,
            tr_ratio=risk_tr_ratio,
            slope_med_pct=risk_slope_med,
            slope_vel_pct=risk_slope_vel,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            shock=bool(shock),
            shock_dir=shock_dir_clean,
        )
        cap_pct_base = float(cfg.spot_max_notional_pct)
        cap_pct = float(cap_pct_base) * float(graph_overlay.cap_mult)

        trace = SpotDecisionTrace(
            action=str(raw_action),
            sizing_mode=str(cfg.sizing_mode),
            lot=int(lot_i),
            quantity_mult=int(cfg.quantity_mult),
            base_signed_qty=int(base_signed_qty),
            entry_price=float(entry_price_f),
            stop_price=float(stop_price_clean) if stop_price_clean is not None else None,
            stop_loss_pct=float(stop_loss_pct_clean) if stop_loss_pct_clean is not None else None,
            equity_ref=float(equity_ref_f),
            cash_ref=float(cash_ref_f) if cash_ref_f is not None else None,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            shock=bool(shock),
            shock_dir=shock_dir_clean,
            shock_dir_down_streak_bars=int(shock_dir_down_streak_clean)
            if shock_dir_down_streak_clean is not None
            else None,
            shock_atr_pct=float(shock_atr_pct_clean) if shock_atr_pct_clean is not None else None,
            shock_drawdown_dist_on_pct=(
                float(shock_drawdown_dist_on_clean) if shock_drawdown_dist_on_clean is not None else None
            ),
            shock_drawdown_dist_on_vel_pp=(
                float(shock_drawdown_dist_on_vel_clean)
                if shock_drawdown_dist_on_vel_clean is not None
                else None
            ),
            shock_drawdown_dist_on_accel_pp=(
                float(shock_drawdown_dist_on_accel_clean)
                if shock_drawdown_dist_on_accel_clean is not None
                else None
            ),
            shock_drawdown_dist_off_pct=(
                float(shock_drawdown_dist_off_clean) if shock_drawdown_dist_off_clean is not None else None
            ),
            shock_prearm_down_streak_bars=(
                int(shock_prearm_down_streak_clean) if shock_prearm_down_streak_clean is not None else None
            ),
            riskoff_mode=str(cfg.riskoff_mode),
            risk_dir=risk_dir_clean,
            signal_entry_dir=signal_entry_dir_clean,
            signal_regime_dir=signal_regime_dir_clean,
            riskoff_long_factor=float(cfg.riskoff_long_factor),
            riskoff_short_factor=float(cfg.riskoff_short_factor),
            riskpanic_long_factor=float(cfg.riskpanic_long_factor),
            riskpanic_short_factor=float(cfg.riskpanic_short_factor),
            riskpop_long_factor=float(cfg.riskpop_long_factor),
            riskpop_short_factor=float(cfg.riskpop_short_factor),
            graph_profile=str(graph.profile_name),
            graph_entry_policy=str(graph.entry_policy),
            graph_exit_policy=str(graph.exit_policy),
            graph_resize_policy=str(graph.resize_policy),
            graph_risk_overlay_policy=str(graph.risk_overlay_policy),
            graph_overlay_trace=graph_overlay.as_payload(),
            cap_pct_base=float(cap_pct_base),
            cap_pct_final=float(cap_pct),
        )

        def _update_trace(**kwargs: object) -> None:
            nonlocal trace
            trace = replace(trace, **kwargs)

        if str(cfg.sizing_mode) == "fixed":
            abs_qty = abs(int(base_signed_qty))
            _update_trace(
                desired_qty_pre_caps=int(abs_qty),
                desired_qty_post_caps=int(abs_qty),
                lot_rounded_qty=int(abs_qty),
                min_effective_qty=max(int(cfg.spot_min_qty), int(lot_i)),
                signed_qty_final=int(base_signed_qty),
            )
            return int(base_signed_qty), trace

        if float(entry_price_f) <= 0:
            _update_trace(zero_reason="invalid_entry_price", signed_qty_final=0)
            return 0, trace

        desired_qty = 0

        if str(cfg.sizing_mode) == "notional_pct":
            if float(cfg.spot_notional_pct) > 0 and float(equity_ref_f) > 0:
                desired_qty = int((float(equity_ref_f) * float(cfg.spot_notional_pct)) / float(entry_price_f))
        else:
            stop_level = None
            if stop_price_clean is not None:
                stop_level = float(stop_price_clean)
            elif stop_loss_pct_clean is not None:
                stop_level = cls.stop_level(
                    entry_price=float(entry_price_f),
                    qty=1 if raw_action == "BUY" else -1,
                    stop_loss_pct=float(stop_loss_pct_clean),
                )

            if stop_level is not None and float(cfg.spot_risk_pct) > 0 and float(equity_ref_f) > 0:
                per_share_risk = abs(float(entry_price_f) - float(stop_level))
                risk_dollars = float(equity_ref_f) * float(cfg.spot_risk_pct)
                _update_trace(
                    per_share_risk=float(per_share_risk),
                    risk_dollars_base=float(risk_dollars),
                    risk_dollars_final=float(risk_dollars),
                )

                if raw_action == "BUY":
                    if bool(riskoff):
                        # In `directional` mode we only apply the long factor when the risk direction is explicitly up.
                        # In `hygiene` mode we apply the long factor unconditionally on riskoff days.
                        # (Default factor=1.0 keeps legacy behavior unchanged.)
                        if str(cfg.riskoff_mode) == "directional":
                            if str(risk_dir_clean) == "up":
                                if float(cfg.riskoff_long_factor) == 0:
                                    _update_trace(zero_reason="riskoff_long_factor_zero", signed_qty_final=0)
                                    return 0, trace
                                risk_dollars *= float(cfg.riskoff_long_factor)
                        else:
                            if float(cfg.riskoff_long_factor) == 0:
                                _update_trace(zero_reason="riskoff_long_factor_zero", signed_qty_final=0)
                                return 0, trace
                            risk_dollars *= float(cfg.riskoff_long_factor)

                    if bool(riskpanic):
                        if float(cfg.riskpanic_long_factor) == 0:
                            _update_trace(zero_reason="riskpanic_long_factor_zero", signed_qty_final=0)
                            return 0, trace
                        risk_dollars *= float(cfg.riskpanic_long_factor)
                    elif risk is not None and str(cfg.riskpanic_long_scale_mode) == "linear":
                        tr_delta = getattr(risk, "tr_median_delta_pct", None)
                        gap_ratio = getattr(risk, "neg_gap_ratio", None)
                        if tr_delta is not None and gap_ratio is not None:
                            gap_min = float(cfg.riskpanic_neg_gap_ratio_min)
                            try:
                                gap_ratio_f = float(gap_ratio)
                            except (TypeError, ValueError):
                                gap_ratio_f = 0.0
                            gap_ratio_f = float(max(0.0, min(1.0, gap_ratio_f)))
                            gap_strength = 0.0
                            if gap_min < 1.0:
                                gap_strength = (gap_ratio_f - gap_min) / (1.0 - gap_min)
                            gap_strength = float(max(0.0, min(1.0, gap_strength)))

                            try:
                                delta_f = float(tr_delta)
                            except (TypeError, ValueError):
                                delta_f = 0.0
                            delta_strength = float(
                                max(0.0, min(1.0, max(0.0, delta_f) / float(cfg.riskpanic_long_scale_tr_delta_max_pct)))
                            )

                            strength = float(gap_strength) * float(delta_strength)
                            if strength > 0:
                                base_factor = float(max(0.0, min(1.0, float(cfg.riskpanic_long_factor))))
                                eff = 1.0 - (float(strength) * (1.0 - float(base_factor)))
                                eff = float(max(0.0, min(1.0, eff)))
                                if eff <= 0:
                                    _update_trace(zero_reason="riskpanic_linear_zero", signed_qty_final=0)
                                    return 0, trace
                                risk_dollars *= float(eff)

                    if bool(riskpop):
                        if float(cfg.riskpop_long_factor) == 0:
                            _update_trace(zero_reason="riskpop_long_factor_zero", signed_qty_final=0)
                            return 0, trace
                        risk_dollars *= float(cfg.riskpop_long_factor)

                    if bool(shock) and shock_dir_clean in ("up", "down"):
                        shock_long_mult = (
                            float(cfg.shock_long_risk_mult_factor)
                            if shock_dir_clean == "up"
                            else float(cfg.shock_long_risk_mult_factor_down)
                        )

                        # Optional "rebound-aware" long boost gate: only apply the >1.0 boost
                        # when drawdown is close to the OFF threshold (still shock-on but recovering).
                        if shock_dir_clean == "up" and float(shock_long_mult) > 1.0:
                            boost_ok = True
                            gate_reason = "ok"
                            max_dist = float(cfg.shock_long_boost_max_dist_off_pp)
                            if max_dist > 0:
                                dist_off = shock_drawdown_dist_off_clean
                                if dist_off is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_off_missing"
                                else:
                                    try:
                                        dist_off_f = float(dist_off)
                                    except (TypeError, ValueError):
                                        dist_off_f = None
                                    if dist_off_f is None:
                                        boost_ok = False
                                        gate_reason = "dd_dist_off_invalid"
                                    elif dist_off_f > 0:
                                        boost_ok = False
                                        gate_reason = "dd_dist_off_gt_0"
                                    else:
                                        dist_off_abs = abs(float(dist_off_f))
                                        if dist_off_abs > float(max_dist):
                                            boost_ok = False
                                            gate_reason = f"dd_dist_off_abs_gt_{float(max_dist):g}pp"

                            if boost_ok and bool(cfg.shock_long_boost_require_regime_up) and signal_regime_dir_clean != "up":
                                boost_ok = False
                                gate_reason = "regime_not_up"
                            if boost_ok and bool(cfg.shock_long_boost_require_entry_up) and signal_entry_dir_clean != "up":
                                boost_ok = False
                                gate_reason = "entry_not_up"

                            if boost_ok:
                                _update_trace(
                                    shock_long_factor=float(shock_long_mult),
                                    shock_long_boost_applied=True,
                                    shock_long_boost_gate_reason=gate_reason,
                                )
                                risk_dollars *= float(shock_long_mult)
                            else:
                                _update_trace(
                                    shock_long_factor=1.0,
                                    shock_long_boost_applied=False,
                                    shock_long_boost_gate_reason=gate_reason,
                                )
                        else:
                            _update_trace(shock_long_factor=float(shock_long_mult))
                            if shock_long_mult == 0:
                                _update_trace(zero_reason="shock_long_factor_zero", signed_qty_final=0)
                                return 0, trace
                            risk_dollars *= float(shock_long_mult)

                    risk_dollars *= float(graph_overlay.long_risk_mult)
                    risk_dollars *= float(graph_overlay.risk_mult)
                    if risk_dollars <= 0:
                        _update_trace(zero_reason="graph_overlay_nonpositive", signed_qty_final=0)
                        return 0, trace

                else:
                    short_mult = float(cfg.spot_short_risk_mult)
                    _update_trace(short_mult_base=float(short_mult), short_mult_final=float(short_mult))

                    # Drawdown depth gate for shorts:
                    # Historically we only applied this when the main shock detector was ON and the direction was DOWN.
                    # For HF chop defense, the same drawdown telemetry can be present even when shock stays OFF
                    # (ex: atr_ratio is muted), so we apply the same gate whenever it's configured.
                    #
                    # Semantics: gate around the ON threshold (dd->on distance in percentage points).
                    # - dist_on == 0: at ON threshold (ex: -20% dd if on=-20)
                    # - dist_on < 0: above ON threshold (milder drawdown / near ATH)
                    # - dist_on > 0: below ON threshold (deeper crash)
                    # We allow short entries only when dist_on is within [-max_dist, +max_dist].
                    max_dist = float(cfg.shock_short_entry_max_dist_on_pp)
                    if max_dist > 0:
                        gate_ok = True
                        gate_reason = "ok"
                        dist_on = shock_drawdown_dist_on_clean
                        if dist_on is None:
                            gate_reason = "dd_dist_on_missing"
                        else:
                            try:
                                dist_on_f = float(dist_on)
                            except (TypeError, ValueError):
                                dist_on_f = None
                            if dist_on_f is None:
                                gate_ok = False
                                gate_reason = "dd_dist_on_invalid"
                            elif dist_on_f < -float(max_dist):
                                gate_ok = False
                                gate_reason = f"dd_dist_on_lt_-{float(max_dist):g}pp"
                            elif dist_on_f > float(max_dist):
                                gate_ok = False
                                gate_reason = f"dd_dist_on_gt_{float(max_dist):g}pp"
                        _update_trace(
                            shock_short_entry_blocked=(not bool(gate_ok)),
                            shock_short_entry_gate_reason=str(gate_reason),
                        )
                        if not bool(gate_ok):
                            _update_trace(zero_reason="shock_short_entry_depth_gate", signed_qty_final=0)
                            return 0, trace

                    if bool(riskoff):
                        # In `directional` mode we only apply the short factor when the risk direction is explicitly down.
                        # In `hygiene` mode we apply the short factor unconditionally on riskoff days.
                        if str(cfg.riskoff_mode) == "directional":
                            if str(risk_dir_clean) == "down":
                                short_mult *= float(cfg.riskoff_short_factor)
                        else:
                            short_mult *= float(cfg.riskoff_short_factor)
                    if bool(riskpanic):
                        short_mult *= float(cfg.riskpanic_short_factor)
                    if bool(riskpop):
                        short_mult *= float(cfg.riskpop_short_factor)
                    if bool(shock) and str(shock_dir_clean) == "down":
                        boost_ok = True
                        gate_reason = "ok"
                        min_streak = max(1, int(cfg.shock_short_boost_min_down_streak_bars))
                        streak = int(shock_dir_down_streak_clean or 0)
                        if streak < min_streak:
                            boost_ok = False
                            gate_reason = f"down_streak_lt_{min_streak}"

                        max_dist = float(cfg.shock_short_boost_max_dist_on_pp)
                        if boost_ok and max_dist > 0:
                            min_dist = float(cfg.shock_short_boost_min_dist_on_pp)
                            dist_on = shock_drawdown_dist_on_clean
                            if dist_on is None:
                                boost_ok = False
                                gate_reason = "dd_dist_on_missing"
                            else:
                                try:
                                    dist_on_f = float(dist_on)
                                except (TypeError, ValueError):
                                    dist_on_f = None
                                if dist_on_f is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_invalid"
                                elif dist_on_f < float(min_dist):
                                    boost_ok = False
                                    gate_reason = (
                                        "dd_dist_on_lt_0" if float(min_dist) <= 0 else f"dd_dist_on_lt_{float(min_dist):g}pp"
                                    )
                                elif dist_on_f > float(max_dist):
                                    boost_ok = False
                                    gate_reason = f"dd_dist_on_gt_{float(max_dist):g}pp"

                        if boost_ok and bool(cfg.shock_short_boost_require_regime_down) and signal_regime_dir_clean != "down":
                            boost_ok = False
                            gate_reason = "regime_not_down"
                        if boost_ok and bool(cfg.shock_short_boost_require_entry_down) and signal_entry_dir_clean != "down":
                            boost_ok = False
                            gate_reason = "entry_not_down"

                        if boost_ok:
                            shock_short_mult = float(cfg.shock_short_risk_mult_factor)
                            _update_trace(
                                shock_short_factor=float(shock_short_mult),
                                shock_short_boost_applied=True,
                                shock_short_boost_gate_reason=gate_reason,
                            )
                            short_mult *= float(shock_short_mult)
                        else:
                            _update_trace(
                                shock_short_boost_applied=False,
                                shock_short_boost_gate_reason=gate_reason,
                            )
                    elif bool(shock):
                        _update_trace(
                            shock_short_boost_applied=False,
                            shock_short_boost_gate_reason="shock_not_down",
                        )
                    else:
                        dd_boost_applied = False
                        dd_boost_reason = "shock_off"

                        # Drawdown-based short boost (shock-off):
                        # When configured, allow the same short boost to fire off drawdown
                        # telemetry even if the main shock detector (ATR/TR ratio) stays off.
                        #
                        # Guarded by `shock_short_boost_max_dist_on_pp > 0` so existing configs
                        # that only rely on factor != 1.0 do not change behavior.
                        if str(shock_dir_clean) == "down":
                            boost_ok = True
                            gate_reason = "ok"
                            max_dist = float(cfg.shock_short_boost_max_dist_on_pp)
                            if max_dist <= 0:
                                boost_ok = False
                                gate_reason = "dd_boost_disabled"
                            else:
                                min_dist = float(cfg.shock_short_boost_min_dist_on_pp)
                                dist_on = shock_drawdown_dist_on_clean
                                if dist_on is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_missing"
                                else:
                                    try:
                                        dist_on_f = float(dist_on)
                                    except (TypeError, ValueError):
                                        dist_on_f = None
                                    if dist_on_f is None:
                                        boost_ok = False
                                        gate_reason = "dd_dist_on_invalid"
                                    elif dist_on_f < float(min_dist):
                                        boost_ok = False
                                        gate_reason = (
                                            "dd_dist_on_lt_0" if float(min_dist) <= 0 else f"dd_dist_on_lt_{float(min_dist):g}pp"
                                        )
                                    elif dist_on_f > float(max_dist):
                                        boost_ok = False
                                        gate_reason = f"dd_dist_on_gt_{float(max_dist):g}pp"

                            # Optional persistence gates (reuse prearm vel/accel knobs).
                            if boost_ok and float(cfg.shock_prearm_min_dist_on_vel_pp) > 0:
                                if shock_drawdown_dist_on_vel_clean is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_vel_missing"
                                elif float(shock_drawdown_dist_on_vel_clean) < float(cfg.shock_prearm_min_dist_on_vel_pp):
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_vel_low"
                            if boost_ok and float(cfg.shock_prearm_min_dist_on_accel_pp) > 0:
                                if shock_drawdown_dist_on_accel_clean is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_accel_missing"
                                elif float(shock_drawdown_dist_on_accel_clean) < float(cfg.shock_prearm_min_dist_on_accel_pp):
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_accel_low"

                            if boost_ok and bool(cfg.shock_short_boost_require_regime_down) and signal_regime_dir_clean != "down":
                                boost_ok = False
                                gate_reason = "regime_not_down"
                            if boost_ok and bool(cfg.shock_short_boost_require_entry_down) and signal_entry_dir_clean != "down":
                                boost_ok = False
                                gate_reason = "entry_not_down"

                            if boost_ok:
                                shock_short_mult = float(cfg.shock_short_risk_mult_factor)
                                short_mult *= float(shock_short_mult)
                                dd_boost_applied = True
                                dd_boost_reason = str(gate_reason)
                                _update_trace(
                                    shock_short_factor=float(shock_short_mult),
                                    shock_short_boost_applied=True,
                                    shock_short_boost_gate_reason=f"dd:{gate_reason}",
                                )
                            else:
                                dd_boost_reason = str(gate_reason)

                        if not bool(dd_boost_applied):
                            _update_trace(
                                shock_short_boost_applied=False,
                                shock_short_boost_gate_reason=str(dd_boost_reason),
                            )

                        prearm_applied = False
                        prearm_reason = "off"
                        prearm_factor = 1.0
                        dist_on = shock_drawdown_dist_on_clean
                        dist_on_vel = shock_drawdown_dist_on_vel_clean
                        dist_on_accel = shock_drawdown_dist_on_accel_clean
                        prearm_streak = shock_prearm_down_streak_clean
                        near_band = float(cfg.shock_prearm_dist_on_max_pp)
                        latch_enabled = int(cfg.shock_prearm_min_streak_bars) > 0
                        if near_band <= 0:
                            prearm_reason = "disabled"
                        elif dist_on is None:
                            prearm_reason = "dist_on_missing"
                        elif bool(latch_enabled) and float(dist_on) < -float(near_band):
                            prearm_reason = "outside_band"
                        elif (not bool(latch_enabled)) and not (-float(near_band) <= float(dist_on) < 0.0):
                            prearm_reason = "outside_band"
                        elif (not bool(latch_enabled)) and dist_on_vel is None:
                            prearm_reason = "dist_vel_missing"
                        elif (not bool(latch_enabled)) and float(dist_on_vel) < float(cfg.shock_prearm_min_dist_on_vel_pp):
                            prearm_reason = "dist_vel_low"
                        elif (not bool(latch_enabled)) and float(cfg.shock_prearm_min_dist_on_accel_pp) > 0 and dist_on_accel is None:
                            prearm_reason = "dist_accel_missing"
                        elif (not bool(latch_enabled)) and float(cfg.shock_prearm_min_dist_on_accel_pp) > 0 and float(dist_on_accel) < float(
                            cfg.shock_prearm_min_dist_on_accel_pp
                        ):
                            prearm_reason = "dist_accel_low"
                        elif int(cfg.shock_prearm_min_streak_bars) > 0 and prearm_streak is None:
                            prearm_reason = "streak_missing"
                        elif int(cfg.shock_prearm_min_streak_bars) > 0 and int(prearm_streak) < int(
                            cfg.shock_prearm_min_streak_bars
                        ):
                            prearm_reason = f"streak_lt_{int(cfg.shock_prearm_min_streak_bars)}"
                        elif bool(cfg.shock_prearm_require_regime_down) and signal_regime_dir_clean != "down":
                            prearm_reason = "regime_not_down"
                        elif bool(cfg.shock_prearm_require_entry_down) and signal_entry_dir_clean != "down":
                            prearm_reason = "entry_not_down"
                        else:
                            prearm_factor = float(cfg.shock_prearm_short_risk_mult_factor)
                            if prearm_factor > 0:
                                short_mult *= float(prearm_factor)
                                prearm_applied = True
                                prearm_reason = "ok"
                            else:
                                prearm_reason = "factor_nonpositive"
                        _update_trace(
                            shock_prearm_applied=bool(prearm_applied),
                            shock_prearm_factor=float(prearm_factor) if prearm_applied else None,
                            shock_prearm_reason=str(prearm_reason),
                        )
                    short_mult *= float(graph_overlay.short_risk_mult)
                    _update_trace(short_mult_final=float(short_mult))
                    if short_mult <= 0:
                        _update_trace(zero_reason="short_mult_nonpositive", signed_qty_final=0)
                        return 0, trace
                    risk_dollars *= float(short_mult)
                    risk_dollars *= float(graph_overlay.risk_mult)
                    if risk_dollars <= 0:
                        _update_trace(zero_reason="graph_overlay_nonpositive", signed_qty_final=0)
                        return 0, trace

                if (
                    cfg.shock_risk_scale_target_atr_pct is not None
                    and shock_atr_pct_clean is not None
                    and float(shock_atr_pct_clean) > 0
                ):
                    target = float(cfg.shock_risk_scale_target_atr_pct)
                    min_mult = float(cfg.shock_risk_scale_min_mult)
                    scale = min(1.0, float(target) / float(shock_atr_pct_clean))
                    scale = float(max(min_mult, min(1.0, scale)))
                    apply_to = str(cfg.shock_risk_scale_apply_to)
                    _update_trace(
                        shock_scale_target_atr_pct=float(target),
                        shock_scale_mult=float(scale),
                        shock_scale_apply_to=str(apply_to),
                    )
                    if apply_to in ("risk", "both"):
                        risk_dollars *= float(scale)
                    if apply_to in ("cap", "both"):
                        cap_pct *= float(scale)

                _update_trace(risk_dollars_final=float(risk_dollars), cap_pct_final=float(cap_pct))

                liq_boost_applied = False
                liq_boost_reason = "off"
                liq_boost_mult = 1.0
                liq_boost_score = None
                if bool(cfg.liq_boost_enable):
                    overlay_trace = graph_overlay.as_payload() if hasattr(graph_overlay, "as_payload") else {}
                    raw_score = overlay_trace.get("trace", {}).get("score") if isinstance(overlay_trace, dict) else None
                    if raw_score is None and isinstance(trace.graph_overlay_trace, dict):
                        raw_score = trace.graph_overlay_trace.get("trace", {}).get("score")
                    try:
                        liq_boost_score = float(raw_score) if raw_score is not None else None
                    except (TypeError, ValueError):
                        liq_boost_score = None
                    if liq_boost_score is None:
                        # Fallback score keeps boost logic usable with non-trend-bias overlay policies.
                        score = 0.0
                        if risk_tr_ratio is not None:
                            score += max(0.0, float(risk_tr_ratio) - 1.0)
                        if risk_slope_med is not None:
                            slope_signed = float(risk_slope_med) if raw_action == "BUY" else -float(risk_slope_med)
                            score += max(0.0, slope_signed / 0.35)
                        if risk_slope_vel is not None:
                            vel_signed = float(risk_slope_vel) if raw_action == "BUY" else -float(risk_slope_vel)
                            score += max(0.0, vel_signed / 0.20)
                        liq_boost_score = float(score)
                    action_dir = "up" if raw_action == "BUY" else "down"
                    align_ok = True
                    if bool(cfg.liq_boost_require_alignment):
                        align_ok = (
                            signal_entry_dir_clean == action_dir and signal_regime_dir_clean == action_dir
                        )
                    if bool(cfg.liq_boost_require_shock) and not bool(shock):
                        align_ok = False
                        liq_boost_reason = "shock_required"
                    elif not align_ok:
                        liq_boost_reason = "align_fail"
                    elif liq_boost_score is None:
                        liq_boost_reason = "score_missing"
                    elif liq_boost_score < float(cfg.liq_boost_score_min):
                        liq_boost_reason = "score_low"
                    else:
                        score_over = float(liq_boost_score) - float(cfg.liq_boost_score_min)
                        intensity = min(1.0, max(0.0, score_over / float(cfg.liq_boost_score_span)))
                        liq_boost_mult = 1.0 + (float(cfg.liq_boost_max_risk_mult) - 1.0) * float(intensity)
                        if liq_boost_mult > 1.0:
                            risk_dollars *= float(liq_boost_mult)
                            liq_boost_applied = True
                            liq_boost_reason = "ok"
                        else:
                            liq_boost_reason = "mult_unity"
                    _update_trace(
                        liq_boost_applied=bool(liq_boost_applied),
                        liq_boost_score=float(liq_boost_score) if liq_boost_score is not None else None,
                        liq_boost_mult=float(liq_boost_mult),
                        liq_boost_reason=str(liq_boost_reason),
                        liq_boost_cap_floor_frac=float(cfg.liq_boost_cap_floor_frac),
                    )
                    if liq_boost_applied:
                        _update_trace(risk_dollars_final=float(risk_dollars), cap_pct_final=float(cap_pct))

                if shock_ramp is not None:
                    ramp_applied = False
                    ramp_reason = "off"
                    ramp_phase = None
                    ramp_dir = "up" if raw_action == "BUY" else "down"
                    ramp_intensity = None
                    ramp_mult = 1.0
                    ramp_floor_frac = 0.0
                    ramp_node = None
                    if isinstance(shock_ramp, Mapping):
                        ramp_node = shock_ramp.get(str(ramp_dir))
                    else:
                        ramp_node = getattr(shock_ramp, str(ramp_dir), None)
                    if isinstance(ramp_node, Mapping):
                        raw_mult = ramp_node.get("risk_mult")
                        raw_floor = ramp_node.get("cap_floor_frac")
                        try:
                            ramp_mult = float(raw_mult) if raw_mult is not None else 1.0
                        except (TypeError, ValueError):
                            ramp_mult = 1.0
                        try:
                            ramp_floor_frac = float(raw_floor) if raw_floor is not None else 0.0
                        except (TypeError, ValueError):
                            ramp_floor_frac = 0.0
                        ramp_floor_frac = float(max(0.0, min(1.0, ramp_floor_frac)))
                        ramp_phase_raw = ramp_node.get("phase")
                        ramp_phase = str(ramp_phase_raw) if ramp_phase_raw is not None else None
                        try:
                            ramp_intensity = float(ramp_node.get("intensity")) if ramp_node.get("intensity") is not None else None
                        except (TypeError, ValueError):
                            ramp_intensity = None
                        ramp_reason_raw = ramp_node.get("reason")
                        ramp_reason = str(ramp_reason_raw) if ramp_reason_raw is not None else "ok"
                        if ramp_mult > 1.0 and risk_dollars > 0:
                            risk_dollars *= float(ramp_mult)
                            ramp_applied = True
                        else:
                            ramp_reason = "mult_unity"
                    elif ramp_node is not None:
                        ramp_reason = "invalid_node"
                    else:
                        ramp_reason = "missing_dir"
                    _update_trace(
                        shock_ramp_applied=bool(ramp_applied),
                        shock_ramp_dir=str(ramp_dir),
                        shock_ramp_phase=str(ramp_phase) if ramp_phase is not None else None,
                        shock_ramp_intensity=float(ramp_intensity) if ramp_intensity is not None else None,
                        shock_ramp_risk_mult=float(ramp_mult),
                        shock_ramp_cap_floor_frac=float(ramp_floor_frac),
                        shock_ramp_reason=str(ramp_reason),
                    )
                    if ramp_applied:
                        _update_trace(risk_dollars_final=float(risk_dollars), cap_pct_final=float(cap_pct))

                if per_share_risk > 1e-9 and risk_dollars > 0:
                    desired_qty = int(risk_dollars / per_share_risk)

        fallback_to_lot = False
        if desired_qty <= 0:
            desired_qty = int(lot_i) * int(cfg.quantity_mult)
            fallback_to_lot = True

        desired_qty_pre_caps = int(desired_qty)
        cap_qty = None
        if cap_pct > 0 and float(equity_ref_f) > 0:
            cap_qty = int((float(equity_ref_f) * float(cap_pct)) / float(entry_price_f))
            desired_qty = min(int(desired_qty), max(0, int(cap_qty)))
            if (
                bool(getattr(trace, "liq_boost_applied", False))
                and float(getattr(cfg, "liq_boost_cap_floor_frac", 0.0)) > 0
                and cap_qty is not None
                and cap_qty > 0
            ):
                floor_qty = int(float(cap_qty) * float(cfg.liq_boost_cap_floor_frac))
                if floor_qty > 0:
                    desired_qty = max(int(desired_qty), int(floor_qty))
                    _update_trace(liq_boost_cap_floor_qty=int(floor_qty))
            if (
                float(getattr(trace, "shock_ramp_cap_floor_frac", 0.0) or 0.0) > 0
                and cap_qty is not None
                and cap_qty > 0
            ):
                floor_qty = int(float(cap_qty) * float(getattr(trace, "shock_ramp_cap_floor_frac", 0.0) or 0.0))
                if floor_qty > 0:
                    desired_qty = max(int(desired_qty), int(floor_qty))
                    _update_trace(shock_ramp_cap_floor_qty=int(floor_qty))

        afford_qty = None
        if raw_action == "BUY" and cash_ref_f is not None and float(cash_ref_f) > 0:
            afford_qty = int(float(cash_ref_f) / float(entry_price_f))
            desired_qty = min(int(desired_qty), max(0, int(afford_qty)))

        if int(cfg.spot_max_qty) > 0:
            desired_qty = min(int(desired_qty), int(cfg.spot_max_qty))

        desired_qty_post_caps = int(desired_qty)
        lot_rounded_qty = (int(desired_qty) // int(lot_i)) * int(lot_i)
        min_effective = max(int(cfg.spot_min_qty), int(lot_i))

        if lot_rounded_qty < min_effective:
            _update_trace(
                desired_qty_pre_caps=int(desired_qty_pre_caps),
                desired_qty_post_caps=int(desired_qty_post_caps),
                cap_qty=int(cap_qty) if cap_qty is not None else None,
                afford_qty=int(afford_qty) if afford_qty is not None else None,
                fallback_to_lot=bool(fallback_to_lot),
                lot_rounded_qty=int(lot_rounded_qty),
                min_effective_qty=int(min_effective),
                min_qty_blocked=True,
                signed_qty_final=0,
                zero_reason="min_qty_gate",
                cap_pct_final=float(cap_pct),
            )
            return 0, trace

        signed_qty = int(lot_rounded_qty) if raw_action == "BUY" else -int(lot_rounded_qty)
        _update_trace(
            desired_qty_pre_caps=int(desired_qty_pre_caps),
            desired_qty_post_caps=int(desired_qty_post_caps),
            cap_qty=int(cap_qty) if cap_qty is not None else None,
            afford_qty=int(afford_qty) if afford_qty is not None else None,
            fallback_to_lot=bool(fallback_to_lot),
            lot_rounded_qty=int(lot_rounded_qty),
            min_effective_qty=int(min_effective),
            min_qty_blocked=False,
            signed_qty_final=int(signed_qty),
            zero_reason=None,
            cap_pct_final=float(cap_pct),
        )
        return int(signed_qty), trace

    @classmethod
    def calc_signed_qty(
        cls,
        *,
        strategy: Mapping[str, object] | object,
        filters: Mapping[str, object] | object | None,
        action: str,
        lot: int,
        entry_price: float,
        stop_price: float | None,
        stop_loss_pct: float | None,
        shock: bool | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_dir_down_streak_bars: int | None = None,
        shock_drawdown_dist_on_pct: float | None = None,
        shock_drawdown_dist_on_vel_pp: float | None = None,
        shock_drawdown_dist_on_accel_pp: float | None = None,
        shock_prearm_down_streak_bars: int | None = None,
        shock_ramp: Mapping[str, object] | object | None = None,
        riskoff: bool = False,
        risk_dir: str | None = None,
        riskpanic: bool = False,
        riskpop: bool = False,
        risk: object | None = None,
        signal_entry_dir: str | None = None,
        signal_regime_dir: str | None = None,
        equity_ref: float = 0.0,
        cash_ref: float | None = None,
    ) -> int:
        qty, _trace = cls.calc_signed_qty_with_trace(
            strategy=strategy,
            filters=filters,
            action=action,
            lot=lot,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_loss_pct=stop_loss_pct,
            shock=shock,
            shock_dir=shock_dir,
            shock_atr_pct=shock_atr_pct,
            shock_dir_down_streak_bars=shock_dir_down_streak_bars,
            shock_drawdown_dist_on_pct=shock_drawdown_dist_on_pct,
            shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
            shock_drawdown_dist_on_accel_pp=shock_drawdown_dist_on_accel_pp,
            shock_prearm_down_streak_bars=shock_prearm_down_streak_bars,
            shock_ramp=shock_ramp,
            riskoff=riskoff,
            risk_dir=risk_dir,
            riskpanic=riskpanic,
            riskpop=riskpop,
            risk=risk,
            signal_entry_dir=signal_entry_dir,
            signal_regime_dir=signal_regime_dir,
            equity_ref=equity_ref,
            cash_ref=cash_ref,
        )
        return int(qty)
