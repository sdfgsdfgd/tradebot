"""Normalized spot policy contracts shared by backtests and live execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

from .fill_modes import SPOT_FILL_MODE_CLOSE, normalize_spot_fill_mode
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
        pack_name, _sget, _fget = _policy_source_getters(
            strategy=strategy,
            filters=filters,
        )

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
            spot_policy_pack=pack_name,
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


def _policy_source_getters(
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
) -> tuple[str | None, Callable[..., object], Callable[..., object]]:
    """One precedence law: explicit input, then pack default, then caller default."""
    pack = resolve_pack(strategy=strategy, filters=filters)
    missing = object()

    def _value(source, defaults, key: str, default: object = None) -> object:
        raw = SpotPolicyConfigView._get(source, key, missing)
        if raw is not missing:
            return raw
        if key in defaults:
            return defaults[key]
        return default

    strategy_defaults = pack.strategy_defaults if pack is not None else {}
    filter_defaults = pack.filter_defaults if pack is not None else {}
    return (
        str(pack.name) if pack is not None else None,
        lambda key, default=None: _value(strategy, strategy_defaults, key, default),
        lambda key, default=None: _value(filters, filter_defaults, key, default),
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
        pack_name, _sget, _ = _policy_source_getters(
            strategy=strategy,
            filters=filters,
        )

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
            spot_policy_pack=pack_name,
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
    regime2_dir: str | None
    regime2_ready: bool

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
