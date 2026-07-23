"""Live signal contract, duration, health, and bar acquisition."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, time, timedelta

from ib_insync import Contract, Stock

from ...contract_identity import future_exchange_for_symbol, is_future_symbol
from ...chart_data.series import BarSeries, BarSeriesMeta
from ...client import IBKRClient, _session_flags
from ...engines.market import full24_post_close_time_et, is_trading_day, session_label_et
from ...signals import parse_bar_size
from ...time_utils import to_et as _to_et_shared
from ..bot_models import _BotInstance
from ..common import _market_session_bucket


class BotSignalDataMixin:
    def _strategy_instrument(self, strategy: dict) -> str:
        value = strategy.get("instrument", "options")
        cleaned = str(value or "options").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    def _spot_sec_type(self, instance: _BotInstance, symbol: str) -> str:
        raw = (instance.strategy or {}).get("spot_sec_type")
        cleaned = str(raw or "").strip().upper()
        if cleaned in ("STK", "FUT"):
            return cleaned
        return "FUT" if is_future_symbol(symbol) else "STK"

    def _spot_exchange(self, instance: _BotInstance, symbol: str, *, sec_type: str) -> str:
        raw = (instance.strategy or {}).get("spot_exchange")
        cleaned = str(raw or "").strip().upper()
        if cleaned:
            return cleaned
        if sec_type == "FUT":
            return future_exchange_for_symbol(symbol) or "CME"
        return "SMART"

    async def _spot_contract(self, instance: _BotInstance, symbol: str) -> Contract | None:
        sec_type = self._spot_sec_type(instance, symbol)
        exchange = self._spot_exchange(instance, symbol, sec_type=sec_type)
        if sec_type == "FUT":
            contract = await self._client.front_future(symbol, exchange=exchange, cache_ttl_sec=3600.0)
            if contract is None:
                return None
            return contract

        contract = Stock(symbol=str(symbol).strip().upper(), exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(contract)
        return qualified[0] if qualified else contract

    async def _signal_contract(self, instance: _BotInstance, symbol: str) -> Contract | None:
        if self._strategy_instrument(instance.strategy) == "spot":
            return await self._spot_contract(instance, symbol)
        contract = Stock(symbol=str(symbol).strip().upper(), exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(contract)
        return qualified[0] if qualified else contract

    def _signal_bar_size(self, instance: _BotInstance) -> str:
        raw = instance.strategy.get("signal_bar_size")
        if raw:
            return str(raw)
        if self._payload:
            return str(self._payload.get("bar_size", "1 hour"))
        return "1 hour"

    def _signal_use_rth(self, instance: _BotInstance) -> bool:
        raw = instance.strategy.get("signal_use_rth")
        if raw is not None:
            return bool(raw)
        if self._payload:
            return bool(self._payload.get("use_rth", False))
        return False

    def _signal_snapshot_kwargs(
        self,
        instance: _BotInstance,
        *,
        strategy: dict | None = None,
        ema_preset_raw: str | None = None,
        entry_signal_raw: str | None = None,
        include_orb: bool = False,
        include_spot_exit: bool = False,
    ) -> dict[str, object]:
        self._heal_instance_effective_filters(instance)
        strat = strategy if isinstance(strategy, dict) else (instance.strategy or {})
        kwargs: dict[str, object] = {
            "base_strategy_raw": strat if isinstance(strat, dict) else None,
            "ema_preset_raw": ema_preset_raw,
            "bar_size": self._signal_bar_size(instance),
            "use_rth": self._signal_use_rth(instance),
            "entry_mode_raw": strat.get("ema_entry_mode"),
            "entry_confirm_bars": strat.get("entry_confirm_bars", 0),
            "spot_dual_branch_enabled_raw": strat.get("spot_dual_branch_enabled"),
            "spot_dual_branch_priority_raw": strat.get("spot_dual_branch_priority"),
            "spot_branch_a_ema_preset_raw": strat.get("spot_branch_a_ema_preset"),
            "spot_branch_a_entry_confirm_bars_raw": strat.get("spot_branch_a_entry_confirm_bars"),
            "spot_branch_a_min_signed_slope_pct_raw": strat.get("spot_branch_a_min_signed_slope_pct"),
            "spot_branch_a_max_signed_slope_pct_raw": strat.get("spot_branch_a_max_signed_slope_pct"),
            "spot_branch_a_size_mult_raw": strat.get("spot_branch_a_size_mult"),
            "spot_branch_b_ema_preset_raw": strat.get("spot_branch_b_ema_preset"),
            "spot_branch_b_entry_confirm_bars_raw": strat.get("spot_branch_b_entry_confirm_bars"),
            "spot_branch_b_min_signed_slope_pct_raw": strat.get("spot_branch_b_min_signed_slope_pct"),
            "spot_branch_b_max_signed_slope_pct_raw": strat.get("spot_branch_b_max_signed_slope_pct"),
            "spot_branch_b_size_mult_raw": strat.get("spot_branch_b_size_mult"),
            "regime_ema_preset_raw": strat.get("regime_ema_preset"),
            "regime_bar_size_raw": strat.get("regime_bar_size"),
            "regime_mode_raw": strat.get("regime_mode"),
            "supertrend_atr_period_raw": strat.get("supertrend_atr_period"),
            "supertrend_multiplier_raw": strat.get("supertrend_multiplier"),
            "supertrend_source_raw": strat.get("supertrend_source"),
            "regime2_ema_preset_raw": strat.get("regime2_ema_preset"),
            "regime2_bar_size_raw": strat.get("regime2_bar_size"),
            "regime2_mode_raw": strat.get("regime2_mode"),
            "regime2_supertrend_atr_period_raw": strat.get("regime2_supertrend_atr_period"),
            "regime2_supertrend_multiplier_raw": strat.get("regime2_supertrend_multiplier"),
            "regime2_supertrend_source_raw": strat.get("regime2_supertrend_source"),
            "filters": instance.filters if isinstance(instance.filters, dict) else None,
        }
        if entry_signal_raw is not None:
            kwargs["entry_signal_raw"] = entry_signal_raw
        if include_orb:
            kwargs["orb_window_mins_raw"] = strat.get("orb_window_mins")
            kwargs["orb_open_time_et_raw"] = strat.get("orb_open_time_et")
        if include_spot_exit:
            kwargs["spot_exit_mode_raw"] = strat.get("spot_exit_mode")
            kwargs["spot_atr_period_raw"] = strat.get("spot_atr_period")
        return kwargs

    def _signal_duration_str(
        self,
        bar_size: str,
        *,
        filters: dict | None = None,
        strategy: dict | None = None,
        use_rth: bool | None = None,
    ) -> str:
        label = str(bar_size or "").strip().lower()

        def _rank(duration: str) -> int:
            order = ("1 D", "2 D", "1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
            cleaned = str(duration or "").strip()
            try:
                return order.index(cleaned)
            except ValueError:
                return 0

        def _max_duration(a: str, b: str) -> str:
            return a if _rank(a) >= _rank(b) else b

        base = "2 W"
        if label.startswith(("1 min", "2 mins")):
            base = "2 D"
        elif label.startswith(("5 mins", "10 mins", "15 mins")):
            base = "1 W"
        elif label.startswith("30 min"):
            # Regime2 often runs on 30m bars (RTH => ~13 bars/day). A 1W request can land
            # just under the 60-bar supertrend warmup requirement depending on the time
            # of week / partial days / holidays. Bump the default so PREWARM converges
            # immediately instead of "waiting" for live closes.
            base = "2 W"
        elif "hour" in label:
            base = "2 W"
        elif "day" in label:
            base = "1 Y"

        # Start at the minimum viable duration for readiness instead of requesting a larger
        # window and then timing out / stitching-incomplete into the floor anyway.
        floor = self._signal_min_duration_str(
            bar_size,
            filters=filters,
            strategy=strategy,
            use_rth=use_rth,
        )
        return _max_duration(base, str(floor)) if floor else base

    @staticmethod
    def _duration_for_days(days_needed: int) -> str:
        try:
            days = int(days_needed)
        except (TypeError, ValueError):
            days = 0
        days = max(0, int(days))
        if days <= 25:
            return "1 M"
        if days <= 50:
            return "2 M"
        if days <= 95:
            return "3 M"
        if days <= 180:
            return "6 M"
        return "1 Y"

    def _signal_min_duration_str(
        self,
        bar_size: str,
        *,
        filters: dict | None = None,
        strategy: dict | None = None,
        use_rth: bool | None = None,
    ) -> str | None:
        _ = bar_size
        floors: list[str] = []

        if isinstance(filters, dict) and filters:
            from ...engines.shock import normalize_shock_detector, normalize_shock_gate_mode

            shock_mode = normalize_shock_gate_mode(filters)
            if shock_mode != "off":
                detector = normalize_shock_detector(filters)
                if detector in ("daily_atr_pct", "daily_drawdown"):
                    if detector == "daily_atr_pct":
                        raw = filters.get("shock_daily_atr_period", 14)
                        try:
                            days_needed = int(raw or 14)
                        except (TypeError, ValueError):
                            days_needed = 14
                        days_needed = max(1, int(days_needed))
                    else:
                        raw = filters.get("shock_drawdown_lookback_days", 20)
                        try:
                            days_needed = int(raw or 20)
                        except (TypeError, ValueError):
                            days_needed = 20
                        days_needed = max(2, int(days_needed))
                    floors.append(self._duration_for_days(days_needed))

        if not floors:
            return None
        return max(floors, key=self._signal_duration_rank)

    @staticmethod
    def _signal_expand_duration(duration: str) -> str:
        order = ("1 D", "2 D", "1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
        cleaned = str(duration or "").strip()
        try:
            idx = order.index(cleaned)
        except ValueError:
            return cleaned
        return order[min(idx + 1, len(order) - 1)]

    @staticmethod
    def _signal_duration_rank(duration: str) -> int:
        order = ("1 D", "2 D", "1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
        cleaned = str(duration or "").strip()
        try:
            return order.index(cleaned)
        except ValueError:
            return -1

    def _signal_timeout_fallback_durations(
        self,
        duration: str,
        *,
        min_duration: str | None = None,
    ) -> tuple[str, ...]:
        requested_rank = self._signal_duration_rank(str(duration))
        min_rank = self._signal_duration_rank(str(min_duration)) if min_duration else -1
        day_rank = self._signal_duration_rank("1 D")
        if requested_rank <= day_rank:
            return ()
        fallbacks = ("3 M", "2 M", "1 M", "1 W", "2 D", "1 D")
        return tuple(
            fallback
            for fallback in fallbacks
            if self._signal_duration_rank(fallback) >= 0
            and self._signal_duration_rank(fallback) < requested_rank
            and (min_rank < 0 or self._signal_duration_rank(fallback) >= min_rank)
        )

    @staticmethod
    def _signal_zero_gap_enabled(filters: dict | None) -> bool:
        _ = filters
        # Strict continuity is now an internal policy (not preset-configurable).
        return True

    def _signal_expected_live_bars(
        self,
        *,
        now_ref: datetime,
        use_rth: bool,
        sec_type: str,
    ) -> bool:
        now_et = _to_et_shared(now_ref, naive_ts_mode="et", default_naive_ts_mode="et")
        weekday = int(now_et.weekday())
        current = now_et.time()
        sec = str(sec_type or "").strip().upper()
        equity_like = sec in ("STK", "OPT")
        if bool(use_rth):
            in_rth = weekday < 5 and time(9, 30) <= current < time(16, 0)
            if not in_rth:
                return False
            if equity_like and not is_trading_day(now_et.date()):
                return False
            return True

        if not equity_like:
            return weekday < 5

        # Keep stale/gap expectations in parity with IBKR session handling:
        # STK/OPT full-session has a maintenance gap between 03:50 and 04:00 ET.
        session_label = session_label_et(current)
        if session_label is None:
            return False
        session_day = now_et.date() + timedelta(days=1) if session_label == "OVERNIGHT_LATE" else now_et.date()
        if not is_trading_day(session_day):
            return False

        outside_rth, include_overnight = _session_flags(now_et)
        if include_overnight:
            return True
        if outside_rth:
            if current >= time(16, 0):
                return current < full24_post_close_time_et(now_et.date())
            return True
        return time(9, 30) <= current < time(16, 0)

    def _signal_stale_threshold_bars(
        self,
        *,
        now_ref: datetime,
        use_rth: bool,
        sec_type: str,
        source: str,
        bar_seconds: float,
    ) -> float:
        # Bars are timestamped at bar start and we trim the in-progress bar.
        # Healthy "latest closed bar" lag therefore lives in [1.0, 2.0) bars.
        # Treat >~2 bars as stale (more than one missed closed bar), with a
        # small epsilon for publication/scheduling jitter.
        threshold = 2.05
        if bar_seconds <= 0:
            return threshold
        return threshold

    def _signal_effective_lag_bars(
        self,
        *,
        last_bar_ts: datetime | None,
        now_ref: datetime,
        bar_seconds: float,
        use_rth: bool,
        sec_type: str,
        stale_threshold_bars: float,
    ) -> float:
        if last_bar_ts is None or bar_seconds <= 0:
            return float("inf")
        age_sec = max(0.0, float((now_ref - last_bar_ts).total_seconds()))
        age_lag_bars = float(age_sec / bar_seconds)
        if bar_seconds < 1800.0:
            return age_lag_bars

        # For HTF bars (RTH or full24), measure lag by expected live slot transitions instead
        # of raw wall-clock time so overnight/weekends/holidays do not false-trigger staleness.
        tolerance_sec = min(60.0, max(1.0, bar_seconds * 0.02))
        probe_cap = max(4, int(math.ceil(stale_threshold_bars)) + 2)
        live_slots = 0
        probe_ts = last_bar_ts
        step = timedelta(seconds=bar_seconds)
        while live_slots < probe_cap:
            probe_ts = probe_ts + step
            if probe_ts > now_ref:
                break
            if (now_ref - probe_ts).total_seconds() < tolerance_sec:
                continue
            if not self._signal_expected_live_bars(
                now_ref=probe_ts,
                use_rth=use_rth,
                sec_type=sec_type,
            ):
                continue
            live_slots += 1
        return float(1 + live_slots)

    def _signal_bar_health(
        self,
        *,
        bars: list | None,
        bar_size: str,
        now_ref: datetime,
        use_rth: bool,
        sec_type: str,
        source: str,
        strict_zero_gap: bool = False,
        heal_attempted: bool = False,
        heal_sources: list[str] | None = None,
    ) -> dict:
        last_bar_ts = getattr(bars[-1], "ts", None) if bars else None
        if not isinstance(last_bar_ts, datetime):
            last_bar_ts = None
        bar_def = parse_bar_size(str(bar_size))
        bar_seconds = (
            float(bar_def.duration.total_seconds())
            if bar_def is not None and bar_def.duration.total_seconds() > 0
            else 300.0
        )
        age_sec = max(0.0, float((now_ref - last_bar_ts).total_seconds())) if last_bar_ts is not None else float("inf")
        expected_live = self._signal_expected_live_bars(
            now_ref=now_ref,
            use_rth=use_rth,
            sec_type=sec_type,
        )
        stale_threshold_bars = self._signal_stale_threshold_bars(
            now_ref=now_ref,
            use_rth=use_rth,
            sec_type=sec_type,
            source=source,
            bar_seconds=bar_seconds,
        )
        lag_bars = self._signal_effective_lag_bars(
            last_bar_ts=last_bar_ts,
            now_ref=now_ref,
            bar_seconds=bar_seconds,
            use_rth=use_rth,
            sec_type=sec_type,
            stale_threshold_bars=stale_threshold_bars,
        )
        stale = bool(last_bar_ts is None or (expected_live and lag_bars > stale_threshold_bars))
        gap_stats = self._signal_gap_stats(
            bars=bars,
            bar_size=bar_size,
            use_rth=use_rth,
            sec_type=sec_type,
            strict_zero_gap=bool(strict_zero_gap),
        )
        return {
            "source": str(source or "TRADES").upper(),
            "last_bar_ts": last_bar_ts,
            "age_sec": None if math.isinf(age_sec) else float(age_sec),
            "lag_bars": None if math.isinf(lag_bars) else float(lag_bars),
            "lag_bars_wall": None if math.isinf(age_sec) else float(age_sec / bar_seconds),
            "stale": bool(stale),
            "expected_live": bool(expected_live),
            "session": _market_session_bucket(now_ref),
            "use_rth": bool(use_rth),
            "sec_type": str(sec_type or "").strip().upper(),
            "stale_threshold_bars": float(stale_threshold_bars),
            "heal_attempted": bool(heal_attempted),
            "heal_sources": list(heal_sources or []),
            "gap_count": int(gap_stats.get("gap_count", 0)),
            "max_gap_bars": float(gap_stats.get("max_gap_bars", 0.0)),
            "recent_gap_count": int(gap_stats.get("recent_gap_count", 0)),
            "recent_max_gap_bars": float(gap_stats.get("recent_max_gap_bars", 0.0)),
            "gap_threshold_bars": float(gap_stats.get("gap_threshold_bars", 2.25)),
            "recent_horizon_bars": int(gap_stats.get("recent_horizon_bars", 0)),
            "gap_detected": bool(gap_stats.get("gap_detected", False)),
            "strict_zero_gap": bool(gap_stats.get("strict_zero_gap", False)),
        }

    def _signal_gap_stats(
        self,
        *,
        bars: list | None,
        bar_size: str,
        use_rth: bool,
        sec_type: str,
        strict_zero_gap: bool = False,
    ) -> dict[str, object]:
        out = {
            "gap_count": 0,
            "max_gap_bars": 0.0,
            "recent_gap_count": 0,
            "recent_max_gap_bars": 0.0,
            "gap_threshold_bars": 1.05 if bool(strict_zero_gap) else 2.25,
            "recent_horizon_bars": 72,
            "gap_detected": False,
            "strict_zero_gap": bool(strict_zero_gap),
            "gap_method": "slot_scan",
            "gap_scan_tolerance_sec": 0.0,
        }
        if not bars or len(bars) < 2:
            return out
        bar_def = parse_bar_size(str(bar_size))
        if bar_def is None or bar_def.duration.total_seconds() <= 0:
            return out

        bar_seconds = float(bar_def.duration.total_seconds())
        gap_threshold = float(out["gap_threshold_bars"])
        recent_horizon_bars = int(out["recent_horizon_bars"])
        recent_horizon_sec = max(bar_seconds, float(recent_horizon_bars) * bar_seconds)
        tolerance_sec = min(60.0, max(1.0, bar_seconds * 0.02))
        out["gap_scan_tolerance_sec"] = float(tolerance_sec)

        last_ts = getattr(bars[-1], "ts", None)
        if not isinstance(last_ts, datetime):
            return out
        recent_cutoff = last_ts - timedelta(seconds=recent_horizon_sec)

        gap_count = 0
        max_gap_bars = 0.0
        recent_gap_count = 0
        recent_max_gap_bars = 0.0

        prev_ts = getattr(bars[0], "ts", None)
        if not isinstance(prev_ts, datetime):
            prev_ts = None
        for bar in bars[1:]:
            curr_ts = getattr(bar, "ts", None)
            if not isinstance(curr_ts, datetime):
                continue
            if not isinstance(prev_ts, datetime):
                prev_ts = curr_ts
                continue
            start_ts = prev_ts
            delta_sec = float((curr_ts - start_ts).total_seconds())
            prev_ts = curr_ts
            if delta_sec <= bar_seconds:
                continue
            slot_span = int(math.floor((delta_sec + tolerance_sec) / bar_seconds))
            if slot_span <= 1:
                continue
            missing_live_slots = 0
            for slot_idx in range(1, slot_span):
                probe_ts = start_ts + timedelta(seconds=(slot_idx * bar_seconds))
                if not self._signal_expected_live_bars(
                    now_ref=probe_ts,
                    use_rth=use_rth,
                    sec_type=sec_type,
                ):
                    continue
                missing_live_slots += 1
            if missing_live_slots <= 0:
                continue
            gap_bars = float(1 + missing_live_slots)
            if gap_bars <= gap_threshold:
                continue
            gap_count += 1
            max_gap_bars = max(max_gap_bars, gap_bars)
            if curr_ts >= recent_cutoff:
                recent_gap_count += 1
                recent_max_gap_bars = max(recent_max_gap_bars, gap_bars)

        out["gap_count"] = int(gap_count)
        out["max_gap_bars"] = float(max_gap_bars)
        out["recent_gap_count"] = int(recent_gap_count)
        out["recent_max_gap_bars"] = float(recent_max_gap_bars)
        out["gap_detected_any"] = bool(gap_count > 0)
        out["gap_detected"] = bool(recent_gap_count > 0)
        return out

    @staticmethod
    def _signal_pick_fresher(
        *,
        current: tuple[BarSeries | None, dict | None] | None,
        candidate: tuple[BarSeries | None, dict | None] | None,
    ) -> tuple[BarSeries | None, dict | None]:
        if candidate is None:
            return current if current is not None else (None, None)
        if current is None:
            return candidate
        current_bars, current_health = current
        cand_bars, cand_health = candidate
        if cand_bars is None:
            return current
        if current_bars is None:
            return candidate
        cur_gap = bool(current_health.get("gap_detected")) if isinstance(current_health, dict) else True
        cand_gap = bool(cand_health.get("gap_detected")) if isinstance(cand_health, dict) else True
        if cur_gap and not cand_gap:
            return candidate
        if cand_gap and not cur_gap:
            return current
        cur_recent_gaps = (
            int(current_health.get("recent_gap_count", 0))
            if isinstance(current_health, dict)
            else 1_000_000
        )
        cand_recent_gaps = (
            int(cand_health.get("recent_gap_count", 0))
            if isinstance(cand_health, dict)
            else 1_000_000
        )
        if cand_recent_gaps < cur_recent_gaps:
            return candidate
        if cand_recent_gaps > cur_recent_gaps:
            return current
        cur_stale = bool(current_health.get("stale")) if isinstance(current_health, dict) else True
        cand_stale = bool(cand_health.get("stale")) if isinstance(cand_health, dict) else True
        if cur_stale and not cand_stale:
            return candidate
        cur_ts = current_health.get("last_bar_ts") if isinstance(current_health, dict) else None
        cand_ts = cand_health.get("last_bar_ts") if isinstance(cand_health, dict) else None
        if isinstance(cand_ts, datetime) and (not isinstance(cur_ts, datetime) or cand_ts > cur_ts):
            return candidate
        return current

    async def _signal_fetch_bars(
        self,
        *,
        contract: Contract,
        duration_str: str,
        min_duration_str: str | None = None,
        bar_size: str,
        use_rth: bool,
        now_ref: datetime,
        strict_zero_gap: bool = False,
        heal_if_stale: bool = False,
    ) -> tuple[BarSeries | None, dict | None]:
        from ...utils.bar_utils import trim_incomplete_last_bar

        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper() or None
        con_id = int(getattr(contract, "conId", 0) or 0)

        async def _fetch(
            *,
            what_to_show: str,
            cache_ttl_sec: float,
            duration_override: str | None = None,
        ) -> tuple[BarSeries | None, dict | None]:
            requested_duration = str(duration_override or duration_str)
            req_duration = str(requested_duration)
            fallback_attempts: list[str] = []
            degraded_from: str | None = None

            def _request_timed_out(diag: dict | None, *, duration: str) -> bool:
                if not isinstance(diag, dict):
                    return False
                status = str(diag.get("status", "") or "").strip().lower()
                if status not in ("timeout", "incomplete"):
                    return False
                request = diag.get("request")
                if not isinstance(request, dict):
                    return True
                if str(request.get("duration_str", "") or "").strip() != str(duration):
                    return False
                if str(request.get("bar_size", "") or "").strip() != str(bar_size):
                    return False
                if str(request.get("what_to_show", "") or "").strip().upper() != str(what_to_show).strip().upper():
                    return False
                if bool(request.get("use_rth")) != bool(use_rth):
                    return False
                return True

            bars = await self._client.historical_bars_ohlcv(
                contract,
                duration_str=req_duration,
                bar_size=bar_size,
                use_rth=use_rth,
                what_to_show=what_to_show,
                cache_ttl_sec=cache_ttl_sec,
            )
            if not bars and IBKRClient._is_intraday_bar_size(str(bar_size)):
                last_diag = self._client.last_historical_request(contract)
                if _request_timed_out(last_diag, duration=req_duration):
                    for fallback_duration in self._signal_timeout_fallback_durations(
                        req_duration,
                        min_duration=min_duration_str,
                    ):
                        fallback_attempts.append(str(fallback_duration))
                        candidate = await self._client.historical_bars_ohlcv(
                            contract,
                            duration_str=str(fallback_duration),
                            bar_size=bar_size,
                            use_rth=use_rth,
                            what_to_show=what_to_show,
                            cache_ttl_sec=0.0,
                        )
                        if candidate:
                            bars = candidate
                            degraded_from = str(req_duration)
                            req_duration = str(fallback_duration)
                            break

            def _annotate_health(payload: dict | None) -> dict | None:
                if not isinstance(payload, dict):
                    return payload
                out = dict(payload)
                out["duration_str"] = str(req_duration)
                if fallback_attempts:
                    out["timeout_fallback_attempts"] = list(fallback_attempts)
                if degraded_from:
                    out["timeout_fallback_used"] = True
                    out["timeout_fallback_from_duration"] = str(degraded_from)
                return out

            if not bars:
                return None, _annotate_health(
                    self._signal_bar_health(
                        bars=None,
                        bar_size=bar_size,
                        now_ref=now_ref,
                        use_rth=use_rth,
                        sec_type=sec_type,
                        source=what_to_show,
                        strict_zero_gap=bool(strict_zero_gap),
                    )
                )
            trimmed = trim_incomplete_last_bar(bars, bar_size=bar_size, now_ref=now_ref)
            if not trimmed:
                return None, _annotate_health(
                    self._signal_bar_health(
                        bars=None,
                        bar_size=bar_size,
                        now_ref=now_ref,
                        use_rth=use_rth,
                        sec_type=sec_type,
                        source=what_to_show,
                        strict_zero_gap=bool(strict_zero_gap),
                    )
                )
            series = BarSeries(
                bars=tuple(trimmed),
                meta=BarSeriesMeta(
                    symbol=symbol,
                    bar_size=str(bar_size),
                    tz_mode="et_naive",
                    session_mode="rth" if bool(use_rth) else "full24",
                    source=f"ibkr:{str(what_to_show).strip().lower()}",
                    extra={"duration_str": str(req_duration), "con_id": int(con_id)},
                ),
            )
            return series, _annotate_health(
                self._signal_bar_health(
                    bars=series.as_list(),
                    bar_size=bar_size,
                    now_ref=now_ref,
                    use_rth=use_rth,
                    sec_type=sec_type,
                    source=what_to_show,
                    strict_zero_gap=bool(strict_zero_gap),
                )
            )

        base = await _fetch(what_to_show="TRADES", cache_ttl_sec=30.0)
        bars, health = base
        if not heal_if_stale:
            return bars, health

        def _annotate_health(base_health: dict | None, **extra: object) -> dict | None:
            if not isinstance(base_health, dict):
                return base_health
            payload = dict(base_health)
            payload.update(extra)
            return payload

        stale_now = bool(health and health.get("stale"))
        gap_now = bool(health and health.get("gap_detected"))
        expected_live = bool(health and health.get("expected_live"))
        if (not stale_now and not gap_now) or not expected_live:
            return bars, health

        heal_key = (
            int(con_id),
            str(symbol or ""),
            str(bar_size),
            bool(use_rth),
        )
        backoff_state = self._signal_heal_backoff.get(heal_key)
        prior_failures = (
            int(backoff_state.get("failures", 0))
            if isinstance(backoff_state, dict)
            else 0
        )
        reconnect_phase = self._client.reconnect_phase()
        if reconnect_phase in ("fast", "slow"):
            return bars, _annotate_health(
                health,
                heal_attempted=False,
                heal_skipped="reconnect",
                heal_reconnect_phase=str(reconnect_phase),
                heal_backoff_failures=int(prior_failures),
            )
        now_mono = asyncio.get_running_loop().time()
        retry_after_mono = (
            float(backoff_state.get("retry_after_mono", 0.0))
            if isinstance(backoff_state, dict)
            else 0.0
        )
        if retry_after_mono > now_mono:
            return bars, _annotate_health(
                health,
                heal_attempted=False,
                heal_skipped="backoff",
                heal_backoff_remaining_sec=max(0.0, float(retry_after_mono - now_mono)),
                heal_backoff_failures=int(prior_failures),
            )

        heal_sources = ["TRADES", "MIDPOINT"]
        heal_durations = [str(duration_str)]
        expanded_duration = self._signal_expand_duration(str(duration_str))
        if expanded_duration and expanded_duration not in heal_durations:
            heal_durations.append(expanded_duration)
        picked = (bars, health)
        for heal_duration in heal_durations:
            for source in heal_sources:
                candidate = await _fetch(
                    what_to_show=source,
                    cache_ttl_sec=0.0,
                    duration_override=heal_duration,
                )
                picked = self._signal_pick_fresher(current=picked, candidate=candidate)

        picked_bars, picked_health = picked
        heal_failed = bool(
            picked_bars is None
            or not isinstance(picked_health, dict)
            or bool(picked_health.get("stale"))
            or bool(picked_health.get("gap_detected"))
        )
        if heal_failed:
            failures = int(prior_failures + 1)
            delay_sec = min(
                float(self._SIGNAL_HEAL_BACKOFF_MAX_SEC),
                float(self._SIGNAL_HEAL_BACKOFF_BASE_SEC) * float(2 ** max(0, failures - 1)),
            )
            self._signal_heal_backoff[heal_key] = {
                "failures": int(failures),
                "retry_after_mono": float(now_mono + delay_sec),
            }
        else:
            failures = 0
            delay_sec = 0.0
            self._signal_heal_backoff.pop(heal_key, None)
        if isinstance(picked_health, dict):
            picked_health = dict(picked_health)
            picked_health["heal_attempted"] = True
            picked_health["heal_sources"] = heal_sources
            picked_health["heal_durations"] = heal_durations
            picked_health["heal_backoff_failures"] = int(failures)
            picked_health["heal_backoff_armed"] = bool(heal_failed)
            if heal_failed:
                picked_health["heal_backoff_delay_sec"] = float(delay_sec)
        return picked_bars, picked_health
