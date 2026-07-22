"""Rich operator-facing diagnostics for live bot journal events."""

from __future__ import annotations

from typing import Mapping


class JournalDiagnostics:
    @staticmethod
    def _coerce_mapping(raw: object) -> dict[str, object]:
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _compact_bar_size(raw: object) -> str:
        text = str(raw or "").strip()
        if not text:
            return "?"
        parts = text.split()
        if len(parts) >= 2 and parts[0].isdigit():
            num = parts[0]
            unit = parts[1].lower()
            if unit.startswith("min"):
                return f"{num}m"
            if unit.startswith("hour"):
                return f"{num}h"
            if unit.startswith("day"):
                return f"{num}D"
        return (
            text.replace(" mins", "m")
            .replace(" min", "m")
            .replace(" hours", "h")
            .replace(" hour", "h")
            .replace(" days", "D")
            .replace(" day", "D")
        )

    @staticmethod
    def _is_disabled(raw: object) -> bool:
        if raw is None:
            return True
        if isinstance(raw, bool):
            return not raw
        if isinstance(raw, (int, float)):
            return float(raw) == 0.0
        text = str(raw).strip().lower()
        return text in ("", "off", "none", "null", "false", "0")

    @staticmethod
    def _dir_badge(raw: object) -> str:
        direction = str(raw or "").strip().lower()
        if direction == "up":
            return "[bold #73d89e]▲ up[/]"
        if direction == "down":
            return "[bold #ff5f87]▼ down[/]"
        if direction:
            return direction
        return "[dim]-[/]"

    @staticmethod
    def _bool_badge(value: bool) -> str:
        return "[bold #73d89e]yes[/]" if bool(value) else "[dim #9aa0a6]no[/]"

    @staticmethod
    def _pass_fail_badge(ok: bool) -> str:
        return "[bold #73d89e]✓[/]" if bool(ok) else "[bold #ff5f87]✗[/]"

    @staticmethod
    def _signed_pct(value: object, *, digits: int = 3) -> str | None:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if parsed > 0:
            color = "#73d89e"
        elif parsed < 0:
            color = "#ff5f87"
        else:
            color = "#9aa0a6"
        return f"[{color}]{parsed:+.{int(digits)}f}%[/]"

    @staticmethod
    def _filter_check_vector(checks: Mapping[str, object]) -> str:
        order = ("rv", "time", "skip_first", "cooldown", "shock_gate", "permission", "volume")
        tokens: list[str] = []
        for key in order:
            if key in checks:
                tokens.append(f"{key}={JournalDiagnostics._pass_fail_badge(bool(checks.get(key)))}")
        for key, raw_ok in checks.items():
            if str(key) in order:
                continue
            tokens.append(f"{key}={JournalDiagnostics._pass_fail_badge(bool(raw_ok))}")
        return JournalDiagnostics._join_tokens(tokens)

    @staticmethod
    def _float_or_none(raw: object) -> float | None:
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or_none(raw: object) -> int | None:
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _time_window_ok(hour: int, *, start: int, end: int) -> bool:
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end

    @staticmethod
    def _filter_fail_reasons(
        *,
        detail: dict[str, object],
        filters: dict[str, object],
        failed_filters: list[str],
    ) -> list[str]:
        sig = detail.get("signal") if isinstance(detail.get("signal"), dict) else {}
        entry_dir_raw = detail.get("entry_dir")
        if entry_dir_raw not in ("up", "down"):
            entry_dir_raw = sig.get("entry_dir") if isinstance(sig, dict) else None
        entry_dir = str(entry_dir_raw) if entry_dir_raw in ("up", "down") else None
        spread = JournalDiagnostics._float_or_none(detail.get("ema_spread_pct"))
        if spread is None and isinstance(sig, dict):
            spread = JournalDiagnostics._float_or_none(sig.get("ema_spread_pct"))
        slope = JournalDiagnostics._float_or_none(detail.get("ema_slope_pct"))
        if slope is None and isinstance(sig, dict):
            slope = JournalDiagnostics._float_or_none(sig.get("ema_slope_pct"))

        rv = JournalDiagnostics._float_or_none(detail.get("rv"))
        bar_hour_et = JournalDiagnostics._int_or_none(detail.get("bar_hour_et"))
        bars_in_day = JournalDiagnostics._int_or_none(detail.get("bars_in_day"))
        cooldown_ok = bool(detail.get("cooldown_ok")) if "cooldown_ok" in detail else None
        cooldown_bars = JournalDiagnostics._int_or_none(detail.get("cooldown_bars"))
        shock = detail.get("shock")
        shock_dir = str(detail.get("shock_dir") or "").strip().lower()
        if shock_dir not in ("up", "down"):
            shock_dir = None
        volume = JournalDiagnostics._float_or_none(detail.get("volume"))
        volume_ema = JournalDiagnostics._float_or_none(detail.get("volume_ema"))
        volume_ema_ready = bool(detail.get("volume_ema_ready")) if "volume_ema_ready" in detail else None

        out: list[str] = []
        for name in failed_filters:
            key = str(name)
            if key == "rv":
                rv_min = JournalDiagnostics._float_or_none(filters.get("rv_min"))
                rv_max = JournalDiagnostics._float_or_none(filters.get("rv_max"))
                parts: list[str] = []
                if rv is None:
                    parts.append("rv unavailable")
                if rv_min is not None and rv is not None and rv < rv_min:
                    parts.append(f"rv {rv:.2f} < rv_min {rv_min:.2f}")
                if rv_max is not None and rv is not None and rv > rv_max:
                    parts.append(f"rv {rv:.2f} > rv_max {rv_max:.2f}")
                out.append(f"rv: {' ; '.join(parts) if parts else 'rv threshold miss'}")
                continue

            if key == "time":
                start = JournalDiagnostics._int_or_none(filters.get("entry_start_hour_et"))
                end = JournalDiagnostics._int_or_none(filters.get("entry_end_hour_et"))
                if start is None or end is None:
                    start = JournalDiagnostics._int_or_none(filters.get("entry_start_hour"))
                    end = JournalDiagnostics._int_or_none(filters.get("entry_end_hour"))
                if start is not None and end is not None and bar_hour_et is not None:
                    ok = JournalDiagnostics._time_window_ok(int(bar_hour_et), start=int(start), end=int(end))
                    if not ok:
                        out.append(f"time: hour {int(bar_hour_et):02d} outside [{int(start):02d},{int(end):02d})")
                    else:
                        out.append("time: window blocked")
                else:
                    out.append("time: window blocked")
                continue

            if key == "skip_first":
                skip_first = JournalDiagnostics._int_or_none(filters.get("skip_first_bars"))
                if skip_first is not None and bars_in_day is not None:
                    out.append(f"skip_first: bar_index {int(bars_in_day)} <= skip_first_bars {int(skip_first)}")
                else:
                    out.append("skip_first: early-bar gate")
                continue

            if key == "cooldown":
                if cooldown_ok is False:
                    if cooldown_bars is not None:
                        out.append(f"cooldown: waiting (cooldown_bars={int(cooldown_bars)})")
                    else:
                        out.append("cooldown: waiting")
                else:
                    out.append("cooldown: waiting")
                continue

            if key == "shock_gate":
                shock_mode = str(filters.get("shock_gate_mode") or filters.get("shock_mode") or "off").strip().lower()
                parts: list[str] = [f"mode={shock_mode or 'off'}"]
                if shock is not None:
                    parts.append(f"shock={'on' if bool(shock) else 'off'}")
                if shock_dir in ("up", "down"):
                    parts.append(f"shock_dir={shock_dir}")
                if entry_dir in ("up", "down"):
                    parts.append(f"entry_dir={entry_dir}")
                out.append(f"shock_gate: {' ; '.join(parts)}")
                continue

            if key == "permission":
                spread_min = JournalDiagnostics._float_or_none(filters.get("ema_spread_min_pct"))
                spread_min_down = JournalDiagnostics._float_or_none(filters.get("ema_spread_min_pct_down"))
                if entry_dir == "down" and spread_min_down is not None:
                    spread_min = spread_min_down
                slope_min = JournalDiagnostics._float_or_none(filters.get("ema_slope_min_pct"))
                slope_signed_up = JournalDiagnostics._float_or_none(filters.get("ema_slope_signed_min_pct_up"))
                slope_signed_down = JournalDiagnostics._float_or_none(filters.get("ema_slope_signed_min_pct_down"))
                parts: list[str] = []
                if spread_min is not None and spread is not None and spread < spread_min:
                    parts.append(f"spread {spread:+.3f}% < ema_spread_min_pct {spread_min:+.3f}%")
                if slope_min is not None and slope is not None and abs(slope) < slope_min:
                    parts.append(f"slope {slope:+.3f}% < ema_slope_min_pct {slope_min:+.3f}%")
                if entry_dir == "up" and slope_signed_up is not None and slope is not None and slope < slope_signed_up:
                    parts.append(f"signed_slope {slope:+.3f}% < ema_slope_signed_min_pct_up {slope_signed_up:+.3f}%")
                if entry_dir == "down" and slope_signed_down is not None and slope is not None and slope > -slope_signed_down:
                    parts.append(
                        f"signed_slope {slope:+.3f}% > -ema_slope_signed_min_pct_down {-slope_signed_down:+.3f}%"
                    )
                out.append(f"permission: {' ; '.join(parts) if parts else 'ema quality gate'}")
                continue

            if key == "volume":
                ratio_min = JournalDiagnostics._float_or_none(filters.get("volume_ratio_min"))
                if volume_ema_ready is False:
                    out.append("volume: volume_ema not ready")
                    continue
                if ratio_min is not None and volume is not None and volume_ema is not None and volume_ema > 0:
                    ratio = float(volume) / float(volume_ema)
                    out.append(f"volume: ratio {ratio:.2f} < volume_ratio_min {ratio_min:.2f}")
                else:
                    out.append("volume: ratio gate")
                continue

            out.append(f"{key}: blocked")
        return out

    @staticmethod
    def _entry_ctx_line(entry_ctx: dict[str, object]) -> str | None:
        tokens: list[str] = []
        entries_today = JournalDiagnostics._int_or_none(entry_ctx.get("entries_today"))
        max_entries = JournalDiagnostics._int_or_none(entry_ctx.get("max_entries_per_day"))
        if entries_today is not None and max_entries is not None:
            limit = "∞" if int(max_entries) <= 0 else str(int(max_entries))
            tokens.append(f"entries_today={int(entries_today)}/{limit}")
        entry_capacity = entry_ctx.get("entry_capacity")
        if isinstance(entry_capacity, bool):
            tokens.append(f"entry_capacity={JournalDiagnostics._pass_fail_badge(bool(entry_capacity))}")
        entry_dir = str(entry_ctx.get("entry_dir") or "")
        if entry_dir in ("up", "down"):
            tokens.append(f"entry_dir={entry_dir}")
        allowed = entry_ctx.get("allowed_directions")
        if isinstance(allowed, list) and allowed:
            compact = ",".join(str(v) for v in allowed if str(v))
            if compact:
                tokens.append(f"allowed={{{compact}}}")
        direction_ok = entry_ctx.get("direction_ok")
        if isinstance(direction_ok, bool):
            tokens.append(f"direction_ok={JournalDiagnostics._pass_fail_badge(bool(direction_ok))}")
        if not tokens:
            return None
        return "[dim]entry_ctx[/] " + JournalDiagnostics._join_tokens(tokens)

    @staticmethod
    def _next_open_ctx_line(next_open_ctx: dict[str, object]) -> str | None:
        tokens: list[str] = []
        allowed = next_open_ctx.get("allowed")
        if isinstance(allowed, bool):
            tokens.append(f"allowed={JournalDiagnostics._pass_fail_badge(bool(allowed))}")
        reason = str(next_open_ctx.get("reason") or "").strip()
        if reason:
            tokens.append(f"reason={reason}")
        due_ts = next_open_ctx.get("due_ts")
        if isinstance(due_ts, str) and len(due_ts) >= 16:
            tokens.append(f"due={due_ts[11:16]}")
        fill_mode = str(next_open_ctx.get("fill_mode") or "").strip()
        if fill_mode:
            tokens.append(f"fill_mode={fill_mode}")
        if not tokens:
            return None
        return "[dim]next_open_ctx[/] " + JournalDiagnostics._join_tokens(tokens)

    @staticmethod
    def _graph_entry_line(graph_entry: dict[str, object]) -> str | None:
        tokens: list[str] = []
        gate = str(graph_entry.get("gate") or "").strip()
        reason = str(graph_entry.get("reason") or "").strip()
        if gate:
            tokens.append(f"gate={gate}")
        if reason:
            tokens.append(f"reason={reason}")
        trace = graph_entry.get("trace") if isinstance(graph_entry.get("trace"), dict) else {}
        if isinstance(trace, dict):
            policy = str(trace.get("policy") or "").strip()
            if policy:
                tokens.append(f"policy={policy}")
            graph_cfg = trace.get("graph") if isinstance(trace.get("graph"), dict) else {}
            if isinstance(graph_cfg, dict):
                profile = str(graph_cfg.get("profile") or "").strip()
                if profile:
                    tokens.append(f"profile={profile}")
            for key in ("tr_ratio", "min", "min_abs", "shock_atr_pct", "max", "slope_med_pct", "slope_vel_pct"):
                if key not in trace:
                    continue
                value = JournalDiagnostics._float_or_none(trace.get(key))
                if value is None:
                    continue
                # Use repr() to preserve full float precision; many thresholds live near ~1e-4.
                tokens.append(f"{key}={value!r}")
        if not tokens:
            return None
        return "[dim]graph_entry[/] " + JournalDiagnostics._join_tokens(tokens)

    @staticmethod
    def _append_float(tokens: list[str], *, label: str, value: object, fmt: str) -> None:
        try:
            if value is not None:
                tokens.append(f"{label}={float(value):{fmt}}")
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _append_int(tokens: list[str], *, label: str, value: object) -> None:
        try:
            if value is not None:
                tokens.append(f"{label}={int(value)}")
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _append_hhmm(tokens: list[str], *, label: str, value: object, min_len: int = 1) -> None:
        if isinstance(value, str) and len(value) >= int(min_len):
            tokens.append(f"{label}={value[11:16]}")

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        return " • ".join(token for token in tokens if token)

    @staticmethod
    def _active_knobs_line(
        *,
        meta: dict[str, object],
        strategy: dict[str, object],
        filters: dict[str, object],
        max_tokens: int,
    ) -> str | None:
        active_knobs = JournalDiagnostics._active_knob_tokens(
            meta=meta,
            strategy=strategy,
            filters=filters,
            max_tokens=max_tokens,
        )
        if not active_knobs:
            return None
        return "[dim]active_knobs[/] " + " • ".join(active_knobs)

    @staticmethod
    def _active_knob_tokens(
        *,
        meta: dict[str, object],
        strategy: dict[str, object],
        filters: dict[str, object],
        max_tokens: int = 10,
    ) -> list[str]:
        def _first(*values: object) -> object:
            for value in values:
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                return value
            return None

        tokens: list[str] = []

        entry_signal_raw = _first(meta.get("entry_signal"), strategy.get("entry_signal"))
        entry_signal = str(entry_signal_raw or "").strip().lower()
        signal_bar = str(_first(meta.get("signal_bar_size"), strategy.get("signal_bar_size")) or "").strip()
        if entry_signal or signal_bar:
            if entry_signal == "ema":
                preset = str(strategy.get("ema_preset") or "").strip()
                sig_name = f"EMA({preset})" if preset else "EMA"
            else:
                sig_name = entry_signal.upper() if entry_signal else "SIGNAL"
            tokens.append(f"signal={sig_name}@{JournalDiagnostics._compact_bar_size(signal_bar)}")
        use_rth_value = _first(meta.get("signal_use_rth"), strategy.get("signal_use_rth"))
        if use_rth_value is not None:
            tokens.append("session=RTH" if bool(use_rth_value) else "session=FULL24")

        if bool(strategy.get("regime_router")):
            fast_raw = strategy.get("regime_router_fast_window_days")
            slow_raw = strategy.get("regime_router_slow_window_days")
            try:
                fast_days = int(fast_raw) if fast_raw is not None else None
            except (TypeError, ValueError):
                fast_days = None
            try:
                slow_days = int(slow_raw) if slow_raw is not None else None
            except (TypeError, ValueError):
                slow_days = None
            if fast_days is None and slow_days is None:
                tokens.append("router=on")
            else:
                fast_label = str(fast_days) if fast_days is not None else "?"
                slow_label = str(slow_days) if slow_days is not None else "?"
                tokens.append(f"router=on({fast_label}/{slow_label}D)")

        regime_mode = str(_first(meta.get("regime_mode"), strategy.get("regime_mode")) or "").strip().lower()
        regime_bar = str(_first(meta.get("regime_bar_size"), strategy.get("regime_bar_size"), signal_bar) or "").strip()
        if regime_mode and not JournalDiagnostics._is_disabled(regime_mode):
            if regime_mode == "supertrend":
                params: list[str] = []
                atr_p = strategy.get("supertrend_atr_period")
                mult = strategy.get("supertrend_multiplier")
                src = strategy.get("supertrend_source")
                if atr_p is not None:
                    try:
                        params.append(str(int(atr_p)))
                    except (TypeError, ValueError):
                        pass
                if mult is not None:
                    try:
                        params.append(f"{float(mult):g}")
                    except (TypeError, ValueError):
                        pass
                if src:
                    params.append(str(src))
                suffix = f"({','.join(params)})" if params else ""
                tokens.append(f"bias=Supertrend{suffix}@{JournalDiagnostics._compact_bar_size(regime_bar)}")
            elif regime_mode == "ema":
                preset = str(strategy.get("regime_ema_preset") or "").strip()
                bias_name = f"EMA({preset})" if preset else "EMA"
                tokens.append(f"bias={bias_name}@{JournalDiagnostics._compact_bar_size(regime_bar)}")
            else:
                tokens.append(f"bias={regime_mode}@{JournalDiagnostics._compact_bar_size(regime_bar)}")

        regime2_mode = str(_first(meta.get("regime2_mode"), strategy.get("regime2_mode")) or "").strip().lower()
        if regime2_mode and not JournalDiagnostics._is_disabled(regime2_mode):
            regime2_bar = str(_first(meta.get("regime2_bar_size"), strategy.get("regime2_bar_size"), signal_bar) or "").strip()
            regime2_apply = str(_first(meta.get("regime2_apply_to"), strategy.get("regime2_apply_to"), "both") or "both")
            tokens.append(f"bias2={regime2_mode}@{JournalDiagnostics._compact_bar_size(regime2_bar)} ({regime2_apply})")

        exec_mode = str(_first(meta.get("spot_exec_feed_mode"), strategy.get("spot_exec_feed_mode")) or "").strip()
        exec_bar = str(_first(meta.get("spot_exec_bar_size"), strategy.get("spot_exec_bar_size"), signal_bar) or "").strip()
        if exec_mode or exec_bar:
            tokens.append(f"execution={exec_mode or '?'}@{JournalDiagnostics._compact_bar_size(exec_bar)}")

        risk_mode = str(_first(filters.get("riskoff_mode"), strategy.get("riskoff_mode")) or "").strip()
        if risk_mode and not JournalDiagnostics._is_disabled(risk_mode):
            tokens.append(f"risk_mode={risk_mode}")

        shock_gate = str(_first(meta.get("shock_gate_mode"), filters.get("shock_gate_mode")) or "").strip()
        shock_detector = str(_first(meta.get("shock_detector"), filters.get("shock_detector")) or "").strip()
        if shock_gate and not JournalDiagnostics._is_disabled(shock_gate):
            shock_token = f"shock_gate={shock_gate}"
            if shock_detector:
                shock_token = f"{shock_token} ({shock_detector})"
            tokens.append(shock_token)
        shock_scale = str(_first(meta.get("shock_scale_detector"), filters.get("shock_scale_detector")) or "").strip()
        if shock_scale and not JournalDiagnostics._is_disabled(shock_scale):
            tokens.append(f"shock_scale={shock_scale}")

        resize_policy = str(strategy.get("spot_resize_policy") or "").strip()
        if resize_policy and not JournalDiagnostics._is_disabled(resize_policy):
            tokens.append(f"resize_policy={resize_policy}")

        risk_overlay = str(strategy.get("spot_risk_overlay_policy") or "").strip()
        if risk_overlay and not JournalDiagnostics._is_disabled(risk_overlay):
            tokens.append(f"risk_overlay={risk_overlay}")

        if bool(strategy.get("spot_dual_branch_enabled")):
            priority = str(strategy.get("spot_dual_branch_priority") or "").strip()
            tokens.append(f"branching=dual ({priority or '?'})")

        tick_gate_mode = str(strategy.get("tick_gate_mode") or "").strip()
        if tick_gate_mode and not JournalDiagnostics._is_disabled(tick_gate_mode):
            tokens.append(f"tick_gate={tick_gate_mode}")

        unique: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            unique.append(token)
            seen.add(token)
        return unique

    @staticmethod
    def _signal_message(*, detail: dict, extra: dict) -> str:
        sig = detail.get("signal") or {}
        meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
        strategy = JournalDiagnostics._coerce_mapping(extra.get("strategy"))
        filters = JournalDiagnostics._coerce_mapping(extra.get("filters"))
        state = str(sig.get("state") or "")
        entry_dir = str(sig.get("entry_dir") or "")
        regime_dir = str(sig.get("regime_dir") or "")
        regime_ready = bool(sig.get("regime_ready"))
        cross_up = bool(sig.get("cross_up"))
        cross_down = bool(sig.get("cross_down"))
        entry_badge = JournalDiagnostics._dir_badge(entry_dir)
        if (
            entry_dir not in ("up", "down")
            and regime_ready
            and state in ("up", "down")
            and regime_dir in ("up", "down")
            and state != regime_dir
        ):
            entry_badge = "[bold #ff5f87]🚫[/]"

        dir_tokens: list[str] = [
            f"state={JournalDiagnostics._dir_badge(state)}",
            f"entry={entry_badge}",
        ]
        if cross_up:
            dir_tokens.append("cross=[bold #73d89e]▲ up[/]")
        elif cross_down:
            dir_tokens.append("cross=[bold #ff5f87]▼ down[/]")
        spread = JournalDiagnostics._signed_pct(sig.get("ema_spread_pct"), digits=3)
        if spread is not None:
            dir_tokens.append(f"spread={spread}")
        slope = JournalDiagnostics._signed_pct(sig.get("ema_slope_pct"), digits=3)
        if slope is not None:
            dir_tokens.append(f"slope={slope}")

        bias_tokens: list[str] = []
        if regime_dir:
            bias_tokens.append(f"regime={JournalDiagnostics._dir_badge(regime_dir)}")
        if not regime_ready:
            bias_tokens.append(f"ready={JournalDiagnostics._pass_fail_badge(False)}")
        shock = detail.get("shock")
        post_bias_tokens: list[str] = []
        if isinstance(shock, dict):
            shock_on = shock.get("shock")
            shock_dir = str(shock.get("dir") or "")
            shock_dir_source = str(shock.get("direction_source_effective") or "").strip()
            shock_dir_ret = JournalDiagnostics._signed_pct(shock.get("dir_ret_sum_pct"), digits=2)
            if bool(shock_on):
                if shock_dir in ("up", "down"):
                    bias_tokens.append(f"shock={JournalDiagnostics._dir_badge(shock_dir)}")
                else:
                    bias_tokens.append("shock=[bold #ffd166]on[/]")
            elif shock_on is False:
                bias_tokens.append("shock=[dim]off[/]")
            if shock_dir_source:
                bias_tokens.append(f"dir_src={shock_dir_source}")
            if shock_dir_ret is not None:
                post_bias_tokens.append(f"dirΣ={shock_dir_ret}")
            JournalDiagnostics._append_float(post_bias_tokens, label="atr", value=shock.get("atr_pct"), fmt=".1f%")
            JournalDiagnostics._append_float(post_bias_tokens, label="peak", value=shock.get("peak_close"), fmt=".2f")
            down_streak = JournalDiagnostics._int_or_none(shock.get("dir_down_streak_bars"))
            up_streak = JournalDiagnostics._int_or_none(shock.get("dir_up_streak_bars"))
            if down_streak is not None and down_streak > 0:
                post_bias_tokens.append(f"dir↓={int(down_streak)}b")
            if up_streak is not None and up_streak > 0:
                post_bias_tokens.append(f"dir↑={int(up_streak)}b")

            dd_pct = JournalDiagnostics._signed_pct(shock.get("drawdown_pct"), digits=1)
            dd_on_raw = shock.get("drawdown_on_pct")
            dd_off_raw = shock.get("drawdown_off_pct")
            if dd_pct is not None:
                try:
                    dd_on = float(dd_on_raw) if dd_on_raw is not None else None
                except (TypeError, ValueError):
                    dd_on = None
                try:
                    dd_off = float(dd_off_raw) if dd_off_raw is not None else None
                except (TypeError, ValueError):
                    dd_off = None
                if dd_on is not None:
                    post_bias_tokens.append(f"dd={dd_pct}/on={dd_on:+.1f}%")
                else:
                    post_bias_tokens.append(f"dd={dd_pct}")
                if dd_off is not None:
                    post_bias_tokens.append(f"off={dd_off:+.1f}%")

            try:
                dd_dist_on = float(shock.get("drawdown_dist_on_pct")) if shock.get("drawdown_dist_on_pct") is not None else None
            except (TypeError, ValueError):
                dd_dist_on = None
            if dd_dist_on is not None:
                if dd_dist_on >= 0:
                    post_bias_tokens.append(f"dd→on=[bold #73d89e]✓ +{dd_dist_on:.1f}pp[/]")
                else:
                    post_bias_tokens.append(f"dd→on=[bold #ff5f87]… {dd_dist_on:+.1f}pp[/]")
            try:
                dd_dist_on_vel = (
                    float(shock.get("drawdown_dist_on_vel_pp")) if shock.get("drawdown_dist_on_vel_pp") is not None else None
                )
            except (TypeError, ValueError):
                dd_dist_on_vel = None
            if dd_dist_on_vel is not None:
                post_bias_tokens.append(f"ddv={dd_dist_on_vel:+.2f}pp")
            try:
                dd_dist_on_accel = (
                    float(shock.get("drawdown_dist_on_accel_pp"))
                    if shock.get("drawdown_dist_on_accel_pp") is not None
                    else None
                )
            except (TypeError, ValueError):
                dd_dist_on_accel = None
            if dd_dist_on_accel is not None:
                post_bias_tokens.append(f"dda={dd_dist_on_accel:+.2f}pp")
            prearm_down_streak = JournalDiagnostics._int_or_none(shock.get("prearm_down_streak_bars"))
            if prearm_down_streak is not None and prearm_down_streak > 0:
                post_bias_tokens.append(f"pre={int(prearm_down_streak)}b")

            ramp = shock.get("ramp")
            if isinstance(ramp, dict):
                for key, glyph in (("down", "↓"), ("up", "↑")):
                    node = ramp.get(key)
                    if not isinstance(node, dict):
                        continue
                    mult = JournalDiagnostics._float_or_none(node.get("risk_mult"))
                    intensity = JournalDiagnostics._float_or_none(node.get("intensity"))
                    phase = str(node.get("phase") or "").strip()
                    if mult is None:
                        continue
                    show = float(mult) > 1.01 or (intensity is not None and float(intensity) > 0.05)
                    if not bool(show):
                        continue
                    phase_clean = phase if phase and phase != "off" else ""
                    token = f"r{glyph}{phase_clean}:x{float(mult):.2f}" if phase_clean else f"r{glyph}x{float(mult):.2f}"
                    post_bias_tokens.append(token)

            try:
                dd_dist_off = float(shock.get("drawdown_dist_off_pct")) if shock.get("drawdown_dist_off_pct") is not None else None
            except (TypeError, ValueError):
                dd_dist_off = None
            if dd_dist_off is not None:
                if dd_dist_off >= 0:
                    post_bias_tokens.append(f"dd→off=[bold #73d89e]↗ {dd_dist_off:+.1f}pp[/]")
                else:
                    post_bias_tokens.append(f"dd→off=[bold #ff5f87]↘ {dd_dist_off:+.1f}pp[/]")

        line1: list[str] = [
            f"dir=[bold #ffd166]⟪[/]{JournalDiagnostics._join_tokens(dir_tokens)}[bold #ffd166]⟫[/]"
        ]
        if bias_tokens:
            line1.append(
                f"bias=[bold #ffd166]⟪[/]{JournalDiagnostics._join_tokens(bias_tokens)}[bold #ffd166]⟫[/]"
            )
        line1.extend(post_bias_tokens)

        risk = detail.get("risk")
        if isinstance(risk, dict):
            risk_flags: list[str] = []
            if bool(risk.get("riskoff")):
                risk_flags.append("risk_off")
            if bool(risk.get("riskpanic")):
                risk_flags.append("risk_panic")
            if bool(risk.get("riskpop")):
                risk_flags.append("risk_pop")
            if risk_flags:
                line1.append(f"risk={','.join(risk_flags)}")
        JournalDiagnostics._append_float(line1, label="rv", value=detail.get("rv"), fmt=".2f")
        JournalDiagnostics._append_int(line1, label="bar_index", value=detail.get("bars_in_day"))
        JournalDiagnostics._append_hhmm(line1, label="time", value=detail.get("bar_ts"), min_len=16)

        bar_health = detail.get("bar_health") if isinstance(detail.get("bar_health"), dict) else None
        if bar_health is not None:
            lag_bars = JournalDiagnostics._float_or_none(bar_health.get("lag_bars"))
            if lag_bars is not None:
                line1.append(f"lag:{max(0, int(round(lag_bars)))}b")
            if "stale" in bar_health:
                stale_ok = not bool(bar_health.get("stale"))
                line1.append(f"stale={JournalDiagnostics._pass_fail_badge(stale_ok)}")
            if "gap_detected" in bar_health:
                gap_ok = not bool(bar_health.get("gap_detected"))
                line1.append(f"gap={JournalDiagnostics._pass_fail_badge(gap_ok)}")

        lines: list[str] = []
        msg = JournalDiagnostics._join_tokens(line1)
        if msg:
            lines.append(msg)
        active_line = JournalDiagnostics._active_knobs_line(
            meta=meta,
            strategy=strategy,
            filters=filters,
            max_tokens=10,
        )
        if active_line:
            lines.append(active_line)

        checks = detail.get("filter_checks")
        if isinstance(checks, dict) and checks:
            lines.append(f"[dim]gate_vector[/] {JournalDiagnostics._filter_check_vector(checks)}")
            failed_filters = detail.get("failed_filters")
            if isinstance(failed_filters, list):
                failed = [str(v) for v in failed_filters if str(v)]
            else:
                failed = [str(name) for name, ok in checks.items() if not bool(ok)]
            fail_reasons = JournalDiagnostics._filter_fail_reasons(
                detail=detail,
                filters=filters,
                failed_filters=failed,
            )
            if fail_reasons:
                lines.append("[dim]gate_fail[/] " + " ; ".join(fail_reasons))

        return "\n".join(lines)

    @staticmethod
    def _gate_message(*, detail: dict, extra: dict) -> str:
        strategy = JournalDiagnostics._coerce_mapping(extra.get("strategy"))
        filters = JournalDiagnostics._coerce_mapping(extra.get("filters"))

        def _health_faults(raw: object) -> tuple[bool, bool]:
            if not isinstance(raw, dict):
                return False, False
            return bool(raw.get("stale")), bool(raw.get("gap_detected"))

        line1: list[str] = []
        JournalDiagnostics._append_hhmm(line1, label="time", value=detail.get("bar_ts"), min_len=16)
        if detail.get("direction") in ("up", "down"):
            line1.append(f"direction={JournalDiagnostics._dir_badge(detail.get('direction'))}")
        if detail.get("entry_dir") in ("up", "down"):
            line1.append(f"entry={JournalDiagnostics._dir_badge(detail.get('entry_dir'))}")
        if detail.get("regime_dir") in ("up", "down"):
            line1.append(f"regime={JournalDiagnostics._dir_badge(detail.get('regime_dir'))}")
        if detail.get("shock_dir") in ("up", "down"):
            line1.append(f"shock={JournalDiagnostics._dir_badge(detail.get('shock_dir'))}")
        if "cooldown_ok" in detail:
            line1.append(f"cooldown_ok={JournalDiagnostics._bool_badge(bool(detail.get('cooldown_ok')))}")
        JournalDiagnostics._append_int(line1, label="bar_index", value=detail.get("bars_in_day"))
        diag_stage = str(detail.get("diag_stage") or "").strip()
        if diag_stage:
            line1.append(f"diag={diag_stage}")

        def _append_have_need(label: str, *, have_key: str, need_key: str) -> None:
            have_raw = detail.get(have_key)
            need_raw = detail.get(need_key)
            try:
                have = int(have_raw) if have_raw is not None else None
            except (TypeError, ValueError):
                have = None
            try:
                need = int(need_raw) if need_raw is not None else None
            except (TypeError, ValueError):
                need = None
            if have is None or need is None:
                return
            line1.append(f"{label}={have}/{need}")

        _append_have_need("sig", have_key="signal_have", need_key="signal_need")
        _append_have_need("reg", have_key="regime_have", need_key="regime_need")
        _append_have_need("reg2", have_key="regime2_have", need_key="regime2_need")
        JournalDiagnostics._append_int(line1, label="seed", value=detail.get("regime_router_seed_count"))
        JournalDiagnostics._append_int(line1, label="positions", value=detail.get("items"))
        mode = detail.get("mode")
        if mode:
            line1.append(f"mode={mode}")
        fill_mode = detail.get("fill_mode")
        if isinstance(fill_mode, str) and fill_mode:
            line1.append(f"fill_mode={fill_mode}")
        why = detail.get("reason")
        if why and isinstance(why, str) and why:
            line1.append(f"why={why}")
        JournalDiagnostics._append_float(line1, label="rv", value=detail.get("rv"), fmt=".2f")
        state = detail.get("state")
        if state in ("up", "down"):
            line1.append(f"state={JournalDiagnostics._dir_badge(state)}")
        signal_stale, signal_gap = _health_faults(detail.get("bar_health"))
        regime_stale, regime_gap = _health_faults(detail.get("regime_bar_health"))
        regime2_stale, regime2_gap = _health_faults(detail.get("regime2_bar_health"))
        health_gate = ""
        health_faults: list[str] = []
        if regime_stale or regime_gap:
            health_gate = "regime"
            if regime_stale:
                health_faults.append("stale")
            if regime_gap:
                health_faults.append("gap")
        elif regime2_stale or regime2_gap:
            health_gate = "regime2"
            if regime2_stale:
                health_faults.append("stale")
            if regime2_gap:
                health_faults.append("gap")
        elif signal_stale or signal_gap:
            health_gate = "signal"
            if signal_stale:
                health_faults.append("stale")
            if signal_gap:
                health_faults.append("gap")
        if health_gate and health_faults:
            line1.append(f"health_gate={health_gate}({','.join(health_faults)})")

        if "regime_router_ready" in detail:
            line1.append(f"router_ready={JournalDiagnostics._pass_fail_badge(bool(detail.get('regime_router_ready')))}")
        router_host = str(detail.get("regime_router_host") or "").strip()
        if router_host:
            line1.append(f"router_host={router_host}")
        router_climate = str(detail.get("regime_router_climate") or "").strip()
        if router_climate:
            line1.append(f"climate={router_climate}")
        router_entry = detail.get("regime_router_entry_dir")
        if router_entry in ("up", "down"):
            line1.append(f"router_entry={JournalDiagnostics._dir_badge(router_entry)}")
        JournalDiagnostics._append_int(line1, label="dwell", value=detail.get("regime_router_dwell_days"))
        JournalDiagnostics._append_float(line1, label="crash_ret", value=detail.get("regime_router_crash_ret"), fmt="+.3f")
        JournalDiagnostics._append_float(line1, label="crash_dd", value=detail.get("regime_router_crash_maxdd"), fmt="+.3f")
        JournalDiagnostics._append_float(line1, label="crash_rv", value=detail.get("regime_router_crash_rv"), fmt="+.3f")
        JournalDiagnostics._append_float(line1, label="fast_ret", value=detail.get("regime_router_fast_ret"), fmt="+.3f")
        JournalDiagnostics._append_float(line1, label="slow_ret", value=detail.get("regime_router_slow_ret"), fmt="+.3f")
        regime4_owner = str(detail.get("regime4_owner") or "").strip()
        regime4_state = str(detail.get("regime4_state") or "").strip()
        if regime4_owner or regime4_state:
            label = f"{regime4_owner}:{regime4_state}" if regime4_owner and regime4_state else (regime4_owner or regime4_state)
            line1.append(f"regime4={label}")
        line1_text = JournalDiagnostics._join_tokens(line1)
        lines: list[str] = [line1_text] if line1_text else []

        line2: list[str] = []
        failed = detail.get("failed_filters")
        if isinstance(failed, list) and failed:
            compact = ",".join(str(v) for v in failed if str(v))
            if compact:
                line2.append(f"failed_filters={compact}")
        checks = detail.get("filter_checks")
        if isinstance(checks, dict):
            bad = [str(name) for name, ok in checks.items() if not bool(ok)]
            if bad and not isinstance(failed, list):
                line2.append(f"failed_filters={','.join(bad)}")
        JournalDiagnostics._append_hhmm(line2, label="next_open_due", value=detail.get("next_open_due"))
        JournalDiagnostics._append_hhmm(line2, label="next_open_from", value=detail.get("next_open_due_from"))
        JournalDiagnostics._append_hhmm(line2, label="now", value=detail.get("now_wall_ts"))

        calc = detail.get("calc")
        if isinstance(calc, dict):
            session = calc.get("session")
            if isinstance(session, str) and session:
                line2.append(f"market_session={session}")
            source = calc.get("market_price_source")
            if source:
                line2.append(f"price_source={source}")
            basis = calc.get("entry_basis_source")
            if isinstance(basis, str) and basis:
                line2.append(f"basis_source={basis}")
            JournalDiagnostics._append_float(line2, label="stop", value=calc.get("stop_level"), fmt=".2f")
            JournalDiagnostics._append_float(line2, label="take_profit", value=calc.get("profit_level"), fmt=".2f")
            JournalDiagnostics._append_float(line2, label="trigger_ref", value=calc.get("exec_hit_ref"), fmt=".2f")

        def _append_health(label: str, raw: object) -> None:
            if not isinstance(raw, dict):
                return
            duration = str(raw.get("duration_str") or "").strip()
            if duration:
                line2.append(f"{label}_dur={duration}")
            last_ts = raw.get("last_bar_ts")
            if isinstance(last_ts, str) and len(last_ts) >= 16:
                line2.append(f"{label}_last={last_ts[11:16]}")
            if bool(raw.get("timeout_fallback_used")):
                from_duration = str(raw.get("timeout_fallback_from_duration") or "").strip()
                if from_duration:
                    line2.append(f"{label}_fallback={from_duration}")

        _append_health("sig", detail.get("bar_health"))
        _append_health("reg", detail.get("regime_bar_health"))
        _append_health("reg2", detail.get("regime2_bar_health"))
        _append_health("seed", detail.get("regime_router_seed_health"))

        lifecycle = detail.get("spot_lifecycle")
        if isinstance(lifecycle, dict):
            fill_mode_lifecycle = str(lifecycle.get("fill_mode") or "").strip()
            if fill_mode_lifecycle and fill_mode_lifecycle != str(fill_mode or ""):
                line2.append(f"fill_mode={fill_mode_lifecycle}")
            trace = lifecycle.get("trace")
            if isinstance(trace, dict):
                stage = str(trace.get("stage") or "").strip()
                path = str(trace.get("path") or "").strip()
                if stage or path:
                    lifecycle_desc = f"{stage}/{path}" if stage and path else (stage or path)
                    line2.append(f"lifecycle={lifecycle_desc}")

        active_line = JournalDiagnostics._active_knobs_line(
            meta={},
            strategy=strategy,
            filters=filters,
            max_tokens=8,
        )
        if active_line:
            line2.append(active_line)
        if line2:
            lines.append(JournalDiagnostics._join_tokens(line2))

        if isinstance(checks, dict) and checks:
            lines.append(f"[dim]filter_map[/] {JournalDiagnostics._filter_check_vector(checks)}")
            if isinstance(failed, list):
                failed_list = [str(v) for v in failed if str(v)]
            else:
                failed_list = [str(name) for name, ok in checks.items() if not bool(ok)]
            fail_reasons = JournalDiagnostics._filter_fail_reasons(
                detail=detail,
                filters=filters,
                failed_filters=failed_list,
            )
            if fail_reasons:
                lines.append("[dim]filter_fail[/] " + " ; ".join(fail_reasons))

        entry_ctx = detail.get("entry_ctx")
        if isinstance(entry_ctx, dict):
            entry_ctx_line = JournalDiagnostics._entry_ctx_line(entry_ctx)
            if entry_ctx_line:
                lines.append(entry_ctx_line)

        next_open_ctx = detail.get("next_open_ctx")
        if isinstance(next_open_ctx, dict):
            next_open_line = JournalDiagnostics._next_open_ctx_line(next_open_ctx)
            if next_open_line:
                lines.append(next_open_line)

        if isinstance(lifecycle, dict):
            trace = lifecycle.get("trace")
            if isinstance(trace, dict):
                graph_entry = trace.get("graph_entry")
                if isinstance(graph_entry, dict):
                    graph_line = JournalDiagnostics._graph_entry_line(graph_entry)
                    if graph_line:
                        lines.append(graph_line)

        return "\n".join(line for line in lines if line)
