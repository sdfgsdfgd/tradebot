"""Bot journal writer + in-app log tail entry builder."""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .bot_models import _BOT_JOURNAL_FIELDS, _BotInstance, _BotOrder


class BotJournal:
    def __init__(self, out_dir: Path) -> None:
        self._lock = threading.Lock()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._path: Path | None = None
            return
        started_at = datetime.now(tz=ZoneInfo("America/New_York"))
        self._path = out_dir / f"bot_journal_{started_at:%Y%m%d_%H%M%S_ET}.csv"

    @property
    def path(self) -> Path | None:
        return self._path

    def write(
        self,
        *,
        event: str,
        instance: _BotInstance | None,
        order: _BotOrder | None,
        reason: str | None,
        data: dict | None,
        strategy_instrument: Callable[[dict], str],
    ) -> dict[str, object]:
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))
        now_utc = datetime.now(tz=timezone.utc)

        row: dict[str, object] = {k: "" for k in _BOT_JOURNAL_FIELDS}
        row["ts_et"] = now_et.isoformat()
        row["ts_utc"] = now_utc.isoformat()
        row["event"] = str(event or "")
        row["reason"] = str(reason) if reason else ""

        if instance is not None:
            row["instance_id"] = str(int(instance.instance_id))
            row["group"] = str(instance.group or "")
            row["symbol"] = str(instance.symbol or "")
            try:
                row["instrument"] = str(strategy_instrument(instance.strategy or {}))
            except Exception:
                row["instrument"] = ""
        if order is not None:
            row["instance_id"] = str(int(order.instance_id))
            row["action"] = str(order.action or "")
            row["qty"] = str(int(order.quantity or 0))
            row["limit_price"] = f"{float(order.limit_price):.6f}"
            row["status"] = str(order.status or "")
            row["order_id"] = str(int(order.order_id)) if order.order_id else ""

        extra: dict[str, object] = {}
        if instance is not None:
            extra["strategy"] = instance.strategy
            extra["filters"] = instance.filters
        if order is not None:
            extra["intent"] = order.intent
            extra["direction"] = order.direction
            extra["signal_bar_ts"] = order.signal_bar_ts.isoformat() if order.signal_bar_ts else None
            extra["order_journal"] = order.journal
            extra["error"] = order.error
            extra["exec_mode"] = order.exec_mode
        if isinstance(data, dict) and data:
            extra.update(data)
        row["data_json"] = json.dumps(extra, sort_keys=True, default=str) if extra else ""

        self._append_row(row)
        return self._in_app_entry(
            now_et=now_et,
            event=event,
            reason=reason,
            instance_id=str(row.get("instance_id") or ""),
            symbol=str(row.get("symbol") or ""),
            extra=extra,
            detail=data if isinstance(data, dict) else {},
        )

    def _append_row(self, row: dict[str, object]) -> None:
        path = self._path
        if path is None:
            return
        try:
            try:
                is_new = (not path.exists()) or path.stat().st_size == 0
            except Exception:
                is_new = True
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=_BOT_JOURNAL_FIELDS)
                    if is_new:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception:
            return

    @staticmethod
    def _in_app_entry(
        *,
        now_et: datetime,
        event: str,
        reason: str | None,
        instance_id: str,
        symbol: str,
        extra: dict,
        detail: dict,
    ) -> dict[str, object]:
        msg = ""
        if event == "SIGNAL" and isinstance(detail.get("signal"), dict):
            sig = detail.get("signal") or {}
            meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
            state = str(sig.get("state") or "")
            entry_dir = str(sig.get("entry_dir") or "")
            regime_dir = str(sig.get("regime_dir") or "")
            regime_ready = bool(sig.get("regime_ready"))
            cross_up = bool(sig.get("cross_up"))
            cross_down = bool(sig.get("cross_down"))
            parts = [f"state={state}", f"entry={entry_dir}"]
            if cross_up:
                parts.append("cross=up")
            elif cross_down:
                parts.append("cross=down")
            if regime_dir:
                parts.append(f"regime={regime_dir}")
            if not regime_ready:
                parts.append("regime_ready=0")
            spread = sig.get("ema_spread_pct")
            slope = sig.get("ema_slope_pct")
            try:
                if spread is not None:
                    parts.append(f"spr={float(spread):+.3f}%")
            except (TypeError, ValueError):
                pass
            try:
                if slope is not None:
                    parts.append(f"slp={float(slope):+.3f}%")
            except (TypeError, ValueError):
                pass
            shock = detail.get("shock")
            if isinstance(shock, dict):
                shock_on = shock.get("shock")
                shock_dir = str(shock.get("dir") or "")
                if bool(shock_on):
                    parts.append(f"shock={shock_dir or 'on'}")
                elif shock_on is False:
                    parts.append("shock=off")
                atr_pct = shock.get("atr_pct")
                try:
                    if atr_pct is not None:
                        parts.append(f"atr={float(atr_pct):.1f}%")
                except (TypeError, ValueError):
                    pass
            risk = detail.get("risk")
            if isinstance(risk, dict):
                risk_flags: list[str] = []
                if bool(risk.get("riskoff")):
                    risk_flags.append("off")
                if bool(risk.get("riskpanic")):
                    risk_flags.append("panic")
                if bool(risk.get("riskpop")):
                    risk_flags.append("pop")
                if risk_flags:
                    parts.append(f"risk={','.join(risk_flags)}")
            rv = detail.get("rv")
            try:
                if rv is not None:
                    parts.append(f"rv={float(rv):.2f}")
            except (TypeError, ValueError):
                pass
            bars_in_day = detail.get("bars_in_day")
            try:
                if bars_in_day is not None:
                    parts.append(f"bar#={int(bars_in_day)}")
            except (TypeError, ValueError):
                pass
            entry_signal = str(meta.get("entry_signal") or "")
            signal_bar_size = str(meta.get("signal_bar_size") or "")
            if entry_signal or signal_bar_size:
                parts.append(f"sig={entry_signal or '?'}@{signal_bar_size or '?'}")
            if "signal_use_rth" in meta:
                parts.append(f"rth={'1' if bool(meta.get('signal_use_rth')) else '0'}")
            regime_mode = str(meta.get("regime_mode") or "")
            regime_bar_size = str(meta.get("regime_bar_size") or signal_bar_size or "")
            if regime_mode:
                parts.append(f"bias={regime_mode}@{regime_bar_size or '?'}")
            regime2_mode = str(meta.get("regime2_mode") or "")
            if regime2_mode and regime2_mode != "off":
                regime2_bar_size = str(meta.get("regime2_bar_size") or signal_bar_size or "")
                regime2_apply = str(meta.get("regime2_apply_to") or "both")
                parts.append(f"bias2={regime2_mode}@{regime2_bar_size or '?'}:{regime2_apply}")
            exec_feed_mode = str(meta.get("spot_exec_feed_mode") or "")
            exec_bar_size = str(meta.get("spot_exec_bar_size") or "")
            if exec_feed_mode or exec_bar_size:
                parts.append(f"exec={exec_feed_mode or '?'}@{exec_bar_size or '?'}")
            msg = " ".join(p for p in parts if p and not p.endswith("="))
        elif event == "CROSS":
            close = detail.get("close")
            try:
                close_f = float(close) if close is not None else None
            except (TypeError, ValueError):
                close_f = None
            if close_f is not None:
                msg = f"close={close_f:.2f}"
        elif event == "GATE":
            parts: list[str] = []
            bar_ts = detail.get("bar_ts")
            if isinstance(bar_ts, str) and len(bar_ts) >= 16:
                parts.append(f"bar={bar_ts[11:16]}")
            if detail.get("direction") in ("up", "down"):
                parts.append(f"dir={detail.get('direction')}")
            if detail.get("entry_dir") in ("up", "down"):
                parts.append(f"entry={detail.get('entry_dir')}")
            if detail.get("regime_dir") in ("up", "down"):
                parts.append(f"regime={detail.get('regime_dir')}")
            if detail.get("shock_dir") in ("up", "down"):
                parts.append(f"shock={detail.get('shock_dir')}")
            if "cooldown_ok" in detail:
                parts.append(f"cool={'1' if bool(detail.get('cooldown_ok')) else '0'}")
            bars_in_day = detail.get("bars_in_day")
            try:
                if bars_in_day is not None:
                    parts.append(f"bar#={int(bars_in_day)}")
            except (TypeError, ValueError):
                pass
            mode = detail.get("mode")
            if mode:
                parts.append(f"mode={mode}")
            fill_mode = detail.get("fill_mode")
            if isinstance(fill_mode, str) and fill_mode:
                parts.append(f"fill={fill_mode}")
            why = detail.get("reason")
            if why and isinstance(why, str) and why:
                parts.append(f"why={why}")
            rv = detail.get("rv")
            try:
                if rv is not None:
                    parts.append(f"rv={float(rv):.2f}")
            except (TypeError, ValueError):
                pass
            state = detail.get("state")
            if state in ("up", "down"):
                parts.append(f"state={state}")
            failed = detail.get("failed_filters")
            if isinstance(failed, list) and failed:
                compact = ",".join(str(v) for v in failed if str(v))
                if compact:
                    parts.append(f"fail={compact}")
            checks = detail.get("filter_checks")
            if isinstance(checks, dict):
                bad = [str(name) for name, ok in checks.items() if not bool(ok)]
                if bad and not isinstance(failed, list):
                    parts.append(f"fail={','.join(bad)}")
                if bad:
                    check_labels = (
                        ("time", "t"),
                        ("skip_first", "sf"),
                        ("cooldown", "cd"),
                        ("shock_gate", "sh"),
                        ("permission", "perm"),
                        ("rv", "rvf"),
                        ("volume", "vol"),
                    )
                    for check_name, label in check_labels:
                        if check_name in checks:
                            parts.append(f"{label}={'1' if bool(checks.get(check_name)) else '0'}")
            due = detail.get("next_open_due")
            if isinstance(due, str) and due:
                parts.append(f"due={due[11:16]}")
            calc = detail.get("calc")
            if isinstance(calc, dict):
                session = calc.get("session")
                if isinstance(session, str) and session:
                    parts.append(f"sess={session}")
                source = calc.get("market_price_source")
                if source:
                    parts.append(f"src={source}")
                basis = calc.get("entry_basis_source")
                if isinstance(basis, str) and basis:
                    parts.append(f"basis={basis}")
                stop_level = calc.get("stop_level")
                try:
                    if stop_level is not None:
                        parts.append(f"stop={float(stop_level):.2f}")
                except (TypeError, ValueError):
                    pass
                profit_level = calc.get("profit_level")
                try:
                    if profit_level is not None:
                        parts.append(f"tp={float(profit_level):.2f}")
                except (TypeError, ValueError):
                    pass
                ref = calc.get("exec_hit_ref")
                try:
                    if ref is not None:
                        parts.append(f"hit={float(ref):.2f}")
                except (TypeError, ValueError):
                    pass
            msg = " ".join(parts)
        elif event.startswith("ORDER") or event in ("SENDING", "SENT", "CANCEL_SENT", "CANCEL_ERROR"):
            msg = str(extra.get("error") or "")
        return {
            "ts_et": now_et,
            "event": str(event or ""),
            "reason": str(reason or ""),
            "instance_id": instance_id,
            "symbol": symbol,
            "msg": msg,
        }
