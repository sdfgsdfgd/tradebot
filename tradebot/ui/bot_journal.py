"""Bot journal writer + in-app log tail entry builder."""

from __future__ import annotations

import atexit
import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..time_utils import UTC as _UTC
from ..time_utils import now_et as _now_et
from .bot_journal_diagnostics import JournalDiagnostics
from .bot_models import _BOT_JOURNAL_FIELDS, _BotInstance, _BotOrder

_REPEAT_COUNT_KEY = "log_repeat_count"
_REPEAT_FROM_ET_KEY = "log_repeat_from_ts_et"
_REPEAT_FROM_UTC_KEY = "log_repeat_from_ts_utc"


def order_attempt_payload(
    instance: _BotInstance,
    base: dict | None = None,
    *,
    required: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = dict(base) if isinstance(base, dict) else {}
    attempt = max(0, int(instance.order_trigger_attempt or 0))
    if required or attempt:
        payload["order_attempt"] = max(1, attempt)
    retry_reason = str(instance.order_trigger_retry_reason or "").strip()
    if retry_reason:
        payload["retry_reason"] = retry_reason
    return payload


def order_build_failure_payload(
    message: str,
    instance: _BotInstance,
    *,
    direction: str | None,
    signal_bar_ts: datetime | None,
    retry_reason: str | None = None,
    quote_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    text = str(message or "")
    lowered = text.strip().lower()
    reason = str(retry_reason or "").strip()
    if not reason:
        reason = next(
            (
                value
                for prefix, value in (
                    ("quote:", "quote_unpriced"),
                    ("signal:", "signal_unavailable"),
                    ("contract:", "contract_unavailable"),
                    ("order: atr not ready", "atr_not_ready"),
                    ("order: spot sizing returned 0 qty", "sizing_zero_qty"),
                    (
                        "order: currency conversion unavailable",
                        "currency_conversion_unavailable",
                    ),
                )
                if lowered.startswith(prefix)
            ),
            "order_build_failed",
        )
    payload = order_attempt_payload(
        instance,
        {
            "error": text,
            "direction": direction,
            "signal_bar_ts": signal_bar_ts.isoformat()
            if signal_bar_ts is not None
            else None,
            "retry_reason": reason,
        },
        required=True,
    )
    payload["retry_reason"] = reason
    if isinstance(quote_payload, dict):
        payload.update(quote_payload)
    return payload


def order_quote_failure_payload(
    *,
    ticker: object | None,
    bid: float | None,
    ask: float | None,
    last: float | None,
    mid: float | None = None,
    proxy_error: object = None,
) -> dict[str, object]:
    md_type_raw = getattr(ticker, "marketDataType", None) if ticker is not None else None
    try:
        md_type = int(md_type_raw) if md_type_raw is not None else None
    except (TypeError, ValueError):
        md_type = None
    age_ms = None
    ticker_ts = getattr(ticker, "time", None) if ticker is not None else None
    if isinstance(ticker_ts, datetime):
        now_ts = (
            datetime.now(tz=ticker_ts.tzinfo)
            if ticker_ts.tzinfo is not None
            else _now_et().replace(tzinfo=None)
        )
        try:
            age_ms = max(0, int((now_ts - ticker_ts).total_seconds() * 1000.0))
        except Exception:
            age_ms = None
    return {
        "quote": {
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "last": float(last) if last is not None else None,
            "mid": float(mid) if mid is not None else None,
            "market_data_type": md_type,
            "live": md_type in (1, 2),
            "delayed": md_type in (3, 4),
            "frozen": md_type in (2, 4),
            "md_ok": any(
                value is not None and float(value) > 0
                for value in (bid, ask, last)
            ),
            "ticker_age_ms": age_ms,
            "proxy_error": proxy_error,
        }
    }


class BotJournal:
    def __init__(self, out_dir: Path) -> None:
        self._lock = threading.Lock()
        self._pending_row: dict[str, object] | None = None
        self._pending_identity: tuple[str, ...] | None = None
        self._pending_repeat_count: int = 0
        self._pending_repeat_from_et: str = ""
        self._pending_repeat_from_utc: str = ""
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._path: Path | None = None
            return
        started_at = _now_et()
        self._path = out_dir / f"bot_journal_{started_at:%Y%m%d_%H%M%S_ET}.csv"
        atexit.register(self.close)

    @property
    def path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        try:
            with self._lock:
                self._flush_pending_locked()
        except Exception:
            return

    @staticmethod
    def _row_identity(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(
            str(row.get(field) or "")
            for field in _BOT_JOURNAL_FIELDS
            if field not in ("ts_et", "ts_utc")
        )

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
        now_et = _now_et()
        now_utc = datetime.now(tz=_UTC)

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

        identity = self._row_identity(row)
        with self._lock:
            self._queue_row_locked(row=row, identity=identity)

        entry = self._in_app_entry(
            now_et=now_et,
            event=event,
            reason=reason,
            instance_id=str(row.get("instance_id") or ""),
            symbol=str(row.get("symbol") or ""),
            extra=extra,
            detail=data if isinstance(data, dict) else {},
        )
        entry["_identity"] = identity
        return entry

    def _queue_row_locked(self, *, row: dict[str, object], identity: tuple[str, ...]) -> None:
        if self._path is None:
            return
        if self._pending_row is None:
            self._pending_row = dict(row)
            self._pending_identity = identity
            self._pending_repeat_count = 1
            self._pending_repeat_from_et = str(row.get("ts_et") or "")
            self._pending_repeat_from_utc = str(row.get("ts_utc") or "")
            return
        if self._pending_identity == identity:
            self._pending_repeat_count = int(self._pending_repeat_count or 0) + 1
            self._pending_row["ts_et"] = row.get("ts_et")
            self._pending_row["ts_utc"] = row.get("ts_utc")
            return
        self._flush_pending_locked()
        self._pending_row = dict(row)
        self._pending_identity = identity
        self._pending_repeat_count = 1
        self._pending_repeat_from_et = str(row.get("ts_et") or "")
        self._pending_repeat_from_utc = str(row.get("ts_utc") or "")

    def _flush_pending_locked(self) -> None:
        row = dict(self._pending_row) if isinstance(self._pending_row, dict) else None
        repeat_count = int(self._pending_repeat_count or 0)
        repeat_from_et = str(self._pending_repeat_from_et or "")
        repeat_from_utc = str(self._pending_repeat_from_utc or "")
        self._pending_row = None
        self._pending_identity = None
        self._pending_repeat_count = 0
        self._pending_repeat_from_et = ""
        self._pending_repeat_from_utc = ""
        if row is None:
            return
        if repeat_count > 1:
            payload: dict[str, object] = {}
            data_json = str(row.get("data_json") or "")
            if data_json:
                try:
                    parsed = json.loads(data_json)
                    if isinstance(parsed, dict):
                        payload = parsed
                except json.JSONDecodeError:
                    payload = {}
            payload[_REPEAT_COUNT_KEY] = int(repeat_count)
            payload[_REPEAT_FROM_ET_KEY] = repeat_from_et
            payload[_REPEAT_FROM_UTC_KEY] = repeat_from_utc
            row["data_json"] = json.dumps(payload, sort_keys=True, default=str)
        self._append_row_locked(row)

    def _append_row_locked(self, row: dict[str, object]) -> None:
        path = self._path
        if path is None:
            return
        try:
            try:
                is_new = (not path.exists()) or path.stat().st_size == 0
            except Exception:
                is_new = True
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
            msg = JournalDiagnostics._signal_message(detail=detail, extra=extra)
        elif event == "CROSS":
            close = detail.get("close")
            try:
                close_f = float(close) if close is not None else None
            except (TypeError, ValueError):
                close_f = None
            if close_f is not None:
                msg = f"close={close_f:.2f}"
        elif event == "GATE":
            msg = JournalDiagnostics._gate_message(detail=detail, extra=extra)
        elif event.startswith("ORDER") or event in ("SENDING", "SENT", "CANCEL_SENT", "CANCEL_ERROR"):
            parts: list[str] = []
            error = str(extra.get("error") or detail.get("error") or "").strip()
            if error:
                parts.append(error)
            order_journal = extra.get("order_journal") if isinstance(extra.get("order_journal"), dict) else {}
            attempt = detail.get("order_attempt")
            if attempt is None:
                attempt = order_journal.get("order_attempt")
            try:
                if attempt is not None and int(attempt) > 0:
                    parts.append(f"attempt={int(attempt)}")
            except (TypeError, ValueError):
                pass
            retry_reason = str(detail.get("retry_reason") or order_journal.get("retry_reason") or "").strip()
            if retry_reason:
                parts.append(f"retry={retry_reason}")
            quote = detail.get("quote") if isinstance(detail.get("quote"), dict) else None
            if quote:
                for key in ("bid", "ask", "last"):
                    value = quote.get(key)
                    try:
                        parts.append(f"{key}={float(value):.2f}" if value is not None else f"{key}=-")
                    except (TypeError, ValueError):
                        parts.append(f"{key}=-")
                for key in ("md_ok", "live", "delayed", "frozen"):
                    if key in quote:
                        parts.append(f"{key}={'1' if bool(quote.get(key)) else '0'}")
                age_ms = quote.get("ticker_age_ms")
                try:
                    if age_ms is not None:
                        parts.append(f"age_ms={int(age_ms)}")
                except (TypeError, ValueError):
                    pass
                proxy_error = str(quote.get("proxy_error") or "").strip()
                if proxy_error:
                    parts.append(f"proxy={proxy_error[:48]}")
            spot_decision = (
                order_journal.get("spot_decision")
                if isinstance(order_journal.get("spot_decision"), dict)
                else (
                    detail.get("spot_decision")
                    if isinstance(detail.get("spot_decision"), dict)
                    else {}
                )
            )
            spot_intent = (
                order_journal.get("spot_intent")
                if isinstance(order_journal.get("spot_intent"), dict)
                else (
                    detail.get("spot_intent")
                    if isinstance(detail.get("spot_intent"), dict)
                    else {}
                )
            )
            size_funnel = (
                order_journal.get("size_funnel")
                if isinstance(order_journal.get("size_funnel"), dict)
                else (
                    detail.get("size_funnel")
                    if isinstance(detail.get("size_funnel"), dict)
                    else {}
                )
            )
            signed_qty_final = JournalDiagnostics._int_or_none(size_funnel.get("signed_qty_final"))
            if signed_qty_final is None:
                signed_qty_final = JournalDiagnostics._int_or_none(spot_decision.get("signed_qty_final"))
            signed_qty_after_branch = JournalDiagnostics._int_or_none(size_funnel.get("signed_qty_after_branch"))
            if signed_qty_after_branch is None:
                signed_qty_after_branch = JournalDiagnostics._int_or_none(spot_decision.get("signed_qty_after_branch"))
            if signed_qty_after_branch is None:
                signed_qty_after_branch = signed_qty_final
            resize_target_qty = JournalDiagnostics._int_or_none(size_funnel.get("resize_target_qty"))
            if resize_target_qty is None:
                resize_target_qty = JournalDiagnostics._int_or_none(spot_intent.get("target_qty"))
            intent_qty = JournalDiagnostics._int_or_none(size_funnel.get("intent_qty"))
            if intent_qty is None:
                intent_qty = JournalDiagnostics._int_or_none(spot_intent.get("order_qty"))
            if (
                signed_qty_final is not None
                and signed_qty_after_branch is not None
                and resize_target_qty is not None
                and intent_qty is not None
            ):
                parts.append(
                    "size="
                    f"{int(signed_qty_final)}→{int(signed_qty_after_branch)}→"
                    f"{int(resize_target_qty)}→{int(intent_qty)}"
                )
            short_mult_final = JournalDiagnostics._float_or_none(spot_decision.get("short_mult_final"))
            if short_mult_final is not None:
                parts.append(f"short_mult={float(short_mult_final):.4f}")
            shock_short_factor = JournalDiagnostics._float_or_none(spot_decision.get("shock_short_factor"))
            if shock_short_factor is not None:
                parts.append(f"shock_short_factor={float(shock_short_factor):.2f}")
            shock_boost_applied = spot_decision.get("shock_short_boost_applied")
            if isinstance(shock_boost_applied, bool):
                parts.append(f"shock_short_boost={'1' if shock_boost_applied else '0'}")
            shock_boost_reason = str(spot_decision.get("shock_short_boost_gate_reason") or "").strip()
            if shock_boost_reason:
                parts.append(f"shock_short_reason={shock_boost_reason}")
            shock_prearm_applied = spot_decision.get("shock_prearm_applied")
            if isinstance(shock_prearm_applied, bool):
                parts.append(f"shock_prearm={'1' if shock_prearm_applied else '0'}")
            shock_prearm_reason = str(spot_decision.get("shock_prearm_reason") or "").strip()
            if shock_prearm_reason:
                parts.append(f"shock_prearm_reason={shock_prearm_reason}")
            ramp_mult = JournalDiagnostics._float_or_none(spot_decision.get("shock_ramp_risk_mult"))
            ramp_floor_frac = JournalDiagnostics._float_or_none(spot_decision.get("shock_ramp_cap_floor_frac"))
            ramp_applied = spot_decision.get("shock_ramp_applied")
            if (
                isinstance(ramp_applied, bool)
                or (ramp_mult is not None and float(ramp_mult) > 1.01)
                or (ramp_floor_frac is not None and float(ramp_floor_frac) > 0.01)
            ):
                ramp_dir = str(spot_decision.get("shock_ramp_dir") or "").strip()
                ramp_phase = str(spot_decision.get("shock_ramp_phase") or "").strip()
                ramp_intensity = JournalDiagnostics._float_or_none(spot_decision.get("shock_ramp_intensity"))
                ramp_reason = str(spot_decision.get("shock_ramp_reason") or "").strip()
                if ramp_dir or ramp_phase:
                    parts.append(f"ramp={ramp_dir or '?'}:{ramp_phase or '?'}")
                if ramp_mult is not None:
                    parts.append(f"ramp_mult={float(ramp_mult):.2f}")
                if ramp_intensity is not None:
                    parts.append(f"ramp_i={float(ramp_intensity):.2f}")
                if ramp_floor_frac is not None and float(ramp_floor_frac) > 0:
                    parts.append(f"ramp_floor={float(ramp_floor_frac):.2f}")
                if ramp_reason:
                    parts.append(f"ramp_reason={ramp_reason}")
            liq_boost_applied = spot_decision.get("liq_boost_applied")
            if isinstance(liq_boost_applied, bool):
                parts.append(f"liq_boost={'1' if liq_boost_applied else '0'}")
            liq_boost_mult = JournalDiagnostics._float_or_none(spot_decision.get("liq_boost_mult"))
            if liq_boost_mult is not None:
                parts.append(f"liq_mult={float(liq_boost_mult):.2f}")
            liq_boost_score = JournalDiagnostics._float_or_none(spot_decision.get("liq_boost_score"))
            if liq_boost_score is not None:
                parts.append(f"liq_score={float(liq_boost_score):+.2f}")
            liq_boost_reason = str(spot_decision.get("liq_boost_reason") or "").strip()
            if liq_boost_reason:
                parts.append(f"liq_reason={liq_boost_reason}")
            msg = " ".join(parts)
        return {
            "ts_et": now_et,
            "event": str(event or ""),
            "reason": str(reason or ""),
            "instance_id": instance_id,
            "symbol": symbol,
            "msg": msg,
        }
