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
            state = str(sig.get("state") or "")
            entry_dir = str(sig.get("entry_dir") or "")
            regime_dir = str(sig.get("regime_dir") or "")
            cross_up = bool(sig.get("cross_up"))
            cross_down = bool(sig.get("cross_down"))
            parts = [f"state={state}", f"entry={entry_dir}"]
            if cross_up:
                parts.append("cross=up")
            elif cross_down:
                parts.append("cross=down")
            if regime_dir:
                parts.append(f"regime={regime_dir}")
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
            if detail.get("direction") in ("up", "down"):
                parts.append(f"dir={detail.get('direction')}")
            mode = detail.get("mode")
            if mode:
                parts.append(f"mode={mode}")
            why = detail.get("reason")
            if why and isinstance(why, str) and why:
                parts.append(f"why={why}")
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
