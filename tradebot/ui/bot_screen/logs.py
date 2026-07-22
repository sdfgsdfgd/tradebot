"""Bot journal filtering, rows, and viewport behavior."""

from __future__ import annotations

from datetime import datetime

from ...time_utils import to_et as _to_et_shared


class BotLogsMixin:
    @staticmethod
    def _log_entry_identity(entry: dict) -> tuple[str, ...]:
        raw = entry.get("_identity")
        if isinstance(raw, tuple) and raw:
            return tuple(str(v) for v in raw)
        return (
            str(entry.get("instance_id") or ""),
            str(entry.get("symbol") or ""),
            str(entry.get("event") or ""),
            str(entry.get("reason") or ""),
            str(entry.get("msg") or ""),
        )

    def _log_entry_visible(self, entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        inst_id = str(entry.get("instance_id") or "")
        active_ids = {str(int(i.instance_id)) for i in self._instances}
        if inst_id and inst_id not in active_ids:
            return False
        scope = self._scope_instance_id()
        if scope is not None and not self._scope_all and inst_id and inst_id != str(int(scope)):
            return False
        return True

    def _logs_at_tail(self) -> bool:
        if not getattr(self, "_logs_table", None):
            return True
        if not self._log_rows:
            return True
        try:
            return bool(getattr(self._logs_table, "is_vertical_scroll_end", False))
        except Exception:
            pass
        return self._logs_table.cursor_coordinate.row >= (len(self._log_rows) - 1)

    @staticmethod
    def _log_row_cells(entry: dict) -> tuple[str, str, str, str, str, str]:
        ts = entry.get("ts_et")
        repeat = int(entry.get("_repeat") or 1)
        repeat_from_ts = entry.get("_repeat_from_ts")
        msg = str(entry.get("msg") or "")
        if repeat > 1:
            repeat_from = (
                _to_et_shared(repeat_from_ts, naive_ts_mode="et", default_naive_ts_mode="et").strftime("%H:%M:%S")
                if isinstance(repeat_from_ts, datetime)
                else ""
            )
            prefix = f"x{repeat}"
            if repeat_from:
                prefix = f"{prefix} from={repeat_from}"
            msg = f"{prefix} {msg}" if msg else prefix
        when = (
            _to_et_shared(ts, naive_ts_mode="et", default_naive_ts_mode="et").strftime("%H:%M:%S")
            if isinstance(ts, datetime)
            else ""
        )
        return (
            when,
            str(entry.get("instance_id") or ""),
            str(entry.get("symbol") or ""),
            str(entry.get("event") or ""),
            str(entry.get("reason") or ""),
            msg,
        )

    @staticmethod
    def _log_row_height(entry: dict) -> int | None:
        event = str(entry.get("event") or "")
        msg = str(entry.get("msg") or "")
        if "\n" in msg:
            return None
        if event in ("SIGNAL", "GATE"):
            return None
        return 1

    def _scroll_logs_to_tail(self) -> None:
        if not self._log_rows:
            self._sync_row_marker(self._logs_table, force=True)
            return
        self._logs_table.cursor_coordinate = (len(self._log_rows) - 1, 0)
        try:
            self._logs_table.scroll_end(animate=False)
        except Exception:
            pass
        self._set_logs_follow_tail(True)
        self._sync_row_marker(self._logs_table, force=True)

    def _append_log_entry(self, entry: dict) -> None:
        if getattr(self, "_logs_table", None) is None:
            return
        if not self._log_entry_visible(entry):
            return
        if self._active_panel == "logs":
            self._set_logs_follow_tail(self._logs_at_tail())
        self._append_log_row(entry, follow_tail=bool(self._logs_follow_tail), sync_marker=False)

    def _append_log_row(self, entry: dict, *, follow_tail: bool, sync_marker: bool) -> None:
        if getattr(self, "_logs_table", None) is None:
            return
        row = dict(entry)
        row["_repeat"] = int(row.get("_repeat") or 1)
        if row["_repeat"] > 1 and not isinstance(row.get("_repeat_from_ts"), datetime):
            if isinstance(row.get("ts_et"), datetime):
                row["_repeat_from_ts"] = row.get("ts_et")

        if self._log_rows and self._log_entry_identity(self._log_rows[-1]) == self._log_entry_identity(row):
            prev = self._log_rows[-1]
            prev_repeat = int(prev.get("_repeat") or 1)
            row_repeat = int(row.get("_repeat") or 1)
            if prev_repeat == 1 and isinstance(prev.get("ts_et"), datetime):
                prev["_repeat_from_ts"] = prev.get("ts_et")
            if not isinstance(prev.get("_repeat_from_ts"), datetime) and isinstance(row.get("_repeat_from_ts"), datetime):
                prev["_repeat_from_ts"] = row.get("_repeat_from_ts")
            prev["_repeat"] = prev_repeat + row_repeat
            prev["ts_et"] = row.get("ts_et")
            when, _inst, _sym, _event, _reason, msg = self._log_row_cells(prev)
            row_index = len(self._log_rows) - 1
            try:
                self._logs_table.update_cell_at((row_index, 0), when)
                self._logs_table.update_cell_at((row_index, 5), msg)
            except Exception:
                return
            if follow_tail:
                self._scroll_logs_to_tail()
            elif sync_marker:
                self._sync_row_marker(self._logs_table, force=True)
            return

        self._log_seq += 1
        row_key = f"log:{self._log_seq}"
        try:
            self._logs_table.add_row(
                *self._log_row_cells(row),
                key=row_key,
                height=self._log_row_height(row),
            )
        except Exception:
            return
        self._log_rows.append(row)
        self._log_row_keys.append(row_key)

        if len(self._log_rows) > int(self._log_max_rows):
            old_key = self._log_row_keys.pop(0)
            self._log_rows.pop(0)
            try:
                self._logs_table.remove_row(old_key)
            except Exception:
                self._refresh_logs_table()
                return

        if follow_tail:
            self._scroll_logs_to_tail()
        elif sync_marker:
            self._sync_row_marker(self._logs_table, force=True)

    def _refresh_logs_table(self) -> None:
        if getattr(self, "_logs_table", None) is None:
            return
        prev_row = self._logs_table.cursor_coordinate.row
        if self._active_panel == "logs":
            self._set_logs_follow_tail(self._logs_at_tail())
        follow_tail = bool(self._logs_follow_tail)

        self._logs_table.clear()
        self._log_rows = []
        self._log_row_keys = []

        active_ids = {str(int(i.instance_id)) for i in self._instances}
        scope = self._scope_instance_id()
        tail = self._log_events[-int(self._log_max_rows) :]
        compacted: list[dict[str, object]] = []
        for entry in tail:
            if not isinstance(entry, dict):
                continue
            inst_id = str(entry.get("instance_id") or "")
            if inst_id and inst_id not in active_ids:
                continue
            if scope is not None and not self._scope_all and inst_id and inst_id != str(int(scope)):
                continue
            if compacted:
                prev = compacted[-1]
                if self._log_entry_identity(prev) == self._log_entry_identity(entry):
                    prev_repeat = int(prev.get("_repeat") or 1)
                    if prev_repeat == 1 and isinstance(prev.get("ts_et"), datetime):
                        prev["_repeat_from_ts"] = prev.get("ts_et")
                    if not isinstance(prev.get("_repeat_from_ts"), datetime) and isinstance(
                        entry.get("_repeat_from_ts"), datetime
                    ):
                        prev["_repeat_from_ts"] = entry.get("_repeat_from_ts")
                    prev["_repeat"] = prev_repeat + int(entry.get("_repeat") or 1)
                    prev["ts_et"] = entry.get("ts_et")
                    continue
            row = dict(entry)
            row["_repeat"] = int(row.get("_repeat") or 1)
            if row["_repeat"] > 1 and not isinstance(row.get("_repeat_from_ts"), datetime):
                if isinstance(row.get("ts_et"), datetime):
                    row["_repeat_from_ts"] = row.get("ts_et")
            compacted.append(row)

        for entry in compacted:
            self._append_log_row(entry, follow_tail=False, sync_marker=False)

        if not self._log_rows:
            self._sync_row_marker(self._logs_table, force=True)
            return
        if follow_tail:
            self._scroll_logs_to_tail()
            return
        target_row = min(max(prev_row, 0), len(self._log_rows) - 1)
        self._logs_table.cursor_coordinate = (target_row, 0)
        self._sync_row_marker(self._logs_table, force=True)

    def _render_status(self) -> None:
        self._render_bot()
