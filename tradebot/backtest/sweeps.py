"""Shared generic helpers for backtest sweeps (CLI tools)."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# region Time / Formatting
def utc_now_iso_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds_int = int(seconds)
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# endregion

# region IO
def write_json(path: Path, payload: object, *, indent: int = 2, sort_keys: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=sort_keys))


# endregion

# region Concurrency / Progress
def normalize_jobs(jobs: int) -> int:
    max_jobs = max(os.cpu_count() or 1, 1)
    jobs = int(jobs)
    if jobs <= 0:
        return max_jobs
    return min(jobs, max_jobs)


def count_total_combos(grid: dict) -> int:
    """Total combinations for the options leaderboard grid."""
    base = (
        len(grid["dte"])
        * len(grid["moneyness_pct"])
        * len(grid["profit_target"])
        * len(grid["stop_loss"])
        * len(grid["ema_preset"])
        * len(grid["ema_entry_mode"])
    )
    per_base = 0
    for flip in grid["exit_on_signal_flip"]:
        per_base += (
            len(grid["flip_exit_min_hold_bars"]) * len(grid["flip_exit_only_if_profit"])
            if flip
            else 1
        )
    return base * per_base


@dataclass
class Progress:
    total: int
    interval_sec: float
    groups: int

    def __post_init__(self) -> None:
        self._total = max(int(self.total), 1)
        self._interval_sec = float(self.interval_sec)
        self._groups = int(self.groups)

        self._done = 0
        self._start = time.monotonic()
        self._last_print = self._start

        self._group_idx = 0
        self._group_name = ""
        self._group_total = 1
        self._group_done = 0
        self._last_line_len = 0

    def start_group(self, group_idx: int, group_name: str, *, total: int) -> None:
        self._group_idx = int(group_idx)
        self._group_name = str(group_name)
        self._group_total = max(int(total), 1)
        self._group_done = 0
        self._print(force=True, newline=True)

    def finish_group(self) -> None:
        self._print(force=True, newline=True)

    def advance(self, n: int = 1) -> None:
        self._done += int(n)
        self._group_done += int(n)
        if self._interval_sec <= 0:
            return
        now = time.monotonic()
        if now - self._last_print >= self._interval_sec:
            self._print(force=True, newline=False)
            self._last_print = now

    def _print(self, *, force: bool, newline: bool) -> None:
        if not force:
            return
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = self._done / elapsed
        remaining = self._total - self._done
        eta = remaining / rate if rate > 0 else None

        overall_pct = (self._done / self._total) * 100.0
        group_pct = (self._group_done / self._group_total) * 100.0

        eta_s = fmt_duration(eta) if eta is not None else "--:--:--"
        line = (
            f"[{self._group_idx}/{self._groups}] {self._group_name} | "
            f"group {group_pct:5.1f}% ({self._group_done}/{self._group_total}) | "
            f"overall {overall_pct:5.1f}% ({self._done}/{self._total}) | "
            f"{rate:5.2f} combos/s | ETA {eta_s}"
        )

        if sys.stdout.isatty() and not newline:
            pad = max(0, self._last_line_len - len(line))
            sys.stdout.write("\r" + line + (" " * pad))
            sys.stdout.flush()
            self._last_line_len = len(line)
            return

        end = "\n" if newline or not sys.stdout.isatty() else ""
        print(line, end=end, flush=True)
        self._last_line_len = 0


# endregion
