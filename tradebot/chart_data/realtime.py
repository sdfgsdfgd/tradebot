"""Render-neutral realtime series used by live views and replay."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from time import monotonic


def resample(points: list[float], width: int) -> list[float]:
    width = max(width, 1)
    if not points:
        return []
    if len(points) == 1 or width == 1:
        return [points[-1]] * width
    if len(points) == width:
        return list(points)
    step = float(len(points) - 1) / float(width - 1)
    out: list[float] = []
    for idx in range(width):
        pos = step * float(idx)
        lo = int(pos)
        hi = min(lo + 1, len(points) - 1)
        frac = pos - float(lo)
        out.append(points[lo] + (points[hi] - points[lo]) * frac)
    return out


def tape_series(
    tape: deque[tuple[float, float]], *, width: int, start: float, end: float
) -> list[float]:
    width = max(width, 1)
    if not tape:
        return []
    points = list(tape)
    start_idx = 0
    while start_idx < len(points) and points[start_idx][0] < start:
        start_idx += 1
    window = [points[start_idx - 1], *points[start_idx:]] if start_idx > 0 else points
    if not window:
        return []
    span = max(float(end - start), 1e-9)
    step = span / float(width)
    cursor = 0
    out: list[float] = []
    for idx in range(width):
        target = float(start) + (step * float(idx + 1))
        while cursor + 1 < len(window) and float(window[cursor + 1][0]) <= target:
            cursor += 1
        t0, v0 = window[cursor]
        if cursor + 1 < len(window):
            t1, v1 = window[cursor + 1]
            if t1 > t0 and t0 <= target <= t1:
                frac = (target - t0) / (t1 - t0)
                out.append(float(v0) + ((float(v1) - float(v0)) * frac))
                continue
        out.append(float(v0))
    return out


def _tape_bins(
    tape: deque[tuple[float, float]],
    *,
    width: int,
    start: float,
    end: float,
    maximum: bool,
) -> list[float]:
    width = max(width, 1)
    if not tape:
        return []
    span = max(float(end - start), 1e-9)
    out = [0.0] * width
    has_data = False
    for ts, value in tape:
        if ts < start or ts > end:
            continue
        ratio = (float(ts) - float(start)) / span
        idx = max(0, min(width - 1, int(ratio * float(width))))
        out[idx] = max(out[idx], float(value)) if maximum else out[idx] + float(value)
        has_data = True
    return out if has_data else []


def tape_bin_max(
    tape: deque[tuple[float, float]], *, width: int, start: float, end: float
) -> list[float]:
    return _tape_bins(tape, width=width, start=start, end=end, maximum=True)


def tape_bin_sum(
    tape: deque[tuple[float, float]], *, width: int, start: float, end: float
) -> list[float]:
    return _tape_bins(tape, width=width, start=start, end=end, maximum=False)


class RealtimeChartData:
    """Own one instrument's bounded realtime samples and flow inference state."""

    def __init__(self, *, window_sec: float, retention_sec: float) -> None:
        self.window_sec = float(window_sec)
        self.retention_sec = max(float(retention_sec), self.window_sec)
        self.mid_samples: deque[float] = deque(maxlen=240)
        self.mid_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self.spread_samples: deque[float] = deque(maxlen=96)
        self.size_samples: deque[float] = deque(maxlen=96)
        self.size_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self.volume_flow_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self.pnl_samples: deque[float] = deque(maxlen=96)
        self.slip_proxy_samples: deque[float] = deque(maxlen=96)
        self.imbalance_samples: deque[float] = deque(maxlen=240)
        self.vol_burst_samples: deque[float] = deque(maxlen=240)
        self.imbalance_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self.vol_burst_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self._cumulative_volume_attr: str | None = None
        self._cumulative_volume: float | None = None
        self._fallback_size: float | None = None
        self._fallback_price: float | None = None
        self._flow_price: float | None = None

    def window_bounds(self, *, now: float | None = None) -> tuple[float, float]:
        end = monotonic() if now is None else float(now)
        return end - self.window_sec, end

    def trim(self, now: float) -> None:
        cutoff = float(now) - self.retention_sec
        for tape in (
            self.mid_tape,
            self.size_tape,
            self.volume_flow_tape,
            self.imbalance_tape,
            self.vol_burst_tape,
        ):
            while tape and tape[0][0] < cutoff:
                tape.popleft()

    def record_mid(self, mid: float, *, epsilon: float, ts: float | None = None) -> None:
        now = monotonic() if ts is None else float(ts)
        value = float(mid)
        if self.mid_tape and abs(value - float(self.mid_tape[-1][1])) <= max(epsilon, 1e-9):
            self.mid_tape[-1] = (now, float(self.mid_tape[-1][1]))
        else:
            self.mid_samples.append(value)
            self.mid_tape.append((now, value))
        self.trim(now)

    def record_size(self, size: float | None, *, ts: float | None = None) -> None:
        if size is None or size < 0:
            return
        now = monotonic() if ts is None else float(ts)
        value = float(size)
        self.size_samples.append(value)
        self.size_tape.append((now, value))
        self.trim(now)

    def record_flow(self, flow: float | None, *, ts: float | None = None) -> None:
        if flow is None or abs(float(flow)) <= 1e-12:
            return
        now = monotonic() if ts is None else float(ts)
        self.volume_flow_tape.append((now, float(flow)))
        self.trim(now)

    def record_aurora(
        self,
        *,
        imbalance: float | None,
        vol_burst: float | None,
        ts: float | None = None,
    ) -> None:
        now = monotonic() if ts is None else float(ts)
        if imbalance is not None:
            value = max(min(float(imbalance), 1.0), -1.0)
            self.imbalance_samples.append(value)
            self.imbalance_tape.append((now, value))
        if vol_burst is not None and vol_burst >= 0:
            value = float(vol_burst)
            self.vol_burst_samples.append(value)
            self.vol_burst_tape.append((now, value))
        self.trim(now)

    def record_market(
        self,
        *,
        mid: float | None,
        mid_epsilon: float,
        spread: float | None,
        size: float | None,
        pnl: float | None,
        slip_proxy: float | None,
        imbalance: float | None,
        vol_burst: float | None,
        ts: float | None = None,
    ) -> None:
        now = monotonic() if ts is None else float(ts)
        if mid is not None:
            self.record_mid(mid, epsilon=mid_epsilon, ts=now)
        if spread is not None and spread >= 0:
            self.spread_samples.append(float(spread))
        self.record_size(size, ts=now)
        if pnl is not None:
            self.pnl_samples.append(float(pnl))
        if slip_proxy is not None and slip_proxy >= 0:
            self.slip_proxy_samples.append(float(slip_proxy))
        self.record_aurora(imbalance=imbalance, vol_burst=vol_burst, ts=now)

    def cumulative_volume_delta(self, volumes: Mapping[str, float | None]) -> float | None:
        if self._cumulative_volume_attr:
            value = volumes.get(self._cumulative_volume_attr)
            if value is None or value < 0:
                self._cumulative_volume_attr = None
                self._cumulative_volume = None
                return None
            previous = self._cumulative_volume
            self._cumulative_volume = float(value)
            if previous is None:
                return None
            delta = float(value) - float(previous)
            return delta if delta >= 0 else None
        for name in ("rtTradeVolume", "rtVolume", "volume"):
            value = volumes.get(name)
            if value is not None and value >= 0:
                self._cumulative_volume_attr = name
                self._cumulative_volume = float(value)
                break
        return None

    def fallback_volume_delta(
        self, *, last_size: float | None, last_price: float | None
    ) -> float | None:
        if last_size is None or last_size <= 0:
            return None
        previous_size = self._fallback_size
        previous_price = self._fallback_price
        self._fallback_size = float(last_size)
        self._fallback_price = float(last_price) if last_price is not None else None
        if previous_size is None:
            return None
        price_changed = (
            last_price is not None
            and previous_price is not None
            and abs(float(last_price) - float(previous_price)) > 1e-9
        )
        if abs(float(last_size) - float(previous_size)) <= 1e-9 and not price_changed:
            return None
        return float(last_size)

    def flow_direction(
        self, *, price: float | None, imbalance: float | None, epsilon: float
    ) -> float:
        if price is not None:
            current = float(price)
            previous = self._flow_price
            self._flow_price = current
            if previous is not None:
                delta = current - float(previous)
                if delta > max(epsilon, 1e-9):
                    return 1.0
                if delta < -max(epsilon, 1e-9):
                    return -1.0
        if imbalance is not None:
            return 1.0 if float(imbalance) >= 0 else -1.0
        return 1.0
