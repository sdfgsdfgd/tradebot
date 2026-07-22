"""Terminal chart encoding for live position detail."""

from __future__ import annotations

from collections import deque

from ib_insync import Contract, Ticker
from rich.text import Text

from ...chart_data.realtime import (
    RealtimeChartData,
    resample,
    tape_bin_max,
    tape_bin_sum,
    tape_series,
)
from ...engines.execution import _tick_size
from ..common import _fmt_quote


class PositionChart(RealtimeChartData):
    """Owns position-detail samples and their terminal visualization."""

    _SPARK_CHARS = "▁▂▃▄▅▆▇█"
    _SPARK_LEVELS = " ▁▂▃▄▅▆▇█"
    _MOMENTUM_CHARS = "░▒▓█"
    _TREND_WINDOW_SEC = 60.0
    _TREND_RETENTION_SEC = 180.0
    _TREND_BRAILLE_X = 2
    _TREND_BRAILLE_Y = 4
    _TREND_ROWS = 5
    _AURORA_TAPE_BIAS = 0.15
    _VOL_MAGENTA_STYLES = ("#4e1e57", "#6f2380", "#922aaa", "#b534d8", "#ff3dee")
    _AURORA_PRESET_ORDER = ("calm", "normal", "feral")
    _AURORA_PRESETS = {
        "calm": {"buy_soft": 0.28, "buy_strong": 0.56, "sell_soft": -0.28, "sell_strong": -0.56, "burst_gain": 0.80},
        "normal": {"buy_soft": 0.16, "buy_strong": 0.34, "sell_soft": -0.16, "sell_strong": -0.34, "burst_gain": 1.00},
        "feral": {"buy_soft": 0.08, "buy_strong": 0.22, "sell_soft": -0.08, "sell_strong": -0.22, "burst_gain": 1.30},
    }

    def __init__(self, refresh_sec: float) -> None:
        super().__init__(
            window_sec=self._TREND_WINDOW_SEC,
            retention_sec=self._TREND_RETENTION_SEC,
        )
        self._refresh_sec = max(float(refresh_sec), 0.1)
        self._contract: Contract | object = Contract()
        self._ticker: Ticker | None = None
        self._aurora_preset = "normal"

    def bind(self, contract: Contract | object, ticker: Ticker | None) -> None:
        self._contract = contract
        self._ticker = ticker

    def cycle_aurora(self) -> str:
        try:
            index = self._AURORA_PRESET_ORDER.index(self._aurora_preset)
        except ValueError:
            index = 1
        self._aurora_preset = self._AURORA_PRESET_ORDER[
            (index + 1) % len(self._AURORA_PRESET_ORDER)
        ]
        return self._aurora_preset

    def sparkline(self, values: deque[float], width: int) -> str:
        width = max(width, 1)
        points = list(values)[-width:]
        if not points:
            return " " * width
        lo = min(points)
        hi = max(points)
        span = hi - lo
        if span <= 1e-12:
            bar = self._SPARK_CHARS[len(self._SPARK_CHARS) // 2]
            return (bar * len(points)).rjust(width)
        out: list[str] = []
        scale = len(self._SPARK_CHARS) - 1
        for value in points:
            ratio = (value - lo) / span
            idx = max(0, min(scale, int(round(ratio * scale))))
            out.append(self._SPARK_CHARS[idx])
        return "".join(out).rjust(width)

    @staticmethod
    def _trend_braille_cell(left: int, right: int, *, top: bool) -> str:
        if left <= 0 and right <= 0:
            return " "
        # Braille gives two 4-dot columns per char (left/right), ideal for 2x supersampling.
        left_order = (64, 4, 2, 1) if top else (1, 2, 4, 64)
        right_order = (128, 32, 16, 8) if top else (8, 16, 32, 128)
        bits = 0
        for idx in range(max(0, min(4, left))):
            bits |= left_order[idx]
        for idx in range(max(0, min(4, right))):
            bits |= right_order[idx]
        return chr(0x2800 + bits) if bits else " "

    def trend_rows(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> tuple[list[str], int]:
        width = max(width, 1)
        cols = width
        rows = max(3, int(self._TREND_ROWS))
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self.window_bounds()
        )
        series = tape_series(self.mid_tape, width=width, start=start, end=end)
        if not series:
            approx = max(2, int(round(self._TREND_WINDOW_SEC / max(self._refresh_sec, 0.1))))
            fallback = list(self.mid_samples)[-approx:]
            series = resample([float(value) for value in fallback], width) if fallback else []
        if not series:
            blank = " " * width
            return ([blank for _ in range(rows)], rows // 2)
        self._trend_bins = list(series)
        lo = min(series)
        hi = max(series)
        span = hi - lo
        tick = _tick_size(self._contract, self._ticker, series[-1])
        min_span = max(float(tick) * 2.0, 1e-9)
        if span < min_span:
            center = (hi + lo) * 0.5
            lo = center - (min_span * 0.5)
            hi = center + (min_span * 0.5)
            span = hi - lo
        pad = span * 0.03
        lo -= pad
        hi += pad
        span = max(hi - lo, min_span)

        hi_rows = rows * int(self._TREND_BRAILLE_Y)
        hi_cols = cols * int(self._TREND_BRAILLE_X)
        raster: list[list[bool]] = [[False] * hi_cols for _ in range(hi_rows)]
        y_points: list[int] = []
        for value in series:
            ratio = (float(value) - lo) / span
            ratio = max(0.0, min(ratio, 1.0))
            y_points.append(int(round((1.0 - ratio) * float(hi_rows - 1))))

        def draw_segment(x0: int, y0: int, x1: int, y1: int) -> None:
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x = x0
            y = y0
            while True:
                if 0 <= y < hi_rows and 0 <= x < hi_cols:
                    raster[y][x] = True
                if x == x1 and y == y1:
                    break
                e2 = err * 2
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

        for idx in range(1, cols):
            x0 = (idx - 1) * int(self._TREND_BRAILLE_X)
            x1 = idx * int(self._TREND_BRAILLE_X)
            draw_segment(x0, y_points[idx - 1], x1, y_points[idx])
        raster[y_points[-1]][hi_cols - 1] = True

        out_rows: list[str] = []
        for row_idx in range(rows):
            top = row_idx * int(self._TREND_BRAILLE_Y)
            chars: list[str] = []
            for col_idx in range(cols):
                left = col_idx * int(self._TREND_BRAILLE_X)
                bits = 0
                if raster[top + 0][left + 0]:
                    bits |= 0x01
                if raster[top + 1][left + 0]:
                    bits |= 0x02
                if raster[top + 2][left + 0]:
                    bits |= 0x04
                if raster[top + 3][left + 0]:
                    bits |= 0x40
                if raster[top + 0][left + 1]:
                    bits |= 0x08
                if raster[top + 1][left + 1]:
                    bits |= 0x10
                if raster[top + 2][left + 1]:
                    bits |= 0x20
                if raster[top + 3][left + 1]:
                    bits |= 0x80
                chars.append(chr(0x2800 + bits) if bits else " ")
            out_rows.append("".join(chars))

        now_row = max(0, min(rows - 1, int(y_points[-1] // int(self._TREND_BRAILLE_Y))))
        return (out_rows, now_row)

    @staticmethod
    def tag_price(row: Text, price: float | None, *, color: str) -> Text:
        if price is None:
            return row
        plain = row.plain
        width = len(plain)
        if width < 4:
            return row
        label = f" {_fmt_quote(price)}"
        usable = min(len(label), width - 1)
        start = width - 1 - usable
        out = row[:start]
        out.append(label[-usable:], style="bold #f8fbff")
        out.append("▐", style=f"bold {color}")
        return out

    def volume_histogram(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> Text:
        width = max(width, 1)
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self.window_bounds()
        )
        flow_events = sum(
            1 for ts, _value in self.volume_flow_tape if start <= ts <= end
        )
        density = float(flow_events) / float(width)
        micro_factor = 2
        if density >= 1.2:
            micro_factor = 3
        if density >= 2.2:
            micro_factor = 4
        if density >= 3.5:
            micro_factor = 5
        micro_width = width * micro_factor
        flow_micro = tape_bin_sum(
            self.volume_flow_tape,
            width=micro_width,
            start=start,
            end=end,
        )

        if not flow_micro:
            size_events = sum(
                1 for ts, _value in self.size_tape if start <= ts <= end
            )
            size_density = float(size_events) / float(width)
            if size_density >= 1.2:
                micro_factor = max(micro_factor, 3)
            if size_density >= 2.2:
                micro_factor = max(micro_factor, 4)
            if size_density >= 3.5:
                micro_factor = max(micro_factor, 5)
            micro_width = width * micro_factor
            size_micro = tape_series(
                self.size_tape,
                width=micro_width,
                start=start,
                end=end,
            )
            if not size_micro:
                size_micro = resample(list(self.size_samples), micro_width)
            if not size_micro:
                return Text(" " * width, style="dim")
            imbalance_micro = tape_series(
                self.imbalance_tape,
                width=micro_width,
                start=start,
                end=end,
            )
            if not imbalance_micro:
                imbalance_micro = [0.0] * micro_width
            flow_micro = []
            for size, imbalance in zip(size_micro, imbalance_micro):
                direction = 1.0 if float(imbalance) >= 0 else -1.0
                flow_micro.append(float(size) * direction)

        if len(flow_micro) != micro_width:
            flow_micro = resample([float(value) for value in flow_micro], micro_width)
        if len(flow_micro) > 2:
            smoothed = [float(flow_micro[0])]
            for idx in range(1, len(flow_micro) - 1):
                left = float(flow_micro[idx - 1])
                center = float(flow_micro[idx])
                right = float(flow_micro[idx + 1])
                blended = (left * 0.18) + (center * 0.64) + (right * 0.18)
                lo = min(left, center, right)
                hi = max(left, center, right)
                smoothed.append(max(lo, min(blended, hi)))
            smoothed.append(float(flow_micro[-1]))
            flow_micro = smoothed

        mags = sorted(abs(float(value)) for value in flow_micro if abs(float(value)) > 1e-12)
        if not mags:
            return Text(" " * width, style="dim")

        def quantile(ratio: float) -> float:
            ratio = max(0.0, min(float(ratio), 1.0))
            if len(mags) == 1:
                return mags[0]
            pos = ratio * float(len(mags) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(mags) - 1)
            frac = pos - float(lo)
            return mags[lo] + ((mags[hi] - mags[lo]) * frac)

        p25 = quantile(0.25)
        p90 = quantile(0.90)
        p98 = quantile(0.98)
        scale = max(p90, p98 * 0.72, mags[-1] * 0.26, 1e-9)
        floor_ratio = max(0.012, min((p25 / scale) * 0.60, 0.18))
        gamma = 0.84 if self._aurora_preset == "feral" else 0.90

        out = Text()
        base_split = max(1, micro_factor // 2)
        for idx in range(width):
            start_idx = idx * micro_factor
            chunk = flow_micro[start_idx : start_idx + micro_factor]
            if not chunk:
                out.append(" ", style="dim")
                continue
            if len(chunk) == 1:
                left_mag = abs(float(chunk[0]))
                right_mag = left_mag
            else:
                split = max(1, min(len(chunk) - 1, base_split))
                left_slice = chunk[:split]
                right_slice = chunk[split:]
                left_mag = sum(abs(float(value)) for value in left_slice) / float(len(left_slice))
                right_mag = sum(abs(float(value)) for value in right_slice) / float(len(right_slice))
            left_ratio = min(left_mag / scale, 1.0)
            right_ratio = min(right_mag / scale, 1.0)
            if left_ratio > 1e-12:
                left_ratio = max(left_ratio, floor_ratio)
            if right_ratio > 1e-12:
                right_ratio = max(right_ratio, floor_ratio)
            left_level = max(0, min(4, int(round(pow(left_ratio, gamma) * 4.0))))
            right_level = max(0, min(4, int(round(pow(right_ratio, gamma) * 4.0))))
            if left_level == 0 and left_ratio > 1e-12:
                left_level = 1
            if right_level == 0 and right_ratio > 1e-12:
                right_level = 1
            cell = self._trend_braille_cell(left_level, right_level, top=True)
            if cell == " ":
                out.append(" ", style="dim")
                continue
            intensity = max(left_level, right_level)
            style_idx = max(0, min(len(self._VOL_MAGENTA_STYLES) - 1, intensity))
            out.append(cell, style=self._VOL_MAGENTA_STYLES[style_idx])
        return out

    @staticmethod
    def mark_now(text: Text, *, style: str = "bold #f8fbff") -> Text:
        plain = text.plain
        if not plain:
            return text
        marked = text[:-1]
        marked.append("▏", style=style)
        return marked

    def _aurora_config(self) -> dict[str, float]:
        return self._AURORA_PRESETS.get(self._aurora_preset, self._AURORA_PRESETS["normal"])

    @staticmethod
    def _aurora_pressure(imbalance: float) -> float:
        x = max(-1.0, min(float(imbalance), 1.0))
        mag = abs(x)
        if mag <= 1e-12:
            return 0.0
        # Lift mid-range imbalance so real pressure is easier to see in color.
        boosted = pow(mag, 0.72)
        return max(-1.0, min(boosted if x >= 0 else -boosted, 1.0))

    def _aurora_drift_series(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> list[float]:
        width = max(width, 1)
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self.window_bounds()
        )
        mids = tape_series(self.mid_tape, width=width + 1, start=start, end=end)
        if len(mids) < 2:
            mids = resample(list(self.mid_samples), width + 1)
        if len(mids) < 2:
            return [0.0] * width
        deltas = [float(mids[idx] - mids[idx - 1]) for idx in range(1, len(mids))]
        mags = sorted(abs(delta) for delta in deltas if abs(delta) > 1e-12)
        if not mags:
            return [0.0] * width
        last = len(mags) - 1
        scale = max(mags[int(round(last * 0.85))], mags[-1] * 0.22, 1e-12)
        norm = [max(-1.0, min(delta / scale, 1.0)) for delta in deltas]
        alpha = 0.36
        smooth: list[float] = [norm[0]]
        for value in norm[1:]:
            prev = smooth[-1]
            smooth.append(prev + alpha * (value - prev))
        return resample(smooth, width) if smooth else [0.0] * width

    def _aurora_blended_imbalance(self, imbalance: float, drift: float) -> float:
        bias = max(0.0, min(self._AURORA_TAPE_BIAS, 1.0))
        return max(-1.0, min((float(imbalance) * (1.0 - bias)) + (float(drift) * bias), 1.0))

    def _aurora_style(self, imbalance: float, *, config: dict[str, float] | None = None) -> str:
        cfg = config or self._aurora_config()
        pressure = self._aurora_pressure(imbalance)
        buy_soft = float(cfg.get("buy_soft", 0.22))
        buy_strong = float(cfg.get("buy_strong", 0.48))
        sell_soft = float(cfg.get("sell_soft", -0.22))
        sell_strong = float(cfg.get("sell_strong", -0.48))
        if pressure >= buy_strong:
            return "#18a63f"  # lots buy pressure (dark green, brightened for terminal contrast)
        if pressure >= buy_soft:
            return "#e6d84e"  # slight buy pressure (yellow)
        if pressure <= sell_strong:
            return "red"  # lots sell pressure (red)
        if pressure <= sell_soft:
            return "#ffaf00"  # slight sell pressure (amber)
        return "#8aa0b6"

    def aurora_now_style(self) -> str:
        if self.imbalance_tape:
            imbalance_now = float(self.imbalance_tape[-1][1])
        elif self.imbalance_samples:
            imbalance_now = float(self.imbalance_samples[-1])
        else:
            return "#8aa0b6"
        drift_now = self._aurora_drift_series(1)[0]
        blended = self._aurora_blended_imbalance(imbalance_now, drift_now)
        return self._aurora_style(blended)

    def aurora_strip(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> Text:
        width = max(width, 1)
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self.window_bounds()
        )
        imbalances = tape_series(
            self.imbalance_tape,
            width=width,
            start=start,
            end=end,
        )
        if not imbalances:
            imbalances = resample(list(self.imbalance_samples), width)
        if not imbalances:
            imbalances = [0.0] * width
        elif len(imbalances) != width:
            imbalances = resample(imbalances, width)

        drifts = self._aurora_drift_series(width, window_start=start, window_end=end)
        if not drifts:
            drifts = [0.0] * width
        elif len(drifts) != width:
            drifts = resample(drifts, width)

        bursts = tape_bin_max(
            self.vol_burst_tape,
            width=width,
            start=start,
            end=end,
        )
        if not bursts:
            bursts = resample(list(self.vol_burst_samples), width)
        if not bursts:
            bursts = [0.0] * width
        elif len(bursts) != width:
            bursts = resample(bursts, width)

        config = self._aurora_config()
        top_burst = max(bursts) if bursts else 0.0
        if top_burst <= 1e-12:
            top_burst = 1.0
        strip = Text()
        for imbalance, drift, burst in zip(imbalances, drifts, bursts):
            gain = float(config.get("burst_gain", 1.0))
            ratio = max(0.0, min((float(burst) / top_burst) * gain, 1.0))
            char = self._SPARK_LEVELS[int(round(ratio * 8.0))]
            blended = self._aurora_blended_imbalance(float(imbalance), float(drift))
            style = self._aurora_style(blended, config=config)
            strip.append(char, style=style)
        return strip

    def render_legend(self) -> Text:
        legend = Text(f"Aurora[{self._aurora_preset}]  ")
        legend.append("BUY+", style="#18a63f")
        legend.append("/", style="dim")
        legend.append("BUY", style="#e6d84e")
        legend.append(" -> ", style="dim")
        legend.append("NEUTRAL", style="#8aa0b6")
        legend.append(" -> ", style="dim")
        legend.append("SELL", style="#ffaf00")
        legend.append("/", style="dim")
        legend.append("SELL+", style="red")
        legend.append("  |  +tape 15%  |  height=vol burst  |  a preset", style="dim")
        return legend

    def momentum(self, width: int) -> str:
        width = max(width, 1)
        mids = list(self.mid_samples)[-(width + 1) :]
        if len(mids) < 2:
            return self._MOMENTUM_CHARS[0] * width
        deltas = [mids[idx] - mids[idx - 1] for idx in range(1, len(mids))]
        scale = max(abs(delta) for delta in deltas) or 1.0
        levels = len(self._MOMENTUM_CHARS) - 1
        chars: list[str] = []
        for delta in deltas[-width:]:
            ratio = min(abs(delta) / scale, 1.0)
            idx = max(0, min(levels, int(round(ratio * levels))))
            chars.append(self._MOMENTUM_CHARS[idx])
        return "".join(chars).rjust(width, self._MOMENTUM_CHARS[0])
