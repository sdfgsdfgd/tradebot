"""Dense, incremental rendering for q-owned normalized strategy traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from rich.text import Text


_UP = "#73d89e"
_DOWN = "#ff4f8b"
_BLUE = "#62b0ff"
_AMBER = "#f2b36f"
_DIM = "#8aa0b6"
_GRID = "#34495a"
_HORIZON_COLUMNS = 5
_HORIZON_CELL_WIDTH = 21
_VOLATILITY_CELL_WIDTH = 22


def _map(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _style(value: object) -> str:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"up", "long", "buy", "supportive", "strengthening"}:
            return _UP
        if lowered in {"down", "short", "sell", "adverse", "weakening"}:
            return _DOWN
        return _DIM
    number = _number(value)
    return _UP if number and number > 0 else _DOWN if number and number < 0 else _DIM


def _direction(value: object) -> str:
    if value in (1, "up", "long", "BUY", "buy"):
        return "↑"
    if value in (-1, "down", "short", "SELL", "sell"):
        return "↓"
    return "→"


def _signed(value: object, digits: int = 3) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:+.{digits}f}"


def _fit(value: Text, width: int, *, align: str = "left") -> Text:
    fitted = value.copy()
    fitted.truncate(width, overflow="ellipsis")
    fitted.align(align, width)
    return fitted


def _append_grid(
    target: Text,
    cells: Sequence[tuple[Text, int, str]],
) -> None:
    for index, (cell, width, align) in enumerate(cells):
        if index:
            target.append("│", style=_GRID)
        target.append(_fit(cell, width, align=align))


def _horizon_identity(item: Mapping[str, object]) -> tuple[str, float] | None:
    bars = _number(item.get("bars"))
    if bars is not None:
        return ("bars", bars)
    minutes = _number(item.get("minutes"))
    return ("minutes", minutes) if minutes is not None else None


class BotTraceMixin:
    """Render compact trace observations without reinterpreting strategy truth."""

    def _init_strategy_traces(self) -> None:
        self._trace_rows_by_key: dict[str, list[dict[str, object]]] = {}
        self._all_trace_rows: list[dict[str, object]] = []

    def _setup_trace_tables(self) -> None:
        for table in self._trace_tables.values():
            table.clear(columns=True)
            table.add_column("When UTC", key="when", width=17)
            table.add_column(
                "Dense evidence · Δ since prior observation", key="trace", width=184
            )

    @staticmethod
    def _trace_row_key(trace: Mapping[str, object]) -> str:
        return "trace:" + str(
            trace.get("trace_id") or trace.get("event_id") or "missing"
        )

    @staticmethod
    def _trace_when(trace: Mapping[str, object]) -> Text:
        first = str(
            trace.get("first_recorded_at_utc") or trace.get("recorded_at_utc") or ""
        )
        last = str(trace.get("last_recorded_at_utc") or first)

        def stamp(value: str) -> str:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.strftime("%H:%M")
            except ValueError:
                return value[11:16] if len(value) >= 16 else value[:5] or "-"

        count = int(trace.get("sample_count") or 1)
        label = stamp(first)
        if count > 1:
            label += f"→{stamp(last)} ×{count}"
        return Text(label, style=_DIM)

    @staticmethod
    def _position(trace: Mapping[str, object]) -> str:
        positions = []
        for symbol, value in sorted(_map(trace.get("holdings")).items()):
            quantity = _number(value)
            if quantity is not None and abs(quantity) >= 1e-9:
                positions.append(f"{quantity:g}{symbol}")
        return "+".join(positions) if positions else "flat"

    @staticmethod
    def _delta_by_horizon(
        trace: Mapping[str, object],
    ) -> dict[tuple[str, float], Mapping[str, object]]:
        return {
            identity: item
            for item in _map(trace.get("delta")).get("horizons", ())
            if isinstance(item, Mapping)
            and (identity := _horizon_identity(item)) is not None
        }

    @classmethod
    def _trace_text(cls, trace: Mapping[str, object]) -> Text:
        result = Text()
        family = str(trace.get("family") or "STATE")
        decision = _map(trace.get("decision"))
        status = str(trace.get("status") or "-")
        status_style = (
            _UP
            if status in {"ACTIONABLE", "ALLOW", "PROMOTE"}
            else "#ff7070"
            if status in {"STOP", "QUARANTINE", "BROKEN"}
            else _AMBER
            if status in {"HOLD", "REVISE"}
            else _BLUE
        )

        status_cell = Text("◆ " if trace.get("episode_start") else "· ", style=_BLUE)
        status_cell.append(status, style=f"bold {status_style}")
        reason = str(trace.get("reason") or "-")
        economics = _map(trace.get("economics"))
        net = _number(economics.get("net"))
        trend = decision.get("trend")
        coherence = _number(decision.get("coherence"))

        net_cell = Text(
            "net=-" if net is None else f"net={net:+.2f}", style=_style(net)
        )
        trend_cell = Text("trend=", style=_DIM)
        if trend is not None:
            trend_cell.append(_direction(trend), style=_style(trend))
        else:
            trend_cell.append("-")
        if coherence is not None:
            trend_cell.append(f"  coh={coherence:.2f}", style=_style(coherence))
        _append_grid(
            result,
            (
                (status_cell, 14, "left"),
                (Text(reason), 24, "left"),
                (
                    Text(
                        f"target={trace.get('target_direction') or 'flat'}", style=_DIM
                    ),
                    14,
                    "center",
                ),
                (Text(f"pos={cls._position(trace)}", style=_DIM), 18, "center"),
                (net_cell, 13, "right"),
                (trend_cell, 19, "center"),
            ),
        )
        if family == "MCL_IMPULSE":
            cl_move = _number(decision.get("cl_move"))
            mcl_move = _number(decision.get("mcl_move"))
            basis_velocity = _number(decision.get("basis_velocity_ticks"))
            parity = decision.get("parity_aligned")
            if any(
                value is not None
                for value in (cl_move, mcl_move, basis_velocity, parity)
            ):
                result.append("│", style=_GRID)
                result.append("TAPE ", style="bold #8fbfff")
                if cl_move is not None:
                    result.append(f"CL={cl_move:+.4f} ", style=_style(cl_move))
                if mcl_move is not None:
                    result.append(f"MCL={mcl_move:+.4f} ", style=_style(mcl_move))
                if basis_velocity is not None:
                    result.append(
                        f"basis-v={basis_velocity:+.2f}t ", style=_style(basis_velocity)
                    )
                if parity is not None:
                    result.append(
                        "parity=✓" if parity else "parity=✗",
                        style=_UP if parity else _DOWN,
                    )
        result.append("\n")

        horizons = sorted(
            (item for item in trace.get("horizons", ()) if isinstance(item, Mapping)),
            key=lambda item: (
                _number(item.get("bars")) is None,
                _number(item.get("bars"))
                if _number(item.get("bars")) is not None
                else _number(item.get("minutes")) or 0.0,
            ),
        )
        horizon_slots: list[Mapping[str, object] | None] = [
            *horizons[:_HORIZON_COLUMNS]
        ]
        horizon_slots.extend([None] * (_HORIZON_COLUMNS - len(horizon_slots)))
        horizon_delta = cls._delta_by_horizon(trace)
        if family in {"IMPULSE", "MCL_IMPULSE"}:
            angle_cells: list[tuple[Text, int, str]] = []
            for item in horizon_slots:
                cell = Text()
                if item is None:
                    angle_cells.append((cell, _HORIZON_CELL_WIDTH, "left"))
                    continue
                minutes = _number(item.get("minutes"))
                bars = _number(item.get("bars"))
                label = (
                    f"{minutes:g}m"
                    if minutes is not None
                    else f"{bars:g}b"
                    if bars is not None
                    else "?"
                )
                angle = _number(item.get("angle"))
                cell.append(f"{label:>4} ", style=_DIM)
                cell.append(f"{_signed(angle, 1) + '°':>7}", style=_style(angle))
                cell.append(" ")
                delta = horizon_delta.get(_horizon_identity(item))
                angle_delta = _number(delta.get("angle")) if delta else None
                if angle_delta is not None and abs(angle_delta) >= 0.05:
                    cell.append(
                        f"{f'Δ{angle_delta:+.1f}':>7}", style=_style(angle_delta)
                    )
                else:
                    cell.append(" " * 7)
                angle_cells.append((cell, _HORIZON_CELL_WIDTH, "left"))
            result.append("θ ", style="bold #8fbfff")
            _append_grid(result, angle_cells)
        elif family == "GOLD_REGIME":
            result.append("REGIME ", style="bold #d6a56f")
            regime_cells: list[tuple[Text, int, str]] = []
            for label, key, width in (
                ("D-hard", "daily_hard", 18),
                ("D-soft", "daily_soft", 18),
                ("H4", "h4_hard", 14),
            ):
                value = decision.get(key)
                regime_cells.append(
                    (
                        Text(f"{label}={_direction(value)}", style=_style(value)),
                        width,
                        "center",
                    )
                )
            regime_cells.append(
                (
                    Text(
                        f"bar={str(decision.get('decision_bar_utc') or '-')[:16]}",
                        style=_DIM,
                    ),
                    31,
                    "center",
                )
            )
            _append_grid(result, regime_cells)
        else:
            result.append("STATE · normalized telemetry unavailable", style=_DIM)
        result.append("\n")

        volatility = _map(trace.get("volatility"))
        volatility_delta = _map(_map(trace.get("delta")).get("volatility"))
        if family in {"IMPULSE", "MCL_IMPULSE"}:
            velocity_cells: list[tuple[Text, int, str]] = []
            for item in horizon_slots:
                cell = Text()
                if item is None:
                    velocity_cells.append((cell, _HORIZON_CELL_WIDTH, "left"))
                    continue
                minutes = _number(item.get("minutes"))
                bars = _number(item.get("bars"))
                label = (
                    f"{minutes:g}m"
                    if minutes is not None
                    else f"{bars:g}b"
                    if bars is not None
                    else "?"
                )
                velocity = _number(item.get("slope_velocity"))
                cell.append(f"{label:>4} ", style=_DIM)
                cell.append(f"{_signed(velocity, 4):>8}", style=_style(velocity))
                velocity_cells.append((cell, _HORIZON_CELL_WIDTH, "left"))
            for label, key, digits in (
                ("r", "atr_ratio", 3),
                ("v", "atr_velocity", 5),
                ("a", "atr_acceleration", 5),
            ):
                value = _number(volatility.get(key))
                cell = Text("ATR " if label == "r" else "", style="bold #d6a56f")
                cell.append(f"{label}=", style="#d6a56f")
                cell.append(_signed(value, digits), style=_style(value))
                delta = _number(volatility_delta.get(key))
                if delta is not None and abs(delta) >= 10 ** (-(digits + 1)):
                    cell.append(" ")
                    cell.append(f"Δ{delta:+.{digits}f}", style=_BLUE)
                velocity_cells.append((cell, _VOLATILITY_CELL_WIDTH, "left"))
            result.append("ω ", style="bold #8fbfff")
            _append_grid(result, velocity_cells)
        elif family == "GOLD_REGIME":
            market = _map(trace.get("market"))
            gold_cells = [
                (
                    Text(
                        f"ATR14={_signed(volatility.get('atr_fast'), 2)}",
                        style="#d6a56f",
                    ),
                    21,
                    "center",
                ),
                (
                    Text(
                        f"r={_signed(volatility.get('atr_ratio'), 3)}",
                        style=_style(volatility.get("atr_ratio")),
                    ),
                    17,
                    "center",
                ),
                (
                    Text(
                        f"v={_signed(volatility.get('atr_velocity'), 5)}",
                        style=_style(volatility.get("atr_velocity")),
                    ),
                    23,
                    "center",
                ),
            ]
            for label, key, width in (
                ("H4 slope", "h4_fast_slope_dollars", 24),
                ("accel", "h4_fast_acceleration_dollars", 20),
                ("spread-v", "h4_spread_velocity_dollars", 25),
            ):
                value = _number(market.get(key))
                gold_cells.append(
                    (
                        Text(f"{label}={_signed(value, 2)}", style=_style(value)),
                        width,
                        "center",
                    )
                )
            _append_grid(result, gold_cells)
        result.append("\n")

        news = _map(trace.get("news"))
        if news:
            direction = news.get("direction")
            authority = (
                str(news.get("authority") or "context").upper().replace("_", "-")
            )
            news_style = _DIM if news.get("usable") is False else _style(direction)
            _append_grid(
                result,
                (
                    (
                        Text(
                            f"NEWS {authority[:11]} {_direction(direction)}",
                            style=f"bold {news_style}",
                        ),
                        20,
                        "left",
                    ),
                    (
                        Text(
                            f"P={_signed(news.get('signed_pressure'), 3)}",
                            style=news_style,
                        ),
                        12,
                        "right",
                    ),
                    (
                        Text(
                            f"ΔP={_signed(news.get('pressure_delta'), 3)}",
                            style=news_style,
                        ),
                        13,
                        "right",
                    ),
                    (
                        Text(
                            f"v/h={_signed(news.get('pressure_velocity_per_hour'), 5)}",
                            style=news_style,
                        ),
                        17,
                        "right",
                    ),
                    (
                        Text(
                            f"conf={_signed(news.get('confidence'), 2)}",
                            style=news_style,
                        ),
                        12,
                        "right",
                    ),
                    (
                        Text(f"impact={news.get('impact', '-')}", style=news_style),
                        12,
                        "right",
                    ),
                    (
                        Text(
                            str(news.get("change") or news.get("reason") or "-"),
                            style=news_style,
                        ),
                        14,
                        "center",
                    ),
                ),
            )
        else:
            result.append("NEWS unavailable", style="dim")
        long_context = _map(trace.get("long_context"))
        directions = _map(long_context.get("directions"))
        if directions:
            result.append("│", style=_GRID)
            result.append("LONG ", style="bold #8fbfff")
            for window, value in sorted(
                directions.items(),
                key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else 10_000,
            ):
                result.append(f"{window}{_direction(value)} ", style=_style(value))
        elif family == "MCL_IMPULSE" and long_context.get("last_raw_turn_utc"):
            result.append(
                f"│raw-turn {str(long_context['last_raw_turn_utc'])[11:16]}"
                f" {_direction(long_context.get('last_raw_turn_direction'))}",
                style=_BLUE,
            )
        elif family == "GOLD_REGIME":
            macro = _map(trace.get("macro"))
            macro_horizons = [
                item for item in macro.get("horizons", ()) if isinstance(item, Mapping)
            ]
            if macro_horizons:
                result.append("│MACRO ", style="bold #8fbfff")
                for item in macro_horizons:
                    value = item.get("direction")
                    result.append(f"{item.get('window')}={value} ", style=_style(value))
        return result

    def _reconcile_trace_table(
        self,
        key: str,
        rows: list[dict[str, object]],
    ) -> bool:
        table = self._trace_tables[key]
        previous = self._trace_rows_by_key.get(key, [])
        if previous == rows:
            return False
        old_keys = [self._trace_row_key(row) for row in previous]
        new_keys = [self._trace_row_key(row) for row in rows]
        selected_row = table.cursor_coordinate.row
        selected_key = (
            old_keys[selected_row] if 0 <= selected_row < len(old_keys) else None
        )
        old_by_key, new_by_key = dict(zip(old_keys, previous)), dict(
            zip(new_keys, rows)
        )
        old_set, new_set = set(old_keys), set(new_keys)
        for row_key in old_keys:
            if row_key not in new_set:
                table.remove_row(row_key)
        for row_key in new_keys:
            trace = new_by_key[row_key]
            if row_key not in old_set:
                table.add_row(
                    self._trace_when(trace),
                    self._trace_text(trace),
                    height=4,
                    key=row_key,
                )
            elif old_by_key[row_key] != trace:
                table.update_cell(row_key, "when", self._trace_when(trace))
                table.update_cell(row_key, "trace", self._trace_text(trace))
        self._trace_rows_by_key[key] = rows
        target = (
            new_keys.index(selected_key)
            if selected_key in new_set
            else len(new_keys) - 1
        )
        if target >= 0:
            table.cursor_coordinate = (target, 0)
        return True

    def _render_strategy_traces(self, traces: list[dict[str, object]]) -> bool:
        self._all_trace_rows = traces
        changed = False
        for key in self._trace_tables:
            rows = (
                [trace for trace in traces if trace.get("episode_start") is True]
                if key == "ALL"
                else [trace for trace in traces if trace.get("trace_key") == key]
            )
            changed = self._reconcile_trace_table(key, rows) or changed
        return changed

    def _selected_trace(self) -> dict[str, object] | None:
        key = self._active_trace_key()
        table = self._trace_tables.get(key)
        rows = self._trace_rows_by_key.get(key, [])
        if table is None:
            return None
        row = table.cursor_coordinate.row
        return rows[row] if 0 <= row < len(rows) else None

    def _latest_trace_for_sleeve(self, sleeve_id: object) -> dict[str, object] | None:
        matches = [
            trace
            for trace in getattr(self, "_all_trace_rows", ())
            if trace.get("sleeve_id") == sleeve_id
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _trace_legend() -> Text:
        legend = Text()
        legend.append("θ", style=f"bold {_BLUE}")
        legend.append("=slope÷TR°  ", style=_DIM)
        legend.append("ω", style=f"bold {_BLUE}")
        legend.append("=Δ(slope%/bar)  ", style=_DIM)
        for symbol, meaning in (
            ("r", "ATRfast÷slow"),
            ("v", "ΔATRfast"),
            ("a", "Δv"),
        ):
            legend.append(symbol, style=f"bold {_AMBER}")
            legend.append(f"={meaning}  ", style=_DIM)
        legend.append("Δ", style=f"bold {_BLUE}")
        legend.append("=vs prior displayed observation", style=_DIM)
        return legend

    @classmethod
    def _trace_context_lines(cls, trace: Mapping[str, object]) -> list[Text]:
        def parsed(value: object) -> datetime | None:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None

        first_raw = trace.get("first_recorded_at_utc") or trace.get("recorded_at_utc")
        last_raw = trace.get("last_recorded_at_utc") or first_raw
        first, last = parsed(first_raw), parsed(last_raw)
        first_label = (
            first.strftime("%m-%d %H:%M:%S") if first else str(first_raw or "-")[:19]
        )
        last_label = (
            last.strftime("%H:%M:%S")
            if first and last and first.date() == last.date()
            else last.strftime("%m-%d %H:%M:%S")
            if last
            else str(last_raw or "-")[:19]
        )
        count = int(trace.get("sample_count") or 1)
        identity = Text(
            str(trace.get("label") or trace.get("trace_key") or "Trace"), style="bold"
        )
        identity.append("  │  ", style=_GRID)
        identity.append(
            str(trace.get("trace_key") or "-").upper(), style=f"bold {_BLUE}"
        )
        identity.append("  │  ", style=_GRID)
        identity.append(f"{first_label} → {last_label} UTC", style=_DIM)
        identity.append("  │  ", style=_GRID)
        identity.append(
            f"{count} observation{'s' if count != 1 else ''}",
            style=_AMBER if count > 1 else _DIM,
        )

        news = _map(trace.get("news"))
        direction = news.get("direction")
        news_line = Text("NEWS ", style=f"bold {_BLUE}")
        news_line.append(
            f"aggregate {_direction(direction)}{news.get('impact', '-')}",
            style=f"bold {_style(direction)}",
        )
        news_line.append("  ", style=_GRID)
        driver_scores = [
            item for item in news.get("driver_scores", ()) if isinstance(item, Mapping)
        ]
        drivers = news.get("drivers")
        driver_cells: list[tuple[Text, int, str]] = []
        if driver_scores:
            for rank, item in enumerate(driver_scores[:5], 1):
                driver_direction = item.get("direction")
                impact = int(_number(item.get("impact")) or 0)
                label = str(item.get("id") or item.get("label") or "driver").replace(
                    "-", " "
                )
                cell = Text(f"{rank} ", style=_DIM)
                emphasis = "bold " if impact >= 70 else ""
                cell.append(
                    f"{_direction(driver_direction)}{impact:02d} ",
                    style=f"{emphasis}{_style(driver_direction)}",
                )
                cell.append(label, style=_style(driver_direction))
                driver_cells.append((cell, 29, "left"))
        elif isinstance(drivers, Sequence) and not isinstance(drivers, (str, bytes)):
            for rank, driver in enumerate(drivers[:5], 1):
                label = str(driver).replace("-", " ")
                driver_cells.append(
                    (Text(f"{rank} {label}", style=_style(direction)), 29, "left")
                )
        if driver_cells:
            _append_grid(news_line, driver_cells)
        else:
            news_line.append("no attributed drivers", style=_DIM)

        provenance = _map(trace.get("provenance"))
        evidence = Text("EVIDENCE ", style=f"bold {_BLUE}")
        evidence_cells: list[tuple[Text, int, str]] = []
        authority = provenance.get("source_authority")
        if authority:
            evidence_cells.append(
                (
                    Text(
                        "AUTH " + str(authority).replace("_", " "),
                        style=_UP,
                    ),
                    49,
                    "left",
                )
            )
        checkpoint = provenance.get("source_checkpoint_id")
        if checkpoint:
            checkpoint_label = str(checkpoint)
            if len(checkpoint_label) > 13:
                checkpoint_label = checkpoint_label[:12] + "…"
            evidence_cells.append(
                (Text(f"CHECKPOINT {checkpoint_label}", style=_DIM), 25, "left")
            )
        schema = provenance.get("source_schema") or provenance.get("context_schema")
        if schema:
            evidence_cells.append((Text(f"SCHEMA {schema}", style=_DIM), 50, "left"))
        recorded = parsed(provenance.get("source_recorded_at_utc"))
        if recorded:
            evidence_cells.append(
                (Text(f"AS OF {recorded:%m-%d %H:%M:%S} UTC", style=_DIM), 29, "left")
            )
        if evidence_cells:
            _append_grid(evidence, evidence_cells)
        else:
            evidence.append("embedded execution context", style=_DIM)

        change_windows = {
            int(hours): item
            for raw in news.get("change_windows", ())
            if isinstance(raw, Mapping)
            and (item := _map(raw))
            and (hours := _number(item.get("hours"))) is not None
        }
        momentum = Text("NEWS CURVE  ", style=f"bold {_BLUE}")
        curve: list[float | None] = []
        momentum_cells: list[tuple[Text, int, str]] = []
        for hours, label in ((4, "4h"), (24, "1d"), (168, "1w")):
            item = change_windows.get(hours, {})
            delta = _number(item.get("pressure_delta"))
            velocity = _number(item.get("pressure_velocity_per_hour"))
            curve.append(delta)
            cell = Text(f"{label} ", style=_DIM)
            if item.get("available") is True and delta is not None and velocity is not None:
                cell.append(f"ΔP={delta:+.3f}", style=_style(delta))
                cell.append(" ")
                cell.append(f"v/h={velocity:+.5f}", style=_style(velocity))
            else:
                cell.append("ΔP=n/a v/h=n/a", style=_DIM)
            momentum_cells.append((cell, 31, "left"))
        _append_grid(momentum, momentum_cells)
        momentum.append("│shape=", style=_GRID)
        for delta in curve:
            direction = 1 if delta and delta > 0 else -1 if delta and delta < 0 else 0
            momentum.append(_direction(direction), style=_style(delta))
        return [identity, news_line, momentum, evidence]

    @classmethod
    def _trace_detail_lines(cls, trace: Mapping[str, object]) -> list[Text]:
        context = cls._trace_context_lines(trace)
        return [context[0], *cls._trace_text(trace).split("\n"), *context[1:]]
