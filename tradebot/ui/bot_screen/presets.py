"""Leaderboard discovery and preset-tree presentation."""

from __future__ import annotations

import math
import re

from rich.text import Text

from ...engine import spot_riskoff_end_hour
from ...spot.champions import load_current_champion_groups
from ..bot_models import _BotPreset, _PresetHeader
from ..common import _pnl_text
from .formatting import _clean_group_label, _filters_for_group, _fmt_pct, _version_tag


class BotPresetsMixin:
    def _load_leaderboard(self) -> None:
        """Load CURRENT champion presets (README-driven).

        Bot Hub presets are intentionally limited to promoted CURRENT champions documented in:
        - `backtests/tqqq/readme-lf.md` (TQQQ LF spot champ)
        - `backtests/tqqq/readme-hf.md` (TQQQ HF spot champ)
        - `backtests/slv/readme-lf.md` (SLV spot champ)
        - `backtests/slv/readme-hf.md` (SLV HF spot champ)
        """

        groups, warnings = load_current_champion_groups(symbols=("TQQQ", "SLV"))
        self._spot_champ_version = next(
            (
                str(group.get("_version"))
                for group in groups
                if group.get("_track") == "LF"
                and any(
                    isinstance(entry, dict)
                    and str(entry.get("symbol") or "").strip().upper() == "TQQQ"
                    for entry in group.get("entries") or ()
                )
            ),
            None,
        )

        has_champions = bool(groups)
        if warnings:
            self._set_status(" | ".join(warnings))
        elif not has_champions:
            self._set_status("No CURRENT champions found. Check README paths.")

        self._payload = {"groups": groups}
        self._group_eval_by_name = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or "")
            eval_payload = group.get("_eval")
            if name and isinstance(eval_payload, dict):
                self._group_eval_by_name[name] = eval_payload
        self._rebuild_presets_table()

    def _rebuild_presets_table(self) -> None:
        self._presets = []
        self._preset_rows = []
        self._presets_table.clear(columns=True)
        self._presets_table.add_column("Preset")
        self._presets_table.add_column("Hours", width=10)
        self._presets_table.add_column("TF/Exec", width=10)
        self._presets_table.add_column("TP", width=7)
        self._presets_table.add_column("SL", width=7)
        self._presets_table.add_column("EMA", width=6)
        self._presets_table.add_column("P/DD 10|2|1", width=14)
        self._presets_table.add_column("PnL 10|2|1", width=18)
        self._presets_table.add_column("DD 10|2|1", width=18)
        self._presets_table.add_column("Active", width=7)
        self._presets_table.add_column("Live Total", width=12)

        payload = self._payload or {}
        groups = payload.get("groups", [])
        if not isinstance(groups, list) or not groups:
            self._move_cursor_to_first_preset()
            self._sync_row_marker(self._presets_table, force=True)
            self._render_status()
            self._presets_table.refresh(repaint=True)
            return

        active_by_preset_key: dict[str, int] = {}
        live_total_by_preset_key: dict[str, float] = {}
        live_seen_by_preset_key: dict[str, bool] = {}
        for instance in self._instances:
            if str(getattr(instance, "state", "")).strip().upper() != "RUNNING":
                continue
            preset_key = str(getattr(instance, "preset_key", "") or "").strip()
            if not preset_key:
                continue
            active_by_preset_key[preset_key] = active_by_preset_key.get(preset_key, 0) + 1
            live_total = self._instance_live_total_by_id.get(int(instance.instance_id))
            if live_total is None:
                continue
            live_total_by_preset_key[preset_key] = live_total_by_preset_key.get(preset_key, 0.0) + float(
                live_total
            )
            live_seen_by_preset_key[preset_key] = True

        def _get_symbol(group_name: str, entry: dict, strat: dict) -> str:
            raw = entry.get("symbol") or strat.get("symbol") or payload.get("symbol") or ""
            cleaned = str(raw or "").strip().upper()
            if cleaned:
                return cleaned
            text = str(group_name or "")
            if "(" in text and ")" in text:
                inside = text.split("(", 1)[1].split(")", 1)[0].strip()
                if inside and inside.replace("-", "").isalnum():
                    return inside.upper()
            return "UNKNOWN"

        def _get_dd(metrics: dict) -> float | None:
            for key in ("max_drawdown", "dd"):
                if metrics.get(key) is None:
                    continue
                try:
                    dd = float(metrics.get(key))
                except (TypeError, ValueError):
                    continue
                if dd >= 0:
                    return dd
            return None

        def _score(metrics: dict) -> float:
            try:
                value = float(metrics.get("pnl_over_dd"))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                return value
            try:
                pnl = float(metrics.get("pnl") or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            dd = _get_dd(metrics)
            return pnl / dd if dd and dd > 0 else float("-inf")

        def _compact_bar_size(raw: str) -> str:
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
                    return f"{num}d"
            return (
                text.replace(" mins", "m")
                .replace(" min", "m")
                .replace(" hours", "h")
                .replace(" hour", "h")
                .replace(" days", "d")
                .replace(" day", "d")
            )

        def _fmt_money_compact(value: float | None) -> str:
            if value is None:
                return "-"
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return "-"
            sign = "-" if parsed < 0 else ""
            parsed = abs(parsed)
            if parsed >= 1_000_000:
                return f"{sign}{parsed / 1_000_000:.1f}M"
            if parsed >= 10_000:
                return f"{sign}{parsed / 1_000:.1f}k"
            return f"{sign}{parsed:,.0f}"

        def _fmt_ratio_compact(value: float | None) -> str:
            if value is None:
                return "-"
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return "-"

        def _eval_windows(group_name: str) -> list[dict]:
            eval_payload = self._group_eval_by_name.get(group_name)
            if not isinstance(eval_payload, dict):
                return []
            windows = eval_payload.get("windows")
            if not isinstance(windows, list) or not windows:
                return []
            cleaned = [w for w in windows if isinstance(w, dict)]
            cleaned.sort(key=lambda w: str(w.get("start") or ""))
            return cleaned

        def _window_triplets(group_name: str, metrics: dict) -> tuple[str, Text, Text]:
            windows = _eval_windows(group_name)
            pnls: list[float | None] = []
            dds: list[float | None] = []
            ratios: list[float | None] = []

            def _as_float(value) -> float | None:
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            if windows:
                for w in windows:
                    pnl = _as_float(w.get("pnl"))
                    dd = _as_float(w.get("dd"))
                    if dd is None:
                        dd = _as_float(w.get("max_drawdown"))
                    ratio = _as_float(w.get("pnl_over_dd"))
                    if ratio is None and pnl is not None and dd is not None and dd > 0:
                        ratio = pnl / dd
                    pnls.append(pnl)
                    dds.append(dd)
                    ratios.append(ratio)
            else:
                pnl = _as_float(metrics.get("pnl"))
                dd = _get_dd(metrics)
                ratio = _as_float(metrics.get("pnl_over_dd"))
                if ratio is None and pnl is not None and dd is not None and dd > 0:
                    ratio = pnl / dd
                pnls = [pnl]
                dds = [dd]
                ratios = [ratio]

            ratio_s = "/".join(_fmt_ratio_compact(v) for v in ratios)
            pnl_s = "/".join(_fmt_money_compact(v) for v in pnls)
            dd_s = "/".join(_fmt_money_compact(v) for v in dds)

            pnl_style = "green" if any(v is not None for v in pnls) and all((v or 0) >= 0 for v in pnls) else ""
            pnl_cell = Text(pnl_s, style=pnl_style) if pnl_s else Text("")
            dd_cell = Text(dd_s, style="red") if dd_s else Text("")
            return ratio_s, pnl_cell, dd_cell

        def _spot_tp_sl(strat: dict) -> tuple[str, str]:
            exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
            if exit_mode == "atr":
                pt = strat.get("spot_pt_atr_mult")
                sl = strat.get("spot_sl_atr_mult")
                pt_s = "-"
                sl_s = "-"
                if pt is not None:
                    try:
                        pt_s = f"x{float(pt):.2f}"
                    except (TypeError, ValueError):
                        pt_s = "-"
                if sl is not None:
                    try:
                        sl_s = f"x{float(sl):.2f}"
                    except (TypeError, ValueError):
                        sl_s = "-"
                return pt_s, sl_s
            pt = strat.get("spot_profit_target_pct")
            sl = strat.get("spot_stop_loss_pct")
            pt_s = "-"
            sl_s = "-"
            if pt is not None:
                try:
                    pt_s = _fmt_pct(float(pt) * 100.0)
                except (TypeError, ValueError):
                    pt_s = "-"
            if sl is not None:
                try:
                    sl_s = _fmt_pct(float(sl) * 100.0)
                except (TypeError, ValueError):
                    sl_s = "-"
            return pt_s, sl_s

        def _options_tp_sl(strat: dict) -> tuple[str, str]:
            try:
                pt = float(strat.get("profit_target", 0.0)) * 100.0
            except (TypeError, ValueError):
                pt = 0.0
            try:
                sl = float(strat.get("stop_loss", 0.0)) * 100.0
            except (TypeError, ValueError):
                sl = 0.0
            return _fmt_pct(pt), _fmt_pct(sl)

        def _short_preset_name(*, group_name: str, symbol: str, instrument: str, source: str) -> str:
            base = _clean_group_label(group_name)
            if instrument == "spot" and symbol and f"Spot ({symbol})" in base:
                base = base.split(f"Spot ({symbol})", 1)[1].strip()
                while base[:1] in ("-", "—", ":", "|"):
                    base = base[1:].strip()
            return base or group_name

        def _track_label(*, source: str) -> str:
            return "HF" if ":HF:" in str(source or "") else "LF"

        contracts: dict[str, dict] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("name") or "")
            source = str(group.get("_source") or "").strip() or "unknown"
            entries = group.get("entries", [])
            if not isinstance(entries, list) or not entries:
                continue
            for entry_idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                strat = entry.get("strategy", {}) if isinstance(entry.get("strategy"), dict) else {}
                metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
                instrument = self._strategy_instrument(strat)

                legs = strat.get("legs", [])
                if instrument == "options" and (not isinstance(legs, list) or not legs):
                    continue

                try:
                    win = float(metrics.get("win_rate", 0.0))
                except (TypeError, ValueError):
                    win = 0.0
                if self._filter_min_win_rate is not None and win < self._filter_min_win_rate:
                    continue

                try:
                    dte = int(strat.get("dte", 0))
                except (TypeError, ValueError):
                    dte = 0
                if instrument == "options" and self._filter_dte is not None and dte != self._filter_dte:
                    continue

                symbol = _get_symbol(group_name, entry, strat)

                signal_bar = str(strat.get("signal_bar_size") or "").strip()
                if not signal_bar and instrument == "options":
                    signal_bar = str(payload.get("bar_size") or "").strip()
                tf = signal_bar or "?"

                name = _short_preset_name(group_name=group_name, symbol=symbol, instrument=instrument, source=source)
                row_id = f"preset:{source}:{group_name}:{entry_idx}"
                preset = _BotPreset(group=group_name, entry=entry, key=row_id)

                item = {
                    "row_id": row_id,
                    "preset": preset,
                    "name": name,
                    "symbol": symbol,
                    "instrument": instrument,
                    "source": source,
                    "version": group.get("_version"),
                    "tf": tf,
                    "dte": dte,
                    "metrics": metrics,
                    "strategy": strat,
                    "score": _score(metrics),
                }
                bucket = contracts.setdefault(symbol, {"spot": {}, "options": {}})
                if instrument == "spot":
                    track = _track_label(source=source)
                    bucket["spot"].setdefault(track, []).append(item)
                else:
                    bucket["options"].setdefault(dte, []).append(item)

        symbols = sorted(contracts.keys(), key=lambda sym: (0 if sym == "TQQQ" else 1, sym))
        if not self._preset_expand_initialized:
            self._preset_expanded = {f"contract:{sym}" for sym in symbols}
            self._preset_expanded |= {f"contract:{sym}|spot" for sym in symbols}
            self._preset_expand_initialized = True
            self._preset_known_contracts = set(symbols)
        else:
            for sym in symbols:
                if sym not in self._preset_known_contracts:
                    self._preset_expanded.add(f"contract:{sym}")
                    self._preset_expanded.add(f"contract:{sym}|spot")
            self._preset_known_contracts.update(symbols)

        def _best(items: list[dict]) -> dict | None:
            return max(items, key=lambda it: float(it.get("score", float("-inf")))) if items else None

        def _best_version(best: dict | None) -> str | None:
            if best is None:
                return None
            version_tag = _version_tag(best.get("version"))
            if version_tag:
                return version_tag
            raw = str(best.get("name") or "")
            match = re.search(r"\bv(?P<ver>\d+(?:\.\d+)?)\b", raw, flags=re.IGNORECASE)
            if match:
                return f"v{match.group('ver')}"
            preset = best.get("preset")
            group_name = getattr(preset, "group", "")
            match = re.search(r"\bv(?P<ver>\d+(?:\.\d+)?)\b", str(group_name or ""), flags=re.IGNORECASE)
            return f"v{match.group('ver')}" if match else None

        def _label_with_version(label: str, best: dict | None) -> str:
            ver = _best_version(best)
            if not ver:
                return label
            if re.search(rf"\b{re.escape(ver)}\b", label, flags=re.IGNORECASE):
                return label
            return f"{label} {ver}"

        def _hours_label(*, strat: dict, filters: dict | None) -> str:
            use_rth_raw = strat.get("signal_use_rth")
            use_rth = None if use_rth_raw is None else bool(use_rth_raw)

            start = end = None
            if isinstance(filters, dict):
                raw_start_et = filters.get("entry_start_hour_et")
                raw_end_et = filters.get("entry_end_hour_et")
                if raw_start_et is not None and raw_end_et is not None:
                    try:
                        start = int(raw_start_et)
                        end = int(raw_end_et)
                    except (TypeError, ValueError):
                        start = None
                        end = None
                else:
                    raw_start = filters.get("entry_start_hour")
                    raw_end = filters.get("entry_end_hour")
                    if raw_start is not None and raw_end is not None:
                        try:
                            start = int(raw_start)
                            end = int(raw_end)
                        except (TypeError, ValueError):
                            start = None
                            end = None

            cutoff = spot_riskoff_end_hour(filters) if isinstance(filters, dict) else None

            if start is not None and end is not None:
                prefix = "R" if use_rth is True else ("F" if use_rth is False else "")
                label = f"{prefix}{start}-{end}"
                if cutoff is not None:
                    label = f"{label}c{cutoff}"
                return label

            if use_rth is True:
                return "RTH"
            if use_rth is False:
                return "24/5"
            return "-"

        def _active_live_cells(items: list[dict]) -> tuple[str, Text]:
            if not items:
                return "", Text("")
            active = 0
            live_total = 0.0
            live_seen = False
            for item in items:
                key = str(item.get("row_id") or "").strip()
                if not key:
                    continue
                active += int(active_by_preset_key.get(key, 0))
                if bool(live_seen_by_preset_key.get(key)):
                    live_total += float(live_total_by_preset_key.get(key, 0.0))
                    live_seen = True
            active_text = str(active) if active > 0 else ""
            live_text = _pnl_text(live_total) if live_seen else Text("")
            return active_text, live_text

        def _add_leaf(item: dict, *, depth: int) -> None:
            preset = item["preset"]
            strat = item["strategy"]
            metrics = item["metrics"]
            instrument = item["instrument"]

            if instrument == "spot":
                exec_bar = str(strat.get("spot_exec_bar_size") or "").strip()
                if exec_bar:
                    tf_dte = f"{_compact_bar_size(item['tf'])}→{_compact_bar_size(exec_bar)}"
                else:
                    tf_dte = _compact_bar_size(item["tf"])
                tp_s, sl_s = _spot_tp_sl(strat)
            else:
                tf_dte = str(int(item["dte"]))
                tp_s, sl_s = _options_tp_sl(strat)

            filters = _filters_for_group(payload, preset.group) if self._payload else None
            hours_s = _hours_label(strat=strat, filters=filters)[:10]
            ema = str(strat.get("ema_preset", ""))[:6]

            ratio_s, pnl_trip, dd_trip = _window_triplets(preset.group, metrics)
            active_text, live_text = _active_live_cells([item])

            label = f"{'  ' * depth}{item['name']}".rstrip()
            self._presets.append(preset)
            self._preset_rows.append(preset)
            self._presets_table.add_row(
                label,
                hours_s,
                tf_dte,
                tp_s,
                sl_s,
                ema,
                ratio_s,
                pnl_trip,
                dd_trip,
                active_text,
                live_text,
                key=item["row_id"],
            )

        def _add_header(
            node_id: str,
            *,
            depth: int,
            label: str,
            best: dict | None,
            items: list[dict],
        ) -> bool:
            expanded = node_id in self._preset_expanded
            caret = "▾" if expanded else "▸"
            left = f"{'  ' * depth}{caret} {label}".rstrip()
            legs_cell = ""
            tf_dte = ""
            tp_s = ""
            sl_s = ""
            ema = ""
            ratio_s = ""
            pnl_trip = Text("")
            dd_trip = Text("")
            active_text, live_text = _active_live_cells(items)

            if best is not None:
                filters = _filters_for_group(payload, best["preset"].group) if self._payload else None
                legs_cell = _hours_label(strat=best["strategy"], filters=filters)[:10]
                strat = best["strategy"]
                instrument = best["instrument"]
                if instrument == "spot":
                    exec_bar = str(strat.get("spot_exec_bar_size") or "").strip()
                    if exec_bar:
                        tf_dte = f"{_compact_bar_size(best['tf'])}→{_compact_bar_size(exec_bar)}"
                    else:
                        tf_dte = _compact_bar_size(best["tf"])
                    tp_s, sl_s = _spot_tp_sl(strat)
                else:
                    tf_dte = str(int(best["dte"]))
                    tp_s, sl_s = _options_tp_sl(strat)
                ema = str(strat.get("ema_preset", ""))[:6]
                metrics = best["metrics"]
                ratio_s, pnl_trip, dd_trip = _window_triplets(best["preset"].group, metrics)

            self._preset_rows.append(_PresetHeader(node_id=node_id, depth=depth, label=label))
            self._presets_table.add_row(
                Text(left, style="bold"),
                Text(legs_cell, style="dim") if legs_cell else Text(""),
                tf_dte,
                tp_s,
                sl_s,
                ema,
                ratio_s,
                pnl_trip,
                dd_trip,
                active_text,
                live_text,
                key=node_id,
            )
            return expanded

        for symbol in symbols:
            bucket = contracts[symbol]
            spot_tracks = list(bucket["spot"].keys())
            multi_track_spot = len(spot_tracks) > 1
            all_items: list[dict] = []
            for track_items in bucket["spot"].values():
                all_items.extend(track_items)
            for dte_items in bucket["options"].values():
                all_items.extend(dte_items)

            contract_node = f"contract:{symbol}"
            contract_best = _best(all_items)
            contract_expanded = _add_header(
                contract_node,
                depth=0,
                label=symbol if multi_track_spot else _label_with_version(symbol, contract_best),
                best=contract_best,
                items=all_items,
            )
            if not contract_expanded:
                continue

            spot_node = f"{contract_node}|spot"
            spot_items = [it for track_items in bucket["spot"].values() for it in track_items]
            spot_best = _best(spot_items)
            spot_expanded = _add_header(
                spot_node,
                depth=1,
                label=f"{symbol} - Spot" if multi_track_spot else _label_with_version(f"{symbol} - Spot", spot_best),
                best=spot_best,
                items=spot_items,
            )
            if spot_expanded:
                spot_tracks.sort(key=lambda t: (0 if t == "LF" else 1 if t == "HF" else 2, t))
                if len(spot_tracks) == 1:
                    track_items = bucket["spot"][spot_tracks[0]]
                    ordered = sorted(
                        track_items,
                        key=lambda it: (-float(it.get("score", float("-inf"))), it["name"]),
                    )
                    for item in ordered:
                        _add_leaf(item, depth=2)
                else:
                    for track in spot_tracks:
                        track_items = bucket["spot"][track]
                        track_node = f"{spot_node}|track:{track}"
                        track_expanded = _add_header(
                            track_node,
                            depth=2,
                            label=_label_with_version(str(track), _best(track_items)),
                            best=_best(track_items),
                            items=track_items,
                        )
                        if track_expanded:
                            ordered = sorted(
                                track_items,
                                key=lambda it: (-float(it.get("score", float("-inf"))), it["name"]),
                            )
                            for item in ordered:
                                _add_leaf(item, depth=3)

            opt_node = f"{contract_node}|options"
            opt_items = [it for items in bucket["options"].values() for it in items]
            if opt_items:
                opt_expanded = _add_header(
                    opt_node,
                    depth=1,
                    label=f"{symbol} - Options",
                    best=_best(opt_items),
                    items=opt_items,
                )
                if opt_expanded:
                    for dte in sorted(bucket["options"].keys()):
                        dte_items = bucket["options"][dte]
                        dte_node = f"{opt_node}|dte:{dte}"
                        dte_expanded = _add_header(
                            dte_node,
                            depth=2,
                            label=f"DTE {dte}",
                            best=_best(dte_items),
                            items=dte_items,
                        )
                        if dte_expanded:
                            ordered = sorted(
                                dte_items,
                                key=lambda it: (-float(it.get("score", float("-inf"))), it["name"]),
                            )
                            for item in ordered:
                                _add_leaf(item, depth=3)

        self._move_cursor_to_first_preset()
        self._sync_row_marker(self._presets_table, force=True)
        self._render_status()
        self._presets_table.refresh(repaint=True)

    def _move_cursor_to_first_preset(self) -> None:
        if not self._preset_rows:
            return
        self._presets_table.cursor_coordinate = (0, 0)

    def _toggle_preset_node(self, node_id: str) -> None:
        if not node_id:
            return
        if node_id in self._preset_expanded:
            self._preset_expanded.remove(node_id)
        else:
            self._preset_expanded.add(node_id)
        self._rebuild_presets_table()
        try:
            row = self._presets_table.get_row_index(node_id)
        except Exception:
            row = None
        if row is not None and self._presets_table.is_valid_row_index(row):
            self._presets_table.cursor_coordinate = (row, 0)
