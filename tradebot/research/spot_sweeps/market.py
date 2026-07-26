"""SweepMarketData capability slice for the canonical spot research runtime."""

from __future__ import annotations

import time as pytime
from collections.abc import Iterable
from datetime import datetime, timedelta
from ...backtest.config import (
    ConfigBundle,
    FiltersConfig,
)
from ...backtest.config_filters import _parse_filters
from ...backtest.spot_context import (
    SpotBarRequirement,
    SpotContextBars,
    load_spot_context_bars,
)
from ...backtest.sweep_parallel import (
    _progress_line,
)
from ...backtest.sweeps import (
    write_json,
)
from ...chart_data.cache import series_cache_service
from ...chart_data.series import BarSeriesSignature, bars_list
from .catalog import (
    AxisExecutionSpec,
    _AXIS_EXECUTION_SPEC_BY_NAME,
    _AXIS_TOTAL_HINT_DIMS_BY_NAME,
    _AXIS_TOTAL_HINT_MODE_BY_NAME,
    _AXIS_TOTAL_HINT_STATIC_BY_NAME,
    _combo_full_preset_key,
    _combo_full_preset_tier,
)
from .dimensions import _AXIS_DIMENSION_REGISTRY
from .profiles import (
    _ATR_EXIT_PROFILE_REGISTRY,
    _SHOCK_SWEEP_PROFILE,
    _SPREAD_PROFILE_REGISTRY,
    _SUPERTREND_SEARCH_PROFILE,
)
from .milestones import (
    _filters_payload,
    _rank_cfg_rows,
    _score_row_pnl,
    _score_row_pnl_dd,
)
from .support import (
    _cardinality,
    _require_offline_cache_or_die,
)


_SERIES_CACHE = series_cache_service()
_SWEEP_BARS_NAMESPACE = "spot.sweeps.bars"


class SweepMarketData:
    def _merge_filters(self, base_filters: FiltersConfig | None, overrides: dict[str, object]) -> FiltersConfig | None:
        """Merge base filters with overrides, where `None` deletes a key.

        Used to build joint permission sweeps without being constrained by the compact preset funnel.
        """
        merged: dict[str, object] = dict(_filters_payload(base_filters) or {})
        for key, val in overrides.items():
            if val is None:
                merged.pop(key, None)
            else:
                merged[key] = val

        # Keep TOD gating consistent (both-or-neither).
        if ("entry_start_hour_et" in merged) ^ ("entry_end_hour_et" in merged):
            merged.pop("entry_start_hour_et", None)
            merged.pop("entry_end_hour_et", None)
        if ("entry_start_hour" in merged) ^ ("entry_end_hour" in merged):
            merged.pop("entry_start_hour", None)
            merged.pop("entry_end_hour", None)

        # Volume gate requires both knobs.
        if merged.get("volume_ratio_min") is None:
            merged.pop("volume_ema_period", None)

        # Riskpanic overlay requires both knobs.
        if ("riskpanic_tr5_med_pct" in merged) ^ ("riskpanic_neg_gap_ratio_min" in merged):
            merged.pop("riskpanic_tr5_med_pct", None)
            merged.pop("riskpanic_neg_gap_ratio_min", None)

        # Riskpop overlay requires both knobs.
        if ("riskpop_tr5_med_pct" in merged) ^ ("riskpop_pos_gap_ratio_min" in merged):
            merged.pop("riskpop_tr5_med_pct", None)
            merged.pop("riskpop_pos_gap_ratio_min", None)

        f = _parse_filters(merged)
        return f if _filters_payload(f) is not None else None

    def _ranked_keys_by_row_scores(self, best_by_key: dict, *, top_pnl: int = 8, top_pnl_dd: int = 8) -> list:
        ranked = _rank_cfg_rows(
            [(key, rec["row"], "") for key, rec in best_by_key.items() if isinstance(rec, dict) and isinstance(rec.get("row"), dict)],
            scorers=[
                (_score_row_pnl, int(top_pnl)),
                (_score_row_pnl_dd, int(top_pnl_dd)),
            ],
            key_fn=lambda key, _row, _note: key,
        )
        return [key for key, _row, _note in ranked]

    def _preflight_offline_requirements(
        self,
        requirements: Iterable[SpotBarRequirement],
        *,
        start_dt: datetime,
        end_dt: datetime,
        honor_warmup: bool = False,
    ) -> None:
        unique: dict[tuple[object, ...], SpotBarRequirement] = {}
        for req in requirements:
            key = (req.symbol, req.exchange, req.bar_size, req.use_rth)
            if key not in unique or req.warmup_days > unique[key].warmup_days:
                unique[key] = req
        for req in unique.values():
            _require_offline_cache_or_die(
                data=self.data,
                cache_dir=self.cache_dir,
                symbol=req.symbol,
                exchange=req.exchange,
                start_dt=start_dt - timedelta(days=req.warmup_days if honor_warmup else 0),
                end_dt=end_dt,
                bar_size=req.bar_size,
                use_rth=req.use_rth,
                cache_policy=self.cache_policy,
            )

    def _bars(self, bar_size: str) -> list:
        if self.offline:
            _require_offline_cache_or_die(
                data=self.data,
                cache_dir=self.cache_dir,
                symbol=self.symbol,
                exchange=None,
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                bar_size=str(bar_size),
                use_rth=self.use_rth,
                cache_policy=self.cache_policy,
            )
            series = self.data.load_cached_bar_series(
                symbol=self.symbol,
                exchange=None,
                start=self.start_dt,
                end=self.end_dt,
                bar_size=str(bar_size),
                use_rth=self.use_rth,
                cache_dir=self.cache_dir,
            )
            return bars_list(series)
        series = self.data.load_or_fetch_bar_series(
            symbol=self.symbol,
            exchange=None,
            start=self.start_dt,
            end=self.end_dt,
            bar_size=str(bar_size),
            use_rth=self.use_rth,
            cache_dir=self.cache_dir,
        )
        return bars_list(series)

    def _bars_cached(self, bar_size: str) -> list:
        key = (
            str(self.symbol).upper(),
            self.start_dt.isoformat(),
            self.end_dt.isoformat(),
            str(bar_size),
            bool(self.use_rth),
            bool(self.offline),
        )
        cached = _SERIES_CACHE.get(namespace=_SWEEP_BARS_NAMESPACE, key=key)
        if isinstance(cached, list):
            return cached
        loaded = self._bars(str(bar_size))
        _SERIES_CACHE.set(namespace=_SWEEP_BARS_NAMESPACE, key=key, value=loaded)
        return loaded

    def _context_bars_for_cfg(
        self,
        *,
        cfg: ConfigBundle,
        bars: list | None = None,
    ) -> SpotContextBars:
        bars_eff = bars if bars is not None else self._bars_cached(str(cfg.backtest.bar_size))

        def _load(req: SpotBarRequirement, _start: datetime, _end: datetime):
            return self._bars_cached(str(req.bar_size))

        def _missing(req: SpotBarRequirement, _start: datetime, _end: datetime) -> None:
            raise SystemExit(f"No {req.bar_size} {req.kind} bars returned (IBKR).")

        return load_spot_context_bars(
            strategy=cfg.strategy,
            signal_bars=bars_eff,
            default_symbol=str(cfg.strategy.symbol),
            default_exchange=cfg.strategy.exchange,
            default_signal_bar_size=str(cfg.backtest.bar_size),
            default_signal_use_rth=bool(cfg.backtest.use_rth),
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            load_requirement=_load,
            on_missing=_missing,
        )

    def _bars_signature(self, series: list | None) -> BarSeriesSignature:
        return _SERIES_CACHE.revision(series or ())

    def _axis_total_hint(self, axis_name: str) -> int | None:
        axis = str(axis_name).strip().lower()
        include_combo_baseline = not bool(getattr(self.args, "combo_full_cartesian_stage", None))
        if axis == "combo_full":
            preset = _combo_full_preset_key(str(getattr(self.args, "combo_full_preset", "") or ""))
            context = self._combo_full_context(preset)
            return (
                int(context.run_total)
                if include_combo_baseline
                else int(context.total)
            )
        spec = _AXIS_EXECUTION_SPEC_BY_NAME.get(str(axis))
        hint_static = int(spec.total_hint_static) if isinstance(spec, AxisExecutionSpec) and isinstance(spec.total_hint_static, int) else None
        hint_mode = str(spec.total_hint_mode or "").strip().lower() if isinstance(spec, AxisExecutionSpec) else ""
        hint_dims = tuple(spec.total_hint_dims or ()) if isinstance(spec, AxisExecutionSpec) else ()
        if hint_static is None:
            raw_static = _AXIS_TOTAL_HINT_STATIC_BY_NAME.get(str(axis))
            if raw_static is not None:
                try:
                    hint_static = int(raw_static)
                except (TypeError, ValueError):
                    hint_static = None
        if not hint_mode:
            hint_mode = str(_AXIS_TOTAL_HINT_MODE_BY_NAME.get(str(axis), "")).strip().lower()
        if not hint_dims:
            hint_dims = tuple(_AXIS_TOTAL_HINT_DIMS_BY_NAME.get(str(axis), ()))
        if isinstance(hint_static, int) and hint_static > 0:
            return int(hint_static)

        def _combo_dim_labels(dim_key: str) -> tuple[str, ...]:
            combo_dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
            if not isinstance(combo_dims, dict):
                return ()
            raw_variants = combo_dims.get(f"{dim_key}_variants")
            labels_from_variants: list[str] = []
            if isinstance(raw_variants, (list, tuple)):
                for row in raw_variants:
                    if not (isinstance(row, (list, tuple)) and len(row) >= 1):
                        continue
                    label = str(row[0] or "").strip()
                    if label:
                        labels_from_variants.append(label)
            return tuple(labels_from_variants)

        if hint_mode == "atr_profile":
            profile = _ATR_EXIT_PROFILE_REGISTRY.get(str(axis)) or {}
            total = _cardinality(
                len(tuple(profile.get("atr_periods") or ())),
                len(tuple(profile.get("pt_mults") or ())),
                len(tuple(profile.get("sl_mults") or ())),
            )
            if total > 0:
                return int(total)
        if hint_mode == "spread_profile":
            profile = _SPREAD_PROFILE_REGISTRY.get(str(axis)) or {}
            total = len(tuple(profile.get("values") or ()))
            if total > 0:
                return int(total)
        if hint_mode == "regime_profile":
            total = _SUPERTREND_SEARCH_PROFILE.cardinality("primary")
            if total > 0:
                return int(total)
        if hint_mode == "regime2_profile":
            total = _SUPERTREND_SEARCH_PROFILE.cardinality("confirmation")
            if total > 0:
                return int(total)
        if hint_mode == "shock_profile":
            profile = _SHOCK_SWEEP_PROFILE
            preset_count = int(
                len(tuple(profile.get("ratio_rows") or ())) + len(tuple(profile.get("daily_atr_rows") or ())) + len(tuple(profile.get("drawdown_rows") or ()))
            )
            total = _cardinality(
                int(preset_count),
                len(tuple(profile.get("modes") or ())),
                len(tuple(profile.get("dir_variants") or ())),
                len(tuple(profile.get("sl_mults") or ())),
                len(tuple(profile.get("pt_mults") or ())),
                len(tuple(profile.get("short_risk_factors") or ())),
            )
            if total > 0:
                return int(total)
        if hint_mode == "combo_subset" and hint_dims:
            sizes: list[int] = []
            risk_tier_hint = str(axis) == "risk_overlays" or _combo_full_preset_tier(str(axis)) == "risk"
            for dim_key in hint_dims:
                labels = list(_combo_dim_labels(str(dim_key)))
                if risk_tier_hint and str(dim_key) == "risk" and bool(getattr(self.args, "risk_overlays_skip_pop", False)):
                    labels = [lbl for lbl in labels if "riskpop" not in str(lbl).lower()]
                sizes.append(max(1, len(labels)))
            total = _cardinality(*sizes)
            if total > 0:
                return int(total) + (1 if include_combo_baseline else 0)
        if hint_mode == "gate_matrix":
            combo_dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
            gate_dims = _AXIS_DIMENSION_REGISTRY.get("gate_matrix", {})
            if isinstance(combo_dims, dict):
                perm_total = len(tuple(gate_dims.get("perm_variants") or ())) if isinstance(gate_dims, dict) else 0
                if perm_total <= 0:
                    perm_total = len(_combo_dim_labels("perm"))
                tod_total = len(tuple(gate_dims.get("tod_variants") or ())) if isinstance(gate_dims, dict) else 0
                if tod_total <= 0:
                    tod_total = len(_combo_dim_labels("tod"))
                short_total = len(tuple(gate_dims.get("short_mults") or ())) if isinstance(gate_dims, dict) else 0
                if short_total <= 0:
                    short_total = len(tuple(combo_dims.get("short_mults") or ()))
                total = _cardinality(
                    max(1, int(perm_total)),
                    max(1, int(tod_total)),
                    max(1, len(_combo_dim_labels("regime2"))),
                    max(1, len(_combo_dim_labels("shock"))),
                    max(1, len(_combo_dim_labels("risk"))),
                    max(1, int(short_total)),
                )
                if total > 0:
                    return int(total) + (1 if include_combo_baseline else 0)
        hist = self.axis_progress_history.get(str(axis))
        if isinstance(hist, int) and hist > 0:
            return int(hist)
        return None

    def _axis_progress_begin(self, *, axis_name: str) -> None:
        axis_key = str(axis_name).strip().lower()
        self.axis_progress_state["active"] = True
        self.axis_progress_state["axis_key"] = str(axis_key)
        self.axis_progress_state["label"] = f"{axis_key} axis"
        self.axis_progress_state["start_calls"] = int(self.run_calls_total)
        self.axis_progress_state["tested"] = 0
        self.axis_progress_state["kept"] = 0
        self.axis_progress_state["total"] = self._axis_total_hint(str(axis_key))
        started_at = float(pytime.perf_counter())
        self.axis_progress_state["started_at"] = started_at
        self.axis_progress_state["last_report"] = started_at
        self.axis_progress_state["last_reported_tested"] = 0
        self.axis_progress_state["report_every"] = 200
        self.axis_progress_state["heartbeat_sec"] = 20.0
        self.axis_progress_state["suppress"] = False

    def _axis_progress_record(self, *, kept: bool) -> None:
        if not bool(self.axis_progress_state.get("active")):
            return
        tested = max(
            0,
            int(self.run_calls_total)
            - int(self.axis_progress_state.get("start_calls") or 0),
        )
        if tested <= int(self.axis_progress_state.get("tested") or 0):
            return
        self.axis_progress_state["tested"] = int(tested)
        if bool(kept):
            self.axis_progress_state["kept"] = int(self.axis_progress_state.get("kept") or 0) + 1
        if bool(self.axis_progress_state.get("suppress")):
            return
        tested = int(self.axis_progress_state.get("tested") or 0)
        report_every = int(self.axis_progress_state.get("report_every") or 0)
        started_at = float(self.axis_progress_state.get("started_at") or 0.0)
        total = self.axis_progress_state.get("total")
        hit_report_every = report_every > 0 and (tested % report_every == 0)
        hit_total = isinstance(total, int) and int(total) > 0 and tested >= int(total)
        now = float(pytime.perf_counter())
        hit_heartbeat = (now - float(self.axis_progress_state.get("last_report") or started_at)) >= float(self.axis_progress_state.get("heartbeat_sec") or 20.0)
        if not (hit_report_every or hit_total or hit_heartbeat):
            return
        print(
            _progress_line(
                label=str(self.axis_progress_state.get("label") or "axis"),
                tested=int(tested),
                total=(int(total) if isinstance(total, int) else None),
                kept=int(self.axis_progress_state.get("kept") or 0),
                started_at=float(started_at),
                rate_unit="cfg/s",
            ),
            flush=True,
        )
        self.axis_progress_state["last_report"] = now
        self.axis_progress_state["last_reported_tested"] = int(tested)

    def _axis_progress_end(self) -> None:
        if not bool(self.axis_progress_state.get("active")):
            return
        tested = max(
            0,
            int(self.run_calls_total)
            - int(self.axis_progress_state.get("start_calls") or 0),
        )
        self.axis_progress_state["tested"] = int(tested)
        axis_key = str(self.axis_progress_state.get("axis_key") or "").strip().lower()
        last_reported_tested = int(self.axis_progress_state.get("last_reported_tested") or 0)
        if tested > 0 and tested != last_reported_tested:
            print(
                _progress_line(
                    label=str(self.axis_progress_state.get("label") or "axis"),
                    tested=tested,
                    total=(int(self.axis_progress_state.get("total")) if isinstance(self.axis_progress_state.get("total"), int) else None),
                    kept=int(self.axis_progress_state.get("kept") or 0),
                    started_at=float(self.axis_progress_state.get("started_at") or 0.0),
                    rate_unit="cfg/s",
                ),
                flush=True,
            )
            if axis_key:
                self.axis_progress_history[str(axis_key)] = int(tested)
                try:
                    write_json(
                        self.axis_progress_history_path,
                        self.axis_progress_history,
                        sort_keys=True,
                    )
                except Exception:
                    pass
        self.axis_progress_state["active"] = False
        self.axis_progress_state["axis_key"] = ""
        self.axis_progress_state["label"] = ""
        self.axis_progress_state["start_calls"] = 0
        self.axis_progress_state["tested"] = 0
        self.axis_progress_state["kept"] = 0
        self.axis_progress_state["total"] = None
        self.axis_progress_state["started_at"] = 0.0
        self.axis_progress_state["last_report"] = 0.0
        self.axis_progress_state["last_reported_tested"] = 0
        self.axis_progress_state["suppress"] = False
