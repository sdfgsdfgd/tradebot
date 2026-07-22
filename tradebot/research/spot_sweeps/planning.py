"""SweepPlanning capability slice for the canonical spot research runtime."""

from __future__ import annotations

import json
import math
from ...backtest.config import (
    ConfigBundle,
)
from ...backtest.spot_codec import (
    filters_from_payload as _codec_filters_from_payload,
    make_bundle as _codec_make_bundle,
    strategy_from_payload as _codec_strategy_from_payload,
)
from .fingerprints import (
    _axis_dimension_fingerprint,
)
from .milestones import (
    _filters_payload,
    _spot_strategy_payload,
)
from .support import (
    _axis_cost_hint,
    _cost_model_weight,
    _registry_float,
    _runtime_policy,
)


class SweepPlanning:
    def _cfg_from_strategy_filters_payload(
        self,
        strategy_payload,
        filters_payload,
        *,
        bar_size: str | None = None,
        use_rth: bool | None = None,
    ) -> ConfigBundle | None:
        if not isinstance(strategy_payload, dict):
            return None
        try:
            filters_obj = _codec_filters_from_payload(
                filters_payload if isinstance(filters_payload, dict) else None
            )
            strategy_obj = _codec_strategy_from_payload(
                strategy_payload, filters=filters_obj
            )
        except Exception:
            return None
        return _codec_make_bundle(
            strategy=strategy_obj,
            start=self.start,
            end=self.end,
            bar_size=str(bar_size or self.signal_bar_size),
            use_rth=bool(self.use_rth if use_rth is None else use_rth),
            cache_dir=self.cache_dir,
            offline=self.offline,
        )

    def _rows_from_kept(
        self, kept: list[tuple[ConfigBundle, dict, str, dict | None]]
    ) -> list[tuple[ConfigBundle, dict, str]]:
        out: list[tuple[ConfigBundle, dict, str]] = []
        for cfg, row, note, _meta in kept:
            out.append((cfg, row, note))
        return out

    def _encode_cfg_payload(
        self,
        cfg: ConfigBundle,
        *,
        note: str | None = None,
        note_key: str = "note",
        extra: dict | None = None,
    ) -> dict:
        payload = {
            "backtest": {
                "bar_size": str(cfg.backtest.bar_size),
                "use_rth": bool(cfg.backtest.use_rth),
            },
            "strategy": _spot_strategy_payload(cfg, meta=self.meta),
            "filters": _filters_payload(cfg.strategy.filters),
        }
        if note is not None:
            payload[str(note_key)] = str(note)
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def _decode_cfg_payload(
        self,
        payload: object,
        *,
        note_key: str = "note",
        default_note: str = "",
        cfg_catalog: dict[str, tuple[dict, dict | None]] | None = None,
    ) -> tuple[ConfigBundle, str] | None:
        if not isinstance(payload, dict):
            return None
        strategy_payload = payload.get("strategy")
        filters_payload = payload.get("filters")
        backtest_payload = payload.get("backtest")
        if not isinstance(backtest_payload, dict):
            backtest_payload = {}
        if filters_payload is not None and not isinstance(filters_payload, dict):
            filters_payload = None
        cfg_ref = str(payload.get("cfg_ref") or "").strip()
        if cfg_ref and isinstance(cfg_catalog, dict):
            resolved = cfg_catalog.get(cfg_ref)
            if isinstance(resolved, tuple) and len(resolved) == 2:
                strategy_payload, filters_payload = resolved
        cfg = self._cfg_from_strategy_filters_payload(
            strategy_payload,
            filters_payload,
            bar_size=str(backtest_payload.get("bar_size") or self.signal_bar_size),
            use_rth=(
                bool(backtest_payload["use_rth"])
                if "use_rth" in backtest_payload
                else bool(self.use_rth)
            ),
        )
        if cfg is None:
            return None
        note = str(payload.get(note_key) or "").strip() or str(default_note)
        return cfg, note

    def _cfg_catalog_from_payload(
        self, payload: dict
    ) -> dict[str, tuple[dict, dict | None]]:
        out: dict[str, tuple[dict, dict | None]] = {}
        cfg_catalog_raw = payload.get("_cfg_catalog")
        if not isinstance(cfg_catalog_raw, list):
            return out
        for item in cfg_catalog_raw:
            if not isinstance(item, dict):
                continue
            cfg_ref = str(item.get("cfg_ref") or "").strip()
            strategy_payload = item.get("strategy")
            filters_payload = item.get("filters")
            if not cfg_ref or not isinstance(strategy_payload, dict):
                continue
            if filters_payload is not None and not isinstance(filters_payload, dict):
                filters_payload = None
            out[cfg_ref] = (
                dict(strategy_payload),
                dict(filters_payload) if isinstance(filters_payload, dict) else None,
            )
        return out

    def _worker_records_from_kept(
        self, kept: list[tuple[ConfigBundle, dict, str, dict | None]]
    ) -> list[dict]:
        records: list[dict] = []
        for cfg, row, note, _meta in kept:
            records.append(self._encode_cfg_payload(cfg, note=note, extra={"row": row}))
        return records

    def _plan_item_cfg(self, item) -> ConfigBundle | None:
        if (
            isinstance(item, tuple)
            and len(item) >= 1
            and isinstance(item[0], ConfigBundle)
        ):
            return item[0]
        return None

    def _plan_item_meta(self, item) -> dict | None:
        if isinstance(item, tuple) and len(item) >= 3 and isinstance(item[2], dict):
            return item[2]
        return None

    def _plan_item_mixed_radix_rank(self, item) -> int | None:
        meta_item = self._plan_item_meta(item)
        if not isinstance(meta_item, dict):
            return None
        raw = meta_item.get("_mr_rank")
        try:
            out = int(raw)
        except (TypeError, ValueError):
            return None
        if out < 0:
            return None
        return int(out)

    def _plan_item_dimension_values(self, item) -> tuple[tuple[str, str], ...]:
        meta_item = self._plan_item_meta(item)
        if not isinstance(meta_item, dict):
            return ()
        out: list[tuple[str, str]] = []
        for raw_key, raw_val in meta_item.items():
            key = str(raw_key or "").strip()
            if not key or key.startswith("_"):
                continue
            out.append((str(key), str(raw_val)))
        out.sort(key=lambda row: row[0])
        return tuple(out)

    def _mixed_radix_warm_ranges(
        self,
        *,
        plan_all,
        max_bins: int = 24,
    ) -> tuple[tuple[int, int, int], ...]:
        ranks: list[int] = []
        for item in plan_all:
            rank = self._plan_item_mixed_radix_rank(item)
            if rank is None:
                continue
            ranks.append(int(rank))
        if not ranks:
            return ()
        r_min = int(min(ranks))
        r_max = int(max(ranks))
        span_total = max(1, int(r_max - r_min + 1))
        bins_target = max(1, min(int(max_bins), max(1, int(math.sqrt(len(ranks))))))
        bin_span = max(1, int(math.ceil(float(span_total) / float(bins_target))))
        bin_counts: dict[int, int] = {}
        for rank in ranks:
            b = int((int(rank) - int(r_min)) // int(bin_span))
            bin_counts[b] = int(bin_counts.get(b, 0)) + 1
        ordered_bins = sorted(
            bin_counts.items(), key=lambda row: (-int(row[1]), int(row[0]))
        )
        out: list[tuple[int, int, int]] = []
        for b, count in ordered_bins:
            lo = int(r_min + (int(b) * int(bin_span)))
            hi = int(min(int(r_max), int(lo + int(bin_span) - 1)))
            out.append((int(lo), int(hi), int(count)))
        return tuple(out)

    def _compress_rank_status_rows(
        self, rank_status: dict[int, str]
    ) -> list[tuple[int, int, str]]:
        rows = [
            (int(rank), str(status))
            for rank, status in rank_status.items()
            if str(status) in ("cached_hit", "evaluated", "dominated")
        ]
        if not rows:
            return []
        rows.sort(key=lambda row: int(row[0]))
        out: list[tuple[int, int, str]] = []
        cur_lo = int(rows[0][0])
        cur_hi = int(rows[0][0])
        cur_status = str(rows[0][1])
        for rank, status in rows[1:]:
            rank_i = int(rank)
            status_s = str(status)
            if status_s == cur_status and rank_i == (int(cur_hi) + 1):
                cur_hi = int(rank_i)
                continue
            out.append((int(cur_lo), int(cur_hi), str(cur_status)))
            cur_lo = int(rank_i)
            cur_hi = int(rank_i)
            cur_status = str(status_s)
        out.append((int(cur_lo), int(cur_hi), str(cur_status)))
        return out

    def _partition_rank_ranges_for_workers(
        self,
        *,
        ranges: tuple[tuple[int, int], ...],
        workers: int,
    ) -> list[list[tuple[int, int]]]:
        workers_n = max(1, int(workers))
        out: list[list[tuple[int, int]]] = [[] for _ in range(workers_n)]
        if workers_n <= 1:
            out[0] = list(ranges)
            return out
        loads: list[int] = [0 for _ in range(workers_n)]
        ordered = sorted(
            [(int(lo), int(hi)) for lo, hi in ranges if int(hi) >= int(lo)],
            key=lambda row: (-(int(row[1]) - int(row[0]) + 1), int(row[0])),
        )
        total_span = sum(
            int(rank_hi) - int(rank_lo) + 1 for rank_lo, rank_hi in ordered
        )
        # Split into finer chunks than worker count so each worker receives multiple disjoint
        # ranges; this reduces straggler risk when rank-local cost is non-uniform.
        target_chunk_span = max(
            1, int(math.ceil(float(total_span) / float(max(1, workers_n * 4))))
        )
        chunks: list[tuple[int, int]] = []
        for rank_lo, rank_hi in ordered:
            cursor = int(rank_lo)
            hi_i = int(rank_hi)
            while int(cursor) <= int(hi_i):
                chunk_hi = min(int(hi_i), int(cursor) + int(target_chunk_span) - 1)
                chunks.append((int(cursor), int(chunk_hi)))
                cursor = int(chunk_hi) + 1
        for rank_lo, rank_hi in chunks:
            span = int(rank_hi) - int(rank_lo) + 1
            target = min(range(workers_n), key=lambda wid: (int(loads[wid]), int(wid)))
            out[int(target)].append((int(rank_lo), int(rank_hi)))
            loads[int(target)] += int(span)
        for bucket in out:
            bucket.sort(key=lambda row: int(row[0]))
        return out

    def _cfg_eval_cost_hint(self, cfg: ConfigBundle) -> float:
        strat = cfg.strategy
        signal_bar = str(cfg.backtest.bar_size)
        filters_payload = _filters_payload(strat.filters) or {}
        axis_fp = _axis_dimension_fingerprint(cfg)
        persisted_cost = float(
            self.run_cfg_dim_index_loaded.get(str(axis_fp), 0.0) or 0.0
        )
        cost = _cost_model_weight("base", 1.0)

        regime_mode = str(getattr(strat, "regime_mode", "ema") or "ema").strip().lower()
        regime_bar = (
            str(getattr(strat, "regime_bar_size", "") or "").strip() or signal_bar
        )
        if regime_bar != signal_bar and (
            regime_mode == "supertrend"
            or bool(getattr(strat, "regime_ema_preset", None))
        ):
            cost += _cost_model_weight("regime_cross_tf", 0.5)

        regime2_mode = (
            str(getattr(strat, "regime2_mode", "off") or "off").strip().lower()
        )
        regime2_bar = (
            str(getattr(strat, "regime2_bar_size", "") or "").strip() or signal_bar
        )
        if regime2_mode != "off" and regime2_bar != signal_bar:
            cost += _cost_model_weight("regime2_cross_tf", 0.5)

        if (
            str(getattr(strat, "tick_gate_mode", "off") or "off").strip().lower()
            != "off"
        ):
            cost += _cost_model_weight("tick_gate_on", 0.75)

        exec_size = str(getattr(strat, "spot_exec_bar_size", "") or "").strip()
        if exec_size and exec_size != signal_bar:
            cost += _cost_model_weight("exec_cross_tf", 0.75)

        perm_on = any(
            filters_payload.get(k) is not None
            for k in (
                "ema_spread_min_pct",
                "ema_spread_min_pct_down",
                "ema_slope_min_pct",
                "ema_slope_signed_min_pct_up",
                "ema_slope_signed_min_pct_down",
            )
        )
        if perm_on:
            cost += _cost_model_weight("perm_gate_on", 0.15) * _axis_cost_hint(
                "perm_joint", "perm_variants", 1.0
            )

        if (
            filters_payload.get("entry_start_hour_et") is not None
            and filters_payload.get("entry_end_hour_et") is not None
        ):
            cost += _cost_model_weight("tod_gate_on", 0.12) * _axis_cost_hint(
                "perm_joint", "tod_windows", 1.0
            )

        if filters_payload.get("volume_ratio_min") is not None:
            cost += _cost_model_weight("volume_gate_on", 0.08) * _axis_cost_hint(
                "perm_joint", "vol_variants", 1.0
            )

        cooldown_raw = filters_payload.get("cooldown_bars")
        skip_raw = filters_payload.get("skip_first_bars")
        try:
            cooldown_on = float(cooldown_raw or 0.0) > 0.0
        except (TypeError, ValueError):
            cooldown_on = bool(cooldown_raw)
        try:
            skip_on = float(skip_raw or 0.0) > 0.0
        except (TypeError, ValueError):
            skip_on = bool(skip_raw)
        if cooldown_on or skip_on:
            cost += _cost_model_weight("cadence_gate_on", 0.05) * _axis_cost_hint(
                "perm_joint", "cadence_variants", 1.0
            )

        if (
            str(filters_payload.get("shock_gate_mode") or "off").strip().lower()
            != "off"
        ):
            cost += _cost_model_weight("shock_gate_on", 0.4) * _axis_cost_hint(
                "shock", "modes", 1.0
            )
        if filters_payload.get("riskoff_tr5_med_pct") is not None:
            cost += _cost_model_weight("riskoff_overlay_on", 0.2) * _axis_cost_hint(
                "risk_overlays", "riskoff", 1.0
            )
        if (
            filters_payload.get("riskpanic_tr5_med_pct") is not None
            and filters_payload.get("riskpanic_neg_gap_ratio_min") is not None
        ):
            cost += _cost_model_weight("riskpanic_overlay_on", 0.35) * _axis_cost_hint(
                "risk_overlays", "riskpanic", 1.0
            )
        if (
            filters_payload.get("riskpop_tr5_med_pct") is not None
            and filters_payload.get("riskpop_pos_gap_ratio_min") is not None
        ):
            cost += _cost_model_weight("riskpop_overlay_on", 0.3) * _axis_cost_hint(
                "risk_overlays", "riskpop", 1.0
            )
        if persisted_cost > 0.0:
            return float((0.65 * float(persisted_cost)) + (0.35 * float(cost)))
        return float(cost)

    def _cfg_locality_bucket_key(self, cfg: ConfigBundle) -> str:
        strat = cfg.strategy
        signal_bar = str(cfg.backtest.bar_size)
        regime_mode = str(getattr(strat, "regime_mode", "ema") or "ema").strip().lower()
        regime_bar = (
            str(getattr(strat, "regime_bar_size", "") or "").strip() or signal_bar
        )
        regime2_mode = (
            str(getattr(strat, "regime2_mode", "off") or "off").strip().lower()
        )
        regime2_bar = (
            str(getattr(strat, "regime2_bar_size", "") or "").strip() or signal_bar
        )
        tick_mode = (
            str(getattr(strat, "tick_gate_mode", "off") or "off").strip().lower()
        )
        exec_size = (
            str(getattr(strat, "spot_exec_bar_size", "") or "").strip() or signal_bar
        )
        filters_payload = _filters_payload(strat.filters) or {}
        raw = {
            "signal_bar": signal_bar,
            "regime": (regime_mode, regime_bar),
            "regime2": (regime2_mode, regime2_bar),
            "tick_mode": tick_mode,
            "exec_size": exec_size,
            "shock_mode": str(filters_payload.get("shock_gate_mode") or "off")
            .strip()
            .lower(),
            "risk_profile": (
                filters_payload.get("riskoff_tr5_med_pct") is not None,
                filters_payload.get("riskpanic_tr5_med_pct") is not None,
                filters_payload.get("riskpop_tr5_med_pct") is not None,
            ),
            "axis_dim_fp": _axis_dimension_fingerprint(cfg),
        }
        return json.dumps(raw, sort_keys=True, default=str)

    def _dimension_value_utility_score(
        self,
        row: dict[str, float] | None,
        *,
        cfg: dict[str, object] | None = None,
    ) -> float:
        if not isinstance(row, dict):
            return 0.0
        cfg_eff = (
            cfg if isinstance(cfg, dict) else _runtime_policy("dimension_value_utility")
        )
        min_eval_count = int(_registry_float(cfg_eff.get("min_eval_count"), 6.0))
        weight_keep = float(_registry_float(cfg_eff.get("weight_keep_rate"), 0.70))
        weight_hit = float(_registry_float(cfg_eff.get("weight_hit_rate"), 0.20))
        weight_conf = float(_registry_float(cfg_eff.get("weight_confidence"), 0.10))
        confidence_eval_scale = float(
            _registry_float(cfg_eff.get("confidence_eval_scale"), 24.0)
        )
        eval_floor = float(_registry_float(cfg_eff.get("eval_sec_floor"), 0.01))

        eval_count = float(max(0.0, float(row.get("eval_count", 0.0) or 0.0)))
        if eval_count < float(max(1, min_eval_count)):
            return 0.0
        keep_rate = float(max(0.0, min(1.0, float(row.get("keep_rate", 0.0) or 0.0))))
        hit_rate = float(max(0.0, min(1.0, float(row.get("hit_rate", 0.0) or 0.0))))
        avg_eval_sec = float(
            max(eval_floor, float(row.get("avg_eval_sec", 0.0) or 0.0))
        )
        confidence = float(
            min(1.0, eval_count / float(max(1.0, confidence_eval_scale)))
        )
        numerator = (
            (weight_keep * keep_rate)
            + (weight_hit * hit_rate)
            + (weight_conf * confidence)
        )
        return float(max(0.0, numerator) / max(eval_floor, avg_eval_sec))

    def _worker_bucketed_indices(
        self,
        *,
        plan_all,
        workers: int,
        bars: list | None = None,
        stage_label: str = "",
        warm_ranges: tuple[tuple[int, int, int], ...] | None = None,
    ) -> list[list[int]]:
        if not bool(self.run_cfg_dim_index_loaded_once):
            loaded = self._run_cfg_dimension_index_load()
            if loaded:
                self.run_cfg_dim_index_loaded.update(loaded)
            self.run_cfg_dim_index_loaded_once = True
        workers_n = max(1, int(workers))
        total = len(plan_all)
        if workers_n <= 1:
            return [list(range(total))]
        if total <= 1:
            out_small: list[list[int]] = [[] for _ in range(workers_n)]
            if total == 1:
                out_small[0].append(0)
            return out_small

        grouped: dict[str, list[tuple[int, float, float]]] = {}
        stage_key = self._stage_cache_scope(stage_label)
        plan_items: list[
            tuple[
                int,
                str,
                float,
                tuple[str, str, str] | None,
                str | None,
                str | None,
                tuple[tuple[str, str], ...],
                int | None,
            ]
        ] = []
        persistent_keys: list[str] = []
        for idx, item in enumerate(plan_all):
            cfg = self._plan_item_cfg(item)
            mixed_radix_rank = self._plan_item_mixed_radix_rank(item)
            dimension_pairs = self._plan_item_dimension_values(item)
            if cfg is None:
                cost = 1.0
                bucket_key = "default"
                plan_items.append(
                    (
                        int(idx),
                        str(bucket_key),
                        float(cost),
                        None,
                        None,
                        None,
                        dimension_pairs,
                        mixed_radix_rank,
                    )
                )
            else:
                cost = self._cfg_eval_cost_hint(cfg)
                bucket_key = self._cfg_locality_bucket_key(cfg)
                _ctx_sig, cache_key, _axis_dim_fp, persistent_key = (
                    self._run_cfg_cache_coords(
                        cfg=cfg,
                        bars=bars,
                        update_dim_index=False,
                    )
                )
                plan_items.append(
                    (
                        int(idx),
                        str(bucket_key),
                        float(cost),
                        cache_key,
                        str(persistent_key),
                        str(cache_key[2]),
                        dimension_pairs,
                        mixed_radix_rank,
                    )
                )
                persistent_keys.append(str(persistent_key))

        persisted_by_key = (
            self._run_cfg_persistent_get_many(cache_keys=persistent_keys)
            if persistent_keys
            else {}
        )
        runtime_cells: list[tuple[str, int]] = []
        utility_cells: list[tuple[str, str, str]] = []
        if stage_key:
            for (
                _idx,
                _bucket_key,
                _cost,
                cache_key,
                _persistent_key,
                window_sig,
                dimension_pairs,
                mixed_radix_rank,
            ) in plan_items:
                if cache_key is None or window_sig is None or mixed_radix_rank is None:
                    if window_sig is not None and dimension_pairs:
                        for dim_key, dim_value in dimension_pairs:
                            utility_cells.append(
                                (str(window_sig), str(dim_key), str(dim_value))
                            )
                    continue
                runtime_cells.append(
                    (
                        str(window_sig),
                        int(self._rank_bin_from_rank(int(mixed_radix_rank))),
                    )
                )
                if dimension_pairs:
                    for dim_key, dim_value in dimension_pairs:
                        utility_cells.append(
                            (str(window_sig), str(dim_key), str(dim_value))
                        )
        runtime_by_cell = (
            self._rank_bin_runtime_get_many(
                stage_label=str(stage_key), cells=runtime_cells
            )
            if runtime_cells
            else {}
        )
        dim_utility_cfg = _runtime_policy("dimension_value_utility")
        dim_utility_by_cell = (
            self._dimension_value_utility_get_many(
                stage_label=str(stage_key), cells=utility_cells
            )
            if utility_cells
            else {}
        )
        prepared_items: list[tuple[int, str, float, int | None, float]] = []
        for (
            idx,
            bucket_key,
            cost,
            cache_key,
            persistent_key,
            window_sig,
            dimension_pairs,
            mixed_radix_rank,
        ) in plan_items:
            adjusted_cost = float(cost)
            utility_score = 0.0
            utility_scores: list[float] = []
            if window_sig is not None and dimension_pairs:
                for dim_key, dim_value in dimension_pairs:
                    utility_row = dim_utility_by_cell.get(
                        (str(window_sig), str(dim_key), str(dim_value))
                    )
                    utility = self._dimension_value_utility_score(
                        utility_row, cfg=dim_utility_cfg
                    )
                    if utility > 0.0:
                        utility_scores.append(float(utility))
                if utility_scores:
                    utility_score = float(
                        sum(float(v) for v in utility_scores)
                        / float(max(1, len(utility_scores)))
                    )
                    self.dimension_utility_hint_hits += len(utility_scores)
            if cache_key is not None and persistent_key is not None:
                cached = self.run_cfg_cache.get(cache_key, self._RUN_CFG_CACHE_MISS)
                if cached is self._RUN_CFG_CACHE_MISS:
                    persisted = persisted_by_key.get(
                        str(persistent_key), self._RUN_CFG_CACHE_MISS
                    )
                    if persisted is not self._RUN_CFG_CACHE_MISS:
                        self.run_cfg_cache[cache_key] = (
                            persisted if isinstance(persisted, dict) else None
                        )
                        cached = persisted
                if cached is not self._RUN_CFG_CACHE_MISS:
                    adjusted_cost = max(0.001, float(adjusted_cost) * 0.03)
                else:
                    if window_sig is not None and mixed_radix_rank is not None:
                        hint = runtime_by_cell.get(
                            (
                                str(window_sig),
                                int(self._rank_bin_from_rank(int(mixed_radix_rank))),
                            )
                        )
                        if isinstance(hint, dict):
                            avg_eval_sec = float(hint.get("avg_eval_sec", 0.0) or 0.0)
                            hit_rate = float(hint.get("hit_rate", 0.0) or 0.0)
                            if avg_eval_sec > 0.0:
                                adjusted_cost = max(
                                    0.001,
                                    (0.45 * float(adjusted_cost))
                                    + (0.55 * float(avg_eval_sec)),
                                )
                            if hit_rate > 0.0:
                                adjusted_cost = max(
                                    0.001,
                                    float(adjusted_cost)
                                    * (1.0 - (0.55 * max(0.0, min(1.0, hit_rate)))),
                                )
            prepared_items.append(
                (
                    int(idx),
                    str(bucket_key),
                    float(adjusted_cost),
                    mixed_radix_rank,
                    float(utility_score),
                )
            )
            grouped.setdefault(str(bucket_key), []).append(
                (int(idx), float(adjusted_cost), float(utility_score))
            )

        mixed_radix_ready = len(prepared_items) > 0 and all(
            (row[3] is not None) for row in prepared_items
        )
        if bool(mixed_radix_ready):
            warm_ranges_eff = tuple(warm_ranges or ())

            def _range_rank_key(rank: int) -> tuple[int, int]:
                if not warm_ranges_eff:
                    return (0, int(rank))
                for idx_r, (lo, hi, _count) in enumerate(warm_ranges_eff):
                    if int(lo) <= int(rank) <= int(hi):
                        return (int(idx_r), int(rank))
                return (len(warm_ranges_eff), int(rank))

            sorted_items = sorted(
                prepared_items,
                key=lambda row: (
                    *_range_rank_key(int(row[3] or 0)),
                    -float(row[4]),
                    int(row[0]),
                ),
            )
            total_cost = sum(float(row[2]) for row in sorted_items)
            target_cost = max(0.001, float(total_cost) / float(max(1, workers_n)))
            out_ranges: list[list[int]] = [[] for _ in range(workers_n)]
            worker_id = 0
            worker_cost = 0.0
            for pos, (
                idx,
                _bucket_key,
                adjusted_cost,
                _rank,
                _utility_score,
            ) in enumerate(sorted_items):
                remaining_items = int(len(sorted_items) - int(pos))
                remaining_workers = int(workers_n - int(worker_id))
                if (
                    int(worker_id) < int(workers_n - 1)
                    and float(worker_cost) >= float(target_cost)
                    and int(remaining_items) > int(remaining_workers - 1)
                ):
                    worker_id += 1
                    worker_cost = 0.0
                out_ranges[int(worker_id)].append(int(idx))
                worker_cost += float(adjusted_cost)
            return out_ranges

        locality_buckets: list[tuple[str, float, list[tuple[int, float, float]]]] = []
        for bucket_key, items in grouped.items():
            items_sorted = sorted(
                items, key=lambda row: (-float(row[2]), -float(row[1]), int(row[0]))
            )
            bucket_cost = sum(
                float(cost) for _idx, cost, _utility_score in items_sorted
            )
            locality_buckets.append((str(bucket_key), float(bucket_cost), items_sorted))
        locality_buckets.sort(key=lambda row: (-float(row[1]), str(row[0])))

        loads = [0.0] * workers_n
        counts = [0] * workers_n
        out: list[list[int]] = [[] for _ in range(workers_n)]
        for _bucket_key, bucket_cost, items in locality_buckets:
            target = min(
                range(workers_n), key=lambda wid: (loads[wid], counts[wid], wid)
            )
            out[target].extend(int(idx) for idx, _cost, _utility_score in items)
            loads[target] += float(bucket_cost)
            counts[target] += len(items)
        return out

    def _ordered_plan_indices_by_dimension_utility(
        self,
        *,
        stage_label: str,
        plan_all,
        bars: list | None = None,
    ) -> list[int]:
        stage_key = self._stage_cache_scope(stage_label)
        total = len(plan_all)
        if total <= 1 or not stage_key:
            return list(range(total))
        utility_cfg = _runtime_policy("dimension_value_utility")
        utility_cells: list[tuple[str, str, str]] = []
        item_meta: list[tuple[int, str, tuple[tuple[str, str], ...], int | None]] = []
        for idx, item in enumerate(plan_all):
            cfg = self._plan_item_cfg(item)
            if cfg is None:
                continue
            dim_pairs = self._plan_item_dimension_values(item)
            if not dim_pairs:
                continue
            _ctx_sig, cache_key, _axis_dim_fp, _persistent_key = (
                self._run_cfg_cache_coords(
                    cfg=cfg,
                    bars=bars,
                    update_dim_index=False,
                )
            )
            window_sig = str(cache_key[2])
            if not window_sig:
                continue
            for dim_key, dim_value in dim_pairs:
                utility_cells.append((str(window_sig), str(dim_key), str(dim_value)))
            item_meta.append(
                (
                    int(idx),
                    str(window_sig),
                    dim_pairs,
                    self._plan_item_mixed_radix_rank(item),
                )
            )
        if not item_meta:
            return list(range(total))
        utility_by_cell = self._dimension_value_utility_get_many(
            stage_label=str(stage_key), cells=utility_cells
        )
        score_by_idx: dict[int, float] = {}
        rank_by_idx: dict[int, int] = {}
        for idx, window_sig, dim_pairs, mr_rank in item_meta:
            scores: list[float] = []
            for dim_key, dim_value in dim_pairs:
                row = utility_by_cell.get(
                    (str(window_sig), str(dim_key), str(dim_value))
                )
                score = self._dimension_value_utility_score(row, cfg=utility_cfg)
                if score > 0.0:
                    scores.append(float(score))
            if scores:
                score_by_idx[int(idx)] = float(
                    sum(float(v) for v in scores) / float(max(1, len(scores)))
                )
            if isinstance(mr_rank, int):
                rank_by_idx[int(idx)] = int(mr_rank)
        if not score_by_idx:
            return list(range(total))
        sentinel_rank = int(10**18)
        return sorted(
            list(range(total)),
            key=lambda idx: (
                -float(score_by_idx.get(int(idx), 0.0)),
                int(rank_by_idx.get(int(idx), sentinel_rank)),
                int(idx),
            ),
        )

    def _ordered_plan_indices_by_upper_bound(
        self,
        *,
        stage_label: str,
        plan_all,
        bars: list | None = None,
    ) -> tuple[list[int], int]:
        stage_key = self._stage_cache_scope(stage_label)
        total = len(plan_all)
        if total <= 1 or not stage_key:
            return list(range(total)), 0
        bound_cells: list[tuple[str, str]] = []
        item_meta: list[tuple[int, str, str, int | None]] = []
        for idx, item in enumerate(plan_all):
            cfg = self._plan_item_cfg(item)
            if cfg is None:
                continue
            _ctx_sig, cache_key, axis_dim_fp, _persistent_key = (
                self._run_cfg_cache_coords(
                    cfg=cfg,
                    bars=bars,
                    update_dim_index=False,
                )
            )
            window_sig = str(cache_key[2])
            axis_fp = str(axis_dim_fp)
            if not window_sig or not axis_fp:
                continue
            bound_cells.append((axis_fp, window_sig))
            item_meta.append(
                (int(idx), axis_fp, window_sig, self._plan_item_mixed_radix_rank(item))
            )
        if not item_meta:
            return list(range(total)), 0
        bound_by_cell = self._dimension_upper_bound_get_many(
            stage_label=str(stage_key), cells=bound_cells
        )
        score_by_idx: dict[int, float] = {}
        rank_by_idx: dict[int, int] = {}
        deferred = 0
        for idx, axis_fp, window_sig, mr_rank in item_meta:
            row = bound_by_cell.get((str(axis_fp), str(window_sig)))
            score = float(self._dimension_upper_bound_score(row))
            score_by_idx[int(idx)] = float(score)
            if score < 0.0:
                deferred += 1
            if isinstance(mr_rank, int):
                rank_by_idx[int(idx)] = int(mr_rank)
        if not score_by_idx:
            return list(range(total)), 0
        if deferred > 0:
            self.dimension_upper_bound_deferred += int(deferred)
        sentinel_rank = int(10**18)
        ordered = sorted(
            list(range(total)),
            key=lambda idx: (
                -(max(0.0, float(score_by_idx.get(int(idx), 0.0)))),
                1 if float(score_by_idx.get(int(idx), 0.0)) < 0.0 else 0,
                int(rank_by_idx.get(int(idx), sentinel_rank)),
                int(idx),
            ),
        )
        return ordered, int(deferred)
