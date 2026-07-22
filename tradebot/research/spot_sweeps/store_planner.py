"""SweepPlannerStore capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import json
import math
import time as pytime
from .fingerprints import (
    _RANK_BIN_SIZE,
)
from .support import (
    _cache_config,
    _registry_float,
)


class SweepPlannerStore:
    def _rank_bin_from_rank(self, rank: int) -> int:
        return int(max(0, int(rank)) // int(max(1, int(_RANK_BIN_SIZE))))

    def _rank_bin_runtime_get_many(
        self,
        *,
        stage_label: str,
        cells: list[tuple[str, int]],
    ) -> dict[tuple[str, int], dict[str, float]]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for window_sig, rank_bin in cells:
            window_key = str(window_sig).strip()
            try:
                bin_i = int(rank_bin)
            except (TypeError, ValueError):
                continue
            key = (window_key, int(bin_i))
            if not window_key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, int], dict[str, float]] = {}
        chunk_size = 160
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(window_signature=? AND rank_bin=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for window_sig, rank_bin in chunk:
                        params.extend((str(window_sig), int(rank_bin)))
                    rows_db = conn.execute(
                        "SELECT window_signature, rank_bin, eval_count, total_eval_sec, cache_hits "
                        "FROM rank_bin_runtime WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        window_sig = str(row[0] or "").strip()
                        try:
                            rank_bin = int(row[1])
                            eval_count = int(row[2] or 0)
                            total_eval_sec = float(row[3] or 0.0)
                            cache_hits = int(row[4] or 0)
                        except Exception:
                            continue
                        if not window_sig or eval_count <= 0:
                            continue
                        avg_eval_sec = float(total_eval_sec) / float(max(1, int(eval_count)))
                        hit_rate = float(cache_hits) / float(max(1, int(eval_count)))
                        out[(window_sig, int(rank_bin))] = {
                            "avg_eval_sec": float(avg_eval_sec),
                            "hit_rate": float(max(0.0, min(1.0, hit_rate))),
                            "eval_count": float(eval_count),
                        }
            self.rank_bin_runtime_reads += len(out)
        except Exception:
            return {}
        return out

    def _rank_bin_runtime_set_many(
        self,
        *,
        stage_label: str,
        rows: list[tuple[str, int, int, float, int]],
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, int], tuple[int, float, int]] = {}
        for window_sig, rank_bin, eval_count, total_eval_sec, cache_hits in rows:
            window_key = str(window_sig).strip()
            try:
                bin_i = int(rank_bin)
                eval_i = int(eval_count)
                sec_f = float(total_eval_sec)
                hit_i = int(cache_hits)
            except Exception:
                continue
            if not window_key or eval_i <= 0 or sec_f < 0.0:
                continue
            key = (window_key, int(bin_i))
            prev = merged.get(key)
            if prev is None:
                merged[key] = (int(eval_i), float(sec_f), int(max(0, hit_i)))
            else:
                merged[key] = (
                    int(prev[0]) + int(eval_i),
                    float(prev[1]) + float(sec_f),
                    int(prev[2]) + int(max(0, hit_i)),
                )
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(window_sig),
                int(rank_bin),
                int(rec[0]),
                float(rec[1]),
                int(rec[2]),
                now_ts,
            )
            for (window_sig, rank_bin), rec in merged.items()
        ]
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO rank_bin_runtime("
                    "stage_label, window_signature, rank_bin, eval_count, total_eval_sec, cache_hits, updated_at"
                    ") VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, window_signature, rank_bin) DO UPDATE SET "
                    "eval_count=rank_bin_runtime.eval_count + excluded.eval_count, "
                    "total_eval_sec=rank_bin_runtime.total_eval_sec + excluded.total_eval_sec, "
                    "cache_hits=rank_bin_runtime.cache_hits + excluded.cache_hits, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            self.rank_bin_runtime_writes += len(payload)
        except Exception:
            return

    def _dimension_value_utility_get_many(
        self,
        *,
        stage_label: str,
        cells: list[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], dict[str, float]]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for window_sig, dim_key, dim_value in cells:
            key = (
                str(window_sig).strip(),
                str(dim_key).strip(),
                str(dim_value).strip(),
            )
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str, str], dict[str, float]] = {}
        chunk_size = 120
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(window_signature=? AND dimension_key=? AND dimension_value=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for window_sig, dim_key, dim_value in chunk:
                        params.extend((str(window_sig), str(dim_key), str(dim_value)))
                    rows_db = conn.execute(
                        "SELECT window_signature, dimension_key, dimension_value, eval_count, keep_count, cache_hits, total_eval_sec "
                        "FROM dimension_value_utility WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        window_sig = str(row[0] or "").strip()
                        dim_key = str(row[1] or "").strip()
                        dim_value = str(row[2] or "").strip()
                        try:
                            eval_count = int(row[3] or 0)
                            keep_count = int(row[4] or 0)
                            cache_hits = int(row[5] or 0)
                            total_eval_sec = float(row[6] or 0.0)
                        except Exception:
                            continue
                        if not window_sig or not dim_key or eval_count <= 0:
                            continue
                        out[(window_sig, dim_key, dim_value)] = {
                            "eval_count": float(eval_count),
                            "keep_rate": float(
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(keep_count) / float(max(1, eval_count)),
                                    ),
                                )
                            ),
                            "hit_rate": float(
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(cache_hits) / float(max(1, eval_count)),
                                    ),
                                )
                            ),
                            "avg_eval_sec": float(total_eval_sec) / float(max(1, eval_count)),
                        }
            self.dimension_utility_reads += len(out)
        except Exception:
            return {}
        return out

    def _dimension_value_utility_set_many(
        self,
        *,
        stage_label: str,
        rows: list[tuple[str, str, str, int, int, int, float]],
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, str, str], tuple[int, int, int, float]] = {}
        for (
            window_sig,
            dim_key,
            dim_value,
            eval_count,
            keep_count,
            cache_hits,
            total_eval_sec,
        ) in rows:
            window_key = str(window_sig).strip()
            dim_key_s = str(dim_key).strip()
            dim_value_s = str(dim_value).strip()
            try:
                eval_i = int(eval_count)
                keep_i = int(keep_count)
                hit_i = int(cache_hits)
                sec_f = float(total_eval_sec)
            except Exception:
                continue
            if not window_key or not dim_key_s or eval_i <= 0 or sec_f < 0.0:
                continue
            key = (window_key, dim_key_s, dim_value_s)
            prev = merged.get(key)
            keep_i = int(max(0, min(eval_i, keep_i)))
            hit_i = int(max(0, min(eval_i, hit_i)))
            if prev is None:
                merged[key] = (int(eval_i), int(keep_i), int(hit_i), float(sec_f))
            else:
                merged[key] = (
                    int(prev[0]) + int(eval_i),
                    int(prev[1]) + int(keep_i),
                    int(prev[2]) + int(hit_i),
                    float(prev[3]) + float(sec_f),
                )
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(window_sig),
                str(dim_key),
                str(dim_value),
                int(rec[0]),
                int(rec[1]),
                int(rec[2]),
                float(rec[3]),
                now_ts,
            )
            for (window_sig, dim_key, dim_value), rec in merged.items()
        ]
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO dimension_value_utility("
                    "stage_label, window_signature, dimension_key, dimension_value, eval_count, keep_count, cache_hits, total_eval_sec, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, window_signature, dimension_key, dimension_value) DO UPDATE SET "
                    "eval_count=dimension_value_utility.eval_count + excluded.eval_count, "
                    "keep_count=dimension_value_utility.keep_count + excluded.keep_count, "
                    "cache_hits=dimension_value_utility.cache_hits + excluded.cache_hits, "
                    "total_eval_sec=dimension_value_utility.total_eval_sec + excluded.total_eval_sec, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            self.dimension_utility_writes += len(payload)
        except Exception:
            return

    def _dimension_upper_bound_get_many(
        self,
        *,
        stage_label: str,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, object]]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for axis_dim_fp, window_sig in cells:
            key = (str(axis_dim_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], dict[str, object]] = {}
        chunk_size = 140
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(axis_dimension_fingerprint=? AND window_signature=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for axis_dim_fp, window_sig in chunk:
                        params.extend((str(axis_dim_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT axis_dimension_fingerprint, window_signature, eval_count, keep_count, best_pnl_over_dd, best_pnl "
                        "FROM dimension_upper_bound WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        axis_dim_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        if not axis_dim_fp or not window_sig:
                            continue
                        out[(axis_dim_fp, window_sig)] = {
                            "eval_count": int(row[2] or 0),
                            "keep_count": int(row[3] or 0),
                            "best_pnl_over_dd": (None if row[4] is None else float(row[4])),
                            "best_pnl": (None if row[5] is None else float(row[5])),
                        }
            self.dimension_upper_bound_reads += len(out)
        except Exception:
            return {}
        return out

    def _dimension_upper_bound_upsert_many(
        self,
        *,
        stage_label: str,
        rows: list[tuple[str, str, dict | None]],
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, str], dict[str, object]] = {}
        for axis_dim_fp, window_sig, row in rows:
            axis_key = str(axis_dim_fp).strip()
            window_key = str(window_sig).strip()
            if not axis_key or not window_key:
                continue
            key = (axis_key, window_key)
            rec = merged.get(key)
            if rec is None:
                rec = {
                    "eval_count": 0,
                    "keep_count": 0,
                    "best_pnl_over_dd": None,
                    "best_pnl": None,
                }
                merged[key] = rec
            rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
            if not isinstance(row, dict):
                continue
            rec["keep_count"] = int(rec.get("keep_count") or 0) + 1
            for metric_key, rec_key, caster in (
                ("pnl_over_dd", "best_pnl_over_dd", float),
                ("pnl", "best_pnl", float),
            ):
                raw_val = row.get(metric_key)
                if raw_val is None:
                    continue
                try:
                    val = caster(raw_val)
                except (TypeError, ValueError):
                    continue
                prev = rec.get(rec_key)
                if prev is None or float(val) > float(prev):
                    rec[rec_key] = val
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(axis_dim_fp),
                str(window_sig),
                int(rec.get("eval_count") or 0),
                int(rec.get("keep_count") or 0),
                rec.get("best_pnl_over_dd"),
                rec.get("best_pnl"),
                now_ts,
            )
            for (axis_dim_fp, window_sig), rec in merged.items()
        ]
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO dimension_upper_bound("
                    "stage_label, axis_dimension_fingerprint, window_signature, eval_count, keep_count, best_pnl_over_dd, best_pnl, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, axis_dimension_fingerprint, window_signature) DO UPDATE SET "
                    "eval_count=dimension_upper_bound.eval_count + excluded.eval_count, "
                    "keep_count=dimension_upper_bound.keep_count + excluded.keep_count, "
                    "best_pnl_over_dd=CASE "
                    "WHEN dimension_upper_bound.best_pnl_over_dd IS NULL THEN excluded.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd IS NULL THEN dimension_upper_bound.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd > dimension_upper_bound.best_pnl_over_dd THEN excluded.best_pnl_over_dd "
                    "ELSE dimension_upper_bound.best_pnl_over_dd END, "
                    "best_pnl=CASE "
                    "WHEN dimension_upper_bound.best_pnl IS NULL THEN excluded.best_pnl "
                    "WHEN excluded.best_pnl IS NULL THEN dimension_upper_bound.best_pnl "
                    "WHEN excluded.best_pnl > dimension_upper_bound.best_pnl THEN excluded.best_pnl "
                    "ELSE dimension_upper_bound.best_pnl END, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            self.dimension_upper_bound_writes += len(payload)
        except Exception:
            return

    def _dimension_upper_bound_score(self, frontier_row: dict[str, object] | None) -> float:
        if not isinstance(frontier_row, dict):
            return 0.0
        cfg = _cache_config("dimension_upper_bound")
        min_eval_count = max(1, int(_registry_float(cfg.get("min_eval_count"), 6.0)))
        low_ceiling_max_keep_count = max(0, int(_registry_float(cfg.get("low_ceiling_max_keep_count"), 0.0)))
        low_ceiling_max_best_pnl = float(_registry_float(cfg.get("low_ceiling_max_best_pnl"), 0.0))
        low_ceiling_max_best_pnl_dd = float(_registry_float(cfg.get("low_ceiling_max_best_pnl_over_dd"), 0.0))
        confidence_eval_scale = max(1.0, float(_registry_float(cfg.get("confidence_eval_scale"), 24.0)))
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        if eval_count < int(min_eval_count):
            return 0.0
        best_pnl_raw = frontier_row.get("best_pnl")
        best_pnl_dd_raw = frontier_row.get("best_pnl_over_dd")
        best_pnl = float(best_pnl_raw) if best_pnl_raw is not None else float("-inf")
        best_pnl_dd = float(best_pnl_dd_raw) if best_pnl_dd_raw is not None else float("-inf")
        is_low_ceiling = bool(
            keep_count <= int(low_ceiling_max_keep_count) and best_pnl <= float(low_ceiling_max_best_pnl) and best_pnl_dd <= float(low_ceiling_max_best_pnl_dd)
        )
        if is_low_ceiling:
            return -1.0
        confidence = float(min(1.0, float(eval_count) / float(confidence_eval_scale)))
        keep_rate = float(max(0.0, min(1.0, float(keep_count) / float(max(1, eval_count)))))
        upside = (0.65 * max(0.0, float(best_pnl_dd if math.isfinite(best_pnl_dd) else 0.0))) + (
            0.35 * max(0.0, float(best_pnl if math.isfinite(best_pnl) else 0.0))
        )
        return float((confidence * upside) + (0.25 * keep_rate))

    def _upper_bound_dominance_signature(self, frontier_row: dict[str, object] | None) -> str:
        if not isinstance(frontier_row, dict):
            return ""
        score = float(self._dimension_upper_bound_score(frontier_row))
        if score >= 0.0:
            return ""
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        best_pnl_raw = frontier_row.get("best_pnl")
        best_pnl_dd_raw = frontier_row.get("best_pnl_over_dd")
        try:
            best_pnl = None if best_pnl_raw is None else round(float(best_pnl_raw), 6)
        except (TypeError, ValueError):
            best_pnl = None
        try:
            best_pnl_dd = None if best_pnl_dd_raw is None else round(float(best_pnl_dd_raw), 6)
        except (TypeError, ValueError):
            best_pnl_dd = None
        raw = {
            "rule": "upper_bound_low_ceiling",
            "eval_count": int(eval_count),
            "keep_count": int(keep_count),
            "best_pnl": best_pnl,
            "best_pnl_over_dd": best_pnl_dd,
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _planner_heartbeat_set(
        self,
        *,
        stage_label: str,
        worker_id: int,
        tested: int,
        cached_hits: int,
        total: int,
        eta_sec: float | None,
        status: str,
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return
        try:
            worker_i = int(worker_id)
            tested_i = int(max(0, int(tested)))
            cached_i = int(max(0, int(cached_hits)))
            total_i = int(max(0, int(total)))
            eta_f = None if eta_sec is None else float(max(0.0, float(eta_sec)))
            status_s = str(status or "").strip().lower() or "running"
        except Exception:
            return
        now_ts = float(pytime.time())
        try:
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "INSERT INTO planner_heartbeat("
                    "stage_label, worker_id, last_seen, tested, cached_hits, total, eta_sec, status, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, worker_id) DO UPDATE SET "
                    "last_seen=excluded.last_seen, "
                    "tested=excluded.tested, "
                    "cached_hits=excluded.cached_hits, "
                    "total=excluded.total, "
                    "eta_sec=excluded.eta_sec, "
                    "status=excluded.status, "
                    "updated_at=excluded.updated_at",
                    (
                        str(stage_key),
                        int(worker_i),
                        float(now_ts),
                        int(tested_i),
                        int(cached_i),
                        int(total_i),
                        (float(eta_f) if eta_f is not None else None),
                        str(status_s),
                        float(now_ts),
                    ),
                )
            self.planner_heartbeat_writes += 1
        except Exception:
            return

    def _planner_heartbeat_clear_stage(self, *, stage_label: str) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return
        try:
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "DELETE FROM planner_heartbeat WHERE stage_label=?",
                    (str(stage_key),),
                )
        except Exception:
            return

    def _planner_heartbeat_get_many(
        self,
        *,
        stage_label: str,
        worker_ids: list[int],
    ) -> dict[int, dict[str, float | int | str | None]]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        worker_ids_clean: list[int] = []
        seen: set[int] = set()
        for raw in worker_ids:
            try:
                worker_i = int(raw)
            except (TypeError, ValueError):
                continue
            if worker_i < 0 or worker_i in seen:
                continue
            seen.add(worker_i)
            worker_ids_clean.append(int(worker_i))
        if not worker_ids_clean:
            return {}
        out: dict[int, dict[str, float | int | str | None]] = {}
        chunk_size = 160
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(worker_ids_clean), chunk_size):
                    chunk = worker_ids_clean[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        "SELECT worker_id, last_seen, tested, cached_hits, total, eta_sec, status "
                        "FROM planner_heartbeat WHERE stage_label=? AND worker_id IN (" + placeholders + ")",
                        tuple([str(stage_key), *[int(worker_i) for worker_i in chunk]]),
                    ).fetchall()
                    for row in rows:
                        try:
                            worker_i = int(row[0])
                            last_seen = float(row[1] or 0.0)
                            tested_i = int(row[2] or 0)
                            cached_i = int(row[3] or 0)
                            total_i = int(row[4] or 0)
                            eta_raw = row[5]
                            eta_f = None if eta_raw is None else float(max(0.0, float(eta_raw)))
                            status_s = str(row[6] or "").strip().lower()
                        except Exception:
                            continue
                        out[int(worker_i)] = {
                            "last_seen": float(last_seen),
                            "tested": int(tested_i),
                            "cached_hits": int(cached_i),
                            "total": int(total_i),
                            "eta_sec": (float(eta_f) if eta_f is not None else None),
                            "status": str(status_s),
                        }
            self.planner_heartbeat_reads += len(out)
        except Exception:
            return {}
        return out
