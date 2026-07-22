"""SweepStatusStore capability slice for the canonical spot research runtime."""

from __future__ import annotations

import time as pytime
from .support import (
    _cache_config,
    _registry_float,
)


class SweepStatusStore:
    def _stage_cell_status_get_many(
        self,
        *,
        stage_label: str,
        cells: list[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], str]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for strategy_fp, axis_dim_fp, window_sig in cells:
            cell_key = (str(strategy_fp), str(axis_dim_fp), str(window_sig))
            if cell_key in seen:
                continue
            seen.add(cell_key)
            deduped.append(cell_key)
        if not deduped:
            return {}
        out: dict[tuple[str, str, str], str] = {}
        chunk_size = 120
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(strategy_fingerprint=? AND axis_dimension_fingerprint=? AND window_signature=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for strategy_fp, axis_dim_fp, window_sig in chunk:
                        params.extend((str(strategy_fp), str(axis_dim_fp), str(window_sig)))
                    rows = conn.execute(
                        "SELECT strategy_fingerprint, axis_dimension_fingerprint, window_signature, status "
                        "FROM stage_cell_status WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows:
                        cell = (str(row[0] or ""), str(row[1] or ""), str(row[2] or ""))
                        status = str(row[3] or "").strip().lower()
                        if not cell[0] or not cell[1] or not cell[2] or status not in self._STAGE_CELL_STATUS_VALUES:
                            continue
                        out[cell] = status
            self.stage_cell_status_reads += len(out)
        except Exception:
            return {}
        return out

    def _stage_cell_status_set_many(
        self,
        *,
        stage_label: str,
        rows: list[tuple[str, str, str, str]],
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        status_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2}
        merged: dict[tuple[str, str, str], str] = {}
        for strategy_fp, axis_dim_fp, window_sig, status_raw in rows:
            strategy_key = str(strategy_fp).strip()
            axis_key = str(axis_dim_fp).strip()
            window_key = str(window_sig).strip()
            status = str(status_raw or "").strip().lower()
            if not strategy_key or not axis_key or not window_key:
                continue
            if status not in self._STAGE_CELL_STATUS_VALUES:
                continue
            cell_key = (strategy_key, axis_key, window_key)
            prev = merged.get(cell_key)
            if prev is None or status_priority.get(status, -1) >= status_priority.get(prev, -1):
                merged[cell_key] = status
        if not merged:
            return
        payload = [
            (
                str(stage_key),
                str(strategy_fp),
                str(axis_dim_fp),
                str(window_sig),
                str(status),
                float(pytime.time()),
            )
            for (strategy_fp, axis_dim_fp, window_sig), status in merged.items()
        ]
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO stage_cell_status("
                    "stage_label, strategy_fingerprint, axis_dimension_fingerprint, window_signature, status, updated_at"
                    ") VALUES(?,?,?,?,?,?)",
                    payload,
                )
                conn.execute(
                    "DELETE FROM stage_unresolved_summary WHERE manifest_name='stage_cell' AND stage_label=?",
                    (str(stage_key),),
                )
            self.stage_cell_status_writes += len(payload)
        except Exception:
            return

    def _cartesian_cell_manifest_get_many(
        self,
        *,
        stage_label: str,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for dim_vec_fp, window_sig in cells:
            key = (str(dim_vec_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], tuple[str, str]] = {}
        chunk_size = 160
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(dimension_vector_fingerprint=? AND window_signature=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for dim_vec_fp, window_sig in chunk:
                        params.extend((str(dim_vec_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT dimension_vector_fingerprint, window_signature, strategy_fingerprint, status "
                        "FROM cartesian_cell_manifest WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        dim_vec_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        strategy_fp = str(row[2] or "").strip()
                        status = str(row[3] or "").strip().lower()
                        if not dim_vec_fp or not window_sig or status not in self._CARTESIAN_CELL_STATUS_VALUES:
                            continue
                        out[(dim_vec_fp, window_sig)] = (status, strategy_fp)
            self.cartesian_manifest_reads += len(out)
        except Exception:
            return {}
        return out

    def _cartesian_cell_manifest_set_many(
        self,
        *,
        stage_label: str,
        rows: list[tuple[str, str, str, str]],
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        status_priority = {
            "pending": 0,
            "cached_hit": 1,
            "evaluated": 2,
            "dominated": 3,
        }
        merged: dict[tuple[str, str], tuple[str, str]] = {}
        for dim_vec_fp, window_sig, strategy_fp, status_raw in rows:
            dim_key = str(dim_vec_fp).strip()
            window_key = str(window_sig).strip()
            strategy_key = str(strategy_fp).strip()
            status = str(status_raw or "").strip().lower()
            if not dim_key or not window_key:
                continue
            if status not in self._CARTESIAN_CELL_STATUS_VALUES:
                continue
            key = (dim_key, window_key)
            prev = merged.get(key)
            prev_status = prev[1] if isinstance(prev, tuple) and len(prev) == 2 else ""
            if prev is None or status_priority.get(status, -1) >= status_priority.get(str(prev_status), -1):
                merged[key] = (strategy_key, status)
        if not merged:
            return
        payload = [
            (
                str(stage_key),
                str(dim_vec_fp),
                str(window_sig),
                str(strategy_fp),
                str(status),
                float(pytime.time()),
            )
            for (dim_vec_fp, window_sig), (strategy_fp, status) in merged.items()
        ]
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO cartesian_cell_manifest("
                    "stage_label, dimension_vector_fingerprint, window_signature, strategy_fingerprint, status, updated_at"
                    ") VALUES(?,?,?,?,?,?)",
                    payload,
                )
            self.cartesian_manifest_writes += len(payload)
        except Exception:
            return

    def _status_span_manifest_spec(self, manifest_name: str) -> dict[str, object]:
        key = str(manifest_name or "").strip().lower()
        if key == "cartesian":
            return {
                "key": "cartesian",
                "table": "cartesian_rank_manifest",
                "cfg_key": "cartesian_rank_manifest",
                "status_values": self._CARTESIAN_RANK_STATUS_VALUES,
                "has_plan_signature": False,
                "compact_min_rows": 1024.0,
                "compact_min_interval_sec": 120.0,
                "pending_ttl_sec": 86400.0,
            }
        if key == "stage":
            return {
                "key": "stage",
                "table": "stage_rank_manifest",
                "cfg_key": "stage_rank_manifest",
                "status_values": self._STAGE_RANK_STATUS_VALUES,
                "has_plan_signature": True,
                "compact_min_rows": 512.0,
                "compact_min_interval_sec": 60.0,
                "pending_ttl_sec": 21600.0,
            }
        return {}

    def _status_span_manifest_counter_add(self, *, manifest_name: str, field: str, value: int) -> None:
        delta = int(max(0, int(value)))
        if delta <= 0:
            return
        key = str(manifest_name).strip().lower()
        field_key = str(field).strip().lower()
        if key == "cartesian":
            if field_key == "reads":
                self.cartesian_rank_manifest_reads += delta
            elif field_key == "writes":
                self.cartesian_rank_manifest_writes += delta
            elif field_key == "compactions":
                self.cartesian_rank_manifest_compactions += delta
            elif field_key == "pending_ttl_prunes":
                self.cartesian_rank_manifest_pending_ttl_prunes += delta
        elif key == "stage":
            if field_key == "reads":
                self.stage_rank_manifest_reads += delta
            elif field_key == "writes":
                self.stage_rank_manifest_writes += delta
            elif field_key == "compactions":
                self.stage_rank_manifest_compactions += delta
            elif field_key == "pending_ttl_prunes":
                self.stage_rank_manifest_pending_ttl_prunes += delta

    def _status_span_manifest_counter_add_hits(self, *, manifest_name: str, covered: int) -> None:
        delta = int(max(0, int(covered)))
        if delta <= 0:
            return
        key = str(manifest_name).strip().lower()
        if key == "cartesian":
            self.cartesian_rank_manifest_hits += delta
        elif key == "stage":
            self.stage_rank_manifest_hits += delta

    def _stage_unresolved_summary_get(
        self,
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        total: int,
        plan_signature: str = "",
    ) -> tuple[int, int] | None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return None
        manifest_key = str(manifest_name or "").strip().lower()
        if manifest_key not in ("cartesian", "stage", "stage_cell"):
            return None
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        plan_key = str(plan_signature).strip() if manifest_key in ("stage", "stage_cell") else ""
        try:
            total_i = int(total)
        except (TypeError, ValueError):
            return None
        if total_i <= 0 or not stage_key or not window_key or (manifest_key in ("stage", "stage_cell") and not plan_key):
            return None
        cfg = _cache_config("stage_unresolved_summary")
        ttl_sec = max(0.0, float(_registry_float(cfg.get("ttl_sec"), 21600.0)))
        now_ts = float(pytime.time())
        row = None
        try:
            with self.run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT unresolved_count, resolved_count, updated_at "
                    "FROM stage_unresolved_summary "
                    "WHERE manifest_name=? AND stage_label=? AND plan_signature=? AND window_signature=? AND total=?",
                    (
                        str(manifest_key),
                        str(stage_key),
                        str(plan_key),
                        str(window_key),
                        int(total_i),
                    ),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        self.stage_unresolved_summary_reads += 1
        try:
            unresolved_i = int(row[0] or 0)
            resolved_i = int(row[1] or 0)
            updated_at = float(row[2] or 0.0)
        except Exception:
            return None
        unresolved_i = int(max(0, min(int(total_i), int(unresolved_i))))
        resolved_i = int(max(0, min(int(total_i), int(resolved_i))))
        if float(ttl_sec) > 0.0 and float(updated_at) > 0.0 and (float(now_ts) - float(updated_at)) > float(ttl_sec):
            try:
                with self.run_cfg_persistent_lock:
                    conn.execute(
                        "DELETE FROM stage_unresolved_summary WHERE manifest_name=? AND stage_label=? AND plan_signature=? AND window_signature=? AND total=?",
                        (
                            str(manifest_key),
                            str(stage_key),
                            str(plan_key),
                            str(window_key),
                            int(total_i),
                        ),
                    )
            except Exception:
                pass
            return None
        self.stage_unresolved_summary_hits += 1
        return int(unresolved_i), int(resolved_i)

    def _stage_unresolved_summary_invalidate(
        self,
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        plan_signature: str = "",
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        manifest_key = str(manifest_name or "").strip().lower()
        if manifest_key not in ("cartesian", "stage", "stage_cell"):
            return
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        plan_key = str(plan_signature).strip() if manifest_key in ("stage", "stage_cell") else ""
        if not stage_key or not window_key or (manifest_key in ("stage", "stage_cell") and not plan_key):
            return
        try:
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "DELETE FROM stage_unresolved_summary WHERE manifest_name=? AND stage_label=? AND plan_signature=? AND window_signature=?",
                    (str(manifest_key), str(stage_key), str(plan_key), str(window_key)),
                )
        except Exception:
            return

    def _stage_unresolved_summary_set(
        self,
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        total: int,
        unresolved_count: int,
        resolved_count: int,
        plan_signature: str = "",
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        manifest_key = str(manifest_name or "").strip().lower()
        if manifest_key not in ("cartesian", "stage", "stage_cell"):
            return
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        plan_key = str(plan_signature).strip() if manifest_key in ("stage", "stage_cell") else ""
        if not stage_key or not window_key or (manifest_key in ("stage", "stage_cell") and not plan_key):
            return
        try:
            total_i = int(max(0, int(total)))
            unresolved_i = int(max(0, int(unresolved_count)))
            resolved_i = int(max(0, int(resolved_count)))
        except (TypeError, ValueError):
            return
        if total_i <= 0:
            return
        unresolved_i = int(max(0, min(int(total_i), int(unresolved_i))))
        resolved_i = int(max(0, min(int(total_i), int(resolved_i))))
        now_ts = float(pytime.time())
        try:
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "INSERT INTO stage_unresolved_summary("
                    "manifest_name, stage_label, plan_signature, window_signature, total, unresolved_count, resolved_count, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(manifest_name, stage_label, plan_signature, window_signature, total) DO UPDATE SET "
                    "unresolved_count=excluded.unresolved_count, "
                    "resolved_count=excluded.resolved_count, "
                    "updated_at=excluded.updated_at",
                    (
                        str(manifest_key),
                        str(stage_key),
                        str(plan_key),
                        str(window_key),
                        int(total_i),
                        int(unresolved_i),
                        int(resolved_i),
                        float(now_ts),
                    ),
                )
            self.stage_unresolved_summary_writes += 1
        except Exception:
            return

    def _status_span_rows_compact(
        self,
        *,
        rows: list[tuple[int, int, str]],
        status_values: frozenset[str],
    ) -> list[tuple[int, int, str]]:
        status_priority = {
            "pending": 0,
            "cached_hit": 1,
            "evaluated": 2,
            "dominated": 3,
        }
        merged: dict[tuple[int, int], str] = {}
        for rank_lo_raw, rank_hi_raw, status_raw in rows:
            try:
                rank_lo = int(rank_lo_raw)
                rank_hi = int(rank_hi_raw)
            except (TypeError, ValueError):
                continue
            status = str(status_raw or "").strip().lower()
            if rank_lo < 0 or rank_hi < rank_lo or status not in status_values:
                continue
            key = (int(rank_lo), int(rank_hi))
            prev = merged.get(key)
            if prev is None or status_priority.get(status, -1) >= status_priority.get(str(prev), -1):
                merged[key] = str(status)
        if not merged:
            return []
        ordered = sorted(
            ((int(lo), int(hi), str(status)) for (lo, hi), status in merged.items()),
            key=lambda row: (int(row[0]), int(row[1])),
        )
        out: list[tuple[int, int, str]] = []
        for rank_lo, rank_hi, status in ordered:
            if out:
                prev_lo, prev_hi, prev_status = out[-1]
                if str(prev_status) == str(status) and int(rank_lo) <= int(prev_hi) + 1:
                    out[-1] = (
                        int(prev_lo),
                        max(int(prev_hi), int(rank_hi)),
                        str(status),
                    )
                    continue
            out.append((int(rank_lo), int(rank_hi), str(status)))
        return out

    def _status_span_manifest_set_many(
        self,
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        rows: list[tuple[int, int, str]],
        plan_signature: str = "",
        replace_scope: bool = False,
    ) -> None:
        spec = self._status_span_manifest_spec(str(manifest_name))
        if not isinstance(spec, dict) or not spec:
            return
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        has_plan_signature = bool(spec.get("has_plan_signature"))
        plan_key = str(plan_signature).strip()
        if not stage_key or not window_key:
            return
        if has_plan_signature and not plan_key:
            return
        status_values = spec.get("status_values")
        if not isinstance(status_values, frozenset):
            return
        compact_rows = self._status_span_rows_compact(rows=list(rows or ()), status_values=status_values)
        did_mutate = bool(replace_scope) or bool(compact_rows)
        now_ts = float(pytime.time())
        try:
            with self.run_cfg_persistent_lock:
                if bool(replace_scope):
                    if has_plan_signature:
                        conn.execute(
                            "DELETE FROM stage_rank_manifest WHERE stage_label=? AND plan_signature=? AND window_signature=?",
                            (str(stage_key), str(plan_key), str(window_key)),
                        )
                    else:
                        conn.execute(
                            "DELETE FROM cartesian_rank_manifest WHERE stage_label=? AND window_signature=?",
                            (str(stage_key), str(window_key)),
                        )
                if compact_rows:
                    if has_plan_signature:
                        payload = [
                            (
                                str(stage_key),
                                str(plan_key),
                                str(window_key),
                                int(rank_lo),
                                int(rank_hi),
                                str(status),
                                float(now_ts),
                            )
                            for rank_lo, rank_hi, status in compact_rows
                        ]
                        conn.executemany(
                            "INSERT OR REPLACE INTO stage_rank_manifest("
                            "stage_label, plan_signature, window_signature, rank_lo, rank_hi, status, updated_at"
                            ") VALUES(?,?,?,?,?,?,?)",
                            payload,
                        )
                    else:
                        payload = [
                            (
                                str(stage_key),
                                str(window_key),
                                int(rank_lo),
                                int(rank_hi),
                                str(status),
                                float(now_ts),
                            )
                            for rank_lo, rank_hi, status in compact_rows
                        ]
                        conn.executemany(
                            "INSERT OR REPLACE INTO cartesian_rank_manifest("
                            "stage_label, window_signature, rank_lo, rank_hi, status, updated_at"
                            ") VALUES(?,?,?,?,?,?)",
                            payload,
                        )
            if bool(did_mutate):
                self._stage_unresolved_summary_invalidate(
                    manifest_name=str(spec.get("key") or ""),
                    stage_label=str(stage_key),
                    plan_signature=(str(plan_key) if has_plan_signature else ""),
                    window_signature=str(window_key),
                )
            self._status_span_manifest_counter_add(
                manifest_name=str(spec.get("key") or ""),
                field="writes",
                value=len(compact_rows),
            )
        except Exception:
            return

    def _status_span_manifest_get_many(
        self,
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        plan_signature: str = "",
    ) -> list[tuple[int, int, str]]:
        spec = self._status_span_manifest_spec(str(manifest_name))
        if not isinstance(spec, dict) or not spec:
            return []
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return []
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        has_plan_signature = bool(spec.get("has_plan_signature"))
        plan_key = str(plan_signature).strip()
        if not stage_key or not window_key:
            return []
        if has_plan_signature and not plan_key:
            return []
        status_values = spec.get("status_values")
        if not isinstance(status_values, frozenset):
            return []
        cfg = _cache_config(str(spec.get("cfg_key") or ""))
        compact_min_rows = max(
            64,
            int(
                _registry_float(
                    cfg.get("compact_min_rows"),
                    float(spec.get("compact_min_rows") or 512.0),
                )
            ),
        )
        compact_min_interval = max(
            5.0,
            float(
                _registry_float(
                    cfg.get("compact_min_interval_sec"),
                    float(spec.get("compact_min_interval_sec") or 120.0),
                )
            ),
        )
        pending_ttl_sec = max(
            0.0,
            float(
                _registry_float(
                    cfg.get("pending_ttl_sec"),
                    float(spec.get("pending_ttl_sec") or 0.0),
                )
            ),
        )
        now_ts = float(pytime.time())
        out: list[tuple[int, int, str]] = []
        stale_pending = 0
        try:
            with self.run_cfg_persistent_lock:
                if has_plan_signature:
                    rows_db = conn.execute(
                        "SELECT rank_lo, rank_hi, status, updated_at FROM stage_rank_manifest WHERE stage_label=? AND plan_signature=? AND window_signature=?",
                        (str(stage_key), str(plan_key), str(window_key)),
                    ).fetchall()
                else:
                    rows_db = conn.execute(
                        "SELECT rank_lo, rank_hi, status, updated_at FROM cartesian_rank_manifest WHERE stage_label=? AND window_signature=?",
                        (str(stage_key), str(window_key)),
                    ).fetchall()
            for row in rows_db:
                try:
                    rank_lo = int(row[0])
                    rank_hi = int(row[1])
                except (TypeError, ValueError):
                    continue
                status = str(row[2] or "").strip().lower()
                try:
                    updated_at = float(row[3] or 0.0)
                except (TypeError, ValueError):
                    updated_at = 0.0
                if rank_lo < 0 or rank_hi < rank_lo or status not in status_values:
                    continue
                if (
                    status == "pending"
                    and float(pending_ttl_sec) > 0.0
                    and float(updated_at) > 0.0
                    and (float(now_ts) - float(updated_at)) > float(pending_ttl_sec)
                ):
                    stale_pending += 1
                    continue
                out.append((int(rank_lo), int(rank_hi), str(status)))
        except Exception:
            return []
        self._status_span_manifest_counter_add(
            manifest_name=str(spec.get("key") or ""),
            field="reads",
            value=len(out),
        )
        if not out:
            if int(stale_pending) > 0:
                self._status_span_manifest_set_many(
                    manifest_name=str(spec.get("key") or ""),
                    stage_label=str(stage_key),
                    plan_signature=str(plan_key),
                    window_signature=str(window_key),
                    rows=[],
                    replace_scope=True,
                )
                self._status_span_manifest_counter_add(
                    manifest_name=str(spec.get("key") or ""),
                    field="pending_ttl_prunes",
                    value=int(stale_pending),
                )
            return []

        compacted = self._status_span_rows_compact(rows=list(out), status_values=status_values)
        compact_key = (
            str(spec.get("key") or ""),
            str(stage_key),
            str(plan_key if has_plan_signature else ""),
            str(window_key),
        )
        last_ts = float(self.status_span_manifest_compact_seen.get(compact_key, 0.0) or 0.0)
        interval_ok = (float(now_ts) - float(last_ts)) >= float(compact_min_interval)
        should_rewrite = bool(int(stale_pending) > 0 or (interval_ok and len(out) >= int(compact_min_rows) and len(compacted) < len(out)))
        if should_rewrite:
            self._status_span_manifest_set_many(
                manifest_name=str(spec.get("key") or ""),
                stage_label=str(stage_key),
                plan_signature=str(plan_key),
                window_signature=str(window_key),
                rows=list(compacted),
                replace_scope=True,
            )
            if len(compacted) < len(out):
                self._status_span_manifest_counter_add(
                    manifest_name=str(spec.get("key") or ""),
                    field="compactions",
                    value=1,
                )
            if int(stale_pending) > 0:
                self._status_span_manifest_counter_add(
                    manifest_name=str(spec.get("key") or ""),
                    field="pending_ttl_prunes",
                    value=int(stale_pending),
                )
            out = list(compacted)
        self.status_span_manifest_compact_seen[compact_key] = float(now_ts)
        return out

    def _status_span_manifest_unresolved_ranges(
        self,
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        total: int,
        plan_signature: str = "",
    ) -> tuple[tuple[int, int], ...]:
        total_i = int(total)
        if total_i <= 0:
            return ()
        spec = self._status_span_manifest_spec(str(manifest_name))
        manifest_key = str(spec.get("key") or str(manifest_name)).strip().lower()
        has_plan_signature = bool(spec.get("has_plan_signature"))
        plan_key = str(plan_signature).strip() if has_plan_signature else ""
        if has_plan_signature and not plan_key:
            return ((0, int(total_i - 1)),)
        summary = self._stage_unresolved_summary_get(
            manifest_name=str(manifest_key),
            stage_label=str(stage_label),
            plan_signature=str(plan_key),
            window_signature=str(window_signature),
            total=int(total_i),
        )
        if isinstance(summary, tuple) and len(summary) == 2:
            try:
                unresolved_count = int(summary[0])
                resolved_count = int(summary[1])
            except (TypeError, ValueError):
                unresolved_count = -1
                resolved_count = -1
            if unresolved_count == 0:
                if resolved_count > 0:
                    self._status_span_manifest_counter_add_hits(
                        manifest_name=str(manifest_key),
                        covered=int(resolved_count),
                    )
                return ()
            if unresolved_count >= int(total_i):
                return ((0, int(total_i - 1)),)
        rows = self._status_span_manifest_get_many(
            manifest_name=str(manifest_name),
            stage_label=str(stage_label),
            window_signature=str(window_signature),
            plan_signature=str(plan_signature),
        )
        if not rows:
            self._stage_unresolved_summary_set(
                manifest_name=str(manifest_key),
                stage_label=str(stage_label),
                plan_signature=str(plan_key),
                window_signature=str(window_signature),
                total=int(total_i),
                unresolved_count=int(total_i),
                resolved_count=0,
            )
            return ((0, int(total_i - 1)),)
        resolved_ranges: list[tuple[int, int]] = []
        for rank_lo, rank_hi, status in rows:
            if str(status) not in ("cached_hit", "evaluated", "dominated"):
                continue
            lo = max(0, int(rank_lo))
            hi = min(int(total_i - 1), int(rank_hi))
            if hi < lo:
                continue
            resolved_ranges.append((int(lo), int(hi)))
        if not resolved_ranges:
            self._stage_unresolved_summary_set(
                manifest_name=str(manifest_key),
                stage_label=str(stage_label),
                plan_signature=str(plan_key),
                window_signature=str(window_signature),
                total=int(total_i),
                unresolved_count=int(total_i),
                resolved_count=0,
            )
            return ((0, int(total_i - 1)),)
        resolved_ranges.sort(key=lambda row: (int(row[0]), int(row[1])))
        merged: list[tuple[int, int]] = []
        for lo, hi in resolved_ranges:
            if not merged:
                merged.append((int(lo), int(hi)))
                continue
            prev_lo, prev_hi = merged[-1]
            if int(lo) <= int(prev_hi) + 1:
                merged[-1] = (int(prev_lo), max(int(prev_hi), int(hi)))
            else:
                merged.append((int(lo), int(hi)))
        unresolved: list[tuple[int, int]] = []
        cursor = 0
        covered = 0
        for lo, hi in merged:
            lo_i = max(0, int(lo))
            hi_i = min(int(total_i - 1), int(hi))
            if hi_i < lo_i:
                continue
            if cursor < lo_i:
                unresolved.append((int(cursor), int(lo_i - 1)))
            covered += max(0, int(hi_i - lo_i + 1))
            cursor = int(max(cursor, hi_i + 1))
        if cursor < total_i:
            unresolved.append((int(cursor), int(total_i - 1)))
        if covered > 0:
            self._status_span_manifest_counter_add_hits(
                manifest_name=str(manifest_key),
                covered=int(covered),
            )
        unresolved_count = sum(max(0, int(hi) - int(lo) + 1) for lo, hi in unresolved)
        self._stage_unresolved_summary_set(
            manifest_name=str(manifest_key),
            stage_label=str(stage_label),
            plan_signature=str(plan_key),
            window_signature=str(window_signature),
            total=int(total_i),
            unresolved_count=int(unresolved_count),
            resolved_count=int(max(0, int(total_i) - int(unresolved_count))),
        )
        return tuple(unresolved)
