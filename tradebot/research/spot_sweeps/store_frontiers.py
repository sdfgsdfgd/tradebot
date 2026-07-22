"""SweepFrontierStore capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time as pytime
from ...backtest.config import (
    ConfigBundle,
)
from .fingerprints import (
    _RUN_CFG_CACHE_ENGINE_VERSION,
    _axis_dimension_fingerprint,
)
from .milestones import (
    _milestone_key,
)
from .support import (
    _cache_config,
    _registry_float,
)


class SweepFrontierStore:
    def _cartesian_rank_manifest_set_many(
        self,
        *,
        stage_label: str,
        window_signature: str,
        rows: list[tuple[int, int, str]],
    ) -> None:
        self._status_span_manifest_set_many(
            manifest_name="cartesian",
            stage_label=str(stage_label),
            window_signature=str(window_signature),
            rows=list(rows or ()),
        )

    def _stage_rank_manifest_set_many(
        self,
        *,
        stage_label: str,
        plan_signature: str,
        window_signature: str,
        rows: list[tuple[int, int, str]],
    ) -> None:
        self._status_span_manifest_set_many(
            manifest_name="stage",
            stage_label=str(stage_label),
            plan_signature=str(plan_signature),
            window_signature=str(window_signature),
            rows=list(rows or ()),
        )

    def _stage_rank_manifest_unresolved_ranges(
        self,
        *,
        stage_label: str,
        plan_signature: str,
        window_signature: str,
        total: int,
    ) -> tuple[tuple[int, int], ...]:
        return self._status_span_manifest_unresolved_ranges(
            manifest_name="stage",
            stage_label=str(stage_label),
            plan_signature=str(plan_signature),
            window_signature=str(window_signature),
            total=int(total),
        )

    def _rank_dominance_stamp_get_many(
        self,
        *,
        stage_label: str,
        window_signature: str,
    ) -> list[tuple[str, int, int]]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return []
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key:
            return []
        cfg = _cache_config("rank_dominance_stamp")
        compact_min_rows = max(
            64, int(_registry_float(cfg.get("compact_min_rows"), 512.0))
        )
        compact_min_interval = max(
            5.0, float(_registry_float(cfg.get("compact_min_interval_sec"), 120.0))
        )
        ttl_sec = max(0.0, float(_registry_float(cfg.get("ttl_sec"), 1209600.0)))
        now_ts = float(pytime.time())
        compact_key = (str(stage_key), str(window_key))
        out: list[tuple[str, int, int]] = []
        stale_rows = 0
        should_consider_compact = False
        try:
            with self.run_cfg_persistent_lock:
                rows_db = conn.execute(
                    "SELECT dominance_signature, rank_lo, rank_hi, updated_at FROM rank_dominance_stamp WHERE stage_label=? AND window_signature=?",
                    (str(stage_key), str(window_key)),
                ).fetchall()
            for row in rows_db:
                sig = str(row[0] or "").strip().lower()
                try:
                    rank_lo = int(row[1])
                    rank_hi = int(row[2])
                except (TypeError, ValueError):
                    continue
                try:
                    updated_at = float(row[3] or 0.0)
                except (TypeError, ValueError):
                    updated_at = 0.0
                if not sig or rank_lo < 0 or rank_hi < rank_lo:
                    continue
                if (
                    float(ttl_sec) > 0.0
                    and float(updated_at) > 0.0
                    and (float(now_ts) - float(updated_at)) > float(ttl_sec)
                ):
                    stale_rows += 1
                    continue
                out.append((str(sig), int(rank_lo), int(rank_hi)))
            self.rank_dominance_stamp_reads += len(out)
            should_consider_compact = (
                len(out) >= int(compact_min_rows) or int(stale_rows) > 0
            )
        except Exception:
            return []
        if not out:
            if int(stale_rows) > 0:
                try:
                    with self.run_cfg_persistent_lock:
                        conn.execute(
                            "DELETE FROM rank_dominance_stamp WHERE stage_label=? AND window_signature=?",
                            (str(stage_key), str(window_key)),
                        )
                except Exception:
                    pass
                self.rank_dominance_stamp_ttl_prunes += int(stale_rows)
            return []
        last_compact_ts = float(
            self.rank_dominance_stamp_compact_seen.get(compact_key, 0.0) or 0.0
        )
        should_compact_now = bool(
            should_consider_compact
            and (float(now_ts) - float(last_compact_ts)) >= float(compact_min_interval)
        )
        if not should_compact_now:
            return out
        grouped: dict[str, list[tuple[int, int]]] = {}
        for sig, rank_lo, rank_hi in out:
            grouped.setdefault(str(sig), []).append((int(rank_lo), int(rank_hi)))
        compacted: list[tuple[str, int, int]] = []
        for sig, ranges in grouped.items():
            ordered = sorted(ranges, key=lambda row: (int(row[0]), int(row[1])))
            if not ordered:
                continue
            cur_lo = int(ordered[0][0])
            cur_hi = int(ordered[0][1])
            for rank_lo, rank_hi in ordered[1:]:
                if int(rank_lo) <= int(cur_hi) + 1:
                    cur_hi = max(int(cur_hi), int(rank_hi))
                else:
                    compacted.append((str(sig), int(cur_lo), int(cur_hi)))
                    cur_lo = int(rank_lo)
                    cur_hi = int(rank_hi)
            compacted.append((str(sig), int(cur_lo), int(cur_hi)))
        rewrite_needed = bool(int(stale_rows) > 0 or len(compacted) < len(out))
        if rewrite_needed:
            try:
                with self.run_cfg_persistent_lock:
                    conn.execute(
                        "DELETE FROM rank_dominance_stamp WHERE stage_label=? AND window_signature=?",
                        (str(stage_key), str(window_key)),
                    )
                self._rank_dominance_stamp_set_many(
                    stage_label=str(stage_key),
                    window_signature=str(window_key),
                    rows=list(compacted),
                )
                out = list(compacted)
                self.rank_dominance_stamp_compactions += 1
            except Exception:
                pass
        if int(stale_rows) > 0:
            self.rank_dominance_stamp_ttl_prunes += int(stale_rows)
        self.rank_dominance_stamp_compact_seen[compact_key] = float(now_ts)
        return out

    def _rank_dominance_stamp_set_many(
        self,
        *,
        stage_label: str,
        window_signature: str,
        rows: list[tuple[str, int, int]],
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key or not rows:
            return
        grouped: dict[str, list[tuple[int, int]]] = {}
        for sig_raw, rank_lo_raw, rank_hi_raw in rows:
            sig = str(sig_raw or "").strip().lower()
            try:
                rank_lo = int(rank_lo_raw)
                rank_hi = int(rank_hi_raw)
            except (TypeError, ValueError):
                continue
            if not sig or rank_lo < 0 or rank_hi < rank_lo:
                continue
            grouped.setdefault(str(sig), []).append((int(rank_lo), int(rank_hi)))
        if not grouped:
            return
        payload: list[tuple[str, str, str, int, int, float]] = []
        now_ts = float(pytime.time())
        for sig, ranges in grouped.items():
            ordered = sorted(ranges, key=lambda row: (int(row[0]), int(row[1])))
            if not ordered:
                continue
            cur_lo = int(ordered[0][0])
            cur_hi = int(ordered[0][1])
            for rank_lo, rank_hi in ordered[1:]:
                if int(rank_lo) <= int(cur_hi) + 1:
                    cur_hi = max(int(cur_hi), int(rank_hi))
                else:
                    payload.append(
                        (
                            str(stage_key),
                            str(window_key),
                            str(sig),
                            int(cur_lo),
                            int(cur_hi),
                            float(now_ts),
                        )
                    )
                    cur_lo = int(rank_lo)
                    cur_hi = int(rank_hi)
            payload.append(
                (
                    str(stage_key),
                    str(window_key),
                    str(sig),
                    int(cur_lo),
                    int(cur_hi),
                    float(now_ts),
                )
            )
        if not payload:
            return
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO rank_dominance_stamp("
                    "stage_label, window_signature, dominance_signature, rank_lo, rank_hi, updated_at"
                    ") VALUES(?,?,?,?,?,?)",
                    payload,
                )
            self.rank_dominance_stamp_writes += len(payload)
        except Exception:
            return

    def _apply_rank_dominance_stamps_to_manifest(
        self,
        *,
        stage_label: str,
        window_signature: str,
        total: int,
    ) -> None:
        total_i = int(total)
        if total_i <= 0:
            return
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key:
            return
        seen_key = (str(stage_key), str(window_key))
        if seen_key in self.rank_dominance_manifest_applied_seen:
            return
        stamp_rows = self._rank_dominance_stamp_get_many(
            stage_label=str(stage_key),
            window_signature=str(window_key),
        )
        self.rank_dominance_manifest_applied_seen.add(seen_key)
        if not stamp_rows:
            return
        manifest_rows: list[tuple[int, int, str]] = []
        covered = 0
        for _sig, rank_lo, rank_hi in stamp_rows:
            lo_i = max(0, int(rank_lo))
            hi_i = min(int(total_i - 1), int(rank_hi))
            if hi_i < lo_i:
                continue
            manifest_rows.append((int(lo_i), int(hi_i), "dominated"))
            covered += max(0, int(hi_i - lo_i + 1))
        if not manifest_rows:
            return
        self._cartesian_rank_manifest_set_many(
            stage_label=str(stage_key),
            window_signature=str(window_key),
            rows=manifest_rows,
        )
        self.rank_dominance_manifest_applies += len(manifest_rows)
        self.rank_dominance_stamp_hits += int(covered)

    def _cartesian_rank_manifest_unresolved_ranges(
        self,
        *,
        stage_label: str,
        window_signature: str,
        total: int,
    ) -> tuple[tuple[int, int], ...]:
        total_i = int(total)
        if total_i <= 0:
            return ()
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        self._apply_rank_dominance_stamps_to_manifest(
            stage_label=str(stage_key),
            window_signature=str(window_key),
            total=int(total_i),
        )
        return self._status_span_manifest_unresolved_ranges(
            manifest_name="cartesian",
            stage_label=str(stage_key),
            window_signature=str(window_key),
            total=int(total_i),
        )

    def _cartesian_rank_manifest_claim_next_range(
        self,
        *,
        stage_label: str,
        window_signature: str,
        total: int,
        max_span: int,
    ) -> tuple[int, int] | None:
        total_i = int(total)
        if total_i <= 0:
            return None
        max_span_i = max(1, int(max_span))
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return None
        stage_key = self._stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key:
            return None
        cfg = _cache_config("cartesian_rank_manifest")
        pending_ttl_sec = max(
            0.0, float(_registry_float(cfg.get("pending_ttl_sec"), 86400.0))
        )
        now_ts = float(pytime.time())
        stale_pending_pruned = 0
        reads_seen = 0
        claim_row: tuple[int, int] | None = None
        attempts = 6
        status_values = ("pending", "cached_hit", "evaluated", "dominated")
        status_placeholders = ",".join("?" for _ in status_values)
        for attempt_idx in range(attempts):
            try:
                with self.run_cfg_persistent_lock:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        if float(pending_ttl_sec) > 0.0:
                            stale_pending_pruned = int(
                                conn.execute(
                                    "SELECT COUNT(*) FROM cartesian_rank_manifest "
                                    "WHERE stage_label=? AND window_signature=? AND status='pending' "
                                    "AND updated_at>0 AND (? - updated_at) > ?",
                                    (
                                        str(stage_key),
                                        str(window_key),
                                        float(now_ts),
                                        float(pending_ttl_sec),
                                    ),
                                ).fetchone()[0]
                                or 0
                            )
                            if int(stale_pending_pruned) > 0:
                                conn.execute(
                                    "DELETE FROM cartesian_rank_manifest "
                                    "WHERE stage_label=? AND window_signature=? AND status='pending' "
                                    "AND updated_at>0 AND (? - updated_at) > ?",
                                    (
                                        str(stage_key),
                                        str(window_key),
                                        float(now_ts),
                                        float(pending_ttl_sec),
                                    ),
                                )

                        cursor_row = conn.execute(
                            "SELECT next_rank FROM cartesian_rank_cursor WHERE stage_label=? AND window_signature=?",
                            (str(stage_key), str(window_key)),
                        ).fetchone()
                        reads_seen += 1
                        cursor = 0
                        if isinstance(cursor_row, tuple) and len(cursor_row) >= 1:
                            try:
                                cursor = int(cursor_row[0])
                            except (TypeError, ValueError):
                                cursor = 0
                        cursor = max(0, min(int(total_i), int(cursor)))
                        if int(stale_pending_pruned) > 0 and int(cursor) > 0:
                            # Pending TTL pruning can reopen gaps behind the persisted cursor.
                            cursor = 0

                        while int(cursor) < int(total_i):
                            cover_row = conn.execute(
                                "SELECT rank_lo, rank_hi FROM cartesian_rank_manifest "
                                "WHERE stage_label=? AND window_signature=? "
                                "AND rank_lo<=? AND rank_hi>=? "
                                f"AND status IN ({status_placeholders}) "
                                "ORDER BY rank_hi DESC LIMIT 1",
                                (
                                    str(stage_key),
                                    str(window_key),
                                    int(cursor),
                                    int(cursor),
                                    *status_values,
                                ),
                            ).fetchone()
                            reads_seen += 1
                            if not (
                                isinstance(cover_row, tuple) and len(cover_row) >= 2
                            ):
                                break
                            try:
                                cover_hi = int(cover_row[1])
                            except (TypeError, ValueError):
                                break
                            if int(cover_hi) < int(cursor):
                                break
                            cursor = int(min(int(total_i), int(cover_hi) + 1))

                        if int(cursor) < int(total_i):
                            next_row = conn.execute(
                                "SELECT rank_lo, rank_hi FROM cartesian_rank_manifest "
                                "WHERE stage_label=? AND window_signature=? AND rank_lo>=? "
                                f"AND status IN ({status_placeholders}) "
                                "ORDER BY rank_lo ASC LIMIT 1",
                                (
                                    str(stage_key),
                                    str(window_key),
                                    int(cursor),
                                    *status_values,
                                ),
                            ).fetchone()
                            reads_seen += 1
                            next_lo = int(total_i)
                            next_hi = None
                            if isinstance(next_row, tuple) and len(next_row) >= 2:
                                try:
                                    next_lo = int(next_row[0])
                                    next_hi = int(next_row[1])
                                except (TypeError, ValueError):
                                    next_lo = int(total_i)
                                    next_hi = None
                            if int(next_lo) <= int(cursor):
                                if isinstance(next_hi, int) and int(next_hi) >= int(
                                    cursor
                                ):
                                    cursor = int(min(int(total_i), int(next_hi) + 1))
                                else:
                                    cursor = int(min(int(total_i), int(cursor) + 1))
                            else:
                                claim_lo = int(cursor)
                                gap_hi = int(min(int(total_i - 1), int(next_lo) - 1))
                                claim_hi = int(
                                    min(
                                        int(gap_hi), int(claim_lo + int(max_span_i) - 1)
                                    )
                                )
                                if int(claim_hi) >= int(claim_lo):
                                    conn.execute(
                                        "INSERT OR REPLACE INTO cartesian_rank_manifest("
                                        "stage_label, window_signature, rank_lo, rank_hi, status, updated_at"
                                        ") VALUES(?,?,?,?,?,?)",
                                        (
                                            str(stage_key),
                                            str(window_key),
                                            int(claim_lo),
                                            int(claim_hi),
                                            "pending",
                                            float(now_ts),
                                        ),
                                    )
                                    claim_row = (int(claim_lo), int(claim_hi))
                                    cursor = int(min(int(total_i), int(claim_hi) + 1))

                        conn.execute(
                            "INSERT OR REPLACE INTO cartesian_rank_cursor(stage_label, window_signature, next_rank, updated_at) VALUES(?,?,?,?)",
                            (
                                str(stage_key),
                                str(window_key),
                                int(cursor),
                                float(now_ts),
                            ),
                        )
                        conn.execute("COMMIT")
                    except Exception:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                break
            except sqlite3.OperationalError as exc:
                msg = str(exc or "").lower()
                if "locked" in msg and int(attempt_idx) < int(attempts - 1):
                    pytime.sleep(min(0.1, 0.01 * float(attempt_idx + 1)))
                    continue
                return None
            except Exception:
                return None

        if int(reads_seen) > 0:
            self._status_span_manifest_counter_add(
                manifest_name="cartesian", field="reads", value=int(reads_seen)
            )
        if int(stale_pending_pruned) > 0:
            self._status_span_manifest_counter_add(
                manifest_name="cartesian",
                field="pending_ttl_prunes",
                value=int(stale_pending_pruned),
            )
        self._status_span_manifest_counter_add(
            manifest_name="cartesian", field="writes", value=1
        )
        if isinstance(claim_row, tuple):
            self._status_span_manifest_counter_add(
                manifest_name="cartesian", field="writes", value=1
            )
        return claim_row

    def _stage_frontier_get_many(
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
        run_min = int(self.run_min_trades)
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
                    where = " OR ".join(
                        "(axis_dimension_fingerprint=? AND window_signature=?)"
                        for _ in chunk
                    )
                    params: list[object] = [str(stage_key), int(run_min)]
                    for axis_dim_fp, window_sig in chunk:
                        params.extend((str(axis_dim_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT axis_dimension_fingerprint, window_signature, eval_count, keep_count, "
                        "best_pnl_over_dd, best_pnl, best_win_rate, best_trades "
                        "FROM stage_frontier WHERE stage_label=? AND run_min_trades=? AND ("
                        + where
                        + ")",
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
                            "best_pnl_over_dd": (
                                None if row[4] is None else float(row[4])
                            ),
                            "best_pnl": (None if row[5] is None else float(row[5])),
                            "best_win_rate": (
                                None if row[6] is None else float(row[6])
                            ),
                            "best_trades": (None if row[7] is None else int(row[7])),
                        }
            self.stage_frontier_reads += len(out)
        except Exception:
            return {}
        return out

    def _stage_frontier_upsert_many(
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
        run_min = int(self.run_min_trades)
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
                    "best_win_rate": None,
                    "best_trades": None,
                }
                merged[key] = rec
            rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
            if not isinstance(row, dict):
                continue
            rec["keep_count"] = int(rec.get("keep_count") or 0) + 1
            for metric_key, rec_key, caster in (
                ("pnl_over_dd", "best_pnl_over_dd", float),
                ("pnl", "best_pnl", float),
                ("win_rate", "best_win_rate", float),
                ("trades", "best_trades", int),
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
                int(run_min),
                int(rec.get("eval_count") or 0),
                int(rec.get("keep_count") or 0),
                rec.get("best_pnl_over_dd"),
                rec.get("best_pnl"),
                rec.get("best_win_rate"),
                rec.get("best_trades"),
                now_ts,
            )
            for (axis_dim_fp, window_sig), rec in merged.items()
        ]
        try:
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO stage_frontier("
                    "stage_label, axis_dimension_fingerprint, window_signature, run_min_trades, "
                    "eval_count, keep_count, best_pnl_over_dd, best_pnl, best_win_rate, best_trades, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, axis_dimension_fingerprint, window_signature, run_min_trades) DO UPDATE SET "
                    "eval_count=stage_frontier.eval_count + excluded.eval_count, "
                    "keep_count=stage_frontier.keep_count + excluded.keep_count, "
                    "best_pnl_over_dd=CASE "
                    "WHEN stage_frontier.best_pnl_over_dd IS NULL THEN excluded.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd IS NULL THEN stage_frontier.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd > stage_frontier.best_pnl_over_dd THEN excluded.best_pnl_over_dd "
                    "ELSE stage_frontier.best_pnl_over_dd END, "
                    "best_pnl=CASE "
                    "WHEN stage_frontier.best_pnl IS NULL THEN excluded.best_pnl "
                    "WHEN excluded.best_pnl IS NULL THEN stage_frontier.best_pnl "
                    "WHEN excluded.best_pnl > stage_frontier.best_pnl THEN excluded.best_pnl "
                    "ELSE stage_frontier.best_pnl END, "
                    "best_win_rate=CASE "
                    "WHEN stage_frontier.best_win_rate IS NULL THEN excluded.best_win_rate "
                    "WHEN excluded.best_win_rate IS NULL THEN stage_frontier.best_win_rate "
                    "WHEN excluded.best_win_rate > stage_frontier.best_win_rate THEN excluded.best_win_rate "
                    "ELSE stage_frontier.best_win_rate END, "
                    "best_trades=CASE "
                    "WHEN stage_frontier.best_trades IS NULL THEN excluded.best_trades "
                    "WHEN excluded.best_trades IS NULL THEN stage_frontier.best_trades "
                    "WHEN excluded.best_trades > stage_frontier.best_trades THEN excluded.best_trades "
                    "ELSE stage_frontier.best_trades END, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            self.stage_frontier_writes += len(payload)
        except Exception:
            return

    def _stage_frontier_is_dominated(
        self, frontier_row: dict[str, object] | None
    ) -> bool:
        if not isinstance(frontier_row, dict):
            return False
        cfg = _cache_config("stage_frontier")
        min_eval_count = max(1, int(_registry_float(cfg.get("min_eval_count"), 3.0)))
        max_keep_count = max(0, int(_registry_float(cfg.get("max_keep_count"), 0.0)))
        max_best_pnl = float(_registry_float(cfg.get("max_best_pnl"), 0.0))
        max_best_pnl_dd = float(_registry_float(cfg.get("max_best_pnl_over_dd"), 0.0))
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        if eval_count < int(min_eval_count):
            return False
        if keep_count > int(max_keep_count):
            return False
        best_pnl = frontier_row.get("best_pnl")
        best_pnl_dd = frontier_row.get("best_pnl_over_dd")
        best_pnl_f = float(best_pnl) if best_pnl is not None else float("-inf")
        best_pnl_dd_f = float(best_pnl_dd) if best_pnl_dd is not None else float("-inf")
        return bool(
            best_pnl_f <= float(max_best_pnl)
            and best_pnl_dd_f <= float(max_best_pnl_dd)
        )

    def _run_cfg_dimension_index_set(
        self, *, fingerprint: str, payload_json: str, est_cost: float
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO axis_dimension_fingerprint_index(fingerprint, payload_json, est_cost, updated_at) VALUES(?,?,?,?)",
                    (
                        str(fingerprint),
                        str(payload_json),
                        float(est_cost),
                        float(pytime.time()),
                    ),
                )
            if float(est_cost) > 0.0:
                self.run_cfg_dim_index_loaded[str(fingerprint)] = float(est_cost)
            self.run_cfg_dim_index_writes += 1
        except Exception:
            return

    def _run_cfg_dimension_index_load(self, *, limit: int = 50000) -> dict[str, float]:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return {}
        try:
            with self.run_cfg_persistent_lock:
                rows = conn.execute(
                    "SELECT fingerprint, est_cost FROM axis_dimension_fingerprint_index WHERE est_cost > 0 ORDER BY updated_at DESC LIMIT ?",
                    (int(max(1, int(limit))),),
                ).fetchall()
        except Exception:
            return {}
        out: dict[str, float] = {}
        for row in rows:
            try:
                fp = str(row[0] or "")
                est = float(row[1] or 0.0)
            except Exception:
                continue
            if not fp or est <= 0.0 or fp in out:
                continue
            out[fp] = est
        return out

    def _worker_plan_cache_key(
        self, *, stage_label: str, workers: int, plan_all
    ) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(_RUN_CFG_CACHE_ENGINE_VERSION).encode("utf-8"))
        hasher.update(str(stage_label).strip().lower().encode("utf-8"))
        hasher.update(str(int(workers)).encode("utf-8"))
        hasher.update(str(len(plan_all)).encode("utf-8"))
        for item in plan_all:
            cfg = (
                item[0]
                if isinstance(item, tuple)
                and len(item) >= 1
                and isinstance(item[0], ConfigBundle)
                else None
            )
            if cfg is None:
                hasher.update(str(item).encode("utf-8"))
                continue
            hasher.update(hashlib.sha1(_milestone_key(cfg).encode("utf-8")).digest())
            hasher.update(
                hashlib.sha1(_axis_dimension_fingerprint(cfg).encode("utf-8")).digest()
            )
        return hasher.hexdigest()

    def _worker_plan_cache_get(self, *, cache_key: str) -> list[list[int]] | None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return None
        try:
            with self.run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT payload_json FROM worker_plan_cache WHERE cache_key=?",
                    (str(cache_key),),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        try:
            payload = json.loads(str(row[0]))
        except Exception:
            return None
        if not isinstance(payload, list):
            return None
        out: list[list[int]] = []
        try:
            for bucket in payload:
                if not isinstance(bucket, list):
                    return None
                out.append([int(idx) for idx in bucket if int(idx) >= 0])
        except Exception:
            return None
        return out

    def _worker_plan_cache_set(
        self, *, cache_key: str, buckets: list[list[int]]
    ) -> None:
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            payload_json = json.dumps(buckets, sort_keys=False, default=str)
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO worker_plan_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (str(cache_key), str(payload_json), float(pytime.time())),
                )
            self.worker_plan_cache_writes += 1
        except Exception:
            return
