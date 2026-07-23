"""SweepCacheStore capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time as pytime
from .fingerprints import (
    _RUN_CFG_CACHE_ENGINE_VERSION,
)


class SweepCacheStore:
    def _stage_cache_scope(self, stage_label: str) -> str:
        stage_key = str(stage_label).strip().lower()
        if not stage_key:
            return ""
        return (
            f"{stage_key}|{_RUN_CFG_CACHE_ENGINE_VERSION}|"
            f"m{int(self.run_min_trades)}"
        )

    def _run_cfg_persistent_conn(self) -> sqlite3.Connection | None:
        if not bool(self.run_cfg_persistent_enabled):
            return None
        if self.run_cfg_persistent_conn is not None:
            return self.run_cfg_persistent_conn
        try:
            conn = sqlite3.connect(
                str(self.run_cfg_persistent_path),
                timeout=15.0,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_cfg_cache (cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS axis_dimension_fingerprint_index ("
                "fingerprint TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL, "
                "est_cost REAL NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS worker_plan_cache (cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_cell_status ("
                "stage_label TEXT NOT NULL, "
                "strategy_fingerprint TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, strategy_fingerprint, axis_dimension_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cartesian_cell_manifest ("
                "stage_label TEXT NOT NULL, "
                "dimension_vector_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "strategy_fingerprint TEXT NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, dimension_vector_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cartesian_rank_manifest ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "rank_lo INTEGER NOT NULL, "
                "rank_hi INTEGER NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, rank_lo, rank_hi))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cartesian_rank_cursor ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "next_rank INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_unresolved_summary ("
                "manifest_name TEXT NOT NULL, "
                "stage_label TEXT NOT NULL, "
                "plan_signature TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "total INTEGER NOT NULL, "
                "unresolved_count INTEGER NOT NULL, "
                "resolved_count INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(manifest_name, stage_label, plan_signature, window_signature, total))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rank_dominance_stamp ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "dominance_signature TEXT NOT NULL, "
                "rank_lo INTEGER NOT NULL, "
                "rank_hi INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, dominance_signature, rank_lo, rank_hi))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_frontier ("
                "stage_label TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "run_min_trades INTEGER NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "best_pnl_over_dd REAL, "
                "best_pnl REAL, "
                "best_win_rate REAL, "
                "best_trades INTEGER, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, axis_dimension_fingerprint, window_signature, run_min_trades))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rank_bin_runtime ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "rank_bin INTEGER NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "total_eval_sec REAL NOT NULL, "
                "cache_hits INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, rank_bin))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dimension_value_utility ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "dimension_key TEXT NOT NULL, "
                "dimension_value TEXT NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "cache_hits INTEGER NOT NULL, "
                "total_eval_sec REAL NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, dimension_key, dimension_value))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dimension_upper_bound ("
                "stage_label TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "best_pnl_over_dd REAL, "
                "best_pnl REAL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, axis_dimension_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS planner_heartbeat ("
                "stage_label TEXT NOT NULL, "
                "worker_id INTEGER NOT NULL, "
                "last_seen REAL NOT NULL, "
                "tested INTEGER NOT NULL, "
                "cached_hits INTEGER NOT NULL, "
                "total INTEGER NOT NULL, "
                "eta_sec REAL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, worker_id))"
            )
            self.run_cfg_persistent_conn = conn
            return conn
        except Exception:
            self.run_cfg_persistent_enabled = False
            self.run_cfg_persistent_conn = None
            return None

    def _run_cfg_persistent_key(
        self,
        *,
        strategy_fingerprint: str,
        axis_dimension_fingerprint: str,
        window_signature: str,
    ) -> str:
        raw = {
            "version": str(_RUN_CFG_CACHE_ENGINE_VERSION),
            "strategy_fingerprint": str(strategy_fingerprint),
            "axis_dimension_fingerprint": str(axis_dimension_fingerprint),
            "window_signature": str(window_signature),
        }
        return hashlib.sha1(
            json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _run_cfg_persistent_cached_lookup(
        self, *, cache_key: str
    ) -> dict | None | object:
        key = str(cache_key)
        if key not in self.run_cfg_persistent_payload_cache:
            return self._RUN_CFG_CACHE_UNSET
        cached_payload = self.run_cfg_persistent_payload_cache.get(key)
        if isinstance(cached_payload, dict):
            return dict(cached_payload)
        if cached_payload is None:
            return None
        return self._RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_cached_store(
        self, *, cache_key: str, payload: dict | None | object
    ) -> dict | None | object:
        key = str(cache_key)
        if isinstance(payload, dict):
            payload_out = dict(payload)
            self.run_cfg_persistent_payload_cache[key] = payload_out
            return dict(payload_out)
        if payload is None:
            self.run_cfg_persistent_payload_cache[key] = None
            return None
        self.run_cfg_persistent_payload_cache[key] = self._RUN_CFG_CACHE_MISS
        return self._RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_decode_payload(
        self, *, payload_json_raw: object
    ) -> dict | None | object:
        try:
            payload = json.loads(str(payload_json_raw))
        except Exception:
            return self._RUN_CFG_CACHE_MISS
        if payload is None:
            return None
        if isinstance(payload, dict):
            return dict(payload)
        return self._RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_get(self, *, cache_key: str) -> dict | None | object:
        key = str(cache_key)
        cached = self._run_cfg_persistent_cached_lookup(cache_key=str(key))
        if cached is not self._RUN_CFG_CACHE_UNSET:
            return cached
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return self._run_cfg_persistent_cached_store(
                cache_key=str(key), payload=self._RUN_CFG_CACHE_MISS
            )
        try:
            with self.run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT payload_json FROM run_cfg_cache WHERE cache_key=?",
                    (key,),
                ).fetchone()
        except Exception:
            return self._run_cfg_persistent_cached_store(
                cache_key=str(key), payload=self._RUN_CFG_CACHE_MISS
            )
        if row is None:
            return self._run_cfg_persistent_cached_store(
                cache_key=str(key), payload=self._RUN_CFG_CACHE_MISS
            )
        decoded = self._run_cfg_persistent_decode_payload(payload_json_raw=row[0])
        return self._run_cfg_persistent_cached_store(
            cache_key=str(key), payload=decoded
        )

    def _run_cfg_persistent_get_many(
        self, *, cache_keys: list[str]
    ) -> dict[str, dict | None]:
        out: dict[str, dict | None] = {}
        missing_db_keys: list[str] = []
        seen: set[str] = set()
        for raw_key in cache_keys:
            key = str(raw_key or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            cached = self._run_cfg_persistent_cached_lookup(cache_key=str(key))
            if cached is not self._RUN_CFG_CACHE_UNSET:
                if cached is not self._RUN_CFG_CACHE_MISS:
                    out[key] = cached if isinstance(cached, dict) else None
                continue
            missing_db_keys.append(key)

        if not missing_db_keys:
            return out

        conn = self._run_cfg_persistent_conn()
        if conn is None:
            for key in missing_db_keys:
                self._run_cfg_persistent_cached_store(
                    cache_key=str(key), payload=self._RUN_CFG_CACHE_MISS
                )
            return out

        found: set[str] = set()
        chunk_size = 400
        try:
            with self.run_cfg_persistent_lock:
                for start_i in range(0, len(missing_db_keys), chunk_size):
                    chunk = missing_db_keys[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT cache_key, payload_json FROM run_cfg_cache WHERE cache_key IN ({placeholders})",
                        tuple(chunk),
                    ).fetchall()
                    for cache_key_raw, payload_json_raw in rows:
                        key = str(cache_key_raw or "").strip()
                        if not key:
                            continue
                        found.add(key)
                        decoded = self._run_cfg_persistent_decode_payload(
                            payload_json_raw=payload_json_raw
                        )
                        stored = self._run_cfg_persistent_cached_store(
                            cache_key=str(key), payload=decoded
                        )
                        if stored is self._RUN_CFG_CACHE_MISS:
                            continue
                        out[key] = stored if isinstance(stored, dict) else None
        except Exception:
            for key in missing_db_keys:
                if key not in self.run_cfg_persistent_payload_cache:
                    self._run_cfg_persistent_cached_store(
                        cache_key=str(key), payload=self._RUN_CFG_CACHE_MISS
                    )
            return out

        for key in missing_db_keys:
            if key not in found:
                self._run_cfg_persistent_cached_store(
                    cache_key=str(key), payload=self._RUN_CFG_CACHE_MISS
                )
        return out

    def _run_cfg_persistent_flush_pending(self, *, force: bool = False) -> None:
        if not self.run_cfg_persistent_pending:
            return
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            self.run_cfg_persistent_pending.clear()
            self.run_cfg_persistent_last_flush_ts = float(pytime.perf_counter())
            return
        now = float(pytime.perf_counter())
        if not bool(force):
            if len(self.run_cfg_persistent_pending) < int(
                self.run_cfg_persistent_batch_write_size
            ):
                if (now - float(self.run_cfg_persistent_last_flush_ts)) < float(
                    self.run_cfg_persistent_batch_write_interval_sec
                ):
                    return
        payload = list(self.run_cfg_persistent_pending.items())
        if not payload:
            self.run_cfg_persistent_last_flush_ts = float(now)
            return
        try:
            wall_ts = float(pytime.time())
            rows = [
                (str(cache_key), str(payload_json), float(wall_ts))
                for cache_key, (payload_json, _payload_obj) in payload
            ]
            with self.run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO run_cfg_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    rows,
                )
            for cache_key, (_payload_json, payload_obj) in payload:
                self._run_cfg_persistent_cached_store(
                    cache_key=str(cache_key),
                    payload=(
                        dict(payload_obj) if isinstance(payload_obj, dict) else None
                    ),
                )
            self.run_cfg_persistent_pending.clear()
            self.run_cfg_persistent_last_flush_ts = float(now)
        except Exception:
            return

    def _run_cfg_persistent_set(self, *, cache_key: str, payload: dict | None) -> None:
        key = str(cache_key)
        conn = self._run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            payload_obj = dict(payload) if isinstance(payload, dict) else None
            payload_json = json.dumps(payload_obj, sort_keys=True, default=str)
            if bool(self.run_cfg_persistent_ram_first_enabled):
                self.run_cfg_persistent_pending[str(key)] = (
                    str(payload_json),
                    payload_obj if isinstance(payload_obj, dict) else None,
                )
                self._run_cfg_persistent_cached_store(
                    cache_key=str(key),
                    payload=(
                        dict(payload_obj) if isinstance(payload_obj, dict) else None
                    ),
                )
                self._run_cfg_persistent_flush_pending(force=False)
                return
            with self.run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO run_cfg_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (key, payload_json, float(pytime.time())),
                )
            self._run_cfg_persistent_cached_store(
                cache_key=str(key),
                payload=(dict(payload_obj) if isinstance(payload_obj, dict) else None),
            )
        except Exception:
            return
