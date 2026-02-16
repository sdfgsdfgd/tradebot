"""Shared in-memory + persistent cache service for derived bar series."""
from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Callable

Validator = Callable[[object], bool]


@dataclass
class SeriesCacheService:
    _memory: dict[tuple[str, object], object] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def get(self, *, namespace: str, key: object) -> object | None:
        with self._lock:
            return self._memory.get((str(namespace), key))

    def set(self, *, namespace: str, key: object, value: object) -> object:
        with self._lock:
            self._memory[(str(namespace), key)] = value
        return value

    def clear(self, *, namespace: str | None = None) -> None:
        with self._lock:
            if namespace is None:
                self._memory.clear()
                return
            ns = str(namespace)
            for key in [k for k in self._memory.keys() if k[0] == ns]:
                self._memory.pop(key, None)

    def get_persistent(
        self,
        *,
        db_path: Path | None,
        namespace: str,
        key_hash: str,
        validator: Validator | None = None,
    ) -> object | None:
        if db_path is None:
            return None
        try:
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS series_cache ("
                    "namespace TEXT NOT NULL, "
                    "cache_key TEXT NOT NULL, "
                    "payload BLOB NOT NULL, "
                    "updated_at TEXT NOT NULL, "
                    "PRIMARY KEY(namespace, cache_key)"
                    ")"
                )
                row = conn.execute(
                    "SELECT payload FROM series_cache WHERE namespace = ? AND cache_key = ?",
                    (str(namespace), str(key_hash)),
                ).fetchone()
                if row is None or not row[0]:
                    return None
                value = pickle.loads(row[0])
                if validator is not None and not bool(validator(value)):
                    return None
                return value
            finally:
                conn.close()
        except Exception:
            return None

    def has_persistent(
        self,
        *,
        db_path: Path | None,
        namespace: str,
        key_hash: str,
    ) -> bool:
        if db_path is None:
            return False
        try:
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS series_cache ("
                    "namespace TEXT NOT NULL, "
                    "cache_key TEXT NOT NULL, "
                    "payload BLOB NOT NULL, "
                    "updated_at TEXT NOT NULL, "
                    "PRIMARY KEY(namespace, cache_key)"
                    ")"
                )
                row = conn.execute(
                    "SELECT 1 FROM series_cache WHERE namespace = ? AND cache_key = ?",
                    (str(namespace), str(key_hash)),
                ).fetchone()
                return bool(row is not None)
            finally:
                conn.close()
        except Exception:
            return False

    def set_persistent(
        self,
        *,
        db_path: Path | None,
        namespace: str,
        key_hash: str,
        value: object,
    ) -> None:
        if db_path is None:
            return
        try:
            payload = sqlite3.Binary(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS series_cache ("
                    "namespace TEXT NOT NULL, "
                    "cache_key TEXT NOT NULL, "
                    "payload BLOB NOT NULL, "
                    "updated_at TEXT NOT NULL, "
                    "PRIMARY KEY(namespace, cache_key)"
                    ")"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO series_cache(namespace, cache_key, payload, updated_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (str(namespace), str(key_hash), payload),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return


_DEFAULT_SERIES_CACHE = SeriesCacheService()


def series_cache_service() -> SeriesCacheService:
    return _DEFAULT_SERIES_CACHE
