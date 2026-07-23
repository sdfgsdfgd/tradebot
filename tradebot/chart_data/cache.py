"""Shared in-memory + persistent cache service for derived bar series."""
from __future__ import annotations

import pickle
import sqlite3
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Callable

from .series import BarSeries, BarSeriesSignature, bar_series_signature

Validator = Callable[[object], bool]
_CACHE_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS series_cache ("
    "namespace TEXT NOT NULL, "
    "cache_key TEXT NOT NULL, "
    "payload BLOB NOT NULL, "
    "updated_at TEXT NOT NULL, "
    "PRIMARY KEY(namespace, cache_key)"
    ")"
)


@dataclass
class SeriesCacheService:
    _memory: dict[tuple[str, object], object] = field(default_factory=dict)
    _revisions: OrderedDict[int, tuple[object, BarSeriesSignature]] = field(
        default_factory=OrderedDict
    )
    _lock: RLock = field(default_factory=RLock)

    def revision(self, bars) -> BarSeriesSignature:
        """Memoize exact tape identity; loaded bar sequences are immutable by contract."""
        source = bars.bars if isinstance(bars, BarSeries) else bars
        identity = id(source)
        with self._lock:
            cached = self._revisions.get(identity)
            if cached is not None and cached[0] is source:
                self._revisions.move_to_end(identity)
                return cached[1]

        revision = bar_series_signature(source)
        with self._lock:
            self._revisions[identity] = (source, revision)
            self._revisions.move_to_end(identity)
            while len(self._revisions) > 16:
                self._revisions.popitem(last=False)
        return revision

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
                self._revisions.clear()
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
        return self.get_persistent_many(
            db_path=db_path,
            namespace=namespace,
            key_hashes=(key_hash,),
            validator=validator,
        ).get(str(key_hash))

    def get_persistent_many(
        self,
        *,
        db_path: Path | None,
        namespace: str,
        key_hashes: Iterable[str],
        validator: Validator | None = None,
    ) -> dict[str, object]:
        keys = tuple(dict.fromkeys(str(key) for key in key_hashes))
        if db_path is None or not keys:
            return {}
        try:
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            try:
                conn.execute(_CACHE_SCHEMA)
                out: dict[str, object] = {}
                for offset in range(0, len(keys), 500):
                    chunk = keys[offset : offset + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        "SELECT cache_key, payload FROM series_cache "
                        f"WHERE namespace = ? AND cache_key IN ({placeholders})",
                        (str(namespace), *chunk),
                    ).fetchall()
                    for key, payload in rows:
                        if not payload:
                            continue
                        try:
                            value = pickle.loads(payload)
                        except Exception:
                            continue
                        if validator is None or bool(validator(value)):
                            out[str(key)] = value
                return out
            finally:
                conn.close()
        except Exception:
            return {}

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
                conn.execute(_CACHE_SCHEMA)
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
        self.set_persistent_many(
            db_path=db_path,
            namespace=namespace,
            values={str(key_hash): value},
        )

    def set_persistent_many(
        self,
        *,
        db_path: Path | None,
        namespace: str,
        values: Mapping[str, object],
    ) -> None:
        if db_path is None or not values:
            return
        try:
            rows = []
            for key, value in values.items():
                try:
                    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    continue
                rows.append(
                    (str(namespace), str(key), sqlite3.Binary(payload))
                )
            if not rows:
                return
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            try:
                conn.execute(_CACHE_SCHEMA)
                conn.executemany(
                    "INSERT OR REPLACE INTO series_cache(namespace, cache_key, payload, updated_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return


_DEFAULT_SERIES_CACHE = SeriesCacheService()


def series_cache_service() -> SeriesCacheService:
    return _DEFAULT_SERIES_CACHE
