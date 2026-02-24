"""Thin async wrapper over ib_insync for snapshot-style position pulls."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from ib_insync import (
    AccountValue,
    Contract,
    ExecutionFilter,
    Forex,
    Future,
    IB,
    LimitOrder,
    PnL,
    PnLSingle,
    PortfolioItem,
    Stock,
    Ticker,
    Trade,
    util,
)

from .backtest.trading_calendar import expected_sessions, session_label_et
from .config import IBKRConfig
from .time_utils import NaiveTsMode, now_et as _now_et, now_et_naive as _now_et_naive, to_et as _to_et_shared

# region Constants
_INDEX_STRIP_SYMBOLS = ("NQ", "ES", "YM")
_INDEX_STRIP_EXCHANGE_HINTS: dict[str, str] = {
    "NQ": "CME",
    "ES": "CME",
    "YM": "CBOT",
}
_PROXY_SYMBOLS = ("QQQ", "SPY", "DIA", "TQQQ")
_PREMARKET_START = dtime(4, 0)
_RTH_START = dtime(9, 30)
_RTH_END = dtime(16, 0)
_AFTER_END = dtime(20, 0)
_OVERNIGHT_END = dtime(3, 50)
_PROXY_CONTRACT_QUOTE_PROBE_INITIAL_SEC = 1.5
_PROXY_CONTRACT_QUOTE_PROBE_RETRY_SEC = 12.0
_INDEX_QUOTE_PROBE_INITIAL_SEC = 2.0
_MAIN_CONTRACT_QUOTE_PROBE_INITIAL_SEC = 0.35
_MAIN_CONTRACT_QUOTE_PROBE_RETRY_SEC = 4.0
_MAIN_CONTRACT_STALE_WATCHDOG_SEC = 3.0
_MAIN_CONTRACT_STALE_REPROBE_SEC = 18.0
_MAIN_CONTRACT_TOPLINE_STALE_REPROBE_SEC = 45.0
_MAIN_CONTRACT_STALE_RESUBSCRIBE_SEC = 75.0
_MAIN_CONTRACT_SNAPSHOT_WAIT_SEC = 0.55
_MAIN_CONTRACT_HISTORICAL_ATTEMPT_TIMEOUT_SEC = 4.5
_HISTORICAL_REQUEST_TIMEOUT_SEC = 12.0
_MAX_HISTORICAL_REQUEST_DIAG_ENTRIES = 256
_ENTITLEMENT_ERROR_CODES = (10167, 354, 10089, 10090, 10091, 10168)
_PRICE_INCREMENT_DETAILS_TIMEOUT_SEC = 1.5
_PRICE_INCREMENT_WAIT_TIMEOUT_SEC = 3.0
_ORDER_RECONCILE_TIMEOUT_SEC = 1.5
_ORDER_RECONCILE_MIN_INTERVAL_SEC = 0.75
_CLIENT_ID_FAST_PROBE_TIMEOUT_SEC = 2.0
_MATCHING_SYMBOL_TIMEOUT_INITIAL_SEC = 5.0
_MATCHING_SYMBOL_TIMEOUT_RETRY_SEC = 7.0
_MATCHING_SYMBOL_MAX_ATTEMPTS = 2
_MATCHING_SYMBOL_RETRY_BASE_SEC = 0.12
_MATCHING_SYMBOL_EMPTY_TIMEOUT_RATIO = 0.9
_CLIENT_ID_CONFLICT_PATTERNS = (
    "client id already in use",
    "clientid already in use",
    "duplicate client id",
    "duplicate clientid",
    "already connected",
    "already in use",
)
_SEARCH_TERM_ALIASES_BY_MODE: dict[str, dict[str, tuple[str, ...]]] = {
    "STK": {
        "SILVER": ("SLV",),
        "XAG": ("SLV",),
        "GOLD": ("GLD",),
        "XAU": ("GLD",),
        "BITCOIN": ("BITU", "IBIT", "BITO", "FBTC", "BTC"),
        "BTC": ("BITU", "IBIT", "BITO", "BITCOIN"),
    },
    "OPT": {
        "SILVER": ("SLV",),
        "XAG": ("SLV",),
        "GOLD": ("GLD",),
        "XAU": ("GLD",),
        "BITCOIN": ("BITU", "IBIT", "BITO", "FBTC", "BTC"),
        "BTC": ("BITU", "IBIT", "BITO", "BITCOIN"),
    },
    "FUT": {
        "SILVER": ("SI",),
        "XAG": ("SI",),
        "GOLD": ("GC",),
        "XAU": ("GC",),
        "MICRO CRUDE": ("MCL",),
        "MICRO CRUDE OIL": ("MCL",),
        "BITCOIN": ("MBT", "BTC"),
        "BTC": ("MBT", "BITCOIN"),
    },
    "FOP": {
        "SILVER": ("SI",),
        "XAG": ("SI",),
        "GOLD": ("GC",),
        "XAU": ("GC",),
        "1OZ": ("GC",),
        "MICRO CRUDE": ("MCL",),
        "MICRO CRUDE OIL": ("MCL",),
        "BITCOIN": ("MBT", "BTC"),
        "BTC": ("MBT", "BITCOIN"),
    },
}
_FUT_EXCHANGE_HINTS: dict[str, tuple[str, ...]] = {
    "SI": ("COMEX", "NYMEX"),
    "GC": ("COMEX", "NYMEX"),
    "CL": ("NYMEX",),
    "MCL": ("NYMEX",),
    "NG": ("NYMEX",),
    "HG": ("COMEX",),
    "ES": ("GLOBEX", "CME"),
    "MES": ("GLOBEX", "CME"),
    "NQ": ("GLOBEX", "CME"),
    "MNQ": ("GLOBEX", "CME"),
    "YM": ("CBOT", "ECBOT", "GLOBEX"),
    "MYM": ("CBOT", "ECBOT", "GLOBEX"),
    "RTY": ("GLOBEX", "CME"),
    "M2K": ("GLOBEX", "CME"),
}
_CONTRACT_LABEL_HINTS: dict[str, str] = {
    # Common futures / futures-option roots where IB matching-symbol descriptions are often empty.
    "CL": "WTI Crude Oil",
    "MCL": "Micro WTI Crude Oil",
    "NG": "Natural Gas",
    "GC": "Gold",
    "SI": "Silver",
    "HG": "Copper",
    "ES": "E-mini S&P 500",
    "MES": "Micro E-mini S&P 500",
    "NQ": "E-mini Nasdaq-100",
    "MNQ": "Micro E-mini Nasdaq-100",
    "YM": "E-mini Dow",
    "MYM": "Micro E-mini Dow",
    "RTY": "E-mini Russell 2000",
    "M2K": "Micro E-mini Russell 2000",
    "BTC": "Bitcoin",
    "MBT": "Micro Bitcoin",
}
# endregion


# region Models
@dataclass(frozen=True)
class OhlcvBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
# endregion


# region Session Helpers
def _session_flags(now: datetime) -> tuple[bool, bool]:
    """Return (outside_rth, include_overnight) for US equity sessions."""
    current = now.time()
    outside_rth = (_PREMARKET_START <= current < _RTH_START) or (
        _RTH_END <= current < _AFTER_END
    )
    include_overnight = current >= _AFTER_END or current < _OVERNIGHT_END
    return outside_rth, include_overnight


def _session_bucket(now: datetime) -> str:
    current = now.time()
    if _RTH_START <= current < _RTH_END:
        return "RTH"
    if _PREMARKET_START <= current < _RTH_START:
        return "PRE"
    if _RTH_END <= current < _AFTER_END:
        return "POST"
    return "OVERNIGHT"


def _futures_session_is_open(now: datetime) -> bool:
    """Approximate CME/COMEX session state in ET.

    Most futures are open Sun 18:00 -> Fri 17:00 ET with a daily 17:00-18:00
    maintenance break and a full Saturday closure.
    """
    current = now.time()
    weekday = int(now.weekday())  # Monday=0 ... Sunday=6

    if weekday == 5:
        return False
    if weekday == 6:
        return current >= dtime(18, 0)
    if weekday == 4 and current >= dtime(17, 0):
        return False
    if dtime(17, 0) <= current < dtime(18, 0):
        return False
    return True


def _futures_md_ladder(now: datetime) -> tuple[int, ...]:
    # Prefer live streams first; gracefully fall back to delayed types when
    # entitlement or venue availability does not permit live quotes.
    if _futures_session_is_open(now):
        return (1, 2, 3, 4)
    return (1, 2, 4, 3)
# endregion


# region Client
class IBKRClient:
    def _store_historical_request_payload(self, payload: dict[str, object]) -> None:
        self._last_historical_request = payload
        contract = payload.get("contract")
        con_id = 0
        if isinstance(contract, dict):
            try:
                con_id = int(contract.get("con_id", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
        if con_id > 0:
            self._last_historical_request_by_con_id[int(con_id)] = payload
            if len(self._last_historical_request_by_con_id) > int(_MAX_HISTORICAL_REQUEST_DIAG_ENTRIES):
                oldest = next(iter(self._last_historical_request_by_con_id))
                self._last_historical_request_by_con_id.pop(oldest, None)

    async def _connect_ib(self, ib: IB, *, client_id: int) -> None:
        timeout = max(1.0, float(self._config.connect_timeout_sec))
        key = int(client_id or 0)
        if key > 0 and key in self._fast_connect_probe_client_ids:
            self._fast_connect_probe_client_ids.discard(key)
            timeout = min(timeout, float(_CLIENT_ID_FAST_PROBE_TIMEOUT_SEC))
        if hasattr(ib, "connectAsync"):
            await ib.connectAsync(
                self._config.host,
                self._config.port,
                clientId=int(client_id),
                timeout=timeout,
            )
            return
        await asyncio.to_thread(
            ib.connect,
            self._config.host,
            self._config.port,
            int(client_id),
            timeout,
        )

    @staticmethod
    def _safe_disconnect(ib: IB) -> None:
        if not ib.isConnected():
            return
        try:
            ib.disconnect()
        except OSError:
            # Avoid noisy shutdown if the socket is already closed.
            pass

    @staticmethod
    def _is_retryable_connect_error(exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionRefusedError, TimeoutError, asyncio.TimeoutError)):
            return True
        if isinstance(exc, OSError):
            err_no = int(getattr(exc, "errno", 0) or 0)
            if err_no in (61, 111, 10061):
                return True
        msg = str(exc)
        return "Connect call failed" in msg or "API connection failed" in msg

    @staticmethod
    def _is_client_id_conflict_error(exc: BaseException) -> bool:
        message = str(exc or "").strip().lower()
        if not message:
            return False
        if "client" not in message and "duplicate" not in message:
            return False
        return any(pattern in message for pattern in _CLIENT_ID_CONFLICT_PATTERNS)

    @staticmethod
    def _is_api_session_init_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return True
        message = str(exc or "").strip().lower()
        if not message:
            return False
        if "socket connection broken while connecting" in message:
            return True
        if "peer closed connection" in message:
            return True
        return "api connection failed" in message and "timeout" in message

    @classmethod
    def _is_pool_rotatable_connect_error(cls, exc: BaseException) -> bool:
        return bool(
            cls._is_client_id_conflict_error(exc)
            or cls._is_api_session_init_error(exc)
        )

    @staticmethod
    def _resolve_path_template(raw_path: str) -> str:
        value = str(raw_path or "").strip()
        if not value:
            return ""

        def _replace(match: re.Match[str]) -> str:
            key = str(match.group(1) or "").strip()
            fallback = str(match.group(2) or "")
            if not key:
                return fallback
            env_value = os.getenv(key)
            return env_value if env_value is not None and env_value != "" else fallback

        templated = re.sub(r"\$\{([^}:]+):-([^}]+)\}", _replace, value)
        return os.path.expandvars(os.path.expanduser(templated))

    def _pair_is_in_pool(self, main_id: int, proxy_id: int) -> bool:
        if proxy_id != (main_id + 1):
            return False
        return bool(
            self._client_id_pool_start <= int(main_id) <= self._client_id_pool_end
            and self._client_id_pool_start <= int(proxy_id) <= self._client_id_pool_end
        )

    def _normalize_seed_pair(self, main_id: int, proxy_id: int) -> tuple[int, int]:
        if self._pair_is_in_pool(main_id, proxy_id):
            return int(main_id), int(proxy_id)
        if self._client_id_ring:
            normalized_main = int(self._client_id_ring[0])
            return normalized_main, int(normalized_main + 1)
        normalized_main = max(1, int(main_id or 1))
        return normalized_main, max(normalized_main + 1, int(proxy_id or 0))

    def _candidate_pairs(
        self,
        *,
        preferred_pair: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        self._prune_pair_quarantine()
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []

        def _push(
            pair: tuple[int, int] | None,
            *,
            require_pool: bool = True,
            allow_quarantined: bool = False,
        ) -> None:
            if pair is None:
                return
            main_raw, proxy_raw = pair
            main_id = int(main_raw or 0)
            proxy_id = int(proxy_raw or 0)
            if require_pool and not self._pair_is_in_pool(main_id, proxy_id):
                return
            key = (main_id, proxy_id)
            if key in seen:
                return
            if not allow_quarantined and self._is_pair_quarantined(main_id, proxy_id):
                return
            seen.add(key)
            out.append(key)

        _push(preferred_pair)
        _push((int(self._main_client_id), int(self._proxy_client_id)))
        max_attempts = max(1, int(self._config.client_id_burst_attempts))
        ring_size = len(self._client_id_ring)
        spins = 0
        while len(out) < max_attempts and ring_size > 0 and spins < (ring_size * 2):
            idx = int(self._client_id_ring_index % ring_size)
            main_id = int(self._client_id_ring[idx])
            self._client_id_ring_index = (idx + 1) % ring_size
            _push((main_id, main_id + 1))
            spins += 1
        if not out:
            _push(
                (
                    int(self._main_client_id),
                    int(self._proxy_client_id),
                ),
                require_pool=False,
                allow_quarantined=True,
            )
        return out[:max_attempts]

    def _prune_pair_quarantine(self) -> None:
        if not self._pair_quarantine_until_mono:
            return
        now = time.monotonic()
        stale = [
            key for key, until_mono in self._pair_quarantine_until_mono.items() if float(until_mono) <= now
        ]
        for key in stale:
            self._pair_quarantine_until_mono.pop(key, None)

    def _is_pair_quarantined(self, main_id: int, proxy_id: int) -> bool:
        self._prune_pair_quarantine()
        key = (int(main_id), int(proxy_id))
        until_mono = self._pair_quarantine_until_mono.get(key)
        if until_mono is None:
            return False
        return bool(float(until_mono) > time.monotonic())

    def _quarantine_pair(
        self,
        main_id: int,
        proxy_id: int,
        *,
        delay_sec: float | None = None,
    ) -> None:
        if not self._pair_is_in_pool(int(main_id), int(proxy_id)):
            return
        ttl = float(delay_sec) if delay_sec is not None else float(self._config.client_id_quarantine_sec)
        ttl = max(1.0, ttl)
        key = (int(main_id), int(proxy_id))
        until_mono = time.monotonic() + ttl
        previous = self._pair_quarantine_until_mono.get(key)
        if previous is None or float(until_mono) > float(previous):
            self._pair_quarantine_until_mono[key] = float(until_mono)

    def _client_id_backoff_remaining_sec(self) -> float:
        until_mono = self._client_id_backoff_until_mono
        if until_mono is None:
            return 0.0
        return max(0.0, float(until_mono) - time.monotonic())

    def _arm_client_id_backoff(self) -> float:
        attempts = max(0, int(self._client_id_backoff_failures))
        base = max(0.5, float(self._config.client_id_backoff_initial_sec))
        multiplier = max(1.0, float(self._config.client_id_backoff_multiplier))
        capped = max(base, float(self._config.client_id_backoff_max_sec))
        delay = min(capped, base * (multiplier ** float(attempts)))
        jitter_ratio = max(0.0, min(0.9, float(self._config.client_id_backoff_jitter_ratio)))
        if jitter_ratio > 0:
            jitter = random.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio)
            delay = max(0.5, float(delay) * float(jitter))
        self._client_id_backoff_failures = attempts + 1
        self._client_id_backoff_until_mono = time.monotonic() + float(delay)
        return float(delay)

    def _reset_client_id_backoff(self) -> None:
        self._client_id_backoff_failures = 0
        self._client_id_backoff_until_mono = None

    def _maybe_persist_client_id_pair(self) -> None:
        if not self._ib.isConnected() or not self._ib_proxy.isConnected():
            return
        main_id = self._connected_main_client_id
        proxy_id = self._connected_proxy_client_id
        if main_id is None or proxy_id is None:
            return
        if not self._pair_is_in_pool(int(main_id), int(proxy_id)):
            return
        path = self._client_id_state_path
        if path is None:
            return
        payload = {
            "main_client_id": int(main_id),
            "proxy_client_id": int(proxy_id),
            "updated_at_epoch": int(time.time()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

    def _load_persisted_client_id_pair(self) -> tuple[int, int] | None:
        path = self._client_id_state_path
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            main_id = int(payload.get("main_client_id") or 0)
            proxy_id = int(payload.get("proxy_client_id") or 0)
        except (TypeError, ValueError):
            return None
        if not self._pair_is_in_pool(main_id, proxy_id):
            return None
        return int(main_id), int(proxy_id)

    async def _connect_main_with_client_id_pool(self) -> None:
        cooldown = self._client_id_backoff_remaining_sec()
        if cooldown > 0:
            raise ConnectionError(f"IBKR client-id backoff active ({cooldown:.1f}s remaining)")
        last_exc: Exception | None = None
        conflict_detected = False
        rotatable_detected = False
        preferred_pair = self._persisted_client_id_pair
        candidates = self._candidate_pairs(preferred_pair=preferred_pair)
        for main_id, proxy_id in candidates:
            self._persisted_client_id_pair = None
            if (
                preferred_pair is not None
                and len(candidates) > 1
                and (int(main_id), int(proxy_id)) == preferred_pair
            ):
                self._fast_connect_probe_client_ids.add(int(main_id))
            self._main_client_id = int(main_id)
            self._proxy_client_id = int(proxy_id)
            try:
                await self._connect_ib(self._ib, client_id=int(self._main_client_id))
                self._connected_main_client_id = int(self._main_client_id)
                self._reset_client_id_backoff()
                self._maybe_persist_client_id_pair()
                return
            except Exception as exc:  # noqa: PERF203 - clarity for selective retry behavior
                if self._is_pool_rotatable_connect_error(exc):
                    if self._is_client_id_conflict_error(exc):
                        conflict_detected = True
                    else:
                        rotatable_detected = True
                    self._quarantine_pair(main_id, proxy_id)
                    self._safe_disconnect(self._ib)
                    self._connected_main_client_id = None
                    last_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                    continue
                raise
        if conflict_detected:
            delay = self._arm_client_id_backoff()
            raise ConnectionError(f"IBKR main client-id pool exhausted; backoff {delay:.1f}s")
        if rotatable_detected:
            delay = self._arm_client_id_backoff()
            detail = type(last_exc).__name__ if last_exc is not None else "error"
            raise ConnectionError(
                f"IBKR main client-id connect retries exhausted; backoff {delay:.1f}s ({detail})"
            )
        if last_exc is not None:
            raise last_exc
        raise ConnectionError("IBKR main client connection failed")

    async def _connect_proxy_with_client_id_pool(self) -> None:
        cooldown = self._client_id_backoff_remaining_sec()
        if cooldown > 0:
            raise ConnectionError(f"IBKR client-id backoff active ({cooldown:.1f}s remaining)")
        preferred = (int(self._main_client_id), int(self._proxy_client_id))
        last_exc: Exception | None = None
        conflict_detected = False
        rotatable_detected = False
        candidates = self._candidate_pairs(preferred_pair=preferred)
        for main_id, proxy_id in candidates:
            if len(candidates) > 1 and (int(main_id), int(proxy_id)) == preferred:
                self._fast_connect_probe_client_ids.add(int(proxy_id))
            self._main_client_id = int(main_id)
            self._proxy_client_id = int(proxy_id)
            try:
                await self._connect_ib(self._ib_proxy, client_id=int(self._proxy_client_id))
                self._connected_proxy_client_id = int(self._proxy_client_id)
                self._proxy_error = None
                self._reset_client_id_backoff()
                self._maybe_persist_client_id_pair()
                return
            except Exception as exc:  # noqa: PERF203 - clarity for selective retry behavior
                if self._is_pool_rotatable_connect_error(exc):
                    if self._is_client_id_conflict_error(exc):
                        conflict_detected = True
                    else:
                        rotatable_detected = True
                    self._quarantine_pair(main_id, proxy_id)
                    self._safe_disconnect(self._ib_proxy)
                    self._connected_proxy_client_id = None
                    last_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                    continue
                raise
        if conflict_detected:
            delay = self._arm_client_id_backoff()
            raise ConnectionError(f"IBKR proxy client-id pool exhausted; backoff {delay:.1f}s")
        if rotatable_detected:
            delay = self._arm_client_id_backoff()
            detail = type(last_exc).__name__ if last_exc is not None else "error"
            raise ConnectionError(
                f"IBKR proxy client-id connect retries exhausted; backoff {delay:.1f}s ({detail})"
            )
        if last_exc is not None:
            raise last_exc
        raise ConnectionError("IBKR proxy client connection failed")

    def _request_reconnect(self) -> None:
        if self._shutdown:
            return
        if not self._reconnect_in_progress() or self._reconnect_fast_deadline is None:
            self._reconnect_fast_deadline = (
                time.monotonic() + float(self._config.reconnect_timeout_sec)
            )
        self._reconnect_requested = True
        self._start_reconnect_loop()

    def _reconnect_in_progress(self) -> bool:
        return bool(
            self._reconnect_requested
            and self._reconnect_task
            and not self._reconnect_task.done()
        )

    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._ib = IB()
        self._ib_proxy = IB()
        self._shutdown = False
        self._connect_lock = asyncio.Lock()
        self._connect_proxy_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self._proxy_lock = asyncio.Lock()
        self._historical_lock = asyncio.Lock()
        self._historical_proxy_lock = asyncio.Lock()
        self._order_reconcile_lock = asyncio.Lock()
        self._account_updates_started = False
        self._index_contracts: dict[str, Contract] = {}
        self._index_tickers: dict[str, Ticker] = {}
        self._index_task: asyncio.Task | None = None
        self._index_probe_task: asyncio.Task | None = None
        self._index_requalify_on_reload = False
        self._index_session_flags: tuple[bool, bool] | None = None
        self._index_futures_session_open: bool | None = None
        self._index_session_include_overnight: bool | None = None
        self._index_error: str | None = None
        # Index strip starts live-first and degrades to delayed when needed.
        self._index_force_delayed = False
        self._proxy_contracts: dict[str, Contract] = {}
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_task: asyncio.Task | None = None
        self._proxy_error: str | None = None
        self._proxy_force_delayed = False
        self._proxy_probe_task: asyncio.Task | None = None
        self._proxy_contract_force_delayed: set[int] = set()
        self._proxy_contract_probe_tasks: dict[int, asyncio.Task] = {}
        self._proxy_contract_live_tasks: dict[int, asyncio.Task] = {}
        self._proxy_contract_delayed_tasks: dict[int, asyncio.Task] = {}
        self._main_contract_probe_tasks: dict[int, asyncio.Task] = {}
        self._main_contract_watchdog_tasks: dict[int, asyncio.Task] = {}
        self._proxy_session_bucket: str | None = _session_bucket(_now_et())
        self._proxy_session_include_overnight: bool | None = None
        self._detail_tickers: dict[int, tuple[IB, Ticker]] = {}
        self._ticker_owners: dict[int, set[str]] = {}
        self._historical_bar_cache: dict[
            tuple[str, int, str, str, bool, str, str],
            tuple[list[tuple[datetime, float]], float] | tuple[list[tuple[datetime, float]], float, float],
        ] = {}
        self._historical_bar_ohlcv_cache: dict[
            tuple[str, int, str, str, bool, str, str],
            tuple[list[OhlcvBar], float] | tuple[list[OhlcvBar], float, float],
        ] = {}
        self._front_future_cache: dict[tuple[str, str], tuple[Contract, float]] = {}
        self._update_callback: Callable[[], None] | None = None
        self._stream_listeners: set[Callable[[], None]] = set()
        self._pnl: PnL | None = None
        self._pnl_account: str | None = None
        self._pnl_single_by_con_id: dict[int, PnLSingle] = {}
        self._pnl_single_account: str | None = None
        self._account_value_cache: dict[tuple[str, str], tuple[float, datetime]] = {}
        self._session_close_cache: dict[
            int, tuple[float | None, float | None, float | None, float]
        ] = {}
        self._last_historical_request: dict[str, object] | None = None
        self._last_historical_request_by_con_id: dict[int, dict[str, object]] = {}
        self._order_error_cache: dict[int, tuple[float, int, str]] = {}
        self._market_rule_increments: dict[int, tuple[tuple[float, float], ...]] = {}
        self._contract_price_increments: dict[int, tuple[tuple[float, float], ...]] = {}
        self._contract_price_increment_tasks: dict[int, asyncio.Task[tuple[tuple[float, float], ...]]] = {}
        self._fast_connect_probe_client_ids: set[int] = set()
        self._fx_rate_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._last_order_reconcile_mono: float = 0.0
        self._farm_connectivity_lost = False
        self._reconnect_requested = False
        self._resubscribe_main_needed = False
        self._resubscribe_proxy_needed = False
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_fast_deadline: float | None = None
        pool_start = max(1, int(self._config.client_id_pool_start))
        pool_end = int(self._config.client_id_pool_end)
        if pool_end <= pool_start:
            pool_start = 500
            pool_end = 899
        self._client_id_pool_start = int(pool_start)
        self._client_id_pool_end = int(pool_end)
        self._client_id_ring: list[int] = [
            candidate
            for candidate in range(self._client_id_pool_start, self._client_id_pool_end + 1, 2)
            if (candidate + 1) <= self._client_id_pool_end
        ]
        if not self._client_id_ring:
            fallback = max(1, int(self._config.client_id or 1))
            self._client_id_ring = [fallback]
        seed_main, seed_proxy = self._normalize_seed_pair(
            int(self._config.client_id),
            int(self._config.proxy_client_id),
        )
        self._main_client_id = int(seed_main)
        self._proxy_client_id = int(seed_proxy)
        self._client_id_ring_index = 0
        if self._client_id_ring and self._main_client_id in self._client_id_ring:
            self._client_id_ring_index = (
                self._client_id_ring.index(self._main_client_id) + 1
            ) % len(self._client_id_ring)
        state_path = self._resolve_path_template(str(self._config.client_id_state_file))
        self._client_id_state_path: Path | None = Path(state_path) if state_path else None
        self._persisted_client_id_pair = self._load_persisted_client_id_pair()
        if self._persisted_client_id_pair is not None:
            persisted_main, persisted_proxy = self._persisted_client_id_pair
            self._main_client_id = int(persisted_main)
            self._proxy_client_id = int(persisted_proxy)
            if self._client_id_ring and self._main_client_id in self._client_id_ring:
                self._client_id_ring_index = (
                    self._client_id_ring.index(self._main_client_id) + 1
                ) % len(self._client_id_ring)
        self._client_id_backoff_failures = 0
        self._client_id_backoff_until_mono: float | None = None
        self._pair_quarantine_until_mono: dict[tuple[int, int], float] = {}
        self._connected_main_client_id: int | None = None
        self._connected_proxy_client_id: int | None = None
        self._ib.errorEvent += self._on_error_main
        self._ib.disconnectedEvent += self._on_disconnected_main
        self._ib.updatePortfolioEvent += self._on_stream_update
        self._ib.pendingTickersEvent += self._on_stream_update
        self._ib.pnlEvent += self._on_stream_update
        self._ib.pnlSingleEvent += self._on_pnl_single
        self._ib.accountValueEvent += self._on_account_value
        for event_name in (
            "openOrderEvent",
            "orderStatusEvent",
            "execDetailsEvent",
            "commissionReportEvent",
            "newOrderEvent",
            "cancelOrderEvent",
        ):
            event = getattr(self._ib, event_name, None)
            if event is None:
                continue
            try:
                event += self._on_stream_update
            except Exception:
                continue
        self._ib_proxy.errorEvent += self._on_error_proxy
        self._ib_proxy.disconnectedEvent += self._on_disconnected_proxy
        self._ib_proxy.pendingTickersEvent += self._on_stream_update

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    def reconnect_phase(self) -> str | None:
        if not self._reconnect_in_progress():
            return None
        deadline = self._reconnect_fast_deadline
        if deadline is None or time.monotonic() < deadline:
            return "fast"
        return "slow"

    def connection_state(self) -> str:
        phase = self.reconnect_phase()
        if phase == "fast":
            return "reconnecting-fast"
        if phase == "slow":
            return "reconnecting-slow"
        main_connected = self._ib.isConnected()
        proxy_connected = self._ib_proxy.isConnected()
        if main_connected and proxy_connected:
            return "connected"
        if main_connected or proxy_connected:
            return "degraded"
        return "disconnected"

    async def connect(self) -> None:
        self._shutdown = False
        if self._ib.isConnected():
            return
        if self._reconnect_in_progress() and asyncio.current_task() is not self._reconnect_task:
            raise ConnectionError("IBKR reconnect in progress")
        async with self._connect_lock:
            if self._ib.isConnected():
                return
            try:
                await self._connect_main_with_client_id_pool()
            except Exception as exc:
                if self._is_retryable_connect_error(exc) or self._is_client_id_conflict_error(exc):
                    self._request_reconnect()
                elif "client-id" in str(exc).lower():
                    self._request_reconnect()
                raise

    async def connect_proxy(self) -> None:
        self._shutdown = False
        if self._ib_proxy.isConnected():
            return
        if self._reconnect_in_progress() and asyncio.current_task() is not self._reconnect_task:
            raise ConnectionError("IBKR reconnect in progress")
        async with self._connect_proxy_lock:
            if self._ib_proxy.isConnected():
                return
            try:
                await self._connect_proxy_with_client_id_pool()
                self._proxy_error = None
            except Exception as exc:
                self._proxy_error = str(exc)
                if self._is_retryable_connect_error(exc) or self._is_client_id_conflict_error(exc):
                    self._request_reconnect()
                elif "client-id" in str(exc).lower():
                    self._request_reconnect()
                raise

    async def disconnect(self) -> None:
        self._shutdown = True
        self._stop_reconnect_loop()
        self._reconnect_requested = False
        self._reconnect_fast_deadline = None
        self._clear_pnl_single_subscriptions(cancel=True)
        self._safe_disconnect(self._ib)
        self._safe_disconnect(self._ib_proxy)
        self._account_updates_started = False
        self._index_tickers = {}
        self._index_task = None
        if self._index_probe_task and not self._index_probe_task.done():
            self._index_probe_task.cancel()
        self._index_probe_task = None
        self._index_requalify_on_reload = False
        self._index_session_flags = None
        self._index_futures_session_open = None
        self._index_session_include_overnight = None
        self._index_error = None
        self._index_force_delayed = False
        self._proxy_tickers = {}
        self._proxy_task = None
        self._proxy_error = None
        self._proxy_force_delayed = False
        self._proxy_probe_task = None
        self._proxy_contract_force_delayed = set()
        self._proxy_session_bucket = None
        self._proxy_session_include_overnight = None
        for task in self._proxy_contract_probe_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._proxy_contract_live_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._proxy_contract_delayed_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._main_contract_probe_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._main_contract_watchdog_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._contract_price_increment_tasks.values():
            if task and not task.done():
                task.cancel()
        self._proxy_contract_probe_tasks = {}
        self._proxy_contract_live_tasks = {}
        self._proxy_contract_delayed_tasks = {}
        self._main_contract_probe_tasks = {}
        self._main_contract_watchdog_tasks = {}
        self._contract_price_increment_tasks = {}
        self._detail_tickers = {}
        self._ticker_owners = {}
        self._pnl = None
        self._pnl_account = None
        self._pnl_single_by_con_id = {}
        self._pnl_single_account = None
        self._account_value_cache = {}
        self._session_close_cache = {}
        self._last_historical_request = None
        self._last_historical_request_by_con_id = {}
        self._fx_rate_cache = {}
        self._fast_connect_probe_client_ids = set()
        self._resubscribe_main_needed = False
        self._resubscribe_proxy_needed = False
        self._reset_client_id_backoff()
        self._connected_main_client_id = None
        self._connected_proxy_client_id = None
        self._last_order_reconcile_mono = 0.0

    async def fetch_portfolio(self) -> list[PortfolioItem]:
        """Fetch a snapshot of portfolio items (filtered by account if provided)."""
        async with self._lock:
            await self._ensure_account_updates()
            account = self._config.account or ""
            items = list(self._ib.portfolio(account))
            self._sync_pnl_single_subscriptions(items, account)
            return items

    async def fetch_index_tickers(self) -> dict[str, Ticker]:
        async with self._index_lock:
            await self._ensure_index_tickers()
            return dict(self._index_tickers)

    def start_index_tickers(self) -> None:
        if self._index_requalify_on_reload:
            self._start_index_resubscribe(requalify=True)
            return
        if self._index_task and not self._index_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._index_task = loop.create_task(self._load_index_tickers())

    def index_tickers(self) -> dict[str, Ticker]:
        return dict(self._index_tickers)

    def index_error(self) -> str | None:
        return self._index_error

    def start_proxy_tickers(self) -> None:
        if self._proxy_task and not self._proxy_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proxy_task = loop.create_task(self._load_proxy_tickers())

    def proxy_tickers(self) -> dict[str, Ticker]:
        return dict(self._proxy_tickers)

    def proxy_error(self) -> str | None:
        return self._proxy_error

    def last_historical_request(self, contract: Contract | None = None) -> dict[str, object] | None:
        payload = None
        if contract is not None:
            try:
                con_id = int(getattr(contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
            if con_id > 0:
                payload = self._last_historical_request_by_con_id.get(con_id)
        if payload is None:
            payload = self._last_historical_request
        if not isinstance(payload, dict):
            return None
        return copy.deepcopy(payload)

    def set_update_callback(self, callback: Callable[[], None]) -> None:
        self._update_callback = callback

    def add_stream_listener(self, callback: Callable[[], None]) -> None:
        self._stream_listeners.add(callback)

    def remove_stream_listener(self, callback: Callable[[], None]) -> None:
        self._stream_listeners.discard(callback)

    def pnl(self) -> PnL | None:
        return self._pnl

    def pnl_unrealized(self) -> float | None:
        if self._pnl is None:
            return None
        return self._clean_pnl_stream_value(getattr(self._pnl, "unrealizedPnL", None))

    def pnl_realized(self) -> float | None:
        if self._pnl is None:
            return None
        return self._clean_pnl_stream_value(getattr(self._pnl, "realizedPnL", None))

    @staticmethod
    def _clean_pnl_stream_value(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        # IB can surface sentinel/invalid values (for example max-float) before streams settle.
        if math.isnan(parsed) or not math.isfinite(parsed):
            return None
        if abs(parsed) >= 1e307:
            return None
        return float(parsed)

    def pnl_single_unrealized(self, con_id: int) -> float | None:
        try:
            key = int(con_id or 0)
        except (TypeError, ValueError):
            key = 0
        if key <= 0:
            return None
        entry = self._pnl_single_by_con_id.get(key)
        if entry is None:
            return None
        return self._clean_pnl_stream_value(getattr(entry, "unrealizedPnL", None))

    def pnl_single_daily(self, con_id: int) -> float | None:
        try:
            key = int(con_id or 0)
        except (TypeError, ValueError):
            key = 0
        if key <= 0:
            return None
        entry = self._pnl_single_by_con_id.get(key)
        if entry is None:
            return None
        return self._clean_pnl_stream_value(getattr(entry, "dailyPnL", None))

    def has_pnl_single_subscription(self, con_id: int) -> bool:
        try:
            key = int(con_id or 0)
        except (TypeError, ValueError):
            return False
        if key <= 0:
            return False
        return key in self._pnl_single_by_con_id

    def portfolio_item(self, con_id: int) -> PortfolioItem | None:
        if not con_id or not self._ib.isConnected():
            return None
        account = self._config.account or ""
        for item in self._ib.portfolio(account):
            try:
                if int(item.contract.conId or 0) == con_id:
                    return item
            except (TypeError, ValueError):
                continue
        return None

    def ticker_for_con_id(self, con_id: int) -> Ticker | None:
        if not con_id:
            return None
        entry = self._detail_tickers.get(int(con_id))
        if not entry:
            return None
        _ib, ticker = entry
        return ticker

    @staticmethod
    def _coerce_price_increments(raw: object) -> tuple[tuple[float, float], ...]:
        if not isinstance(raw, (list, tuple)):
            return ()
        rows: list[tuple[float, float]] = []
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                low_edge = float(row[0])
                increment = float(row[1])
            except (TypeError, ValueError):
                continue
            if increment <= 0:
                continue
            rows.append((max(0.0, low_edge), increment))
        if not rows:
            return ()
        rows.sort(key=lambda entry: entry[0])
        deduped: list[tuple[float, float]] = []
        for low_edge, increment in rows:
            if deduped and abs(deduped[-1][0] - low_edge) < 1e-9:
                deduped[-1] = (low_edge, increment)
            else:
                deduped.append((low_edge, increment))
        return tuple(deduped)

    @classmethod
    def _price_increment_for_value(
        cls,
        increments: tuple[tuple[float, float], ...],
        *,
        price: float | None,
    ) -> float | None:
        normalized = cls._coerce_price_increments(list(increments))
        if not normalized:
            return None
        try:
            ref = float(price) if price is not None else 0.0
        except (TypeError, ValueError):
            ref = 0.0
        ref = max(0.0, ref)
        selected = normalized[0][1]
        for low_edge, increment in normalized:
            if ref >= low_edge:
                selected = increment
            else:
                break
        return selected if selected > 0 else None

    @staticmethod
    def _attach_price_increments(target: object | None, increments: tuple[tuple[float, float], ...]) -> None:
        if target is None or not increments:
            return
        setattr(target, "tbPriceIncrements", increments)

    async def _prime_contract_price_increments(
        self,
        contract: Contract,
        *,
        ticker: Ticker | None = None,
    ) -> tuple[tuple[float, float], ...]:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id <= 0:
            return await self._prime_contract_price_increments_impl(contract, ticker=ticker)

        task = self._contract_price_increment_tasks.get(con_id)
        if task is None or task.done():
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._prime_contract_price_increments_impl(contract, ticker=ticker)
            )
            self._contract_price_increment_tasks[con_id] = task

            def _cleanup(done_task: asyncio.Task, *, key: int = con_id) -> None:
                current = self._contract_price_increment_tasks.get(key)
                if current is done_task:
                    self._contract_price_increment_tasks.pop(key, None)

            task.add_done_callback(_cleanup)

        try:
            increments = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=float(_PRICE_INCREMENT_WAIT_TIMEOUT_SEC),
            )
        except asyncio.TimeoutError:
            increments = self._contract_price_increments.get(con_id, ())
        except Exception:
            increments = self._contract_price_increments.get(con_id, ())
        if increments:
            self._attach_price_increments(contract, increments)
            if ticker is not None:
                self._attach_price_increments(ticker, increments)
                self._attach_price_increments(getattr(ticker, "contract", None), increments)
        return increments

    async def _prime_contract_price_increments_impl(
        self,
        contract: Contract,
        *,
        ticker: Ticker | None = None,
    ) -> tuple[tuple[float, float], ...]:
        source_contract = contract
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id > 0:
            cached = self._contract_price_increments.get(con_id)
            if cached:
                # Self-heal older flat-minTick cache entries for FOP that can cause
                # invalid prices above tier boundaries.
                if (
                    sec_type == "FOP"
                    and len(cached) == 1
                    and float(cached[0][0]) <= 0.0
                    and float(cached[0][1]) <= 0.05
                ):
                    self._contract_price_increments.pop(con_id, None)
                else:
                    self._attach_price_increments(contract, cached)
                    if source_contract is not contract:
                        self._attach_price_increments(source_contract, cached)
                    if ticker is not None:
                        self._attach_price_increments(ticker, cached)
                        self._attach_price_increments(getattr(ticker, "contract", None), cached)
                    return cached
        await self.connect()

        async def _req_details(req_contract: Contract) -> list[object]:
            try:
                return list(
                    await asyncio.wait_for(
                        self._ib.reqContractDetailsAsync(req_contract),
                        timeout=float(_PRICE_INCREMENT_DETAILS_TIMEOUT_SEC),
                    )
                    or []
                )
            except Exception:
                return []

        details: list[object] = []
        details = await _req_details(contract)
        if con_id > 0 and not details:
            details = await _req_details(Contract(conId=con_id))
        detail = None
        if con_id > 0:
            for candidate in details:
                detail_contract = getattr(candidate, "contract", None)
                try:
                    detail_con_id = int(getattr(detail_contract, "conId", 0) or 0)
                except (TypeError, ValueError):
                    detail_con_id = 0
                if detail_con_id and detail_con_id == con_id:
                    detail = candidate
                    break
        if detail is None and details:
            detail = details[0]
        detail_contract = getattr(detail, "contract", None) if detail is not None else None
        if detail_contract is not None:
            contract = detail_contract
            try:
                con_id = int(getattr(detail_contract, "conId", 0) or con_id)
            except (TypeError, ValueError):
                pass
        increments: tuple[tuple[float, float], ...] = ()
        market_rule_ids = ""
        if detail is not None:
            market_rule_ids = str(getattr(detail, "marketRuleIds", "") or "").strip()
            try:
                detail_min_tick = float(getattr(detail, "minTick", 0.0) or 0.0)
            except (TypeError, ValueError):
                detail_min_tick = 0.0
            if detail_min_tick > 0:
                setattr(contract, "minTick", detail_min_tick)
                if source_contract is not contract:
                    setattr(source_contract, "minTick", detail_min_tick)
                if ticker is not None:
                    setattr(ticker, "minTick", detail_min_tick)
                    ticker_contract = getattr(ticker, "contract", None)
                    if ticker_contract is not None:
                        setattr(ticker_contract, "minTick", detail_min_tick)
        if con_id > 0 and not market_rule_ids:
            retry_details = await _req_details(Contract(conId=con_id))
            if retry_details:
                retry_detail = retry_details[0]
                retry_contract = getattr(retry_detail, "contract", None)
                if retry_contract is not None:
                    contract = retry_contract
                    try:
                        con_id = int(getattr(retry_contract, "conId", 0) or con_id)
                    except (TypeError, ValueError):
                        pass
                market_rule_ids = str(getattr(retry_detail, "marketRuleIds", "") or "").strip()
                try:
                    retry_min_tick = float(getattr(retry_detail, "minTick", 0.0) or 0.0)
                except (TypeError, ValueError):
                    retry_min_tick = 0.0
                if retry_min_tick > 0:
                    setattr(contract, "minTick", retry_min_tick)
                    if source_contract is not contract:
                        setattr(source_contract, "minTick", retry_min_tick)
                    if ticker is not None:
                        setattr(ticker, "minTick", retry_min_tick)
                        ticker_contract = getattr(ticker, "contract", None)
                        if ticker_contract is not None:
                            setattr(ticker_contract, "minTick", retry_min_tick)
        for raw_rule_id in [segment.strip() for segment in market_rule_ids.split(",") if segment.strip()]:
            try:
                rule_id = int(raw_rule_id)
            except (TypeError, ValueError):
                continue
            rule_increments = self._market_rule_increments.get(rule_id)
            if not rule_increments:
                rule_increments = ()
                for _attempt in range(2):
                    try:
                        rows = list(
                            await asyncio.wait_for(
                                self._ib.reqMarketRuleAsync(rule_id),
                                timeout=float(_PRICE_INCREMENT_DETAILS_TIMEOUT_SEC),
                            )
                            or []
                        )
                    except Exception:
                        rows = []
                    rule_increments = self._coerce_price_increments(
                        [
                            (
                                getattr(row, "lowEdge", None),
                                getattr(row, "increment", None),
                            )
                            for row in rows
                        ]
                    )
                    if rule_increments:
                        break
                    await asyncio.sleep(0.05)
                if rule_increments:
                    self._market_rule_increments[rule_id] = rule_increments
            if rule_increments:
                increments = rule_increments
                break
        if not increments and con_id > 0:
            entry = self._detail_tickers.get(con_id)
            if entry is not None:
                _entry_ib, entry_ticker = entry
                increments = self._coerce_price_increments(getattr(entry_ticker, "tbPriceIncrements", None))
                if not increments:
                    increments = self._coerce_price_increments(
                        getattr(getattr(entry_ticker, "contract", None), "tbPriceIncrements", None)
                    )
                if increments and ticker is not None:
                    self._attach_price_increments(ticker, increments)
                    self._attach_price_increments(getattr(ticker, "contract", None), increments)
        if not increments:
            increments = self._coerce_price_increments(getattr(source_contract, "tbPriceIncrements", None))
        if con_id > 0 and increments:
            self._contract_price_increments[con_id] = increments
        self._attach_price_increments(contract, increments)
        if source_contract is not contract:
            self._attach_price_increments(source_contract, increments)
        if ticker is not None:
            self._attach_price_increments(ticker, increments)
            self._attach_price_increments(getattr(ticker, "contract", None), increments)
        return increments

    def _normalize_limit_price_increment(self, contract: Contract, limit_price: float) -> float:
        try:
            price_value = float(limit_price)
        except (TypeError, ValueError):
            raise ValueError("limit_price must be numeric") from None
        increments = self._coerce_price_increments(getattr(contract, "tbPriceIncrements", None))
        if not increments:
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id > 0:
                increments = self._contract_price_increments.get(con_id, ())
                if not increments:
                    entry = self._detail_tickers.get(con_id)
                    if entry is not None:
                        _ib, detail_ticker = entry
                        increments = self._coerce_price_increments(
                            getattr(detail_ticker, "tbPriceIncrements", None)
                        )
                        if not increments:
                            increments = self._coerce_price_increments(
                                getattr(getattr(detail_ticker, "contract", None), "tbPriceIncrements", None)
                            )
                        if increments:
                            self._contract_price_increments[con_id] = increments
                            self._attach_price_increments(contract, increments)
        increment = self._price_increment_for_value(increments, price=price_value)
        if increment is None:
            try:
                raw_tick = float(getattr(contract, "minTick", 0.0) or 0.0)
            except (TypeError, ValueError):
                raw_tick = 0.0
            increment = raw_tick if raw_tick > 0 else None
        if increment is None or increment <= 0:
            return float(price_value)
        return round(float(price_value) / float(increment)) * float(increment)

    @staticmethod
    def _owner_defers_price_increment_prime(owner: str) -> bool:
        owner_key = str(owner or "").strip().lower()
        if not owner_key:
            return False
        deferred_prefixes = ("details", "details-chase", "search", "positions", "favorites")
        for prefix in deferred_prefixes:
            if owner_key == prefix or owner_key.startswith(f"{prefix}:"):
                return True
        return False

    @staticmethod
    def _consume_background_task(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except Exception:
            return

    async def _prime_price_increments_for_ticker(
        self,
        contract: Contract,
        *,
        sec_type: str,
        owner: str,
        ticker: Ticker | None,
    ) -> None:
        if sec_type not in ("OPT", "FOP", "FUT"):
            return
        if self._owner_defers_price_increment_prime(owner):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            task = loop.create_task(self._prime_contract_price_increments(contract, ticker=ticker))
            task.add_done_callback(self._consume_background_task)
            return
        try:
            await self._prime_contract_price_increments(contract, ticker=ticker)
        except Exception:
            pass

    async def ensure_ticker(self, contract: Contract, *, owner: str = "default") -> Ticker:
        con_id = int(contract.conId or 0)
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        use_proxy = sec_type in ("STK", "OPT")
        contract_force_delayed = bool(
            use_proxy and con_id and con_id in self._proxy_contract_force_delayed
        )
        requested_md_type = 1
        proxy_md_type = 1
        if use_proxy:
            await self.connect_proxy()
            proxy_md_type = 3 if self._proxy_force_delayed or contract_force_delayed else 1
            self._ib_proxy.reqMarketDataType(proxy_md_type)
            requested_md_type = int(proxy_md_type)
            ib = self._ib_proxy
        else:
            await self.connect()
            md_ladder = _futures_md_ladder(_now_et()) if sec_type in ("FUT", "FOP") else (3,)
            self._ib.reqMarketDataType(int(md_ladder[0]))
            requested_md_type = int(md_ladder[0])
            ib = self._ib
        req_contract = contract
        if contract.secType == "STK":
            _, include_overnight = _session_flags(_now_et())
            req_contract = self._stock_market_data_contract(
                contract,
                include_overnight=include_overnight,
                delayed=bool(use_proxy and proxy_md_type == 3),
            )
        elif sec_type in ("FUT", "OPT", "FOP"):
            req_contract = self._normalize_derivative_market_data_contract(
                contract,
                sec_type=sec_type,
            )
        cached = self._detail_tickers.get(con_id) if con_id else None
        if cached:
            if con_id:
                self._ticker_owners.setdefault(con_id, set()).add(owner)
            cached_ib, cached_ticker = cached
            desired_exchange = str(getattr(req_contract, "exchange", "") or "").strip().upper()
            current_exchange = str(getattr(cached_ticker.contract, "exchange", "") or "").strip().upper()
            exchange_sensitive = sec_type in ("STK", "OPT", "FUT", "FOP")
            if exchange_sensitive and desired_exchange and desired_exchange != current_exchange:
                try:
                    cached_ib.cancelMktData(cached_ticker.contract)
                except Exception:
                    pass
                ticker = ib.reqMktData(req_contract)
                self._detail_tickers[con_id] = (ib, ticker)
                if use_proxy and con_id:
                    if con_id in self._proxy_contract_force_delayed:
                        if not self._ticker_has_data(ticker):
                            self._start_proxy_contract_delayed_resubscribe(req_contract)
                    else:
                        self._start_proxy_contract_quote_probe(req_contract)
                elif con_id and sec_type in ("FUT", "FOP"):
                    self._start_main_contract_quote_watchdog(req_contract)
                    if self._ticker_has_data(ticker):
                        self._tag_ticker_quote_meta(ticker, source="stream")
                    elif self._ticker_has_close_data(ticker):
                        self._tag_ticker_quote_meta(ticker, source="stream-close-only")
                    else:
                        self._start_main_contract_quote_probe(req_contract)
                await self._prime_price_increments_for_ticker(
                    req_contract,
                    sec_type=sec_type,
                    owner=owner,
                    ticker=ticker,
                )
                return ticker
            if use_proxy and con_id:
                if con_id in self._proxy_contract_force_delayed:
                    if not self._ticker_has_data(cached_ticker):
                        self._start_proxy_contract_delayed_resubscribe(req_contract)
                elif not self._ticker_has_data(cached_ticker):
                    self._start_proxy_contract_quote_probe(req_contract)
            elif con_id and sec_type in ("FUT", "FOP"):
                self._start_main_contract_quote_watchdog(req_contract)
                if self._ticker_has_data(cached_ticker):
                    self._tag_ticker_quote_meta(cached_ticker, source="stream")
                elif self._ticker_has_close_data(cached_ticker):
                    self._tag_ticker_quote_meta(cached_ticker, source="stream-close-only")
                else:
                    self._start_main_contract_quote_probe(req_contract)
            await self._prime_price_increments_for_ticker(
                req_contract,
                sec_type=sec_type,
                owner=owner,
                ticker=cached_ticker,
            )
            return cached_ticker
        ticker = ib.reqMktData(req_contract)
        try:
            ticker.marketDataType = int(requested_md_type)
        except Exception:
            pass
        if con_id:
            self._detail_tickers[con_id] = (ib, ticker)
            self._ticker_owners.setdefault(con_id, set()).add(owner)
        if use_proxy and con_id:
            if con_id in self._proxy_contract_force_delayed:
                if not self._ticker_has_data(ticker):
                    self._start_proxy_contract_delayed_resubscribe(req_contract)
            else:
                self._start_proxy_contract_quote_probe(req_contract)
        elif con_id and sec_type in ("FUT", "FOP"):
            self._start_main_contract_quote_watchdog(req_contract)
            if self._ticker_has_data(ticker):
                self._tag_ticker_quote_meta(ticker, source="stream")
            elif self._ticker_has_close_data(ticker):
                self._tag_ticker_quote_meta(ticker, source="stream-close-only")
            else:
                self._start_main_contract_quote_probe(req_contract)
        await self._prime_price_increments_for_ticker(
            req_contract,
            sec_type=sec_type,
            owner=owner,
            ticker=ticker,
        )
        return ticker

    async def refresh_live_snapshot_once(self, contract: Contract) -> str | None:
        """Try a one-shot live snapshot for an already-bound FUT/FOP detail ticker.

        Returns the applied quote-source label (for status text) on success, else None.
        """
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id <= 0:
            return None
        entry = self._detail_tickers.get(con_id)
        if entry is None:
            return None
        ib, ticker = entry
        if ib is not self._ib:
            return None
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type not in ("FUT", "FOP"):
            return None
        await self.connect()
        req_contract = getattr(ticker, "contract", None) or contract
        ok = await self._attempt_main_contract_snapshot_quote(
            req_contract,
            ticker=ticker,
            md_types=(1, 2),
        )
        if not ok:
            return None
        source = str(getattr(ticker, "tbQuoteSource", "") or "").strip()
        return source or "snapshot"

    async def place_limit_order(
        self,
        contract: Contract,
        action: str,
        quantity: float,
        limit_price: float,
        outside_rth: bool,
    ) -> Trade:
        await self.connect()
        order_contract = contract
        tif = "GTC"
        outside_session = False
        include_overnight = False
        if contract.secType == "STK":
            outside_session, include_overnight = _session_flags(_now_et())
            if include_overnight:
                order_contract = copy.copy(contract)
                order_contract.exchange = "OVERNIGHT"
                # IBKR rejects STK OVERNIGHT orders with GTC; DAY is required.
                tif = "DAY"
        if str(getattr(order_contract, "secType", "") or "").strip().upper() in ("OPT", "FOP", "FUT"):
            try:
                await self._prime_contract_price_increments(order_contract)
            except Exception:
                pass
        limit_price = self._normalize_limit_price_increment(order_contract, float(limit_price))
        order = LimitOrder(action, quantity, limit_price, tif=tif)
        if contract.secType == "STK" and outside_rth and outside_session and not include_overnight:
            order.outsideRth = True
        order_contract = _normalize_order_contract(order_contract)
        return self._ib.placeOrder(order_contract, order)

    async def modify_limit_order(self, trade: Trade, limit_price: float) -> Trade:
        """Modify an existing LIMIT order's price in place."""
        await self.connect()
        order = trade.order
        if not hasattr(order, "lmtPrice"):
            raise ValueError("modify_limit_order: trade has no lmtPrice")
        try:
            await self._prime_contract_price_increments(trade.contract)
        except Exception:
            pass
        order.lmtPrice = self._normalize_limit_price_increment(trade.contract, float(limit_price))
        return self._ib.placeOrder(trade.contract, order)

    def open_trades_for_conids(self, con_ids: Iterable[int]) -> list[Trade]:
        if not self._ib.isConnected():
            return []
        targets = {int(con_id) for con_id in con_ids if con_id}
        if not targets:
            return []
        trades: list[Trade] = []
        for trade in self._ib.openTrades():
            try:
                trade_con_id = int(getattr(trade.contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                continue
            if trade_con_id in targets:
                trades.append(trade)
        return trades

    @staticmethod
    def _trade_order_ids(trade: Trade) -> tuple[int, int]:
        order = getattr(trade, "order", None)
        try:
            order_id = int(getattr(order, "orderId", 0) or 0)
        except (TypeError, ValueError):
            order_id = 0
        try:
            perm_id = int(getattr(order, "permId", 0) or 0)
        except (TypeError, ValueError):
            perm_id = 0
        return max(0, order_id), max(0, perm_id)

    @staticmethod
    def _ids_match(
        *,
        order_id: int,
        perm_id: int,
        candidate_order_id: int,
        candidate_perm_id: int,
    ) -> bool:
        return bool(
            (order_id > 0 and candidate_order_id > 0 and order_id == candidate_order_id)
            or (perm_id > 0 and candidate_perm_id > 0 and perm_id == candidate_perm_id)
        )

    def trade_for_order_ids(
        self,
        *,
        order_id: int = 0,
        perm_id: int = 0,
        include_closed: bool = True,
    ) -> Trade | None:
        try:
            wanted_order_id = int(order_id or 0)
        except (TypeError, ValueError):
            wanted_order_id = 0
        try:
            wanted_perm_id = int(perm_id or 0)
        except (TypeError, ValueError):
            wanted_perm_id = 0
        if wanted_order_id <= 0 and wanted_perm_id <= 0:
            return None
        if not self._ib.isConnected():
            return None
        try:
            trades = self._ib.trades() if include_closed else self._ib.openTrades()
        except Exception:
            trades = []
        for trade in reversed(list(trades or [])):
            candidate_order_id, candidate_perm_id = self._trade_order_ids(trade)
            if self._ids_match(
                order_id=wanted_order_id,
                perm_id=wanted_perm_id,
                candidate_order_id=candidate_order_id,
                candidate_perm_id=candidate_perm_id,
            ):
                return trade
        return None

    def _has_open_trade_for_order_ids(self, *, order_id: int, perm_id: int) -> bool:
        if not self._ib.isConnected():
            return False
        try:
            trades = self._ib.openTrades()
        except Exception:
            trades = []
        for trade in trades or []:
            candidate_order_id, candidate_perm_id = self._trade_order_ids(trade)
            if self._ids_match(
                order_id=order_id,
                perm_id=perm_id,
                candidate_order_id=candidate_order_id,
                candidate_perm_id=candidate_perm_id,
            ):
                return True
        return False

    def _executed_qty_for_order_ids(self, *, order_id: int, perm_id: int) -> float:
        if not self._ib.isConnected():
            return 0.0
        try:
            fills = list(self._ib.fills() or [])
        except Exception:
            fills = []
        executed_qty = 0.0
        cumulative_only_max = 0.0
        has_incremental_shares = False
        for fill in fills:
            execution = getattr(fill, "execution", None)
            if execution is None:
                continue
            try:
                candidate_order_id = int(getattr(execution, "orderId", 0) or 0)
            except (TypeError, ValueError):
                candidate_order_id = 0
            try:
                candidate_perm_id = int(getattr(execution, "permId", 0) or 0)
            except (TypeError, ValueError):
                candidate_perm_id = 0
            if not self._ids_match(
                order_id=order_id,
                perm_id=perm_id,
                candidate_order_id=candidate_order_id,
                candidate_perm_id=candidate_perm_id,
            ):
                continue
            try:
                shares = float(getattr(execution, "shares", None))
            except (TypeError, ValueError):
                shares = 0.0
            if shares > 0:
                has_incremental_shares = True
                executed_qty += float(shares)
                continue
            try:
                cum_qty = float(getattr(execution, "cumQty", None))
            except (TypeError, ValueError):
                cum_qty = 0.0
            if cum_qty > cumulative_only_max:
                cumulative_only_max = float(cum_qty)
        if has_incremental_shares:
            return max(0.0, float(executed_qty))
        return max(0.0, float(cumulative_only_max))

    @staticmethod
    def _effective_order_status(
        *,
        raw_status: str,
        open_present: bool,
        is_done: bool,
        filled_qty: float,
        remaining_qty: float,
        executed_qty: float,
    ) -> str:
        status = str(raw_status or "").strip()
        terminal_statuses = ("Filled", "Cancelled", "ApiCancelled", "Inactive")
        if status in terminal_statuses:
            return status
        total_filled = max(float(filled_qty), float(executed_qty))
        remaining = max(0.0, float(remaining_qty))
        if total_filled > 0 and remaining <= 1e-9:
            return "Filled"
        if open_present and status in ("", "PendingSubmit", "PendingSubmission", "ApiPending"):
            return "Submitted"
        if total_filled > 0 and status in ("", "PendingSubmit", "PendingSubmission", "ApiPending"):
            return "Submitted"
        if status:
            return status
        if open_present:
            return "Submitted"
        if is_done:
            return "Done"
        return "PendingSubmission"

    def _order_state_for_ids(self, *, order_id: int, perm_id: int) -> dict[str, object] | None:
        if order_id <= 0 and perm_id <= 0:
            return None
        trade = self.trade_for_order_ids(
            order_id=order_id,
            perm_id=perm_id,
            include_closed=True,
        )
        raw_status = ""
        filled_qty = 0.0
        remaining_qty = 0.0
        is_done = False
        if trade is not None:
            order_status = getattr(trade, "orderStatus", None)
            raw_status = str(getattr(order_status, "status", "") or "").strip()
            try:
                filled_qty = float(getattr(order_status, "filled", 0.0) or 0.0)
            except (TypeError, ValueError):
                filled_qty = 0.0
            try:
                remaining_qty = float(getattr(order_status, "remaining", 0.0) or 0.0)
            except (TypeError, ValueError):
                remaining_qty = 0.0
            try:
                is_done = bool(trade.isDone())
            except Exception:
                is_done = False
            trade_order_id, trade_perm_id = self._trade_order_ids(trade)
            if order_id <= 0:
                order_id = trade_order_id
            if perm_id <= 0:
                perm_id = trade_perm_id
        open_present = self._has_open_trade_for_order_ids(order_id=order_id, perm_id=perm_id)
        executed_qty = self._executed_qty_for_order_ids(order_id=order_id, perm_id=perm_id)
        effective_status = self._effective_order_status(
            raw_status=raw_status,
            open_present=open_present,
            is_done=is_done,
            filled_qty=filled_qty,
            remaining_qty=remaining_qty,
            executed_qty=executed_qty,
        )
        is_terminal = bool(
            effective_status in ("Filled", "Cancelled", "ApiCancelled", "Inactive")
            or (is_done and effective_status not in ("Submitted", "PreSubmitted"))
        )
        if (
            trade is None
            and order_id <= 0
            and perm_id <= 0
            and executed_qty <= 0
            and not open_present
        ):
            return None
        payload: dict[str, object] = {
            "order_id": int(order_id),
            "perm_id": int(perm_id),
            "raw_status": str(raw_status),
            "effective_status": str(effective_status),
            "filled_qty": float(max(filled_qty, executed_qty)),
            "remaining_qty": float(max(0.0, remaining_qty)),
            "executed_qty": float(max(0.0, executed_qty)),
            "open_present": bool(open_present),
            "is_done": bool(is_done),
            "is_terminal": bool(is_terminal),
        }
        if trade is not None:
            payload["trade"] = trade
        return payload

    def current_order_state(
        self,
        *,
        order_id: int = 0,
        perm_id: int = 0,
    ) -> dict[str, object] | None:
        try:
            wanted_order_id = int(order_id or 0)
        except (TypeError, ValueError):
            wanted_order_id = 0
        try:
            wanted_perm_id = int(perm_id or 0)
        except (TypeError, ValueError):
            wanted_perm_id = 0
        if wanted_order_id <= 0 and wanted_perm_id <= 0:
            return None
        if not self._ib.isConnected():
            return None
        return self._order_state_for_ids(
            order_id=wanted_order_id,
            perm_id=wanted_perm_id,
        )

    async def _sync_order_snapshots(self, *, include_completed: bool) -> None:
        requests = [
            asyncio.wait_for(
                self._ib.reqAllOpenOrdersAsync(),
                timeout=float(_ORDER_RECONCILE_TIMEOUT_SEC),
            ),
            asyncio.wait_for(
                self._ib.reqOpenOrdersAsync(),
                timeout=float(_ORDER_RECONCILE_TIMEOUT_SEC),
            ),
            asyncio.wait_for(
                self._ib.reqExecutionsAsync(ExecutionFilter()),
                timeout=float(_ORDER_RECONCILE_TIMEOUT_SEC),
            ),
        ]
        if include_completed:
            requests.append(
                asyncio.wait_for(
                    self._ib.reqCompletedOrdersAsync(apiOnly=False),
                    timeout=float(_ORDER_RECONCILE_TIMEOUT_SEC),
                )
            )
        await asyncio.gather(*requests, return_exceptions=True)

    async def reconcile_order_state(
        self,
        *,
        order_id: int = 0,
        perm_id: int = 0,
        force: bool = False,
    ) -> dict[str, object] | None:
        try:
            wanted_order_id = int(order_id or 0)
        except (TypeError, ValueError):
            wanted_order_id = 0
        try:
            wanted_perm_id = int(perm_id or 0)
        except (TypeError, ValueError):
            wanted_perm_id = 0
        if not self._ib.isConnected():
            try:
                await self.connect()
            except Exception:
                return None
        now = time.monotonic()
        if bool(force) or (
            now - float(self._last_order_reconcile_mono)
        ) >= float(_ORDER_RECONCILE_MIN_INTERVAL_SEC):
            async with self._order_reconcile_lock:
                now_locked = time.monotonic()
                if bool(force) or (
                    now_locked - float(self._last_order_reconcile_mono)
                ) >= float(_ORDER_RECONCILE_MIN_INTERVAL_SEC):
                    await self._sync_order_snapshots(include_completed=bool(force))
                    self._last_order_reconcile_mono = time.monotonic()
        if wanted_order_id <= 0 and wanted_perm_id <= 0:
            return None
        return self._order_state_for_ids(
            order_id=wanted_order_id,
            perm_id=wanted_perm_id,
        )

    async def cancel_trade(self, trade: Trade) -> None:
        await self.connect()
        self._ib.cancelOrder(trade.order)

    async def resolve_underlying_contract(self, contract: Contract) -> Contract | None:
        if contract.secType == "OPT":
            candidate = Stock(
                symbol=contract.symbol,
                exchange="SMART",
                currency=contract.currency or "USD",
            )
            qualified = await self._qualify_contract(candidate, use_proxy=True)
            return qualified or candidate
        if contract.secType == "FOP":
            under_con_id = int(getattr(contract, "underConId", 0) or 0)
            if not under_con_id:
                symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
                if not symbol:
                    return None
                preferred: list[str] = []
                for exchange in (
                    str(getattr(contract, "exchange", "") or "").strip().upper(),
                    str(getattr(contract, "primaryExchange", "") or "").strip().upper(),
                ):
                    if not exchange or exchange in preferred:
                        continue
                    preferred.append(exchange)
                for exchange in self._future_exchange_candidates(symbol, preferred):
                    future = await self.front_future(
                        symbol,
                        exchange=exchange,
                        cache_ttl_sec=900.0,
                    )
                    if future is not None:
                        return future
                return None
            candidate = Contract(conId=under_con_id)
            qualified = await self._qualify_contract(candidate, use_proxy=False)
            return qualified or candidate
        return None

    @classmethod
    def _historical_request_contract(cls, contract: Contract, *, sec_type: str) -> Contract:
        req_contract = contract
        if sec_type in ("STK", "OPT") and not getattr(contract, "exchange", ""):
            req_contract = copy.copy(contract)
            req_contract.exchange = "SMART"
        elif sec_type == "FOP" and not getattr(contract, "exchange", ""):
            req_contract = copy.copy(contract)
            primary_exchange = getattr(contract, "primaryExchange", "") or ""
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            preferred = [str(primary_exchange).strip().upper()] if primary_exchange else []
            candidates = cls._future_exchange_candidates(symbol, preferred)
            req_contract.exchange = candidates[0] if candidates else (primary_exchange or "CME")
        elif sec_type == "FUT" and not getattr(contract, "exchange", ""):
            req_contract = copy.copy(contract)
            primary_exchange = str(getattr(contract, "primaryExchange", "") or "").strip().upper()
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            preferred = [primary_exchange] if primary_exchange else []
            candidates = cls._future_exchange_candidates(symbol, preferred)
            req_contract.exchange = candidates[0] if candidates else (primary_exchange or "CME")
        return req_contract

    async def _request_historical_data(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        use_proxy = sec_type in ("STK", "OPT")
        timeout_sec = self._historical_timeout_sec(duration_str)

        def _record(
            status: str,
            *,
            req_contract: Contract,
            error: BaseException | None = None,
            detail: str | None = None,
            bars_count: int | None = None,
            elapsed_sec: float | None = None,
        ) -> None:
            request_ts = _now_et().isoformat()
            con_id = 0
            try:
                con_id = int(getattr(req_contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
            payload: dict[str, object] = {
                "status": str(status),
                "ts": request_ts,
                "timeout_sec": float(timeout_sec),
                "request": {
                    "duration_str": str(duration_str),
                    "bar_size": str(bar_size),
                    "what_to_show": str(what_to_show),
                    "use_rth": bool(use_rth),
                    "use_proxy": bool(use_proxy),
                },
                "contract": {
                    "con_id": int(con_id),
                    "sec_type": str(getattr(req_contract, "secType", "") or "").strip().upper(),
                    "symbol": str(getattr(req_contract, "symbol", "") or "").strip().upper(),
                    "exchange": str(getattr(req_contract, "exchange", "") or "").strip().upper(),
                    "primary_exchange": str(getattr(req_contract, "primaryExchange", "") or "").strip().upper(),
                    "currency": str(getattr(req_contract, "currency", "") or "").strip().upper(),
                },
            }
            if bars_count is not None:
                payload["bars_count"] = int(bars_count)
            if elapsed_sec is not None:
                payload["elapsed_sec"] = max(0.0, float(elapsed_sec))
            if error is not None:
                payload["error"] = str(error)
                payload["error_type"] = type(error).__name__
            if detail:
                payload["detail"] = str(detail)
            self._store_historical_request_payload(payload)

        req_contract = self._historical_request_contract(contract, sec_type=sec_type)
        try:
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib
        except Exception as exc:
            if use_proxy:
                self._proxy_error = str(exc)
            _record(
                "connect_error",
                req_contract=req_contract,
                error=exc,
                detail="historical connect failed",
            )
            return []
        started_mono = time.monotonic()
        try:
            # Use ib_insync's native historical timeout handling so the underlying
            # request is canceled and cleaned up on timeout.
            bars = await ib.reqHistoricalDataAsync(
                req_contract,
                endDateTime="",
                durationStr=str(duration_str),
                barSizeSetting=str(bar_size),
                whatToShow=str(what_to_show),
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
                timeout=float(timeout_sec),
            )
        except asyncio.TimeoutError as exc:
            elapsed_sec = max(0.0, float(time.monotonic() - started_mono))
            _record(
                "timeout",
                req_contract=req_contract,
                error=exc,
                detail=f"historical request timed out after {timeout_sec:.3f}s",
                elapsed_sec=elapsed_sec,
            )
            return []
        except Exception as exc:
            elapsed_sec = max(0.0, float(time.monotonic() - started_mono))
            _record(
                "request_error",
                req_contract=req_contract,
                error=exc,
                detail="historical request failed",
                elapsed_sec=elapsed_sec,
            )
            return []
        elapsed_sec = max(0.0, float(time.monotonic() - started_mono))
        try:
            bars_count = int(len(bars)) if bars is not None else 0
        except Exception:
            bars_count = 0
        if bars_count <= 0:
            # ib_insync timeouts return an empty container after canceling the
            # request, so classify long-empty waits as timeouts for diagnostics.
            timeout_threshold = max(
                float(timeout_sec) * 0.9,
                float(timeout_sec) - min(0.05, float(timeout_sec) * 0.1),
            )
            if elapsed_sec >= timeout_threshold:
                _record(
                    "timeout",
                    req_contract=req_contract,
                    error=asyncio.TimeoutError(),
                    detail=f"historical request timed out after {timeout_sec:.3f}s",
                    bars_count=0,
                    elapsed_sec=elapsed_sec,
                )
                return []
            _record(
                "empty",
                req_contract=req_contract,
                detail="historical response returned no bars",
                bars_count=0,
                elapsed_sec=elapsed_sec,
            )
            return []
        _record(
            "ok",
            req_contract=req_contract,
            bars_count=bars_count,
            elapsed_sec=elapsed_sec,
        )
        return bars

    @staticmethod
    def _is_intraday_bar_size(bar_size: str) -> bool:
        label = str(bar_size or "").strip().lower()
        if not label:
            return True
        return not any(token in label for token in ("day", "week", "month"))

    @staticmethod
    def _historical_timeout_sec(duration_str: str) -> float:
        base = float(_HISTORICAL_REQUEST_TIMEOUT_SEC)
        if base <= 0:
            base = 0.001
        cleaned = " ".join(str(duration_str or "").strip().upper().split())
        if cleaned.endswith("M") and " " not in cleaned:
            prefix = cleaned[:-1]
            if prefix.isdigit():
                cleaned = f"{int(prefix)} M"
        month_overrides = {
            "1 M": 80.0,
            "2 M": 100.0,
            "3 M": 120.0,
        }
        override = month_overrides.get(cleaned)
        if override is None:
            return float(base)
        return max(float(base), float(override))

    @staticmethod
    def _bar_time_et(ts: datetime) -> dtime:
        return _to_et_shared(ts, naive_ts_mode=NaiveTsMode.ET).timetz().replace(tzinfo=None)

    @classmethod
    def _is_overnight_bar(cls, ts: datetime) -> bool:
        current = cls._bar_time_et(ts)
        return current >= _AFTER_END or current < _PREMARKET_START

    @classmethod
    def _merge_full24_raw_bars(cls, *, smart: list, overnight: list) -> list:
        by_ts: dict[datetime, object] = {}
        for bar in smart or []:
            dt = cls._ib_bar_datetime(getattr(bar, "date", None))
            if dt is None:
                continue
            by_ts[dt] = bar
        for bar in overnight or []:
            dt = cls._ib_bar_datetime(getattr(bar, "date", None))
            if dt is None:
                continue
            if dt not in by_ts or cls._is_overnight_bar(dt):
                by_ts[dt] = bar
        return [by_ts[ts] for ts in sorted(by_ts.keys())]

    @classmethod
    def _full24_stitch_quality(cls, *, bars: list) -> dict[str, object]:
        first_ts: datetime | None = None
        last_ts: datetime | None = None
        sessions_by_day: dict[date, set[str]] = {}
        for bar in bars or []:
            ts = cls._ib_bar_datetime(getattr(bar, "date", None))
            if ts is None:
                continue
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts
            label = session_label_et(ts.time())
            if label is None:
                continue
            sessions_by_day.setdefault(ts.date(), set()).add(str(label))
        out: dict[str, object] = {
            "complete": True,
            "first_bar_ts": first_ts.isoformat() if isinstance(first_ts, datetime) else None,
            "last_bar_ts": last_ts.isoformat() if isinstance(last_ts, datetime) else None,
            "interior_trading_days": 0,
            "missing_days": 0,
            "missing_by_day": {},
        }
        if not isinstance(first_ts, datetime) or not isinstance(last_ts, datetime):
            out["complete"] = False
            return out
        if first_ts.date() >= last_ts.date():
            return out
        missing_by_day: dict[str, list[str]] = {}
        interior_trading_days = 0
        day = first_ts.date() + timedelta(days=1)
        last_interior = last_ts.date() - timedelta(days=1)
        while day <= last_interior:
            expected = expected_sessions(day, session_mode="full24")
            if expected:
                interior_trading_days += 1
                present = sessions_by_day.get(day, set())
                missing = sorted(str(label) for label in (set(expected) - set(present)))
                if missing:
                    missing_by_day[day.isoformat()] = missing
            day += timedelta(days=1)
        out["interior_trading_days"] = int(interior_trading_days)
        out["missing_days"] = int(len(missing_by_day))
        out["missing_by_day"] = missing_by_day
        out["complete"] = bool(not missing_by_day)
        return out

    async def _request_historical_data_for_stream(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        # For stocks, IBKR SMART full-session intraday misses OVERNIGHT. Stitch SMART+OVERNIGHT.
        if bool(use_rth) or sec_type != "STK" or not self._is_intraday_bar_size(str(bar_size)):
            return await self._request_historical_data(
                contract,
                duration_str=duration_str,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )

        smart_contract = copy.copy(contract)
        smart_exchange = str(getattr(smart_contract, "exchange", "") or "").strip().upper()
        if not smart_exchange or smart_exchange == "OVERNIGHT":
            smart_contract.exchange = "SMART"

        smart = await self._request_historical_data(
            smart_contract,
            duration_str=duration_str,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=False,
        )
        smart_diag = self.last_historical_request(smart_contract)
        overnight_contract = copy.copy(contract)
        overnight_contract.exchange = "OVERNIGHT"
        overnight = await self._request_historical_data(
            overnight_contract,
            duration_str=duration_str,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=False,
        )
        overnight_diag = self.last_historical_request(overnight_contract)
        merged = self._merge_full24_raw_bars(smart=smart, overnight=overnight)
        quality = self._full24_stitch_quality(bars=merged)
        if not bool(quality.get("complete", True)):
            smart_status = (
                str(smart_diag.get("status", "")).strip().lower()
                if isinstance(smart_diag, dict)
                else ""
            )
            overnight_status = (
                str(overnight_diag.get("status", "")).strip().lower()
                if isinstance(overnight_diag, dict)
                else ""
            )
            status = "timeout" if "timeout" in {smart_status, overnight_status} else "incomplete"
            con_id = 0
            try:
                con_id = int(getattr(contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
            payload: dict[str, object] = {
                "status": str(status),
                "ts": _now_et().isoformat(),
                "timeout_sec": float(self._historical_timeout_sec(duration_str)),
                "request": {
                    "duration_str": str(duration_str),
                    "bar_size": str(bar_size),
                    "what_to_show": str(what_to_show),
                    "use_rth": bool(use_rth),
                    "use_proxy": True,
                },
                "contract": {
                    "con_id": int(con_id),
                    "sec_type": str(sec_type),
                    "symbol": str(getattr(contract, "symbol", "") or "").strip().upper(),
                    "exchange": str(getattr(contract, "exchange", "") or "").strip().upper(),
                    "primary_exchange": str(getattr(contract, "primaryExchange", "") or "").strip().upper(),
                    "currency": str(getattr(contract, "currency", "") or "").strip().upper(),
                },
                "bars_count": int(len(merged)),
                "detail": "historical full24 stitch incomplete; missing expected sessions",
                "stream_quality": quality,
                "stream_legs": {
                    "smart_rows": int(len(smart or [])),
                    "overnight_rows": int(len(overnight or [])),
                    "smart_status": smart_status or None,
                    "overnight_status": overnight_status or None,
                },
            }
            self._store_historical_request_payload(payload)
            return []
        return merged

    @staticmethod
    def _ib_bar_datetime(value) -> datetime | None:
        dt = value
        if isinstance(dt, str):
            dt = util.parseIBDatetime(dt)
        if dt is None:
            return None
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, dtime(0, 0))
        return _to_et_shared(dt, naive_ts_mode=NaiveTsMode.ET).replace(tzinfo=None)

    async def session_close_anchors(
        self,
        contract: Contract,
        *,
        cache_ttl_sec: float = 900.0,
    ) -> tuple[float | None, float | None, float | None]:
        """Return (prev_close, close_1_session_ago, close_3_sessions_ago).

        Uses daily bars first, then falls back to intraday TRADES/MIDPOINT anchoring
        for sparse option/FOP history, with a small in-memory TTL cache to
        avoid pacing issues.
        """
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id:
            return None, None, None
        cached = self._session_close_cache.get(con_id)
        if cached:
            prev_close, close_1ago, close_3ago, cached_at = cached
            ttl = 30.0 if close_3ago is None else cache_ttl_sec
            if time.monotonic() - cached_at < ttl:
                return prev_close, close_1ago, close_3ago
        use_proxy = contract.secType in ("STK", "OPT")
        lock = self._historical_proxy_lock if use_proxy else self._historical_lock
        async with lock:
            cached = self._session_close_cache.get(con_id)
            if cached:
                prev_close, close_1ago, close_3ago, cached_at = cached
                ttl = 30.0 if close_3ago is None else cache_ttl_sec
                if time.monotonic() - cached_at < ttl:
                    return prev_close, close_1ago, close_3ago
            duration_str = "1 M" if sec_type in ("OPT", "FOP") else "2 W"
            use_rth = sec_type not in ("OPT", "FOP")

            def _extract_closes(raw_bars: list | None) -> list[float]:
                closes: list[float] = []
                for bar in raw_bars or []:
                    try:
                        value = float(getattr(bar, "close", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        closes.append(value)
                return closes

            bars = await self._request_historical_data(
                contract,
                duration_str=duration_str,
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=use_rth,
            )
            closes = _extract_closes(bars)
            if sec_type in ("OPT", "FOP") and len(closes) < 4:
                midpoint_bars = await self._request_historical_data(
                    contract,
                    duration_str=duration_str,
                    bar_size="1 day",
                    what_to_show="MIDPOINT",
                    use_rth=use_rth,
                )
                midpoint_closes = _extract_closes(midpoint_bars)
                if len(midpoint_closes) > len(closes):
                    closes = midpoint_closes
            prev_close = closes[-1] if closes else None
            close_1ago = closes[-2] if len(closes) >= 2 else None
            close_3ago = closes[-4] if len(closes) >= 4 else None
            if close_3ago is None and sec_type in ("OPT", "FOP"):
                intraday: list[tuple[datetime, float]] = []
                best_intraday: list[tuple[datetime, float]] = []
                best_days = 0
                best_last: datetime | None = None
                for what in ("TRADES", "MIDPOINT"):
                    bars_1h = await self._request_historical_data(
                        contract,
                        duration_str="10 D",
                        bar_size="1 hour",
                        what_to_show=what,
                        use_rth=False,
                    )
                    current: list[tuple[datetime, float]] = []
                    for bar in bars_1h or []:
                        dt = self._ib_bar_datetime(getattr(bar, "date", None))
                        if dt is None:
                            continue
                        try:
                            value = float(getattr(bar, "close", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            continue
                        if value > 0:
                            current.append((dt, value))
                    if not current:
                        continue
                    current.sort(key=lambda entry: entry[0])
                    day_count = len({entry[0].date() for entry in current})
                    last_dt = current[-1][0]
                    if (
                        day_count > best_days
                        or (day_count == best_days and (best_last is None or last_dt > best_last))
                    ):
                        best_intraday = current
                        best_days = day_count
                        best_last = last_dt
                intraday = best_intraday
                if intraday:
                    by_day: dict[date, float] = {}
                    for dt, value in sorted(intraday, key=lambda entry: entry[0]):
                        by_day[dt.date()] = value
                    days = sorted(by_day.keys())
                    values = [by_day[key] for key in days]
                    ref_idx = len(days) - 1
                    today_et = _now_et().date()
                    for idx in range(len(days) - 1, -1, -1):
                        if days[idx] < today_et:
                            ref_idx = idx
                            break
                    if prev_close is None and values:
                        prev_close = values[ref_idx]
                    if close_1ago is None:
                        prev1_idx = ref_idx - 1
                        if prev1_idx >= 0:
                            close_1ago = values[prev1_idx]
                    target_idx = ref_idx - 3
                    if target_idx >= 0:
                        close_3ago = values[target_idx]
                    else:
                        # Keep 72h truthful: only use an observation at/before T-72h.
                        target = _now_et_naive() - timedelta(hours=72)
                        candidates = [entry for entry in intraday if entry[0] <= target]
                        close_3ago = candidates[-1][1] if candidates else None
            self._session_close_cache[con_id] = (
                prev_close,
                close_1ago,
                close_3ago,
                time.monotonic(),
            )
            return prev_close, close_1ago, close_3ago

    async def session_closes(
        self,
        contract: Contract,
        *,
        cache_ttl_sec: float = 900.0,
    ) -> tuple[float | None, float | None]:
        """Return (prev_close, close_3_sessions_ago) for the given contract."""
        prev_close, _close_1ago, close_3ago = await self.session_close_anchors(
            contract,
            cache_ttl_sec=cache_ttl_sec,
        )
        return prev_close, close_3ago

    async def historical_bars(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str = "TRADES",
        cache_ttl_sec: float = 30.0,
    ) -> list[tuple[datetime, float]]:
        """Return [(bar_ts, close), ...] for the given contract.

        Uses a small in-memory TTL cache to avoid pacing issues when the bot is running.
        """
        con_id = int(getattr(contract, "conId", 0) or 0)
        sec_type = str(getattr(contract, "secType", "") or "")
        symbol = str(getattr(contract, "symbol", "") or "")
        key = (
            symbol,
            con_id,
            sec_type,
            str(bar_size),
            bool(use_rth),
            str(what_to_show),
            str(duration_str),
        )
        requested_ttl = max(0.0, float(cache_ttl_sec))
        empty_ttl_sec = 1.0

        def _cached_bars(
            cached_entry: tuple[list[tuple[datetime, float]], float] | tuple[list[tuple[datetime, float]], float, float] | None,
        ) -> list[tuple[datetime, float]] | None:
            if cached_entry is None:
                return None
            if len(cached_entry) >= 3:
                bars, cached_at, cached_ttl = cached_entry[0], cached_entry[1], cached_entry[2]
            else:
                bars, cached_at = cached_entry
                cached_ttl = requested_ttl
            ttl = max(0.0, float(cached_ttl))
            if time.monotonic() - float(cached_at) < ttl:
                return list(bars)
            return None

        cached_bars = _cached_bars(self._historical_bar_cache.get(key))
        if cached_bars is not None:
            return cached_bars

        use_proxy = sec_type in ("STK", "OPT")
        lock = self._historical_proxy_lock if use_proxy else self._historical_lock
        async with lock:
            cached_bars = _cached_bars(self._historical_bar_cache.get(key))
            if cached_bars is not None:
                return cached_bars
            raw = await self._request_historical_data_for_stream(
                contract,
                duration_str=str(duration_str),
                bar_size=str(bar_size),
                what_to_show=str(what_to_show),
                use_rth=bool(use_rth),
            )

            bars: list[tuple[datetime, float]] = []
            for bar in raw or []:
                dt = self._ib_bar_datetime(getattr(bar, "date", None))
                if dt is None:
                    continue
                try:
                    close = float(getattr(bar, "close", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if close <= 0:
                    continue
                bars.append((dt, close))

            cache_ttl = min(requested_ttl, empty_ttl_sec) if not bars else requested_ttl
            self._historical_bar_cache[key] = (bars, time.monotonic(), cache_ttl)
            return list(bars)

    async def historical_bars_ohlcv(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str = "TRADES",
        cache_ttl_sec: float = 30.0,
    ) -> list[OhlcvBar]:
        """Return OHLCV bars for the given contract.

        Uses a small in-memory TTL cache to avoid pacing issues when the bot is running.
        """
        con_id = int(getattr(contract, "conId", 0) or 0)
        sec_type = str(getattr(contract, "secType", "") or "")
        symbol = str(getattr(contract, "symbol", "") or "")
        key = (
            symbol,
            con_id,
            sec_type,
            str(bar_size),
            bool(use_rth),
            str(what_to_show),
            str(duration_str),
        )
        requested_ttl = max(0.0, float(cache_ttl_sec))
        empty_ttl_sec = 1.0

        def _cached_bars(
            cached_entry: tuple[list[OhlcvBar], float] | tuple[list[OhlcvBar], float, float] | None,
        ) -> list[OhlcvBar] | None:
            if cached_entry is None:
                return None
            if len(cached_entry) >= 3:
                bars, cached_at, cached_ttl = cached_entry[0], cached_entry[1], cached_entry[2]
            else:
                bars, cached_at = cached_entry
                cached_ttl = requested_ttl
            ttl = max(0.0, float(cached_ttl))
            if time.monotonic() - float(cached_at) < ttl:
                return list(bars)
            return None

        cached_bars = _cached_bars(self._historical_bar_ohlcv_cache.get(key))
        if cached_bars is not None:
            return cached_bars

        use_proxy = sec_type in ("STK", "OPT")
        lock = self._historical_proxy_lock if use_proxy else self._historical_lock
        async with lock:
            cached_bars = _cached_bars(self._historical_bar_ohlcv_cache.get(key))
            if cached_bars is not None:
                return cached_bars
            raw = await self._request_historical_data_for_stream(
                contract,
                duration_str=str(duration_str),
                bar_size=str(bar_size),
                what_to_show=str(what_to_show),
                use_rth=bool(use_rth),
            )

            bars: list[OhlcvBar] = []
            for bar in raw or []:
                dt = self._ib_bar_datetime(getattr(bar, "date", None))
                if dt is None:
                    continue
                try:
                    open_p = float(getattr(bar, "open", 0.0) or 0.0)
                    high = float(getattr(bar, "high", 0.0) or 0.0)
                    low = float(getattr(bar, "low", 0.0) or 0.0)
                    close = float(getattr(bar, "close", 0.0) or 0.0)
                    volume = float(getattr(bar, "volume", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if close <= 0:
                    continue
                bars.append(
                    OhlcvBar(
                        ts=dt,
                        open=open_p,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                    )
                )

            cache_ttl = min(requested_ttl, empty_ttl_sec) if not bars else requested_ttl
            self._historical_bar_ohlcv_cache[key] = (bars, time.monotonic(), cache_ttl)
            return list(bars)

    @staticmethod
    def _nearest_expiry(raw_expirations: Iterable[object]) -> str | None:
        today = _now_et().date()
        options: list[tuple[date, str]] = []
        for raw in raw_expirations or ():
            text = str(raw or "").strip()
            if len(text) < 8 or not text[:8].isdigit():
                continue
            try:
                parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
            except ValueError:
                continue
            options.append((parsed, text[:8]))
        if not options:
            return None
        options.sort(key=lambda pair: pair[0])
        for parsed, text in options:
            if parsed >= today:
                return text
        return options[0][1]

    @staticmethod
    def _median_strike(raw_strikes: Iterable[object]) -> float | None:
        values: list[float] = []
        for raw in raw_strikes or ():
            try:
                strike = float(raw)
            except (TypeError, ValueError):
                continue
            if strike > 0:
                values.append(strike)
        if not values:
            return None
        values.sort()
        return values[len(values) // 2]

    def _search_reference_price_from_ticker(self, ticker: object) -> float | None:
        if ticker is None:
            return None
        bid = self._quote_num(getattr(ticker, "bid", None))
        ask = self._quote_num(getattr(ticker, "ask", None))
        last = self._quote_num(getattr(ticker, "last", None))
        close = self._quote_num(getattr(ticker, "close", None))
        if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
            return (float(bid) + float(ask)) / 2.0
        if last is not None and last > 0:
            return float(last)
        if close is not None and close > 0:
            return float(close)
        return None

    def _search_actionable_reference_price_from_ticker(self, ticker: object) -> float | None:
        """Return a near-live search reference price from top-of-book/last fields."""
        if ticker is None:
            return None
        bid = self._quote_num(getattr(ticker, "bid", None))
        ask = self._quote_num(getattr(ticker, "ask", None))
        last = self._quote_num(getattr(ticker, "last", None))
        if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
            return (float(bid) + float(ask)) / 2.0
        if last is not None and last > 0:
            return float(last)
        return None

    @staticmethod
    def _search_terms(query: str, *, mode: str | None = None) -> list[str]:
        cleaned = str(query or "").strip().upper()
        if not cleaned:
            return []
        parts = [p for p in re.split(r"[^A-Z0-9]+", cleaned) if p]
        terms: list[str] = []
        mode_clean = str(mode or "").strip().upper()
        aliases_by_seed = _SEARCH_TERM_ALIASES_BY_MODE.get(mode_clean, {})

        def _add(term: str) -> None:
            value = str(term or "").strip().upper()
            if not value or value in terms:
                return
            terms.append(value)

        _add(cleaned)
        for part in parts:
            _add(part)
        if len(parts) > 1:
            # Preserve compact symbol-like variants for inputs such as "1 oz".
            _add("".join(parts))
        # Alias expansion for both full query and individual words.
        for seed in tuple(terms):
            for alias in aliases_by_seed.get(seed, ()):
                _add(alias)
        return terms[:10]

    @staticmethod
    def _desc_text(desc: object) -> str:
        chunks = [
            str(getattr(desc, "description", "") or ""),
            str(getattr(desc, "longName", "") or ""),
            str(getattr(desc, "companyName", "") or ""),
        ]
        contract = getattr(desc, "contract", None)
        if contract is not None:
            chunks.extend(
                [
                    str(getattr(contract, "localSymbol", "") or ""),
                    str(getattr(contract, "tradingClass", "") or ""),
                ]
            )
        return " ".join(chunks).upper()

    @staticmethod
    def _desc_label(desc: object) -> str:
        for attr in ("description", "longName", "companyName"):
            text = str(getattr(desc, attr, "") or "").strip()
            if text:
                return text
        contract = getattr(desc, "contract", None)
        if contract is None:
            return ""
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        return symbol

    async def search_contract_labels(
        self,
        query: str,
        *,
        mode: str = "STK",
        symbols: Iterable[str] | None = None,
    ) -> dict[str, str]:
        token = str(query or "").strip().upper()
        if not token:
            return {}
        mode_clean = str(mode or "STK").strip().upper()
        if mode_clean not in ("STK", "FUT", "OPT", "FOP"):
            mode_clean = "STK"
        target_symbols = {
            str(symbol or "").strip().upper()
            for symbol in (symbols or ())
            if str(symbol or "").strip()
        }
        labels: dict[str, str] = {}
        matches = await self._matching_symbols(
            token,
            use_proxy=mode_clean in ("STK", "OPT"),
            mode=mode_clean,
            raise_on_error=False,
        )
        for desc in matches:
            contract = getattr(desc, "contract", None)
            if contract is None:
                continue
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            if not symbol:
                continue
            if target_symbols and symbol not in target_symbols:
                continue
            if symbol in labels:
                continue
            label = str(self._desc_label(desc) or "").strip()
            if not label:
                continue
            if label.upper() == symbol:
                continue
            labels[symbol] = label
        for symbol in target_symbols:
            if symbol in labels:
                continue
            hint = str(_CONTRACT_LABEL_HINTS.get(symbol, "") or "").strip()
            if not hint:
                continue
            labels[symbol] = hint
        return labels

    @staticmethod
    def _future_exchange_candidates(symbol: str, preferred: Iterable[str]) -> list[str]:
        sym = str(symbol or "").strip().upper()
        out: list[str] = []

        def _add(exchange: str) -> None:
            value = str(exchange or "").strip().upper()
            if not value:
                return
            if value in ("SMART", "IDEALPRO"):
                return
            if value in out:
                return
            out.append(value)

        for exchange in preferred or ():
            _add(exchange)
        for exchange in _FUT_EXCHANGE_HINTS.get(sym, ()):
            _add(exchange)
        for exchange in ("COMEX", "NYMEX", "GLOBEX", "CME", "CBOT", "ECBOT"):
            _add(exchange)
        if not out:
            out = ["CME"]
        return out

    @staticmethod
    def _is_retryable_matching_symbols_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)):
            return True
        text = str(exc or "").strip().lower()
        if not text:
            return False
        return any(
            needle in text
            for needle in (
                "timeout",
                "timed out",
                "already pending",
                "failed to request matching symbols",
                "pacing",
                "connect",
                "socket",
                "temporarily",
            )
        )

    async def _matching_symbols(
        self,
        query: str,
        *,
        use_proxy: bool,
        mode: str | None = None,
        raise_on_error: bool = False,
    ) -> list[object]:
        terms = self._search_terms(query, mode=mode)
        if not terms:
            return []
        search_terms = terms[:6]
        lock = self._proxy_lock if use_proxy else self._lock
        async with lock:
            try:
                if use_proxy:
                    await self.connect_proxy()
                    ib = self._ib_proxy
                else:
                    await self.connect()
                    ib = self._ib
            except Exception as exc:
                if raise_on_error:
                    raise RuntimeError(f"IBKR symbol lookup connect failed: {exc}") from exc
                return []
            out: list[object] = []
            seen: set[tuple[int, str, str, str, str]] = set()
            had_successful_term = False
            term_errors: list[str] = []
            for term in search_terms:
                matches = []
                last_exc: Exception | None = None
                for attempt in range(_MATCHING_SYMBOL_MAX_ATTEMPTS):
                    timeout_sec = (
                        _MATCHING_SYMBOL_TIMEOUT_INITIAL_SEC
                        if attempt == 0
                        else _MATCHING_SYMBOL_TIMEOUT_RETRY_SEC
                    )
                    try:
                        started = time.monotonic()
                        matches = await asyncio.wait_for(
                            ib.reqMatchingSymbolsAsync(term),
                            timeout=float(timeout_sec),
                        )
                        elapsed = time.monotonic() - started
                        # ib_insync can occasionally return an empty list right around the
                        # timeout boundary for delayed/canceled requests; treat that as retryable.
                        if not matches:
                            timeout_threshold = max(
                                float(timeout_sec) * float(_MATCHING_SYMBOL_EMPTY_TIMEOUT_RATIO),
                                float(timeout_sec) - min(0.05, float(timeout_sec) * 0.1),
                            )
                            if elapsed >= timeout_threshold:
                                raise asyncio.TimeoutError(
                                    f"matching symbols request timed out after {float(timeout_sec):.3f}s"
                                )
                        had_successful_term = True
                        last_exc = None
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        last_exc = exc
                        retryable = self._is_retryable_matching_symbols_error(exc)
                        if not retryable or (attempt + 1) >= _MATCHING_SYMBOL_MAX_ATTEMPTS:
                            break
                        await asyncio.sleep(_MATCHING_SYMBOL_RETRY_BASE_SEC * float(attempt + 1))
                if last_exc is not None:
                    detail = str(last_exc or "").strip() or type(last_exc).__name__
                    term_errors.append(f"{term}: {detail}")
                    continue
                for desc in list(matches or []):
                    contract = getattr(desc, "contract", None)
                    if not contract:
                        continue
                    key = (
                        int(getattr(contract, "conId", 0) or 0),
                        str(getattr(contract, "secType", "") or "").strip().upper(),
                        str(getattr(contract, "symbol", "") or "").strip().upper(),
                        str(getattr(contract, "exchange", "") or "").strip().upper(),
                        str(getattr(contract, "currency", "") or "").strip().upper(),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(desc)
                if len(out) >= 160:
                    break
            if not out and raise_on_error and term_errors and not had_successful_term:
                raise RuntimeError(
                    "IBKR symbol lookup unavailable: "
                    + "; ".join(term_errors[:2])
                )
        return out

    @staticmethod
    def _rank_opt_underlyer_symbols(
        matches: Iterable[object],
        *,
        term_set: set[str],
        token: str,
        token_is_alias_seed: bool,
        max_symbols: int,
    ) -> list[str]:
        exact_symbols: list[str] = []
        prefix_symbols: list[str] = []
        contains_symbols: list[str] = []
        desc_symbols: list[str] = []
        seen_symbols: set[str] = set()
        for desc in matches:
            contract = getattr(desc, "contract", None)
            if not contract:
                continue
            if str(getattr(contract, "secType", "") or "").strip().upper() != "STK":
                continue
            deriv = {
                str(sec).strip().upper()
                for sec in (getattr(desc, "derivativeSecTypes", None) or [])
            }
            if "OPT" not in deriv:
                continue
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            if not symbol or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            desc_text = IBKRClient._desc_text(desc)
            if symbol in term_set and not (token_is_alias_seed and symbol == token):
                exact_symbols.append(symbol)
            elif any(symbol.startswith(term) for term in term_set):
                prefix_symbols.append(symbol)
            elif any(term in symbol for term in term_set):
                contains_symbols.append(symbol)
            elif any(term in desc_text for term in term_set):
                desc_symbols.append(symbol)
            if len(seen_symbols) >= max_symbols:
                break
        return [*exact_symbols, *prefix_symbols, *contains_symbols, *desc_symbols]

    async def _opt_underlyer_fallback_from_symbol(self, symbol: str) -> tuple[str, str] | None:
        candidate = str(symbol or "").strip().upper()
        if not candidate:
            return None
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", candidate):
            return None
        try:
            resolved = await self.stock_option_chain(candidate)
        except Exception:
            return None
        if not resolved:
            return None
        underlying, _chain = resolved
        normalized = str(getattr(underlying, "symbol", "") or "").strip().upper() or candidate
        return normalized, ""

    async def search_option_underlyers(
        self,
        query: str,
        *,
        limit: int = 8,
        timing: dict[str, object] | None = None,
    ) -> list[tuple[str, str]]:
        started_mono = time.monotonic()
        token = str(query or "").strip().upper()
        if timing is not None:
            timing.clear()
            timing.update(
                {
                    "query": token,
                    "limit": int(limit or 8),
                    "mode": "OPT",
                    "source": "",
                }
            )

        def _finish(
            rows: list[tuple[str, str]],
            *,
            source: str = "",
            error: str = "",
        ) -> list[tuple[str, str]]:
            if timing is not None:
                if source:
                    timing["source"] = str(source)
                if error:
                    timing["error"] = str(error)
                timing["result_count"] = len(rows)
                timing["total_ms"] = (time.monotonic() - started_mono) * 1000.0
            return rows

        if not token:
            return _finish([], source="empty-query")
        if len(token) >= 3 and re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", token):
            direct_started = time.monotonic()
            direct = await self._opt_underlyer_fallback_from_symbol(token)
            if timing is not None:
                timing["direct_ms"] = (time.monotonic() - direct_started) * 1000.0
            if direct is not None:
                return _finish([direct], source="direct")
        terms = self._search_terms(token, mode="OPT")
        if not terms:
            return _finish([], source="no-terms")
        term_set = set(terms)
        mode_aliases = _SEARCH_TERM_ALIASES_BY_MODE.get("OPT", {})
        token_is_alias_seed = token in mode_aliases
        max_rows = max(1, min(int(limit or 8), 20))
        matches: list[object] = []
        match_error: Exception | None = None
        matching_started = time.monotonic()
        try:
            matches = await self._matching_symbols(
                token,
                use_proxy=True,
                mode="OPT",
                raise_on_error=True,
            )
        except Exception as exc:
            match_error = exc
        if timing is not None:
            timing["matching_ms"] = (time.monotonic() - matching_started) * 1000.0
            timing["matching_rows"] = len(matches)
        if not matches:
            fallback_candidates: list[str] = [token, *terms]
            seen_candidates: set[str] = set()
            fallback_calls = 0
            fallback_ms_total = 0.0
            for candidate in fallback_candidates:
                normalized = str(candidate or "").strip().upper()
                if not normalized or normalized in seen_candidates:
                    continue
                seen_candidates.add(normalized)
                fallback_calls += 1
                fallback_started = time.monotonic()
                fallback = await self._opt_underlyer_fallback_from_symbol(normalized)
                fallback_ms_total += (time.monotonic() - fallback_started) * 1000.0
                if fallback is not None:
                    if timing is not None:
                        timing["fallback_chain_calls"] = fallback_calls
                        timing["fallback_chain_ms"] = fallback_ms_total
                    return _finish([fallback], source="fallback")
            if timing is not None:
                timing["fallback_chain_calls"] = fallback_calls
                timing["fallback_chain_ms"] = fallback_ms_total
            if match_error is not None:
                if timing is not None:
                    _finish([], source="matching-error", error=str(match_error))
                raise match_error
        if not matches:
            return _finish([], source="no-matches")
        rank_started = time.monotonic()
        ranked_symbols = self._rank_opt_underlyer_symbols(
            matches,
            term_set=term_set,
            token=token,
            token_is_alias_seed=token_is_alias_seed,
            max_symbols=max_rows * 6,
        )
        if timing is not None:
            timing["rank_ms"] = (time.monotonic() - rank_started) * 1000.0
            timing["ranked_symbol_count"] = len(ranked_symbols)
        if not ranked_symbols:
            return _finish([], source="rank-empty")
        label_by_symbol: dict[str, str] = {}
        for desc in matches:
            contract = getattr(desc, "contract", None)
            if not contract:
                continue
            if str(getattr(contract, "secType", "") or "").strip().upper() != "STK":
                continue
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            if not symbol or symbol in label_by_symbol:
                continue
            label_by_symbol[symbol] = self._desc_label(desc)
        out: list[tuple[str, str]] = []
        seen_symbols: set[str] = set()
        for symbol in ranked_symbols:
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            out.append((symbol, label_by_symbol.get(symbol, "")))
            if len(out) >= max_rows:
                break
        return _finish(out, source="matching")

    async def search_contracts(
        self,
        query: str,
        *,
        mode: str = "STK",
        limit: int = 5,
        opt_underlyer_symbol: str | None = None,
        timing: dict[str, object] | None = None,
        opt_first_limit: int | None = None,
        opt_progress: Callable[[list[Contract], dict[str, object]], object] | None = None,
        expiry_offset: int = 0,
    ) -> list[Contract]:
        started_mono = time.monotonic()
        token = str(query or "").strip().upper()
        try:
            expiry_offset_clean = max(0, int(expiry_offset or 0))
        except (TypeError, ValueError):
            expiry_offset_clean = 0
        if timing is not None:
            timing.clear()
            timing.update(
                {
                    "query": token,
                    "mode": str(mode or "").strip().upper(),
                    "limit": int(limit or 5),
                    "source": "",
                    "candidate_count": 0,
                    "qualified_count": 0,
                    "expiry_offset": int(expiry_offset_clean),
                }
            )
        if not token:
            return []
        mode_clean = str(mode or "STK").strip().upper()
        if mode_clean not in ("STK", "FUT", "OPT", "FOP"):
            mode_clean = "STK"
        terms = self._search_terms(token, mode=mode_clean)
        if not terms:
            return []
        term_set = set(terms)
        mode_aliases = _SEARCH_TERM_ALIASES_BY_MODE.get(mode_clean, {})
        token_is_alias_seed = token in mode_aliases
        max_cap = 160 if mode_clean in ("OPT", "FOP") else 20
        max_rows = max(1, min(int(limit or 5), max_cap))
        opt_symbol_override = ""
        if mode_clean == "OPT":
            opt_symbol_override = str(opt_underlyer_symbol or "").strip().upper()
        if timing is not None:
            timing["mode"] = mode_clean
            timing["limit"] = int(max_rows)
            timing["opt_symbol_override"] = bool(mode_clean == "OPT" and opt_symbol_override)

        def _opt_timing_finish(
            rows: list[Contract],
            *,
            stage: str,
            reason: str,
        ) -> list[Contract]:
            if timing is not None and mode_clean == "OPT":
                timing["source"] = "search_contracts_opt"
                timing["stage"] = str(stage or "")
                timing["reason"] = str(reason or "")
                timing["result_count"] = len(rows)
                timing["total_ms"] = (time.monotonic() - started_mono) * 1000.0
            return rows

        matches: list[object] = []
        if not (mode_clean == "OPT" and opt_symbol_override):
            lookup_terms = [token]
            if mode_clean in ("FUT", "FOP"):
                for term in terms:
                    normalized = str(term or "").strip().upper()
                    if not normalized or normalized in lookup_terms:
                        continue
                    lookup_terms.append(normalized)
            matching_started = time.monotonic()
            if timing is not None and mode_clean == "OPT":
                timing["lookup_terms"] = list(lookup_terms)
            for lookup in lookup_terms:
                matches = await self._matching_symbols(
                    lookup,
                    use_proxy=mode_clean in ("STK", "OPT"),
                    mode=mode_clean,
                    raise_on_error=True,
                )
                if matches or mode_clean not in ("FUT", "FOP"):
                    break
            if timing is not None and mode_clean == "OPT":
                timing["matching_ms"] = (time.monotonic() - matching_started) * 1000.0
                timing["matching_rows"] = len(matches)
            if not matches and mode_clean not in ("FUT", "FOP"):
                if mode_clean == "OPT":
                    return _opt_timing_finish([], stage="matching", reason="no-matches")
                return []

        if mode_clean == "STK":
            exact_symbols: list[str] = []
            prefix_symbols: list[str] = []
            contains_symbols: list[str] = []
            desc_symbols: list[str] = []
            seen_symbols: set[str] = set()
            for desc in matches:
                contract = getattr(desc, "contract", None)
                if not contract:
                    continue
                if str(getattr(contract, "secType", "") or "").strip().upper() != "STK":
                    continue
                symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
                if not symbol or symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)
                desc_text = self._desc_text(desc)
                if symbol in term_set and not (token_is_alias_seed and symbol == token):
                    exact_symbols.append(symbol)
                elif any(symbol.startswith(term) for term in term_set):
                    prefix_symbols.append(symbol)
                elif any(term in symbol for term in term_set):
                    contains_symbols.append(symbol)
                elif any(term in desc_text for term in term_set):
                    desc_symbols.append(symbol)
                if len(seen_symbols) >= max_rows * 6:
                    break
            symbols = [*exact_symbols, *prefix_symbols, *contains_symbols, *desc_symbols]
            if not symbols:
                return []
            if exact_symbols:
                symbols = exact_symbols
            candidates = [Stock(symbol=symbol, exchange="SMART", currency="USD") for symbol in symbols]
            qualified = await self.qualify_proxy_contracts(*candidates)
            by_symbol = {
                str(getattr(contract, "symbol", "") or "").strip().upper(): contract
                for contract in qualified
                if contract
            }
            out: list[Contract] = []
            for candidate in candidates:
                symbol = str(getattr(candidate, "symbol", "") or "").strip().upper()
                resolved = by_symbol.get(symbol)
                if resolved is None:
                    continue
                out.append(resolved)
                if len(out) >= max_rows:
                    break
            return out

        if mode_clean == "FUT":
            exact_roots: list[tuple[str, str]] = []
            prefix_roots: list[tuple[str, str]] = []
            contains_roots: list[tuple[str, str]] = []
            desc_roots: list[tuple[str, str]] = []
            seen_roots: set[tuple[str, str]] = set()
            for desc in matches:
                contract = getattr(desc, "contract", None)
                if not contract:
                    continue
                if str(getattr(contract, "secType", "") or "").strip().upper() != "FUT":
                    continue
                symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
                if not symbol:
                    continue
                exchange = str(getattr(contract, "exchange", "") or "").strip().upper() or "CME"
                key = (symbol, exchange)
                if key in seen_roots:
                    continue
                seen_roots.add(key)
                desc_text = self._desc_text(desc)
                if symbol in term_set and not (token_is_alias_seed and symbol == token):
                    exact_roots.append(key)
                elif any(symbol.startswith(term) for term in term_set):
                    prefix_roots.append(key)
                elif any(term in symbol for term in term_set):
                    contains_roots.append(key)
                elif any(term in desc_text for term in term_set):
                    desc_roots.append(key)
                if len(seen_roots) >= max_rows * 6:
                    break
            ranked_roots = [*exact_roots, *prefix_roots, *contains_roots, *desc_roots]
            symbol_exchange_map: dict[str, list[str]] = {}
            for symbol, exchange in ranked_roots:
                symbol_exchange_map.setdefault(symbol, [])
                if exchange not in symbol_exchange_map[symbol]:
                    symbol_exchange_map[symbol].append(exchange)

            if exact_roots:
                candidate_symbols: list[str] = []
                for symbol, _exchange in exact_roots:
                    if symbol not in candidate_symbols:
                        candidate_symbols.append(symbol)
            else:
                candidate_symbols = []
                for symbol, _exchange in ranked_roots:
                    if symbol not in candidate_symbols:
                        candidate_symbols.append(symbol)

            # Hard fallback: derive plausible future roots directly from expanded terms.
            if not candidate_symbols:
                for term in terms:
                    if not term:
                        continue
                    if len(term) > 6:
                        continue
                    if not term.isalnum():
                        continue
                    if term not in candidate_symbols:
                        candidate_symbols.append(term)
                    if len(candidate_symbols) >= (max_rows * 4):
                        break

            if not candidate_symbols:
                return []
            out: list[Contract] = []
            seen_con_ids: set[int] = set()
            for symbol in candidate_symbols:
                preferred = symbol_exchange_map.get(symbol, [])
                future = None
                for exchange in self._future_exchange_candidates(symbol, preferred):
                    future = await self.front_future(symbol, exchange=exchange, cache_ttl_sec=1800.0)
                    if future is not None:
                        break
                if future is None:
                    continue
                con_id = int(getattr(future, "conId", 0) or 0)
                if con_id and con_id in seen_con_ids:
                    continue
                if con_id:
                    seen_con_ids.add(con_id)
                out.append(future)
                if len(out) >= max_rows:
                    break
            return out

        if mode_clean == "OPT":
            symbol = opt_symbol_override
            if not symbol:
                ranked_symbols = self._rank_opt_underlyer_symbols(
                    matches,
                    term_set=term_set,
                    token=token,
                    token_is_alias_seed=token_is_alias_seed,
                    max_symbols=max_rows * 6,
                )
                if not ranked_symbols:
                    return _opt_timing_finish([], stage="rank", reason="no-ranked-symbols")
                symbol = ranked_symbols[0]
            if timing is not None:
                timing["symbol"] = str(symbol)

            chain_started = time.monotonic()
            chain_entry = await self.stock_option_chain(symbol)
            if timing is not None:
                timing["chain_ms"] = (time.monotonic() - chain_started) * 1000.0
            if not chain_entry:
                return _opt_timing_finish([], stage="chain", reason="chain-missing")
            underlying, chain = chain_entry
            expiries_raw = getattr(chain, "expirations", ()) or ()
            expiries: list[str] = []
            for raw in expiries_raw:
                text = str(raw or "").strip()
                if len(text) >= 8 and text[:8].isdigit():
                    expiries.append(text[:8])
            expiries = sorted(set(expiries))
            if timing is not None:
                timing["expiry_count"] = len(expiries)
            if not expiries:
                return _opt_timing_finish([], stage="chain", reason="no-expiries")
            # Favor surfacing multiple expiries while keeping each expiry
            # deep enough for a usable C/P ladder near ATM.
            pair_budget = max(1, int(max_rows) // 2)
            min_strikes_per_expiry = 6
            max_expiries = max(1, pair_budget // min_strikes_per_expiry)
            expiry_page_size = max(1, int(max_expiries))
            expiry_start = min(int(expiry_offset_clean), len(expiries))
            expiry_end = min(len(expiries), int(expiry_start) + int(expiry_page_size))
            selected_expiries = expiries[expiry_start:expiry_end]
            next_expiry_offset = int(expiry_end)
            has_more_expiries = bool(next_expiry_offset < len(expiries))
            if timing is not None:
                timing["has_more_expiries"] = bool(has_more_expiries)
                timing["next_expiry_offset"] = int(next_expiry_offset)
                timing["expiry_page_size"] = int(expiry_page_size)
            if not selected_expiries:
                return _opt_timing_finish([], stage="chain", reason="expiry-page-empty")
            strikes_raw = getattr(chain, "strikes", ()) or ()
            strikes: list[float] = []
            for raw in strikes_raw:
                try:
                    strike = float(raw)
                except (TypeError, ValueError):
                    continue
                if strike > 0:
                    strikes.append(strike)
            if not strikes:
                return _opt_timing_finish([], stage="chain", reason="no-strikes")
            strikes = sorted(set(strikes))
            if timing is not None:
                timing["strike_count"] = len(strikes)
            ref_price = None
            ref_source = ""
            ref_started = time.monotonic()
            try:
                ticker = await self.ensure_ticker(underlying, owner="search")
                ref_price = self._search_actionable_reference_price_from_ticker(ticker)
                if ref_price is None:
                    settle_wait_sec = 0.3
                    settle_poll_sec = 0.05
                    settle_deadline = time.monotonic() + settle_wait_sec
                    while time.monotonic() < settle_deadline:
                        await asyncio.sleep(settle_poll_sec)
                        ref_price = self._search_actionable_reference_price_from_ticker(ticker)
                        if ref_price is not None:
                            break
                if ref_price is None:
                    ref_price = self._search_reference_price_from_ticker(ticker)
                if ref_price is not None:
                    ref_source = "ticker"
                if ref_price is None:
                    for what_to_show in ("TRADES", "MIDPOINT"):
                        try:
                            bars = await asyncio.wait_for(
                                self.historical_bars(
                                    underlying,
                                    duration_str="10800 S",
                                    bar_size="1 min",
                                    use_rth=False,
                                    what_to_show=what_to_show,
                                    cache_ttl_sec=10.0,
                                ),
                                timeout=1.0,
                            )
                        except asyncio.TimeoutError:
                            bars = []
                        except Exception:
                            bars = []
                        if not bars:
                            continue
                        _ts, close = bars[-1]
                        close_num = self._quote_num(close)
                        if close_num is None:
                            continue
                        ref_price = float(close_num)
                        ref_source = f"historical-{what_to_show.lower()}"
                        break
            except Exception:
                ref_price = None
            finally:
                under_con_id = int(getattr(underlying, "conId", 0) or 0)
                if under_con_id:
                    self.release_ticker(under_con_id, owner="search")
            if timing is not None:
                timing["ref_price_ms"] = (time.monotonic() - ref_started) * 1000.0

            if ref_price is None:
                ref_price = self._median_strike(strikes)
                if ref_price is not None:
                    ref_source = "median"
            if ref_price is None:
                return _opt_timing_finish([], stage="reference", reason="no-reference-price")
            if timing is not None:
                timing["ref_price_source"] = ref_source or "unknown"
            rows_per_expiry = max(1, pair_budget // max(1, len(selected_expiries)))
            if timing is not None:
                timing["selected_expiry_count"] = len(selected_expiries)
                timing["rows_per_expiry"] = int(rows_per_expiry)

            symbol_clean = str(getattr(underlying, "symbol", "") or symbol).strip().upper()
            exchange = str(getattr(chain, "exchange", "") or "SMART").strip().upper() or "SMART"
            currency = str(getattr(underlying, "currency", "") or "USD").strip().upper() or "USD"
            multiplier = str(getattr(chain, "multiplier", "") or "")
            trading_class = str(getattr(chain, "tradingClass", "") or "")
            candidates: list[Contract] = []
            for expiry in selected_expiries:
                ordered_strikes = sorted(
                    strikes,
                    key=lambda value: (abs(float(value) - float(ref_price)), float(value)),
                )
                nearby = sorted(ordered_strikes[:rows_per_expiry])
                for strike in nearby:
                    for right in ("C", "P"):
                        candidates.append(
                            Contract(
                                secType="OPT",
                                symbol=symbol_clean,
                                exchange=exchange,
                                currency=currency,
                                lastTradeDateOrContractMonth=expiry,
                                strike=float(strike),
                                right=right,
                                multiplier=multiplier,
                                tradingClass=trading_class,
                            )
                        )
            if not candidates:
                return _opt_timing_finish([], stage="candidate-build", reason="no-candidates")
            if timing is not None:
                timing["candidate_count"] = len(candidates)

            def _opt_key(contract: Contract | None) -> tuple[str, str, str, float] | None:
                if contract is None:
                    return None
                try:
                    strike = float(getattr(contract, "strike", 0.0) or 0.0)
                except (TypeError, ValueError):
                    return None
                return (
                    str(getattr(contract, "symbol", "") or "").strip().upper(),
                    str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip(),
                    str(getattr(contract, "right", "") or "").strip().upper()[:1],
                    round(strike, 6),
                )

            def _ordered_resolved(
                source_candidates: list[Contract],
                resolved_by_key: dict[tuple[str, str, str, float], Contract],
            ) -> list[Contract]:
                out_rows: list[Contract] = []
                for candidate in source_candidates:
                    key = _opt_key(candidate)
                    if key is None:
                        continue
                    resolved = resolved_by_key.get(key)
                    if resolved is None:
                        continue
                    con_id = int(getattr(resolved, "conId", 0) or 0)
                    if con_id <= 0:
                        continue
                    out_rows.append(resolved)
                return out_rows

            split_limit = 0
            if opt_progress is not None:
                try:
                    split_limit = int(opt_first_limit or 0)
                except (TypeError, ValueError):
                    split_limit = 0
                split_limit = max(0, min(split_limit, len(candidates), max_rows))
            split_active = 0 < split_limit < len(candidates)
            if timing is not None:
                timing["first_limit"] = int(split_limit)
                timing["split_active"] = bool(split_active)

            qualified: list[Contract] = []
            qualify_ms_total = 0.0
            if split_active:
                first_candidates = candidates[:split_limit]
                rest_candidates = candidates[split_limit:]

                first_started = time.monotonic()
                first_qualified = await self.qualify_proxy_contracts(*first_candidates)
                first_ms = (time.monotonic() - first_started) * 1000.0
                qualify_ms_total += first_ms
                qualified.extend(first_qualified)

                first_by_key: dict[tuple[str, str, str, float], Contract] = {}
                for contract in first_qualified:
                    key = _opt_key(contract)
                    if key is not None:
                        first_by_key[key] = contract
                first_rows = _ordered_resolved(first_candidates, first_by_key)
                first_qualified_count = len(first_by_key)
                if timing is not None:
                    timing["qualify_ms_first"] = float(first_ms)
                    timing["first_candidate_count"] = len(first_candidates)
                    timing["first_qualified_count"] = int(first_qualified_count)
                    timing["first_result_count"] = len(first_rows)
                if opt_progress is not None:
                    progress_timing: dict[str, object] = {}
                    if timing is not None:
                        progress_timing.update(timing)
                    progress_timing.update(
                        {
                            "source": "search_contracts_opt",
                            "stage": "qualify-first",
                            "reason": "progress",
                            "candidate_count": len(first_candidates),
                            "qualified_count": int(first_qualified_count),
                            "qualify_ms": float(first_ms),
                            "result_count": len(first_rows),
                            "total_ms": (time.monotonic() - started_mono) * 1000.0,
                        }
                    )
                    try:
                        progress_result = opt_progress(first_rows[:max_rows], progress_timing)
                        if asyncio.iscoroutine(progress_result):
                            await progress_result
                    except Exception:
                        pass

                rest_started = time.monotonic()
                rest_qualified = await self.qualify_proxy_contracts(*rest_candidates)
                rest_ms = (time.monotonic() - rest_started) * 1000.0
                qualify_ms_total += rest_ms
                qualified.extend(rest_qualified)
                if timing is not None:
                    timing["qualify_ms_rest"] = float(rest_ms)
                    timing["rest_candidate_count"] = len(rest_candidates)
                    timing["rest_qualified_count"] = len(rest_qualified)
            else:
                qualify_started = time.monotonic()
                qualified = await self.qualify_proxy_contracts(*candidates)
                qualify_ms_total = (time.monotonic() - qualify_started) * 1000.0
            if timing is not None:
                timing["qualify_ms"] = float(qualify_ms_total)

            by_key: dict[tuple[str, str, str, float], Contract] = {}
            for contract in qualified:
                key = _opt_key(contract)
                if key is not None:
                    by_key[key] = contract
            if timing is not None:
                timing["qualified_count"] = int(len(by_key))
            out = _ordered_resolved(candidates, by_key)
            return _opt_timing_finish(out[:max_rows], stage="done", reason="ok")

        exact_roots: list[tuple[str, str]] = []
        prefix_roots: list[tuple[str, str]] = []
        contains_roots: list[tuple[str, str]] = []
        desc_roots: list[tuple[str, str]] = []
        seen_roots: set[tuple[str, str]] = set()
        for desc in matches:
            contract = getattr(desc, "contract", None)
            if not contract:
                continue
            deriv = {
                str(sec).strip().upper()
                for sec in (getattr(desc, "derivativeSecTypes", None) or [])
            }
            if "FOP" not in deriv:
                continue
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            if not symbol:
                continue
            exchange = str(getattr(contract, "exchange", "") or "").strip().upper() or "CME"
            key = (symbol, exchange)
            if key in seen_roots:
                continue
            seen_roots.add(key)
            desc_text = self._desc_text(desc)
            if symbol in term_set and not (token_is_alias_seed and symbol == token):
                exact_roots.append(key)
            elif any(symbol.startswith(term) for term in term_set):
                prefix_roots.append(key)
            elif any(term in symbol for term in term_set):
                contains_roots.append(key)
            elif any(term in desc_text for term in term_set):
                desc_roots.append(key)
            if len(seen_roots) >= max_rows * 8:
                break

        ranked_roots = [*exact_roots, *prefix_roots, *contains_roots, *desc_roots]
        symbol_exchange_map: dict[str, list[str]] = {}
        for symbol, exchange in ranked_roots:
            symbol_exchange_map.setdefault(symbol, [])
            if exchange not in symbol_exchange_map[symbol]:
                symbol_exchange_map[symbol].append(exchange)

        if exact_roots:
            candidate_symbols: list[str] = []
            for symbol, _exchange in exact_roots:
                if symbol not in candidate_symbols:
                    candidate_symbols.append(symbol)
        else:
            candidate_symbols = []
            for symbol, _exchange in ranked_roots:
                if symbol not in candidate_symbols:
                    candidate_symbols.append(symbol)

        if not candidate_symbols:
            for term in terms:
                if not term:
                    continue
                if len(term) > 8:
                    continue
                if not term.isalnum():
                    continue
                if term not in candidate_symbols:
                    candidate_symbols.append(term)
                if len(candidate_symbols) >= (max_rows * 6):
                    break
        if not candidate_symbols:
            return []

        out: list[Contract] = []
        seen_contracts: set[tuple[str, str, str, str, float]] = set()
        for symbol in candidate_symbols:
            preferred = symbol_exchange_map.get(symbol, [])
            future = None
            for exchange in self._future_exchange_candidates(symbol, preferred):
                future = await self.front_future(symbol, exchange=exchange, cache_ttl_sec=1800.0)
                if future is not None:
                    break
            if future is None:
                continue
            try:
                await self.connect()
            except Exception:
                continue

            future_symbol = str(getattr(future, "symbol", "") or symbol).strip().upper()
            future_ex = str(getattr(future, "exchange", "") or "").strip().upper()
            fut_fop_exchanges: list[str] = []
            for ex_value in (future_ex, *preferred):
                ex_clean = str(ex_value or "").strip().upper()
                if not ex_clean or ex_clean in fut_fop_exchanges:
                    continue
                fut_fop_exchanges.append(ex_clean)
            if not fut_fop_exchanges:
                fut_fop_exchanges = ["CME"]

            chains: list[object] = []
            for fut_fop_exchange in fut_fop_exchanges:
                try:
                    chains = await self._ib.reqSecDefOptParamsAsync(
                        future_symbol,
                        fut_fop_exchange,
                        str(getattr(future, "secType", "") or "FUT"),
                        int(getattr(future, "conId", 0) or 0),
                    )
                except Exception:
                    chains = []
                if chains:
                    break
            if not chains:
                alt_details: list[object] = []
                try:
                    alt_details = await self._ib.reqContractDetailsAsync(
                        Future(
                            symbol=future_symbol,
                            lastTradeDateOrContractMonth="",
                            exchange=future_ex or "CME",
                            currency=str(getattr(future, "currency", "") or "USD"),
                        )
                    )
                except Exception:
                    alt_details = []

                def _parse_exp(raw: object) -> date | None:
                    text = str(raw or "").strip()
                    if len(text) >= 8 and text[:8].isdigit():
                        try:
                            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
                        except ValueError:
                            return None
                    if len(text) >= 6 and text[:6].isdigit():
                        try:
                            return date(int(text[:4]), int(text[4:6]), 1)
                        except ValueError:
                            return None
                    return None

                ranked_alts: list[tuple[int, date, Contract]] = []
                today = _now_et().date()
                for detail in alt_details or []:
                    alt_contract = getattr(detail, "contract", None)
                    if alt_contract is None:
                        continue
                    if str(getattr(alt_contract, "secType", "") or "").strip().upper() != "FUT":
                        continue
                    alt_con_id = int(getattr(alt_contract, "conId", 0) or 0)
                    if alt_con_id <= 0:
                        continue
                    if alt_con_id == int(getattr(future, "conId", 0) or 0):
                        continue
                    exp_raw = getattr(detail, "realExpirationDate", None) or getattr(
                        alt_contract, "lastTradeDateOrContractMonth", None
                    )
                    exp = _parse_exp(exp_raw)
                    if exp is None:
                        continue
                    past_flag = 0 if exp >= today else 1
                    ranked_alts.append((past_flag, exp, alt_contract))
                ranked_alts.sort(key=lambda row: (row[0], row[1]))

                for _past_flag, _exp, alt_contract in ranked_alts[:14]:
                    alt_symbol = str(getattr(alt_contract, "symbol", "") or future_symbol).strip().upper()
                    alt_ex = str(getattr(alt_contract, "exchange", "") or "").strip().upper() or future_ex or "CME"
                    try:
                        alt_chains = await self._ib.reqSecDefOptParamsAsync(
                            alt_symbol,
                            alt_ex,
                            str(getattr(alt_contract, "secType", "") or "FUT"),
                            int(getattr(alt_contract, "conId", 0) or 0),
                        )
                    except Exception:
                        alt_chains = []
                    if not alt_chains:
                        continue
                    future = alt_contract
                    future_symbol = alt_symbol
                    future_ex = alt_ex
                    chains = list(alt_chains)
                    break
            if not chains:
                continue

            chain_rows: list[tuple[str, str, str, list[str], list[float]]] = []
            for value in chains:
                exchange_value = str(getattr(value, "exchange", "") or "").strip().upper()
                trading_class = str(getattr(value, "tradingClass", "") or "").strip().upper()
                multiplier = str(getattr(value, "multiplier", "") or getattr(future, "multiplier", "") or "")
                chain_expiries: list[str] = []
                for raw in (getattr(value, "expirations", ()) or ()):
                    text = str(raw or "").strip()
                    if len(text) >= 8 and text[:8].isdigit():
                        chain_expiries.append(text[:8])
                chain_expiries = sorted(set(chain_expiries))
                if not chain_expiries:
                    continue
                chain_strikes: list[float] = []
                for raw in (getattr(value, "strikes", ()) or ()):
                    try:
                        strike = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if strike > 0:
                        chain_strikes.append(strike)
                chain_strikes = sorted(set(chain_strikes))
                if not chain_strikes:
                    continue
                chain_rows.append(
                    (
                        exchange_value,
                        trading_class,
                        multiplier,
                        chain_expiries,
                        chain_strikes,
                    )
                )
            if not chain_rows:
                continue

            preferred_chain_rows = [row for row in chain_rows if row[0] == future_ex]
            if not preferred_chain_rows:
                preferred_chain_rows = list(chain_rows)

            expiry_chain: dict[str, tuple[tuple[int, int], str, str, str, list[float]]] = {}
            all_strikes: set[float] = set()
            for exchange_value, trading_class, multiplier, chain_expiries, chain_strikes in preferred_chain_rows:
                score = (len(chain_strikes), len(chain_expiries))
                all_strikes.update(chain_strikes)
                for expiry in chain_expiries:
                    existing = expiry_chain.get(expiry)
                    if existing is not None and existing[0] >= score:
                        continue
                    expiry_chain[expiry] = (
                        score,
                        exchange_value,
                        trading_class,
                        multiplier,
                        chain_strikes,
                    )

            expiries = sorted(expiry_chain.keys())
            if not expiries:
                continue
            future_ref: float | None = None
            try:
                future_ticker = await self.ensure_ticker(future, owner="search")
                future_ref = self._search_reference_price_from_ticker(future_ticker)
                if future_ref is None:
                    try:
                        await self._attempt_main_contract_snapshot_quote(future, ticker=future_ticker)
                    except Exception:
                        pass
                    future_ref = self._search_reference_price_from_ticker(future_ticker)
                if future_ref is None:
                    try:
                        await self._attempt_main_contract_historical_quote(future, ticker=future_ticker)
                    except Exception:
                        pass
                    future_ref = self._search_reference_price_from_ticker(future_ticker)
            except Exception:
                future_ref = None
            finally:
                future_con_id = int(getattr(future, "conId", 0) or 0)
                if future_con_id:
                    self.release_ticker(future_con_id, owner="search")

            strikes = sorted(all_strikes)
            if not strikes:
                continue
            if future_ref is None:
                future_ref = self._median_strike(strikes)
            if future_ref is None:
                continue
            target_strikes_per_expiry = max(1, min(20, int(max_rows) // 2))
            max_expiries = max(1, int(max_rows) // (2 * target_strikes_per_expiry))
            def _expiry_rank(expiry: str) -> tuple[int, float, str]:
                entry = expiry_chain.get(expiry)
                expiry_strikes = entry[4] if entry is not None and entry[4] else strikes
                if not expiry_strikes:
                    return (1, float("inf"), expiry)
                in_range = 0 if expiry_strikes[0] <= float(future_ref) <= expiry_strikes[-1] else 1
                nearest = min(abs(float(strike) - float(future_ref)) for strike in expiry_strikes)
                return (in_range, nearest, expiry)

            ranked_expiries = sorted(expiries, key=_expiry_rank)
            expiry_page_size = max(1, int(max_expiries))
            expiry_start = min(int(expiry_offset_clean), len(ranked_expiries))
            expiry_end = min(len(ranked_expiries), int(expiry_start) + int(expiry_page_size))
            selected_ranked_expiries = ranked_expiries[expiry_start:expiry_end]
            selected_expiries = sorted(selected_ranked_expiries)
            next_expiry_offset = int(expiry_end)
            has_more_expiries = bool(next_expiry_offset < len(ranked_expiries))
            if timing is not None:
                timing["expiry_count"] = len(expiries)
                timing["selected_expiry_count"] = len(selected_expiries)
                timing["has_more_expiries"] = bool(has_more_expiries)
                timing["next_expiry_offset"] = int(next_expiry_offset)
                timing["expiry_page_size"] = int(expiry_page_size)
            if not selected_expiries:
                continue
            rows_per_expiry = max(
                target_strikes_per_expiry,
                int(max_rows) // max(1, (2 * len(selected_expiries))),
            )
            if timing is not None:
                timing["rows_per_expiry"] = int(rows_per_expiry)
            currency = str(getattr(future, "currency", "") or "USD").strip().upper() or "USD"
            candidates: list[Contract] = []
            for expiry in selected_expiries:
                entry = expiry_chain.get(expiry)
                if entry is None:
                    continue
                _score, exchange, trading_class, multiplier, expiry_strikes = entry
                exchange = exchange or (future_ex or "CME")
                ordered_strikes = sorted(
                    expiry_strikes,
                    key=lambda value: (abs(float(value) - float(future_ref)), float(value)),
                )
                nearby = sorted(ordered_strikes[:rows_per_expiry])
                for strike in nearby:
                    for right in ("C", "P"):
                        candidate_key = (
                            future_symbol,
                            str(expiry),
                            trading_class,
                            right,
                            round(float(strike), 6),
                        )
                        if candidate_key in seen_contracts:
                            continue
                        seen_contracts.add(candidate_key)
                        candidates.append(
                            Contract(
                                secType="FOP",
                                symbol=future_symbol,
                                exchange=exchange,
                                currency=currency,
                                lastTradeDateOrContractMonth=expiry,
                                strike=float(strike),
                                right=right,
                                multiplier=multiplier,
                                tradingClass=trading_class,
                            )
                        )
            if not candidates:
                continue

            qualified_rows: list[Contract] = []
            batch_size = max(1, min(24, len(candidates)))
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start : start + batch_size]
                try:
                    qualified_rows.extend(list(await self._ib.qualifyContractsAsync(*batch) or []))
                except Exception:
                    continue

            by_key: dict[tuple[str, str, str, str, float], Contract] = {}
            for contract in qualified_rows:
                try:
                    strike = float(getattr(contract, "strike", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                key = (
                    str(getattr(contract, "symbol", "") or "").strip().upper(),
                    str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip(),
                    str(getattr(contract, "tradingClass", "") or "").strip().upper(),
                    str(getattr(contract, "right", "") or "").strip().upper()[:1],
                    round(strike, 6),
                )
                by_key[key] = contract

            for candidate in candidates:
                key = (
                    str(getattr(candidate, "symbol", "") or "").strip().upper(),
                    str(getattr(candidate, "lastTradeDateOrContractMonth", "") or "").strip(),
                    str(getattr(candidate, "tradingClass", "") or "").strip().upper(),
                    str(getattr(candidate, "right", "") or "").strip().upper()[:1],
                    round(float(getattr(candidate, "strike", 0.0) or 0.0), 6),
                )
                resolved = by_key.get(key)
                if resolved is None:
                    resolved = await self._qualify_contract(candidate, use_proxy=False)
                if resolved is None:
                    continue
                con_id = int(getattr(resolved, "conId", 0) or 0)
                if con_id <= 0:
                    continue
                out.append(resolved)
                if len(out) >= max_rows:
                    break
            if len(out) >= max_rows:
                break
        if timing is not None:
            timing["source"] = "search_contracts_fop"
            timing["stage"] = "done"
            timing["reason"] = "ok" if out else "empty"
            timing["result_count"] = len(out)
            timing["total_ms"] = (time.monotonic() - started_mono) * 1000.0
        return out

    async def stock_option_chain(self, symbol: str):
        """Return (qualified_underlying, chain) for an equity option underlyer."""
        candidate = Stock(symbol=symbol, exchange="SMART", currency="USD")
        underlying = await self._qualify_contract(candidate, use_proxy=True) or candidate
        try:
            await self.connect_proxy()
        except Exception:
            return None
        try:
            chains = await self._ib_proxy.reqSecDefOptParamsAsync(
                underlying.symbol,
                "",
                underlying.secType,
                int(getattr(underlying, "conId", 0) or 0),
            )
        except Exception:
            chains = []
        if not chains:
            return None
        chain = next((c for c in chains if getattr(c, "exchange", None) == "SMART"), chains[0])
        return underlying, chain

    async def qualify_proxy_contracts(self, *contracts: Contract) -> list[Contract]:
        try:
            await self.connect_proxy()
        except Exception:
            return []
        if not contracts:
            return []

        cleaned = [contract for contract in contracts if contract is not None]
        if not cleaned:
            return []

        try:
            result = await self._ib_proxy.qualifyContractsAsync(*cleaned)
            if result:
                return list(result)
        except Exception:
            pass

        # Fallback path for market-open bursts/pacing: recover partial results
        # via smaller batches and finally single-contract qualification.
        resolved: list[Contract] = []
        seen_con_ids: set[int] = set()
        chunk_size = max(1, min(24, len(cleaned)))
        for start in range(0, len(cleaned), chunk_size):
            batch = cleaned[start : start + chunk_size]
            try:
                qualified = list(await self._ib_proxy.qualifyContractsAsync(*batch) or [])
            except Exception:
                qualified = []

            if not qualified and len(batch) > 1:
                for candidate in batch:
                    try:
                        single = list(await self._ib_proxy.qualifyContractsAsync(candidate) or [])
                    except Exception:
                        continue
                    qualified.extend(single)

            for contract in qualified:
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id > 0:
                    if con_id in seen_con_ids:
                        continue
                    seen_con_ids.add(con_id)
                resolved.append(contract)
        return resolved

    async def front_future(
        self,
        symbol: str,
        *,
        exchange: str = "CME",
        cache_ttl_sec: float = 3600.0,
    ) -> Contract | None:
        """Resolve a tradable front-month future contract.

        Uses a TTL cache to avoid repeated contract-details requests in the live UI.
        """
        sym = str(symbol or "").strip().upper()
        ex = str(exchange or "").strip().upper() or "CME"
        key = (sym, ex)
        cached = self._front_future_cache.get(key)

        if cached:
            contract, cached_at = cached
            cached_month = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            if cached_month and time.monotonic() - cached_at < float(cache_ttl_sec):
                return contract

        def _parse_expiry(raw: str | None) -> datetime | None:
            if not raw:
                return None
            cleaned = str(raw).strip()
            if not cleaned:
                return None
            m8 = re.search(r"(\d{8})", cleaned)
            if m8:
                try:
                    text = m8.group(1)
                    return datetime(int(text[:4]), int(text[4:6]), int(text[6:8]))
                except ValueError:
                    return None
            m6 = re.search(r"(\d{6})", cleaned)
            if m6:
                try:
                    text = m6.group(1)
                    return datetime(int(text[:4]), int(text[4:6]), 1)
                except ValueError:
                    return None
            return None

        async with self._lock:
            cached = self._front_future_cache.get(key)
            if cached:
                contract, cached_at = cached
                cached_month = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
                if cached_month and time.monotonic() - cached_at < float(cache_ttl_sec):
                    return contract
            try:
                await self.connect()
            except Exception:
                return None
            candidate = Future(symbol=sym, lastTradeDateOrContractMonth="", exchange=ex, currency="USD")
            try:
                details = await self._ib.reqContractDetailsAsync(candidate)
            except Exception:
                details = []
            if not details:
                return None

            today = _now_et().date()
            best = None
            best_dt = None
            latest = None
            latest_dt = None
            with_month = None
            for d in details:
                contract = getattr(d, "contract", None)
                if not contract:
                    continue
                if getattr(contract, "secType", "") != "FUT":
                    continue
                if with_month is None:
                    month_text = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
                    if month_text:
                        with_month = contract
                exp_raw = getattr(d, "realExpirationDate", None) or getattr(
                    contract, "lastTradeDateOrContractMonth", None
                )
                exp_dt = _parse_expiry(str(exp_raw) if exp_raw is not None else None)
                if exp_dt is None:
                    continue
                exp_date = exp_dt.date()
                if latest_dt is None or exp_date > latest_dt:
                    latest_dt = exp_date
                    latest = contract
                if exp_date < today:
                    continue
                if best_dt is None or exp_date < best_dt:
                    best_dt = exp_date
                    best = contract

            if best is None:
                if latest is not None:
                    best = latest
                elif with_month is not None:
                    best = with_month
                else:
                    for d in details:
                        contract = getattr(d, "contract", None)
                        if contract and getattr(contract, "secType", "") == "FUT":
                            best = contract
                            break
            if best is None:
                return None

            try:
                qualified = await self._ib.qualifyContractsAsync(best)
            except Exception:
                qualified = []
            resolved = qualified[0] if qualified else best
            self._front_future_cache[key] = (resolved, time.monotonic())
            return resolved

    def release_ticker(self, con_id: int, *, owner: str = "default") -> None:
        if not con_id:
            return
        owners = self._ticker_owners.get(con_id)
        if owners is not None:
            owners.discard(owner)
            if owners:
                return
            self._ticker_owners.pop(con_id, None)

        entry = self._detail_tickers.pop(con_id, None)
        if entry:
            ib, ticker = entry
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        probe_task = self._proxy_contract_probe_tasks.pop(con_id, None)
        if probe_task and not probe_task.done():
            probe_task.cancel()
        live_task = self._proxy_contract_live_tasks.pop(con_id, None)
        if live_task and not live_task.done():
            live_task.cancel()
        delayed_task = self._proxy_contract_delayed_tasks.pop(con_id, None)
        if delayed_task and not delayed_task.done():
            delayed_task.cancel()
        main_probe_task = self._main_contract_probe_tasks.pop(con_id, None)
        if main_probe_task and not main_probe_task.done():
            main_probe_task.cancel()
        main_watchdog_task = self._main_contract_watchdog_tasks.pop(con_id, None)
        if main_watchdog_task and not main_watchdog_task.done():
            main_watchdog_task.cancel()

    def account_value(
        self,
        tag: str,
        *,
        currency: str | None = None,
    ) -> tuple[float | None, str | None, datetime | None]:
        desired_currency = str(currency or "").strip().upper() or None
        account = self._config.account or ""
        if desired_currency:
            cached_exact = self._account_value_cache.get((tag, desired_currency))
            if cached_exact:
                value, updated_at = cached_exact
                return value, desired_currency, updated_at
            cached = None
        else:
            cached = _pick_cached_value(self._account_value_cache, tag)
        if cached:
            value, currency, updated_at = cached
            return value, currency, updated_at
        values = [v for v in self._ib.accountValues(account) if v.tag == tag]
        if not values:
            return None, None, None
        if desired_currency:
            chosen = next(
                (
                    value
                    for value in values
                    if str(getattr(value, "currency", "") or "").strip().upper() == desired_currency
                ),
                None,
            )
        else:
            chosen = _pick_account_value(values)
        if not chosen:
            return None, None, None
        parsed = self._clean_pnl_stream_value(getattr(chosen, "value", None))
        if parsed is None:
            return None, chosen.currency, None
        return parsed, chosen.currency, None

    def account_exchange_rate(self, currency: str) -> float | None:
        target = str(currency or "").strip().upper()
        if not target:
            return None
        rate, _currency, _updated = self.account_value("ExchangeRate", currency=target)
        try:
            parsed = float(rate) if rate is not None else None
        except (TypeError, ValueError):
            parsed = None
        if parsed is None or parsed <= 0:
            return None
        return float(parsed)

    async def fx_rate(
        self,
        from_currency: str,
        to_currency: str,
        *,
        max_age_sec: float = 15.0,
    ) -> float | None:
        src = str(from_currency or "").strip().upper()
        dst = str(to_currency or "").strip().upper()
        if not src or not dst:
            return None
        if src == dst:
            return 1.0

        key = (src, dst)
        now = time.monotonic()
        cached = self._fx_rate_cache.get(key)
        if cached and (now - float(cached[1])) <= float(max_age_sec):
            return float(cached[0])

        await self.connect()
        contract = Forex(f"{src}{dst}")
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
        except Exception:
            qualified = []
        if qualified:
            contract = qualified[0]

        ticker = await self.ensure_ticker(contract, owner="fx")

        def _as_pos_float(value: object) -> float | None:
            try:
                parsed = float(value) if value is not None else None
            except (TypeError, ValueError):
                parsed = None
            if parsed is None or parsed <= 0:
                return None
            return parsed

        bid = _as_pos_float(getattr(ticker, "bid", None))
        ask = _as_pos_float(getattr(ticker, "ask", None))
        last = _as_pos_float(getattr(ticker, "last", None))
        close = _as_pos_float(getattr(ticker, "close", None))
        market_price = None
        market_price_fn = getattr(ticker, "marketPrice", None)
        if callable(market_price_fn):
            try:
                market_price = _as_pos_float(market_price_fn())
            except Exception:
                market_price = None

        rate = ((bid + ask) / 2.0) if bid is not None and ask is not None else (last or market_price or close)
        if rate is not None and rate > 0:
            self._fx_rate_cache[key] = (float(rate), now)

        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id:
            self.release_ticker(con_id, owner="fx")
        return float(rate) if rate is not None and rate > 0 else None

    async def convert_currency_value(
        self,
        value: float,
        *,
        from_currency: str,
        to_currency: str,
    ) -> tuple[float | None, float | None]:
        src = str(from_currency or "").strip().upper()
        dst = str(to_currency or "").strip().upper()
        if not src or not dst:
            return None, None
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None, None
        if src == dst:
            return amount, 1.0

        from_rate = self.account_exchange_rate(src)
        to_rate = self.account_exchange_rate(dst)
        if from_rate is not None and to_rate is not None and from_rate > 0 and to_rate > 0:
            rate = float(from_rate) / float(to_rate)
            return amount * float(rate), float(rate)

        direct = await self.fx_rate(src, dst)
        if direct is not None and direct > 0:
            return amount * float(direct), float(direct)

        inverse = await self.fx_rate(dst, src)
        if inverse is not None and inverse > 0:
            rate = 1.0 / float(inverse)
            return amount * float(rate), float(rate)

        return None, None

    async def hard_refresh(self) -> None:
        async with self._lock:
            await self.connect()
            account = self._config.account or ""
            try:
                self._ib.client.reqAccountUpdates(False, account)
            except Exception:
                pass
            self._clear_pnl_single_subscriptions(cancel=True)
            if self._pnl_account:
                try:
                    self._ib.cancelPnL(self._pnl_account)
                except Exception:
                    pass
            self._pnl = None
            self._pnl_account = None
            self._account_updates_started = False
            self._session_close_cache = {}
            await self._ensure_account_updates()
        async with self._index_lock:
            for ticker in self._index_tickers.values():
                try:
                    self._ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
            if self._index_probe_task and not self._index_probe_task.done():
                self._index_probe_task.cancel()
            self._index_probe_task = None
            self._index_requalify_on_reload = False
            self._index_session_flags = None
            self._index_futures_session_open = None
            self._index_session_include_overnight = None
            self._index_tickers = {}
            self._index_task = None
            await self._ensure_index_tickers()
            self._start_index_probe()
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception as exc:
                self._proxy_error = str(exc)
                self._proxy_tickers = {}
                self._proxy_task = None
                return
            self._proxy_force_delayed = False
            self._proxy_contract_force_delayed = set()
            self._proxy_session_bucket = None
            self._proxy_session_include_overnight = None
            for task in self._proxy_contract_probe_tasks.values():
                if task and not task.done():
                    task.cancel()
            for task in self._proxy_contract_live_tasks.values():
                if task and not task.done():
                    task.cancel()
            for task in self._proxy_contract_delayed_tasks.values():
                if task and not task.done():
                    task.cancel()
            for task in self._main_contract_probe_tasks.values():
                if task and not task.done():
                    task.cancel()
            for task in self._main_contract_watchdog_tasks.values():
                if task and not task.done():
                    task.cancel()
            self._proxy_contract_probe_tasks = {}
            self._proxy_contract_live_tasks = {}
            self._proxy_contract_delayed_tasks = {}
            self._main_contract_probe_tasks = {}
            self._main_contract_watchdog_tasks = {}
            for ticker in self._proxy_tickers.values():
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._proxy_tickers = {}
            self._proxy_task = None
            await self._ensure_proxy_tickers()

    async def _ensure_account_updates(self) -> None:
        if self._account_updates_started:
            return
        await self.connect()
        account = self._config.account or ""
        # Start streaming account/portfolio updates without blocking UI startup.
        self._ib.client.reqAccountUpdates(True, account)
        self._account_updates_started = True
        self._ensure_pnl(account)

    def _ensure_pnl(self, account: str) -> None:
        if self._pnl:
            return
        account = self._resolve_account(account)
        if not account:
            return
        try:
            self._pnl = self._ib.reqPnL(account)
            self._pnl_account = account
        except Exception:
            self._pnl = None
            self._pnl_account = None

    def _resolve_account(self, account: str | None = None) -> str:
        candidate = str(account or "").strip()
        if candidate:
            return candidate
        if self._pnl_account:
            return str(self._pnl_account)
        try:
            accounts = self._ib.managedAccounts()
        except Exception:
            return ""
        return str(accounts[0]) if accounts else ""

    def _sync_pnl_single_subscriptions(self, items: list[PortfolioItem], account: str | None) -> None:
        if not hasattr(self._ib, "reqPnLSingle") or not hasattr(self._ib, "cancelPnLSingle"):
            return
        resolved_account = self._resolve_account(account)
        if not resolved_account:
            return

        desired_con_ids: set[int] = set()
        for item in items:
            contract = getattr(item, "contract", None)
            try:
                con_id = int(getattr(contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
            if con_id > 0:
                desired_con_ids.add(con_id)

        if self._pnl_single_account and self._pnl_single_account != resolved_account:
            self._clear_pnl_single_subscriptions(cancel=True)

        active_con_ids = set(self._pnl_single_by_con_id.keys())
        remove_con_ids = active_con_ids - desired_con_ids
        add_con_ids = desired_con_ids - active_con_ids

        if remove_con_ids:
            cancel_account = self._pnl_single_account or resolved_account
            for con_id in remove_con_ids:
                try:
                    self._ib.cancelPnLSingle(cancel_account, "", int(con_id))
                except Exception:
                    pass
                self._pnl_single_by_con_id.pop(int(con_id), None)

        if add_con_ids:
            for con_id in add_con_ids:
                try:
                    pnl_single = self._ib.reqPnLSingle(resolved_account, "", int(con_id))
                except Exception:
                    continue
                self._pnl_single_by_con_id[int(con_id)] = pnl_single

        if desired_con_ids:
            self._pnl_single_account = resolved_account
        elif not self._pnl_single_by_con_id:
            self._pnl_single_account = None

    def _clear_pnl_single_subscriptions(self, *, cancel: bool) -> None:
        if cancel and self._pnl_single_account and hasattr(self._ib, "cancelPnLSingle"):
            account = str(self._pnl_single_account)
            for con_id in list(self._pnl_single_by_con_id.keys()):
                try:
                    self._ib.cancelPnLSingle(account, "", int(con_id))
                except Exception:
                    continue
        self._pnl_single_by_con_id = {}
        self._pnl_single_account = None

    def _index_market_data_type(self, contract: Contract, *, now: datetime) -> int:
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type in ("FUT", "FOP"):
            ladder = _futures_md_ladder(now)
            if self._index_force_delayed:
                delayed_types = [int(md) for md in ladder if int(md) in (3, 4)]
                return int(delayed_types[0]) if delayed_types else 3
            return int(ladder[0]) if ladder else 1
        return 3 if self._index_force_delayed else 1

    def _index_market_data_contract(
        self,
        contract: Contract,
        *,
        md_type: int,
        now: datetime,
    ) -> Contract:
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type == "STK":
            _, include_overnight = _session_flags(now)
            return self._stock_market_data_contract(
                contract,
                include_overnight=include_overnight,
                delayed=bool(int(md_type) in (3, 4)),
            )
        if sec_type in ("FUT", "FOP", "OPT"):
            return self._normalize_derivative_market_data_contract(
                contract,
                sec_type=sec_type,
            )
        return contract

    async def _ensure_index_tickers(self) -> None:
        await self.connect()
        now = _now_et()
        current_session_flags = tuple(bool(v) for v in _session_flags(now))
        self._index_futures_session_open = bool(_futures_session_is_open(now))
        self._index_session_flags = current_session_flags
        self._index_session_include_overnight = bool(current_session_flags[1])
        if not self._index_contracts:
            self._index_contracts = await self._qualify_index_contracts()

        desired_specs: dict[str, tuple[Contract, int]] = {}
        for symbol, contract in self._index_contracts.items():
            md_type = int(self._index_market_data_type(contract, now=now))
            req_contract = self._index_market_data_contract(
                contract,
                md_type=md_type,
                now=now,
            )
            desired_specs[symbol] = (req_contract, md_type)

        reload_needed = set(self._index_tickers.keys()) != set(desired_specs.keys())
        if not reload_needed:
            for symbol, (req_contract, req_md_type) in desired_specs.items():
                ticker = self._index_tickers.get(symbol)
                if ticker is None:
                    reload_needed = True
                    break
                current = getattr(ticker, "contract", None) or Contract()
                cur_con_id = int(getattr(current, "conId", 0) or 0)
                req_con_id = int(getattr(req_contract, "conId", 0) or 0)
                cur_exchange = str(getattr(current, "exchange", "") or "").strip().upper()
                req_exchange = str(getattr(req_contract, "exchange", "") or "").strip().upper()
                current_req_md_raw = getattr(ticker, "tbRequestedMdType", None)
                try:
                    current_req_md = int(current_req_md_raw) if current_req_md_raw is not None else None
                except (TypeError, ValueError):
                    current_req_md = None
                if (
                    cur_exchange != req_exchange
                    or (req_con_id and cur_con_id != req_con_id)
                    or current_req_md != int(req_md_type)
                ):
                    reload_needed = True
                    break
        if reload_needed:
            for ticker in self._index_tickers.values():
                try:
                    self._ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._index_tickers = {}
        if not self._index_tickers:
            for symbol, (req_contract, req_md_type) in desired_specs.items():
                self._ib.reqMarketDataType(int(req_md_type))
                ticker = self._ib.reqMktData(req_contract)
                setattr(ticker, "tbRequestedMdType", int(req_md_type))
                self._index_tickers[symbol] = ticker

    async def _ensure_proxy_tickers(self) -> None:
        await self.connect_proxy()
        self._maybe_reset_proxy_contract_delay_on_session_change()
        md_type = 3 if self._proxy_force_delayed else 1
        self._ib_proxy.reqMarketDataType(md_type)
        if not self._proxy_contracts:
            self._proxy_contracts = await self._qualify_proxy_contracts()
        _, include_overnight = _session_flags(_now_et())
        delayed = bool(md_type == 3)
        desired_contracts: dict[str, Contract] = {}
        for symbol, contract in self._proxy_contracts.items():
            req_contract = contract
            if str(getattr(contract, "secType", "") or "").strip().upper() == "STK":
                req_contract = self._stock_market_data_contract(
                    contract,
                    include_overnight=include_overnight,
                    delayed=delayed,
                )
            desired_contracts[symbol] = req_contract
        self._proxy_session_include_overnight = include_overnight
        reload_needed = set(self._proxy_tickers.keys()) != set(desired_contracts.keys())
        if not reload_needed:
            for symbol, req_contract in desired_contracts.items():
                ticker = self._proxy_tickers.get(symbol)
                if ticker is None:
                    reload_needed = True
                    break
                current = ticker.contract
                cur_con_id = int(getattr(current, "conId", 0) or 0)
                req_con_id = int(getattr(req_contract, "conId", 0) or 0)
                cur_exchange = str(getattr(current, "exchange", "") or "").strip().upper()
                req_exchange = str(getattr(req_contract, "exchange", "") or "").strip().upper()
                if cur_exchange != req_exchange or (req_con_id and cur_con_id != req_con_id):
                    reload_needed = True
                    break
        if reload_needed:
            for ticker in self._proxy_tickers.values():
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._proxy_tickers = {}
        if not self._proxy_tickers:
            for symbol, req_contract in desired_contracts.items():
                self._proxy_tickers[symbol] = self._ib_proxy.reqMktData(req_contract)

    async def _qualify_proxy_contracts(self) -> dict[str, Contract]:
        async def _resolve(symbol: str) -> tuple[str, Contract | None]:
            candidate = Stock(symbol=symbol, exchange="SMART", currency="USD")
            try:
                result = await self._ib_proxy.qualifyContractsAsync(candidate)
            except Exception:
                return symbol, None
            return symbol, (result[0] if result else None)

        tasks = [_resolve(symbol) for symbol in _PROXY_SYMBOLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        qualified: dict[str, Contract] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            symbol, contract = result
            if contract is not None:
                qualified[symbol] = contract
        return qualified

    async def _qualify_contract(self, contract: Contract, use_proxy: bool) -> Contract | None:
        try:
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib
        except Exception:
            return None
        try:
            result = await ib.qualifyContractsAsync(contract)
        except Exception:
            return None
        return result[0] if result else None

    async def _qualify_index_contracts(self) -> dict[str, Contract]:
        async def _resolve(symbol: str) -> tuple[str, Contract | None]:
            contract = None
            preferred_exchange = str(
                _INDEX_STRIP_EXCHANGE_HINTS.get(str(symbol or "").strip().upper(), "")
            ).strip().upper()
            exchanges: list[str] = []
            if preferred_exchange:
                exchanges.append(preferred_exchange)
            for candidate in self._future_exchange_candidates(symbol, ()):
                exchange = str(candidate or "").strip().upper()
                if not exchange or exchange in exchanges:
                    continue
                exchanges.append(exchange)
            for exchange in exchanges:
                contract = await self.front_future(
                    symbol,
                    exchange=exchange,
                    cache_ttl_sec=1800.0,
                )
                if contract is not None:
                    break
            return symbol, contract

        tasks = [_resolve(symbol) for symbol in _INDEX_STRIP_SYMBOLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        qualified: dict[str, Contract] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            symbol, contract = result
            if contract is not None:
                qualified[symbol] = contract
        return qualified

    async def _load_index_tickers(self) -> None:
        try:
            async with self._index_lock:
                await self._ensure_index_tickers()
            self._index_error = None
            self._start_index_probe()
        except Exception as exc:
            self._index_error = str(exc)

    async def _reload_index_tickers(self) -> None:
        try:
            async with self._index_lock:
                await self.connect()
                for ticker in self._index_tickers.values():
                    try:
                        self._ib.cancelMktData(ticker.contract)
                    except Exception:
                        pass
                self._index_tickers = {}
                if self._index_requalify_on_reload:
                    self._index_requalify_on_reload = False
                    self._index_contracts = {}
                await self._ensure_index_tickers()
            self._index_error = None
            self._start_index_probe()
        except Exception as exc:
            self._index_error = str(exc)

    def _start_index_resubscribe(self, *, requalify: bool = False) -> None:
        if requalify:
            self._index_requalify_on_reload = True
        if self._index_probe_task and not self._index_probe_task.done():
            self._index_probe_task.cancel()
        self._index_probe_task = None
        if self._index_task and not self._index_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._index_task = loop.create_task(self._reload_index_tickers())

    def _start_index_probe(self) -> None:
        if not self._index_tickers:
            return
        if self._index_probe_task is not None:
            if not self._index_probe_task.done():
                return
            self._index_probe_task = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._index_probe_task = loop.create_task(self._probe_index_quotes())

    async def _probe_index_quotes(self) -> None:
        await asyncio.sleep(float(_INDEX_QUOTE_PROBE_INITIAL_SEC))
        if not self._index_tickers:
            return
        if self._index_has_data():
            return
        if not self._index_force_delayed:
            self._index_force_delayed = True
            self._start_index_resubscribe(requalify=False)
        else:
            self._start_index_resubscribe(requalify=False)
        await asyncio.sleep(float(_INDEX_QUOTE_PROBE_INITIAL_SEC))
        if not self._index_tickers:
            return
        if self._index_has_data():
            return
        self._start_index_resubscribe(requalify=False)

    def _index_has_data(self) -> bool:
        for ticker in self._index_tickers.values():
            if self._ticker_has_data(ticker):
                return True
        return False

    def _is_index_contract(self, contract: Contract | None) -> bool:
        if contract is None:
            return False
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        if symbol in _INDEX_STRIP_SYMBOLS:
            return True
        try:
            con_id = int(getattr(contract, "conId", 0) or 0)
        except (TypeError, ValueError):
            con_id = 0
        if con_id <= 0:
            return False
        for candidate in self._index_contracts.values():
            try:
                if int(getattr(candidate, "conId", 0) or 0) == con_id:
                    return True
            except (TypeError, ValueError):
                continue
        for ticker in self._index_tickers.values():
            try:
                if int(getattr(getattr(ticker, "contract", None), "conId", 0) or 0) == con_id:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    async def _load_proxy_tickers(self) -> None:
        try:
            async with self._proxy_lock:
                await self._ensure_proxy_tickers()
            self._proxy_error = None
            self._start_proxy_probe()
        except Exception as exc:
            self._proxy_error = str(exc)

    @classmethod
    def _ticker_has_quote_or_close(cls, ticker: Ticker | None) -> bool:
        return bool(cls._ticker_has_data(ticker) or cls._ticker_has_close_data(ticker))

    @staticmethod
    def _set_ticker_quote_error(ticker: Ticker, *, error_code: int, error_text: str | None = None) -> None:
        setattr(ticker, "tbQuoteErrorCode", int(error_code))
        setattr(ticker, "tbQuoteError", str(error_text or "").strip())

    @staticmethod
    def _clear_ticker_quote_error(ticker: Ticker) -> None:
        if hasattr(ticker, "tbQuoteErrorCode"):
            setattr(ticker, "tbQuoteErrorCode", None)
        if hasattr(ticker, "tbQuoteError"):
            setattr(ticker, "tbQuoteError", None)

    def _on_error_main(self, reqId, errorCode, errorString, contract) -> None:
        self._remember_order_error(reqId, errorCode, errorString)
        if errorCode in _ENTITLEMENT_ERROR_CODES:
            if self._is_index_contract(contract):
                self._index_force_delayed = True
                self._start_index_resubscribe(requalify=True)
            con_id = int(getattr(contract, "conId", 0) or 0) if contract else 0
            sec_type = (
                str(getattr(contract, "secType", "") or "").strip().upper()
                if contract is not None
                else ""
            )
            if con_id and sec_type in ("FUT", "FOP"):
                entry = self._detail_tickers.get(con_id)
                if entry:
                    _ib, ticker = entry
                    req_contract = getattr(ticker, "contract", None) or contract
                    if self._ticker_has_quote_or_close(ticker):
                        self._clear_ticker_quote_error(ticker)
                        if req_contract is not None:
                            self._start_main_contract_quote_watchdog(req_contract)
                    else:
                        self._set_ticker_quote_error(
                            ticker,
                            error_code=int(errorCode),
                            error_text=str(errorString or "").strip(),
                        )
                        self._resubscribe_main_contract_stream(ticker, md_type_override=4)
                        if req_contract is not None:
                            self._start_main_contract_quote_probe(req_contract)
                elif contract is not None:
                    self._start_main_contract_quote_probe(contract)
            elif contract is None:
                for _con_id, (ib, ticker) in list(self._detail_tickers.items()):
                    if ib is not self._ib:
                        continue
                    req_contract = getattr(ticker, "contract", None)
                    sec = str(getattr(req_contract, "secType", "") or "").strip().upper()
                    if sec not in ("FUT", "FOP"):
                        continue
                    if self._ticker_has_quote_or_close(ticker):
                        self._clear_ticker_quote_error(ticker)
                        continue
                    self._set_ticker_quote_error(
                        ticker,
                        error_code=int(errorCode),
                        error_text=str(errorString or "").strip(),
                    )
                    if req_contract is not None:
                        self._start_main_contract_quote_probe(req_contract)
        self._handle_conn_error(errorCode)

    def _on_error_proxy(self, reqId, errorCode, errorString, contract) -> None:
        self._remember_order_error(reqId, errorCode, errorString)
        if errorCode == 10167:
            con_id = int(getattr(contract, "conId", 0) or 0) if contract else 0
            if con_id:
                self._proxy_contract_force_delayed.add(con_id)
                self._start_proxy_contract_delayed_resubscribe(contract)
            elif not self._proxy_force_delayed:
                self._proxy_force_delayed = True
                self._start_proxy_resubscribe()
        if errorCode in (354, 10089, 10090, 10091, 10168) and contract:
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._proxy_contract_force_delayed.add(con_id)
                self._start_proxy_contract_delayed_resubscribe(contract)
        self._handle_conn_error(errorCode)

    def _remember_order_error(self, req_id, error_code, error_text) -> None:
        try:
            order_id = int(req_id or 0)
        except (TypeError, ValueError):
            return
        if order_id <= 0:
            return
        try:
            code = int(error_code or 0)
        except (TypeError, ValueError):
            code = 0
        message = str(error_text or "").strip()
        if not message:
            return
        # Keep order-related errors only; reqId is shared across other request types.
        if code not in (110, 201, 202, 10147, 10148, 10149) and "order" not in message.lower():
            return
        self._order_error_cache[order_id] = (time.monotonic(), code, message)
        if len(self._order_error_cache) > 512:
            stale = sorted(
                self._order_error_cache.items(),
                key=lambda item: item[1][0],
            )[:64]
            for key, _value in stale:
                self._order_error_cache.pop(int(key), None)

    def pop_order_error(self, order_id: int, *, max_age_sec: float = 120.0) -> dict | None:
        try:
            key = int(order_id or 0)
        except (TypeError, ValueError):
            key = 0
        if key <= 0:
            return None
        payload = self._order_error_cache.pop(key, None)
        if payload is None:
            return None
        ts_mono, code, message = payload
        if (time.monotonic() - float(ts_mono)) > float(max_age_sec):
            return None
        return {"code": int(code), "message": str(message)}

    def _on_disconnected_main(self) -> None:
        if self._shutdown:
            return
        main_id = int(self._connected_main_client_id or self._main_client_id or 0)
        proxy_id = int(self._connected_proxy_client_id or self._proxy_client_id or 0)
        if main_id > 0 and proxy_id > 0:
            self._quarantine_pair(main_id, proxy_id)
        self._connected_main_client_id = None
        self._resubscribe_main_needed = True
        self._account_updates_started = False
        self._pnl = None
        self._pnl_account = None
        self._clear_pnl_single_subscriptions(cancel=False)
        self._account_value_cache = {}
        self._fx_rate_cache = {}
        self._index_tickers = {}
        self._index_task = None
        if self._index_probe_task and not self._index_probe_task.done():
            self._index_probe_task.cancel()
        self._index_probe_task = None
        self._index_requalify_on_reload = False
        self._index_session_flags = None
        self._index_futures_session_open = None
        self._index_session_include_overnight = None
        for task in self._main_contract_probe_tasks.values():
            if task and not task.done():
                task.cancel()
        self._main_contract_probe_tasks = {}
        for task in self._main_contract_watchdog_tasks.values():
            if task and not task.done():
                task.cancel()
        self._main_contract_watchdog_tasks = {}
        self._reconnect_requested = True
        self._start_reconnect_loop()
        if self._update_callback:
            self._update_callback()

    def _on_disconnected_proxy(self) -> None:
        if self._shutdown:
            return
        main_id = int(self._connected_main_client_id or self._main_client_id or 0)
        proxy_id = int(self._connected_proxy_client_id or self._proxy_client_id or 0)
        if main_id > 0 and proxy_id > 0:
            self._quarantine_pair(main_id, proxy_id)
        self._connected_proxy_client_id = None
        self._resubscribe_proxy_needed = True
        self._proxy_task = None
        self._proxy_tickers = {}
        self._proxy_session_bucket = None
        self._proxy_session_include_overnight = None
        self._proxy_probe_task = None
        for task in self._proxy_contract_probe_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._proxy_contract_live_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._proxy_contract_delayed_tasks.values():
            if task and not task.done():
                task.cancel()
        self._proxy_contract_probe_tasks = {}
        self._proxy_contract_live_tasks = {}
        self._proxy_contract_delayed_tasks = {}
        self._reconnect_requested = True
        self._start_reconnect_loop()
        if self._update_callback:
            self._update_callback()

    def _start_proxy_contract_delayed_resubscribe(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        probe_task = self._proxy_contract_probe_tasks.pop(con_id, None)
        if probe_task and not probe_task.done():
            probe_task.cancel()
        existing = self._proxy_contract_delayed_tasks.get(con_id) if con_id else None
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._resubscribe_proxy_contract_delayed(contract))
        if con_id:
            self._proxy_contract_delayed_tasks[con_id] = task

            def _cleanup(done_task: asyncio.Task, key: int = con_id) -> None:
                current = self._proxy_contract_delayed_tasks.get(key)
                if current is done_task:
                    self._proxy_contract_delayed_tasks.pop(key, None)

            task.add_done_callback(_cleanup)

    def _start_proxy_contract_live_resubscribe(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id or con_id in self._proxy_contract_force_delayed:
            return
        delayed_task = self._proxy_contract_delayed_tasks.pop(con_id, None)
        if delayed_task and not delayed_task.done():
            delayed_task.cancel()
        existing = self._proxy_contract_live_tasks.get(con_id)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._resubscribe_proxy_contract_live(contract))
        self._proxy_contract_live_tasks[con_id] = task

        def _cleanup(done_task: asyncio.Task, key: int = con_id) -> None:
            current = self._proxy_contract_live_tasks.get(key)
            if current is done_task:
                self._proxy_contract_live_tasks.pop(key, None)

        task.add_done_callback(_cleanup)

    @staticmethod
    def _stock_market_data_contract(
        contract: Contract,
        *,
        include_overnight: bool,
        delayed: bool,
    ) -> Contract:
        req_contract = contract
        exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
        primary_exchange = str(getattr(contract, "primaryExchange", "") or "").strip().upper()
        if include_overnight:
            if exchange != "OVERNIGHT":
                req_contract = copy.copy(contract)
                req_contract.exchange = "OVERNIGHT"
            return req_contract
        if delayed and exchange in ("", "SMART") and primary_exchange:
            req_contract = copy.copy(contract)
            req_contract.exchange = primary_exchange
            return req_contract
        if not exchange:
            req_contract = copy.copy(contract)
            req_contract.exchange = "SMART"
        return req_contract

    def _normalize_derivative_market_data_contract(
        self,
        contract: Contract,
        *,
        sec_type: str | None = None,
    ) -> Contract:
        normalized = contract
        resolved_sec_type = str(sec_type or getattr(contract, "secType", "") or "").strip().upper()
        if resolved_sec_type not in ("FUT", "FOP", "OPT"):
            return normalized
        current_exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
        if current_exchange:
            return normalized
        normalized = copy.copy(contract)
        if resolved_sec_type in ("FUT", "FOP"):
            primary_exchange = str(getattr(contract, "primaryExchange", "") or "").strip().upper()
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            preferred = [primary_exchange] if primary_exchange else []
            candidates = self._future_exchange_candidates(symbol, preferred)
            normalized.exchange = candidates[0] if candidates else (primary_exchange or "CME")
            return normalized
        normalized.exchange = "SMART"
        return normalized

    @staticmethod
    def _ticker_has_data(ticker: Ticker | None) -> bool:
        if ticker is None:
            return False
        bid = getattr(ticker, "bid", None)
        ask = getattr(ticker, "ask", None)
        try:
            bid_num = float(bid) if bid is not None else None
        except (TypeError, ValueError):
            bid_num = None
        try:
            ask_num = float(ask) if ask is not None else None
        except (TypeError, ValueError):
            ask_num = None
        if (
            bid_num is not None
            and ask_num is not None
            and not math.isnan(bid_num)
            and not math.isnan(ask_num)
            and bid_num > 0
            and ask_num > 0
            and ask_num >= bid_num
        ):
            return True
        last = getattr(ticker, "last", None)
        try:
            last_num = float(last) if last is not None else None
        except (TypeError, ValueError):
            last_num = None
        return bool(last_num is not None and not math.isnan(last_num) and last_num > 0)

    @staticmethod
    def _ticker_has_close_data(ticker: Ticker | None) -> bool:
        if ticker is None:
            return False
        for attr in ("close", "prevLast"):
            value = getattr(ticker, attr, None)
            try:
                number = float(value) if value is not None else None
            except (TypeError, ValueError):
                number = None
            if number is None or math.isnan(number) or number <= 0:
                continue
            return True
        return False

    @staticmethod
    def _quote_num(value: object) -> float | None:
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        if number is None or math.isnan(number) or number <= 0:
            return None
        return float(number)

    @staticmethod
    def _quote_signature_num(value: object, *, allow_zero: bool = True) -> float | None:
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        if number is None or math.isnan(number):
            return None
        if allow_zero and number < 0:
            return None
        if not allow_zero and number <= 0:
            return None
        return float(number)

    @classmethod
    def _ticker_quote_signature(cls, ticker: Ticker | None) -> tuple[float | int | None, ...] | None:
        if ticker is None:
            return None
        md_type_raw = getattr(ticker, "marketDataType", None)
        try:
            md_type = int(md_type_raw) if md_type_raw is not None else None
        except (TypeError, ValueError):
            md_type = None
        return (
            cls._quote_num(getattr(ticker, "bid", None)),
            cls._quote_num(getattr(ticker, "ask", None)),
            cls._quote_num(getattr(ticker, "last", None)),
            cls._quote_num(getattr(ticker, "close", None)),
            cls._quote_num(getattr(ticker, "prevLast", None)),
            cls._quote_signature_num(getattr(ticker, "bidSize", None)),
            cls._quote_signature_num(getattr(ticker, "askSize", None)),
            cls._quote_signature_num(getattr(ticker, "lastSize", None)),
            cls._quote_signature_num(getattr(ticker, "rtTradeVolume", None)),
            cls._quote_signature_num(getattr(ticker, "rtVolume", None)),
            cls._quote_signature_num(getattr(ticker, "volume", None)),
            md_type,
        )

    @classmethod
    def _ticker_top_quote_signature(cls, ticker: Ticker | None) -> tuple[float | None, ...] | None:
        if ticker is None:
            return None
        return (
            cls._quote_num(getattr(ticker, "bid", None)),
            cls._quote_num(getattr(ticker, "ask", None)),
            cls._quote_num(getattr(ticker, "last", None)),
        )

    @staticmethod
    def _ticker_quote_age_sec(ticker: Ticker | None) -> float | None:
        if ticker is None:
            return None
        updated_mono = getattr(ticker, "tbQuoteUpdatedMono", None)
        try:
            age_sec = (
                max(0.0, time.monotonic() - float(updated_mono))
                if updated_mono is not None
                else None
            )
        except (TypeError, ValueError):
            age_sec = None
        return age_sec

    @staticmethod
    def _ticker_top_quote_age_sec(ticker: Ticker | None) -> float | None:
        if ticker is None:
            return None
        updated_mono = getattr(ticker, "tbTopQuoteUpdatedMono", None)
        try:
            age_sec = (
                max(0.0, time.monotonic() - float(updated_mono))
                if updated_mono is not None
                else None
            )
        except (TypeError, ValueError):
            age_sec = None
        return age_sec

    @staticmethod
    def _looks_like_ticker(value: object) -> bool:
        return bool(
            value is not None
            and hasattr(value, "contract")
            and any(
                hasattr(value, attr)
                for attr in ("bid", "ask", "last", "close", "prevLast", "marketDataType")
            )
        )

    @classmethod
    def _event_tickers(cls, *args, **kwargs) -> list[Ticker]:
        seen: set[int] = set()
        out: list[Ticker] = []

        def _maybe_add(candidate: object) -> None:
            if not cls._looks_like_ticker(candidate):
                return
            key = id(candidate)
            if key in seen:
                return
            seen.add(key)
            out.append(candidate)  # type: ignore[arg-type]

        def _consume(value: object) -> None:
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _maybe_add(item)
                return
            _maybe_add(value)

        for arg in args:
            _consume(arg)
        for value in kwargs.values():
            _consume(value)
        return out

    def _sync_top_quote_meta_for_ticker(self, ticker: Ticker) -> bool:
        signature = self._ticker_top_quote_signature(ticker)
        if signature is None:
            return False
        previous = getattr(ticker, "tbTopQuoteSignature", None)
        if previous == signature:
            return False
        setattr(ticker, "tbTopQuoteSignature", signature)
        if not any(value is not None for value in signature):
            return False
        setattr(ticker, "tbTopQuoteUpdatedMono", time.monotonic())
        previous_moves = getattr(ticker, "tbTopQuoteMoveCount", None)
        try:
            moves = int(previous_moves) if previous_moves is not None else 0
        except (TypeError, ValueError):
            moves = 0
        setattr(ticker, "tbTopQuoteMoveCount", max(0, moves) + 1)
        return True

    def _sync_stream_quote_meta_for_ticker(self, ticker: Ticker) -> bool:
        signature = self._ticker_quote_signature(ticker)
        if signature is None:
            return False
        self._sync_top_quote_meta_for_ticker(ticker)
        previous = getattr(ticker, "tbQuoteSignature", None)
        if previous == signature:
            return False
        setattr(ticker, "tbQuoteSignature", signature)
        if self._ticker_has_data(ticker):
            self._tag_ticker_quote_meta(ticker, source="stream")
            return True
        if self._ticker_has_close_data(ticker):
            self._tag_ticker_quote_meta(ticker, source="stream-close-only")
            return True
        return False

    def _tag_ticker_quote_meta(
        self,
        ticker: Ticker,
        *,
        source: str,
        error_code: int | None = None,
        error_text: str | None = None,
        as_of: datetime | None = None,
    ) -> None:
        self._sync_top_quote_meta_for_ticker(ticker)
        setattr(ticker, "tbQuoteSource", str(source or "").strip())
        setattr(ticker, "tbQuoteUpdatedMono", time.monotonic())
        if as_of is not None:
            setattr(ticker, "tbQuoteAsOf", as_of.isoformat())
        elif hasattr(ticker, "tbQuoteAsOf"):
            setattr(ticker, "tbQuoteAsOf", None)
        if error_code is not None:
            setattr(ticker, "tbQuoteErrorCode", int(error_code))
        elif hasattr(ticker, "tbQuoteErrorCode"):
            setattr(ticker, "tbQuoteErrorCode", None)
        if error_text:
            setattr(ticker, "tbQuoteError", str(error_text).strip())
        elif hasattr(ticker, "tbQuoteError"):
            setattr(ticker, "tbQuoteError", None)

    def _apply_ticker_fallback_quote(
        self,
        *,
        ticker: Ticker,
        source: str,
        md_type: int | None = None,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
        close: float | None = None,
        prev_last: float | None = None,
        as_of: datetime | None = None,
    ) -> bool:
        applied = False
        clean_bid = self._quote_num(bid)
        clean_ask = self._quote_num(ask)
        clean_last = self._quote_num(last)
        clean_close = self._quote_num(close)
        clean_prev_last = self._quote_num(prev_last)
        if clean_bid is not None:
            ticker.bid = clean_bid
            applied = True
        if clean_ask is not None:
            ticker.ask = clean_ask
            applied = True
        if clean_last is not None:
            ticker.last = clean_last
            applied = True
        if clean_close is not None:
            ticker.close = clean_close
            applied = True
        if clean_prev_last is not None:
            ticker.prevLast = clean_prev_last
            applied = True
        if md_type is not None:
            ticker.marketDataType = int(md_type)
        if applied:
            self._tag_ticker_quote_meta(ticker, source=source, as_of=as_of)
            self._on_stream_update()
        return applied

    def _start_main_contract_quote_probe(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id:
            return
        self._start_main_contract_quote_watchdog(contract)
        existing = self._main_contract_probe_tasks.get(con_id)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._probe_main_contract_quote(contract))
        self._main_contract_probe_tasks[con_id] = task

        def _cleanup(done_task: asyncio.Task, key: int = con_id) -> None:
            current = self._main_contract_probe_tasks.get(key)
            if current is done_task:
                self._main_contract_probe_tasks.pop(key, None)

        task.add_done_callback(_cleanup)

    def _start_main_contract_quote_watchdog(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id:
            return
        existing = self._main_contract_watchdog_tasks.get(con_id)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._watch_main_contract_quote(contract))
        self._main_contract_watchdog_tasks[con_id] = task

        def _cleanup(done_task: asyncio.Task, key: int = con_id) -> None:
            current = self._main_contract_watchdog_tasks.get(key)
            if current is done_task:
                self._main_contract_watchdog_tasks.pop(key, None)

        task.add_done_callback(_cleanup)

    def _resubscribe_main_contract_stream(
        self,
        ticker: Ticker,
        *,
        md_type_override: int | None = None,
    ) -> Ticker | None:
        if not self._ib.isConnected():
            return None
        contract = getattr(ticker, "contract", None)
        if contract is None:
            return None
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type not in ("FUT", "FOP"):
            return None
        req_contract = self._normalize_derivative_market_data_contract(contract, sec_type=sec_type)
        md_type = int(md_type_override) if md_type_override is not None else int(_futures_md_ladder(_now_et())[0])
        try:
            self._ib.reqMarketDataType(int(md_type))
        except Exception:
            pass
        try:
            self._ib.cancelMktData(contract)
        except Exception:
            pass
        try:
            refreshed = self._ib.reqMktData(req_contract)
        except Exception:
            return None
        con_id = int(getattr(req_contract, "conId", 0) or 0)
        if con_id:
            self._detail_tickers[con_id] = (self._ib, refreshed)
        return refreshed

    async def _watch_main_contract_quote(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        last_probe_mono = 0.0
        last_resubscribe_mono = 0.0
        try:
            await asyncio.sleep(float(_MAIN_CONTRACT_QUOTE_PROBE_INITIAL_SEC))
            while True:
                if not con_id:
                    return
                entry = self._detail_tickers.get(con_id)
                if not entry:
                    return
                ib, ticker = entry
                if ib is not self._ib:
                    return
                now_mono = time.monotonic()
                age_sec = self._ticker_quote_age_sec(ticker)
                top_age_sec = self._ticker_top_quote_age_sec(ticker)
                has_actionable = self._ticker_has_data(ticker)
                has_close = self._ticker_has_close_data(ticker)
                stale_needs_probe = (
                    (not has_actionable and not has_close)
                    or age_sec is None
                    or age_sec >= float(_MAIN_CONTRACT_STALE_REPROBE_SEC)
                    or top_age_sec is None
                    or top_age_sec >= float(_MAIN_CONTRACT_TOPLINE_STALE_REPROBE_SEC)
                )
                if stale_needs_probe and (
                    now_mono - last_probe_mono
                ) >= float(_MAIN_CONTRACT_QUOTE_PROBE_RETRY_SEC):
                    req_contract = getattr(ticker, "contract", None) or contract
                    self._start_main_contract_quote_probe(req_contract)
                    last_probe_mono = now_mono
                md_type_raw = getattr(ticker, "marketDataType", None)
                try:
                    md_type = int(md_type_raw) if md_type_raw is not None else None
                except (TypeError, ValueError):
                    md_type = None
                promote_to_frozen = bool(
                    top_age_sec is not None
                    and top_age_sec >= float(_MAIN_CONTRACT_TOPLINE_STALE_REPROBE_SEC)
                    and md_type == 3
                )
                promote_live_frozen_to_live = bool(
                    md_type == 2
                    and _futures_session_is_open(_now_et())
                    and (now_mono - last_resubscribe_mono)
                    >= float(_MAIN_CONTRACT_STALE_RESUBSCRIBE_SEC)
                )
                resubscribe_due = bool(
                    (age_sec is not None and age_sec >= float(_MAIN_CONTRACT_STALE_RESUBSCRIBE_SEC))
                    or promote_to_frozen
                    or promote_live_frozen_to_live
                )
                if (
                    resubscribe_due
                    and (now_mono - last_resubscribe_mono) >= float(_MAIN_CONTRACT_STALE_REPROBE_SEC)
                ):
                    md_type_override: int | None = None
                    if promote_to_frozen:
                        md_type_override = 4
                    elif promote_live_frozen_to_live:
                        md_type_override = 1
                    refreshed = self._resubscribe_main_contract_stream(
                        ticker,
                        md_type_override=md_type_override,
                    )
                    if refreshed is not None:
                        if self._ticker_has_data(refreshed):
                            self._tag_ticker_quote_meta(refreshed, source="stream")
                        elif self._ticker_has_close_data(refreshed):
                            self._tag_ticker_quote_meta(refreshed, source="stream-close-only")
                        self._on_stream_update()
                    last_resubscribe_mono = now_mono
                await asyncio.sleep(float(_MAIN_CONTRACT_STALE_WATCHDOG_SEC))
        finally:
            current = self._main_contract_watchdog_tasks.get(con_id)
            if current is asyncio.current_task():
                self._main_contract_watchdog_tasks.pop(con_id, None)

    async def _probe_main_contract_quote(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        try:
            await asyncio.sleep(float(_MAIN_CONTRACT_QUOTE_PROBE_INITIAL_SEC))
            if not con_id:
                return
            entry = self._detail_tickers.get(con_id)
            if not entry:
                return
            ib, ticker = entry
            if ib is not self._ib:
                return
            if self._ticker_has_data(ticker):
                self._tag_ticker_quote_meta(ticker, source="stream")
                return
            if self._ticker_has_close_data(ticker):
                self._tag_ticker_quote_meta(ticker, source="stream-close-only")
            if await self._attempt_main_contract_snapshot_quote(contract, ticker=ticker):
                return
            if await self._attempt_main_contract_historical_quote(contract, ticker=ticker):
                return
            await asyncio.sleep(float(_MAIN_CONTRACT_QUOTE_PROBE_RETRY_SEC))
            entry = self._detail_tickers.get(con_id)
            if not entry:
                return
            ib, ticker = entry
            if ib is not self._ib:
                return
            if self._ticker_has_data(ticker):
                self._tag_ticker_quote_meta(ticker, source="stream")
                return
            if self._ticker_has_close_data(ticker):
                self._tag_ticker_quote_meta(ticker, source="stream-close-only")
            if await self._attempt_main_contract_snapshot_quote(contract, ticker=ticker):
                return
            await self._attempt_main_contract_historical_quote(contract, ticker=ticker)
        finally:
            current = self._main_contract_probe_tasks.get(con_id)
            if current is asyncio.current_task():
                self._main_contract_probe_tasks.pop(con_id, None)

    async def _attempt_main_contract_snapshot_quote(
        self,
        contract: Contract,
        *,
        ticker: Ticker,
        md_types: Iterable[int] | None = None,
    ) -> bool:
        req_contract = contract
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type in ("FUT", "OPT", "FOP"):
            req_contract = self._normalize_derivative_market_data_contract(
                contract,
                sec_type=sec_type,
            )

        ladder: list[int] = []
        md_candidates: Iterable[int] = (
            _futures_md_ladder(_now_et()) if md_types is None else md_types
        )
        for raw_md_type in md_candidates:
            try:
                md_type = int(raw_md_type)
            except (TypeError, ValueError):
                continue
            if md_type <= 0:
                continue
            ladder.append(md_type)
        if not ladder:
            ladder = list(_futures_md_ladder(_now_et()))

        for md_type in ladder:
            try:
                self._ib.reqMarketDataType(int(md_type))
                snap = self._ib.reqMktData(req_contract, "", True, False)
            except Exception:
                continue
            await asyncio.sleep(float(_MAIN_CONTRACT_SNAPSHOT_WAIT_SEC))
            bid = self._quote_num(getattr(snap, "bid", None))
            ask = self._quote_num(getattr(snap, "ask", None))
            last = self._quote_num(getattr(snap, "last", None))
            close = self._quote_num(getattr(snap, "close", None))
            prev_last = self._quote_num(getattr(snap, "prevLast", None))
            has_actionable = bool(
                (bid is not None and ask is not None and ask >= bid)
                or last is not None
            )
            has_close = bool(close is not None or prev_last is not None)
            if not has_actionable and not has_close:
                continue
            source = {
                1: "live-snapshot",
                2: "live-frozen-snapshot",
                3: "delayed-snapshot",
                4: "delayed-frozen-snapshot",
            }.get(int(md_type), f"snapshot-md{int(md_type)}")
            applied = self._apply_ticker_fallback_quote(
                ticker=ticker,
                source=source,
                md_type=int(md_type),
                bid=bid,
                ask=ask,
                last=last,
                close=close,
                prev_last=prev_last,
            )
            if applied:
                return True
        return False

    async def _attempt_main_contract_historical_quote(self, contract: Contract, *, ticker: Ticker) -> bool:
        req_contract = contract
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type in ("FUT", "OPT", "FOP"):
            req_contract = self._normalize_derivative_market_data_contract(
                contract,
                sec_type=sec_type,
            )

        async def _fetch(
            *,
            duration_str: str,
            bar_size: str,
            use_rth: bool,
            what_to_show: str,
            cache_ttl_sec: float,
        ) -> list[tuple[datetime, float]]:
            try:
                return await asyncio.wait_for(
                    self.historical_bars(
                        req_contract,
                        duration_str=duration_str,
                        bar_size=bar_size,
                        use_rth=use_rth,
                        what_to_show=what_to_show,
                        cache_ttl_sec=cache_ttl_sec,
                    ),
                    timeout=float(_MAIN_CONTRACT_HISTORICAL_ATTEMPT_TIMEOUT_SEC),
                )
            except asyncio.TimeoutError:
                return []
            except Exception:
                return []

        for what_to_show in ("TRADES", "MIDPOINT", "BID_ASK"):
            bars = await _fetch(
                duration_str="10800 S",
                bar_size="1 min",
                use_rth=False,
                what_to_show=what_to_show,
                cache_ttl_sec=20.0,
            )
            if not bars:
                continue
            ts, close = bars[-1]
            if close is None or float(close) <= 0:
                continue
            source = f"historical-{what_to_show.lower()}"
            if self._apply_ticker_fallback_quote(
                ticker=ticker,
                source=source,
                md_type=3 if _futures_session_is_open(_now_et()) else 4,
                last=float(close),
                close=float(close),
                as_of=ts,
            ):
                return True
        if sec_type in ("FUT", "OPT", "FOP"):
            for what_to_show in ("TRADES", "MIDPOINT"):
                bars = await _fetch(
                    duration_str="2 M",
                    bar_size="1 day",
                    use_rth=False,
                    what_to_show=what_to_show,
                    cache_ttl_sec=120.0,
                )
                if not bars:
                    continue
                ts, close = bars[-1]
                if close is None or float(close) <= 0:
                    continue
                source = f"historical-daily-{what_to_show.lower()}"
                if self._apply_ticker_fallback_quote(
                    ticker=ticker,
                    source=source,
                    md_type=3 if _futures_session_is_open(_now_et()) else 4,
                    last=float(close),
                    close=float(close),
                    as_of=ts,
                ):
                    return True
        if not self._ticker_has_data(ticker) and not self._ticker_has_close_data(ticker):
            self._tag_ticker_quote_meta(
                ticker,
                source="unavailable",
                error_text=str(getattr(ticker, "tbQuoteError", "") or "").strip(),
            )
            self._on_stream_update()
        return False

    def _start_proxy_contract_quote_probe(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id or con_id in self._proxy_contract_force_delayed:
            return
        existing = self._proxy_contract_probe_tasks.get(con_id)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._probe_proxy_contract_quote(contract))
        self._proxy_contract_probe_tasks[con_id] = task

    async def _probe_proxy_contract_quote(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        try:
            await asyncio.sleep(float(_PROXY_CONTRACT_QUOTE_PROBE_INITIAL_SEC))
            if not con_id or con_id in self._proxy_contract_force_delayed:
                return
            entry = self._detail_tickers.get(con_id)
            if not entry:
                return
            ib, ticker = entry
            if ib is not self._ib_proxy:
                return
            if self._ticker_has_data(ticker):
                return
            self._start_proxy_contract_live_resubscribe(contract)
            await asyncio.sleep(float(_PROXY_CONTRACT_QUOTE_PROBE_RETRY_SEC))
            if not con_id or con_id in self._proxy_contract_force_delayed:
                return
            entry = self._detail_tickers.get(con_id)
            if not entry:
                return
            ib, ticker = entry
            if ib is not self._ib_proxy:
                return
            if self._ticker_has_data(ticker):
                return
            if contract.secType == "OPT":
                self._start_proxy_contract_live_resubscribe(contract)
        finally:
            current = self._proxy_contract_probe_tasks.get(con_id)
            if current is asyncio.current_task():
                self._proxy_contract_probe_tasks.pop(con_id, None)

    async def _resubscribe_proxy_contract_live(self, contract: Contract) -> None:
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception:
                return
            self._ib_proxy.reqMarketDataType(1)
            req_contract = contract
            if contract.secType == "STK":
                _, include_overnight = _session_flags(_now_et())
                req_contract = self._stock_market_data_contract(
                    contract,
                    include_overnight=include_overnight,
                    delayed=False,
                )
            elif contract.secType in ("OPT", "FOP"):
                if not contract.exchange:
                    req_contract = copy.copy(contract)
                    if contract.secType == "FOP":
                        primary_exchange = getattr(contract, "primaryExchange", "") or ""
                        req_contract.exchange = primary_exchange or "CME"
                    else:
                        req_contract.exchange = "SMART"
            try:
                self._ib_proxy.cancelMktData(contract)
            except Exception:
                pass
            ticker = self._ib_proxy.reqMktData(req_contract)
            con_id = int(getattr(req_contract, "conId", 0) or 0)
            if con_id and con_id in self._detail_tickers:
                self._detail_tickers[con_id] = (self._ib_proxy, ticker)
            symbol = getattr(req_contract, "symbol", "") or ""
            if symbol and symbol in self._proxy_tickers:
                self._proxy_tickers[symbol] = ticker

    async def _resubscribe_proxy_contract_delayed(self, contract: Contract) -> None:
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception:
                return
            self._ib_proxy.reqMarketDataType(3)
            req_contract = contract
            if contract.secType == "STK":
                _, include_overnight = _session_flags(_now_et())
                req_contract = self._stock_market_data_contract(
                    contract,
                    include_overnight=include_overnight,
                    delayed=True,
                )
                if str(getattr(req_contract, "exchange", "") or "").strip().upper() == "OVERNIGHT":
                    primary_exchange = str(
                        getattr(contract, "primaryExchange", "")
                        or getattr(req_contract, "primaryExchange", "")
                        or ""
                    ).strip().upper()
                    if primary_exchange:
                        req_contract = copy.copy(contract)
                        req_contract.exchange = primary_exchange
            elif contract.secType in ("OPT", "FOP"):
                if not contract.exchange:
                    req_contract = copy.copy(contract)
                    if contract.secType == "FOP":
                        primary_exchange = getattr(contract, "primaryExchange", "") or ""
                        req_contract.exchange = primary_exchange or "CME"
                    else:
                        req_contract.exchange = "SMART"
            try:
                self._ib_proxy.cancelMktData(contract)
            except Exception:
                pass
            ticker = self._ib_proxy.reqMktData(req_contract)
            con_id = int(getattr(req_contract, "conId", 0) or 0)
            if con_id and con_id in self._detail_tickers:
                self._detail_tickers[con_id] = (self._ib_proxy, ticker)
            symbol = getattr(req_contract, "symbol", "") or ""
            if symbol and symbol in self._proxy_tickers:
                self._proxy_tickers[symbol] = ticker

    def _maybe_reset_proxy_contract_delay_on_session_change(self) -> None:
        now = _now_et()
        current_bucket = _session_bucket(now)
        _, include_overnight = _session_flags(now)
        current_include_overnight = bool(include_overnight)
        if self._proxy_session_bucket is None:
            self._proxy_session_bucket = current_bucket
            self._proxy_session_include_overnight = current_include_overnight
            return
        if current_bucket == self._proxy_session_bucket:
            return
        previous_include_overnight = self._proxy_session_include_overnight
        self._proxy_session_bucket = current_bucket
        self._proxy_session_include_overnight = current_include_overnight
        overnight_route_changed = (
            previous_include_overnight is not None
            and bool(previous_include_overnight) != current_include_overnight
        )
        if self._proxy_contract_force_delayed:
            self._proxy_contract_force_delayed = set()
        for _con_id, (ib, ticker) in list(self._detail_tickers.items()):
            if ib is not self._ib_proxy:
                continue
            contract = getattr(ticker, "contract", None)
            if contract is None:
                continue
            sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
            if sec_type not in ("OPT", "FOP"):
                continue
            self._start_proxy_contract_live_resubscribe(contract)
            self._start_proxy_contract_quote_probe(contract)
        if overnight_route_changed and self._proxy_tickers:
            self._start_proxy_resubscribe()

    def _handle_conn_error(self, error_code: int) -> None:
        if error_code == 1100:
            self._farm_connectivity_lost = True
        elif error_code in (1101, 1102):
            self._farm_connectivity_lost = False

    def _start_proxy_resubscribe(self) -> None:
        if self._proxy_task and not self._proxy_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proxy_task = loop.create_task(self._reload_proxy_tickers())

    def _start_proxy_probe(self) -> None:
        if self._proxy_force_delayed:
            return
        if self._proxy_probe_task and not self._proxy_probe_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proxy_probe_task = loop.create_task(self._probe_proxy_quotes())

    async def _probe_proxy_quotes(self) -> None:
        await asyncio.sleep(2)
        if self._proxy_force_delayed or not self._proxy_tickers:
            return
        if self._proxy_has_data():
            return
        self._proxy_force_delayed = True
        self._start_proxy_resubscribe()

    def _proxy_has_data(self) -> bool:
        for ticker in self._proxy_tickers.values():
            if self._ticker_has_data(ticker):
                return True
        return False

    async def _reload_proxy_tickers(self) -> None:
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception as exc:
                self._proxy_error = str(exc)
                self._proxy_tickers = {}
                self._proxy_task = None
                return
            md_type = 3 if self._proxy_force_delayed else 1
            self._ib_proxy.reqMarketDataType(md_type)
            _, include_overnight = _session_flags(_now_et())
            self._proxy_session_include_overnight = include_overnight
            for ticker in self._proxy_tickers.values():
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._proxy_tickers = {}
            for con_id, (ib, ticker) in list(self._detail_tickers.items()):
                if ib is not self._ib_proxy:
                    continue
                req_contract = ticker.contract
                if req_contract.secType == "STK":
                    req_contract = self._stock_market_data_contract(
                        req_contract,
                        include_overnight=include_overnight,
                        delayed=bool(md_type == 3),
                    )
                elif req_contract.secType in ("OPT", "FOP") and not req_contract.exchange:
                    req_contract = copy.copy(req_contract)
                    if req_contract.secType == "FOP":
                        primary_exchange = getattr(req_contract, "primaryExchange", "") or ""
                        req_contract.exchange = primary_exchange or "CME"
                    else:
                        req_contract.exchange = "SMART"
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
                self._detail_tickers[con_id] = (
                    self._ib_proxy,
                    self._ib_proxy.reqMktData(req_contract),
                )
            await self._ensure_proxy_tickers()

    def _maybe_resubscribe_index_on_session_transition(self) -> None:
        now = _now_et()
        futures_open = bool(_futures_session_is_open(now))
        previous = self._index_futures_session_open
        self._index_futures_session_open = futures_open
        if previous is None or previous == futures_open:
            return
        if self._index_force_delayed and self._index_tickers:
            self._start_index_resubscribe(requalify=False)

    def _on_stream_update(self, *_, **__) -> None:
        self._maybe_resubscribe_index_on_session_transition()
        self._maybe_reset_proxy_contract_delay_on_session_change()
        event_tickers = self._event_tickers(*_, **__)
        for ticker in event_tickers:
            self._sync_stream_quote_meta_for_ticker(ticker)
        if self._update_callback:
            self._update_callback()
        if not self._stream_listeners:
            return
        for callback in tuple(self._stream_listeners):
            try:
                callback()
            except Exception:
                continue

    def _on_pnl_single(self, value: PnLSingle) -> None:
        if self._config.account and value.account and value.account != self._config.account:
            return
        try:
            con_id = int(getattr(value, "conId", 0) or 0)
        except (TypeError, ValueError):
            con_id = 0
        if con_id > 0:
            self._pnl_single_by_con_id[con_id] = value
        self._on_stream_update()

    def _on_account_value(self, value: AccountValue) -> None:
        if self._config.account and value.account != self._config.account:
            return
        parsed = self._clean_pnl_stream_value(getattr(value, "value", None))
        if parsed is not None:
            key = (value.tag, value.currency)
            self._account_value_cache[key] = (parsed, datetime.now(timezone.utc))
        if self._update_callback:
            self._update_callback()

    def _start_reconnect_loop(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._reconnect_task = loop.create_task(self._reconnect_until_deadline())

    def _stop_reconnect_loop(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

    async def _reconnect_until_deadline(self) -> None:
        fast_deadline = self._reconnect_fast_deadline
        if fast_deadline is None:
            fast_deadline = time.monotonic() + float(self._config.reconnect_timeout_sec)
            self._reconnect_fast_deadline = fast_deadline
        fast_interval = max(0.5, float(self._config.reconnect_interval_sec))
        slow_interval = max(fast_interval, float(self._config.reconnect_slow_interval_sec))
        slow_notified = False
        while self._reconnect_requested:
            if not slow_notified and time.monotonic() >= fast_deadline:
                slow_notified = True
                if self._update_callback:
                    self._update_callback()
            await self._reconnect_once()
            if not self._reconnect_requested:
                break
            interval = fast_interval if time.monotonic() < fast_deadline else slow_interval
            cooldown = self._client_id_backoff_remaining_sec()
            if cooldown > interval:
                interval = cooldown
            await asyncio.sleep(interval)

    async def _reconnect_once(self) -> None:
        async with self._lock:
            if not self._ib.isConnected():
                try:
                    await self.connect()
                except Exception:
                    return
                self._resubscribe_main_needed = True
            if self._resubscribe_main_needed and self._ib.isConnected():
                self._account_updates_started = False
                await self._ensure_account_updates()
                for con_id, (ib, ticker) in list(self._detail_tickers.items()):
                    if ib is not self._ib:
                        continue
                    try:
                        req_contract = ticker.contract
                        sec_type = str(getattr(req_contract, "secType", "") or "").strip().upper()
                        if sec_type in ("FUT", "OPT", "FOP"):
                            req_contract = self._normalize_derivative_market_data_contract(
                                req_contract,
                                sec_type=sec_type,
                            )
                        if sec_type in ("FUT", "FOP"):
                            self._ib.reqMarketDataType(int(_futures_md_ladder(_now_et())[0]))
                        else:
                            self._ib.reqMarketDataType(3)
                        refreshed = self._ib.reqMktData(req_contract)
                        self._detail_tickers[con_id] = (
                            self._ib,
                            refreshed,
                        )
                        if sec_type in ("FUT", "FOP"):
                            self._start_main_contract_quote_watchdog(req_contract)
                            if self._ticker_has_data(refreshed):
                                self._tag_ticker_quote_meta(refreshed, source="stream")
                            elif self._ticker_has_close_data(refreshed):
                                self._tag_ticker_quote_meta(refreshed, source="stream-close-only")
                            else:
                                self._start_main_contract_quote_probe(req_contract)
                    except Exception:
                        continue
                self._index_tickers = {}
                self._index_task = None
                if self._index_probe_task and not self._index_probe_task.done():
                    self._index_probe_task.cancel()
                self._index_probe_task = None
                await self._ensure_index_tickers()
                self._start_index_probe()
                self._resubscribe_main_needed = False
        async with self._proxy_lock:
            if not self._ib_proxy.isConnected():
                try:
                    await self.connect_proxy()
                except Exception as exc:
                    self._proxy_error = str(exc)
                    return
                self._resubscribe_proxy_needed = True
            if self._resubscribe_proxy_needed and self._ib_proxy.isConnected():
                md_type = 3 if self._proxy_force_delayed else 1
                self._ib_proxy.reqMarketDataType(md_type)
                _, include_overnight = _session_flags(_now_et())
                self._proxy_session_include_overnight = include_overnight
                for con_id, (ib, ticker) in list(self._detail_tickers.items()):
                    if ib is not self._ib_proxy:
                        continue
                    req_contract = ticker.contract
                    if req_contract.secType == "STK":
                        req_contract = self._stock_market_data_contract(
                            req_contract,
                            include_overnight=include_overnight,
                            delayed=bool(md_type == 3),
                        )
                    elif req_contract.secType in ("OPT", "FOP") and not req_contract.exchange:
                        req_contract = copy.copy(req_contract)
                        if req_contract.secType == "FOP":
                            primary_exchange = getattr(req_contract, "primaryExchange", "") or ""
                            req_contract.exchange = primary_exchange or "CME"
                        else:
                            req_contract.exchange = "SMART"
                    try:
                        self._detail_tickers[con_id] = (
                            self._ib_proxy,
                            self._ib_proxy.reqMktData(req_contract),
                        )
                    except Exception:
                        continue
                self._proxy_tickers = {}
                self._proxy_task = None
                await self._ensure_proxy_tickers()
                self._proxy_probe_task = None
                self._start_proxy_probe()
                self._resubscribe_proxy_needed = False
        if (
            self._ib.isConnected()
            and self._ib_proxy.isConnected()
            and not self._resubscribe_main_needed
            and not self._resubscribe_proxy_needed
        ):
            try:
                await self.reconcile_order_state(force=True)
            except Exception:
                pass
            self._reconnect_requested = False
            self._reconnect_fast_deadline = None
            if self._update_callback:
                self._update_callback()
# endregion


# region Helpers
def _pick_account_value(values: list[AccountValue]) -> AccountValue | None:
    for currency in ("BASE", "USD", "AUD"):
        for value in values:
            if value.currency == currency:
                return value
    return values[0] if values else None


def _pick_cached_value(
    cache: dict[tuple[str, str], tuple[float, datetime]], tag: str
) -> tuple[float, str, datetime] | None:
    for currency in ("BASE", "USD", "AUD"):
        key = (tag, currency)
        if key in cache:
            value, updated = cache[key]
            return value, currency, updated
    for (cached_tag, currency), (value, updated) in cache.items():
        if cached_tag == tag:
            return value, currency, updated
    return None


def _normalize_order_contract(contract: Contract) -> Contract:
    if contract.exchange:
        return contract
    normalized = copy.copy(contract)
    sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
    if sec_type == "FUT":
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        exchange_hints = _FUT_EXCHANGE_HINTS.get(symbol, ())
        primary_exchange = str(getattr(contract, "primaryExchange", "") or "").strip().upper()
        normalized.exchange = primary_exchange or (exchange_hints[0] if exchange_hints else "CME")
        return normalized
    if sec_type == "FOP":
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        exchange_hints = _FUT_EXCHANGE_HINTS.get(symbol, ())
        primary_exchange = str(getattr(contract, "primaryExchange", "") or "").strip().upper()
        normalized.exchange = primary_exchange or (exchange_hints[0] if exchange_hints else "CME")
        return normalized
    if sec_type in ("STK", "OPT"):
        normalized.exchange = "SMART"
    return normalized
# endregion
