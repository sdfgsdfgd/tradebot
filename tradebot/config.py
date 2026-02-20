"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os

DEFAULT_REFRESH_SEC = 0.25
DEFAULT_DETAIL_REFRESH_SEC = 0.25
DEFAULT_RECONNECT_INTERVAL_SEC = 5.0
DEFAULT_RECONNECT_TIMEOUT_SEC = 240.0
DEFAULT_RECONNECT_SLOW_INTERVAL_SEC = 60.0
DEFAULT_CLIENT_ID_POOL_START = 500
DEFAULT_CLIENT_ID_POOL_END = 899
DEFAULT_CLIENT_ID_BURST_ATTEMPTS = 8
DEFAULT_CLIENT_ID_BACKOFF_INITIAL_SEC = 5.0
DEFAULT_CLIENT_ID_BACKOFF_MAX_SEC = 300.0
DEFAULT_CLIENT_ID_BACKOFF_MULTIPLIER = 2.0
DEFAULT_CONNECT_TIMEOUT_SEC = 15.0
DEFAULT_CLIENT_ID_QUARANTINE_SEC = 90.0


@dataclass(frozen=True)
class IBKRConfig:
    host: str
    port: int
    client_id: int
    proxy_client_id: int
    account: str | None
    refresh_sec: float
    detail_refresh_sec: float
    reconnect_interval_sec: float
    reconnect_timeout_sec: float
    reconnect_slow_interval_sec: float
    client_id_pool_start: int = DEFAULT_CLIENT_ID_POOL_START
    client_id_pool_end: int = DEFAULT_CLIENT_ID_POOL_END
    client_id_burst_attempts: int = DEFAULT_CLIENT_ID_BURST_ATTEMPTS
    client_id_backoff_initial_sec: float = DEFAULT_CLIENT_ID_BACKOFF_INITIAL_SEC
    client_id_backoff_max_sec: float = DEFAULT_CLIENT_ID_BACKOFF_MAX_SEC
    client_id_backoff_multiplier: float = DEFAULT_CLIENT_ID_BACKOFF_MULTIPLIER
    client_id_backoff_jitter_ratio: float = 0.15
    client_id_state_file: str = "${TMPDIR:-/tmp}/tradebot_ib_client_ids.json"
    connect_timeout_sec: float = DEFAULT_CONNECT_TIMEOUT_SEC
    client_id_quarantine_sec: float = DEFAULT_CLIENT_ID_QUARANTINE_SEC


def load_config() -> IBKRConfig:
    """Load config from environment with safe defaults for local IB Gateway."""
    pool_start = int(os.getenv("IBKR_CLIENT_ID_POOL_START", str(DEFAULT_CLIENT_ID_POOL_START)))
    pool_end = int(os.getenv("IBKR_CLIENT_ID_POOL_END", str(DEFAULT_CLIENT_ID_POOL_END)))
    if pool_end <= pool_start:
        pool_start = DEFAULT_CLIENT_ID_POOL_START
        pool_end = DEFAULT_CLIENT_ID_POOL_END
    client_id = int(os.getenv("IBKR_CLIENT_ID", str(pool_start)))
    proxy_client_id = int(os.getenv("IBKR_PROXY_CLIENT_ID", str(client_id + 1)))
    return IBKRConfig(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=int(os.getenv("IBKR_PORT", "4001")),
        client_id=client_id,
        proxy_client_id=proxy_client_id,
        account=os.getenv("IBKR_ACCOUNT") or None,
        refresh_sec=float(os.getenv("IBKR_REFRESH_SEC", DEFAULT_REFRESH_SEC)),
        detail_refresh_sec=float(
            os.getenv("IBKR_DETAIL_REFRESH_SEC", DEFAULT_DETAIL_REFRESH_SEC)
        ),
        reconnect_interval_sec=float(
            os.getenv("IBKR_RECONNECT_INTERVAL_SEC", DEFAULT_RECONNECT_INTERVAL_SEC)
        ),
        reconnect_timeout_sec=float(
            os.getenv("IBKR_RECONNECT_TIMEOUT_SEC", DEFAULT_RECONNECT_TIMEOUT_SEC)
        ),
        reconnect_slow_interval_sec=float(
            os.getenv(
                "IBKR_RECONNECT_SLOW_INTERVAL_SEC",
                DEFAULT_RECONNECT_SLOW_INTERVAL_SEC,
            )
        ),
        client_id_pool_start=pool_start,
        client_id_pool_end=pool_end,
        client_id_burst_attempts=max(
            1,
            int(os.getenv("IBKR_CLIENT_ID_BURST_ATTEMPTS", str(DEFAULT_CLIENT_ID_BURST_ATTEMPTS))),
        ),
        client_id_backoff_initial_sec=max(
            0.5,
            float(
                os.getenv(
                    "IBKR_CLIENT_ID_BACKOFF_INITIAL_SEC",
                    str(DEFAULT_CLIENT_ID_BACKOFF_INITIAL_SEC),
                )
            ),
        ),
        client_id_backoff_max_sec=max(
            1.0,
            float(
                os.getenv(
                    "IBKR_CLIENT_ID_BACKOFF_MAX_SEC",
                    str(DEFAULT_CLIENT_ID_BACKOFF_MAX_SEC),
                )
            ),
        ),
        client_id_backoff_multiplier=max(
            1.0,
            float(
                os.getenv(
                    "IBKR_CLIENT_ID_BACKOFF_MULTIPLIER",
                    str(DEFAULT_CLIENT_ID_BACKOFF_MULTIPLIER),
                )
            ),
        ),
        client_id_backoff_jitter_ratio=max(
            0.0,
            min(0.9, float(os.getenv("IBKR_CLIENT_ID_BACKOFF_JITTER_RATIO", "0.15"))),
        ),
        client_id_state_file=os.getenv(
            "IBKR_CLIENT_ID_STATE_FILE",
            "${TMPDIR:-/tmp}/tradebot_ib_client_ids.json",
        ),
        connect_timeout_sec=max(
            1.0,
            float(os.getenv("IBKR_CONNECT_TIMEOUT_SEC", str(DEFAULT_CONNECT_TIMEOUT_SEC))),
        ),
        client_id_quarantine_sec=max(
            1.0,
            float(
                os.getenv(
                    "IBKR_CLIENT_ID_QUARANTINE_SEC",
                    str(DEFAULT_CLIENT_ID_QUARANTINE_SEC),
                )
            ),
        ),
    )
