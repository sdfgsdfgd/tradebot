"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os

DEFAULT_REFRESH_SEC = 0.25
DEFAULT_DETAIL_REFRESH_SEC = 0.5
DEFAULT_RECONNECT_INTERVAL_SEC = 5.0
DEFAULT_RECONNECT_TIMEOUT_SEC = 240.0


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


def load_config() -> IBKRConfig:
    """Load config from environment with safe defaults for local IB Gateway."""
    client_id = int(os.getenv("IBKR_CLIENT_ID", "0"))
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
    )
