"""One-shot causal news research service and timestamp-correct observation."""

from .contract import (
    NewsError,
    NewsSignalObservation,
    load_news_history,
    observe_news_signal,
    select_news_snapshot_at,
)
from .pipeline import main, run_once, verify_published

__all__ = (
    "NewsError",
    "NewsSignalObservation",
    "load_news_history",
    "main",
    "observe_news_signal",
    "run_once",
    "select_news_snapshot_at",
    "verify_published",
)
