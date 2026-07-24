"""One-shot causal news research service."""

from .contract import NewsError
from .pipeline import main, run_once

__all__ = ("NewsError", "main", "run_once")
