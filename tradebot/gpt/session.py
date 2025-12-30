"""Reserved GPT session hooks.

Later we will inject live options pricing and position context into
multi-round conversations and tool calls.
"""
from __future__ import annotations


def build_context_payload() -> dict:
    """Placeholder for GPT context injection (options prices, positions, signals)."""
    return {}
