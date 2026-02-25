"""README string retrievers used by the Bot preset loader."""

from __future__ import annotations

import re

_CURRENT_HEADER_RE = re.compile(
    r"^###\s+CURRENT(?:\s+\(v(?P<ver>[^)]+)\))?",
    flags=re.MULTILINE | re.IGNORECASE,
)
_SLV_JSON_BACKTICK_RE = re.compile(r"`(?P<path>backtests/slv/[^`]+\.json)`")
_SLV_JSON_INLINE_RE = re.compile(r"(?P<path>backtests/slv/[^\s)]+\.json)")
_TQQQ_JSON_BACKTICK_RE = re.compile(r"`(?P<path>backtests/(?:out|tqqq)/[^`]+\.json)`")
_TQQQ_JSON_INLINE_RE = re.compile(r"(?P<path>backtests/(?:out|tqqq)/[^\s)]+\.json)")


def _extract_current(
    readme_text: str,
    *,
    backtick_re: re.Pattern[str],
    inline_re: re.Pattern[str],
) -> tuple[str | None, str | None]:
    head = _CURRENT_HEADER_RE.search(str(readme_text or ""))
    if head is None:
        return None, None

    version = str(head.group("ver") or "").strip() or None
    tail = readme_text[head.end() :]
    next_head = re.search(r"^###\s+", tail, flags=re.MULTILINE)
    section = tail[: next_head.start()] if next_head else tail

    path_match = backtick_re.search(section) or inline_re.search(section)
    path = path_match.group("path") if path_match else None
    return version, path


def extract_current_slv_lf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/slv/readme-lf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_SLV_JSON_BACKTICK_RE,
        inline_re=_SLV_JSON_INLINE_RE,
    )


def extract_current_slv_hf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/slv/readme-hf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_SLV_JSON_BACKTICK_RE,
        inline_re=_SLV_JSON_INLINE_RE,
    )


def extract_current_tqqq_lf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/tqqq/readme-lf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_TQQQ_JSON_BACKTICK_RE,
        inline_re=_TQQQ_JSON_INLINE_RE,
    )


def extract_current_tqqq_hf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/tqqq/readme-hf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_TQQQ_JSON_BACKTICK_RE,
        inline_re=_TQQQ_JSON_INLINE_RE,
    )
