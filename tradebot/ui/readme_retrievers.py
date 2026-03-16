"""README string retrievers used by the Bot preset loader."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HEADING_RE = re.compile(r"^(?P<hash>#{3,6})\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_CURRENT_TITLE_RE = re.compile(r"^CURRENT(?:\s|$|\()", flags=re.IGNORECASE)
_VERSION_RE = re.compile(r"\(v(?P<ver>[^)]+)\)", flags=re.IGNORECASE)
_PATH_VERSION_RE = re.compile(r"(?:^|[_/.-])v(?P<ver>\d+(?:\.\d+)?)\b", flags=re.IGNORECASE)
_PRESET_FILE_LINE_RE = re.compile(
    r"^-\s*Preset file(?:\s*\([^)]*\))?:\s*(?P<body>.+)$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_SLV_JSON_BACKTICK_RE = re.compile(r"`(?P<path>backtests/slv/[^`]+\.json)`")
_SLV_JSON_INLINE_RE = re.compile(r"(?P<path>backtests/slv/[^\s)]+\.json)")
_TQQQ_JSON_BACKTICK_RE = re.compile(r"`(?P<path>backtests/(?:out|tqqq)/[^`]+\.json)`")
_TQQQ_JSON_INLINE_RE = re.compile(r"(?P<path>backtests/(?:out|tqqq)/[^\s)]+\.json)")
_MNQ_JSON_BACKTICK_RE = re.compile(r"`(?P<path>backtests/mnq/[^`]+\.json)`")
_MNQ_JSON_INLINE_RE = re.compile(r"(?P<path>backtests/mnq/[^\s)]+\.json)")


def _version_sort_key(version: str | None) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", str(version or ""))]
    return tuple(parts) or (-1,)


def _section_for_heading(text: str, headings: list[re.Match[str]], idx: int) -> str:
    head = headings[idx]
    level = len(head.group("hash"))
    end = len(text)
    for nxt in headings[idx + 1 :]:
        if len(nxt.group("hash")) <= level:
            end = nxt.start()
            break
    return text[head.end() : end]


def _extract_version(title: str, path: str | None = None) -> str | None:
    match = _VERSION_RE.search(str(title or ""))
    if match is not None:
        version = str(match.group("ver") or "").strip()
        if version:
            return version
    path_match = _PATH_VERSION_RE.search(str(path or ""))
    if path_match is not None:
        version = str(path_match.group("ver") or "").strip()
        if version:
            return version
    return None


def _extract_path(
    section: str,
    *,
    backtick_re: re.Pattern[str],
    inline_re: re.Pattern[str],
) -> str | None:
    preset_line = _PRESET_FILE_LINE_RE.search(section)
    if preset_line is not None:
        body = str(preset_line.group("body") or "")
        match = backtick_re.search(body) or inline_re.search(body)
        if match is not None:
            return str(match.group("path") or "").strip() or None
    match = backtick_re.search(section) or inline_re.search(section)
    if match is None:
        return None
    return str(match.group("path") or "").strip() or None


def _extract_current_candidates(
    readme_text: str,
    *,
    backtick_re: re.Pattern[str],
    inline_re: re.Pattern[str],
) -> list[tuple[str | None, str | None]]:
    text = str(readme_text or "")
    headings = list(_HEADING_RE.finditer(text))
    candidates: list[tuple[tuple[int, ...], int, str | None, str | None]] = []
    for idx, head in enumerate(headings):
        title = str(head.group("title") or "").strip()
        if not _CURRENT_TITLE_RE.match(title):
            continue
        section = _section_for_heading(text, headings, idx)
        path = _extract_path(section, backtick_re=backtick_re, inline_re=inline_re)
        if not path:
            continue
        version = _extract_version(title, path)
        candidates.append((_version_sort_key(version), head.start(), version, path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(version, path) for _sort_key, _start, version, path in candidates]


def _extract_current(
    readme_text: str,
    *,
    backtick_re: re.Pattern[str],
    inline_re: re.Pattern[str],
) -> tuple[str | None, str | None]:
    candidates = _extract_current_candidates(
        readme_text,
        backtick_re=backtick_re,
        inline_re=inline_re,
    )
    for version, path in candidates:
        if path and (_REPO_ROOT / path).exists():
            return version, path
    return candidates[0] if candidates else (None, None)


def extract_current_slv_lf_json_candidates(readme_text: str) -> list[tuple[str | None, str | None]]:
    return _extract_current_candidates(
        readme_text,
        backtick_re=_SLV_JSON_BACKTICK_RE,
        inline_re=_SLV_JSON_INLINE_RE,
    )


def extract_current_slv_lf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/slv/readme-lf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_SLV_JSON_BACKTICK_RE,
        inline_re=_SLV_JSON_INLINE_RE,
    )


def extract_current_slv_hf_json_candidates(readme_text: str) -> list[tuple[str | None, str | None]]:
    return _extract_current_candidates(
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


def extract_current_tqqq_lf_json_candidates(readme_text: str) -> list[tuple[str | None, str | None]]:
    return _extract_current_candidates(
        readme_text,
        backtick_re=_TQQQ_JSON_BACKTICK_RE,
        inline_re=_TQQQ_JSON_INLINE_RE,
    )


def extract_current_tqqq_lf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/tqqq/readme-lf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_TQQQ_JSON_BACKTICK_RE,
        inline_re=_TQQQ_JSON_INLINE_RE,
    )


def extract_current_tqqq_hf_json_candidates(readme_text: str) -> list[tuple[str | None, str | None]]:
    return _extract_current_candidates(
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


def extract_current_mnq_hf_json_candidates(readme_text: str) -> list[tuple[str | None, str | None]]:
    return _extract_current_candidates(
        readme_text,
        backtick_re=_MNQ_JSON_BACKTICK_RE,
        inline_re=_MNQ_JSON_INLINE_RE,
    )


def extract_current_mnq_hf_json_path(readme_text: str) -> tuple[str | None, str | None]:
    """Return (version, json_path) from backtests/mnq/readme-hf.md CURRENT section."""
    return _extract_current(
        readme_text,
        backtick_re=_MNQ_JSON_BACKTICK_RE,
        inline_re=_MNQ_JSON_INLINE_RE,
    )
