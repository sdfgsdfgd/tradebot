from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tradebot.news as news
from tradebot.news import (
    DEFAULT_MODEL,
    NewsError,
    build_prompt,
    output_schema,
    parse_finviz_news,
    run_once,
    select_candidates,
    validate_analysis,
    validate_memory_markdown,
)


FIXTURE = Path(__file__).parent / "fixtures" / "news" / "finviz_news.html"
NOW = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _score(direction: int, impact: int) -> dict[str, object]:
    remaining = impact
    components = {}
    for name, ceiling in news.SCORE_COMPONENT_LIMITS.items():
        components[name] = min(remaining, ceiling)
        remaining -= components[name]
    return {
        "direction": direction,
        "impact": impact,
        "components": components,
        "calibration": "Compared with the matching retained high-water anchor.",
    }


def _memory() -> str:
    return """# Trade Research Memory

## Mission

Retain only causal trend evidence for XSP and MCL.

## Calibration Anchors

- **XSP reference ceiling — 100.** System-scale US economic or market dysfunction.
- **MCL reference ceiling — 100.** Confirmed sustained physical oil-chokepoint closure.

### Oil chokepoint closure
- MCL 100 requires confirmed cross-source physical evidence.

## Active Regimes

### Gulf shipping risk
- Thesis: Physical transit loss raises oil scarcity and freight risk.
- Active event IDs: bab-el-mandeb-closure.

## Durable Causal Priors

### Evidence is not impact
- Duplicate coverage raises confidence, not magnitude.
"""


def _event(
    urls: list[str],
    *,
    as_of: datetime = NOW,
    previous: dict[str, object] | None = None,
    review_hours: int = 12,
) -> dict[str, object]:
    first_seen = str(previous["first_seen_utc"]) if previous else _iso(as_of)
    last_change = str(previous["last_material_change_utc"]) if previous else _iso(as_of)
    return {
        "id": "bab-el-mandeb-closure",
        "umbrella": "Oil chokepoint disruption",
        "event": "Bab el-Mandeb shipping closure",
        "state": "active",
        "status": "confirmed",
        "basis": "cross_source_content",
        "channel": "supply",
        "confidence": 0.95,
        "first_seen_utc": first_seen,
        "last_material_change_utc": last_change,
        "last_verified_utc": _iso(as_of),
        "review_after_utc": _iso(as_of + timedelta(hours=review_hours)),
        "mechanism": "Physical transit loss removes effective oil transport capacity.",
        "invalidation": "Verified reopening and normalized tanker transit.",
        "evidence_urls": urls[:2],
        "xsp": _score(-1, 70),
        "mcl": _score(1, 100),
    }


def _analysis(
    urls: list[str],
    *,
    as_of: datetime = NOW,
    previous: dict[str, object] | None = None,
    review_hours: int = 12,
) -> dict[str, object]:
    event = _event(
        urls if previous is None else list(previous["evidence_urls"]),
        as_of=as_of,
        previous=previous,
        review_hours=review_hours,
    )
    return {
        "active_events": [event],
        "removals": [],
        "assets": {
            "XSP": {
                "direction": -1,
                "impact": 70,
                "confidence": 0.9,
                "horizon_hours": 24,
                "change": "new" if previous is None else "unchanged",
                "mechanism": "Oil scarcity lifts inflation and compresses index multiples.",
                "calibration": "Below the XSP system-function ceiling.",
                "drivers": [event["id"]],
            },
            "MCL": {
                "direction": 1,
                "impact": 100,
                "confidence": 0.95,
                "horizon_hours": 24,
                "change": "new" if previous is None else "unchanged",
                "mechanism": "Confirmed closure removes effective transport capacity.",
                "calibration": "Matches the physical-closure ceiling.",
                "drivers": [event["id"]],
            },
        },
        "memory_markdown": _memory(),
    }


def _zero_analysis() -> dict[str, object]:
    zero = {
        "direction": 0,
        "impact": 0,
        "confidence": 0.95,
        "horizon_hours": 24,
        "change": "unchanged",
        "mechanism": "No supplied fact has material contract transmission.",
        "calibration": "No retained anchor is engaged.",
        "drivers": [],
    }
    return {
        "active_events": [],
        "removals": [],
        "assets": {"XSP": dict(zero), "MCL": dict(zero)},
        "memory_markdown": _memory(),
    }


def test_finviz_parser_keeps_mainstream_rows_and_canonicalizes_tracking() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)

    assert [article.source for article in articles] == [
        "Reuters",
        "Bloomberg",
        "BBC",
        "New York Times",
    ]
    assert articles[0].url == (
        "https://reuters.com/world/middle-east/bab-el-mandeb-closed-2026-07-24"
    )
    assert articles[2].url == "https://bbc.co.uk/news/articles/fed-rates"
    assert all(article.observed_at_utc == "2026-07-24T09:00:00Z" for article in articles)


def test_candidate_selection_has_no_topical_keyword_sieve() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    selected, acknowledged, deferred = select_candidates(articles, seen=set(), limit=3)

    assert selected == articles[:3]
    assert acknowledged == {article.id for article in articles[:3]}
    assert deferred == 1
    assert articles[3].id not in acknowledged


def test_chokepoint_maximum_is_valid_and_cross_source_is_required() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    value = _analysis([article.url for article in articles])

    assert validate_analysis(value, previous_events=[], as_of=NOW) == value

    value["active_events"][0]["evidence_urls"] = [articles[0].url]
    with pytest.raises(NewsError, match="distinct source hosts"):
        validate_analysis(value, previous_events=[], as_of=NOW)


def test_maximum_mcl_rejects_headline_only_claims() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    value = _analysis([article.url for article in articles])
    value["active_events"][0]["basis"] = "summary_only"

    with pytest.raises(NewsError, match="summaries only"):
        validate_analysis(value, previous_events=[], as_of=NOW)


def test_prior_event_cannot_disappear_without_explicit_removal() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    prior = _event([article.url for article in articles])
    value = _zero_analysis()

    with pytest.raises(NewsError, match="requires exactly one removal"):
        validate_analysis(value, previous_events=[prior], as_of=NOW + timedelta(hours=1))

    value["removals"] = [
        {
            "id": prior["id"],
            "reason": "Verified reopening removed physical and risk-premium transmission.",
            "resolved_at_utc": _iso(NOW + timedelta(hours=1)),
        }
    ]
    assert validate_analysis(
        value,
        previous_events=[prior],
        as_of=NOW + timedelta(hours=1),
    ) == value


def test_unchanged_event_preserves_material_timestamp() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    prior = _event([article.url for article in articles])
    later = NOW + timedelta(hours=13)
    value = _analysis(
        [article.url for article in articles],
        as_of=later,
        previous=prior,
    )

    assert validate_analysis(value, previous_events=[prior], as_of=later) == value

    value["active_events"][0]["last_material_change_utc"] = _iso(later)
    with pytest.raises(NewsError, match="does not match its material diff"):
        validate_analysis(value, previous_events=[prior], as_of=later)


def test_codex_schema_avoids_unsupported_unique_items_keyword() -> None:
    schema = json.dumps(output_schema())
    assert "uniqueItems" not in schema
    assert "active_events" in schema
    assert "removals" in schema


def test_codex_invocation_pins_sol_and_max_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs) -> SimpleNamespace:
        calls.append(command)
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="codex-cli test\n")
        return SimpleNamespace(returncode=0, stdout="{}")

    monkeypatch.setattr(news.subprocess, "run", fake_run)
    _, receipt = news.invoke_codex(
        "probe", {"type": "object"}, codex="codex", model=DEFAULT_MODEL, timeout_sec=30
    )

    assert ["--model", "gpt-5.6-sol"] == calls[0][
        calls[0].index("--model") : calls[0].index("--model") + 2
    ]
    assert 'model_reasoning_effort="max"' in calls[0]
    assert "--strict-config" in calls[0]
    assert receipt["model"] == "gpt-5.6-sol"
    assert receipt["reasoning_effort"] == "max"


def test_memory_contract_rejects_growth_and_old_horizon_sections() -> None:
    assert validate_memory_markdown(_memory()).endswith("\n")

    with pytest.raises(NewsError, match="exactly one"):
        validate_memory_markdown("# Trade Research Memory\n\n## Mission\n")

    too_long = _memory() + "\n".join(f"- line {index}" for index in range(160))
    with pytest.raises(NewsError, match="160 lines"):
        validate_memory_markdown(too_long)

    with pytest.raises(NewsError, match="unexpected section"):
        validate_memory_markdown(_memory() + "\n## 1D - Active Trend Tape\n")


def test_prompt_contains_state_paths_and_compact_causal_contract() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    prior = _event([article.url for article in articles])
    previous = {
        "schema": news.SCHEMA,
        "analysis": {"assets": _analysis([article.url for article in articles])["assets"]},
    }

    prompt = build_prompt(
        articles,
        previous,
        memory_path=Path("/Users/x/.codex/trade-research.md"),
        events_path=Path("/Users/x/.codex/trade-events.jsonl"),
        memory_markdown=_memory(),
        active_events=[prior],
        event_snapshot=news._event_snapshot([prior], as_of=NOW),
        due_event_ids=[str(prior["id"])],
        as_of_utc=_iso(NOW),
    )

    assert '"previous_assets":{"XSP"' in prompt
    assert "/Users/x/.codex/trade-research.md" in prompt
    assert "/Users/x/.codex/trade-events.jsonl" in prompt
    assert "There is no topical keyword\nfilter" in prompt
    assert "Open at most eight substantive pages" in prompt
    assert "fact -> changed physical/economic variable -> contract transmission" in prompt
    assert "complete replacement" in prompt
    assert "Every old ID omitted" not in prompt
    assert "Never emit buy/sell/order advice" in prompt


def test_event_snapshot_uses_exclusive_material_change_buckets() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    events = []
    for event_id, age in [
        ("breaking-event", timedelta(hours=4)),
        ("day-event", timedelta(hours=5)),
        ("week-event", timedelta(days=2)),
        ("month-event", timedelta(days=8)),
        ("persistent-event", timedelta(days=32)),
    ]:
        event = _event([article.url for article in articles])
        event["id"] = event_id
        event["first_seen_utc"] = _iso(NOW - age)
        event["last_material_change_utc"] = _iso(NOW - age)
        event["last_verified_utc"] = _iso(NOW - age)
        events.append(event)

    snapshot = news._event_snapshot(events, as_of=NOW)

    assert {
        bucket: [event["id"] for event in values]
        for bucket, values in snapshot.items()
    } == {
        "breaking": ["breaking-event"],
        "day": ["day-event"],
        "week": ["week-event"],
        "month": ["month-event"],
        "persistent": ["persistent-event"],
    }


def test_run_once_publishes_then_refreshes_without_second_codex_session(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(_url: str, *, timeout_sec: int) -> str:
        assert timeout_sec == 30
        return _html()

    def grader(prompt: str, _schema: dict, **_kwargs) -> tuple[dict, dict]:
        calls.append(prompt)
        inputs = json.loads(prompt.split("INPUT:\n", 1)[1])
        return _analysis([article["url"] for article in inputs["articles"]]), {"version": "test"}

    first = run_once(
        data_dir=tmp_path,
        now=NOW,
        timeout_sec=30,
        fetcher=fetcher,
        grader=grader,
    )
    second = run_once(
        data_dir=tmp_path,
        now=NOW + timedelta(hours=1),
        timeout_sec=30,
        fetcher=fetcher,
        grader=grader,
    )

    assert first["status"] == "published"
    assert second["status"] == "no_session"
    assert len(calls) == 1
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["schema"] == news.SCHEMA
    assert latest["score_version"] == "causal-impact-100.v2"
    assert latest["run_status"] == "no_new_evidence"
    assert latest["analysis"]["assets"]["MCL"]["impact"] == 100
    assert latest["event_snapshot"]["breaking"][0]["id"] == "bab-el-mandeb-closure"
    assert latest["memory"]["lines"] <= 160
    assert (tmp_path / "trade-events.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "trade-research.md").read_text(encoding="utf-8").startswith(
        "# Trade Research Memory"
    )
    assert len((tmp_path / "history" / "2026-07.jsonl").read_text().splitlines()) == 1


def test_run_once_normalizes_runtime_clock_to_schema_precision(
    tmp_path: Path,
) -> None:
    def grader(prompt: str, _schema: dict, **_kwargs) -> tuple[dict, dict]:
        inputs = json.loads(prompt.split("INPUT:\n", 1)[1])
        as_of = datetime.fromisoformat(inputs["as_of_utc"].replace("Z", "+00:00"))
        return _analysis(
            [article["url"] for article in inputs["articles"]],
            as_of=as_of,
        ), {"version": "test"}

    run_once(
        data_dir=tmp_path,
        now=NOW + timedelta(microseconds=999_999),
        fetcher=lambda *_args, **_kwargs: _html(),
        grader=grader,
    )

    event = json.loads(
        (tmp_path / "trade-events.jsonl").read_text(encoding="utf-8")
    )
    assert event["first_seen_utc"] == _iso(NOW)
    assert event["last_material_change_utc"] == _iso(NOW)
    assert event["last_verified_utc"] == _iso(NOW)


def test_due_event_runs_codex_without_new_articles(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def grader(prompt: str, _schema: dict, **_kwargs) -> tuple[dict, dict]:
        inputs = json.loads(prompt.split("INPUT:\n", 1)[1])
        calls.append(inputs)
        previous = inputs["active_events"][0] if inputs["active_events"] else None
        urls = [article["url"] for article in inputs["articles"]]
        return _analysis(
            urls,
            as_of=datetime.fromisoformat(inputs["as_of_utc"].replace("Z", "+00:00")),
            previous=previous,
            review_hours=12 if previous else 1,
        ), {"version": "test"}

    run_once(
        data_dir=tmp_path,
        now=NOW,
        fetcher=lambda *_args, **_kwargs: _html(),
        grader=grader,
    )
    result = run_once(
        data_dir=tmp_path,
        now=NOW + timedelta(hours=2),
        fetcher=lambda *_args, **_kwargs: _html(),
        grader=grader,
    )

    assert result["status"] == "published"
    assert len(calls) == 2
    assert calls[1]["articles"] == []
    assert calls[1]["due_event_ids"] == ["bab-el-mandeb-closure"]


def test_mainstream_noise_is_sent_to_codex_for_semantic_rejection(tmp_path: Path) -> None:
    html = """
    <tr class="news_table-row">
      <td class="news_date-cell">09:00AM</td>
      <td class="news_link-cell" data-boxover-text="A family owns a cycling race.">
        <a href="https://www.nytimes.com/sports/cycling">Tour de France ownership</a>
      </td>
    </tr>
    """
    calls = 0

    def grader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _zero_analysis(), {"version": "test"}

    result = run_once(
        data_dir=tmp_path,
        now=NOW,
        fetcher=lambda *_args, **_kwargs: html,
        grader=grader,
    )

    assert result["status"] == "published"
    assert calls == 1
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["analysis"]["assets"]["XSP"]["impact"] == 0


def test_failed_grade_preserves_latest_state_memory_and_events(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    state_path = tmp_path / "state.json"
    memory_path = tmp_path / "trade-research.md"
    events_path = tmp_path / "trade-events.jsonl"
    latest_path.write_text('{"legacy":true}\n', encoding="utf-8")
    state_path.write_text(
        '{"schema":"tradebot.news-state.v1","last_successful_fetch_utc":null,"seen":{}}\n',
        encoding="utf-8",
    )
    memory_path.write_text(_memory(), encoding="utf-8")
    events_path.write_text("", encoding="utf-8")
    before = {
        path: path.read_bytes()
        for path in (latest_path, state_path, memory_path, events_path)
    }

    def fail_grader(*_args, **_kwargs):
        raise NewsError("synthetic grader failure")

    with pytest.raises(NewsError, match="synthetic grader failure"):
        run_once(
            data_dir=tmp_path,
            now=NOW,
            fetcher=lambda *_args, **_kwargs: _html(),
            grader=fail_grader,
        )

    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not (tmp_path / "history").exists()
