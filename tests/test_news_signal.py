from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tradebot.news as news_package
from tradebot.news import __main__ as news_entrypoint
from tradebot.news import pipeline as news
from tradebot.news.contract import (
    NewsError,
    SCORE_COMPONENT_LIMITS,
    load_news_history,
    observe_news_signal,
    output_schema,
    publication_id,
    select_news_snapshot_at,
    validate_analysis,
    validate_memory_markdown,
)
from tradebot.news.pipeline import (
    DEFAULT_FETCH_TIMEOUT_SEC,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SEC,
    build_prompt,
    parse_finviz_news,
    run_once,
    select_candidates,
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
    for name, ceiling in SCORE_COMPONENT_LIMITS.items():
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


def test_news_package_preserves_public_and_module_entrypoints() -> None:
    assert news_package.NewsError is NewsError
    assert news_package.run_once is run_once
    assert news_entrypoint.main is news.main
    assert news._parser().prog == "python -m tradebot.news"


def test_published_asset_observation_is_fresh_then_fails_closed_at_horizon() -> None:
    value = {
        "schema": news.SCHEMA,
        "score_version": news.SCORE_VERSION,
        "run_status": "published",
        "signal_as_of_utc": _iso(NOW),
        "snapshot_as_of_utc": _iso(NOW),
        "analysis": {"assets": _analysis([])["assets"]},
    }
    value["analysis"]["assets"]["XSP"]["horizon_hours"] = 4

    fresh = observe_news_signal(
        value,
        symbol="XSP",
        as_of=NOW + timedelta(hours=3, minutes=59),
    )
    stale = observe_news_signal(
        value,
        symbol="XSP",
        as_of=NOW + timedelta(hours=4, seconds=1),
    )

    assert fresh.usable is True
    assert fresh.reason == "fresh"
    assert fresh.as_payload()["drivers"] == ["bab-el-mandeb-closure"]
    assert stale.usable is False
    assert stale.reason == "stale"


def test_asset_observation_validates_new_content_addresses_and_reads_legacy() -> None:
    value = {
        "schema": news.SCHEMA,
        "score_version": news.SCORE_VERSION,
        "run_status": "published",
        "signal_as_of_utc": _iso(NOW),
        "snapshot_as_of_utc": _iso(NOW),
        "analysis": {"assets": _analysis([])["assets"]},
    }

    assert observe_news_signal(value, symbol="XSP", as_of=NOW).usable is True
    value["publication_id"] = publication_id(value)
    assert observe_news_signal(value, symbol="XSP", as_of=NOW).usable is True

    value["run_status"] = "no_new_evidence"
    with pytest.raises(NewsError, match="publication ID mismatch"):
        observe_news_signal(value, symbol="XSP", as_of=NOW)


def test_news_history_selects_the_latest_causally_available_publication(
    tmp_path,
) -> None:
    def snapshot(as_of: datetime) -> dict[str, object]:
        return {
            "schema": news.SCHEMA,
            "score_version": news.SCORE_VERSION,
            "run_status": "published",
            "signal_as_of_utc": _iso(as_of),
            "snapshot_as_of_utc": _iso(as_of),
            "analysis": {"assets": _analysis([])["assets"]},
        }

    prior = snapshot(NOW - timedelta(minutes=5))
    future = snapshot(NOW + timedelta(minutes=1))
    path = tmp_path / "2026-07.jsonl"
    path.write_text(
        "\n".join(json.dumps(value) for value in (prior, future)) + "\n",
        encoding="utf-8",
    )

    loaded = load_news_history(path)

    assert loaded == (prior, future)
    assert select_news_snapshot_at(loaded, as_of=NOW) == prior
    assert (
        select_news_snapshot_at(
            loaded,
            as_of=NOW - timedelta(minutes=6),
        )
        is None
    )

    path.write_bytes(path.read_bytes() + b'{"torn":')
    with pytest.raises(NewsError, match="invalid news snapshot JSON"):
        load_news_history(path)


def test_asset_observation_rejects_future_or_incompatible_state() -> None:
    value = {
        "schema": news.SCHEMA,
        "score_version": news.SCORE_VERSION,
        "run_status": "published",
        "signal_as_of_utc": _iso(NOW),
        "snapshot_as_of_utc": _iso(NOW),
        "analysis": {"assets": _analysis([])["assets"]},
    }

    future = observe_news_signal(
        value,
        symbol="XSP",
        as_of=NOW - timedelta(seconds=1),
    )
    assert future.usable is False
    assert future.reason == "future"

    value["snapshot_as_of_utc"] = _iso(NOW + timedelta(hours=1))
    unavailable = observe_news_signal(
        value,
        symbol="XSP",
        as_of=NOW + timedelta(minutes=30),
    )
    assert unavailable.usable is False
    assert unavailable.reason == "future"

    value["snapshot_as_of_utc"] = _iso(NOW - timedelta(seconds=1))
    with pytest.raises(NewsError, match="precedes signal"):
        observe_news_signal(value, symbol="XSP", as_of=NOW)

    value["snapshot_as_of_utc"] = _iso(NOW)
    value["schema"] = "tradebot.news-signal.v2"
    with pytest.raises(NewsError, match="schema/version"):
        observe_news_signal(value, symbol="XSP", as_of=NOW)


def test_systemd_budget_leaves_atomic_publication_grace() -> None:
    deploy = Path(__file__).parents[1] / "deploy/systemd"
    unit = (deploy / "tradebot-news.service").read_text()
    guide = (deploy / "README.md").read_text()

    assert "--timeout-sec" not in unit
    assert DEFAULT_FETCH_TIMEOUT_SEC == 30
    assert DEFAULT_TIMEOUT_SEC is None
    assert "--codex %h/.local/bin/codex" in unit
    assert "TimeoutStartSec=infinity" in unit
    assert "StartLimitIntervalSec=2h" in unit
    assert "StartLimitBurst=31" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=30s" in unit
    assert "systemctl --user stop tradebot-news.timer" in guide
    assert 'test "$news_state" = inactive || test "$news_state" = failed' in guide
    assert (
        "install -m 0644 deploy/systemd/tradebot-news.{service,timer} "
        "~/.config/systemd/user/"
    ) in guide
    producer = guide.index("tradebot-news.timer tradebot-xsp-quotes.timer")
    manual_shadow = guide.index(
        "systemctl --user start tradebot-xsp-shadow.service"
    )
    shadow_timer = guide.index(
        "systemctl --user enable --now tradebot-xsp-shadow.timer"
    )
    assert producer < manual_shadow < shadow_timer
    assert (
        "cmp -s deploy/systemd/tradebot-news.service "
        "~/.config/systemd/user/tradebot-news.service"
    ) in guide
    assert "git diff --quiet origin/main -- deploy/systemd/tradebot-news.service" not in guide
    assert guide.index("cmp -s deploy/systemd/tradebot-news.service") < guide.index(
        "git restore --source=HEAD --worktree deploy/systemd/tradebot-news.service"
    )


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


def test_truncated_causal_prose_is_rejected() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    value = _analysis([article.url for article in articles])
    value["assets"]["XSP"]["calibration"] = "Below the systemic ceiling because systemic U"

    with pytest.raises(NewsError, match="complete sentence"):
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


def test_material_timestamp_is_derived_from_the_event_diff() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    prior = _event([article.url for article in articles])
    later = NOW + timedelta(hours=13)
    unchanged = _analysis(
        [article.url for article in articles],
        as_of=later,
        previous=prior,
    )

    unchanged["active_events"][0]["first_seen_utc"] = _iso(later)
    unchanged["active_events"][0]["last_material_change_utc"] = _iso(later)
    validated = validate_analysis(unchanged, previous_events=[prior], as_of=later)
    assert validated["active_events"][0]["first_seen_utc"] == prior["first_seen_utc"]
    assert (
        validated["active_events"][0]["last_material_change_utc"]
        == prior["last_material_change_utc"]
    )

    changed = _analysis(
        [article.url for article in articles],
        as_of=later,
        previous=prior,
    )
    changed["active_events"][0]["evidence_urls"].append(articles[2].url)
    validated = validate_analysis(changed, previous_events=[prior], as_of=later)
    assert validated["active_events"][0]["last_material_change_utc"] == _iso(later)


def test_unknown_asset_driver_is_dropped_from_the_model_cross_reference() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    value = _analysis([article.url for article in articles], as_of=NOW)
    value["assets"]["XSP"]["drivers"].append("not-an-active-event")

    validated = validate_analysis(value, previous_events=[], as_of=NOW)

    assert validated["assets"]["XSP"]["drivers"] == ["bab-el-mandeb-closure"]


def test_codex_schema_avoids_unsupported_unique_items_keyword() -> None:
    schema = json.dumps(output_schema())
    assert "uniqueItems" not in schema
    assert "active_events" in schema
    assert "removals" in schema


def test_codex_invocation_pins_sol_and_max_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdin = StringIO()
            self.stdout = StringIO("{}")
            self.stderr = StringIO()

        def wait(self, *, timeout: int | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    process = FakeProcess()

    def fake_run(command: list[str], **_kwargs) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="codex-cli test\n")

    monkeypatch.setattr(news.subprocess, "run", fake_run)
    monkeypatch.setattr(
        news.subprocess,
        "Popen",
        lambda command, **_kwargs: (calls.append(command), process)[1],
    )
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


def test_codex_upstream_refusal_exits_for_systemd_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RefusedProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdin = StringIO()
            self.stdout = StringIO()
            self.stderr = StringIO(
                'ERROR: {"detail":{"source":"concurrency_limit"}}\n'
            )

        def wait(self, *, timeout: int | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(
        news.subprocess,
        "Popen",
        lambda *_args, **_kwargs: RefusedProcess(),
    )
    monkeypatch.setattr(
        news.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="codex-cli test\n"
        ),
    )

    with pytest.raises(NewsError, match="upstream capacity refused"):
        news.invoke_codex(
            "probe",
            {"type": "object"},
            codex="codex",
            model=DEFAULT_MODEL,
            timeout_sec=None,
        )


def test_memory_contract_rejects_growth_and_old_horizon_sections() -> None:
    assert validate_memory_markdown(_memory()).endswith("\n")

    with pytest.raises(NewsError, match="exactly one"):
        validate_memory_markdown("# Trade Research Memory\n\n## Mission\n")

    too_long = _memory() + "\n".join(f"- line {index}" for index in range(400))
    with pytest.raises(NewsError, match="400 lines"):
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
    assert "republishers must not upgrade basis" in prompt
    assert "fact -> changed physical/economic variable -> contract transmission" in prompt
    assert "no\nword or thought is truncated" in prompt
    assert "complete replacement" in prompt
    inputs = json.loads(prompt.split("INPUT:\n", 1)[1])
    assert "capacity_feedback" not in inputs
    assert "If the 24-event capacity is full" in prompt
    assert "Every old ID omitted" not in prompt
    assert "Never emit buy/sell/order advice" in prompt

    capacity_prompt = build_prompt(
        articles,
        previous,
        memory_path=Path("/Users/x/.codex/trade-research.md"),
        events_path=Path("/Users/x/.codex/trade-events.jsonl"),
        memory_markdown=_memory(),
        active_events=[prior] * 22,
        event_snapshot=news._event_snapshot([prior], as_of=NOW),
        due_event_ids=[str(prior["id"])],
        as_of_utc=_iso(NOW),
    )
    capacity_inputs = json.loads(capacity_prompt.split("INPUT:\n", 1)[1])
    assert capacity_inputs["capacity_feedback"] == {
        "active_event_capacity": 24,
        "prior_active_event_count": 22,
        "prior_event_slots_free": 2,
        "reason": "The prior active-event ledger is at least 90% full; apply the capacity "
        "policy before retaining or adding an event.",
    }


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
        assert _kwargs["timeout_sec"] == 840
        calls.append(prompt)
        inputs = json.loads(prompt.split("INPUT:\n", 1)[1])
        return _analysis([article["url"] for article in inputs["articles"]]), {"version": "test"}

    first = run_once(
        data_dir=tmp_path,
        now=NOW,
        timeout_sec=840,
        fetcher=fetcher,
        grader=grader,
    )
    first_publication_id = json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )["publication_id"]
    second = run_once(
        data_dir=tmp_path,
        now=NOW + timedelta(hours=1),
        timeout_sec=840,
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
    assert latest["publication_id"] != first_publication_id
    assert latest["publication_id"] == publication_id(latest)
    assert latest["analysis"]["assets"]["MCL"]["impact"] == 100
    assert latest["event_snapshot"]["breaking"][0]["id"] == "bab-el-mandeb-closure"
    assert latest["memory"]["lines"] <= 400
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


@pytest.mark.parametrize(
    ("failure_write", "history_snapshot"),
    (
        (2, NOW + timedelta(hours=1)),
        (3, NOW),
    ),
)
def test_interrupted_publication_recovers_one_complete_generation(
    tmp_path: Path,
    monkeypatch,
    failure_write: int,
    history_snapshot: datetime,
) -> None:
    grader_calls = 0

    def grader(prompt: str, _schema: dict, **_kwargs) -> tuple[dict, dict]:
        nonlocal grader_calls
        grader_calls += 1
        inputs = json.loads(prompt.split("INPUT:\n", 1)[1])
        return _analysis([row["url"] for row in inputs["articles"]]), {
            "version": "test"
        }

    original = news._write_json_atomic
    writes = 0

    def interrupt_before_latest(path: Path, value: dict[str, object]) -> None:
        nonlocal writes
        writes += 1
        if writes == failure_write:
            raise OSError("synthetic publication interruption")
        original(path, value)

    monkeypatch.setattr(news, "_write_json_atomic", interrupt_before_latest)
    with pytest.raises(OSError, match="synthetic publication interruption"):
        run_once(
            data_dir=tmp_path,
            now=NOW,
            fetcher=lambda *_args, **_kwargs: _html(),
            grader=grader,
        )
    pending = tmp_path / "pending-publication.json"
    history = tmp_path / "history" / "2026-07.jsonl"
    assert pending.exists()
    assert not history.exists()
    history.parent.mkdir(parents=True)
    with history.open("ab") as handle:
        handle.write(b'{"torn":')

    monkeypatch.setattr(news, "_write_json_atomic", original)
    result = run_once(
        data_dir=tmp_path,
        now=NOW + timedelta(hours=1),
        fetcher=lambda *_args, **_kwargs: _html(),
        grader=grader,
    )

    assert result["status"] == "no_session"
    assert grader_calls == 1
    assert not pending.exists()
    assert history.read_text().count("\n") == 1
    published = json.loads(history.read_text())
    latest = json.loads((tmp_path / "latest.json").read_text())
    state = json.loads((tmp_path / "state.json").read_text())
    assert latest["publication_id"]
    assert latest["signal_as_of_utc"] == _iso(NOW)
    assert latest["snapshot_as_of_utc"] == _iso(NOW + timedelta(hours=1))
    assert published["signal_as_of_utc"] == _iso(NOW)
    assert published["snapshot_as_of_utc"] == _iso(history_snapshot)
    half_hour = observe_news_signal(
        published,
        symbol="XSP",
        as_of=NOW + timedelta(minutes=30),
    )
    assert half_hour.usable is (failure_write == 3)
    assert half_hour.reason == ("fresh" if failure_write == 3 else "future")
    assert observe_news_signal(
        published,
        symbol="XSP",
        as_of=NOW + timedelta(hours=1),
    ).usable
    assert state["last_successful_fetch_utc"] == _iso(NOW + timedelta(hours=1))
    assert latest["memory"]["sha256"] == news.sha256(
        (tmp_path / "trade-research.md").read_bytes()
    ).hexdigest()
    assert latest["events"]["sha256"] == news.sha256(
        (tmp_path / "trade-events.jsonl").read_bytes()
    ).hexdigest()
