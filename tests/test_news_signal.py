from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tradebot.news as news
from tradebot.news import (
    Article,
    DEFAULT_MODEL,
    NewsError,
    build_prompt,
    is_causally_relevant,
    output_schema,
    parse_finviz_news,
    run_once,
    select_candidates,
    validate_analysis,
    validate_memory_markdown,
)


FIXTURE = Path(__file__).parent / "fixtures" / "news" / "finviz_news.html"
NOW = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)


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
        "calibration": "Compared with the matching retained high-water umbrella.",
    }


def _analysis(ids: list[str]) -> dict[str, object]:
    oil_ids = ids[:2]
    return {
        "events": [
            {
                "event": "Bab el-Mandeb shipping closure",
                "umbrella": "Oil chokepoint disruption",
                "status": "confirmed",
                "basis": "cross_source_content",
                "channel": "supply",
                "mechanism": "Physical transit loss removes effective oil transport capacity.",
                "evidence": oil_ids,
                "xsp": _score(-1, 70),
                "mcl": _score(1, 100),
            }
        ],
        "assets": {
            "XSP": {
                "direction": -1,
                "impact": 70,
                "confidence": 0.9,
                "horizon_hours": 24,
                "change": "new",
                "mechanism": "Oil scarcity lifts inflation and yields while compressing index multiples.",
                "calibration": "Below the XSP system-function ceiling.",
                "drivers": oil_ids,
            },
            "MCL": {
                "direction": 1,
                "impact": 100,
                "confidence": 0.95,
                "horizon_hours": 24,
                "change": "new",
                "mechanism": "Confirmed chokepoint closure removes effective transport capacity.",
                "calibration": "Matches the retained MCL physical-closure ceiling.",
                "drivers": oil_ids,
            },
        },
        "memory_markdown": """# Trade Research Memory

## Mission

Retain only causal trend evidence for XSP and MCL.

## Calibration Ledger

- **XSP reference ceiling — 100.** System-scale US economic or market dysfunction.
- **MCL reference ceiling — 100.** Confirmed sustained physical oil-chokepoint closure.

## 1D - Active Trend Tape

### Oil chokepoint disruption
- Thesis: Confirmed closure raises physical scarcity and freight risk.
- Transmission: XSP bearish through inflation; MCL bullish through supply.
- Invalidation: Verified reopening and normalized tanker transit.

## 1W - Persistent Themes

- None.

## 1M - Regime Shifts

- None.

## 1Y - Secular Priors

- None.
""",
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


def test_causal_sieve_spans_chokepoints_and_us_macro_but_rejects_noise() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)

    assert is_causally_relevant(articles[0])
    assert is_causally_relevant(articles[1])
    assert is_causally_relevant(articles[2])
    assert not is_causally_relevant(articles[3])
    assert not is_causally_relevant(
        Article("n", "WSJ", "Nike market share is good for its stock", "Sneaker demand.", "https://wsj.com/n", "2026-07-24T09:00:00Z", "")
    )


def test_candidate_overflow_remains_unacknowledged() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    selected, acknowledged, deferred = select_candidates(articles, seen=set(), limit=2)

    assert len(selected) == 2
    assert deferred == 1
    assert articles[2].id not in acknowledged
    assert articles[3].id in acknowledged


def test_chokepoint_maximum_is_valid_and_unknown_evidence_is_rejected() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    relevant = [article for article in articles if is_causally_relevant(article)]
    value = _analysis([article.id for article in relevant])

    assert validate_analysis(value, article_ids={article.id for article in relevant}) == value

    value["events"][0]["evidence"] = ["unknown"]
    with pytest.raises(NewsError, match="unknown article ID"):
        validate_analysis(value, article_ids={article.id for article in relevant})


def test_maximum_mcl_rejects_headline_only_claims() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)
    relevant = [article for article in articles if is_causally_relevant(article)]
    value = _analysis([article.id for article in relevant])
    value["events"][0]["basis"] = "summary_only"

    with pytest.raises(NewsError, match="evidence-basis ceiling"):
        validate_analysis(value, article_ids={article.id for article in relevant})


def test_codex_schema_avoids_unsupported_unique_items_keyword() -> None:
    assert "uniqueItems" not in json.dumps(output_schema())


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


def test_memory_contract_rejects_growth_and_missing_horizons() -> None:
    memory = _analysis(["known"])["memory_markdown"]
    assert validate_memory_markdown(memory).endswith("\n")

    with pytest.raises(NewsError, match="exactly one"):
        validate_memory_markdown("# Trade Research Memory\n\n## Mission\n")

    too_long = memory + "\n".join(f"- line {index}" for index in range(5_000))
    with pytest.raises(NewsError, match="under 5,000 lines"):
        validate_memory_markdown(too_long)


def test_prompt_contains_previous_assets_and_forbids_execution_advice() -> None:
    articles = parse_finviz_news(_html(), observed_at=NOW)[:1]
    previous = {
        "schema": "tradebot.news-signal.v2",
        "analysis": {"assets": _analysis([articles[0].id])["assets"]},
    }

    prompt = build_prompt(
        articles,
        previous,
        memory_path=Path("/Users/x/.codex/trade-research.md"),
        memory_markdown=_analysis([articles[0].id])["memory_markdown"],
        as_of_utc="2026-07-24T09:00:00Z",
    )

    assert '"previous_assets":{"XSP"' in prompt
    assert "/Users/x/.codex/trade-research.md" in prompt
    assert "1D/1W/1M/1Y" in prompt
    assert "strictly under\n  5,000 lines" in prompt
    assert "Bab el-Mandeb/Hormuz" in prompt
    assert "actionable impact 0..100" in prompt
    assert "Calibration Ledger" in prompt
    assert "Open at most eight supplied links" in prompt
    assert "Never call an event confirmed without readable page content" in prompt
    assert "Never emit buy/sell/order advice" in prompt


def test_run_once_publishes_once_then_skips_seen_batch(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(_url: str, *, timeout_sec: int) -> str:
        assert timeout_sec == 30
        return _html()

    def grader(prompt: str, _schema: dict, **_kwargs) -> tuple[dict, dict]:
        calls.append(prompt)
        articles = json.loads(prompt.split("INPUT:\n", 1)[1])["articles"]
        return _analysis([article["id"] for article in articles]), {"version": "test"}

    first = run_once(
        data_dir=tmp_path,
        now=NOW,
        timeout_sec=30,
        fetcher=fetcher,
        grader=grader,
    )
    second = run_once(
        data_dir=tmp_path,
        now=datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc),
        timeout_sec=30,
        fetcher=fetcher,
        grader=grader,
    )

    assert first["status"] == "published"
    assert second["status"] == "no_candidates"
    assert len(calls) == 1
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["schema"] == "tradebot.news-signal.v2"
    assert latest["score_version"] == "causal-impact-100.v1"
    assert latest["analysis"]["assets"]["MCL"]["impact"] == 100
    assert "memory_markdown" not in latest["analysis"]
    assert latest["memory"]["lines"] < 5_000
    assert (tmp_path / "trade-research.md").read_text(encoding="utf-8").startswith(
        "# Trade Research Memory"
    )
    assert len((tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_run_once_does_not_call_codex_for_irrelevant_mainstream_news(tmp_path: Path) -> None:
    html = """
    <tr class="news_table-row">
      <td class="news_date-cell">09:00AM</td>
      <td class="news_link-cell" data-boxover-text="A family owns a cycling race.">
        <a href="https://www.nytimes.com/sports/cycling">Tour de France ownership</a>
      </td>
    </tr>
    """

    def fail_grader(*_args, **_kwargs):
        raise AssertionError("Codex must not run without a relevant candidate")

    result = run_once(
        data_dir=tmp_path,
        now=NOW,
        fetcher=lambda *_args, **_kwargs: html,
        grader=fail_grader,
    )

    assert result["status"] == "no_candidates"
    assert (tmp_path / "state.json").exists()
    assert not (tmp_path / "latest.json").exists()


def test_failed_grade_preserves_latest_and_state(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    state_path = tmp_path / "state.json"
    memory_path = tmp_path / "trade-research.md"
    latest_path.write_text('{"analysis":{"assets":{}}}\n', encoding="utf-8")
    state_path.write_text(
        '{"schema":"tradebot.news-state.v1","last_successful_fetch_utc":null,"seen":{}}\n',
        encoding="utf-8",
    )
    memory_path.write_text(_analysis(["known"])["memory_markdown"], encoding="utf-8")
    before_latest = latest_path.read_bytes()
    before_state = state_path.read_bytes()
    before_memory = memory_path.read_bytes()

    def fail_grader(*_args, **_kwargs):
        raise NewsError("synthetic grader failure")

    with pytest.raises(NewsError, match="synthetic grader failure"):
        run_once(
            data_dir=tmp_path,
            now=NOW,
            fetcher=lambda *_args, **_kwargs: _html(),
            grader=fail_grader,
        )

    assert latest_path.read_bytes() == before_latest
    assert state_path.read_bytes() == before_state
    assert memory_path.read_bytes() == before_memory
    assert not (tmp_path / "history.jsonl").exists()
