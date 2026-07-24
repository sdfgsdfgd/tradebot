# Quest I — Causal News Memory Gate

- **Status:** `[LOCAL VALIDATION PASSED — REBASE AND q DEPLOYMENT PENDING]`
- **Worktree:** `/Users/x/Desktop/py/tradebot-news-intelligence`
- **Branch:** `codex/news-intelligence-gate`
- **Runtime host:** `q` (`192.168.1.4`)
- **Primary contracts:** `XSP`, `MCL`
- **Authority:** research signal only; never order intent or execution
- **Confirmed design:** two user decision gates completed on 2026-07-24

This file is the task center, architecture contract, decision journal, evidence
ledger, and resume source for the news-intelligence service.

## 1. Mission

Wake approximately every four hours, fetch the Finviz news surface once, send
every unseen mainstream article to one ephemeral Sol/max Codex session, and
publish a compact causal state for XSP and MCL.

The result is not generic sentiment. It answers:

1. What materially changed?
2. Is the claim rhetoric, a single report, corroborated, or confirmed?
3. What physical or economic variable changed?
4. How does that variable transmit independently into XSP and MCL?
5. What are direction, decomposed 0–100 impact, evidence confidence, and horizon?
6. How does the score compare with a historical high-water or boundary anchor?
7. Is an existing event created, materially updated, merely re-verified, merged,
   resolving, or removed?

There is no polling loop, Python article crawler, resident Codex conversation,
secondary grader, or order integration.

## 2. Architecture

```text
q systemd timer (~4h, coalesced)
  -> one Python process
  -> one Finviz GET
  -> mainstream publisher boundary + canonical URL identity
  -> every unseen mainstream item, bounded at 128
  -> no topical or keyword sieve
  -> inject Markdown + active-event JSONL + previous aggregate + age snapshot
  -> one gpt-5.6-sol / max / ephemeral / live-search session
  -> at most eight substantive page reads
  -> strict Python validation
  -> atomic Markdown, event JSONL, and latest snapshot
  -> monthly 13-month audit + bounded seen state
  -> exit
```

If there are no unseen links and no event is due for review, Codex does not run.
Python performs the single Finviz GET, refreshes the mechanical age snapshot,
and exits. A failed fetch, inference, validation, or publication leaves seen
state unchanged so the batch is retried.

## 3. Discovery contract

Finviz is a replaceable discovery index. Python retains:

```json
{
  "id": "canonical-url-sha256-prefix",
  "source": "Reuters",
  "title": "...",
  "summary": "...",
  "url": "https://publisher/article",
  "observed_at_utc": "2026-07-24T09:00:00Z",
  "publisher_label": "Finviz display metadata"
}
```

`observed_at_utc` is the system knowledge time, not a fabricated publication
timestamp. Tracking parameters are stripped before identity is computed.

Mainstream publisher boundary:

- Reuters
- Bloomberg
- Wall Street Journal
- CNBC
- New York Times
- BBC
- MarketWatch
- Associated Press
- Financial Times

The former causal keyword regex is deleted. Sol, not Python vocabulary, decides
whether Chinese tariffs, credit seizure, logistics disruption, court action,
technology concentration, or an unforeseen mechanism matters.

Overflow remains unseen for the next run rather than being discarded.

## 4. Codex contract

```text
codex exec
  --ephemeral
  --ignore-user-config
  --ignore-rules
  --strict-config
  --model gpt-5.6-sol
  -c web_search="live"
  -c model_reasoning_effort="max"
  -c model_reasoning_summary="concise"
  --sandbox read-only
  --output-schema <temporary-schema>
  -
```

stderr is inherited. Reasoning summaries and live-search progress therefore
stream directly to the operator or systemd journal; stdout is reserved for the
schema-bound final JSON.

The prompt uses a causal ontology rather than a keyword list:

- XSP: expected cash flows, inflation, discount rates, liquidity/credit,
  systemic function, index concentration, and risk premium.
- MCL: physical production, transport, inventories, sanctions, global demand,
  and supply-risk premium.

Each event must follow:

```text
fact
  -> changed physical/economic variable
  -> contract transmission
  -> direction and horizon
  -> decomposed impact
  -> nearest historical comparator
```

Impact is conditional contract displacement. It is not drama, probability, or
confidence. Confidence is evidence quality. Duplicate coverage can increase
corroboration but cannot increase causal magnitude.

The model may read at most eight pages chosen for maximum information gain:
physical status, official implementation, independent corroboration,
contradiction, or rhetoric-versus-reality. Due events may use exact-name live
search and primary sources even when the current Finviz page has no new link.

## 5. Scoring

Every event/asset impact is an exact sum:

| Component | Maximum |
|---|---:|
| Magnitude | 30 |
| Contract transmission | 25 |
| Surprise versus expectations | 20 |
| Immediacy | 15 |
| Persistence | 10 |
| **Total** | **100** |

Evidence gates:

- rhetoric: impact ≤30;
- summary only: impact ≤49;
- one substantive page: impact ≤79;
- 80–100: at least two independent substantive source hosts;
- MCL 100: confirmed sustained physical closure or equivalent supply loss.

Permanent boundaries:

- XSP 100: verified shock breaking US economic or market function.
- MCL 100: verified sustained closure of Hormuz, Bab el-Mandeb, or equivalent
  physical oil-supply loss.

The seeded 2025 reciprocal-tariff comparator is XSP -1 / impact 93 /
confidence .99. Its observed April 2–8 S&P decline was 12.14%, the pre-shock
high was not regained until June 27, and the May 12 de-escalation produced a
3.26% one-day rise. WTI is explicitly marked mixed attribution because OPEC+
accelerated supply concurrently.

## 6. Canonical semantic state

### `~/.codex/trade-research.md`

Complete replacement, not append:

```text
# Trade Research Memory
## Mission
## Calibration Anchors        <= 16 entries
## Active Regimes             <= 10 entries
## Durable Causal Priors      <= 12 entries
```

- target: at most 100 lines;
- hard limit: 160 lines and 32 KiB;
- anchors retain singular precedents or umbrella high-water records;
- active regimes synthesize interpretation and reference event IDs;
- priors are compact, falsifiable transmission rules;
- obsolete prose must be merged, compressed, corrected, or forgotten.

### `~/.codex/trade-events.jsonl`

This JSONL is not append-only. Each line is one current event, and every
successful Codex run returns the complete replacement. Maximum: 24 records.

Each event contains:

- stable kebab-case ID, umbrella, and event;
- lifecycle state: `watch`, `active`, or `resolving`;
- evidence status and content basis;
- confidence and causal channel;
- immutable first-seen time;
- last material change, last substantive verification, and next review;
- concise mechanism and falsifiable invalidation;
- one to three exact HTTPS evidence URLs;
- complete XSP and MCL decomposed scores.

Every omitted prior ID requires an explicit removal reason. Python independently
derives `added`, `materially_updated`, `reviewed_unchanged`, and `removed`.

Review ceilings:

- impact ≥80 or resolving: within 24h;
- impact 50–79: within 72h;
- otherwise: within seven days.

Quiet is not resolution. A war, tariff, sanction, credit regime, or physical
disruption remains while its trend transmission remains plausible.

## 7. Published consumer state

`~/.local/state/tradebot/news/latest.json` contains:

- current XSP/MCL aggregate;
- stable event-ID drivers;
- input article metadata;
- event diff;
- Codex receipt;
- memory/event paths, hashes, and counts;
- separate signal and snapshot timestamps;
- every active event exactly once in:

```text
breaking   <= 4h since last material change
day        > 4h and <= 24h
week       > 24h and <= 7d
month      > 7d and <= 31d
persistent > 31d while still active
```

Future UI, backtest, or policy code consumes this file. It does not share the
research service process and receives no order authority in this quest.

Supporting state:

```text
~/.local/state/tradebot/news/
  state.json
  latest.json
  history/
    YYYY-MM.jsonl
```

Seen identity is bounded at 5,000 records. Audit partitions are never prompt
memory and are retained for 13 months.

## 8. Repository surfaces

```text
tradebot/news.py
tradebot/news_contract.py
tests/test_news_signal.py
tests/fixtures/news/finviz_news.html
deploy/systemd/tradebot-news.service
deploy/systemd/tradebot-news.timer
deploy/systemd/README.md
README.md
tests/ledgers/capability_contracts.json
```

Primary capability ownership: `signal-regime-intelligence`.

No live UI, backtest engine, policy kernel, client, order, or broker surface is
modified.

## 9. Systemd contract

- `Type=oneshot`
- `OnUnitInactiveSec=4h`
- `AccuracySec=15min`
- `Nice=10`
- `IOSchedulingClass=idle`
- `PrivateTmp=true`
- `NoNewPrivileges=true`
- `UMask=0077`
- 15-minute service timeout

The four-hour interval begins after the previous run becomes inactive, avoiding
overlap. Timer accuracy permits wakeup coalescing; it is not a 15-minute poll.

## 10. Validation contract

- parser admits mainstream rows and canonicalizes URLs;
- candidate selection has no topical sieve;
- overflow remains pending;
- schema supports full event replacement and explicit removals;
- missing removals fail closed;
- event IDs and first-seen times are stable;
- unchanged events cannot falsify material-change time;
- review deadlines are impact-bounded;
- cross-source basis requires distinct hosts;
- MCL 100 requires confirmed physical evidence;
- Markdown headings, entry counts, lines, bytes, and ceilings are enforced;
- snapshots are exclusive across five age windows;
- irrelevant mainstream news reaches Sol and may produce a neutral result;
- no unseen links and no due event skip Codex;
- due events invoke Codex even without new links;
- failed inference preserves latest, state, memory, and event ledger;
- Sol and max reasoning are command- and receipt-tested;
- systemd units must pass q's `systemd-analyze verify`;
- focused and full repository suites must pass before deployment.

## 11. Deployment quest

- [x] isolate a dedicated worktree and branch;
- [x] complete two user decision gates;
- [x] remove keyword filtering;
- [x] implement bounded Markdown;
- [x] implement bounded active-event JSONL;
- [x] implement explicit removals and lifecycle validation;
- [x] implement breaking/day/week/month/persistent snapshot;
- [x] implement no-session mechanical refresh;
- [x] set four-hour timer and 15-minute accuracy;
- [x] pin `gpt-5.6-sol` and `max`;
- [x] finish focused and full validation;
- [ ] rebase onto current `origin/main` and preserve concurrent ledger changes;
- [ ] push `codex/news-intelligence-gate`;
- [ ] clone exact branch on q;
- [ ] seed q Markdown;
- [ ] install and verify user units;
- [ ] run one manual service while watching stderr;
- [ ] inspect exact persistent artifacts and causal output;
- [ ] enable timer only after the manual run succeeds;
- [ ] record final commit, runtime, and journal evidence here.

Local validation evidence before rebase:

- focused news + capability-ledger + architecture suite: `23 passed`;
- complete repository suite under Python 3.12: `681 passed, 4 deselected,
  1 warning` in `10.65s`;
- both production modules remain below the 1,000-line architecture ceiling:
  `tradebot/news.py` 695 lines and `tradebot/news_contract.py` 716 lines.

## 12. Decision journal

| ID | Decision | Reason |
|---|---|---|
| N-001 | Dedicated worktree | Another agent owns the primary checkout |
| N-002 | One-shot four-hour timer | Trend inference does not justify polling |
| N-003 | One Finviz GET | Avoid a crawler and redundant network work |
| N-004 | Mainstream provenance only | Reject pundit/blog noise without topic blindness |
| N-005 | No keyword filter | Novel causal mechanisms must reach the model |
| N-006 | One Sol/max session | Joint grouping and comparison are the intelligence center |
| N-007 | Eight page reads | Bound information cost while requiring substantive content |
| N-008 | Two curated memories | Qualitative anchors and structured active state serve different consumers |
| N-009 | Full replacement | Makes additions, changes, merges, and removals explicit and bounded |
| N-010 | Stable IDs | Future tradebot consumers need durable causal identity |
| N-011 | Exclusive age snapshot | Avoid duplicate events across horizons |
| N-012 | Confidence separate from impact | Evidence certainty is not conditional severity |
| N-013 | Physical consequence outranks rhetoric | Contract transmission, not language intensity, moves scores |
| N-014 | Audit is not model memory | Preserve calibration evidence without prompt bloat |
| N-015 | No policy integration | A research signal must earn authority empirically |

## Conclusion

- **Quest:** Quest I — Causal News Memory Gate
- **Status:** `[LOCAL VALIDATION PASSED — REBASE AND q DEPLOYMENT PENDING]`
- **Current seam:** one schema-bound research publisher producing a stable file
  contract for the future tradebot service.
- **Predictive observation:** the active-event ledger, not headline volume, will
  become the decisive interface between causal research and later policy.
