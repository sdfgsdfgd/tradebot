# News Signal One-Shot — XSP And MCL

- **Status:** `[READY — LOCAL AND q COMPATIBILITY PROVEN; NOT ENABLED]`
- **Goal:** `019f935a-5774-7623-b08e-fd912c4ffe2f`
- **Worktree:** `/Users/x/Desktop/py/tradebot-news-intelligence`
- **Branch:** `codex/news-intelligence-gate`
- **Baseline:** `8c2e65968785b8dbf98f51fc1fb9c4b48b90f135`
- **Primary assets:** `XSP` and `MCL`
- **Authority:** signal fact only; never order intent or execution

This file is the task tree, decision journal, evidence ledger, and resume source.
Update it whenever cadence, input, output, runtime, or policy authority changes.

## 0. Correction

The original `30s ± 5s` polling proposal was architecturally wrong for this
objective. It optimized headline completeness instead of inference efficiency.

The accepted mandate is:

> Wake once per configured one-, two-, or four-hour interval; fetch the news
> surface once; group the unseen relevant links; run one ephemeral Codex
> analysis; atomically publish one compact XSP/MCL JSON signal; exit.

There is no polling loop, resident Python worker, fast grader, secondary
reconciler, immortal Codex thread, or Python article crawler. The single Codex
run may selectively open at most eight high-value supplied pages.

The default timer is **one hour** because the requested atomic unit is an
hourly link group. Changing the timer to two or four hours changes scheduling,
not application architecture.

## 1. Outcome

The result is not generic sentiment. It is a compact causal answer:

1. What materially changed?
2. Is it confirmed physical reality, corroborated reporting, a single report,
   or rhetoric?
3. Through what channel does it affect a stable US index or crude oil?
4. In what direction, with what impact, confidence, and horizon?
5. Did that asset signal strengthen, weaken, reverse, or remain unchanged
   relative to the previous published snapshot?
6. How does its 0–100 causal-impact score compare with the retained high-water
   event under the same umbrella?

```text
systemd timer
  -> one Python process
  -> one Finviz GET
  -> mainstream source + compact causal-keyword filter
  -> all unseen candidates in one immutable batch
  -> inject ~/.codex/trade-research.md + previous JSON
  -> one `codex exec` with ephemeral state, live native search, and output schema
  -> validate signals, evidence IDs, and full curated memory replacement
  -> atomic memory/latest/state + append history.jsonl
  -> exit
```

No valid new candidates means no Codex invocation and no replacement signal.
Consumers must inspect `as_of_utc` and reject stale snapshots.

## 2. Minimal input contract

Finviz is only a replaceable discovery index. Python sends headline, hover
summary, publisher URL, and knowledge timestamp:

```json
{
  "id": "sha256-prefix",
  "source": "Reuters",
  "title": "…",
  "summary": "Finviz-provided hover summary when present",
  "url": "canonical publisher URL",
  "observed_at_utc": "2026-07-24T09:00:00Z"
}
```

`observed_at_utc` is when this system learned the link. Publisher labels are
kept as metadata but never used to pretend the system knew an article earlier.

### Mainstream source boundary

- Reuters
- Bloomberg
- Wall Street Journal
- CNBC
- New York Times
- BBC
- MarketWatch
- Associated Press
- Financial Times

Publisher tracking parameters are stripped before URL identity is computed.
Raw title and summary text are treated as untrusted data, never instructions.

### Compact causal lexicon

The cheap lexical sieve exists only to avoid spending a Codex run on cycling,
celebrity, or other clearly unrelated news. It deliberately spans mechanisms,
not fashionable sentiment words:

```text
US macro/index:
fed fomc rate yield treasury inflation cpi pce jobs payroll unemployment
gdp recession tariff sanction bank credit default bailout shutdown dollar
stocks futures s&p nasdaq earnings ai chip

crude/geopolitics:
oil crude brent wti opec inventory refinery pipeline tanker
chokepoint hormuz mandeb red sea suez houthi iran russia saudi
war invasion attack missile strike embargo blockade closure ceasefire
```

This is one centralized tuple, not per-source or per-contract helper logic.
It is a prefilter only. Codex must infer event identity and causal transmission.

## 3. One-shot Codex contract

Invocation:

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
  --output-schema <schema>
  -
```

One prompt contains:

- the immutable candidate batch;
- the previous published XSP/MCL signals, if any;
- the absolute memory path and current curated Markdown;
- the short causal rubric below; and
- an instruction to output JSON only through the supplied schema.

The model may use native live web search to open at most eight supplied links
whose contents could change event identity, physical status, impact, or
direction. It may not use the shell, edit files, or persist a session. Python
owns all local I/O, validation, state, publication, timeout, and failure
behavior.
Codex stderr is inherited and streamed to the operator or systemd journal;
stdout remains exclusively the structured final JSON.

### Curated cross-run memory

`~/.codex/trade-research.md` is the only semantic memory. Codex does not edit it
with tools. It returns the complete replacement in the same structured response;
Python validates and atomically publishes it.

The file is a causal research map, not a headline log:

- `Calibration Ledger`: permanent XSP/MCL 100-point ceilings and at most twelve
  umbrella high-water observations; a weaker repetition never adds a record;
- `1D - Active Trend Tape`: active facts no older than 24 hours;
- `1W - Persistent Themes`: merged themes supported within seven days;
- `1M - Regime Shifts`: compressed changes supported within 31 days;
- `1Y - Secular Priors`: only durable priors supported within 366 days.

Every item must expire, merge, be promoted, or be rewritten more compactly.
Each theme states XSP/MCL transmission, confidence, representative dated
headlines, last confirmation, and invalidation. The prompt budgets
`12/10/8/6` themes across the horizons, targets under 250 lines, and Python
rejects anything at or above 5,000 lines.

## 4. Minimal output contract

The Codex response contains at most five distinct events plus exactly one XSP
and one MCL aggregate:

```json
{
  "events": [
    {
      "event": "Verified shipping closure at Bab el-Mandeb",
      "umbrella": "Oil chokepoint disruption",
      "status": "confirmed",
      "basis": "cross_source_content",
      "channel": "supply",
      "mechanism": "Physical crude/product transit disruption raises scarcity and freight risk.",
      "evidence": ["a1b2c3d4e5f6", "b2c3d4e5f6a1"],
      "xsp": {
        "direction": -1,
        "impact": 70,
        "components": {"magnitude": 22, "transmission": 18, "surprise": 13, "immediacy": 11, "persistence": 6},
        "calibration": "Below the XSP system-function ceiling."
      },
      "mcl": {
        "direction": 1,
        "impact": 100,
        "components": {"magnitude": 30, "transmission": 25, "surprise": 20, "immediacy": 15, "persistence": 10},
        "calibration": "Matches the retained physical-closure ceiling."
      }
    }
  ],
  "assets": {
    "XSP": {
      "direction": -1,
      "impact": 76,
      "confidence": 0.9,
      "horizon_hours": 24,
      "change": "strengthening",
      "mechanism": "Oil shock lifts inflation/yields and compresses broad-index multiples.",
      "calibration": "Reinforcing channels remain below system-scale dysfunction.",
      "drivers": ["a1b2c3d4e5f6", "b2c3d4e5f6a1"]
    },
    "MCL": {
      "direction": 1,
      "impact": 100,
      "confidence": 0.95,
      "horizon_hours": 24,
      "change": "reversal",
      "mechanism": "A confirmed chokepoint closure removes effective transport capacity.",
      "calibration": "Matches the physical-closure ceiling.",
      "drivers": ["a1b2c3d4e5f6", "b2c3d4e5f6a1"]
    }
  }
}
```

The publisher-independent wrapper adds:

- `schema = "tradebot.news-signal.v2"`
- `score_version = "causal-impact-100.v1"`
- `as_of_utc`
- `window_started_at_utc`
- `article_count`
- `articles`
- `analysis`
- Codex version/model/run receipt when observable

Tradebot may deterministically derive a scalar:

```text
signed_strength = direction * impact * confidence
```

The model does not emit `buy`, `sell`, quantity, contract selection, or an
order action.

## 5. Causal rubric

### Shared

- `direction`: `-1`, `0`, or `1`.
- `impact`: `0..100`, measuring actionable causal severity, not drama or
  probability. It equals exactly:
  `magnitude 0..30 + transmission 0..25 + surprise 0..20 +
  immediacy 0..15 + persistence 0..10`.
- `confidence`: `0..1`, based on specificity and corroboration.
- `horizon_hours`: `1`, `4`, or `24`.
- `change`: `new`, `strengthening`, `weakening`, `reversal`, or `unchanged`.
- Duplicate coverage is corroboration, never independent impact addition.
- Rhetoric, threat, proposal, action, and confirmed physical consequence are
  materially different states.
- Unsupported or contradictory evidence reduces confidence; irrelevant news
  is omitted.
- Each event records `summary_only`, `single_content`, or
  `cross_source_content`; `confirmed` is illegal without readable page content.
- Rhetoric is capped at 30, summary-only evidence at 49, and one readable page
  at 79. Scores `80..100` require cross-source page content.
- `direction = 0` exactly when `impact = 0`; every component sum and evidence
  ceiling is independently revalidated in Python.

This score is an auditable causal prior, not yet an empirically fitted return
probability. Real 1/4/24-hour XSP and MCL outcomes must later calibrate it
without lookahead before the signal receives policy authority.

### MCL

The dominant channels are physical supply, transport, inventory, production,
sanctions, geopolitical risk premium, and global demand.

- Confirmed closure/blockage of Bab el-Mandeb, Hormuz, or another major oil
  chokepoint may receive **MCL impact 100**, but only after at least two
  independent source pages are read.
- A credible imminent closure or verified attack is major but remains below
  100 until the physical maximum boundary is actually evidenced.
- War-hawk rhetoric without action cannot score above 30 merely because it is
  loud.
- Reopening, ceasefire, production growth, inventory builds, or demand
  destruction normally pressure MCL downward.

### XSP

The dominant channels are US growth, inflation, rates/yields, liquidity,
credit, tariffs, systemic risk, and sufficiently large mega-cap shocks.

- An oil supply shock can hurt XSP through inflation, yields, consumer margins,
  and risk appetite; it is not simply copied from MCL.
- Easing inflation/rates or credible relief can support XSP.
- Idiosyncratic company news is low impact unless its index weight or systemic
  channel makes the broad-contract effect tangible.

## 6. State and failure contract

State directory:

```text
db/news/
  state.json
  latest.json
  history.jsonl
```

- `state.json` contains recent canonical URL hashes and last successful fetch.
- `latest.json` is replaced atomically only after response validation.
- `history.jsonl` contains the complete published wrapper, including inputs.
- `~/.codex/trade-research.md` contains only the latest curated causal memory;
  forgotten detail is intentionally absent.
- State advances only after successful publication.
- A fetch, timeout, process, parse, schema, evidence-reference, or write failure
  leaves the last valid signal untouched and exits nonzero.
- Seen identity is bounded; it is not an ever-growing database.
- If more candidates exist than the bounded batch, overflow remains unseen for
  the next timer run rather than being discarded.

This deliberately avoids SQLite, migrations, queues, and a resident scheduler.

## 7. Minimal repository surfaces

```text
tradebot/news.py
  parser, source/keyword filtering, URL identity, state, prompt/schema,
  Codex invocation, validation, atomic publication, and CLI

tests/fixtures/news/finviz_news.html
tests/test_news_signal.py
  parser, filtering, grouping, validation, retry/idempotency, publication

deploy/systemd/tradebot-news.service
deploy/systemd/tradebot-news.timer
  q-ready oneshot foundation; not enabled during local proof
```

No live UI, policy gate, backtest engine, order path, or existing GPT placeholder
changes in this phase. Those become consumers of a proven `latest.json` later.

Capability ownership:

- production and test: `signal-regime-intelligence`
- q service template: `runtime-configuration-state`
- future consumption/gating: `policy-risk-sizing`
- future causal replay: `market-realism-parity`

## 8. Quest

### Phase 0 — corrected contract `[DONE]`

- [x] isolate the concurrently dirty primary checkout;
- [x] reject high-frequency polling and multi-stage orchestration;
- [x] freeze one fetch + one ephemeral Codex run + one signal;
- [x] make XSP and MCL primary assets;
- [x] encode physical chokepoint severity as a first-class mechanism;
- [x] activate the long-horizon goal after user authorization.

### Phase 1 — deterministic one-shot spine `[DONE]`

- [x] implement the Finviz parser with standard-library HTML handling;
- [x] centralize source, URL, and keyword filtering;
- [x] implement bounded seen state and candidate overflow;
- [x] implement compact prompt and embedded JSON schema;
- [x] invoke one read-only ephemeral Codex process;
- [x] selectively inspect high-value article contents through native live search;
- [x] validate every range, enum, asset, score component, and evidence reference;
- [x] inject and atomically curate calibration plus 1D/1W/1M/1Y memory;
- [x] publish latest/history/state in failure-safe order.

### Phase 2 — proof `[DONE]`

- [x] fixture-test publisher extraction and blog rejection;
- [x] prove unrelated links do not trigger Codex;
- [x] prove duplicate URLs are not reprocessed;
- [x] prove overflow is not lost;
- [x] prove malformed/stale evidence cannot replace `latest.json`;
- [x] prove an MCL chokepoint event can express impact 100;
- [x] prove MCL impact 100 is rejected without cross-source page contents;
- [x] pin and regression-test `gpt-5.6-sol` plus `max` reasoning;
- [x] update the MECC capability ledger and README crosswalk;
- [x] run focused architecture/ledger tests: `20 passed`.

### Phase 3 — authentic dry run `[DONE]`

- [x] run real Finviz fetches locally;
- [x] inspect exact candidate batches and remove the noisy singular `stock` term;
- [x] run real schema-bound Codex invocations with native page search;
- [x] inspect XSP/MCL JSON, content basis, causal arithmetic, and receipt;
- [x] keep it disconnected from trade policy.

First-run evidence: Codex rejected standard JSON Schema `uniqueItems` in its
structured-output dialect. Publication failed closed and wrote no state or
signal. Uniqueness remains enforced by Python; the unsupported schema keyword
was removed and regression-tested before retry.

Max-effort evidence on 2026-07-24:

- Codex's own startup trace printed `model: gpt-5.6-sol` and
  `reasoning effort: max`; receipt independently agreed.
- The focused six-link probe read Reuters and Bloomberg material and returned
  three `cross_source_content` events rather than headline-only guesses.
- Aggregate output was XSP `-1 / 78 / confidence .89 / 24h` and MCL
  `+1 / 80 / confidence .83 / 24h`.
- The Gulf high-water was XSP `72`, MCL `84`, explicitly below 100 because
  two-page sustained-closure evidence was absent and crude retraced.
- The replacement memory was 48 lines and admitted three first-version
  umbrella high-water marks.

Final compatibility evidence:

- q Codex `0.144.1` opened live Finviz under Sol/max and returned valid
  schema-bound JSON.
- q Python `3.13` compiled the module and its one Finviz GET found 78
  whitelisted rows, 42 surviving the compact causal sieve.
- q `systemd-analyze verify` accepted the updated service and timer.
- Full local regression: `678 passed, 4 deselected, 1 warning` under Python
  `3.12` in `12.04s`.

### Phase 4 — q foundation `[DONE: templates only, not installed]`

- [x] add a user `Type=oneshot` service and timer template;
- [x] default to `1h` with coarse timer accuracy for wakeup coalescing;
- [x] document the one-line `2h` and `4h` timer alternatives;
- [x] verify the template with q's `systemd-analyze`;
- [x] prove q's Codex `0.144.1` accepts Sol/max/schema/live-search together;
- [x] compile and one-fetch smoke the module with q's Python 3.13;
- [x] run the full repository suite with Python 3.12;
- [x] do not install or enable the service before final compatibility proof.

### Phase 5 — later tradebot consumption `[BLOCKED: signal evidence]`

- [ ] define freshness and default-neutral consumer semantics;
- [ ] surface the signal in the live UI/journal;
- [ ] replay exact historical wrappers without lookahead;
- [ ] attach realized 1/4/24-hour contract moves and fit monotonic
  score-to-outcome calibration by asset, horizon, umbrella, and regime;
- [ ] evaluate shadow counterfactuals;
- [ ] seek separate authority before any entry veto.

## 9. Decision journal

| ID | Decision | Reason |
|---|---|---|
| N-001 | Dedicated worktree only | The primary checkout is changing concurrently |
| N-002 | One-shot timer, no daemon | The desired information changes slowly |
| N-003 | Default one-hour run; timer may be 2h/4h | Hourly link grouping is the requested atomic unit |
| N-004 | One Codex call for the whole batch | Cross-source grouping is cheaper and smarter jointly |
| N-005 | Ephemeral session plus prior JSON | Trend comparison needs explicit state, not conversation memory |
| N-006 | Summary batch plus at most eight native page reads | Content matters, but a Python crawler and indiscriminate fetching do not |
| N-007 | JSON files, not SQLite | One producer and one latest snapshot do not justify a database |
| N-008 | XSP and MCL are peers | Oil/geopolitical mechanisms are primary, not a later afterthought |
| N-009 | Physical consequence outranks rhetoric | Price impact follows causal transmission, not emotional language |
| N-010 | No policy integration yet | First prove the signal fact before granting it authority |
| N-011 | Full memory replacement, never model file edits | Atomic validation prevents append bloat and corruption |
| N-012 | 1D/1W/1M/1Y promotion and forgetting | Durable causal themes survive while stale detail loses space |
| N-013 | Confirmed and MCL 100 require content evidence | Headlines alone cannot establish physical consequence |
| N-014 | Explicit `gpt-5.6-sol` with `max` reasoning and strict config | One infrequent quality-first inference should spend cognition, not wakeups |
| N-015 | Versioned 0–100 additive score | Exact components make severity explainable and machine-checkable |
| N-016 | Permanent umbrella high-water ledger in the same bounded Markdown | Every score has a retained comparator without an ever-growing news diary |
| N-017 | Confidence remains separate from impact | Evidence certainty and conditional contract severity are different facts |
| N-018 | Empirical return calibration remains a later admission gate | A precise causal prior is not yet a proven trading edge |

## Conclusion

- **Quest:** News Signal One-Shot — XSP And MCL
- **Status:** `[READY — LOCAL AND q COMPATIBILITY PROVEN; NOT ENABLED]`
- **Current seam:** one independent signal publisher, designed to become the
  first small systemd-managed tradebot component on q.
- **Predictive observation:** model intelligence will matter most in separating
  rhetoric from physical consequence and in transmitting one event differently
  into oil and broad US-index contracts. Everything around that inference
  should remain brutally ordinary.
