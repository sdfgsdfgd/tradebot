# XSP Mastery — Live Research Kata

- **Status:** `[WIP] Phase 0.4 — freeze realistic economics, score, and risk`
- **Management role:** canonical task tree, evidence ledger, decision journal, and resume source
- **Code baseline:** `25513267225908b7288530c1ec0762f7656bbf5b`
- **Management-brain commit:** `3c38af635fcc6ce8b3b0a88e1f2de567345d1bf0`
- **Instrument scope:** XSP first; no expansion until its data, economics, execution, and live drift are mastered
- **Capital premise:** user-reported net liquidity near USD 1,000; re-read broker truth before every capital decision
**Operating rule:** update this file before stopping, changing phase, promoting a strategy, or changing live risk

---

## 0. The mandate

Build a sharp, evolving XSP system whose backtests, live trading, and continuous
evaluations form one learning loop.

While the live bot remains active across days and weeks, evaluations must
re-hydrate and re-score every eligible leaderboard champion. If another proven
strategy becomes materially stronger for the current market context, the system
may eventually promote, demote, combine, or cautiously switch approaches. It
must not revive an opaque regime router, chase recent noise, switch an open
position's governing strategy, or let an unproved selector control live capital.

This kata gives equal weight to three truths:

1. **Actual backtests:** authenticate the data; construct realistic, causal,
   reproducible XSP research over appropriate 1-, 2-, and selectively 5-year
   windows; discover safer and stronger leaders rather than merely optimizing
   the existing grid.
2. **Live trade:** support explicit safety profiles—for example, a defined-risk
   income sleeve built first around vertical credit spreads, a carefully
   justified iron-condor extension, and a separate alpha-hunting sleeve using
   bounded directional/debit structures. Execution must use the right
   patient-to-relentless chase policy for the order's intent, not one ladder
   blindly applied everywhere.
3. **Evaluation:** continuously compare expected behavior with broker-qualified
   quotes, previews, fills, commissions, slippage, Greeks, buying-power effects,
   position state, and realized outcomes. Every divergence must improve either
   the model, execution policy, risk policy, or candidate ranking.

The primary ambition is an extremely reliable, self-healing strategy system
that can keep a minimal defined-risk XSP package—initially no more than one
carefully admitted package—active when conditions truly justify it, and grow
liquidity predictably over repeated weeks. The system should seek extreme
reliability, but it must never claim certainty, guaranteed profit, or treat one
lucky trade as success. The measurable target is stable net expectancy with
bounded drawdown, calibrated confidence, repeatable execution, and explicit
evidence about when **not** to trade.

Before any live-capital activation, spend at least 2–4 hours on extensive,
novel but reproducible backtesting and the necessary improvements to its data,
cache, option-economics, and evaluation seams. Use the available IB Gateway for
read-only qualification, live quotes, option-chain hydration, broker `what-if`
previews, and paper/canary evidence before submitting capital. Research may run
for days or weeks, but every individual backtest must publish an ETA and remain
bounded to 20 minutes; long research is a resumable sequence of useful runs,
not one opaque multi-hour process.

Candidate ideas include:

- a precisely defined opening-volatility reversal that observes an early
  breakdown/bear trap and participates in a causal reclaim rather than
  hindsight-labeling the day's low;
- a separate opening continuation candidate for days where no reversal is
  confirmed;
- highly selective, defined-risk premium selling when implied compensation
  demonstrably exceeds realized and execution risk;
- bounded debit structures for asymmetric directional alpha;
- rare, heavily filtered material-news risk used first as a defensive veto or
  sizing input, never as an unverified prediction engine.

The desired blend is not one universal strategy. It is a small set of real
champions, separated by horizon and risk sleeve, that share one canonical data,
economics, execution, and evaluation spine.

---

## 1. Non-negotiable truth

### 1.1 Financial truth

- XSP is the one-tenth-scale Mini-SPX options product; signals may come from
  authentic index/market evidence, but the live package is an XSP option
  structure—not fictional "XSP spot."
- "Defined risk" does not mean "safe." A one-point XSP vertical has USD 100 of
  gross width exposure before premium, fees, and settlement effects; that is
  material relative to USD 1,000.
- No naked short options, uncovered tail risk, martingale sizing, averaging
  down, or correlated package stacking.
- No live order because a deadline, milestone, or leaderboard row demands one.
  `NO_TRADE` is a valid and often superior decision.
- Broker qualification, trading permissions, quote freshness, buying-power
  effect, maximum loss, commission, liquidity, settlement, and current account
  state are admission evidence—not assumptions.
- XSP's European-style, cash-settled exercise removes early-assignment risk but
  does not remove expiration, settlement, gap, liquidity, or operational risk.
- Every result must distinguish:
  - package economics;
  - per-leg marks and fills;
  - broker/account P&L;
  - fees and cash movements.
  A prior live XSP investigation proved why: a combo-parent liability-like mark
  could look roughly `-97` while leg-reconciled economic P&L was only a few
  dollars. False labels must never train promotion logic.

### 1.2 Research truth

- Synthetic option prices are useful for hypothesis search and stress tests,
  but synthetic-only evidence cannot promote a strategy to live capital.
- No look-ahead, revised-data leakage, same-bar clairvoyance, end-of-day
  hindsight, future-aware strike selection, or optimistic fill assumption.
- A best parameter point is not a champion. Prefer stable parameter plateaus,
  walk-forward survival, lower confidence bounds, and adverse-condition
  resilience.
- One, two, and five years are different evidence roles:
  - **1 year:** recent development and execution relevance;
  - **2 years:** out-of-sample and changing-condition robustness;
  - **5 years:** selective stress coverage only where authentic,
    structurally comparable data exists.
- Never silently mix synthetic chains, live captured chains, delayed quotes,
  adjusted underlying bars, or different session calendars.

### 1.3 System truth

- Backtest, replay, paper, preview, and live must consume the same canonical
  scenario, option-package economics, strategy decision, and execution-policy
  vocabulary.
- Context/regime evidence may classify, stratify, veto, and score candidates.
  It may not become another unobservable authority that silently rewrites a
  strategy.
- Promotion must be hysteretic and slow; risk shutdown must be immediate.
- Never switch a strategy owner while its package is open. Rescue/exit policy
  may override only to reduce risk.
- Caches are evidence: every artifact needs provenance, content identity,
  covered intervals, and gap semantics. A missing day must hydrate that day,
  not refetch six months or masquerade as a complete window.
- Restart and reconciliation behavior is part of correctness. A profitable
  strategy that cannot recover state is not promotable.

---

## 2. One canonical learning loop

```text
authentic market + option evidence
                │
                ▼
      normalized, provenance-bound tape
                │
                ▼
 deterministic scenario + canonical package economics
                │
          ┌─────┴─────┐
          ▼           ▼
  SAFE-INCOME      ALPHA-HUNT
   champions        champions
          └─────┬─────┘
                ▼
 shadow tournament + context-stratified evaluator
                │
                ▼
 eligibility → admission → execution policy → IBKR
                │
                ▼
 quotes / previews / fills / positions / realized economics
                │
                └──────────────► drift + calibration + leaderboard
```

The loop owns seven durable receipts:

1. **Data receipt:** source, timestamp, session, rights, completeness, gaps,
   adjustment policy, and content fingerprint.
2. **Research receipt:** hypothesis, frozen configuration, train/test boundary,
   run ETA, wall time, cache identity, and causal outputs.
3. **Economics receipt:** canonical package, maximum profit/loss, break-evens,
   fees, modeled buying power, and exit assumptions.
4. **Broker-preview receipt:** qualified contracts, NBBO/Greeks, `what-if`
   commission, margin/buying-power response, warnings, and freshness.
5. **Execution receipt:** arrival quote, every price revision, fill latency,
   partial/no-fill path, price improvement/adverse selection, and terminal
   reconciliation.
6. **Drift receipt:** expected versus observed signal, price, fill, P&L, Greeks,
   and state-machine behavior.
7. **Promotion receipt:** candidate evidence, incumbent comparison, confidence
   margin, risk budget, decision, expiry, and rollback trigger.

No separate live-only arithmetic or backtest-only policy copy may define these
truths.

---

## 3. Strategy topology

### 3.1 Safe-income sleeve

First candidate family:

- XSP defined-risk vertical credit spreads;
- one package at a time initially;
- expiry, width, strike distance, entry time, and exit policy selected from
  evidence rather than hardcoded folklore;
- credit after all commissions and likely slippage must justify width risk;
- skip on stale/wide markets, insufficient depth, event risk, missing Greeks,
  preview disagreement, or weak statistical edge.

Iron condors are a later extension, not the default. Four-leg friction,
asymmetric fill risk, and doubled surface assumptions must beat the simpler
vertical **net of all costs** before a condor is eligible.

Initial hypotheses to falsify:

- short-dated but not automatically 0DTE spreads may provide a better
  liquidity/gamma/overnight compromise than either same-day or long-duration
  structures;
- event-filtered and volatility-compensated entry may dominate mechanically
  selling every day;
- patient entries with a strict minimum-credit/edge floor may outperform
  relentless entry chasing;
- risk-off exits may justify faster escalation than entries.

### 3.2 Alpha-hunting sleeve

First candidate family:

- long call/put or debit vertical packages with loss bounded by debit;
- opening-range reversal/bear-trap hypothesis;
- separate opening continuation hypothesis;
- no "low of day" or "peak of day" labels available at decision time;
- profit-taking and rescue behavior derived from excursion, volatility, time
  decay, and execution evidence.

Operational bear-trap research must define, before running:

- observation window;
- breakdown threshold;
- reclaim threshold and minimum persistence;
- volume/volatility/breadth evidence that is actually available at that time;
- earliest legal entry timestamp;
- maximum wait and explicit invalidation;
- contract selection without future IV or strike knowledge.

### 3.3 Horizon families

Maintain distinct evidence and crowns:

- **HF:** intraday/opening-window opportunities, execution-sensitive;
- **LF:** multi-session directional or premium opportunities, gap- and
  carry-sensitive.

At most one admitted champion per `sleeve × horizon × declared context` begins
in shadow evaluation. Combination requires portfolio-level tail correlation,
capital reservation, and conflict-resolution evidence; it is not permitted
merely because each component backtested well alone.

---

## 4. Context adaptation without an opaque regime router

Regime work was directionally valuable but previously threatened to become a
second strategy authority. Preserve the valuable part as explicit evidence:

- volatility level and change;
- trend/range strength;
- gap/opening behavior;
- liquidity and spread quality;
- scheduled-event proximity;
- realized-versus-implied volatility;
- session and time-to-expiry;
- recent live/backtest drift.

Use it in three observable places:

1. **Stratified evaluation:** show where each fixed strategy wins, loses, and
   abstains.
2. **Eligibility/veto:** a champion declares the contexts it proved; outside
   them it becomes ineligible.
3. **Shadow selection:** challengers are scored beside the incumbent on the
   same tape without controlling capital.

Only after sufficient shadow evidence may a selector affect live admission. It
must then have:

- a small closed set of already-admitted champions;
- minimum tenure and promotion/demotion margins;
- sample-size and confidence gates;
- drawdown and drift shutdowns;
- deterministic reason codes;
- `NO_TRADE` fallback;
- no mid-position owner swap;
- full replay from its decision journal.

This is evidence-driven champion rotation, not speculative regime routing.

---

## 5. Current baseline and known gaps

### 5.1 What already exists

- Canonical multi-leg option-package economics cover vertical credit/debit
  spreads, butterflies, iron butterflies, iron condors, and generic
  defined-risk combinations.
- Live XSP/SPX BAG admission fails closed behind broker preview, canonical
  identity, maximum-loss, capacity, status, and minimum-credit checks.
- Active XSP package capacity is reserved from maximum loss rather than
  optimistic buying-power guesses.
- The live UI can qualify, quote, preview, stage, submit, close, and reconcile
  atomic option packages.
- The execution engine has centralized `OPTIMISTIC → MID → AGGRESSIVE → CROSS
  → RELENTLESS` behavior, including a delayed relentless mode.
- Spot research already contains resumable Cartesian sweeps, stability filters,
  promotion artifacts, and HF/LF historical lessons worth reusing.

### 5.2 What is not yet proof

- There is no established XSP research crown or authenticated XSP historical
  artifact in this repository.
- The present options backtest primarily derives contracts and prices from
  underlying bars plus synthetic Black-Scholes/Black-76 surfaces. It is not an
  authentic multi-year XSP NBBO replay engine.
- The existing options grid begins from a USD 10,000 default and includes
  undefined-risk families that are inadmissible for this USD 1,000 mission.
- Current option ranking is too close to raw P&L/win-rate ordering; it does not
  yet establish calibrated, cost-adjusted, walk-forward reliability.
- Option realism still needs explicit evidence for commissions, bid/ask
  dynamics, partial/no fills, cancel/replace latency, assignment/settlement,
  quote staleness, Greeks drift, and broker buying-power parity.
- The option engine currently limits concurrency simply; portfolio sleeve
  interaction and capital reservation require measured design.
- Existing chase timings are a capable mechanism, not proof that their
  durations are optimal for XSP entry, rescue, and exit intents.

### 5.3 First architectural rule

Do not build a parallel "new XSP engine." Extend the existing canonical owners:

- package economics;
- order admission and reservation;
- execution policy and journals;
- cache interval/provenance ownership;
- research sweep and promotion machinery;
- live/backtest drift contracts.

Centralize a shared truth once. Remove or absorb duplicate semantics as each
slice proves parity.

---

## 6. Data and realism plan

### 6.1 Underlying and context tape

Hydrate exact interval coverage for:

- 1-year recent development;
- 2-year robustness;
- selective 5-year stress;
- exchange calendar, RTH/GTH identity, daylight-saving behavior, holidays, and
  early closes;
- volatility/context features whose timestamps prove they were observable.

Cache reads must union valid existing slices and request only missing intervals.
Every requested range must return an explicit completeness receipt.

### 6.2 XSP option tape

Use IB Gateway now to determine and record what can be captured legitimately:

- qualified XSP contract identifiers;
- expiries and strikes available at each timestamp;
- bid/ask/last, sizes, model Greeks, implied volatility, quote timestamp, and
  market-data type;
- underlying/index reference;
- trading-class, multiplier, exchange, session, and settlement metadata;
- broker preview commission and buying-power effect.

Do not assume IBKR provides a complete multi-year historical option-chain tape.
If it does not, separate:

- **forward authentic capture** for ongoing evaluation;
- **approved historical provider data** if acquired;
- **synthetic/calibrated research** clearly labeled as model evidence.

Synthetic evidence can prune ideas; only authentic replay, broker preview,
paper/canary behavior, and drift receipts can graduate them.

### 6.3 Fill and economics realism

Replay must model or bound:

- combo versus leg execution behavior;
- NBBO width and quote age;
- limit-price queue uncertainty;
- partial/no fill and cancel/replace;
- commissions and minimums per package/leg;
- price-chase revisions and arrival-price slippage;
- expiry and cash-settlement timing;
- risk/margin reservation;
- missing data and disconnects;
- overnight and scheduled-event gaps.

Every optimistic unknown gets an adverse sensitivity run, not a silent zero.

---

## 7. Backtest and leaderboard discipline

### 7.1 Search sequence

1. Establish a do-nothing benchmark and simple fixed-policy baselines.
2. Define a small causal hypothesis family before widening combinations.
3. Reuse immutable tapes and precomputed features across combinations.
4. Use walk-forward folds with locked final holdouts.
5. Rank stable neighborhoods, not isolated maxima.
6. Stress spread, commission, fill probability, delay, IV error, and gap risk.
7. Bootstrap days/trade order for drawdown and ruin distributions.
8. Re-run finalists on authentic option replay or forward-captured tapes.
9. Submit only qualified finalists to shadow/paper evaluation.

### 7.2 Champion score

No single metric may crown a champion. The promotion score must expose:

- net expectancy after all modeled costs;
- lower confidence bound on expectancy;
- maximum drawdown and duration;
- tail loss/CVaR and estimated ruin probability;
- profit factor and payoff asymmetry;
- turnover, no-fill rate, and capital occupancy;
- stability across folds, years, parameters, and market contexts;
- dependence on a handful of days/trades;
- execution sensitivity;
- backtest-to-live drift;
- calibration of predicted versus observed outcomes.

Win rate and total P&L remain diagnostics, not authorities.

### 7.3 Runtime contract

- Every run prints ETA before expensive work.
- No individual run exceeds 20 minutes.
- Large searches checkpoint and resume.
- Report cold, warm, and small-delta timings separately.
- Warm repeat and partial-delta runs should approach instant reuse.
- Cache hits must prove semantic compatibility, not merely matching filenames.
- Kill or narrow a low-information run early; do not spend hours confirming a
  dominated hypothesis.

---

## 8. Live execution and safety profiles

### 8.1 Intent-aware price chasing

Measure and tune the existing centralized ladder rather than adding another
executor:

- **income/debit entry:** start patient; never chase beyond a frozen
  minimum-credit or maximum-debit edge boundary;
- **profit-taking exit:** patient while risk remains bounded and edge persists;
- **risk exit/rescue:** escalate faster when model risk, time-to-expiry,
  liquidity, or drift justifies urgency;
- **relentless mode:** a bounded terminal mechanism, not permission to abandon
  package economics.

For every attempt record:

- decision and arrival quote;
- theoretical/fair package value;
- each limit revision and elapsed time;
- NBBO movement and quote freshness;
- fill/no-fill/partial-fill;
- price improvement or adverse selection;
- commission and final economic result.

### 8.2 Initial USD 1,000 envelope

Exact limits are derived from evidence and broker preview, then frozen before a
canary. Until then:

- maximum one open XSP package;
- defined maximum loss only;
- no overlapping short-premium structures;
- no undefined-risk or naked legs;
- no automatic size increase after a loss;
- no 0DTE short-premium exposure without dedicated proof;
- no overnight exposure without dedicated gap/settlement proof;
- daily and weekly loss shutdowns;
- quote-age, spread-width, permission, commission, margin, and preview-drift
  gates;
- immediate fail-closed on reconciliation ambiguity or stale account state.

The first live unit must be the smallest broker-supported package that still has
positive expected value after friction. If no such unit fits the risk envelope,
the correct live allocation is zero.

### 8.3 Graduation ladder

```text
deterministic unit/economics proof
  → historical causal backtest
  → walk-forward + stress
  → authentic replay
  → live broker preview
  → shadow decision
  → paper execution
  → restart/reconciliation drill
  → one tightly bounded live canary
  → repeated canary
  → cautious size or strategy expansion
```

Skipping a rung requires written evidence that it is inapplicable, not
convenience.

---

## 9. Continuous evaluation and promotion

All eligible champions replay the same normalized tape in shadow, even while
only one strategy—or none—controls capital.

At every decision:

- log each champion's signal, abstention, package, expected edge, risk, and
  confidence;
- compare the selected decision with counterfactual champions without
  fabricating fills;
- reconcile actual quotes, previews, fills, and P&L;
- attribute drift to data, signal, option pricing, execution, fees, state, or
  market change;
- update evidence only after the outcome horizon closes;
- avoid learning twice from overlapping outcomes.

Promotion rules:

- challenger is already admissible and fully replayable;
- evidence spans minimum samples and more than one context;
- lower confidence bound beats the incumbent by a frozen margin;
- improvement survives costs, stress, and locked holdout;
- selector change respects minimum tenure/hysteresis;
- no open position is re-owned;
- rollback trigger is predeclared.

Demotion rules:

- hard safety violation: immediate disable;
- material model/live drift: quarantine;
- statistical decay: slow demotion after confidence threshold;
- temporary data outage: fail closed, do not rewrite the leaderboard.

---

## 10. Hierarchical questchain

### Phase 0 — Anchor reality `[WIP]`

- [x] Re-read repository, ledger, existing leader archives, and canonical option
      owners from baseline `2551326`.
- [x] Connect read-only to the available IB Gateway; record account,
      permissions, market-data type, XSP contract identity, multiplier,
      exchange, valid expiries, and session facts.
- [x] Inventory every underlying, option, feature, and calibration cache,
      including provenance, intervals, gaps, and authenticity.
- [x] Smoke-test the existing options runner before trusting any output; record
      every generated family, count, runtime, exclusion, and failure.
- [~] Produce one canonical XSP vertical and one iron-condor economics receipt
      from known legs; prove maximum profit/loss and package/leg/account
      attribution. Vertical complete; condor remains pending.
- [~] Produce broker `what-if` previews for the smallest realistic packages;
      do not submit. Contract and status proof exists, but commission and
      buying-power fields were absent and therefore do not satisfy admission.
- [ ] Freeze the initial research metrics, walk-forward boundaries, stress
      matrix, and run-time budget.
- [ ] Accumulate at least 2–4 hours of meaningful research/backtesting evidence
      before live-capital eligibility.

**Phase exit:** data and broker truth are known; current simulator limitations
are quantified; no option result is mislabeled as authentic.

#### Phase 0 measured truth — 2026-07-24

**Broker and contract**

- IB Gateway server version `176` was reached through isolated probe clients;
  every client disconnected after its receipt and no order was submitted.
- The account identity is retained only as fingerprint `bddaf5682a86`.
  Broker values were denominated in AUD and were above the deliberately
  conservative USD 1,000 design envelope. Exact balances are not persisted in
  the repository. The snapshot had seven positions, zero XSP positions, and no
  visible open orders.
- XSP qualified as index contract `137851301`, CBOE, USD, multiplier `100`.
  The chain exposed `45` expirations and `509` strikes across CBOE, SMART, and
  IBUSOPT parameters.
- Real-time market data was not subscribed. Explicit delayed data supplied XSP
  close `740.83` and historical index bars. Pre-RTH option snapshots qualified
  the legs but supplied no usable bid, ask, last, or Greeks; repeat capture
  during RTH remains mandatory.

**Underlying and cache**

- The existing canonical historical loader already owns interval-aware sparse
  hydration: it reuses covered bars and fetches only missing ranges. Do not
  duplicate that behavior.
- XSP 5-minute RTH history now covers `2025-07-24` through `2026-07-23`:
  `19,506` valid bars over `251` sessions, no missing session ranges, `249`
  normal sessions with `78` bars and two early closes with `42`.
- Raw cache fingerprints:
  - `2026-07-01..2026-07-23`:
    `eaa6da6c015c8b26c9935c1ce3091902053e5daf5ba2818eaa98c0f566ca7e98`
  - `2025-07-24..2026-06-30`:
    `dfcf7bf27fbf75fc4536abeea71404c7edbbcdc32a61bdf77dd44130cc9c20db`
- All XSP index volume values are zero. Any opening-volume or bear-trap
  confirmation must use a separately sourced, provenance-bound context tape
  such as SPY, ES, or breadth; it must never reinterpret absent XSP volume.
- The local cache root is roughly `39 GiB`, dominated by a `25 GiB` packed
  series cache and `12 GiB` core-series database. No authentic historical XSP
  option-chain, NBBO, Greek, or fill tape exists yet.

**Existing options runner**

- The hardcoded smoke grid evaluates `12` strategy groups and `31,104`
  configurations against synthetic Black-Scholes/mid-edge option prices,
  USD `10,000` starting cash, no option commissions, and no realistic
  no-fill/slippage model.
- On the 16-session July slice, the cold run completed in about `60 s`; exact
  warm reuse completed in `1.78 s`. Economic outputs were byte-equivalent after
  removing generation time. Output fingerprint:
  `5588cf170959abfa7d6903e185695b06a264d49db1f8a73a486a263d58727667`.
- The synthetic leaders reported implausibly clean results, including several
  `100%` win rates over only `8–19` trades. They are speed/cache receipts, not
  candidate evidence.
- Current leg geometry uses percentage moneyness. A `1%` wing near XSP `740`
  creates roughly `7.4` points or USD `740` gross width—not the intended
  one-point, USD `100` package. A one-year full grid would therefore spend
  hundreds of millions of bar/config evaluations testing the wrong mission.

**Vertical preview**

- A canonical hypothetical `2026-07-27` XSP `734/735` put-credit vertical at
  USD `0.20` credit produced USD `20` maximum profit and USD `80` maximum loss.
- IBKR qualified both legs and returned `PreSubmitted` from `what-if`, but
  commission, margin change, equity-with-loan, and buying-power evidence were
  absent/sentinel values. The result repeated with a non-readonly probe, so it
  was not caused by read-only mode.
- Current admission policy deliberately allows this sparse preview. For this
  kata that is **insufficient live evidence**: broker status cannot substitute
  for bounded package economics, conservative fees, fresh quotes, permissions,
  and explicit account-capacity proof.

### Phase 1 — Authentic XSP data spine `[TODO]`

- [ ] Centralize interval-aware cache ownership and gap hydration.
- [ ] Establish authenticated 1-year and 2-year underlying/context tapes.
- [ ] Admit the 5-year window only for comparable, complete evidence.
- [ ] Capture forward XSP chains/NBBO/Greeks with provenance and restart safety.
- [ ] Bind synthetic calibration to explicit source/effective intervals.
- [ ] Add completeness and freshness gates consumed identically by research,
      replay, evaluation, and live admission.

**Phase exit:** identical evidence fingerprints can hydrate backtest, replay,
shadow, and live comparison without refetching complete cached ranges.

### Phase 2 — Candidate birth and causal tournament `[TODO]`

- [ ] Establish safe-income vertical baselines.
- [ ] Test whether iron condors add net value after four-leg friction.
- [ ] Formalize opening bear-trap reversal without hindsight.
- [ ] Formalize opening continuation as a separate candidate.
- [ ] Establish LF directional/premium baselines.
- [ ] Partition HF/LF and safe-income/alpha crowns.
- [ ] Encode `NO_TRADE` and event/liquidity vetoes.
- [ ] Remove dominated or redundant candidates; keep the frontier compact.

**Phase exit:** a small, interpretable candidate frontier exists; every
candidate declares its evidence, contexts, risk, and invalidation.

### Phase 3 — Realistic backtest and promotion receipts `[TODO]`

- [ ] Complete bounded 1-year development runs.
- [ ] Complete locked 2-year robustness/walk-forward runs.
- [ ] Complete selective 5-year stress only where authentic.
- [ ] Apply cost, fill, latency, IV, spread, gap, and missing-data stresses.
- [ ] Bootstrap confidence/drawdown rather than trust point P&L.
- [ ] Test stable parameter neighborhoods.
- [ ] Re-evaluate finalists on authentic option evidence.
- [ ] Produce explicit promote/hold/reject receipts.

**Phase exit:** at least one candidate—or an honest `none`—meets the frozen
research gate without hidden optimistic assumptions.

### Phase 4 — Shadow, preview, and paper `[TODO]`

- [ ] Run every admitted champion on the same live tape.
- [ ] Compare expected and broker-preview economics.
- [ ] Calibrate patient/aggressive/relentless execution by intent.
- [ ] Exercise reject, disconnect, stale quote, partial/no-fill, and timeout
      paths.
- [ ] Restart during an open paper package and prove exact reconciliation.
- [ ] Measure daily live/backtest drift and root-cause every material delta.

**Phase exit:** paper decisions, execution, accounting, restart, and evaluator
receipts agree with canonical expectations.

### Phase 5 — Tightly bounded live canary `[BLOCKED: Phases 0–4]`

- [ ] Freeze package, maximum loss, maximum debit/minimum credit, daily/weekly
      shutdowns, allowed session, chase ceiling, and rollback triggers.
- [ ] Re-read account/permissions/capacity and obtain a fresh broker preview.
- [ ] Submit at most one smallest eligible XSP package.
- [ ] Observe and reconcile without strategy mutation.
- [ ] Exit according to the frozen policy or safety override.
- [ ] Publish complete economic and drift receipts.

**Phase exit:** the canary is fully reconciled and yields a truthful decision:
repeat, revise, hold, or stop.

### Phase 6 — Weekly self-healing kata `[TODO]`

- [ ] Schedule resumable research/evaluation cycles; "weekly" means a complete
      combinatorial/walk-forward refresh, not merely a calendar report.
- [ ] Hydrate new tapes and re-score every eligible champion.
- [ ] Run challengers in shadow before promotion.
- [ ] Apply hysteretic promotion/demotion and risk shutdowns.
- [ ] Update HF/LF and safe-income/alpha leaderboards.
- [ ] Publish model/live drift, P&L, drawdown, calibration, and remaining risks.
- [ ] Compound only after repeated evidence; never mechanically scale by recent
      profit.

**Phase exit:** one full weekly cycle is reproducible, restart-safe, and makes
an evidence-backed promote/hold/demote decision.

---

## 11. Time-anchored milestone receipts

Milestones are evidence anchors, not pressure to trade. If a weekend, holiday,
market-data outage, or absent setup prevents live evidence, record that truth
and advance only the lanes that remain valid.

### Within 24 hours

- exact repository and management state persisted;
- Gateway read-only connection and XSP contract facts verified;
- account premise and market-data type recorded;
- cache/data authenticity inventory complete;
- existing options runner smoke-tested;
- one simple XSP vertical economics + broker-preview receipt;
- first underlying/context baseline and at least one causal candidate run;
- all runs carry ETA and remain under 20 minutes;
- current gaps, blockers, and next 24-hour sequence written here.

**Live-capital requirement:** none.

### Within 48 hours or two eligible market sessions, whichever is later

- forward XSP quote/chain capture survives restart;
- safe-income and alpha baselines replay on the same normalized tape;
- first walk-forward/stress comparison exists;
- chase modes have preview/paper measurements by intent;
- paper/shadow evaluation produces package/leg/account attribution;
- disconnect, stale-data, rejection, and recovery paths have receipts;
- first provisional promote/hold/reject ranking exists with uncertainty.

**Live-capital requirement:** none; canary remains conditional.

### Within 1 week or five eligible market sessions, whichever is later

- five-session shadow ledger is complete;
- all eligible champions were re-hydrated and scored consistently;
- at least one safe-income and one alpha hypothesis received an honest verdict;
- backtest/live drift is attributed and bounded;
- execution quality, fees, fill probability, drawdown, and calibration are
  reported;
- restart and reconciliation remain exact;
- risk shutdowns were exercised in paper/replay;
- a formal `PROMOTE`, `HOLD`, `REVISE`, or `STOP` decision is issued;
- any live canary occurred only if every prior gate passed and is fully
  reconciled;
- the remaining-risk register and next weekly kata are frozen.

**Success is not "profitable every day."** Success is a repeatable evidence
loop, preserved capital discipline, and a strategy decision we can defend.

### Four-week mastery extension

One week can validate machinery and expose obvious drift; it cannot establish
extreme reliability. Continue for four or more weeks before materially scaling:

- multiple volatility/context states;
- repeated walk-forward and shadow promotions;
- stable lower-confidence expectancy after real friction;
- bounded and recovered drawdowns;
- no unexplained accounting/reconciliation drift;
- promotion decisions that outperform simple fixed-policy baselines.

---

## 12. Active task tree

1. **Phase 0.1 — Management and truth freeze `[DONE]`**
   - this artifact created from the full mission;
   - goal points here as the canonical resume source;
   - repository, Gateway, contract, data, and runner receipts recorded.
2. **Phase 0.2 — XSP broker and account census `[WIP]`**
   - contract, account currency, delayed data, chain, and sparse preview proven;
   - next: fresh RTH option quotes/Greeks and complete preview economics.
3. **Phase 0.3 — Backtest authenticity census `[DONE]`**
   - cache inventory and sparse hydration verified;
   - current runner cold/warm smoke measured;
   - synthetic-only boundary and incorrect spread geometry frozen.
4. **Phase 0.4 — Metrics and risk freeze `[WIP]`**
   - scorecard;
   - walk-forward;
   - stress matrix and realistic one-point XSP economics;
   - conservative capacity and canary eligibility.
5. **Phase 1 — Data spine `[TODO]`**
6. **Phase 2 — Candidate frontier `[TODO]`**
7. **Phase 3 — Research tournament `[TODO]`**
8. **Phase 4 — Shadow/paper `[TODO]`**
9. **Phase 5 — Bounded live canary `[BLOCKED]`**
10. **Phase 6 — Weekly self-healing loop `[TODO]`**

---

## 13. Evidence registry

Add rows; never rewrite an unfavorable receipt.

| ID | Time | Phase | Evidence | Source / artifact | Fingerprint | Result |
|---|---|---|---|---|---|---|
| E-000 | 2026-07-24 | 0.1 | Repository baseline | Git `main` | `2551326` | Clean code anchor; management added at `3c38af6` |
| E-001 | 2026-07-24 06:35 UTC | 0.2 | Gateway/account census | IB Gateway `176` | account `bddaf5682a86` | AUD account above conservative design envelope; 7 positions; no XSP/open order |
| E-002 | 2026-07-24 | 0.3 | XSP data/cache census | `db/XSP/*.csv` | `eaa6da6…`, `dfcf7bf…` | 19,506 bars/251 sessions/no gaps; volume absent; no option replay tape |
| E-003 | 2026-07-24 | 0.3 | Options-runner smoke | `/tmp/xsp-options-smoke.json` + result cache | `5588cf17…` | 31,104 cells; ~60 s cold/1.78 s warm; synthetic, cost-free, wrong wing geometry |
| E-004 | 2026-07-24 | 0.2 | XSP vertical broker preview | 734/735P `20260727`, no submit | qualified conIds + sparse `PreSubmitted` | Canonical max +20/-80; broker commission/capacity proof missing; not live-eligible |
| E-005 | pending | 0.4 | Frozen score/risk contract | pending | pending | pending |

---

## 14. Decision journal

| ID | Decision | Why | Revisit when |
|---|---|---|---|
| D-001 | XSP only | Cheapest bounded-risk mastery must precede product sprawl | One full weekly loop is exact and stable |
| D-002 | Separate safe-income and alpha sleeves | Their payoff, execution, and failure modes differ | Portfolio interaction is measured |
| D-003 | No opaque regime router | Context should explain eligibility, not become hidden strategy authority | Shadow selector proves hysteretic superiority |
| D-004 | Synthetic evidence cannot promote live capital | Current options history is model-derived, not authentic NBBO replay | Authentic replay and live drift agree |
| D-005 | Broker preview before paper/live admission | Package identity, fees, margin, and warnings are broker facts | Never |
| D-006 | Minimum 2–4 hours research before live eligibility | Prevent milestone-driven premature submission | Never |
| D-007 | Every run ETA; 20-minute hard bound | Keep research observable and resumable | Only with explicit written reason |
| D-008 | No strategy-owner swap on an open package | Avoid incoherent lifecycle and attribution | Never; rescue may only reduce risk |
| D-009 | One package initially | USD 1,000 makes correlation/capacity mistakes material | Repeated canaries prove headroom |
| D-010 | Iron condor must beat vertical net of four-leg friction | Complexity is not value | Authentic execution proves value |
| D-011 | Rare news begins as defensive veto/sizing evidence | Causality and historical availability are fragile | Forward evidence proves incremental value |
| D-012 | `NO_TRADE` is a first-class champion action | Capital preservation outranks activity | Never |
| D-013 | Keep a USD 1,000 design envelope despite a higher broker snapshot | Account values drift and are AUD-denominated; the mission should remain robust under the user's conservative premise | User explicitly changes the envelope after a fresh broker census |
| D-014 | Do not run the current one-year 31K-cell options grid | It omits realistic friction and maps `1%` to about 7.4 XSP points, so scale would amplify invalid evidence | Canonical point geometry, costs, and fill realism are proven |
| D-015 | Missing XSP index volume is not zero-volume evidence | The index tape does not carry authentic volume | A separately provenance-bound context tape is admitted |
| D-016 | Sparse `what-if` status alone is not enough for this live canary | `PreSubmitted` did not return usable commission, margin, or buying-power effects | Canonical risk plus conservative fees, fresh quotes, permissions, and capacity are jointly proven |

---

## 15. Prospective capability births

Encode these into the capability ledger only when implementation begins; reuse
an existing contract where it already owns the outcome.

- `benchmark.future.xsp-data-authenticity`
- `integration.replay.future.xsp-option-chain-replay`
- `benchmark.future.xsp-champion-tournament`
- `benchmark.future.xsp-live-backtest-drift`
- `e2e.live.future.xsp-bounded-canary`
- reuse the existing shadow-order digital-twin, walk-forward overfit-defense,
  package-economics, live admission, and restart-recovery capabilities.

The ledger must drive system behavior, not become a substitute for it.

---

## 16. Resume and stopping protocol

At the beginning of every resumed round:

1. Read this file completely.
2. Verify repo path, `HEAD`, worktree, remote, Gateway state, and current time.
3. Resume the first `[WIP]`, then the highest unblocked `[TODO]`.
4. Revalidate time-sensitive account, quote, and session facts.

Before stopping or changing phase:

1. Update the active task tree.
2. Append evidence and decisions.
3. Record exact commands/artifacts, fingerprints, and results.
4. State what remains unknown; do not convert assumptions into memory.
5. Preserve partial data/cache work atomically and restartably.
6. Keep the repository and this management brain synchronized.

No objective completion is valid until the one-week receipt exists and the
system issues an honest promote/hold/revise/stop verdict with its remaining-risk
register.

---

## 17. Authoritative product anchors

- Cboe XSP product overview:
  <https://www.cboe.com/tradable_products/sp_500/mini_spx_options>
- Cboe XSP contract specifications:
  <https://www.cboe.com/tradable_products/sp_500/mini_spx_options/specifications>
- IBKR API order types and combination-order behavior:
  <https://ibkrcampus.com/campus/ibkr-api-page/order-types/>
- IBKR Client Portal API combination and `what-if` behavior:
  <https://ibkrcampus.com/campus/ibkr-api-page/cpapi-v1/>
- IBKR TWS API and Gateway documentation:
  <https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/>

---

## Conclusion

- **Quest:** XSP Mastery — Live Research Kata
- **Current status:** Phase 0.4 realistic economics, score, and risk freeze
- **Next action:** centralize one-point XSP package geometry and realistic
  fees/fills, then require conservative package-plus-broker capacity evidence
  before any candidate can enter the real research tournament.

**Predictive observation:** the first major gain is likely to come from making
the experiment truthful, not adding a signal. Correct one-point geometry,
authentic option evidence, explicit friction, and strict abstention can reverse
the current synthetic leaderboard before any indicator changes.
