# XSP Mastery — Live Research Kata

- **Status:** `[WIP] Phase 1 — authentic XSP evidence spine`
- **Management role:** canonical task tree, evidence ledger, decision journal, and resume source
- **Code baseline:** `25513267225908b7288530c1ec0762f7656bbf5b`
- **Resumable pushed anchor before current WIP:** `b8ce7b90096a4b47dfe89c30db85623017a10fc8`
- **Management brain introduced:** `3c38af635fcc6ce8b3b0a88e1f2de567345d1bf0`
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

The concrete economic objective is stricter than infrastructure readiness: the
selected strategy must finish one complete 24-hour shadow/paper run and one
complete five-session week net positive after all applicable costs, inside
frozen drawdown limits, with package/leg/account attribution. Neither a passing
calibration benchmark nor correct abstention alone completes those targets.

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
- a causal opening-state matrix over separately frozen 30-, 60-, 90-, and
  120-minute observation windows: test whether an extreme upside extension
  tends to fade through the remaining session, whether an extreme downside
  liquidation tends to rebound quickly and/or continue recovering slowly, and
  whether the middle state should abstain. These are hypotheses to falsify,
  never assumed daily laws;
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

The broader opening-state family must remain a compact causal classifier, not a
new regime router:

- freeze the observation boundary before seeing the outcome;
- measure opening return, range expansion, excursion, gap, realized volatility,
  and separately provenance-bound breadth/volume context when available;
- classify only explicit `UPSIDE_EXTENSION`, `DOWNSIDE_LIQUIDATION`, or
  `NEUTRAL/AMBIGUOUS` facts;
- test fade, fast rebound, slow rebound, and continuation as separate strategy
  owners with separate attribution—never silently swap their meaning;
- compare each branch with `NO_TRADE`, unconditional time-of-day baselines, and
  stable neighboring thresholds;
- keep direction, option structure, exit horizon, and execution policy outside
  the classifier so backtest, shadow, and live can share the same facts.

Reuse the existing capability spine instead of creating a parallel framework:

- `signal-regime-intelligence` owns causal opening facts/classification;
- `research-optimization-calibration` owns branch/threshold tournaments;
- `market-realism-parity` proves identical live and replay classification;
- `backtest-simulation-accounting` owns fill/P&L consequences;
- `live-execution-orders` owns the admitted package and chase lifecycle.

**Frozen opening-state discovery contract**

- source: canonical XSP 5-minute RTH bar-close tape;
- research cutoff: `2026-01-22`; the family-specific
  `2026-01-23..2026-07-23` holdout remains unread;
- observation boundaries: 30, 60, 90, and 120 minutes after 9:30 a.m. ET;
- state: boundary close return is compared only with the prior 60 complete
  sessions for that same boundary; lower/upper empirical 10%, 15%, 20%, and
  25% tails define downside liquidation/upside extension, otherwise abstain;
- branches: upside fade, upside continuation, downside rebound, and downside
  continuation remain separately attributed;
- outcomes: 15-, 30-, and 60-minute forward return plus session close;
- temporal stability: report `2024-10-24..2025-07-23` and
  `2025-07-24..2026-01-22` separately;
- discovery gate: at least 30 total observations, at least 12 per temporal
  block, positive means in both blocks, positive ordinary 95% lower bound,
  positive family-wise Bonferroni lower bound, and support from neighboring
  tail thresholds. No option or live promotion follows directly from this
  underlying event study.

**Frozen causal-context extension — `xsp.opening_context.study.v1`**

Registered before reading contextual outcomes. Reuse the same discovery
cutoff, holdout, rolling 60-session reference, observation boundaries, XSP
tails, branches, horizons, temporal blocks, and gates above. Join only exact
same-timestamp, provenance-bound 5-minute RTH bars from XSP, SPY, and VIX.
At each frozen boundary, compute SPY cumulative volume relative to the median
of the prior 60 complete sessions at that boundary and VIX return from that
session's open. Test exactly three causal contexts per branch:

- downside branches: SPY participation at or above its rolling median; VIX
  rising; and both together;
- upside branches: SPY participation at or above its rolling median; VIX
  falling; and both together.

This is one 768-cell family (`4 windows × 4 tails × 4 branches × 4 horizons ×
3 contexts`) with one family-wise correction. A contextual cell must also beat
its unconditional parent mean and retain neighboring-tail support. Missing or
misaligned context abstains. No combinations, thresholds, or holdout outcomes
may be added after results are read under this version.

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
IBKR explicitly excludes expired options and option EOD data from historical
retrieval and does not store native combo history. Therefore a weekly or 0DTE
XSP contract disappears as historical evidence after expiry; current
longer-dated contracts expose only their own finite listed life, not a
chain-as-of archive. Separate:

- **forward authentic capture** for ongoing evaluation;
- **approved historical provider data** if acquired;
- **synthetic/calibrated research** clearly labeled as model evidence.

Synthetic evidence can prune ideas; only authentic replay, broker preview,
paper/canary behavior, and drift receipts can graduate them.

Multi-year XSP underlying RTH bars remain useful only after explicit coverage,
calendar, revision, and fingerprint checks. Their availability does not prove
after-hours completeness, historical NBBO, option-chain membership, or
executable spread economics. The XSP index tape has no authentic volume and
IBKR's index historical surface does not provide bid/ask bars; separately
proven context is mandatory for any volume, breadth, or overnight claim.

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

#### Frozen Phase-0 score/risk contract — `research.daily.v1`

The atomic statistical unit is a complete market session's close-to-close
economic equity change, including zero-activity sessions. Closed-package P&L
is net of the run's explicit commission and slippage model. This prevents a
high-frequency or highly selective strategy from improving its apparent sample
by omitting the days on which it abstained.

Every run exposes, without collapsing them into one magic number:

- sessions, active sessions, closed packages, and net P&L;
- mean daily P&L and its conservative normal `95%` lower bound;
- daily volatility, worst session, and worst-`5%` daily CVaR;
- maximum drawdown and net-P&L/drawdown;
- profit factor and payoff ratio;
- the share of gross wins contributed by the largest five trades;
- commission, holding time, and the existing strategy identity.

The per-fold evidence floor is `60` complete sessions and `30` closed
packages. Exploration ordering first asks whether that floor and a positive
daily lower bound both pass, then compares the lower bound, P&L/drawdown,
profit factor, low concentration, and net P&L. Win rate is deliberately absent.
This ordering is only a shortlist aid: **no synthetic row is promotable** and
no one-run ordering can crown a champion.

Freeze time boundaries before opening results:

1. **Recent 1-year development:** first `50%` of sessions for discovery, next
   `25%` for validation, final `25%` as a locked holdout. Any parameter or
   rule changed after seeing the locked holdout becomes a new candidate and
   restarts the split.
2. **2-year robustness:** run the unchanged candidate over four sequential
   half-year blocks. At least three of four and the final block must be net
   positive; pooled out-of-sample daily LCB must remain positive.
3. **Selective 5-year stress:** yearly/context blocks with no retuning; use
   only when tape provenance and product comparability are defensible.

Freeze the first friction matrix:

| Level | Commission / contract / side | Extra package-price slippage |
|---|---:|---:|
| Baseline | USD `1.00` | `1` tick |
| Adverse | USD `1.50` | `2` ticks |
| Severe diagnostic | USD `2.00` | `3` ticks |

Authentic replay must additionally stress no-fill, stale/wide quotes,
cancel/replace delay, IV error, gap/settlement behavior, and missing evidence.
Those cannot be fabricated from underlying-only synthetic bars.

Initial safe-income research gates, all required on validation and locked
evidence, are: net P&L positive after adverse friction; positive pooled daily
LCB; profit factor at least `1.20`; maximum drawdown no more than `15%` of the
USD `1,000` envelope; worst session no worse than `-10%`; and top-five wins no
more than `50%` of gross wins. The alpha sleeve may use a `20%` drawdown and
`1.10` adverse profit-factor floor, but retains the `-10%` worst-session
limit. These are rejection gates, not profit promises.

Preregister `xsp.directional-debit.discovery.v1` before inspecting outcomes:

- discovery remains `2025-07-24..2026-01-22`; validation and holdout stay
  sealed;
- compare filtered and unfiltered EMA-directed one-point verticals: BUY CALL /
  SELL the next higher CALL on up evidence, BUY PUT / SELL the next lower PUT
  on down evidence;
- search DTE `0/5/10/20`, anchors `0/0.5/1%`, profit targets
  `0.25/0.5/0.75/1`, stops `0.25/0.5/0.75`, EMA `3/7`, `9/21`, `20/50`,
  trend/cross entry, and fixed/eligible profitable-flip exit: `3,456` cells;
- use adverse USD `1.50` per contract per side plus two package-price ticks;
- a candidate must pass all alpha gates, remain positive-LCB in both exact
  chronological discovery halves, and have at least two immediate
  target/stop/EMA/DTE neighbors with positive LCB. Otherwise reject the family
  without reading validation. Do not add a permanent strategy catalog entry
  unless this discovery contract passes.

For the first possible live canary, canonical maximum loss plus conservative
round-trip fees must be no more than `10%` of the lesser of fresh usable
capacity and the USD `1,000` design envelope. Only one package may be open;
the first daily and weekly loss shutdowns are `10%` and `15%` respectively.
Fresh broker identity, permission, quotes, preview, capacity, and complete
paper/replay receipts remain mandatory and may impose stricter limits.

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

### 9.1 Frozen live-calibration benchmark

Reuse `benchmark.future.live-backtest-drift-score`; do not create an
XSP-specific parallel scorer. Each `live_calibration.v1` result is
content-addressed and append-only, with the forecast frozen before its outcome:

This benchmark governs and authenticates the live runs; it does not replace
their economic objective. Completion still requires an actual net-positive
24-hour selected-strategy run and an actual net-positive one-week/five-session
run after all applicable costs. A complete, well-calibrated benchmark attached
to a flat, losing, or abstaining run remains valuable evidence, but it is not
achievement of either profitable-run milestone.

```text
identity       strategy/version, tape/config fingerprints, capital sleeve
forecast       decision/no-trade, package, P&L distribution, risk, costs, fills
observed       package/leg/account P&L, drawdown, fills, chase, fees, margin
drift          data, decision, pricing, execution, economic and safety deltas
context        causal hourly/session facts and long-horizon state observations
counterfactual every eligible champion replayed on the exact same live tape
gates          evidence completeness, hard vetoes, calibration and uncertainty
verdict        PROMOTE | HOLD | REVISE | QUARANTINE | STOP
```

The benchmark is veto-first, not one opaque scalar. It reports five independent
axes: data/decision parity; execution and buying-power drift; net package P&L
and drawdown calibration; safety/tail behavior; and counterfactual opportunity
cost. Actual P&L is never sufficient by itself. A profitable but miscalibrated
fill can fail; a correctly predicted `NO_TRADE` session can pass.

Record causal market-state facts at decision and approximately hourly
boundaries—opening extension/liquidation/neutral, gap, realized volatility,
trend/excursion, context freshness, and material-event vetoes—so drift can be
explained across intraday, daily, and longer-horizon changes. These observations
do not directly swap strategies. Promotion/demotion runs only at frozen daily
and weekly cadences after minimum samples, with hysteresis and an open-position
ownership guard. This preserves responsiveness without recreating noisy regime
routing.

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
- [x] Freeze the initial research metrics, walk-forward boundaries, stress
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

**Corrected synthetic baseline**

- Canonical leg intent now supports additive `otm_offset_points`; percentage
  moneyness remains the short-strike anchor while both deterministic replay and
  live chain resolution derive a fixed-point wing from the same target.
- Synthetic option execution now keeps slippage in package-price ticks and
  commissions in cash/P&L, including liquidation marks, drawdown, and
  conservative credit-package capital. Research assumptions remain explicit
  CLI/config inputs.
- One mission-shaped receipt used USD `1,000`, one-point wings, USD `1.00`
  commission per contract per side, and one extra slippage tick:
  - full mixed grid: `31,104` cells, `1:21` cold;
  - safe-income-only grid: `4,608` cells, `12.3 s` cold and `<0.6 s` warm;
  - safe-income cold/warm semantic fingerprint:
    `2622c16b246a0ad3948e0eee212da705d8ba2a58858db07b52835b5c33eb20bf`.
- The safe-income sleeve contains only put-credit vertical and iron-condor
  families, filtered and unfiltered. It removes unreachable credit profit
  targets above `1.0`; legacy `all` remains available for reproduction.
- Repricing was material. For example, the best unfiltered put-credit row moved
  from synthetic `+504.73`, `100%` wins to `+51.43`, `75%`; the best
  unfiltered condor moved from `+1,056.49`, `100%` to `+121.64`, `80%`.
  This is not an ablation and proves no edge: geometry, capital, commissions,
  and slippage all changed together, while the sample still contains only
  16 sessions and synthetic options.

**Chronological discovery verdicts**

- The frozen discovery window is `2025-07-24..2026-01-22` (`126` complete
  sessions). Validation (`2026-01-23..2026-04-23`) and locked holdout
  (`2026-04-24..2026-07-23`) remain unopened because no family passed its
  discovery gates.
- The refreshed safe-income baseline evaluated `4,608` cells with USD `1.00`
  per contract per side and one slippage tick. Only three rows passed sample
  plus positive-LCB gates; all three are the same filtered `5`-DTE,
  two-percent-anchor iron-condor path repeated by stop settings that never
  activated: `30` trades, synthetic P&L `+106.83`, daily LCB `+0.5277`,
  drawdown `18.34`, top-five win share `31.5%`.
- The identical `4,608` cells under adverse friction—USD `1.50` per contract
  per side and two ticks—produced **zero** rows with both the sample gate and a
  positive daily LCB. The baseline condor therefore fails the frozen
  safe-income contract before validation, irrespective of its visually perfect
  synthetic baseline win rate.
- The first causal opening-reclaim alpha family evaluated `1,728` cells across
  opening windows, breakdown depth, reclaim persistence, deadline, DTE,
  moneyness, and exits. It produced zero positive daily LCBs; all `109`
  sample-qualified rows were negative. The most seductive low-sample row
  reported synthetic `+426.73` over only `16` trades, daily LCB `-0.1768`,
  `147.39` drawdown, and `67.4%` of gross wins in its top five wins. It is
  rejected, not promoted.
- Timestamp-audited replay proved those `16` entries exactly match `16` causal
  signal events on the canonical `9,756`-bar tape. At fixed 15-, 30-, and
  60-minute horizons, underlying-return 95% lower bounds were `-2.22`,
  `-7.49`, and `-11.74` bps; EOD was `-29.35` bps. The synthetic option result
  is path-dependent hypothesis evidence, not an underlying edge or live claim.
- Two generic simulator defects were found while challenging the result:
  warmup bars could previously enter trades before the requested scoring
  boundary, and every 0DTE option retained a full `6.5` hours of time value at
  every intraday bar. Execution/equity are now bounded to the requested window,
  while OPT valuation counts exact ET time to the 4:00 p.m. expiration close
  (`1:00 p.m.` on half-days). The result-cache namespace is bumped so stale
  economics cannot masquerade as current evidence.
- The preregistered two-year opening-state study evaluated `256` causal cells
  over `316` eligible sessions per boundary and produced **zero** family-wise
  discovery passes. The only ordinary positive 95% lower bound was a
  bottom-decile 90-minute liquidation followed by a 30-minute rebound:
  `42` events, mean `+6.93` bps, ordinary LCB `+0.19` bps, but Bonferroni LCB
  `-5.27` bps and no neighboring-threshold support. The family-specific
  `2026-01-23..2026-07-23` holdout remains unopened. This is a research hint,
  not a strategy or option-edge receipt.
- The preregistered SPY/VIX causal-context extension evaluated `768` additional
  cells without opening that holdout and also produced **zero** passes. Same-
  boundary SPY participation and VIX direction did not stabilize the original
  XSP states: every contextual cell retained a negative family-wise lower
  bound, and the best-looking cases lacked neighboring-tail support or durable
  later-block strength. No context strategy knob or selector was born.

Receipt fingerprints:

- safe-income baseline:
  `e4a325ffdb15532850a973811819fac3538364d132eaff815ed069e0c9aea733`;
- safe-income adverse:
  `801548592728c6c1ca14c2e3584bdcd3078a17a53677f1658d6b612b9f11953a`;
- opening-reclaim baseline:
  `dab6f0b9ac166027f08551e5207301ca781e25139ff3630abfba4bcd4bec7f41`.

### Phase 1 — Authentic XSP data spine `[WIP]`

- [x] Centralize interval-aware cache ownership and gap hydration. Existing
      slices are unioned; only contiguous holes are requested; each contract is
      serialized while independent tapes may hydrate in parallel; ambiguous
      empty/timeout responses fail closed rather than advancing the cursor.
- [x] Establish authenticated 1-year and 2-year XSP underlying tapes.
- [x] Establish separately provenance-bound SPY participation and VIX
      volatility-context tapes. Both cover the same 501 RTH sessions as XSP;
      SPY volume is observed while XSP/VIX volume remains explicitly absent.
- [ ] Admit the 5-year window only for comparable, complete evidence.
- [~] Capture forward XSP chains/NBBO/Greeks with provenance and restart safety.
      Two-process append, repair, and manifest reuse are proven premarket;
      fresh RTH evidence remains pending.
- [ ] Bind synthetic calibration to explicit source/effective intervals.
- [~] Add completeness and freshness gates consumed identically by research,
      replay, evaluation, and live admission. Capture, captured replay,
      execution, UI, and journal share one quote classifier; evaluation binding
      remains.

**Phase exit:** identical evidence fingerprints can hydrate backtest, replay,
shadow, and live comparison without refetching complete cached ranges.

### Phase 2 — Candidate birth and causal tournament `[TODO]`

- [~] Establish safe-income vertical baselines. Synthetic discovery was
      rejected under adverse friction; one exact delayed captured vertical now
      proves replay/live pricing and risk parity, while authentic RTH
      time-series evidence remains pending.
- [~] Establish alpha defined-risk baselines. The exact same delayed captured
      snapshot now prices a one-point call-debit vertical through the canonical
      package kernel. A 5,184-cell adverse-cost directional-credit family
      produced one aggregate positive-LCB singleton, but its early chronological
      half failed sample, concentration, and LCB gates; validation and holdout
      remain sealed. A preregistered 3,456-cell directional-debit family then
      produced zero positive daily LCBs under adverse friction. Decision edge,
      RTH execution, and authentic time-series replay remain pending.
- [ ] Test whether iron condors add net value after four-leg friction.
- [ ] Formalize opening bear-trap reversal without hindsight.
- [x] Build the frozen-window opening-state matrix; falsify upside-fade,
      downside-fast-rebound, downside-slow-rebound, continuation, and
      `NO_TRADE` branches independently. Both the 256-cell XSP family and its
      preregistered 768-cell SPY/VIX causal-context extension produced zero
      corrected passes; the holdout remains sealed.
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

**Economic target:** one complete 24-hour selected-strategy shadow or paper
evaluation closes net positive after modeled/observed fees and execution costs,
with bounded drawdown and reconciled package/leg/account economics. A safe
`NO_TRADE` preserves capital but does not satisfy this profitable-run target.

**Live-capital requirement:** none; the first 24-hour economic proof may remain
shadow/paper.

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

**Economic target:** the selected strategy's complete five-session ledger is
net positive after all observed costs, stays inside the frozen drawdown/loss
limits, and is not carried by one lucky fill. If a live canary was admissible,
its actual realized economics—not synthetic or shadow P&L—must be reported
separately and be net positive for the live-profit target to pass.

**Success is not "profitable every day."** The required target is positive
aggregate 24-hour and one-week economics with a repeatable evidence loop,
preserved capital discipline, and a strategy decision we can defend. A loss,
an uneconomic no-fill, or a week of correct abstention is valuable evidence but
does not get relabeled as achievement of the profitable-run objective.

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
4. **Phase 0.4 — Metrics and risk freeze `[DONE]`**
   - daily scorecard, zero-session accounting, concentration, and confidence
     bounds frozen;
   - threshold-independent economic receipts and near-instant warm reuse proven;
   - baseline/adverse friction, one-point XSP economics, conservative capacity,
     and canary eligibility frozen;
   - failed opening-reclaim and safe-income discovery families rejected without
     opening validation or holdout.
5. **Phase 1 — Data spine `[WIP]`**
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
| E-005 | 2026-07-24 | 0.4 | Frozen score/risk contract | Git `5308293`; `research.daily.v1` | focused tests `74/74` | Daily zero-session-aware evidence; threshold-independent cache; win rate removed from authority |
| E-006 | 2026-07-24 | 0.4 | Canonical one-point wings | Git `521f16c` | focused tests `60/60` | Shared percentage anchor plus point offset reaches replay and live resolution |
| E-007 | 2026-07-24 | 0.4 | Explicit friction and safe-income smoke | Git `eddb3bd`, `49f517e`; `/tmp/xsp-safe-income-smoke.json` | `2622c16…` | 4,608 cells; 12.3 s cold/<0.6 s warm; 240 synthetic rows remain quarantined |
| E-008 | 2026-07-24 | 0.4 | Chronological safe-income discovery | `/tmp/xsp-safe-income-discovery-{baseline,adverse}-v3.json` | `e4a325ff…`, `80154859…` | Baseline has 3 duplicate sample+LCB rows; adverse has 0; validation/holdout sealed |
| E-009 | 2026-07-24 | 0.4 | Canonical hypothetical iron-condor economics | 733P/734P/746C/747C, expiry `20260727`, no submit | shared package kernel | USD 0.40 credit gives +40/-60 before costs; conservative USD 8 round-trip fees gives +32/-68; arithmetic only |
| E-010 | 2026-07-24 | 0.4 | Causal opening-reclaim discovery | `/tmp/xsp-opening-reclaim-discovery-baseline-v3.json` | `dab6f0b9…` | 1,728 cells; 0 positive LCB; 109 sample-qualified rows all negative; validation/holdout sealed |
| E-011 | 2026-07-24 | 0.4 | Causal research and parity implementation | Git `743ef06` | full suite `662 passed`; `engine_options.py` 992 lines | Exact intraday OPT expiry clock, causal reclaim mode, frozen research groups, ledger ownership, and architecture ratchet pass; threshold-only rerank reused all 4,608 receipts in 0.66 s with no workers |
| E-012 | 2026-07-24 | 1 | Two-year XSP underlying hydration | `/tmp/xsp-cache-sync-2y.json`; six stitched cache shards | canonical rows `a1154bba…` | 38,898 bars; 501 complete sessions; 496×78 bars plus five half-days×42; zero missing ranges; volume absent throughout |
| E-013 | 2026-07-24 | 2 | Preregistered opening-state matrix | `/tmp/xsp-opening-state-study-v1.json` | `03af6fb1…` | 256 cells; 316 eligible sessions/boundary; zero family-wise passes; narrow 90m downside/30m rebound hint fails corrected and neighborhood gates; holdout sealed |
| E-014 | 2026-07-24 08:48 UTC | 1 | XSP forward quote-capture smoke | `/tmp/xsp-forward-capture-smoke-v4/XSP/2026-07-24.jsonl` | `dcf24c2e…` | Exact `IND/CBOE` underlier; 12 qualified option rows, zero invalid conIds, six NBBO/full-Greek rows; requested delayed mode and preserved actual `1/3` provenance; subscription/definition errors retained; premarket plumbing evidence only |
| E-015 | 2026-07-24 09:16 UTC | 1 | XSP capture restart continuity | `/tmp/xsp-forward-restart-proof.BCWo2l/XSP/` | tape `c091adf4…`; chain `ae4679a…` | Two independent recorder processes appended two schema-v2 snapshots to one valid JSONL tape; one content-addressed chain manifest reused; 28 qualified contracts/snapshot, zero invalid conIds; actual `1/3` provenance and errors preserved; premarket delayed evidence, not RTH admission |
| E-016 | 2026-07-24 09:34 UTC | 1/2 | Captured/live XSP package parity | `/tmp/xsp-forward-restart-proof.BCWo2l/XSP/2026-07-24.jsonl` | tape `c091adf4…`; full suite `671 passed` | The exact delayed `20260731` 734/733 put-credit vertical replayed through the shared live-intended quote kernel at `-0.24` debit units: max profit USD 24, max loss USD 76; 28/28 qualified fresh delayed NBBO/Greek rows. Adapter parity is exact, but premarket delayed evidence cannot promote a strategy |
| E-017 | 2026-07-24 10:23 UTC | 1 | Two-year causal context tapes + fail-closed hydration | `/tmp/xsp-context-sync-2y.json`; `/tmp/xsp-vix-context-refresh-2y.json`; `db/{SPY,VIX}/*5mins_rth.csv` | SPY `c47f19e9…`; VIX `1797fc3e…`; focused `36 passed` | Exact canonical `STK/SMART` SPY and `IND/CBOE` VIX tapes: each 38,898 bars/501 sessions, 496×78 full days + 5×42 half-days, zero missing ranges/anomalies/duplicates. SPY healed 41 missing sessions using two contiguous requests; VIX was independently regenerated after extended-index bars exposed a half-day audit defect. Historical acquisition now uses one-day-sized repairs, adaptive duration fallback, per-contract serialization, bounded backoff, IBKR error/head-timestamp evidence, and never skips an ambiguous empty response. A live 2004 VIX negative probe retained three HMDS no-data errors and failed closed against IBKR's exact `2005-10-03T13:30Z` head |
| E-018 | 2026-07-24 10:32 UTC | 1 | Monotonic IBKR concurrency backoff | Git this commit | full suite `681 passed` | The shared adaptive planner previously widened a retry from ceilings `1` or `2` to `3`; it now emits only strictly descending concurrency (`1`; `2→1`; `6→3→1`; `10→5→3→1`). Independent primary contracts retain bounded parallelism while residual day repairs remain serialized |
| E-019 | 2026-07-24 10:37 UTC | 2 | Preregistered XSP/SPY/VIX opening-context study | `/tmp/xsp-opening-context-study-v1.json` | `6ef12945…`; preregistration `521230c` | 768 cells over 316 eligible discovery sessions/boundary; zero passes; zero positive family-wise lower bounds. SPY cumulative participation and VIX direction did not rescue the failed XSP opening families; holdout remained sealed |
| E-020 | 2026-07-24 10:42 UTC | 1/2 | Same-tape safe-income and alpha package baselines | `/tmp/xsp-same-tape-package-baselines-v1.json` | `6ec25b2e…`; source tape `c091adf4…` | One delayed snapshot priced both sleeves through the shared live-intended kernel: 734/733P credit vertical at `-0.25`, max +25/-75 USD; 741/742C debit vertical at `+0.65`, max +35/-65 USD. Geometry/economics parity is proven; delayed premarket quotes prove neither edge nor live eligibility |
| E-021 | 2026-07-24 10:47 UTC | 2 | Adverse-cost directional-credit discovery + chronological halves | `/tmp/xsp-directional-credit-discovery-adverse-v1.json`; `/tmp/xsp-directional-credit-discovery-halves-v1.json` | `f2b45a91…`; halves `7ccc10b6…` | 5,184 synthetic cells at USD 1.50/contract and two ticks retained 2,255 sample rows. One filtered DTE5/EMA3-7/PT0.5/SL0.35 cell had 38 trades, +117.61 PnL and +0.117 daily LCB, but no stable parameter neighborhood. Its exact early half had 15 trades, -0.537 LCB and 55.2% top-five-win concentration; late had 23 trades and +0.092 LCB. The singleton is rejected as a champion; validation and holdout stay sealed |
| E-022 | 2026-07-24 10:56 UTC | 1 | Official-rule-aware historical retry contract | Git current WIP | focused `43 passed`; full `687 passed, 4 deselected` | Minute-and-larger tapes retain bounded independent-contract parallelism; same-contract requests serialize and day repairs descend to one worker. Ambiguous failures retry with smaller windows and exponential delay; explicit pacing waits 15/30 seconds. `reqHeadTimestamp` proof is reused for one hour and failed probes cool down for 15 seconds because IBKR subjects head requests to strict small-bar pacing. Only rejection, expiry, or request-before-head is called unavailable; repeated broker no-data remains unresolved and cannot delete or bless a cache gap |
| E-023 | 2026-07-24 11:02 UTC | 2 | Preregistered directional-debit discovery | This document at pushed preregistration anchor | `xsp.directional-debit.discovery.v1` | Frozen 3,456-cell filtered/unfiltered one-point CALL-up/PUT-down vertical family, adverse friction, exact discovery/validation/holdout boundary, two-half repeatability, alpha risk, concentration, and neighborhood gates. No permanent catalog entry exists before evidence |
| E-024 | 2026-07-24 11:05 UTC | 2 | Directional-debit discovery verdict | `/tmp/xsp-directional-debit-discovery-adverse-v1.json`; persistent receipts `/tmp/xsp-directional-debit-discovery-adverse-v1b.sqlite3` | semantic `886861ec…` | 3,456 adverse-friction cells completed in 82.29 seconds cold and 0.26 seconds warm; 2,049 sample-retained rows, zero positive daily LCBs, therefore zero alpha-gate passes. Validation/holdout remain sealed and no permanent strategy catalog entry is born |

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
| D-017 | Percentage anchor plus additive OTM point offset is the canonical wing geometry | It preserves scale-aware strike placement while producing exact one-point defined-risk wings in replay and live | Authentic chains prove a better shared selector |
| D-018 | Model slippage in fill price and commission in cash/P&L | Combining them would obscure execution drift and corrupt payoff geometry | Never |
| D-019 | Safe-income research excludes unbounded legacy families | Naked puts and risk reversals violate this sleeve's defined-risk mandate | A separately governed sleeve explicitly admits them |
| D-020 | Credit profit targets above `1.0` are not distinct candidates | A credit package cannot earn more than its entry credit; higher thresholds only duplicate other exit paths | Payoff semantics change |
| D-021 | Score complete sessions, including abstentions | Trade-only samples hide opportunity cost and selection frequency | Never |
| D-022 | Shortlist thresholds do not belong in economic cache identity | Changing `min_trades` must reuse identical simulations | Economic semantics change |
| D-023 | Synthetic ordering is exploratory only | Underlying-derived option prices cannot prove execution or live expectancy | Authentic replay and paper drift agree |
| D-024 | First canary loss plus fees is capped at 10% of the conservative envelope | One-point width is already material near USD 1,000 | Fresh account truth or repeated canaries justify a lower limit; never loosen from recent profit alone |
| D-025 | Opening context is an observable classifier, not strategy authority | Explicit extension/liquidation/neutral facts can be shared and falsified without reviving opaque regime routing | Shadow evidence proves a more compact fact vocabulary |
| D-026 | Discovery failure keeps validation and holdout sealed | Repeatedly inspecting future windows converts research into selection leakage | A predeclared discovery family passes its frozen gates |
| D-027 | OPT replay uses exact intraday time to expiration close | A constant full-session 0DTE clock suppresses theta and can manufacture option P&L | Authentic option replay replaces synthetic valuation |
| D-028 | Pre-register the opening-state matrix before reading outcomes | Opening folklore is easy to hindsight-label; rolling prior-session thresholds and a sealed family-specific holdout preserve causality | Only through a new versioned research contract before viewing new outcomes |
| D-029 | Do not bank on opening folklore as a daily law | The first causal two-year matrix found no corrected, neighborhood-stable edge; one narrow downside-rebound hint is insufficient | A new predeclared feature family passes development and sealed holdout |
| D-030 | IBKR is not a historical XSP option-chain archive | Expired options and option EOD data are unavailable; native combo history is not stored, so successful underlying requests cannot authenticate old spread economics | A provenance-complete specialist dataset is admitted or sufficient forward tape accumulates |
| D-031 | Chain expiry and strike sets are discovery unions, not exact pairs | IBKR returned chain-wide strikes that lacked a security definition for the selected expiry; only broker-qualified contracts prove exact membership | A provider supplies an authenticated expiry-by-strike matrix with equivalent broker proof |
| D-032 | Neutral package quote arithmetic belongs to canonical execution | Captured replay and live BAG pricing must share signed debit units, quote modes, ticks, and payoff risk; replay binds tape provenance while live owns qualification and broker projection | The ownership model changes |
| D-033 | Historical ambiguity fails closed | IBKR timeouts return an empty container indistinguishable from absent bars unless errors, retries, and availability are retained; cursor-skipping silently created two month-scale SPY holes | Never accept an empty response as coverage: retry the same cursor with bounded backoff and smaller windows, classify broker rejection/unavailability, consult earliest availability, then fail with evidence if unresolved |
| D-034 | SPY/VIX context remains evidence, not strategy authority | A preregistered 768-cell extension produced no corrected, neighborhood-stable edge and no durable rescue of the opening-state family | A new causal feature contract passes development before the sealed holdout is read |
| D-035 | A no-data message is not proof of global unavailability | IBKR can return no rows for a valid contract/range under transport, farm, session, entitlement, or sparse-history conditions; only permanent broker rejection, expiry, or a requested end before `reqHeadTimestamp` proves absence | Never advance a cursor, erase a gap, or promote cache completeness from an unresolved empty response |

---

## 15. Prospective capability births

Encode these into the capability ledger only when implementation begins; reuse
an existing contract where it already owns the outcome.

- `benchmark.future.xsp-data-authenticity`
- `integration.replay.future.xsp-option-chain-replay`
- `benchmark.future.xsp-champion-tournament`
- reuse `benchmark.future.live-backtest-drift-score` for XSP live calibration
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

No objective completion is valid until both the net-positive 24-hour and
net-positive one-week/five-session economic receipts exist, the one-week
calibration receipt is complete, and the system issues an honest
promote/hold/revise/stop verdict with its remaining-risk register.

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
- IBKR TWS API and Gateway documentation, including historical-data
  limitations:
  <https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/>

---

## Conclusion

- **Quest:** XSP Mastery — Live Research Kata
- **Current status:** Phase 0.4 is complete. The first bounded chronological
  tournaments rejected both the opening-reclaim alpha family and the
  baseline-only safe-income condor. The preregistered opening-state matrix also
  found no family-wise edge; SPY/VIX context did not rescue it, and the first
  directional-credit singleton failed chronological repeatability. The
  preregistered directional-debit family produced zero positive daily LCBs.
  All family-specific holdouts remain unobserved.
- **Next action:** capture fresh RTH XSP chain/NBBO/Greeks evidence and bind its
  completeness verdict to same-tape replay. Continue bounded preregistered
  safe-income and alpha discovery without tuning rejected families against
  sealed holdouts or promoting synthetic option evidence.

**Predictive observation:** authentic option quotes may show that execution
friction dominates the small underlying effects seen so far. If so, the best
near-term XSP champion will be a rarer, wider-margin opportunity—or
`NO_TRADE`—rather than a busier selector.
