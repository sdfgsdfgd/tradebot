# XSP Mastery — Research Kata Archive

- **Active authority:** [`q_XSP_live_research_kata.md`](q_XSP_live_research_kata.md)
- **Role:** cold, lossless historical context; never the resume or runtime authority
- **Distillation checkpoint:** 2026-07-27 AEST
- **Pre-distillation active snapshot SHA-256:** `816bcb87ae8d5b9cfa0e679974997b34243f5acb2be51292773da13d6d50367e`
- **Original narrative body SHA-256:** `dfc97f209c8190bfd7dc255eb3014d5588745c682b7a0efbbf286f865dbe6f80`
- **Mutation law:** preserve unfavorable and superseded outcomes; append a new
  receipt or decision instead of rewriting historical evidence.

Routine unit-test, lint, compilation, and line-count narration does not belong
here unless a failure changed a capability, conclusion, or recovery boundary.

## Archive map

| Marker | Archived material |
|---|---|
| [A01](#xsp-archive-a01-original-mandate) | Original detailed mandate, topology, phase prose, and historical task tree through E-128 |
| [A02](#xsp-archive-a02-pre-distillation-frontier) | Exact pre-distillation frontier, mission, architecture, hypotheses, and closed campaign detail |
| [A03](#xsp-archive-a03-completed-task-tree) | Exact 2026-07-27 ordered task-tree snapshot, including completed work |
| [A04](#xsp-archive-a04-evidence-registry) | Full append-only evidence registry E-000…E-174 |
| [A05](#xsp-archive-a05-decision-journal) | Full frozen decision journal D-001…D-145 |
| [A06](#xsp-archive-a06-prior-conclusion) | Pre-distillation accumulated-status conclusion |

---

<a id="xsp-archive-a01-original-mandate"></a>
<!-- XSP-ARCHIVE:A01:BEGIN -->
# Archive A01 — Original mandate and narrative

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
   blindly applied everywhere. The live system is session-aware across XSP
   Global Trading Hours (`20:15..09:25` ET), RTH (`09:30..16:15` ET), and Curb
   (`16:15..17:00` ET), subject to holidays, broker permissions, current market
   state, and evidence-based liquidity admission. Opening-hour research is one
   specialist alpha hypothesis; it never narrows the product mandate to RTH.
3. **Evaluation:** continuously compare expected behavior with broker-qualified
   quotes, previews, fills, commissions, slippage, Greeks, buying-power effects,
   position state, and realized outcomes. Every divergence must improve either
   the model, execution policy, risk policy, or candidate ranking.

The primary ambition is an extremely reliable, self-healing strategy system.
First prove its directional cause with one unlevered XSP-equivalent shadow unit;
then graduate the unchanged signal to no more than one carefully admitted,
defined-risk XSP package when conditions truly justify it. The system should
seek extreme reliability, but it must never claim certainty, guaranteed profit,
or treat one lucky trade as success. The measurable target is stable net
expectancy with bounded drawdown, calibrated confidence, repeatable execution,
and explicit evidence about when **not** to trade.

For this quest, the `24h → 48h → five-session` profitability gates belong
exclusively to that one XSP-index-equivalent directional shadow/paper unit.
XSP itself is a non-tradable index, so no fictional spot order is allowed;
debit options, credit spreads, iron condors, and live option capital graduate
only in a later quest after this simplest directional baseline passes.

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

The target operating envelope is XSP's nearly continuous `24x5` exchange
surface, not forced continuous exposure. Every quote, forecast, preview, chase,
fill, and outcome is labeled `GTH`, `RTH`, or `CURB`; each session earns its own
liquidity, spread, fill, and drift calibration. A champion may abstain or be
session-specific. Evidence from RTH cannot silently authorize GTH/Curb capital,
and thinner overnight markets require tighter—not looser—execution admission.

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
- skip on stale/wide markets, insufficient depth, event risk, missing
  strategy-required model inputs, preview disagreement, or weak statistical
  edge. Do not require unused Greeks merely because the broker exposes them.

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

Primary proof lane: `xsp.opening-directional-unit.v1`.

- Prove the causal directional engine before adding option leverage: backtest
  and run one synthetic XSP unit at `$1` per index point through the live-data
  shadow/paper UI for the `24h → 48h → five-session` gates.
- XSP is a non-tradable index, so this unit is an evaluation instrument, never
  a fabricated broker fill. The first strict-XSP executable graduation is one
  bounded-debit call or put vertical driven by the unchanged signal.
- Treat the preceding `2/4/6` GTH/pre-open hours as separately provenance-bound
  context, then decide only from completed `5/10/15/20`-minute RTH evidence.
  Test upward continuation and downside-sweep/reclaim as distinct branches.
- Own exits canonically across backtest and live: an initial structure/volatility
  stop, a high-water trailing ratchet after sufficient favorable excursion,
  and a time/fizzle exit. Every update uses information available at that
  instant; report captured excursion, giveback, stop lag, and adverse gaps.
- Never score hindsight-perfect bottom or peak fills. The target is repeatable
  participation in the middle of the move while surviving failed openings.

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

**Frozen pre-open proxy extension — `xsp.preopen-directional.study.v1`**

Registered before reading outcomes. Use canonical XSP five-minute RTH bars and
canonical SPY five-minute SMART/full24 bars only through `2024-07-23`;
`2024-07-24..2025-07-23` remains sealed validation. SPY is provenance-bound
context, never a substitute execution price for XSP.

- Freeze every feature at `09:25 ET`: SPY returns from `04:00`, `07:25`,
  `08:25`, and `08:55`; the gap versus the prior `16:00` close; the
  `04:00..09:25` range and closing-range location. Missing exact anchors
  abstain.
- Test only four interpretable long contexts: broad upside consensus,
  late pre-open reversal after weakness, gap-down recovery, and high-range
  strength. Compare every contextual candidate with its identical unfiltered
  RTH signal.
- Reuse the causal v2 opening signals and stop/trail/fizzle lifecycle. Context
  may admit or veto an entry; it may not own direction, exits, sizing, or
  strategy switching.
- Require at least `4/3` trades per complete discovery session, positive daily
  95% lower bound, profit factor at least `1.10`, positive results in all but
  one half-year, bounded drawdown, improvement over the matched unfiltered
  candidate, and support from neighboring context thresholds.
- Search at most `6,000` discovery cells with a ten-minute hard cap. Open
  validation only for a stable discovery pocket. No option or live promotion
  follows directly from this synthetic `$1/point` directional study.

**Frozen one-position opening ride — `xsp.opening-ride.study.v1`**

Registered after the v3 pre-open rejection and before reading new outcomes.
This is the directional proof baseline, not the still-open HF crown.

- Enter at the next five-minute open after one causal admission: fixed-time
  baseline, opening strength, flush/reclaim, or renewed breakout. Observe from
  `5/10/15/20/30` minutes and stop seeking entries by
  `30/45/60/90/120` minutes.
- Permit at most one `$1/point` synthetic XSP position per session and no
  re-entry. Use the existing conservative lifecycle: prior-bar
  structure/ATR initial stop, high-water trail effective no earlier than the
  next bar, break-even ratchet, fizzle/time exit, and forced session close.
- Compare every gated entry with the identical fixed-time entry and identical
  exit policy. Never score low-of-day entry, high-of-day exit, same-bar
  favorable ordering, or zero friction.
- Discovery must produce at least `0.5` trades per complete session (roughly
  the `>120/year` baseline), positive daily 95% lower bound, profit factor at
  least `1.10`, positive results in all but one half-year, median favorable
  excursion capture at least `0.20`, worst day no worse than `-$5`, drawdown
  no greater than `-$25`, and stable neighboring stop/trail thresholds.
- Evaluate at most `2,400` broad cells plus bounded local neighbors under ten
  minutes. Open validation only for stable discovery. A later HF champion must
  still meet the separate `4/3` trades/session contract.

**Frozen symmetric opening ride — `xsp.symmetric-opening-ride.study.v1`**

Registered after the long-only rejection and before reading new outcomes.

- Choose exactly one causal direction per session from opening continuation,
  prior-range breakout/breakdown, or symmetric sweep/reclaim/rejection.
  Long/short facts share one owner and remain separately attributed.
- Enter only at the next five-minute open, permit one position and no re-entry,
  and mirror the same structure/ATR stop, next-bar high/low-water trail,
  break-even, fizzle/time, friction, and session-close rules across direction.
- Compare with identical always-long and always-short fixed-time twins. Require
  at least `25%` of trades from each side and non-negative net P&L from each;
  aggregate profitability may not conceal a structurally losing direction.
- Retain the one-position gate above: at least `0.5` trades/session, positive
  daily 95% lower bound, profit factor `>=1.10`, all but one half-year
  positive, median excursion capture `>=0.20`, worst day `>=-$5`, drawdown
  `<= $25`, and neighboring stop/trail support.
- Evaluate at most `2,400` broad cells plus bounded local neighbors under ten
  minutes. Validation remains sealed absent stable discovery; option leverage
  remains forbidden.

**Outcome: rejected.** The frozen discovery produced no profitable
cadence-qualified cell and no positive daily lower bound. Do not retune this
same opening-OHLC family or inspect its sealed validation year; any successor
must add independently justified information or a materially different
mechanism.

**Frozen multiscale directional-impulse census —
`xsp.multiscale-directional-impulse.census.v1`**

Registered after the bar-pattern families were rejected and before reading
multiscale outcomes. This is a materially different causal mechanism derived
from telemetry already shared by live and replay (`RATS-V` slope/velocity,
true-range expansion, shock ATR velocity/acceleration, and signed return
pressure); it is not another opening-pattern retune.

- Use XSP five-minute RTH execution bars and provenance-bound SPY five-minute
  full-session proxy bars only through `2024-07-23`.
  `2024-07-24..2025-07-23` remains sealed validation.
- This is strictly intraday evidence. Daily, weekly, and higher-timeframe
  regime labels cannot vote. XSP uses only `5/15/30/60/120m`; SPY's pre-open
  warm context uses only `15/30/60/120/240/360m`.
- Evaluate completed bars from `09:35..11:30 ET`. XSP contributes signed price
  velocity over `5/15/30/60/120` minutes as each horizon becomes available.
  SPY contributes `15/30/60/120/240/360`-minute velocity, including causal
  pre-open history; its deterministic warm-start weight decays to zero as the
  first twelve XSP RTH bars complete.
- Direction comes only from signed price velocity, its change across fast and
  slow horizons, signed return efficiency, and cross-horizon coherence.
  True-range/ATR level, velocity, acceleration, and range expansion may raise
  or lower conviction but can never choose up versus down.
- Replay the complete causal evidence already exposed by live diagnostics:
  fast/slow slope medians and velocities, slope-velocity consistency, signed
  shock return sum, shock/true-range strength, ATR velocity and acceleration,
  drawdown-distance velocity and acceleration, fast/hard directional peers,
  hard-state release age, and transition heat. Decompose legacy
  `regime2/regime4` labels into those facts; the labels themselves receive no
  new authority. Drawdown evidence is slow downside/recovery context, not a
  symmetric intraday direction oracle.
- Census exactly three interpretable owners: XSP-native velocity, SPY-warmed
  velocity, and cross-instrument coherence. Each emits `up`, `down`, or
  `abstain`; EMA and Supertrend are measured as peer confirmations, never
  hidden prerequisites.
- Report fixed conviction bands against next-bar-entered `15/30/60/120`-minute
  returns, MFE, MAE, directional hit rate, long/short attribution, per-session
  attribution, and chronological half-year stability. Dependence-aware
  confidence is clustered by session; family-wise correction spans every
  owner, band, and horizon.
- This first pass is a feature census, not a trade-parameter tournament:
  no stop, trail, sizing, threshold, or validation optimization is permitted.
  Promote only a monotonic, neighboring-band-stable directional relationship
  with adequate observations, positive session-clustered lower bound, and
  support across all but one half-year.
- If the census passes, preregister one bounded `$1/point` lifecycle study and
  then extract only its compact winning state machine into the shared signal
  owner as an alternative `entry_signal`. Backtest and live must consume the
  same owner and diagnostics. If it fails, preserve the telemetry for
  explanation/risk and do not create a new regime router.

**Frozen directional-turn lifecycle study —
`xsp.directional-turn-lifecycle.study.v1`**

Registered after the production-engine census passed its material-turn
acceptance and before reading any lifecycle P&L. The unchanged
`DirectionalImpulseEngine` and frozen `DirectionalTurnPolicy` own direction;
this study may vary exits but may not retune the detector.

- Use only canonical XSP five-minute RTH bars from
  `2021-07-26..2024-07-23`. Keep `2024-07-24..2025-07-23` sealed validation.
  SPY remains diagnostic and cannot admit, veto, or choose direction.
- Select `entry_signal=directional_impulse` through the normal
  `SpotSignalEvaluator` and shared entry-control/lifecycle path. Enter one
  synthetic `$1/point` XSP-equivalent unit at the next five-minute open after
  an eligible `up` or `down` turn. Hold at most one position, permit at most
  four entries per session, and allow an opposite turn to close/reverse only
  through the normal controlled-flip contract.
- Charge `$0.10` round-trip friction per unit. XSP itself is not tradable; this
  remains a directional proof instrument and cannot be represented as a broker
  fill or options result.
- Evaluate exactly `1,296` discovery cells:
  initial stop `0.50/0.75/1.00 ATR`; trail activation
  `0.25/0.50/0.75/1.00 ATR`; trail distance `0.25/0.50/0.75 ATR`;
  break-even ratchet `off/0.50 ATR`; fizzle pairs
  `off`, `6 bars + 0.25 ATR MFE`, or `12 bars + 0.50 ATR MFE`;
  maximum hold `6/12/24` bars; and post-exit cooldown `0/3` bars.
  The full declared grid is the neighborhood test—no outcome-shaped local
  expansion.
- Initial ATR is frozen at entry from completed information. A gap through the
  existing stop fills at the adverse open. Intrabar stop checks precede the
  current bar's favorable excursion; break-even/trailing ratchets computed
  from that completed bar become effective only on the next bar. Fizzle and
  maximum-hold decisions execute at the next tradable open. Force any remainder
  flat at the session close.
- Report net P&L, daily clustered 95% lower bound, profit factor, drawdown,
  worst day, trade/session cadence, longest idle interval, long/short count and
  P&L, exit reasons, MFE/MAE, captured-excursion ratio, false-turn cost,
  half-year attribution, and top-five-day concentration. Compare with
  `NO_TRADE` and the identical opposite-turn-only lifecycle.
- A discovery pocket passes only with at least `4/3` trades per complete
  session, positive daily lower bound, profit factor `>=1.10`, positive P&L in
  all but one half-year, at least `25%` of trades from each direction,
  non-negative net P&L on each direction, median favorable-excursion capture
  `>=0.20`, worst day `>=-$5`, drawdown `<= $25`, top-five-day concentration
  `<50%`, and at least two immediate stop/trail/fizzle neighbors that preserve
  positive daily lower bound. Otherwise reject without reading validation.
- Hard cap: ten minutes. Emit ETA before execution and retain the exact
  preregistration, source fingerprint, detector policy, grid, and result
  artifact. No live or option activation follows from discovery alone.

**Result:** rejected in discovery. The normal backtest path completed all
`1,296` cells in `193.2s`; zero cells passed, zero had positive net P&L or a
positive daily lower bound, and validation remained sealed. The best cell
lost `146.31` points across `2,510` trades (PF `0.814`, drawdown `153.22`);
both directions and every half-year were negative. Its trailing exits earned
`+636.47`, but `1,574` initial stops lost `-784.07`. This is useful mechanism
evidence: the causal turn sensor finds excursions, yet indiscriminate admission
turns too many small/noisy reversals into friction and stop loss. Do not retune
the same exit grid; next test a preregistered admission-quality layer using the
existing snapshot's strength, coherence, acceleration, and session-relative
context while leaving the turn owner and lifecycle frozen.

**Frozen directional-turn admission study —
`xsp.directional-turn-admission.study.v1`**

Registered after rejecting the lifecycle grid and after inspecting only the
unconditional availability/ranges of causal entry evidence—not
feature-conditioned outcomes.

- Keep the detector, XSP discovery tape, sealed validation year, next-open
  execution, `$0.10` round-trip friction, one-position/four-entry limit, and
  SPY diagnostic-only boundary unchanged.
- Freeze the best rejected lifecycle solely as the common measurement vehicle:
  initial stop `0.75 ATR`, trail activation `1.00 ATR`, trail distance
  `0.50 ATR`, break-even off, fizzle `12 bars + 0.50 ATR MFE`, maximum hold
  `24` bars, and post-exit cooldown `0`.
- Evaluate exactly `432` admission cells:
  conviction floor `0/.05/.10/.20`; retrace floor `.75/1.00/1.50 ATR`;
  short-horizon alignment `off`, at least two of `5/15/30m` signed slopes
  aligned, at least two signed slope velocities aligned, or both; volatility
  trend `off`, fast ATR velocity positive, or both ATR velocity and
  acceleration positive; session scope `09:30..11:45`, `09:30..10:30`, or
  `10:30..11:45 ET`.
- Every value is computed at the completed signal bar that emits the turn.
  Missing required `5/15/30m` evidence fails closed. The `60/120m` horizons,
  turn sequence, and SPY remain recorded diagnostics because only `271/2,510`
  baseline entries had the complete `120m` horizon; they cannot veto early
  opportunities in this study.
- Project admission onto the immutable prepared evaluator tape, then run the
  unchanged normal execution/lifecycle engine. This is a research projection,
  not a second signal or fill simulator. If a stable pocket passes, extract its
  minimal rule into the shared `SpotEntryControlPlan`; otherwise leave
  production unchanged.
- Reuse every frozen economic gate from the lifecycle study and require at
  least two immediate conviction/retrace/alignment/volatility neighbors with
  positive daily lower bounds. Compare with the exact ungated E-051 lifecycle
  and `NO_TRADE`. Keep validation sealed unless discovery and neighborhood
  gates pass.
- Hard cap: ten minutes; publish ETA, source/code fingerprints, all `432`
  cells, gate failures, direction and half-year attribution, and the exact
  research projection contract.

**Result:** rejected as an HF champion. The `432` cells completed in `26.0s`;
`33` produced positive net P&L and `8` reached PF `>=1.10`, proving that the
shared evidence can remove substantial noise. However, no cell had a positive
daily lower bound, no profitable cell met `4/3` trades/session, no cell made
money in both directions, and validation remained sealed. The strongest
stability-ranked pocket admitted only `153/3,162` turns, earned `+6.04` points
with PF `1.154`, drawdown `9.88`, and five positive half-years, but traded only
`0.203/session`; downside still lost `-2.65`. The best net cell earned `+7.84`
over `243` trades (`0.323/session`) with a negative daily lower bound. Preserve
the late-session high-retrace/rising-ATR upside pocket as a diagnostic
hypothesis only. The next causal family must improve timing and downside
symmetry—not loosen cadence or confidence gates.

**Frozen directional reversal-cascade study —
`xsp.directional-reversal-cascade.study.v1`**

Registered before viewing cascade-conditioned labels or economics.

- Preserve the shared `DirectionalImpulseEngine`, best E-051 lifecycle,
  next-open fill, `$0.10` friction, one-position/four-entry limit, XSP-only
  authority, and external `2024-07-24..2025-07-23` sealed validation.
- Split the existing discovery tape chronologically: calibrate on
  `2021-07-26..2023-07-23`; require an untouched internal audit on
  `2023-07-24..2024-07-23`. A candidate must pass both independently before
  external validation can open.
- Define one symmetric causal phase change from completed `5/15/30m` evidence.
  The `30m` slope supplies the old direction; `5m` slope and velocity must
  reverse against it, and `15m` velocity must confirm. The grid may additionally
  require the `15m` slope to have crossed and/or the `30m` velocity to be
  decelerating toward the new direction. No daily/regime label or SPY vote.
- Emit once on entry into the qualifying phase, reset only after the phase
  clears, and enforce a `3/6`-bar event cooldown. This prevents a persistent
  condition from masquerading as repeated opportunities.
- Evaluate exactly `144` cells: `15m` slope `optional/aligned`; `30m` velocity
  `optional/aligned`; normalized `5m` velocity floor `0/.25/.50 TR`; normalized
  `15m` velocity floor `0/.10/.25 TR`; ATR trend `off/positive fast velocity`;
  cooldown `3/6` bars. The session window is fixed at `09:30..11:45 ET`.
- Alongside normal lifecycle economics, label every emitted event from its next
  bar open as favorable `1.00 ATR` reached before adverse `0.75 ATR` within
  `24` bars; same-bar ties lose conservatively. Report hit rate, direction,
  time bucket, lead/lag, event/session cadence, and half-year attribution.
  This label explains timing but cannot promote a cell without economic gates.
- Reuse every economic gate from E-051 on both chronological partitions,
  including `>=4/3` trades/session and non-negative P&L on both directions.
  Require two immediate parameter neighbors with positive daily lower bounds
  on both partitions. Compare to E-051, E-052, and `NO_TRADE`.
- Run through research-projected immutable evaluator tapes and the unchanged
  normal execution/lifecycle owner. Hard cap ten minutes; emit ETA and full
  fingerprints. Extract no production rule unless the stable chronological
  contract passes.

**Result:** rejected on both chronological partitions. All `144` cells
completed in `117.6s`; all lost in calibration and internal audit, no daily
lower bound was positive, no direction pair was profitable, and external
validation remained sealed. `71` cells met HF cadence on both partitions, so
frequency was not the failure. The best chronological compromise emitted
`0.56/0.53` events/session, scored only `40.7%/46.6%` on the frozen tradability
label, and lost `-14.43/-9.09` points on train/audit. Across the full tape it
lost `-23.93` with PF `0.815`. Retire this bar-only precursor family; do not
extract it into production or retune adjacent slope/velocity thresholds.

**Frozen VIX-confirmed turn admission study —
`xsp.vix-turn-admission.study.v1`**

Registered before reading any VIX-conditioned event or economic outcome. This
is one bounded test of independent implied-volatility pressure—not another XSP
slope, ATR, lifecycle, or exit search.

- Reuse the unchanged production `DirectionalImpulseEngine`, the E-051
  lifecycle, next-open fill, `$0.10` friction, one-position/four-entry limit,
  and canonical XSP five-minute discovery tape through `2024-07-23`.
  `2024-07-24..2025-07-23` remains sealed validation.
- Join only exact same-timestamp completed five-minute VIX `IND/CBOE` RTH bars.
  Require the current and exact within-session `5/15/30m` anchors; missing or
  misaligned VIX evidence abstains. VIX may admit or veto an existing XSP turn,
  but cannot choose direction, alter exits/sizing, or enter production.
- For an XSP up turn, confirming VIX pressure is a falling VIX; for a down
  turn, it is a rising VIX. Measure signed VIX returns over `5/15/30m`,
  confirming-horizon count, and whether the signed five-minute pressure rate
  exceeds the signed fifteen-minute rate. Normalize the five-minute pressure
  magnitude by the mean completed VIX true-range percentage over the same
  causal thirty-minute window.
- Evaluate exactly `13` cells over the full `09:30..11:45 ET` window: one
  VIX-off matched baseline; and `fast`, `majority`, `unanimous`, or
  `majority+accelerating` confirmation at normalized pressure floors
  `0/.25/.50 TR`. No time-window, XSP threshold, side-specific, or local
  expansion is permitted.
- A non-baseline cell must improve on its matched VIX-off result and pass every
  E-051 economic gate on both the chronological
  `2021-07-26..2023-07-23` calibration and
  `2023-07-24..2024-07-23` internal audit, including `>=4/3`
  trades/session, positive daily lower bound, PF `>=1.10`, both-direction
  profitability, drawdown, concentration, capture, and half-year stability.
  Its adjacent pressure floor must also preserve a positive daily lower bound.
- Hard cap ten minutes; retain exact XSP/VIX fingerprints, cache-sync health,
  formulas, cell results, and partition attribution. External validation
  remains sealed and production remains unchanged unless the complete
  contract passes. If it fails, keep VIX as observable context and close this
  historical confirmation lane.

**Result:** rejected on both chronological partitions. The canonical cache
sync admitted `58,554` exact VIX `IND/CBOE` RTH bars with one thread, zero
failed batches, zero missing days, and no repair. All `13` frozen cells then
completed in `2.9s`; none passed. Every VIX-confirmed cell improved on the
unconditional `-146.31` baseline, but even the least-bad unanimous/high-pressure
cell lost `-32.64` in calibration and `-20.43` in internal audit, with negative
P&L on both directions, zero positive half-years, PF `0.803` full-period, and
only `1.16` trades/session. The external year remains sealed and production is
unchanged. VIX remains useful context; this historical confirmation lane is
closed rather than retuned.

**Frozen NASDAQ-breadth turn admission study —
`xsp.nasdaq-breadth-turn-admission.study.v1`**

Registered before hydrating the historical breadth tape or reading any
breadth-conditioned XSP outcome. This is one bounded test of an independent
market-participation mechanism—not another XSP slope, ATR, lifecycle, exit,
SPY, or VIX threshold search.

- Reuse the unchanged production `DirectionalImpulseEngine`, frozen E-051
  lifecycle, next-open fill, `$0.10` friction, one-position/four-entry limit,
  and canonical XSP five-minute discovery tape through `2024-07-23`.
  `2024-07-24..2025-07-23` remains sealed validation.
- Join exact same-timestamp completed five-minute `TICK-NASD` and `TRIN-NASD`
  `IND/NASDAQ` bars (`conId=26719259/26719262`). Although IBKR's generic index
  `useRTH` label can include extended hours, this study admits only explicit
  `09:30..<16:00 ET` bars (or `09:30..<13:00` on an early close) through the
  canonical session filter. Missing or misaligned evidence abstains.
- Freeze causal participation pressure before outcomes:
  `tick_current=TICK[t]`, `tick_fast=mean(TICK[t-2:t])`,
  `tick_slow=mean(TICK[t-5:t-3])`, and
  `trin_fast=mean(-log(TRIN[t-2:t]))`. Positive means bullish and negative
  bearish; each is multiplied by the existing XSP turn sign only for an
  aligned/not-aligned admission decision. No raw-magnitude threshold is tuned.
- Evaluate exactly nine matched cells: `off`, `tick_current`, `tick_fast`,
  `tick_fast+improving`, `tick_reversal`, `trin_fast`,
  `tick_fast+trin_fast`, `tick_fast+improving+trin_fast`, and
  `tick_reversal+trin_fast`. Improving requires
  signed `(tick_fast-tick_slow)>0`; reversal additionally requires signed
  `tick_slow<=0`. The matched `off` baseline requires the same exact breadth
  coverage so missing sessions cannot manufacture improvement.
- A non-baseline cell must improve net P&L and daily lower bound over its
  matched baseline and pass every E-051 economic gate independently on
  `2021-07-26..2023-07-23` calibration and
  `2023-07-24..2024-07-23` internal audit, including `>=4/3`
  trades/session, positive daily lower bound, PF `>=1.10`, both-direction
  profitability, drawdown, concentration, capture, and half-year stability.
  At least one adjacent TICK-only/combined interpretation must preserve a
  positive daily lower bound; no side-specific or local expansion is allowed.
- Hydration is sequential and read-only through the canonical historical
  retry/backoff/filter owner. Hard cap ten minutes for the study and twenty
  minutes for each data operation; retain exact contract, tape, source,
  formula, partition, and runtime fingerprints. Production remains unchanged
  unless the full contract passes.

**Result:** rejected on the internal audit. Read-only canonical hydration
admitted exact `TICK-NASD` and `TRIN-NASD` tapes of `58,554` bars and `753`
sessions each (`748` normal, five early closes) in `489.2s`, with no empty
response, retry, timeout, gap, or extended-hours row. The nine cells completed
in `2.6s`; the matched `off` cell reproduced E-051 exactly. Every breadth mode
improved net P&L and daily lower bound over that losing baseline on both
partitions, but zero passed. `tick_reversal+trin_fast` was strongest by frozen
ranking: calibration `+4.97` over `267` trades, then audit `-3.85` over `130`;
its daily lower bounds were `-0.044/-0.062`, cadence only `0.53/session`, and
both audit directions lost. TICK reversal alone similarly changed
`+24.66` calibration into `-13.14` audit. Validation remains sealed and no
breadth rule entered production. Preserve these tapes as authentic diagnostics;
close this historical admission lane rather than threshold-mine it.

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

Capture forward evidence across GTH, RTH, and Curb whenever the exchange and
Gateway are available. Requested market-data type is not provenance: preserve
the actual type returned per contract, and separate contract/chain continuity
from executable live-NBBO eligibility. A mixed or delayed snapshot remains
useful research evidence but cannot pass a streaming-live capital gate.

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

Discovery has full initiative over parameter ranges, signal/gate interactions,
and new causal dimensions. Parameters exist to expose unusual profitable
pockets; do not ritualistically rerun a small conventional grid after it has
gone dry. Expand, narrow, recenter, or compose those axes when evidence gives a
reason. Use diagnostics, coverage maps, ablations, and adaptive follow-up
searches to sharpen promising mechanisms. Rigor begins at interpretation:
version every changed family, distinguish a stable pocket from an isolated
maximum, and restart sealed validation whenever the design learned from prior
results.

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

Frequency is also a hard eligibility axis, not a way for a sparse strategy to
game P&L. Over every complete scoring fold, an **LF** crown requires at least
one closed package per two eligible sessions (about `126/year`); an **HF** crown
requires at least four per three sessions (about `336/year`). Report
active-session coverage, longest idle run, and trade concentration so bursts or
duplicate re-entry cannot manufacture cadence. `NO_TRADE` remains correct for
capital safety, but an overly dormant strategy cannot satisfy a champion crown.

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

Preregister `xsp.opening-directional-unit.v1` as the proof-first HF family:

- one serial long XSP-equivalent shadow unit at `$1/point`; never more than one
  open position, but test `1/2/3` causally renewed entries per session rather
  than silently imposing the UI's default one-entry ceiling;
- use `2021-07-26..2024-07-23` for adaptive discovery and
  `2024-07-24..2025-07-23` for unchanged validation. Treat
  `2025-07-24..2026-07-23` as researcher-exposed diagnostics, not a sealed
  holdout; `2026-07-24` onward is the first true forward evidence;
- begin with authentic five-minute XSP RTH tape. Add `2/4/6`-hour pre-open
  context only after a provenance-complete full-session proxy/reference tape
  exists; do not invent historical XSP GTH bars;
- search causal EMA/supertrend trend changes, opening drawdown/rebound,
  range/volatility expansion, renewed pullback-reclaim, confirmation, cooldown,
  entry deadline, and interaction gates. Adaptive discovery may reshape these
  axes, but every learned family/version restarts unchanged validation;
- search structure/ATR initial stops, favorable-excursion activation,
  high-water trailing distance, and time/fizzle exits. Stops only ratchet
  toward profit and consume finalized information available to live runtime;
- require the HF cadence, daily alpha gates, chronological repeatability,
  stable parameter neighborhoods, and honest failed-day attribution before the
  exact configuration may enter the `24h → 48h → five-session` live-data
  shadow ladder.

Preregister the matched condor incremental-value audit before reading pairwise
outcomes:

- use only the already-open `2025-07-24..2026-01-22` discovery artifacts;
  adverse friction is authoritative and baseline friction is descriptive;
- match each filtered/unfiltered iron-condor cell to the put-credit vertical
  with identical DTE, moneyness, target, stop, EMA, entry, and exit semantics;
- a condor adds value only if it independently passes every safe-income gate,
  exceeds its matched vertical's net P&L and pooled daily LCB, and is no worse
  on maximum drawdown or worst session after its four-leg costs;
- require at least two immediate matched parameter neighbors to do the same.
  Otherwise reject the condor extension without reading validation or holdout.

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
- [x] Admit the 5-year underlying window as comparable, complete 5-minute RTH
      evidence. It contains 1,254 sessions with exact normal/early-close row
      counts, no effective gaps or duplicates, and still carries no authentic
      XSP volume or historical option evidence.
- [~] Capture forward XSP chains/NBBO/Greeks with provenance and restart safety.
      Two-process append, repair, and manifest reuse are proven premarket. Ten
      strict-live `09:31..09:40` RTH snapshots now prove this account lacks
      streaming XSP entitlement; none fabricated a close-based contract.
      Separately labeled delayed-mechanics jobs collect an opening sequence in
      `/tmp/xsp-rth-forward-delayed-20260724`, exact
      `10:00/10:30/11:00/11:30` boundaries in
      `/tmp/xsp-rth-boundaries-delayed-20260724`, and fixed five-minute
      non-boundary samples into durable `db/quotes/XSP/2026-07-24.jsonl`.
      A client-183 close collector fills the original schedule's
      `16:00/16:05/16:10 ET` core-session gap, then a separate client-182
      collector covers `16:15..16:55 ET` Curb in the same file; none of the
      three request windows overlap.
      Concurrent recorders have distinct IBKR client IDs, every one-shot
      subprocess has a 150-second bound, and no order is submitted. The
      non-submitting shadow owns the isolated `979/980/981` triplet while the
      recorder remains on `989`; neither can reuse the live runtime pool or
      its persisted client-ID state. The prospective producer is now frozen as
      one read-only, restart-safe process per exchange session: it starts at
      `20:15 ET` Sunday through Thursday, appends every five minutes across
      GTH/RTH/Curb, and exits at `17:00 ET` the following exchange date. It
      reuses the existing tape/universe/repair kernel and on-demand tunnel;
      no reconnect-per-snapshot service or second cache was created. The units
      remain uninstalled until the combined revision reaches q.
- [x] Use one exchange trading date for the entire forward evidence session.
      Evening GTH maps to the following XSP trading date; pre-open GTH, RTH,
      and Curb remain on that same date. Recorder paths, restart restoration,
      expiry DTE selection, and shadow option-tape lookup now share that owner.
      A continuous recorder makes no broker request while XSP is normally
      closed, resumes the correct tape/universe at the next session, and uses
      the existing shared/exclusive JSONL locks so shadow reads cannot observe
      a partial append.
- [x] Bind synthetic calibration to explicit source/effective intervals. New
      records retain broker observation time, actual underlying-tape bounds,
      source kind, and a next-date effective boundary; source-less RV
      overrides fail closed and legacy records remain readable.
- [~] Add completeness and freshness gates consumed identically by research,
      replay, evaluation, and live admission. Capture, captured replay,
      execution, UI, and journal share one quote classifier. The recorder now
      emits a strict provenance/NBBO/freshness/live verdict plus diagnostic
      Greek coverage, and captured package replay refuses missing or mismatched
      chain provenance; evaluation binding remains.

**Phase exit:** identical evidence fingerprints can hydrate backtest, replay,
shadow, and live comparison without refetching complete cached ranges.

### Phase 2 — Candidate birth and causal tournament `[TODO]`

- [~] Establish safe-income vertical baselines. Synthetic discovery was
      rejected under adverse friction; one exact delayed captured vertical now
      proves replay/live pricing and risk parity, while authentic RTH
      time-series evidence remains pending.
- [x] Establish a five-year short-strike barrier census before proposing
      another credit family. The fixed v1 contract reproduced all 128 cells
      from the admitted tape byte-for-byte in 1.80 seconds. It sets empirical
      quote hurdles but cannot promote historical option expectancy.
- [~] Establish alpha defined-risk baselines. The exact same delayed captured
      snapshot now prices a one-point call-debit vertical through the canonical
      package kernel. A 5,184-cell adverse-cost directional-credit family
      produced one aggregate positive-LCB singleton, but its early chronological
      half failed sample, concentration, and LCB gates; validation and holdout
      remain sealed. A preregistered 3,456-cell directional-debit family then
      produced zero positive daily LCBs under adverse friction. Decision edge,
      RTH execution, and authentic time-series replay remain pending.
- [x] Test whether iron condors add net value after four-leg friction. Across
      `2,304` exact adverse-friction matched cells, no condor independently
      passed the safe-income gate; therefore none could add stable incremental
      value over its vertical counterpart.
- [ ] Formalize opening bear-trap reversal without hindsight.
- [x] Build the frozen-window opening-state matrix; falsify upside-fade,
      downside-fast-rebound, downside-slow-rebound, continuation, and
      `NO_TRADE` branches independently. Both the 256-cell XSP family and its
      preregistered 768-cell SPY/VIX causal-context extension produced zero
      corrected passes; the holdout remains sealed.
- [ ] Formalize early opening trend-change/continuation as a separate HF
      candidate at causal `5/10/15/20`-minute decision boundaries; measure
      signal lead time and captured excursion without using the eventual peak.
- [x] Run one preregistered NASDAQ breadth-participation admission test against
      the frozen XSP turn/lifecycle baseline. Exact entitled contract and
      session semantics, exact three-year tapes, and the nine-cell chronological
      rejection are proven; validation stayed sealed and production unchanged.
- [ ] Establish LF directional/premium baselines.
- [ ] Partition HF/LF and safe-income/alpha crowns.

#### Phase 2.1 — Fixed XSP credit-barrier census v1

This is an underlying-risk screen, not an option-PnL backtest or strategy
promotion. It asks what executable credit a one-point vertical would minimally
need to compensate for historically observed expiration breach risk.

- Evidence: complete `2021-07-26..2026-07-23` XSP 5-minute RTH tape.
- Decision boundaries: `10:00`, `10:30`, `11:00`, and `11:30` ET, exactly
  30/60/90/120 minutes after the open.
- Sides: put-credit and call-credit, independently.
- Short-strike distances: `0.25%`, `0.50%`, `0.75%`, and `1.00%` from spot at
  the decision boundary.
- Geometry: one-point width; put shorts round upward and call shorts downward
  to the nearest whole XSP point, conservatively toward spot.
- Expiration horizons: same-session close, then `1`, `3`, and `5` subsequent
  trading-session closes. The intervening RTH path includes opening gaps.
- Outcomes: short-strike touch, expiration beyond the short strike, maximum
  adverse excursion, pooled and annual rates, Wilson 95% upper bounds, and
  exact eligible-session counts.
- Adverse friction: USD `1.50` per contract per side plus two package ticks on
  both entry and exit: USD `10` round trip for one two-leg spread.
- Conservative price-unit hurdle:
  `required_credit = breach_rate_upper95 * 1.00 + 0.10`. Any expiration beyond
  the short strike is treated as a full-width loss even when settlement inside
  the long strike would produce only a partial loss.
- No cell is a candidate unless a fresh, strict-admission RTH package quote
  offers at least its required credit after tick rounding. Barrier evidence
  alone cannot open validation, claim expectancy, or control capital.

Observed v1 verdict:

- The most conservative same-session one-percent cells were the `11:30` ET
  put-credit and call-credit barriers. Their Wilson-plus-friction minimum
  credits were `0.1831` and `0.1620`.
- For the next-session horizon closest to the scheduled Friday-to-Monday
  capture, the best one-percent barriers still required `0.3026` put credit
  at `10:30` and `0.3267` call credit at `11:00`.
- Three- and five-session one-percent hurdles rose to `0.3659..0.5059`.
  Annual breach rates varied materially, so pooled performance is not stable
  enough to waive the fresh-quote or forward-replay gates.
- The census is now one reproducible function in the shared research-evidence
  owner. It consumes the admitted tape and its fingerprint, computes each
  boundary/horizon path only once, and regenerated the original artifact with
  identical SHA-256 `29c4d73b…`.

#### Phase 2.2 — Causal XSP credit-eligibility screen v1

Preregister before computing conditional outcomes. This asks whether a compact,
observable `NO_TRADE` filter can reduce short-strike risk; it remains an
underlying-risk screen and cannot establish historical option expectancy.

- Source: the same admitted five-year XSP 5-minute RTH tape and fingerprint.
- Family-specific chronological boundaries:
  - discovery: `2021-07-26..2024-07-23`;
  - validation: `2024-07-24..2025-07-23`;
  - locked holdout: `2025-07-24..2026-07-23`.
  Only discovery may be read initially. These are family-specific boundaries,
  not globally untouched data; the unconditional annual barrier rates are
  already known.
- Geometry: decision times `10:30/11:30` ET, short offsets `0.75/1.00%`,
  same-session and next-session horizons, both credit sides, one-point width,
  and the exact v1 strike/path/friction semantics.
- Compute exactly eight matched causal contexts:
  `unfiltered`, `direction`, `gap`, `direction+gap`, `quiet`,
  `quiet+direction`, `quiet+gap`, and `quiet+direction+gap`.
- `direction` means boundary close is at or above the session open for a put
  credit and at or below it for a call credit. `gap` applies the same
  side-aware sign to session open versus the previous session close.
- `quiet` means the current open-to-boundary high-low range divided by the
  previous close is no greater than the median from the prior 60 complete
  sessions at that exact boundary. Fewer than 60 prior sessions abstains.
- The fixed family contains `128` cells:
  `2 times × 2 offsets × 2 horizons × 2 sides × 8 contexts`.
- Report observations, touches, expiration breaches, Wilson upper bounds,
  annual point rates, worst adverse excursion, and required credit exactly as
  in the parent census.
- A discovery context passes only with at least `120` observations and `30`
  in every included July-to-July block; pooled required credit at least `0.05`
  below its matched unfiltered parent; pooled touch upper bound lower than its
  parent; expiration-breach point rate no worse in every block; and one
  immediate time/offset neighbor passing the same rules.
- If no discovery context passes, reject this eligibility family without
  reading validation or holdout. If one passes, freeze the frontier and require
  the same directional improvement in validation before opening the holdout.
- Even a full pass only defines an eligibility/abstention fact. A fresh
  strict-admission RTH package quote must clear its registered credit hurdle,
  followed by authentic replay and execution/economic gates.

Observed v1 verdict:

- Discovery evaluated all `128` cells over `58,554` bars and `753` sessions in
  `1.49` seconds. All structural, nested-context, parent, block, and gate
  invariants passed independently.
- Zero conditional cells achieved the preregistered `0.05` required-credit
  improvement, so zero reached the neighbor gate. Validation and the locked
  holdout remain unread.
- The nearest apparent improvement was the `11:30` ET, `0.75%`, same-session
  put-credit cell under `quiet+direction+gap`: `0.0493` lower required credit,
  but only `113` observations. It fails both the `0.05` and `120` gates; neither
  threshold is moved after seeing the result.
- Simple side-aware direction, gap, and rolling-quiet facts therefore do not
  justify an XSP premium-selling selector. They remain diagnostics, not live
  policy. The rejected one-off driver was removed rather than adding another
  permanent research surface.

#### Phase 2.3 — Fresh RTH XSP package screen v1

Preregister before the `2026-07-24` RTH capture begins:

- Sources are the opening, exact-boundary, and durable five-minute quote tapes
  already scheduled above. Preserve their independent files, append order,
  chain manifests, errors, actual market-data types, and content fingerprints.
- Opening `09:31..09:40` observations prove live-session capture health only;
  they cannot satisfy a `10:00..11:30` strategy decision.
- Freeze the boundary anchor before selecting strikes: use a fresh underlying
  NBBO midpoint, then fresh last, then the robust median of timestamped
  option-model underlying values. A close-only or stale index value is
  diagnostic and cannot anchor an executable package.
- At each exact boundary construct only the `0.75%` and `1.00%` one-point
  put-credit and call-credit verticals using the frozen whole-point,
  toward-spot short-strike rule. Monday expiry maps to the next-session
  historical barrier. Once selected, package legs and conIds remain fixed
  through follow-up replay; never roll strikes after seeing the path.
- A boundary package is evidence-eligible only when the snapshot has exact
  chain provenance, every captured contract is qualified, both selected legs
  have streaming-live NBBO no older than 30 seconds, and the canonical package
  kernel produces identical captured/live-intended geometry and risk. Preserve
  available Greeks for diagnostics; no Greek is a blanket execution gate.
  `model_under_price` is required only when it supplies the strike anchor.
- Natural executable entry credit—not midpoint—must meet the registered
  Wilson-plus-friction hurdle. Midpoint and optimistic credit remain
  diagnostics. Natural-to-mid credit difference may not exceed `0.10`, ten
  percent of a one-point width.
- The exact boundary and at least one of its fixed-leg neighboring
  five-minute snapshots must independently retain strict quote evidence and
  natural credit at or above the same hurdle. A transient one-snapshot cross
  does not qualify.
- Canonical maximum loss plus conservative remaining round-trip fees must fit
  the frozen USD `100` first-canary ceiling. Broker preview/capacity evidence
  may only tighten that limit.
- For any package that clears, freeze its boundary forecast, then replay its
  fixed legs through subsequent snapshots. Record natural liquidation cost,
  midpoint, package/leg Greeks, quote loss/staleness, spread, maximum favorable
  and adverse excursion, and end-of-tape state. This is mark/replay evidence,
  never a fabricated fill.
- One fresh session can birth a shadow candidate only. It cannot satisfy the
  24-hour, 48-hour, five-session, paper, or live-profit gates.
- If no package clears, retain the complete tape and rejection reasons; do not
  relax the historical hurdle or quote requirements post hoc.

Observed v1 verdict:

- All four preregistered `10:00/10:30/11:00/11:30 ET` snapshots were captured
  with fresh XSP last-price anchors (`0.000..1.600` seconds old), exact Monday
  expiry, stable chain fingerprints, and `100/100` qualified contracts.
- The canonical captured/live-intended package kernel evaluated all `16`
  frozen side/offset cells and their next five-minute fixed-leg neighbors.
  Twelve retained identical conIds, seven boundary packages and three
  neighbors were mechanically priceable from delayed/mixed evidence, but zero
  boundary or neighbor packages had a complete streaming-live selected-leg
  quote. Therefore zero cells reached strict admission.
- The economic veto was independently decisive: every priceable boundary
  natural credit missed its registered hurdle. The closest was the `11:00 ET`
  `0.75%` call credit at `0.25` versus `0.39664` required; its fixed-leg
  neighbor improved only to `0.28`. No quote requirement or hurdle was moved
  after the observation.
- Verdict: preserve the tape and emit `NO_TRADE`. This session proves durable
  capture, exact package projection, and fail-closed admission—not a champion,
  paper candidate, entitlement, or profit result.

This preregistration is intentionally RTH-only because it tests the U.S.-open
behavioral hypothesis. In parallel, accumulate separately labeled GTH and Curb
forward tapes and build session-conditioned baselines from them. Do not reuse
RTH thresholds, fill assumptions, or quote-quality distributions outside RTH
until same-session evidence supports them.

#### Phase 2.4 — Forward option-parity participation observer v1

Preregister before any outcome scoring:

- This is a materially independent, observation-only admission hypothesis for
  future sessions. The complete `2026-07-24` tape may prove deterministic
  mechanics and availability only; its already-observed cash path cannot
  establish value, tune thresholds, or promote a selector.
- Reuse the canonical option quote owner. At each snapshot, match qualified
  call/put contracts by exact expiry and strike, require non-crossed timestamped
  NBBO no older than `30s`, and choose at most the five strikes nearest the
  causal anchor. The anchor is fresh XSP last/mid when available, otherwise the
  already-declared option-model consensus diagnostic.
- Require exact chain provenance and at least three matched pairs. Persist
  actual market-data types, pair count, maximum quote age, median relative
  spread, parity-anchor median `strike + call_mid - put_mid`, and parity
  dispersion. These are diagnostics, not executable package prices.
- Align only the latest snapshot whose capture timestamp is at or before a
  directional decision, from the same session, and no more than seven minutes
  old. Never look forward to a later option snapshot.
- The prospective hypothesis is narrow: cash-direction turns whose
  option-parity anchor is moving coherently in the same direction under
  adequate paired coverage and spread quality may have fewer false
  admissions than the unchanged `TA-only` observer. Compare paired
  `TA-only` versus `TA+option-observe` receipts on identical forward decisions;
  the option observer initially records agreement, disagreement, and
  unavailable only and has no veto, sizing, selection, or order authority.
- Freeze `60m` as the one primary non-overlapping outcome horizon before the
  first prospective session. The benchmark reports aligned, opposed, flat,
  and unavailable counts; each cohort's wins, losses, net/mean points; exact
  chain/current/prior provenance; and overall TA points unchanged. It performs
  no hypothetical veto and cannot call any cohort an edge before at least
  `30` usable pairs across five complete sessions. This prevents the three
  overlapping observer horizons from masquerading as independent evidence.
- Do not mine a velocity threshold from July 24. Accumulate independent
  sessions first, then freeze any candidate threshold from discovery and
  require chronological stability, adequate HF cadence, both directions, and
  positive economic lower bounds before opening validation.
- Delayed/mixed evidence may validate plumbing and causal alignment only.
  Promotion still requires streaming or broker-preview evidence, restart-safe
  replay, the one-unit directional profitability gates, and the ordinary
  safety contract.

Implementation status: the shared `live_calibration.v1` ledger now emits this
exact `60m` classification receipt. It reuses the already-frozen current/prior
parity context, preserves `NO_TRADE`, and reports no action field. Replaying the
already-seen July mechanics yielded four pairs—one aligned and three
unavailable—across one diagnostic session. The benchmark now reports those
rows separately from prospective evidence: July contributes exactly zero
prospective pairs and zero prospective sessions, so it can never help satisfy
the frozen `30 pairs / 5 sessions` sample gate. This receipt proves mechanics
only. A prospective pair enters the gate numerator only when its own trading
date has every canonical shadow checkpoint (`78` normal or `42` early-close)
in one coherent `EVALUATED` state; one missing/conflicting invocation keeps
that day's pairs diagnostic. Monday is the first eligible prospective evidence.

Before Monday's first prospective RTH decision, freeze one mechanics-only
pre-open bridge without reading an outcome:

- Reuse the same qualified option-parity observation; do not add another price,
  Greek, cache, or signal implementation.
- For an RTH decision, retain the last usable GTH parity observation from the
  same XSP trading date only when it lands within ten minutes of the `09:30 ET`
  boundary. Resolve exact elapsed `2h`, `4h`, and `6h` anchors at or before
  their targets with at most seven minutes of capture tolerance and the same
  target expiry.
- Persist each anchor's timestamp, chain fingerprint, pair count, dispersion,
  median relative spread, exact selected strikes, maximum quote age, price
  anchor and anchor source, actual market-data-type census, value change,
  per-minute velocity, and raw `up/down/flat` sign. Preserve the same fields at
  the final GTH boundary so local-surface, freshness, spread, and provenance
  evolution can be evaluated without reconstructing it after outcomes.
  Missing, stale, cross-date, cross-expiry, or incomplete horizons remain
  explicit and make the composite path unusable.
- Extend the existing parity benchmark with exactly five descriptive cohorts:
  `aligned_all` when all three path directions match the frozen TA turn;
  `opposed_all` when all three oppose it; `reversal_into` when the `6h` path
  opposes but the `2h` path has turned into the TA direction; `mixed`; and
  `unavailable`. Do not rank, filter, or select from these cohorts before
  prospective sample evidence exists.
- This path is causal observation only. It neither changes the current
  seven-minute RTH parity context nor grants a veto, confirmation, sizing,
  selector, package, fill, or order action. July may prove mechanics only;
  prospective sessions must establish any value under a separately frozen
  contract.

Implementation status: the existing parity benchmark now emits those five
pre-open cohorts for all, prospective, and complete-session rows while keeping
the original short-horizon parity sample gate and aligned candidate completely
separate. A `reversal_into` fixture proves `6h` opposition followed by `2h`
alignment is classified without promotion; an incomplete session contributes
zero complete-session pre-open evidence.

Before the first prospective cash outcome, freeze one value interpretation
without changing the observer or mining a magnitude threshold:

- The sole candidate is exact sign alignment: admit an unchanged TA direction
  only when first-seen option-parity movement is usable and points the same
  way. Opposed, flat, unavailable, or incomplete-session evidence abstains.
- Construct independent causal `TA-only` and `TA+aligned-parity` sequences.
  Within each sequence, sort by decision time, take the earliest eligible turn,
  hold its frozen `60m` horizon, and ignore further turns before that outcome
  boundary. A turn exactly at the prior boundary is eligible. This prevents
  overlapping forecasts from becoming a fictitious equity curve.
- The TA baseline considers every prospective decision from each complete
  session—including parity-unavailable rows—so thin option coverage cannot
  flatter the candidate by silently shrinking the comparison universe.
- The existing `30` usable-pair / `5` complete-session gate must pass first.
  The aligned sequence then needs at least two UP and two DOWN observations,
  at least one observation per two complete sessions, positive total points,
  a positive two-sided 95% daily lower-confidence bound with zero-trade
  complete sessions included, positive net points after removing any one
  complete session, and no single win contributing more than half of gross
  winning points.
- To establish incremental admission value rather than merely sparse profit,
  the aligned sequence must have both a higher mean outcome and a lower loss
  rate than the independently non-overlapping TA baseline. Report drawdown and
  profit factor, but leave the ordinary selected-strategy risk contract to
  govern later `24h → 48h → five-session` shadow graduation.
- Passing this contract may create an observation-only shadow candidate. It
  never changes `NO_TRADE`, grants promotion/order authority, starts a
  profitability clock, or weakens the later broker-preview, execution,
  restart, safety, and selected-economics gates.

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

- [x] Record every forward shadow invocation independently of signal activity.
      The append-only checkpoint distinguishes fresh RTH evaluation,
      stale/empty RTH data, unsupported GTH/Curb strategy coverage, and closed
      time. Exact retries are idempotent; none has order authority.
- [x] Freeze the one-shot shadow cadence as a non-persistent systemd timer:
      `09:37..09:57`, `10:02..15:57`, and `16:02 ET`, two minutes after all
      `78` normal XSP cash-RTH five-minute bars. The service has a two-minute
      hard bound and fails closed until its managed q runtime exists.
- [ ] Provision that isolated q runtime, install/enable the units, and capture
      their first real scheduler receipt only after one combined source
      revision reaches q. Direct runtime pins and the q-local read-only Gateway
      transport are frozen and proven in disposable environments; the
      persistent runtime and units remain deliberately absent. Convergence now
      stops only the news timer, refuses to switch source while its one-shot is
      running, proves the sole dirty checkout file is the currently loaded
      legacy unit, switches cleanly, then installs and byte-verifies the new
      news/XSP units before re-enabling any timer.
- [x] Encode one fail-closed `xsp.live-profitability.v1` authority over the
      existing calibration ledger. It accepts only one non-`NO_TRADE`
      strategy/version/config/run; exact normal/early-close RTH checkpoints;
      cumulative reconciled `$1/XSP-point` gross, cost, net,
      realized/unrealized, drawdown, session-loss, trade-count and
      concentration evidence; and complete attribution with zero safety
      breaches. Missing/conflicting slots, identity drift, nonzero baselines,
      counterfactual economics, or observer abstention cannot advance a clock.
      Every `24h`, `48h`, and five-session verdict is anchored to its own
      earliest wall-clock-plus-complete-session evidence prefix, so later
      gains, losses, gaps, or retries cannot rewrite an earlier milestone.
- [ ] Start the `24h → 48h → five-session` clocks only after one selected,
      admissible strategy exists and a continuous session-coverage receipt can
      distinguish evaluated abstention from missing execution. Turn-triggered
      forecasts, overlapping counterfactual horizons, recorder uptime, and
      `NO_TRADE` observations cannot start or satisfy an economic milestone.
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
      shutdowns, allowed session(s), session-specific liquidity/chase ceilings,
      and rollback triggers.
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

**Weekend boundary:** finish every valid research, replay, recorder, preview,
and shadow-preflight task first. If the final live/shadow profitability drive
then reaches the weekend closure, pause the goal at that boundary and resume
on Monday's next eligible market window; closed hours do not count toward the
`24h → 48h → five-session` evidence clocks or profitable abstention.

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
evaluation spans every interval for which that exact strategy has frozen
authority and closes net positive after modeled/observed fees and execution
costs, with bounded drawdown and reconciled package/leg/account economics. The
current directional baseline is explicitly RTH-only; GTH/Curb remain
unsupported diagnostics and cannot count as coverage until a separate strategy
earns those sessions. A safe `NO_TRADE` preserves capital but does not satisfy
this profitable-run target.

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

1. **Current WIP — Authentic forward evidence + shared entry spine `[WIP]`**
   - one typed `SpotEntryControlPlan` now resolves the active signal owner,
     active source gates (`dual_branch`, branch slope, `RATS-V`),
     primary/secondary confirmation and their direction scopes, bear takeover
     and its risk/shock scope, the five ordered legacy regime-veto families,
     shock permission mode, named signal filters, TICK mode, allowed
     directions, lifecycle checks, and graph policy for both live and
     backtest. The same plan is embedded in every entry-control trace.
     Explicit `regime_mode=off` is truly off instead of silently normalizing
     back to EMA;
   - physical ownership now matches that contract: multihorizon slope/turn
     state lives in `engines/directional_impulse.py`, entry-source and
     permission configuration lives in `spot/entry_control.py`, and the
     remaining generic signal, gate, and journal owners are all below the
     architecture ceiling. No compatibility facade or debt exemption was
     added; architecture, capability ownership, and the full selected suite
     pass (`720 passed`, `4 deselected`);
   - one shared `decide_flat_position_intent` kernel exclusively arbitrates
     strategy permission in live and backtest: current-day admission,
     signal-day admission, numeric capacity, signal readiness, allowed side,
     the named filter registry, ATR readiness, deferred-open validity, and
     graph policy. Live retains only operational readiness outside it
     (contract/snapshot/data/preflight/pending-broker state); its former
     weekday and filter pre-vetoes are removed. Signal-day eligibility is no
     longer ambiguously folded into `BLOCKED_ENTRY_LIMIT`;
   - the XSP-native `5/15/30/60/120m` observer explicitly reserves `25`
     five-minute price samples (24 completed intervals) in both cache warmup
     and live preflight. A turn is separately eligible after exact
     `5/15/30m` evidence; the full snapshot remains unready until all five
     horizons exist. All `753` discovery and `17` recent sessions contain
     complete, gap-free `09:30..11:45 ET` windows, and no recorded event fired
     below the required three horizons. Recent SPY
     full-session context is now hydrated across four provenance-bound shards:
     `4,814` stitched rows, zero missing ranges, and exact `03:30/09:30 ET`
     anchors for all `17` matching XSP sessions. Its `15..360m` engine resolves
     by elapsed timestamp—not observation count—because the scheduled
     `03:50..04:00 ET` break leaves only `71` observations inside the exact
     six-hour span. SPY failed the owner ablation, so it remains diagnostic
     context and cannot vote on XSP direction;
   - centralize the causal live vocabulary—signed slope/return velocity,
     acceleration, ATR/true-range trend, drawdown velocity, fast/hard peers,
     release age, transition heat, and cross-timeframe coherence—behind one
     compact snapshot produced identically for backtest, live UI, journal, and
     replay;
   - the production observer now owns ATR-normalized signed slope, efficiency,
     multihorizon coherence, slope velocity, ATR level/velocity/acceleration,
     and one hysteretic causal turn state. It emits `up / down / abstain`
     evidence independently of EMA. It is selectable only through the explicit
     research source and remains inactive in live profiles; EMA/Supertrend
     remain peers or explicit confirmations;
   - the discovery-selected policy is frozen at `alpha=.90`,
     `initial=.075`, `turn=.02`, `retrace=.75 ATR`, state age/cooldown `3/3`,
     and at least three observed horizons. The production engine—not a
     research copy—scored `0.660` discovery F1 and `0.710` recent F1 with
     `66.7%` precision, `75.9%` recall, and `1.5`-bar median lag across the
     latest `17` sessions;
   - the recent timestamped ledger records causal evidence, false turns,
     misses, excursion, lag, and boundary censoring. Absolute-window extrema
     recall is only `16/34`; `18/34` absolute extrema sit at the `09:30/09:35`
     or `11:25/11:30` boundaries. Therefore material local turns are the honest
     sensor acceptance target and no “perfect daily top/bottom” claim is made;
   - `entry_signal=directional_impulse` now selects the same production sensor
     through the normal evaluator contract; all downstream confirmations,
     vetoes, permission, sizing, lifecycle, deferred fills, and traces remain
     shared. The observer stays non-authoritative for every other source;
   - `xsp.directional-turn-lifecycle.study.v1` completed all `1,296` cells
     through the normal backtest path in `193.2s`. Zero passed; every cell lost,
     every daily lower bound was negative, and validation remained sealed.
     The best lifecycle improved the opposite-turn-only baseline from
     `-159.10` to `-146.31`, but still had PF `0.814`, drawdown `153.22`, and
     negative P&L on both sides and in all seven half-years. The source remains
     inactive in live profiles;
   - **next:** freeze one compact admission-quality study over the existing
     snapshot—not another exit tournament. Measure whether strength,
     cross-horizon coherence, signed slope acceleration, ATR expansion, turn
     age, and session-relative timing can remove false turns while preserving
     the profitable trailing excursions, cadence, both directions, and
     chronological stability. Keep the lifecycle, detector, validation year,
     and SPY diagnostic-only boundary unchanged;
   - that `432`-cell admission study is now complete and rejected for HF use.
     It found `33` positive-net cells but zero positive daily lower bounds,
     zero profitable cadence-qualified cells, and zero cells with profitable
     downside. The strongest pocket was a sparse late-session upside niche,
     not a symmetric opening champion;
   - **next:** study the causal reversal cascade before the completed turn:
     short-horizon slope reversal, medium-horizon velocity confirmation, and
     long-horizon deceleration while the old trend is still present. Freeze a
     tradable-excursion label and chronological internal split before reading
     results. The purpose is earlier entry with fewer immediate stops, not a
     wider threshold search over the already rejected turn event;
   - that `144`-cell cascade study is complete and rejected. All cells lost in
     both chronological partitions; `71` met HF cadence on both, proving the
     failure is expectancy rather than inactivity. Do not add a production
     cascade gate;
   - the one permitted independent historical confirmation test is also
     complete: exact same-bar VIX pressure improved every matched XSP-turn
     baseline but all `13` preregistered cells still lost in both chronological
     partitions. The least-bad cell was too sparse, lost on both directions,
     and had zero positive half-years. Keep VIX observable and do not retune
     VIX return, coherence, pressure, or timing thresholds;
   - the final independent historical admission family is complete and
     rejected. Exact entitled NASDAQ TICK/TRIN participation improved every
     matched losing baseline, but no mode produced positive daily confidence
     on both chronological partitions. The strongest reversal/pressure
     combination flipped from `+4.97` calibration to `-3.85` audit and became
     too sparse. Preserve the exact tapes for diagnostics; add no breadth gate
     and do not retune sign, magnitude, window, or side thresholds;
   - **next frontier:** return to the authentic evidence spine and add
     materially new causal information—forward top-of-book/package
     microstructure, spread/liquidity state, and same-tape shadow outcomes.
     Preserve the directional sensor for telemetry and counterfactual scoring,
     not order authority. No further OHLC slope/velocity threshold mining is
     admissible without a new independent mechanism.
   - the XSP-native pre-open bridge mechanics are now frozen, rather than
     adding another SPY vote. The existing qualified parity owner retains the
     last usable same-trading-date GTH observation within ten minutes of
     `09:30 ET`, then resolves exact causal `2h/4h/6h` same-expiry anchors with
     seven-minute capture tolerance. Each raw path records provenance, pair
     count, dispersion, median relative spread, exact selected strikes, maximum
     quote age, price anchor/source, actual market-data-type census, point
     change, velocity, and direction at every horizon and the final boundary;
     any missing horizon makes it unusable. This does not relax the ordinary
     same-session RTH context or grant confirmation, veto, sizing, selector,
     fill, or order authority. Prospective evidence must determine whether it
     reduces the current `09:30/09:35` extrema censoring;
   - authenticated historical XSP GTH underlying bars are unavailable through
     the current broker/cache path. Every canonical XSP shard is explicitly
     RTH. A bounded read-only Gateway comparison on exact XSP
     `IND/CBOE` conId `137851301` requested the same Friday endpoint with
     `TRADES/useRTH=true` and `TRADES/useRTH=false`; both returned the identical
     `78` bars from `09:30..15:55 ET`. `MIDPOINT/useRTH=false` returned IBKR
     error `162` and zero bars. Treat this as an unsupported evidence class,
     not a missing-range repair: do not retry it through cache healing or
     fabricate historical pre-open XSP. The forward GTH option-parity tape is
     the first admissible XSP-native pre-open lane;
   - the scheduled forward recorder now requests delayed market data (`3`)
     rather than strict live (`1`). This is an evidence-availability correction,
     not an admission relaxation: the broker previously proved requested-live
     GTH had no strike-selection price and strict-live RTH produced empty
     snapshots, while an explicit delayed request captured qualified,
     timestamped NBBO/Greeks and preserved actual mixed `1/3` provenance.
     Every consumer still gates on the actual per-contract market-data type;
     delayed evidence can mature mechanics and observation cohorts but cannot
     satisfy streaming-live capital admission;
   - each synchronous broker request in that recorder is now bounded to `45s`.
     `ib_insync` otherwise defaults `RequestTimeout` to zero, so one hung
     qualification or quote snapshot could consume the entire `20h50m`
     service lifetime and bypass the existing recovery loop. A timeout now
     follows the same proven disconnect, bounded exponential backoff, reconnect,
     and no-duplicate append path as a transport failure;
   - the preregistered forward RTH screen is complete and rejected: all `16`
     fixed one-point packages failed strict streaming-live admission, and all
     seven mechanically priceable delayed boundary packages also missed their
     frozen historical credit hurdle. The closest remained `0.14664` credit
     short. Preserve `NO_TRADE`; do not mine this one session or relabel mixed
     delayed/type-1 rows as live entitlement;
   - the July 24 append-only collector, exact-source completeness/restart
     receipt, immutable q archive, and same-tape mechanics replay are finished.
     Its already-observed outcome remains diagnostic only. Monday July 27 is
     the first prospective session eligible for the unchanged parity and
     pre-open cohort contracts;
   - **next:** deploy one combined revision before the Sunday `20:15 ET`
     producer boundary. Arm news plus the read-only quote producer immediately,
     but keep the shadow timer disabled until its first manual `EVALUATED`,
     `order_authority=none` checkpoint at Monday `09:37 ET`; then enable the
     remaining `09:42..16:02` schedule. Collect the exact delayed-request/
     actual-provenance GTH path and every RTH checkpoint. Reuse that same tape for
     counterfactual package marks and session-conditioned quote-quality
     baselines. A materially new candidate must arise from authentic
     microstructure/execution evidence or a different causal mechanism—not
     another bar-threshold permutation;
   - **sequencing override:** the first candidate that honestly clears its
     prospective admission contract must earn the `24h → 48h → five-session`
     gates with one index-equivalent shadow/paper unit. Archive option
     evidence, but defer debit/credit/condor promotion and all live option
     capital to a later quest.
   - **integrated forward causal-news gate:** after the completed recorder
     handoff, delivered head
     `f837e21e555ca4f1902a9ba2efd848ddd4ae3452` from
     `origin/codex/news-intelligence-gate` was added without resetting the live
     q checkout. Its canonical owner is `tradebot/news/`; q runs
     `python -m tradebot.news` on a battery-efficient systemd timer and
     publishes qualitative memory, current events, a per-contract aggregate,
     and monthly audit snapshots. First trace that real service and file
     contract read-only; do not trigger an extra AI/news session merely for
     testing.
   - the read-only service census found one bounded operational defect: the
     `2026-07-25 05:42 AEST` run reached the pipeline's internal `600s` Codex
     timeout and exited `1` at `05:52`, before systemd's outer `15m` limit.
     Atomic publication worked correctly: qualitative memory, event ledger,
     latest aggregate, state, and two-row monthly history remain byte-for-byte
     at the last successful `01:42 AEST` publication. The nested timeout
     contract is corrected without a manual rerun: the
     deployed q unit and canonical source give the ephemeral Codex subprocess `840s`
     beneath the initially retained `900s` hard wall. A later whole-envelope
     audit rejected that exact-sum budget: `30s` discovery + `840s` inference
     + `30s` validation/publication left no allowance for interpreter startup,
     subprocess teardown, or filesystem synchronization. The canonical
     combined unit therefore uses a `960s` outer wall and a real `90s`
     post-inference reserve. At `2026-07-25 16:15 AEST`, while the one-shot was
     idle/failed and nearly two hours before its next natural timer, the exact
     canonical unit was installed and daemon-reloaded on q without starting or
     restarting the service; q's checkout copy and loaded copy are now
     byte-identical to the canonical `209b5b05…` unit. The timer remains
     naturally scheduled for `18:11:07 AEST`; source revision, publication
     state, and the broader one-revision integration remain unchanged. Unit
     verification, focused tests, and the full combined suite passed. This is
     evidence-sized rather than
     arbitrary: the preceding successful cycle consumed `9m52s`, only eight
     seconds below the old nested ceiling.
     The combined WIP also removed a hidden budget coupling: Finviz discovery
     now receives at most `30s` while Codex retains `840s`, leaving a real
     `90s` reserve inside the `960s` service wall for validation, publication,
     startup, teardown, and filesystem synchronization. The next
     natural q run still uses the delivered pre-WIP pipeline with the corrected
     explicit Codex allowance, so it proves that allowance only; the complete
     `30 + 840 + 90` invariant becomes operational when this WIP is integrated.
     That natural `09:52:05` run exercised the corrected `840s` path and Codex
     returned its complete schema-shaped payload at `10:01:31` (`9m26s`).
     Because `566s < 600s`, this confirms the loaded path and successful
     one-shot completion, but is not empirical proof of surviving beyond the
     former child wall; the deployed unit and behavioral regression pin the
     actual `840s` ceiling.
     Publication then failed safely on a separate derived-clock mismatch:
     one event added evidence while retaining its old
     `last_material_change_utc`. All five last-good publication files and both
     history rows remained byte-identical. The canonical validator now owns
     `first_seen_utc` and `last_material_change_utc`: it preserves stable
     first-seen time, compares material fields with the prior stable ID, and
     stamps this run only when that diff changes. `last_verified_utc` remains
     model-authored and strict because it means substantive content was read.
     The exact rejected `20,598`-byte payload validates under this repair;
     the next natural `14:01:31` run independently returned in `576s`, again
     beneath the old child wall, then the old reducer rejected another material
     clock mismatch. Its exact `20,274`-byte response validates under the
     combined reducer after three deterministic retained-event clock
     corrections, yielding eight events and a 71-line memory. q still
     published nothing; all last-good files remained byte-identical.
     publication recovery is also causally exact: `signal_as_of_utc` owns the
     signal horizon, while `snapshot_as_of_utc` is the earliest time that
     particular generation may enter a decision. A generation interrupted
     before `latest.json` becomes visible is restamped and re-addressed at
     recovery before any output or history row is exposed; a generation
     already visible keeps its original availability. Historical replay
     therefore cannot borrow a recovered signal from the interval before it
     was durably published. A no-new-evidence refresh likewise recomputes its
     content address after changing run status, availability, or event
     buckets; it can no longer retain an ID that hashes a previous wrapper.
     The same contract owns producer and consumer hashing. Legacy q snapshots
     without an ID remain readable during migration, but every addressed
     generation fails closed if any wrapper field no longer matches its ID.
     q retains only the timeout override until the combined WIP can be
     integrated without colliding with the pending external work.
   - integrate XSP fundamental pressure beside technical evidence through the
     same centralized entry-control and receipt vocabulary, defaulting to
     `off/observe`. It may later earn an explicit defensive veto, bounded
     sizing, or structure-eligibility role; it is never a selector, order
     trigger, opaque regime router, or alpha claim. A stale, missing, failed,
     horizon-expired, or insufficient-confidence aggregate means no
     fundamental gate—not fabricated direction.
     The shared timestamp-aware consumer and entry-control declaration now
     enforce exactly that boundary: only `off|observe` exists, and the current
     real q XSP aggregate resolves to `stale` and `usable=false`.
   - `live_calibration.v1` now freezes selected and counterfactual decisions
     before their outcome horizons, binds each one to the causal bar-tape
     prefix and exact configuration, and settles append-only observed/drift
     records. Torn tails are repaired durably; missing five-minute entry or
     outcome bars remain unsettled instead of borrowing a later price. The
     first deterministic recent-session replay retained `NO_TRADE` for every
     selected decision and gave the rejected directional sensor
     observation-only counterfactual outcomes with no broker-fill fiction.
     This historical replay proves the evidence kernel, not the economic
     `24h → 48h → five-session` milestones.
     Forward invocations now append a separate checkpoint even when no turn
     fires. `EVALUATED`, `STALE_DATA`, `NO_DATA`, `UNSUPPORTED_SESSION`, and
     `CLOSED` are mutually exclusive evidence states; this prevents evaluator
     silence from masquerading as safe abstention. Known GTH/Curb windows,
     exchange holidays, and the tail after an early close now emit their
     truthful idempotent checkpoint before contract qualification or history
     acquisition, so the scheduled observer makes no pointless broker request.
     A normal RTH invocation still reaches the broker and fails closed on
     unexpected halt, permission, empty-history, or freshness evidence.
     Broker runtime and pure option/news ledger benchmarks now have separate
     cohesive owners (`786` and `465` lines); the shared observer identity
     belongs to the `1,000`-line append-only calibration contract rather than
     forcing the pure reducer to import broker runtime. The `98`-line CLI
     imports each directly, with no compatibility facade. All owners remain
     below the architecture ceiling while preserving the existing command and
     isolated client-ID contract.
     Every broker await on the scheduled shadow's exact path is now bounded:
     the isolated proxy and main connections each have a `15s` ceiling,
     proxy contract qualification has a `15s` ceiling at its existing owner,
     and the `2 D` historical request has a `30s` broker-native ceiling. The
     serialized broker budget is therefore at most `75s`, leaving `45s` beneath
     the unit's `2min` wall for evaluation, durable checkpointing, disconnect,
     and process teardown. A hung qualification returns no contract and fails
     before writing a checkpoint; there is no shadow-specific retry stack,
     fabricated coverage, or global request-policy change.
   - the same ledger now has one non-submitting forward command:
     `python -m tradebot.research.xsp_shadow`. It qualifies XSP exactly as
     `IND/CBOE`, reuses `IBKRClient.historical_bars_ohlcv` and the canonical
     sparse disk cache, converts IBKR bar-start rows through the same
     close-alignment owner used by live UI/backtest, and then advances the
     shared evaluator. Forecasts at or after their first outcome boundary are
     rejected; missing exact next-open or horizon-close bars remain unsettled;
     stale active-RTH data disables new freezes. A real Gateway smoke on the
     closed-session frontier produced zero forecasts, results, fills, or
     orders. Its warmed repeat reduced the broker fetch from `1 D / 78 bars /
     2.57s` to only the uncached `8580 S / 29 bars / 0.52s`, while preserving
     the same `156`-bar canonical tape.
   - the preregistered option-parity participation observer now reuses the
     canonical quote snapshot and execution-quality owners. It selects at most
     five nearest exact-expiry call/put pairs with timestamped non-crossed
     NBBO, records parity anchor, dispersion, relative spread, quote age, and
     actual market-data provenance, and aligns only the latest same-session
     capture at or before a directional decision within seven minutes.
     `2026-07-24` mechanics admitted `43/91` snapshots (`8/8` GTH, `30/74`
     RTH, `5/9` Curb); the other `48` failed solely for insufficient paired
     NBBO, and every admitted selected row remained delayed type `3`. The
     exact same-tape causal join found four unchanged directional turns, with
     one usable option context, two paired-NBBO failures, and one missing
     preceding seven-minute capture. The forward calibration receipt carries
     this as `observation_only`; it has no veto, sizing, selector,
     package-price, fill, or order authority. The already-seen one-day cash
     path is not scored for value.
     Before Monday outcomes, each first-seen option context now also freezes
     the nearest prior usable observation from the same session and expiry,
     within an untuned `15m` evidence span. It records the two capture times,
     chain fingerprints, exact elapsed seconds, parity-anchor point change,
     per-minute velocity, and raw `up/down/flat` sign. Future snapshots,
     different expiries, unusable parity, or absent history remain explicit;
     they are never reconstructed after settlement. This completes the
     preregistered “anchor moving coherently” evidence vocabulary without
     choosing a threshold or granting admission authority. One compact
     benchmark now stratifies the primary `60m` settled outcome as aligned,
     opposed, flat, or unavailable while leaving TA points and `NO_TRADE`
     untouched. The July mechanics receipt contains four pairs, only one
     usable and one diagnostic session, but zero prospective pairs/sessions;
     therefore `sample_gate=false` and those rows can never mature the forward
     gate. Forward pairs likewise remain outside the gate numerator until
     their own date has complete canonical RTH checkpoint coverage. No value
     claim, veto, or selector was born.
   - Monday's forward producer/consumer boundary is deterministic before
     deployment: `xsp_trading_date` owns the GTH/RTH/Curb tape date, the
     existing expiry selector accepts that same date for DTE, and recorder
     restart restoration and shadow auto-discovery address one file. A real
     lock-contention test proves append waits behind an active shadow reader,
     then commits one complete next row; closed weekly hours trigger no broker
     call. Direct historical, calibration, and recorder clients are
     deterministically reserved above the live runtime's rotating client-ID
     pool, preventing the continuous recorder from colliding with a shadow or
     UI connection. Keep this as one continuous existing recorder plus bounded
     non-submitting shadow invocations—not a second cache service or strategy
     daemon.
     The final adapter seam is also frozen: one chronological same-date JSONL
     containing exact `03:20/05:20/07:20/09:20 ET` GTH anchors and the first
     RTH snapshot is read in full by the real shadow CLI. The unchanged option
     reducer then returns both usable current-session parity and a complete
     `120/240/360m` pre-open path, and the isolated client disconnects. The CLI
     never prefilters the file to RTH or silently loses its GTH prefix.
     The producer is now genuinely one exchange-session process rather than a
     week-long daemon hidden behind `Restart=on-failure`. One central
     `xsp_capture_window_date` maps the scheduled `20:15..17:00 ET` window to
     its trading date while retaining the intentional `09:25..09:30`
     GTH-to-RTH transition gap. An indefinite XSP recorder started outside that
     window exits successfully without touching the broker; an in-window run
     remains bound to its original date, emits its final content-addressed
     receipt, and exits at the window boundary. Genuine in-window failures may
     still restart and restore the existing tape, while the `20h50m` systemd
     wall is only a five-minute post-close backstop. This prevents the former
     `20h45m` timeout from restarting the producer at 17:00 and silently
     carrying one service process across later sessions.
     An exact combined-source process-level q preflight then ran that real
     command under a disposable user systemd unit with
     `Restart=on-failure` and an unreachable broker port. The closed-window
     command exited `0`, systemd reported `Result=success` and `NRestarts=0`,
     and the journal contained only the explicit broker-skip receipt. The
     transient unit was stopped and left inactive; no persistent unit, timer,
     tunnel, or broker connection was created.
     Cboe's current C1 schedule independently confirms the normal boundaries:
     XSP GTH `20:15..09:25`, RTH `09:30..16:15`, and Curb `16:15..17:00 ET`.
     Its published holiday table also has exceptional GTH and early-close
     hours. Therefore the capture-window date owns process lifetime only; it
     is not evidence that a holiday venue is open, that a quote is executable,
     or that a row belongs to RTH. Actual broker timestamp/provenance and the
     explicit exchange calendar retain those authorities.
     The launch contract is frozen, but must not run until this WIP is
     integrated onto the execution host and its managed runtime is
     provisioned. The localhost authorization boundary is now proven end to
     end from q: one self-cleaning SSH forward connected isolated clients
     `979/980/981` through Mac `127.0.0.1:4001` in broker-enforced read-only
     mode, with no order or market-data request. The scheduled service depends
     on that on-demand, strict-host-key tunnel and sets `IBKR_READONLY=1`;
     the live UI remains writable by default. The pinned runtime bootstrap now
     also requires the proven Python `3.12..3.13` band and imports its two
     direct dependencies before installing units. q's clean Python `3.13.7`
     venv passes; Python `3.14.4` fails inside upstream `eventkit` before any
     TradeBot code runs, so the launch must fail explicitly rather than hide
     that incompatibility behind a process-global event-loop side effect. Run
     exactly one producer for the nearest expiry:

     ```bash
     python -m tradebot.backtest.tools.record_quotes \
       --symbol XSP --exchange CBOE --md-type 3 --dte 0 \
       --moneyness 1,2.5,5 --interval 300 --count 0 --out-dir db/quotes
     ```

     Invoke the existing consumer once after eligible completed five-minute
     RTH bars—never as a second resident evaluator:

     ```bash
     python -m tradebot.research.xsp_shadow \
       --ledger db/calibration/xsp_live_calibration.jsonl --duration '2 D'
     ```

     The producer spans GTH/RTH/Curb and records actual market-data provenance;
     the cash-direction observer remains RTH-only until independently proven
     GTH evidence earns authority. Stop the producer only with `SIGINT` or
     `SIGTERM` so it emits the final content-addressed receipt. Before launch,
     prove one source revision, one q-local Gateway endpoint/tunnel, one
     producer PID, distinct auxiliary client IDs, a fresh causal-news snapshot
     or explicit unusable reason, and empty order authority. This first forward
     session tests admission evidence and restart continuity; it does not start
     the profitable `24h` gate while `NO_TRADE` remains selected.
     The 2026-07-25 weekend preflight remains deliberately closed: q still
     executes the earlier news-only source `f837e21`, no local Gateway port is
     listening there, and this combined WIP has not been integrated. The last
     good news publication remains intact while the failed 09:52 and 14:01
     timer cycles are retained as evidence. Do not start either process from
     mixed revisions.
   - preregister timestamp-correct paired `TA-only` versus
     `TA+fundamental` decisions. Persist the exact `signal_as_of_utc`,
     `run_status`, horizon, confidence, XSP direction/impact, change class,
     drivers, and event IDs in backtest/replay/preview/shadow/live receipts.
     Historical evaluation may read only the monthly snapshot appended before
     each decision; never backfill old trades with later causal knowledge.
     Preserve separate attribution for directional/debit, credit-spread, and
     condor sleeves. Promote influence only if forward out-of-sample evidence
     improves expectancy, drawdown, adverse-selection avoidance, or structure
     choice. The first aggregate and its low-authority AI-capex corroboration
     are infrastructure evidence, not a value claim. The forward shadow now
     freezes this exact observation vocabulary and source fingerprint beside
     each first-seen forecast; missing, invalid, future, stale, and failed
     evidence remain explicitly unusable. The real CLI reads the append-time
     current and immediately preceding monthly ledgers under the producer's
     file lock, covering the contract's maximum `24h` horizon without a broad
     history scan. It adds the current `latest.json` and passes the complete
     publication sequence to the shared reducer. Each replayed turn selects
     the latest `snapshot_as_of_utc` that
     was actually visible at its own decision—not whichever publication later
     overwrote `latest.json`. A torn or structurally invalid history disables
     fundamental context without blocking the cash observer. Restart cannot
     enrich an already
     frozen decision; exact decision/outcome-slot deduplication also prevents a
     harmless earlier tape-prefix repair from creating a second, richer
     forecast. The news context still has no selector, veto, sizing, fill, or
     order authority.
     The first paired observer is frozen before Monday outcomes:
     `xsp.fundamental-defensive-observer.v1` scores only the non-overlapping
     primary `60m` horizon. It hypothetically vetoes—but never opens, reverses,
     sizes, or selects—a TA direction only when the timestamp-valid XSP
     aggregate is usable, points in the opposite direction, has impact `>=70`,
     and confidence `>=0.80`. Missing, stale, failed, neutral, aligned, or
     weaker evidence leaves the TA observation unchanged. Report paired
     TA-only versus defended points after the same `0.10`-point friction,
     avoided losses, foregone gains, veto count, and exact source fingerprint.
     These remain overlapping-event diagnostics rather than a tradable equity
     curve; no value claim, activation, or threshold retuning is allowed from
     one day or one publication.
     A mechanics-only replay of q's two genuine append-time July 24 history
     rows aligned fresh context to all four `60m` turns. The frozen bearish
     veto would have removed both `UP` observations, which later earned
     `+0.27` and `+3.51` points, while leaving two `DOWN` observations whose
     combined result was `-3.42`. Baseline diagnostic sum was `+0.36`;
     hypothetically defended was `-3.42` (`-3.78` delta). The rule was
     registered after those outcomes, so this is neither out-of-sample value
     evidence nor grounds for outcome-shaped retuning. It is an explicit
     negative warning before Monday's first prospective paired decisions.
2. **Phase 0.1 — Management and truth freeze `[DONE]`**
   - this artifact created from the full mission;
   - goal points here as the canonical resume source;
   - repository, Gateway, contract, data, and runner receipts recorded.
3. **Phase 0.2 — XSP broker and account census `[WIP]`**
   - contract, account currency, delayed data, chain, and sparse preview proven;
   - next: session-conditioned GTH/RTH/Curb option quotes/Greeks and complete
     preview economics; requested-live but actually delayed/mixed data remains
     non-executable evidence.
4. **Phase 0.3 — Backtest authenticity census `[DONE]`**
   - cache inventory and sparse hydration verified;
   - current runner cold/warm smoke measured;
   - synthetic-only boundary and incorrect spread geometry frozen.
5. **Phase 0.4 — Metrics and risk freeze `[DONE]`**
   - daily scorecard, zero-session accounting, concentration, and confidence
     bounds frozen;
   - threshold-independent economic receipts and near-instant warm reuse proven;
   - baseline/adverse friction, one-point XSP economics, conservative capacity,
    and canary eligibility frozen;
   - failed opening-reclaim and safe-income discovery families rejected without
     opening validation or holdout.
6. **Phase 1 — Data spine `[WIP]`**
7. **Phase 2 — Candidate frontier `[WIP: no admitted champion]`**
8. **Phase 3 — Research tournament `[WIP: historical families rejected]`**
9. **Phase 4 — Shadow/paper `[WIP: infrastructure only, clock not started]`**
10. **Phase 5 — Bounded live canary `[BLOCKED]`**
11. **Phase 6 — Weekly self-healing loop `[TODO]`**

---

<!-- XSP-ARCHIVE:A01:END -->

---

<a id="xsp-archive-a02-pre-distillation-frontier"></a>
<!-- XSP-ARCHIVE:A02:BEGIN -->
# Archive A02 — Pre-distillation frontier and research detail

## 0. Resume here — authoritative frontier

- **Selected strategy:** `NO_TRADE`; no candidate has earned order authority.
- **Profitability clock:** `NOT_STARTED`; the `24h → 48h → five-session` gates
  belong first to one synthetic `$1_per_XSP_point` directional shadow unit.
- **Capital authority:** none. XSP is a non-tradable index; do not fabricate a
  spot fill. Debit options, credit spreads, condors, and live option capital are
  evidence-only in this kata and graduate in a later quest.
- **Combined source:** production foundation
  `b943ea4fcd911022704278424bcb3450c5fc7d94` semantically absorbs every
  unique behavior from q's eleven-commit `f837e21..b094ce6` news delta;
  operational corrections through `0af37a1540a077e9e6e7d9be928c8aa87dcd3136`
  and the fixed-unit campaign through `64c4cd1` are exact on Mac, GitHub
  `main`, and q at E-149/E-157.
- **q execution runtime:** pinned Python `3.13.7`, `ib-insync 0.9.86`, and
  `textual 6.12.0`; all seven installed user units verify, with the changed
  tunnel/producer units byte-identical to `0af37a1`.
- **q news runtime:** the timer launches one process approximately four hours
  after completion. Its first natural combined-source cycle at
  `08:06:02..08:06:03 AEST` correctly published `no_new_evidence` without
  opening Codex or aging the causal XSP `-1/76/.94/24h` signal.
- **Historical campaign verdict:** the fixed-unit `5m/15m` families and all
  `7,556` cap-independent `30m` candidates remain rejected under their exact
  tested contracts. E-161's static pocket, E-162's walk-forward law, and E-163's
  stop-dominated expansion lifecycle remain immutable failures. E-165/E-166
  corrected the untested seam: persistent directional-impulse ownership plus
  inverse-source flip/EOD exits, without stop/trail/fizzle preemption. One
  compact opening-edge policy now leads research at `+10.98/17` recent,
  `+131.74/204` one-year, and `+120.59/1,019` five-year after frozen `$0.10`
  friction. It is XSP's first **research crown, not an operational crown**;
  its immutable lineage and future crown protocol live in
  `backtests/xsp/leaderboard.md`. The five-year daily LCB remains `-0.0202`,
  drawdown is `59.95`, and 2021/2023 lose. Freeze this plateau before
  prospective evaluation; do not keep outcome-mining the same historical
  slopes or call positive point estimates validation.
- **Forward frontier:** let the resilient read-only producer begin Sunday
  `20:15 ET` / Monday `10:15 AEST`; prove process start and restart continuity;
  inspect the exact one-shot Monday `09:37 ET` / `23:37 AEST` checkpoint made
  reboot-durable at E-151 and require `EVALUATED`,
  `order_authority=none`; only then enable `09:42..16:02 ET`.
- **Do not:** trigger an extra news run, deploy mixed revisions, backfill forward
  evidence, blindly retune rejected source thresholds, promote delayed quotes,
  submit an order, or call abstention/profitability infrastructure an economic
  win. Lifecycle sharpening must explain signal landing and exit causality,
  preserve `$0.10` friction, and challenge frozen survivors chronologically.

The original detailed mandate, hypotheses, phase prose, and execution narrative
are preserved verbatim in
[`docs/xsp-live-research-kata-narrative-archive.md`](docs/xsp-live-research-kata-narrative-archive.md)
(body SHA-256 `dfc97f209c8190bfd7dc255eb3014d5588745c682b7a0efbbf286f865dbe6f80`). The root file remains the only live task
brain, evidence ledger, decision journal, and resume authority.

---

## 1. Mission and success contract

Build one XSP-first learning system in which authentic backtests, non-submitting
forward observation, and continuous calibration share the same data, signal,
permission, economics, and evidence owners. Re-score eligible champions only on
causal, timestamp-correct evidence; never revive opaque regime routing or let a
recent winner seize an open position.

The practical target is stable, risk-adjusted positive expectancy for a
user-reported account near USD 1,000—not guaranteed income. Never force a trade
for a deadline. A safe `NO_TRADE` preserves capital but does not pass a profit
milestone. One lucky fill, synthetic option P&L, infrastructure uptime, or an
unreconciled balance cannot satisfy the quest.

### Required sequence

1. Authenticate underlying, context, option, news, and broker provenance.
2. Discover a materially new causal admission mechanism through frozen,
   chronological research; do not threshold-mine already rejected features.
3. Replay the same normalized tape through research and live-intended semantics.
4. Freeze one selected, reconciled, index-equivalent directional shadow.
5. Pass immutable net-positive `24h`, `48h`, and five-session prefixes inside
   explicit drawdown/session-loss limits.
6. Issue an honest `PROMOTE | HOLD | REVISE | QUARANTINE | STOP` verdict and
   remaining-risk register.
7. Only a later quest may translate proven direction into bounded option risk,
   broker preview, paper/canary, and tightly bounded live capital.

---

## 2. One centralized architecture

| Concern | Canonical owner | Boundary |
|---|---|---|
| Exchange calendar/session/trading date | `tradebot/engines/market.py` | One clock for cache, capture, replay, and live |
| Multihorizon direction telemetry | `tradebot/engines/directional_impulse.py` | XSP-native `5/15/30/60/120m`; no hidden selector |
| Signal source and permission plan | `tradebot/spot/entry_control.py` | One normalized plan for backtest/live/UI/journal |
| Final flat-position admission | shared `decide_flat_position_intent` kernel | Strategy permission once; operational broker readiness stays live-side |
| Historical/cache truth | `tradebot/chart_data/history.py` plus canonical cache owners | Sparse, provenance-bound, close-aligned, no fabricated gaps |
| Forward option evidence | `tradebot/backtest/tools/record_quotes.py` | One restart-safe GTH/RTH/Curb tape with actual MD provenance |
| Non-submitting broker adapter | `tradebot/research/xsp_shadow.py` | Exact XSP `IND/CBOE`; bounded requests; no order path |
| Append-only evidence/economics | `tradebot/research/live_calibration.py` | Content-addressed forecasts, results, checkpoints, immutable milestones |
| Pure candidate reducers | `tradebot/research/xsp_benchmarks.py` | Observation-only parity/news comparisons; no broker imports |
| Causal fundamental context | `tradebot/news/` | Timestamp-valid `off/observe`; no selector/order authority |
| Runtime activation | `deploy/systemd/` | One revision, read-only tunnel, producer before observer |

**Reuse-first law:** extend these owners before creating a new surface. Shared
semantics exist once and are consumed everywhere. No adapter policy copies,
parallel models, facade forests, helper sprawl, generic dumping grounds, or
module growth beyond the architecture ratchet. Centralize or delete genuinely
redundant behavior; preserve outcomes, evidence classes, and fail-closed safety.

---

## 3. Frozen research truth

### Authenticated assets

- Five-year XSP 5-minute RTH underlying: `97,530` bars through July 24; causal calendar and
  close alignment admitted. XSP volume is absent.
- Two-year SPY/VIX context: complete and provenance-bound. SPY did not improve
  XSP direction and remains diagnostic; VIX confirmation improved losing
  baselines but did not produce an admissible strategy.
- July 24 forward option tape: `91` restart-safe snapshots (`8 GTH / 74 RTH /
  9 Curb`), immutable local/q copy SHA `5d88d9a…`.
- IBKR cannot supply a historical expired-XSP chain, native BAG history, or the
  required historical NBBO/Greeks. Underlying-driven option results remain
  synthetic and cannot promote capital.
- Causal news history begins only at append time. Missing, future, stale,
  failed, or expired news means no fundamental gate.

### Directional sensor verdict

The shared XSP-native observer measures signed slope/return velocity, reversal
and acceleration, ATR-normalized movement, volatility velocity/acceleration,
efficiency, and cross-horizon coherence. Frozen historical performance:

- discovery: F1 `0.660`;
- latest 17-session tape: `66.7%` precision, `75.9%` recall, F1 `0.710`, median
  lag `1.5` bars;
- absolute-window extrema: only `16/34`; another `18/34` were boundary-censored.

This is useful causal telemetry, not yet a profitable strategy. The frozen
directional-turn family (`1,296` lifecycle cells plus its bounded admission,
reversal-cascade, VIX, and TICK/TRIN extensions) failed its chronological
economics. That verdict closes those exact families only: it did **not** fairly
exercise the wider canonical EMA/source, timing, permission, lifecycle, exit,
shock/risk, and Cartesian space on XSP. Preserve the observer for traces and
counterfactuals; do not threshold-mine its rejected one-variable slope/ATR
neighborhood. The active interaction campaign below is a separate causal
mechanism with an explicit high-density floor, not a relaxation of E-158.

### Active prospective hypotheses

- **Option parity movement:** exact-sign aligned vs TA-only, independently
  non-overlapping, complete-session-only. Minimum `30` usable pairs across at
  least five complete prospective sessions before value interpretation.
- **Option liquidity evolution:** classify pair count, parity dispersion,
  relative spread, and quote age with one threshold-free Pareto rule. The
  aligned-plus-strengthening candidate must beat both TA-only and parity-aligned
  alone on the same complete prospective sessions; it remains observation-only.
- **Pre-open bridge:** exact causal `2h/4h/6h` same-expiry GTH path into the
  first RTH context; descriptive until prospective completeness exists.
- **Fundamental defense:** timestamp-valid opposite XSP pressure with frozen
  impact/confidence thresholds, observation-only. The first retrospective
  mechanics counterexample was negative; do not retune from it. Historical
  headline sampling is plausibility-only: weak-cluster losses occur on both
  sides, while several worst days visibly reverse their opening narrative.
  Keep the frozen `60m` veto counterfactual unchanged and stratify prospective
  attribution by pressure sign plus `new/strengthening/weakening/reversal`.
- **Materially new microstructure:** top-of-book/package liquidity, spread
  state, quote evolution, and same-tape outcomes. This is the next admissible
  candidate frontier.
- **Champion namespace:** XSP starts with no HF or LF crown. Existing MNQ,
  SLV, and TQQQ crowns are diagnostic legacy artifacts, not transferable XSP
  candidates; the first XSP crown must be born from this prospective ledger.

### Primary premarket fixed-unit campaign — frozen before outcomes

- **Discovery tape:** `2026-06-29..2026-07-24`, `1,482` five-minute bars across
  `19` complete XSP RTH sessions. This is four-week selection, not validation.
- **Search identity:** existing `combo_full` engine with preset
  `xsp_candidate`; exactly `23,040` cells plus one control through the one
  canonical Cartesian runner.
- **Signal sources:** EMA `2/4, 3/7, 4/9, 8/21` in cross/trend modes; ORB
  `15/30m × RR 1/2`; opening reclaim `15/30m × 1/2` causal confirmations.
  Directional impulse remains counterfactual telemetry because its frozen
  admission/lifecycle family already failed.
- **Shared permissions:** RTH/`09-12`/`10-12 ET`; realized volatility
  off/`>=0.06`/`>=0.10`/`0.06..0.20`; base or skip-one/cooldown-two; primary
  confirmation off or Supertrend at `30m/1h/4h`.
- **Lifecycle:** three ATR target/stop geometries and two causal
  stop/trail/breakeven/fizzle/max-hold geometries; at most four entries/session;
  one open position; flatten every session.
- **Shock evidence:** off, block, or direction-aligned surf using XSP-scaled
  five-minute TR hysteresis (`1.40` on / `1.20` off / `0.05%` minimum). It was
  frozen from causal distribution measurements before campaign P&L.
- **Forbidden evidence:** XSP volume, TICK-AMEX daily bars, SPY direction,
  historical-news backfill, EMA-only spread/slope gates on non-EMA sources,
  opaque regime routing, option P&L, and delayed/live quotes.
- **Economics:** one synthetic `$1_per_XSP_point` unit, USD `1,000` starting
  equity, next-tradable-bar fills, intrabar stop/gap/drawdown accounting, and
  `$0.10` round-trip friction with no fabricated broker fill.
- **Freeze law:** require at least `10` four-week trades, retain at most `500`
  objective-diverse identities, and content-address the unchanged strategy,
  filters, tape, costs, dates, and code before challenge.
- **Challenge law:** first `2025-07-25..2026-07-24`, then
  `2021-07-26..2026-07-24`, with at least `120` trades/year independently.
  These overlapping windows are robustness challenges, not sealed validation.
- **Economic survivor:** positive net P&L after friction in every window,
  positive clustered daily lower bound and profit factor, bounded drawdown and
  concentration, and honest side/year attribution. Removing a losing side or
  retuning a gate creates a new candidate that must repeat the sequence.

### Five-minute campaign result and horizon follow-up

- The five-minute campaign completed all `23,041` cells; `10,570` met the
  four-week `10`-trade floor and `500` exact identities were frozen under
  freeze `7fe6e443…`. Four-week best net was `+14.1` points over `41` trades.
- The unchanged one-year challenge found `464/500` cadence-qualified,
  `103/500` net-positive, and `67/500` satisfying both. Every one of those `67`
  exceeded the five-year `600`-trade floor; **zero** remained net-positive.
  Best five-year net was `-17.79` points over `786` trades.
- This rejects the five-minute family economically, not for inactivity. Its
  best five-year path was roughly `+60.81` gross points before the frozen
  `78.6` points of friction, motivating a new horizon mechanism rather than a
  cheaper cost assumption or adjacent threshold retune.
- **Preregistered follow-up:** repeat the same source-safe control plane at
  `15m` and `30m`, independently run four-week discovery, retain at most `250`
  objective-diverse identities per horizon, freeze them before outcomes, then
  apply the unchanged one-year and five-year challenges and `120` trades/year.
  The hypothesis is that fewer, larger causal moves can preserve gross edge
  after the same `$0.10` friction. All forbidden evidence, economics, side/year
  attribution, and survivor gates remain unchanged.

### Fifteen/thirty-minute verdict and final breadth audit

- The `15m` lane tested all `23,041` cells, froze `250`, retained `21` after the
  one-year cadence/net gate, and retained **zero** after five years. Its best
  five-year path was `-21.79` points over `841` trades: rejected for economics,
  not inactivity.
- The `30m` lane tested all `23,041` cells and froze `250` before challenge.
  It retained `38` after one year and seven nominal finalists after five years.
  Full trade-ledger replay proves those seven collapse exactly into only two
  economic behaviors:
  - RTH: `+25.34` points / `851` trades / `36.76` drawdown;
  - 09–12 ET: `+13.10` / `723` / `27.41`.
- Neither behavior is eligible for selection. Their daily LCB95 values are
  `-0.0467/-0.0517`; both contain one rolling year that loses before friction;
  latest-year net is only `+2.29/+0.83`; both lose at `0.15` round-trip
  friction. The RTH path is positive in only `27/60` months and becomes
  negative after removing its five best days.
- RR `1/2` and shock `block/surf` are dead labels under these exact paths:
  ordered entry/exit/side/price/reason ledgers are byte-identical. They are not
  seven independent champions, and `NO_TRADE` remains selected.
- **Preregistered final bar-only breadth audit:** materialize and
  content-address every `30m` four-week candidate already satisfying the frozen
  ten-trade discovery floor before any additional outcome is read. Challenge
  the unchanged identities over five independent one-year slices, newest to
  oldest for bounded pruning, requiring at least `120` trades/year and positive
  net after the same `$0.10` friction in every slice. Finalists must then pass
  full-ledger daily-LCB, concentration, side, drawdown, and cost-sensitivity
  gates. This tests whether the `250`-identity cap hid robustness; it does not
  retune the exposed family. If none pass, historical bar-only XSP admission is
  closed and the next candidate frontier is prospective microstructure.
- The breadth cohort is now frozen before annual evaluation: all `7,556`
  qualifying identities, exact ordered records, tape/code/cost provenance, and
  the five newest-to-oldest annual gates are bound by freeze `b22a9619…`.
- The cap-independent audit is complete. Successive independent annual gates
  reduced the frozen population `7,556 → 734 → 189 → 133 → 0`; the fifth
  slice was correctly skipped because no identity survived the fourth. The
  successful run took `771.5s`, resumed `1,782` exact receipts from its
  interrupted predecessor, and changed no candidate identity or economics.
- This closes historical bar-only XSP admission under the registered
  source/gate/lifecycle family. The top-`250` cap hid no robust champion.
  Adjacent bar horizon, gate, friction, or threshold retuning is now forbidden;
  prospective quote/liquidity evolution and same-tape microstructure are the
  next admissible candidate evidence. `NO_TRADE` remains selected.
- All ignored campaign artifacts and terminal logs are checksum-identical in
  `~/Desktop/tradebot-backup/2026-07-26_18-53_AEST-xsp-campaign/` on both Mac
  and q; generated evidence remains outside the source revision by design.

### Final frozen directional meta-admission falsification

The production detector itself is stable across the seven discovery
half-years: precision remains `0.575..0.606` and recall `0.727..0.802`.
Its failure is economic admission, not a year-specific loss of slope/ATR
detection. One final bar-only test was allowed without reopening its thresholds:

- freeze the current detector, next-open fill, best E-051 lifecycle, one-unit
  economics, `$0.10` friction, and at most one open position;
- fit one L2-regularized linear admission model on `2021-07-26..2022-12-30`
  using only timestamp-available detector telemetry: direction/time/ordinal,
  readiness, direction and smoothed scores, coherence, conviction, retrace,
  ATR state, and aligned per-horizon slope/velocity/efficiency/TR;
- choose once among thresholds `0.45..0.75` on that training partition only,
  requiring positive net and daily LCB95, PF `>=1.10`, at least `0.40`
  trades/session, nonnegative P&L on both directions, drawdown `<=25`, and
  top-five positive-day concentration `<0.50`;
- rerun the normal lifecycle engine with the frozen projection over calendar
  `2023`; inspect `2024-01-01..2024-07-23` only if the identical audit gates
  pass. Do not fit features, coefficients, thresholds, or detector policy to
  either challenge;
- this is a falsification of whether the existing causal telemetry can separate
  trail winners from initial-stop losers. Failure permanently closes bar-only
  directional curation; success remains research-only until untouched and
  prospective evidence pass.

E-158 completed this exact contract: no training threshold passed, so 2023 and
2024 remained sealed and bar-only directional curation is closed.

### Multiscale interaction campaign — frozen before challenge

The recovered E-158 generator proves its best dense pocket was not a simple
gate: every single and two-label discovery bucket with at least `234` normal
engine trades remained net-negative. The new user-authorized campaign may
therefore test only the interaction the linear main-effect model could not
express:

- frozen detector, E-051 lifecycle, next-open fill, one `$1_per_XSP_point`
  unit, `$0.10` round-trip friction, one open position, four entries/session,
  both directions, and EOD flattening remain unchanged;
- slow `60/120m` signed slope/TR is a permission context; fast `5/15/30m`
  slope/velocity reversal and causal turn order provide timing; ATR ratio,
  velocity, and acceleration identify compression-to-expansion phase;
- only compact continuous products among those three families may augment the
  exact E-158 main effects. Detector thresholds, hidden regimes, volume,
  future bars, news backfill, options, and outcome-derived exits are forbidden;
- discovery is still `2021-07-26..2022-12-30`. Candidate identities include
  every coefficient, scaler, interaction, threshold, source hash, and ordered
  decision/trade ledger. No later partition may repair them;
- **density floor:** at least the E-158 dense pocket's `234` trades across
  `363` sessions (`0.6446/session`), with both directions represented. Sparse
  high-PF pockets are immediate failures;
- discovery additionally requires positive net, daily LCB95 `>0`, PF
  `>=1.10`, drawdown `<=25`, top-five positive-day concentration `<0.50`, and
  nonnegative P&L on each side. Only then open calendar `2023`; only an
  unchanged audit pass opens `2024-01-01..2024-07-23`;
- an unchanged survivor must next pass the remaining authenticated annual
  slices and current forward shadow evidence before selection. Failure closes
  this interaction family without another neighboring retune.

E-161 rejects that static identity. One adaptive mechanism is now
preregistered before its first outcome:

- the algorithm—not any monthly coefficient—is the candidate. At each calendar
  month boundary it fits one L2 ridge net-points permission model using only
  raw directional-turn shadow outcomes whose exits already matured before that
  boundary;
- trailing shadow windows are exactly `126`, `252`, or all available sessions;
  L2 is `10/40/100`; prior-score admission rates are
  `0.18/0.25/0.35/0.45`. Those `36` identities are the entire search;
- the feature vocabulary, E-051 lifecycle, costs, density floor, both-side
  requirement, and all safety/reliability gates remain E-161-identical. Each
  monthly threshold is derived only from scores available before that month;
- calendar `2023` is algorithm discovery because its static outcome is already
  exposed. Freeze the highest daily-LCB passing identity once. Challenge that
  unchanged window/L2/rate law sequentially over calendar `2024`, `2025`, and
  `2026` through July 24. A failed challenge seals every later partition;
- this requires a future live shadow ledger that matures every raw turn,
  including rejected ones. Until that owner exists and prospective parity is
  proven, even a historical pass has research authority only.

E-162 rejects all `36` adaptive identities before opening 2024. The remaining
bar-only experiment changes the source semantics rather than its admission:

- derive one causal **directional expansion** event from every existing shared
  impulse snapshot instead of waiting for `turn_event`. Use only the available
  `5/15/30m` signed slope/TR, aligned slope velocity, coherence, and ATR
  velocity/acceleration; `60/120m` slope may veto opposition only when already
  warmed and can never make an early event ready;
- grid exactly: score floor `0.25/0.35/0.45`; coherence `2/3` or `1`; aligned
  velocity floor `0/0.2/0.4`; ATR phase `off/velocity/acceleration`; end time
  `10:30/11:00/11:45 ET`; cooldown `6/12` bars; slow permission `off` or
  `aligned_if_present` — `648` identities;
- fire only when a qualified direction becomes newly active or flips after the
  cooldown. Falling below qualification rearms the transition; continuously
  qualified bars never spray repeated entries;
- freeze the same E-051 lifecycle, next-open fills, both directions, one unit,
  `$0.10` friction, EOD flattening, and the `234/363` discovery-trade floor.
  Rank passing discovery identities by daily LCB then net;
- calendar `2021-07-26..2022-12-30` is discovery. Freeze one identity before
  calendar `2023`; open `2024-01-01..2024-07-23` only if the unchanged 2023
  gates pass. No static/adaptive admission model, hidden regime, XSP volume,
  news backfill, option evidence, or future bar may participate.

E-163 rejects the directional-expansion family in discovery. Of `648`
preregistered identities, `588` cleared the `234`-event density floor and
seven were net-positive, but none had positive daily LCB95 or PF `>=1.10`.
The least-bad reliable-rank identity made `+1.59/264`, PF `1.017`, daily
LCB95 `-0.0718`; the best net was only `+4.06/357`, PF `1.033`, with its up
side negative. Calendar 2023 and the 2024 holdout remained sealed. This closes
historical bar-only direction-source curation; the next admission information
must be timestamp-correct prospective microstructure/news or another
materially independent causal class.

---

<!-- XSP-ARCHIVE:A02:END -->

---

<a id="xsp-archive-a03-completed-task-tree"></a>
<!-- XSP-ARCHIVE:A03:BEGIN -->
# Archive A03 — Completed task-tree snapshot

## 6. Ordered active task tree

1. `[DONE]` Inspect q's naturally scheduled news run exactly once after terminal;
   its exact output validates under combined source and last-good state is intact.
2. `[DONE]` Preserve a source-faithful 14-hour retrospective and reorganize the
   3,000-line task brain into one compact active authority plus a byte-preserved
   narrative archive; no research result or provenance was discarded.
3. `[DONE]` Preregister the threshold-free option-liquidity cohort/candidate and
   prove its mechanics on the already-seen July tape without prospective credit.
4. `[DONE]` Freeze one deterministic candidate recommendation: liquidity first
   only when independently eligible, then parity-aligned, otherwise `HOLD`;
   explicit run selection remains required and order authority remains zero.
5. `[DONE]` Audit every promoted HF/LF crown; no XSP crown exists and no legacy
   MNQ/SLV/TQQQ artifact is eligible for XSP calibration or selection.
6. `[DONE]` Freeze the original candidate-independent selected-shadow limits
   into every recommendation receipt before prospective outcomes.
7. `[DONE]` Define one content-addressed selected-shadow run freeze that
   rederives the current ledger recommendation, rejects stale/`HOLD`/tampered
   input, retains zero order authority, and supplies the only profitability
   policy projection.
8. `[DONE]` Semantically absorb q's complete eleven-commit news delta into the
   stronger local transactional publication owner; no separate incoming work
   or missing q behavior remains.
9. `[DONE]` Publish the full-suite-green combined tree and content-address the
   final integration manifest as clean revision `b943ea4`.
10. `[DONE]` The `5m/15m` families and all `7,556` cap-independent `30m`
    candidates are rejected. The frozen annual population fell
    `7,556 → 734 → 189 → 133 → 0`; historical bar-only admission is closed
    without changing identities, costs, evidence, or cadence. A final frozen
    multivariate admission falsification also failed its training reliability
    gate before opening 2023/2024, so detector-threshold curation is closed.
11. `[DONE]` The static multiscale interaction campaign retained 12 dense
    discovery cells; its frozen leader earned `+54.57/413` with positive
    daily LCB and both directions, then lost `-20.39/189` unchanged in calendar
    2023. The 2024 holdout stayed sealed and the static family is rejected.
12. `[DONE]` All `36` causal walk-forward identities failed calendar 2023;
    zero opened 2024. The best made only `118` trades and lost `-7.99`, while
    density-qualified variants also lost. Monthly adaptation is rejected.
13. `[DONE]` Preserve the frozen `648`-identity expansion failure and correct
    its overbroad conclusion. The shared lifecycle now follows persistent
    directional-impulse ownership; the stopless inverse-source/EOD campaign
    traced every landing, tested EMA/permission/hold interactions, retained
    `>200` trades/year, and froze the stable E-166 opening-edge plateau without
    relaxing `$0.10` friction.
14. `[WIP]` The natural combined-source news cycle passed at E-147. Producer
    recovery was strengthened at E-148 and deployed exactly at E-149 so
    recorder-owned reconnect survives a sleeping Mac/tunnel outage. E-159 proves
    full-process same-date/torn-tail/universe/read-only restart behavior.
    E-164 records the first later material natural publication; prove the
    actual Sunday recorder start and continuity. The shadow timer remains
    disabled.
15. `[WIP]` An exact reboot-durable one-shot timer now owns Monday's first
    checkpoint without arming the recurring cadence. Inspect and prove it,
    remove that temporary timer, then collect every canonical RTH slot and exact
    option/news context without backfill.
16. `[DONE]` Carry the E-166 research champion unchanged into the Monday
    non-submitting evaluator. One central candidate owner now reproduces its
    exact prefix economics, persists restart-stable counterfactual checkpoints,
    and retains zero order authority. The E-168 exit-to-flat ablation rejected
    deleting its admitted reverse handoff. Accumulate preregistered parity,
    pre-open, news, and microstructure cohorts beside it and compare identical
    TA-only and augmented decisions.
17. `[DONE]` The frozen `2/3/4 ATR × 1/1.5/2 ATR` late-profit-lock family
    improved only the latest 19 sessions and weakened every one-/five-year
    result, so E-169 rejects it without changing the crown. The shared kernel
    now truthfully permits trail/fizzle/max-hold tracking without a mandatory
    initial stop; no rejected policy was wired into live state.
18. `[DONE]` Reconstruct every crown path and separate the weak intervals into
    quiet expansion starvation and turbulent giveback/whipsaw. No one of `42`
    causal entry features binds all three clusters, and a real low-energy
    state-machine veto either changes one immaterial signal or violates the
    `>200` trades/year floor. The common expansion-financing failure remains
    evaluator telemetry; crash/rebound transitions remain a distinct peer
    research state. Crown identity and authority are unchanged.
19. `[DONE]` Repair the five-minute cache clock without inventing a successor:
    Opening Edge v1 now declares `close-time-parity-r1`, reproduces all
    `1,019` physical trades and exact economics, and remains the sole research
    crown. Retire the semantically false TICK-width direction gate; retain one
    causal signed breadth observation and an unpromoted breadth-150 challenger.
20. `[TODO]` Admit the directional candidate only after positive prospective
    economics; freeze its identity/risk limits before selection. Historical
    point estimates alone cannot start the profitability clock.
21. `[TODO]` Run the immutable `24h → 48h → five-session` shadow sequence and
    issue the final promotion/drift/risk verdict.
22. `[BLOCKED]` Any option structure or live-capital canary belongs to a later
    quest after directional causality is proven.

---

<!-- XSP-ARCHIVE:A03:END -->

---

<a id="xsp-archive-a04-evidence-registry"></a>
<!-- XSP-ARCHIVE:A04:BEGIN -->
# Archive A04 — Full evidence registry

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
| E-022 | 2026-07-24 10:56 UTC | 1 | Official-rule-aware historical retry contract | Git `957ef10` | focused `43 passed`; full `687 passed, 4 deselected` | Minute-and-larger tapes retain bounded independent-contract parallelism; same-contract requests serialize and day repairs descend to one worker. Ambiguous failures retry with smaller windows and exponential delay; explicit pacing waits 15/30 seconds. `reqHeadTimestamp` proof is reused for one hour and failed probes cool down for 15 seconds because IBKR subjects head requests to strict small-bar pacing. Only rejection, expiry, or request-before-head is called unavailable; repeated broker no-data remains unresolved and cannot delete or bless a cache gap |
| E-023 | 2026-07-24 11:02 UTC | 2 | Preregistered directional-debit discovery | This document at pushed preregistration anchor | `xsp.directional-debit.discovery.v1` | Frozen 3,456-cell filtered/unfiltered one-point CALL-up/PUT-down vertical family, adverse friction, exact discovery/validation/holdout boundary, two-half repeatability, alpha risk, concentration, and neighborhood gates. No permanent catalog entry exists before evidence |
| E-024 | 2026-07-24 11:05 UTC | 2 | Directional-debit discovery verdict | `/tmp/xsp-directional-debit-discovery-adverse-v1.json`; persistent receipts `/tmp/xsp-directional-debit-discovery-adverse-v1b.sqlite3` | semantic `886861ec…` | 3,456 adverse-friction cells completed in 82.29 seconds cold and 0.26 seconds warm; 2,049 sample-retained rows, zero positive daily LCBs, therefore zero alpha-gate passes. Validation/holdout remain sealed and no permanent strategy catalog entry is born |
| E-025 | 2026-07-24 11:15 UTC | 1 | Historical pacing-flexibility refinement | Git `e66f82b` | focused `48 passed`; full `689 passed, 4 deselected` | Bars at or below 30 seconds now obey IBKR's 15-second identical-request floor before any pacing error, while minute-and-larger XSP tapes retain the faster ordinary retry ladder. Explicit pacing escalates cooldown; broker no-data without head proof remains retryable across adaptive passes. Same-contract serialization and the normal two-independent-contract ceiling remain unchanged; no global token bucket or extra cache subsystem was added |
| E-026 | 2026-07-24 11:24 UTC | 1 | Strict forward-tape admission profile | Git `c99d31f`; prior tapes `/tmp/xsp-forward-{restart-proof.BCWo2l,capture-smoke-v4}/XSP/2026-07-24.jsonl` | focused `26 passed`; full `691 passed, 4 deselected` | One shared quote classifier now produces a declared complete/incomplete snapshot verdict from chain provenance, qualification, NBBO, age, actual live/delayed provenance, and Greeks. Exact package replay rejects missing or mismatched chain expiry before canonical pricing. Both premarket tapes correctly fail the new live profile: zero eligible streaming-live rows; the older smoke also lacks chain provenance. No historical artifact was relabeled or promoted |
| E-027 | 2026-07-24 11:28 UTC | 2 | Matched condor incremental-value audit | `/tmp/xsp-condor-incremental-value-v1.json`; preregistration `5fc6d26` | artifact `e54a19b8…`; adverse source `80154859…` | `1,152` filtered and `1,152` unfiltered condors were matched exactly to put-credit verticals after USD `1.50`/contract and two ticks. Zero condors passed the safe-income gate, so zero met the preregistered incremental-value contract or its two-neighbor rule. All unfiltered condors had negative daily LCB; filtered rows were primarily sample-, concentration-, P&L-, and profit-factor-limited. Validation and holdout remain sealed; the four-leg extension is rejected |
| E-028 | 2026-07-24 11:46 UTC | 1 | Calibration provenance and effective boundary | Git this commit | focused `16 passed`; full `693 passed, 4 deselected` | Schema-v2 records bind the delayed broker observation to source kind, actual first/last underlying bars, and a next-date effective boundary. A caller-provided RV without both source bounds is rejected; same-day replay cannot consume a newly observed surface; legacy `asof` records remain readable |
| E-029 | 2026-07-24 11:55 UTC | 1 | Five-year XSP underlying admission | `db/XSP/*5mins_rth.csv` | stitched-source manifest `591e581f…`; new 3-year shard `b7b90395…` | Canonical sparse hydration reused the existing two years and fetched only `2021-07-26..2024-07-23`: 36/36 sequential month requests, 244.18 seconds, no retries/fallbacks. The complete `2021-07-26..2026-07-23` tape has 97,452 bars/1,254 sessions, 1,244×78 normal and 10×42 early-close rows, zero unexpected counts/duplicates/effective gaps, strict timestamp order, and no nonzero XSP volume |
| E-030 | 2026-07-24 12:01 UTC | 2 | Preregistered five-year XSP credit-barrier census | This document at pushed preregistration anchor | `xsp.credit-barrier-census.v1` | Frozen 128-cell descriptive matrix: four decision times × four OTM distances × four expiration horizons × two sides; conservative whole-point geometry, touch/expiration/adverse-excursion evidence, Wilson upper bounds, annual stability, and adverse USD 10 round-trip friction. No filtering, tuning, option-PnL claim, or promotion authority |
| E-031 | 2026-07-24 12:10 UTC | 2 | Reproducible five-year XSP credit-barrier verdict | `/tmp/xsp-credit-barrier-census-v1.json`; Git this commit | artifact `29c4d73b…`; source `591e581f…`; full suite `694 passed, 4 deselected` | Shared research evidence regenerated the prior 128-cell artifact byte-identically from 97,452 bars in 1.80 seconds. Best one-percent Wilson-plus-friction hurdles were `0.1620..0.1831` same-session, `0.3026..0.3267` next-session, `0.3659..0.4461` three-session, and `0.3935..0.5059` five-session. Annual breach dispersion remains material; this screens strict-admission quotes but proves no historical option expectancy |
| E-032 | 2026-07-24 12:16 UTC | 2 | Preregistered causal XSP credit-eligibility screen | This document at pushed preregistration anchor | `xsp.credit-eligibility-screen.v1` | Frozen 128-cell discovery family over two times, two offsets, two horizons, two sides, and eight nested side-aware direction/gap/rolling-quiet contexts. Discovery must materially beat its matched unconditional barrier with block stability and neighbor support before validation or locked holdout can be read |
| E-033 | 2026-07-24 12:20 UTC | 2 | Causal XSP credit-eligibility rejection | `/tmp/xsp-credit-eligibility-screen-v1.json`; preregistration `40f2541` | artifact `e8e2e657…`; source `591e581f…` | Discovery-only run completed 128 cells over 58,554 bars/753 sessions in 1.49 seconds. Zero cells reached the frozen `0.05` required-credit improvement, so zero passed before or after neighbor support. The nearest row improved `0.0493` with only 113 observations and remains rejected; validation/holdout stay sealed |
| E-034 | 2026-07-24 12:29 UTC | 1/2 | Preregistered fresh-RTH XSP package screen | This document at pushed preregistration anchor | `xsp.forward-package-screen.v1` | Frozen interpretation of the already-scheduled opening, exact-boundary, and five-minute tapes: fixed one-point legs, live/provenance/NBBO/age/Greek gates, executable natural credit versus registered barrier, 0.10 natural-to-mid ceiling, adjacent-snapshot persistence, USD 100 risk ceiling, and fixed-leg forward replay. One session can birth only a shadow candidate |
| E-035 | 2026-07-24 12:35 UTC | 1 | XSP GTH availability and provenance probe | `/tmp/xsp-gth-{forward,delayed}-20260724/XSP/2026-07-24.jsonl` | chain `ae4679a1…` | At `08:34..08:35` ET, inside XSP GTH, requested-live underlier data returned IBKR `354` and no strike-selection price. An explicit delayed request qualified and timestamped 44 Monday-expiry options with 44 NBBO rows and 36 full-Greek rows; actual per-option provenance was mixed (`36` delayed, `8` live), with `10090` subscription warnings retained. This proves GTH chain/quote availability but fails the strict all-contract live/Greek capital gate |
| E-036 | 2026-07-24 12:38 UTC | 1 | Canonical XSP session provenance | Git this commit | focused `15 passed`; full `696 passed, 4 deselected` | Quote schema v4 records the centralized weekly Cboe session label (`GTH`, `RTH`, `CURB`, or closed) while reading schema-v3 tapes unchanged. The label explicitly does not override holiday, halt, permission, expiry-specific close, or actual quote provenance evidence |
| E-037 | 2026-07-24 12:53 UTC | 1/2 | Strategy-required quote admission amendment | Git this commit; `xsp.forward-package-screen.v1.1` | focused `13 passed` | Before RTH evidence, the shared classifier replaced the recorder's blanket all-Greeks gate with declared strategy inputs; Greeks remain captured diagnostics. Strict admission still requires exact provenance, qualification, fresh streaming NBBO for selected legs, canonical package risk, and a fresh strike anchor |
| E-038 | 2026-07-24 13:11 UTC | 2 | Prior-period put-credit neighborhood and cadence verdict | `/tmp/xsp-put-credit-prior-neighborhood-v1.json` | `2227c85b…` | `36` fixed filtered 2%-OTM one-point variants completed `180` exact runs in `42.9s` over four preceding half-years plus pooled evidence. Thirty produced positive pooled daily LCB and the largest synthetic P&L was `+1005.30`, but zero met LF cadence: the busiest had `141/502` trades versus `251` required. No sealed 2026 validation/holdout was read; this dormant family is rejected regardless of headline P&L |
| E-039 | 2026-07-24 13:28 UTC | 2 | Opening directional-unit discovery v1 | `/tmp/xsp-opening-directional-unit-v1.json` | `689bde04…` | Adaptive discovery evaluated `1,200` broad and `179` local causal signal/stop/trail/cooldown combinations in `24.4s`; zero passed and validation stayed sealed. The best momentum mechanism earned `+26.06` over `394/753` trades, PF `1.219`, six positive half-years, but only `0.523` trades/session and a negative daily LCB. The next version must add evidence-motivated renewed microstructure and authenticated pre-open context—not rerun this dry space |
| E-040 | 2026-07-24 13:40 UTC | 1 | Strict-live XSP opening entitlement receipt | `/tmp/xsp-rth-forward-20260724/XSP/2026-07-24.jsonl` | `fd5dbe9a…` | Ten exact `09:31..09:40` ET snapshots retained one stable chain fingerprint but zero qualified options: IBKR returned `354`, then `10168`, for unsubscribed XSP streaming data. Every snapshot failed closed with `no_eligible_options` and `provenance_incomplete`; delayed mechanics must remain separately labeled and cannot pass a live gate |
| E-041 | 2026-07-24 13:43 UTC | 2 | Opening directional-unit discovery v2 | `/tmp/xsp-opening-directional-unit-v2.json` | `8618ac7e…` | `2,400` renewed-breakout, flush/reclaim, pullback/reclaim, range-quality, and ratcheting-exit combinations plus local neighborhoods completed in `49.7s`; zero passed and validation stayed sealed. HF-cadence variants reached `1.34..2.17` trades/session but all lost after friction. The best sparse flush/reclaim pocket earned `+17.25`, PF `1.925`, but only `82/753` trades. Do not mine this RTH-only space again; add authenticated pre-open context |
| E-042 | 2026-07-24 14:28 UTC | 1/2 | Three-year SPY pre-open proxy hydration | `db/SPY/SPY_{2021-07-26_2022-07-23,2022-07-24_2023-07-23,2023-07-24_2024-07-23}_5mins_full24.csv`; `/tmp/xsp-spy-full24-discovery-audit.json`; `/tmp/xsp-spy-full24-2022-2023-heal.json` | `b58f7cbf…`, `36a3f249…`, `49ccd3b0…` | `48,154`/`48,142`/`71,638` canonical rows persisted with zero anomalies in the first and third shards. IBKR's SPY OVERNIGHT head is exactly `2023-07-17T00:00Z`; pre-head sessions correctly use SMART-extended expectations. Two post-head late-overnight gaps (`2023-07-18`, `2023-07-20`) survived three exact one-day retries each with explicit HMDS no-data evidence and remain declared gaps. All `04:00..09:25` pre-open anchors remain independently auditable; no bars were fabricated. |
| E-043 | 2026-07-24 14:31 UTC | 2 | Preregistered pre-open directional study v3 | `/tmp/xsp-preopen-directional-v3.json` | `9eb5837b…` | `684` matched cells over `19` genuinely HF v2 bases and `36` causal SPY pre-open contexts completed in `55.9s`; `750/753` discovery sessions had exact features, zero cells passed, and validation stayed sealed. Static upside consensus, late reversal, gap recovery, and high-range strength improved some losing bases but collapsed maximum cadence to `0.408/session`; every daily lower bound remained negative. Sparse positives (`+7.97/131` and `+6.10/24` trades) are not champions. |
| E-044 | 2026-07-24 14:43 UTC | 2 | Preregistered one-position opening ride | `/tmp/xsp-opening-ride-v1.json` | `8d77912d…` | `2,400` causal one-entry long rides and matched fixed-time twins completed in `91.8s`; zero passed and validation stayed sealed. The best cadence-qualified ride lost `-6.27` over `387/753` trades; the best fixed-time twin lost `-31.93` over `753`. Gates reduced damage but no candidate had positive P&L or daily lower bound after `0.10` friction; the visual opening spike is not a reliable long-only law. |
| E-045 | 2026-07-24 15:01 UTC | 2 | Preregistered symmetric opening ride | `/tmp/xsp-symmetric-opening-ride-v1.json` | `865c0c99…` | `2,400` one-position continuation, breakout/breakdown, and sweep-reversal cells completed reproducibly in `144.5s`; zero passed, zero stable pockets formed, and validation stayed sealed. Only `5` cells made money at any cadence. All `1,978` cells meeting `0.5` trades/session and `25%` per-side balance lost money and had negative daily lower bounds. Best cadence-qualified P&L was `-3.85/588` trades; best stability-ranked was `-7.37/686`. Symmetry reduced blind-exposure losses but did not create an opening edge. |
| E-046 | 2026-07-24 | 1 | Central entry-control and impulse-readiness spine | Current WIP | focused `135 passed` | One normalized live/backtest plan exposes source, confirmations, active permission filters, TICK, allowed directions, graph policy, and exact order. XSP full impulse readiness is a hard 25-sample cache/preflight requirement; explicit primary-regime off no longer revives EMA. The sensor remains observation-only; SPY context is separately auditable diagnostics, not an XSP entry-readiness dependency. |
| E-047 | 2026-07-24 16:26 UTC | 1 | Single spot permission-kernel convergence | Current WIP | focused `136 passed`; compile and diff checks clean | Live no longer pre-vetoes weekday or filters before lifecycle. One kernel returns the first current-day, signal-day, capacity, direction, named-filter, ATR, deferred-open, or graph blocker; backtest calls the same owner. A live-loop regression proves filter failure arrives through `spot_lifecycle`, and a mixed day/filter case proves `BLOCKED_ENTRY_DAY` wins in the canonical order. Operational contract/data/preflight/pending-order safety remains outside strategy authority. |
| E-048 | 2026-07-24 16:29 UTC | 1 | Recent SPY elapsed-horizon hydration and consumer proof | `/tmp/xsp-spy-recent-full24-sync.json`; `db/SPY/SPY_2026-06-30_2026-07-06_5mins_full24.csv`; `...07-07_07-13...`; `...07-14_07-20...`; `...07-21_07-23...` | shards `c3973ecc…`, `64f5df6b…`, `60c3d9ab…`, `aac10291…`; focused `137 passed` | Four bounded seven-day IBKR requests completed `4/4` with zero unresolved days. The canonical loader stitches `4,814` rows and zero missing ranges; every one of `17` matching XSP sessions has exact SPY `03:30/09:30 ET` anchors. The shared impulse owner now supports strict elapsed-time anchors: the scheduled overnight break yields `71` observations but exactly `360` elapsed minutes, while a naïve 72-observation sensor correctly remains unready. |
| E-049 | 2026-07-24 17:23 UTC | 1 | Entry-authority control-plane completion | Current WIP | focused `140 passed`; compile and diff checks clean | The shared plan now declares active source gates, confirmation and bear scopes, all five legacy regime-veto families, named filters, direction map, lifecycle, TICK, and graph policy. Evaluator setup consumes the same typed regime policy and scopes; the old private decoder was removed rather than retained as a facade. |
| E-050 | 2026-07-24 17:26 UTC | 1/2 | Production XSP directional-turn census | `/tmp/xsp-directional-turn-census-v1.json` | `a2f8a263ead862e4b775592d45eac3333296088ef7503efd0f96da9e91bae89e` | The production engine directly replayed `753` discovery and `17` recent complete `09:30..11:45` sessions with zero under-horizon events. Discovery: `2489` material labels, `3162` events, precision `0.590`, recall `0.749`, F1 `0.660`, median lag `2` bars. Recent: `58/66/44`, precision `0.667`, recall `0.759`, F1 `0.710`, median lag `1.5` bars. Absolute-extrema recall was only `16/34`, with `18/34` boundary-censored. SPY failed to improve recent ownership and remains diagnostics only. |
| E-051 | 2026-07-24 18:03 UTC | 2 | Preregistered directional-turn lifecycle discovery | `/tmp/xsp-directional-turn-lifecycle-v1.json` | `95ea493fd022847703b5523e358535b7a97fd14ec4b7a0447c14531f63832d28` | The normal backtest owner completed all `1,296` frozen cells over `58,554` bars / `753` discovery sessions in `193.2s` with eight workers. Zero cells passed, zero had positive net P&L/daily LCB/PF>=1/acceptable drawdown/both-direction P&L, and validation stayed sealed. Best: `-146.31`, PF `0.814`, drawdown `153.22`, `2,510` trades (`3.33/session`), all seven half-years negative. Trail exits earned `+636.47`; `1,574` initial stops lost `-784.07`, locating the next seam in admission quality rather than exit-grid breadth. |
| E-052 | 2026-07-24 18:14 UTC | 2 | Preregistered directional-turn admission discovery | `/tmp/xsp-directional-turn-admission-v1.json` | `d5788407d257bd88aeb1cbd092444a8c66e9104266dc386676fc3ec75bd7d099` | The normal lifecycle engine evaluated all `432` frozen causal admission projections in `26.0s`; the gates-off cell exactly reproduced E-051. `33` cells had positive net P&L and `8` reached PF `>=1.10`, but zero had positive daily LCB, zero profitable cells met HF cadence, zero made money in both directions, and validation stayed sealed. Best stability-ranked: `+6.04/153` trades, PF `1.154`, drawdown `9.88`, five positive half-years, but only `0.203/session` and downside `-2.65`. Best net: `+7.84/243`, only `0.323/session`. |
| E-053 | 2026-07-24 18:23 UTC | 2 | Preregistered directional reversal-cascade study | `/tmp/xsp-directional-reversal-cascade-v1.json` | `412c0e671cd419950517c6b1516b59f75ceada316628aafcf2f0570889394aa1` | All `144` causal `5/15/30m` cascade cells ran through train/audit/full normal lifecycle paths in `117.6s`. All cells lost on both `501`-session calibration and `252`-session internal audit; zero daily LCBs or two-sided P&Ls were positive, while `71` cells met cadence on both. Best compromise: train `-14.43`, audit `-9.09`, full `-23.93`, PF `0.815`; tradability hit rate `40.7%/46.6%`. External validation stayed sealed and production was unchanged. |
| E-055 | 2026-07-24 19:40 UTC | 1 | Directional/control-plane structural completion | Current WIP | architecture + ledger `8 passed`; focused `145 passed`; full `720 passed, 4 deselected` | The causal multihorizon subsystem moved intact from the oversized generic signal owner into `tradebot/engines/directional_impulse.py`; normalized entry authority moved into `tradebot/spot/entry_control.py`. Generic signals are now `422` lines, gates `984`, entry control `488`, directional impulse `622`, and journal diagnostics `997`. No architecture debt exception, compatibility facade, or parallel policy stack was introduced. |
| E-054 | 2026-07-24 18:34 UTC | 1/2 | Frozen forward RTH package-screen verdict | `/tmp/xsp-forward-package-screen-v1.json` | artifact `ed517b9d…`; boundaries `dcbf9843…`; neighbors `5c927537…`; barriers `29c4d73b…` | Four exact boundaries yielded `16` frozen one-point packages. `100/100` captured boundary contracts were qualified and anchors were fresh, but zero boundary/neighbor packages had strict streaming-live selected-leg quotes. Seven delayed/mixed boundary packages were mechanically priceable; all missed their hurdle. Closest: `11:00` 0.75% call credit `0.25` versus `0.39664`, neighbor `0.28`. Verdict `NO_TRADE`; no thresholds moved and no validation opened. |
| E-056 | 2026-07-24 20:56 UTC | 1 | Complete XSP forward-session evidence tape | q archive `/home/x/Desktop/tradebot-evidence/XSP/2026-07-24/`; audit `/tmp/xsp_session_tape_audit.json` | tape `5d88d9a…`; boundary `dcbf9843…`; audit `ac8c7f3a…`; archive `c35035eb…` | Exact final census: `91` main rows (`8` GTH / `74` RTH / `9` Curb), `83/83` scheduled rows matched, `4/4` registered boundaries matched, zero timeouts, zero invalid conIds, strict timestamps, and all chain manifests present. The original GTH runtime log is partial, so it is not misrepresented as exact `7/7` log proof; tape, source, schedules, and manifests remain independently preserved and hash-bound. |
| E-057 | 2026-07-24 21:18 UTC | 1 | Restart-safe option-recorder kernel integration | q archive `/home/x/Desktop/tradebot-evidence/XSP/2026-07-24/integration/`; content receipt `db/quotes/XSP/receipts/2026-07-24.5d88d9a9d1dd651a586368155b5769102d3800ca6100fabf14aab008b6ecdee3.json` | integration archive `477dd0ec…`; focused `56 passed`; full `726 passed, 4 deselected` | The canonical recorder now repairs torn tails, fsyncs append state, uses coherent shared-lock readers, emits content-addressed receipts, preserves absolute cadence and retained contract identity across restart/spot motion, batches requests, and reconnects with bounded backoff without duplicate rows or empty-chain fake success. The central RTH normalizer now discards after-close bars rather than collapsing them onto duplicate `16:00` timestamps. |
| E-058 | 2026-07-25 06:08 AEST | 1 | Causal-news timeout-budget correction and source integration | q unit `~/.config/systemd/user/tradebot-news.service`; canonical `deploy/systemd/tradebot-news.service` | unit/source `5fa5e092…`; focused `27 passed`; full `745 passed, 4 deselected`; systemd verify clean | The `05:42 AEST` one-shot failed at the pipeline's exact internal `600s` Codex timeout, ten minutes before systemd's outer `900s` wall; last-good qualitative memory, event ledger, aggregate, state, and history remained byte-identical. The loaded unit and canonical template now grant the ephemeral Codex `840s`, reserving `60s` for schema validation and atomic publication, with a regression pinning the nested budget. The four-hour timer remained scheduled and no manual run was triggered. |
| E-059 | 2026-07-25 07:11 AEST | 1 | Timestamp-correct fundamental observation boundary | q `~/.local/state/tradebot/news/latest.json`; canonical `tradebot/news/contract.py` and `tradebot/spot/entry_control.py` | focused `45 passed`; full `748 passed, 4 deselected` | One shared consumer validates schema/version, decision timestamp, actual run status, signal/snapshot times, XSP/MCL direction, impact, confidence, horizon, change, and driver IDs. Entry control exposes only `off/observe`; fundamental pressure is never a source gate. The real q XSP aggregate (`-1/74/.90/4h`) resolved to `usable=false`, `reason=stale`, age `20,312s` after the failed run—proving stale output cannot silently become bearish authority. |
| E-060 | 2026-07-25 07:25 AEST | 1/2 | Option-model underlier consensus census | `/tmp/xsp-option-model-consensus-census-v1.json` | artifact `5219dd7d…`; source tape `5d88d9a…`; focused `51 passed` | A fixed provenance/age/cross-sectional contract admitted `39/91` snapshots: all `8` GTH, `27` RTH, and `4` Curb. The GTH model path was coherent and its final `741.3906` anchor preceded the first captured RTH cash `741.35`, but RTH median absolute disagreement was `0.650` points, only `9/27` were within `0.25`, same-interval direction agreement was `53.8%`, and next-observed-cash agreement was `68.0%` on just `25` transitions. Verdict: useful missing-cash GTH context only; not an executable quote, directional claim, or admission gate. |
| E-061 | 2026-07-25 07:44 AEST | 1/2 | Append-only selected/counterfactual live-calibration spine | `/tmp/xsp-live-calibration-replay-v1.ydegtq/live_calibration.jsonl`; canonical `tradebot/research/{live_calibration,xsp_shadow}.py` | ledger `a092a185…` reproduced byte-identically; focused `26 passed`; full `755 passed, 4 deselected` | The recent `2026-07-01..23` RTH tape replayed `1,248` bars in `3.36s`, freezing and settling `186` causal `30/60/120m` forecasts. All selected decisions remained `NO_TRADE`, broker-fill and selected-P&L counts remained zero, and only the rejected directional observer received synthetic same-tape outcomes (`97` positive / `89` negative; overlapping-horizon sum `+9.19` points). The sum is diagnostic—not a tradable equity curve, champion, or profitable-run milestone. Prefix-bound provenance, immutable pre-outcome forecasts, exact-bar settlement, durable torn-tail repair, and deterministic replay are proven. |
| E-062 | 2026-07-25 08:09 AEST | 1/2 | Qualified non-submitting XSP forward-shadow smoke | `/tmp/xsp-forward-shadow-smoke.fJ6X97/`; canonical `tradebot/research/xsp_shadow.py` | XSP conId `137851301`; focused `99 passed`; full `759 passed, 4 deselected` | The real Gateway qualified exact `IND/CBOE` XSP and returned a canonical `156`-bar close-aligned tape. Closed-session advancement retained an empty `live_calibration.v1` ledger: zero selected/counterfactual forecasts, results, broker fills, or orders. A warmed repeat proved sparse live-tail reuse: `1 D / 78 bars / 2.57s` became `8580 S / 29 bars / 0.52s` without changing the merged tape. The forward owner rejects forecasts at/after their outcome boundary, refuses non-exact entry/outcome substitutions, and disables new active-RTH freezes when the latest complete bar is more than ten minutes old. |
| E-063 | 2026-07-25 08:29 AEST | 1/2 | Preregistered option-parity participation observer | `db/quotes/XSP/2026-07-24.jsonl`; canonical `tradebot/backtest/quotes.py` and `tradebot/research/xsp_shadow.py` | tape `5d88d9a…`; focused architecture/calibration `31 passed`; full `762 passed, 4 deselected` | Before outcome scoring, one shared exact-expiry paired-NBBO observer and a causal same-session seven-minute aligner were frozen. Mechanics admitted `43/91` snapshots: `8/8` GTH, `30/74` RTH, and `5/9` Curb; all selected observations retained actual delayed type `3`, while all `48` rejections were `insufficient_nbbo_pairs`. An exact causal join against the unchanged July 24 sensor found four turns: one had usable option context, two failed paired NBBO, and one lacked a preceding seven-minute snapshot. Forward forecasts can now retain only the context available at first freeze without changing `NO_TRADE`, retroactive enrichment, invented fills, or veto/order authority. July 24 is explicitly ineligible for value tuning because its cash path was already observed. |
| E-064 | 2026-07-25 08:42 AEST | 1 | Canonical XSP trading-date and concurrent-tape boundary | canonical `tradebot/config.py`, `tradebot/engines/market.py`, `tradebot/backtest/{data.py,calibration.py,tools/record_quotes.py}`, and `tradebot/research/xsp_shadow.py` | WIP diff `1ca71401…`; focused client/tape matrix `117 passed`; architecture/data `33 passed`; full `766 passed, 4 deselected` | One exchange-date owner maps evening GTH into the following trade date and keeps pre-open GTH/RTH/Curb together. Recorder path, restart-universe restoration, expiry DTE, and shadow auto-tape lookup now agree; the indefinite recorder makes no broker call while the normal weekly session is closed. A real shared-reader/exclusive-writer test proved the writer blocks during a shadow read and then appends a complete ordered row with no torn JSON. Direct historical, calibration, and recorder IDs are now reserved above the rotating live pool (`949/979/989` under defaults), removing a cross-process collision without adding a second allocator. No new cache, calendar, daemon, selector, fill, or order authority was introduced. |
| E-065 | 2026-07-25 08:55 AEST | 1/2 | First-seen causal-news binding in forward shadow | canonical `tradebot/research/xsp_shadow.py` and `tradebot/news/contract.py` | source `8112e416…`; focused news/calibration `31 passed`; architecture/capability `8 passed`; full `767 passed, 4 deselected` | Every new XSP shadow forecast now records the exact news snapshot fingerprint, signal/snapshot timestamps, age, run status, horizon, direction, impact, confidence, change, and driver IDs available at its causal decision time. Missing, invalid, future, stale, or failed evidence freezes an explicit unusable reason. Identity remains independent of later context, so restart returns the first record instead of retroactively enriching it. The selected decision remains `NO_TRADE`; the news state is observation-only and has no veto, sizing, selector, fill, or order authority. |
| E-066 | 2026-07-25 09:04 AEST | 1/2 | Preregistered fundamental defensive paired observer | canonical `tradebot/research/xsp_shadow.py`; null-history replay `/tmp/xsp-fundamental-null.5MtR7C/live_calibration.jsonl` | source `0f40b36a…`; null ledger `ef519eee…`; focused `36 passed`; architecture/capability `8 passed`; full `772 passed, 4 deselected` | Before any Monday outcome, one primary `60m` paired rule was frozen: only usable opposite-direction XSP news with impact `>=70` and confidence `>=.80` hypothetically vetoes the unchanged directional observation. Weak, aligned, neutral, stale, failed, missing, `30m`, and `120m` evidence cannot change it. A deterministic replay of the existing `186` settled recent forecasts produced `62` primary pairs, zero usable historical news, zero vetoes, and identical TA/defended diagnostic points (`+1.30` each), proving no retrospective enrichment. The benchmark retains `NO_TRADE`, observation-only authority, and an explicit prohibition on interpreting overlapping events as a tradable equity curve. |
| E-067 | 2026-07-25 09:24 AEST | 1 | Crash-recoverable causal-news publication generation | canonical `tradebot/news/pipeline.py` | source `768ed6fa…`; focused news `22 passed`; architecture/capability `8 passed`; full `774 passed, 4 deselected` | Individual atomic renames were replaced by one durable pending-generation journal around events, qualitative memory, monthly history, latest aggregate, and cursor state. A forced interruption after history but before latest/state, plus a deliberately torn history tail, recovered the exact frozen generation on restart, repaired the tail, appended no duplicate, advanced the cursor once, and invoked Codex only once. Existing public paths and current schema consumers remain unchanged; q's next natural run still proves only the already-deployed timeout correction until this WIP is integrated. |
| E-068 | 2026-07-25 09:24 AEST | 1 | Semantic first-seen XSP forecast slot | canonical `tradebot/research/xsp_shadow.py` | source `fad4b59b…`; focused calibration `16 passed`; full `774 passed, 4 deselected` | XSP forward forecasts now deduplicate by exact strategy-version decision/outcome slot rather than by a tape fingerprint that may legitimately change after an earlier cache repair. A replay changed only an old bar's volume—altering the causal prefix hash without altering the signal—and produced zero new forecasts; the original tape, context, timestamp, and forecast IDs remained immutable. |
| E-069 | 2026-07-25 09:34 AEST | 1/2 | First-seen causal option-parity movement | canonical `tradebot/research/xsp_shadow.py`; mechanics ledger `/tmp/xsp-parity-movement.PV4sfX/ledger.jsonl` | source `b96a207a…`; ledger `8e0ed273…`; focused calibration/quote/architecture `42 passed`; full `775 passed, 4 deselected` | The existing option-parity context now freezes the latest causal snapshot plus the nearest prior usable same-session/same-expiry observation within `15m`, retaining exact timestamps, chain provenance, elapsed seconds, point delta, per-minute velocity, and raw direction. A focused future/old/prior/latest matrix proved the later snapshot cannot leak and the closest eligible prior wins. Deterministic replay retained `12` settled horizon records from four July 24 turns: the one previously usable turn now had a causal prior (`+0.69` parity points over `602.17s`, matching the unchanged `UP` observation); the other three remained honestly unavailable. This is mechanics-only because the cash path was already seen. Missing history remains explicit; no threshold, veto, selector, fill, or order authority was added. |
| E-070 | 2026-07-25 09:43 AEST | 1 | Honest nested news runtime budget | canonical `tradebot/news/pipeline.py` and `deploy/systemd/tradebot-news.service` | source `2ed124ef…`; focused news/calibration/quote/architecture `64 passed`; full `775 passed, 4 deselected` | The timeout audit found that the same `840s` argument still reached both Finviz and Codex, so a stalled discovery request could consume the service wall before inference. The existing interface now caps discovery at `min(30s, caller ceiling)` while preserving the `840s` Codex allowance and `900s` systemd ceiling. A behavioral regression proves a default-size run sends `30s` to discovery and `840s` to inference. This WIP—not the upcoming natural q run—completes the truthful `30 + 840 + 30` budget. |
| E-071 | 2026-07-25 09:46 AEST | 1/2 | Append-time causal-news alignment counterexample | q history `/home/x/.local/state/tradebot/news/history/2026-07.jsonl`; mechanics ledger `/tmp/xsp-parity-movement.PV4sfX/ledger.jsonl` | history `59dec698…`; two publications/four unique `60m` decisions | The exact `11:15:20Z` and `15:32:09Z` publications were causally available and fresh at all four July 24 decisions. Applying the later-frozen bearish veto mechanically would remove two profitable `UP` observations (`+0.27`, `+3.51`) and retain two `DOWN` observations totaling `-3.42`, changing the overlapping diagnostic sum from `+0.36` to `-3.42` (`-3.78`). Because the policy was registered after these outcomes, this is a non-promotional negative sensitivity receipt—not out-of-sample evidence. No threshold moved; Monday remains the first prospective paired test. |
| E-072 | 2026-07-25 10:14 AEST | 1 | Natural corrected-budget exercise and canonical event clocks | q service journal `09:52:05..10:01:31 AEST`; canonical `tradebot/news/contract.py` | exact response `20,598` bytes; source `9ced1990…`; focused `64 passed`; full `775 passed, 4 deselected` | The first natural q run under the explicit `840s` child ceiling returned complete JSON in `9m26s`; it confirms the corrected path was loaded and completed, while its `566s` duration does not itself prove survival beyond the former `600s` wall. The deployed unit and behavioral regression pin the actual ceiling. Strict validation then rejected one stale `last_material_change_utc` after its evidence set changed; no publication file, state cursor, or two-row history changed. The existing validator boundary now derives stable first-seen and material-change clocks from prior ID plus exact material diff while retaining strict model-authored verification time. Replaying the exact rejected payload against the exact prior six-event ledger validated eight events and correctly classified all six retained events as materially updated. The q source remains unmodified pending safe combined-WIP integration. |
| E-073 | 2026-07-25 10:31 AEST | 1/2 | Exact VIX discovery tape and turn-admission rejection | `db/VIX/VIX_2021-07-26_2024-07-23_5mins_rth.csv`; `/tmp/xsp-vix-cache-sync-report.json`; `/tmp/xsp-vix-turn-admission-v1.json` | VIX `7200ee6f…`; sync report `d2e66885…`; result `65e0fae0…` | Canonical sparse hydration resolved exact VIX `IND/CBOE` conId `13455763` and admitted `58,554` same-timestamp bars through one successful primary batch: one thread, zero remaining days, zero repairs, zero failed requests. The preregistered `13`-cell dynamic VIX-pressure admission matrix completed in `2.9s`; zero cells passed and validation stayed sealed. Every filter improved the losing `-146.31` baseline, but the best still lost `-32.64/-20.43` in calibration/audit, lost on both directions, had zero positive half-years, and missed HF cadence. |
| E-074 | 2026-07-25 10:46 AEST | 1/2 | Forward option-parity participation benchmark | canonical `tradebot/research/xsp_shadow.py`; mechanics `/tmp/xsp-parity-participation-mechanics-v1.json` | source `d12a45c2…`; receipt `cd052963…`; full `779 passed, 4 deselected` | The existing first-seen current/prior option-parity context now feeds one `60m` classify-only receipt through the shared calibration ledger. It reports aligned/opposed/flat/unavailable cohort counts and economics while preserving exact forecast and chain provenance, `NO_TRADE`, and zero veto/selector/order authority. The already-seen July mechanics ledger deterministically produced four pairs: one aligned `+0.27`, three unavailable totaling `+0.09`, one usable session, and `sample_gate=false`; these are plumbing facts, not a value claim. The production module remains under the architecture ceiling at `997` lines with no debt exception. |
| E-075 | 2026-07-25 10:53 AEST | 1 | Isolated Monday producer/observer client ownership | canonical `tradebot/config.py` and `tradebot/research/xsp_shadow.py` | config `49ef9912…`; shadow `80b67868…`; focused `47 passed`; full `780 passed, 4 deselected` | The launch audit found that the recorder already reserved client `989`, but the non-submitting shadow still loaded the live UI's main/proxy pool and persisted-ID state. One canonical config transform now allocates the shadow's transient main/proxy/index triplet as `979/980/981`, clears the live state path, and publishes the IDs plus `order_authority=none` in its receipt. This proves the frozen one-producer/one-shadow topology cannot collide with the live dashboard or silently acquire order authority. |
| E-076 | 2026-07-25 10:56 AEST | 1 | Fail-closed weekend launch census | q execution checkout, timer/service units, news state, Gateway listeners, and local combined WIP | q source `f837e21`; last-good news `a5b30dc1…`; history `59dec698…`; local full `780 passed, 4 deselected` | systemd owns the four-hour cadence (`OnUnitInactiveSec=4h`); Codex is a one-shot child, not a sleeping resident process. The q unit now exposes the intended `840s` child inside a `15min` service wall and its next timer is 14:01 AEST. The 09:52 child returned after `9m26s`, then the older q reducer failed strict event-clock validation without mutating the 01:42 last-good publication. No q-local Gateway port is listening and the combined WIP is not deployed, so producer/shadow launch correctly remains closed. |
| E-077 | 2026-07-25 11:01 AEST | 1 | Combined-WIP shadow restart determinism | `/tmp/xsp-weekend-restart-v1/{primary,fresh}.jsonl`; `/tmp/xsp-weekend-restart-v1/receipt.json` | ledger `4512b5f6…`; receipt `b9f69d6a…`; option tape `5d88d9a…` | A bounded `9.98s` mechanics replay consumed `1,326` authentic July XSP bars with zero missing ranges plus all `91` captured option snapshots, producing `198` forecasts and `198` settled outcomes. A same-ledger rerun changed zero bytes and a fresh ledger reproduced the exact `1,114,583`-byte SHA. Current news was intentionally omitted to prevent historical look-ahead; option-parity remained unavailable instead of being retroactively enriched, and both observers retained `promotion_eligible=false`. |
| E-078 | 2026-07-25 11:05 AEST | 1 | Profitability-clock truth audit | canonical `tradebot/research/live_calibration.py`, `tradebot/research/xsp_shadow.py`, and `benchmark.future.live-backtest-drift-score` | ledger owner `9ad586ba…`; capability ledger `41bfe899…`; E-077 replay | The append-only kernel proves immutable forecasts, exact settlements, restart, and observer attribution, but intentionally has no continuous session-coverage records or milestone scorer. Every current selected decision remains `NO_TRADE`; the `198` results are overlapping observation-only counterfactual horizons. Therefore no `24h`, `48h`, or five-session profitability clock has started. A future clock must require one admissible selected strategy, continuous eligible-session evidence, reconciled selected economics, costs/drawdown, and explicit missing-data separation before elapsed time can count. |
| E-079 | 2026-07-25 11:11 AEST | 1 | Signal-independent forward evaluation checkpoints | canonical `tradebot/research/{live_calibration,xsp_shadow,xsp_shadow_cli}.py` | sources `f3f4b2b1…`, `807643f7…`, `9ccb685b…`; focused `31 passed`; full `782 passed, 4 deselected` | Each non-submitting IBKR shadow invocation now appends an idempotent coverage checkpoint only after the evaluator succeeds. Fresh close-aligned RTH is `EVALUATED`; empty/stale RTH, unsupported GTH/Curb, and closed time remain distinct. A forced evaluator crash proves it leaves zero checkpoints rather than fabricating coverage. Successful checkpoints retain trading date, session, cash-tape fingerprint, complete bars, latest close/age, option-snapshot count, and `order_authority=none` independently of forecasts. The unchanged module command delegates only CLI concerns to a `97`-line adapter; the research kernel is `966` lines and the central ledger owner `378`, with no architecture exception. |
| E-080 | 2026-07-25 11:22 AEST | 1 | DST-safe one-shot XSP shadow scheduler | canonical `deploy/systemd/tradebot-xsp-shadow.{service,timer}` | service `bb1ae1e…`; timer `756053d6…`; q `systemd-analyze verify` clean; focused `32 passed`; full `783 passed, 4 deselected` | Three New York-time calendar expressions schedule exactly `78` normal cash-RTH invocations: five at `09:37..09:57`, 72 at `10:02..15:57`, and one at `16:02`, each two minutes after a completed bar. The timer is non-persistent with zero randomized delay, so missed jobs never burst. systemd suppresses overlap for the active oneshot; a two-minute hard wall bounds each run. The service uses the existing isolated shadow clients and durable ledger, and an `ExecCondition` refuses launch until `%h/.local/share/tradebot/venv` exists. q currently has neither that runtime nor `ib_insync`, so no unit was installed or enabled. |
| E-081 | 2026-07-25 11:24 AEST | 1 | Reproducible q shadow-runtime dependency contract | canonical `requirements.txt`; disposable q `/tmp/tradebot-runtime-smoke.diSFLr` | requirements `9e05894a…`; q Python `3.13.7`; `ib-insync 0.9.86`; `textual 6.12.0` | The two direct runtime dependencies are exactly pinned to the versions already proven by the Mac project environment. A fresh disposable q virtual environment installed those pins, passed `pip check`, imported both packages, and constructed the exact `Index('XSP', 'CBOE', 'USD')` contract. No persistent q runtime was created and no unit was installed or enabled; operational launch still requires one combined source revision and proven q-local Gateway transport. |
| E-082 | 2026-07-25 11:35 AEST | 1/4 | On-demand read-only q-to-Mac shadow transport | `/tmp/xsp-q-readonly-shadow-transport-v1.json`; canonical `deploy/systemd/tradebot-ib-gateway-tunnel.service` | receipt `2dcebe4d…`; tunnel `0c0cb3c9…`; config `169f9149…`; client `e58c6d6a…`; focused `95 passed`; full `785 passed, 4 deselected` | The transport audit rejected the stale `id_rsa_mac` assumption and proved q's explicit `~/.ssh/id_rsa` under strict host-key checking. A disposable localhost forward reached Mac Gateway `127.0.0.1:4001`; the exact combined-WIP client stack connected isolated IDs `979/980/981`, reported server version `176` and one managed account, and set IBKR `readonly=True` on all three sessions. The probe made zero order and market-data calls and self-cleaned. The shadow unit now requires a non-enabled, `StopWhenUnneeded` tunnel, sets `IBKR_READONLY=1`, and leaves the live UI writable by default. q `systemd-analyze verify` passed; no persistent runtime or unit was installed and no profitability clock started. |
| E-083 | 2026-07-25 11:42 AEST | 1 | Single-session forward XSP producer contract | `/tmp/xsp-forward-producer-scheduler-v1.json`; canonical `deploy/systemd/tradebot-xsp-quotes.{service,timer}` | receipt `37366819…`; recorder `d8a519ad…`; service `de065419…`; timer `39f9fb7a…`; focused `114 passed`; full `786 passed, 4 deselected` | One non-persistent New York-time timer starts exactly one recorder at `20:15 ET` Sunday through Thursday; its `20h45m` bound ends at the following `17:00 ET` Curb close. The producer reuses the existing append-only, trading-date-partitioned tape, retained-universe restart, torn-tail repair, five-minute absolute cadence, client `989`, and on-demand tunnel. It now asks IBKR for read-only mode on every reconnect while still requesting live data and preserving the actual returned provenance. q parsed six calendar iterations and verified the complete unit graph. No unit was installed, no tape collection started, and no clock was backfilled. |
| E-084 | 2026-07-25 11:44 AEST | 1/4 | Gateway-loss coverage integrity | canonical `tests/test_live_calibration.py` | test source `3aa58971…`; focused `115 passed`; full `787 passed, 4 deselected` | A simulated unavailable tunnel fails before qualification and leaves the append-only shadow ledger with zero checkpoints. Together with the existing forced-evaluator-crash receipt, this proves neither transport loss nor evaluation failure can fabricate an `EVALUATED`, abstention, or elapsed-time record. |
| E-085 | 2026-07-25 12:03 AEST | 1/4 | Selected-economics profitability authority | canonical `tradebot/research/live_calibration.py`, `tradebot/engines/market.py`, and `tests/test_live_calibration.py`; replay `/tmp/xsp-weekend-restart-v1/primary.jsonl` | sources `9b872f95…`, `fcf43540…`, `38284450…`; focused `43 passed`; architecture/capability `8 passed`; full `790 passed, 4 deselected` | One central receipt now proves `24h`, `48h`, and seven-day/five-session gates only from an exact non-`NO_TRADE` strategy/version/config/run and complete canonical RTH slots, including 78 normal or 42 early-close evaluations. Cumulative gross-cost-net identities, realized/open marks, costs, monotonic trade attribution, drawdown, session loss, win concentration, reconciliation, and safety are independently checked. One omitted five-minute slot invalidated an otherwise profitable synthetic run; a complete six-session positive run passed all gates. Replaying the actual `4512b5f6…` weekend ledger returned `NOT_STARTED`: 198 overlapping observer forecasts/results, zero checkpoints, selected P&L `0`, and no economic milestone. |
| E-086 | 2026-07-25 12:35 AEST | 1/2 | Exact NASDAQ breadth tapes and turn-admission rejection | `/tmp/xsp-breadth-cache-sync-v1.json`; `/tmp/xsp-nasdaq-breadth-turn-admission-v1.json` | sync `bb213a5c…`; TICK `f9b0a9fc…`; TRIN `5623fbb7…`; result `42e3c8ec…`; source `00304598…`/`3e060476…`; full `792 passed, 4 deselected` | Read-only canonical hydration resolved `TICK-NASD/TRIN-NASD` as `IND/NASDAQ` conIds `26719259/26719262` and produced exact matched tapes of `58,554` bars across `753` sessions each, with explicit cash-RTH filtering and no gap/retry/timeout. Nine preregistered sign/coherence cells ran in `2.6s`; zero passed and validation stayed sealed. Every mode improved the matched losing E-051 baseline, but the strongest changed from `+4.97` calibration to `-3.85` audit, retained negative daily lower bounds, lost both audit directions, and fell to `0.53` trades/session. The combined tree is green after making one RTH fixture independent of overnight wall-clock starvation; production strategy semantics remained unchanged. |
| E-087 | 2026-07-25 12:58 AEST | 1/2 | Prospective observer provenance isolation | canonical `tradebot/research/{live_calibration,xsp_shadow}.py`; mechanics ledger `/tmp/xsp-parity-movement.PV4sfX/ledger.jsonl` | sources `fd4e7eb8…`/`ac9e1823…`; test `493c7353…`; focused `29 passed`; architecture/capability `8 passed`; full `793 passed, 4 deselected` | The immutable ledger now owns the forecast/result/evidence-mode join and observer benchmarks explicitly separate all diagnostic rows from first-seen `forward_broker_history` rows. The four already-seen July pairs remain inspectable but resolve to zero prospective pairs and zero prospective sessions. A regression built five retrospective plus 25 prospective usable pairs across five forward sessions: the diagnostic total reached 30 while `sample_gate` remained false; only after 30 genuinely prospective pairs across six sessions did the gate become true. Fundamental-veto receipts likewise expose prospective counts/economics separately. Both production owners remain below the architecture ceiling (`945`/`954` lines); no decision, veto, selector, fill, or order authority changed. |
| E-088 | 2026-07-25 13:06 AEST | 1/2/4 | Exact complete-session observer gate | canonical `tradebot/research/{live_calibration,xsp_shadow}.py` | sources `25e2de04…`/`258026ef…`; test `fbe5ebe8…`; focused `29 passed`; architecture/capability `8 passed`; full `793 passed, 4 deselected` | The preregistered “five complete sessions” condition previously counted event-bearing dates, not complete evaluator coverage. One central ledger query now admits a date only when every canonical RTH slot has one coherent `EVALUATED` signature: `78` normal or `42` early-close checkpoints within the frozen tolerance. The regression first created 25 prospective pairs across five complete sessions, then added five pairs on a sixth day with one missing checkpoint: raw prospective pairs reached `30`, but sample-eligible pairs stayed `25` and the gate remained false. Appending the missing idempotent checkpoint yielded `30` eligible pairs across six complete sessions and only then opened the mechanical sample gate. Incomplete-day pairs remain visible diagnostics. The owners remain below the ratchet (`997`/`977` lines); no strategy or order authority changed. |
| E-089 | 2026-07-25 13:09 AEST | 1/2/4 | Exact RTH session-identity gate | canonical `tradebot/research/live_calibration.py` | source `7159c748…`; test `4fd7a928…`; focused `30 passed`; architecture/capability `8 passed`; full `794 passed, 4 deselected` | Complete-session authority now requires the checkpoint's explicit `session=RTH`, not merely a timestamp near an RTH slot. A regression filled every canonical slot with GTH-labeled checkpoints and proved zero complete sessions, then proved one missing RTH slot remains incomplete and only the final idempotent RTH checkpoint admits the date. The central ledger owner remains below the architecture ceiling at `998` lines; no strategy, selector, veto, fill, or order authority changed. |
| E-090 | 2026-07-25 13:15 AEST | 1/4 | One-revision q convergence preflight | Mac `main`/combined WIP; q execution checkout `codex/news-intelligence-gate`; canonical `deploy/systemd/README.md` | local anchor `9546bad`; q head `f837e21`; merge base `a5d026c`; deployment guide `001917b5…` | The branches are intentionally divergent (`9` local-only / `13` news-branch-only commits), but all `12` files delivered by the news branch exist in the combined local tree: five remain byte-identical and seven extend the same owners with the proven timeout split, derived event clocks, crash-recoverable publication, timestamp-correct consumer, tests, capability ownership, and XSP unit instructions. No delivered news surface is missing. The deployment guide now hard-gates a clean checkout at the pushed `origin/main`, defines one isolated pinned runtime, all five forward-evidence units, user-unit verification, a manual non-submitting checkpoint proof, and only then timer enablement. q's worktree remains on its old source with the single 840-second service override; no source switch, runtime installation, timer enablement, commit, or push occurred. |
| E-091 | 2026-07-25 13:23 AEST | 1/2 | Causal publication-availability recovery | canonical `tradebot/news/{contract,pipeline}.py` | sources `c421abbd…`/`83de2539…`; test `573b5572…`; focused news/calibration `53 passed`; architecture/capability `8 passed`; full `795 passed, 4 deselected` | The audit found that an interrupted generation computed at time A could become durable at time B while its consumer gated only on A. The central consumer now requires the decision to follow both the signal and snapshot timestamps, rejects a snapshot predating its signal, and still ages the horizon from the signal. Recovery checks whether the content-addressed generation was already visible: if not, it records B and a new publication ID back into the pending journal before exposing memory, events, latest, state, or history; if already visible, it preserves A. Two forced interruptions prove both sides: a pre-`latest.json` crash produced no history row, then one restart repaired a torn tail, called Codex zero additional times, emitted exactly one generation at B, rejected it at A+30m, and admitted it at B; a post-`latest.json` crash preserved A as the already-visible availability while still reconstructing exactly one audit row. The news owners remain below the architecture ceiling (`895`/`853` lines). |
| E-092 | 2026-07-25 13:29 AEST | 1/4 | Exact q branch-convergence rehearsal | disposable `/tmp/tradebot-q-convergence.pIkB9z/repo`; canonical `deploy/systemd/README.md` | simulated target `502d748`; guide `d3543322…` | q's branch-specific fetch refspec does not create `origin/main`, and Git refuses to switch away from `f837e21` while its one service override is dirty even when those bytes exactly equal the target. The exact-lineage disposable rehearsal first reproduced that refusal, then proved the minimal safe sequence: explicitly fetch `main`; require the override to be the only dirty path; byte-compare it with both pushed `origin/main` and the installed user unit; restore only that redundant checkout copy to the old branch index; and switch directly to tracked `main`. The result was clean and byte-identical to the target. No stash, launch-time merge, force switch, reset, q worktree mutation, or running-unit change occurred. |
| E-093 | 2026-07-25 13:40 AEST | 1/4 | Closed-window broker skip and benchmark ownership split | canonical `tradebot/research/{xsp_shadow,xsp_benchmarks,xsp_shadow_cli}.py` | sources `92769272…`/`8b2bb588…`/`99eeaafc…`; test `fcee7fe3…`; focused `31 passed`; architecture/capability `8 passed`; full `796 passed, 4 deselected` | The one-shot observer now classifies known GTH/Curb, exchange-holiday, and post-early-close invocations before XSP qualification or history acquisition, appending one explicit `UNSUPPORTED_SESSION` or `CLOSED` checkpoint with `broker_request_skipped` and `order_authority=none`. A client that raises on any broker touch proves all three paths; ordinary RTH still reaches broker evidence, so an unexpected halt or outage cannot be mistaken for a calendar close. Pure defensive-news and option-parity benchmark reduction moved intact into one `286`-line ledger owner; the broker runtime fell to `781` lines and the CLI remains `98`, with direct imports, no facade, no policy duplication, and no strategy/order-authority change. |
| E-094 | 2026-07-25 13:46 AEST | 1/4 | Exact q Python compatibility boundary | disposable Mac `/tmp/tradebot-runtime-preflight.b11Quh`; disposable q `/tmp/tradebot-runtime-preflight.sCfpQQ`; canonical `deploy/systemd/README.md` | guide `ee24d5e…`; q Python `3.13.7`; Mac probe Python `3.14.4` | Both clean environments installed the exact two-line requirements and passed `pip check`. q's Python `3.13.7` imported `ib-insync 0.9.86` and `eventkit` successfully; Python `3.14.4` failed at upstream `eventkit` import because its obsolete default-loop lookup raises before TradeBot loads. The q bootstrap now invokes `/usr/bin/python3`, explicitly admits only tested `3.12..3.13`, and requires clean `ib_insync`/`textual` imports before any unit installation. No compatibility shim, persistent q venv, source switch, Gateway call, timer mutation, or publication occurred. |
| E-095 | 2026-07-25 13:47 AEST | 1/4 | Combined-source q entrypoint preflight | disposable q `/tmp/tradebot-runtime-preflight.sCfpQQ/{venv,src}` | q Python `3.13.7`; exact current WIP source | The exact current `tradebot/` tree was copied only into the disposable q runtime. Under the clean pinned environment, both `python -m tradebot.backtest.tools.record_quotes --help` and `python -m tradebot.research.xsp_shadow --help` exited zero, and the separated benchmark and broker-runtime owners imported together. This proves the intended q interpreter can load the combined entrypoints; it does not prove the unpushed Git revision, Gateway transport, market data, timer installation, or economics. No canonical q checkout, service, timer, broker, or publication was touched. |
| E-096 | 2026-07-25 13:49 AEST | 1/4 | Exact weekend shadow-command no-broker receipt | `/tmp/xsp-weekend-cli.OklCFC/{ledger.jsonl,receipt.json}` | ledger `96afd36d…`; receipt `76699540…` | The real `python -m tradebot.research.xsp_shadow` adapter ran with `IBKR_PORT=1` during a canonically closed Friday-night ET window and still exited zero. It appended exactly one `CLOSED` checkpoint, reported `broker_request_skipped=closed_calendar`, `historical_request=null`, `contract=null`, zero bars/forecasts/results, `order_authority=none`, and both observer sample gates false. This command-level proof confirms calendar truth short-circuits qualification/history even through config, tape/news lookup, CLI serialization, and disconnect cleanup; it is infrastructure evidence only and starts no profitability clock. |
| E-097 | 2026-07-25 13:52 AEST | 1/4 | Honest whole-service news timeout envelope | canonical `deploy/systemd/tradebot-news.service`; `tests/test_news_signal.py`; disposable q `/tmp/tradebot-news-16min.service` | unit `209b5b05…`; test `0f3f445e…`; focused `62 passed`; full `796 passed, 4 deselected`; q systemd verify clean | The initial correction still summed exactly to the service wall: `30s` discovery + `840s` inference + `30s` validation/publication equaled `900s`, leaving no startup, teardown, scheduling, or durable-sync margin. The combined unit now retains the evidence-sized `840s` inference ceiling under a `960s` outer wall, producing a real `90s` completion reserve after the bounded discovery call. q's loaded `900s` unit was deliberately not changed immediately before its natural scheduled run; this stronger envelope activates only with the one-revision deployment. |
| E-098 | 2026-07-25 13:53 AEST | 1/4 | Disposable q full-command no-broker preflight | q `/tmp/xsp-q-weekend-cli.hjHZhP/{ledger.jsonl,receipt.json}` under the clean combined runtime | ledger `c21bbd8b…`; receipt `b2a84838…` | The exact current combined source and clean pinned q Python `3.13.7` runtime executed the complete shadow CLI with an intentionally unreachable broker port. It returned `status=ok`, one explicit `CLOSED` checkpoint, `broker_request_skipped=closed_calendar`, no contract/history/bars/forecasts/results, broker read-only true, and order authority none. This closes the source + interpreter + adapter weekend-preflight seam without changing q's canonical checkout, persistent runtime, units, Gateway, or publication state. |
| E-099 | 2026-07-25 13:56 AEST | 1/4 | Central observer-identity ownership | canonical `tradebot/research/{live_calibration,xsp_shadow,xsp_benchmarks}.py` | sources `30fff970…`/`c8e8e94b…`/`a7fd4423…`; test `fb063635…`; focused/architecture `39 passed`; full `796 passed, 4 deselected` | The pure option/news benchmark reducer no longer imports the broker-facing shadow runtime merely to obtain its version string. `XSP_DIRECTIONAL_OBSERVER_VERSION` now belongs to the append-only calibration contract used by both consumers; tests import the same owner. Ledger/runtime/benchmark sizes are `999/786/289` lines, so the dependency direction is clean without a new constants module, duplicate literal, compatibility re-export, or architecture exception. Runtime behavior and all strategy/order authority remain unchanged. |
| E-100 | 2026-07-25 14:04 AEST | 1/4 | Content-addressed no-new-evidence refresh | canonical `tradebot/news/pipeline.py`; `tests/test_news_signal.py` | source `363b431d…`; test `636882b2…`; focused `62 passed`; full `796 passed, 4 deselected` | The no-Codex refresh path changed `run_status`, `snapshot_as_of_utc`, and time-derived event buckets while retaining the prior wrapper's `publication_id`. That made the advertised content address false even though ordinary consumers did not yet validate it. The existing refresh site now recomputes the ID after all mutations; a regression proves it differs from the prior generation and exactly hashes the new wrapper. Memory, events, signal time, horizon, one-session behavior, and order authority are unchanged; no helper or second publication protocol was added. |
| E-101 | 2026-07-25 14:07 AEST | 1/4 | Shared news content-address contract | canonical `tradebot/news/{contract,pipeline}.py`; `tests/test_news_signal.py` | sources `fd26d73a…`/`36c19e61…`; test `f00b09b8…`; focused `63 passed`; full `797 passed, 4 deselected` | Publication hashing moved from a private producer helper into the strict news contract used by creation, pending recovery, no-new refresh, and observation. The consumer accepts q's existing unaddressed legacy snapshot during migration, accepts a correctly addressed generation, and fails closed after any addressed wrapper field is mutated without re-addressing. An exact read of q's current unaddressed last-good snapshot returned both XSP and MCL as honestly `stale`/unusable rather than rejecting migration or fabricating freshness. Contract/pipeline remain `913/844` lines; no duplicate hash rule, schema bump, compatibility facade, signal-policy change, or order authority was introduced. |
| E-102 | 2026-07-25 14:12 AEST | 1/4 | Second natural timeout/derived-clock live receipt | q `tradebot-news.service` journal `14:01:31..14:11:07 AEST`; exact response replayed through current combined `validate_analysis` | raw response `94d6dc7b…` (`20,274` bytes); `295,582` Codex tokens; all five last-good hashes unchanged | The natural one-shot returned after `576s`, proving the loaded `840s` child path completed again without approaching systemd's `900s` wall. q's old reducer then rejected `active event 2.last_material_change_utc`; publication remained atomic and all last-good files retained their E-076 hashes. The exact response validates under the combined reducer: `8` active events, `71` memory lines, XSP `-1 / 79 / .94 / 24h`, and MCL `+1 / 91 / .96 / 24h`. Three retained events changed materially while copying old clocks; the canonical reducer correctly derived `2026-07-25T04:01:31Z` for all three. No manual republish, service change, source switch, Gateway call, or order occurred; the next timer is naturally scheduled for `18:11:07 AEST`. |
| E-103 | 2026-07-25 14:28 AEST | 1/4 | One-session XSP producer lifecycle | canonical `tradebot/engines/market.py`, `tradebot/backtest/tools/record_quotes.py`, and `deploy/systemd/tradebot-xsp-quotes.service` | sources `98e43d84…`/`3fe40ff4…`; unit `c889a4d1…`; focused `31 passed`; full `800 passed, 4 deselected`; q systemd verify clean | The scheduled producer previously combined an infinite closed-window loop, `Restart=on-failure`, and a `20h45m` runtime wall. A normal 17:00 timeout could therefore restart the recorder and turn one session into a continuously recycled week-long process. One central capture-window date now spans Sunday/weekday 20:15 GTH through 17:00 ET, deliberately keeps the 09:25–09:30 transition gap inside the same run, and excludes closed launch times. A command-level regression proves a closed invocation makes zero broker connections; another proves an in-window indefinite run writes one snapshot and final receipt, then exits when its window ends. The systemd wall is now a `20h50m` backstop after natural close; genuine in-window crashes still restart against the same durable tape. No quote, chain, cadence, market-data, strategy, selector, fill, or order semantics changed. |
| E-104 | 2026-07-25 14:31 AEST | 1/4 | Authoritative XSP session-boundary audit | [Cboe C1 Hours & Holidays](https://www.cboe.com/about/hours/us-options) and [XSP product specification](https://www.cboe.com/tradable_products/sp_500/mini_spx_options/specifications) | current normal hours `20:15..09:25 GTH`, `09:30..16:15 RTH`, `16:15..17:00 Curb`; 2026 holiday exceptions published separately | The central normal-session labels and one-session producer end agree with Cboe's current XSP schedule. Holiday GTH and early-close hours are explicitly exceptional, so `xsp_capture_window_date` is classified only as a process-lifetime boundary. It cannot establish market openness, RTH identity, quote freshness, entitlement, or executability; those remain broker- and exchange-calendar-evidenced. The upcoming July 27 forward session is not a published holiday. No code, timer, broker, or order action followed this source audit. |
| E-105 | 2026-07-25 14:35 AEST | 1/4 | Exact q systemd closed-producer receipt | disposable q unit `xsp-lifecycle-preflight-884EK9.service`; exact combined source `/tmp/xsp-lifecycle-preflight.884EK9/src`; clean pinned runtime from E-095 | invocation `a8f34b16edde486da018ffe5aec934e8`; `Result=success`; `ExecMainStatus=0`; `NRestarts=0` | The real indefinite XSP recorder command executed under user systemd with `Restart=on-failure`, `IBKR_PORT=1`, and the exact combined source. During the closed Saturday window it emitted only `{"broker_request_skipped":"closed_capture_window","status":"closed"}`, exited successfully, and performed no restart despite the restart policy. The unit was stopped and is inactive/dead. No persistent unit, timer, tunnel, Gateway request, tape mutation, checkpoint, order, commit, or deployment occurred. |
| E-106 | 2026-07-25 14:50 AEST | 1/2/4 | Preregistered non-overlapping parity-aligned candidate | canonical `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py`; July mechanics `/tmp/xsp-parity-movement.PV4sfX/ledger.jsonl`; restart ledger `/tmp/xsp-weekend-restart-v1/primary.jsonl` | source `f59904cc…`; test `1aac5d6c…`; mechanics receipt `8fbb2920…`; restart receipt `7cd15031…`; focused/architecture `40 passed`; full `801 passed, 4 deselected` | Before any prospective outcome, the classify-only parity observer gained one nested, observation-only value contract without changing its v1 cohorts. Independent TA-only and exact-sign-aligned sequences take their earliest eligible turn, hold the frozen `60m` horizon, and ignore all intervening turns; an adversarial `+99` point overlap was excluded while a decision exactly at the prior boundary remained eligible. The TA baseline retains every prospective complete-session decision, including unavailable parity, so thin quote coverage cannot flatter the filter. A synthetic five-session/35-pair contract proved the aligned sequence can pass only with the existing sample gate, at least two profitable trades on each side, one trade per two sessions, positive net and daily 95% LCB, positive leave-one-session-out net, bounded single-win concentration, higher mean, and lower loss rate than TA. Replaying both existing real ledgers yielded `4` and `66` diagnostic pairs but exactly zero prospective pairs, zero eligible sequence trades, and every value gate false. Passing creates only a shadow-candidate receipt; root promotion/order authority remain false and no profitability clock starts. |
| E-107 | 2026-07-25 15:08 AEST | 1/2/4 | Content-addressed ledger reads and exact directional outcomes | canonical `tradebot/research/live_calibration.py`; `tests/test_live_calibration.py`; retained July mechanics/restart ledgers from E-106 | source `10a9ccee…`; test `ded904c4…`; focused/architecture `41 passed`; full `802 passed, 4 deselected` | The ledger previously wrote content-addressed forecasts, results, and checkpoints but did not revalidate those addresses while reading, so a hand-edited selected-equity checkpoint could theoretically enter the eventual profitability clock. Every read, receipt, and append scan now rejects an unknown schema/kind, a stale forecast/result/checkpoint ID, or a stale forecast identity before any consumer receives the row. The directional join additionally admits only canonical `NO_TRADE` observer identity, exactly one matching frozen/result counterfactual, an observed timestamp equal to the frozen `60m` boundary, and an unchanged direction. A five-minute-late result, direction-reversed result, hand-edited result P&L, forecast context, and selected-equity checkpoint all failed before option, news, receipt, or profitability consumption. The real ledgers retained their exact hashes `8e0ed273…`/`4512b5f6…` and reproduced `4/4/4` and `66/66/66` pair counts. The owner remains exactly at the `1,000`-line ceiling; stored evidence, arithmetic, strategy, and authority were unchanged. |
| E-108 | 2026-07-25 15:31 AEST | 4 | Time-anchored profitability milestone authority | canonical `tradebot/research/live_calibration.py`; `tests/test_live_calibration.py` | source `f7d28a70…`; test `72c75e47…`; focused/architecture `48 passed`; full `809 passed, 4 deselected` | The original receipt evaluated every milestone from the latest aggregate, so a losing 24-hour prefix could be retroactively relabeled by a profitable week and a later gap could erase an earlier clean result. Each gate now freezes the earliest instant satisfying both its wall-clock threshold and required complete RTH-session count, then recomputes coverage, net economics, drawdown, session loss, trades, concentration, reconciliation, attribution, and safety from only evidence durably recorded by that instant. New selected checkpoints bind `recorded_at_utc` into their content address while legacy observer ledgers remain readable but cannot enter profitability. Regressions prove distinct `+2.5/+5.0/+12.5` milestone prefixes; a final `+8.0` run retaining `-4.5/-2.0` failed 24h/48h prefixes; later gaps preserve prior gates while invalidating current continuity; mid-session ownership does not demand pre-selection slots; a new run ignores an older same-version run; and GTH impersonation, false session rollups, future-start checkpoints, evidence edits, and recorded-time edits fail closed. The owner remains under the ratchet at `999` lines; no strategy, selector, broker, schedule, order, or clock-start state changed. |
| E-109 | 2026-07-25 15:42 AEST | 1/4 | Race-free one-revision q convergence transaction | canonical `deploy/systemd/README.md`; `tests/test_news_signal.py`; read-only q execution-checkout/unit census | guide `dd30d871…`; test `57bc713a…`; focused news `24 passed`; full `809 passed, 4 deselected` | The prior guide would switch q's source while its four-hour timer remained armed, omitted reinstalling the combined news unit, and required q's dirty 15-minute override to equal the new 16-minute target—so the corrected timeout envelope could remain stranded in Git or the handoff could fail exactly as designed. The transaction now stops only the timer, accepts only `inactive|failed` one-shot state, proves the sole dirty checkout file equals the still-loaded legacy unit, cleans that redundant copy, switches to pushed `origin/main`, installs all news/tunnel/XSP units from that one revision, verifies them, byte-compares the newly loaded news template, manually proves the non-submitting shadow after an eligible bar, and only then enables the three timers together. q remains unchanged on `f837e21`; its timer remains enabled and scheduled, no service was stopped, no source switched, and no commit/push/deployment/order occurred. |

| E-110 | 2026-07-25 15:58 AEST | 1/2/4 | Causal GTH-to-RTH XSP parity path | canonical `tradebot/research/xsp_shadow.py`; `tests/test_live_calibration.py` | source `fb2722fc…`; test `c519901f…`; focused/architecture `81 passed`; full `811 passed, 4 deselected` | The existing option-context owner previously required exact session-label equality, correctly preventing accidental borrowing but making every final GTH observation invisible to an RTH decision. Before any prospective outcome, one observation-only bridge now reuses the same qualified parity reduction and XSP trading-date owner: an RTH decision may retain the final usable GTH parity row no more than ten minutes before `09:30 ET`, plus exact elapsed `2h/4h/6h` same-expiry anchors no more than seven minutes before each target. It freezes timestamps, chain fingerprints, pair counts, dispersion, point changes, velocities, and raw directions; a missing horizon fails the composite path closed. A regression caught the intentional `09:25..09:30` transition gap and proves the actual final five-minute GTH close is `09:20`, while cross-date/session borrowing remains unavailable. The retained 91-row July tape honestly returns `no_usable_gth_boundary_observation` because its eight seed rows stop at `09:16 ET`; the frozen Monday five-minute cadence contains all required `03:20/05:20/07:20/09:20 ET` anchors. The ordinary seven-minute RTH parity context, `NO_TRADE`, sealed outcomes, selector, fill, order, and profitability-clock authority are unchanged. |

| E-111 | 2026-07-25 16:07 AEST | 2/4 | Fixed pre-open parity cohorts | canonical `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py` | source `d3435c47…`; test `8d1f851e…`; focused `10 passed`; focused/architecture `72 passed`; full `812 passed, 4 deselected` | Before any prospective outcome, the existing parity benchmark gained one descriptive view over the E-110 path: `aligned_all`, `opposed_all`, `reversal_into` (`6h` opposite, `2h` aligned), `mixed`, and `unavailable`. A real path fixture proves the reversal classification while root promotion, the short-horizon parity candidate, and order authority remain false. Pre-open completeness is evaluated over all prospective rows from fully checkpointed sessions rather than borrowing the existing short-horizon parity sample subset, so missing current parity cannot selectively erase an otherwise observable GTH path. No cohort is ranked, filtered, selected, or assigned a threshold; Monday remains the first eligible prospective observation. Runtime and benchmark owners remain `893/542` lines. |

| E-112 | 2026-07-25 16:15 AEST | 1/4 | Live q news unit-envelope convergence | canonical/q checkout/q loaded `deploy/systemd/tradebot-news.service` | all three unit hashes `209b5b05…`; focused timeout regression `1 passed, 23 deselected`; loaded child argument `840s`; loaded wall `960s` | The original exact `600s` child failure is no longer possible: q's already-idle unit passes `840s` beneath a `960s` whole-service wall. This changed only the loaded/template unit; q still executes the older `f837e21` pipeline, which passes that same caller ceiling to discovery and inference. Therefore the live unit has a larger wall but does not yet prove the combined source's separate `30s` fetch cap or guaranteed `90s` post-inference reserve. The user timer stayed active with its next natural run at `18:11:07 AEST`; no source switch, service start/restart, publication, Gateway call, order, commit, or push occurred. |

| E-113 | 2026-07-25 16:19 AEST | 1/2 | Historical XSP GTH availability boundary | canonical `db/XSP/`; read-only Mac Gateway `127.0.0.1:4001`; exact XSP `IND/CBOE` conId `137851301` | eight cache shards, all `*_5mins_rth.csv`; `TRADES RTH=78`; `TRADES FULL=78`; `MIDPOINT FULL=0`, IBKR `162` | The complete local XSP history contains only explicit RTH shards. Against the same `2026-07-24 17:00 ET` endpoint, `1 D / 5 mins / TRADES` with `useRTH=true` and `false` returned byte-for-byte-equivalent time coverage—`78` bars from `09:30..15:55 ET`; the full-session flag exposed no GTH rows. The only plausible index alternative, `MIDPOINT/useRTH=false`, is unsupported. This is not a transient timeout, partial cache gap, or repair candidate: historical XSP GTH underlying evidence is unavailable through this path. The isolated read-only client disconnected; no cache write, subscription, order, or account mutation occurred. |

| E-114 | 2026-07-25 16:28 AEST | 1/2/4 | Entitlement-correct forward recorder request mode | canonical `deploy/systemd/tradebot-xsp-quotes.service`; `tests/test_option_quote_capture.py`; prior broker receipts E-014/E-035/E-040 | unit `5a3294fa…`; test `a8c3fbff…`; focused `8 passed, 13 deselected`; full `812 passed, 4 deselected`; q systemd verify clean | The scheduled all-session producer incorrectly requested strict-live mode even though the authenticated account has no proven XSP streaming entitlement: live GTH could not select strikes and strict-live RTH yielded ten empty snapshots. The evidence recorder now explicitly requests delayed mode, which previously qualified and timestamped GTH contracts while retaining actual mixed `1/3` per-contract provenance. Consumers—not the request flag—continue to enforce actual streaming-live, freshness, NBBO, and preview requirements; the regression explicitly uses requested-delayed with mixed actual live/delayed rows. Mechanics/observation evidence therefore becomes reliable without weakening capital admission. No q persistent unit, timer, source, Gateway request, order, commit, or push changed. |

| E-115 | 2026-07-25 16:35 AEST | 1/4 | Exact Monday scheduler and tunnel-auth preflight | canonical `deploy/systemd/tradebot-{ib-gateway-tunnel,xsp-quotes,xsp-shadow}.*`; q `systemd-analyze calendar`; q-to-Mac batch SSH | unit hashes tunnel `0c0cb3c…`, quotes `5a3294fa…`, quotes timer `39f9fb7a…`, shadow `8a1c69d3…`, shadow timer `756053d6…`; auth `ok` | q's systemd parser resolves the next producer start to `2026-07-27 10:15 AEST` (`Sunday 20:15 ET`), the first RTH observer to `23:37 AEST` (`09:37 ET`), and the final observer to `2026-07-28 06:02 AEST` (`Monday 16:02 ET`). The exact tunnel identity and strict known-host file are readable, and the unit's batch SSH command authenticates successfully to the Mac. This proves calendar and transport preconditions only; persistent runtime/units remain deliberately undeployed until one combined revision is available, and no tunnel, Gateway request, timer, order, commit, or push was started. |

| E-116 | 2026-07-25 16:37 AEST | 1/4 | Bounded quote-recorder broker requests | canonical `tradebot/backtest/tools/record_quotes.py`; `tests/test_option_quote_capture.py` | source `4a9022ce…`; test `a8c3fbff…`; focused `4 passed, 17 deselected`; full `812 passed, 4 deselected` | `ib_insync` defaults synchronous `RequestTimeout` to zero, so the recorder's existing reconnect/backoff could never run if qualification, chain discovery, or `reqTickers` stalled. The existing recorder now sets one `45s` request ceiling; the regression injects an actual `TimeoutError` during option snapshot acquisition and proves disconnect/reconnect, one qualification, two complete nonduplicate rows, and the exact timeout policy. Connect remains separately bounded to ten seconds and repeated capture failures retain the existing `1..60s` exponential backoff. No helper, flag, second retry policy, broker call, persistent service, order, commit, or push was added. |

| E-117 | 2026-07-25 16:40 AEST | 1/2/4 | Lossless pre-open parity microstructure path | canonical `tradebot/research/xsp_shadow.py`; `tests/test_live_calibration.py` | source `9bc837ed…`; test `8ccce2b3…`; focused `7 passed, 36 deselected`; full `812 passed, 4 deselected` | The qualified parity reducer already produced median relative spread, actual per-leg market-data-type census, anchor source/value, exact selected strikes, and maximum quote age, but the new `2h/4h/6h` bridge discarded them and retained only direction/value geometry. Before the first prospective outcome, every anchor and final GTH boundary now freezes all six alongside pair count, dispersion, chain, expiry, parity value, velocity, and sign. A requested-delayed/actual-type-3 fixture proves the exact deterministic strike ordering and five-second age. The benchmark's existing path fingerprint therefore binds future local-surface, freshness, liquidity, and provenance evolution without adding an indicator, threshold, cohort, selector, veto, fill, or order authority. |

| E-118 | 2026-07-25 16:49 AEST | 1/4 | Bounded XSP shadow broker path | canonical `tradebot/client.py`; `tests/test_option_search_qualification.py`; `deploy/systemd/tradebot-xsp-shadow.service` | client `b32f576a…`; test `18c6b228…`; focused client/qualification/calibration `144 passed`; full `814 passed, 4 deselected`; outer wall `2min` | The exact scheduled observer already bounded its isolated proxy/main connections to `15s` each and its `2 D` history request to `30s`, but the single XSP `IND/CBOE` qualification await remained indefinite. The existing proxy-qualification owner now caps every batch at `15s`; a deliberately hung coroutine is cancelled and returns no qualified contract. A singleton failure is not redundantly retried as the same singleton, while multi-contract burst recovery remains unchanged. The observer therefore has a maximum serialized broker budget of `75s`, leaves `45s` for deterministic evaluation/checkpoint/disconnect beneath systemd, and fails before checkpointing if qualification never resolves. No global timeout, retry subsystem, fabricated history, broker call, persistent unit, order, commit, or push was added. |

| E-119 | 2026-07-25 16:55 AEST | 1/2/4 | Real CLI same-date GTH-to-RTH handoff | canonical `tradebot/research/xsp_shadow_cli.py`; `tradebot/research/xsp_shadow.py`; `tests/test_live_calibration.py` | test `cd0a33ae…`; focused `6 passed, 38 deselected`; full `814 passed, 4 deselected` | An adapter-level receipt writes one chronological option tape with exact `03:20/05:20/07:20/09:20 ET` GTH observations followed by the first RTH row, invokes the real shadow CLI with that file, and proves the full ordered five-row tuple reaches the shared reducer. Current RTH parity and every `120/240/360m` pre-open anchor are usable together, and the isolated client disconnects. This closes the final file-to-reducer seam without session prefiltering, alternate cache ownership, a synthetic broker fill, selector influence, or order authority. |

| E-120 | 2026-07-25 17:04 AEST | 1/4 | News unit/source lineage boundary | q pipeline `f837e21:tradebot/news/pipeline.py`; local combined pipeline; q loaded unit | q pipeline `945cf2a2…`; combined pipeline `36c19e61…`; unit `209b5b05…` | An exact source audit corrected an overstatement in E-112. q's old pipeline still declares a `600s` default, receives the unit's explicit `840s`, and passes that same value to both Finviz and Codex; the combined local pipeline alone caps discovery at `30s`. Thus the next natural q run can prove survival under the `840/960s` unit envelope and atomic old-source behavior, but cannot prove the complete `30 + 840 + 90` nested contract. That contract activates only in the already-planned one-revision deployment. No file, service, timer, publication, broker, order, commit, or push changed during this read-only audit. |

| E-121 | 2026-07-25 17:06 AEST | 1 | Exact pre-run causal-news publication baseline | q last-good publication and natural timer | memory `d6db6889…`; events `c636ba11…`; latest `a5b30dc1…`; state `69506946…`; July history `59dec698…`; pending generations `0` | Immediately before the scheduled `18:11:07 AEST` natural run, all five publication surfaces still share the last-good `01:42:01 AEST` modification epoch and retain the exact hashes that survived both derived-clock failures. No pending-generation file exists. The service remains failed from its prior invocation while its timer is active and untouched. This freezes a byte-exact before-state for one post-completion comparison; it does not inspect, trigger, delay, restart, or influence the upcoming inference. |

| E-122 | 2026-07-25 17:14 AEST | 1/2/4 | Decision-time causal-news history selection | canonical `tradebot/news/contract.py`; `tradebot/research/xsp_{shadow,shadow_cli}.py`; news/calibration tests | contract `33b9e044…`; shadow `4f7384db…`; CLI `72d79d72…`; focused news/calibration/architecture `78 passed`; full `816 passed, 4 deselected` | The scheduled shadow previously loaded only mutable `latest.json`. If a new four-hour publication landed while a missed turn from the prior 30 minutes was being replayed, that new row was correctly future-dated for the turn but the prior causally available publication had already vanished from the input. One shared-lock reader now loads the complete-replacement current and immediately preceding monthly JSONL files, covering the declared maximum `24h` signal horizon without a broad scan, rejects torn/non-object rows, and selects the latest `snapshot_as_of_utc <= decision` for each turn; `latest.json` remains an additional current generation, not a replacement for append-time causality. A fixture proves the CLI hands history plus latest to the reducer, the prior publication—not a one-second-future row—is frozen, and no-causal-history/invalid-history remains explicitly unusable. Historical replay can consume the same sequence. No threshold, selector, veto activation, order, service, publication, commit, or push changed. |

| E-123 | 2026-07-25 17:30 AEST | 1/2/4 | Current-source same-tape timestamp-parity replay | canonical close normalizer `tradebot/chart_data/history.py`; observer `tradebot/research/xsp_shadow.py`; immutable July 24 cash/option tapes | cash normalizer `c0f70183…`; observer `4f7384db…`; tape `5d88d9a…`; replay ledger `4ec73db2…` | The current combined WIP replayed `1,326` raw XSP five-minute rows through the production bar-start-to-close normalizer, then the shared evaluator against all `91` retained option snapshots. It reproduced the frozen four July 24 turns exactly at `10:10`, `10:35`, `10:50`, and `11:45 ET`: one usable causal option context, two `insufficient_nbbo_pairs`, one `no_causal_same_session_snapshot`, `189/189` forecasts/results, and zero unsettled outcomes. A deliberately incorrect diagnostic that bypassed close alignment shifted every turn five minutes earlier and reduced usable context to zero; it was discarded rather than changing policy. This proves the retained tape detects a one-bar timestamp error and that the real CLI's canonical normalization remains indispensable. No outcome, threshold, context, selector, order, service, commit, or push changed. |

| E-124 | 2026-07-25 17:37 AEST | 1/4 | Monday runtime-identity and installation-closure audit | canonical `tradebot/config.py`; XSP systemd units; read-only q runtime census | live pool `500..899`; shadow `979/980/981`; recorder `989`; focused `3 passed`; q XSP units/runtime/port absent | The persistent quote producer and five-minute observer cannot evict or reuse a live UI/Gateway API identity: the recorder owns deterministic client `989`, while each isolated shadow owns main/proxy/index `979/980/981` with no persisted client-ID state. Both run broker-read-only as q user `x`; the recorder writes the trading-date tape under `db/quotes/XSP`, the ledger atomically creates `db/calibration`, and `UMask=0077` keeps forward evidence private. q currently has no XSP/tunnel unit installed, no managed runtime, and no local `4001` listener, exactly matching the fail-closed predeployment state; the one-revision transaction must create all three together before Sunday `20:15 ET`. No runtime, directory, unit, tunnel, broker call, order, commit, or push changed. |

| E-125 | 2026-07-25 17:37 AEST | 1/4 | GTH-preserving split timer activation | canonical `deploy/systemd/README.md`; deployment-order regression | guide `d053bb44…`; test `9b00b572…`; focused `3 passed`; full `816 passed, 4 deselected` | The one-revision guide contained an impossible ordering: it withheld both XSP timers until a manual shadow proof after an eligible RTH bar, but Monday's option producer must already be armed by Sunday `20:15 ET` to retain the GTH prefix. The transaction now enables only news plus the read-only quote producer before Sunday and explicitly keeps the shadow timer disabled. At Monday `09:37 ET`, the first completed cash bar is advanced manually and must append `EVALUATED` with `order_authority=none`; only then is the `09:42..16:02` shadow schedule enabled. A regression pins producer-before-manual-before-shadow ordering. No extra unit, automatic order authority, service start, timer change on q, broker call, commit, or push occurred. |

| E-126 | 2026-07-25 17:38 AEST | 1/4 | Truthful scheduled-shadow process verdict | canonical `tradebot/research/xsp_{shadow,shadow_cli}.py`; CLI regression | observer `ff607256…`; CLI `a9eb0be4…`; test `54c1a096…`; focused `3 passed`; full `817 passed, 4 deselected` | The observer already wrote distinct `EVALUATED`, `STALE_DATA`, `NO_DATA`, `CLOSED`, and `UNSUPPORTED_SESSION` checkpoints, but its command returned success for stale active-RTH history whenever any bars existed. systemd could therefore advertise a successful slot that the complete-session reducer correctly rejected. Every receipt now exposes the exact evaluation status, and the command exits zero only for `EVALUATED`; all non-evaluated states remain durably diagnosed and nonzero so the next timer slot can retry without manufacturing coverage. A stale-command fixture proves the failure boundary. No checkpoint meaning, retry loop, benchmark, selector, order, service, commit, or push changed. |
| E-127 | 2026-07-25 17:42 AEST | 1/4 | Exact closed-session command verdict | `/tmp/xsp-shadow-closed-verdict-confirm-20260725.jsonl` | ledger `e15a225d…`; command exit `2`; one checkpoint | The real combined-source command classified Saturday as `CLOSED`, wrote exactly one content-addressed `NO_TRADE` checkpoint, recorded `broker_request_skipped=closed_calendar`, zero bars/options, `cash_history_fresh=false`, and `order_authority=none`, and returned nonzero without qualifying XSP or opening any broker session. This supersedes the old successful-process behavior recorded in E-098 while preserving its durable diagnostic. No source, service, timer, broker, order, commit, or push changed. |
| E-128 | 2026-07-25 17:56 AEST | 4 | Jitter-safe immutable profitability prefixes | canonical `tradebot/research/live_calibration.py`; profitability regression | source `95e22660…`; test `08b05787…`; live-calibration `47 passed`; architecture/capability `8 passed`; owner `999` lines | The immutable milestone reducer previously evaluated each `24h/48h/five-session` prefix at the exact canonical slot. A real timer checkpoint evaluated ten seconds later was valid under the frozen 90-second slot tolerance but absent from that prefix, so the just-due endpoint could make an otherwise continuous 24-hour run permanently incomplete. Each economic window now remains exact while its evidence cutoff deterministically includes the same frozen slot tolerance. A real-jitter fixture proves one complete profitable Monday plus Tuesday's delayed endpoint passes the 24-hour prefix at `09:38:30 ET`; no later record can rewrite it. The owner stayed below the architecture ceiling without a helper or parallel clock. No policy limit, selected strategy, service, timer, broker, order, commit, or push changed. |
| E-129 | 2026-07-25 18:01 AEST | management | Lossless task-brain centralization | canonical root brain; `docs/xsp-live-research-kata-narrative-archive.md` | root `2,935→655` lines before receipt append; archive `2,535` lines / `78ec1287…`; archived body `dfc97f20…` | The injected management brain no longer repeats completed research as mandate prose, phase prose, and a 510-line “active” narrative. One compact root now owns the authoritative state, centralized architecture, frozen research boundary, forward-runtime contract, milestone contract, ordered task tree, immutable evidence registry, and decision journal. Original Sections 0–12 were moved byte-for-byte into a linked historical archive whose body hash is verified; no adverse result, hypothesis, command, threshold, or provenance was discarded. The root remains the only resume authority and the archive is read-only historical context. No production source, test, service, timer, broker, order, commit, or push changed. |
| E-130 | 2026-07-25 18:06 AEST | 1/management | Central-owner residue audit and combined verification | XSP research/news/entry owners; combined suite | focused `125 passed`; full `818 passed, 4 deselected`; Ruff/diff clean | The surviving WIP has one direction engine, one permission plan, one append-only evidence owner, one broker adapter, one pure benchmark owner, and one news contract/pipeline pair; no parallel selector or policy stack was found. Two genuinely unused imports were removed. A test-only score constant stopped relying on a pipeline re-export and now imports its canonical contract owner directly. Two local policy lambdas became named local functions without adding a public helper. The cohesive near-ceiling owners remain intact (`live_calibration.py` 999, `xsp_shadow.py` 942, `entry_control.py` 503) rather than being filename-split into granular facades. No strategy semantics, threshold, service, timer, broker, order, commit, or push changed. |
| E-131 | 2026-07-25 18:17 AEST | 1/2/4 | Lossless prior option-liquidity state | canonical `tradebot/research/xsp_shadow.py`; `tests/test_live_calibration.py` | source `36eafecb…`; test `00dddafe…`; focused `47 passed`; full `818 passed, 4 deselected`; Ruff/diff clean | The causal parity context already retained current quote quality and prior parity value but discarded the prior pair count, dispersion, relative spread, quote age, strikes, reference, market-data provenance, and anchor source. The existing immutable `parity_change` record now freezes those already-computed prior fields so prospective evidence can measure whether liquidity strengthened or deteriorated into a turn. No derived score, threshold, cohort, signal, selector, service, broker call, order, commit, or push was added; the cohesive broker-shadow owner remains `952` lines. |
| E-132 | 2026-07-25 18:33 AEST | 1/4 | Third natural q news run and exact combined-source replay | q service journal `18:12:22..18:21:42 AEST`; frozen E-121 baseline; combined `validate_analysis` | session `019f9855-2545-70c2-9aa0-1b6a98136f8f`; raw `c7953b57…`; `100,325` tokens; runtime `560s`; all five baseline hashes unchanged | The one-shot completed inside the loaded `840/960s` envelope, then q's old `f837e21` reducer rejected `active event 3.last_material_change_utc`. The event/memory/latest/state/history files remained byte- and mtime-identical, no pending generation exists, and systemd performed zero restarts. Replaying the exact returned JSON through the combined reducer validates all eight events and 61 memory lines after centrally deriving three stale model clocks (`us-section301-forced-labor-tariffs`, `us-yield-fed-hike-repricing`, `global-equity-fund-inflow-streak`) as `2026-07-25T08:12:22Z`. The unpublished aggregate would be XSP `-1/77/.94/24h strengthening` and MCL `+1/86/.95/24h strengthening`. It was deliberately not manually republished: only the one-revision transaction may create a new live generation. No source switch, service/timer mutation, broker call, order, commit, or push occurred. |
| E-133 | 2026-07-25 18:50 AEST | 1/2/4 | Preregistered threshold-free option-liquidity candidate | canonical `tradebot/research/xsp_{shadow,benchmarks}.py`; `tests/test_live_calibration.py`; `/tmp/xsp-liquidity-mechanics.kjvl95/ledger.jsonl` | benchmark `80e464fa…`; shadow `36eafecb…`; test `f2d19cf5…`; mechanics ledger `eb63ea70…`; focused `52 passed`; full `823 passed, 4 deselected` | Before inspecting forward outcomes, the existing parity benchmark gained one Pareto classification over already-frozen current/prior pair count, dispersion, median relative spread, and quote age: improving-only `strengthening`, degrading-only `weakening`, both `mixed`, unchanged `stable`, incomplete `unavailable`. No tunable cutoff or score was added. Its independently non-overlapping aligned-plus-strengthening candidate must pass the existing sample/cadence/two-side/LCB/leave-one-session-out/concentration gates and improve both mean and loss rate over TA-only and parity-aligned alone. A synthetic five-session receipt proves the complete gate can pass without granting authority. The current-source July mechanics replay used the canonical close normalizer over `1,326` bars and all `91` option snapshots: `189/189` forecasts/results, `63` 60-minute pairs, only one usable parity/liquidity pair (`mixed`), `62 unavailable`, zero prospective/sample-eligible pairs, and both candidates false. The replay is explicitly historical and cannot mature any forward gate. All output remains observation-only, root promotion is false, and no selector, service, broker call, order, commit, or push changed. |
| E-134 | 2026-07-25 19:01 AEST | 2/4 | Deterministic shadow-candidate recommendation | canonical `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py`; E-133 mechanics ledger | benchmark `5c2bbe84…`; test `59caea92…`; recommendation `2091e4c2…`; focused `52 passed`; full `823 passed, 4 deselected` | The benchmark previously exposed two independently gated candidates but left their eventual arbitration implicit. One pure content-addressed recommendation now ranks only the preregistered choices: nominate aligned-plus-Pareto-strengthening when its stricter incremental gate passes, otherwise nominate parity-aligned when its gate passes, otherwise `HOLD`. Synthetic receipts prove each nomination path. The real July mechanics ledger returns `HOLD`, no recommended schema, and every failed gate explicitly. Even `PROMOTE` is scoped only to an explicit future selected-shadow freeze: selection authority remains absent, open-position switching is forbidden, the profitability clock remains stopped, and order authority is `none`. No state store, runtime selector, service, broker call, order, commit, or push changed; the pure benchmark owner remains `774` lines. |
| E-135 | 2026-07-25 19:06 AEST | 2/management | Exact HF/LF champion inventory and XSP eligibility boundary | canonical `tradebot/spot/champions.py`; repository `backtests/` declarations/artifacts | inventory `a80c8919…`; five refs; XSP crowns `0` | The operational loader resolves exactly five legacy README crowns: MNQ HF, SLV HF/LF, and TQQQ HF/LF. None is a machine `tradebot.spot.champion.v1` declaration; all originate in February/March research, and TQQQ LF explicitly falls back from missing v39 to v34. The artifacts retain their own symbol/track backtest semantics but have no XSP identity, current option/news provenance, complete prospective XSP sessions, or `live_calibration.v1` recommendation. Therefore none is an eligible XSP candidate and no cross-symbol bridge was added. XSP's HF/LF namespace intentionally remains empty until a prospectively nominated selected run completes its frozen evidence contract. No artifact, declaration, loader, UI, benchmark, service, broker, order, commit, or push changed. |
| E-136 | 2026-07-25 19:15 AEST | 2/4 | Candidate-independent selected-shadow risk preregistration | canonical `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py`; E-133 mechanics ledger | benchmark `91dfd49b…`; test `f211ca93…`; recommendation `a8b50dd1…`; focused `52 passed`; full `823 passed, 4 deselected` | The selected-economics verifier already enforced limits, but the active recommendation did not carry the original pre-outcome numbers. Every recommendation now content-addresses the same frozen `$1_per_XSP_point` evidence envelope: user-reported USD `1,000` design reference only, `25` points maximum drawdown, `5` points maximum session loss, at least two weekly closed trades, top-five win share `<=0.50`, and `90s` evidence tolerance. The canonical tests instantiate those shared values instead of parallel literals. Authentic July mechanics still returns `HOLD`; no strategy is selected, no clock starts, and the policy grants no broker, order, or capital authority. |
| E-137 | 2026-07-25 19:24 AEST | 2/4 | Content-addressed selected-shadow run freeze contract | canonical `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py` | benchmark `d95eab47…`; test `99a62137…`; focused `53 passed`; full `824 passed, 4 deselected`; owner `876` lines | The recommendation boundary previously required an explicit future freeze but supplied no canonical receipt. One pure owner now converts only an intact `PROMOTE` recommendation into `xsp.selected-shadow-run.v1`, binding its exact recommendation fingerprint, source-ledger hash, winning candidate evidence, selection timestamp, run/strategy/config/capital identity, and preregistered risk policy. Identical inputs reproduce the same `selection_id`; `HOLD`, a tampered fingerprint, incomplete identity, risk drift, or invalid timestamp is refused. The receipt remains selected-shadow evidence only: open-position switching is false, the profitability clock remains stopped, and order authority is `none`. It creates no file, selector, service, broker call, order, commit, or push; persistence remains closed until a real prospective candidate passes. |
| E-138 | 2026-07-25 19:31 AEST | 2/4 | Ledger-current selected-shadow freeze binding | canonical `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py` | benchmark `5c98df70…`; test `654323e5…`; focused `53 passed`; full `824 passed, 4 deselected`; owner `881` lines | An adversarial audit showed that content addressing alone proves immutability, not that a caller's recommendation came from the supplied ledger. The freeze now rederives the current recommendation from that exact ledger and requires byte-equivalent structured content before selection. A synthetic passing receipt freezes deterministically; appending one later complete-session checkpoint makes the old recommendation stale and unselectable, while its already-created `selection_id` remains unchanged. A fabricated or stale internally rehashed receipt cannot cross the boundary. No persistence, selected strategy, clock, service, broker call, order, commit, or push was added. |
| E-139 | 2026-07-26 06:35 AEST | 2/4 | Receipt-bound profitability policy | `tradebot/research/xsp_benchmarks.py`; `tests/test_live_calibration.py` | benchmark `70a25d5d…`; test `d585f2ab…`; focused `53 passed`; full `824 passed, 4 deselected` | The selected-run receipt now owns the canonical profitability `run_id` and policy; manual identity, altered receipt, or policy substitution is refused. The content-addressed selection remains non-submitting and the profitability clock remains stopped. No selection, economics, service, broker call, order, commit, or push was created. |
| E-140 | 2026-07-26 06:48 AEST | management/1/4 | Sunday pre-session truth census and timeout-boundary repair | local combined WIP; q `b094ce6`; systemd/Gateway/Cboe clocks; `tradebot/news/pipeline.py` | q clean; natural news `04:02..04:06 AEST`, status `published`, zero restarts; Mac `4001` listening; q `4001/4002` closed; zero XSP q units; pipeline `1a995808…`; news `25 passed`; full `824 passed, 4 deselected` | q has eleven newer news-only commits whose retry, clock, CLI, memory, channel, and capacity behavior overlaps the local publication/runtime work; no pull, merge, source switch, or deployment occurred. The census also caught six local failures caused by passing optional inference timeout `None` into the independently bounded HTTP fetch. One owner-local correction retains a `30s` discovery ceiling while allowing the intended unbounded inference policy; all six failures and the full suite now pass. Cboe's next ordinary XSP GTH open is Sunday `20:15 ET` / Monday `10:15 AEST`, about `27.5h` after census—not Sunday evening AEST; the first eligible RTH observer proof is Monday `09:37 ET` / `23:37 AEST`. No additional retrospective slope/OHLC tuning is justified before the independent prospective tape exists. |
| E-141 | 2026-07-26 07:12 AEST | management/1/4 | Complete q news-delta absorption and combined verification | local combined WIP; q commits `d58b6ee..b094ce6`; `tradebot/news/{pipeline.py,contract.py}`; `deploy/systemd/tradebot-news.service`; `tests/test_news_signal.py` | pipeline `0dc994e2…`; contract `6022c2ee…`; unit `799b8eed…`; test `cc807958…`; news `27 passed`; cross-boundary `145 passed`; full `826 passed, 4 deselected` | Every unique behavior in q's eleven-commit delta is now represented once inside the stronger local architecture: code-owned lifecycle clocks, canonical active drivers, stable `400`-line/`64 KiB` memory, distinct-channel capacity feedback, user Codex CLI, bounded systemd retry, and immediate capacity-refusal detection. Local content-addressed pending-publication recovery, append-time history, timestamp-correct consumers, and independent `30s` discovery budget remain intact. The q subprocess implementation was strengthened while integrating it: stdout and stderr drain concurrently, so large output cannot block timeout enforcement. No q source switch, runtime install, selection, clock start, broker call, or order occurred. |
| E-142 | 2026-07-26 07:21 AEST | 1/4 | One-revision publication and q activation preflight | Mac/GitHub/q Git identities; q pinned venv; installed systemd units and timers | all source `b943ea4fcd911022704278424bcb3450c5fc7d94`; Python `3.13.7`; `ib-insync 0.9.86`; `textual 6.12.0`; unit verify/import/help probes pass | The clean combined revision is now identical on Mac, GitHub `main`, and q. q's news and XSP quote timers are enabled; the quote producer is scheduled for Monday `10:15 AEST`; the shadow timer and all three one-shot services remain inactive. All installed unit templates are byte-identical to source and the runtime imports each entrypoint. The switch occurred while the old news one-shot was inactive. No extra news session, early capture, Gateway connection, shadow evaluation, profitability clock, broker call, or order was triggered. |
| E-143 | 2026-07-26 07:30 AEST | 1/2/4 | Synchronized q same-tape observer replay | q clean `bb3e70e`; deployed runtime code `b943ea4`; authenticated `2026-07-01..24` XSP cash shards and July 24 option tape | `1,326` cash bars; `91` option snapshots; two isolated replays `8.070s/8.063s`; byte-identical ledger `1efb5f12…`; `198/198` forecasts/results | The exact deployed q revision replayed the canonical recent tape twice through temporary ledgers and reproduced identical content. Every record remained historical diagnostic evidence: zero prospective pairs, zero complete prospective sessions, both candidates ineligible, recommendation `HOLD`, selected strategy absent, profitability clock stopped, and order authority `none`. A deliberately bounded all-five-year double replay exceeded its `300s` diagnostic wall and wrote nothing; full-history replay remains research-only while deployment smoke uses the recent chronological partition. No canonical ledger, source, timer, broker session, order, or cache changed. |
| E-144 | 2026-07-26 07:44 AEST | 1/4 | Completion-safe news inference wall | canonical `tradebot/news/pipeline.py`; `deploy/systemd/tradebot-news.service`; `tests/test_news_signal.py` | discovery `30s`; inference `840s`; service `960s`; news `27 passed`; full `826 passed, 4 deselected`; Ruff/diff clean | The q news delta had made both the child wait and service startup wall infinite. Because the four-hour timer is completion-based, a hung Codex process could then suppress every future run and never reach `Restart=on-failure`. The existing independent timeout owners now restore the previously measured envelope: Finviz remains capped at `30s`, Codex gets `840s`, and systemd retains `90s` for validation, atomic publication, and teardown. Capacity refusal still terminates immediately for bounded restart. No scoring, prompt, source selection, state, publication, broker, strategy, or order semantics changed. |
| E-145 | 2026-07-26 07:51 AEST | 1/4 | Exact synchronized q closed-window shadow command | q clean `e0c535d`; pinned runtime; isolated `/tmp/xsp-synced-closed.*` ledger | exit `2`; one `CLOSED` checkpoint; ledger `fc631593…` | The exact deployed CLI ran with an intentionally unreachable broker port during a canonically closed window. It returned broker read-only true, `historical_request=null`, `contract=null`, one content-addressed `CLOSED` checkpoint, and `order_authority=none`. This proves the synchronized adapter skips qualification/history rather than merely importing successfully. No tunnel, broker session, canonical ledger, repository file, timer, strategy, clock, or order changed. |
| E-146 | 2026-07-26 07:55 AEST | 1/4 | Monday directional warm-up window | canonical `tradebot/research/xsp_{shadow,shadow_cli}.py`; authenticated XSP cache; `tests/test_live_calibration.py` | shared duration `1 W`; actual Monday window `2026-07-20 09:37..2026-07-27 09:37 ET`; `388` cached bars; only missing range `2026-07-27`; focused `137 passed`; full `827 passed, 4 deselected`; Ruff/diff clean | The former `2 D` default was literal calendar time: Monday's first observer began on Saturday, excluded Friday, and could append a fresh `EVALUATED` checkpoint while the 25-bar directional engine remained underwarmed. One shared `1 W` constant now gives the CLI and broker adapter the same window. Against the real cache it retains 388 prior close-aligned bars and delegates only Monday's live tail to the existing sparse fetcher. No sensor threshold, direction, permission gate, candidate, broker authority, schedule, or order semantics changed. |
| E-147 | 2026-07-26 08:06 AEST | 1/4 | Natural combined-source no-evidence publication | q service `08:06:02..08:06:03 AEST`; synchronized `4df8a96`; exact pre-run hashes | success; zero restarts; `71` whitelisted, `10` active, `0` deferred; next timer `11:45 AEST`; no pending generation | The first natural cycle on the combined source found no unseen evidence and correctly opened no Codex session. It atomically advanced only `latest.json` and `state.json` to `run_status=no_new_evidence`; memory `8171c962…`, events `33964739…`, and history `6ed7b909…` remained byte- and mtime-identical. The causal signal retained its original `2026-07-25T18:02:10Z` timestamp and unchanged XSP `-1/76/.94/24h` plus MCL `+1/84/.95/24h` values rather than fabricating freshness. No broker, strategy, selector, clock, order, source, unit, or timer mutation occurred. |
| E-148 | 2026-07-26 08:09 AEST | 1/4 | Recorder-owned tunnel recovery | canonical tunnel/quote units and deployment guide; recorder retry kernel; focused unit contracts | tunnel `30s` uncapped retry; producer soft dependency; shadow hard dependency retained; q systemd parser exit `0`; focused `84 passed`; full `827 passed, 4 deselected`; Ruff/diff clean | The recorder already reconnects indefinitely with exponential backoff, resumes the same trading-date tape, repairs torn tails, restores the retained option universe, and skips catch-up bursts. Its start transaction nevertheless hard-required a tunnel capped at three rapid attempts, so a sleeping Mac at Sunday GTH open could exhaust recovery before the application loop began. The tunnel now retries calmly without a start ceiling, while only the long-running producer soft-depends on it and therefore remains alive until Gateway returns. The bounded shadow one-shot still hard-requires the tunnel and fails closed. No capture cadence, market-data type, chain, quote, strategy, clock, broker write, or order semantics changed. |
| E-149 | 2026-07-26 08:10 AEST | 1/4 | Exact producer-recovery deployment | Mac/GitHub/q `0af37a1540a077e9e6e7d9be928c8aa87dcd3136`; installed q tunnel/producer units | q clean; both units byte-identical; systemd verify clean; tunnel `RestartUSec=30s`, `StartLimitIntervalUSec=0`; producer `Wants` tunnel and has no tunnel `Requires`; services inactive; news/quote timers enabled; shadow timer disabled | The source switch and unit replacement occurred only after the natural news one-shot reached terminal and while tunnel/producer were inactive. The loaded manager now exposes the intended eventual-recovery properties without starting the tunnel, Gateway, producer, shadow, profitability clock, or any broker/order path. |
| E-150 | 2026-07-26 08:15 AEST | 1/4 | Exact first-checkpoint trigger without cadence authority | q transient `tradebot-xsp-shadow-first.{timer,service}`; canonical installed shadow one-shot | next `2026-07-27 09:37:00 America/New_York` / `23:37:00 AEST`; `AccuracyUSec=1s`; zero jitter; non-persistent; recurring shadow timer disabled | A transient timer now invokes the existing hardened shadow service exactly once after Monday's first completed XSP cash bar. It duplicates no broker, cache, evaluator, or policy configuration and cannot arm later slots. The wrapper waits up to three minutes for the canonical two-minute fail-closed one-shot; the resulting journal and calibration checkpoint remain the only acceptance evidence. No process, tunnel, Gateway, observer, clock, or order started when the timer was created. |
| E-151 | 2026-07-26 08:19 AEST | 1/4 | Reboot-durable first-checkpoint trigger | q installed `~/.config/systemd/user/tradebot-xsp-shadow-first.timer`; canonical shadow service | transient pair fully unloaded; durable timer byte-verified and enabled; `Persistent=yes`; next `2026-07-27 09:37 ET` / `23:37 AEST`; `AccuracyUSec=1s`; zero jitter; direct `Unit=tradebot-xsp-shadow.service`; recurring timer disabled | The transient trigger survived logout through the lingering user manager but not a q reboot. It was replaced before firing by one temporary installed timer that survives both and directly invokes the canonical hardened service without a wrapper. A missed wall-clock edge after downtime triggers on return, but freshness and session gates still prevent stale evidence from becoming `EVALUATED`. The timer must be disabled and removed after its one receipt is inspected. No tunnel, Gateway, observer, clock, broker, or order started during replacement. |
| E-152 | 2026-07-26 17:40 AEST | 2 | Primary five-minute fixed-unit campaign rejection | freeze `backtests/out/xsp/xsp_candidate_discovery_freeze_20260726.json`; stability `backtests/out/xsp/xsp_candidate_stability_20260726.json`; persistent cache `/tmp/xsp-discovery-v13.ocVnrU` | freeze `7fe6e443…`, file `e9036224…`; stability `2e67b2ee…`; log `24ac2148…`; `23,041` cells; `500` frozen; `67` one-year survivors; `0` five-year finalists | The source-aware five-minute campaign completed the registered four-week discovery, content-addressed freeze, ordered one-year challenge, and five-year challenge without inspecting a later window before selection. The four-week leader earned `+14.1/41` trades, while one year retained `67` candidates with both positive net and at least `120` trades. Every one cleared the five-year `600`-trade floor, but all lost after frozen friction; best was `-17.79/786` trades. Cadence therefore did not cause rejection. The least-bad path was approximately `+60.81` gross before `78.6` points of frozen friction, which preregisters a lower-turnover `15m/30m` horizon test without loosening costs or retuning the failed five-minute neighborhood. XSP volume, options, news backfill, broker fills, selection, profitability clock, and order authority remained absent. |
| E-153 | 2026-07-26 18:20 AEST | 2 | Lower-turnover horizon challenge and exact economic deduplication | `backtests/out/xsp/xsp_candidate_{15m,30m}_{discovery_freeze,stability,challenge}_20260726.json`; `backtests/out/xsp/xsp_candidate_30m_robustness_20260726.json`; persistent cache `/tmp/xsp-discovery-v13.ocVnrU` | 15m freeze `83ab25e0…`, challenge `a92040a6…`; 30m freeze `d9335aea…`, challenge `18ea10c5…`, robustness `39e07479…`; 15m `0` finalists; 30m `7` nominal / `2` economic behaviors | Deterministic resampling produced complete `15m/30m` tapes from the authenticated five-minute source with no broker request. The unchanged 15-minute challenge retained `21/250` after one year and zero after five years. The 30-minute challenge retained `38/250` after one year and seven after five years, but full-result replay hashes every ordered entry/exit/side/price/reason ledger into exactly two groups of four and three. The better RTH group earned `+25.34/851` trades over five years but daily LCB95 was `-0.0467`, one annual slice lost even before friction, latest-year net was `+2.29`, break-even friction was only `0.1298`, and removing its five best days made net negative. The 09–12 behavior was weaker. Both remain `HOLD`; no strategy, profitability clock, broker, or order authority was created. |
| E-154 | 2026-07-26 18:26 AEST | 2 | All-qualified 30-minute breadth freeze before annual outcomes | `backtests/out/xsp/xsp_candidate_30m_all_qualified_{manifest_20260726.json,freeze_20260726.json.gz}`; `/tmp/xsp-campaign-30m-breadth-freeze-20260726.log`; persistent stage key `35134189…` | `23,041` tested; `7,556` retained; freeze `b22a9619…`; stage `acb7299b…` / `72,641,400` bytes; deterministic gzip `8a9aa2e2…`; manifest `2ea97842…`; candidate IDs `27a6be88…`; log `dd42ecd5…`; replay `47.3s`, cached `23,041/23,041` | The warm canonical Cartesian runner reconstructed every four-week-qualified identity from existing receipts with zero economic recomputation, eliminating the previous top-`250` selection cap as an untested blind spot. Before any further annual result was exposed, one deterministic manifest bound every identity and exact ordered record to the same tape, code, `$1` unit, `$0.10` friction, forbidden-evidence law, and five independent annual gates in newest-to-oldest order. The compressed payload round-trips cleanly. The freeze grants research-challenge authority only; `NO_TRADE`, stopped profitability clock, and zero broker/order authority remain unchanged. |
| E-155 | 2026-07-26 18:53 AEST | 2 | Thread-safe persistent planner heartbeat | `tradebot/research/spot_sweeps/store_cache.py`; `tests/test_spot_combo_full_signature.py`; failed attempt `/tmp/xsp-campaign-30m-breadth-stability-attempt1-20260726.log` | failed log `3ed2225c…`; source `d63d216b…`; test `c0ffb8d6…`; focused `4 passed`; full `831 passed, 4 deselected`; Ruff/diff clean | The first breadth challenge exposed a runtime defect: evaluation progress came from a background thread, but the shared SQLite connection retained its default creator-thread restriction. Its swallowed thread-affinity error left the persisted heartbeat stale and made the parent falsely recycle healthy workers after `210s`. The canonical connection now permits the heartbeat thread while the existing store lock continues to serialize every access. A real-thread regression proves persisted progress/ETA round-trip; the resumed campaign reused `1,782` exact receipts and completed with advancing parent heartbeat. Strategy, cache identity, economics, broker, and order semantics did not change. |
| E-156 | 2026-07-26 18:53 AEST | 2 | Cap-independent 30-minute bar-only rejection | freeze `b22a9619…`; `backtests/out/xsp/xsp_candidate_30m_breadth_stability_20260726.json`; `/tmp/xsp-campaign-30m-breadth-stability-20260726.log` | stability `80f653db…`; log `f47265e5…`; `7,556 → 734 → 189 → 133 → 0`; `771.5s`; full `831 passed, 4 deselected` | Every four-week-qualified identity entered the preregistered newest-to-oldest annual challenge unchanged. Positive net after frozen friction plus `120` trades/year reduced the population to zero in the fourth independent slice, so the fifth was correctly skipped and no full-ledger promotion gate was opened. This eliminates the top-`250` cap as the explanation for prior rejection and closes adjacent historical bar-only XSP mining. No XSP crown, selection, profitability clock, broker session, or order authority was created; prospective microstructure is now the sole admissible candidate frontier. |
| E-157 | 2026-07-26 19:03 AEST | management/1/2/4 | Fixed-unit campaign publication and evidence durability | Mac/GitHub/q `64c4cd12bb4f2af193b35aa0d785c6f3d4f3c9ac`; `~/Desktop/tradebot-backup/2026-07-26_18-53_AEST-xsp-campaign/` on Mac and q | all Git refs exact and clean; q Python `3.13.7` compile/import smoke passed; backup checksum dry-run empty; services inactive; quote timer enabled; recurring shadow disabled; first-proof timer enabled | The audited 16-file campaign transaction is now one clean shared source revision. Generated freezes, manifests, outcomes, task brain, and terminal logs remain outside Git but are checksum-identical across both backup roots. q source was fast-forwarded only while news, quote, and shadow services were inactive; no unit file, timer authority, selected strategy, profitability clock, broker session, or order changed. |
| E-158 | 2026-07-26 19:18 AEST | 2 | Frozen directional meta-admission falsification | `backtests/out/xsp/xsp_directional_meta_admission_20260726.json`; authenticated 2021–2024 XSP tape | artifact `276d4031…`; source `b7b90395…`; `363` training sessions; `1,214` labeled normal-engine trades; AUC `0.6016`; `0/7` thresholds passed | The production detector remained frozen after a half-year audit showed precision `0.575..0.606` and recall `0.727..0.802`. One preregistered L2-linear admission owner used only timestamp-available direction/time/ordinal, readiness, slope/velocity/efficiency/TR, ATR, retrace, coherence, and conviction evidence from 2021–2022 normal-lifecycle trades. At threshold `0.45`, both sides and aggregate P&L were positive (`+23.22/234`, PF `1.269`) but clustered daily LCB95 remained negative (`-0.0191`). Higher thresholds manufactured attractive PF by collapsing cadence from `102` to `2` trades and increasing concentration. No threshold jointly passed reliability, `0.40/session` cadence, both-side economics, drawdown, and concentration, so calendar 2023 and the 2024 holdout remained sealed. No coefficients, detector policy, production source, candidate, selection, clock, broker, or order changed. |
| E-159 | 2026-07-26 19:32 AEST | 1/4 | Full-process Monday tape restart regression | `tests/test_option_quote_capture.py`; unchanged canonical recorder/calendar/ledger owners | test `eee07541…`; recorder `4a9022ce…`; market `98e43d84…`; ledger `95e22660…`; focused `86 passed`; full `832 passed, 4 deselected`; Ruff/diff clean | The previously separate tail-repair, cadence, retained-universe, trading-date, and read-only proofs are now exercised through two complete recorder processes. Process one writes the Monday exchange-date tape; a torn JSON tail is injected; process two repairs the same file, restores the qualified option universe, performs zero redundant contract qualification, appends one valid row, and connects read-only. This closes the deterministic restart-proof gap without changing recorder, calendar, quote, broker, strategy, timer, selection, clock, or order behavior. The actual Sunday scheduled start/restart receipt remains required. |
| E-160 | 2026-07-26 19:40 AEST | 1/4 | Reboot-durable session-recorder trigger | Mac/GitHub/q `48d53cf755582083387780224a1ac4c934541235`; installed q `tradebot-xsp-quotes.timer` | timer `a57969bc…`; test `827efbd9…`; focused `86 passed`; full `832 passed, 4 deselected`; systemd verify and byte comparison clean; `Persistent=yes`; next `2026-07-27 10:15 AEST`; quote/shadow/news services inactive | The recorder already recovered correctly after process failure but its non-persistent daily trigger could disappear if q rebooted across Sunday `20:15 ET`. Only the once-per-session quote timer is now persistent. A catch-up inside the exchange capture window resumes the same trading-date tape; a catch-up after `17:00 ET` exits through the existing broker-silent closed-window guard. The high-frequency shadow timer remains non-persistent. Loading the unit did not start the tunnel, recorder, shadow, news, profitability clock, broker, or order path. The actual Sunday start/restart receipt remains required. |
| E-161 | 2026-07-26 20:02 AEST | 2 | Dense multiscale interaction falsification | `backtests/xsp/xsp_directional_interaction_campaign.py`; `backtests/out/xsp/xsp_directional_interaction_admission_20260726.json` | campaign `9b312514…`; artifact `3228f984…`; `45` cells in `19.3s`; `12` discovery passes; audit opened; holdout sealed | The exact E-158 generator was recovered from the durable session, including its feature order, scaler, optimizer, regularization, thresholds, and normal-engine projection. A discovery census first proved that every single or two-label slow/fast/ATR bucket dense enough for `234` trades still lost. The preregistered nonlinear family then retained complete coefficients/scalers/decisions/trades for positive-trade, trail-capture, and net-points objectives across three regularizations and five dense admission counts. Its frozen leader (`net_points`, L2 `40`, target `450`) earned `+54.57` over `413` trades (`1.138/session`), PF `1.390`, daily LCB95 `+0.0390`, drawdown `13.70`, and positive P&L on both directions. Unchanged in calendar 2023 it lost `-20.39/189`, PF `0.660`, daily LCB95 `-0.1492`, with both directions negative. The 2024 holdout remained sealed. This proves a strong static discovery pocket but rejects its permanence; no detector, lifecycle, candidate, selection, clock, broker, or order changed. |
| E-162 | 2026-07-26 20:11 AEST | 2 | Causal monthly walk-forward admission rejection | shared interaction campaign; `backtests/out/xsp/xsp_directional_walk_forward_admission_20260726.json`; authenticated eight-shard five-year tape | campaign-at-run `84fc3803…`; artifact `fed94470…`; `97,530` bars / `1,255` sessions / `5,177` raw events / `4,125` matured shadow outcomes; `36` identities in `24.1s`; `0` passes | Every calendar-month model used only raw-turn outcomes exited before that month and derived its threshold from the same prior-score window. The frozen grid covered `126/252/expanding` sessions, L2 `10/40/100`, and admission rates `0.18/0.25/0.35/0.45`. None passed calendar 2023. The best daily-LCB identity admitted `124` events but yielded only `118` trades (`0.472/session`) and lost `-7.99`, PF `0.776`, with both sides negative; density-qualified variants also lost. No 2024 challenge opened. This rejects monthly coefficient adaptation of the turn-entry mechanism, not the shared sensor or a distinct expansion source. No production source, selector, clock, broker, or order changed. |
| E-163 | 2026-07-26 20:29 AEST | 2 | Directional-expansion source rejection | shared interaction campaign; `backtests/out/xsp/xsp_directional_expansion_source_20260726.json`; authenticated XSP five-minute tape | campaign `9cb602c2…`; artifact `b69ca78f…`; `648` identities / `637` unique entry tapes in `138.5s`; `588` density-qualified; `7` net-positive; `0` passes | The preregistered source fired on new or flipped coherent `5/15/30m` slope/TR and aligned-velocity expansion, optionally requiring positive ATR velocity/acceleration and already-warmed `60/120m` alignment. Every identity retained the same next-open execution, E-051 lifecycle, one-unit economics, `$0.10` friction, both directions, EOD flattening, and `234/363` density floor. Seven dense identities were net-positive, but none achieved positive daily LCB95 or PF `>=1.10`. The least-bad reliability rank made `+1.59/264`, PF `1.017`, daily LCB95 `-0.0718`, with both sides slightly positive; the best net made `+4.06/357`, PF `1.033`, but lost on up entries. Calendar 2023 and 2024 remained sealed. Research execution deduplicated byte-equivalent entry tapes only; identity/economics stayed unchanged. No production source, selector, candidate, clock, broker, or order changed. |
| E-164 | 2026-07-26 20:38 AEST | 1/2 | Natural material causal-news publication before Monday evidence | q `tradebot-news.service`; `~/.local/state/tradebot/news/latest.json` | service `20:31:47..20:38:20 AEST`, exit `0`; publication `2f3f94d0…`; file `ebb43816…`; `8` articles / `12` active events / `2` additions | The completion-scheduled service ran naturally—no manual trigger—and atomically published a fresh `tradebot.news-signal.v3` snapshot at `2026-07-26T10:31:47Z`. Joint causal reduction moved XSP to `-1 / 79 / confidence .95 / 24h / strengthening` and retained MCL at `+1 / 84 / .95 / 24h / unchanged`; exact drivers and event snapshot remain content-addressed. The next cycle is scheduled for `00:38:20 AEST`. This is timestamp-valid observation-only input for the preregistered opposing-signal defensive comparison, not a selector, trade claim, backfill, profitability event, broker call, or order authority. |
| E-165 | 2026-07-26 22:32 AEST | 2/3 | Source-consistent directional lifecycle and complete landing trace | canonical `tradebot/spot/{evaluator_common,evaluator_policy,entry_control}.py`; `tradebot/engines/directional_impulse.py`; backtest/UI lifecycle consumers; `tests/test_directional_impulse.py` | focused `146 passed`; full `836 passed, 4 deselected`; Ruff/diff/compile clean; architecture ratchet clean; `5,177` five-year source events; `1,019/1,019` leader trades carry exact signal-bar and entry-control evidence | The prior expansion campaign rewrote entry direction while flip exits still read a transient raw turn and its `0.75 ATR` stop preempted `150/264` paths. The shared lifecycle now exposes persistent directional-impulse trend ownership separately from one-bar admission proposals; backtest and live UI consume the same source direction for inverse-signal exits. Same-timeframe EMA confirmation is genuinely advanced when selected, and entry-control traces retain every active pass/block. The corrected campaign uses no initial stop, trail, profit target, or fizzle exit: only source-consistent controlled flips after the configured hold or EOD flattening, next-open fills, one fixed unit, and `$0.10` round-trip friction. Exact traces show `986` five-minute entries, `33` controlled-flip ten-minute entries, median hold `13` bars, EOD `+376.51`, and flip exits `-255.92`; removing flips still destroys reverse-position ownership, so local exit loss is not evidence to disable them. No selector, clock, broker call, or order authority changed. |
| E-166 | 2026-07-26 22:45 AEST | 2 | Tight opening-edge plateau and frozen research champion | canonical `DirectionalImpulseAdmissionPolicy`; `backtests/xsp/xsp_directional_interaction_campaign.py`; `backtests/out/xsp/xsp_directional_lifecycle_anatomy_20260726.json` | artifact `d3c5801f…`; recent freeze `fac8d314…`; `192 → 55 → 24` tested; leader `opening_edge/off/hold12`; recent `+10.98/17`; one-year `+131.74/204`; five-year `+120.59/1,019` | Tight causal sweeps—not a broad new model—selected the stable center: `09:30..11:15 ET`, ATR velocity `(0,.055)`, down retrace `>=1.25`, plus up-only `11:20..11:25` continuation when retrace is `1.25..1.70` and coherence `>=.75`; inverse-source hold is `12` bars. Neighboring down-retrace `1.20..1.30` and ATR ceilings `.050..060` remain net-positive; the center preserves lower drawdown than the absolute endpoint winner. The same typed policy run through the normal evaluator exactly reproduces the research result. Five-year PF is `1.1705`, cadence `204.61/year`, up/down P&L `+56.48/+64.11`, and concentration `0.0987`, but daily LCB95 is `-0.0202`, drawdown `59.95`, and 2021/2023 lose. It is frozen for prospective shadow/counterfactual evaluation only: `NO_TRADE`, stopped profitability clock, and `order_authority=none` remain authoritative. |
| E-167 | 2026-07-26 23:21 AEST | 2/3 | First XSP research crown and exact prospective prefix runtime | canonical `tradebot/research/xsp_candidate.py`; `tradebot/research/xsp_shadow.py`; shared backtest engine/tape; `backtests/xsp/leaderboard.md`; rerun artifact/log | full realized-config `77e285f3…`; candidate source `b9997bf7…`; artifact `753889ac…`; log `75dfa3d6…`; recent replay `+12.68 gross -1.70 cost = +10.98 net`, `17` trades, `7.83` drawdown, zero breaches; focused `81 passed`; full `837 passed, 4 deselected` | One typed candidate owner now supplies the exact campaign and prospective configuration, eliminating campaign-local admission/lifecycle literals. Its content address binds the full realized strategy, filters, synthetic inputs, bar/session contract, and capital reference; the campaign artifact separately hashes that owner so default drift cannot reuse the crown identity. A causal normal-engine prefix reproduces the E-166 recent ledger exactly and records restart-stable `xsp.candidate-equity.v1` counterfactual economics beside—never instead of—the `NO_TRADE` observer. The shared engine now distinguishes a complete final session from an incomplete live prefix: completed EOD liquidation cannot reopen on the same terminal bar, while a live prefix retains its open mark. The rerun preserved exact crown economics through all three windows. The first durable XSP leaderboard crowns Opening Edge v1 as research leader and freezes its edge lineage, active/inactive gates, neighborhood, failure boundary, promotion ladder, and capability frontier. The profitability clock remains stopped and both candidate and checkpoint retain `order_authority=none`. |
| E-168 | 2026-07-26 23:41 AEST | 2 | Controlled-reverse ownership ablation | unchanged central candidate/engine; `spot_controlled_flip=true` crown versus `false` exit-to-flat challenger | six unchanged-window runs in `32.9s`; five-year handoff anatomy in `23.7s`; crown `+120.59/1,019`, challenger `+104.81/986`; cadence `204.61` versus `197.99/year`; drawdown `59.95` versus `58.46` | The central lifecycle already separates protective inverse-source exit from opposite-side admission, so the challenger required no code or campaign-local policy: both variants kept `exit_on_signal_flip=true`, while only the challenger stopped carrying a simultaneously admitted opposite proposal through the deferred exit. It was identical over the recent 19 sessions, then lost `14.63` points and eight trades over one year and `15.78` points plus 33 trades over five years, crossing below the required `>200/year` cadence for only `1.49` points of drawdown relief. Exact ledger differencing found zero altered common trades and exactly 33 crown-only admitted handoffs worth `+15.78`: bearish `+22.44/7`, bullish `-6.66/26`. The global controlled-reverse mechanism is therefore retained; only the weak bullish handoff subset may be challenged later with independent causal evidence. Crown identity, source, thresholds, authority, profitability clock, broker state, and orders remained unchanged. |
| E-169 | 2026-07-26 23:54 AEST | 2/3 | Optional-stop kernel repair and late-profit-lock rejection | canonical `tradebot/spot/lifecycle.py`; normal backtest engine; frozen Opening Edge v1; preregistered `2/3/4 × 1/1.5/2 ATR` grid | `30` unchanged-window runs in `67.6s`; focused/architecture `69 passed`; full `839 passed, 4 deselected`; owner `999` lines; crown control exact `+10.98/+131.74/+120.59` | The excursion policy advertised independent trail/fizzle/max-hold behavior but enabled its state only when `initial_stop_atr>0`, forcing every prior trail experiment to install an initial stop. The central state now represents a genuinely absent stop as `None`, activates only configured behavior, and retains the completed-bar ratchet so a trail cannot reprice its source bar. Existing configurations are unchanged. The nine frozen trail-only challengers used no initial stop, breakeven, target, fizzle, max hold, admission/EMA change, cost relief, or authority. Every variant improved the recent 19 sessions (`+12.41..+15.00`, `19..20` trades) yet weakened the one-year crown (best `+105.45` versus `+131.74`) and five-year crown (best `+65.99` versus `+120.59`); lowest challenger drawdown was `60.30` versus crown `59.95`. The family is rejected as recency-biased. No challenger identity, live excursion state, selector, clock, broker call, or order was added. |
| E-170 | 2026-07-27 01:02 AEST | 1/2/3 | Opening-edge degradation anatomy and unconditional path telemetry | `backtests/out/xsp/xsp_opening_edge_degradation_signatures_20260727.json`; normal engine; `backtests/xsp/leaderboard.md` | artifact `2ebd3644…`; engine `23d3ce07…`; leaderboard `bfbd3cae…` | All eight authenticated shards, `97,530` bars, sliced/prepared/standalone paths, costs, and DST reproduce the exact `+120.59/1,019` crown; no mechanics defect explains its `623/1,004` positive rolling years or `692`-session underwater interval. Every weak cluster shares flip churn without enough EOD financing, but 2023/early 2025 mostly starve while Apr–Jul 2025 gives back large excursions under elevated volatility. Zero of `42` causal pre-entry features shifts `0.1` pooled IQR consistently across all three; a strictly prior `20/40/63` health consensus marks `80.2%/82.5%/84.1%` of them but also profitable controls. Real state-machine low-energy vetoes improve net/drawdown only by falling below `200` trades/year; the sole cadence-preserving point changes one already-observed signal and is rejected. The engine now records bars held and MFE/MAE for every trade even when stop/trail policy is disabled, reusing the existing `SpotTrade` fields without altering exits; the crown remains exact. Selective contemporaneous headlines make causal news plausible but not a backfill: weak-cluster P&L loses on both sides and includes narrative reversals, so the frozen defensive observer remains unchanged and prospective pressure-change is attribution only. Crash/rebound remains a distinct peer state. `NO_TRADE`, stopped clock, and `order_authority=none` remain authoritative. |
| E-171 | 2026-07-27 02:50 AEST | 1/2/3 | Opening Edge v1 close-clock parity revision | clean pre-repair q artifact/source; current canonical cache normalizer, directional policy, candidate owner, normal engine, and lifecycle campaign | old/new artifact `753889ac…` / `b86deb3b…`; run log `ea69e627…`; config `bbb0a391…`; original config `77e285f3…`; semantic physical-ledger `4099fe9f…` | The cached IBKR five-minute rows were bar-start timestamps while the evaluator interpreted strategy clocks as causal closes. One canonical normalizer now translates every row once; turn/admission labels move exactly five minutes to `09:35..11:50`, core through `11:20`, and late-up through `11:30`. A read-only replay from q's clean pre-repair source and the repaired local source produced the same ordered `1,019` physical trades after that deterministic clock translation: identical prices, sides, exits, `+120.59` net, PF `1.170486`, and `59.95` drawdown. The bounded parity campaign completed `192 → 62 → 29` in `109.5s` and retained `opening_edge/off/hold12` as the five-year leader. The strategy remains `xsp.opening-edge-directional.v1`; `close-time-parity-r1` is a runtime revision, not Opening Edge v2 or a new crown. `NO_TRADE`, stopped clock, and zero order authority remain unchanged. |
| E-172 | 2026-07-27 02:50 AEST | 1/2/3 | Retired false TICK direction authority and causal breadth challenger | canonical `tradebot/engines/market.py`; centralized config/context/backtest/live/sweep owners; authenticated `TICK-NASD` five-minute cache | market `67361a0a…`; TICK shards `f9b0a9fc…` / `3c29ca11…`; zero enabled `raschke` configs; `108` historical `off` snapshots | The legacy path converted daily TICK-band width into direction (`wide → up`, `narrow → down`), although dispersion has no sign. No current or archived strategy enabled it, so its executable config fields, cache requirement, engine clamp, live placeholder, CLI/sweep axis, presets, and research runner were removed together; old journals remain readable. One shared observation now exposes causal same-session current TICK, 3/6-bar means, cumulative breadth, freshness, provenance, direction-relative alignment, and improving/deteriorating transition without thresholds or order authority. The exact close-aligned breadth-150 challenger earned `+127.64/1,004`, PF `1.1819`, DD `52.67` over five years versus crown `+120.59/1,019`, but worsened recent (`+8.99/16` versus `+10.98/17`) and latest-year (`+128.34/202` versus `+131.74/204`) evidence, still lost 2021/2023, and uses NASDAQ TICK as an XSP proxy discovered on the same tape. It remains observation-only and uncoronated; Opening Edge v1 is unchanged. |
| E-173 | 2026-07-27 AEST | management | Priority-ordered active brain and lossless root archive | `q_XSP_live_research_kata.md`; `q_XSP_live_research_kata-archive.md` | pre-distillation active snapshot `816bcb87…`; active `1,076→355` lines; archive markers `A01…A06`; `173/173` prior evidence IDs and `144/144` prior decision IDs preserved uniquely | The root brain now answers only current mission, authority, crown, risks, Monday sequence, active hypotheses/tasks, canonical ownership, and critical history. Original narrative, exact pre-distillation frontier, completed task tree, every full receipt/decision, and the prior conclusion remain behind stable reciprocal markers in the same-named cold archive. Routine verification narration is explicitly excluded unless it changes a capability or conclusion. No strategy, runtime, evidence outcome, selected state, clock, broker, service, or order authority changed. |
| E-174 | 2026-07-27 03:12 AEST | 1/2/3 | Crown parity/breadth runtime publication and cross-device evidence convergence | Mac/GitHub/q `main`; repaired and raw-clock lifecycle artifacts | source `d20ed6e8…`; active brain `d9d30d2e…`; archive `6196588b…`; leaderboard `9baa447a…`; raw/repaired artifacts `753889ac…` / `b86deb3b…` | The close-time crown parity repair, false TICK-direction retirement, signed breadth observation, unconditional path telemetry, leaderboard lineage, and active/archive management split were published as one clean revision and fast-forwarded to q while quote, shadow, and news services were inactive. Mac and q source/management files are byte-identical. The old raw-clock artifact was preserved explicitly on both machines before the repaired close-clock artifact became canonical, so representation forensics remain reproducible. Timers, selected state, profitability clock, Gateway activity, broker state, and order authority were not changed. |
| E-175 | 2026-07-27 AEST | 1/2/3 | Signed breadth prospective boundary and exact crown-entry attribution | runtime `f146c946…`; `tradebot/contract_identity.py`; `tradebot/client.py`; `tradebot/engines/market.py`; `tradebot/research/{xsp_context.py,xsp_shadow.py,xsp_candidate.py}`; authenticated TICK-NASD cache | context `c23116c5…`; shadow `d2945284…`; market `ec56c285…`; client `6671786a…`; candidate `3997f4a8…`; cache shards `f9b0a9fc…` / `3c29ca11…`; `97,530` rows including `47,114` negative and `186` zero closes | The Monday observer previously promised breadth but recorded none, and both canonical client history adapters silently discarded every non-positive close even though TICK is signed. Contract identity now marks only explicit TICK indices as signed-valued; ordinary price series still reject non-positive values. One external-context owner centralizes option parity, pre-open, causal news, and signed breadth without wrappers or policy copies; the shadow owner shrank from `990` to `849` lines. Every forward turn and checkpoint now carries exact proxy identity, causal close, 3/6-bar means, cumulative breadth, readiness/staleness, direction-relative transition, and zero order authority. The Opening Edge candidate exposes its exact signal-decision timestamp so the latest entry can be paired with breadth at decision—not checkpoint time. Missing, unqualified, underwarmed, stale, or failed breadth leaves crown evaluation unchanged. Exact `TICK-NYSE` exists at IBKR but current NYSE index-data permission rejects its history, so `TICK-NASD` remains explicitly named as a proxy; its two authenticated shards are byte-identical on Mac/q. `NO_TRADE`, the crown, historical economics, stopped profitability clock, broker state, and capital authority are unchanged. |
| E-176 | 2026-07-27 AEST | management/runtime | Signed-breadth scope correction | canonical market/client/context owners; XSP shadow and active task brain | exact cleanup revision recorded by repository history | The breadth challenger never earned a crown or prospective authority, yet its automatic TICK broker request and propagation through every shadow forecast, checkpoint, receipt, and active task item made it operationally prominent. That surface was removed before Monday. The retained core is only explicit signed-index history semantics plus one pure timestamp-causal research reducer; it performs no automatic fetch, direction vote, veto, selector action, shadow persistence, broker submission, or order action. Opening Edge v1, its economics, option/news context, candidate equity, `NO_TRADE`, and the stopped profitability clock are unchanged. |
| E-177 | 2026-07-27 AEST | 1/2/3 | Corrected-lifecycle tournament closure and Monday runtime preflight | normal XSP engine; repaired source/permission/flip ownership; Opening Edge v1; q installed runtime | broad `402`-cell source tournament; bounded EMA-veto, supplement, profit-only-flip, and `28`-cell handoff challenges; q `83cfc58`; crown fingerprint `bbb0a391…`; producer `10:15 AEST`; observer `23:37 AEST` | The broad source tournament and four focused lifecycle challenges used one fixed XSP unit, next-open fills, unchanged `$0.10` friction, exact gate traces, and recent → one-year → five-year ordering. EMA `5/10` produced `+17.93/72` recent trades but its best impulse veto still lost `-47.85/958` over one year. An impulse-owned EMA supplement improved recent economics to `+14.38/19`, but its best later path fell to `+124.08/215` over one year and `+97.38/1,110` over five years. Profit-only flips and asymmetric handoffs likewise improved only the recent tape; only the unchanged crown survived both later crown gates. No experimental policy entered the engine. Mac/GitHub/q remain synchronized at `83cfc58`; q imports the exact crown fingerprint, installed units match source, `order_authority=none`, and a transient systemd dependency proved the on-demand localhost tunnel reaches the live Mac Gateway before stopping correctly under `StopWhenUnneeded=yes`. |

---

<!-- XSP-ARCHIVE:A04:END -->

---

<a id="xsp-archive-a05-decision-journal"></a>
<!-- XSP-ARCHIVE:A05:BEGIN -->
# Archive A05 — Full decision journal

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
| D-036 | Calibration becomes effective after its observed date | A same-day date-only record can leak later broker evidence into earlier replay; explicit source bounds and next-date eligibility preserve causality while live orders continue to use broker quotes | Authentic timestamped option replay replaces synthetic calibration |
| D-037 | Admit five-year XSP only as underlying RTH evidence | The hydrated tape is complete and comparable, but IBKR still supplies no historical expired-option chain/NBBO/Greek tape and XSP index volume remains absent | A provenance-complete option provider or accumulated forward tape supplies the missing evidence |
| D-038 | Barrier evidence screens credit geometry but cannot prove option expectancy | Historical XSP spot can authenticate touches and settlement breaches, but not old NBBO, IV, fill probability, commissions, or executable package credit | Fresh strict-admission package quotes and forward replay clear the empirical required-credit hurdle |
| D-039 | Reject unconditional multi-session XSP credit carry at current evidence | Even the best one-percent next-session cells require `0.3026..0.3267` executable credit after a deliberately conservative full-loss model, with substantial annual breach dispersion; longer holds require more | A fresh strict-admission package quote clears the registered hurdle and forward replay then passes calibration, execution, safety, and economic gates |
| D-040 | Do not turn simple direction/gap/quiet facts into a premium-selling selector | No preregistered discovery cell materially lowered matched required credit; moving the `0.05` or sample gates for a near miss would be post-result tuning | A new causal feature family is preregistered from independent rationale or authentic forward option evidence reveals a specific execution-compensation mechanism |
| D-041 | XSP mastery is 24x5 and session-conditioned, not RTH-only | Cboe supports XSP in GTH, RTH, and Curb; the opening-volatility study is only one edge hypothesis, while liquidity, spread, fill, and data provenance differ materially by session | Never collapse sessions; widen a strategy only after same-session replay, preview, and drift evidence passes |
| D-042 | Close-only XSP cannot anchor an executable package | During GTH the option surface can quote while the cash-index top remains unavailable or stale; using yesterday's close would select hindsight-wrong geometry | Use fresh index midpoint/last or a timestamped robust option-model/futures reference, and retain exact provenance |
| D-043 | Greeks are strategy inputs and diagnostics, not universal execution gates | A fixed-width defined-risk package is economically admitted by exact identity, executable quote, tick, fees, max loss, capacity, and broker state; requiring unused vega or other missing model fields discards valid evidence without improving safety | A strategy that selects or sizes from a Greek must declare that field as required and fail closed when it is absent |
| D-044 | This account has no proven streaming-live XSP quote entitlement | The complete strict opening probe returned only IBKR `354/10168` subscription failures and zero qualified options | Preserve delayed tapes for mechanics and research, but require a later streaming entitlement receipt or broker preview before any executable XSP promotion |
| D-045 | RTH-only long-direction frequency is not an edge | Two independently structured searches found sparse profitable mechanisms, while every `>=4/3`-trade/session v2 variant lost after friction; simple 15/30/60-minute direction facts also left upside and downside opportunity nearly symmetric | Hydrate authenticated pre-open proxy context and test its incremental value; do not tune the same RTH feature family until it wins by new evidence |
| D-046 | Full24 completeness begins at the broker-supported overnight head | SPY SMART history predates IBKR's OVERNIGHT venue by years; demanding overnight sessions before `reqHeadTimestamp` caused impossible day repairs, while two post-head gaps remain explicitly broker-unresolved | Preserve SMART data before the head, require full24 after it, and never fabricate or silently bless a post-head gap |
| D-047 | Static pre-open filters do not rescue repeated opening re-entry | Exact SPY context reduced exposure and some losses, but no cell produced a positive daily lower bound and cadence fell below the HF contract | Change the lifecycle/mechanism—test one deliberate opening ride with causal admission and robust trailing/fizzle exits—rather than tuning these filters |
| D-048 | Opening volatility must not be assumed upward | The deliberately long-only ride improved on fixed-time exposure but remained negative across discovery; visual upside days omit equally important failed and downside openings | Test one symmetric, explicitly attributed long/short directional owner before abandoning the opening-volatility family; do not loosen friction or gates |
| D-049 | Retire bar-only opening-direction mining | Long-only, symmetric, static pre-open-context, and repeated-entry searches all failed frozen cadence/stability gates; the sparse positive cells changed sides or disappeared as cadence rose | Keep validation sealed and stop retuning this OHLC family. Resume directional discovery only with independently justified information—preferably authentic forward microstructure, executable option compensation, or a materially different horizon |
| D-050 | Wall-clock context uses timestamp anchors, not bar counts | SPY's scheduled `03:50..04:00 ET` break means `03:30..09:30` contains `71`, not `73`, observations; interpreting `72 bars` as six hours silently reaches ten minutes too far back | Only if the venue supplies a formally continuous, gap-free series |
| D-051 | XSP-native evidence owns XSP direction; SPY is diagnostics only | XSP-only matched discovery and beat fused/SPY-only recent precision and F1, so proxy authority adds complexity without measured value | An independently preregistered proxy study proves stable incremental signal |
| D-052 | Three observed horizons admit a turn; five define full readiness | Recent precision was `0.667` with three horizons versus `0.662` with two, while four horizons collapsed recall/F1; exact anchors and the upstream data-gap veto prevent silent underwarming | A new frozen calibration proves a better neighboring contract |
| D-053 | Material local extrema—not absolute daily extrema—are the sensor acceptance target | The causal production sensor reached `0.710` recent F1 on material turns, while only `16/34` absolute extrema matched and `18/34` were censored at the measurement boundaries | A wider, causal observation contract removes boundary censoring without hindsight |
| D-054 | Directional turns remain observation-only until lifecycle economics pass | Feature detection is not profitability; false turns, friction, trailing/fizzle exits, cadence, and adverse path must be measured in the normal backtest owner | The preregistered lifecycle study passes discovery, neighbors, and sealed validation |
| D-055 | Reject indiscriminate directional-turn entry; preserve the sensor as evidence | All `1,296` frozen lifecycle variants lost despite adequate cadence and balanced sides. The best trail path captured real excursions, but repeated false turns and initial stops overwhelmed it; exit retuning cannot create admission edge | A preregistered admission-quality family preserves cadence and both directions while passing discovery, neighboring stability, and sealed validation |
| D-056 | Do not promote the sparse late-upside admission pocket | Causal conviction/retrace/ATR filters created positive cells, but all had negative daily lower bounds or inadequate cadence; every downside slice lost and all cadence-qualified cells remained negative | An earlier, preregistered short-to-medium-to-long reversal cascade passes symmetric tradable-excursion and economic gates on chronological evidence |
| D-057 | Retire bar-only directional reversal mining from the current XSP frontier | Completed-turn lifecycle, causal admission, and earlier reversal-cascade studies all failed frozen economic gates. In the cascade, `71/144` cells met HF cadence on both partitions yet every cell lost, so more adjacent threshold tuning is outcome mining | Materially new causal evidence—preferably authenticated forward microstructure/execution—or a genuinely different horizon/mechanism is preregistered |
| D-058 | Preserve `NO_TRADE` for the first frozen RTH package screen | Neither evidence nor economics admitted a package: strict streaming-live selected-leg quotes were absent, and every delayed diagnostic natural credit missed its preregistered barrier | A later independently captured session produces persistent strict quotes, clears its unchanged registered hurdle, and remains within canonical risk/cost gates |
| D-059 | Option-model consensus may anchor missing GTH cash but cannot own direction | All eight GTH snapshots had a coherent model underlier and bridged plausibly into the first captured RTH print, yet one session is not validation and RTH accuracy/direction agreement was inadequate | Multiple independent forward sessions prove timestamp-correct incremental admission value over the unchanged XSP cash-only observer |
| D-060 | Historical calibration must preserve `NO_TRADE` while failed candidates remain counterfactual | No candidate passed the preregistered economic gates. Treating a detector's overlapping-horizon diagnostic sum as selected P&L would fabricate a strategy and double-count outcomes; the new ledger keeps selected, counterfactual, package, leg, and account economics separate | A preregistered candidate passes causal discovery, stability, cadence, and sealed validation, then survives authentic same-tape shadow evidence |
| D-061 | Forward shadow consumes the canonical close-aligned tape and owns no orders | IBKR timestamps intraday bars at their open, while the shared UI/backtest evaluator consumes bars normalized to their causal close. Feeding raw broker timestamps directly would freeze five minutes early; letting a shadow submit would confuse observation with promotion | Only replace this boundary if one stronger canonical market-series contract preserves identical close time, freshness, provenance, restart, and no-order guarantees |
| D-062 | Option-parity participation begins as forward observation, not a selector | One complete session proves deterministic paired-call/put mechanics but every selected row is delayed and the cash outcome is already known. Scoring or thresholding July 24 would turn availability work into hindsight mining | Multiple independent forward sessions compare timestamp-identical `TA-only` and `TA+option-observe` decisions with stable cadence, both directions, economic lower bounds, and execution-grade provenance |
| D-063 | Partition forward XSP evidence by exchange trading date, not UTC or wall-calendar date | Sunday/weekday evening GTH belongs to the following XSP session; UTC or ET-calendar filenames split one session, misalign shadow lookup, and can alter DTE selection across restart | Only replace this if a broker-native session identifier subsumes tape path, expiry basis, restart restoration, and shadow lookup without losing causal provenance |
| D-064 | Fixed auxiliary IBKR clients live above the rotating runtime pool | Historical fetch, calibration, and recorder offsets were previously based on the pool start, placing `550/580/590` inside `500..899`; a continuous fixed-ID recorder could lose a race to UI/shadow clients and retry the same occupied ID | A single cross-process client-ID lease owner replaces both deterministic auxiliary IDs and the resilient live pool |
| D-065 | Fundamental pressure is immutable forecast context before it can become a gate | Reconstructing the latest news state after an outcome creates lookahead; including it in strategy identity would also create duplicate forecasts when a later publication arrives | Paired forward evidence proves a preregistered defensive threshold improves economics or drawdown, at which point a new versioned gate contract must be frozen before activation |
| D-066 | Fundamental value is scored first as one fixed 60-minute defensive-veto observer | Multiple horizons double-count the same turn, while letting news open, reverse, or amplify exposure would confuse defensive context with an unproved alpha selector | Several independent forward sessions show stable avoided-loss benefit over foregone gains on the frozen threshold; only then may a separate versioned gate seek promotion |
| D-067 | One news publication is a recoverable multi-file generation | Per-file atomic rename prevents torn files but not mixed events/memory/history/latest/state generations after a mid-handoff kill | A stronger storage owner provides an equally compact transaction across the same externally stable paths |
| D-068 | First-seen forward decisions outrank later tape-prefix enrichment | A benign historical cache repair changes provenance but must not rewrite or duplicate the forecast that was actually available at decision time | A versioned research run intentionally uses a separate ledger and explicitly permits alternate provenance for the same timestamp |
| D-069 | Freeze raw parity movement before scoring it | The preregistered coherence hypothesis needs two causal observations; reconstructing a prior anchor after the outcome would weaken provenance, while choosing a movement threshold from the observed July 24 path would be outcome mining | Independent forward sessions accumulate enough timestamp-correct pairs to preregister and test a threshold without opening old outcomes |
| D-070 | Discovery and inference have separate runtime ceilings | A large inference allowance must not silently become an equally large network-connect timeout; otherwise the outer service deadline cannot reserve publication time | Measured discovery latency justifies a different bounded cap while preserving the same total wall |
| D-071 | Preserve the frozen defensive-news observer through its first negative mechanics receipt | Retuning after seeing that July 24 bearish context would have vetoed profitable upside would convert a defensive hypothesis into outcome mining; one post-registered day cannot prove or disprove forward value | Preregistered prospective sessions accumulate enough paired veto opportunities to compare avoided losses, foregone gains, drawdown, and cadence |
| D-072 | Event clocks are deterministic reducer state, not model judgment | Stable first-seen time and whether the material event view changed are exactly derivable from prior ID, current payload, and run time; asking the model to reproduce that bookkeeping turned a valid researched payload into a failed cycle | A stronger event-store transaction subsumes identity, diff, and clock derivation without weakening verification provenance |
| D-073 | Close historical VIX turn-confirmation tuning | Exact dynamic VIX pressure reduced losses but did not produce positive expectancy, chronological stability, both-direction profitability, or HF cadence; changing thresholds after all `13` frozen cells failed would mine another bar-derived context | New timestamp-correct forward XSP microstructure or a genuinely different causal mechanism passes a preregistered paired observer |
| D-074 | Option parity remains a classify-only forward observer | One already-seen aligned pair cannot establish incremental value, and inventing a movement threshold or veto from it would leak the known cash outcome | At least 30 timestamp-correct usable pairs across five prospective sessions support a preregistered threshold or defensive action with chronological stability |
| D-075 | Auxiliary observers own isolated client triplets above the live pool | Reusing the dashboard's main/proxy IDs or persisted allocator state lets a harmless shadow collide with live connectivity and makes the launch topology depend on process timing | A cross-process lease owner subsumes both deterministic auxiliary ranges and the live rotating pool without weakening restart isolation |
| D-076 | Profitability clocks require selected economics plus continuous coverage | An event-driven observer ledger cannot distinguish a quiet evaluated interval from a crashed or missing evaluator, and overlapping counterfactual horizons are not a tradable equity curve | One admissible selected strategy emits complete session-conditioned coverage and reconciled net-P&L/drawdown receipts under the frozen milestone contract |
| D-077 | Coverage state is orthogonal to signal state | A quiet invocation, stale data, an unsupported session, and a stopped evaluator must not collapse into the same absence-of-forecast representation | A stronger canonical runtime receipt subsumes the same five fail-closed states without weakening first-seen forecasts or restart idempotence |
| D-078 | Schedule the current observer only after completed XSP cash-RTH bars | The selected directional baseline is RTH-only, and repeatedly connecting through GTH/Curb would waste Gateway work while implying authority the strategy has not earned; host-local times also drift across DST | A proven GTH/Curb strategy gets its own versioned evidence cadence, or one exchange-calendar scheduler subsumes both without extra broker requests or catch-up bursts |
| D-079 | Pin direct shadow-runtime dependencies before provisioning q | Floating installs could silently move the scheduled evaluator away from the versions used by the tested Mac runtime; exact direct pins keep the persistent environment reproducible without maintaining a redundant transitive lockfile | A repository-wide environment owner replaces `requirements.txt` with a stronger reproducible contract shared by every supported runtime |
| D-080 | Broker read-only mode is mandatory for non-submitting observers | `order_authority=none` in a receipt describes intended code flow but cannot prevent an accidental order call; IBKR's connection-level `readonly` flag provides an independent broker-session boundary while preserving ordinary live UI order authority by default | Only a stronger broker-enforced capability boundary can replace the explicit observer flag |
| D-081 | One session-scoped recorder owns the forward option tape | Reconnecting and requalifying hundreds of one-shot recorders wastes Gateway capacity and weakens chain continuity, while an immortal weekly daemon holds transport through closed weekends; one bounded daily process preserves exact cadence and restart semantics with minimal authority | A broker-native streaming/archive owner proves stronger continuity and provenance without widening order authority or creating a second cache |
| D-082 | Profitability is a selected equity-curve property, never an observer sum | Overlapping forecast horizons, counterfactual points, recorder uptime, `NO_TRADE`, gross P&L, or a count of arbitrary checkpoints can all manufacture a false milestone; exact scheduled coverage and one reconciled cumulative run are the minimum defensible authority | A broker-native immutable equity journal proves the same identity, coverage, cost, drawdown, attribution, and safety invariants more strongly without widening order authority |
| D-083 | Close historical NASDAQ-breadth admission tuning | Exact TICK/TRIN participation consistently reduced the rejected baseline's losses but did not survive the chronological audit, positive-confidence, cadence, or both-direction gates; changing windows, magnitude floors, or sides after this frozen result would outcome-mine a third historical confirmation family | Timestamp-correct prospective XSP microstructure or another independently preregistered mechanism produces stable same-tape value without retuning the rejected bar-derived families |
| D-084 | Prospective gates count only first-seen forward broker evidence | Historical replay and already-seen mechanics are valuable deterministic diagnostics, but letting their pairs or session dates mature a forward sample gate would relabel known outcomes as prospective evidence | A stronger immutable evidence provenance proves an equivalent or stricter forward-only boundary without discarding diagnostic rows |
| D-085 | Forward pairs earn sample authority only inside completely evaluated sessions | Counting a date because one event fired cannot distinguish a healthy full-session observer from 77 missing normal-day invocations; allowing incomplete-day pairs into the numerator would let a partial or crashed run mature the gate | A stronger canonical coverage receipt proves every expected session-conditioned slot and its evidence signature without weakening pair provenance |
| D-086 | RTH coverage requires explicit RTH checkpoint identity | A GTH or Curb observation can coincide with an RTH wall-clock slot yet does not prove the RTH evaluator ran under the intended session contract | A stricter canonical session receipt supersedes the explicit identity without accepting timestamp coincidence |
| D-087 | Deploy q from one pushed combined `main`, not by merging the stale execution branch at launch time | The combined tree already retains every delivered news surface and extends its canonical owners, while q's branch is intentionally divergent and carries only one local timeout override; a launch-time merge would reintroduce conflict and mixed-revision risk without adding behavior | A different pushed revision proves the news branch contains unique behavior absent from the combined tree |
| D-088 | News availability and news age are separate clocks | `signal_as_of_utc` describes when evidence was reduced and owns horizon expiry; `snapshot_as_of_utc` describes when that exact artifact became durable and prevents a recovered or refreshed generation from entering an earlier decision | A stronger append-only publication record proves the same two-clock causality without mutable latest-state dependence |
| D-089 | Clear q's one redundant override only after three-way identity proof | Git correctly refuses the branch switch even when the dirty file equals the target; restoring it blindly could remove a live-only correction, while stashing or merging would preserve unnecessary divergence. Exact dirty-path, pushed-target, and installed-unit equality makes the one-file restore safe and leaves the loaded service unchanged | q is already clean on tracked `main`, or a different dirty path exists and requires a new census |
| D-090 | Known closed or unsupported XSP cash windows never reach the broker | Calendar/session truth is sufficient for GTH/Curb, holidays, and completed early-close tails; qualifying XSP or requesting RTH history there wastes transport and can turn expected closure into a false outage. Normal RTH remains broker-evidenced so unexpected halts still fail closed through data/freshness state | A proven GTH/Curb strategy earns its own versioned observer cadence, or a stronger exchange-session owner subsumes the same no-broker boundary |
| D-091 | Fail deployment outside the proven Python 3.12–3.13 band | The pinned q dependencies are complete and work on tested Python 3.12/3.13, while upstream `eventkit` fails before TradeBot import on 3.14. Injecting a global asyncio loop merely to bypass that failure would create hidden process-wide semantics | A maintained IBKR client stack explicitly supports Python 3.14 and passes the clean recorder/shadow runtime preflight |
| D-092 | The outer service wall must strictly exceed every nested worst-case budget | Equaling discovery + inference + publication ceilings ignores interpreter startup, process teardown, scheduling, and durable filesystem synchronization; that makes the advertised publication reserve fictional | Measured stages justify a different explicit envelope while retaining positive non-overlap margin and atomic last-good preservation |
| D-093 | Every latest-wrapper mutation recomputes its publication ID | A content address is an integrity contract over the complete wrapper, not merely the underlying signal; retaining an old ID after a no-new refresh defeats recovery validation and provenance comparison | A stronger immutable store derives the address at write time and makes stale IDs unrepresentable |
| D-094 | Validate every addressed news generation while reading legacy absence | Requiring an ID immediately would discard q's last-good pre-generation snapshot during one-revision migration; ignoring a present ID would make the new integrity contract decorative | The legacy q publication has naturally been replaced and an explicit schema migration can require the field universally |
| D-095 | Do not manually republish a response rejected by q's older reducer | The exact response proves the combined reducer works, but publishing it from a different source revision would create mixed-revision provenance and bypass the tested pending-generation path | The combined revision is deployed and a natural one-shot publishes through that exact source and unit contract |
| D-096 | Bind each indefinite XSP quote recorder to one scheduled capture window | A systemd runtime timeout is a failure and `Restart=on-failure` can recycle an otherwise normal session process after close. The recorder must own its exchange-date boundary, preserve the GTH/RTH transition gap, emit its final receipt, and exit successfully; the service wall is only a deadlock backstop | A stronger broker-native session identifier subsumes both tape date and producer lifetime without losing restart or closed-window guarantees |
| D-097 | A scheduled capture window is not proof of an executable exchange session | Cboe's normal and holiday schedules differ, and a broker can return delayed, frozen, absent, or mixed-provenance evidence inside a nominal window. Process lifecycle must not silently become session or execution authority | One broker-native session-status contract plus a versioned Cboe holiday calendar proves the same distinction end to end |
| D-098 | Test parity alignment through independent non-overlapping sequences | Overlapping observer horizons are not a tradable equity curve, while filtering the TA universe before constructing its baseline would let missing option evidence manufacture apparent value. Exact-sign alignment is the only outcome-independent candidate available before Monday | Thirty usable prospective pairs across five complete sessions pass first, then the frozen aligned sequence passes cadence, both-side, daily-confidence, leave-one-session-out, concentration, and incremental-value gates; any later magnitude rule requires a separately preregistered discovery/validation contract |
| D-099 | Validate calibration content addresses on every read, then benchmark only exact precommitted outcomes | Write-time hashes do not protect later benchmarks or selected profitability if a reader accepts stale IDs. A generic result may legitimately arrive after its minimum horizon, but a fixed-horizon directional benchmark also cannot substitute that later observation, reverse the frozen side, or duplicate its named counterfactual | A stronger immutable evidence store makes malformed rows unrepresentable and a versioned variable-duration outcome schema supplies an equally strict causal identity contract |
| D-100 | Milestones are immutable evidence prefixes, not labels over the latest balance | Later gains cannot prove that an earlier 24h/48h interval was profitable, while later failure cannot erase a milestone that was already clean. Each verdict must use only records available at the earliest instant satisfying its own wall-clock and complete-session conditions; final quest success requires all three gates, not merely the latest week | A broker-native immutable milestone journal records equally strict prefix-bound economics and coverage at source without weakening any current invariant |
| D-101 | Switch q source only inside a quiescent, one-revision unit transaction | Replacing checkout files while the enabled news timer can launch a one-shot creates mixed-revision evidence; leaving the old loaded unit in place silently defeats the combined timeout fix. The legacy dirty copy need only equal the currently loaded unit, not the intentionally newer target | A deployment manager atomically binds source revision, isolated runtime, loaded unit hashes, and timer activation with equivalent or stronger receipts |

| D-102 | Bridge GTH evidence explicitly; never weaken ordinary session causality | Relaxing `same_session` for all option context would let stale GTH, RTH, or Curb rows silently enter unrelated decisions. The pre-open hypothesis needs one separately labeled same-trading-date path with fixed boundary, horizons, expiry, and completeness while the existing RTH context remains exact-session | Prospective same-tape evidence proves a stronger canonical session-transition feature without weakening quote provenance, temporal alignment, or observation-only authority |

| D-103 | Keep pre-open parity cohorts descriptive until prospective completeness exists | Classifying the raw `2h/4h/6h` path before outcomes prevents retrospective naming, but filtering the TA universe, sharing the short-horizon parity sample subset, or ranking one cohort now would still manufacture evidence from availability | Multiple fully checkpointed prospective sessions establish a separately preregistered value contract without changing the existing parity candidate or opening sealed historical outcomes |

| D-104 | Classify historical XSP GTH underlying bars as unsupported, not missing | Local provenance contains only RTH shards; the broker returned identical RTH-only `TRADES` coverage under both session flags and no historical `MIDPOINT`. Sending this through missing-range repair would create repeated requests without an evidence source, while synthesizing it would violate the causal admission boundary | A provenance-complete provider or broker capability supplies timestamped historical XSP GTH underlying evidence beyond the identical RTH result |

| D-105 | Bound the shadow at existing broker owners; keep systemd as the final backstop | A shadow-only retry loop would duplicate client policy, while a global IBKR timeout would alter unrelated UI and order paths. The exact observer needs only the existing connection/history ceilings plus one missing bound at proxy qualification; together they leave deterministic checkpoint and teardown reserve beneath the outer wall | A shared broker-operation budget with stronger diagnostics can replace these owner-local ceilings without weakening UI behavior or expanding the scheduled observer |

| D-106 | Treat unit arguments and loaded source as separate deployment evidence | An updated systemd timeout can override one child value without installing the source logic that partitions fetch, inference, and publication budgets. Claiming the complete nested envelope from unit text alone would hide a mixed-revision runtime | The source revision, managed runtime, loaded unit, and one natural receipt are content-addressed together by an atomic deployment authority |

| D-107 | Resolve causal news from append-time history at each decision, never mutable latest alone | A four-hour publication can legitimately occur between an earlier turn and its bounded replay. Using only the new latest row turns valid prior context into a false future/missing observation, while falling back by signal outcome would create look-ahead | A transactional time-indexed signal store exposes the same publication-availability contract more directly than the locked monthly ledger |

| D-108 | Arm forward capture before proving the RTH observer; gate only the shadow timer | The producer's Sunday GTH prefix is irrecoverable, while the first eligible shadow proof cannot exist until Monday `09:37 ET`. Coupling both timers makes complete Monday evidence impossible. The quote recorder is broker-read-only and has no order path, so it can safely run before the non-submitting observer's manual checkpoint | A single transactional supervisor can activate producer, verify the first exact RTH checkpoint, and arm subsequent observer slots without human timing while preserving the same fail-closed gates |

| D-109 | Treat only `EVALUATED` as command success | A durable stale/no-data checkpoint is valuable evidence but not a successful scheduled evaluation. Returning zero for it hides missing coverage from systemd and operators even though the benchmark later rejects the date | A supervisor consumes structured status directly and provides equivalent failure visibility without relying on process exit |
| D-110 | Separate an economic window from its deterministic evidence-finalization cutoff | A canonical slot is an economic boundary, while the corresponding systemd checkpoint can be recorded seconds later within the same frozen tolerance. Evaluating both at the slot instant makes valid real evidence impossible; evaluating at query time lets later data rewrite the prefix | A broker-native append-only milestone record freezes the same exact economic window and causal finalization lag at source |
| D-111 | Keep one compact live brain and one immutable narrative archive | Reinjecting 3,000 lines made obsolete “next” markers compete with the actual frontier; deleting that prose would lose valuable research context, while splitting the evidence and decision ledgers would weaken resume authority | The root stays the sole task/evidence/decision authority; historical prose remains byte-for-byte addressable in its linked archive and is consulted only when the compact frontier points to it |
| D-112 | Freeze raw option-liquidity evolution with a threshold-free Pareto rule before prospective outcomes | A weighted score or post-outcome cutoff would make sparse quote evidence easy to overfit, while any single improving metric can hide simultaneous deterioration elsewhere. The four already-computed causal dimensions admit a compact dominance classification and preserve conflicting evidence as `mixed` | At least 30 usable pairs across five complete prospective sessions show that a separately preregistered alternative classification improves out-of-sample economics without changing evidence provenance |
| D-113 | Rank eligible candidates deterministically, but require an explicit selected-run freeze | Two passing candidates cannot leave promotion order to operator discretion, while automatically turning a benchmark result into runtime state would let evidence mutate strategy ownership and start an unapproved clock. The stricter liquidity candidate therefore outranks parity-aligned only after its own incremental gate; otherwise parity-aligned outranks `HOLD` | A transactional selection authority freezes equivalent evidence, run identity, lifecycle, and safety policy without allowing an open-position strategy switch or any order authority |
| D-114 | Begin XSP with an empty crown namespace; never port a foreign legacy champion by analogy | MNQ, SLV, and TQQQ crowns encode different instruments, data, sessions, economics, and mostly README-era promotion provenance. Reusing their loader is valid only after an XSP-native artifact exists; treating an old crown as an XSP candidate would bypass this kata's prospective option/news/calibration evidence | An XSP-native HF or LF run earns nomination, completes an explicit selected-run freeze and profitability evidence, and can be promoted through the existing content-addressed machine-crown mechanism |
| D-115 | Freeze one candidate-independent shadow risk envelope before selection | Leaving numeric limits to the future operator would permit outcome-aware risk selection; duplicating them in tests and task prose could drift. The original `25/5/2/0.50/90s` contract now travels inside every content-addressed candidate recommendation and remains synthetic evidence-only | A later quest with verified broker/account/product economics may preregister a different bounded-capital policy before its own evidence begins |
| D-116 | Make recommendation-to-selection an explicit content-addressed boundary | Treating `PROMOTE` as mutable runtime state would let a later benchmark revision silently change ownership; requiring a separate receipt binds exact evidence and run identity without granting order authority or starting a clock | A transactional append-only selection store preserves the same immutable receipt more strongly when the first real prospective candidate becomes eligible |
| D-117 | Recompute the recommendation from the supplied ledger at freeze time | A digest proves that a payload is internally unchanged, but not that it is authentic or current. Re-deriving the pure benchmark closes fabricated and stale selection paths while retaining deterministic restart behavior | A signed transactional evidence authority provides stronger provenance without changing the same eligibility contract |
| D-118 | Derive profitability identity and limits only from a validated selected-run receipt | Allowing a manually assembled run ID or risk policy after selection would break attribution and permit outcome-aware policy drift. The receipt's content address therefore becomes the canonical profitability run ID and its preregistered limits are projected without reinterpretation | A transactional selected-run store emits the same validated policy directly while preserving immutable evidence identity and zero order authority |
| D-119 | Keep discovery and inference timeout domains independent | Network discovery must remain tightly bounded even when a long Codex inference is intentionally unbounded or externally supervised; reusing an optional inference value made `None` reach numeric fetch arithmetic and broke six existing paths | One shared operation-budget contract expresses both independent ceilings without coupling their failure semantics |
| D-120 | Absorb q's news delta semantically into one transactional owner | Replaying eleven commits onto the combined tree would duplicate or weaken its pending-publication, append-time history, and timestamp-valid consumer contracts. The integrated owner retains those stronger invariants while adding q's lifecycle, memory, capacity, CLI, and bounded-retry behavior exactly once; concurrent pipe draining also preserves timeout authority under large output | A future transactional supervisor can own source revision, inference capacity retry, publication generation, and unit activation without weakening the present one-process/one-publication boundary |
| D-121 | Bound every completion-scheduled news run | An infinite child plus infinite service wall can wedge `OnUnitInactiveSec=4h` forever, bypass failure restart, and silently age the causal signal. The measured `30s` discovery, `840s` inference, and `90s` completion reserve preserve long research while guaranteeing release or retry | A heartbeat-aware supervisor can prove forward progress and safely extend a specific run without weakening the four-hour freshness contract |
| D-122 | Warm the Monday observer by trading context, not calendar adjacency | `2 D` excludes Friday on Monday and can make checkpoint freshness look healthy while the multihorizon sensor has no prior session. A `1 W` request safely spans ordinary and holiday weekends; canonical disk reuse still asks IBKR only for sparse gaps and the live tail | A central warm-up planner derives the smallest calendar window from the active signal plan and exchange calendar while proving at least the same 25-bar readiness |
| D-123 | Let the session recorder, not tunnel startup timing, own forward-evidence recovery | The application already has durable resume and bounded exponential reconnect, while a hard dependency on a three-attempt tunnel could prevent that recovery owner from starting at all. An uncapped 30-second tunnel retry plus producer `Wants` preserves eventual recovery; the bounded observer retains `Requires` and remains fail-closed | A q-local or broker-independent forward source removes the Mac/tunnel boundary while preserving the same append-only provenance contract |
| D-124 | Schedule only the first observer proof; never pre-authorize the cadence | Missing the exact first slot would lose causal evidence, while enabling all 78 slots before inspecting the first receipt would bypass the deployment gate. A one-shot transient timer reuses the canonical shadow service and leaves the recurring timer disabled | The first checkpoint is `EVALUATED`, provenance-complete, and `order_authority=none`; then explicitly arm the remaining schedule |
| D-125 | Make the temporary first-proof trigger reboot-durable, then remove it | A transient timer survives logout but not a manager/machine restart; missing the first slot is costlier than one temporary installed timer. `Persistent=yes` preserves the trigger, while the canonical service still rejects late/stale evidence and the recurring cadence remains disabled | Immediately after inspecting the first terminal receipt, disable and delete the temporary timer before deciding whether to arm the canonical cadence |
| D-126 | Reject the five-minute source/gate/lifecycle family; test a lower-turnover horizon without relaxing economics | All `67` one-year survivors exceeded five-year cadence yet lost net. The best path retained positive gross movement but surrendered it to the frozen per-trade friction, so adjacent five-minute thresholds or cheaper assumed costs would outcome-mine the same mechanism | Independently frozen `15m/30m` candidates preserve positive net economics, cadence, clustered confidence, side/year attribution, and bounded concentration through both unchanged challenges |
| D-127 | Hold the weak 30-minute behaviors and run one cap-independent breadth audit before closing bar-only discovery | Seven labels reduce to two ledgers; neither has positive daily confidence, all-year gross strength, adequate friction margin, or unconcentrated net economics. Promoting `+25.34` over `851` trades would mistake a fragile historical residue for a champion. The only untested structural question is whether the preregistered top-250 cap discarded a stable member of the already-frozen four-week qualified population | An all-qualified, content-addressed 30-minute freeze yields an unchanged identity that is positive after frozen friction and cadence in each of five annual slices, then passes full-ledger confidence, concentration, side, drawdown, and cost-sensitivity gates; otherwise prospective microstructure is the sole remaining admission frontier |
| D-128 | Permit the heartbeat thread on the one locked sweep-store connection | Background progress is part of the canonical evaluator, while SQLite's default creator-thread rule silently prevented its persisted heartbeat and caused false worker recycling. The store already serializes every connection access, so changing only the connection affinity is the smallest repair | A dedicated heartbeat connection or process-local progress authority can prove less contention while preserving identical restart receipts |
| D-129 | Close adjacent historical bar-only XSP admission | The cap-independent preregistered audit gave every `30m` four-week-qualified identity the same five annual gates and reached zero before the final slice. More nearby horizons, thresholds, friction relief, or gate permutations would outcome-mine a rejected evidence class | A materially independent prospective quote/liquidity or same-tape microstructure mechanism earns positive chronological and forward economics under a new preregistered identity |
| D-130 | Do not curate the stable turn detector into sparse in-sample wins | The detector's half-year precision/recall is stable; the economic failure comes from adverse path and false admission. A frozen multivariate model found positive in-sample pockets only by retaining a negative daily lower bound or starving cadence/concentrating gains, and therefore never earned authority to expose later partitions | Timestamp-correct prospective microstructure or another independent causal feature improves the unchanged detector's admission economics across complete sessions without relaxing cadence, reliability, side, cost, or concentration gates |
| D-131 | Make only the once-per-session quote trigger persistent | In-process recovery cannot help if q reboots across the sole daily trigger. Persistence is safe because an in-window catch-up resumes the same durable tape and an expired catch-up is rejected before broker work; applying it to the high-frequency shadow cadence could instead create bursts and stale evaluations | A broker-native durable capture scheduler subsumes the trigger while preserving trading-date identity, exact cadence, restart deduplication, and the same closed-window no-broker boundary |
| D-132 | Reject static multiscale coefficients; test causal walk-forward adaptation once | Twelve independently regularized/density-varied discovery cells passed, yet the preregistered leader reversed both sides and lost sharply in the next calendar year. Nearby static thresholds would therefore tune to a known regime. A fixed walk-forward law is materially different: it may recalibrate only from already-matured prior shadow outcomes and must preserve the same density/economic gates | A content-addressed monthly walk-forward identity passes untouched annual slices and prospective shadow evidence without outcome-aware window, feature, threshold, or lifecycle changes |
| D-133 | Stop curating turn events; test coherent directional expansion as a distinct source | Static and monthly-adaptive admission both failed because the raw event remains a retracement/turn that repeatedly reaches its initial stop. The opening-drive hypothesis is instead a new transition into aligned fast slope, velocity, and volatility expansion. Projecting that event from the same causal snapshots isolates source semantics without changing the engine or exits | One frozen expansion identity preserves density and positive chronology through 2023/2024, then earns prospective shadow parity; otherwise historical bar-only direction-source research closes |
| D-134 | Close historical bar-only direction-source curation | The final materially distinct expansion source produced adequate cadence and a few positive point estimates, but zero reliable discovery pass before any later partition was exposed. Retuning neighboring slope, velocity, ATR, clock, cooldown, or permission thresholds would now select against known outcomes rather than test new information | Timestamp-correct prospective quote/liquidity evolution, causal-news context, or another independently sourced mechanism improves the unchanged detector's admission economics under a preregistered paired comparison |
| D-135 | Persistent source ownership, not a one-bar proposal, governs directional lifecycle | A raw turn is a causal entry proposal that may disappear on the next bar; using it as an open-position owner made entry and flip semantics disagree and allowed a tight stop/trail stack to mask the directional engine. One shared lifecycle direction now follows the persistent impulse trend while admission remains separately observable | A stronger canonical position-intent contract subsumes both source ownership and entry proposals without losing exact backtest/live trace parity |
| D-136 | Freeze the opening-edge plateau as a research champion; let prospective economics decide admission | Tight neighboring values reproduce positive net economics at the required cadence, but repeated historical inspection, a negative five-year daily LCB, `59.95` drawdown, and losing 2021/2023 slices forbid promotion. Further same-tape threshold polishing would weaken rather than strengthen the claim | The unchanged identity earns positive, coverage-complete prospective economics and drift evidence through the frozen selection and `24h → 48h → five-session` contracts |
| D-137 | Crown Opening Edge v1 for research while leaving the operational crown vacant | The candidate now has stable-neighborhood, chronological, trace, shared-owner, exact-prefix, and full-suite proof, so withholding a durable research identity would lose valuable lineage. Those receipts still do not repair its negative five-year daily LCB, weak years, or absent prospective economics | The content-addressed candidate passes complete prospective `24h`, `48h`, and five-session economics/safety/drift gates and receives an explicit operational promotion |
| D-138 | Preserve controlled reverse admission; reject global exit-to-flat | The central lifecycle already auto-reverses only when the inverse source and a fresh opposite admission coincide. Removing that carry changed no common trade, discarded 33 independently admitted handoffs worth `+15.78`, and fell below `200` trades/year for negligible drawdown relief | Independent prospective evidence identifies a causal veto for the losing bullish handoff subset without weakening bearish handoffs, cadence, or unchanged-crown economics |
| D-139 | Repair optional excursion semantics but reject late profit locks | A trail should not secretly require an initial stop, so the shared pure state now models absence honestly. The complete preregistered high-activation family improved only the already-inspected recent month while sacrificing one-/five-year edge and drawdown, so operationalizing it would encode recency bias and unnecessary live state | A materially independent prospective mechanism—not a wider/higher adjacent trail—improves the unchanged crown across complete forward sessions and earns exact live/restart parity before authority |
| D-140 | Model weak intervals as one strategy-health failure with distinct market-path causes, not one hidden regime | Every weak cluster retains flip churn while EOD expansion stops financing it, but the causal paths differ: 2023/early 2025 mostly starve, while spring 2025 gives back large excursions under elevated volatility. Zero of `42` pre-entry features shifts even `0.1` pooled IQR consistently across all three. A trailing `20/40/63` health consensus detects most weak sessions but also labels profitable 2022/2024 intervals, so using it as an admission veto or selector would be a lagging regime router by another name | Timestamp-correct prospective quote/news evidence distinguishes quiet starvation from turbulent giveback and improves unchanged-crown economics without suppressing cadence; until then the state remains evaluator telemetry only |
| D-141 | Keep crash/rebound transition as a peer research state | The shared directional engine sees the relevant multiscale reversals and already captured profitable Mar 3, Mar 9, and Jun 5 paths, but missed Jun 9 when the opening down turn failed the crown's ATR-velocity admission. Crash displacement is therefore neither identical to weak-period degradation nor justification to revive legacy TQQQ shock defaults. One observable peer state may reuse directional impulse, calibrated XSP displacement, forward GTH context, and causal news without owning direction or becoming a regime router | A preregistered same-tape comparison proves incremental crash/rebound expectancy across multiple independent episodes and prospective sessions while preserving exact source/lifecycle ownership |
| D-142 | Preserve the frozen defensive-news observer; treat headline history as plausibility only | The three weak clusters lose on both sides (`2023 -23.91 up/-25.66 down`; `Dec24–Mar25 -23.72/-4.56`; `Apr–Jul25 -8.41/-19.21`), and selective contemporaneous headlines include both persistent stress and sharp narrative reversals. A generic negative-sentiment gate therefore cannot bind the degradation, while adding historical labels now would create hindsight. The existing four-hour service is suitable for persistent causal context, not exact intraday turn timing | Unchanged prospective `60m` pairs show that fresh opposite pressure improves avoided-loss versus foregone-gain economics; pressure sign and `new/strengthening/weakening/reversal/unchanged` remain attribution strata unless separately preregistered evidence earns a new versioned gate |
| D-143 | A representation repair updates the existing crown; it does not create a successor | Translating raw cache bar starts to causal closes changes clock labels and content addresses, but the ordered physical ledger, prices, sides, exits, and economics are identical after the exact five-minute mapping. Calling that Opening Edge v2 would manufacture strategy lineage from a bugfix | A materially different candidate improves the crown across recent, latest-year, annual-slice, five-year, cadence, drawdown, and prospective evidence—not merely code or timestamp representation |
| D-144 | Breadth may describe signed participation but dispersion cannot own direction | The retired TICK-width path inferred `up/down` from wide/narrow activity and was enabled nowhere. Causal 3/6-bar signed breadth modestly improves aggregate five-year economics yet worsens the freshest windows and does not cure losing years; proxy and same-tape discovery risk remain | A preregistered prospective paired challenger using authentic, fresh breadth materially improves unchanged-crown expectancy/drawdown without losing cadence, recent stability, or both-side attribution |
| D-145 | Keep the root brain priority-ordered and move cold history behind stable markers | Re-reading full closed campaigns, completed task narration, and hundreds of verbose receipts on every resume obscures the live authority and invites process noise such as routine test counts into strategy state. A cold archive preserves forensic context without competing with the active brain | Restore material to the root only when it becomes an active authority, unresolved risk, current candidate requirement, or next action; never delete unfavorable evidence |
| D-146 | Preserve signed index values by explicit contract identity and bind breadth only as timestamp-causal context | A global `close<=0` price guard erased legitimate bearish/zero TICK values, while a generic exception would admit invalid prices everywhere. Explicit TICK identity keeps the price invariant intact. Exact NYSE TICK is not currently entitled, so NASDAQ TICK must remain a named proxy; missing/stale/underwarmed proxy evidence cannot alter the unchanged crown | A preregistered prospective paired ledger proves that breadth-150 improves unchanged-crown economics and cadence; exact NYSE data may replace the proxy only after entitlement and same-tape parity are proven |
| D-147 | Keep TICK as a pure research primitive, not an active XSP subsystem | The signed-data correction and causal reducer are valid, but the failed challenger did not justify a second broker request or repeated receipt/task-brain surface. Runtime prominence without earned incremental value distracts from the higher-priority champion and prospective-profit quest | Reintroduce a paired TICK observer only in a separately preregistered future study that materially improves recent, annual-slice, five-year, cadence, and prospective evidence |
| D-148 | Close pre-Monday historical lifecycle tuning with Opening Edge v1 unchanged | Corrected source ownership reopened the prior overbroad rejection and justified one fair tournament. Every distinct recent improvement then weakened one-year or five-year evidence, while the unchanged crown alone retained the required cadence and chronological economics. Continuing to vary the same historical bars would now outcome-mine rather than prepare Monday | Reopen challenger work only for materially independent timestamp-causal evidence, especially prospective liquidity, causal news change, or crash/rebound observations; first collect the scheduled unchanged crown evidence |

---

<!-- XSP-ARCHIVE:A05:END -->

---

<a id="xsp-archive-a06-prior-conclusion"></a>
<!-- XSP-ARCHIVE:A06:BEGIN -->
# Archive A06 — Prior accumulated-status conclusion

## Conclusion

- **Quest:** XSP Mastery — Live Research Kata
- **Current authoritative status (E-172):** Phase 1 has its first XSP research
  crown, Opening Edge v1, frozen at `+10.98/17` recent,
  `+131.74/204` one-year, and `+120.59/1,019` five-year with
  `204.61` annualized trades. Its `close-time-parity-r1` runtime revision
  exactly preserves the `1,019` physical trades while correcting every causal
  clock label; this is not a second crown. The false TICK-width directional
  gate is retired, and the honest signed breadth-150 challenger remains
  unpromoted because it worsens recent and latest-year evidence despite a
  modest five-year improvement. The central candidate owner exactly reproduces
  causal prefix economics and is wired into Monday's non-submitting evaluator.
  It remains a research crown:
  `NO_TRADE` is selected, order authority is `none`, and the profitability
  clock is `NOT_STARTED`.
- **Historical accumulated status through E-149 (retained):** Phase 0.4 is complete. The first bounded chronological
  tournaments rejected both the opening-reclaim alpha family and the
  baseline-only safe-income condor; a later exact matched audit confirmed that
  none of `2,304` adverse-friction condors passed the safe-income gate or added
  stable value over a vertical. The preregistered opening-state matrix also
  found no family-wise edge; SPY/VIX context did not rescue it, and the first
  directional-credit singleton failed chronological repeatability. The
  preregistered directional-debit family produced zero positive daily LCBs.
  All family-specific holdouts remain unobserved. Phase 1 now has a complete
  five-year XSP underlying tape, causal calibration records, and a reproducible
  128-cell credit-barrier census. The census rejects unconditional
  multi-session carry at current evidence; the subsequent preregistered
  direction/gap/rolling-quiet eligibility family also produced zero discovery
  passes without opening validation or holdout. The first GTH probe confirmed
  qualified chain/quote availability but returned mixed live/delayed provenance,
  so it remains non-executable evidence. Fresh session-conditioned forward
  option evidence is the active seam; the RTH boundary study remains only the
  preregistered U.S.-open hypothesis. The restart-safe forward
  `live_calibration.v1` command now runs against exact qualified XSP and the
  shared cache/evaluator without order authority; its first closed-session
  Gateway smoke correctly produced no forecasts or economics.
  The shared multihorizon directional observer, reversal cascade, VIX
  confirmation, and exact NASDAQ TICK/TRIN participation families have now
  all completed their frozen chronological studies without producing an
  admissible champion; their holdouts and live authority remain closed. The
  forward evidence spine now includes restart-safe option tapes, first-seen
  option-parity movement, timestamp-valid causal-news context, signal-
  independent shadow checkpoints, a read-only q-to-Mac Gateway boundary, and
  one fail-closed selected-economics authority. The sole prospective
  option-parity interpretation is now frozen as an exact-sign, independently
  non-overlapping aligned shadow candidate with explicit reliability gates.
  The selected-economics authority now freezes independent, content-addressed
  `24h`, `48h`, and five-session evidence prefixes; later P&L, gaps, or retries
  cannot rewrite them. The combined WIP is green at
  `832 passed, 4 deselected`, and every unique behavior in q's eleven-commit
  news delta has been absorbed into the stronger combined publication
  architecture. The production foundation at `b943ea4` and operational
  corrections through E-149 are fully verified; q's pinned runtime and
  installed units verify, the completion-safe news wall is restored, the
  read-only quote producer is armed, Monday's shadow window retains
  prior-session warm-up, and the shadow timer remains disabled.
  The deployed recent-tape replay is byte-deterministic and honestly remains
  `HOLD`. Therefore
  `NO_TRADE` remains selected and the `24h → 48h → five-session` economic
  clock remains exactly `NOT_STARTED`.
- **Next action:** let the already-armed producer begin without backfill at
  Sunday `20:15 ET` / Monday `10:15 AEST`; inspect the already-scheduled first
  Monday `09:37 ET` / `23:37 AEST` non-submitting observer **and Opening Edge
  candidate-equity** checkpoints, verify exact run/config/tape continuity, then
  arm the remaining observer slots only if both pass. Accumulate the
  preregistered option-parity observer toward `30` timestamp-correct pairs over
  five prospective sessions and compare the frozen independent non-overlapping
  TA-only, exact-sign-aligned, and aligned-plus-Pareto-strengthening receipts.
  Keep `NO_TRADE`
  selected until a materially new
  causal admission mechanism passes its preregistered economics; do not retune
  rejected slope/OHLC families or promote synthetic option evidence.

**Predictive observation:** authentic option quotes may show that execution
friction dominates the small underlying effects seen so far. If so, the best
near-term XSP champion will be a rarer, wider-margin opportunity—or
`NO_TRADE`—rather than a busier selector.

<!-- XSP-ARCHIVE:A06:END -->
