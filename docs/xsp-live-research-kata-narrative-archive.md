# XSP Mastery — Narrative Archive

- **Canonical live brain:** [`q_XSP_live_research_kata.md`](../q_XSP_live_research_kata.md)
- **Archived scope:** the original detailed Sections 0–12 through evidence `E-128`
- **Body SHA-256:** `dfc97f209c8190bfd7dc255eb3014d5588745c682b7a0efbbf286f865dbe6f80`
- **Contract:** historical context is preserved verbatim below. Do not resume from
  this file or mutate old outcomes; append current evidence and decisions to the
  canonical root brain.

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
