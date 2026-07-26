# XSP Leaderboard

This is the durable XSP crown registry: current leaders, exact edge lineage,
reproduction evidence, failure boundaries, and the next highest-value research
seams. New crown records are prepended. Historical records are never rewritten
to make later outcomes look cleaner.

The leaderboard separates two authorities:

- **Research crown:** historically reproducible candidate worthy of unchanged
  prospective evaluation.
- **Operational crown:** a selected run that has passed complete prospective
  `24h → 48h → five-session` economics and safety gates.

An impressive backtest can earn the first. It cannot grant the second.

---

## Crowns

| Rank | Track | Crown | Identity | State |
|---:|---|---|---|---|
| 1 | RTH directional, one `$1/XSP-point` unit | **Opening Edge v1** | `xsp.opening-edge-directional.v1` | Frozen research champion; prospective counterfactual only |
| — | Operational/live | **Vacant** | `NO_TRADE` | Profitability clock not started; order authority `none` |

### #1 — Opening Edge v1

**Crown thesis:** detect a causal multitimeframe XSP turn, admit only the
opening-window subset whose ATR velocity and retrace geometry distinguish
material movement from noise, then let the persistent direction source—not a
one-bar proposal—own the position until a confirmed inverse source or EOD.

| Contract | Frozen value |
|---|---|
| Symbol / tape | XSP, authenticated five-minute RTH |
| Unit | One synthetic `$1 per XSP point` directional unit |
| Source | XSP-native `directional_impulse`; SPY has no direction authority |
| Horizons | `5/15/30/60/120m` signed slope, velocity, acceleration, ATR-normalized movement, efficiency, coherence |
| Turn observation | `09:35..11:50 ET` causal bar-close clock |
| Core admission | `09:35..11:20 ET` close clock; `0 < ATR velocity < .055` |
| Down admission | Core plus retrace `>= 1.25 ATR` |
| Late-up admission | `11:25..11:30 ET` close clock; retrace `1.25..1.70 ATR`; coherence `>= .75` |
| EMA confirmation | Off |
| Entry | Next open; maximum five entries/session |
| Exit | Persistent inverse-source flip after `12` bars, or EOD |
| Disabled lifecycle | Initial stop, trailing stop, profit target, fizzle |
| Frozen friction | `$0.10` points per round trip |
| Runtime revision | `close-time-parity-r1`; same crown, not a successor |
| Config fingerprint | `bbb0a39166dabf6d6237563c7ed08ecba377dd96abac0d622405f02117c0e1d9` |
| Original config fingerprint | `77e285f377f17115ea01e5d37bef9af53d2f902858fb2fc579dff210efee7e9b` |
| Original recent identity freeze | `fac8d3147cbf45c93a8e26e19ce4af5ad52e76c6bb41ca1778b2eddd54aabc8a` |
| Parity-rerun freeze | `4ba13f3874b1b27ef88ccce99616b0305202e0e88b4aa2dda125a6a0be870943` |
| Campaign artifact | `backtests/out/xsp/xsp_directional_lifecycle_anatomy_20260726.json` |
| Original / parity artifact SHA-256 | `753889ac…` / `b86deb3b…` |
| Tape manifest | `564deda9e4e22d20a649dacf78f27d4b5de0df2e95457f1410591d35065b3b9d` |

The runtime revision fixes representation, not economics. IBKR cache rows are
bar-start timestamps; the canonical evaluator now converts them once to
bar-close timestamps. The old and repaired engines produce the same ordered
physical ledger after that deterministic five-minute clock translation:
`1,019` trades and semantic-ledger fingerprint `4099fe9f…`, with identical
prices, sides, exits, P&L, and crown metrics. The old clock labels above are
therefore preserved only through the original artifact; they do not define a
second strategy.

### Crown economics

All results use the same normal engine, one unit, next-open fills, EOD
flattening, and frozen `$0.10` round-trip cost.

| Window | Sessions | Trades | Annualized | Net points | PF | Drawdown | Daily LCB95 | Up / down P&L | Top-five-day concentration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-29..07-24 | 19 | 17 | 225.47 | **+10.98** | 1.950 | 7.83 | -0.3971 | +10.18 / +0.80 | 0.6943 |
| 2025-07-25..2026-07-24 | 251 | 204 | 204.81 | **+131.74** | 2.044 | 21.29 | **+0.1841** | +56.17 / +75.57 | 0.2805 |
| 2021-07-26..2026-07-24 | 1,255 | 1,019 | 204.61 | **+120.59** | 1.170 | 59.95 | -0.0202 | +56.48 / +64.11 | 0.0987 |

The five-year result clears the required `>200 trades/year`, is net-positive
on both directions, and is not concentrated in a handful of days. It is not
yet reliable enough for promotion: the five-year daily lower bound is
negative, drawdown is large relative to the proposed sleeve, and 2021/2023
lose.

---

## Where the edge came from

### 1. The sensor found turns; the original lifecycle hid them

The shared directional sensor causally combines signed multiscale slope and
return velocity, slope acceleration/reversal, ATR-normalized movement,
volatility velocity/acceleration, efficiency, retrace, and cross-horizon
coherence.

- Five-year material-turn census: 58.8% precision, 73.8% recall, F1 `0.655`,
  ten-minute median lag over 4,124 labels.
- Recent 19-session census: 65.8% precision, 73.8% recall, F1 `0.696`,
  7.5-minute median lag over 65 labels.
- The final opening admission raises recent material-extrema match precision
  to **81.0%** across its 21 causal events; five-year precision is 72.5% across
  1,503 events.

This is a noisy causal turn detector, not a hindsight maximum/minimum oracle.

### 2. Entry and exit ownership were made coherent

The first expansion experiment changed `entry_dir`, while flip exits still read
the transient raw-turn signal. Entry and exit therefore followed different
authorities. The shared evaluator now separates:

- **proposal:** the current bar may request entry;
- **source owner:** the persistent directional state owns an open position;
- **permission:** the centralized entry-control plan says exactly why the
  proposal passed or was blocked.

Backtest, live UI, journal, and prospective replay now consume the same
ownership semantics.

### 3. Tight protective exits were masking the directional hypothesis

The old `0.75 ATR` stop ended 150 of 264 representative paths; another 113
ended at the trail. That experiment tested stop sensitivity, not whether the
direction source could hold a move.

The crown therefore tests the clean hypothesis:

```text
qualified opening turn
    → one fixed unit
    → no stop / trail / target / fizzle
    → persistent inverse-source flip after 12 bars
    → otherwise flatten at EOD
```

Across the five-year crown ledger:

- all `1,019/1,019` trades retain signal and control-plane evidence;
- `986` entries occur on the normal five-minute path;
- `33` controlled-flip entries occur after the ten-minute deferred handoff;
- median hold is `13` bars;
- EOD exits contribute `+376.51` points;
- flip exits contribute `-255.92` points.

The frozen exit-to-flat ablation kept inverse-source exits but disabled only
the admitted reverse handoff:

- recent economics were identical;
- one year fell from `+131.74/204` to `+117.11/196`;
- five years fell from `+120.59/1,019` to `+104.81/986`;
- annualized cadence fell from `204.61` to `197.99`;
- drawdown improved only from `59.95` to `58.46`;
- zero common trades changed; the crown's 33 extra admitted handoffs earned
  `+15.78` (`+22.44/7` down, `-6.66/26` up).

Controlled reversal is therefore retained. The precise open seam is the weak
bullish handoff subset, which requires independent prospective evidence—not
global flip deletion or more same-tape threshold polishing.

### 4. The tight knob search found a plateau, not a magic decimal

The evolution was deliberately local:

1. Extending every direction from 11:15 to 11:25 restored recent cadence but
   admitted roughly 130 noisy historical trades and nearly halved the stronger
   five-year edge.
2. Extending only upward turns preserved more signal: `+84.47/1,081` over five
   years and `+120.96/218` over one year, but late-up noise still diluted the
   core.
3. Inspecting only timestamp-available sensor evidence showed that late-up
   retrace geometry was discriminative.
4. Requiring `1.25..1.75 ATR` retrace and `.75` coherence produced
   `+120.59/1,019`.
5. Tight band-edge and hold checks found:
   - upper retrace `1.70` and `1.75` are identical;
   - lower `1.25` and `1.275` differ by one five-year trade;
   - hold `12` balances edge and cadence;
   - hold `13` improves recent P&L but falls below `200/year`.
6. A final `5×5` neighborhood kept down-retrace `1.20..1.30` and ATR ceilings
   `.0525..0575` positive. The center `.055 / 1.25 / hold12` was frozen for
   stability and lower drawdown, not the endpoint’s slightly higher
   `+122.65`.

No news, option quote, volume, SPY authority, or future outcome was used to
manufacture this crown.

---

## Crown authority and Monday handoff

The same typed `DirectionalImpulseAdmissionPolicy` now owns campaign,
backtest, live diagnostics, and prospective replay configuration. A causal
prefix run through the normal engine reproduces the recent crown exactly:

```text
gross +12.68
cost   -1.70
net   +10.98
trades     17
drawdown 7.83
safety breaches 0
```

The prospective checkpoint is separately identified as
`xsp.candidate-equity.v1`, restart-stable, content-addressed, and
`prospective_counterfactual_only`. It cannot submit an order, replace
`NO_TRADE`, or start the profitability clock.

Promotion requires, in order:

1. complete, fresh, causal checkpoint coverage;
2. unchanged config and tape identity;
3. positive 24-hour economics within frozen safety limits;
4. positive 48-hour economics without one-trade concentration;
5. a positive complete five-session week after costs;
6. backtest/live drift, restart, attribution, and selector-stability receipts;
7. an explicit promote/hold/demote verdict.

---

## Rolling robustness truth

The crown is profitable over the complete five-year tape, but it is not yet a
stable-income strategy:

- `623/1,004` rolling 252-session windows are positive (`62.1%`).
- Worst year: `2022-12-09..2023-12-11`, `-47.34/206` trades.
- Best year: `2025-07-24..2026-07-24`, `+132.67/205` trades.
- The worst year retained flip churn (`-79.13`) while persistent EOD moves paid
  only `+31.79`; the best year retained smaller flip losses (`-16.29`) while
  EOD persistence paid `+148.96`.
- The longest session-close underwater interval lasted `692` sessions
  (`2023-02-17..2025-11-20`); engine intraday drawdown remains `59.95`.

The failure mechanism is therefore **missing session expansion after an
otherwise valid turn**, not one globally removable clock bucket. The crown
stays research-only until independent prospective evidence identifies that
condition without suppressing its required cadence.

### Degradation signatures

The exact five-year replay, all eight authenticated cache shards, and the normal
engine path reproduce the crown at `+120.59/1,019`; no cache, DST, cost, or
optimized-path defect explains the weak periods.

| Signature | Evidence | Authority |
|---|---|---|
| Expansion-financing failure | All three weak clusters retain flip churn while EOD persistence stops covering it | Strategy-health telemetry only |
| Quiet expansion starvation | 2023 and Dec 2024–Mar 2025 contain many entries with little post-entry MFE; a real state-machine chop veto improves net/DD but falls to at most `191.16` trades/year | Rejected as an admission gate |
| Turbulent giveback/whipsaw | Apr–Jul 2025 retains large favorable movement but incurs still larger adverse/flip paths under elevated trailing volatility | One historical episode; observe prospectively |
| Crash/rebound transition | Mar/Jun 2026 shows the shared multitimeframe sensor can catch some large reversals and miss others; this is distinct from weak-period degradation | Peer research state, never a hidden regime router |

No one causal entry scalar binds the weak areas: `42` pre-entry
slope/velocity/ATR/coherence/range features were tested, and zero shifted by
even `0.1` pooled IQR in the same direction across all three weak clusters.
A strictly prior `20/40/63`-session expansion-financing consensus marks
`80.2%/82.5%/84.1%` of the three weak clusters, but also marks profitable
intervals in 2022 and 2024. It is therefore an honest degradation alarm—not a
selector, veto, or permission gate.

### Hindsight headline plausibility

Contemporaneous headlines make causal news a credible **prospective
discriminator**, but not a historical explanation or standalone selector:

- Weak-cluster P&L is not uniformly bullish exposure: 2023 was
  `-23.91 up / -25.66 down`, Dec 2024–Mar 2025 was
  `-23.72 up / -4.56 down`, and Apr–Jul 2025 was
  `-8.41 up / -19.21 down`. An always-bearish rule cannot bind or cure all
  three.
- On 2023-03-16 the crown's worst short lost `-5.12` while an announced
  [First Republic rescue triggered a broad market reversal](https://www.axios.com/2023/03/16/first-republic-rescue-markets).
  Static bearish pressure would not have protected that trade; a timely
  `weakening/reversal` observation might have explained it.
- On 2023-08-24 the crown lost `-4.42` long after
  [Nvidia optimism at the open reversed under rising yields](https://apnews.com/article/a893e995e462f797e0160e28e11b9a72).
- The later clusters include visibly persistent hostile contexts: the
  [post-Fed rebound faded on 2024-12-19](https://www.investing.com/news/economy-news/futures-steady-after-wall-street-swoons-on-fed-view-of-fewer-rate-cuts-3781357),
  [tariff relief reversed into China escalation on 2025-04-10](https://ca.investing.com/news/stock-market-news/trumps-tariff-pause-focuses-trade-war-on-china-markets-bounce-3950735),
  and [Israel–Iran conflict pressured stocks on 2025-06-17](https://www.investing.com/news/economy-news/wall-street-futures-edge-lower-as-mideast-conflict-continues-4098734).

This selective sample is plausibility evidence only. The news service has no
timestamp-correct archive for those years, runs roughly every four hours, and
cannot be credited with catching an intraday headline it never observed.
Monday therefore preserves the frozen `60m` opposite-pressure counterfactual
and attributes each pair by pressure sign and
`new/strengthening/weakening/reversal/unchanged`; missing, stale, aligned, or
late evidence leaves the crown unchanged.

---

## Capability frontier

| Capability | State | Exact next proof |
|---|---|---|
| Shared multitimeframe directional source | Covered | Keep frozen; observe prospective drift |
| Central admission/control trace | Covered | Every candidate entry retains pass/block causes |
| Source-consistent flip/EOD lifecycle | Covered | Compare exit counterfactuals without mutating crown |
| Exact normal-engine prefix replay | Covered | Monday restart and incomplete-session receipt |
| Content-addressed candidate equity | Covered locally | Persist first fresh q checkpoint; verify restart identity |
| Authentic XSP option/underlier tape | Covered infrastructure | Continue forward capture as independent context |
| Causal news context | Experimental | Paired TA-only vs TA+news veto; never historical-backfill |
| Quote/liquidity admission | Open | Test whether spread, depth, freshness, and quote movement reject adverse entries |
| Weak-year explanation | Partly covered | Two path signatures explain the collapse; require timestamp-correct prospective discrimination |
| Unconditional path telemetry | Covered | Every backtest trade now records bars held, MFE, and MAE even when stop/trail policy is absent |
| Exit quality | Partly covered | Exit-to-flat and high-activation trails rejected; use independent prospective evidence |
| Operational selection | Blocked | Only after the complete 24h/48h/week promotion ladder |

### Highest-value arcane seams

These are hypotheses, not permission to retune the historical crown:

- **Prospective news veto:** does fresh, high-confidence opposing XSP causal
  pressure prevent the worst admissions without strangling cadence?
- **Forward microstructure:** do widening spread, one-sided books, quote churn,
  or adverse option-skew evolution identify false turns before entry?
- **Flip-quality state:** can persistent-source disagreement plus deteriorating
  MFE/MAE distinguish necessary reversals from costly churn?
- **Weak-period causal split:** can prospective quote/news evidence distinguish
  quiet expansion starvation from turbulent giveback before the path resolves?
- **Crash/rebound peer state:** can large displacement plus short-to-long
  direction reversal earn incremental value without overriding the crown?
- **Cost frontier:** does the edge survive measured live slippage rather than a
  convenient fixed assumption?
- **Future top-signal premium sleeve:** after this directional quest, test
  whether timestamp-valid IV/skew overprices realized post-top movement through
  defined-risk XSP call spreads or neutral structures. Spot bars cannot prove
  this, and naked short premium is outside scope.

Each seam must compare the unchanged crown against one preregistered variant on
the same prospective tape. It earns authority only through incremental,
out-of-sample value.

---

## Crown record protocol

Prepend every future crown or challenger using this exact evidence order:

1. identity, status, timestamp, and predecessor;
2. causal thesis and exact active/inactive gates;
3. code/config/tape/artifact fingerprints;
4. fills, costs, unit sizing, lifecycle, and safety limits;
5. discovery, frozen challenge, annual slices, and prospective windows;
6. P&L, PF, cadence, drawdown, confidence, concentration, and both-side split;
7. signal landing and entry/exit attribution;
8. stable-neighborhood and cost-sensitivity evidence;
9. failed variants and why they failed;
10. live authority, remaining blockers, and the next falsifiable seam.

Research crowns are immutable. A successor gets a new record; it does not edit
its predecessor’s numbers.

---

## Crown history — newest first

### CH-004 · 2026-07-27 · Corrected-lifecycle tournament · Rejected

- EMA, ORB/reclaim, impulse confirmation/veto, supplemental EMA timing,
  profit-only flips, and asymmetric controlled handoffs all used the repaired
  source-consistent lifecycle and unchanged `$0.10` friction.
- EMA `5/10` reached `+17.93/72` recently but lost at least `-47.85` over the
  latest year after its best impulse veto.
- The best supplement reached `+14.38/19` recently, then
  `+124.08/215` over one year and only `+97.38/1,110` over five years.
- Profit-only and directional-handoff variants also improved only the recent
  window. Opening Edge v1 was the sole strategy to survive both later crown
  gates and remains unchanged.

### CH-003 · 2026-07-27 · Signed-breadth challenger · Rejected

- Five-year `+127.64/1,004`, PF `1.182`, DD `52.67`; worse recent and
  latest-year economics, still negative in 2021/2023, and proxy-discovered on
  the same tape.
- Retain only signed-value parsing and the pure causal research reducer. No
  automatic fetch, shadow receipt, gate, selector, or active campaign remains.

### CH-002 · 2026-07-26 · Late profit-lock grid · Rejected

- **Predecessor:** Opening Edge v1, unchanged.
- **Change:** no initial stop; trail activation `2/3/4 ATR` × distance
  `1/1.5/2 ATR`; all other gates, fills, costs, and lifecycle fixed.
- **Result:** all nine improved the latest month to `+12.41..+15.00`, but the
  best one-year result fell to `+105.45` and the best five-year result to
  `+65.99`; no challenger improved crown drawdown.
- **Verdict:** recency-biased family rejected. Keep the optional-stop kernel
  correction, but add no challenger configuration or live trail state.

### CH-001 · 2026-07-26 · Exit-to-flat ablation · Rejected

- **Predecessor:** Opening Edge v1, unchanged.
- **Change:** retain inverse-source exits; disable only the deferred opposite
  admission handoff.
- **Result:** identical recent economics; one year `+117.11/196`; five years
  `+104.81/986`, PF `1.153`, drawdown `58.46`, and `197.99` trades/year.
- **Attribution:** zero common trades changed; 33 omitted crown handoffs were
  worth `+15.78`, split `+22.44/7` down and `-6.66/26` up.
- **Verdict:** reject globally. Preserve the crown and test the weak bullish
  subset only when independent prospective evidence can preregister a veto.

### CR-001 · 2026-07-26 · Opening Edge v1

- **Change:** first XSP research crown.
- **Predecessor:** none; the XSP namespace was intentionally vacant.
- **Runtime repair, 2026-07-27:** `close-time-parity-r1` translated the
  cache's bar-start representation to the canonical causal close clock and
  removed dead TICK-width configuration fields. It reproduced the same
  `1,019` physical trades and all economics; this updates CR-001 rather than
  creating CR-002.
- **Earned by:** source-consistent lifecycle repair, exact signal/control
  traces, tight side-specific admission geometry, stable local neighborhood,
  positive recent/one-year/five-year economics, and `>200` annualized trades.
- **Did not earn:** selected strategy, profitability milestone, broker/order
  authority, or a claim of reliable income.
- **Next contest:** unchanged prospective candidate equity versus authentic
  Monday tape, then independent news/microstructure admission evidence.

### Reproduce

```bash
PYTHONUNBUFFERED=1 venv/bin/python -u \
  -m backtests.xsp.xsp_directional_interaction_campaign --mode lifecycle
```

Expected parity-rerun campaign shape: `192 → 62 → 29`, about 110 seconds on the
current machine, with the unchanged five-year leader
`gate=opening_edge:ema=off:hold=12`.

Canonical quest/evidence journal: `q_XSP_live_research_kata.md`.
