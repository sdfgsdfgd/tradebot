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
| Core admission | `09:30..11:15 ET`; `0 < ATR velocity < .055` |
| Down admission | Core plus retrace `>= 1.25 ATR` |
| Late-up admission | `11:20..11:25 ET`; retrace `1.25..1.70 ATR`; coherence `>= .75` |
| EMA confirmation | Off |
| Entry | Next open; maximum five entries/session |
| Exit | Persistent inverse-source flip after `12` bars, or EOD |
| Disabled lifecycle | Initial stop, trailing stop, profit target, fizzle |
| Frozen friction | `$0.10` points per round trip |
| Config fingerprint | `77e285f377f17115ea01e5d37bef9af53d2f902858fb2fc579dff210efee7e9b` |
| Recent identity freeze | `fac8d3147cbf45c93a8e26e19ce4af5ad52e76c6bb41ca1778b2eddd54aabc8a` |
| Campaign artifact | `backtests/out/xsp/xsp_directional_lifecycle_anatomy_20260726.json` |
| Artifact SHA-256 | `753889ac8b6866a98808e449fd766571ad515e05115ab1c8f4c83fc3cf41ef81` |
| Tape manifest | `564deda9e4e22d20a649dacf78f27d4b5de0df2e95457f1410591d35065b3b9d` |

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
| Weak-year explanation | Open | Explain 2021/2023 loss with causal, independently available evidence |
| Exit quality | Partly covered | Global exit-to-flat rejected; challenge only weak bullish handoffs with independent evidence |
| Operational selection | Blocked | Only after the complete 24h/48h/week promotion ladder |

### Highest-value arcane seams

These are hypotheses, not permission to retune the historical crown:

- **Prospective news veto:** does fresh, high-confidence opposing XSP causal
  pressure prevent the worst admissions without strangling cadence?
- **Forward microstructure:** do widening spread, one-sided books, quote churn,
  or adverse option-skew evolution identify false turns before entry?
- **Flip-quality state:** can persistent-source disagreement plus deteriorating
  MFE/MAE distinguish necessary reversals from costly churn?
- **Weak-year causal split:** are the losing years explained by opening-range
  compression, delayed expansion, or path asymmetry already observable at
  decision time?
- **Cost frontier:** does the edge survive measured live slippage rather than a
  convenient fixed assumption?

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

Expected bounded campaign shape: `192 → 55 → 24`, about 103 seconds on the
current machine, with five-year leader
`gate=opening_edge:ema=off:hold=12`.

Canonical quest/evidence journal: `q_XSP_live_research_kata.md`.
