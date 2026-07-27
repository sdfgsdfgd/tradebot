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
| 1 | Balanced 24/5, one `$1/XSP-point` research unit | **Opening Edge v2 — Balanced 24/5** | `xsp.opening-edge-v2-balanced-24x5.v1` | Frozen historical crown; SPY selector is shadow-only |
| 2 | RTH directional, one `$1/XSP-point` unit | **Opening Edge v1** | `xsp.opening-edge-directional.v1` | Immutable predecessor |
| — | Previously selected synthetic shadow | **Opening Edge v1** | `9fac460e…` | Unchanged; v2 has not inherited its prospective evidence |
| — | Operational/live | **Vacant** | `NO_TRADE` | Profitability clock not started; order authority `none` |

### #1 — Opening Edge v2 — Balanced 24/5

**Crown thesis:** retain Opening Edge’s causal XSP RTH turn owner, remove two
portable classes of already-mature bullish admission, and add a non-overlapping
SPY-derived GTH down sleeve whose fast-to-slow velocity cascade can own direction
without EMA, stops, trails, or a daily quota.

| Contract | Frozen value |
|---|---|
| Historical unit | One synthetic `$1 per XSP point`; `$0.10` round-trip friction |
| Live execution mapping | One whole SPY share; RTH signal remains XSP IND, GTH signal/execution is SPY STK |
| RTH source | XSP-native `directional_impulse`, five-minute authenticated index bars |
| RTH horizons | `5/15/30/60/120m` signed slope/velocity/acceleration, ATR-normalized geometry, efficiency, retrace, coherence |
| RTH lifecycle | Opening Edge v1 next-open entry, source-consistent flip after 12 bars, EOD flat |
| GTH source | SPY five-minute all-hours path projected onto the prior XSP close for historical comparison |
| GTH cascade | At least two fast slope-velocity votes; exactly one `60/120m` slow velocity vote; fast/slow ATR `>=1` |
| GTH ownership | Down entries only; qualified up evidence remains exit authority; flat by `09:25 ET` |
| Disabled | EMA authority, regime routing, initial stop, trail, target, fizzle, GTH trade quota |
| Crown artifact | `backtests/xsp/opening_edge_v2_balanced_24x5.json` |
| Historical finalist | `backtests/out/xsp/xsp_24x5_finalists_20260727.json` |
| Finalist / config fingerprint | `59bbe144…` / `d032c247…` |
| Combined ledger / tape | `5af0c8b2…` / `6d239c88…` |
| Authority | Historical research crown only; `order_authority=none`; profitability clock not started |

The frozen results are positive in every available July-to-July annual slice,
the recent acceptance tape, both directions, and both non-overlapping lanes:

| Window | Trades | Annualized | Net points | PF | Drawdown |
|---|---:|---:|---:|---:|---:|
| 2026-06-29..07-24 | 16 | 212.21 | **+11.6675** | 2.3674 | 8.6825 |
| 2023-07-24..2024-07-23 | 244 | 244.00 | **+15.6348** | 1.1364 | 19.0612 |
| 2024-07-24..2025-07-23 | 230 | 230.00 | **+29.3083** | 1.1980 | 26.4689 |
| 2025-07-24..2026-07-24 | 251 | 251.00 | **+134.7224** | 2.0272 | 23.8600 |
| 2023-07-24..2026-07-24 | **725** | **242.31** | **+179.6656** | **1.4562** | **26.4689** |

- Direction attribution: `down +108.4056`, `up +71.2600`.
- Lane attribution: `RTH +169.4600/546`, `GTH +10.2056/179`.
- Full-three-year P&L/drawdown: `6.7878`.
- On the identical three-year tape, Opening Edge v1 produced
  `+121.17/622`; v2 improves net, cadence, PF, drawdown, all three annual
  slices, and the recent window. This is why v2 is a crown rather than a
  parity-fix rename.

#### The decisive RTH signatures

The final improvement did not come from another generic trend filter. Exact
entry anatomy isolated two scale-free **bullish exhaustion/maturation**
signatures. They veto only flat `up` admission; the unchanged raw source still
owns exits from an opposing position.

1. **Already-expanded impulse:** block an up entry when the mean of the
   `5/15/30m` slope-velocity values, each divided by its own horizon TR, is
   `>=0.58`, while the `15m` slope/TR is already `>=0.04`.
2. **Developed, maturing upslope:** block an up entry when `30m` slope/TR is
   `>=0.13` and the `5m - 15m` slope/TR curve is `<=0.32`.

The first signature matched 57 former longs. Every annual cohort lost
(`-13.63/-11.11/-11.86`), for `-36.60` total and PF `0.384`.
Its 24 non-extrema entries lost `-31.22`, PF `0.11`, with 120-minute
MAE `2.18` versus MFE `0.67`. Replaying the lifecycle rather than deleting
trades statically improved net by `40.43` and drawdown by `16.95`.

The second signature was not a magic decimal: the complete
`30m slope/TR .11..15 × 5m-15m curve .28..36` neighborhood kept all annual
slices positive and recent RTH economics at `+12.63`. The robust center
`.13/.32` was frozen.

#### The GTH cascade and why it is not an EMA sleeve

The all-hours lane is an alternative directional owner:

- at least two of the `5/15/30m` slope-velocity horizons point into the
  proposal;
- exactly one of the `60/120m` horizons has joined the turn;
- fast ATR is not below slow ATR;
- an up proposal additionally requires the `30m` slope still to be opposed,
  identifying a reversal cascade rather than an already-developed chase;
- a down reaffirmation must re-prove the ordered cascade;
- a first down with coherence `<=0.75` may establish immediately; otherwise it
  gets one causal bar to prove a lower price and at least two fast velocity
  votes;
- only down proposals enter, but valid up proposals close the down state.

The lane contributes `+10.2056/179`, PF `1.2098`, DD `13.9744`.
Its exact shared-runtime ledger matches the frozen research ledger trade for
trade: SHA-256 `9e04807f…`. The RTH lane likewise matches exactly at
`+169.46/546`, with ledger SHA-256 `1c511d2f…`. This proved that GTH is not a
gate downstream of Opening Edge’s RTH ATR policy; it is a peer admission owner,
so no hidden RTH rule can silently veto it.

#### How the 24/5 signal was sharpened

The pure continuous sensor exposed the useful vocabulary before it exposed a
stationary strategy:

- Literal continuous 24/5 produced `3,144` three-year trades. On the recent
  tape it made `187` trades, lost `-5.45` after friction, but captured about
  `+13.25` gross movement. The information existed; re-flip churn consumed it.
- Requiring fast slope-velocity votes `>=2`, exactly one slow vote, and ATR
  ratio `>=1.05` produced `+11.01/40`, PF `2.26`, DD `3.09`.
- Relaxing ATR ratio to the stable `>=1.00` center and adding three-bar flip
  hysteresis produced `+12.06/44`, PF `2.20`, DD `3.90`, with a slightly
  positive daily LCB95. Positive ATR-velocity, ATR-acceleration, and coherence
  floors were unnecessary.
- `65.2%` of those entries landed from one bar before through three bars after
  a material `±30m` top/bottom. The 28 extrema-adjacent trades earned
  `+13.88`, with MFE `1.25` versus MAE `0.33`; the 15 other trades lost
  `-4.34`, won only `13.3%`, and had MAE greater than MFE.
- Three-bar price follow-through raised purity but fell to `+7.87/31`.
  An acceleration cap reached `+12.17/29`, PF `3.32`, but had no older-window
  proof.
- Persistent turn-state hysteresis reached a spectacular recent
  `+11.16/17`, PF `6.51`, DD `3.76`; the same fixed rule lost
  `-38.36/258` in the first older year. Recent extrema precision alone was
  therefore not crown evidence.

Older anatomy explained the nonstationarity:

- `GTH→RTH +21.18/29` and `RTH→Curb +24.47/77` were strong;
- `GTH→GTH -104.03/965` and `RTH→RTH -78.12/564` were poisonous;
- first daily establishment was a larger failure surface than same-session
  re-flips;
- cross-session state continuity mattered more than a fixed cooldown;
- ordered reversal tension—`15m` slope velocity newly aligned while `30m`
  slope remained opposed—had stable extrema-label AUC
  `0.712/0.725/0.731`, but a binary gate was too sparse to satisfy cadence.
- After the Sunday-evening trading-date repair, all `838/838` older entries
  reconciled to their causal proposals: 358 extrema-adjacent trades earned
  `+64.51`, while the other 480 lost `-134.64`. This is the durable mastery
  target—not merely suppressing a few inverse flips.

Five frozen pure-transition variants all lost over three years
(`-163.78..-51.83` across `791..1,757` trades). Ordered proof for up
proposals cut one continuous path from `-73.05` to `-18.53` at `205/year`;
direct `30m` opposition reached `-16.37` at `200/year`. Re-proving the cascade
for GTH down re-entry then produced `+9.31/520` at `174/year`; restricting
immediate initial downs by coherence reached `+23.47/396`, and one-bar
maturation reached `+25.47/436`, but both missed cadence. The final
non-overlapping down sleeve preserves the useful mechanism without pretending
the pure continuous lane had earned coronation.

#### ATR/velocity lessons retained for future challengers

- Fast/slow ATR level is insufficient: fast ATR can remain above slow ATR
  while the slower volatility background is collapsing.
- Year-1 reversals were especially toxic when slow ATR changed by roughly less
  than `-3%` or more than `+20%` over 15 minutes.
- Very large reversal score, retrace, or ATR expansion was often exhaustion,
  not stronger truth; one observed event reached `6.18×` fast/slow ATR.
- Adaptive confirmation reduced the fixed state rule’s Year-1 loss from
  `-38.36` to `-28.66` at 208 trades, and to `-22.68` at only 152 trades.
  No relative-ATR cell achieved both positive economics and cadence, so this
  mechanism remains telemetry rather than another shipped gate.
- A 66-field regularized linear admission scorer, sparse EMA GTH sleeves,
  daily-incumbent rules, and structural-long bearish overrides also failed
  their development law. The sparse EMA search completed all `108` cells with
  zero Year-1/Year-2 survivors; the clock-free linear scorer lost in all `48`
  identities. They are closed families, not reasons to restart a broad
  indicator sweep.
- A shared calendar defect had treated Sunday-evening XSP GTH bars as civil
  Sunday at the entry-day gate while session accounting called them Monday.
  Correcting the trading date restored `502/502` expected admissions, but the
  exact rerun still rejected all 37 structural-long identities. This was an
  engine truth repair, not alpha.

The next materially new challenger should compare opposing evidence against an
incumbent directional state, normalized by strictly prior volatility and
time-of-session noise. It should preserve session-boundary state and use
prospective news pressure/pressure-delta only as independently timestamped
context—not backfill history or become a hidden regime router.

#### Data, execution, and authority boundary

The historical all-hours tape contains `184,957` five-minute bars over `754`
sessions. XSP supplies authenticated RTH observations; GTH/Curb uses SPY
returns anchored to the prior XSP close. Reliable all-hours history begins in
July 2023, so three years is the maximum defensible challenge for this branch.

The Bot UI exposes two named SPY leaves:

- **RTH Core:** XSP IND signal, one SPY STK execution proxy;
- **GTH Down Sleeve:** SPY STK signal and one SPY STK execution proxy.

Both are visibly shadow-only and return an empty allowed-direction set because
`order_authority=none`. The historical `$0.10` friction is not a claim about
actual IBKR SPY commission or overnight spread. Exact SPY fills/costs must be
replayed before v2 can inherit a prospective selection, start a profitability
clock, or approach an order path.

#### Aggressive variant — recorded, not crowned

The higher-return union is a valuable challenger:

| Window | Net | Trades | PF | Drawdown |
|---|---:|---:|---:|---:|
| Recent | `+9.7675` | 14 | 2.8351 | 5.4200 |
| Year 1 | `+22.1648` | — | — | — |
| Year 2 | `+26.2683` | — | — | — |
| Year 3 | `+142.4924` | — | — | — |
| Full three years | **`+190.9256`** | **705** | **1.5144** | **31.8189** |

It is not the crown: balanced v2 has lower drawdown, stronger Year 2, stronger
recent P&L, higher recent and full cadence, and one fewer admission clause.
The aggressive union remains the correct reference for a future independently
frozen risk/return contest.

### #2 — Opening Edge v1

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

### Opening Edge v1 economics

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

## Opening Edge v1 edge lineage

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

## Prospective authority and handoff

Opening Edge v2 supersedes v1 as the historical leader, but historical
coronation cannot silently transfer v1's already-frozen prospective identity.
The Bot UI selector is therefore visible and inspectable while remaining
non-submitting. Before a v2 selection can be frozen, the exact one-share SPY
ledger must include measured commission, spread/slippage, RTH XSP-to-SPY
tracking, GTH availability, and restart-safe signal/execution provenance.

The same typed `DirectionalImpulseAdmissionPolicy` now owns campaign,
backtest, live diagnostics, and prospective replay configuration. A causal
prefix run through the normal engine reproduces the **v1** selected-shadow
baseline exactly:

```text
gross +12.68
cost   -1.70
net   +10.98
trades     17
drawdown 7.83
safety breaches 0
```

That predecessor's prospective checkpoint emits restart-stable,
content-addressed
`xsp.candidate-equity.v1`. Before any Monday outcome, selection
`9fac460e…` bound that unchanged ledger and config `bbb0a391…` to one
`xsp.selected-shadow-run.v1`; its receipt SHA-256 is `faf294f7…`. This makes
the first exact `EVALUATED` checkpoint eligible under the predecessor contract,
but that clock remains unstarted. It cannot submit an order, replace broker
`NO_TRADE`, transfer authority to v2, or grant capital authority.

Promotion requires, in order:

1. complete, fresh, causal checkpoint coverage;
2. unchanged config and tape identity;
3. positive 24-hour economics within frozen safety limits;
4. positive 48-hour economics without one-trade concentration;
5. a positive complete five-session week after costs;
6. backtest/live drift, restart, attribution, and selector-stability receipts;
7. an explicit promote/hold/demote verdict.

---

## Opening Edge v1 rolling robustness truth

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
| Opening Edge v2 RTH/GTH runtime parity | Covered | Preserve exact ledgers `1c511d2f…` / `9e04807f…` |
| SPY selector transport | Shadow-only | Replay one-share SPY costs/tracking before v2 selection |
| Content-addressed candidate equity | v1 predecessor frozen | Do not let `9fac460e…` imply v2 authority |
| Authentic XSP option/underlier tape | Covered infrastructure | Continue forward capture as independent context |
| Causal news context | Experimental | Paired TA-only vs TA+news veto; never historical-backfill |
| Quote/liquidity admission | Open | Test whether spread, depth, freshness, and quote movement reject adverse entries |
| Weak-year explanation | Partly covered | Two path signatures explain the collapse; require timestamp-correct prospective discrimination |
| Unconditional path telemetry | Covered | Every backtest trade now records bars held, MFE, and MAE even when stop/trail policy is absent |
| Exit quality | Partly covered | Exit-to-flat and high-activation trails rejected; use independent prospective evidence |
| Operational selection | Paused / `NO_TRADE` | Exact SPY replay, then a new v2 freeze and the complete 24h/48h/week ladder |

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

### CR-002 · 2026-07-28 · Opening Edge v2 — Balanced 24/5

- **Change:** first balanced all-hours XSP research crown; explicitly crowned
  by the user after exact annual, recent, neighborhood, and runtime-parity
  review.
- **Predecessor:** Opening Edge v1.
- **Earned by:** `+179.6656/725`, PF `1.4562`, DD `26.4689`,
  `242.31/year`, positive recent and every annual slice, positive RTH/GTH and
  up/down attribution, scale-free RTH poison signatures, and exact centralized
  RTH/GTH ledgers.
- **Execution mapping:** RTH signal XSP IND → one SPY share; GTH signal and
  execution SPY. UI leaves are named `RTH Core` and `GTH Down Sleeve`.
- **Did not earn:** selected-run inheritance, profitability-clock state, broker
  or order authority. Actual SPY cost/tracking replay remains mandatory.

### CH-006 · 2026-07-28 · Opening Edge v2 aggressive union · Retained

- Full three years `+190.9256/705`, PF `1.5144`, DD `31.8189`;
  annual `+22.1648/+26.2683/+142.4924`; recent `+9.7675/14`.
- It remains below the balanced crown because balanced has materially lower
  drawdown, stronger Year 2 and recent P&L, higher cadence, and a simpler
  admission identity.
- Retain as an independently frozen risk/return challenger; do not merge its
  extra weak-fast clause into the crown.

### CH-005 · 2026-07-27 · Inverse-source quality confirmation · Rejected

- A source-consistent `3×4×3` family required raw reversal coherence,
  instantaneous signed strength, and retrace quality before either entry or
  exit could accept the new persistent direction.
- The true no-quality control reproduced the crown exactly. Every material
  confirmation failed the recent cadence/economics gate.
- The sole survivor merely rejected a smoothed reversal when the current raw
  score already pointed the other way: recent and one-year ledgers were
  identical; five years changed four paths and moved from `+120.59/1,019`,
  PF `1.1705`, DD `59.95` to `+121.47/1,018`, PF `1.1720`, DD `60.26`.
- Annual deltas were `0/0/-0.31/+0.06/+1.13/0` for `2021..2026`; the change
  worsened the weak 2023 slice and removed one 2025 trade. No stable mechanism
  or successor was earned, so no runtime policy was added.

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
- **Did not earn:** operational/live strategy, profitability milestone,
  broker/order authority, or a claim of reliable income. Its synthetic shadow
  identity was frozen only to make prospective attribution honest.
- **Next contest:** unchanged selected-shadow equity versus authentic Monday
  tape, then independent news/microstructure admission evidence.

### Reproduce

Opening Edge v2 is reproduced by its content-addressed artifact and the two
exact central-runtime parity receipts:

```text
backtests/xsp/opening_edge_v2_balanced_24x5.json
backtests/out/xsp/xsp_rth_balanced_runtime_parity_20260727.json
backtests/out/xsp/xsp_gth_balanced_runtime_parity_20260727.json
```

Opening Edge v1's historical campaign remains:

```bash
PYTHONUNBUFFERED=1 venv/bin/python -u \
  -m backtests.xsp.xsp_directional_interaction_campaign --mode lifecycle
```

Expected parity-rerun campaign shape: `192 → 62 → 29`, about 110 seconds on the
current machine, with the unchanged five-year leader
`gate=opening_edge:ema=off:hold=12`.

Canonical quest/evidence journal: `q_XSP_live_research_kata.md`.
