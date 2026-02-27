# TQQQ HF Research (High-Frequency Spot Track)

This file is the dedicated TQQQ high-frequency evolution track.
- `readme-lf.md` holds the low-frequency and broader historical TQQQ lineage.
- This file tracks the throughput-biased HF line and its own HF crowns.

Promotion contract (current):
- Promote based on `1Y` first, then reproduce on `2Y`.
- `10Y` is a later reality check (deferred for now).

Canonical execution paths:
- Spot sweeps/evolution: `python -m tradebot.backtest spot ...`
- Multiwindow kingmaker eval: `python -m tradebot.backtest spot_multitimeframe ...`

## Current Champions (stack)

### CURRENT (v19-km01-riskpanic(tr_med>=5.0 neg_gap_ratio>=0.6 long_factor=0.4)-linear(tr_delta_max=0.5)-overlay(atr_compress hi=1.4 min=0.30)) — v18 + riskpanic linear sizing (1Y/2Y promotion)

- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v19_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_atrC_hi1p4_min0p3_20260228.json`
- Dojo replay (warmup+focus tape):
  - Warmup window: `2026-02-10 -> 2026-02-25` (so TR5/gap overlays have state)
  - Focus window: `2026-02-19 -> 2026-02-25` (the last-5-trading-days chop tape)
  - Replay config: `backtests/tqqq/replays/tqqq_hf_v19_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_atrC_hi1p4_min0p3_dojo_warmup_20260210_20260225.json`
- Timeframe: `signal=5 mins`, `exec=1 min`, `RTH`
- Entry window: `09:00–16:00 ET` (RTH-only data; first tradable entries begin after 09:30 ET)
- Risk overlay: `riskoff_tr5_med_pct=8.5` + `risk_entry_cutoff_hour_et=15` (`riskoff_mode=hygiene`)
- Riskpanic sizing overlay (chop/crisis belt):
  - Trigger: `riskpanic_tr5_med_pct=5.0` + `riskpanic_neg_gap_ratio_min=0.6`
  - Effect: `riskpanic_long_risk_mult_factor=0.4` (short factor stays `1.0`)
- Riskpanic linear sizing overlay (pre-panic downshift, volatility ramp aware):
  - `riskpanic_long_scale_mode=linear`
  - `riskpanic_long_scale_tr_delta_max_pct=0.5`
- Cooldown: `cooldown_bars=3`
- Shock detect (no entry gating; enables `atr_fast_pct` for overlay):
  - `shock_gate_mode=detect`, `shock_detector=atr_ratio`, `shock_atr_fast_period=7`, `shock_atr_slow_period=50`
- Graph risk overlay (ATR compress):
  - `spot_risk_overlay_policy=atr_compress`
  - `spot_graph_overlay_atr_hi_pct=1.4`, `spot_graph_overlay_atr_hi_min_mult=0.30`
- Permission gate (needle-thread in v8): `ema_slope_min_pct=0.03`, `ema_spread_min_pct=0.003`, `ema_spread_min_pct_down=0.05`
- Graph entry gate (needle-thread in v9):
  - `spot_entry_policy=slope_tr_guard`
  - `spot_entry_slope_vel_abs_min_pct=0.00012` (leave other graph entry thresholds off)
- Graph exit flip-hold gate (needle-thread in v10):
  - `spot_exit_policy=slope_flip_guard`
  - `spot_exit_flip_hold_slope_min_pct=0.00008` (leave other flip-hold thresholds off)
- RATS-V entry gate:
  - `ratsv_enabled=true`, `ratsv_slope_window_bars=5`, `ratsv_tr_fast_bars=5`, `ratsv_tr_slow_bars=20`
  - `ratsv_rank_min=0.10`, `ratsv_slope_med_min_pct=0.00010`, `ratsv_slope_vel_min_pct=0.00006`
- 1Y (`2025-01-01 -> 2026-01-19`): trades **577**, pnl **46,585.6**, dd **7,870.8**, pnl/dd **5.919**
- 2Y (`2024-01-01 -> 2026-01-19`): trades **1,119**, pnl **63,400.7**, dd **11,732.0**, pnl/dd **5.404**
- Dojo focus window (`2026-02-19 -> 2026-02-25`): pnl **+535.8** (v16 was **-331.9**)

Replay / verify:
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v19_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_atrC_hi1p4_min0p3_20260228.json \
  --symbol TQQQ --bar-size "5 mins" --use-rth --offline --cache-dir db \
  --top 1 --min-trades 0 \
  --window 2025-01-01:2026-01-19 \
  --window 2024-01-01:2026-01-19
```

## Evolutions (stack)

### v19 (2026-02-28) — dethroned v18 (riskpanic linear sizing overlay)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v18 unchanged, but enable the existing linear sizing overlay so we downshift long risk earlier during volatility ramp (without forcing a full panic day):
    - `riskpanic_long_scale_mode: off -> linear`
    - `riskpanic_long_scale_tr_delta_max_pct: off -> 0.5`
  - Outcome: small but real stability-floor lift (throughput unchanged):
    - stability floor (min `1Y/2Y` pnl/dd): **5.380 -> 5.404**
    - `1Y` pnl/dd: **5.783 -> 5.919**
    - `2Y` pnl/dd: **5.380 -> 5.404**
  - Dojo focus: unchanged (**+535.8**) on the chop tape slice.
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v19_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_atrC_hi1p4_min0p3_20260228.json`

### v18 (2026-02-28) — dethroned v17 (ATR compress tune)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v17 unchanged (riskpanic sizing belt stays), but retune the ATR compress envelope to lift the `2Y` floor:
    - `spot_graph_overlay_atr_hi_pct: 1.3 -> 1.4`
    - `spot_graph_overlay_atr_hi_min_mult: 0.4 -> 0.30`
  - Outcome: stability floor lifted again (same throughput band):
    - stability floor (min `1Y/2Y` pnl/dd): **5.341 -> 5.380**
    - `1Y` pnl/dd: **5.756 -> 5.783**
    - `2Y` pnl/dd: **5.341 -> 5.380**
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v18_km01_panicTr5med5p0_neg0p6_long0p4_atrC_hi1p4_min0p3_20260228.json`

### v17 (2026-02-28) — dethroned v16 (riskpanic sizing overlay)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v16 unchanged, but add a targeted TR%+gap-driven sizing overlay so we downshift long risk on chop/crisis tapes:
    - `riskpanic_tr5_med_pct: off -> 5.0`
    - `riskpanic_neg_gap_ratio_min: off -> 0.6`
    - `riskpanic_long_risk_mult_factor: 1.0 -> 0.4`
  - Outcome: stability floor lifted again (and the chop tape improved materially):
    - stability floor (min `1Y/2Y` pnl/dd): **5.178 -> 5.341**
    - `1Y` pnl/dd: **6.104 -> 5.756**
    - `2Y` pnl/dd: **5.178 -> 5.341**
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v17_km01_panicTr5med5p0_neg0p6_long0p4_20260228.json`

### v16 (2026-02-27) — dethroned v15 (ATR compress tune)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v15 unchanged, but retune the ATR compress envelope:
    - `spot_graph_overlay_atr_hi_pct: 1.2 -> 1.3`
    - `spot_graph_overlay_atr_hi_min_mult: 0.5 -> 0.4`
  - Outcome: stability floor lifted again:
    - stability floor (min `1Y/2Y` pnl/dd): **5.111 -> 5.178**
    - `1Y` pnl/dd: **6.050 -> 6.104**
    - `2Y` pnl/dd: **5.111 -> 5.178**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v16_km01_riskoff8p5_cut15_ratsv_rank0p1_slope0p0001_vel0p00006_cd3_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p00008_shockDetect_atrRatio_f7s50_overlayAtrCompress_hi1p3_min0p4_20260227.json`

### v15 (2026-02-27) — dethroned v14 (ATR compress overlay)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Enable shock detector in `detect` mode (no entry gating) so we can compute intraday `atr_fast_pct`:
    - `shock_gate_mode: off -> detect` (detector: `atr_ratio`, `fast=7`, `slow=50`)
  - Use ATR-aware risk overlay to compress both risk + cap in high-volatility bars (without touching the base entry/exit logic):
    - `spot_risk_overlay_policy: legacy -> atr_compress`
    - `spot_graph_overlay_atr_hi_pct: off -> 1.2`
    - `spot_graph_overlay_atr_hi_min_mult: 0.5` (floor)
  - Outcome: massive stability-floor lift (same throughput band):
    - stability floor (min `1Y/2Y` pnl/dd): **4.898 -> 5.111**
    - `1Y` pnl/dd: **4.898 -> 6.050**
    - `2Y` pnl/dd: **4.924 -> 5.111**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v15_km01_riskoff8p5_cut15_ratsv_rank0p1_slope0p0001_vel0p00006_cd3_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p00008_shockDetect_atrRatio_f7s50_overlayAtrCompress_hi1p2_min0p5_20260227.json`

### v14 (2026-02-27) — dethroned v13 (RATS-V loosen needle)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v13 unchanged, but loosen the RATS-V gate slightly:
    - `ratsv_rank_min: 0.11 -> 0.10`
    - `ratsv_slope_vel_min_pct: 0.00008 -> 0.00006`
  - Outcome: stability floor lifted:
    - stability floor (min `1Y/2Y` pnl/dd): **4.816 -> 4.898**
    - `1Y` pnl/dd: **5.329 -> 4.898**
    - `2Y` pnl/dd: **4.816 -> 4.924**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v14_km01_riskoff8p5_cut15_ratsv_rank0p1_slope0p0001_vel0p00006_cd3_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p00008_20260227.json`

### v13 (2026-02-27) — dethroned v12 (cooldown needle cd2->cd3)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v12 unchanged, but increase the anti-churn spacing slightly:
    - `cooldown_bars: 2 -> 3`
  - Outcome: massive stability-floor lift with essentially unchanged throughput:
    - `1Y` pnl/dd: **4.876 -> 5.329**
    - `2Y` pnl/dd: **4.726 -> 4.816**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v13_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd3_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p00008_20260227.json`

### v12 (2026-02-27) — dethroned v11 (cooldown needle cd4->cd2)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v11 unchanged, but re-arm faster after a fill:
    - `cooldown_bars: 4 -> 2`
  - Outcome: this is a pure stability-floor dethrone (2Y lifts materially, 1Y stays strong):
    - `1Y` pnl/dd: **4.908 -> 4.876**
    - `2Y` pnl/dd: **4.606 -> 4.726**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v12_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd2_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p00008_20260227.json`

### v11 (2026-02-27) — dethroned v10 (lower exit flip-hold slope threshold)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v10 unchanged, but lower the flip-hold slope threshold so we suppress flips slightly more often in true trend continuation:
    - `spot_exit_flip_hold_slope_min_pct: 0.00010 -> 0.00008`
  - Outcome: stability floor lifted again (and DD tightened):
    - `1Y` pnl/dd: **4.481 -> 4.908**
    - `2Y` pnl/dd: **4.508 -> 4.606**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v11_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd4_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p00008_20260227.json`

### v10 (2026-02-27) — dethroned v9 (graph exit flip-hold slope gate)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v9 unchanged (riskoff overlay + permission gate + RATS-V + graph entry vel gate), but stop flipping out of winners while slope is still strong:
    - `spot_exit_policy: priority -> slope_flip_guard`
    - `spot_exit_flip_hold_slope_min_pct: off -> 0.00010` (leave other flip-hold thresholds off)
  - Outcome: stability floor jumped hard while maintaining HF throughput:
    - `1Y` pnl/dd: **3.682 -> 4.481**
    - `2Y` pnl/dd: **3.936 -> 4.508**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v10_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd4_hold0_permDn0p05_graphEntryVel0p00012_graphExitHoldSlope0p0001_20260227.json`

### v9 (2026-02-27) — dethroned v8 (graph slope-velocity entry gate)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v8 structure unchanged (riskoff overlay + permission gate + RATS-V), but add a micro guard at the lifecycle gate:
    - `spot_entry_policy=slope_tr_guard`
    - `spot_entry_slope_vel_abs_min_pct: off -> 0.00012` (leave other graph entry thresholds off)
  - Outcome: stability floor lifted materially while keeping HF throughput:
    - `1Y` pnl/dd: **3.559 -> 3.682**
    - `2Y` pnl/dd: **3.553 -> 3.936**
- Preset: `backtests/tqqq/archive/champion_history_20260227/tqqq_hf_champions_v9_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd4_hold0_permDn0p05_graphEntryVel0p00012_20260227.json`

### v8 (2026-02-26) — dethroned v7 (tighten down-permission gate)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - v7 was already extremely stable; we only tightened the down-direction permission gate:
    - `ema_spread_min_pct_down: 0.04 -> 0.05`
  - Outcome: big stability floor lift (both windows improved, throughput essentially unchanged):
    - `1Y` pnl/dd: **3.123 -> 3.559**
    - `2Y` pnl/dd: **3.057 -> 3.553**
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v8_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd4_hold0_permDn0p05_20260226.json`

### v7 (2026-02-26) — dethroned v6 (cooldown+flip-hold rebalance)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - v6 was already elite, so we only touched the anti-churn knobs (and left riskoff + RATS-V intact):
    - `cooldown_bars: 3 -> 4`
    - `flip_exit_min_hold_bars: 1 -> 0`
  - Outcome: stability floor lifted (2Y improved materially while 1Y stayed huge):
    - `1Y` pnl/dd: **3.154 -> 3.123**
    - `2Y` pnl/dd: **2.906 -> 3.057**
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v7_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_cd4_hold0_20260226.json`

### v6 (2026-02-26) — dethroned v5 (RATS-V threshold tightened)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v5 structure unchanged (riskoff overlay + RATS-V gate), only tighten the RATS-V “velocity” edge:
    - `ratsv_rank_min: 0.10 -> 0.11`
    - `ratsv_slope_vel_min_pct: 0.00006 -> 0.00008` (keep `slope_med_min_pct=0.00010`)
  - Outcome: huge stability jump (same trade floor):
    - `1Y` pnl/dd: **1.975 -> 3.154**
    - `2Y` pnl/dd: **2.182 -> 2.906**
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v6_km01_riskoff8p5_cut15_ratsv_rank0p11_slope0p0001_vel0p00008_20260226.json`

### v5 (2026-02-26) — dethroned v4 (RATS-V entry gate)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep the riskoff overlay from v4 unchanged.
  - Add a strict-but-simple RATS-V entry gate (global thresholds; no probe-cancel/release exits):
    - `ratsv_rank_min=0.10` + `ratsv_slope_med_min_pct=0.00010` + `ratsv_slope_vel_min_pct=0.00006`
  - Outcome: materially higher stability floor:
    - `1Y` pnl/dd: **1.749 -> 1.975**
    - `2Y` pnl/dd: **1.918 -> 2.182**
    - Throughput remains HF-grade (still **~600 trades/year**).
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v5_km01_riskoff8p5_cut15_ratsv_rank0p1_slope0p0001_vel0p00006_20260226.json`

### v4 (2026-02-26) — dethroned v3 (riskoff threshold tuned)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Tighten the TR5 riskoff trigger: `riskoff_tr5_med_pct: 9.0 -> 8.5` (keep `cutoff=15`).
  - Outcome: higher `1Y` and `2Y` pnl/dd with essentially unchanged throughput.
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v4_km01_riskoff8p5_cut15_20260226.json`

### v3 (2026-02-26) — dethroned v2 (TR5 riskoff overlay + 3pm cutoff)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Add TR5-based risk overlay: `riskoff_tr5_med_pct=9.0`, `riskoff_mode=hygiene`, `risk_entry_cutoff_hour_et=15`.
  - Outcome: materially higher `1Y` and `2Y` pnl/dd (better DD control + higher PnL) with essentially unchanged throughput.
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v3_km01_riskoff9_cut15_20260226.json`

### v2 (2026-02-26) — dethroned v1 (flip hold=1)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - `flip_exit_min_hold_bars: 0 -> 1` (no other changes) improved stability floor across `1Y/2Y`
    while keeping throughput essentially unchanged.
  - This is specifically anti-churn: stop rapid flip-flop re-entries on the next tradable bar.
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v2_km01_cd3_hold1_20260226.json`

### v1 (2026-02-26) — first HF promotion (KINGMAKER #34 lineage; cooldown needle)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - This is the only config found in the current HF search lane that held positive economics on both `1Y` and `2Y`
    while still printing on the last-5-days timing tape.
  - `cooldown_bars=3` improved `1Y` and `2Y` pnl/dd without killing throughput.
- Preset: `backtests/tqqq/archive/champion_history_20260226/tqqq_hf_champions_v1_km34_cd3_20260226.json`

## Research Notes (migrated)

## Quest & Upcoming Feature: News aggregation overlay, sentiment scoring, and event-driven capabilities
## (multiround GPT parsing on breaking news / events for the final overlay / scoring step)

# Research on 2024 vs 2025 TQQQ volatility spikes, early detection analysis of weekly TR and gap patterns, before sharpening reg gates

• 1. Feb 28 onward: is there a “distinct slope” you can catch (rare, and also hits COVID etc)?

    Yes — but the cleanest “early warning” in the data is not ATR14% level (that lags), it’s TR% velocity / weekly TR regime, and it shows up late Feb → early Mar.

  
  Key facts from the daily RTH bars (db/TQQQ/TQQQ_2016-01-01_2026-01-19_1day_rth.csv), using TR% = TrueRange / prev_close:

  - The week containing Feb 28, 2025 (ISO week starting 2025-02-24) has:
      - weekly TR% median 6.10%
      - week-over-week jump (delta) +2.94
      - that delta is 93.9th percentile vs all week-over-week TR% median changes in 2016–2026
      - drawdown proxy at Feb 28 close (dd60) already about -23%
  - The week starting 2025-03-03 is the real “oh, this is different” week:
      - weekly TR% median 9.50% (95th percentile of weekly TR% medians)
      - delta vs prior week +3.40 (95.6th percentile)
      - 4-week “ramp” (TR_med now vs 4 weeks prior) +5.51 (96.5th percentile)

  So if you’re looking for a slope/angle signal (like your screenshot), it’s real — it just starts showing up as volatility acceleration in the week of Feb 24, and becomes full-blown stress in the week of Mar 3.

  A very sharp “panic week” rule that barely over-triggers
  If you want something that:

  - is rare across 10y,
  - also catches COVID / 2022 / 2018Q4-type events,
  - and is clearly “risk-off downtrend stress”…

  This is the sharpest thing I found from your requested ingredients (TR% + gaps), in weekly form:

  - Weekly panic = (weekly TR% median ≥ 9.0%) AND (≥ 60% of gaps are negative)

  Results over 2016–2026:

  - Flags 17 weeks total (very low).
  - 16/17 (94.1%) of flagged weeks have dd60 ≤ -30% by that week’s last trading day.
  - It flags the exact week you care about: 2025-03-03 (and the extreme week 2025-04-07).
  - It also flags the big historical stress clusters:
      - 2020-02-24, 2020-03-09, 2020-03-16…
      - 2022-03-07, 2022-04/05 clusters…
      - 2018-12 etc.

  This meets your “doesn’t trigger except huge downtrends” requirement better than ATR-based rules.

  2. What do the ATR/TR “waves” look like in the 1–2 years before (weekly view + why March ramps / April peaks)

  Your monthly read is correct: Feb isn’t the volatility peak — March ramps, April peaks. Weekly makes the shape obvious.

  Here’s the “wave” context (weekly medians):

  2024 summer spike (non-crisis-like compared to 2025)

  - 2024-07-29: TR_med 9.87%, neg_gaps 20%
  - 2024-08-05: TR_med 9.70%, neg_gaps 40%
  - Then it mean-reverts quickly.

  So 2024 can print TR_med ~9–10, but it wasn’t “persistent negative-gap stress”.

  2025 ramp + peak

  - 2025-02-24: TR_med 6.10%, neg_gaps 40%  (the “velocity” week)
  - 2025-03-03: TR_med 9.50%, neg_gaps 60%  (panic threshold week)
  - 2025-03-10: TR_med 7.52%, neg_gaps 60%
  - 2025-03-31: TR_med 8.70%, neg_gaps 100%
  - 2025-04-07: TR_med 23.56%, neg_gaps 80% (the peak/reversal chaos)

  That Apr 7 week is exactly your “ATR peaks during reversal” intuition — and you can also see why ATR14% is lagging: the TR and gaps explode first, then ATR catches up.

  For comparison, COVID looks similar in “weekly TR shock” shape, just bigger:

  - 2020-02-24: TR_med 13.08%, |gap|_med 7.54%
  - 2020-03-09: TR_med 20.60%, |gap|_med 16.79%
  - 2020-03-16: TR_med 24.54%, ATR14_med 24.93%

  That’s the same pattern class: gap/true-range explosion first, ATR ramps after.


  3. ATR-expansion vs TR-velocity: which actually catches Mar 5 early enough?

  This part matters for your “prevent the March 5 overnight gap loss” goal.

  - The ATR14%-expansion idea (fast5 median ATR% vs slow20 median ATR%) is too laggy for Mar 5 in your data.
      - With a reasonable “expansion” rule (fast5 ≥ ~6 and diff ≥ ~1.5), it turns on around Mar 6–7, not by Mar 5 close.
  - A TR-based rolling median triggers in time:
      - If you define risk_off_TR5 = median(last 5 daily TR%) ≥ 9.0%
      - It flags 2025-03-04, 03-05, 03-06, 03-07, 03-10 (and then the April cluster)
      - So it is ON at Mar 5 close, which is exactly what you need to avoid holding into Mar 6 open.

  Also, your key “bad overnight” is absolutely consistent with this:

  - On 2025-03-06: daily gap% ≈ -5.05% (big negative gap day)
  - Your strategy did have exposure over the Mar 5 → Mar 6 boundary.

  ## How rare is this TR5 detector (does it overtrigger)?
  - TR5_med ≥ 9 flags 159 days out of 2516 trading days (~6.3% of days).
  - It clusters mainly in the “obvious” stress regimes (2018Q4, COVID, 2022 bear, 2025).
  - It’s fairly selective for deep drawdown: ~84% of flagged days have dd60 ≤ -30%.

  - The champ is in-position overnight on 2101 close→next-open boundaries.
  - If you applied risk-off only on TR5_med≥9 days:
      - it would affect 114 of those overnights (5.4%).
  - If you applied the stricter weekly-panic regime:
      - it would affect 51 overnights (2.4%).

  So: it’s rare enough to plausibly be “surgical”.
  But: the hard trade-off
  Those risk-off nights are not free to avoid:
  - They have fatter left tails (bad gaps), but also fatter right tails (good gaps).
  - For TR5 risk-off overnights, gap-PnL distribution (per overnight boundary) looked like:
      - p05 about -1405 vs normal p05 about -587
      - min about -2950
      - but mean is actually higher than normal (because some huge rebounds exist)



  ### Even with a no‑lookahead “online weekly” implementation, it will NOT catch the Mar 5 overnight loss
  - Option A (most aligned with your intent): cancel next-open entries + forbid new late-day entries when risk_off is ON (prevents “entered near end of day then got gapped”)
  - Detector 2: weekly panic (TR_week_med ≥ 9 AND neg gaps ≥ 60%)

  So the clean integration is: add a new “risk_off overlay” under filters (parallel to shock, not replacing it), and in the spot engine:

  - refuse scheduling pending_entry_dir if it would fill next morning (overnight),
  - refuse fills/scheduling after a cutoff time when risk_off is ON.


  - Weekly panic = (weekly TR% median ≥ 9.0%) AND (≥ 60% of gaps are negative)
  …was computed using the full completed week in our analysis.
  
  - If we naïvely apply “this week is panic” during the week, that’s future leakage (you don’t know Friday’s gap sign on Tuesday).
  This is important because your motivation was specifically the Mar 5 15:05 long → Mar 6 09:35 stop.

  That trade is in your focus CSV:

  - 2025-03-05T15:05:00 → 2025-03-06T09:35:00, pnl -1231.52

  Now, check the actual daily gaps that week (from the daily bars):

  - Mar 3 gap +
  - Mar 4 gap -
  - Mar 5 gap +
  - Mar 6 gap -
  - Mar 7 gap -

  So the neg-gap fraction only reaches 60% on Friday (3/5).
  Meaning: weekly panic turns ON too late to block the Mar 5 → Mar 6 gap unless you allow lookahead (which we can’t).

  So: weekly panic is extremely selective (good), but it doesn’t solve the exact Mar 5 event (bad). It will still help April and other crisis clusters though.

  ———

  ## Solution Candidates (with concise ratings):

  1. TR5 Risk‑Off Overlay (no lookahead) + Option A hygiene [Score: 9/10] ← IDEAL for your Mar‑5 objective
     • Define risk_off = median(last 5 *completed* daily TR%) >= 9.0 (computed from daily bars; no future leak).
     • Apply Option A: block late-day entries + cancel overnight next_open fills when risk_off ON.
     Pros:
      - Actually ON by Mar 5 intraday (using completed days up to Mar 4), so it can block the 2025-03-05T15:05 entry.
      - Still fairly selective (clusters mainly in 2018Q4 / 2020 / 2022 / 2025).
        Cons:
      - Less “rare” than weekly panic (more flagged days than 17 weeks).


• Yes — a news/GPT layer can very plausibly complement your system, and it complements it in exactly the place where your purely market‑derived detectors are structurally weak:

  - Market detectors (TR%, gaps, ATR) are great at saying “volatility regime changed”.
  - They’re much worse at saying “this is the start of a sustained risk-off narrative” early enough (especially if the first big move is a weekend gap), and they’re also bad at saying “the narrative has flipped; this is now a
    rebound regime” (because TR stays elevated into rebounds).

  That’s why your TR5 / weekly‑panic research is still correct and still valuable, even though the naïve “cancel overnights during riskoff” didn’t beat the champ: the detectors are real, but they’re missing the missing variable:
  narrative direction + de‑escalation detection.

  ### Why your TR% + negative-gap signals are still “the right ingredients”

  From your pasted research:

  - They were selective (weekly panic barely triggers).
  - TR5 triggers early enough to matter for the Mar 5 → Mar 6 gap-loss.
  - But they stay “ON” into April rebound conditions because the market is still chaotic.

  So the problem isn’t “TR5 is wrong”. The problem is: TR5 is an amplitude detector; it’s not a direction+phase detector.

  ### Where a news/GPT layer actually helps (and why it could fix the April rebound problem)

  A news/GPT layer can provide two things the market-only layer can’t reliably provide without lag:

  1. Pre-gap anticipation (weekend / overnight)
      - If the first real damage happens between sessions, the only thing you can “see” beforehand is:
          - the narrative worsening (tariff escalation, retaliation headlines, etc.), and/or
          - the market already being in a fragile state (TR ramp, more negative gaps, worsening dd proxy).
      - TR5 only “turns on” after you’ve already had several large TR days.
  2. Reversal / de-escalation detection
      - This is the killer: market volatility stays high during rebounds.
      - A good news classifier could flip from:
          - “escalation / risk-off / negative expected drift”
          - to “de‑escalation / relief / positive expected drift”
      - …even while TR remains high.
      - That’s precisely how you avoid the failure mode we saw: “risk-off stays elevated into the rebound, so you miss the big rebound long(s).”

  ### The right “shape” of an overlay once you have news

  The overlay that makes sense is not “TR5 cancels overnights.” It’s more like a risk-state machine that uses both market data and news.

  Think in terms of a state with at least these outputs:

  - risk_on (bool / intensity)
  - risk_dir ∈ {down, up, neutral}
  - risk_phase ∈ {escalation, panic, cooling, relief}

  Then your actions become directional and phase-aware, e.g.:

  - If risk_dir=down and risk_on=true:
      - block/cancel only late-day long entries (especially ones that will be held overnight),
      - keep/allow shorts (maybe scaled, but carefully),
      - tighten long stops, allow more aggressive loss-cutting,
      - optionally disallow holding long over weekends specifically.
  - If risk_dir=up and risk_on=true (high-vol rebound / relief rally):
      - do not block the rebound longs (this is exactly where the earlier TR-based overlay accidentally hurt you),
      - if you do anything, you scale down size rather than block participation.

  This is exactly the missing lever: the TR/gap regime says “this is a tail-risk environment”; the news regime says “tail-risk is currently skewed down vs skewed up.”

  ### Where it should plug into your pipeline (conceptually)

  With news available, I would still not jam it into EMA/supertrend math directly. I’d treat it like shock: a parallel overlay that can influence:

  - Entry permission (block/cancel specific kinds of entries; especially near EOD / weekend)
  - Sizing (risk budget scaling)
  - Exit semantics (e.g., allow flip-at-loss only when risk_dir opposes your position)

  That keeps your core alpha logic stable, and isolates the “narrative risk” system.
