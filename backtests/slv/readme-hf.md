# SLV HF Research (High-Frequency Spot Track)

This file is the dedicated SLV high-frequency evolution track, separate from `backtests/slv/README.md`.
- `README.md` remains the low-frequency/live-parity representative champion track.
- This file tracks the high-turnover RATS line and its own HF crowns.

Canonical execution paths:
- Full comprehensive suite (includes migrated HF profiles): `python -m tradebot.backtest spot --axis combo_full --offline`
- HF timing corridor replay: `python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline`

Current HF champion replay (v12-strict kingmaker; Feb-14 1Y/2Y/10Y contract):
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v12_strict_20260216.json \
  --symbol SLV --bar-size "10 mins" --spot-exec-bar-size "5 mins" --offline --cache-dir db \
  --top 1 --min-trades 0 \
  --window 2016-02-14:2026-02-14 \
  --window 2024-02-14:2026-02-14 \
  --window 2025-02-14:2026-02-14
```

Historical evolution commands below are normalized to current wrappers:
- Spot sweeps/evolution: `python -m tradebot.backtest spot ...`
- Multiwindow kingmaker eval: `python -m tradebot.backtest spot_multitimeframe ...`

## Current Champions (stack)

### CURRENT (v12-strict) — strict 1Y+2Y+10Y dethrone crown
Current promoted crown on the Feb-14 windows after a full strict contract pass (no waiver) on 2026-02-16.

**v12-strict kingmaker #01 [HF strict dethrone]**
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v12_strict_20260216.json`
- Source eval (1Y screens): `backtests/slv/slv_hf_r4_track3_slope_snipe_1y_ranked_20260216_v1.json`
- Source eval (10Y/2Y/1Y): `backtests/slv/slv_hf_r4_alltracks_top32_10y2y1y_ranked_20260216_v1.json`
- Strict audit vs prior crown: `backtests/slv/slv_hf_r4_alltracks_top32_10y2y1y_vs_v11exception_audit_20260216_v1.tsv`
- Variant id: `r4c_074_km03_tight_h3_sm0p0388_sl0p0192`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- 1y (`2025-02-14 -> 2026-02-14`): trades **711**, pnl **14,383.39**, dd **13,007.06**, pnl/dd **1.1058**
- 2y (`2024-02-14 -> 2026-02-14`): trades **1,261**, pnl **15,374.80**, dd **16,006.52**, pnl/dd **0.9605**
- 10y (`2016-02-14 -> 2026-02-14`): trades **4,109**, pnl **-55,742.15**, dd **64,620.33**, pnl/dd **-0.8626**

Promotion contract check:
- `1y trades >= 700 OR higher than champion`: **PASS** (`711`)
- `beat v11-exception on 1y pnl + pnl/dd`: **PASS** (`pnl +1,116.09`, `pnl/dd +0.1674`)
- `beat v11-exception on 2y pnl + pnl/dd`: **PASS** (`pnl +699.26`, `pnl/dd +0.0326`)
- `beat v11-exception on 10y pnl + pnl/dd`: **PASS** (`pnl +469.32`, `pnl/dd +0.0070`)

Backup strict row (higher long-horizon repair, lower 1Y uplift):
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v12_strict_alt_km02_20260216.json`
- Variant id: `r4c_062_km02_tight_h7_sm0p0388_sl0p0192`
- Delta vs v11-exception: `1y pnl +799.84`, `2y pnl +1,025.51`, `10y pnl +877.77`

## Current Hardening Result (investigation)

### v2.1 — target-window long-loss hardening (trade lock)
Status: **DONE (not promoted)**

Contract:
- Hard lock: `1y trades >= 700`
- Hard lock: `1y pnl/dd >= 2.5`
- Explicit target windows:
  - `2025-12-28T20:03:00 -> 2025-12-30T13:06:00`
  - `2025-10-16T21:04:00 -> 2025-10-17T15:28:00`
  - `2026-01-05T03:17:00 -> 2026-01-07T13:04:00`

Best lock-pass reducer found:
- Variant: `v5_002`
- 1y: trades **729**, pnl **33,600.09**, dd **12,695.70**, pnl/dd **2.6466**
- 6m: trades **516**, pnl **27,926.66**, dd **12,162.20**, pnl/dd **2.2962**
- Window long-loss mass: **28,427.39** (vs `rnd_016` **36,366.18**)

Delta vs v2 crown (`rnd_016`):
- `trades -4`, `pnl -6,490.84`, `dd -2,821.67`, `pnl/dd +0.0630`
- `target-window long-loss -7,938.79`

Decision:
- Not promoted because absolute 1Y pnl was lower than the crown at the time.

## Previous Crowns (references)

### v9 — `timing_symm_r0245_c5_m24_v12` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json`
- Source eval: `backtests/slv/archive/champion_history_20260214/slv_hf_v9_timing_true_symmetry_20260213.json`
- 1y: trades **752**, pnl **54,540.13**, dd **14,649.91**, pnl/dd **3.7229**
- 2y: trades **1,285**, pnl **51,403.67**, dd **17,066.21**, pnl/dd **3.0120**

### v8 — `overlay_probe_neutral` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v8.json`
- Source eval: `backtests/slv/archive/champion_history_20260214/slv_hf_v7_round3b_tradefloor752_20260213.json`
- 1y: trades **752**, pnl **54,350.24**, dd **14,633.78**, pnl/dd **3.7140**
- 2y: trades **1,285**, pnl **51,263.25**, dd **17,066.21**, pnl/dd **3.0038**

### v7 — `r2_slope_early_rank024` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v7.json`
- Source eval: `backtests/slv/archive/champion_history_20260214/slv_hf_v6_timing_true_attack_round2_20260213.json`
- 1y: trades **752**, pnl **54,054.17**, dd **14,604.84**, pnl/dd **3.7011**
- 2y: trades **1,285**, pnl **50,977.95**, dd **17,066.31**, pnl/dd **2.9871**

### v6 — `off_combo_04` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v6.json`
- Source eval: `backtests/slv/archive/champion_history_20260214/slv_hf_v5_permissionoff_slope_sniper_20260213.json`
- 1y: trades **751**, pnl **53,341.13**, dd **14,538.14**, pnl/dd **3.6690**
- 2y: trades **1,284**, pnl **50,301.82**, dd **17,066.31**, pnl/dd **2.9474**

### v5 — `on_t03_a_rank_025` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v5.json`
- Source eval: `backtests/slv/archive/champion_history_20260214/slv_hf_perm_slope_hunt_v3_targeted_20260213.json`
- 1y: trades **751**, pnl **52,329.44**, dd **17,346.07**, pnl/dd **3.0168**
- 2y: trades **1,284**, pnl **47,836.49**, dd **16,956.92**, pnl/dd **2.8211**


### v4 — `idealplan_v2_fast_top1` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v4.json`
- Source eval: `[PURGED false-timebars artifact]`
- 1y: trades **751**, pnl **51,284.79**, dd **17,226.97**, pnl/dd **2.9770**
- 2y: trades **1,283**, pnl **45,578.79**, dd **17,197.81**, pnl/dd **2.6503**

### v3 — `attackpass_v2_top1` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v3.json`
- Source eval: `backtests/slv/archive/champion_history_20260214/slv_hf_rats_attackpass_v2_20260213.json`
- 1y: trades **746**, pnl **43,770.01**, dd **16,338.51**, pnl/dd **2.6789**
- 2y: trades **1,271**, pnl **27,696.87**, dd **23,622.59**, pnl/dd **1.1725**

### v2 — `rnd_016` 1Y PnL crown
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v2.json`
- Source eval: `[PURGED false-timebars artifact]`
- 1y: trades **733**, pnl **40,090.93**, dd **15,517.36**, pnl/dd **2.5836**
- 6m: trades **520**, pnl **30,851.19**, dd **14,492.85**, pnl/dd **2.1287**

### v1 — `rand_025` ultra-tight exploit dethrone (trade-locked)
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v1.json`
- Source eval: `[PURGED false-timebars artifact]`
- 6m: trades **335**, pnl **37,517.96**, dd **12,791.19**, pnl/dd **2.9331**
- 1y: trades **646**, pnl **33,287.38**, dd **12,399.62**, pnl/dd **2.6845**

### v0.2 — `rand_221` stability hardening winner
- Source eval: `[PURGED false-timebars artifact]`
- 6m: trades **333**, pnl **35,238.38**, dd **12,714.22**, pnl/dd **2.7716**
- 1y: trades **634**, pnl **30,090.27**, dd **12,229.15**, pnl/dd **2.4605**

### v0.1 — `rand_204` reliability sibling
- Source eval: `[PURGED false-timebars artifact]`
- 6m: trades **339**, pnl **32,867.16**, dd **11,860.89**, pnl/dd **2.7711**
- 1y: trades **639**, pnl **28,265.14**, dd **11,446.03**, pnl/dd **2.4694**

## Evolutions (stack)

### v14.0 — strict dethrone from Round-4 all-track strike (PROMOTED)
Status: **DONE (promoted; strict 1Y+2Y+10Y contract pass)**

Promotion decision (record):
- Date: **2026-02-16**
- Promotion target: strongest strict hit vs `v11-exception` with `1y trades >= 700`.
- Strict audit outcome on shortlist: **7 strict hits**, **11 exception hits**.

Promoted row:
- `r4c_074_km03_tight_h3_sm0p0388_sl0p0192`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- Core deltas vs prior crown `v11-exception`:
  - 1Y: `tr +2`, `pnl +1,116.09`, `pnl/dd +0.1674`
  - 2Y: `tr +6`, `pnl +699.26`, `pnl/dd +0.0326`
  - 10Y: `tr +8`, `pnl +469.32`, `pnl/dd +0.0070`

What made this run distinct:
- High-frequency 5m execution was preserved, but entries were sharpened by micro-timing and risk symmetry:
  - `flip_exit_mode=entry`
  - `flip_exit_min_hold_bars=3`
  - `spot_short_risk_mult=0.0388`
  - `spot_stop_loss_pct=0.0192`
- Stress gate remained explicit (`filters.shock_on_ratio=1.35`, `filters.shock_off_ratio=1.25`).
- The slope-heavy branch was explored in this round, but the strict winner emerged from the tighter non-slope micro corridor.

Artifacts:
- `backtests/slv/slv_hf_r4_track1_recovery_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_r4_track2_conditional_overlay_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_r4_track3_slope_snipe_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_r4_alltracks_top32_for_multi_20260216_v1.json`
- `backtests/slv/slv_hf_r4_alltracks_top32_10y2y1y_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_v11_exception_recheck_10y2y1y_20260216_v1.json`
- `backtests/slv/slv_hf_r4_alltracks_top32_10y2y1y_vs_v11exception_audit_20260216_v1.tsv`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v12_strict_20260216.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v12_strict_alt_km02_20260216.json`

### v13.0 — v11-exception promotion from Strike-3 gate/flip family (PROMOTED BY EXCEPTION)
Status: **DONE (promoted; user-approved 10Y waiver)**

Promotion decision (record):
- Date: **2026-02-16**
- User explicitly approved promotion even with 10Y contract miss.
- Promotion target: strongest 1Y+2Y row with `trades >= 700` from the Strike-3 family.

Promoted row:
- `s3g_gateoff_flipentry_h5_sl0p0192_sm0p039_ptnone`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- Core deltas vs prior `v10`:
  - 1Y: `tr +1`, `pnl +81.29`, `pnl/dd +0.0069`
  - 2Y: `tr +2`, `pnl +73.68`, `pnl/dd +0.0051`
  - 10Y: `tr +6`, `pnl -699.64`, `pnl/dd -0.0035` (waived)

What made this run distinct:
- Flip-exit mode changed from `state` to **`entry`**.
- Hold tightened to `flip_exit_min_hold_bars=5`.
- `spot_short_risk_mult` tightened to `0.039`.
- Stop remained `spot_stop_loss_pct=0.0192`, preserving stability envelope.

Artifacts:
- `backtests/slv/slv_hf_strike3_gateflip_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_strike3_gateflip_1y_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_strike3_gateflip_10y2y1y_ranked_20260216_v1.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v11_exception_20260216.json`

### v12.0 — v10-seeded complex slope/velocity/adaptive replay (NOT PROMOTED)
Status: **DONE (investigation, no dethrone)**

Commands used:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v10.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 40 --jobs 4 \
  --write-milestones --milestones-out backtests/slv/slv_hf_v10_sniper_complex_1y_trade700_20260216_round1_milestones.json \
  --milestone-min-win 0 --milestone-min-trades 700 --milestone-min-pnl-dd -99

python -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/slv_hf_v10_sniper_complex_1y_trade700_20260216_round1_milestones.json \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db --jobs 4 \
  --top 6 --min-trades 700 \
  --window 2016-02-14:2026-02-14 \
  --window 2024-02-14:2026-02-14 \
  --window 2025-02-14:2026-02-14 \
  --write-top 6 --out backtests/slv/slv_hf_v10_sniper_complex_10y2y1y_20260216_round1_top6.json
```

Distinct run signature (what evolved):
- Seeded from the Feb-14 window king (`v10`) to force a true dethrone attempt instead of fresh baseline drift.
- Used the most complex `hf_timing_sniper` pocket:
  - profile modes: `hf_anchor_overlay`, `hf_guard_soft`, `hf_guard_medium`, `hf_crash_probe`
  - `ratsv_branch_a_rank_min in {0.0035, 0.0085}`
  - `ratsv_branch_a_cross_age_max_bars in {3, 6}`
  - `ratsv_branch_a_slope_med_min_pct / slope_vel_min_pct in {(0.000002, 0.000001), (0.000006, 0.000002)}`
  - `spot_branch_b_size_mult in {1.00, 1.20}`
- Adaptive sizing + policy graph were active in the profile modes (target resize, hybrid adaptive multipliers, trend-bias overlay, slope/velocity/TR-ratio references).

Outcome summary:
- 1Y sweep (`>=700`) kept **49** candidates.
- Best 1Y row remained the current crown itself:
  - `sl0192_state_h6_prof1_gateoff_eod0`, `tr=708`, `pnl=13,186.01`, `pnl/dd=0.9316`.
- 10Y/2Y/1Y top-6 validation:
  - #1 remained unchanged current crown (`signal=10 mins`, `exec=5 mins`).
  - No candidate beat current crown on the full 1Y+2Y+10Y contract.
  - Distinct failure mode observed: `exec=1 min` siblings raised trade count (~5,284-5,291 on 10Y) but collapsed pnl to about **-99,922** with `pnl/dd ~ -0.996`.

Decision:
- **Not promoted**.
- Jan-Feb crash monetization did not improve under this complex profile family; added churn and fill-frequency dominated gains.

Artifacts:
- `backtests/slv/slv_hf_v10_sniper_complex_1y_trade700_20260216_round1.log`
- `backtests/slv/slv_hf_v10_sniper_complex_1y_trade700_20260216_round1_milestones.json`
- `backtests/slv/slv_hf_v10_sniper_complex_10y2y1y_20260216_round1.log`
- `backtests/slv/slv_hf_v10_sniper_complex_10y2y1y_20260216_round1_top6.json`

### v11.0 — adaptive overlay high-trade recovery baseline (NOT PROMOTED)
Status: **DONE (investigation baseline, keep for HF crash-ride R&D)**

---
High-trade milestone (important even without dethrone):
- Recovered high-frequency behavior with the new adaptive overlay path.
- Observed high-turnover variants around **6,833 trades/year** (widened pass) and **8,888 trades/year** (clean cache pass).
- This confirms the new slope/ATR-velocity policy path can generate dense HF action, which we can later sharpen for crash/fakeout precision.
---

Core exploration command:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --offline --cache-dir db --realism2 \
  --min-trades 700 --top 20 --jobs 8
```

Adaptive/velocity corridor used:
- `spot_policy_graph=aggressive`
- `spot_risk_overlay_policy=trend_bias`
- `spot_resize_mode=target`
- `spot_resize_adaptive_mode=hybrid`
- `spot_resize_min_delta_qty=3`
- `spot_resize_max_step_qty=2`
- `spot_resize_cooldown_bars=6`
- `spot_resize_adaptive_min_mult=0.90`
- `spot_resize_adaptive_max_mult=1.40`
- `spot_resize_adaptive_slope_ref_pct=0.06`
- `spot_resize_adaptive_vel_ref_pct=0.04`
- `spot_resize_adaptive_tr_ratio_ref=1.00`
- `spot_graph_overlay_trend_boost_max=1.35`
- `spot_graph_overlay_slope_ref_pct=0.06`
- `spot_graph_overlay_tr_ratio_ref=1.05`
- `spot_graph_overlay_trend_floor_mult=0.90`
- branch timing probes:
  - `ratsv_branch_a_rank_min in {0.0035, 0.0085, 0.0185}`
  - `ratsv_branch_a_cross_age_max_bars in {3, 6, 10}`
  - `ratsv_branch_a_slope_med_min_pct in {0.000002, 0.000006}`
  - `ratsv_branch_a_slope_vel_min_pct in {0.000001, 0.000002}`

Outcome summary (1Y, `trades >= 700`):
- Base remained best:
  - `tr=3079`, `pnl=-6173.9`, `pnl/dd=-1.00`
- High-trade adaptive rows (research baseline):
  - `tr=6833`, `pnl=-13625.2` (widened pass)
  - `tr=8888`, `pnl=-17879.6` (clean cache pass)

Decision:
- **Not promoted** (underperformed base on pnl and pnl/dd).
- **Retained as HF baseline** for future work on Jan-Feb 2026 style crash riding with better anti-churn and earlier directional discrimination.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_widened.log`
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_throttle.log`
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_fpfix_v8.log`

### v11.1 — sub-1D bias-gate probe (NOT PROMOTED)
Status: **DONE (investigation, no dethrone)**

Commands used:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 20 --jobs 8

python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 0 --top 40 --jobs 8
```

Corridor focus:
- `ratsv_branch_a_rank_min in {0.0085, 0.0185, 0.0240}`
- `ratsv_branch_a_cross_age_max_bars in {3, 5, 8}`
- `ratsv_branch_a_slope_med_min_pct in {0.000002, 0.000006}`
- `ratsv_branch_a_slope_vel_min_pct in {0.000001, 0.000002}`
- `spot_branch_b_size_mult in {1.10, 1.40}`
- Bias-gate profile mix included `ST@4h` overlays (plus baseline `ST@1d`) under adaptive resize/overlay policies.

Outcome:
- Trade-floor run (`>=700`) stayed negative:
  - best kept: `tr=773`, `pnl=-22981.4`, `pnl/dd=-0.55` (`overlay_only_hybrid`)
- Diagnostic run (`min_trades=0`) showed:
  - base positive but low-frequency: `tr=119`, `pnl=26293.8`, `pnl/dd=0.83`
  - `strict_slow_confirm` had small positive rows (`tr~27..29`, `pnl~+900`) but far below HF trade floor.

Decision:
- **Not promoted**.
- Interpretation: sub-1D gate settings improved micro-quality but over-pruned participation; high-trade path remains loss-heavy in this Feb-anchored year.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_bias4h1h.log`
- `backtests/slv/slv_hf_sniper_1y_diag_min0_20260216_bias4h1h.log`

### v11.2 — guard-heavy rescue corridor (NOT PROMOTED)
Status: **DONE (investigation, rejected)**

Commands used:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 5000 --top 25 --jobs 8

python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 30 --jobs 8
```

Outcome:
- `tested=96`, `kept=0` at `min_trades=5000`
- `tested=96`, `kept=0` at `min_trades=700`

Interpretation:
- Enabling aggressive entry guarding (`slope_tr_guard`-led profile stack) collapsed participation too hard.
- Not a viable HF rescue direction.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade5000_20260216_rescueA.log`
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_rescueA.log`

### v11.3 — entry-permissive rescue corridor (NOT PROMOTED)
Status: **DONE (investigation, no dethrone)**

Command used:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 30 --jobs 8
```

Corridor:
- Profiles: `hf_anchor_overlay`, `hf_guard_soft`, `hf_guard_medium`, `hf_crash_probe`
- `rank in {0.0035,0.0085}`, `cross in {3,6}`, `slope pairs={(2e-6,1e-6),(6e-6,2e-6)}`, `b_mult in {1.00,1.20}`

Outcome:
- `tested=64`, `kept=48`
- Best kept:
  - `tr=899`, `pnl=-11577.9`, `pnl/dd=-0.38` (`hf_anchor_overlay`)
- Guard-medium cluster degraded further:
  - `tr=975`, `pnl~-22.4k to -24.1k`

Interpretation:
- Survivorship recovered, but profitability and participation quality did not.
- In this corridor, trade counts stayed in ~900-975 range, far below the old 6.8k/8.8k signatures.

Artifact:
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_rescueB.log`

### v11.4 — high-floor re-anchor attempt (ABORTED)
Status: **ABORTED (investigation)**

Command attempted:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 3000 --top 25 --jobs 8
```

Abort reason:
- At ~25% progress, all workers still reported `kept=0` with low-hundreds trade footprints.
- Probability of `>=3000` survivors was negligible for this corridor; run terminated early.

Artifact:
- `backtests/slv/slv_hf_sniper_1y_trade3000_20260216_rescueC.log`

### v10.0 — timing-true symmetric hardening round2 promotion (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Promotion basis:
- Continued the timing-only corridor from v9 (no sizing sweep).
- Focused again on `ratsv_branch_a_rank_min`, `ratsv_branch_a_cross_age_max_bars`, `ratsv_branch_a_slope_med_min_pct`, `ratsv_branch_a_slope_vel_min_pct`, with probe/release overlays.
- Hard gate preserved: `1y trades >= 733` and beat prior king on 1Y+2Y for pnl and pnl/dd.

Artifacts:
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`

Outcome:
- New king: `timing_symm_r0240_c4_m24_v12`
- 1y: trades **752**, pnl **54,828.72**, dd **14,678.58**, pnl/dd **3.7353**
- 2y: trades **1,285**, pnl **51,685.81**, dd **17,066.21**, pnl/dd **3.0285**
- Dethrone delta vs v9: 1y `+288.59 pnl`, `+0.0124 pnl/dd`; 2y `+282.13 pnl`, `+0.0165 pnl/dd`.

### v9.0 — timing-true symmetric hardening promotion (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Promotion basis:
- Kept the timing-only corridor (branch sizing frozen during search, production-size verification).
- Focused on `ratsv_branch_a_rank_min`, `ratsv_branch_a_cross_age_max_bars`, `ratsv_branch_a_slope_med_min_pct`, `ratsv_branch_a_slope_vel_min_pct`, plus mild probe/release overlays.
- Hard gate preserved: `1y trades >= 733` and beat prior king on 1Y+2Y for pnl and pnl/dd.

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_v9_timing_true_symmetry_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json`

Outcome:
- New king: `timing_symm_r0245_c5_m24_v12`
- 1y: trades **752**, pnl **54,540.13**, dd **14,649.91**, pnl/dd **3.7229**
- 2y: trades **1,285**, pnl **51,403.67**, dd **17,066.21**, pnl/dd **3.0120**
- Dethrone delta vs v8: 1y `+189.89 pnl`, `+0.0089 pnl/dd`; 2y `+140.42 pnl`, `+0.0082 pnl/dd`.

### v8.0 — strict eq+ trade-floor promotion (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Promotion basis:
- Follow-up to timing-true round3 where best candidates improved pnl/pnl-dd but dropped 1Y trades by 1.
- Enforced strict crown contract against current king:
  - `1y trades >= 752` (equal-or-higher vs prior king),
  - beat prior king on 1Y and 2Y for pnl and pnl/dd.

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_v7_timing_true_round3_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_v7_round3b_tradefloor752_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v8.json`

Outcome:
- New king: `overlay_probe_neutral`
- 1y: trades **752**, pnl **54,350.24**, dd **14,633.78**, pnl/dd **3.7140**
- 2y: trades **1,285**, pnl **51,263.25**, dd **17,066.21**, pnl/dd **3.0038**
- Notes: timing drift is material (`changed_any`: 1Y=388, 2Y=389); short pnl is slightly worse, but this remains a preference axis, not a hard gate.

### v7.0 — timing-true round2 dethrone promotion (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Promotion basis:
- Took `r2_slope_early_rank024` from the round-2 stage-3 production shortlist.
- Applied updated crown contract:
  - short-delta uplift is a preference (not hard gate),
  - hard gates are `1y trades >= 733` + beat prior king on 1Y and 2Y for pnl and pnl/dd.

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_v6_timing_true_attack_round2_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v7.json`

Outcome:
- New king: `r2_slope_early_rank024`
- 1y: trades **752**, pnl **54,054.17**, dd **14,604.84**, pnl/dd **3.7011**
- 2y: trades **1,285**, pnl **50,977.95**, dd **17,066.31**, pnl/dd **2.9871**
- Contract check: **PASS**

### v6.0 — permission-off slope sniper pass (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search grid (tight, non-broad):
- branch sizing held constant for this pass lineage (timing-focused corridor)
- timing-active knobs only around v5 core: `ratsv_branch_a_*`, `ratsv_probe_cancel_*`, `ratsv_adverse_release_*`
- `entry_permission_mode=off` in search path
- stage flow: 1Y sniper then 2Y verify on shortlist
- trade floor: `1y trades >= 733`
- workers: dynamic (`cpu-2`, capped)

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_v5_permissionoff_slope_sniper_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_v5_forensics_combo04_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v6.json`

Outcome:
- Stage counts: `1y=36`, `2y_verify=10`
- Dethrone hits after 2Y verify: **8**
- Promoted `off_combo_04` as new HF 1Y/2Y crown.
- Notes: gain is primarily from improved long extraction and lower 1Y DD; short side remains net negative and remains the timing frontier.

### v6.1 — timing-true attack pass round1 (INVESTIGATION)
Status: **DONE (not promoted)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search contract:
- Freeze branch sizing in search (`spot_branch_a_size_mult == spot_branch_b_size_mult`) to remove sizing confound.
- Sweep timing-active knobs only: `ratsv_branch_a_*`, `ratsv_probe_cancel_*`, `ratsv_adverse_release_*`.
- Hard acceptance gates:
  - material timestamp drift (entry/exit changes),
  - material short-pnl improvement,
  - `1y trades >= 733`,
  - beat king on 1Y+2Y pnl and pnl/dd after reapplying production sizing.

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_v6_timing_true_attack_20260213.json`

Outcome:
- Stage counts: `stage1=30`, `stage2_frozen=14`, `stage3_prod=6`
- `accepted_timing_true_frozen=0`, `accepted_timing_true_prod=0`, `accepted_full_contract=0`
- Best production row (`smed_2e5`) improved behavior metrics but did not satisfy dethrone+timing together.

### v6.2 — timing-true attack pass round2 (INVESTIGATION)
Status: **DONE (not promoted)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search corridor:
- Narrowed around `combo_slope_early` family to recover 2Y continuity while retaining timing drift.
- Kept timing-only knobs and frozen-size search design.

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_v6_timing_true_attack_round2_20260213.json`

Outcome:
- Stage counts: `stage1=20`, `stage2=12`, `stage3=6`
- Under the original strict short-improvement gate, `accepted_full_contract=0`.
- Dethrone hit `r2_slope_early_rank024` was later promoted in `v7.0` when crown contract was updated to prioritize 1Y/2Y pnl+pnl/dd with trade floor.

### v5.0 — targeted permission+slope micro-pass (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search grid (tight, non-broad):
- Focused around v4 crown timing knobs (`ratsv_branch_a_*`, `ratsv_branch_b_*`, `ratsv_probe_cancel_*`, `ratsv_adverse_release_*`)
- permission A/B checks only on top movers
- trade floor: `1y trades >= 733`
- workers: `6`

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_perm_slope_hunt_v2_tight_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_perm_slope_hunt_v3_targeted_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v5.json`

Outcome:
- `candidate_count_on=20`, `off_checks=4`
- Promoted `on_t03_a_rank_025` as new HF 1Y/2Y crown.
- Permission A/B for this winner replayed equivalently in-window; key dethrone lever was branch-A rank sensitivity.

### v4.0 — RATS ideal-plan v2 fast exploit (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search grid (staged, reliability-first):
- Stage B: `ratsv_adverse_release_*` + `flip_exit_*` around the v3 crown core
- Stage C: `ratsv_probe_cancel_*` around stage-B survivors
- trade floor: `1y trades >= 733`
- workers: `8`

Artifacts:
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v4.json`

Outcome:
- Stage counts: `b_1y=576`, `b_2y_verified=80`, `c_1y=216`, `c_2y_verified=100`
- Dethrone hits: **172**
- Promoted `idealplan_v2_fast_top1` as new HF 1Y/2Y crown.
- Notes: short-side/downturn monetization still needs targeted follow-up despite stronger aggregate continuity.

### v3.1 — RATS attackpass v2 exploit (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search grid (tight exploit around v3.0 winner):
- `ratsv_adverse_release_tr_ratio_min`: `1.0325, 1.035, 1.0375, 1.04, 1.0425, 1.045`
- `spot_branch_a_size_mult`: `0.53, 0.54, 0.55, 0.56, 0.57`
- `ema_spread_min_pct`: `0.00075, 0.0008, 0.00085, 0.0009`
- workers: `6`

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_rats_attackpass_v2_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_attackpass_forensic_compare_v2top_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v3.json`

Outcome:
- Total candidates: **120**
- Dethrone-primary hits: **80**
- Strict 2Y trade-lock hits: **0** (known limitation in this family)
- Promoted top candidate (`attackpass_v2_top1`) as new HF 1Y/2Y crown.

### v3.0 — RATS attackpass v1 bridge (PROMOTION FEEDER)
Status: **DONE**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Search grid:
- `ratsv_adverse_release_tr_ratio_min`: `1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05`
- `spot_branch_a_size_mult`: `0.55, 0.60, 0.65`
- `ema_spread_min_pct`: `0.0008, 0.0009, 0.0010, 0.0011`
- workers: `6`

Artifacts:
- `backtests/slv/archive/champion_history_20260214/slv_hf_rats_attackpass_v1_20260213.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_attackpass_forensic_compare_20260213.json`

Outcome:
- Total candidates: **140**
- Dethrone-primary hits: **67**
- Best bridge candidate prepared the v3.1 exploit corridor.

### v2.1 — constrained window hardening around `rnd_016`
Status: **DONE (investigation)**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Artifacts:
- `[PURGED false-timebars artifact]`

Outcome:
- Found lock-pass variant `v5_002` that reduced target-window long-loss concentration.
- Not promoted due meaningful 1Y pnl give-up versus then-current 1Y crown.

### v2 — 1m PnL push dethrone (`rnd_016`)
Status: **DONE**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Artifacts:
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v2.json`

Outcome:
- Promoted `rnd_016` as HF 1Y crown (high-trade 1m/1m line).
- Preserved `rand_025` as separate HF 6M crown reference.

### v1 — rand_221 ultra-tight exploit (hour-leak targeting + hard trade lock)
Status: **DONE**

Command used:
```bash
python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline
```

Artifacts:
- `[PURGED false-timebars artifact]`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v1.json`

Outcome:
- Promoted `rand_025` as v1 HF crown.
- Found strict siblings: `rand_039`, `rand_190`, `rand_041`, `probe_short_003`.

### v11.5 — crash-predref corridor (investigation, not promoted)

What was tested:
- High-trade backbone with two modes:
  - `overlay_only_hybrid_baseline` (control)
  - `overlay_only_hybrid_predref` (new predictive refs)
- Corridor:
  - `rank_min ∈ {0.0035, 0.0185}`
  - `cross_age ∈ {6, 10}`
  - `slope_med/slope_vel ∈ {(0.000002,0.000001), (0.000006,0.000002)}`
  - `spot_branch_b_size_mult ∈ {1.20, 1.60}`

Key predref knobs:
- `spot_resize_adaptive_atr_vel_ref_pct=0.25`
- `spot_graph_overlay_atr_vel_ref_pct=0.25`
- `spot_graph_overlay_trend_boost_max=1.75`
- `spot_graph_overlay_trend_floor_mult=0.88`
- `spot_exit_flip_hold_tr_ratio_min=1.00`
- `spot_exit_flip_hold_slow_slope_min_pct=0.000002`
- `spot_exit_flip_hold_slow_slope_vel_min_pct=0.000001`

Results:
- 1Y floor run (`2025-02-14 -> 2026-02-14`, `min_trades=700`):
  - Survivors recovered (`kept=16`), but all survivors were baseline mode.
  - Best kept: `tr=773`, `pnl=-22147.4`, `pnl/dd=-0.52`.
- Crash slice (`2026-01-01 -> 2026-02-14`, `min_trades=0`):
  - Predref became top performer:
    - `tr=65`, `pnl=+2602.2`, `pnl/dd=0.20`.
  - Baseline in same slice was strongly negative (`pnl≈-19.7k`).

Conclusion:
- Predref knobs are promising for Jan-Feb crash monetization, but not yet compatible with the current 1Y `>=700` trade-floor target.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_probeF.log`
- `backtests/slv/slv_hf_sniper_crash_20260101_20260214_min0_20260216_probeF.log`

### Pinned finding (requested): best crash-capture family

- Best crash-capture family:
  - `tag=overlay_only_hybrid_predref`
  - Strong row (multiple equivalent rows):
    - `tr=65`, `pnl=+2602.2`, `dd=12754.7`, `pnl/dd=0.20`
    - `rank=0.0035`, `cross=6`, `slope=(0.000002,0.000001)` (same outcome across paired slope/mult variants)
- Critical predictive knobs on this family:
  - `spot_resize_adaptive_atr_vel_ref_pct=0.25`
  - `spot_graph_overlay_atr_vel_ref_pct=0.25`
  - `spot_graph_overlay_trend_boost_max=1.75`
  - `spot_graph_overlay_trend_floor_mult=0.88`
  - `spot_exit_flip_hold_tr_ratio_min=1.00`
  - `spot_exit_flip_hold_slow_slope_min_pct=0.000002`
  - `spot_exit_flip_hold_slow_slope_vel_min_pct=0.000001`

### v11.6 — predref bridge pass (1Y, investigation)

Command:
```bash
TB_HF_TIMING_SNIPER_BRIDGE=1 python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 0 --top 40 --jobs 8 \
  --write-milestones --milestones-out backtests/slv/slv_hf_bridge_predref_1y_20260216_tight_milestones.json \
  --milestone-min-win 0 --milestone-min-trades 0 --milestone-min-pnl-dd -99
```

Bridge corridor:
- `cross_age in {4,6}`
- `spot_exit_flip_hold_tr_ratio_min in {0.97,0.99}`
- `spot_graph_overlay_trend_floor_mult in {0.84,0.86}`
- ATR-velocity refs fixed at `0.25` on resize+overlay.

Outcome:
- `tested=33` (`32` bridge rows + base), `kept=33`.
- Best bridge row family:
  - `cross=4`, `floor=0.86`, `tr_ratio_min in {0.97,0.99}`
  - `tr=638`, `pnl=+2969.9`, `pnl/dd=0.12`.
- Trade-floor edge:
  - `cross=6` lifted trades (`~650`) but damaged pnl toward flat/negative.
- Not promoted: still below `>=700` trade floor in sweep output and far from current 1Y HF crown pnl.

Targeted replay probe (`--no-write`) around best row:
- `cross=4 floor=0.86 tr=0.99`: `tr=669`, `pnl=+3853.6`, `pnl/dd=0.142`
- `cross=5 floor=0.86 tr=0.99`: `tr=674`, `pnl=+3327.1`, `pnl/dd=0.122`
- `cross=6 floor=0.86 tr=0.99`: `tr=681`, `pnl=+579.0`, `pnl/dd=0.020`
- continuation at same floor/tr-ratio:
  - `cross=7`: `tr=687`, `pnl=-1301.3`
  - `cross=8`: `tr=695`, `pnl=-1065.4`
  - `cross=9`: `tr=700`, `pnl=-2816.0`

Trade diagnostics artifact:
- `backtests/slv/slv_hf_bridge_trade_diag_20260216.json`
- Key delta (`cross6` vs `cross4` replay): `+12` trades, `-3274.5` pnl.
