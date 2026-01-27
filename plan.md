# TQQQ Spot: Regime Shock / Drawdown Detection Investigation

Status: **active** (started 2026-01-23)

This file is a **temporary, persisted** plan/log for:
1) Understanding where the CURRENT champ loses (loss clusters / regimes).
2) Designing a **new gate** that detects “shock” regimes (Feb–Apr 2025 drawdown, COVID, etc.) using
   **TQQQ-only** volume/volatility/contraction features.

Delete this file when we’re done with:
- Implementing the chosen new gate + running the follow-up sweeps (1/2/3), and
- Updating the spot champs README accordingly.

---

## Fixed evaluation setup (do not drift)

Ticker: **TQQQ only**

Windows:
- 10y: `2016-01-01 → 2026-01-19`
- 2y: `2024-01-01 → 2026-01-19`
- 1y: `2025-01-01 → 2026-01-19`

Execution model:
- Signal on `30 mins` (RTH)
- Execution + intrabar stops + flip exits on `5 mins` (RTH) via `spot_exec_bar_size=5 mins`

Constraints:
- Always require **positive PnL** in all evaluated windows for “champ” consideration.
- For CURRENT champ comparisons, require “no regression” in `roi/dd` across **10y+2y+1y** (unless explicitly relaxing for a high-activity section).

---

## CURRENT champ (as of 2026-01-23)

Source set:
- `backtests/out/tqqq_exec5m_v31_shock_block_longs_30m_10y2y1y_mintr100_top36.json`

Champion identity:
- EMA `4/9` trend, Supertrend bias `4 hours` `ATR=7 mult=0.50 src=hl2`
- Permission: `ema_spread_min_pct=0.003`, `ema_slope_min_pct=0.03`, `ema_spread_min_pct_down=0.05`
- Session: `entry_start_hour_et=10`, `entry_end_hour_et=15`
- Exits: `PT=None`, `SL=0.04`, `exit_on_signal_flip=true`, `flip_exit_only_if_profit=true`, `hold=2`
- Shorts: `spot_short_risk_mult=0.01`
- Shock overlay: `shock_detector=daily_atr_pct`, `shock_gate_mode=surf`, `on=13.5 off=13.0`, `sl_mult=0.75`

Stats:
- 10y: `tr=1079`, `win=51.1%`, `roi=+67.91%`, `dd%=22.04%`, `roi/dd=3.08`
- 2y: `tr=207`, `win=54.6%`, `roi=+27.01%`, `dd%=5.94%`, `roi/dd=4.55`
- 1y: `tr=109`, `win=54.1%`, `roi=+13.06%`, `dd%=4.42%`, `roi/dd=2.96`

---

## This round: “shock regime” investigation (vol/volume + EMA contraction)

Goal:
- Identify whether loss clusters (esp. **Feb–Apr 2025**) align with:
  - **high realized volatility** (RV)
  - **high ATR%** “shock”
  - **volume spikes**
  - **EMA spread contraction** (trend weakening / whipsaw zone)
- Convert that into a new gate axis we can sweep (minimal, shared by live+backtest via `tradebot/engine.py`).
  (core shared module is now `tradebot/engine.py`; `tradebot/decision_core.py` remains as a shim)

Artifacts:
- Loss clustering / drawdown context: `backtests/out/tqqq_exec5m_v11b_regime_adaptability.md`
- Trade dump for deeper feature analysis: `backtests/out/tqqq_exec5m_v11b_kingmaker03_trades_10y.json`
- Feature probe (daily ATR/RV/vol ratio + EMA spread/slope at entry): `backtests/out/tqqq_exec5m_v11b_kingmaker03_regime_features.md`
- ATR shock-gate probe (trade-level approximation): `backtests/out/tqqq_exec5m_v11b_kingmaker03_atr_gate_probe.md`
- v25 champ regime/loss-cluster report: `backtests/out/tqqq_exec5m_v25_kingmaker01_regime_adaptability.md`

Key questions:
1) Are the Apr 2025 losses concentrated during high daily ATR% / RV spikes?
2) Does EMA spread contraction (negative spread delta) predict loss clusters?
3) Can we design a “shock gate” that blocks a small fraction of trades but improves worst-window stability?

---

## Next iteration: sweeps (after we add the new gate)

### 1) Micro squeeze around the CURRENT pocket
Sweep:
- `supertrend_multiplier ∈ {0.47..0.53 step 0.01}`
- `flip_exit_min_hold_bars ∈ {1,2,3}`
- `ema_slope_min_pct ∈ {0.025, 0.03, 0.035}`

### 2) Short-side stability shaping (keep two-way)
Sweep:
- `ema_spread_min_pct_down ∈ {0.05,0.06,0.07,0.08}`
- `spot_short_risk_mult ∈ {0.005,0.01,0.015}`

### 3) Entry debounce / confirmation
Sweep:
- `entry_confirm_bars ∈ {0,1,2}`
- `ema_spread_min_pct ∈ {0.0025,0.0030,0.0035}`

### 4) New gate sweep (to be chosen)
Candidate axes (pick 1 first; more later if needed):
- Daily ATR% shock gate: `daily_atr14_max_pct ∈ {10,12,13,14,15}`
- EMA spread contraction gate: `ema_spread_delta_min_pct ∈ {-0.50,-0.25,-0.10,-0.05}` (units: spread% points per signal bar)
- Loss-streak circuit breaker: `loss_pause_after_losses ∈ {3,4,5}` × `loss_pause_bars ∈ {4,8,12}`

---

## Log

- 2026-01-23: Generated loss clustering report: `backtests/out/tqqq_exec5m_v11b_regime_adaptability.md`.
- 2026-01-23: Generated feature probes for Feb–Apr 2025 + COVID + Oct 2025:
  - `backtests/out/tqqq_exec5m_v11b_kingmaker03_regime_features.md`
  - `backtests/out/tqqq_exec5m_v11b_kingmaker03_atr_gate_probe.md`
- 2026-01-23: Micro-squeeze around the CURRENT pocket found a near-miss no-regression candidate (v14) that improved 10y+2y but did not improve 1y:
  - `backtests/out/tqqq_exec5m_v14_micro_squeeze_variants_30m.json` → `backtests/out/tqqq_exec5m_v14_micro_squeeze_30m_10y2y1y_mintr100_top80.json`
- 2026-01-23: Shock overlay experiments (v12/v13/v17) did not improve worst-window stability (kept for reference, not promoted):
  - `backtests/out/tqqq_exec5m_v12b_shock_gate_30m_10y2y1y_mintr100_top80.json` (block-all)
  - `backtests/out/tqqq_exec5m_v13_shock_surf_30m_10y2y1y_mintr100_top80.json` (block_longs + short scaling)
  - `backtests/out/tqqq_exec5m_v17_shock_detect_shortscale_30m_10y2y1y_mintr100_top80.json` (detect-only + short scaling)
- 2026-01-23: Added a configurable shock direction source + asymmetric long sizing:
  - `shock_direction_source`: `"regime"` (default) or `"signal"` for `shock_dir` smoothing in multi-timeframe runs
  - `shock_long_risk_mult_factor_down`: scales (or disables) long sizing when `shock_dir=down`
- 2026-01-23: Additional shock sweeps (none beat CURRENT across all 3 windows; kept for reference):
  - v18 `"surf"` directional overlay: `backtests/out/tqqq_exec5m_v18_shock_surf_dir_30m_10y2y1y_mintr100_top80.json` (stability min roi/dd ≈ 2.70)
  - v20 `"surf"` + signal-direction: `backtests/out/tqqq_exec5m_v20_shock_surf_signal_dir_30m_10y2y1y_mintr100_top80.json` (stability min roi/dd ≈ 2.42)
  - v21 `"detect"` risk-tilt: `backtests/out/tqqq_exec5m_v21_shock_detect_risk_tilt_30m_10y2y1y_mintr100_top80.json` (no improvement)
  - v22 `"detect"` risk-tilt + long-down scaling: `backtests/out/tqqq_exec5m_v22_shock_detect_risk_tilt2_30m_10y2y1y_mintr100_top80.json` (no improvement)
- 2026-01-23: v25 daily ATR% shock sweep found a new CURRENT champ that strictly beats v11b roi/dd in all 3 windows:
  - `backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_variants_30m.json` → `backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_30m_10y2y1y_mintr100_top80.json`
  - Regime/loss report: `backtests/out/tqqq_exec5m_v25_kingmaker01_regime_adaptability.md`
- 2026-01-23: v31 shock threshold sweep found a new CURRENT champ that strictly beats v25 roi/dd in all 3 windows:
  - `backtests/out/tqqq_exec5m_v31_shock_block_longs_variants_30m.json` → `backtests/out/tqqq_exec5m_v31_shock_block_longs_30m_10y2y1y_mintr100_top36.json`
  - Key delta: `shock_daily_on_atr_pct=13.5` (was `14.0`), `shock_daily_off_atr_pct=13.0`

- 2026-01-23: Generated v31 regime/loss-cluster report (this shows why the “Feb–Apr” bucket still feels bad):
  - `backtests/out/tqqq_exec5m_v31_kingmaker01_regime_adaptability.md`
  - Key finding: in `2025-02-14→2025-04-04` the PnL is still negative and the big losses are long stop-outs (bias ST @4h stayed UP through Mar 4/5).

- 2026-01-23: TR% trigger experiment:
  - Implemented “sticky shock day” behavior for `shock_daily_on_tr_pct` (no effect unless that knob is set).
  - v32 TR% sweep did not improve stability; still kept for reference:
    - `backtests/out/tqqq_exec5m_v32_trpct_trigger_variants_30m.json` → `backtests/out/tqqq_exec5m_v32_trpct_trigger_30m_10y2y1y_mintr100_top12.json`

- 2026-01-23: Shock SL-mult sweep:
  - `backtests/out/tqqq_exec5m_v34_shock_slmult_variants_30m.json` → `backtests/out/tqqq_exec5m_v34_shock_slmult_30m_10y2y1y_mintr100_top6.json`
  - Baseline `shock_stop_loss_pct_mult=0.75` remains best (Apr 2025 pocket gets worse with wider stops).

- 2026-01-23: Entry-confirm debounce sweep:
  - `backtests/out/tqqq_exec5m_v35_entry_confirm_variants_30m.json` → `backtests/out/tqqq_exec5m_v35_entry_confirm_30m_10y2y1y_mintr100_top9.json`
  - No promotable improvement vs CURRENT (best candidate matched 1y roi/dd exactly; trade count was unchanged).

- 2026-01-23: TradingView Supertrend sanity check (Feb→Apr 2025):
  - On cached `TQQQ` RTH `4 hours` bars, ST(10,3,hl2) flips once (up→down) and then stays down through `2025-04-04`.
  - CURRENT bias ST(7,0.5,hl2) flips repeatedly in the same slice (incl. flipping up on `2025-03-04`, which aligns with the big long stop-outs).

- 2026-01-23: v36 Supertrend “TV 10/3” neighborhood sweep (bias gate replacement attempt):
  - `backtests/out/tqqq_exec5m_v36_supertrend_tv10_3_neighborhood_variants_30m.json` → `backtests/out/tqqq_exec5m_v36_supertrend_tv10_3_neighborhood_30m_10y2y1y_all_top19.json`
  - Result: ST(10,3) is visually accurate for the Feb→Apr downtrend, but as a bias gate it fails “positive PnL in all 3 windows” (10y/2y/1y stability goes negative). Not promotable.

- 2026-01-23: v37 Supertrend multiplier squeeze (bias ST @4h ATR=7, mult ∈ {0.5..1.5}):
  - `backtests/out/tqqq_exec5m_v37_supertrend_mult_squeeze_variants_30m.json` → `backtests/out/tqqq_exec5m_v37_supertrend_mult_squeeze_30m_10y2y1y_mintr100_top6.json`
  - Result: increasing mult reduces worst-window stability; baseline mult=0.5 remains best.

- 2026-01-23: Implemented `regime2_apply_to` (optional; default `both`):
  - Allows regime2 to gate only `longs` or only `shorts` (for “permission layer” experiments).
  - Code: `tradebot/backtest/config.py`, `tradebot/backtest/engine.py`, `tradebot/ui/app.py` (tests still pass).

- 2026-01-23: v38 daily drawdown shock detector sweep:
  - `backtests/out/tqqq_exec5m_v38_daily_drawdown_shock_variants_30m.json` → `backtests/out/tqqq_exec5m_v38_daily_drawdown_shock_30m_10y2y1y_mintr100_top9.json`
  - Result: worse 1y stability vs CURRENT; not promotable.

- 2026-01-23: v40 regime2 Supertrend as “longs-only permission” sweep:
  - `backtests/out/tqqq_exec5m_v40_regime2_longs_only_supertrend_variants_30m.json` → `backtests/out/tqqq_exec5m_v40_regime2_longs_only_supertrend_30m_10y2y1y_mintr100_top9.json`
  - Result: passes min-trades only in a couple settings, but 1y/2y roi/dd collapses; not promotable.

- 2026-01-23: v41 TR% triggered shock day sweep (adds `shock_daily_on_tr_pct`):
  - `backtests/out/tqqq_exec5m_v41_trpct_shock_trigger_variants_30m.json` → `backtests/out/tqqq_exec5m_v41_trpct_shock_trigger_30m_10y2y1y_mintr100_top8.json`
  - Result: no stability improvement vs CURRENT; not promotable.

- 2026-01-23: v42 TR% trigger × short scaling sweep:
  - `backtests/out/tqqq_exec5m_v42_trpct_shortscale_variants_30m.json` → `backtests/out/tqqq_exec5m_v42_trpct_shortscale_30m_10y2y1y_mintr100_top13.json`
  - Result: no stability improvement vs CURRENT; not promotable.

- 2026-01-23: v43 bias-off + ST(10,3) as “permission gate” (regime2) check:
  - Variants: `backtests/out/tqqq_exec5m_v43_bias_off_regime2_st10_3_variants_30m.json`
  - Full eval (includes losing configs): `backtests/out/tqqq_exec5m_v43_bias_off_regime2_st10_3_30m_10y2y1y_all_top6.json`
  - Key outcomes:
    - Turning shock off on the CURRENT family regresses stability: worst-window `roi/dd` drops `2.96 → 2.80` (candidate #02).
    - Disabling the bias gate and using ST(10,3) as regime2 (“permission”) loses in 1y and 2y (fails the champ constraints).
