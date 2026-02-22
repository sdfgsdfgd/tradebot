# SLV HF Research (High-Frequency Spot Track)

This file is the dedicated SLV high-frequency evolution track, separate from `backtests/slv/README.md`.
- `README.md` remains the low-frequency/live-parity representative champion track.
- This file tracks the high-turnover RATS line and its own HF crowns.

Canonical execution paths:
- Full comprehensive suite (includes migrated HF profiles): `python -m tradebot.backtest spot --axis combo_full --offline`
- HF timing corridor replay: `python -m tradebot.backtest spot --axis combo_full --combo-full-preset hf_timing_sniper --offline`

Current HF champion replay (v28 base short-mult retune sweet spot; 2Y-scoped):
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v28_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_shortmult0p028_20260222.json \
  --symbol SLV --bar-size "10 mins" --spot-exec-bar-size "5 mins" --offline --cache-dir db \
  --top 1 --min-trades 0 \
  --window 2024-02-14:2026-02-14 \
  --window 2025-02-14:2026-02-14
```

Historical evolution commands below are normalized to current wrappers:
- Spot sweeps/evolution: `python -m tradebot.backtest spot ...`
- Multiwindow kingmaker eval: `python -m tradebot.backtest spot_multitimeframe ...`

## Current Champions (stack)

### CURRENT (v28-exception-ddshock-lb10-on10-off5-depth1p25pp-streak1-shortmult0p028) — base short-mult retune sweet spot (2Y scope)
Current promoted crown on Feb-14 windows under the active HF scope contract (`<=2Y`; 10Y excluded due overnight bar reliability corruption concerns before `2023-07-24`).

**v28-exception kingmaker #01 [dd_lb10_on10_off5 + streak1 + depth-band short boost + base short throttle retune]**
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v28_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_shortmult0p028_20260222.json`
- Source eval: `backtests/slv/slv_hf_v31_short_mult_sweetspot_20260222.json`
- Base seed: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v27_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_20260222.json`
- Variant id: `r15_s02_sm_dn_fhard + dd_lb10_on10_off5 + shock_short_boost(streak>=1, dist_on<=1.25pp, require_regime_down, require_entry_down) + spot_short_risk_mult=0.028`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- 1y (`2025-02-14 -> 2026-02-14`): trades **694**, pnl **33,432.50**, dd **9,326.23**, pnl/dd **3.5848**
- 2y (`2024-02-14 -> 2026-02-14`): trades **1,244**, pnl **51,500.91**, dd **13,887.22**, pnl/dd **3.7085**
- Full working window (`2025-01-08 -> 2026-02-14`): pnl **39,266.04**, dd **9,848.74**, pnl/dd **3.9869**
- Jan-Feb 2026 slice (`2026-01-01 -> 2026-02-14`): pnl **19,164.12** (long **20,782.41**, short **-1,618.29**)
- Crash-leg slice (`2026-01-24 -> 2026-02-14`): pnl **7,498.10** (long **8,438.79**, short **-940.69**)
- Last-2m slice (`2025-12-14 -> 2026-02-14`): pnl **27,592.78** (long **29,230.68**, short **-1,637.90**)
- Jan-Feb short avg qty: **38.9**
- Trace probe (`2025-01-08 -> 2026-02-14`):
  - `entry_traces=747`
  - `shock_prearm_applied=1` (rare; remains effectively inert in this lane)
  - `shock_short_boost_applied=11` (boosted in early shock band; still **0** boosts in Jan-Feb crash window)

Unpromoted probes (artifact-only; informative only):

- artifact v28: **Long “recovering shock” boost off-gate** (attempted rebound-aware long sizing)
  - Artifact: `backtests/slv/slv_hf_v28_long_boost_offgate_micro_20260222.json`
  - New knobs added (defaults keep baseline behavior):
    - `shock_long_boost_max_dist_off_pp`
    - `shock_long_boost_require_regime_up`
    - `shock_long_boost_require_entry_up`
  - Result: **no PnL or DD change** across tested variants (long entries are frequently notional-cap bound in this lane; boosting risk dollars does not change final qty).

- artifact v29: **Widen short-boost depth band** (try to catch more of the crash with shock short boost)
  - Artifact: `backtests/slv/slv_hf_v29_short_boost_maxdist_widen_micro_20260222.json`
  - Result: widening `shock_short_boost_max_dist_on_pp` beyond the v27 sweet spot **hurts** 2Y economics (more deep-in-drawdown shorts; more rebound donation).

- artifact v30: **Short entry depth gate** (block shock-down shorts that start too deep into drawdown)
  - Artifact: `backtests/slv/slv_hf_v30_short_entry_depth_gate_micro_20260222.json`
  - New knob added (default `0` disables):
    - `shock_short_entry_max_dist_on_pp`
  - Best tested: `shock_short_entry_max_dist_on_pp=1.25`
    - 1y: trades **681**, pnl **33,075.26**, dd **9,519.90**, pnl/dd **3.4736** (baseline 1y pnl **32,739.53**, pnl/dd **3.4778**)
    - 2y: trades **1,233**, pnl **46,186.41**, dd **15,140.71**, pnl/dd **3.0505** (baseline 2y pnl **45,627.81**, pnl/dd **3.0124**)
  - Interpretation: this gate is directionally correct, but the **net uplift is ~1%** on 1Y/2Y, below the 3-5% improvement threshold.

Promotion contract check (exception policy; user approved):
- Hard throughput gate `>=700/year` on 1Y and 2Y: **FAIL** (`1Y=694`, `2Y=1244 -> 622/year`)
- Deterministic micro-matrix replay check: **PASS** (stable; spot_short_risk_mult micro-matrix replayed deterministically)
- 1Y/2Y dethrone vs prior crown `v27-exception`: **PASS**
  - `1Y pnl +692.97` and `pnl/dd +0.10698`
  - `2Y pnl +5,873.11` and `pnl/dd +0.69607`
- Stress-slice confirmation vs `v27-exception`: **PASS**
  - Jan-Feb and crash-leg remain long-dominant; short sleeve stays negative but is materially less destructive.

Immediate predecessor (now dethroned):
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v27_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_20260222.json`
- Source eval: `backtests/slv/slv_hf_v27_short_boost_maxdist_sweetspot_20260222.json`
- Variant id: `r15_s02_sm_dn_fhard + dd_lb10_on10_off5 + shock_short_boost(streak>=1, dist_on<=1.25pp, require_regime_down, require_entry_down)`

Earlier predecessor:
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v26_exception_ddshock_lb10_on10_off5_depth1pp_streak1_20260222.json`
- Variant id: `r15_s02_sm_dn_fhard + dd_lb10_on10_off5 + shock_short_boost(dist_on<=1.0pp)`

Earlier predecessor:
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v20_exception_ddshock_lb10_on10_off5_streak6_prearmtight_20260218.json`
- Variant id: `r15_s02_sm_dn_fhard + dd_lb10_on10_off5 + streak6 + prearm_tight`

Historical strict predecessor:
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v16_strict_1y2y_r15_20260217.json`
- Variant id: `r15_s02_sm_dn_fhard`

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

### v20.0 — pre-threshold drawdown-velocity + telemetry hardening dethrone (PROMOTED BY USER EXCEPTION)
Status: **DONE (promoted; deterministic 6-run pass + user-approved throughput exception)**

Promotion decision (record):
- Date: **2026-02-18**
- Scope rule: **exclude 10Y** from promotion decisions for this HF lane due overnight-bar corruption risk beyond 2Y.
- User-approved exception: crown can be promoted on 1Y/2Y economics even when strict throughput `>=700/year` is missed.
- Round objective: execute requested #1/#2/#3 exploration package while preserving v19 broad fitness:
  - `#1` adaptive resize/sizing path experiments
  - `#2` pre-threshold shock approach (distance-to-on + velocity)
  - `#3` confidence/liquidity boost path

Promotion winner:
- Variant: `prearm_tight_f1p5`
- Base profile: `r15_s02_sm_dn_fhard + dd_lb10_on10_off5 + streak6`
- New pre-threshold knobs:
  - `shock_prearm_dist_on_max_pp=1.0`
  - `shock_prearm_min_dist_on_vel_pp=0.25`
  - `shock_prearm_short_risk_mult_factor=1.5`
  - `shock_prearm_require_regime_down=true`
  - `shock_prearm_require_entry_down=true`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`

Measured dethrone deltas vs `v19-exception-dd_lb10_on10_off5-streak6`:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `697 vs 697` (`+0`)
  - pnl `32,649.19 vs 32,596.27` (`+52.92`)
  - pnl/dd `3.4682 vs 3.4626` (`+0.0056`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,252 vs 1,252` (`+0`)
  - pnl `45,611.02 vs 45,575.04` (`+35.98`)
  - pnl/dd `3.0050 vs 3.0026` (`+0.0024`)
- Full working window (`2025-01-08 -> 2026-02-14`):
  - pnl `38,252.49 vs 38,222.00` (`+30.50`)
  - dd% `9.93% vs 9.93%` (`+0.00pp`)

Expanded stress-slice confirmation (state-warm full-run slicing):
- Pre-2026 (`2025-01-08 -> 2025-12-31`): `19,783.10 vs 19,754.38` (`+28.72`)
- Last-2m (`2025-12-14 -> 2026-02-14`): `26,828.01 vs 26,827.70` (`+0.32`)
- Jan-Feb 2026 (`2026-01-01 -> 2026-02-14`): `18,469.39 vs 18,467.62` (`+1.77`)
- Crash leg (`2026-01-24 -> 2026-02-14`): `7,100.04 vs 7,099.02` (`+1.02`)
- Crash short sleeve: unchanged (`-1,281.24 vs -1,281.24`)

Determinism proof:
- Replays: **6**
- Variants: `baseline_v19`, `prearm_tight_f1p5`
- Windows: `1Y` and `2Y`
- Result: exact stability (all 6 tuples identical per variant/window).

Exploration package result (#1/#2/#3):
- `#1 resize`: broad degradation in this lane; not promoted.
- `#2 pre-threshold`: tight profile improved both 1Y and 2Y while staying neutral-to-positive on stress slices.
- `#3 liq boost`: triggered in multiple probes, but generally reduced broad fitness vs baseline in this 2Y contract.

Telemetry/logging hardening completed in this round:
- Signal/runtime now carries drawdown distance velocity (`shock_drawdown_dist_on_vel_pp`; shown as `ddv=...pp` in journal signal lines).
- Order journal now emits explicit pre-threshold and liquidity-boost decisions:
  - `shock_prearm`, `shock_prearm_reason`
  - `liq_boost`, `liq_mult`, `liq_score`, `liq_reason`
- Sizing trace now includes:
  - `shock_drawdown_dist_on_pct`, `shock_drawdown_dist_on_vel_pp`
  - cap-floor detail when liq boost is active.

Throughput contract note:
- Strict throughput rule `>=700/year` remains unmet (`1Y=697`, `2Y=626/year`), but promotion is approved under explicit user exception after deterministic/stress confirmation.

Artifacts:
- `backtests/slv/slv_hf_v19_track123_resize_prearm_liqboost_1y2y_20260218_v1.json`
- `backtests/slv/slv_hf_v19_track23_refine_noresize_1y2y_20260218_v1.json`
- `backtests/slv/slv_hf_v19_vs_prearmtight_6replay_1y2y_20260218_v1.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v20_exception_ddshock_lb10_on10_off5_streak6_prearmtight_20260218.json`

### v21.0 — slow-in/fast-out shock_ramp sizing controller + ramp telemetry (NOT PROMOTED)
Status: **DONE (implemented; not promoted)**

Objective:
- Add a maintainable, centralized “slow-in / fast-out” sizing ramp that can pre-arm into crash regimes using *distance-to-drawdown-threshold* + *velocity* + *directional slope stability*, while keeping v20 reverse-compatible (disabled unless enabled in filters).

Implementation (reverse-compatible; default OFF):
- Added new shock-ramp knobs (filters):
  - `shock_ramp_enable`
  - `shock_ramp_apply_to=down|up|both` (safe default `down`)
  - `shock_ramp_max_risk_mult`
  - `shock_ramp_max_cap_floor_frac` (optional cap-floor funnel)
  - `shock_ramp_min_slope_streak_bars`
  - `shock_ramp_min_slope_abs_pct`
- Added per-bar ramp snapshot: `shock_ramp` carried by signal runtime (live + backtest) and threaded into spot sizing.
- Added unambiguous live telemetry:
  - SIGNAL line: ramp tokens `r↓...` / `r↑...` appear when mult/intensity is non-trivial.
  - ORDER line: ramp decisions are explicit (`ramp=dir:phase`, `ramp_mult`, `ramp_i`, optional `ramp_floor`).

Key caution discovered (why not promoted):
- Enabling `spot_resize_mode=target` in this lane remains **catastrophic** in broad 1Y/2Y economics (very large drawdown blowups), even with ramp gating.
  - See: `backtests/slv/slv_hf_v20_shock_ramp_sweep_v1_20260221.json` (resize-on variants show negative pnl + extreme dd%).
- Entry-only ramp variants *did* increase crash short sizing, but **still reduced** broad 1Y/2Y pnl and pnl/dd vs v20.

Measured sweeps (<=2Y only; 10Y excluded):
- Resize-on exploration: `backtests/slv/slv_hf_v20_shock_ramp_sweep_v1_20260221.json`
- Entry-only exploration: `backtests/slv/slv_hf_v20_shock_ramp_sweep_v2_20260221.json`
- Down-only + dd-coupled ramp exploration: `backtests/slv/slv_hf_v20_shock_ramp_sweep_v3_downonly_20260221.json`

Bottom line:
- The telemetry + ramp controller are now available for future crash-regime R&D.
- Under the current HF promotion contract (1Y/2Y economics), no ramp configuration tested here dethroned v20.

### v22.0 — unbiased prearm latch + drawdown-depth gates micro-matrix (NOT PROMOTED)
Status: **DONE (investigation; not promoted)**

Objective:
- Test whether a *depth gate* can prevent prearm from waking up in mild pullbacks while still giving us earlier crash monetization.

Setup:
- Base: v20 king preset (`slv-hf-v20-exception-dd-lb10-on10-off5-streak6-prearm-tight`)
- Enable unbiased latch persistence (`shock_prearm_min_streak_bars=1`) and widen band for test visibility:
  - `shock_prearm_dist_on_max_pp=8.0`
  - `shock_prearm_min_dist_on_vel_pp=0.10`
  - `shock_prearm_min_dist_on_accel_pp=0.0`
- Sweep only the depth gate:
  - `shock_prearm_min_drawdown_pct ∈ {-6.0, -8.0, -9.0}` (note: `-10.0` is incompatible with prearm because prearm only runs while `shock=off`)

Artifact:
- `backtests/slv/slv_hf_v22_prearm_depth_gate_micro_20260221.json`

Result (deterministic replay):
- Baseline (v20): `2Y pnl=+45,611.0 pnl/dd=3.005 prearm_applied=1`
- Depth `-6`: `2Y pnl=+44,992.1 pnl/dd=2.908 prearm_applied=20` (more prearms, worse broad fitness)
- Depth `-8`: `2Y pnl=+45,480.7 pnl/dd=2.997 prearm_applied=3` (near-inert)
- Depth `-9`: `2Y pnl=+45,469.3 pnl/dd=2.988 prearm_applied=1` (inert)

Conclusion:
- Depth gating alone does not create positive EV prearm; when it increases prearm frequency (dd<=-6), it slightly degrades 2Y economics.

### v23.0 — prearm short-factor micro-matrix (NOT PROMOTED)
Status: **DONE (investigation; not promoted)**

Objective:
- If prearm frequency is higher (dd<=-6 latch), see whether increasing `shock_prearm_short_risk_mult_factor` creates real edge.

Artifact:
- `backtests/slv/slv_hf_v23_prearm_factor_micro_20260221.json`

Result (deterministic replay):
- `shock_prearm_short_risk_mult_factor=1.5`: `2Y pnl=+44,992.1 pnl/dd=2.908`
- `shock_prearm_short_risk_mult_factor=2.0`: `2Y pnl=+44,432.0 pnl/dd=2.810`
- `shock_prearm_short_risk_mult_factor=2.5`: `2Y pnl=+43,854.6 pnl/dd=2.715`

Conclusion:
- Increasing prearm sizing in this band is toxic for broad 2Y fitness (monotonic degradation).

### v24.0 — narrow-band prearm velocity tweak (NOT PROMOTED)
Status: **DONE (investigation; not promoted)**

Artifact:
- `backtests/slv/slv_hf_v24_prearm_narrow_band_micro_20260221.json`

Result:
- Lowering `shock_prearm_min_dist_on_vel_pp` from `0.25 -> 0.10` while keeping tight band `dist_on<=1.0pp` did **not** increase `prearm_applied` beyond `1`.

Conclusion:
- v20’s prearm tightness is not “just the velocity threshold” problem; it’s mostly an *alignment/opportunity rarity* problem in this lane.

### v25.0 — shock-on short sleeve EV vs drawdown depth (INSIGHT)
Status: **DONE (analysis; used to guide next regime ideas)**

Artifact:
- `backtests/slv/slv_hf_v25_short_ev_by_dd_depth_20260221.json`

Finding (v20 baseline, shock-on short trades; bucketed by `dd→on` in pp beyond activation):
- `dd→on ∈ [0,2)pp`: **positive** short-trade EV (avg pnl per trade > 0)
- `dd→on >= 2pp`: **negative** short-trade EV (avg pnl per trade < 0, gets worse deeper)

Implication:
- For this HF lane, “aggressive crash monetization” is not “keep boosting shorts deeper into drawdown”.
- It looks more like a **fast-in then stop boosting** (or even flip contrarian) once the drawdown is *already deep*.

### v26.0 — depth-aware shock short-boost dethrone (PROMOTED)
Status: **DONE (promoted; deterministic micro-matrix replay)**

Objective:
- Let the short boost actually fire (reduce streak gate), but only in the “early shock-on” band where shorts have positive EV.
- Avoid the classic failure mode: boosting deeper into drawdown, then getting snapbacked.

Promotion winner:
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v26_exception_ddshock_lb10_on10_off5_depth1pp_streak1_20260222.json`
- Base: v20 king (`slv-hf-v20-exception-dd-lb10-on10-off5-streak6-prearm-tight`)

New knobs (short boost only):
- `shock_short_boost_min_down_streak_bars=1` (was `6`; allows boost to fire)
- `shock_short_boost_max_dist_on_pp=1.0` (NEW; only boost when `0 <= dd→on <= 1.0pp`)
- `shock_short_boost_require_regime_down=true`
- `shock_short_boost_require_entry_down=true`

Deterministic dethrone vs v20 baseline:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `697 vs 697` (`+0`)
  - win `64.28% vs 63.99%` (`+0.29pp`)
  - pnl `32,733.34 vs 32,649.19` (`+84.15`)
  - pnl/dd `3.4772 vs 3.4682` (`+0.0089`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,252 vs 1,252` (`+0`)
  - win `66.85% vs 66.77%` (`+0.08pp`)
  - pnl `45,612.18 vs 45,611.02` (`+1.15`)
  - pnl/dd `3.0113 vs 3.0050` (`+0.0063`)

Behavioral proof (from micro-matrix):
- Boost applied **10** times total in the full working run, always in the intended early band:
  - boosted `dd→on` range: `+0.12pp .. +0.98pp` (avg `+0.60pp`)
- This implements the discovered shape: **fast-in boost**, then stop boosting once drawdown is deeper.

Artifact:
- `backtests/slv/slv_hf_v26_depth_aware_short_boost_micro_20260222.json`

### v27.0 — short-boost depth sweet spot (1.25pp) dethrone (PROMOTED)
Status: **DONE (promoted; deterministic micro-matrix replay)**

Objective:
- Find the least-invasive sweet spot for `shock_short_boost_max_dist_on_pp` without touching resizing.
- Preserve trade count (no behavior churn), but harvest any remaining edge in the early shock-on band.

Promotion winner:
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v27_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_20260222.json`
- Base: v26 king (`slv-hf-v26-exception-dd-lb10-on10-off5-depth1pp-streak1`)

Knob change (only):
- `shock_short_boost_max_dist_on_pp: 1.0 -> 1.25`

Deterministic dethrone vs v26 baseline:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `697 vs 697` (`+0`)
  - pnl `32,739.53 vs 32,733.34` (`+6.19`)
  - pnl/dd `3.4778 vs 3.4772` (`+0.00066`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,252 vs 1,252` (`+0`)
  - pnl `45,627.81 vs 45,612.18` (`+15.63`)
  - pnl/dd `3.0124 vs 3.0113` (`+0.00103`)

Notes:
- The short boost still does **not** trigger inside the Jan-Feb 2026 crash window in this lane (0 boosted entries there).
- So the improvement is a tiny broad stability edge, not “short sleeve crash monetization”.

Artifact:
- `backtests/slv/slv_hf_v27_short_boost_maxdist_sweetspot_20260222.json`

### v28.0 — base short throttle retune sweet spot (0.028) dethrone (PROMOTED)
Status: **DONE (promoted; deterministic micro-matrix replay)**

Objective:
- Reduce persistent short-side drag while preserving flip/neutralization behavior (no attempt here to “max shorts”).
- Hit the user’s improvement floor (**>=3%**) on at least one core metric without changing the shock detector or resizing logic.

Promotion winner:
- Preset file: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v28_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_shortmult0p028_20260222.json`
- Base: v27 king (`slv-hf-v27-exception-dd-lb10-on10-off5-depth1p25pp-streak1`)

Knob change (only):
- `spot_short_risk_mult: 0.0384 -> 0.028`

Deterministic dethrone vs v27 baseline:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `694 vs 697` (`-3`)
  - pnl `33,432.50 vs 32,739.53` (`+692.97`, `+2.12%`)
  - pnl/dd `3.5848 vs 3.4778` (`+0.10698`, `+3.08%`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,244 vs 1,252` (`-8`)
  - pnl `51,500.91 vs 45,627.81` (`+5,873.11`, `+12.87%`)
  - pnl/dd `3.7085 vs 3.0124` (`+0.69607`, `+23.11%`)

Notes:
- This does **not** “solve shorts” (short sleeve still negative), it simply stops the short sleeve from bleeding as much.
- Attempting to brute-force downturn monetization via larger shorts in this lane remains a timing problem, not a sizing problem.

Artifact:
- `backtests/slv/slv_hf_v31_short_mult_sweetspot_20260222.json`

### v19.0 — shock short-boost stability gate dethrone (PROMOTED BY USER EXCEPTION)
Status: **DONE (promoted; deterministic 6-run pass + user-approved throughput exception)**

Promotion decision (record):
- Date: **2026-02-18**
- Scope rule: **exclude 10Y** from promotion decisions for this HF lane due overnight-bar corruption risk beyond 2Y.
- User-approved exception: crown can be promoted on 1Y/2Y economics even when strict throughput `>=700/year` is missed.
- Promotion objective for this round: keep v18 drawdown-shock structure, but force short-boost to activate only after stable down-streak confirmation.

Promoted row:
- Base profile: `r15_s02_sm_dn_fhard`
- Shock regime base: `dd_lb10_on10_off5`
- New stability gate:
  - `shock_short_boost_min_down_streak_bars=6`
- Preserved shock knobs:
  - `shock_detector=daily_drawdown`
  - `shock_drawdown_lookback_days=10`
  - `shock_on_drawdown_pct=-10.0`, `shock_off_drawdown_pct=-5.0`
  - `shock_short_risk_mult_factor=2.0`
  - `shock_long_risk_mult_factor_down=0.1`
  - `shock_direction_source=regime`, `shock_direction_lookback=2`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`

Deterministic replay proof:
- Replays: **6**
- Variants: `v18_baseline`, `streak6`
- Windows: `1Y` (`2025-02-14..2026-02-14`) and `2Y` (`2024-02-14..2026-02-14`)
- Result: exact stability (`unique=1`) for all variant/window tuples.

Measured dethrone deltas vs `v18-exception-dd_lb10_on10_off5`:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `697 vs 698` (`-1`)
  - pnl `32,596.27 vs 31,077.08` (`+1,519.18`)
  - pnl/dd `3.4626 vs 3.2520` (`+0.2106`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,252 vs 1,252` (`+0`)
  - pnl `45,575.04 vs 43,716.00` (`+1,859.04`)
  - pnl/dd `3.0026 vs 2.8670` (`+0.1356`)
- Full working window (`2025-01-08 -> 2026-02-14`):
  - pnl `38,222.00 vs 36,697.18` (`+1,524.82`)
  - dd% `9.93% vs 10.08%` (`-0.15pp`)

Expanded stress-slice proof (state-warm full-run slicing):
- Pre-2026 (`2025-01-08 -> 2025-12-31`): `19,754.38 vs 19,606.91` (`+147.47`)
- Last-2m (`2025-12-14 -> 2026-02-14`): `26,827.70 vs 25,340.14` (`+1,487.56`)
- Jan-Feb 2026 (`2026-01-01 -> 2026-02-14`): `18,467.62 vs 17,090.27` (`+1,377.35`)
- Crash leg (`2026-01-24 -> 2026-02-14`): `7,099.02 vs 5,672.30` (`+1,426.72`)
- Crash short sleeve: `-1,281.24 vs -2,663.20` (`+1,381.96`)
- Jan-Feb short avg qty: `53.4 vs 84.2` (less short overextension while still improving crash-leg total and broad windows)

Mechanics note (why this worked):
- In this 2Y-scoped sample, observed `shock_dir_down_streak_bars` reached at most **4** during shock-short entries.
- With gate at `>=6`, `shock_short_boost_applied` fell from `69/69` (v18) to `0/70` (v19).
- Net effect: better global 1Y/2Y economics and drawdown efficiency, plus reduced crash short-sleeve damage.

Throughput contract note:
- Strict throughput rule `>=700/year` remains unmet (`1Y=697`, `2Y=626/year`), but promotion is approved under explicit user exception after deterministic/stress confirmation.

Artifacts:
- `backtests/slv/slv_hf_v18_stability_regimealign_1y2y_20260218_v2.json`
- `backtests/slv/slv_hf_v18_vs_streak6_6replay_1y2y_20260218_v2.json`
- `backtests/slv/slv_hf_v18_streak6_predictive_frontier_1y2y_20260218.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v19_exception_ddshock_lb10_on10_off5_streak6_20260218.json`

### v18.0 — earlier drawdown trigger dethrone (PROMOTED BY USER EXCEPTION)
Status: **DONE (promoted; deterministic 6-run pass + user-approved throughput exception)**

Promotion decision (record):
- Date: **2026-02-17**
- Scope rule: **exclude 10Y** from promotion decisions for this HF lane due overnight-bar corruption risk beyond 2Y.
- User-approved exception: crown can be promoted on 1Y/2Y economics even when strict throughput `>=700/year` is missed.
- Additional reliability gate added for this promotion: **6-run deterministic replay** on both 1Y and 2Y.

Promoted row:
- Base profile: `r15_s02_sm_dn_fhard`
- Shock regime add-on: `dd_lb10_on10_off5`
- Key shock knobs:
  - `shock_detector=daily_drawdown`
  - `shock_drawdown_lookback_days=10`
  - `shock_on_drawdown_pct=-10.0`, `shock_off_drawdown_pct=-5.0`
  - `shock_short_risk_mult_factor=2.0`
  - `shock_long_risk_mult_factor_down=0.1`
  - `shock_direction_source=regime`, `shock_direction_lookback=2`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`

Deterministic replay proof:
- Replays: **6**
- Variants: `v17_baseline`, `dd_lb10_on10_off5`
- Windows: `1Y` (`2025-02-14..2026-02-14`) and `2Y` (`2024-02-14..2026-02-14`)
- Result: exact stability (`distinct_rows=1`) for all variant/window pairs.

Measured dethrone deltas vs `v17-exception-dd_u2_l0p1`:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `698 vs 697` (`+1`)
  - pnl `31,079.10 vs 27,040.95` (`+4,038.16`)
  - pnl/dd `3.2523 vs 2.8297` (`+0.4226`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,252 vs 1,254` (`-2`)
  - pnl `43,716.80 vs 38,893.30` (`+4,823.51`)
  - pnl/dd `2.8669 vs 2.5040` (`+0.3629`)
- Full working window (`2025-01-08 -> 2026-02-14`):
  - pnl `36,697.18 vs 32,429.92` (`+4,267.26`)
  - dd% `10.08% vs 10.08%` (`+0.00pp`)

Expanded stress-slice proof (state-warm full-run slicing):
- Pre-2026 (`2025-01-08 -> 2025-12-31`): `19,606.91 vs 18,393.70` (`+1,213.21`)
- Last-2m (`2025-12-14 -> 2026-02-14`): `25,340.14 vs 21,167.59` (`+4,172.55`)
- Jan-Feb 2026 (`2026-01-01 -> 2026-02-14`): `17,090.27 vs 14,036.21` (`+3,054.06`)
- Crash leg (`2026-01-24 -> 2026-02-14`): `5,672.30 vs 2,799.30` (`+2,873.00`)
- Rebound tail (`2026-02-01 -> 2026-02-14`): `3,525.13 vs 673.44` (`+2,851.70`)
- Early 2025 (`2025-01-08 -> 2025-04-30`): unchanged (`-1,715.11`)
- Mid 2025 (`2025-05-01 -> 2025-10-31`): `+69.58` uplift
- Late 2025 (`2025-11-01 -> 2025-12-31`): `+1,143.63` uplift

Throughput contract note:
- Strict throughput rule `>=700/year` remains unmet (`1Y=698`, `2Y=626/year`), but promotion is approved under explicit user exception after deterministic/stress confirmation.

Artifacts:
- `backtests/slv/slv_hf_v18_probe_v17_vs_dd_lb10_on10_off5_6replay_20260217.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v18_exception_ddshock_lb10_on10_off5_20260217.json`

### v17.0 — daily-drawdown shock asymmetry dethrone (PROMOTED BY USER EXCEPTION)
Status: **DONE (promoted; user-approved throughput exception)**

Promotion decision (record):
- Date: **2026-02-17**
- Scope rule: **exclude 10Y** from promotion decisions for this HF lane due overnight-bar corruption risk beyond 2Y.
- User-approved exception: allow promotion even if `>=700/year` throughput gate fails, as long as crash-regime monetization clearly dethrones current crown on 1Y/2Y economics.

Promoted row:
- Base profile: `r15_s02_sm_dn_fhard`
- Shock regime add-on: `dd_u2_l0p1`
- Key shock knobs:
  - `shock_detector=daily_drawdown`
  - `shock_drawdown_lookback_days=20`
  - `shock_on_drawdown_pct=-12.0`, `shock_off_drawdown_pct=-5.0`
  - `shock_short_risk_mult_factor=2.0`
  - `shock_long_risk_mult_factor_down=0.1`
  - `shock_direction_source=regime`, `shock_direction_lookback=2`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`

Measured dethrone deltas vs `v16-strict` baseline:
- 1Y (`2025-02-14 -> 2026-02-14`):
  - trades `697 vs 697` (`+0`)
  - pnl `27,040.95 vs 24,148.14` (`+2,892.80`)
  - pnl/dd `2.8297 vs 1.9076` (`+0.9221`)
- 2Y (`2024-02-14 -> 2026-02-14`):
  - trades `1,254 vs 1,252` (`+2`)
  - pnl `38,893.30 vs 36,273.91` (`+2,619.39`)
  - pnl/dd `2.5040 vs 2.3898` (`+0.1141`)
- Full working window (`2025-01-08 -> 2026-02-14`):
  - pnl `32,429.92 vs 29,391.14` (`+3,038.77`)
  - dd% `10.08% vs 13.20%` (`-3.12pp`)

Crash-regime slice proof (state-warm full-run slicing):
- Pre-2026 (`2025-01-08 -> 2025-12-31`): `18,393.70 vs 18,535.82` (`-142.12`)
- Jan-Feb 2026 (`2026-01-01 -> 2026-02-14`): `14,036.21 vs 10,855.32` (`+3,180.89`)
- Crash leg (`2026-01-24 -> 2026-02-14`): `2,799.30 vs -393.51` (`+3,192.81`)
- Jan-Feb short avg qty: `80.9 vs 52.2` (aggression shifted into downturn window)

Throughput contract note:
- Strict throughput rule `>=700/year` was not met (`1Y=697`, `2Y=627/year`), and promotion proceeded under explicit user exception for crash-monetization R&D.

Reproduction spine (normalized from session runbook):
```bash
python -u - <<'PY'
import json
from pathlib import Path
from copy import deepcopy
from tradebot.backtest.config import load_config
from tradebot.backtest.engine import run_backtest

raw = json.loads(Path('backtests/slv/archive/champion_history_20260214/slv_hf_champions_v16_strict_1y2y_r15_20260217.json').read_text())
g = raw['groups'][0]
base_strategy = dict(g['entries'][0]['strategy'])
base_filters = {}
base_filters.update(base_strategy.get('filters') or {})
base_filters.update(dict(g.get('filters') or {}))

mods = {
  'shock_detector': 'daily_drawdown',
  'shock_drawdown_lookback_days': 20,
  'shock_on_drawdown_pct': -12.0,
  'shock_off_drawdown_pct': -5.0,
  'shock_short_risk_mult_factor': 2.0,
  'shock_long_risk_mult_factor_down': 0.1,
  'shock_long_risk_mult_factor': 1.0,
  'shock_direction_source': 'regime',
  'shock_direction_lookback': 2,
}
for k, v in mods.items():
    base_filters[k] = v
base_strategy['filters'] = base_filters

# run_backtest on:
# 2025-01-08..2026-02-14, 2025-02-14..2026-02-14, 2024-02-14..2026-02-14
PY
```

Artifacts:
- `backtests/slv/slv_hf_v17_exception_ddshock_2y_replay_20260217.json`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v17_exception_ddshock_20260217.json`
- Session checkpoints preserved in `/tmp/slv_downturn_segmented_probe.json` and `/tmp/slv_dd_u2_l0p1_final_metrics.json`

### v16.0 — strict 1Y/2Y dethrone after post-2023 reliability pivot (PROMOTED)
Status: **DONE (promoted; strict 1Y/2Y contract pass)**

Promotion decision (record):
- Date: **2026-02-17**
- New contract basis: 10Y gate retired for HF stability work due missing/flat overnight bars in IBKR history before `2023-07-24`.
- Promotion target: beat current crown on both `1Y` and `2Y` with `1y trades >= 700 OR higher than champion`.

Promoted row:
- `r15_s02_sm_dn_fhard`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- Core deltas vs prior crown `v15-exception`:
  - 1Y: `tr +0`, `pnl +16.09`, `pnl/dd +0.0014`
  - 2Y: `tr +0`, `pnl +28.78`, `pnl/dd +0.0018`

What made this run distinct:
- Champion-local micro-anneal around the proven v15 core (not a broad rewrite), with tight lattice on short risk and stop.
- Best row tightened `spot_short_risk_mult` from `0.0386` to `0.0384` while preserving the rest of the execution/entry architecture.
- Stress-overlay profile remained enabled (`shock_*`, `riskpanic_*`, `atr_compress`, `adaptive_hybrid_aggressive`) and yielded small but strict 1Y+2Y uplift.

Artifacts:
- `backtests/slv/slv_hf_r14_2y1y_dynamic_guard_candidates_20260217_v1.json`
- `backtests/slv/slv_hf_r14_2y1y_dynamic_guard_1y_ranked_20260217_v1.json`
- `backtests/slv/slv_hf_r15_champion_micro_direct_candidates_20260217_v1.json`
- `backtests/slv/slv_hf_r15_champion_micro_direct_1y_ranked_20260217_v1.json`
- `backtests/slv/slv_hf_r15_champion_micro_direct_top100_for_1y2y_20260217_v1.json`
- `backtests/slv/slv_hf_r15_champion_micro_direct_top100_1y2y_ranked_20260217_v1.json`
- `backtests/slv/slv_hf_r15_champion_micro_direct_top100_vs_v15_1y2y_strict_audit_20260217_v1.tsv`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v16_strict_1y2y_r15_20260217.json`

### v15.0 — dual-regime exception dethrone from r7 strike cycle (PROMOTED BY USER DIRECTION)
Status: **DONE (promoted; user-directed 10Y pnl waiver)**

Promotion decision (record):
- Date: **2026-02-16**
- User requested promotion of the strongest discovered row in the current cycle.
- Promotion target: maximize 1Y/2Y extraction with `1y trades >= 700`, while recording explicit 10Y pnl miss.

Promoted row:
- `r7t2_r5a_km03_h1_sm0_0386_sl0_0192_sh1_35_med_c2_bfirst_bal`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- Core deltas vs prior strict crown `v12-strict`:
  - 1Y: `tr -9`, `pnl +8,242.27`, `pnl/dd +0.6945`
  - 2Y: `tr -4`, `pnl +21,103.68`, `pnl/dd +1.1853`
  - 10Y: `tr -34`, `pnl -14,433.14`, `pnl/dd +0.0303` (pnl miss waived)

What made this run distinct:
- Pivoted from static corridor into stress-conditional dual-branch logic:
  - `spot_dual_branch_enabled=true`, `spot_dual_branch_priority=b_then_a`
  - branch sizing asymmetry: `spot_branch_a_size_mult=0.6`, `spot_branch_b_size_mult=1.2`
- RATS branch thresholds were explicitly separated by branch (rank/TR minima), keeping entries high-frequency while sharpening quality.
- Overlay + adaptive stack switched on with stress filters:
  - `spot_risk_overlay_policy=atr_compress`
  - `spot_resize_policy=adaptive_hybrid_aggressive`
  - stress filters centered around `shock_on_ratio=1.35`, `shock_off_ratio=1.25`, `shock_scale_detector=tr_ratio`, `shock_risk_scale_apply_to=both`.

Artifacts:
- `backtests/slv/slv_hf_regime_probe_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_regime_probe_10y_yearslice_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_r7_track1_overlay_repair_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_r7_track1_overlay_repair_1y_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_r7_track2_dual_regime_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_r7_track2_dual_regime_1y_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_r7_track3_slope_dynamic_candidates_20260216_v1.json`
- `backtests/slv/slv_hf_r7_track3_slope_dynamic_1y_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_r7_focus_contract_shortlist_10y2y1y_ranked_20260216_v1.json`
- `backtests/slv/slv_hf_r7_focus_contract_shortlist_vs_v12strict_audit_20260216_v1.tsv`
- `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v15_exception_r7_20260216.json`

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
