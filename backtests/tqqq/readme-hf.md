# TQQQ HF Research (High-Frequency Spot Track)

This file is the dedicated TQQQ high-frequency evolution track.
- `readme-lf.md` holds the low-frequency and broader historical TQQQ lineage.
- This file tracks the throughput-biased HF line and its own HF crowns.

Promotion contract (current):
- Promote based on `1Y` first, then reproduce on `2Y`.
- `10Y` is a later reality check (deferred for now).

Sovereignty contract (current phase):
- Promote based on coexistence across:
  - `2020` crash-transition / hot repair
  - `2022` persistent downtrend
  - `2025` persistent uptrend
- `v34` remains the raw `1Y/2Y` modern crown under the original contract.
- `v35` remains the bridge-only dethrone that first unified `2022/2025`.
- `v47` Router-On Composite Context Confidence is the active router-on crown for live deployment.
  - `v46` remains the prior router-on crown (lineage truth).
  - `v45` remains the prior router-on crown (lineage truth).
  - `v44` remains the first router-on dethrone (baseline truth).
  - `v43` remains the raw HF-host organism (router-off), preserved as the base sovereign.

Canonical execution paths:
- Spot sweeps/evolution: `python -m tradebot.backtest spot ...`
- Multiwindow kingmaker eval: `python -m tradebot.backtest spot_multitimeframe ...`

## Climate Router Log

- `regime_router` is now the top-level opt-in contract for host routing.
- Current central implementation lives in `tradebot/climate_router.py`.
- This is intentionally above the HF host line; it exists to stop forcing one host to trade every climate.

### Router v1
- First prototype insight only:
  - split the world into a bull-grind long host, a defensive host, and the HF host
  - proved the architecture, not kept as a durable implementation

### Router v2
- First validated annual prototype:
  - `bull_grind_low_vol -> buyhold`
  - `negative_transition_bear -> sma200`
  - `positive_high_stress_transition / negative_extreme_bear -> v43`
- Result:
  - worst-year chosen `pnl/dd ≈ 0.312`
- Status:
  - diagnostic only; later steward-host evaluation uncovered lookahead in the daily host prototypes
  - kept for lineage truth, not for promotion

### Router v3
- First steward-host dethrone:
  - replace `sma200` with `lf_defensive_long_v1`
  - `lf_defensive_long_v1 = ma50, entry_buffer=2%, exit_buffer=0`
- Result:
  - worst-year chosen `pnl/dd ≈ 0.713`
- Status:
  - diagnostic only; later recalibration superseded this host under corrected no-lookahead semantics

### Router v4
- Steward-host upgrade:
  - replace `lf_defensive_long_v1` with `lf_defensive_long_v2`
  - `lf_defensive_long_v2 = long-only daily drawdown kill-switch`
  - `on_dd=15%`, `off_dd=8%`
- Result:
  - worst-year chosen `pnl/dd ≈ 1.500`
  - chosen annual ladder:
    - `2017 8.059`
    - `2018 3.045`
    - `2019 4.346`
    - `2020 1.732`
    - `2021 2.724`
    - `2022 3.013`
    - `2023 5.774`
    - `2024 1.500`
    - `2025 6.951`
- Status:
  - invalid as a promotion result; later found to contain lookahead in the daily host evaluation path
  - the architecture survived, the quoted ladder did not

### Router v5 (2026-04-01) — Switch-Twitch Exorcism; canonical-cache stability crown
- Fix:
  - put the bull-sovereign `buyhold -> bull_ma200_v1` override behind confirm-days hysteresis
    - eliminates daily host strobing when features wobble around the threshold
  - warm up enough days for router readiness at the scoring start in per-year backtests
  - offline backtests no longer hard-fail when warmup reaches earlier than cache coverage
- Steward-host map:
  - `negative_transition_bear -> lf_defensive_long_v1` (ma50, entry_buffer=2%)
- New crash / slow-crash minimal gates:
  - `regime_router_crash_hf_slow_ret_max=-0.25` (prevents `crash_now` from always forcing `hf_host` during mild slow-crashes)
  - `damage_positive_lock` (route to `lf_defensive_long_v1` when slow-window is weak-positive but inefficient with high dd)
- Result (TQQQ, v43 preset, router ON; fast/slow/dwell=`63/84/10`, bull confirm=`1/7`):
  - `2017 pnl/dd=4.071` switches=0
  - `2018 pnl/dd=0.006` switches=7  (positive; no more strobe-spam)
  - `2019 pnl/dd=0.191` switches=4
  - `2020 pnl/dd=0.610` switches=10
  - `2021 pnl/dd=0.341` switches=4
  - `2022 pnl/dd=2.257` switches=6
  - `2023 pnl/dd=1.125` switches=7
  - `2024 pnl/dd=1.077` switches=8
  - `2025 pnl/dd=1.312` switches=7
  - Avg (2017–2025): **1.221**
  - Worst (2017–2025): **0.006**

### Router v6 (2026-04-01) — Oracle Episode Takeover; bad-window distilled
- New rule:
  - `episode_crash_takeover -> hf_host`
  - knobs:
    - `regime_router_hf_takeover_crash_ret_max=-0.08`
    - `regime_router_hf_takeover_crash_maxdd_min=0.20`
    - `regime_router_hf_takeover_crash_rv_max=0.55`
- Result (TQQQ, v43 preset, router ON; fast/slow/dwell=`63/84/10`, bull confirm=`1/7`, crash gate `-0.25`):
  - `2017 pnl/dd=4.071` switches=0
  - `2018 pnl/dd=0.006` switches=7  (positive; no more strobe-spam)
  - `2019 pnl/dd=0.255` switches=5
  - `2020 pnl/dd=0.610` switches=10
  - `2021 pnl/dd=0.326` switches=6
  - `2022 pnl/dd=2.257` switches=6
  - `2023 pnl/dd=1.125` switches=7
  - `2024 pnl/dd=1.051` switches=8
  - `2025 pnl/dd=1.620` switches=8
  - Avg (2017–2025): **1.258**
  - Worst (2017–2025): **0.006**

### Router v7 (2026-04-02) — Bull-Sovereign Ret Cap; stop nuking 2023 to fix 2018
- Fix:
  - bull-sovereign override now requires `slow_ret <= 0.40` (stops bull-ma200 from eating strong bull expansions like `2023`)
  - removed the `slow_eff <= 0.08` constraint (so 2018’s Feb/March fragility actually trips bull-ma200)
- Status:
  - policy remains interpretable and knob-light (no new strategy knobs; one rule change)

## Current Champions (stack)

### CURRENT (v49) — Router-On Composite Context Confidence; episode-crash goes flat (live preset)

- Delta vs `v48`:
  - in `tradebot/climate_router.py`:
    - `episode_crash_takeover` now routes to `lf_defensive_long_v2` (drawdown-kill) when slow is not deeply negative
    - this prevents MA200 “false defense” during correction episodes (April-2024 was the poster child)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v49_routerOnCompositeContextConfidence_20260404.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.219**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **298**, pnl **24,820.2**, dd **14,639.7**, pnl/dd **1.695**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **20**, pnl **7,412.6**, dd **19,000.7**, pnl/dd **0.390**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **390**, pnl **13,723.8**, dd **7,922.3**, pnl/dd **1.732**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **58**, pnl **32,043.5**, dd **22,955.5**, pnl/dd **1.396**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **20**, pnl **4,673.5**, dd **21,364.3**, pnl/dd **0.219**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **127**, pnl **13,236.9**, dd **12,202.1**, pnl/dd **1.085**
- Underlying HF host base filters: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`

### PREVIOUS (v48) — Router-On Composite Context Confidence; 2016-positive reality fix (live preset)

- Delta vs `v47`:
  - backtest semantic fix (no more phantom stop exits in host-managed lanes):
    - persist `regime_router_host_managed` / `regime_router_ready` across exec bars (multi-res backtests)
  - loosen crash-prearm choke (stops 2016 from getting strangled by false “crash prearm”):
    - `regime2_crash_prearm_shock_dir_ret_sum_pct_max=-10.0` (was `-5.0`)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v48_routerOnCompositeContextConfidence_20260404.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.116**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **298**, pnl **24,820.2**, dd **14,639.7**, pnl/dd **1.695**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **21**, pnl **11,321.6**, dd **19,000.7**, pnl/dd **0.596**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **390**, pnl **13,723.8**, dd **7,922.3**, pnl/dd **1.732**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **58**, pnl **32,043.5**, dd **22,955.5**, pnl/dd **1.396**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **20**, pnl **2,337.1**, dd **20,083.0**, pnl/dd **0.116**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **127**, pnl **13,236.9**, dd **12,202.1**, pnl/dd **1.085**
- Underlying HF host base filters: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`

### PREVIOUS (v47) — Router-On Composite Context Confidence; overextended-bull HF takeover (live preset)

- Delta vs `v46`:
  - add overextended-bull escape hatch:
    - if the router would pick `bull_grind_low_vol -> buyhold` but both fast+slow windows are overextended,
      route to `hf_host` instead (prevents bull-only long forcing in mean-reversion traps)
  - new knobs:
    - `regime_router_bull_overextended_hf_fast_ret_min=0.40`
    - `regime_router_bull_overextended_hf_slow_ret_min=0.60`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v47_routerOnCompositeContextConfidence_20260403_2.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.823**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **364**, pnl **27,803.3**, dd **12,044.8**, pnl/dd **2.308**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **81**, pnl **14,808.1**, dd **10,497.9**, pnl/dd **1.411**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **400**, pnl **14,812.0**, dd **7,999.3**, pnl/dd **1.852**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **155**, pnl **20,516.7**, dd **16,874.6**, pnl/dd **1.216**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **111**, pnl **14,112.6**, dd **17,151.1**, pnl/dd **0.823**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **224**, pnl **17,683.1**, dd **9,450.9**, pnl/dd **1.871**
- Underlying HF host base filters: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`

### PREVIOUS (v46) — Router-On Composite Context Confidence; universal-floor lift via trend-down long gate (live preset)

- Delta vs `v45`:
  - add `regime4_trenddown_block_longs=True` (blocks new long entries when `regime4_state=trend_down` and `hard_dir=down`)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v46_routerOnCompositeContextConfidence_20260403.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.659**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **313**, pnl **10,176.3**, dd **15,448.7**, pnl/dd **0.659**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **81**, pnl **14,808.1**, dd **10,497.9**, pnl/dd **1.411**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **400**, pnl **14,812.0**, dd **7,999.3**, pnl/dd **1.852**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **137**, pnl **20,388.9**, dd **14,567.0**, pnl/dd **1.400**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **111**, pnl **14,112.6**, dd **17,151.1**, pnl/dd **0.823**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **224**, pnl **17,683.1**, dd **9,450.9**, pnl/dd **1.871**
- Underlying HF host base filters: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`

### PREVIOUS (v45) — Router-On Composite Context Confidence; 2018 ≥ 0.5 floor crown (live preset)

- Delta vs `v44`:
  - Router v7 bull-sovereign ret-cap (keeps 2018 protection but stops 2023 cannibalization)
  - HF-host loss-cut mode: `flip_exit_only_if_profit=False` + `flip_exit_min_hold_bars=30`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v45_routerOnCompositeContextConfidence_20260402.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.398**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **328**, pnl **8,880.4**, dd **16,432.7**, pnl/dd **0.540**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **106**, pnl **6,132.0**, dd **15,422.4**, pnl/dd **0.398**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **406**, pnl **13,767.5**, dd **7,872.5**, pnl/dd **1.749**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **154**, pnl **19,101.0**, dd **16,615.1**, pnl/dd **1.150**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **152**, pnl **13,983.8**, dd **15,340.3**, pnl/dd **0.912**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **244**, pnl **15,657.7**, dd **11,611.3**, pnl/dd **1.348**
- Underlying HF host base filters: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`

### PREVIOUS (v44) — Router-On Composite Context Confidence; first router-on dethrone (baseline)

- Delta vs `v43`:
  - enable `regime_router` (Router v6) for stable host routing + episode-takeover defense
  - underlying HF host organism remains `v43` (no changes to the champion filters/knobs)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v44_routerOnCompositeContextConfidence_20260401.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.326**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **327**, pnl **9,677.3**, dd **15,854.8**, pnl/dd **0.610**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **111**, pnl **5,263.7**, dd **16,164.1**, pnl/dd **0.326**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **438**, pnl **22,798.7**, dd **10,100.6**, pnl/dd **2.257**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **156**, pnl **18,625.6**, dd **16,549.3**, pnl/dd **1.125**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **151**, pnl **16,298.2**, dd **15,508.9**, pnl/dd **1.051**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **248**, pnl **18,707.6**, dd **10,802.7**, pnl/dd **1.732**
- Underlying HF host (router-off): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`

### PREVIOUS (v43) — Composite Context Confidence; canonical-cache reigning organism over the `2020..2025` universal-floor contract

- Delta vs `v42`:
  - Keep the all-green `v42` organism intact.
  - Add one centralized thing only:
    - `spot_entry_context_confidence_mode="continuation_v1"`
  - That mode is fed by the actual entry decision context:
    - `regime4_state`
    - `branch`
    - `shock_dir`
    - `hard_dir`
    - `release_age_bars`
  - Real new truth:
    - the post-`v42` scalar guard-knob path was dead
    - the next edge was a composite ambiguity score, not another threshold
    - the winning motifs were:
      - `trend_up_clean` long continuation with `shock_dir=down`, `hard_dir=up`, `age 500..1599`
      - `trend_down / branch-b` long continuation with `shock_dir=down`, `hard_dir=up`, `age 500..1599`
      - `transition_up_hot / branch-a` long continuation with `shock_dir=up`, `hard_dir=up`, `age 1600+`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.449**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **488**, pnl **14,711.6**, dd **8,492.9**, pnl/dd **1.732**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **430**, pnl **4,047.9**, dd **6,987.0**, pnl/dd **0.579**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **506**, pnl **32,942.8**, dd **10,933.1**, pnl/dd **3.013**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **470**, pnl **5,139.3**, dd **11,447.9**, pnl/dd **0.449**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **413**, pnl **5,942.0**, dd **11,813.6**, pnl/dd **0.503**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **453**, pnl **30,401.3**, dd **5,013.3**, pnl/dd **6.064**
- Distinct anatomy:
  - `2023`: weak-year floor lifted materially, from `0.046` to `0.449`
  - `2020`, `2021`, `2024`, and `2025`: all improved
  - `2022`: only a tiny tax versus `v42`
  - this is not a threshold hack; it is the first post-`v42` composite-confidence dethrone
- Adjacent crowns that still define the truthful-cache ridge:
  - Previous crown, `v42` All-Green Continuation Confidence:
    - Result across `2020..2025`: floor **0.046**, median **0.779**, positive years **6/6**
    - Distinct truth: the first all-green line in the lineage; branch-`b` continuation confidence plus branch-`a` fresh-transition skepticism
  - Previous crown, `v41` Universal Floor Continuation Confidence:
    - Result across `2020..2025`: floor **-0.039**, median **0.801**, positive years **5/6**
    - Distinct truth: one centralized branch-`b` continuation-confidence rule nearly neutralized the last bad year
  - Universal-floor sibling:
    - branch-`b` low-ATR recovery shelf only
    - Result across `2020..2025`: floor **-0.278**, median **0.589**, positive years **5/6**
    - Preset file: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v40_branchBLowAtrRecoveryUniversalFloorSibling_20260317.json`
    - Distinct truth: real floor lift, but not enough by itself
  - Hard-shelf one-knob sibling:
    - only change `regime2_bear_hard_supertrend_multiplier = 2.30`
    - Result: `2020 pnl/dd = 0.928`, `2022 pnl/dd = 3.031`, `2025 pnl/dd = 6.953`
    - Distinct truth: the hard-regime shelf itself was the real precondition for the universal-floor jump

## Evolutionary Log

### v49 (2026-04-04) — Router-On Composite Context Confidence; episode-crash goes flat

- Delta vs `v48`:
  - router rule in `tradebot/climate_router.py`:
    - `episode_crash_takeover` now routes to `lf_defensive_long_v2` (instead of `bull_ma200_v1`)
      when the slow window is not deeply negative
  - add investigation tooling:
    - `tradebot/backtest/tools/regime_router_flip_autopsy.py`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v49_routerOnCompositeContextConfidence_20260404.json`

### v48 (2026-04-04) — Router-On Composite Context Confidence; 2016-positive reality fix

- Delta vs `v47`:
  - fix multi-resolution backtest semantics in `tradebot/backtest/engine.py`:
    - persist the router’s `host_managed` metadata across exec bars (prevents phantom `stop_loss_pct` exits)
  - update the preset knob:
    - `regime2_crash_prearm_shock_dir_ret_sum_pct_max=-10.0` (was `-5.0`)
  - add a regression test:
    - `tests/test_regime_router_exec_bar_state_persistence.py`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v48_routerOnCompositeContextConfidence_20260404.json`

### v47 (2026-04-03) — Router-On Composite Context Confidence; overextended-bull HF takeover

- Delta vs `v46`:
  - add the overextended-bull escape hatch inside `tradebot/climate_router.py`:
    - when the router would force `bull_grind_low_vol -> buyhold` but both windows are overextended,
      route to `hf_host` instead
  - knobs:
    - `regime_router_bull_overextended_hf_fast_ret_min=0.40`
    - `regime_router_bull_overextended_hf_slow_ret_min=0.60`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v47_routerOnCompositeContextConfidence_20260403_2.json`
- Router year table (2017–2025): avg **1.495**, worst **0.774**, and `2020 pnl/dd=2.308`

### v46 (2026-04-03) — Router-On Composite Context Confidence; trend-down long-gate universal-floor lift

- Delta vs `v45`:
  - add `regime4_trenddown_block_longs=True`
    - blocks new long entries when `regime4_state=trend_down` and `hard_dir=down`
    - used to stop weak-year bleed clusters without harming the router itself
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v46_routerOnCompositeContextConfidence_20260403.json`
- Router year table (2017–2025): avg **1.332**, worst **0.659**, and `2018 pnl/dd=0.774`

### v45 (2026-04-02) — Router-On Composite Context Confidence; 2018 ≥ 0.5 floor crown

- Delta vs `v44`:
  - Router v7 bull-sovereign ret cap: preserve 2018 defense without consuming 2023
  - `flip_exit_only_if_profit=False` + `flip_exit_min_hold_bars=30` (loss-cut without strobe-churn)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v45_routerOnCompositeContextConfidence_20260402.json`
- Router year table (2017–2025): avg **1.200**, worst **0.193**, and `2018 pnl/dd=0.538`

### v44 (2026-04-01) — Router-On Composite Context Confidence; first router-on dethrone (stability contract)

- Delta vs `v43`:
  - enable `regime_router` (switch-twitch hysteresis + oracle episode takeover)
  - per-year router backtests warm up correctly; offline cache gaps no longer hard-fail
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v44_routerOnCompositeContextConfidence_20260401.json`
- Underlying HF host (router-off): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json`
- Router year table (2017–2025): avg **1.258**, worst **0.006**

### v42 (2026-03-18) — All-Green Continuation Confidence; first all-green organism over the `2020..2025` universal-floor contract

- Delta vs `v41` and the earlier truthful-cache crowns:
  - Keep the whole `v41` organism intact:
    - corrected `v38` host still owns the broader sovereignty map
    - split-age `trend_up_clean / branch-b` repair still lives
    - `trend_down / branch-b` purifier still lives
    - the hard-regime shelf still lives
    - branch-`b` continuation confidence still lives
  - Add only one thing:
    - a branch-`a` transition continuation-confidence rule for:
      - `transition_up_hot`
      - `shock_dir = up`
      - `hard_dir = down`
      - `release_age <= 4`
      - `ATR 0.2 .. 0.35`
      - `ddv <= 0.0`
  - Real new truth:
    - the last bad year was not solved by more regime splitting
    - it was solved by refusing fresh low-ATR branch-`a` transition optimism when the hard regime was still down
    - this was the first all-green line in the lineage
- Preset file: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v42_allGreenContinuationConfidence_20260318.json`
- Universal-floor floor (min `2020..2025` pnl/dd): **0.046**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **497**, pnl **10,298.5**, dd **9,621.8**, pnl/dd **1.070**
- `2021` (`2021-01-01 -> 2022-01-01`): trades **438**, pnl **3,272.4**, dd **6,805.0**, pnl/dd **0.481**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **514**, pnl **33,018.3**, dd **10,912.0**, pnl/dd **3.026**
- `2023` (`2023-01-01 -> 2024-01-01`): trades **478**, pnl **639.8**, dd **13,789.1**, pnl/dd **0.046**
- `2024` (`2024-01-01 -> 2025-01-01`): trades **423**, pnl **5,745.6**, dd **11,789.4**, pnl/dd **0.487**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **460**, pnl **27,974.6**, dd **4,951.4**, pnl/dd **5.650**
- Distinct anatomy:
  - `2020`: stayed above `1.0` even after the weak-year fix
  - `2021`: flipped materially positive; the old floor year was no longer red
  - `2023`: became the thinnest year in the contract
  - `2022` and `2025`: stayed strong enough that this was not just a floor hack

### PREVIOUS MULTI-REGIME DETHRONE (v37-Heat-Lip Sovereignty) — dethroned v36 for the `2020/2022/2025` sovereignty contract

- Delta vs `v36` (needle-thread):
  - Keep the whole v36 sovereignty seam intact:
    - Bridge host remains the default sovereign
    - alternate host still owns `crash_down + transition_up_hot`
    - keep the living crash / repair / pocket scaffold unchanged
  - Discover the real next truth:
    - the broad heat family was not wrong, it was too low
    - the real signal is the upper-lip transition heat seam around `0.79 - 0.80`
    - `0.81+` already loses too much of the `2020` magic
  - Final heat-lip addition:
    - `regime2_transition_hot_shock_atr_pct_min=0.8`
  - Why it dethroned:
    - `2020` gain came mostly from cutting rebound poison, especially `branch-b long` drag
    - `2022` gain came from killing fake rebound `branch-b long` damage inside the persistent bear
    - `2025` stayed strong because the sacred `branch-a long` kingdom remained mostly intact; the tax was mainly clipped `branch-b` continuation
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v37_heatLipSovereignty_20260315.json`
- Sovereignty floor (min `2020/2022/2025` pnl/dd): **1.994**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **492**, pnl **10,748.3**, dd **5,390.3**, pnl/dd **1.994**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **536**, pnl **28,147.8**, dd **9,183.5**, pnl/dd **3.065**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **478**, pnl **25,907.4**, dd **4,983.2**, pnl/dd **5.199**
- Distinct anatomy:
  - `2020`: still not a raw crash god, but rebound poison gets cut materially; `branch-b long` drag collapses from about `-6.8k` to `-3.2k`, which is what pulls the year to the edge of `2.0`
  - `2022`: the heat lip acts like a fake-rebound long killer in a real bear; `branch-b long` poison collapses from about `-9.3k` to `-3.7k` while `branch-a short` stays broad and real
  - `2025`: the modern host survives; `branch-a long` only gives back modestly, and most of the tax comes from trimming `branch-b long` continuation
- Adjacent crowns that still define the ridge:
  - Previous king, v36 Balanced Combined Sovereignty:
    - `regime2_clean_host_takeover_state="crash_or_transition_up_hot"`
    - `regime2_clean_host_supertrend_multiplier=2.32`
    - `regime2_clean_host_bear_supertrend_multiplier=1.45`
    - `regime2_clean_host_bear_hard_supertrend_multiplier=2.18`
    - Result: `2020 pnl/dd = 1.459`, `2022 pnl/dd = 2.828`, `2025 pnl/dd = 5.468`
  - Soft heat-lip sibling:
    - add `regime2_transition_hot_shock_atr_pct_min=0.78`
    - Result: `2020 pnl/dd = 1.866`, `2022 pnl/dd = 3.052`, `2025 pnl/dd = 5.228`
    - Distinct truth: nearly the same beast, a bit less hostile, a bit more modern grace
  - `0.82+` heat siblings:
    - `regime2_transition_hot_shock_atr_pct_min=0.82` and above
    - Result example: `2020 pnl/dd = 1.423`, `2022 pnl/dd = 3.067`, `2025 pnl/dd = 5.392`
    - Distinct truth: once the lip gets too high, `2020` drops off a cliff; the useful edge really does live near `0.79 - 0.80`

### EARLIER MULTI-REGIME DETHRONE (v36-Balanced Combined Sovereignty) — dethroned v35 for the `2020/2022/2025` sovereignty contract

- Delta vs `v35` (needle-thread):
  - Keep the Bridge Crown host as the default sovereign instead of throwing away the modern kingdom.
  - Keep the pocket-law / crash-repair scaffolding that made `2020 > 1` real:
    - `regime2_crash_atr_pct_min=0.9`
    - `regime2_crash_block_longs=true`
    - `regime2_repair_block_branch_b_longs=true`
    - up-corridor `branch-a long` pocket map:
      - mid ATR band `0.35-0.5`
      - extreme ATR `>= 0.75`
      - fresh hard-release age `<= 1`
      - stale hard-release age `>= 17`
  - Real breakthrough:
    - the mutant never truly beat the bridge in the raw COVID crash
    - its edge lived in the violent `transition_up_hot` corridor
    - the dethrone came from letting the alternate host own `crash_down + transition_up_hot`, not from generic crash detection
  - Combined sovereignty saddle that held:
    - `regime2_clean_host_enable=true`
    - `regime2_clean_host_takeover_state="crash_or_transition_up_hot"`
    - `regime2_clean_host_supertrend_multiplier=2.32`
    - `regime2_clean_host_bear_supertrend_multiplier=1.45`
    - `regime2_clean_host_bear_hard_supertrend_multiplier=2.18`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v36_balancedCombinedSovereignty_20260315.json`
- Sovereignty floor (min `2020/2022/2025` pnl/dd): **1.459**
- `2020` (`2020-01-01 -> 2021-01-01`): trades **498**, pnl **8,970.5**, dd **6,149.6**, pnl/dd **1.459**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **535**, pnl **27,671.4**, dd **9,783.3**, pnl/dd **2.828**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **478**, pnl **27,408.8**, dd **5,012.5**, pnl/dd **5.468**

### PREVIOUS BIDIRECTIONAL DETHRONE (v35-Bridge Crown Throne) — dethroned v34 for the `2022/2025` bridge contract

- Delta vs raw `v34` (needle-thread):
  - Keep `v34` as the host instead of replacing the organism.
  - Add a two-strength bear architecture:
    - soft wakeup: `regime2_mode=supertrend`, `regime2_bar_size="30 mins"`, `regime2_supertrend_multiplier=2.5`
    - hard takeover confirm: `regime2_bear_hard_mode=supertrend`, `regime2_bear_hard_bar_size="4 hours"`, `regime2_bear_hard_supertrend_multiplier=2.5`
  - Semantic fix that unlocked the bridge: soft bear no longer clears host entries when hard bear does not explicitly fire.
  - Hard-bear takeover sweet spot: `regime2_bear_takeover_mode=always`
  - Bear primary sweet spot: `regime2_bear_supertrend_multiplier=1.5`
  - Keep rebound participation alive: `regime2_bear_allow_long_recovery=true`
  - Modern-protection nerve: `regime2_soft_bear_branch_a_slope_vel_slow_min_pct=0.00002`
  - Keep soft bear slow-med hugging the base instead of over-tightening it: `regime2_soft_bear_branch_a_slope_med_slow_min_pct=0.00008`
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v35_bridgeCrownThrone_bidirectional_20260315.json`
- Contract floor (min `2022/2025` pnl/dd): **2.327**
- `2022` (`2022-01-01 -> 2023-01-01`): trades **639**, pnl **22,816.1**, dd **9,803.4**, pnl/dd **2.327**
- `2025` (`2025-01-01 -> 2026-01-19`): trades **513**, pnl **31,666.8**, dd **5,972.8**, pnl/dd **5.302**
- Distinct anatomy:
  - `2022`: real downside alpha is alive, led by `branch-a short pnl = +18.1k` while `branch-a long` still contributes `+12.6k`
  - `2025`: the host long kingdom is alive again, led by `branch-a long pnl = +17.5k`, with meaningful help from `branch-b short pnl = +6.0k`
- Adjacent crowns that still define the ridge:
  - Shared base for the ridge:
    - `spot_short_risk_mult=0.03`
    - `ratsv_branch_a_slope_med_slow_min_pct=0.00008`
    - `riskoff_long_risk_mult_factor=0.25`
    - `riskpanic_long_risk_mult_factor=0.05`
    - `regime2_mode=supertrend`, `regime2_apply_to=off`, `regime2_bar_size="30 mins"`
    - `regime2_supertrend_atr_period=10`, `regime2_supertrend_source="hl2"`
    - `regime2_bear_entry_mode=supertrend`
    - `regime2_bear_supertrend_atr_period=10`, `regime2_bear_supertrend_source="hl2"`
    - `regime2_bear_hard_mode=supertrend`, `regime2_bear_hard_bar_size="4 hours"`
    - `regime2_bear_hard_supertrend_atr_period=10`, `regime2_bear_hard_supertrend_source="hl2"`
  - Modern Crown:
    - `regime2_bear_takeover_mode=riskpanic`
    - `regime2_supertrend_multiplier=2.5`
    - `regime2_bear_hard_supertrend_multiplier=2.5`
    - `regime2_bear_supertrend_multiplier=1.5`
    - `regime2_bear_allow_long_recovery=false`
    - `regime2_soft_bear_branch_a_slope_med_slow_min_pct=0.00012`
    - `regime2_soft_bear_branch_a_slope_vel_slow_min_pct=null`
    - Result: `2022 pnl/dd = 1.478`, `2025 pnl/dd = 7.308`
  - Bear Crown:
    - `regime2_bear_takeover_mode=always`
    - `regime2_supertrend_multiplier=2.5`
    - `regime2_bear_hard_supertrend_multiplier=2.0`
    - `regime2_bear_supertrend_multiplier=1.75`
    - `regime2_bear_allow_long_recovery=true`
    - `regime2_soft_bear_branch_a_slope_med_slow_min_pct=0.00012`
    - `regime2_soft_bear_branch_a_slope_vel_slow_min_pct=null`
    - Result: `2022 pnl/dd = 2.844`, `2025 pnl/dd = 2.844`
  - Bear-leaning bridge sibling:
    - `regime2_bear_takeover_mode=always`
    - `regime2_supertrend_multiplier=2.5`
    - `regime2_bear_hard_supertrend_multiplier=2.5`
    - `regime2_bear_supertrend_multiplier=1.6`
    - `regime2_bear_allow_long_recovery=true`
    - `regime2_soft_bear_branch_a_slope_med_slow_min_pct=0.00008`
    - `regime2_soft_bear_branch_a_slope_vel_slow_min_pct=0.00002`
    - Result: `2022 pnl/dd = 2.455`, `2025 pnl/dd = 5.076`

### RAW MODERN CROWN (v34-km01-shockMin(1.25)-shockDownLong(0.10)-dualBranch(minSlope=0.00075 b_mult=0.70 priority=a_then_b)-riskpanic(tr_med>=5.0 neg_gap_ratio>=0.5 long_factor=0.10 short_factor=1.5)-linear(tr_delta_max=0.5)-overlay(atr_compress+shock_dir lb=78 floor=0.65 boost=1.0 hi=1.4 min=0.30)-cd4-ddBoost(lb=20 on=-20 off=-15 max_dist=15 factor=18 vel_gate=0.2)-dynGuard(atr_vel_direct min_mult=1.0)-shortEntryBand(max_dist=20)) — dethroned v33 (shock detector armed; shock-down long downshift; 1Y/2Y promotion)

- Delta vs v33 (needle-thread):
  - `shock_min_atr_pct: 7.0 -> 1.25` (shock detector no longer inert on `signal=5 mins` bars)
  - `shock_long_risk_mult_factor_down: 1.0 -> 0.10` (downshift long sizing only when `shock=on` and `shock_dir=down`)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v34_km01_shockMin1p25_downLong0p10_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_rpLong0p10_ddFac18_20260305.json`
- Dojo replay (warmup+focus tape):
  - Warmup window: `2026-02-10 -> 2026-02-28`
  - Focus window: `2026-02-23 -> 2026-02-27`
  - Replay config: `backtests/tqqq/replays/tqqq_hf_v34_km01_shockMin1p25_downLong0p10_dualbranch_slope0p00075_bmult0p70_panicNeg0p50_rpLong0p10_ddFac18_dojo_warmup_20260210_20260228.json`
- Stability floor (min `1Y/2Y` pnl/dd): **7.901**
- 1Y (`2025-01-01 -> 2026-01-19`): trades **556**, pnl **52,189.5**, dd **5,579.7**, pnl/dd **9.353**
- 2Y (`2024-01-01 -> 2026-01-19`): trades **1,095**, pnl **75,995.6**, dd **9,618.5**, pnl/dd **7.901**

### v33 (2026-03-02) — dethroned by v34 (shock detector was effectively inert at `shock_min_atr_pct=7.0` for 5m bars)

- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v33_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_rpLong0p10_ddFac18_20260302.json`
- Dojo replay (warmup+focus tape):
  - Warmup window: `2026-02-10 -> 2026-02-28`
  - Focus window: `2026-02-23 -> 2026-02-27`
  - Replay config: `backtests/tqqq/replays/tqqq_hf_v33_km01_dualbranch_slope0p00075_bmult0p70_panicNeg0p50_rpLong0p10_ddFac18_dojo_warmup_20260210_20260228.json`
- Stability floor (min `1Y/2Y` pnl/dd): **7.732**
- 1Y (`2025-01-01 -> 2026-01-19`): trades **556**, pnl **50,902.0**, dd **5,531.8**, pnl/dd **9.202**
- 2Y (`2024-01-01 -> 2026-01-19`): trades **1,095**, pnl **74,367.0**, dd **9,618.5**, pnl/dd **7.732**

### v32 (2026-03-02) — dethroned by v33 (riskpanic long downshift)

- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v32_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_ddFac18_20260301.json`
- Dojo replay (warmup+focus tape):
  - Warmup window: `2026-02-10 -> 2026-02-28` (so TR5/gap overlays have state)
  - Focus window: `2026-02-23 -> 2026-02-27` (the newest choppy-week tape)
  - Replay config: `backtests/tqqq/replays/tqqq_hf_v32_km01_dualbranch_slope0p00075_bmult0p70_panicNeg0p50_ddFac18_dojo_warmup_20260210_20260228.json`
- Timeframe: `signal=5 mins`, `exec=1 min`, `RTH`
- Entry window: `09:00–16:00 ET` (RTH-only data; first tradable entries begin after 09:30 ET)
- Risk overlay: `riskoff_tr5_med_pct=8.5` + `risk_entry_cutoff_hour_et=15` (`riskoff_mode=hygiene`)
- Dual-branch slope-gated sizing downshift (timing sniper, without changing the base entry signal family):
  - `spot_dual_branch_enabled=true`
  - `spot_dual_branch_priority=a_then_b` (try strict A first, else fall back to B)
  - Branch A (full size): `spot_branch_a_size_mult=1.0`, `spot_branch_a_min_signed_slope_pct=0.00075`
  - Branch B (downshift): `spot_branch_b_size_mult=0.70`
- Riskpanic sizing overlay (chop/crisis belt):
  - Trigger: `riskpanic_tr5_med_pct=5.0` + `riskpanic_neg_gap_ratio_min=0.5`
  - Effect: `riskpanic_long_risk_mult_factor=0.4` + `riskpanic_short_risk_mult_factor=1.5`
- Riskpanic linear sizing overlay (pre-panic downshift, volatility ramp aware):
  - `riskpanic_long_scale_mode=linear`
  - `riskpanic_long_scale_tr_delta_max_pct=0.5`
- Cooldown: `cooldown_bars=4`
- Shock detect (no entry gating; enables `atr_fast_pct` for overlay):
  - `shock_gate_mode=detect`, `shock_detector=atr_ratio`, `shock_atr_fast_period=7`, `shock_atr_slow_period=50`
- Crash-regime short boost (drawdown-based, shock-off compatible, *velocity gated*):
  - `shock_drawdown_lookback_days=20`
  - `shock_on_drawdown_pct=-20`, `shock_off_drawdown_pct=-15`
  - Boost band: `0 <= (dd→on) <= 15pp` (meaning roughly `-20% .. -35%` drawdown on the rolling lookback)
  - `shock_short_risk_mult_factor=18.0` (scales `spot_short_risk_mult=0.01 -> 0.18` only inside the crash band)
  - Velocity gate (persistence precondition):
    - `shock_prearm_dist_on_max_pp=0.0` (keep prearm disabled)
    - `shock_prearm_min_dist_on_vel_pp=0.2` (ddBoost only when drawdown-velocity clears the floor)
- Short-entry depth band (tail-chase blocker; only matters in the deep tail):
  - `shock_short_entry_max_dist_on_pp=20.0`
  - Interpreted as: allow new shorts only when `(dd→on) ∈ [-20pp, +20pp]` (practically blocks opening new shorts beyond ~`-40%` drawdown on the rolling lookback)
- Graph risk overlay (ATR compress + direction-bias via shock direction):
  - `spot_risk_overlay_policy=atr_compress_shock_dir_bias`
  - `spot_graph_overlay_atr_hi_pct=1.4`, `spot_graph_overlay_atr_hi_min_mult=0.30`
  - `shock_direction_lookback=78` (direction smoothing for the overlay)
  - `spot_graph_overlay_trend_boost_max=1.0`, `spot_graph_overlay_trend_floor_mult=0.65` (downshift-only)
- Permission gate (needle-thread in v8): `ema_slope_min_pct=0.03`, `ema_spread_min_pct=0.003`, `ema_spread_min_pct_down=0.05`
- Graph entry gate (needle-thread in v9):
  - `spot_entry_policy=slope_tr_guard`
  - `spot_entry_slope_vel_abs_min_pct=0.00012` (leave other graph entry thresholds off)
- Graph exit flip-hold gate (needle-thread in v10):
  - `spot_exit_policy=slope_flip_guard`
  - `spot_exit_flip_hold_slope_min_pct=0.00008` (leave other flip-hold thresholds off)
- Dynamic guard scaling (vol-adaptive threshold scaling):
  - `spot_guard_threshold_scale_mode=atr_vel_direct`
  - `spot_guard_threshold_scale_min_mult=1.0` (no loosening; only scaling up in high ATR-velocity)
- RATS-V entry gate:
  - `ratsv_enabled=true`, `ratsv_slope_window_bars=5`, `ratsv_tr_fast_bars=5`, `ratsv_tr_slow_bars=20`
  - `ratsv_rank_min=0.10`, `ratsv_slope_med_min_pct=0.00010`, `ratsv_slope_vel_min_pct=0.00006`
- Stability floor (min `1Y/2Y` pnl/dd): **7.361**
- 1Y (`2025-01-01 -> 2026-01-19`): trades **556**, pnl **50,355.1**, dd **5,926.1**, pnl/dd **8.497**
- 2Y (`2024-01-01 -> 2026-01-19`): trades **1,095**, pnl **73,740.8**, dd **10,017.2**, pnl/dd **7.361**
- Dojo focus tape (`2026-02-23 -> 2026-02-27`, realized-trade dd): trades **10**, pnl **+511.9** (trade-dd **769.0**, pnl/dd **0.666**)
- Crashlab scored (`2025-01-01 -> 2025-03-31`, realized-trade dd): trades **147**, pnl **+4,590.4** (trade-dd **3,630.2**, pnl/dd **1.265**)

Replay / verify:
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v32_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_ddFac18_20260301.json \
  --symbol TQQQ --bar-size "5 mins" --use-rth --offline --cache-dir db \
  --top 1 --min-trades 0 \
  --window 2025-01-01:2026-01-19 \
  --window 2024-01-01:2026-01-19
```

## Evolutions (stack)

### v37 (2026-03-15) — Heat-Lip Sovereignty dethroned v36 for the multi-regime sovereignty contract
- Contract: coexistence across `2020` crash-transition, `2022` persistent downtrend, and `2025` persistent uptrend.
- Needle-thread:
  - Keep the whole v36 ownership seam unchanged.
  - Stop treating broad transition heat as one monotonic god.
  - Discover the real next signal:
    - the useful edge lives at the upper-lip transition heat seam around `0.79 - 0.80`
    - below that, heat is too broad
    - above that, `2020` falls off a cliff
  - Final dethrone cut:
    - add `regime2_transition_hot_shock_atr_pct_min=0.8`
  - What it really fixed:
    - `2020` rebound poison, especially `branch-b long`
    - fake `2022` rebound-long damage
    - while only shaving some `2025` continuation, mostly on `branch-b long`
- Outcome:
  - sovereignty floor (min `2020/2022/2025` pnl/dd): **1.459 -> 1.994** (dethrone)
  - `2020` pnl/dd: **1.459 -> 1.994**
  - `2022` pnl/dd: **2.828 -> 3.065**
  - `2025` pnl/dd: **5.468 -> 5.199**
  - soft sibling also proved the lip is real:
    - `regime2_transition_hot_shock_atr_pct_min=0.78`
    - `2020 = 1.866`, `2022 = 3.052`, `2025 = 5.228`
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v37_heatLipSovereignty_20260315.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v36_balancedCombinedSovereignty_20260315.json`

### v36 (2026-03-15) — Balanced Combined Sovereignty dethroned v35 for the multi-regime sovereignty contract
- Contract: coexistence across `2020` crash-transition, `2022` persistent downtrend, and `2025` persistent uptrend.
- Needle-thread:
  - Keep the v35 Bridge Crown host.
  - Stop treating crash-only invasion as the answer.
  - Discover the real truth:
    - the mutant did not materially beat the bridge in the raw COVID crash
    - its edge lived in `transition_up_hot`
  - Extend alternate-host sovereignty from:
    - `trend_up_clean` / `crash_down` experiments
    - to the real winning corridor:
      - `regime2_clean_host_takeover_state="crash_or_transition_up_hot"`
  - Final balanced saddle:
    - `regime2_clean_host_supertrend_multiplier=2.32`
    - `regime2_clean_host_bear_supertrend_multiplier=1.45`
    - `regime2_clean_host_bear_hard_supertrend_multiplier=2.18`
  - Keep the living `2020` scaffolding:
    - `regime2_crash_atr_pct_min=0.9`
    - `regime2_crash_block_longs=true`
    - `regime2_repair_block_branch_b_longs=true`
    - up-corridor `branch-a long` pocket map unchanged
- Outcome:
  - sovereignty floor (min `2020/2022/2025` pnl/dd): **-0.597 -> 1.459** (dethrone)
  - `2020` pnl/dd: **-0.597 -> 1.459**
  - `2022` pnl/dd: **2.327 -> 2.828**
  - `2025` pnl/dd: **5.302 -> 5.468**
  - final probe also exposed a separate heat crown:
    - `regime2_transition_hot_shock_atr_pct_min=0.65`
    - `2020 = 1.852`, `2022 = 1.917`, `2025 = 6.196`
    - preserved as a clue, not the dethrone, because it taxed `2022` too hard
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v36_balancedCombinedSovereignty_20260315.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v35_bridgeCrownThrone_bidirectional_20260315.json`

### v35 (2026-03-15) — Bridge Crown Throne dethroned v34 for the bidirectional bridge contract (`v34` remains the raw modern crown)
- Contract: opposite-regime bridge before `2020` (`2022` persistent downtrend + `2025` persistent uptrend).
- Needle-thread:
  - Keep `v34` as the host and stop trying to replace the core modern long engine.
  - Add a two-strength bear regime:
    - soft wakeup on `30 mins` supertrend
    - hard takeover confirm on `4 hours` supertrend
  - Fix the real semantic bug:
    - soft bear must not zero host entries when hard bear does not explicitly fire
  - Bridge saddle that actually held:
    - `regime2_bear_takeover_mode=always`
    - `regime2_bear_hard_supertrend_multiplier: 2.25/2.75 probes -> 2.5` (sweet spot)
    - `regime2_bear_supertrend_multiplier: 1.45/1.6 probes -> 1.5` (modern-leaning bridge throne)
    - `regime2_soft_bear_branch_a_slope_med_slow_min_pct: 0.00012+ -> 0.00008` (hug the base floor)
    - `regime2_soft_bear_branch_a_slope_vel_slow_min_pct: null -> 0.00002` (the real modern-protection nerve)
    - `regime2_bear_allow_long_recovery=true`
- Outcome:
  - opposite-regime floor (min `2022/2025` pnl/dd): **-0.567 -> 2.327** (dethrone)
  - `2022` pnl/dd: **-0.567 -> 2.327**
  - `2025` pnl/dd: **9.353 -> 5.302**
  - `2020` preview: **-0.909 -> -0.597** (material lift before any shock-transition overlay work)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v35_bridgeCrownThrone_bidirectional_20260315.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v34_km01_shockMin1p25_downLong0p10_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_rpLong0p10_ddFac18_20260305.json`

### v33 (2026-03-02) — dethroned v32 (riskpanic long downshift)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Riskpanic long downshift: reduce long sizing only in riskpanic tapes (chop/crisis belt), without touching the core entry/exit family:
    - `riskpanic_long_risk_mult_factor: 0.4 -> 0.10`
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **7.361 -> 7.732** (dethrone)
  - `1Y` pnl/dd: **8.497 -> 9.202**
  - `2Y` pnl/dd: **7.361 -> 7.732**
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v33_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_rpLong0p10_ddFac18_20260302.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v32_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_ddFac18_20260301.json`

### v32 (2026-03-01) — dethroned v31 (crash ddBoost factor lift)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Increase the crash-band short boost factor (still vel-gated) so the Jan–Mar 2025 downturn tape is monetized harder without changing normal-regime behavior:
    - `shock_short_risk_mult_factor: 16.0 -> 18.0`
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **7.356 -> 7.361** (dethrone)
  - `1Y` pnl/dd: **8.470 -> 8.497**
  - `2Y` pnl/dd: **7.356 -> 7.361**
  - crashlab scored (`2025-01-01 -> 2025-03-31`, realized-trade dd): **1.242 -> 1.265** (recovered while also improving floor)
  - dojo focus (`2026-02-23 -> 2026-02-27`, realized-trade dd): **unchanged** (**+511.9**, pnl/dd **0.666**)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v32_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_ddFac18_20260301.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v31_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_20260301.json`

### v31 (2026-03-01) — dethroned v30 (riskpanic neg-gap sensitivity tighten)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Trigger riskpanic slightly earlier on gap-down tapes (defensive downshift without touching the core entry/exit family):
    - `riskpanic_neg_gap_ratio_min: 0.6 -> 0.5`
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **7.337 -> 7.356** (dethrone)
  - `1Y` pnl/dd: **8.109 -> 8.470**
  - `2Y` pnl/dd: **7.337 -> 7.356**
  - crashlab scored (`2025-01-01 -> 2025-03-31`, realized-trade dd): **1.265 -> 1.242** (regressed slightly; tracked for next evolution)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v31_km01_dualbranchSlope0p00075_bmult0p70_panicNeg0p50_20260301.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v30_km01_dualbranchSlope0p00075_bmult0p70_20260301.json`

### v30 (2026-03-01) — dethroned v29 (dual-branch slope-gated sizing downshift)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Enable dual-branch selection and apply a size downshift only when slope is weak (so throughput stays HF, but tail losses shrink):
    - `spot_dual_branch_enabled: false -> true`
    - `spot_dual_branch_priority: b_then_a -> a_then_b`
    - `spot_branch_a_min_signed_slope_pct: null -> 0.00075` (strict A qualifies only when slope is aligned)
    - `spot_branch_b_size_mult: 1.0 -> 0.70` (fallback B is smaller)
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.743 -> 7.337** (dethrone)
  - `1Y` pnl/dd: **6.762 -> 8.109**
  - `2Y` pnl/dd: **6.743 -> 7.337**
  - crashlab scored (`2025-01-01 -> 2025-03-31`, realized-trade dd): pnl/dd **0.629 -> 1.265** (material lift in the Jan–Mar 2025 downturn tape)

### v29 (2026-03-01) — dethroned v28 (crash dd boost widen, but vel-gated)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Restore a wider crash drawdown short-boost band and higher boost factor, but add a persistence gate so we only boost when drawdown is actually moving (avoid false boosts / whipsaw tails):
    - `shock_short_boost_max_dist_on_pp: 12.0 -> 15.0`
    - `shock_short_risk_mult_factor: 14.0 -> 16.0`
    - `shock_prearm_min_dist_on_vel_pp: 0.0 -> 0.2` (vel gate)
    - `shock_prearm_dist_on_max_pp: None -> 0.0` (keep prearm disabled; vel gate still binds ddBoost)
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.676 -> 6.743** (dethrone)
  - `1Y` pnl/dd: **6.734 -> 6.762**
  - `2Y` pnl/dd: **6.676 -> 6.743**
  - dojo focus (`2026-02-23 -> 2026-02-27`): unchanged (**+465.7**, pnl/dd **0.264**)
  - crashlab scored (`2025-01-01 -> 2025-03-31`): regressed vs v28 (**0.657 -> 0.469**, still positive; target for next evolution)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v29_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max15_fac16_velGate0p2_dynAtrVelDirect_shortEntryBand20_20260301.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v28_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max12_fac14_dynAtrVelDirect_shortEntryBand20_20260301.json`

### v28 (2026-03-01) — dethroned v27 (dyn guard scaling + tighten crash dd band)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Tightened the crash drawdown short-boost band and reduced the boost factor (keep the crash weapon, stop over-chasing the tail):
    - `shock_short_boost_max_dist_on_pp: 15.0 -> 12.0`
    - `shock_short_risk_mult_factor: 16.0 -> 14.0`
  - Enabled volatility-adaptive guard scaling so chop spikes tighten our entry/exit guards instead of turning into churn:
    - `spot_guard_threshold_scale_mode: off -> atr_vel_direct`
    - `spot_guard_threshold_scale_min_mult: 0.7 -> 1.0` (no loosening; scale-up only)
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.628 -> 6.676** (dethrone)
  - `1Y` pnl/dd: **6.717 -> 6.734**
  - `2Y` pnl/dd: **6.628 -> 6.676**
  - dojo focus (`2026-02-23 -> 2026-02-27`): **unchanged** (same tape behavior)
  - crashlab scored (`2025-01-01 -> 2025-03-31`): **regressed vs v27** (**0.777 -> 0.657**, tracked for next evolution)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v28_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max12_fac14_dynAtrVelDirect_shortEntryBand20_20260301.json`
- Previous: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v27_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max15_fac16_shortEntryBand20_20260301.json`

### v27 (2026-03-01) — dethroned v26 (short-entry tail-chase band)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Added a *banded* short-entry depth gate around the crash ON threshold. This blocks opening new shorts only in the ultra-deep tail, where the strategy tends to chase and give back risk-adjusted performance:
    - `shock_short_entry_max_dist_on_pp: 0.0 -> 20.0` (banded gate around `dd→on`)
  - Policy semantics (engine):
    - Interpret `(dd→on)` as a signed distance in percentage points from the ON threshold:
      - negative = milder drawdown (above the ON threshold)
      - positive = deeper drawdown (below the ON threshold)
    - Gate allows shorts only when `(dd→on) ∈ [-max_dist, +max_dist]`.
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.591 -> 6.628** (dethrone)
  - `1Y` pnl/dd: **6.684 -> 6.717**
  - `2Y` pnl/dd: **6.591 -> 6.628**
  - crash lab (warmup `2024-01-01 -> 2025-03-31`, scored `2025-01-01 -> 2025-03-31`): unchanged (**0.777**)
  - chop dojo focus pnl (2026-02-23..2026-02-27): unchanged (**+465.7**, pnl/dd **0.264**)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v27_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max15_fac16_shortEntryBand20_20260301.json`

### v26 (2026-03-01) — dethroned v25 (crash-regime drawdown boost v3)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Same crash band as v25, but increased the short boost factor so crash tapes are monetized harder (while leaving normal regime behavior unchanged):
    - `shock_short_risk_mult_factor: 12.0 -> 16.0` (scales `spot_short_risk_mult=0.01 -> 0.16` only inside the crash band)
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.561 -> 6.591** (dethrone)
  - `1Y` pnl/dd: **6.623 -> 6.684**
  - `2Y` pnl/dd: **6.561 -> 6.591**
  - crash lab (warmup `2024-01-01 -> 2025-03-31`, scored `2025-01-01 -> 2025-03-31`):
    - pnl/dd **0.666 -> 0.777** (material lift in the Jan–Mar 2025 downturn tape)
  - chop dojo focus pnl (2026-02-23..2026-02-27): unchanged (**+465.7**, pnl/dd **0.264**)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v26_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max15_fac16_20260301.json`

### v25 (2026-03-01) — dethroned v24 (crash-regime drawdown boost v2)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Expanded the crash short-boost so it can actually monetize the deeper drawdown tail (without touching normal uptrend behavior):
    - `shock_short_boost_max_dist_on_pp: 10 -> 15`
    - `shock_short_risk_mult_factor: 10.0 -> 12.0` (scales `spot_short_risk_mult=0.01 -> 0.12` only inside the crash band)
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.536 -> 6.561** (dethrone)
  - `1Y` pnl/dd: **6.556 -> 6.623**
  - `2Y` pnl/dd: **6.536 -> 6.561**
  - crash lab (warmup `2024-01-01 -> 2025-03-31`, scored `2025-01-01 -> 2025-03-31`):
    - pnl/dd **0.565 -> 0.666** (material lift in the Jan–Mar 2025 downturn tape)
  - chop dojo focus pnl (2026-02-23..2026-02-27): unchanged (**+465.7**, pnl/dd **0.264**)
- Preset: `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v25_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max15_fac12_20260301.json`

### v24 (2026-02-28) — dethroned v23 (crash-regime drawdown boost)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Implemented crash-regime weaponization without touching the main shock detector:
    - Engine: compute auxiliary daily-drawdown telemetry when dd-gate knobs are present (so dd gates work even when `shock_detector=atr_ratio`).
    - Policy: allow drawdown-based short boost to apply even when shock is off (guarded by `shock_short_boost_max_dist_on_pp > 0` so existing configs don’t change behavior).
  - Enabled the crash short boost in a narrow crash band:
    - `shock_drawdown_lookback_days: 20`
    - `shock_on_drawdown_pct: -20`, `shock_off_drawdown_pct: -15`
    - `shock_short_boost_max_dist_on_pp: 10`
    - `shock_short_risk_mult_factor: 10.0` (scales `spot_short_risk_mult=0.01 -> 0.10` only inside the crash band)
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.398 -> 6.536** (massive lift; dethrone)
  - `1Y` pnl/dd: **6.472 -> 6.556**
  - `2Y` pnl/dd: **6.398 -> 6.536**
  - chop dojo focus pnl (2026-02-19..2026-02-25): unchanged (**+1,206.1**)
  - crash lab (warmup `2024-01-01 -> 2025-03-31`, scored `2025-01-01 -> 2025-03-31`):
    - pnl/dd **0.218 -> 0.565** (material crash survival + monetization)
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v24_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_ddBoost_lb20_on20_off15_max10_fac10_20260228.json`

### v23 (2026-02-28) — dethroned v22 (riskpanic short weaponization)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v22 unchanged, but weaponize shorts only during true riskpanic days (so we monetize crash/chop tapes without touching the normal uptrend behavior):
    - `riskpanic_short_risk_mult_factor: 1.0 -> 1.5`
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.392 -> 6.398**
  - `1Y` pnl/dd: **6.507 -> 6.472** (small giveback)
  - `2Y` pnl/dd: **6.392 -> 6.398** (small lift; the dethrone)
  - dojo focus pnl (2026-02-19..2026-02-25): **+547.2 -> +1,206.1** (big lift in the chop tape)
  - downturn lab (`2024-12-02 -> 2025-03-31`): pnl/dd **0.700 -> 0.706**
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v23_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_rpShort1p5_20260228.json`

### v22 (2026-02-28) — dethroned v21 (anti-churn spacing cd3->cd4)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Increase the cooldown slightly so we stop re-entering too quickly after a fill (this is the smallest change that materially changes chop/downturn behavior without widening the surface area):
    - `cooldown_bars: 3 -> 4`
- Outcome:
  - stability floor (min `1Y/2Y` pnl/dd): **6.144 -> 6.392**
  - `1Y` pnl/dd: **6.519 -> 6.507** (tiny giveback, still elite)
  - `2Y` pnl/dd: **6.144 -> 6.392** (big lift; this is the dethrone)
  - dojo focus pnl (2026-02-19..2026-02-25): **+547.2 -> +547.2** (no regression in the chop tape)
  - downturn lab (`2024-12-02 -> 2025-03-31`): pnl/dd **0.464 -> 0.700**
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v22_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_20260228.json`

### v21 (2026-02-28) — dethroned v20 (ATR compress + shock-direction bias overlay)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v20 unchanged, but swap the trend overlay for a direction-aware overlay driven by smoothed shock direction (sign of recent returns):
    - `spot_risk_overlay_policy: atr_compress_trend_bias -> atr_compress_shock_dir_bias`
    - `shock_direction_lookback: 2 -> 78` (stabilizes direction, so we stop thrashing in chop / drift-down)
  - Outcome: strong lift on both stability + chop tape (throughput unchanged):
    - stability floor (min `1Y/2Y` pnl/dd): **5.775 -> 6.144**
    - `1Y` pnl/dd: **6.391 -> 6.519**
    - `2Y` pnl/dd: **5.775 -> 6.144**
    - dojo focus pnl: **+441.8 -> +547.2**
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v21_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCShockDir_lb78_floor0p65_boost1p0_hi1p4_min0p3_20260228.json`

### v20 (2026-02-28) — dethroned v19 (ATR compress + trend-bias overlay)
- Contract: `1Y` then `2Y` (10Y deferred).
- Needle-thread:
  - Keep v19 unchanged, but swap the graph overlay from pure ATR compression to an ATR+trend hybrid so drift-down tapes don’t get treated as “no event” just because volatility is stable:
    - `spot_risk_overlay_policy: atr_compress -> atr_compress_trend_bias`
  - Make the trend overlay actually bind on real intraday slope magnitudes and disable leverage-up:
    - `spot_graph_overlay_slope_ref_pct: 0.08 -> 0.00015`
    - `spot_graph_overlay_trend_boost_max: 1.35 -> 1.0` (no boosting; downshift-only)
  - Outcome: real stability-floor lift (trade count stays in the HF band):
    - stability floor (min `1Y/2Y` pnl/dd): **5.404 -> 5.775**
    - `1Y` pnl/dd: **5.919 -> 6.391**
    - `2Y` pnl/dd: **5.404 -> 5.775**
- Preset: `backtests/tqqq/archive/champion_history_20260228/tqqq_hf_champions_v20_km01_panicTr5med5p0_neg0p6_long0p4_linDmax0p5_overlayAtrCTrendBias_sref0p00015_boost1p0_hi1p4_min0p3_20260228.json`

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
