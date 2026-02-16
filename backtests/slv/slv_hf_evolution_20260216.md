# SLV HF Evolution Log — 2026-02-16

## High-Trade Adaptive Overlay Baseline (Key Discovery)
- Recovered high-turnover behavior with the new adaptive overlay path:
  - `tr=6833` (widened pass)
  - `tr=8888` (clean cache pass)
- These did **not** dethrone on pnl, but this is an important HF baseline for future crash/fakeout precision research.

## Mission Context
- Goal: detect Jan-Feb 2026 style cascading downturns earlier and ride them with better precision.
- Constraint: keep strong annual trade participation (`>=700`) while improving pnl/pnl-dd.
- Current base anchor in this corridor:
  - `tr=3079`, `pnl=-6173.9`, `pnl/dd=-1.00`

## Commands Used (Current Session)
```bash
# Adaptive/velocity combo corridor
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --offline --cache-dir db --realism2 \
  --min-trades 700 --top 20 --jobs 8

# PT/SL recovery probe at same window/floor
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis ptsl --base champion --offline --cache-dir db --realism2 \
  --min-trades 700 --top 40
```

## Adaptive Overlay Corridor Snapshot
- Policy/graph knobs explored:
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
- RATS timing probes:
  - `rank_min in {0.0035, 0.0085, 0.0185}`
  - `cross_age in {3, 6, 10}`
  - `slope_med in {0.000002, 0.000006}`
  - `slope_vel in {0.000001, 0.000002}`

## Results Summary
- Adaptive overlay high-trade rows:
  - `tr=6833`, `pnl=-13625.2`
  - `tr=8888`, `pnl=-17879.6`
- Best result in current 1Y/700 sweep family remained base:
  - `tr=3079`, `pnl=-6173.9`, `pnl/dd=-1.00`
- PT/SL sweep still strongest non-base recovery line in this window:
  - best row: `tr=712`, `pnl=-1413.2`, `pnl/dd=-1.00`
  - top parameters:
    - `PT=0.0180`
    - `SL=0.0360`
    - `flip=entry`
    - `hold=6`
    - `only_profit=0`
    - `gate=regime_or_permission`
    - `close_eod=0`

## Critical Engine/Backtest Integrity Fixes Logged
- Fixed cache poisoning across different trade floors:
  - stage cache keys now scoped by `run_min_trades`.
- Fixed incomplete axis-dimension fingerprint collisions:
  - fingerprint now uses full strategy+filters payload, not a narrow legacy subset.
- Bumped cache engine version to invalidate stale rank/stage manifests:
  - `_RUN_CFG_CACHE_ENGINE_VERSION=spot_stage_v8`.

## Interpretation
- High-trade adaptive behavior is now reproducible and traceable.
- Current overlay settings are too churn-heavy and directionally weak in this window.
- PT/SL remains the best current recovery layer for this specific 1Y window, but still negative.

## Next Iteration Ideas
- Couple adaptive resizing with stricter anti-churn regime:
  - larger `spot_resize_min_delta_qty`
  - stronger cooldown
  - tighter max step
- Add lower-timeframe bias gate (below 1D) to reduce fakeout flips before resize amplification.
- Increase weight on negative-slope velocity consistency + ATR velocity acceleration only in confirmed downside bursts, with symmetric upside release.

## Sub-1D Bias-Gate Probe (v11.1, Investigation)
Commands:
```bash
# Trade-floor constrained pass
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 20 --jobs 8

# Diagnostic pass (no trade floor) to inspect hidden survivors
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 0 --top 40 --jobs 8
```

Knob corridor used:
- `rank_min in {0.0085, 0.0185, 0.0240}`
- `cross_age in {3, 5, 8}`
- `slope_med/slope_vel in {(0.000002,0.000001), (0.000006,0.000002)}`
- `spot_branch_b_size_mult in {1.10, 1.40}`
- Timing profile gate overlays:
  - `overlay_only_hybrid` -> `ST@1d` baseline
  - `early_sizeup_hybrid` -> `ST@4h + regime2 ST@4h`
  - `strict_slow_confirm` -> `ST@4h` tighter confirmation
  - `loose_probe_hybrid` -> `ST@4h` loose probe

Outcomes:
- `min_trades=700`: best kept row was still high-trade but deeply negative:
  - `tr=773`, `pnl=-22981.4`, `pnl/dd=-0.55` (`overlay_only_hybrid`)
- `min_trades=0` diagnostic:
  - Base was positive but low-frequency: `tr=119`, `pnl=26293.8`, `pnl/dd=0.83`
  - Positive adaptive rows existed only in `strict_slow_confirm`, but tiny activity:
    - `tr=27..29`, `pnl~+900`

Interpretation:
- Sub-1D gate variants currently over-prune; they improve local quality but collapse trade count.
- The only profile clearing the `>=700` floor remained `overlay_only_hybrid`, which is loss-heavy in this Feb-anchored year.
- Practical next move: run a dedicated high-trade rescue around `overlay_only_hybrid` (exit/flip/PTSL/cadence/resize-throttle), while keeping sub-1D gates as low-frequency quality overlays.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_bias4h1h.log`
- `backtests/slv/slv_hf_sniper_1y_diag_min0_20260216_bias4h1h.log`

## HF Rescue Corridor A (v11.2, investigation)
Commands:
```bash
# Strict HF floor probe
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 5000 --top 25 --jobs 8

# Same corridor with lower floor
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 30 --jobs 8
```

Outcome:
- `min_trades=5000`: `tested=96`, `kept=0`
- `min_trades=700`: `tested=96`, `kept=0`

Interpretation:
- Making `spot_entry_policy=slope_tr_guard` the default across all profiles over-pruned participation.
- This corridor did not preserve the HF trade floor and was discarded.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade5000_20260216_rescueA.log`
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_rescueA.log`

## HF Rescue Corridor B (v11.3, investigation)
Command:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 30 --jobs 8
```

Knob family:
- Entry-permissive anchor with selective guard variants (`hf_anchor_overlay`, `hf_guard_soft`, `hf_guard_medium`, `hf_crash_probe`)
- `rank_min in {0.0035, 0.0085}`
- `cross_age in {3, 6}`
- `slope_med/slope_vel in {(0.000002,0.000001), (0.000006,0.000002)}`
- `spot_branch_b_size_mult in {1.00, 1.20}`

Outcome:
- `tested=64`, `kept=48` (`min_trades=700`)
- Best kept row:
  - `tr=899`, `pnl=-11577.9`, `pnl/dd=-0.38` (`hf_anchor_overlay`)
- Guard-medium rows were materially worse:
  - `tr=975`, `pnl~ -22389 ... -24114`

Interpretation:
- Survivorship recovered, but not profitability.
- Observed participation remained around `~900-975` trades, not `~6.8k/8.8k`.

Artifacts:
- `backtests/slv/slv_hf_sniper_1y_trade700_20260216_rescueB.log`

## HF Rescue Corridor C (v11.4, aborted early)
Command attempted:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 3000 --top 25 --jobs 8
```

Why aborted:
- At ~25% progress, all workers still had `kept=0`, and per-candidate footprints were in the low-hundreds range.
- This made `>=3000` survivor probability negligible for this profile family.

Interpretation:
- Under current semantics/fixes, the old 6.8k/8.8k signature was not reproduced by the current rescue corridors.
- Next step should be explicit reproducibility for the legacy high-trade line in a single-config replay before further corridor mutation.

Artifact:
- `backtests/slv/slv_hf_sniper_1y_trade3000_20260216_rescueC.log`

## HF Crash-Predref Probe (v11.5, investigation)
Commands:
```bash
# 1Y trade-floor validation
python -m tradebot.backtest spot \
  --symbol SLV --start 2025-02-14 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 700 --top 30 --jobs 8

# Jan-Feb crash slice diagnostic
python -m tradebot.backtest spot \
  --symbol SLV --start 2026-01-01 --end 2026-02-14 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --axis combo_full --combo-full-preset hf_timing_sniper \
  --base champion --seed-milestones backtests/slv/archive/champion_history_20260214/slv_hf_champions_v9.json \
  --offline --cache-dir db --realism2 --min-trades 0 --top 30 --jobs 8
```

Tested corridor:
- `rank_min in {0.0035, 0.0185}`
- `cross_age in {6, 10}`
- `slope_med/slope_vel in {(0.000002,0.000001), (0.000006,0.000002)}`
- `spot_branch_b_size_mult in {1.20, 1.60}`
- Mode A `overlay_only_hybrid_baseline`
- Mode B `overlay_only_hybrid_predref` (new predictive refs):
  - `spot_resize_adaptive_atr_vel_ref_pct=0.25`
  - `spot_graph_overlay_atr_vel_ref_pct=0.25`
  - `spot_graph_overlay_trend_boost_max=1.75`
  - `spot_graph_overlay_trend_floor_mult=0.88`
  - `spot_exit_flip_hold_tr_ratio_min=1.00`
  - `spot_exit_flip_hold_slow_slope_min_pct=0.000002`
  - `spot_exit_flip_hold_slow_slope_vel_min_pct=0.000001`

Outcomes:
- 1Y floor (`min_trades=700`): `tested=32`, `kept=16`
  - Survivors were baseline-only; best kept row:
    - `tr=773`, `pnl=-22147.4`, `pnl/dd=-0.52`
- Crash slice (`2026-01-01 -> 2026-02-14`, `min_trades=0`):
  - Predref beat baseline decisively around `cross=6`:
    - `tr=65`, `pnl=+2602.2`, `pnl/dd=0.20` (`overlay_only_hybrid_predref`)
  - Baseline in same slice remained deeply negative:
    - `tr=114`, `pnl~ -19728` (`overlay_only_hybrid_baseline`)

Interpretation:
- New ATR-velocity + trend-bias reference knobs can meaningfully improve the Jan-Feb crash regime.
- Current predref settings still under-shoot the 1Y `>=700` floor (no predref rows survived 1Y floor gate).
- Best next mutation is to raise predref annual participation while preserving crash-edge (e.g., soften flip-hold, slightly reduce atr_vel refs only on full-year corridor).

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

## v11.6 — predref bridge pass (cross/floor/tr-ratio mutation, 1Y)

Status: **DONE (investigation, not promoted)**

Command used:
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

Bridge corridor (requested):
- `cross_age in {4,6}` (toward annual participation recovery)
- `spot_exit_flip_hold_tr_ratio_min in {0.97,0.99}`
- `spot_graph_overlay_trend_floor_mult in {0.84,0.86}`
- ATR-velocity refs fixed:
  - `spot_resize_adaptive_atr_vel_ref_pct=0.25`
  - `spot_graph_overlay_atr_vel_ref_pct=0.25`
- Tight lock for runtime:
  - `rank_min=0.0035` only
  - `slope pairs in {(0.000006,0.000002), (0.000002,0.000001)}`
  - `spot_branch_b_size_mult in {1.20,1.60}`

Sweep outcome:
- Runtime path: `tested=33` (`32` bridge rows + base), `kept=33`
- Best bridge family rows:
  - `cross=4`, `floor=0.86`, `tr_ratio_min in {0.97,0.99}`
  - `tr=638`, `pnl=+2969.9`, `dd=25558.9`, `pnl/dd=0.12`
- Trade-floor side:
  - `cross=6` increased participation (`tr~650`) but pnl collapsed toward flat/negative.
- Contract status:
  - Did not clear `>=700` 1Y trade floor in sweep output.
  - Did not approach current 1Y HF crown pnl.

Focused replay probe (`--no-write`) around best row:
- `cross=4 floor=0.86 tr=0.99`: `tr=669`, `pnl=+3853.6`, `pnl/dd=0.142`
- `cross=5 floor=0.86 tr=0.99`: `tr=674`, `pnl=+3327.1`, `pnl/dd=0.122`
- `cross=6 floor=0.86 tr=0.99`: `tr=681`, `pnl=+579.0`, `pnl/dd=0.020`
- extended cross-age continuation at same floor/tr-ratio:
  - `cross=7`: `tr=687`, `pnl=-1301.3`, `pnl/dd=-0.045`
  - `cross=8`: `tr=695`, `pnl=-1065.4`, `pnl/dd=-0.038`
  - `cross=9`: `tr=700`, `pnl=-2816.0`, `pnl/dd=-0.102`
- Signal: pushing cross-age higher raises trade count modestly but bleeds edge faster than it gains frequency.

Trade forensics:
- Artifact: `backtests/slv/slv_hf_bridge_trade_diag_20260216.json`
- Core readout:
  - best-cross4 replay: `669` trades, pnl `+3853.6`
  - cross6 sibling: `681` trades, pnl `+579.0`
  - delta (`cross6 - cross4`): `+12` trades, pnl `-3274.5`
  - losses increased on both sides (long delta `-1827.5`, short delta `-1447.0`)
  - worst degradation months: `2025-05`, `2025-12`, `2025-09`, `2026-01`
