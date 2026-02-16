# Spot Policy Kernel

`tradebot/spot/policy.py`, `tradebot/spot/lifecycle.py`, and `tradebot/spot/graph.py` are the shared spot kernels for:
- backtest execution (`tradebot/backtest/engine.py`)
- live/UI runtime (`tradebot/ui/bot_order_builder.py`, `tradebot/ui/bot_signal_runtime.py`)

The goal is to keep strategy-evolution logic centralized and readable.

Core typed primitives:
- `SpotPolicyConfigView`: single sanitized/defaulted knob view used by policy methods.
- `SpotDecisionTrace`: typed sizing trace emitted into live/backtest payloads.
- `SpotIntentDecision`: typed quantity transition decision (`hold|enter|exit|resize`).
- `SpotLifecycleDecision`: typed lifecycle decision for pending/flat/open stages.
- `SpotPolicyGraph`: pluggable node registry (entry/exit/resize/risk overlay) with profile selection.

## Lifecycle Kernel

`tradebot/spot/lifecycle.py` centralizes lifecycle decisions and exit arbitration:

- `decide_pending_next_open()`:
  - pending next-open cancel/trigger/hold
  - risk-day/riskoff directional cancellation behavior
- `decide_flat_position_intent()`:
  - flat-side gate evaluation + entry intent
- `decide_open_position_intent()`:
  - exit reason arbitration by priority
  - controlled flip queueing (`spot_controlled_flip`)
  - in-position target resize intent + cooldown gate
  - optional adaptive target curves (`spot_resize_adaptive_*`)

All three are called by both live and backtest adapters.

`tradebot/spot/graph.py` provides graph node orchestration:
- graph profiles: `neutral`, `defensive`, `aggressive`, `hf_probe`
- node families:
  - `entry_policy`
  - `exit_policy`
  - `resize_policy`
  - `risk_overlay_policy`
- selectors:
  - `spot_policy_graph` / `spot_graph_profile`
  - per-node overrides:
    - `spot_entry_policy`
    - `spot_exit_policy`
    - `spot_resize_policy`
    - `spot_risk_overlay_policy`

## Sizing Pipeline

Entry sizing pipeline in `SpotPolicy`:
1. `resolve_entry_action_qty()` resolves directional spot action/qty mapping.
2. `risk_overlay_policy()` normalizes riskoff/riskpanic/riskpop factors.
3. `shock_exit_pct_multipliers()` + `scale_exit_pcts()` apply shock-aware pct exit scaling.
4. `calc_signed_qty()` computes base qty by `spot_sizing_mode`:
   - `fixed`
   - `notional_pct`
   - `risk_pct`
   - `calc_signed_qty_with_trace()` returns `(qty, SpotDecisionTrace)`.
5. Overlay multipliers are applied in `risk_pct` mode:
   - riskoff/riskpanic/riskpop
   - shock long/short multipliers
6. Dynamic shock ATR throttle (`shock_risk_scale_*`) can scale risk and/or cap.
7. Final clamps are applied:
   - `spot_max_notional_pct`
   - buying power cap (BUY only)
   - `spot_max_qty`
   - lot rounding + `spot_min_qty`
8. Optional branch post-scaling:
   - `branch_size_mult()`
   - `apply_branch_size_mult()`

Pending-entry next-open cancellation policy:
- `risk_entry_cutoff_hour_et()`
- `pending_entry_should_cancel()`

## Shared Knob Surface

Primary sizing knobs:
- `spot_sizing_mode`
- `spot_notional_pct`
- `spot_risk_pct`
- `spot_short_risk_mult`
- `spot_max_notional_pct`
- `spot_min_qty`
- `spot_max_qty`
- `spot_resize_mode`
- `spot_resize_min_delta_qty`
- `spot_resize_max_step_qty`
- `spot_resize_allow_scale_in`
- `spot_resize_allow_scale_out`
- `spot_resize_cooldown_bars`
- `spot_resize_adaptive_mode`
- `spot_resize_adaptive_min_mult`
- `spot_resize_adaptive_max_mult`
- `spot_resize_adaptive_atr_target_pct`
- `spot_resize_adaptive_slope_ref_pct`
- `spot_resize_adaptive_vel_ref_pct`
- `spot_resize_adaptive_tr_ratio_ref`
- `spot_controlled_flip`
- `spot_policy_pack`
- `spot_policy_graph`
- `spot_graph_profile`
- `spot_entry_policy`
- `spot_exit_policy`
- `spot_resize_policy`
- `spot_risk_overlay_policy`

Graph entry gate experiment knobs:
- `spot_entry_tr_ratio_min`
- `spot_entry_slope_med_abs_min_pct`
- `spot_entry_shock_atr_max_pct`

Graph exit experiment knobs:
- `spot_exit_flip_hold_slope_min_pct`
- `spot_exit_flip_hold_tr_ratio_min`

Graph overlay experiment knobs:
- `spot_graph_overlay_atr_hi_pct`
- `spot_graph_overlay_atr_hi_min_mult`
- `spot_graph_overlay_tr_ratio_ref`
- `spot_graph_overlay_slope_ref_pct`
- `spot_graph_overlay_trend_boost_max`
- `spot_graph_overlay_trend_floor_mult`

Overlay knobs:
- `riskoff_mode`
- `riskoff_long_risk_mult_factor`
- `riskoff_short_risk_mult_factor`
- `riskpanic_long_risk_mult_factor`
- `riskpanic_short_risk_mult_factor`
- `riskpanic_long_scale_mode`
- `riskpanic_long_scale_tr_delta_max_pct`
- `riskpanic_neg_gap_ratio_min`
- `riskpop_long_risk_mult_factor`
- `riskpop_short_risk_mult_factor`

Shock knobs:
- `shock_long_risk_mult_factor`
- `shock_long_risk_mult_factor_down`
- `shock_short_risk_mult_factor`
- `shock_risk_scale_target_atr_pct`
- `shock_risk_scale_min_mult`
- `shock_risk_scale_apply_to`
- `shock_stop_loss_pct_mult`
- `shock_profit_target_pct_mult`

Branch knobs:
- `spot_dual_branch_enabled`
- `spot_branch_a_size_mult`
- `spot_branch_b_size_mult`

Pending-entry/cutoff knobs:
- `risk_entry_cutoff_hour_et`
- legacy fallback: `entry_end_hour_et` or `entry_end_hour` when start hour is unset

Policy packs:
- `tradebot/spot/packs.py` exposes baseline profiles (`neutral`, `defensive`, `aggressive`, `hf_probe`)
- explicit strategy/filter fields always override pack defaults.
- graph profile defaults follow `spot_policy_pack` unless explicitly overridden.

## Adapter Responsibilities

`tradebot/engine.py` exposes compatibility wrappers so existing callers can keep importing from `engine`.

Backtest/UI should remain thin adapters:
- gather runtime context (signal/account/shock/risk)
- call shared policy helpers
- avoid re-implementing policy math in local modules

Trace emission:
- live UI gate/order journals include:
  - `spot_lifecycle`
  - `spot_intent`
  - `spot_decision`
- backtest spot trades carry `decision_trace` with:
  - `spot_lifecycle`
  - `spot_intent`
  - optional resize rows in `decision_trace.resizes`

Scenario runner helpers:
- `tradebot/spot/scenario.py` can produce:
  - normalized lifecycle trace rows (`lifecycle_trace_row`)
  - why-not exit/resize report (`why_not_exit_resize_report`)
  - csv output (`write_rows_csv`)
- Backtest capture knobs:
  - `spot_capture_lifecycle_trace=true` to keep `BacktestResult.lifecycle_trace`
  - `spot_lifecycle_trace_path=/path/file.csv` optional direct csv dump
  - `spot_why_not_report_path=/path/file.csv` optional why-not report csv
