# How It Works (Backtest + Live-Equivalent Spot Pipeline)

This document is the source-of-truth explainer for how the spot signal/lifecycle/sizing system is wired today in this repo.

It answers the exact confusion you called out:
- why logs show multiple direction-ish fields (`state`, `entry`, `regime`, `shock`)
- whether those fields are duplicated (they are not)
- what `shock_dir` really is
- what “return sign” in `_ShockDirectionMixin` means
- how every major knob family interacts

The explanations below are grounded in current code paths, primarily:
- `tradebot/engine.py`
- `tradebot/spot_engine.py`
- `tradebot/spot/lifecycle.py`
- `tradebot/spot/policy.py`
- `tradebot/spot/graph.py`
- `tradebot/backtest/config.py`
- `tradebot/ui/bot_journal.py`
- `tradebot/ui/bot_signal_runtime.py`

## 1) Fast answer to your core question

No, the system is not simply cloning one direction three times.

These fields are intentionally different layers:
- `state`: raw EMA trend state (fast vs slow EMA relationship).
- `entry` / `entry_dir`: entry candidate after entry-mode logic (`trend`/`cross`) + confirm bars + regime gating.
- `regime` / `regime_dir`: independent bias gate direction (EMA regime or Supertrend regime).
- `regime2`: secondary bias gate, optional, side-selective (`longs`/`shorts`/`both`).
- `shock` (bool): volatility stress state (detector ON/OFF with hysteresis).
- `shock_dir`: short-horizon return-sign direction from `_ShockDirectionMixin`; this is not the same object as `regime_dir`.

So yes, they can disagree, and most disagreements are expected behavior, not duplication bugs.

## 2) What each logged token means exactly

When journal lines print stuff like:
- `dir<state=... entry=... spread=... slope=...>`
- `bias<regime=... shock=... dir_src=...>`

the mapping is:

- `state`
  - from `EmaDecisionSnapshot.state`
  - computed via `ema_state_direction(ema_fast, ema_slow)`
  - values: `up`, `down`, or `None` when not ready.

- `entry`
  - from `EmaDecisionSnapshot.entry_dir` (or top-level entry direction for actual entry attempts)
  - produced by entry mode logic:
    - `ema_entry_mode=trend`: requires streak/confirm bars in the same `state`
    - `ema_entry_mode=cross`: requires crossover (with optional confirm bars)
  - then can be nulled by regime gates.

- `regime`
  - from `EmaDecisionSnapshot.regime_dir`
  - the current primary bias direction (EMA regime or Supertrend regime path)
  - with `regime_ready` showing whether that gate is active/valid yet.

- `shock`
  - from shock engine snapshot (`shock=True/False`)
  - detector-specific ON/OFF state with hysteresis thresholds.

- `shock_dir`
  - from `_ShockDirectionMixin` inside shock engines
  - sign of summed recent close-to-close returns over `shock_direction_lookback` bars.

- `dirΣ` (`shock_dir_ret_sum_pct`)
  - summed return over lookback, in percent
  - this is the direct numeric basis for `shock_dir`.

## 3) Why `entry` is not a clone of `state`

`state` is just “which side are EMAs on right now”.

`entry` adds additional logic layers:
- entry mode (`trend` vs `cross`)
- confirm-bar requirements (`entry_confirm_bars`)
- primary regime alignment (`regime_mode` + readiness)
- secondary regime alignment (`regime2_mode`, with `regime2_apply_to`)
- dual-branch/RATS-V branch selection (when enabled)
- tick gate clamping (if enabled)

So even if `state=down`, `entry` can be `None` for many valid reasons:
- no fresh cross in cross mode
- confirm bars not met
- regime says opposite side
- regime gate not ready
- regime2 blocked only this side
- branch filters failed
- tick gate blocked direction

## 4) Why `regime` and `shock_dir` are often different

They are different signals by design:

- `regime_dir`
  - medium-structure trend/bias gate
  - EMA regime or Supertrend regime
  - can run on signal TF or separate MTF regime bars.

- `shock_dir`
  - short-horizon return-sign momentum of raw closes
  - based on rolling sum over small lookback (`shock_direction_lookback`, default 2)
  - can be updated from regime-side updates or signal-side updates (`shock_direction_source`).

They can diverge when:
- market is noisy/choppy (short-term return sign flips while regime remains stable)
- regime uses MTF bars but shock direction uses signal bars
- regime is smoothed (EMA/ST), while shock_dir is direct return sign
- shock direction source is `signal` but regime gate is driven off exec/regime updates.

## 5) “Return sign” in `_ShockDirectionMixin`: exact meaning

Inside `tradebot/engine.py`, `_ShockDirectionMixin` does:

1. Per update, compute close-to-close return:
   - `r_t = close_t / close_{t-1} - 1`

2. Keep a rolling deque of last `N = shock_direction_lookback` returns.

3. Once deque is full, sum:
   - `R_sum = Σ r_t`
   - `direction_ret_sum_pct = 100 * R_sum`

4. Direction is sign of that sum:
   - `R_sum > 0` => `direction = "up"`
   - `R_sum < 0` => `direction = "down"`
   - `R_sum == 0` => `direction = None`

5. `direction_ready` is true only when:
   - direction is `up/down`, and
   - deque length reached lookback.

Important consequences:
- It is not ATR direction.
- It is not regime direction.
- It is not entry direction.
- It is not a price-level trendline; it is a sign of summed recent returns.

## 6) What “bias.shock is computed from it” means

The journal “bias” bundle includes shock information, but it is two separate things:

- `shock` bool comes from detector hysteresis state:
  - ATR ratio / TR ratio / daily ATR% / daily drawdown
  - ON when crossing ON threshold, OFF when crossing OFF threshold.

- `shock_dir` comes from `_ShockDirectionMixin` return-sign sum.

So `bias` can show:
- `shock=on`, `shock_dir=up`
- `shock=on`, `shock_dir=down`
- `shock=off`, `shock_dir=up/down` (if direction stream still valid)

That is expected and intentional.

## 7) Primary signal/gate pipeline (actual order)

For each new signal bar (`SpotSignalEvaluator.update_signal_bar`):

1. Update signal engine
   - EMA or ORB
   - produce base `state`, `entry_dir`, cross flags.

2. Apply primary regime gate
   - `regime_mode=ema` or `supertrend`
   - optional shock/cooling supertrend variants can modify regime gating behavior.

3. Optionally apply `shock_regime_override_dir`
   - if enabled and shock ON with direction ready, regime dir can be overridden by shock direction.

4. Apply secondary regime gate (`regime2`)
   - `regime2_mode=ema|supertrend|off`
   - only applied on selected sides via `regime2_apply_to`.

5. Reconcile branch-selected entry direction
   - branch-A/B selection + RATS-V branch thresholds (if dual branch enabled)
   - final entry direction for entries can be narrower than signal’s internal fields.

6. Build shock view
   - produce `shock` bool
   - `shock_dir`
   - scale stream (`shock_atr_pct`), optionally from alternate `shock_scale_detector`.

7. Derive shock telemetry
   - drawdown distances/velocity/accel
   - direction streaks
   - prearm streak
   - ATR velocity/acceleration
   - optional shock-ramp nodes (`up`/`down`).

8. Emit `SpotSignalSnapshot`
   - includes both raw signal object and higher-level fields used by lifecycle/sizing.

## 8) Entry filter stack order (`gate_vector`)

Signal filters are evaluated in fixed order:
- `rv`
- `time`
- `skip_first`
- `cooldown`
- `shock_gate`
- `permission`
- `volume`

So when you see `gate_vector`, this is exactly the stack being applied.

## 9) Shock gate modes (entry filter behavior)

`shock_gate_mode` semantics:
- `off`: ignore shock in filter stage.
- `detect`: compute shock only; do not block entries.
- `block`: block all entries while shock ON.
- `block_longs`: block only long entries during shock.
- `block_shorts`: block only short entries during shock.
- `surf`: during shock, allow only entries aligned with `shock_dir`.

## 10) Flip-exit gate mode semantics

`flip_exit_gate_mode` controls whether an EMA flip exit is allowed only under extra conditions:
- `off`
- `regime`
- `permission`
- `regime_or_permission`
- `regime_and_permission`

Note: in current implementation, “blocked” means the selected condition evaluates true for the active trade direction, so this mode is effectively an additional arbitration layer on top of flip detection.

## 11) Sizing pipeline order (critical)

In `SpotPolicy.calc_signed_qty_with_trace`, order is:

1. Base qty from sizing mode
   - `fixed`, `notional_pct`, or `risk_pct`.

2. Risk overlays and directional factors
   - `riskoff`, `riskpanic`, `riskpop`
   - long/short factors
   - optional riskpanic linear scaling
   - graph overlay multipliers.

3. Shock multipliers
   - long factors (`shock_long_risk_mult_factor`, `_down`)
   - long boost gates (`shock_long_boost_*`)
   - short boost gates (`shock_short_*`)
   - short prearm (`shock_prearm_*`).

4. Shock risk scaling
   - uses `shock_risk_scale_target_atr_pct`
   - floor via `shock_risk_scale_min_mult`
   - applies to `risk`, `cap`, or `both`.

5. Liquidity boost (`liq_boost_*`)
   - optional score-based risk multiplier
   - optional alignment/shock requirements
   - optional cap floor.

6. Shock ramp (`shock_ramp_*`)
   - directional ramp node (`up/down/both`)
   - can boost risk and set cap floor fraction.

7. Caps + affordability
   - notional cap (`spot_max_notional_pct`, plus cap multipliers)
   - cash affordability for buys
   - `spot_max_qty` clamp.

8. Lot rounding and min gate
   - round to lot
   - enforce `spot_min_qty`
   - return 0 if below minimum effective qty.

9. Branch size multiplier (if dual-branch enabled)
   - branch A/B size multipliers applied after sizing result.

## 11.1) Shock long/short/prearm gates (exact semantics)

These three families are easy to confuse because they all live in sizing but trigger under different states.

### Long side (`BUY`) in shock

When `shock=True` and `shock_dir` is available:
- Base long factor:
  - if `shock_dir=up`: use `shock_long_risk_mult_factor`
  - if `shock_dir=down`: use `shock_long_risk_mult_factor_down`
- If `shock_dir=up` and base long factor is `> 1.0`, an optional rebound gate is enforced:
  - `shock_long_boost_max_dist_off_pp`:
    - requires derived `shock_drawdown_dist_off_pct` to be in `[-max_dist, 0]`
    - this means: still in shock-on territory, but close to OFF threshold (recovery zone)
  - optional alignment:
    - `shock_long_boost_require_regime_up` -> require `signal_regime_dir=up`
    - `shock_long_boost_require_entry_up` -> require `signal_entry_dir=up`
- If the boost gate fails, the multiplier is forced back to `1.0` for that boost path.

Trace diagnostics:
- `shock_long_boost_applied` (bool)
- `shock_long_boost_gate_reason`
  - `ok`
  - `dd_dist_off_missing`
  - `dd_dist_off_invalid`
  - `dd_dist_off_gt_0`
  - `dd_dist_off_abs_gt_{N}pp`
  - `regime_not_up`
  - `entry_not_up`

### Short side (`SELL`) in shock

When `shock=True` and `shock_dir=down`, short boost gating runs:
- `shock_short_boost_min_down_streak_bars`
  - requires `shock_dir_down_streak_bars >= min_streak`
- optional drawdown-distance gate:
  - `shock_short_boost_max_dist_on_pp > 0` requires
    - `shock_drawdown_dist_on_pct` present
    - `0 <= dist_on <= max_dist`
- optional alignment:
  - `shock_short_boost_require_regime_down`
  - `shock_short_boost_require_entry_down`
- If all pass, apply `shock_short_risk_mult_factor`.

Trace diagnostics:
- `shock_short_boost_applied` (bool)
- `shock_short_boost_gate_reason`
  - `ok`
  - `down_streak_lt_{N}`
  - `dd_dist_on_missing`
  - `dd_dist_on_invalid`
  - `dd_dist_on_lt_0`
  - `dd_dist_on_gt_{N}pp`
  - `regime_not_down`
  - `entry_not_down`
  - `shock_not_down`
  - `shock_off`

### Short prearm (`SELL`) while shock is OFF

Prearm is explicitly an OFF-state mechanism. It runs only on the short path when `shock=False`.

Core idea:
- allow controlled short-risk boost near the ON threshold before detector flips ON.

Key knobs:
- `shock_prearm_dist_on_max_pp`:
  - near-band width around ON threshold.
- `shock_prearm_min_streak_bars`:
  - when `>0`, enables streak-latch behavior (uses `shock_prearm_down_streak_bars`).
- `shock_prearm_min_dist_on_vel_pp` and `shock_prearm_min_dist_on_accel_pp`:
  - only enforced in non-latched mode.
- `shock_prearm_require_regime_down`
- `shock_prearm_require_entry_down`
- `shock_prearm_short_risk_mult_factor`

Behavior:
- If `shock_prearm_dist_on_max_pp <= 0`, feature is effectively disabled.
- Non-latched mode (`shock_prearm_min_streak_bars == 0`) requires:
  - `dist_on` in `[-near_band, 0)`
  - `dist_on_vel` threshold
  - optional `dist_on_accel` threshold.
- Latched mode (`shock_prearm_min_streak_bars > 0`) uses streak gating and wider distance acceptance:
  - rejects only when `dist_on < -near_band`
  - then requires `prearm_streak >= min_streak`.
- If all gates pass and factor is positive, apply `shock_prearm_short_risk_mult_factor`.

Trace diagnostics:
- `shock_prearm_applied` (bool)
- `shock_prearm_factor`
- `shock_prearm_reason`
  - `off`
  - `disabled`
  - `dist_on_missing`
  - `outside_band`
  - `dist_vel_missing`
  - `dist_vel_low`
  - `dist_accel_missing`
  - `dist_accel_low`
  - `streak_missing`
  - `streak_lt_{N}`
  - `regime_not_down`
  - `entry_not_down`
  - `ok`
  - `factor_nonpositive`

### Distances used by these gates

These names are easy to misread:
- `shock_drawdown_dist_on_pct`
  - distance to ON threshold; derived so that `>=0` means at/through ON threshold.
- `shock_drawdown_dist_off_pct`
  - distance to OFF threshold; derived so that `<=0` means still below OFF threshold, `0` is exact OFF threshold, `>0` is already above OFF threshold.

## 12) Resize behavior (not a separate position model)

Resize uses target intent logic from current qty -> target qty:
- `spot_resize_mode=off` => no resizing (except enter/exit intents)
- `spot_resize_mode=target` => allow scale-in/out deltas
- gates:
  - `spot_resize_allow_scale_in`
  - `spot_resize_allow_scale_out`
  - `spot_resize_min_delta_qty`
  - `spot_resize_max_step_qty`
  - `spot_resize_cooldown_bars`

Graph resize policy can transform base target before intent resolution.

## 13) Policy graph and profile system

Graph profiles define coherent node sets:
- `neutral`
- `defensive`
- `aggressive`
- `hf_probe`

Nodes can be overridden explicitly:
- `spot_entry_policy`
- `spot_exit_policy`
- `spot_resize_policy`
- `spot_risk_overlay_policy`

Default node families:
- Entry: `default`, `slope_tr_guard`
- Exit: `priority`, `slope_flip_guard`
- Resize: `adaptive`, `adaptive_atr_defensive`, `adaptive_hybrid_aggressive`, `adaptive_slope_probe`
- Risk overlay: `legacy`, `atr_compress`, `trend_bias`

## 14) Knob precedence rules (important)

When policy/runtime views are built:

1. explicit `strategy` / `filters` value
2. else `spot_policy_pack` defaults
3. else hardcoded fallback in parser/policy

So pack defaults never override explicit knobs.

## 15) Extra runtime keys that are real but not config-schema knobs

The following keys are consumed in runtime adapters even though they are not core schema fields in `StrategyConfigBase`:
- `signal_use_rth`
- `spot_sec_type`
- graph-node alias keys:
  - `spot_graph_entry_policy`
  - `spot_graph_exit_policy`
  - `spot_graph_resize_policy`
  - `spot_graph_risk_overlay_policy`

Also detector alias inputs exist in filters parsing:
- `shock_tr_fast_period` / `shock_tr_slow_period` (TR detector aliases)
- `shock_min_tr_pct` alias for TR detector minimum percent.

Sizing also consumes runtime telemetry keys (not static schema knobs) for shock gates:
- `shock_dir_down_streak_bars`
- `shock_drawdown_dist_on_pct`
- `shock_drawdown_dist_on_vel_pp`
- `shock_drawdown_dist_on_accel_pp`
- `shock_prearm_down_streak_bars`
- derived in policy: `shock_drawdown_dist_off_pct`

## 16) Practical sanity-check heuristics

When diagnosing logs, read in this order:

1. `state`
   - Is EMA state directional yet?

2. `entry`
   - Did entry mode + confirm produce a candidate?

3. `regime` + `ready`
   - Is primary bias gate active and aligned?

4. `regime2` (if enabled)
   - Did secondary bias null that side?

5. `shock` + `shock_dir` + `dirΣ`
   - Is volatility stress ON, and what does short-horizon return sign say?

6. `gate_vector`
   - Which exact filter blocked the candidate?

7. lifecycle decision
   - blocked vs enter vs deferred next-open vs pending.

8. sizing trace
   - where quantity got cut/boosted (risk overlay, shock scale, caps, floors, min-qty gate).

## 17) Parser normalization and alias behavior (high-value details)

These normalization rules are why knobs sometimes appear to behave differently than raw JSON strings suggest.

- `ema_entry_mode`
  - accepted aliases:
    - `trend`, `state` -> `trend`
    - `cross`, `crossover` -> `cross`
  - default -> `trend`.

- `flip_exit_mode`
  - accepted aliases:
    - `entry`, `default`, `same`, `auto` -> `entry`
    - `state`, `trend` -> `state`
    - `cross`, `crossover` -> `cross`
  - default -> `entry`.

- `flip_exit_gate_mode`
  - accepted aliases:
    - `bias` -> `regime`
    - `perm` -> `permission`
    - `bias_or_perm` -> `regime_or_permission`
    - `bias_and_perm` -> `regime_and_permission`
  - `off` aliases: `none`, `disabled`, `false`, `0`, empty, `default`.

- `regime_mode`
  - `ema` or `supertrend` (`st` alias); default `ema`.

- `regime2_mode`
  - `off`, `ema`, `supertrend` (`st` alias).
  - if `ema` is selected but `regime2_ema_preset` is missing, runtime forces `regime2_mode=off`.

- `regime2_apply_to`
  - aliases:
    - `both`, `all`, empty -> `both`
    - `long`, `up`, `buy` -> `longs`
    - `short`, `down`, `sell` -> `shorts`.

- `tick_gate_mode`
  - aliases:
    - `raschke`, `tick`, `tick_width`, `tickwidth` -> `raschke`
    - off aliases -> `off`.

- `spot_fill_mode` knobs (`spot_entry_fill_mode`, `spot_flip_exit_fill_mode`)
  - canonical:
    - `close`
    - `next_bar`
    - `next_tradable_bar`
  - aliases include:
    - `next_open`, `open`, `nextbaropen` -> `next_tradable_bar`
    - `nextbar` -> `next_bar`
    - `bar_close`, `at_close` -> `close`.

- `spot_next_open_session`
  - canonical:
    - `auto`, `rth`, `extended`, `always`, `tradable_24x5`
  - aliases:
    - `full24`, `tradable`, `overnight_plus_extended` -> `tradable_24x5`.

- `spot_mark_to_market`
  - `close` aliases: `mid`, `bar_close`
  - `liquidation` aliases: `liq`, `bidask`, `bid_ask`, `bid-ask`.

- `spot_drawdown_mode`
  - `close` or `intrabar` (`intra`, `ohlc` aliases).

- `spot_sizing_mode`
  - canonical: `fixed`, `notional_pct`, `risk_pct`
  - common aliases normalized accordingly.

## 18) Shock-family coercion and clamping (important for “why is this value different?”)

In filters parsing, several safety clamps are applied:

- `shock_gate_mode`
  - booleans are converted:
    - `true` -> `block`
    - `false` -> `off`.

- `shock_detector` and `shock_scale_detector`
  - canonical set: `atr_ratio`, `tr_ratio`, `daily_atr_pct`, `daily_drawdown`
  - many aliases are normalized to one of those.

- Hysteresis consistency
  - daily ATR detector:
    - if `off > on`, parser sets `off = on`.
  - daily drawdown detector:
    - if `off < on` (more negative), parser sets `off = on`.

- Risk-scale clamp
  - `shock_risk_scale_min_mult` is clamped to `[0.0, 1.0]`.
  - `shock_risk_scale_apply_to` is normalized to one of:
    - `risk`, `cap`, `both`.

- Cap-floor clamps
  - `liq_boost_cap_floor_frac` and `shock_ramp_max_cap_floor_frac` are clamped to `[0.0, 1.0]`.

- Positive-only thresholds
  - many slope/ATR/TR threshold knobs are coerced to `None` when non-positive.

## 19) Tick gate mechanics (why directional entries get nulled)

With `tick_gate_mode=raschke`, the gate computes a market-width regime from tick bars:

1. Build moving upper/lower envelopes from tick highs/lows (`tick_band_ma_period`).
2. Compute width = upper - lower.
3. Compute width z-score over rolling window (`tick_width_z_lookback`).
4. Compute width slope/delta over `tick_width_slope_lookback`.
5. State transitions:
   - neutral -> wide if `z >= tick_width_z_enter` and slope positive
   - neutral -> narrow if `z <= -tick_width_z_enter` and slope negative
   - wide -> neutral when z drops below `tick_width_z_exit`
   - narrow -> neutral when z rises above `-tick_width_z_exit`.
6. Direction mapping:
   - wide -> `up`
   - narrow -> `down` (or blocked when `tick_direction_policy=wide_only`)
   - neutral -> no tick direction.
7. Final entry clamp:
   - if tick direction conflicts with entry direction, entry is nulled
   - if tick not ready and `tick_neutral_policy=block`, entry is nulled.

## 20) One concrete divergence scenario (so this feels real)

Example:
- `ema_entry_mode=trend`
- `state=down`
- `regime=down`
- `regime2=off`
- `shock=on`
- `shock_dir=up`

This can happen when:
- medium trend is still down (EMA/supertrend regime),
- but the last 2 close-to-close returns are net positive (`shock_direction_lookback=2`), so `shock_dir=up`.

No contradiction exists here. It means:
- trend gate says “still bearish structure,”
- short-horizon shock return sign says “recent recoil upward.”

That is exactly the sort of separation the design intentionally preserves.

---

## 21) Current SLV-HF champion sizing map (v27 dd_lb10_on10_off5 depth1p25 streak1)

This section supersedes the older v18 map and tracks the current promoted SLV HF champion.

As of **February 22, 2026**, `backtests/slv/readme-hf.md` points to:
- preset: `backtests/slv/archive/champion_history_20260214/slv_hf_champions_v27_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_20260222.json`
- source sweep: `backtests/slv/slv_hf_v27_short_boost_maxdist_sweetspot_20260222.json`
- strategy row: group `0`, entry `0`

Important merge rule for replayed milestone payloads:
- effective filters are `group.filters` merged with `strategy.filters`
- `strategy.filters` overrides any duplicate keys from `group.filters`

This is intentionally practical and specific, so you can answer:
- "Will this scale into meaningful size?"
- "Which knobs are actually active vs just present in schema?"
- "When shock or risk scaling is ON, what changes in size?"

### 21.1) Resolved graph/runtime nodes for current champion

Resolved via `SpotPolicyGraph.from_sources(strategy, filters)`:
- `profile = neutral`
- `entry_policy = default`
- `exit_policy = priority`
- `resize_policy = adaptive_hybrid_aggressive`
- `risk_overlay_policy = atr_compress`

Interpretation:
- entry graph gate is permissive by default (`default` node returns allow=true)
- resize and risk overlay are non-trivial and do affect sizing

### 21.2) Active, size-moving knobs (current values)

Core sizing:
- `spot_sizing_mode = risk_pct`
- `spot_risk_pct = 0.019`
- `spot_short_risk_mult = 0.0384`
- `spot_max_notional_pct = 0.85`
- `spot_min_qty = 1`
- `spot_max_qty = 0` (no explicit hard qty ceiling)
- `spot_stop_loss_pct = 0.0192`

Branch sizing:
- `spot_dual_branch_enabled = true`
- `spot_branch_a_size_mult = 0.6`
- `spot_branch_b_size_mult = 1.2`
- `spot_dual_branch_priority = b_then_a`

Shock state engine (state + direction context):
- `shock_detector = daily_drawdown`
- `shock_drawdown_lookback_days = 10`
- `shock_on_drawdown_pct = -10.0`
- `shock_off_drawdown_pct = -5.0`
- `shock_direction_source = regime`
- `shock_direction_lookback = 2`
- `shock_on_ratio = 1.4`
- `shock_off_ratio = 1.3`
- `shock_gate_mode = detect`
- `shock_min_atr_pct = 4.0`
- `shock_atr_fast_period = 5`
- `shock_atr_slow_period = 50`

Shock side multipliers and gates:
- `shock_short_risk_mult_factor = 2.0`
- `shock_short_boost_min_down_streak_bars = 1`
- `shock_short_boost_max_dist_on_pp = 1.25`
- `shock_short_boost_require_regime_down = true`
- `shock_short_boost_require_entry_down = true`
- `shock_prearm_short_risk_mult_factor = 1.5`
- `shock_prearm_dist_on_max_pp = 1.0`
- `shock_prearm_min_dist_on_vel_pp = 0.25`
- `shock_prearm_require_regime_down = true`
- `shock_prearm_require_entry_down = true`
- `shock_long_risk_mult_factor = 1.0`
- `shock_long_risk_mult_factor_down = 0.1`

Shock risk scaling:
- `shock_scale_detector = tr_ratio`
- `shock_risk_scale_target_atr_pct = 12.0`
- `shock_risk_scale_min_mult = 0.6`
- `shock_risk_scale_apply_to = both`

Risk overlays and gating:
- `riskoff_mode = directional`
- `riskpanic_long_risk_mult_factor = 0.25`
- `riskpanic_long_scale_mode = linear`
- `riskpanic_neg_gap_ratio_min = 0.65`
- `riskpanic_tr5_med_delta_min_pct = 0.35` (parsed into `riskpanic_long_scale_tr_delta_max_pct`)
- `riskpanic_short_risk_mult_factor = 1.0`
- `risk_entry_cutoff_hour_et = 15`

Graph overlay/ref knobs used by active policies:
- `spot_graph_overlay_tr_ratio_ref = 1.0`
- `spot_graph_overlay_slope_ref_pct = 0.08`
- `spot_graph_overlay_atr_vel_ref_pct = 0.35`
- `spot_graph_overlay_trend_boost_max = 1.25`
- `spot_graph_overlay_trend_floor_mult = 0.75`

Notes:
- `atr_compress` overlay is active and uses defaults when explicit hi-threshold knobs are unset:
  - `atr_hi_pct default = 2.5`
  - `min_mult default = 0.5`
- `adaptive_hybrid_aggressive` resize is active and uses defaults when adaptive min/max/mode are unset:
  - `default_mode = hybrid`
  - `default_min_mult = 0.5`
  - `default_max_mult = 2.25`

### 21.3) Active-but-neutral knobs (parsed, but currently multiplicative no-ops)

These are live in code paths but currently evaluate to neutral multipliers:
- `riskoff_long_risk_mult_factor = 1.0`
- `riskoff_short_risk_mult_factor = 1.0`
- `riskpanic_short_risk_mult_factor = 1.0`
- `riskpop_long_risk_mult_factor = 1.0`
- `riskpop_short_risk_mult_factor = 1.0`
- `shock_stop_loss_pct_mult = 1.0`
- `shock_profit_target_pct_mult = 1.0`

### 21.4) Inert/unset for this champion (schema exists, effect absent)

Examples of families not currently changing behavior:
- `spot_notional_pct` path (`spot_sizing_mode=risk_pct`, so notional sizing path is inactive)
- `shock_short_entry_max_dist_on_pp = 0.0` (depth gate disabled; no short hard-block on this check)
- `shock_prearm_min_dist_on_accel_pp = 0.0` (accel threshold disabled)
- `shock_prearm_min_streak_bars = 0` (prearm latch/streak mode disabled)
- long-shock boost gate family effectively off:
  - `shock_long_boost_max_dist_off_pp = 0.0`
  - `shock_long_boost_require_regime_up = false`
  - `shock_long_boost_require_entry_up = false`
- graph entry guard thresholds unset:
  - `spot_entry_tr_ratio_min`
  - `spot_entry_slope_med_abs_min_pct`
  - `spot_entry_slope_vel_abs_min_pct`
  - `spot_entry_slow_slope_med_abs_min_pct`
  - `spot_entry_slow_slope_vel_abs_min_pct`
- guard threshold scaling family unset:
  - `spot_guard_threshold_scale_mode` and refs
- dynamic flip-hold family unset:
  - `spot_flip_hold_dynamic_*`
- liquidity boost family disabled:
  - `liq_boost_enable = false`

### 21.5) Exact sizing order for this champion (with formulas)

The active sizing order is:

1. Base risk budget:
   - `risk_dollars_base = equity * spot_risk_pct`

2. Long/short branch multipliers:
   - BUY branch:
     - `riskoff`/`riskpanic`/`riskpop` long factors apply when their states are active
     - if `riskpanic_long_scale_mode=linear`, a dynamic long factor can apply in non-panic state
     - shock long factor applies when shock is active (`up -> 1.0`, `down -> 0.1` in this champion)
     - graph overlay long channel factors then apply
   - SELL branch:
     - start with `short_mult = spot_short_risk_mult`
     - optional short entry depth block can run (`shock_short_entry_max_dist_on_pp`) but is disabled here (`0.0`)
     - `riskoff`/`riskpanic`/`riskpop` short factors apply when their states are active
     - if `shock=true` and `shock_dir=down`, short boost gate can apply `shock_short_risk_mult_factor=2.0` when all gates pass
     - if `shock=false`, prearm gate can apply `shock_prearm_short_risk_mult_factor=1.5` when all prearm gates pass
     - graph overlay short channel factors then apply

3. Graph risk overlay (`atr_compress` here):
   - if `shock_metric > atr_hi_pct`, apply compression:
     - `overlay_scale = clamp(atr_hi_pct / shock_metric, min_mult..1.0)`
   - this policy applies compression to both risk and cap channels

4. Shock risk-scale stream (`shock_scale_detector=tr_ratio`):
   - `scale = clamp(target_atr_pct / shock_metric, min_mult..1.0)`
   - with `apply_to=both`:
     - `risk_dollars *= scale`
     - `cap_pct *= scale`

5. Convert risk dollars to qty (risk_pct mode):
   - `desired_qty = floor(risk_dollars / per_share_risk)`
   - `per_share_risk = abs(entry_price - stop_level)`

6. Apply caps:
   - notional cap from `spot_max_notional_pct` and overlay/scaling cap multipliers
   - cash affordability (buy side)
   - `spot_max_qty` if greater than 0

7. Lot/min handling:
   - lot rounding
   - enforce `spot_min_qty`
   - may return 0 if minimum constraints fail

8. Branch multiplier (post-sizing):
   - apply `spot_branch_a_size_mult` or `spot_branch_b_size_mult` if dual-branch is enabled

9. Resize policy transforms target qty:
   - `adaptive_hybrid_aggressive` can scale target in default `[0.5x, 2.25x]` band
   - this can apply through target-intent resolution, including flat-to-open transitions

Practical caveat:
- cap clamps happen before branch/resize transforms; branch and resize can still move final target afterward

### 21.6) Current champion rough exposure envelope (first-order intuition)

Using `spot_risk_pct=0.019` and `spot_stop_loss_pct=0.0192` as a rough pct-stop approximation:

- Long baseline (no shock-down suppression):
  - effective risk ~= `1.90%` equity
  - implied notional ~= `1.90 / 1.92 ~= 98.96%` equity
  - then capped by `spot_max_notional_pct=85%` before later branch/resize transforms

- Long in shock-down:
  - effective risk ~= `1.90% * 0.1 = 0.19%` equity
  - implied notional ~= `0.19 / 1.92 ~= 9.90%` equity

- Short baseline (shock off, no prearm):
  - effective risk ~= `1.90% * 0.0384 = 0.0730%` equity
  - implied notional ~= `0.0730 / 1.92 ~= 3.80%` equity

- Short, prearm applied (shock off + prearm gates pass):
  - effective risk ~= `1.90% * 0.0384 * 1.5 = 0.1094%` equity
  - implied notional ~= `0.1094 / 1.92 ~= 5.70%` equity

- Short, shock down + short boost gates pass:
  - effective risk ~= `1.90% * 0.0384 * 2.0 = 0.1459%` equity
  - implied notional ~= `0.1459 / 1.92 ~= 7.60%` equity

These are first-order approximations. Final qty still depends on:
- stop-level geometry (actual per-share risk)
- cap/cash clamps
- shock/overlay scaling
- branch multiplier
- resize policy transform

### 21.7) Shock state vs scaling stream (decoupled by design)

For this champion:
- shock ON/OFF state is driven by `shock_detector=daily_drawdown`
- size scaling stream is driven by `shock_scale_detector=tr_ratio`

So logs can correctly show:
- drawdown-based `shock=true`
- plus a separate scaling metric (`shock.atr_pct` payload field) from TR-ratio detector stream

This is expected, not contradictory:
- state detector answers "are we in stress?"
- scale detector answers "how much should we compress risk/cap right now?"

### 21.8) What can increase size, decrease size, or block entries

Increase size:
- branch multiplier `spot_branch_b_size_mult=1.2` (branch B)
- short prearm multiplier `shock_prearm_short_risk_mult_factor=1.5` (when prearm gates pass while shock is off)
- short shock boost multiplier `shock_short_risk_mult_factor=2.0` (when shock-down gates pass)
- adaptive resize policy (`adaptive_hybrid_aggressive`, up to default 2.25x target transform)

Decrease size:
- `spot_short_risk_mult=0.0384` (major base short-side compression)
- `shock_long_risk_mult_factor_down=0.1` (long compression in shock-down)
- `riskpanic_long_risk_mult_factor=0.25` and linear panic scaling path
- `atr_compress` overlay when metric exceeds threshold
- shock risk-scale when scaling metric exceeds target
- branch A multiplier `0.6`

Block entries:
- regular filter stack (`rv/time/skip_first/cooldown/shock_gate/permission/volume`)
- regime/regime2 gates
- lifecycle gating (capacity, allowed directions, next-open timing)
- `risk_entry_cutoff_hour_et=15` entry-time cut
- optional short depth hard-gate if enabled in future (`shock_short_entry_max_dist_on_pp > 0`)

Note on `shock_gate_mode=detect`:
- detect mode computes shock and exposes telemetry, but does not block entries by itself

## Appendix A: Exhaustive knob catalogs

The sections below are generated from the actual schema/dataclass definitions so the knob list is complete.


### A1) Backtest Section (`backtest`) (11 knobs)
| knob | default |
|---|---|
| `start` | `<required>` |
| `end` | `<required>` |
| `bar_size` | `'1 hour'` |
| `use_rth` | `False` |
| `starting_cash` | `100000.0` |
| `risk_free_rate` | `0.02` |
| `cache_dir` | `'db'` |
| `calibration_dir` | `'db/calibration'` |
| `output_dir` | `'backtests/out'` |
| `calibrate` | `False` |
| `offline` | `False` |

### A2) Strategy Section (`strategy`) (150 knobs)
| knob | default |
|---|---|
| `name` | `'credit_spread'` |
| `instrument` | `'options'` |
| `symbol` | `'MNQ'` |
| `exchange` | `None` |
| `right` | `'PUT'` |
| `entry_days` | `[]` |
| `max_entries_per_day` | `1` |
| `dte` | `0` |
| `otm_pct` | `2.5` |
| `width_pct` | `1.0` |
| `profit_target` | `0.5` |
| `stop_loss` | `0.35` |
| `exit_dte` | `0` |
| `quantity` | `1` |
| `stop_loss_basis` | `'max_loss'` |
| `min_credit` | `None` |
| `ema_preset` | `None` |
| `ema_entry_mode` | `None` |
| `entry_confirm_bars` | `0` |
| `regime_ema_preset` | `None` |
| `regime_bar_size` | `None` |
| `ema_directional` | `False` |
| `exit_on_signal_flip` | `False` |
| `flip_exit_mode` | `None` |
| `flip_exit_gate_mode` | `None` |
| `flip_exit_min_hold_bars` | `0` |
| `flip_exit_only_if_profit` | `False` |
| `spot_controlled_flip` | `False` |
| `direction_source` | `None` |
| `directional_legs` | `None` |
| `directional_spot` | `None` |
| `legs` | `None` |
| `filters` | `None` |
| `spot_profit_target_pct` | `None` |
| `spot_stop_loss_pct` | `None` |
| `spot_close_eod` | `False` |
| `entry_signal` | `None` |
| `orb_window_mins` | `15` |
| `orb_risk_reward` | `2.0` |
| `orb_target_mode` | `None` |
| `orb_open_time_et` | `None` |
| `spot_exit_mode` | `None` |
| `spot_atr_period` | `14` |
| `spot_pt_atr_mult` | `1.5` |
| `spot_sl_atr_mult` | `1.0` |
| `spot_exit_time_et` | `None` |
| `spot_exec_bar_size` | `None` |
| `regime_mode` | `None` |
| `regime2_mode` | `None` |
| `regime2_apply_to` | `None` |
| `regime2_ema_preset` | `None` |
| `regime2_bar_size` | `None` |
| `regime2_supertrend_atr_period` | `10` |
| `regime2_supertrend_multiplier` | `3.0` |
| `regime2_supertrend_source` | `None` |
| `supertrend_atr_period` | `10` |
| `supertrend_multiplier` | `3.0` |
| `supertrend_source` | `None` |
| `tick_gate_mode` | `None` |
| `tick_gate_symbol` | `'TICK-NYSE'` |
| `tick_gate_exchange` | `'NYSE'` |
| `tick_band_ma_period` | `10` |
| `tick_width_z_lookback` | `252` |
| `tick_width_z_enter` | `1.0` |
| `tick_width_z_exit` | `0.5` |
| `tick_width_slope_lookback` | `3` |
| `tick_neutral_policy` | `None` |
| `tick_direction_policy` | `None` |
| `spot_entry_fill_mode` | `'close'` |
| `spot_flip_exit_fill_mode` | `'close'` |
| `spot_next_open_session` | `'auto'` |
| `spot_intrabar_exits` | `False` |
| `spot_spread` | `0.0` |
| `spot_commission_per_share` | `0.0` |
| `spot_commission_min` | `0.0` |
| `spot_slippage_per_share` | `0.0` |
| `spot_mark_to_market` | `None` |
| `spot_drawdown_mode` | `None` |
| `spot_sizing_mode` | `None` |
| `spot_notional_pct` | `0.0` |
| `spot_risk_pct` | `0.0` |
| `spot_short_risk_mult` | `1.0` |
| `spot_max_notional_pct` | `1.0` |
| `spot_min_qty` | `1` |
| `spot_max_qty` | `0` |
| `spot_dual_branch_enabled` | `False` |
| `spot_dual_branch_priority` | `None` |
| `spot_branch_a_ema_preset` | `None` |
| `spot_branch_a_entry_confirm_bars` | `None` |
| `spot_branch_a_min_signed_slope_pct` | `None` |
| `spot_branch_a_max_signed_slope_pct` | `None` |
| `spot_branch_a_size_mult` | `1.0` |
| `spot_branch_b_ema_preset` | `None` |
| `spot_branch_b_entry_confirm_bars` | `None` |
| `spot_branch_b_min_signed_slope_pct` | `None` |
| `spot_branch_b_max_signed_slope_pct` | `None` |
| `spot_branch_b_size_mult` | `1.0` |
| `spot_policy_pack` | `None` |
| `spot_policy_graph` | `None` |
| `spot_graph_profile` | `None` |
| `spot_entry_policy` | `None` |
| `spot_exit_policy` | `None` |
| `spot_resize_policy` | `None` |
| `spot_risk_overlay_policy` | `None` |
| `spot_resize_mode` | `None` |
| `spot_resize_min_delta_qty` | `1` |
| `spot_resize_max_step_qty` | `0` |
| `spot_resize_allow_scale_in` | `True` |
| `spot_resize_allow_scale_out` | `True` |
| `spot_resize_cooldown_bars` | `0` |
| `spot_resize_adaptive_mode` | `None` |
| `spot_resize_adaptive_min_mult` | `0.5` |
| `spot_resize_adaptive_max_mult` | `1.75` |
| `spot_resize_adaptive_atr_target_pct` | `None` |
| `spot_resize_adaptive_atr_vel_ref_pct` | `0.4` |
| `spot_resize_adaptive_slope_ref_pct` | `0.1` |
| `spot_resize_adaptive_vel_ref_pct` | `0.08` |
| `spot_resize_adaptive_tr_ratio_ref` | `1.0` |
| `spot_entry_tr_ratio_min` | `None` |
| `spot_entry_slope_med_abs_min_pct` | `None` |
| `spot_entry_slope_vel_abs_min_pct` | `None` |
| `spot_entry_slow_slope_med_abs_min_pct` | `None` |
| `spot_entry_slow_slope_vel_abs_min_pct` | `None` |
| `spot_entry_shock_atr_max_pct` | `None` |
| `spot_entry_atr_vel_min_pct` | `None` |
| `spot_entry_atr_accel_min_pct` | `None` |
| `spot_guard_threshold_scale_mode` | `None` |
| `spot_guard_threshold_scale_min_mult` | `0.7` |
| `spot_guard_threshold_scale_max_mult` | `1.8` |
| `spot_guard_threshold_scale_tr_ref` | `None` |
| `spot_guard_threshold_scale_atr_vel_ref_pct` | `None` |
| `spot_guard_threshold_scale_tr_median_ref_pct` | `None` |
| `spot_exit_flip_hold_slope_min_pct` | `None` |
| `spot_exit_flip_hold_tr_ratio_min` | `None` |
| `spot_exit_flip_hold_slow_slope_min_pct` | `None` |
| `spot_exit_flip_hold_slope_vel_min_pct` | `None` |
| `spot_exit_flip_hold_slow_slope_vel_min_pct` | `None` |
| `spot_flip_hold_dynamic_mode` | `None` |
| `spot_flip_hold_dynamic_min_mult` | `0.5` |
| `spot_flip_hold_dynamic_max_mult` | `2.5` |
| `spot_flip_hold_dynamic_tr_ref` | `None` |
| `spot_flip_hold_dynamic_atr_vel_ref_pct` | `None` |
| `spot_flip_hold_dynamic_tr_median_ref_pct` | `None` |
| `spot_graph_overlay_atr_hi_pct` | `None` |
| `spot_graph_overlay_atr_hi_min_mult` | `0.5` |
| `spot_graph_overlay_atr_vel_ref_pct` | `0.4` |
| `spot_graph_overlay_tr_ratio_ref` | `1.0` |
| `spot_graph_overlay_slope_ref_pct` | `0.08` |
| `spot_graph_overlay_trend_boost_max` | `1.35` |
| `spot_graph_overlay_trend_floor_mult` | `0.65` |

### A3) Synthetic Section (`synthetic`) (7 knobs)
| knob | default |
|---|---|
| `rv_lookback` | `60` |
| `rv_ewma_lambda` | `0.94` |
| `iv_risk_premium` | `1.2` |
| `iv_floor` | `0.05` |
| `term_slope` | `0.02` |
| `skew` | `-0.25` |
| `min_spread_pct` | `0.1` |

### A4) Filters Object (`strategy.filters` / `FiltersConfig`) (134 knobs)
| knob | default |
|---|---|
| `rv_min` | `<required>` |
| `rv_max` | `<required>` |
| `ema_spread_min_pct` | `<required>` |
| `ema_slope_min_pct` | `<required>` |
| `entry_start_hour` | `<required>` |
| `entry_end_hour` | `<required>` |
| `skip_first_bars` | `<required>` |
| `cooldown_bars` | `<required>` |
| `entry_start_hour_et` | `None` |
| `entry_end_hour_et` | `None` |
| `volume_ema_period` | `None` |
| `volume_ratio_min` | `None` |
| `ema_spread_min_pct_down` | `None` |
| `ema_slope_signed_min_pct_up` | `None` |
| `ema_slope_signed_min_pct_down` | `None` |
| `shock_gate_mode` | `'off'` |
| `shock_detector` | `'atr_ratio'` |
| `shock_scale_detector` | `None` |
| `shock_atr_fast_period` | `7` |
| `shock_atr_slow_period` | `50` |
| `shock_on_ratio` | `1.55` |
| `shock_off_ratio` | `1.3` |
| `shock_min_atr_pct` | `7.0` |
| `shock_daily_atr_period` | `14` |
| `shock_daily_on_atr_pct` | `13.0` |
| `shock_daily_off_atr_pct` | `11.0` |
| `shock_daily_on_tr_pct` | `None` |
| `shock_drawdown_lookback_days` | `20` |
| `shock_on_drawdown_pct` | `-20.0` |
| `shock_off_drawdown_pct` | `-10.0` |
| `shock_short_risk_mult_factor` | `1.0` |
| `shock_short_boost_min_down_streak_bars` | `1` |
| `shock_short_boost_require_regime_down` | `False` |
| `shock_short_boost_require_entry_down` | `False` |
| `shock_short_boost_max_dist_on_pp` | `0.0` |
| `shock_prearm_dist_on_max_pp` | `0.0` |
| `shock_prearm_min_drawdown_pct` | `0.0` |
| `shock_prearm_min_dist_on_vel_pp` | `0.0` |
| `shock_prearm_min_dist_on_accel_pp` | `0.0` |
| `shock_prearm_min_streak_bars` | `0` |
| `shock_prearm_short_risk_mult_factor` | `1.0` |
| `shock_prearm_require_regime_down` | `True` |
| `shock_prearm_require_entry_down` | `True` |
| `shock_long_risk_mult_factor` | `1.0` |
| `shock_long_risk_mult_factor_down` | `1.0` |
| `shock_long_boost_require_regime_up` | `False` |
| `shock_long_boost_require_entry_up` | `False` |
| `shock_long_boost_max_dist_off_pp` | `0.0` |
| `shock_stop_loss_pct_mult` | `1.0` |
| `shock_profit_target_pct_mult` | `1.0` |
| `shock_direction_lookback` | `2` |
| `shock_direction_source` | `'regime'` |
| `shock_regime_override_dir` | `False` |
| `shock_regime_supertrend_multiplier` | `None` |
| `shock_cooling_regime_supertrend_multiplier` | `None` |
| `shock_daily_cooling_atr_pct` | `None` |
| `shock_risk_scale_target_atr_pct` | `None` |
| `shock_risk_scale_min_mult` | `0.2` |
| `shock_risk_scale_apply_to` | `'risk'` |
| `shock_ramp_enable` | `False` |
| `shock_ramp_apply_to` | `'down'` |
| `shock_ramp_max_risk_mult` | `1.0` |
| `shock_ramp_max_cap_floor_frac` | `0.0` |
| `shock_ramp_min_slope_streak_bars` | `0` |
| `shock_ramp_min_slope_abs_pct` | `0.0` |
| `liq_boost_enable` | `False` |
| `liq_boost_score_min` | `2.0` |
| `liq_boost_score_span` | `2.0` |
| `liq_boost_max_risk_mult` | `1.0` |
| `liq_boost_cap_floor_frac` | `0.0` |
| `liq_boost_require_alignment` | `True` |
| `liq_boost_require_shock` | `False` |
| `risk_entry_cutoff_hour_et` | `None` |
| `riskoff_tr5_med_pct` | `None` |
| `riskoff_tr5_lookback_days` | `5` |
| `riskoff_mode` | `'hygiene'` |
| `riskoff_short_risk_mult_factor` | `1.0` |
| `riskoff_long_risk_mult_factor` | `1.0` |
| `riskpanic_tr5_med_pct` | `None` |
| `riskpanic_neg_gap_ratio_min` | `None` |
| `riskpanic_neg_gap_abs_pct_min` | `None` |
| `riskpanic_lookback_days` | `5` |
| `riskpanic_tr5_med_delta_min_pct` | `None` |
| `riskpanic_tr5_med_delta_lookback_days` | `1` |
| `riskpanic_long_risk_mult_factor` | `1.0` |
| `riskpanic_long_scale_mode` | `'off'` |
| `riskpanic_long_scale_tr_delta_max_pct` | `None` |
| `riskpanic_short_risk_mult_factor` | `1.0` |
| `riskpop_tr5_med_pct` | `None` |
| `riskpop_pos_gap_ratio_min` | `None` |
| `riskpop_pos_gap_abs_pct_min` | `None` |
| `riskpop_lookback_days` | `5` |
| `riskpop_tr5_med_delta_min_pct` | `None` |
| `riskpop_tr5_med_delta_lookback_days` | `1` |
| `riskpop_long_risk_mult_factor` | `1.0` |
| `riskpop_short_risk_mult_factor` | `1.0` |
| `ratsv_enabled` | `False` |
| `ratsv_slope_window_bars` | `5` |
| `ratsv_slope_slow_window_bars` | `None` |
| `ratsv_tr_fast_bars` | `5` |
| `ratsv_tr_slow_bars` | `20` |
| `ratsv_rank_min` | `None` |
| `ratsv_tr_ratio_min` | `None` |
| `ratsv_slope_med_min_pct` | `None` |
| `ratsv_slope_vel_min_pct` | `None` |
| `ratsv_slope_med_slow_min_pct` | `None` |
| `ratsv_slope_vel_slow_min_pct` | `None` |
| `ratsv_slope_vel_consistency_bars` | `0` |
| `ratsv_slope_vel_consistency_min` | `None` |
| `ratsv_cross_age_max_bars` | `None` |
| `ratsv_branch_a_rank_min` | `None` |
| `ratsv_branch_a_tr_ratio_min` | `None` |
| `ratsv_branch_a_slope_med_min_pct` | `None` |
| `ratsv_branch_a_slope_vel_min_pct` | `None` |
| `ratsv_branch_a_slope_med_slow_min_pct` | `None` |
| `ratsv_branch_a_slope_vel_slow_min_pct` | `None` |
| `ratsv_branch_a_slope_vel_consistency_bars` | `None` |
| `ratsv_branch_a_slope_vel_consistency_min` | `None` |
| `ratsv_branch_a_cross_age_max_bars` | `None` |
| `ratsv_branch_b_rank_min` | `None` |
| `ratsv_branch_b_tr_ratio_min` | `None` |
| `ratsv_branch_b_slope_med_min_pct` | `None` |
| `ratsv_branch_b_slope_vel_min_pct` | `None` |
| `ratsv_branch_b_slope_med_slow_min_pct` | `None` |
| `ratsv_branch_b_slope_vel_slow_min_pct` | `None` |
| `ratsv_branch_b_slope_vel_consistency_bars` | `None` |
| `ratsv_branch_b_slope_vel_consistency_min` | `None` |
| `ratsv_branch_b_cross_age_max_bars` | `None` |
| `ratsv_probe_cancel_max_bars` | `0` |
| `ratsv_probe_cancel_slope_adverse_min_pct` | `None` |
| `ratsv_probe_cancel_tr_ratio_min` | `None` |
| `ratsv_adverse_release_min_hold_bars` | `0` |
| `ratsv_adverse_release_slope_adverse_min_pct` | `None` |
| `ratsv_adverse_release_tr_ratio_min` | `None` |

### A5) Runtime/Adapter-Only Strategy Keys
These keys are consumed in runtime adapters even when not first-class schema fields.

| key | meaning |
|---|---|
| `signal_use_rth` | Session handling for next-open alignment and runtime day logic. |
| `spot_sec_type` | Security/session model hint (`STK`/`FUT` semantics in adapters). |
| `spot_graph_entry_policy` | Alias override for graph entry node. |
| `spot_graph_exit_policy` | Alias override for graph exit node. |
| `spot_graph_resize_policy` | Alias override for graph resize node. |
| `spot_graph_risk_overlay_policy` | Alias override for graph risk-overlay node. |
| `shock_tr_fast_period` | Alias for TR-ratio detector fast period. |
| `shock_tr_slow_period` | Alias for TR-ratio detector slow period. |
| `shock_min_tr_pct` | Alias for TR-ratio detector minimum TR%. |

### A6) Policy Graph Profiles
| profile | entry policy | exit policy | resize policy | risk overlay policy |
|---|---|---|---|---|
| `neutral` | `default` | `priority` | `adaptive` | `legacy` |
| `defensive` | `slope_tr_guard` | `slope_flip_guard` | `adaptive_atr_defensive` | `atr_compress` |
| `aggressive` | `default` | `priority` | `adaptive_hybrid_aggressive` | `trend_bias` |
| `hf_probe` | `slope_tr_guard` | `slope_flip_guard` | `adaptive_slope_probe` | `trend_bias` |

### A7) Policy Packs
| pack | behavior intent |
|---|---|
| `neutral` | No-op defaults; explicit knobs dominate. |
| `defensive` | Drawdown-first behavior and tighter adaptive sizing. |
| `aggressive` | Faster trend-following scale-up behavior. |
| `hf_probe` | Probe-oriented small-step, fast-feedback behavior. |
