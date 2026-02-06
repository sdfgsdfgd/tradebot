# Spot Sweep Coverage Map

Generated: 2026-02-06

Legend: `sharded` = stage-level worker sharding inside the axis; `cached` = run_cfg+context cache applies; `fast_path` = expected fast-summary eligibility (`yes`/`partial`/`no`).

| axis | sharded | cached | fast_path | notes |
|---|---|---|---|---|
| `ema` | no | yes | yes |  |
| `entry_mode` | no | yes | yes |  |
| `combo_fast` | yes | yes | partial | stage worker sharding |
| `combo_full` | no | yes | partial | axis subprocess orchestration |
| `squeeze` | no | yes | partial |  |
| `volume` | no | yes | yes |  |
| `rv` | no | yes | yes |  |
| `tod` | no | yes | yes |  |
| `tod_interaction` | no | yes | yes |  |
| `perm_joint` | no | yes | yes |  |
| `weekday` | no | yes | yes |  |
| `exit_time` | no | yes | yes |  |
| `atr` | no | yes | no |  |
| `atr_fine` | no | yes | no |  |
| `atr_ultra` | no | yes | no |  |
| `r2_atr` | no | yes | no |  |
| `r2_tod` | no | yes | yes |  |
| `ema_perm_joint` | no | yes | yes |  |
| `tick_perm_joint` | no | yes | no |  |
| `regime_atr` | no | yes | no |  |
| `ema_regime` | no | yes | yes |  |
| `chop_joint` | no | yes | yes |  |
| `ema_atr` | no | yes | no |  |
| `tick_ema` | no | yes | no |  |
| `ptsl` | no | yes | yes |  |
| `hf_scalp` | no | yes | partial |  |
| `hold` | no | yes | yes |  |
| `spot_short_risk_mult` | no | yes | yes |  |
| `orb` | no | yes | no |  |
| `orb_joint` | no | yes | no |  |
| `frontier` | no | yes | partial |  |
| `regime` | no | yes | yes |  |
| `regime2` | no | yes | yes |  |
| `regime2_ema` | no | yes | yes |  |
| `joint` | no | yes | yes |  |
| `micro_st` | no | yes | yes |  |
| `flip_exit` | no | yes | partial |  |
| `confirm` | no | yes | yes |  |
| `spread` | no | yes | yes |  |
| `spread_fine` | no | yes | yes |  |
| `spread_down` | no | yes | yes |  |
| `slope` | no | yes | yes |  |
| `slope_signed` | no | yes | yes |  |
| `cooldown` | no | yes | yes |  |
| `skip_open` | no | yes | yes |  |
| `shock` | no | yes | yes |  |
| `risk_overlays` | yes | yes | yes | stage worker sharding |
| `loosen` | no | yes | yes |  |
| `loosen_atr` | no | yes | yes |  |
| `tick` | no | yes | no |  |
| `gate_matrix` | yes | yes | partial | stage worker sharding |
| `champ_refine` | yes | yes | partial | stage worker sharding |
| `st37_refine` | yes | yes | partial | stage worker sharding |
| `shock_alpha_refine` | no | yes | partial |  |
| `shock_velocity_refine` | yes | yes | partial | stage worker sharding |
| `shock_velocity_refine_wide` | yes | yes | partial | stage worker sharding |
| `shock_throttle_refine` | no | yes | partial |  |
| `shock_throttle_tr_ratio` | no | yes | partial |  |
| `shock_throttle_drawdown` | no | yes | partial |  |
| `riskpanic_micro` | no | yes | partial |  |
| `overlay_family` | no | yes | partial |  |
| `exit_pivot` | no | yes | partial |  |

## Notes
- `combo_full` and `--axis all` additionally support axis-level subprocess orchestration.
- Persistent cross-process run_cfg cache is enabled via sqlite (`spot_sweeps_run_cfg_cache.sqlite3`).
- Fast-path labels are conservative and reflect the current gate in `_can_use_fast_summary_path`.
