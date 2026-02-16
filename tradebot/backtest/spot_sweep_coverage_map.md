# Spot Sweep Coverage Map

Generated: 2026-02-15

Legend: `sharded` = stage-level worker sharding inside the axis; `cached` = run_cfg+context cache applies; `fast_path` = expected fast-summary eligibility (`yes`/`partial`/`no`).

| axis | sharded | cached | fast_path | notes |
|---|---|---|---|---|
| `ema` | no | yes | yes |  |
| `entry_mode` | no | yes | yes |  |
| `combo_full` | yes | yes | partial | unified Cartesian core (mixed-radix shard ranges) |
| `volume` | no | yes | yes |  |
| `rv` | no | yes | yes |  |
| `tod` | no | yes | yes |  |
| `weekday` | no | yes | yes |  |
| `exit_time` | no | yes | yes |  |
| `atr` | no | yes | no |  |
| `atr_fine` | no | yes | no |  |
| `atr_ultra` | no | yes | no |  |
| `chop_joint` | no | yes | yes |  |
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
| `loosen` | no | yes | yes |  |
| `tick` | no | yes | no |  |

## Notes
- `combo_full` and `--axis all` additionally support axis-level subprocess orchestration.
- Persistent cross-process run_cfg cache is enabled via sqlite (`spot_sweeps_run_cfg_cache.sqlite3`).
- Fast-path labels are conservative and reflect the current gate in `_can_use_fast_summary_path`.
