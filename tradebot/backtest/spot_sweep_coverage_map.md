# Spot Sweep Coverage Map

Generated: 2026-07-22

Legend: `sharded` = stage-level worker sharding inside the axis; `cached` = run_cfg+context cache applies.

| axis | sharded | cached | notes |
|---|---|---|---|
| `ema` | no | yes |  |
| `entry_mode` | no | yes |  |
| `combo_full` | yes | yes | unified Cartesian core (mixed-radix shard ranges) |
| `volume` | no | yes |  |
| `rv` | no | yes |  |
| `tod` | no | yes |  |
| `weekday` | no | yes |  |
| `exit_time` | no | yes |  |
| `atr` | no | yes |  |
| `atr_fine` | no | yes |  |
| `atr_ultra` | no | yes |  |
| `chop_joint` | no | yes |  |
| `ptsl` | no | yes |  |
| `hold` | no | yes |  |
| `spot_short_risk_mult` | no | yes |  |
| `orb` | no | yes |  |
| `orb_joint` | no | yes |  |
| `frontier` | no | yes |  |
| `regime` | no | yes |  |
| `regime2` | no | yes |  |
| `regime2_ema` | no | yes |  |
| `joint` | no | yes |  |
| `micro_st` | no | yes |  |
| `flip_exit` | no | yes |  |
| `confirm` | no | yes |  |
| `spread` | no | yes |  |
| `spread_fine` | no | yes |  |
| `spread_down` | no | yes |  |
| `slope` | no | yes |  |
| `slope_signed` | no | yes |  |
| `cooldown` | no | yes |  |
| `skip_open` | no | yes |  |
| `shock` | no | yes |  |
| `loosen` | no | yes |  |
| `tick` | no | yes |  |

## Notes
- `combo_full` and `--axis all` additionally support axis-level subprocess orchestration.
- Persistent cross-process run_cfg cache is enabled via sqlite (`spot_sweeps_run_cfg_cache.sqlite3`).
- Engine labels describe canonical summary execution; all axes share one lifecycle implementation.
