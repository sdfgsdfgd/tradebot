"""SweepCartesian capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from ...backtest.cli_utils import parse_window
from ...backtest.config import ConfigBundle
from ...backtest.spot_context import spot_bar_requirements_from_strategy
from ...backtest.sweeps import utc_now_iso_z, write_json
from ...spot.champions import (
    load_current_champion_groups,
    promote_champion,
    promotion_receipt,
)
from ...spot.codec import bool_from_payload
from .combo_presets import ComboPresetContext
from .catalog import (
    _COMBO_FULL_CARTESIAN_DIM_ORDER,
    _COMBO_FULL_PAIR_DIM_VARIANT_SPECS,
    _combo_full_preset_axes,
    _combo_full_preset_key,
    _combo_full_preset_tier,
)
from .dimensions import _AXIS_DIMENSION_REGISTRY
from .fingerprints import (
    _RUN_CFG_CACHE_ENGINE_VERSION,
)
from .milestones import (
    _collect_milestone_items_from_payload,
    _filters_payload,
    _milestone_key,
    _print_leaderboards,
    _spot_strategy_payload,
)
from .support import (
    _registry_float,
    _require_offline_cache_or_die,
    _runtime_policy,
)


# region Full-combo finalist and crown policy
def _metric(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _objective_shortlist(
    candidates: list[tuple[object, dict, str]],
    *,
    limit: int,
) -> list[tuple[object, dict, str]]:
    """Preserve each economic leader, then rank broad strength across objectives."""
    best: dict[str, tuple[object, dict, str]] = {}
    for cfg, row, note in candidates:
        key = _milestone_key(cfg)
        previous = best.get(key)
        score = (
            _metric(row, "pnl_over_dd"),
            _metric(row, "pnl"),
            _metric(row, "win_rate"),
            _metric(row, "trades"),
        )
        if previous is None or score > (
            _metric(previous[1], "pnl_over_dd"),
            _metric(previous[1], "pnl"),
            _metric(previous[1], "win_rate"),
            _metric(previous[1], "trades"),
        ):
            best[key] = (cfg, row, note)
    keyed = sorted(best.items())
    if not keyed:
        return []
    objectives = (
        lambda row: _metric(row, "pnl_over_dd"),
        lambda row: _metric(row, "pnl"),
        lambda row: _metric(row, "win_rate"),
        lambda row: _metric(row, "trades"),
    )
    rankings = [
        sorted(keyed, key=lambda item, score=score: (score(item[1][1]), item[0]), reverse=True)
        for score in objectives
    ]
    fused = {key: 0.0 for key, _candidate in keyed}
    for ranking in rankings:
        for rank, (key, _candidate) in enumerate(ranking, start=1):
            fused[key] += 1.0 / (60.0 + float(rank))
    ordered: list[str] = []
    for ranking in rankings:
        if ranking[0][0] not in ordered:
            ordered.append(ranking[0][0])
    ordered.extend(
        key
        for key, _candidate in sorted(
            keyed,
            key=lambda item: (fused[item[0]], item[0]),
            reverse=True,
        )
        if key not in ordered
    )
    return [best[key] for key in ordered[: max(1, int(limit))]]


def _stability_score(item: dict) -> tuple:
    stability = item.get("stability") if isinstance(item.get("stability"), dict) else {}
    primary = item.get("primary") if isinstance(item.get("primary"), dict) else {}
    return (
        _metric(stability, "min_roi_over_dd"),
        _metric(stability, "min_pnl"),
        _metric(primary, "roi_over_dd_pct"),
        _metric(primary, "pnl"),
    )


def _required_trades(
    start: date,
    end: date,
    *,
    minimum: int,
    per_year: float | None,
) -> int:
    annualized = (
        int(math.ceil(max(1, (end - start).days + 1) / 365.25 * float(per_year)))
        if per_year is not None
        else 0
    )
    return max(int(minimum), annualized)


# endregion


class SweepCartesian:
    def _combo_full_context(self, preset: str | None = None) -> ComboPresetContext:
        """Build the single search-space authority shared by progress and execution."""

        dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
        if not isinstance(dims, dict):
            raise SystemExit("combo_full dimension registry missing: combo_full_cartesian_tight")
        requested_preset = getattr(self.args, "combo_full_preset", "") if preset is None else preset
        combo_full_preset = _combo_full_preset_key(str(requested_preset or ""))
        valid_combo_presets = set(_combo_full_preset_axes())
        if combo_full_preset and combo_full_preset not in valid_combo_presets:
            raise SystemExit(f"Unknown combo_full preset: {combo_full_preset!r}")

        def _pairs_from_registry(
            *,
            dim_name: str,
            variants_key: str,
        ) -> list[tuple[str, dict[str, object]]]:
            out: list[tuple[str, dict[str, object]]] = []
            raw_variants = dims.get(str(variants_key))
            if isinstance(raw_variants, (list, tuple)):
                for row in tuple(raw_variants):
                    if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                        continue
                    label = str(row[0] or "").strip()
                    payload = row[1]
                    if not label or not isinstance(payload, dict):
                        continue
                    out.append((label, dict(payload)))
            if not out:
                raise SystemExit(f"combo_full requires at least one {dim_name} variant.")
            return out

        def _timing_profiles_from_registry(
            *,
            variants_key: str,
        ) -> list[tuple[str, dict[str, object], dict[str, object]]]:
            out: list[tuple[str, dict[str, object], dict[str, object]]] = []
            raw_variants = dims.get(str(variants_key))
            if isinstance(raw_variants, (list, tuple)):
                for row in tuple(raw_variants):
                    if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                        continue
                    label = str(row[0] or "").strip()
                    payload = row[1]
                    if not label or not isinstance(payload, dict):
                        continue
                    strat_over = payload.get("strategy_overrides")
                    filt_over = payload.get("filter_overrides")
                    strat_dict = dict(strat_over) if isinstance(strat_over, dict) else {}
                    filt_dict = dict(filt_over) if isinstance(filt_over, dict) else {}
                    out.append((str(label), strat_dict, filt_dict))
            return out

        pair_variants_by_dim: dict[str, list[tuple[str, dict[str, object]]]] = {
            str(dim_name): _pairs_from_registry(
                dim_name=str(dim_name),
                variants_key=str(variants_key),
            )
            for dim_name, variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }
        if not bool(getattr(self.args, "combo_full_include_tick", False)):
            tick_rows = [
                (str(label), dict(payload))
                for label, payload in tuple(pair_variants_by_dim.get("tick") or ())
                if str((payload or {}).get("tick_gate_mode") or "off").strip().lower() in ("off", "", "none", "false", "0")
            ]
            if not tick_rows:
                tick_rows = [("tick=off", {"tick_gate_mode": "off"})]
            pair_variants_by_dim["tick"] = tick_rows
        confirm_bars = [int(v) for v in tuple(dims.get("confirm_bars") or ())]
        timing_profile_variants = _timing_profiles_from_registry(variants_key="timing_profile_variants")
        if not timing_profile_variants:
            timing_profile_variants = [("timing=base", {}, {})]
        short_mults = [float(v) for v in tuple(dims.get("short_mults") or ())]
        if not confirm_bars or not short_mults:
            raise SystemExit("combo_full requires non-empty confirm_bars and short_mults.")

        dim_state: dict[str, list[object]] = {
            "timing_profile": list(timing_profile_variants),
            "confirm": list(confirm_bars),
            **{str(dim_name): list(rows) for dim_name, rows in pair_variants_by_dim.items()},
            "short_mult": list(short_mults),
        }
        preset_context = ComboPresetContext(self, dims, dim_state, _timing_profiles_from_registry)

        preset_context.apply(combo_full_preset)
        return preset_context

    def _combo_full_seed_candidates(self) -> list[tuple[ConfigBundle, dict, str]]:
        """Decode every matching seed entry through the canonical spot codec."""
        payload = getattr(self, "milestones", None)
        if not isinstance(payload, dict):
            return []
        candidates: list[tuple[ConfigBundle, dict, str]] = []
        for item in _collect_milestone_items_from_payload(payload, symbol=self.symbol):
            strategy = item.get("strategy")
            if not isinstance(strategy, dict):
                continue
            bar_size = str(strategy.get("signal_bar_size") or self.signal_bar_size).strip()
            use_rth = bool_from_payload(
                strategy.get("signal_use_rth"),
                default=self.use_rth,
            )
            if bar_size.lower() != str(self.signal_bar_size).strip().lower() or use_rth != bool(self.use_rth):
                continue
            cfg = self._cfg_from_strategy_filters_payload(
                strategy,
                item.get("filters"),
                bar_size=bar_size,
                use_rth=use_rth,
            )
            metrics = item.get("metrics")
            if cfg is not None and isinstance(metrics, dict):
                candidates.append((cfg, dict(metrics), str(item.get("note") or "seed")))
        return candidates

    def _combo_full_stability(
        self,
        candidates: list[tuple[ConfigBundle, dict, str]],
    ) -> None:
        """Gate the full-combo shortlist and optionally advance one machine crown."""
        raw_windows = tuple(getattr(self.args, "stability_window", ()) or ())
        promote = bool(getattr(self.args, "promote", False))
        if not raw_windows and not promote:
            return

        seeded = self._combo_full_seed_candidates()
        if seeded:
            candidates = [*candidates, *seeded]
            print(f"combo_full stability: added {len(seeded)} seed candidates", flush=True)

        requested_track = str(getattr(self.args, "track", "auto") or "auto").strip().upper()
        track_filter = None if requested_track == "AUTO" else (requested_track,)
        incumbent_groups, warnings = load_current_champion_groups(
            symbols=(self.symbol,),
            tracks=track_filter,
        )
        for warning in warnings:
            print(f"champion source warning: {warning}", flush=True)

        def _matches_lane(group: dict) -> bool:
            entries = group.get("entries") if isinstance(group.get("entries"), list) else []
            entry = next((row for row in entries if isinstance(row, dict)), None)
            strategy = entry.get("strategy") if isinstance(entry, dict) else None
            if not isinstance(strategy, dict):
                return False
            use_rth = bool_from_payload(
                strategy.get("signal_use_rth"),
                default=self.use_rth,
            )
            return (
                str(strategy.get("signal_bar_size") or "").strip().lower()
                == str(self.signal_bar_size).strip().lower()
                and use_rth == bool(self.use_rth)
            )

        lane_groups = [group for group in incumbent_groups if _matches_lane(group)]
        lane_tracks = {str(group.get("_track") or "").strip().upper() for group in lane_groups}
        if requested_track == "AUTO" and len(lane_tracks) == 1:
            requested_track = next(iter(lane_tracks))
            lane_groups = [
                group
                for group in lane_groups
                if str(group.get("_track") or "").strip().upper() == requested_track
            ]
        if promote and requested_track not in {"HF", "LF"}:
            raise SystemExit("--promote requires --track hf|lf when the current lane is ambiguous")
        incumbent = lane_groups[0] if len(lane_groups) == 1 else None
        incumbent_eval = (
            incumbent.get("_eval")
            if isinstance(incumbent, dict) and isinstance(incumbent.get("_eval"), dict)
            else {}
        )
        incumbent_windows = tuple(
            row
            for row in incumbent_eval.get("windows", ())
            if isinstance(row, dict)
        )

        try:
            windows = [parse_window(raw) for raw in raw_windows]
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if not windows and promote:
            windows = [
                (date.fromisoformat(str(row["start"])), date.fromisoformat(str(row["end"])))
                for row in incumbent_windows
            ]
        if not windows:
            raise SystemExit("combo_full stability requires --stability-window or incumbent promotion windows")
        if any(end < start for start, end in windows):
            raise SystemExit("combo_full stability windows must end on or after their start")

        shortlist = _objective_shortlist(
            candidates,
            limit=max(1, int(getattr(self.args, "stability_top", 200) or 200)),
        )
        if not shortlist:
            print("combo_full stability: no eligible primary-window candidates", flush=True)
            return
        requirements = {
            _milestone_key(cfg): tuple(
                spot_bar_requirements_from_strategy(
                    strategy=cfg.strategy,
                    default_symbol=str(cfg.strategy.symbol),
                    default_exchange=cfg.strategy.exchange,
                    default_signal_bar_size=str(cfg.backtest.bar_size),
                    default_signal_use_rth=bool(cfg.backtest.use_rth),
                    include_signal=True,
                )
            )
            for cfg, _row, _note in shortlist
        }
        rows_by_key: dict[str, list[dict]] = {
            _milestone_key(cfg): [] for cfg, _row, _note in shortlist
        }
        active = set(rows_by_key)
        per_year_raw = getattr(self.args, "stability_min_trades_per_year", None)
        try:
            per_year = float(per_year_raw) if per_year_raw is not None else None
        except (TypeError, ValueError) as exc:
            raise SystemExit("--stability-min-trades-per-year must be a non-negative number") from exc
        if per_year is not None and (not math.isfinite(per_year) or per_year < 0.0):
            raise SystemExit("--stability-min-trades-per-year must be a non-negative number")
        window_plan = sorted(
            enumerate(windows),
            key=lambda item: (
                -_required_trades(
                    item[1][0],
                    item[1][1],
                    minimum=int(self.args.min_trades),
                    per_year=per_year,
                ),
                (item[1][1] - item[1][0]).days,
                item[0],
            ),
        )
        print(
            "combo_full stability: "
            f"shortlist={len(shortlist)} windows={len(windows)} "
            f"track={requested_track if requested_track in {'HF', 'LF'} else 'unclassified'}",
            flush=True,
        )

        for window_position, (window_index, (window_start, window_end)) in enumerate(
            window_plan,
            start=1,
        ):
            window_candidates = [
                item for item in shortlist if _milestone_key(item[0]) in active
            ]
            if not window_candidates:
                break
            required = _required_trades(
                window_start,
                window_end,
                minimum=int(self.args.min_trades),
                per_year=per_year,
            )
            unique_requirements = {
                (
                    req.symbol,
                    req.exchange,
                    req.bar_size,
                    req.use_rth,
                    req.warmup_days,
                ): req
                for cfg, _row, _note in window_candidates
                for req in requirements[_milestone_key(cfg)]
            }
            warmup_days = max(
                (max(0, int(req.warmup_days)) for req in unique_requirements.values()),
                default=0,
            )
            warm_start = window_start - timedelta(days=warmup_days)
            if self.offline:
                self._preflight_offline_requirements(
                    unique_requirements.values(),
                    start_dt=datetime.combine(window_start, time(0, 0)),
                    end_dt=datetime.combine(window_end, time(23, 59)),
                    honor_warmup=True,
                )

            child_args = type(self.args)(**vars(self.args))
            child_args.start = warm_start.isoformat()
            child_args.end = window_end.isoformat()
            child_args.min_trades = required
            child_args.jobs = getattr(self, "jobs", 1) if self.offline else 1
            child_args.base = "default"
            child_args.seed_milestones = None
            child_args.write_milestones = False
            child_args.promote = False
            child_args.stability_window = []
            child_args.combo_full_cartesian_stage = None
            child_args.combo_full_cartesian_worker = None
            child_args.combo_full_cartesian_workers = None
            child_args.combo_full_cartesian_out = None
            child_args.combo_full_cartesian_run_min_trades = None
            child_args.cfg_stage = None
            child_args.cfg_worker = None
            child_args.cfg_workers = None
            child_args.cfg_out = None
            child = type(self)(child_args)
            try:
                evaluated: dict[str, dict] = {}
                window_cfgs: list[tuple[ConfigBundle, str]] = []
                for cfg, _primary_row, _note in window_candidates:
                    key = _milestone_key(cfg)
                    window_cfgs.append(
                        (
                            replace(
                                cfg,
                                backtest=replace(
                                    cfg.backtest,
                                    start=window_start,
                                    end=window_end,
                                ),
                            ),
                            key,
                        )
                    )

                def _capture_window(cfg: ConfigBundle, row: dict, _note: str) -> None:
                    evaluated[_milestone_key(cfg)] = row

                child._run_cfg_pairs_grid(
                    axis_tag="combo_full_stability",
                    cfg_pairs=window_cfgs,
                    rows=[],
                    on_row=_capture_window,
                    report_every=max(1, len(window_cfgs) // 20),
                    heartbeat_sec=30.0,
                )
                for cfg, _primary_row, _note in window_candidates:
                    key = _milestone_key(cfg)
                    row = evaluated.get(key)
                    if row is None:
                        active.discard(key)
                        continue
                    ratio = (
                        _metric(row, "roi") / _metric(row, "dd_pct")
                        if _metric(row, "dd_pct") > 0.0
                        else _metric(row, "pnl_over_dd")
                    )
                    rows_by_key[key].append(
                        {
                            "start": window_start.isoformat(),
                            "end": window_end.isoformat(),
                            **row,
                            "roi_over_dd_pct": ratio,
                            "_window_index": int(window_index),
                        }
                    )
            finally:
                child._run_cfg_persistent_flush_pending(force=True)
                if child.run_cfg_persistent_conn is not None:
                    child.run_cfg_persistent_conn.close()
                child.data.disconnect()
            print(
                f"combo_full stability window {window_position}/{len(windows)} "
                f"{window_start.isoformat()}->{window_end.isoformat()} "
                f"required_trades={required} survivors={len(active)}",
                flush=True,
            )

        results: list[dict] = []
        for cfg, _primary_row, note in shortlist:
            key = _milestone_key(cfg)
            if key not in active:
                continue
            window_rows = [
                {name: value for name, value in row.items() if name != "_window_index"}
                for row in sorted(
                    rows_by_key[key],
                    key=lambda row: int(row["_window_index"]),
                )
            ]
            if len(window_rows) != len(windows):
                continue
            stability = {
                "min_pnl_over_dd": min(_metric(row, "pnl_over_dd") for row in window_rows),
                "min_pnl": min(_metric(row, "pnl") for row in window_rows),
                "min_roi_over_dd": min(_metric(row, "roi_over_dd_pct") for row in window_rows),
                "min_roi": min(_metric(row, "roi") for row in window_rows),
            }
            results.append(
                {
                    "key": key,
                    "cfg": cfg,
                    "note": note,
                    "primary": window_rows[0],
                    "stability": stability,
                    "windows": window_rows,
                    "promotion": promotion_receipt(
                        window_rows,
                        incumbent_windows,
                        objective=str(self.args.promotion_objective),
                    ),
                }
            )
        results.sort(key=_stability_score, reverse=True)

        generated_at = utc_now_iso_z()
        write_top = max(1, int(getattr(self.args, "stability_write_top", 200) or 200))
        written_results = results if promote else results[:write_top]
        groups: list[dict] = []
        for rank, item in enumerate(written_results, start=1):
            cfg = item["cfg"]
            primary = item["primary"]
            group = {
                "name": (
                    f"Spot ({self.symbol}) STABILITY #{rank:02d} "
                    f"roi/dd={_metric(primary, 'roi_over_dd_pct'):.2f} "
                    f"pnl={_metric(primary, 'pnl'):.1f} "
                    f"tr={int(_metric(primary, 'trades'))}"
                ),
                "filters": _filters_payload(cfg.strategy.filters),
                "entries": [
                    {
                        "symbol": self.symbol,
                        "metrics": primary,
                        "strategy": _spot_strategy_payload(cfg, meta=self.meta),
                    }
                ],
                "_key": item["key"],
                "_eval": {
                    "stability": item["stability"],
                    "windows": item["windows"],
                    "promotion": item["promotion"],
                },
            }
            if requested_track in {"HF", "LF"}:
                group["_track"] = requested_track
            groups.append(group)
        output = Path(str(self.args.stability_out))
        payload = {
            "schema": "tradebot.research.stability.v1",
            "name": "combo_full_stability",
            "generated_at": generated_at,
            "symbol": self.symbol,
            "track": requested_track if requested_track in {"HF", "LF"} else None,
            "source": {
                "axis": "combo_full",
                "preset": str(getattr(self.args, "combo_full_preset", "") or "full"),
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "shortlist": len(shortlist),
            },
            "windows": [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in windows
            ],
            "groups": groups,
        }
        write_json(output, payload, sort_keys=False)
        print(f"Wrote {output} ({len(groups)} stable finalists).", flush=True)

        if not promote:
            return
        winner = next(
            (
                group
                for group in groups
                if bool(group.get("_eval", {}).get("promotion", {}).get("eligible"))
            ),
            None,
        )
        if winner is None:
            raise SystemExit("No stable finalist passed the incumbent promotion receipt")
        declaration = promote_champion(
            root=Path(__file__).resolve().parents[3],
            symbol=self.symbol,
            track=requested_track,
            version=str(self.args.promotion_version or generated_at),
            artifact_path=output,
            strategy_key=str(winner["_key"]),
            receipt=dict(winner["_eval"]["promotion"]),
        )
        print(f"Promoted {self.symbol} {requested_track} crown: {declaration}", flush=True)

    def _sweep_combo_full(self) -> None:
        """Unified tight Cartesian sweep over centralized combo dimensions."""

        bars_sig = self._bars_cached(self.signal_bar_size)
        combo_full_preset = _combo_full_preset_key(str(getattr(self.args, "combo_full_preset", "") or ""))
        preset_context = self._combo_full_context(combo_full_preset)
        pair_variants_by_dim = {
            dim_name: list(preset_context.rows[dim_name])
            for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }

        requires_tick_daily = any(
            str((payload or {}).get("tick_gate_mode") or "off").strip().lower() != "off" for _label, payload in pair_variants_by_dim["tick"]
        )
        if self.offline and requires_tick_daily:
            tick_warm_start = self.start_dt - timedelta(days=400)
            tick_ok = False
            for tick_sym in ("TICK-AMEX", "TICK-NYSE"):
                try:
                    _require_offline_cache_or_die(
                        data=self.data,
                        cache_dir=self.cache_dir,
                        symbol=tick_sym,
                        exchange=None,
                        start_dt=tick_warm_start,
                        end_dt=self.end_dt,
                        bar_size="1 day",
                        use_rth=True,
                        cache_policy=self.cache_policy,
                    )
                    tick_ok = True
                    break
                except SystemExit:
                    continue
            if not tick_ok:
                raise SystemExit(
                    "combo_full requires cached daily $TICK bars when running with --offline "
                    "(expected under db/TICK-AMEX or db/TICK-NYSE). Run once without --offline to fetch."
                )

        size_by_dim = preset_context.size_by_dim
        cartesian_total = preset_context.total
        total = preset_context.run_total
        if cartesian_total <= 0:
            raise SystemExit("combo_full has empty Cartesian dimensions.")
        base = preset_context.base
        ordered_dims = preset_context.ordered_dims
        combo_dim_space_sig = preset_context.dimension_signature

        def _combo_full_worker_stage_window_signature() -> str:
            raw = {
                "version": str(_RUN_CFG_CACHE_ENGINE_VERSION),
                "stage": "combo_full_cartesian",
                "symbol": str(self.symbol),
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "signal_bar_size": str(self.signal_bar_size),
                "use_rth": bool(self.use_rth),
                "preset": str(combo_full_preset or ""),
                "ordered_dims": tuple(str(v) for v in ordered_dims),
                "size_by_dim": tuple((str(k), int(v)) for k, v in size_by_dim.items()),
                "dim_space_sig": str(combo_dim_space_sig),
                "run_total": int(total),
                "bars_sig": self._bars_signature(bars_sig),
            }
            return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

        if self.args.combo_full_cartesian_stage:
            self._run_sharded_stage_worker(
                stage_label="combo_full_cartesian",
                worker_raw=self.args.combo_full_cartesian_worker,
                workers_raw=self.args.combo_full_cartesian_workers,
                out_path_raw=str(self.args.combo_full_cartesian_out or ""),
                out_flag_name="combo-full-cartesian-out",
                plan_all=None,
                bars=bars_sig,
                report_every=0,
                heartbeat_sec=30.0,
                plan_total=int(total),
                plan_item_from_rank=preset_context.run_item_from_rank,
                rank_manifest_window_signature=_combo_full_worker_stage_window_signature(),
                rank_batch_size=2048,
            )
            return

        print("")
        print("=== combo_full: unified tight Cartesian core ===")
        if combo_full_preset:
            print(
                f"combo_full preset active: {combo_full_preset} (tier={_combo_full_preset_tier(str(combo_full_preset))})",
                flush=True,
            )
        print(
            "combo_full dimensions: "
            f"cartesian={int(cartesian_total)} evaluations={int(total)} "
            + " ".join(f"{str(dim_name)}={int(size_by_dim.get(str(dim_name), 0) or 0)}" for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER),
            flush=True,
        )
        print(
            f"combo_full sharding order: {','.join(ordered_dims)}",
            flush=True,
        )
        print("")

        base_key = _milestone_key(base)

        rows: list[dict] = []
        candidates: list[tuple[ConfigBundle, dict, str]] = []
        all_candidates: list[tuple[ConfigBundle, dict, str]] = []
        combo_stage_args = tuple(("--combo-full-preset", str(combo_full_preset)) if combo_full_preset else ())
        combo_manifest_window_sig = _combo_full_worker_stage_window_signature()
        snapshot_full = bool(getattr(self.args, "write_milestones", False))
        snapshot_limit = max(
            200,
            int(getattr(self.args, "top", 0) or 0),
            int(getattr(self.args, "stability_top", 0) or 0),
        )
        snapshot_scope = (
            f"{combo_manifest_window_sig}|"
            f"m{int(self.run_min_trades)}|"
            f"{'full' if snapshot_full else f'top{snapshot_limit}'}"
        )
        combo_snapshot_key = self._run_cfg_persistent_key(
            strategy_fingerprint="__stage_result__",
            axis_dimension_fingerprint="combo_full_cartesian",
            window_signature=str(snapshot_scope),
        )
        complete_snapshot_limit = max(
            0,
            int(
                _registry_float(
                    _runtime_policy("stage_result_snapshot").get(
                        "complete_max_total"
                    ),
                    2048.0,
                )
            ),
        )
        complete_snapshot_enabled = bool(
            0 < int(total) <= int(complete_snapshot_limit)
        )
        complete_snapshot_key = self._run_cfg_persistent_key(
            strategy_fingerprint="__stage_result__",
            axis_dimension_fingerprint="combo_full_cartesian",
            window_signature=f"{combo_manifest_window_sig}|complete",
        )
        combo_snapshot = self._run_cfg_persistent_get(
            cache_key=str(combo_snapshot_key)
        )
        try:
            snapshot_tested = int(
                combo_snapshot.get("tested") if isinstance(combo_snapshot, dict) else -1
            )
        except (TypeError, ValueError):
            snapshot_tested = -1
        snapshot_valid = bool(
            isinstance(combo_snapshot, dict)
            and combo_snapshot.get("schema") == "tradebot.research.stage-result.v1"
            and int(snapshot_tested) == int(total)
            and isinstance(combo_snapshot.get("records"), list)
        )
        complete_snapshot = (
            self._run_cfg_persistent_get(cache_key=str(complete_snapshot_key))
            if complete_snapshot_enabled and not snapshot_valid
            else None
        )
        try:
            complete_snapshot_tested = int(
                complete_snapshot.get("tested")
                if isinstance(complete_snapshot, dict)
                else -1
            )
        except (TypeError, ValueError):
            complete_snapshot_tested = -1
        complete_snapshot_valid = bool(
            isinstance(complete_snapshot, dict)
            and complete_snapshot.get("schema")
            == "tradebot.research.stage-result.v2"
            and complete_snapshot.get("complete") is True
            and int(complete_snapshot_tested) == int(total)
            and isinstance(complete_snapshot.get("records"), list)
            and len(complete_snapshot.get("records") or ()) == int(total)
        )
        snapshot_records = (
            [
                dict(record)
                for record in combo_snapshot.get("records", ())
                if isinstance(record, dict)
            ]
            if snapshot_valid and isinstance(combo_snapshot, dict)
            else []
        )
        if complete_snapshot_valid and isinstance(complete_snapshot, dict):
            snapshot_valid = True
            snapshot_records = [
                dict(record)
                for record in complete_snapshot.get("records", ())
                if isinstance(record, dict)
                and _metric(
                    record.get("row")
                    if isinstance(record.get("row"), dict)
                    else {},
                    "trades",
                )
                >= int(self.run_min_trades)
            ]

        def _combo_full_parallel_totals() -> tuple[int, int]:
            if snapshot_valid:
                return 0, int(total)

            def _unresolved_total() -> int:
                try:
                    unresolved_ranges = self._cartesian_rank_manifest_unresolved_ranges(
                        stage_label="combo_full_cartesian",
                        window_signature=str(combo_manifest_window_sig),
                        total=int(total),
                    )
                    return sum(
                        max(0, int(rank_hi) - int(rank_lo) + 1)
                        for rank_lo, rank_hi in tuple(unresolved_ranges)
                    )
                except Exception:
                    return int(total)

            unresolved_total = _unresolved_total()
            if int(unresolved_total) < int(total) and not snapshot_valid:
                print(
                    "combo_full Cartesian result snapshot missing; replaying cached "
                    "rank receipts",
                    flush=True,
                )
                self._cartesian_rank_manifest_reset(
                    stage_label="combo_full_cartesian",
                    window_signature=str(combo_manifest_window_sig),
                )
                unresolved_total = _unresolved_total()
            unresolved_i = max(0, int(unresolved_total))
            prefetched_i = max(0, int(total) - int(unresolved_i))
            return int(unresolved_i), int(prefetched_i)

        def _combo_full_parallel_payloads() -> dict[int, dict]:
            unresolved_i, prefetched_i = _combo_full_parallel_totals()
            if unresolved_i > 0 and self.offline:
                requirements = []
                for rank in range(int(total)):
                    cfg, _note, _meta = preset_context.run_item_from_rank(rank)
                    requirements.extend(
                        req
                        for req in spot_bar_requirements_from_strategy(
                            strategy=cfg.strategy,
                            default_symbol=self.symbol,
                            default_exchange=cfg.strategy.exchange,
                            default_signal_bar_size=self.signal_bar_size,
                            default_signal_use_rth=self.use_rth,
                        )
                        if req.kind not in {"signal", "tick"}
                    )
                self._preflight_offline_requirements(
                    requirements,
                    start_dt=self.start_dt,
                    end_dt=self.end_dt,
                )
            payloads = self._run_parallel_stage(
                axis_name="combo_full",
                stage_label="combo_full Cartesian",
                total=int(unresolved_i),
                jobs=int(self.jobs),
                worker_tmp_prefix="tradebot_combo_full_cartesian_",
                worker_tag="cfc",
                out_prefix="combo_full_cartesian_out",
                stage_flag="--combo-full-cartesian-stage",
                stage_value="1",
                worker_flag="--combo-full-cartesian-worker",
                workers_flag="--combo-full-cartesian-workers",
                out_flag="--combo-full-cartesian-out",
                strip_flags_with_values=(
                    "--combo-full-cartesian-stage",
                    "--combo-full-cartesian-worker",
                    "--combo-full-cartesian-workers",
                    "--combo-full-cartesian-out",
                    "--combo-full-cartesian-run-min-trades",
                    "--combo-full-preset",
                ),
                run_min_trades_flag="--combo-full-cartesian-run-min-trades",
                run_min_trades=(
                    0
                    if complete_snapshot_enabled
                    else int(self.run_min_trades)
                ),
                stage_args=combo_stage_args,
                capture_error="Failed to capture combo_full Cartesian worker stdout.",
                failure_label="combo_full Cartesian worker",
                missing_label="combo_full Cartesian",
                invalid_label="combo_full Cartesian",
                planner_stage_label="combo_full_cartesian",
            )
            if int(prefetched_i) > 0 or snapshot_valid:
                payloads[max(payloads, default=-1) + 1] = {
                    "tested": int(prefetched_i),
                    "kept": len(snapshot_records),
                    "records": list(snapshot_records),
                }
            return payloads

        def _capture_candidate(cfg: ConfigBundle, row: dict, note: str) -> None:
            if _milestone_key(cfg) == base_key:
                row = dict(row)
                row["note"] = "base"
                note = "base"
            all_candidates.append((cfg, row, note))
            if _metric(row, "trades") < int(self.run_min_trades):
                return
            self._record_milestone(cfg, row, note)
            rows.append(row)
            candidates.append((cfg, row, note))

        tested_total = self._run_stage_cfg_rows(
            stage_label="combo_full_cartesian",
            total=int(total),
            jobs_req=int(self.jobs),
            bars=bars_sig,
            report_every=200,
            heartbeat_sec=30.0,
            on_row=_capture_candidate,
            serial_plan_builder=preset_context.iter_run_plan,
            parallel_payloads_builder=_combo_full_parallel_payloads,
            parallel_default_note="combo_full Cartesian",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=False,
        )
        if complete_snapshot_enabled and len(all_candidates) == int(total):
            complete_candidates = sorted(
                all_candidates, key=lambda item: _milestone_key(item[0])
            )
            self._run_cfg_persistent_set(
                cache_key=str(complete_snapshot_key),
                payload={
                    "schema": "tradebot.research.stage-result.v2",
                    "complete": True,
                    "tested": int(total),
                    "records": [
                        self._encode_cfg_payload(
                            cfg, note=note, extra={"row": row}
                        )
                        for cfg, row, note in complete_candidates
                    ],
                },
            )
        snapshot_candidates = (
            list(candidates)
            if snapshot_full
            else _objective_shortlist(
                candidates,
                limit=int(snapshot_limit),
            )
        )
        self._run_cfg_persistent_set(
            cache_key=str(combo_snapshot_key),
            payload={
                "schema": "tradebot.research.stage-result.v1",
                "tested": int(total),
                "records": [
                    self._encode_cfg_payload(cfg, note=note, extra={"row": row})
                    for cfg, row, note in snapshot_candidates
                ],
            },
        )
        self._run_cfg_persistent_flush_pending(force=True)
        print(
            f"combo_full Cartesian tested={int(tested_total)} kept={len(rows)} min_trades={int(self.run_min_trades)}",
            flush=True,
        )
        _print_leaderboards(
            rows,
            title="combo_full sweep (unified tight Cartesian)",
            top_n=int(self.args.top),
        )
        self._combo_full_stability(candidates)
