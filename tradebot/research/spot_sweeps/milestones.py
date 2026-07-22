"""Spot research ranking, milestone persistence, and champion decoding."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from ...backtest.config import ConfigBundle, FiltersConfig
from ...backtest.data import ContractMeta
from ...backtest.spot_codec import (
    filters_from_payload as _codec_filters_from_payload,
    filters_payload as _codec_filters_payload,
    spot_strategy_payload as _codec_spot_strategy_payload,
    strategy_from_payload as _codec_strategy_from_payload,
)
from ...backtest.sweep_fingerprint import _strategy_fingerprint
from ...backtest.sweeps import utc_now_iso_z, write_json
from ...spot.codec import effective_filters_payload as _codec_effective_filters_payload
from ...spot.fill_modes import (
    SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    normalize_spot_fill_mode,
)
from ...spot.champions import load_current_champion_groups


def _milestone_metrics_from_row(row: dict) -> dict:
    return {
        "pnl": float(row.get("pnl") or 0.0),
        "roi": float(row.get("roi") or 0.0),
        "win_rate": float(row.get("win_rate") or 0.0),
        "trades": int(row.get("trades") or 0),
        "max_drawdown": float(row.get("dd") or row.get("max_drawdown") or 0.0),
        "max_drawdown_pct": float(row.get("dd_pct") or row.get("max_drawdown_pct") or 0.0),
        "pnl_over_dd": row.get("pnl_over_dd"),
    }


def _milestone_item(
    *,
    strategy: dict,
    filters: dict | None,
    note: str | None,
    metrics: dict,
) -> dict:
    return {
        "key": _strategy_fingerprint(strategy, filters=filters),
        "strategy": strategy,
        "filters": filters,
        "note": note,
        "metrics": {
            "pnl": float(metrics.get("pnl") or 0.0),
            "roi": float(metrics.get("roi") or 0.0),
            "win_rate": float(metrics.get("win_rate") or 0.0),
            "trades": int(metrics.get("trades") or 0),
            "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct") or 0.0),
            "pnl_over_dd": metrics.get("pnl_over_dd"),
        },
    }


def _note_from_group_name(raw_name: str) -> str | None:
    text = str(raw_name or "")
    if text.endswith("]") and "[" in text:
        try:
            return text[text.rfind("[") + 1 : -1].strip() or None
        except Exception:
            return None
    return None


def _collect_milestone_items_from_rows(
    rows: list[tuple[ConfigBundle, dict, str]],
    *,
    meta: ContractMeta,
    min_win: float,
    min_trades: int,
    min_pnl_dd: float,
) -> list[dict]:
    out: list[dict] = []
    for cfg, row, note in rows:
        try:
            win = float(row.get("win_rate") or 0.0)
        except (TypeError, ValueError):
            win = 0.0
        try:
            trades = int(row.get("trades") or 0)
        except (TypeError, ValueError):
            trades = 0
        pnl_dd_raw = row.get("pnl_over_dd")
        try:
            pnl_dd = float(pnl_dd_raw) if pnl_dd_raw is not None else None
        except (TypeError, ValueError):
            pnl_dd = None
        if win < float(min_win) or trades < int(min_trades) or pnl_dd is None or pnl_dd < float(min_pnl_dd):
            continue
        strategy = _spot_strategy_payload(cfg, meta=meta)
        out.append(
            _milestone_item(
                strategy=strategy,
                filters=_filters_payload(cfg.strategy.filters),
                note=str(note),
                metrics=_milestone_metrics_from_row(row),
            )
        )
    return out


def _collect_milestone_items_from_payload(payload: dict, *, symbol: str) -> list[dict]:
    out: list[dict] = []
    if not isinstance(payload, dict):
        return out
    symbol_key = str(symbol).strip().upper()
    for group in payload.get("groups") or []:
        if not isinstance(group, dict):
            continue
        filters = group.get("filters") if isinstance(group.get("filters"), dict) else None
        note = _note_from_group_name(str(group.get("name") or ""))
        for entry in group.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            strategy = entry.get("strategy") or {}
            metrics = entry.get("metrics") or {}
            if not isinstance(strategy, dict) or not isinstance(metrics, dict):
                continue
            entry_symbol = str(entry.get("symbol") or symbol_key).strip().upper()
            if entry_symbol != symbol_key:
                continue
            out.append(
                _milestone_item(
                    strategy=dict(strategy),
                    filters=filters,
                    note=note,
                    metrics=metrics,
                )
            )
    return out


def _milestone_sort_key(item: dict) -> tuple:
    m = item.get("metrics") or {}
    return (
        float(m.get("pnl_over_dd") or float("-inf")),
        float(m.get("pnl") or 0.0),
        float(m.get("win_rate") or 0.0),
        int(m.get("trades") or 0),
    )


def _milestone_sort_key_pnl(item: dict) -> tuple:
    m = item.get("metrics") or {}
    return (
        float(m.get("pnl") or float("-inf")),
        float(m.get("pnl_over_dd") or 0.0),
        float(m.get("win_rate") or 0.0),
        int(m.get("trades") or 0),
    )


def _dedupe_best_milestones(items: list[dict]) -> list[dict]:
    best_by_key: dict[str, dict] = {}
    for item in items:
        key = str(item.get("key") or "")
        if not key:
            continue
        prev = best_by_key.get(key)
        if prev is None or _milestone_sort_key(item) > _milestone_sort_key(prev):
            best_by_key[key] = item
    return sorted(best_by_key.values(), key=_milestone_sort_key, reverse=True)


def _merge_and_write_milestones(
    *,
    out_path: Path,
    eligible_new: list[dict],
    merge_existing: bool,
    add_top_pnl_dd: int,
    add_top_pnl: int,
    symbol: str,
    start: date,
    end: date,
    signal_bar_size: str,
    use_rth: bool,
    milestone_min_win: float,
    milestone_min_trades: int,
    milestone_min_pnl_dd: float,
) -> int:
    items = list(eligible_new)
    add_top_dd = max(0, int(add_top_pnl_dd or 0))
    add_top_pnl = max(0, int(add_top_pnl or 0))
    if merge_existing and (add_top_dd > 0 or add_top_pnl > 0):
        by_dd = sorted(items, key=_milestone_sort_key, reverse=True)[:add_top_dd] if add_top_dd > 0 else []
        by_pnl = sorted(items, key=_milestone_sort_key_pnl, reverse=True)[:add_top_pnl] if add_top_pnl > 0 else []
        seen: set[str] = set()
        selected: list[dict] = []
        for item in by_dd + by_pnl:
            key = str(item.get("key") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(item)
        items = selected

    if merge_existing and out_path.exists():
        try:
            existing_payload = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing_payload = {}
        items.extend(_collect_milestone_items_from_payload(existing_payload, symbol=symbol))

    unique = _dedupe_best_milestones(items)
    groups: list[dict] = []
    for idx, item in enumerate(unique, start=1):
        metrics = item["metrics"]
        groups.append(
            {
                "name": _milestone_group_name_from_strategy(
                    rank=idx,
                    strategy=item["strategy"],
                    metrics=metrics,
                    note=str(item.get("note") or "").strip(),
                ),
                "filters": item["filters"],
                "entries": [{"symbol": symbol, "metrics": metrics, "strategy": item["strategy"]}],
            }
        )
    payload = {
        "name": "spot_milestones",
        "generated_at": utc_now_iso_z(),
        "notes": (
            f"Auto-generated via evolve_spot.py (post-fix). "
            f"window={start.isoformat()}→{end.isoformat()}, bar_size={signal_bar_size}, use_rth={use_rth}. "
            f"thresholds: win>={float(milestone_min_win):.2f}, trades>={int(milestone_min_trades)}, "
            f"pnl/dd>={float(milestone_min_pnl_dd):.2f}."
        ),
        "groups": groups,
    }
    write_json(out_path, payload, sort_keys=False)
    return len(groups)


def _filters_payload(filters: FiltersConfig | None) -> dict | None:
    return _codec_filters_payload(filters)


def _spot_strategy_payload(cfg: ConfigBundle, *, meta: ContractMeta) -> dict:
    return _codec_spot_strategy_payload(cfg, meta=meta)


def _milestone_key(cfg: ConfigBundle) -> str:
    strategy = asdict(cfg.strategy)
    strategy.pop("filters", None)
    return _strategy_fingerprint(
        strategy,
        filters=_filters_payload(cfg.strategy.filters),
        signal_bar_size=str(cfg.backtest.bar_size),
        signal_use_rth=bool(cfg.backtest.use_rth),
    )


def _milestone_group_name(*, rank: int, cfg: ConfigBundle, metrics: dict, note: str | None) -> str:
    pnl = float(metrics.get("pnl") or 0.0)
    win = float(metrics.get("win_rate") or 0.0) * 100.0
    trades = int(metrics.get("trades") or 0)
    pnl_dd = float(metrics.get("pnl_over_dd") or 0.0)
    strat = cfg.strategy
    rbar = str(getattr(strat, "regime_bar_size", "") or "").strip() or "?"
    tag = ""
    if str(getattr(strat, "regime_mode", "") or "").strip().lower() == "supertrend":
        tag = f"ST({getattr(strat, 'supertrend_atr_period', '?')},{getattr(strat, 'supertrend_multiplier', '?')},{getattr(strat, 'supertrend_source', '?')})@{rbar}"
    elif getattr(strat, "regime_ema_preset", None):
        tag = f"EMA({getattr(strat, 'regime_ema_preset', '?')})@{rbar}"
    if str(getattr(strat, "regime2_mode", "off") or "off").strip().lower() != "off":
        r2bar = str(getattr(strat, "regime2_bar_size", "") or "").strip() or "?"
        tag += f" + R2@{r2bar}"
    base = f"Spot (MNQ) 12m (post-fix) #{rank:02d} pnl/dd={pnl_dd:.2f} pnl={pnl:.0f} win={win:.1f}% tr={trades}"
    if tag:
        base += f" — {tag}"
    if note:
        base += f" [{note}]"
    return base


def _milestone_group_name_from_strategy(*, rank: int, strategy: dict, metrics: dict, note: str | None) -> str:
    pnl = float(metrics.get("pnl") or 0.0)
    win = float(metrics.get("win_rate") or 0.0) * 100.0
    trades = int(metrics.get("trades") or 0)
    pnl_dd = float(metrics.get("pnl_over_dd") or 0.0)
    rbar = str(strategy.get("regime_bar_size") or "").strip() or "?"
    tag = ""
    if str(strategy.get("regime_mode") or "").strip().lower() == "supertrend":
        tag = f"ST({strategy.get('supertrend_atr_period', '?')},{strategy.get('supertrend_multiplier', '?')},{strategy.get('supertrend_source', '?')})@{rbar}"
    elif strategy.get("regime_ema_preset"):
        tag = f"EMA({strategy.get('regime_ema_preset', '?')})@{rbar}"
    if str(strategy.get("regime2_mode") or "off").strip().lower() != "off":
        r2bar = str(strategy.get("regime2_bar_size") or "").strip() or "?"
        tag += f" + R2@{r2bar}"
    base = f"Spot (MNQ) 12m (post-fix) #{rank:02d} pnl/dd={pnl_dd:.2f} pnl={pnl:.0f} win={win:.1f}% tr={trades}"
    if tag:
        base += f" — {tag}"
    if note:
        base += f" [{note}]"
    return base


def _score_row_pnl_dd(row: dict) -> tuple:
    return (
        float(row.get("pnl_over_dd") or float("-inf")),
        float(row.get("pnl") or 0.0),
        float(row.get("win_rate") or 0.0),
        int(row.get("trades") or 0),
    )


def _score_row_pnl(row: dict) -> tuple:
    return (
        float(row.get("pnl") or float("-inf")),
        float(row.get("pnl_over_dd") or 0.0),
        float(row.get("win_rate") or 0.0),
        int(row.get("trades") or 0),
    )


def _score_row_roi(row: dict) -> float:
    return float(row.get("roi") or 0.0)


def _score_row_win_rate(row: dict) -> float:
    return float(row.get("win_rate") or 0.0)


def _score_row_roi_dd(row: dict) -> float:
    roi = float(row.get("roi") or 0.0)
    dd_pct = float(row.get("dd_pct") or 0.0)
    if dd_pct <= 0.0:
        return float("-inf") if roi <= 0.0 else float("inf")
    return float(roi / dd_pct)


def _rank_cfg_rows(
    items: list[tuple[object, dict, str]],
    *,
    scorers: list[tuple[object, int]],
    limit: int | None = None,
    key_fn=None,
) -> list[tuple[object, dict, str]]:
    def _identity(item: tuple[object, dict, str]) -> str:
        cfg, row, note = item
        if callable(key_fn):
            raw = key_fn(cfg, row, note)
        elif isinstance(cfg, ConfigBundle):
            raw = _milestone_key(cfg)
        else:
            raw = (note, row)
        return raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True, default=str)

    stable_items = sorted(items, key=_identity)
    ranked: list[tuple[object, dict, str]] = []
    for score_fn, top_n in scorers:
        n = int(top_n)
        if n <= 0:
            continue
        ranked.extend(
            sorted(stable_items, key=lambda t, fn=score_fn: fn(t[1]), reverse=True)[
                :n
            ]
        )
    seen: set[str] = set()
    out: list[tuple[object, dict, str]] = []
    max_items = None if limit is None else max(0, int(limit))
    for cfg, row, note in ranked:
        if callable(key_fn):
            key = key_fn(cfg, row, note)
        else:
            if not isinstance(cfg, ConfigBundle):
                continue
            key = _milestone_key(cfg)
        try:
            already_seen = key in seen
        except TypeError:
            key = json.dumps(key, sort_keys=True, default=str)
            already_seen = key in seen
        if already_seen:
            continue
        seen.add(key)
        out.append((cfg, row, note))
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _rank_cfg_rows_with_meta(
    items: list[tuple[ConfigBundle, dict, str, object]],
    *,
    scorers: list[tuple[object, int]],
    limit: int | None = None,
) -> list[tuple[ConfigBundle, dict, str, object]]:
    ranked_core = _rank_cfg_rows(
        [(cfg, row, note) for cfg, row, note, _meta in items],
        scorers=scorers,
        limit=limit,
    )
    meta_by_key: dict[str, object] = {}
    for cfg, _row, _note, meta in items:
        key = _milestone_key(cfg)
        if key not in meta_by_key:
            meta_by_key[key] = meta
    out: list[tuple[ConfigBundle, dict, str, object]] = []
    for cfg, row, note in ranked_core:
        out.append((cfg, row, note, meta_by_key.get(_milestone_key(cfg))))
    return out


def _print_top(rows: list[dict], *, title: str, top_n: int, sort_key) -> None:
    print("")
    print(title)
    print("-" * len(title))
    rows_sorted = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))
    rows_sorted.sort(key=sort_key, reverse=True)
    for idx, row in enumerate(rows_sorted[: max(1, int(top_n))], start=1):
        pnl = float(row.get("pnl") or 0.0)
        dd = float(row.get("dd") or 0.0)
        roi = float(row.get("roi") or 0.0) * 100.0
        dd_pct = float(row.get("dd_pct") or 0.0) * 100.0
        trades = int(row.get("trades") or 0)
        win = float(row.get("win_rate") or 0.0) * 100.0
        pnl_over_dd = float(row.get("pnl_over_dd") or 0.0)
        note = row.get("note") or ""
        print(
            f"{idx:>2}. tr={trades:>4} win={win:>5.1f}% pnl={pnl:>10.1f} dd={dd:>8.1f} pnl/dd={pnl_over_dd:>6.2f} roi={roi:>6.2f}% dd%={dd_pct:>6.2f}% {note}"
        )


def _print_leaderboards(rows: list[dict], *, title: str, top_n: int) -> None:
    _print_top(rows, title=f"{title} — Top by pnl/dd", top_n=top_n, sort_key=_score_row_pnl_dd)
    _print_top(rows, title=f"{title} — Top by pnl", top_n=top_n, sort_key=_score_row_pnl)


def _milestone_entry_for(
    milestones: dict | None,
    *,
    symbol: str,
    signal_bar_size: str,
    use_rth: bool,
    sort_by: str,
    prefer_realism: bool = False,
) -> tuple[dict, dict | None, dict] | None:
    if not milestones:
        return None

    groups = milestones.get("groups") or []
    candidates: list[tuple[dict, dict | None, dict]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        strategy = entry.get("strategy") or {}
        metrics = entry.get("metrics") or {}
        if not isinstance(strategy, dict) or not isinstance(metrics, dict):
            continue
        if str(entry.get("symbol") or "").strip().upper() != str(symbol).strip().upper():
            continue
        if str(strategy.get("signal_bar_size") or "").strip().lower() != str(signal_bar_size).strip().lower():
            continue
        if bool(strategy.get("signal_use_rth")) != bool(use_rth):
            continue
        if prefer_realism:
            fill_mode = normalize_spot_fill_mode(strategy.get("spot_entry_fill_mode"), default="close")
            if fill_mode != SPOT_FILL_MODE_NEXT_TRADABLE_BAR:
                continue
            if not bool(strategy.get("spot_intrabar_exits")):
                continue
            try:
                comm = float(strategy.get("spot_commission_per_share") or 0.0)
            except (TypeError, ValueError):
                comm = 0.0
            try:
                comm_min = float(strategy.get("spot_commission_min") or 0.0)
            except (TypeError, ValueError):
                comm_min = 0.0
            if comm <= 0.0 and comm_min <= 0.0:
                continue
        group_filters = group.get("filters") if isinstance(group.get("filters"), dict) else None
        effective_filters = _codec_effective_filters_payload(group_filters=group_filters, strategy=strategy)
        candidates.append((strategy, effective_filters, metrics))

    if not candidates:
        return None

    def _score(c: tuple[dict, dict | None, dict]) -> tuple:
        _, _, m = c
        if str(sort_by).strip().lower() == "pnl":
            return _score_row_pnl(m)
        return _score_row_pnl_dd(m)

    return sorted(candidates, key=_score, reverse=True)[0]


def load_current_champion_milestones(
    *,
    symbol: str,
    signal_bar_size: str,
    use_rth: bool,
    track: str,
    prefer_realism: bool,
) -> tuple[dict, str, list[str]]:
    requested_track = str(track or "auto").strip().upper()
    track_filter = None if requested_track == "AUTO" else (requested_track,)
    groups, warnings = load_current_champion_groups(
        symbols=(str(symbol).strip().upper(),),
        tracks=track_filter,
    )
    matching = [
        group
        for group in groups
        if _milestone_entry_for(
            {"groups": [group]},
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            sort_by="pnl_dd",
            prefer_realism=prefer_realism,
        )
        is not None
    ]
    matched_tracks = sorted(
        {str(group.get("_track") or "").strip().upper() for group in matching}
    )
    if not matching:
        raise ValueError(
            f"no promoted {requested_track} champion matches "
            f"{str(symbol).upper()} bar={signal_bar_size!r} rth={bool(use_rth)}"
        )
    if requested_track == "AUTO" and len(matched_tracks) != 1:
        raise ValueError(
            f"promoted champion is ambiguous across {', '.join(matched_tracks)}; "
            "select --track lf or --track hf"
        )
    selected_track = matched_tracks[0]
    return (
        {
            "name": f"current_{selected_track.lower()}_champion",
            "groups": [
                group
                for group in matching
                if str(group.get("_track") or "").strip().upper() == selected_track
            ],
        },
        selected_track,
        warnings,
    )


def _apply_milestone_base(cfg: ConfigBundle, *, strategy: dict, filters: dict | None) -> ConfigBundle:
    # Decode milestone payload through the shared codec so sweep baselines inherit
    # the same shape as runtime/backtest payloads (including dual-branch/rats filters).
    merged_filters = _codec_effective_filters_payload(
        group_filters=filters if isinstance(filters, dict) else None,
        strategy=strategy,
    )

    parsed_filters: FiltersConfig | None = None
    if isinstance(merged_filters, dict):
        parsed_filters = _codec_filters_from_payload(merged_filters)
        if _filters_payload(parsed_filters) is None:
            parsed_filters = None

    parsed_strategy = _codec_strategy_from_payload(strategy, filters=parsed_filters)
    return replace(cfg, strategy=parsed_strategy)
