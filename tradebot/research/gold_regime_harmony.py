"""Frozen Stage-76 gold signal owner and deterministic historical replay."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..backtest.data import ContractMeta
from ..backtest.engine import _backtest_window, _run_spot_backtest
from ..backtest.models import BacktestResult, SpotTrade
from ..backtest.spot_tape import PreparedSpotEvaluatorTape
from ..chart_data.history import normalize_bars_to_close, read_cache
from ..spot.champions import discover_current_champions, load_champion_group
from ..spot.fill_modes import SPOT_FILL_MODE_NEXT_TRADABLE_BAR
from .gold_context import (
    gold_daily_timeline,
    gold_h4_timeline,
    gold_macro_timeline,
    gold_utc,
)
from .spot_sweeps.support import _bundle_base


GOLD_REGIME_HARMONY_VERSION = "gold.1oz-regime-harmony-stage76.v1"
GOLD_REGIME_HARMONY_AUTHORITY = "research_only_no_selection_no_capital_no_orders"
GOLD_REGIME_HARMONY_STRATEGY_KEY = "one-oz-regime-harmony-stage76-77"
GOLD_REGIME_HARMONY_SOURCE_START = date(2016, 7, 1)
GOLD_REGIME_HARMONY_SOURCE_END = date(2026, 6, 30)
GOLD_REGIME_HARMONY_FINANCING_USD = 2.32
GOLD_REGIME_HARMONY_TARGET_OUTCOMES = frozenset(("prior_win",))
GOLD_REGIME_HARMONY_FULL3_LEDGER = (
    "c3add7cde287718b895de70936aa9aeab18b465bd6e68d3b8140c262650a3733"
)
GOLD_REGIME_HARMONY_FULL10_LEDGER = (
    "f5c212b19b9c5b6f697f01b6162a2f4382d8503afa4ebaa3844add45326a35a5"
)
_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc
_XAU_CACHE_PATHS = {
    "1 hour": (
        "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_1hour_full24.csv",
        "db/XAUUSD/XAUUSD_2016-07-01_2026-08-02_1hour_full24.csv",
    ),
    "4 hours": (
        "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_4hours_full24.csv",
        "db/XAUUSD/XAUUSD_2016-01-01_2026-08-02_4hours_full24.csv",
    ),
    "1 day": (
        "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_1day_full24.csv",
        "db/XAUUSD/XAUUSD_2016-01-01_2026-08-02_1day_full24.csv",
    ),
}
_MACRO_CACHE_PATHS = {
    "UUP": "db/UUP/UUP_2015-07-01_2026-08-02_1day_rth.csv",
    "TIP": "db/TIP/TIP_2015-07-01_2026-08-02_1day_rth.csv",
}


def _canonical_hash(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _naive_utc(value: object) -> datetime:
    return gold_utc(value).replace(tzinfo=None)


@dataclass(frozen=True)
class GoldRegimeHarmonyTape:
    h1: tuple[object, ...]
    h4: tuple[object, ...]
    daily: tuple[object, ...]
    uup: tuple[object, ...]
    tip: tuple[object, ...]

    @property
    def as_of(self) -> datetime:
        if not self.h4:
            raise ValueError("gold tape has no H4 bars")
        return gold_utc(getattr(self.h4[-1], "ts"))


def _combined_cache(root: Path, paths: Sequence[str]) -> list[object]:
    rows = {
        getattr(bar, "ts"): bar
        for relative in paths
        for bar in read_cache(root / relative)
    }
    return [rows[stamp] for stamp in sorted(rows)]


def load_gold_regime_harmony_tape(
    *, root: Path | None = None
) -> GoldRegimeHarmonyTape:
    """Load the frozen causal research tape without broker or order authority."""

    base = (root or Path(__file__).resolve().parents[2]).resolve()
    xau = {
        size: tuple(
            normalize_bars_to_close(
                _combined_cache(base, paths),
                symbol="XAUUSD",
                bar_size=size,
                use_rth=False,
            )
        )
        for size, paths in _XAU_CACHE_PATHS.items()
    }
    macro = {
        symbol: tuple(
            normalize_bars_to_close(
                read_cache(base / relative),
                symbol=symbol,
                bar_size="1 day",
                use_rth=True,
            )
        )
        for symbol, relative in _MACRO_CACHE_PATHS.items()
    }
    return GoldRegimeHarmonyTape(
        h1=xau["1 hour"],
        h4=xau["4 hours"],
        daily=xau["1 day"],
        uup=macro["UUP"],
        tip=macro["TIP"],
    )


def load_gold_regime_harmony_crown(
    *, root: Path | None = None
) -> dict[str, object]:
    """Validate the machine crown and return its immutable identity."""

    base = (root or Path(__file__).resolve().parents[2]).resolve()
    refs = discover_current_champions(
        root=base,
        symbols=("1OZ",),
        tracks=("LF",),
    )
    if len(refs) != 1:
        raise ValueError("gold Stage-76 crown is not uniquely declared")
    ref = refs[0]
    if ref.strategy_key != GOLD_REGIME_HARMONY_STRATEGY_KEY:
        raise ValueError("gold Stage-76 strategy identity drifted")
    declaration_bytes = ref.declaration_path.read_bytes()
    artifact_bytes = ref.artifact_path.read_bytes()
    declaration = json.loads(declaration_bytes)
    artifact = json.loads(artifact_bytes)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if declaration.get("artifact_sha256") != artifact_hash:
        raise ValueError("gold Stage-76 artifact hash drifted")
    group = load_champion_group(ref)
    if group is None or group.get("_key") != GOLD_REGIME_HARMONY_STRATEGY_KEY:
        raise ValueError("gold Stage-76 crown group is not loadable")
    if any(
        artifact.get(field) != "none"
        for field in ("order_authority", "selection_authority", "capital_authority")
    ):
        raise ValueError("gold historical crown unexpectedly owns live authority")
    return {
        "strategy_version": GOLD_REGIME_HARMONY_VERSION,
        "authority": GOLD_REGIME_HARMONY_AUTHORITY,
        "declaration_path": ref.declaration_path.relative_to(base).as_posix(),
        "declaration_sha256": hashlib.sha256(declaration_bytes).hexdigest(),
        "artifact_path": ref.artifact_path.relative_to(base).as_posix(),
        "artifact_sha256": artifact_hash,
        "strategy_key": ref.strategy_key,
        "version": ref.version,
    }


def gold_regime_harmony_config(start: date, end: date):
    base = _bundle_base(
        symbol="XAUUSD",
        start=start,
        end=end,
        bar_size="4 hours",
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=None,
        starting_cash=100_000.0,
        spot_profit_target_pct=None,
        spot_stop_loss_pct=None,
        flip_exit_min_hold_bars=0,
        spot_close_eod=False,
    )
    strategy = replace(
        base.strategy,
        ema_preset="8/21",
        ema_entry_mode="cross",
        entry_signal="ema",
        direction_source="ema",
        entry_confirm_bars=1,
        entry_days=tuple(range(7)),
        max_entries_per_day=0,
        regime_mode="off",
        regime2_mode="ema",
        regime2_apply_to="both",
        regime2_ema_preset="21/50",
        regime2_bar_size="1 day",
        exit_on_signal_flip=True,
        spot_controlled_flip=True,
        flip_exit_mode="entry",
        flip_exit_gate_mode="off",
        flip_exit_min_hold_bars=0,
        flip_exit_only_if_profit=False,
        spot_profit_target_pct=None,
        spot_stop_loss_pct=None,
        spot_close_eod=False,
        spot_entry_fill_mode=SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
        spot_flip_exit_fill_mode=SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
        spot_next_open_session="tradable_24x5",
        spot_intrabar_exits=False,
        spot_exec_bar_size="1 hour",
        spot_spread=0.50,
        spot_commission_per_share=0.66,
        spot_commission_min=0.66,
        spot_slippage_per_share=0.25,
        spot_mark_to_market="liquidation",
        spot_drawdown_mode="intrabar",
        spot_sizing_mode="fixed",
        spot_min_qty=1,
        spot_max_qty=1,
        spot_sec_type="FUT",
        spot_exchange="SMART",
        quantity=1,
    )
    return replace(base, strategy=strategy)


class GoldHardRegimeOwner:
    """Apply the frozen six-completed-D1 hard-regime admission law."""

    def __init__(
        self,
        daily_context: Sequence[Mapping[str, object]],
        *,
        minimum_hard_age: int = 6,
    ) -> None:
        self._rows = tuple(
            (
                gold_utc(row["end"]),
                row.get("hard_direction"),
                row.get("hard_age"),
            )
            for row in daily_context
        )
        self._timestamps = tuple(row[0] for row in self._rows)
        self._minimum_hard_age = int(minimum_hard_age)

    def state_id(self, timestamp: object, direction: str) -> str:
        index = bisect_right(self._timestamps, gold_utc(timestamp)) - 1
        if index < 0:
            raise ValueError("gold decision has no completed D1 state")
        _stamp, hard_direction, hard_age = self._rows[index]
        if hard_direction != direction or hard_age is None:
            raise ValueError("gold decision disagrees with completed D1 hard state")
        birth_index = index - int(hard_age) + 1
        if birth_index < 0:
            raise ValueError("gold D1 state birth precedes causal tape")
        return f"{direction}:{self._timestamps[birth_index].isoformat()}"

    def project_evaluator_tape(
        self,
        prepared: PreparedSpotEvaluatorTape,
        bars: Sequence[object],
        *,
        sig_idx_by_exec_idx: Sequence[int] | None = None,
    ) -> PreparedSpotEvaluatorTape:
        del bars, sig_idx_by_exec_idx
        output = []
        for snapshot in prepared.signals:
            if snapshot is None or snapshot.entry_dir not in ("up", "down"):
                output.append(snapshot)
                continue
            index = bisect_right(self._timestamps, gold_utc(snapshot.bar_ts)) - 1
            hard_direction, hard_age = (
                (self._rows[index][1], self._rows[index][2])
                if index >= 0
                else (None, None)
            )
            output.append(
                snapshot
                if hard_direction == snapshot.entry_dir
                and hard_age is not None
                and int(hard_age) >= self._minimum_hard_age
                else replace(
                    snapshot,
                    entry_dir=None,
                    entry_blocked_by="gold_hard_regime_maturation",
                )
            )
        return replace(prepared, signals=tuple(output))

    @staticmethod
    def excursion_policy_for_trade(_trade: object) -> None:
        return None

    @staticmethod
    def resolve_flip(
        *, trade: object, bar: object, snapshot: object, hit: bool
    ) -> bool:
        del trade, bar, snapshot
        return bool(hit)


class GoldRegimeHarmonyOwner(GoldHardRegimeOwner):
    """Own Stage-76 admission memory and full-cycle financed surrender."""

    def __init__(
        self,
        daily_context: Sequence[Mapping[str, object]],
        h4_context: Sequence[Mapping[str, object]],
        macro_context: Sequence[Mapping[str, object]],
        *,
        predecessors: Mapping[str, Sequence[Mapping[str, object]]],
        source_records: Mapping[str, Sequence[float]],
    ) -> None:
        super().__init__(daily_context, minimum_hard_age=6)
        self._predecessors = {
            key: tuple(rows) for key, rows in predecessors.items()
        }
        self._source_records = {
            key: tuple(float(value) for value in values)
            for key, values in source_records.items()
        }
        self._h4 = tuple(h4_context)
        self._h4_times = tuple(gold_utc(row["end"]) for row in self._h4)
        self._macro = tuple(macro_context)
        self._macro_times = tuple(gold_utc(row["end"]) for row in self._macro)
        self._epoch_order = tuple(
            f"{direction}:{stamp.isoformat()}"
            for stamp, direction, age in self._rows
            if direction in ("up", "down") and age == 1
        )
        self._epoch_index = {
            epoch: index for index, epoch in enumerate(self._epoch_order)
        }
        self._active_trade = None
        self._active_epoch = None
        self._record_by_trade: dict[int, float | None] = {}
        self._entry_atr: dict[int, float] = {}
        self.events: list[dict[str, object]] = []
        self.classifications: list[dict[str, object]] = []
        self.tail_hits: list[dict[str, object]] = []

    @staticmethod
    def _source_direction(snapshot: object) -> str | None:
        signal = getattr(snapshot, "signal", None)
        state = getattr(signal, "state", None)
        return str(state) if state in ("up", "down") else None

    def _full_cycle_epochs(self, epoch: str) -> tuple[str, str] | None:
        index = self._epoch_index.get(epoch)
        if index is None or index < 2:
            return None
        direction = epoch.split(":", 1)[0]
        opposite = self._epoch_order[index - 1]
        same = next(
            (
                candidate
                for candidate in reversed(self._epoch_order[: index - 1])
                if candidate.split(":", 1)[0] == direction
            ),
            None,
        )
        if same is None or opposite.split(":", 1)[0] == direction:
            return None
        return same, opposite

    def _macro_state(self, timestamp: object) -> dict[str, object]:
        index = bisect_right(self._macro_times, gold_utc(timestamp)) - 1
        if index < 0:
            return {
                "ready": False,
                "neutral": False,
                "supportive_horizons": 0,
                "adverse_horizons": 0,
                "completed_at": None,
            }
        row = self._macro[index]
        horizons = row["horizons"]
        labels = [horizons[key]["direction"] for key in ("5", "21", "63")]
        return {
            "ready": True,
            "neutral": all(label == "mixed" for label in labels),
            "supportive_horizons": sum(label == "supportive" for label in labels),
            "adverse_horizons": sum(label == "adverse" for label in labels),
            "completed_at": gold_utc(row["end"]).isoformat(),
        }

    def _prior(self, timestamp: object, direction: str):
        state_id = self.state_id(timestamp, direction)
        stamp = gold_utc(timestamp)
        for row in reversed(self._predecessors.get(state_id, ())):
            if gold_utc(row["exit_time"]) < stamp:
                return state_id, row
        return state_id, None

    def _bind_trade(self, trade: SpotTrade) -> None:
        if self._active_trade is trade:
            return
        direction = "up" if int(trade.qty) > 0 else "down"
        epoch = self.state_id(trade.entry_time, direction)
        cycle = self._full_cycle_epochs(epoch)
        values = (
            [
                value
                for key in cycle
                for value in self._source_records.get(key, ())
            ]
            if cycle is not None
            else []
        )
        self._active_trade = trade
        self._active_epoch = epoch
        self._record_by_trade[id(trade)] = max(values) if values else None

    def resolve_flip(
        self, *, trade: SpotTrade, bar: object, snapshot: object, hit: bool
    ) -> bool:
        self._bind_trade(trade)
        if hit:
            return True
        if snapshot is None or getattr(snapshot, "bar_ts", None) != getattr(bar, "ts"):
            return False
        prior_record = self._record_by_trade[id(trade)]
        if prior_record is None:
            return False
        running_mfe_pct = float(trade.max_favorable_excursion) / float(
            trade.entry_price
        )
        if running_mfe_pct < prior_record:
            return False
        index = bisect_right(self._h4_times, gold_utc(snapshot.bar_ts)) - 1
        entry_index = bisect_right(self._h4_times, gold_utc(trade.entry_time)) - 1
        if index < 0 or entry_index < 0:
            return False
        key = id(trade)
        entry_atr = self._entry_atr.get(key)
        if entry_atr is None:
            entry_context = self._h4[entry_index]
            atr = entry_context.get("atr14_dollars")
            source_close = entry_context.get("close")
            if (
                atr is None
                or source_close is None
                or float(atr) <= 0.0
                or float(source_close) <= 0.0
            ):
                return False
            entry_atr = (
                float(atr) / float(source_close) * float(trade.entry_price)
            )
            self._entry_atr[key] = entry_atr
        direction = "up" if int(trade.qty) > 0 else "down"
        sign = 1.0 if direction == "up" else -1.0
        context = self._h4[index]
        fast = context.get("fast_slope_pct")
        velocity = context.get("spread_velocity_pct")
        acceleration = context.get("fast_acceleration_pct")
        if fast is None or velocity is None or acceleration is None:
            return False
        running_mfe = float(trade.max_favorable_excursion)
        terminal = sign * (float(getattr(bar, "close")) - float(trade.entry_price))
        giveback = running_mfe - terminal
        triggered = bool(
            giveback >= 2.0 * entry_atr
            and sign * float(fast) <= 0.0
            and sign * float(velocity) <= 0.0
            and sign * float(acceleration) <= 0.0
        )
        if triggered:
            self.tail_hits.append(
                {
                    "entry_time": trade.entry_time.isoformat(),
                    "trigger_time": snapshot.bar_ts.isoformat(),
                    "direction": direction,
                    "epoch": self._active_epoch,
                    "cycle_record_mfe_pct": prior_record,
                    "running_mfe_pct": running_mfe_pct,
                    "giveback_atr": giveback / entry_atr,
                }
            )
        return triggered

    def project_evaluator_tape(
        self,
        prepared: PreparedSpotEvaluatorTape,
        bars: Sequence[object],
        *,
        sig_idx_by_exec_idx: Sequence[int] | None = None,
    ) -> PreparedSpotEvaluatorTape:
        gated = super().project_evaluator_tape(
            prepared,
            bars,
            sig_idx_by_exec_idx=sig_idx_by_exec_idx,
        )
        output = []
        projected_by_ts = {}
        source_epoch = 0
        last_source = object()
        blocked_epochs: set[int] = set()
        for raw, admitted in zip(prepared.signals, gated.signals, strict=True):
            if raw is None:
                output.append(None)
                continue
            if raw.bar_ts in projected_by_ts:
                output.append(projected_by_ts[raw.bar_ts])
                continue
            source_state = self._source_direction(raw)
            if source_state != last_source:
                source_epoch += 1
                last_source = source_state
            projected = admitted
            if admitted is not None and admitted.entry_dir in ("up", "down"):
                direction = str(admitted.entry_dir)
                if source_epoch not in blocked_epochs:
                    macro = self._macro_state(raw.bar_ts)
                    state_id, predecessor = self._prior(raw.bar_ts, direction)
                    targeted = bool(
                        macro["ready"]
                        and macro["neutral"]
                        and predecessor is not None
                        and predecessor["outcome"]
                        in GOLD_REGIME_HARMONY_TARGET_OUTCOMES
                    )
                    classification = {
                        "trigger_time": raw.bar_ts.isoformat(),
                        "source_epoch": source_epoch,
                        "direction": direction,
                        "state_id": state_id,
                        "predecessor": predecessor,
                        "macro": macro,
                        "targeted": targeted,
                    }
                    self.classifications.append(classification)
                    if targeted:
                        blocked_epochs.add(source_epoch)
                        self.events.append(
                            {**classification, "verdict": "reject_source_epoch"}
                        )
                if source_epoch in blocked_epochs:
                    projected = replace(
                        admitted,
                        entry_dir=None,
                        entry_blocked_by="gold_winner_exhaustion_veto",
                        entry_controls=(
                            *admitted.entry_controls,
                            "gold_winner_exhaustion_veto:reject_source_epoch",
                        ),
                    )
                    if projected.lifecycle_inputs()["signal_entry_dir"] is not None:
                        raise AssertionError("gold winner veto retained admission")
            projected_by_ts[raw.bar_ts] = projected
            output.append(projected)
        return replace(prepared, signals=tuple(output))

    def state_payload(self, result: BacktestResult) -> dict[str, object]:
        open_trades = [trade for trade in result.trades if trade.exit_time is None]
        if len(open_trades) > 1:
            raise ValueError("gold owner projected multiple open positions")
        trade = open_trades[0] if open_trades else None
        return {
            "schema": "gold.1oz-regime-harmony-state.v1",
            "strategy_version": GOLD_REGIME_HARMONY_VERSION,
            "authority": GOLD_REGIME_HARMONY_AUTHORITY,
            "target_direction": (
                "up" if trade is not None and int(trade.qty) > 0 else
                "down" if trade is not None else None
            ),
            "synthetic_midcycle_entry_authority": "none",
            "admission_events": len(self.classifications),
            "blocked_source_epochs": len(self.events),
            "tail_surrenders": len(self.tail_hits),
            "state_sha256": _canonical_hash(
                {
                    "predecessors": self._predecessors,
                    "events": self.events,
                    "tail_hits": self.tail_hits,
                    "target": (
                        None
                        if trade is None
                        else {
                            "direction": "up" if int(trade.qty) > 0 else "down",
                            "entry_time": trade.entry_time.isoformat(),
                            "entry_price": trade.entry_price,
                        }
                    ),
                }
            ),
            "order_authority": "none",
            "submitted_orders": 0,
        }


def _signal_time(trade: SpotTrade) -> datetime:
    trace = trade.decision_trace
    value = (
        trace.get("entry_guard_inputs", {}).get("signal_bar_ts")
        if isinstance(trace, dict)
        else None
    )
    if not isinstance(value, str):
        raise ValueError("gold closed trade lacks exact signal timestamp")
    return datetime.fromisoformat(value)


def _closed_rows(
    result: BacktestResult,
    owner: GoldHardRegimeOwner,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for raw in result.trades:
        if not isinstance(raw, SpotTrade) or raw.exit_time is None:
            continue
        direction = "up" if int(raw.qty) > 0 else "down"
        decision_time = _signal_time(raw)
        state_id = owner.state_id(decision_time, direction)
        pnl = float(raw.pnl(1.0))
        mfe = float(raw.max_favorable_excursion)
        outcome = (
            "prior_win"
            if pnl >= 0.0
            else "prior_financed_loss"
            if mfe >= GOLD_REGIME_HARMONY_FINANCING_USD
            else "prior_unfinanced_loss"
        )
        grouped.setdefault(state_id, []).append(
            {
                "state_id": state_id,
                "direction": direction,
                "decision_time": decision_time.isoformat(),
                "entry_time": raw.entry_time.isoformat(),
                "exit_time": raw.exit_time.isoformat(),
                "outcome": outcome,
                "pnl": round(pnl, 8),
                "mfe": round(mfe, 8),
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda row: (str(row["exit_time"]), str(row["entry_time"])))
    return dict(sorted(grouped.items()))


def _open_identity(result: BacktestResult) -> dict[str, object] | None:
    open_trades = [
        trade
        for trade in result.trades
        if isinstance(trade, SpotTrade) and trade.exit_time is None
    ]
    if not open_trades:
        return None
    if len(open_trades) != 1:
        raise ValueError("gold replay has multiple open positions")
    trade = open_trades[0]
    return {
        "direction": "up" if int(trade.qty) > 0 else "down",
        "entry_time": trade.entry_time.isoformat(),
        "entry_price": round(float(trade.entry_price), 8),
        "mfe": round(float(trade.max_favorable_excursion), 8),
        "mae": round(float(trade.max_adverse_excursion), 8),
    }


class GoldRegimeHarmonyReplay:
    """Cold-replay the exact Stage-76 owner from explicit causal bar inputs."""

    def __init__(self, tape: GoldRegimeHarmonyTape) -> None:
        self.tape = tape
        self.daily_context = tuple(
            gold_daily_timeline(tape.daily, as_of=tape.as_of)
        )
        self.h4_context = tuple(gold_h4_timeline(tape.h4, as_of=tape.as_of))
        self.macro_context = tuple(
            gold_macro_timeline(tape.uup, tape.tip, as_of=tape.as_of)
        )
        self._source_cache: dict[
            tuple[date, date, bool], dict[str, tuple[float, ...]]
        ] = {}

    def _run(
        self,
        start: date,
        end: date,
        *,
        owner: object,
        final_session_complete: bool,
    ) -> BacktestResult:
        cfg = gold_regime_harmony_config(start, end)
        entry_not_before, resolved_end, warmup_start = _backtest_window(cfg)
        return _run_spot_backtest(
            cfg,
            [
                bar
                for bar in self.tape.h4
                if warmup_start <= getattr(bar, "ts") <= resolved_end
            ],
            ContractMeta(
                symbol="XAUUSD",
                exchange="SMART",
                multiplier=1.0,
                min_tick=0.01,
            ),
            final_session_complete=final_session_complete,
            regime2_bars=[
                bar for bar in self.tape.daily if getattr(bar, "ts") <= resolved_end
            ],
            exec_bars=[
                bar for bar in self.tape.h1 if getattr(bar, "ts") <= resolved_end
            ],
            entry_not_before=entry_not_before,
            spot_state_owner=owner,
        )

    def stage7_window(
        self,
        start: date,
        end: date,
        *,
        final_session_complete: bool = True,
    ) -> tuple[BacktestResult, GoldHardRegimeOwner]:
        owner = GoldHardRegimeOwner(self.daily_context)
        return (
            self._run(
                start,
                end,
                owner=owner,
                final_session_complete=final_session_complete,
            ),
            owner,
        )

    def source_records(
        self,
        start: date,
        end: date,
        *,
        final_session_complete: bool,
    ) -> dict[str, tuple[float, ...]]:
        key = (start, end, final_session_complete)
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        result, owner = self.stage7_window(
            start,
            end,
            final_session_complete=final_session_complete,
        )
        records: dict[str, list[float]] = defaultdict(list)
        for raw in result.trades:
            if not isinstance(raw, SpotTrade) or raw.exit_time is None:
                continue
            direction = "up" if int(raw.qty) > 0 else "down"
            records[owner.state_id(raw.entry_time, direction)].append(
                float(raw.max_favorable_excursion) / float(raw.entry_price)
            )
        frozen = {key: tuple(values) for key, values in records.items()}
        self._source_cache[key] = frozen
        return frozen

    def converged_window(
        self,
        start: date,
        end: date,
        *,
        source_start: date = GOLD_REGIME_HARMONY_SOURCE_START,
        source_end: date = GOLD_REGIME_HARMONY_SOURCE_END,
        final_session_complete: bool = True,
    ) -> tuple[BacktestResult, GoldRegimeHarmonyOwner, list[dict[str, object]], bool]:
        source_records = self.source_records(
            source_start,
            source_end,
            final_session_complete=final_session_complete,
        )
        prior_result, prior_owner = self.stage7_window(
            start,
            end,
            final_session_complete=final_session_complete,
        )
        predecessors = _closed_rows(prior_result, prior_owner)
        prior_identity = (
            gold_regime_harmony_summary(prior_result, start, end)["ledger_sha256"],
            _canonical_hash(predecessors),
            _canonical_hash(_open_identity(prior_result)),
        )
        seen = {prior_identity}
        trace: list[dict[str, object]] = [
            {"iteration": 0, "ledger_sha256": prior_identity[0]}
        ]
        result = prior_result
        owner = GoldRegimeHarmonyOwner(
            self.daily_context,
            self.h4_context,
            self.macro_context,
            predecessors=predecessors,
            source_records=source_records,
        )
        for iteration in range(1, 33):
            owner = GoldRegimeHarmonyOwner(
                self.daily_context,
                self.h4_context,
                self.macro_context,
                predecessors=predecessors,
                source_records=source_records,
            )
            result = self._run(
                start,
                end,
                owner=owner,
                final_session_complete=final_session_complete,
            )
            summary = gold_regime_harmony_summary(result, start, end)
            next_predecessors = _closed_rows(result, owner)
            identity = (
                summary["ledger_sha256"],
                _canonical_hash(next_predecessors),
                _canonical_hash(_open_identity(result)),
            )
            trace.append(
                {
                    "iteration": iteration,
                    "ledger_sha256": identity[0],
                    "trades": summary["trades"],
                    "targets": sum(
                        bool(row["targeted"]) for row in owner.classifications
                    ),
                    "events": len(owner.events),
                    "tail_hits": len(owner.tail_hits),
                }
            )
            if identity == prior_identity:
                return result, owner, trace, True
            if identity in seen:
                trace[-1]["oscillation"] = True
                return result, owner, trace, False
            seen.add(identity)
            predecessors = next_predecessors
            prior_identity = identity
        trace[-1]["iteration_limit"] = True
        return result, owner, trace, False


def _gold_clocks(timestamp: datetime) -> tuple[str, str]:
    stamp = _naive_utc(timestamp).replace(tzinfo=_UTC).astimezone(_ET)
    legacy_rth = (
        stamp.weekday() < 5
        and (stamp.hour > 9 or (stamp.hour == 9 and stamp.minute >= 30))
        and stamp.hour < 16
    )
    if stamp.hour >= 18 or stamp.hour < 2:
        session = "asia_reopen"
    elif stamp.hour < 8:
        session = "london"
    elif stamp.hour < 14:
        session = "comex_core"
    elif stamp.hour < 17:
        session = "post_settlement"
    else:
        session = "maintenance_boundary"
    return ("rth" if legacy_rth else "gth", session)


def gold_regime_harmony_summary(
    result: BacktestResult,
    start: date,
    end: date,
) -> dict[str, object]:
    trades = [
        trade
        for trade in result.trades
        if isinstance(trade, SpotTrade) and trade.exit_time is not None
    ]
    pnls = [float(trade.pnl(1.0)) for trade in trades]
    positive = [value for value in pnls if value > 0.0]
    negative = [value for value in pnls if value < 0.0]
    directions = {
        "up": {"trades": 0, "net": 0.0},
        "down": {"trades": 0, "net": 0.0},
    }
    legacy_sessions = {
        "rth": {"trades": 0, "net": 0.0},
        "gth": {"trades": 0, "net": 0.0},
    }
    gold_sessions: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"trades": 0, "net": 0.0}
    )
    months: dict[str, dict[str, float | int]] = {}
    ledger = []
    for trade, pnl in zip(trades, pnls, strict=True):
        direction = "up" if int(trade.qty) > 0 else "down"
        legacy, gold = _gold_clocks(trade.entry_time)
        month = trade.entry_time.strftime("%Y-%m")
        directions[direction]["trades"] += 1
        directions[direction]["net"] += pnl
        legacy_sessions[legacy]["trades"] += 1
        legacy_sessions[legacy]["net"] += pnl
        gold_sessions[gold]["trades"] += 1
        gold_sessions[gold]["net"] += pnl
        month_row = months.setdefault(month, {"trades": 0, "net": 0.0})
        month_row["trades"] += 1
        month_row["net"] += pnl
        ledger.append(
            {
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "direction": direction,
                "entry_price": round(float(trade.entry_price), 8),
                "exit_price": round(float(trade.exit_price), 8),
                "exit_reason": trade.exit_reason,
                "pnl": round(pnl, 8),
                "mfe": round(float(trade.max_favorable_excursion), 8),
                "mae": round(float(trade.max_adverse_excursion), 8),
            }
        )
    gross_win = sum(positive)
    gross_loss = -sum(negative)
    elapsed_years = max(1.0 / 365.25, ((end - start).days + 1) / 365.25)
    return {
        "trades": len(trades),
        "trades_per_year": len(trades) / elapsed_years,
        "net": sum(pnls),
        "gross_before_cost": sum(pnls)
        + GOLD_REGIME_HARMONY_FINANCING_USD * len(trades),
        "profit_factor": gross_win / gross_loss if gross_loss > 0.0 else None,
        "max_drawdown": float(result.summary.max_drawdown),
        "win_rate": float(result.summary.win_rate),
        "directions": directions,
        "legacy_sessions": legacy_sessions,
        "gold_sessions": dict(sorted(gold_sessions.items())),
        "positive_months": sum(float(row["net"]) > 0.0 for row in months.values()),
        "months": len(months),
        "top_five_positive_share": (
            sum(sorted(positive, reverse=True)[:5]) / gross_win
            if gross_win > 0.0
            else None
        ),
        "ledger_sha256": _canonical_hash(ledger),
    }
