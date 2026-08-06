"""Causal opening-minute authority for the XSP dual-clock successor.

The opening bridge owns only one narrow seam: a completed native one-minute
XSP/SPY auction may open an RTH target before the unchanged five-minute v3
owner is ready.  Volume establishes seriousness, joint price transport owns
direction, and the existing v3 owner keeps all later lifecycle authority.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timezone
import hashlib
import math
import statistics

from ..backtest.engine import _run_spot_backtest
from ..backtest.models import BacktestResult, SpotTrade
from ..backtest.spot_tape import (
    PreparedSpotEvaluatorTape,
    prepare_spot_evaluator_tape,
)
from ..chart_data.series import OhlcvBar
from ..time_utils import ET_ZONE
from .live_calibration import calibration_fingerprint
from .xsp_opening_edge_state import XspOpeningEdgeV3StateOwner


XSP_DUAL_CLOCK_VERSION = "xsp.opening-edge-v4-dual-clock-arbitration-p009.v1"
XSP_DUAL_CLOCK_SOURCE_VERSION = "xsp.opening-edge-v4-dual-clock-source-p009.v1"
XSP_DUAL_CLOCK_PAIRED_SCHEMA = "xsp.opening-edge-v4-dual-clock-paired-equity.v1"
XSP_DUAL_CLOCK_TARGET_SCHEMA = "xsp.opening-edge-v4-dual-clock-target.v1"

XSP_DUAL_CLOCK_VOLUME_LEVELS = (5.0, 10.0, 12.0, 20.0)
XSP_DUAL_CLOCK_MINUTE_WINDOW = (5, 7)
XSP_DUAL_CLOCK_TRUE_RANGE_FLOOR = 0.5
XSP_DUAL_CLOCK_VOLUME_FLOOR = 20.0


@dataclass(frozen=True, slots=True)
class XspOpeningEmission:
    """One immutable completed-minute opening authority event."""

    emission_id: str
    trading_date: str
    mechanism: str
    direction: str
    signal_close_utc: str
    entry_clock_utc: str
    minute_index: int
    volume_multiple: float | None
    true_range_multiple: float | None
    volume_level: float
    authority_level: float
    slow_15bar_front: Mapping[str, object]
    recoil_seen: bool
    aligned_streak: int

    def as_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "schema": "xsp.opening-edge-v4-opening-emission.v1",
            "authority": "causal_signal_only_no_orders_no_capital",
            "submitted_orders": 0,
        }


def _utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def _sign(value: float, tolerance: float = 1e-12) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def _direction_name(direction: int) -> str:
    if direction not in {-1, 1}:
        raise ValueError("XSP opening direction must be signed")
    return "up" if direction > 0 else "down"


def _log_bps(right: float, left: float) -> float:
    if not all(math.isfinite(value) and value > 0.0 for value in (right, left)):
        raise ValueError("XSP opening prices must be finite and positive")
    return 10_000.0 * math.log(right / left)


def _true_range_bps(bar: OhlcvBar) -> float:
    return _log_bps(float(bar.high), float(bar.low))


def _positive_median(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if float(value) > 0.0]
    return float(statistics.median(clean)) if clean else None


def _multiple(value: float, baseline: float | None) -> float | None:
    return value / baseline if baseline is not None and baseline > 0.0 else None


def _volume_level(value: float | None) -> float:
    return max(
        (
            bound
            for bound in XSP_DUAL_CLOCK_VOLUME_LEVELS
            if (value or 0.0) >= bound
        ),
        default=0.0,
    )


def _joined_direction(values: tuple[float, ...]) -> int:
    directions = {_sign(value) for value in values}
    return directions.pop() if len(directions) == 1 and 0 not in directions else 0


def _slow_front(
    rows: Sequence[OhlcvBar], index: int, direction: int
) -> dict[str, object]:
    """Project the frozen 15-five-minute-bar (75-minute) front exactly.

    The development evaluator called this a 15-minute front, but its executable
    law spans fifteen completed five-minute bars.  Production preserves the
    economics and corrects the description rather than silently retuning it.
    """

    if index < 17:
        raise ValueError("XSP slow front is underwarmed")

    def slope(offset: int) -> float:
        stop = index - offset
        return _log_bps(float(rows[stop].close), float(rows[stop - 15].close))

    current, prior1, prior2 = (slope(offset) for offset in range(3))
    velocity = current - prior1
    acceleration = velocity - (prior1 - prior2)
    adjusted = (
        direction * current,
        direction * velocity,
        direction * acceleration,
    )
    state = (
        "CONTINUATION"
        if adjusted[0] > 0.0 and adjusted[1] > 0.0
        else "PIVOT_REPAIR"
        if adjusted[0] <= 0.0 and adjusted[1] > 0.0 and adjusted[2] > 0.0
        else "CONFLICT"
    )
    return {
        "shape": "".join(
            "+" if value > 0.0 else "-" if value < 0.0 else "0"
            for value in adjusted
        ),
        "state": state,
        "slope_bps": adjusted[0],
        "velocity_bps": adjusted[1],
        "acceleration_bps": adjusted[2],
        "bars": 15,
        "bar_size": "5 mins",
        "horizon_minutes": 75,
    }


def _emission_id(day: date, mechanism: str, direction: int, at: datetime) -> str:
    identity = "|".join(
        (
            "xsp.p007-opening-minute-acceptance",
            day.isoformat(),
            mechanism,
            str(direction),
            at.isoformat(),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _ordered(rows: Sequence[OhlcvBar], *, name: str) -> tuple[OhlcvBar, ...]:
    values = tuple(rows)
    if any(
        _utc(values[index].ts) <= _utc(values[index - 1].ts)
        for index in range(1, len(values))
    ):
        raise ValueError(f"{name} bars must be ordered and unique")
    return values


def _opening_rows(
    rows: Sequence[OhlcvBar],
) -> dict[date, list[OhlcvBar]]:
    grouped: dict[date, list[OhlcvBar]] = defaultdict(list)
    for row in rows:
        observed = _utc(row.ts).astimezone(ET_ZONE)
        close_time = observed.time().replace(tzinfo=None)
        if time(9, 31) <= close_time <= time(9, 46):
            grouped[observed.date()].append(row)
    return grouped


def _first_p007_emission(
    day: date,
    xrows: Sequence[OhlcvBar],
    srows: Sequence[OhlcvBar],
    spy_full: Sequence[OhlcvBar],
    full_times: Sequence[datetime],
) -> XspOpeningEmission | None:
    if not xrows or len(xrows) != len(srows):
        return None
    if tuple(_utc(row.ts) for row in xrows) != tuple(_utc(row.ts) for row in srows):
        raise ValueError("XSP/SPY opening-minute clocks disagree")

    rth_open = datetime.combine(day, time(9, 30), tzinfo=ET_ZONE).astimezone(
        timezone.utc
    )
    preopen_at = bisect_right(full_times, rth_open) - 1
    preopen_rows = spy_full[max(0, preopen_at - 1) : preopen_at + 1]
    if len(preopen_rows) != 2 or full_times[preopen_at] != rth_open:
        return None
    volume_baseline = _positive_median(
        tuple(float(row.volume) / 5.0 for row in preopen_rows)
    )
    true_range_baseline = _positive_median(
        tuple(_true_range_bps(row) for row in preopen_rows)
    )

    authority = 0
    authority_level = 0.0
    aligned_streak = 0
    opposite_streak = 0
    recoil = False
    session_x_open = float(xrows[0].open)
    session_s_open = float(srows[0].open)
    for index, (xbar, sbar) in enumerate(zip(xrows, srows), start=1):
        close_at = _utc(xbar.ts)
        volume_multiple = _multiple(float(sbar.volume), volume_baseline)
        true_range_multiple = _multiple(
            _true_range_bps(sbar), true_range_baseline
        )
        current_level = _volume_level(volume_multiple)
        path_direction = _joined_direction(
            (
                _log_bps(float(xbar.close), session_x_open),
                _log_bps(float(sbar.close), session_s_open),
            )
        )
        velocity_direction = _joined_direction(
            (
                _log_bps(float(xbar.close), float(xbar.open)),
                _log_bps(float(sbar.close), float(sbar.open)),
            )
        )
        raw_direction = (
            path_direction if path_direction == velocity_direction else 0
        )
        full_at = bisect_right(full_times, close_at) - 1
        probe_direction = raw_direction or authority
        front = (
            _slow_front(spy_full, full_at, probe_direction)
            if probe_direction and full_at >= 17
            else None
        )

        if authority == 0:
            if (
                current_level >= 10.0
                and raw_direction in {-1, 1}
                and front is not None
                and front["state"] in {"CONTINUATION", "PIVOT_REPAIR"}
            ):
                authority = raw_direction
                authority_level = current_level
                aligned_streak = 1
            continue

        same = raw_direction == authority
        opposite = raw_direction == -authority
        recoil_now = velocity_direction == -authority and path_direction == authority
        aligned_streak = aligned_streak + 1 if same else 0
        opposite_streak = opposite_streak + 1 if opposite else 0
        recoil = recoil or recoil_now

        mechanism = None
        if (
            current_level >= 12.0
            and same
            and aligned_streak >= 2
            and not recoil
            and front is not None
            and front["state"] in {"CONTINUATION", "PIVOT_REPAIR"}
        ):
            mechanism = "MINUTE_ACCEPTANCE"
        elif (
            current_level >= 12.0
            and same
            and recoil
            and front is not None
            and front["state"] in {"CONTINUATION", "PIVOT_REPAIR"}
        ):
            mechanism = "RECOIL_REACQUISITION"
        elif (
            current_level >= 12.0
            and opposite_streak >= 2
            and current_level >= authority_level
            and path_direction == -authority
            and front is not None
            and front["state"] in {"CONTINUATION", "PIVOT_REPAIR"}
        ):
            authority = -authority
            authority_level = current_level
            mechanism = "AUTHORITATIVE_HANDOFF"
        if mechanism is None:
            continue
        return XspOpeningEmission(
            emission_id=_emission_id(day, mechanism, authority, close_at),
            trading_date=day.isoformat(),
            mechanism=mechanism,
            direction=_direction_name(authority),
            signal_close_utc=close_at.isoformat(),
            entry_clock_utc=close_at.isoformat(),
            minute_index=index,
            volume_multiple=volume_multiple,
            true_range_multiple=true_range_multiple,
            volume_level=current_level,
            authority_level=authority_level,
            slow_15bar_front=front,
            recoil_seen=recoil,
            aligned_streak=aligned_streak,
        )
    return None


def xsp_dual_clock_emissions(
    *,
    xsp_rth_one_minute: Sequence[OhlcvBar],
    spy_rth_one_minute: Sequence[OhlcvBar],
    spy_full_five_minute: Sequence[OhlcvBar],
    require_complete_windows: bool = False,
) -> tuple[XspOpeningEmission, ...]:
    """Return only the frozen P-009 interior-cell opening emissions."""

    xsp = _ordered(xsp_rth_one_minute, name="XSP one-minute")
    spy = _ordered(spy_rth_one_minute, name="SPY one-minute")
    full = _ordered(spy_full_five_minute, name="SPY full-session five-minute")
    xsp_by_day = _opening_rows(xsp)
    spy_by_day = _opening_rows(spy)
    full_times = tuple(_utc(row.ts) for row in full)
    output = []
    for day in sorted(set(xsp_by_day) & set(spy_by_day)):
        xrows = xsp_by_day[day]
        srows = spy_by_day[day]
        if require_complete_windows and (len(xrows) != 16 or len(srows) != 16):
            raise ValueError(f"XSP opening window is incomplete: {day.isoformat()}")
        emission = _first_p007_emission(day, xrows, srows, full, full_times)
        if (
            emission is not None
            and emission.mechanism == "MINUTE_ACCEPTANCE"
            and XSP_DUAL_CLOCK_MINUTE_WINDOW[0]
            <= emission.minute_index
            <= XSP_DUAL_CLOCK_MINUTE_WINDOW[1]
            and float(emission.true_range_multiple or 0.0)
            >= XSP_DUAL_CLOCK_TRUE_RANGE_FLOOR
            and emission.volume_level >= XSP_DUAL_CLOCK_VOLUME_FLOOR
        ):
            output.append(emission)
    return tuple(output)


def _trade_direction(trade: SpotTrade) -> str:
    return "up" if int(trade.qty) > 0 else "down"


def _marker(snapshot: object, prefix: str) -> str | None:
    controls = getattr(snapshot, "entry_controls", ()) if snapshot is not None else ()
    return next(
        (str(value) for value in controls if str(value).startswith(prefix)),
        None,
    )


class XspDualClockBridgeOwner:
    """Inject frozen opening emissions into the exact five-minute v3 state.

    Only bridge-owned exposure advances on the native one-minute execution
    clock.  Authentic v3 decisions remain projected from completed five-minute
    bars and keep their original lifecycle owner.
    """

    def __init__(
        self,
        *,
        cfg: object,
        v3_bars: Sequence[OhlcvBar],
        exec_bars: Sequence[OhlcvBar],
        emissions: Sequence[XspOpeningEmission],
        daily_context: Sequence[object],
    ) -> None:
        base = XspOpeningEdgeV3StateOwner(daily_context)
        signal_index = {bar.ts: index for index, bar in enumerate(v3_bars)}
        alignment = tuple(signal_index.get(bar.ts, -1) for bar in exec_bars)
        prepared = prepare_spot_evaluator_tape(
            cfg=cfg,
            signal_bars=v3_bars,
            exec_bars=exec_bars,
            sig_idx_by_exec_idx=alignment,
            exec_dates=tuple(
                _utc(bar.ts).astimezone(ET_ZONE).date() for bar in exec_bars
            ),
        )
        projected = base.project_evaluator_tape(
            prepared,
            v3_bars,
            sig_idx_by_exec_idx=alignment,
        )
        self.base = base
        self.exec_bars = tuple(exec_bars)
        self.v3_rows = tuple(
            (
                bar.ts,
                projected.signals[index],
                projected.risks[index],
                projected.prior_shocks[index],
            )
            for index, bar in enumerate(exec_bars)
        )
        self.risk_overlay_enabled = projected.risk_overlay_enabled
        self.shock_enabled = projected.shock_enabled
        self.emissions = {
            _utc(datetime.fromisoformat(row.signal_close_utc)).replace(tzinfo=None): row
            for row in emissions
        }
        self.base_direction_by_time: dict[datetime, str] = {}
        self.bridge_fill_owner: dict[tuple[datetime, str], str] = {}
        self.v3_fill_owner: set[tuple[datetime, str]] = set()
        self.promoted_trade_ids: set[int] = set()

    def hybrid_signal_bars(
        self, v3_bars: Sequence[OhlcvBar]
    ) -> tuple[OhlcvBar, ...]:
        by_time = {bar.ts: bar for bar in v3_bars}
        exec_by_time = {bar.ts: bar for bar in self.exec_bars}
        for observed in self.emissions:
            if observed in exec_by_time:
                by_time.setdefault(observed, exec_by_time[observed])
        return tuple(by_time[observed] for observed in sorted(by_time))

    def _owner(self, trade: SpotTrade) -> str:
        key = (trade.entry_time, _trade_direction(trade))
        if id(trade) in self.promoted_trade_ids or key in self.v3_fill_owner:
            return "v3"
        return "bridge" if key in self.bridge_fill_owner else "v3"

    def project_evaluator_tape(
        self,
        prepared: PreparedSpotEvaluatorTape,
        bars: Sequence[OhlcvBar],
        *,
        sig_idx_by_exec_idx: Sequence[int] | None = None,
    ) -> PreparedSpotEvaluatorTape:
        del bars, sig_idx_by_exec_idx
        output = [None] * len(self.exec_bars)
        risks = [None] * len(self.exec_bars)
        shocks = [(None, None, None)] * len(self.exec_bars)
        latest_snapshot = None
        for index, bar in enumerate(self.exec_bars):
            observed, current_v3, latest_risk, latest_shock = self.v3_rows[index]
            if observed != bar.ts:
                raise AssertionError("XSP dual-clock v3 projection drift")
            latest_snapshot = current_v3 or latest_snapshot
            output[index] = current_v3
            risks[index] = latest_risk
            shocks[index] = latest_shock
            if current_v3 is not None and current_v3.entry_dir in {"up", "down"}:
                direction = str(current_v3.entry_dir)
                self.base_direction_by_time[bar.ts] = direction
                if index + 1 < len(self.exec_bars):
                    self.v3_fill_owner.add((self.exec_bars[index + 1].ts, direction))

            emission = self.emissions.get(bar.ts)
            if emission is None or (
                current_v3 is not None and current_v3.entry_dir in {"up", "down"}
            ):
                continue
            template = current_v3 or latest_snapshot or prepared.signals[index]
            if template is None:
                continue
            direction = emission.direction
            output[index] = replace(
                template,
                bar_ts=bar.ts,
                close=float(bar.close),
                entry_dir=direction,
                entry_branch=None,
                entry_proposed_dir=direction,
                entry_blocked_by=None,
                entry_source="p009_opening_minute_acceptance",
                entry_controls=(
                    *template.entry_controls,
                    f"p009_emission:{emission.emission_id}:{direction}",
                ),
            )
            if index + 1 < len(self.exec_bars):
                self.bridge_fill_owner[
                    (self.exec_bars[index + 1].ts, direction)
                ] = emission.emission_id
        return PreparedSpotEvaluatorTape(
            prior_shocks=tuple(shocks),
            risks=tuple(risks),
            signals=tuple(output),
            risk_overlay_enabled=self.risk_overlay_enabled,
            shock_enabled=self.shock_enabled,
        )

    def excursion_policy_for_trade(self, trade: SpotTrade):
        return self.base.excursion_policy_for_trade(trade)

    def resolve_flip(
        self,
        *,
        trade: SpotTrade,
        bar: OhlcvBar,
        snapshot: object,
        hit: bool,
    ) -> bool:
        owner = self._owner(trade)
        emission = _marker(snapshot, "p009_emission:")
        base_direction = self.base_direction_by_time.get(bar.ts)
        incumbent = _trade_direction(trade)
        if owner == "v3" and emission is not None and base_direction is None:
            return self.base.resolve_flip(
                trade=trade, bar=bar, snapshot=snapshot, hit=False
            )
        if owner == "bridge":
            if base_direction is not None:
                if base_direction == incumbent:
                    self.promoted_trade_ids.add(id(trade))
                    return self.base.resolve_flip(
                        trade=trade, bar=bar, snapshot=snapshot, hit=False
                    )
                return self.base.resolve_flip(
                    trade=trade, bar=bar, snapshot=snapshot, hit=True
                )
            if emission is not None:
                return emission.rsplit(":", maxsplit=1)[-1] != incumbent
        return self.base.resolve_flip(
            trade=trade, bar=bar, snapshot=snapshot, hit=hit
        )

    def state_payload(self) -> dict[str, object]:
        return {
            "schema": "xsp.opening-edge-v4-dual-clock-bridge-state.v1",
            "base": self.base.state_payload(),
            "emission_ids": sorted(row.emission_id for row in self.emissions.values()),
            "order_authority": "none",
        }


def xsp_dual_clock_bridge_result(
    *,
    cfg: object,
    v3_bars: Sequence[OhlcvBar],
    exec_bars: Sequence[OhlcvBar],
    emissions: Sequence[XspOpeningEmission],
    daily_context: Sequence[object],
    meta: object,
    entry_not_before: datetime | None = None,
    final_session_complete: bool = False,
) -> tuple[BacktestResult, XspDualClockBridgeOwner]:
    owner = XspDualClockBridgeOwner(
        cfg=cfg,
        v3_bars=v3_bars,
        exec_bars=exec_bars,
        emissions=emissions,
        daily_context=daily_context,
    )
    result = _run_spot_backtest(
        cfg,
        owner.hybrid_signal_bars(v3_bars),
        meta,
        exec_bars=exec_bars,
        final_session_complete=final_session_complete,
        spot_state_owner=owner,
        entry_not_before=entry_not_before,
    )
    return result, owner


def xsp_dual_clock_target(
    *,
    bridge_result: BacktestResult,
    bridge_owner: XspDualClockBridgeOwner,
    v3_position: Mapping[str, object] | None,
    observed_at: datetime,
) -> dict[str, object] | None:
    """Project the current bridge target; otherwise preserve exact v3 state."""

    observed_naive = _utc(observed_at).replace(tzinfo=None)
    active_bridge = next(
        (
            (trade, bridge_owner.bridge_fill_owner[key])
            for trade in reversed(bridge_result.trades)
            if (
                key := (trade.entry_time, _trade_direction(trade))
            ) in bridge_owner.bridge_fill_owner
            and trade.entry_time <= observed_naive
            and (trade.exit_reason == "end" or trade.exit_time > observed_naive)
        ),
        None,
    )
    if active_bridge is None:
        if not isinstance(v3_position, Mapping):
            return None
        return {
            "schema": XSP_DUAL_CLOCK_TARGET_SCHEMA,
            "lane": v3_position.get("lane"),
            "direction": v3_position.get("direction"),
            "entry_time": v3_position.get("entry_time"),
            "trading_date": v3_position.get("trading_date"),
            "entry_price": v3_position.get("entry_price"),
            "exit_reason": v3_position.get("exit_reason"),
            "owner": "v3",
            "emission_id": None,
            "bridge_fill_time_utc": None,
            "execution_signal_context": None,
            "order_authority": "none",
        }

    trade, emission_id = active_bridge
    emission = next(
        row for row in bridge_owner.emissions.values() if row.emission_id == emission_id
    )
    signal_context = {
        "schema": "xsp.execution-signal-context.v1",
        "lane": "rth",
        "direction": emission.direction,
        "entry_time_utc": emission.signal_close_utc,
        "signal_bar_ts": emission.signal_close_utc,
        "decision_trace_fingerprint": calibration_fingerprint(
            {
                "emission": emission.as_payload(),
                "owner": XSP_DUAL_CLOCK_VERSION,
            }
        ),
        "control": {
            "source": "p009_opening_minute_acceptance",
            "mechanism": emission.mechanism,
            "volume_level": emission.volume_level,
        },
        "directional_impulse": {
            "source": "joint_XSP_SPY_opening_transport",
            "direction": emission.direction,
            "volume_multiple": emission.volume_multiple,
            "true_range_multiple": emission.true_range_multiple,
            "slow_front": dict(emission.slow_15bar_front),
        },
        "market_state": {
            "owner": "opening_bridge",
            "authority_level": emission.authority_level,
            "recoil_seen": emission.recoil_seen,
            "aligned_streak": emission.aligned_streak,
        },
        "local_extrema": None,
    }
    return {
        "schema": XSP_DUAL_CLOCK_TARGET_SCHEMA,
        "lane": "rth",
        "direction": emission.direction,
        "entry_time": emission.signal_close_utc,
        "trading_date": emission.trading_date,
        "entry_price": float(trade.entry_price),
        "exit_reason": "end",
        "owner": (
            "v3_handoff"
            if id(trade) in bridge_owner.promoted_trade_ids
            else "opening_bridge"
        ),
        "emission_id": emission_id,
        "bridge_fill_time_utc": _utc(trade.entry_time).isoformat(),
        "execution_signal_context": signal_context,
        "order_authority": "none",
    }
