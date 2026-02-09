"""Shared bot UI data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ib_insync import Contract, Trade

from ..engine import EmaDecisionSnapshot, RiskOverlaySnapshot

_BOT_JOURNAL_FIELDS = (
    "ts_et",
    "ts_utc",
    "event",
    "instance_id",
    "group",
    "symbol",
    "instrument",
    "action",
    "qty",
    "limit_price",
    "order_id",
    "status",
    "reason",
    "data_json",
)


@dataclass(frozen=True)
class _BotPreset:
    group: str
    entry: dict


@dataclass(frozen=True)
class _PresetHeader:
    node_id: str
    depth: int
    label: str


@dataclass
class _BotInstance:
    instance_id: int
    group: str
    symbol: str
    strategy: dict
    filters: dict | None
    metrics: dict | None = None
    state: str = "RUNNING"
    last_order_date: date | None = None
    open_direction: str | None = None
    last_entry_bar_ts: datetime | None = None
    last_exit_bar_ts: datetime | None = None
    exit_retry_bar_ts: datetime | None = None
    exit_retry_count: int = 0
    exit_retry_cooldown_until: datetime | None = None
    entries_today: int = 0
    entries_today_date: date | None = None
    error: str | None = None
    spot_profit_target_price: float | None = None
    spot_stop_loss_price: float | None = None
    spot_entry_basis_price: float | None = None
    spot_entry_basis_source: str | None = None
    spot_entry_basis_set_ts: datetime | None = None
    touched_conids: set[int] = field(default_factory=set)
    last_signal_fingerprint: tuple | None = None
    last_cross_bar_ts: datetime | None = None
    last_gate_status: str | None = None
    last_gate_fingerprint: tuple | None = None
    pending_entry_direction: str | None = None
    pending_entry_signal_bar_ts: datetime | None = None
    pending_entry_due_ts: datetime | None = None
    pending_exit_reason: str | None = None
    pending_exit_signal_bar_ts: datetime | None = None
    pending_exit_due_ts: datetime | None = None
    exec_bar_ts: datetime | None = None
    exec_bar_open: float | None = None
    exec_bar_high: float | None = None
    exec_bar_low: float | None = None
    exec_tick_cursor: int = 0
    exec_tick_by_tick_cursor: int = 0
    order_trigger_intent: str | None = None
    order_trigger_reason: str | None = None
    order_trigger_mode: str | None = None
    order_trigger_direction: str | None = None
    order_trigger_signal_bar_ts: datetime | None = None
    order_trigger_ts: datetime | None = None
    order_trigger_deadline_ts: datetime | None = None


@dataclass(frozen=True)
class _BotConfigResult:
    mode: str  # "create" or "update"
    instance_id: int | None
    group: str
    symbol: str
    strategy: dict
    filters: dict | None


@dataclass(frozen=True)
class _BotConfigField:
    label: str
    kind: str  # "int" | "float" | "bool" | "enum" | "text"
    path: str
    options: tuple[str, ...] = ()


@dataclass
class _BotLegOrder:
    contract: Contract
    action: str  # BUY/SELL
    ratio: int


@dataclass
class _BotOrder:
    instance_id: int
    preset: _BotPreset | None
    underlying: Contract
    order_contract: Contract
    legs: list[_BotLegOrder]
    action: str  # BUY/SELL (order action for order_contract)
    quantity: int  # combo quantity (BAG) or contracts (single-leg)
    limit_price: float
    created_at: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    status: str = "STAGED"
    order_id: int | None = None
    trade: Trade | None = None
    error: str | None = None
    intent: str | None = None
    direction: str | None = None
    reason: str | None = None
    signal_bar_ts: datetime | None = None
    journal: dict | None = None
    sent_at: float | None = None  # asyncio loop time (monotonic)
    exec_mode: str | None = None  # OPTIMISTIC/MID/AGGRESSIVE/CROSS


@dataclass(frozen=True)
class _SignalSnapshot:
    bar_ts: datetime
    close: float
    signal: EmaDecisionSnapshot
    bars_in_day: int
    rv: float | None
    volume: float | None = None
    volume_ema: float | None = None
    volume_ema_ready: bool = True
    shock: bool | None = None
    shock_dir: str | None = None
    shock_atr_pct: float | None = None
    risk: RiskOverlaySnapshot | None = None
    atr: float | None = None
    or_high: float | None = None
    or_low: float | None = None
    or_ready: bool = False
    bar_health: dict | None = None
