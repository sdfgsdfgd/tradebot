from __future__ import annotations

from types import SimpleNamespace

from tradebot.ui.bot import BotScreen


def _portfolio_item_stub() -> SimpleNamespace:
    return SimpleNamespace(
        contract=SimpleNamespace(conId=123, secType="STK", symbol="SLV", localSymbol="SLV"),
        position=3.0,
        averageCost=79.44,
        marketPrice=79.21,
        marketValue=237.63,
        unrealizedPNL=-0.69,
        realizedPNL=0.0,
    )


class _OrdersTableStub:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def clear(self) -> None:
        self.rows = []

    def add_row(self, *values) -> None:
        self.rows.append(values)


class _OrdersRefreshProbe:
    _scope_all = False
    _item_con_id = staticmethod(BotScreen._item_con_id)
    _edge_glyph_style = staticmethod(BotScreen._edge_glyph_style)
    _live_unrealized_value = BotScreen._live_unrealized_value
    _official_unrealized_value = BotScreen._official_unrealized_value
    _position_unrealized_values = BotScreen._position_unrealized_values
    _position_entry_now_cell = BotScreen._position_entry_now_cell
    _position_px_change_cell = BotScreen._position_px_change_cell
    _position_md_badge_cell = BotScreen._position_md_badge_cell

    def __init__(self, item: SimpleNamespace) -> None:
        self._orders_table = _OrdersTableStub()
        self._orders: list = []
        self._order_rows: list = []
        self._positions = [item]
        self._instances = [SimpleNamespace(instance_id=1, touched_conids={123})]
        ticker = SimpleNamespace(
            bid=82.37,
            ask=82.39,
            last=82.38,
            close=79.06,
            marketDataType=1,
        )
        self._client = SimpleNamespace(
            pnl_single_unrealized=lambda con_id: 7.02 if int(con_id) == 123 else None,
            ticker_for_con_id=lambda con_id: ticker if int(con_id) == 123 else None,
        )

    @staticmethod
    def _scope_instance_id() -> int:
        return 1

    @staticmethod
    def _sync_row_marker(*_args, **_kwargs) -> None:
        return

    @staticmethod
    def _position_mark_price(_item) -> float:
        # Simulate fresh live quote while portfolio snapshot is stale.
        return 81.78


def _position_row_after_refresh() -> tuple:
    item = _portfolio_item_stub()
    probe = _OrdersRefreshProbe(item)
    BotScreen._refresh_orders_table(probe)
    assert len(probe._orders_table.rows) == 3
    return probe._orders_table.rows[2]


def test_bot_orders_positions_row_prefers_live_official_unrealized() -> None:
    row = _position_row_after_refresh()
    # stale snapshot unrealized is -0.69, live broker unrealized is +7.02.
    assert row[8].plain == "7.02"


def test_bot_orders_positions_row_uses_live_mark_for_now_column() -> None:
    row = _position_row_after_refresh()
    # stale snapshot marketPrice is 79.21, live mark is 81.78.
    assert "81.78" in row[5].plain
