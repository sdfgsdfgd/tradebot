from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import tradebot.option_package as option_package
import tradebot.backtest.config_values as config_values
import tradebot.backtest.spot_codec as spot_codec
import tradebot.knobs.models as knob_models
import tradebot.live.options as live_options
import tradebot.ui.bot_order_builder as bot_order_builder


def _valid_leg(**overrides):
    raw = {
        "action": "SELL",
        "right": "PUT",
        "moneyness_pct": 1.25,
        "qty": 2,
    }
    raw.update(overrides)
    return raw


def test_option_package_owns_the_legacy_named_canonical_leg_model():
    assert hasattr(option_package, "LegConfig")
    assert hasattr(option_package, "OptionLegSpec")
    assert option_package.OptionLegSpec is option_package.LegConfig
    assert knob_models.LegConfig is option_package.LegConfig
    assert option_package.LegConfig.__name__ == "LegConfig"

    legacy = knob_models.LegConfig(
        action="SELL",
        right="PUT",
        moneyness_pct=1.0,
        qty=1,
    )
    assert legacy.delta is None
    assert legacy.otm_offset_points == 0.0


def test_valid_leg_is_normalized_once_and_delta_survives_hydration():
    raw = _valid_leg(
        action=" sell ",
        right=" put ",
        moneyness_pct="1.25",
        qty="2",
        delta="-0.30",
        otm_offset_points="1.0",
    )
    expected = option_package.normalize_option_leg(raw, path="leg")

    assert expected.action == "SELL"
    assert expected.right == "PUT"
    assert expected.moneyness_pct == pytest.approx(1.25)
    assert expected.qty == 2
    assert expected.delta == pytest.approx(-0.30)
    assert expected.otm_offset_points == pytest.approx(1.0)
    assert config_values._parse_legs([raw]) == (expected,)
    assert spot_codec.leg_from_payload(raw) == expected


def test_omitted_quantity_defaults_to_one_but_explicit_null_is_rejected():
    omitted = _valid_leg()
    omitted.pop("qty")
    assert option_package.normalize_option_leg(omitted, path="leg").qty == 1

    raw = _valid_leg(qty=None)
    with pytest.raises(ValueError, match="qty"):
        option_package.normalize_option_leg(raw, path="leg")
    with pytest.raises(ValueError, match="qty"):
        config_values._parse_legs([raw])
    with pytest.raises(ValueError, match="qty"):
        spot_codec.leg_from_payload(raw)


def test_missing_moneyness_is_rejected_by_every_ingestion_surface():
    raw = _valid_leg()
    raw.pop("moneyness_pct")

    with pytest.raises(ValueError, match="moneyness_pct"):
        option_package.normalize_option_leg(raw, path="leg")
    with pytest.raises(ValueError, match="moneyness_pct"):
        config_values._parse_legs([raw])
    with pytest.raises(ValueError, match="moneyness_pct"):
        spot_codec.leg_from_payload(raw)


@pytest.mark.parametrize(
    "value",
    [None, "bad", float("nan"), float("inf"), float("-inf")],
)
def test_invalid_or_nonfinite_moneyness_is_rejected_everywhere(value):
    raw = _valid_leg(moneyness_pct=value)

    with pytest.raises(ValueError, match="moneyness_pct"):
        option_package.normalize_option_leg(raw, path="leg")
    with pytest.raises(ValueError, match="moneyness_pct"):
        config_values._parse_legs([raw])
    with pytest.raises(ValueError, match="moneyness_pct"):
        spot_codec.leg_from_payload(raw)


@pytest.mark.parametrize(
    "value",
    [None, "bad", float("nan"), float("inf"), float("-inf")],
)
def test_invalid_or_nonfinite_otm_point_offset_is_rejected_everywhere(value):
    raw = _valid_leg(otm_offset_points=value)

    with pytest.raises(ValueError, match="otm_offset_points"):
        option_package.normalize_option_leg(raw, path="leg")
    with pytest.raises(ValueError, match="otm_offset_points"):
        config_values._parse_legs([raw])
    with pytest.raises(ValueError, match="otm_offset_points"):
        spot_codec.leg_from_payload(raw)


def test_point_offset_preserves_percentage_anchor_and_exact_wing_width():
    intent = option_package.option_package_entry_intent(
        {
            "legs": [
                _valid_leg(moneyness_pct=1.0, otm_offset_points=0.0),
                _valid_leg(
                    action="BUY",
                    moneyness_pct=1.0,
                    otm_offset_points=1.0,
                ),
            ]
        }
    )

    put_strikes = [
        leg.strike
        for leg in intent.resolved_legs(spot=740.0, expiry="20260727")
    ]
    assert put_strikes == pytest.approx([732.6, 731.6])

    call_intent = option_package.option_package_entry_intent(
        {
            "legs": [
                _valid_leg(right="CALL", moneyness_pct=1.0),
                _valid_leg(
                    action="BUY",
                    right="CALL",
                    moneyness_pct=1.0,
                    otm_offset_points=1.0,
                ),
            ]
        }
    )
    call_strikes = [
        leg.strike
        for leg in call_intent.resolved_legs(spot=740.0, expiry="20260727")
    ]
    assert call_strikes == pytest.approx([747.4, 748.4])


@pytest.mark.parametrize(
    "value",
    [None, "", "bad", 0, "0", -2, float("nan"), float("inf")],
)
def test_explicit_invalid_quantity_is_rejected_everywhere(value):
    raw = _valid_leg(qty=value)

    with pytest.raises(ValueError, match="qty"):
        option_package.normalize_option_leg(raw, path="leg")
    with pytest.raises(ValueError, match="qty"):
        config_values._parse_legs([raw])
    with pytest.raises(ValueError, match="qty"):
        spot_codec.leg_from_payload(raw)


@pytest.mark.parametrize(
    "value",
    ["bad", 0, 1.1, -1.1, float("nan"), float("inf")],
)
def test_invalid_or_nonfinite_delta_is_rejected_everywhere(value):
    raw = _valid_leg(delta=value)

    with pytest.raises(ValueError, match="delta"):
        option_package.normalize_option_leg(raw, path="leg")
    with pytest.raises(ValueError, match="delta"):
        config_values._parse_legs([raw])
    with pytest.raises(ValueError, match="delta"):
        spot_codec.leg_from_payload(raw)


def test_leg_collection_preserves_order_and_fails_atomically():
    first = _valid_leg(action="SELL", right="PUT", moneyness_pct=1.0)
    second = _valid_leg(action="BUY", right="PUT", moneyness_pct=2.0)
    normalized = option_package.normalize_option_legs(
        [first, second],
        path="legs",
    )
    assert [leg.action for leg in normalized] == ["SELL", "BUY"]

    malformed = dict(second)
    malformed.pop("moneyness_pct")
    with pytest.raises(ValueError, match=r"legs\[2\].*moneyness_pct"):
        option_package.normalize_option_legs(
            [first, malformed],
            path="legs",
        )


def test_directional_collections_reject_nonlist_branches_instead_of_skipping():
    malformed = {"up": {"action": "SELL", "right": "PUT"}}

    with pytest.raises(ValueError, match=r"directional_legs.*up.*list"):
        config_values._parse_directional_legs(malformed)
    with pytest.raises(ValueError, match=r"directional_legs.*up.*list"):
        spot_codec.strategy_from_payload(
            {"instrument": "options", "directional_legs": malformed},
            filters=None,
        )


def test_ingestion_consumers_reference_the_shared_normalizers_without_ui_facades():
    assert config_values.normalize_option_legs is option_package.normalize_option_legs
    assert spot_codec.normalize_option_leg is option_package.normalize_option_leg
    assert spot_codec.normalize_option_legs is option_package.normalize_option_legs
    assert not hasattr(bot_order_builder, "normalize_option_legs")


def test_nonfinite_delta_is_rejected_before_any_provider_interaction():
    class _NoProvider:
        async def qualify_proxy_contracts(self, *contracts):
            raise AssertionError("provider interaction occurred")

    result = asyncio.run(
        live_options._strike_by_delta(
            _NoProvider(),
            symbol="XSP",
            expiry="20260724",
            right="P",
            strikes=[100.0],
            trading_class="XSP",
            target_strike=100.0,
            target_delta=float("nan"),
            owner="bot",
        )
    )
    assert result is None
