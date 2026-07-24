"""Record option chain quote snapshots to JSONL.

Primary use cases:
- build a forward dataset of real option bid/ask/last (OPRA, CME, etc.)
- compare synthetic vs real premiums for calibration/validation

This is intentionally small and append-only. It does not try to backfill
entire chains; it records a small strike set around spot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ib_insync import IB, FuturesOption, Option

from ...config import load_config
from ..calibration import _nearest_strike, _pick_expiry
from ..quotes import QuoteError, append_snapshot, make_snapshot, resolve_option_chain


def main() -> None:
    parser = argparse.ArgumentParser(description="Record option quote snapshots (JSONL)")
    parser.add_argument("--symbol", required=True, help="Underlying symbol (e.g. SLV, MNQ)")
    parser.add_argument(
        "--exchange",
        default=None,
        help="Exchange override. Defaults to the canonical future/index exchange or SMART for stocks.",
    )
    parser.add_argument(
        "--md-type",
        type=int,
        default=1,
        help="IB market data type (1=live, 3=delayed, 4=delayed-frozen).",
    )
    parser.add_argument(
        "--dte",
        type=int,
        default=30,
        help="Target DTE for expiry selection.",
    )
    parser.add_argument(
        "--moneyness",
        default="1,2.5,5",
        help="Comma-separated percent offsets from spot (ATM is always included).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between snapshots.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of snapshots to record.",
    )
    parser.add_argument(
        "--out-dir",
        default="db/quotes",
        help="Output directory (default: db/quotes).",
    )
    args = parser.parse_args()

    cfg = load_config()
    ib = IB()
    errors: list[QuoteError] = []

    def on_error(req_id, code, message, contract) -> None:
        try:
            con_id = int(getattr(contract, "conId", 0) or 0) if contract else None
        except (TypeError, ValueError):
            con_id = None
        errors.append(
            QuoteError(
                req_id=int(req_id) if req_id is not None else None,
                code=int(code),
                message=str(message),
                con_id=con_id,
                symbol=getattr(contract, "symbol", None) if contract else None,
                local_symbol=getattr(contract, "localSymbol", None) if contract else None,
                sec_type=getattr(contract, "secType", None) if contract else None,
                exchange=getattr(contract, "exchange", None) if contract else None,
            )
        )

    ib.errorEvent += on_error
    ib.connect(cfg.host, cfg.port, clientId=cfg.client_id + 90, timeout=10)

    try:
        symbol = args.symbol
        out_dir = Path(args.out_dir) / symbol
        out_path = out_dir / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"

        moneyness = []
        for part in str(args.moneyness).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                moneyness.append(float(part))
            except ValueError:
                continue
        if not moneyness:
            moneyness = [1.0, 2.5, 5.0]

        for _ in range(max(1, int(args.count))):
            errors.clear()

            ib.reqMarketDataType(int(args.md_type))
            under_contract, spot, chain, is_future = resolve_option_chain(
                ib,
                symbol,
                args.exchange,
            )
            if spot is None or not chain:
                snap = make_snapshot(
                    symbol=symbol,
                    md_type=int(args.md_type),
                    underlying_contract=under_contract,
                    underlying_ticker=type("T", (), {"bid": None, "ask": None, "last": None, "close": None})(),
                    option_contracts=[],
                    option_tickers=[],
                    errors=list(errors),
                )
                append_snapshot(out_path, snap)
                ib.sleep(args.interval)
                continue

            expiry = _pick_expiry(chain.expirations, 0, 3650, int(args.dte))
            if not expiry:
                ib.sleep(args.interval)
                continue

            strikes = sorted(chain.strikes)
            selections = set()
            selections.add(_nearest_strike(strikes, spot))
            for pct in moneyness:
                selections.add(_nearest_strike(strikes, spot * (1 - pct / 100.0)))
                selections.add(_nearest_strike(strikes, spot * (1 + pct / 100.0)))

            contracts = []
            for strike in sorted(selections):
                for right in ("P", "C"):
                    if is_future:
                        contracts.append(
                            FuturesOption(
                                symbol,
                                expiry,
                                float(strike),
                                right,
                                exchange=chain.exchange,
                                currency="USD",
                                tradingClass=chain.tradingClass,
                            )
                        )
                    else:
                        contracts.append(
                            Option(
                                symbol,
                                expiry,
                                float(strike),
                                right,
                                exchange=chain.exchange,
                                currency="USD",
                                tradingClass=chain.tradingClass,
                            )
                        )

            contracts = list(ib.qualifyContracts(*contracts) or [])
            tickers = ib.reqTickers(*contracts) if contracts else []

            # Re-snapshot the exact qualified underlyer beside the option quotes.
            [under_ticker] = ib.reqTickers(under_contract)

            snap = make_snapshot(
                symbol=symbol,
                md_type=int(args.md_type),
                underlying_contract=under_contract,
                underlying_ticker=under_ticker,
                option_contracts=contracts,
                option_tickers=tickers,
                errors=list(errors),
                ts=datetime.now(timezone.utc),
            )
            append_snapshot(out_path, snap)
            ib.sleep(args.interval)
    finally:
        try:
            ib.disconnect()
        except OSError:
            pass


if __name__ == "__main__":
    main()
