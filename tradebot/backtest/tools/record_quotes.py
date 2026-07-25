"""Record option chain quote snapshots to JSONL.

Primary use cases:
- build a forward dataset of real option bid/ask/last (OPRA, CME, etc.)
- compare synthetic vs real premiums for calibration/validation

This is intentionally small and append-only. It does not try to backfill
entire chains; it records a small strike set around spot.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ib_insync import IB, FuturesOption, Option

from ...config import auxiliary_client_id, load_config
from ...engines.market import xsp_capture_window_date, xsp_trading_date
from ..calibration import _nearest_strike, _pick_expiry
from ..quotes import (
    QuoteError,
    append_snapshot,
    iter_snapshots,
    make_chain_manifest,
    make_snapshot,
    persist_chain_manifest,
    persist_quote_tape_receipt,
    repair_snapshot_tail,
    resolve_option_chain,
    snapshot_quality,
)

_REQUEST_BATCH_SIZE = 50


@dataclass
class CaptureCadence:
    """Absolute cadence: capture cost and restarts never accumulate drift."""

    interval_sec: float
    due_mono: float

    @classmethod
    def resume(
        cls,
        interval_sec: float,
        *,
        now_mono: float,
        now_utc: datetime,
        last_captured_at: datetime | None,
    ) -> "CaptureCadence":
        interval = max(0.0, float(interval_sec))
        delay = 0.0
        if interval and last_captured_at is not None:
            if last_captured_at.tzinfo is None:
                last_captured_at = last_captured_at.replace(tzinfo=timezone.utc)
            delay = max(
                0.0,
                (
                    last_captured_at.astimezone(timezone.utc)
                    + timedelta(seconds=interval)
                    - now_utc.astimezone(timezone.utc)
                ).total_seconds(),
            )
        return cls(interval_sec=interval, due_mono=now_mono + delay)

    def wait(self, ib: IB) -> None:
        delay = self.due_mono - time.monotonic()
        if delay > 0.0:
            ib.sleep(delay)

    def advance(self, finished_mono: float) -> None:
        if self.interval_sec <= 0.0:
            self.due_mono = finished_mono
            return
        periods = max(
            1,
            math.floor((finished_mono - self.due_mono) / self.interval_sec) + 1,
        )
        self.due_mono += periods * self.interval_sec


@dataclass
class RetainedOptionUniverse:
    """Keep every qualified session leg addressable as spot moves."""

    key: tuple[str, str] | None = None
    _contracts: dict[int, object] = field(default_factory=dict)
    _identities: set[tuple[object, ...]] = field(default_factory=set)

    @staticmethod
    def identity(contract: object) -> tuple[object, ...]:
        return (
            str(getattr(contract, "symbol", "") or "").strip().upper(),
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip(),
            float(getattr(contract, "strike", 0.0) or 0.0),
            str(getattr(contract, "right", "") or "").strip().upper()[:1],
            str(getattr(contract, "exchange", "") or "").strip().upper(),
            str(getattr(contract, "tradingClass", "") or "").strip().upper(),
        )

    def begin(self, key: tuple[str, str]) -> None:
        if key != self.key:
            self.key = key
            self._contracts.clear()
            self._identities.clear()

    def unseen(self, contracts: list[object]) -> list[object]:
        return [
            contract
            for contract in contracts
            if self.identity(contract) not in self._identities
        ]

    def retain(self, contracts: list[object]) -> None:
        for contract in contracts:
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id <= 0:
                continue
            self._contracts[con_id] = contract
            self._identities.add(self.identity(contract))

    def restore(self, options: list[object]) -> None:
        """Rehydrate the union from durable rows after a process restart."""

        contracts = []
        for option in options:
            con_id = int(getattr(option, "con_id", 0) or 0)
            expiry = str(getattr(option, "expiry", "") or "")
            if con_id <= 0 or (self.key and expiry != self.key[1]):
                continue
            option_type = (
                FuturesOption
                if str(getattr(option, "sec_type", "") or "").upper() == "FOP"
                else Option
            )
            contracts.append(
                option_type(
                    str(getattr(option, "symbol", "") or ""),
                    expiry,
                    float(getattr(option, "strike", 0.0) or 0.0),
                    str(getattr(option, "right", "") or ""),
                    exchange=str(getattr(option, "exchange", "") or ""),
                    multiplier=str(getattr(option, "multiplier", "") or ""),
                    currency=str(getattr(option, "currency", "") or ""),
                    tradingClass=str(getattr(option, "trading_class", "") or ""),
                    conId=con_id,
                    localSymbol=str(getattr(option, "local_symbol", "") or ""),
                )
            )
        self.retain(contracts)

    @property
    def contracts(self) -> list[object]:
        return sorted(
            self._contracts.values(),
            key=lambda contract: (*self.identity(contract), int(contract.conId)),
        )


def _persist_receipt(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    receipt_path = persist_quote_tape_receipt(path)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "tape": str(path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record option quote snapshots (JSONL)"
    )
    parser.add_argument(
        "--symbol", required=True, help="Underlying symbol (e.g. SLV, MNQ)"
    )
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
        help="Number of snapshots to record; 0 runs until interrupted.",
    )
    parser.add_argument(
        "--out-dir",
        default="db/quotes",
        help="Output directory (default: db/quotes).",
    )
    args = parser.parse_args()

    cfg = load_config()
    ib = IB()
    ib.RequestTimeout = 45.0
    errors: list[QuoteError] = []
    signal.signal(signal.SIGTERM, signal.default_int_handler)

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
                local_symbol=getattr(contract, "localSymbol", None)
                if contract
                else None,
                sec_type=getattr(contract, "secType", None) if contract else None,
                exchange=getattr(contract, "exchange", None) if contract else None,
            )
        )

    ib.errorEvent += on_error

    last_out_path: Path | None = None
    try:
        symbol = str(args.symbol).strip().upper()
        out_dir = Path(args.out_dir) / symbol

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

        snapshot_count = int(args.count)
        now_utc = datetime.now(timezone.utc)
        capture_window_day = (
            xsp_capture_window_date(now_utc)
            if symbol == "XSP" and not snapshot_count
            else None
        )
        if symbol == "XSP" and not snapshot_count and capture_window_day is None:
            print(
                json.dumps(
                    {
                        "broker_request_skipped": "closed_capture_window",
                        "status": "closed",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        starting_day = (
            capture_window_day or xsp_trading_date(now_utc)
            if symbol == "XSP"
            else now_utc.date()
        )
        today_path = (
            out_dir / f"{starting_day.isoformat()}.jsonl"
            if starting_day is not None
            else None
        )
        last_captured_at = None
        if today_path is not None and today_path.exists():
            repair_snapshot_tail(today_path)
            for prior in iter_snapshots(today_path):
                last_captured_at = datetime.fromisoformat(prior.ts)
            last_out_path = today_path
        cadence = CaptureCadence.resume(
            float(args.interval),
            now_mono=time.monotonic(),
            now_utc=datetime.now(timezone.utc),
            last_captured_at=last_captured_at if not snapshot_count else None,
        )
        universe = RetainedOptionUniverse()
        active_day = None
        under_contract = chain = None
        expiry = chain_fingerprint = None
        is_future = False
        completed = 0

        while not snapshot_count or completed < max(1, snapshot_count):
            cadence.wait(ib)
            loop_now = datetime.now(timezone.utc)
            if (
                capture_window_day is not None
                and xsp_capture_window_date(loop_now) != capture_window_day
            ):
                break
            session_day = (
                xsp_trading_date(loop_now) if symbol == "XSP" else loop_now.date()
            )
            if session_day is None:
                if snapshot_count:
                    raise RuntimeError("XSP is outside its normal weekly session")
                cadence.advance(time.monotonic())
                continue
            capture_day = session_day.isoformat()
            attempts = 0
            while True:
                errors.clear()
                try:
                    if not ib.isConnected():
                        ib.connect(
                            cfg.host,
                            cfg.port,
                            clientId=auxiliary_client_id(cfg, 90),
                            timeout=10,
                            readonly=bool(getattr(cfg, "readonly", False)),
                        )
                    if not ib.isConnected():
                        raise ConnectionError("IB connection did not become ready")

                    ib.reqMarketDataType(int(args.md_type))
                    if (
                        capture_day != active_day
                        or under_contract is None
                        or chain is None
                        or expiry is None
                    ):
                        under_contract, spot, chain, is_future = resolve_option_chain(
                            ib,
                            symbol,
                            args.exchange,
                        )
                        expiry = (
                            _pick_expiry(
                                chain.expirations,
                                0,
                                3650,
                                int(args.dte),
                                as_of=session_day,
                            )
                            if chain
                            else None
                        )
                        chain_fingerprint = (
                            persist_chain_manifest(
                                out_dir,
                                make_chain_manifest(under_contract, chain),
                            )
                            if chain
                            else None
                        )
                        active_day = capture_day
                        universe.begin((active_day, str(expiry or "")))
                        session_path = out_dir / f"{active_day}.jsonl"
                        if session_path.exists():
                            repair_snapshot_tail(session_path)
                            prior_options = [
                                option
                                for prior in iter_snapshots(session_path)
                                for option in prior.options
                            ]
                            universe.restore(prior_options)
                            last_out_path = session_path
                        under_ticker = None
                        if chain is None or expiry is None:
                            raise ConnectionError(
                                "IBKR returned no usable option chain or expiry"
                            )
                    else:
                        [under_ticker] = ib.reqTickers(under_contract)
                        spot = None
                        for value in (
                            under_ticker.marketPrice(),
                            under_ticker.last,
                            under_ticker.close,
                        ):
                            try:
                                candidate = float(value)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(candidate) and candidate > 0.0:
                                spot = candidate
                                break

                    if (
                        spot is not None
                        and chain
                        and expiry
                        and getattr(chain, "strikes", None)
                    ):
                        strikes = sorted(chain.strikes)
                        targets = {float(spot)}
                        for pct in moneyness:
                            targets.add(spot * (1 - pct / 100.0))
                            targets.add(spot * (1 + pct / 100.0))
                        selected_indices: set[int] = set()
                        for target in targets:
                            nearest_index = strikes.index(
                                _nearest_strike(strikes, target)
                            )
                            selected_indices.update(
                                range(
                                    max(0, nearest_index - 1),
                                    min(len(strikes), nearest_index + 2),
                                )
                            )
                        option_type = FuturesOption if is_future else Option
                        candidates = [
                            option_type(
                                symbol,
                                expiry,
                                float(strikes[index]),
                                right,
                                exchange=chain.exchange,
                                currency="USD",
                                tradingClass=chain.tradingClass,
                            )
                            for index in sorted(selected_indices)
                            for right in ("P", "C")
                        ]
                        missing = universe.unseen(candidates)
                        if missing:
                            qualified = []
                            for start in range(0, len(missing), _REQUEST_BATCH_SIZE):
                                qualified.extend(
                                    ib.qualifyContracts(
                                        *missing[start : start + _REQUEST_BATCH_SIZE]
                                    )
                                    or []
                                )
                            universe.retain(qualified)

                    contracts = universe.contracts
                    if not contracts:
                        raise ConnectionError(
                            "IBKR returned no qualified option contracts"
                        )
                    tickers = []
                    for start in range(0, len(contracts), _REQUEST_BATCH_SIZE):
                        tickers.extend(
                            ib.reqTickers(
                                *contracts[start : start + _REQUEST_BATCH_SIZE]
                            )
                        )
                    if under_ticker is None:
                        [under_ticker] = ib.reqTickers(under_contract)
                    captured_at = datetime.now(timezone.utc)
                    captured_day = (
                        xsp_trading_date(captured_at)
                        if symbol == "XSP"
                        else captured_at.date()
                    )
                    if captured_day is None or captured_day.isoformat() != active_day:
                        active_day = None
                        continue
                    snap = make_snapshot(
                        symbol=symbol,
                        md_type=int(args.md_type),
                        underlying_contract=under_contract,
                        underlying_ticker=under_ticker,
                        option_contracts=contracts,
                        option_tickers=tickers,
                        errors=list(errors),
                        ts=captured_at,
                        chain_fingerprint=chain_fingerprint,
                        target_expiry=expiry,
                    )
                    out_path = out_dir / f"{active_day}.jsonl"
                    if last_out_path is not None and out_path != last_out_path:
                        _persist_receipt(last_out_path)
                    append_snapshot(out_path, snap)
                    last_out_path = out_path
                    output = json.dumps(
                        {
                            "path": str(out_path),
                            "ts": snap.ts,
                            "chain_fingerprint": chain_fingerprint,
                            "target_expiry": expiry,
                            "retained_options": len(contracts),
                            **snapshot_quality(
                                snap,
                                max_age_sec=30.0,
                                require_live=int(args.md_type) == 1,
                                require_provenance=True,
                                require_all_options=True,
                            ),
                        },
                        sort_keys=True,
                    )
                    break
                except (ConnectionError, OSError, TimeoutError) as exc:
                    attempts += 1
                    try:
                        ib.disconnect()
                    except OSError:
                        pass
                    if snapshot_count and attempts >= 5:
                        raise
                    delay = min(60.0, float(2 ** min(attempts - 1, 6)))
                    print(
                        json.dumps(
                            {
                                "capture_retry": attempts,
                                "delay_sec": delay,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    time.sleep(delay)

            print(output, flush=True)
            completed += 1
            cadence.advance(time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ib.disconnect()
        except OSError:
            pass
        _persist_receipt(last_out_path)


if __name__ == "__main__":
    main()
