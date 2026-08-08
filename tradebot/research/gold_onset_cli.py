"""Read-only broker/cache adapter for the prospective 1OZ onset tape."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ib_insync import Contract, Future, IB, Stock

from ..chart_data.history import normalize_bars_to_close, read_cache
from ..chart_data.series import OhlcvBar
from ..news.contract import load_news_history
from ..live.capital import load_live_capital_plan, publish_live_capital_plan
from ..live.capital_packages import (
    load_allocated_live_selection,
    publish_immutable_live_selection,
)
from ..live.capital_stability import (
    PORTFOLIO_CAPITAL_STABILITY_DIRECTORY,
    PORTFOLIO_CAPITAL_STABILITY_PATH,
    PORTFOLIO_PACKAGE_GENERATION_DIRECTORY,
    portfolio_capital_owner_stability_gate,
    publish_portfolio_capital_owner_stability,
    publish_portfolio_package_generation,
)
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_LEDGER_PATH,
    GOLD_OPEN_POSITION_STRESS_PATH,
    GOLD_LIVE_SELECTION_PATH,
    advance_gold_regime_harmony_source,
    build_gold_portfolio_capital_plan,
    gold_selection_preview,
    load_gold_live_selection_from_mapping,
    publish_gold_live_selection,
    reallocate_gold_live_transport,
    select_gold_live_transport,
)
from .gold_onset import (
    advance_gold_onset_tape,
    build_gold_onset_context,
    gold_signal_context,
    select_gold_contract_pair,
)
from .gold_regime_harmony import GoldRegimeHarmonyTape
from .live_calibration import LiveCalibrationLedger
from .live_portfolio_packages import build_xsp_gold_mcl_portfolio_package_plan
from .mcl_live_transport import MCL_LIVE_CAPITAL_SLEEVE
from .xsp_live_transport import XSP_V3_TRANSPORT_CAPITAL_SLEEVE


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATHS = {
    "xau_h1": (
        ROOT / "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_1hour_full24.csv",
        ROOT / "db/XAUUSD/XAUUSD_2016-07-01_2026-08-02_1hour_full24.csv",
    ),
    "xau_h4": (
        ROOT / "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_4hours_full24.csv",
        ROOT / "db/XAUUSD/XAUUSD_2016-01-01_2026-08-02_4hours_full24.csv",
    ),
    "xau_d1": (
        ROOT / "db/XAUUSD/XAUUSD_2015-07-01_2016-06-30_1day_full24.csv",
        ROOT / "db/XAUUSD/XAUUSD_2016-01-01_2026-08-02_1day_full24.csv",
    ),
    "uup_d1": (ROOT / "db/UUP/UUP_2015-07-01_2026-08-02_1day_rth.csv",),
    "tip_d1": (ROOT / "db/TIP/TIP_2015-07-01_2026-08-02_1day_rth.csv",),
}
_MONTH_CODES = set("FGHJKMNQUVXZ")
_GOLD_ROLLOVER_INTENT_SCHEMA = "gold.1oz-fail-closed-rollover-intent.v1"


def _aware(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bars_from_cache(paths: Iterable[Path]) -> list[OhlcvBar]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"gold onset cache missing: {missing}")
    return [
        bar
        for _stamp, bar in sorted(
            {bar.ts: bar for path in paths for bar in read_cache(path)}.items()
        )
    ]


def _raw_bars(raw: Iterable[object]) -> list[OhlcvBar]:
    output = []
    for source in raw:
        stamp = getattr(source, "date", None)
        if isinstance(stamp, datetime):
            ts = _aware(stamp).replace(tzinfo=None)
        elif stamp is not None:
            try:
                ts = datetime.fromisoformat(str(stamp)).replace(tzinfo=None)
            except ValueError:
                continue
        else:
            continue
        values = [_finite(getattr(source, name, None)) for name in ("open", "high", "low", "close", "volume")]
        if any(value is None for value in values):
            continue
        output.append(OhlcvBar(ts, *[float(value) for value in values]))
    return output


def _merge(*groups: Iterable[OhlcvBar]) -> list[OhlcvBar]:
    return [
        bar
        for _stamp, bar in sorted(
            {bar.ts: bar for group in groups for bar in group}.items()
        )
    ]


def _request_bars(
    ib: IB,
    contract: Contract,
    *,
    duration: str,
    bar_size: str,
    what: str,
    use_rth: bool,
    symbol: str,
) -> list[OhlcvBar]:
    raw = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what,
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
        timeout=60,
    )
    return normalize_bars_to_close(
        _raw_bars(raw),
        symbol=symbol,
        bar_size=bar_size,
        use_rth=use_rth,
    )


def _qualify_one(ib: IB, contract: Contract) -> Contract:
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"contract did not qualify: {contract}")
    return qualified[0]


def _expiry(value: object) -> datetime | None:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) < 6:
        return None
    day = digits[6:8] if len(digits) >= 8 else "28"
    try:
        return datetime.strptime(digits[:6] + day, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _chain(ib: IB, symbol: str, *, now: datetime) -> list[Contract]:
    details = ib.reqContractDetails(Future(symbol=symbol, exchange="COMEX", currency="USD"))
    output = []
    for detail in details:
        contract = detail.contract
        expiry = _expiry(
            getattr(detail, "realExpirationDate", "")
            or getattr(contract, "lastTradeDateOrContractMonth", "")
        )
        if expiry is None or not now.date() <= expiry.date() <= (now + timedelta(days=400)).date():
            continue
        local = str(getattr(contract, "localSymbol", "") or "").upper()
        if not re.search(rf"[{''.join(sorted(_MONTH_CODES))}]\d$", local):
            continue
        output.append(contract)
    output.sort(key=lambda contract: str(getattr(contract, "lastTradeDateOrContractMonth", "") or ""))
    return output[:10]


def _chain_quotes(
    ib: IB, *, now: datetime
) -> tuple[list[dict[str, object]], dict[int, Contract]]:
    contracts = [contract for symbol in ("GC", "1OZ") for contract in _chain(ib, symbol, now=now)]
    qualified = []
    for contract in contracts:
        rows = ib.qualifyContracts(contract)
        if rows:
            qualified.append(rows[0])
    ib.reqMarketDataType(1)
    tickers = [ib.reqMktData(contract, "233", False, False) for contract in qualified]
    ib.sleep(5.0)
    quotes = []
    by_id = {}
    for contract, ticker in zip(qualified, tickers, strict=True):
        by_id[int(contract.conId)] = contract
        quotes.append(
            {
                "symbol": contract.symbol,
                "local_symbol": contract.localSymbol,
                "con_id": contract.conId,
                "expiry": contract.lastTradeDateOrContractMonth,
                "market_data_type": ticker.marketDataType,
                "bid": _finite(ticker.bid),
                "bid_size": _finite(ticker.bidSize),
                "ask": _finite(ticker.ask),
                "ask_size": _finite(ticker.askSize),
                "last": _finite(ticker.last),
                "volume": _finite(ticker.volume),
                "time": _aware(ticker.time or now).isoformat(),
            }
        )
        ib.cancelMktData(contract)
    return quotes, by_id


def _point_at(rows: Sequence[OhlcvBar], at: datetime) -> dict[str, object]:
    aware = sorted(((_aware(row.ts), row) for row in rows), key=lambda item: item[0])
    candidates = [(stamp, row) for stamp, row in aware if stamp <= at]
    if not candidates:
        return {"close": None, "bar_end_utc": None, "age_seconds": None}
    stamp, row = candidates[-1]
    return {
        "close": float(row.close),
        "bar_end_utc": stamp.isoformat(),
        "age_seconds": (_aware(at) - stamp).total_seconds(),
    }


def _news_history(path: Path) -> list[dict[str, object]]:
    return [
        row
        for source in sorted(path.glob("*.jsonl"))
        for row in load_news_history(source)
    ]


def _unmanaged_stress(preview: dict[str, object]) -> float:
    positions = preview.get("positions")
    if not isinstance(positions, list):
        raise ValueError("gold rollover preview has no broker positions")
    owned = {"UPRO", "SPXU", "1OZ", "MCL"}
    return sum(
        abs(float(row["market_value_base"]))
        for row in positions
        if isinstance(row, dict)
        and str(row.get("symbol") or "").upper() not in owned
    )


def _gold_rollover_boundary(
    *,
    capital_path: Path,
    intent_path: Path,
    root: Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Persist the predecessor before mutation or recover its active successor."""

    plan = load_live_capital_plan(capital_path)
    gold, gold_path, gold_sha = load_allocated_live_selection(
        plan,
        sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
        repository_root=root,
    )
    if intent_path.is_symlink():
        raise ValueError("gold rollover intent must not be a symlink")
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
    else:
        body = {
            "schema": _GOLD_ROLLOVER_INTENT_SCHEMA,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "predecessor_plan_id": plan["plan_id"],
            "predecessor_selection_id": gold["selection_id"],
        }
        intent = {
            **body,
            "intent_id": hashlib.sha256(
                json.dumps(
                    body,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
        payload = json.dumps(
            intent, allow_nan=False, indent=2, sort_keys=True
        ).encode() + b"\n"
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=intent_path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, intent_path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if not isinstance(intent, Mapping):
        raise ValueError("gold rollover intent must be an object")
    frozen = dict(intent)
    intent_id = str(frozen.pop("intent_id", ""))
    expected = str(frozen.get("predecessor_selection_id") or "")
    if (
        frozen.get("schema") != _GOLD_ROLLOVER_INTENT_SCHEMA
        or set(frozen)
        != {
            "schema",
            "registered_at_utc",
            "predecessor_plan_id",
            "predecessor_selection_id",
        }
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(frozen.get("predecessor_plan_id") or "")
        )
        or intent_id
        != hashlib.sha256(
            json.dumps(
                frozen,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("gold rollover intent identity is invalid")
    if gold["selection_id"] == expected:
        return dict(intent), None

    successor = gold.get("allocation_successor")
    if (
        not isinstance(successor, Mapping)
        or successor.get("predecessor_selection_id") != expected
    ):
        raise ValueError("active Gold selection crossed the rollover boundary")
    load_gold_live_selection_from_mapping(gold)

    root = root.resolve()
    generation_path = (
        PORTFOLIO_PACKAGE_GENERATION_DIRECTORY / f"{plan['plan_id']}.json"
    )
    generation_sha = hashlib.sha256(
        (root / generation_path).read_bytes()
    ).hexdigest()
    current_stability = root / PORTFOLIO_CAPITAL_STABILITY_PATH
    stability_payload = current_stability.read_bytes()
    stability_sha = hashlib.sha256(stability_payload).hexdigest()
    stability_path = (
        PORTFOLIO_CAPITAL_STABILITY_DIRECTORY / f"{stability_sha}.json"
    )
    if (root / stability_path).read_bytes() != stability_payload:
        raise ValueError("Gold rollover capital stability archive changed")
    for sleeve in plan["sleeves"]:
        decision = portfolio_capital_owner_stability_gate(
            current_stability,
            repo_root=root,
            sleeve_id=str(sleeve["sleeve_id"]),
            selection_id=str(sleeve["run_id"]),
            selection_file_sha256=str(sleeve["selection_file_sha256"]),
        )
        if decision["status"] != "PASS":
            raise ValueError("Gold rollover capital stability proof is invalid")
    xsp, _xsp_path, _xsp_sha = load_allocated_live_selection(
        plan,
        sleeve_id=XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
        repository_root=root,
    )
    mcl, _mcl_path, _mcl_sha = load_allocated_live_selection(
        plan,
        sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
        repository_root=root,
    )
    return dict(intent), {
        "rollover": {
            "predecessor_selection_id": expected,
            "selection_id": gold["selection_id"],
            "selection_path": gold_path.relative_to(root).as_posix(),
            "selection_file_sha256": gold_sha,
            "capital_plan_id": plan["plan_id"],
            "portfolio_generation_path": generation_path.as_posix(),
            "portfolio_generation_sha256": generation_sha,
            "capital_stability_path": stability_path.as_posix(),
            "capital_stability_sha256": stability_sha,
            "retained_xsp_selection_id": xsp["selection_id"],
            "retained_mcl_selection_id": mcl["selection_id"],
            "rollover_intent_id": intent_id,
            "recovered_after_interruption": True,
            "submitted_orders": 0,
            "verdict": "FRESH_FAIL_CLOSED_GOLD_RUN_SELECTED_FLAT",
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Advance the causal gold onset and Stage-76 source owners."
    )
    parser.add_argument("--commission-canary", action="store_true")
    parser.add_argument("--rollover-canary", action="store_true")
    parser.add_argument("--rollover-intent")
    parser.add_argument("--live-ledger", default=str(GOLD_LIVE_LEDGER_PATH))
    parser.add_argument("--selection", default=str(GOLD_LIVE_SELECTION_PATH))
    parser.add_argument(
        "--capital-plan", default="db/calibration/live_capital_plan.json"
    )
    args = parser.parse_args(argv)
    if args.commission_canary and args.rollover_canary:
        raise ValueError("gold canary commission and rollover are exclusive")
    rollover_intent: dict[str, object] | None = None
    if args.rollover_canary:
        if not args.rollover_intent:
            raise ValueError("gold rollover requires an immutable intent path")
        rollover_intent, recovered = _gold_rollover_boundary(
            capital_path=Path(args.capital_plan).expanduser(),
            intent_path=Path(args.rollover_intent).expanduser(),
            root=ROOT,
        )
        if recovered is not None:
            print(json.dumps(recovered, indent=2, sort_keys=True, allow_nan=False))
            return
    request_started_at = datetime.now(timezone.utc)
    ib = IB()
    ib.connect(
        os.environ.get("IBKR_HOST", "127.0.0.1"),
        int(os.environ.get("IBKR_PORT", "4001")),
        clientId=int(os.environ.get("IBKR_CLIENT_ID", "3199")),
        readonly=True,
        timeout=12,
    )
    try:
        xau = _qualify_one(
            ib,
            Contract(secType="CMDTY", symbol="XAUUSD", exchange="SMART", currency="USD"),
        )
        uup = _qualify_one(ib, Stock("UUP", "SMART", "USD"))
        tip = _qualify_one(ib, Stock("TIP", "SMART", "USD"))
        xau_h1 = _merge(
            _bars_from_cache(CACHE_PATHS["xau_h1"]),
            _request_bars(ib, xau, duration="10 D", bar_size="1 hour", what="MIDPOINT", use_rth=False, symbol="XAUUSD"),
        )
        xau_h4 = _merge(
            _bars_from_cache(CACHE_PATHS["xau_h4"]),
            _request_bars(ib, xau, duration="1 M", bar_size="4 hours", what="MIDPOINT", use_rth=False, symbol="XAUUSD"),
        )
        xau_d1 = _merge(
            _bars_from_cache(CACHE_PATHS["xau_d1"]),
            _request_bars(ib, xau, duration="1 Y", bar_size="1 day", what="MIDPOINT", use_rth=False, symbol="XAUUSD"),
        )
        uup_d1 = _merge(
            _bars_from_cache(CACHE_PATHS["uup_d1"]),
            _request_bars(ib, uup, duration="1 M", bar_size="1 day", what="TRADES", use_rth=True, symbol="UUP"),
        )
        tip_d1 = _merge(
            _bars_from_cache(CACHE_PATHS["tip_d1"]),
            _request_bars(ib, tip, duration="1 M", bar_size="1 day", what="TRADES", use_rth=True, symbol="TIP"),
        )
        quotes, contracts = _chain_quotes(ib, now=request_started_at)
        quote_as_of = datetime.now(timezone.utc)
        pair = select_gold_contract_pair(quotes, observed_at=quote_as_of)
        if not pair.get("usable"):
            raise RuntimeError(
                "gold shared-month book unavailable: "
                + json.dumps(pair, sort_keys=True, allow_nan=False)
            )
        gc_contract = contracts[int(dict(pair["gc"])["con_id"])]
        one_contract = contracts[int(dict(pair["one_oz"])["con_id"])]
        gc_h1 = _request_bars(ib, gc_contract, duration="10 D", bar_size="1 hour", what="TRADES", use_rth=False, symbol="GC")
        one_m30 = _request_bars(ib, one_contract, duration="10 D", bar_size="30 mins", what="TRADES", use_rth=False, symbol="1OZ")
    finally:
        ib.disconnect()

    observed_at = datetime.now(timezone.utc)
    signal = gold_signal_context(
        xau_h4, xau_d1, uup_d1, tip_d1, as_of=observed_at
    )
    decision_at = (
        datetime.fromisoformat(str(signal["decision_bar_end_utc"]).replace("Z", "+00:00"))
        if signal.get("usable")
        else observed_at
    )
    source_points = {
        "XAUUSD": _point_at(xau_h1, decision_at),
        "GC": _point_at(gc_h1, decision_at),
        "1OZ": _point_at(one_m30, decision_at),
    }
    news_dir = Path(
        os.environ.get(
            "TRADEBOT_NEWS_HISTORY",
            str(Path.home() / ".local/state/tradebot/news/history"),
        )
    ).expanduser()
    context = build_gold_onset_context(
        xau_h4=xau_h4,
        xau_daily=xau_d1,
        uup_daily=uup_d1,
        tip_daily=tip_d1,
        quotes=quotes,
        news_history=_news_history(news_dir),
        source_points=source_points,
        observed_at=observed_at,
    )
    ledger = LiveCalibrationLedger(
        Path(
            os.environ.get(
                "GOLD_ONSET_LEDGER",
                str(Path.home() / ".local/state/tradebot/research/gold_onset.jsonl"),
            )
        ).expanduser()
    )
    output = advance_gold_onset_tape(
        ledger,
        context=context,
        outcome_bars={"XAUUSD": xau_h1, "GC": gc_h1, "1OZ": one_m30},
        observed_at=observed_at,
    )
    live_ledger = LiveCalibrationLedger(Path(args.live_ledger).expanduser())
    source = advance_gold_regime_harmony_source(
        live_ledger,
        tape=GoldRegimeHarmonyTape(
            h1=tuple(xau_h1),
            h4=tuple(xau_h4),
            daily=tuple(xau_d1),
            uup=tuple(uup_d1),
            tip=tuple(tip_d1),
        ),
        onset_context=context,
        observed_at=observed_at,
    )
    output["stage76_source"] = source
    if args.commission_canary or args.rollover_canary:
        selection_path = Path(args.selection).expanduser()
        capital_path = Path(args.capital_plan).expanduser()
        if args.commission_canary and selection_path.exists():
            raise ValueError("gold canary selection already exists")
        preview_ib = IB()
        preview_ib.connect(
            os.environ.get("IBKR_HOST", "127.0.0.1"),
            int(os.environ.get("IBKR_PORT", "4001")),
            clientId=int(os.environ.get("IBKR_CLIENT_ID", "3199")) + 2,
            readonly=True,
            timeout=12,
        )
        try:
            preview_quotes, preview_contracts = _chain_quotes(
                preview_ib, now=datetime.now(timezone.utc)
            )
            preview_pair = select_gold_contract_pair(
                preview_quotes,
                observed_at=datetime.now(timezone.utc),
            )
            preview = gold_selection_preview(
                preview_ib,
                pair=preview_pair,
                contracts=preview_contracts,
                observed_at=datetime.now(timezone.utc),
            )
        finally:
            preview_ib.disconnect()
        selected_at = datetime.now(timezone.utc)
        if args.commission_canary:
            selection = select_gold_live_transport(
                source_checkpoint=source["checkpoint"],
                preview=preview,
                selected_at_utc=selected_at,
                root=ROOT,
            )
            publish_gold_live_selection(selection_path, selection)
            capital = build_gold_portfolio_capital_plan(
                selection,
                selection_path=selection_path,
                current_plan=load_live_capital_plan(capital_path),
            )
            publish_live_capital_plan(capital_path, capital)
            output["commissioning"] = {
                "selection_id": selection["selection_id"],
                "capital_plan_id": capital["plan_id"],
                "order_authority": selection["order_authority"],
                "submitted_orders": 0,
                "verdict": "CANARY_SELECTED_FLAT_AWAITING_FRESH_STAGE76_ADMISSION",
            }
        else:
            assert rollover_intent is not None
            predecessor = load_live_capital_plan(capital_path)
            gold, _gold_path, _gold_sha = load_allocated_live_selection(
                predecessor,
                sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
                repository_root=ROOT,
            )
            if gold["selection_id"] != rollover_intent["predecessor_selection_id"]:
                raise ValueError("active Gold selection crossed the rollover boundary")
            selection = reallocate_gold_live_transport(
                predecessor=gold,
                records=tuple(live_ledger.records()),
                source_checkpoint=source["checkpoint"],
                preview=preview,
                selected_at_utc=selected_at,
                stress_receipt_path=ROOT / GOLD_OPEN_POSITION_STRESS_PATH,
                root=ROOT,
            )
            gold_path, gold_sha = publish_immutable_live_selection(ROOT, selection)
            xsp, xsp_path, xsp_sha = load_allocated_live_selection(
                predecessor,
                sleeve_id=XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
                repository_root=ROOT,
            )
            mcl, mcl_path, mcl_sha = load_allocated_live_selection(
                predecessor,
                sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
                repository_root=ROOT,
            )
            account = preview["account_values"]
            assert isinstance(account, dict)
            capital = build_xsp_gold_mcl_portfolio_package_plan(
                xsp_selection=xsp,
                gold_selection=selection,
                mcl_selection=mcl,
                xsp_selection_path=xsp_path.relative_to(ROOT).as_posix(),
                xsp_selection_file_sha256=xsp_sha,
                gold_selection_path=gold_path,
                gold_selection_file_sha256=gold_sha,
                mcl_selection_path=mcl_path.relative_to(ROOT).as_posix(),
                mcl_selection_file_sha256=mcl_sha,
                account_resources={
                    "account_id": preview["account_id"],
                    "account_type": preview["account_type"],
                    "base_currency": preview["base_currency"],
                    "settled_cash_usd": account["settled_cash_usd"],
                    "available_funds_base": account["available_funds_aud"],
                    "excess_liquidity_base": account["excess_liquidity_aud"],
                    "usd_to_base_rate": account["usd_to_aud"],
                    "unmanaged_position_stress_base": _unmanaged_stress(preview),
                },
                repository_root=ROOT,
                created_at_utc=selected_at,
                supersedes_plan_id=str(predecessor["plan_id"]),
            )
            generation_path, generation_sha = publish_portfolio_package_generation(
                ROOT, capital
            )
            stability_path, stability_sha = publish_portfolio_capital_owner_stability(
                ROOT,
                generation_path=generation_path,
                generation_sha256=generation_sha,
                observed_at_utc=selected_at,
            )
            publish_live_capital_plan(capital_path, capital)
            output["rollover"] = {
                "predecessor_selection_id": gold["selection_id"],
                "selection_id": selection["selection_id"],
                "selection_path": gold_path,
                "selection_file_sha256": gold_sha,
                "capital_plan_id": capital["plan_id"],
                "portfolio_generation_path": generation_path,
                "portfolio_generation_sha256": generation_sha,
                "capital_stability_path": stability_path,
                "capital_stability_sha256": stability_sha,
                "retained_xsp_selection_id": xsp["selection_id"],
                "retained_mcl_selection_id": mcl["selection_id"],
                "rollover_intent_id": rollover_intent["intent_id"],
                "recovered_after_interruption": False,
                "submitted_orders": 0,
                "verdict": "FRESH_FAIL_CLOSED_GOLD_RUN_SELECTED_FLAT",
            }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
