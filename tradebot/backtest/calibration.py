"""Calibration against delayed option LAST prices."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterable

from ib_insync import IB, ContFuture, Option, FuturesOption, Stock, util

from .config import ConfigBundle
from ..config import load_config
from .data import IBKRHistoricalData
from .synth import IVSurfaceParams, black_76, black_scholes, ewma_vol, iv_atm, iv_for_strike


@dataclass(frozen=True)
class CalibrationParams:
    iv_floor: float
    iv_risk_premium: float
    skew: float
    term_slope: float


@dataclass(frozen=True)
class CalibrationRecord:
    asof: str
    params: CalibrationParams
    mae: float
    mape: float
    samples: int


@dataclass
class CalibrationBucket:
    min_dte: int
    max_dte: int
    records: list[CalibrationRecord]


@dataclass
class CalibrationBook:
    symbol: str
    buckets: list[CalibrationBucket]

    def latest_params(self, dte: int) -> CalibrationParams | None:
        for bucket in self.buckets:
            if bucket.min_dte <= dte <= bucket.max_dte and bucket.records:
                return bucket.records[-1].params
        return None

    def surface_params(self, dte: int, base: IVSurfaceParams) -> IVSurfaceParams:
        override = self.latest_params(dte)
        if not override:
            return base
        return IVSurfaceParams(
            rv_lookback=base.rv_lookback,
            rv_ewma_lambda=base.rv_ewma_lambda,
            iv_risk_premium=override.iv_risk_premium,
            iv_floor=override.iv_floor,
            term_slope=override.term_slope,
            skew=override.skew,
        )


DEFAULT_BUCKETS: list[tuple[int, int]] = [
    (0, 2),
    (3, 7),
    (8, 30),
    (31, 60),
    (61, 120),
    (121, 3650),
]

_FUTURE_EXCHANGES = {
    "MNQ": "CME",
    "MBT": "CME",
}


def load_calibration(calibration_dir: Path, symbol: str) -> CalibrationBook | None:
    path = calibration_dir / f"{symbol}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    buckets: list[CalibrationBucket] = []
    for bucket in raw.get("buckets", []):
        records = [
            CalibrationRecord(
                asof=rec["asof"],
                params=CalibrationParams(**rec["params"]),
                mae=rec["mae"],
                mape=rec["mape"],
                samples=rec["samples"],
            )
            for rec in bucket.get("records", [])
        ]
        buckets.append(
            CalibrationBucket(
                min_dte=bucket["min_dte"],
                max_dte=bucket["max_dte"],
                records=records,
            )
        )
    return CalibrationBook(symbol=raw.get("symbol", symbol), buckets=buckets)


def save_calibration(calibration_dir: Path, book: CalibrationBook) -> None:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": book.symbol,
        "buckets": [
            {
                "min_dte": bucket.min_dte,
                "max_dte": bucket.max_dte,
                "records": [
                    {
                        "asof": rec.asof,
                        "params": {
                            "iv_floor": rec.params.iv_floor,
                            "iv_risk_premium": rec.params.iv_risk_premium,
                            "skew": rec.params.skew,
                            "term_slope": rec.params.term_slope,
                        },
                        "mae": rec.mae,
                        "mape": rec.mape,
                        "samples": rec.samples,
                    }
                    for rec in bucket.records
                ],
            }
            for bucket in book.buckets
        ],
    }
    path = calibration_dir / f"{book.symbol}.json"
    path.write_text(json.dumps(payload, indent=2))


def ensure_calibration(cfg: ConfigBundle) -> CalibrationBook | None:
    symbol = cfg.strategy.symbol
    book = load_calibration(cfg.backtest.calibration_dir, symbol)
    asof = datetime.now().date().isoformat()
    if book and all(_has_record_today(bucket, asof) for bucket in book.buckets):
        return book
    updated = calibrate_symbol(cfg, book, asof)
    if updated:
        save_calibration(cfg.backtest.calibration_dir, updated)
    return updated


def calibrate_symbol(cfg: ConfigBundle, book: CalibrationBook | None, asof: str) -> CalibrationBook | None:
    symbol = cfg.strategy.symbol
    if book is None:
        book = CalibrationBook(
            symbol=symbol,
            buckets=[CalibrationBucket(min_dte=a, max_dte=b, records=[]) for a, b in DEFAULT_BUCKETS],
        )

    ibkr = load_config()
    ib = IB()
    ib.connect(ibkr.host, ibkr.port, clientId=ibkr.client_id + 80, timeout=8)
    ib.reqMarketDataType(3)

    spot, chain, is_future = _resolve_chain(ib, symbol, cfg.strategy.exchange)
    if spot is None or chain is None:
        ib.disconnect()
        return book

    rv = _recent_realized_vol(cfg, symbol, cfg.strategy.exchange)

    for bucket in book.buckets:
        if _has_record_today(bucket, asof):
            continue
        expiry = _pick_expiry(chain.expirations, bucket.min_dte, bucket.max_dte)
        if not expiry:
            continue
        samples = _sample_contracts(ib, symbol, chain, expiry, spot, is_future)
        if not samples:
            continue
        params, mae, mape, count = _fit_params(samples, spot, rv, cfg, is_future)
        bucket.records.append(
            CalibrationRecord(
                asof=asof,
                params=params,
                mae=mae,
                mape=mape,
                samples=count,
            )
        )

    ib.disconnect()
    return book


@dataclass(frozen=True)
class _Sample:
    strike: float
    right: str
    expiry: date
    last: float


def _resolve_chain(ib: IB, symbol: str, exchange: str | None) -> tuple[float | None, object | None, bool]:
    if exchange is None and symbol in _FUTURE_EXCHANGES:
        exchange = _FUTURE_EXCHANGES[symbol]
    if exchange and exchange != "SMART":
        cont = ContFuture(symbol, exchange, "USD")
        ib.qualifyContracts(cont)
        ticker = ib.reqTickers(cont)[0]
        spot = ticker.marketPrice() or ticker.last or ticker.close
        chains = ib.reqSecDefOptParams(cont.symbol, exchange, "FUT", cont.conId)
        chain = chains[0] if chains else None
        return spot, chain, True
    underlying = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(underlying)
    ticker = ib.reqTickers(underlying)[0]
    spot = ticker.marketPrice() or ticker.last or ticker.close
    chains = ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0] if chains else None)
    return spot, chain, False


def _pick_expiry(expirations: Iterable[str], min_dte: int, max_dte: int) -> str | None:
    today = datetime.now().date()
    chosen = None
    for exp in sorted(expirations):
        try:
            exp_date = datetime.strptime(exp, "%Y%m%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if min_dte <= dte <= max_dte:
            chosen = exp
            break
    return chosen


def _sample_contracts(ib: IB, symbol: str, chain, expiry: str, spot: float, is_future: bool) -> list[_Sample]:
    if spot is None:
        return []
    targets = [1.0, 2.5, 5.0]
    strikes = sorted(chain.strikes)
    selections: set[float] = set()
    selections.add(_nearest_strike(strikes, spot))
    for pct in targets:
        selections.add(_nearest_strike(strikes, spot * (1 - pct / 100.0)))
        selections.add(_nearest_strike(strikes, spot * (1 + pct / 100.0)))

    rights = ["P", "C"]
    contracts = []
    for strike in sorted(selections):
        for right in rights:
            if is_future:
                contracts.append(
                    FuturesOption(
                        symbol,
                        expiry,
                        strike,
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
                        strike,
                        right,
                        exchange=chain.exchange,
                        currency="USD",
                        tradingClass=chain.tradingClass,
                    )
                )

    if not contracts:
        return []
    ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*contracts)
    samples: list[_Sample] = []
    for contract, ticker in zip(contracts, tickers):
        last = ticker.last
        if last is None or (isinstance(last, float) and math.isnan(last)):
            continue
        samples.append(
            _Sample(
                strike=float(contract.strike),
                right=str(contract.right),
                expiry=datetime.strptime(expiry, "%Y%m%d").date(),
                last=float(last),
            )
        )
    return samples


def _fit_params(samples: list[_Sample], spot: float, rv: float, cfg: ConfigBundle, is_future: bool) -> tuple[CalibrationParams, float, float, int]:
    base = cfg.synthetic
    iv_floors = _grid(base.iv_floor, [0.03, 0.05, 0.08, 0.1])
    risk_premiums = _grid(base.iv_risk_premium, [1.0, 1.2, 1.4, 1.6])
    skews = _grid(base.skew, [-0.5, -0.3, -0.2, -0.1, 0.0])
    term_slopes = _grid(base.term_slope, [0.0, 0.02, 0.04])

    best_params = CalibrationParams(
        iv_floor=base.iv_floor,
        iv_risk_premium=base.iv_risk_premium,
        skew=base.skew,
        term_slope=base.term_slope,
    )
    best_mae = float("inf")
    best_mape = float("inf")
    count = len(samples)

    for iv_floor in iv_floors:
        for risk in risk_premiums:
            for skew in skews:
                for term in term_slopes:
                    params = CalibrationParams(iv_floor=iv_floor, iv_risk_premium=risk, skew=skew, term_slope=term)
                    mae, mape = _score_params(samples, spot, rv, cfg, params, is_future)
                    if mae < best_mae:
                        best_mae = mae
                        best_mape = mape
                        best_params = params

    return best_params, best_mae, best_mape, count


def _score_params(
    samples: list[_Sample],
    spot: float,
    rv: float,
    cfg: ConfigBundle,
    params: CalibrationParams,
    is_future: bool,
) -> tuple[float, float]:
    abs_errors = []
    pct_errors = []
    for sample in samples:
        dte = max((sample.expiry - datetime.now().date()).days, 0)
        iv_params = IVSurfaceParams(
            rv_lookback=cfg.synthetic.rv_lookback,
            rv_ewma_lambda=cfg.synthetic.rv_ewma_lambda,
            iv_risk_premium=params.iv_risk_premium,
            iv_floor=params.iv_floor,
            term_slope=params.term_slope,
            skew=params.skew,
        )
        atm_iv = iv_atm(rv, dte, iv_params)
        opt_iv = iv_for_strike(atm_iv, spot, sample.strike, iv_params)
        t = max(dte / 365.0, 1 / (24 * 365))
        if is_future:
            mid = black_76(spot, sample.strike, t, cfg.backtest.risk_free_rate, opt_iv, sample.right)
        else:
            mid = black_scholes(spot, sample.strike, t, cfg.backtest.risk_free_rate, opt_iv, sample.right)
        abs_errors.append(abs(mid - sample.last))
        denom = max(sample.last, 0.01)
        pct_errors.append(abs(mid - sample.last) / denom)
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("inf")
    mape = sum(pct_errors) / len(pct_errors) if pct_errors else float("inf")
    return mae, mape


def _grid(default: float, candidates: list[float]) -> list[float]:
    values = sorted(set([default] + candidates))
    return values


def _nearest_strike(strikes: list[float], target: float) -> float:
    return min(strikes, key=lambda s: abs(s - target))


def _recent_realized_vol(cfg: ConfigBundle, symbol: str, exchange: str | None) -> float:
    data = IBKRHistoricalData()
    end = datetime.now()
    start = end - timedelta(days=10)
    bars = data.load_or_fetch_bars(
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=cfg.backtest.bar_size,
        use_rth=cfg.backtest.use_rth,
        cache_dir=cfg.backtest.cache_dir,
    )
    closes = [bar.close for bar in bars if bar.close]
    returns = []
    for idx in range(1, len(closes)):
        if closes[idx - 1] > 0:
            returns.append(math.log(closes[idx] / closes[idx - 1]))
    if not returns:
        return cfg.synthetic.iv_floor
    rv = ewma_vol(returns[-cfg.synthetic.rv_lookback :], cfg.synthetic.rv_ewma_lambda)
    return rv * math.sqrt(_annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))


def _annualization_factor(bar_size: str, use_rth: bool) -> float:
    label = bar_size.lower()
    if "hour" in label:
        return 252 * (6.5 if use_rth else 24)
    if "day" in label:
        return 252
    return 252 * (6.5 if use_rth else 24)


def _has_record_today(bucket: CalibrationBucket, asof: str) -> bool:
    return any(rec.asof == asof for rec in bucket.records)
