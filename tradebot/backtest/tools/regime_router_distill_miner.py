"""Mine simple host-path rules from the regime-router distillation dataset."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _mine_bull_toxic(rows: list[dict[str, str]], *, toxic_ret: float, min_support: int) -> list[tuple]:
    bull = [r for r in rows if r.get("climate") == "bull_grind_low_vol" and _float(r, "fwd_21d_ret") is not None]
    out: list[tuple] = []
    for slow_ret, slow_dd, slow_rv, slow_eff, fast_ret, fast_dd in product(
        (0.00, 0.05, 0.10, 0.15),
        (0.20, 0.24, 0.28, 0.32),
        (0.50, 0.55, 0.60),
        (0.05, 0.08, 0.10, 0.12),
        (-0.10, -0.05, 0.00),
        (0.10, 0.14, 0.18, 0.22),
    ):
        toxic: list[float] = []
        safe: list[float] = []
        for row in bull:
            cond = (
                (_float(row, "slow_ret") or 0.0) <= float(slow_ret)
                and (_float(row, "slow_maxdd") or 0.0) >= float(slow_dd)
                and (_float(row, "slow_rv") or 0.0) >= float(slow_rv)
                and (_float(row, "slow_eff") or 0.0) <= float(slow_eff)
                and (_float(row, "fast_ret") or 0.0) <= float(fast_ret)
                and (_float(row, "fast_maxdd") or 0.0) >= float(fast_dd)
            )
            target = toxic if cond else safe
            target.append(float(_float(row, "fwd_21d_ret") or 0.0))
        if len(toxic) < int(min_support) or len(safe) < int(min_support):
            continue
        toxic_mean = mean(toxic)
        safe_mean = mean(safe)
        toxic_rate = sum(1 for v in toxic if v <= float(toxic_ret)) / len(toxic)
        safe_rate = sum(1 for v in safe if v <= float(toxic_ret)) / len(safe)
        score = (safe_mean - toxic_mean) + (toxic_rate - safe_rate)
        out.append(
            (
                score,
                len(toxic),
                toxic_mean,
                safe_mean,
                toxic_rate,
                safe_rate,
                slow_ret,
                slow_dd,
                slow_rv,
                slow_eff,
                fast_ret,
                fast_dd,
            )
        )
    out.sort(reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mine simple rules from the regime-router distillation dataset.")
    ap.add_argument("--csv", default="/tmp/tqqq_regime_router_distill.csv")
    ap.add_argument("--toxic-ret", type=float, default=-0.05)
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--out-md", default="/tmp/tqqq_regime_router_distill_miner.md")
    args = ap.parse_args(argv)

    rows = _rows(Path(args.csv))
    top = _mine_bull_toxic(rows, toxic_ret=float(args.toxic_ret), min_support=int(args.min_support))

    lines: list[str] = []
    lines.append("# Regime Router Distillation Miner")
    lines.append("")
    lines.append(f"Rows: **{len(rows)}**")
    lines.append(f"Toxic return threshold: **{float(args.toxic_ret):.4f}**")
    lines.append(f"Min support: **{int(args.min_support)}**")
    lines.append("")
    lines.append("| Rank | score | support | toxic mean | safe mean | toxic rate | safe rate | slow_ret<= | slow_dd>= | slow_rv>= | slow_eff<= | fast_ret<= | fast_dd>= |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(top[: int(args.top_k)], start=1):
        score, support, toxic_mean, safe_mean, toxic_rate, safe_rate, slow_ret, slow_dd, slow_rv, slow_eff, fast_ret, fast_dd = row
        lines.append(
            f"| {rank} | {score:.4f} | {support} | {toxic_mean:.4f} | {safe_mean:.4f} | "
            f"{toxic_rate:.3f} | {safe_rate:.3f} | {slow_ret:.2f} | {slow_dd:.2f} | {slow_rv:.2f} | {slow_eff:.2f} | {fast_ret:.2f} | {fast_dd:.2f} |"
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
