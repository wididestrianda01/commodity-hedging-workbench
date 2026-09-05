"""Data gates: count/completeness assertions, fail-closed to the pre-declared
gold fallback. Pattern inherited from the P18 C-gates.

CLI:  python -m hedging_workbench.data.gates
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class GateError(RuntimeError):
    """Both universes failed their gates — no usable curve, fail closed."""


MIN_ROWS = 120          # ~6 months of daily bars per contract
MAX_NAN_FRAC = 0.05
MAX_PRICE = {"coffee": 2000.0,   # cents/lb — coffee sanity ceiling
             "gold": 20000.0}    # $/oz — gold sanity ceiling


@dataclass
class GateReport:
    universe: str
    ok: bool
    fallback_used: bool = False
    series: dict[str, dict] = field(default_factory=dict)   # symbol -> stats
    failures: list[str] = field(default_factory=list)
    reason: str = ""

    def summary(self) -> str:
        state = "PASS" if self.ok else "FAIL"
        fb = " (gold fallback)" if self.fallback_used else ""
        return f"[{state}]{fb} {self.universe}: {len(self.series)} series, {len(self.failures)} failures {self.reason}".strip()


def check_series(s: pd.Series, min_rows: int = MIN_ROWS,
                 max_nan_frac: float = MAX_NAN_FRAC,
                 max_price: float = 20000.0) -> list[str]:
    """Return failure messages for one series (empty = pass)."""
    fails = []
    if len(s) < min_rows:
        fails.append(f"rows {len(s)} < {min_rows}")
    if not s.index.is_monotonic_increasing:
        fails.append("index not monotonic")
    if s.index.has_duplicates:
        fails.append("duplicate dates")
    nan_frac = float(s.isna().mean())
    if nan_frac > max_nan_frac:
        fails.append(f"nan_frac {nan_frac:.2%} > {max_nan_frac:.0%}")
    if (s <= 0).any():
        fails.append("non-positive prices")
    if (s > max_price).any():
        fails.append(f"price above sanity ceiling {max_price}")
    return fails


def run_gates(series: dict[str, pd.Series], universe: str) -> GateReport:
    """Assert count/completeness on every series of a universe."""
    report = GateReport(universe=universe, ok=True)
    for symbol, s in series.items():
        fails = check_series(s, max_price=MAX_PRICE.get(universe, 20000.0))
        report.series[symbol] = {
            "rows": int(len(s)),
            "last_close": float(s.dropna().iloc[-1]),
            "last_date": str(s.dropna().index[-1].date()),
        }
        if fails:
            report.ok = False
            report.failures.extend(f"{symbol}: {f}" for f in fails)
    if not series:
        report.ok = False
        report.failures.append("no series at all")
    return report


def evaluate(coffee: dict[str, pd.Series],
             gold: dict[str, pd.Series]) -> GateReport:
    """Coffee primary, gold fallback, fail closed if both fail."""
    report = run_gates(coffee, "coffee")
    if report.ok:
        return report
    fallback = run_gates(gold, "gold")
    fallback.fallback_used = True
    fallback.reason = f"coffee failed: {'; '.join(report.failures[:3])}"
    if fallback.ok:
        return fallback
    raise GateError(
        "coffee and gold gates both failed — fail closed, no curve.\n"
        f"  coffee: {'; '.join(report.failures)}\n"
        f"  gold:   {'; '.join(fallback.failures)}"
    )


def main() -> None:
    from pathlib import Path

    from hedging_workbench.data.download import FROZEN_DIR, load_frozen, verify_manifest
    from hedging_workbench.data.universe import UNIVERSES

    bad = (verify_manifest(Path(FROZEN_DIR), "coffee")
           + verify_manifest(Path(FROZEN_DIR), "gold"))
    if bad:
        raise GateError(f"manifest checksum mismatch for {bad}")
    coffee = load_frozen(list(UNIVERSES["coffee"]))
    gold = load_frozen(list(UNIVERSES["gold"]))
    report = evaluate(coffee, gold)
    print(report.summary())
    for symbol, stats in report.series.items():
        print(f"  {symbol:<12} rows={stats['rows']:<6} last={stats['last_close']:>9.2f}  {stats['last_date']}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
