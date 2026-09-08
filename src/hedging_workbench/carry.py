"""Carry layer: implied convenience-yield term structure from the frozen chain.

Cost-of-carry identity between consecutive maturities:

    F2 = F1 * exp((r + d - y) * tau)

so the implied total carry is c = ln(F2/F1) / tau and, given an assumed
storage/financing outlay d, the implied convenience yield is y = r + d - c.

Conventions (documented per ticket 10-02):
- Contract expiry is approximated as the 15th of the delivery month. Only
  maturity *differences* enter the spread identity, so the convention
  cancels to first order.
- d is not separately identifiable from spreads alone: it requires a
  storage/insurance outlay assumption. Default 0 (yield reported gross of
  storage); pass `storage` to decompose.
- The curve is fail-closed: every frozen series in the chain must end on
  the same date or load_curve raises (mixed as-of dates would corrupt
  spreads). Yahoo data; ICE Coffee C contract is 37,500 lb (¢/lb quotes).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from hedging_workbench.data.frozen import FROZEN_DIR, latest_rate, load
from hedging_workbench.data.universe import MONTH_NUM, UNIVERSES, contract_label

_EXPIRY_DAY = 15  # mid-month convention, see module docstring


def expiry(symbol: str) -> pd.Timestamp | None:
    """Delivery-month expiry approximation; continuous symbols -> None."""
    if symbol.endswith("=F"):
        return None
    root = symbol.split(".")[0][2:]
    return pd.Timestamp(
        year=2000 + int(root[1:]), month=MONTH_NUM[root[0]], day=_EXPIRY_DAY
    )


def implied_carry(f_near: float, f_far: float, tau: float) -> float:
    """Annualised total carry c = ln(F_far/F_near)/tau (negative = backwardation)."""
    return math.log(f_far / f_near) / tau


def implied_yield(
    f_near: float, f_far: float, tau: float, r: float, storage: float = 0.0
) -> float:
    """Implied convenience yield y = r + storage - carry."""
    return r + storage - implied_carry(f_near, f_far, tau)


def load_curve(universe: str = "coffee", frozen_dir: Path = FROZEN_DIR) -> pd.DataFrame:
    """Contract-level curve from the frozen chain: expiry, last price, ttm."""
    symbols = [s for s in UNIVERSES[universe] if not s.endswith("=F")]
    series = load(symbols, frozen_dir=frozen_dir)
    last_dates = {sym: s.dropna().index.max() for sym, s in series.items()}
    if len(set(last_dates.values())) > 1:
        raise ValueError(
            f"mixed as-of dates across chain: {last_dates} — "
            "refreeze before calibrating"
        )
    rows = []
    for sym in symbols:
        s = series[sym].dropna()
        rows.append(
            {
                "symbol": sym,
                "label": contract_label(sym),
                "expiry": expiry(sym),
                "price": float(s.iloc[-1]),
            }
        )
    df = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    curve_date = next(iter(last_dates.values()))
    df["ttm"] = (df["expiry"] - curve_date).dt.days / 365.25
    df.attrs["curve_date"] = curve_date
    df.attrs["universe"] = universe
    return df


def yield_term_structure(
    curve: pd.DataFrame, r: float | None = None, storage: float = 0.0
) -> pd.DataFrame:
    """Implied yield per consecutive spread pair + per-pair curve state."""
    if r is None:
        r = latest_rate()
    rows = []
    for near, far in zip(curve.index[:-1], curve.index[1:]):
        f1, f2 = curve.at[near, "price"], curve.at[far, "price"]
        tau = curve.at[far, "ttm"] - curve.at[near, "ttm"]
        carry = implied_carry(f1, f2, tau)
        rows.append(
            {
                "near": curve.at[near, "label"],
                "far": curve.at[far, "label"],
                "tau": tau,
                "f_near": f1,
                "f_far": f2,
                "carry": carry,
                "implied_yield": r + storage - carry,
                "state": "backwardation" if f2 < f1 else "contango",
            }
        )
    ts = pd.DataFrame(rows)
    ts.attrs.update(curve.attrs)
    ts.attrs["r"] = r
    ts.attrs["storage"] = storage
    return ts


def curve_state(ts: pd.DataFrame) -> str:
    """Overall state: 'backwardation', 'contango', or 'mixed'."""
    states = set(ts["state"])
    return states.pop() if len(states) == 1 else "mixed"
