"""Historical tail risk: VaR and Expected Shortfall (Phase 5, ticket 12-01).

Unconditional historical estimators on a P&L or return series:
    VaR_q   = -Q_{1-q}(pnl)          (loss positive)
    ES_q    = -E[pnl | pnl <= Q_{1-q}]

Sign convention: inputs are P&L (or returns) as they happen — positive is
gain; VaR/ES come back POSITIVE numbers expressed as losses. Units pass
through untouched (percent daily returns, USD P&L, whatever the caller
parks in the series). Conditional vol lives in vol.py (GARCH); this module
is the unconditional tail view — plus the FILTERED estimator added with
the Phase 6 remediation (ticket 13-05): vol-scaled day-by-day VaR using
the lagged EWMA variance, which fixes the static historical estimator's
under-coverage on fat-tailed coffee returns (see validate.py findings).
"""

from __future__ import annotations

import numpy as np

import pandas as pd
from scipy.stats import norm

from hedging_workbench.vol import EWMA_LAMBDA


def filtered_var(
    price: pd.Series,
    multiplier: float = 1.0,
    lam: float = EWMA_LAMBDA,
    level: float = 0.95,
) -> pd.Series:
    """Day-by-day filtered VaR: z × lagged EWMA daily vol × lagged price
    × multiplier. Strictly no look-ahead: the variance at t−1 (EWMA over
    returns ≤ t−1) and the price at t−1 are both known at t−1.
    Returns a VaR series aligned to `price` (first rows NaN).
    """
    if not 0.5 < level < 1.0:
        raise ValueError(f"confidence level {level} outside (0.5, 1)")
    px = pd.Series(price).dropna()
    sig = np.sqrt(px.pct_change().ewm(alpha=1 - lam).var()).shift(1)
    z = float(norm.ppf(level))
    return z * sig * px.shift(1) * multiplier


def var_es(pnl: pd.Series, levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
    """Historical VaR and ES at each confidence level. Index: level."""
    pnl = pd.Series(pnl).dropna()
    if pnl.empty:
        raise ValueError("empty P&L series")
    rows = []
    for q in levels:
        if not 0.5 < q < 1.0:
            raise ValueError(f"confidence level {q} outside (0.5, 1)")
        tail_cut = pnl.quantile(1 - q)
        tail = pnl[pnl <= tail_cut]
        rows.append({"level": q, "var": -tail_cut, "es": -tail.mean()})
    out = pd.DataFrame(rows).set_index("level")
    bad = out[out["es"] < out["var"] - 1e-12]
    if not bad.empty:
        raise AssertionError(f"ES < VaR at {list(bad.index)}")
    return out
