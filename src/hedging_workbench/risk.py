"""Historical tail risk: VaR and Expected Shortfall (Phase 5, ticket 12-01).

Unconditional historical estimators on a P&L or return series:
    VaR_q   = -Q_{1-q}(pnl)          (loss positive)
    ES_q    = -E[pnl | pnl <= Q_{1-q}]

Sign convention: inputs are P&L (or returns) as they happen — positive is
gain; VaR/ES come back POSITIVE numbers expressed as losses. Units pass
through untouched (percent daily returns, USD P&L, whatever the caller
parks in the series). Conditional vol lives in vol.py (GARCH); this module
is the unconditional tail view.
"""

from __future__ import annotations

import pandas as pd


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
