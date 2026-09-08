"""Volatility layer: GARCH(1,1) working vol with EWMA (λ=0.94) reference.

Returns are in PERCENT (arch's convention); all reported vols are
annualised %/yr. GARCH long-run variance = omega/(1 - alpha - beta),
annualised ×252. EWMA is the lambda-fixed special case of the same
filter — the upgrade is estimated persistence, not a likelihood win
(documented per ticket 10-04).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

EWMA_LAMBDA = 0.94


@dataclass
class VolFit:
    garch_longrun: float  # %/yr annualised long-run vol
    garch_last: float  # %/yr conditional vol at the last observation
    omega: float
    alpha: float
    beta: float
    loglikelihood: float
    ewma_annual: float  # %/yr EWMA reference (lambda = 0.94)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def garch_vol(returns_pct: pd.Series, lam: float = EWMA_LAMBDA) -> VolFit:
    """Fit GARCH(1,1) on percent daily returns; annualise; EWMA reference."""
    res = arch_model(returns_pct, vol="GARCH", p=1, q=1, mean="Constant").fit(
        disp="off"
    )
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    ewma = float(np.sqrt(252 * returns_pct.ewm(alpha=1 - lam).var().iloc[-1]))
    persistence = alpha + beta
    # Non-stationary fit (persistence >= 1, IGARCH on short/trending windows)
    # has no finite long-run variance — report NaN instead of dividing by ~0.
    longrun = (
        float(np.sqrt(252 * omega / (1 - persistence)))
        if persistence < 1 - 1e-8
        else float("nan")
    )
    return VolFit(
        omega=omega,
        garch_longrun=longrun,
        garch_last=float(res.conditional_volatility.iloc[-1] * np.sqrt(252)),
        alpha=alpha,
        beta=beta,
        loglikelihood=float(res.loglikelihood),
        ewma_annual=ewma,
    )


def vol_from_frozen() -> tuple[pd.Series, VolFit]:
    """Frozen KC=F daily % returns + the fit. Returns (returns, fit)."""
    from hedging_workbench.data.frozen import load

    kc = load(["KC=F"])["KC=F"].dropna()
    rets = kc.pct_change().dropna() * 100
    return rets, garch_vol(rets)
