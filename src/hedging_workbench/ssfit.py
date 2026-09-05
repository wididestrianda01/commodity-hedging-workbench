"""Schwartz–Smith (2000) two-factor model fit to the frozen curve.

Model: log spot = chi_t + xi_t, with chi an OU short-term deviation
(mean-reversion kappa) and xi a GBM equilibrium level. Futures:

    ln F(tau) = chi * exp(-kappa * tau) + xi + A(tau)

A(tau) collects the drift and volatility terms (mu_xi, sigma_chi, sigma_xi,
rho). With a SINGLE frozen snapshot those are not separately identifiable,
so this fit uses the reduced form A(tau) -> slope * tau: a 4-parameter
nonlinear least squares on the snapshot (documented choice per ticket
10-03). Vol parameters need the full time-series MLE (Kalman) and are
out of scope here. With only ~8 maturities vs 4 parameters, kappa is
weakly identified — approximate standard errors are reported and the
notebook must surface them, not hide them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from hedging_workbench.carry import load_curve, yield_term_structure

LN2 = np.log(2.0)


@dataclass
class SSFit:
    chi: float            # short-term deviation of the last observation
    xi: float             # log long-run equilibrium level
    kappa: float          # mean-reversion speed (1/years)
    slope: float          # reduced-form drift slope (annualised, log space)
    rmse: float           # log-space residual RMSE
    se: dict[str, float]  # approximate parameter standard errors
    converged: bool

    @property
    def half_life(self) -> float:
        """Years for a spot deviation to decay by half: ln(2)/kappa."""
        return LN2 / self.kappa

    @property
    def long_run_price(self) -> float:
        return float(np.exp(self.xi))

    def forward(self, tau):
        """Model log-forward curve at time-to-maturity vector tau."""
        return self.chi * np.exp(-self.kappa * np.asarray(tau)) + self.xi \
            + self.slope * np.asarray(tau)


def _residuals(p, tau, ln_f):
    chi, xi, kappa, slope = p
    return chi * np.exp(-kappa * tau) + xi + slope * tau - ln_f


def fit_curve(curve: pd.DataFrame) -> SSFit:
    """Fit (chi, xi, kappa, slope) to a snapshot curve (see module docstring)."""
    tau = curve["ttm"].to_numpy(float)
    ln_f = np.log(curve["price"].to_numpy(float))
    # linear-regression start: intercept ~ xi + chi, slope ~ slope
    slope0, intercept0 = np.polyfit(tau, ln_f, 1)
    # kappa bounded >0: mean reversion is the model's premise; an unbounded
    # LM could land on negative kappa and abs() would misreport the optimum
    res = least_squares(_residuals, x0=[0.05, intercept0, 1.0, slope0],
                        bounds=([-np.inf, -np.inf, 1e-6, -np.inf],
                                [np.inf, np.inf, 50.0, np.inf]),
                        args=(tau, ln_f), method="trf", max_nfev=20000)
    chi, xi, kappa, slope = res.x
    dof = max(len(tau) - 4, 1)
    s2 = float(res.cost * 2 / dof)
    cov = s2 * np.linalg.pinv(res.jac.T @ res.jac)
    se = np.sqrt(np.abs(np.diag(cov)))
    return SSFit(chi=chi, xi=xi, kappa=kappa, slope=slope,
                 rmse=float(np.sqrt(np.mean(res.fun ** 2))),
                 se={"chi": se[0], "xi": se[1], "kappa": se[2], "slope": se[3]},
                 converged=bool(res.success))


def fit_from_frozen(universe: str = "coffee") -> tuple[pd.DataFrame, SSFit, pd.DataFrame]:
    """Convenience: frozen curve + fit + yield term structure in one call."""
    curve = load_curve(universe)
    ts = yield_term_structure(curve)
    return curve, fit_curve(curve), ts
