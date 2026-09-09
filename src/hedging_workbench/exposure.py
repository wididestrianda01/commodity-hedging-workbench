"""Hedge-book exposure simulation: EE / EPE / PFE (Phase 5, tickets 12-02, 12-05).

Book view of a long futures (or collar-wrapped) position. Futures are
martingales under Q (zero-cost carry), so the front-contract price walks
a lognormal martingale

    F_{t+dt} = F_t * exp(-0.5*sigma^2*dt + sigma*sqrt(dt)*Z)

with sigma the Phase 2 GARCH working vol (decimal/yr). Documented choice:
the reduced-form SS fit carries no vol parameters (ssfit.py), so the
martingale + GARCH vol is the honest simulated dynamics for exposure
purposes; the SS fit anchors the LEVEL of the curve, not the paths.
MtM in USD = contracts * DOLLARS_PER_CENT * (F_t - entry).

Exposure profile:
    EE(t)   = E[max(V_t, 0)]
    EPE     = time-average of EE(t)
    PFE(t)  = quantile_q of max(V_t, 0)  (q = 0.95 default)

Netting and collateral (12-05) are pure transforms on MtM path arrays:
    netted MtM      = elementwise sum of position paths
    collateralized  = max(V - (IA + threshold), 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedging_workbench.conventions import DOLLARS_PER_CENT
from hedging_workbench.pricing import Collar, black76
from hedging_workbench.sim import martingale_paths


@dataclass(frozen=True)
class FuturesBook:
    """Long futures book: total contracts, entry price (cents/lb)."""

    contracts: float
    entry: float

    def mtm(self, f_paths: np.ndarray) -> np.ndarray:
        """USD MtM paths, shape (n_paths, n_steps+1)."""
        return self.contracts * DOLLARS_PER_CENT * (f_paths - self.entry)

    def pnl_from_returns(self, returns_pct) -> np.ndarray:
        """USD P&L from PERCENT returns of the front: r/100 * notional."""
        return (
            np.asarray(returns_pct, dtype=float)
            / 100.0
            * self.contracts
            * DOLLARS_PER_CENT
            * self.entry
        )


def simulate_front(
    f0: float,
    sigma: float,
    horizon_years: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Lognormal martingale paths for the front futures price.

    Shape (n_paths, steps+1); column 0 = f0. E[F_t] = f0 by construction.
    Delegates to the shared engine in sim.martingale_paths.
    """
    return martingale_paths(f0, sigma, horizon_years, steps, n_paths, seed)


def profile(mtm: np.ndarray, pfe_q: float = 0.95, dt: float = 1 / 252) -> pd.DataFrame:
    """EE(t)/PFE(t) profile from MtM paths (n_paths, n_steps+1)."""
    expo = np.maximum(mtm, 0.0)
    return pd.DataFrame(
        {
            "t": np.arange(mtm.shape[1]) * dt,
            "ee": expo.mean(axis=0),
            "pfe": np.quantile(expo, pfe_q, axis=0),
        }
    )


def epe(prof: pd.DataFrame) -> float:
    """Expected positive exposure: time-average of EE(t)."""
    return float(prof["ee"].mean())


# -- 12-05: netting and collateral -------------------------------------------


def netted_mtm(paths: list[np.ndarray]) -> np.ndarray:
    """Elementwise sum of position MtM paths (ISDA netting-set view)."""
    if not paths:
        raise ValueError("empty netting set")
    out = paths[0].copy()
    for p in paths[1:]:
        if p.shape != out.shape:
            raise ValueError("netting-set paths must share a shape")
        out += p
    return out


def collateralized(
    mtm: np.ndarray, threshold: float, ia: float = 0.0, mta: float = 0.0
) -> np.ndarray:
    """Exposure after collateral: max(V - held, 0).

    Collateral mechanics (12-05): IA + threshold are held in full, and
    above the threshold further transfers happen in COMPLETED MTA round
    lots (floor) — so the residual uncollateralised exposure is bounded
    by the MTA, the granularity that makes small moves post nothing.
    mta=0 keeps the continuous threshold block (no granularity).
    """
    v = np.asarray(mtm, dtype=float)
    if mta <= 0:
        return np.maximum(v - (ia + threshold), 0.0)
    lots = np.floor(np.maximum(v - threshold, 0.0) / mta)
    held = ia + threshold + lots * mta
    return np.maximum(v - held, 0.0)


def collar_book_mtm(
    contracts: float,
    collar: Collar,
    f_paths: np.ndarray,
    t_ttm: np.ndarray,
    sigma: float,
    r: float,
) -> np.ndarray:
    """MtM paths of long futures + long put + short call (collar wrap).

    t_ttm: remaining time to each column's option expiry (years), same
    length as f_paths columns. The collar is repriced along each path.
    """
    fut = FuturesBook(contracts, float(f_paths[0, 0])).mtm(f_paths)
    opt = np.zeros_like(f_paths)
    for j in range(f_paths.shape[1]):
        if t_ttm[j] <= 0:
            continue
        f = f_paths[:, j]
        put = black76("put", f, collar.put_strike, t_ttm[j], sigma, r)
        call = black76("call", f, collar.call_strike, t_ttm[j], sigma, r)
        opt[:, j] = contracts * DOLLARS_PER_CENT * (put - call)
    return fut + opt
