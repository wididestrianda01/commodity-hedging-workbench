"""Black-76 pricing on futures (ticket 10-06, reused by the Phase 4 note).

European option on a futures contract, premiums DISCOUNTED:
    c = exp(-rT) [F N(d1) - K N(d2)]
    p = exp(-rT) [K N(-d2) - F N(-d1)]
    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)

Conventions:
- F, K in the quote unit of the underlying (cents/lb for coffee); premiums
  come back in the SAME unit. USD premium = cents * 375 per contract.
- sigma is a decimal (0.385 = 38.5%/yr) — the Phase 2 GARCH working vol.
- T in years, r a decimal. No early-exercise premium (European assumption;
  American futures options exist but no free coffee option quotes — stated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from hedging_workbench.conventions import lb_to_usd


def d1(f, k, t, sigma: float):
    """Black-76 d1: (ln(F/K) + sigma^2 t / 2) / (sigma sqrt(t)).
    Scalar or array (broadcast) inputs."""
    f, k, t = np.asarray(f, float), np.asarray(k, float), np.asarray(t, float)
    return (np.log(f / k) + 0.5 * sigma * sigma * t) / (sigma * np.sqrt(t))


def black76(kind: str, f, k, t, sigma: float, r: float):
    """Black-76 premium in the underlying's quote unit ('call' or 'put').

    Scalar or array inputs (broadcast); arrays repriced along paths share
    this exact implementation with the scalar callers.
    """
    f, k, t = np.asarray(f, float), np.asarray(k, float), np.asarray(t, float)
    if np.any(t <= 0):
        intrinsic = np.maximum(f - k, 0.0) if kind == "call" else np.maximum(k - f, 0.0)
        return intrinsic.item() if intrinsic.ndim == 0 else intrinsic
    d1v, d2 = d1(f, k, t, sigma), d1(f, k, t, sigma) - sigma * np.sqrt(t)
    df = np.exp(-r * t)
    if kind == "call":
        return df * (f * norm.cdf(d1v) - k * norm.cdf(d2))
    return df * (k * norm.cdf(-d2) - f * norm.cdf(-d1v))


@dataclass
class Collar:
    """Zero-cost collar on the long futures position."""

    f0: float  # futures at inception (cents/lb)
    put_strike: float  # protection level (floor)
    call_strike: float  # cap given up (set by zero-cost identity)
    put_premium: float
    call_premium: float
    t: float
    sigma: float
    r: float

    @property
    def premium_gap(self) -> float:
        """|put - call| in cents/lb; ~0 by construction."""
        return abs(self.put_premium - self.call_premium)

    def payoff(self, f_terminal: float) -> float:
        """Per-cent/lb terminal P&L of futures + long put + short call."""
        fut = f_terminal - self.f0
        put = max(self.put_strike - f_terminal, 0.0) - self.put_premium
        call = self.call_premium - max(f_terminal - self.call_strike, 0.0)
        return fut + put + call


def zero_cost_collar(
    f0: float,
    put_strike: float,
    t: float,
    sigma: float,
    r: float,
    hi: float | None = None,
) -> Collar:
    """Find the call strike making the short call pay for the long put.

    put_strike < f0 < call strike, and the call premium is strictly
    decreasing in its strike, so brentq on [f0, hi] brackets the root
    (hi defaults to 2*f0).
    """
    put_p = black76("put", f0, put_strike, t, sigma, r)
    g = lambda k: black76("call", f0, k, t, sigma, r) - put_p
    hi = hi if hi is not None else 2.0 * f0
    call_k = brentq(g, f0 + 1e-9, hi)
    call_p = black76("call", f0, call_k, t, sigma, r)
    return Collar(
        f0=f0,
        put_strike=put_strike,
        call_strike=call_k,
        put_premium=put_p,
        call_premium=call_p,
        t=t,
        sigma=sigma,
        r=r,
    )


def collar_vs_futures(volume_lb: float, collar: Collar, f_terminal: float) -> dict:
    """Terminal USD outcome, long futures vs collar, same volume."""
    usd = lb_to_usd(volume_lb)
    fut = (f_terminal - collar.f0) * usd
    return {
        "f_terminal": f_terminal,
        "futures_usd": fut,
        "collar_usd": collar.payoff(f_terminal) * usd,
    }


def margin_relief(volume_lb: float, collar: Collar, shock_floor: float) -> dict:
    """Worst-case additional variation margin under a price shock to shock_floor.

    Long futures loses unboundedly down to the floor; the collar's put caps
    the loss at f0 - put_strike. Returns worst extra margin per program (USD).
    """
    usd = lb_to_usd(volume_lb)
    return {
        "shock_floor": shock_floor,
        "futures_worst_usd": (collar.f0 - shock_floor) * usd,
        "collar_worst_usd": max(collar.f0 - collar.put_strike, 0.0) * usd,
    }
