"""Unilateral CVA on simulated exposure (Phase 5, tickets 12-03, 12-04).

Discretized unilateral CVA over exposure buckets [t_{i-1}, t_i]:

    CVA = LGD * sum_i  EE(t_i) * df(t_i) * [ S(t_{i-1}) - S(t_i) ]

with S(t) = exp(-hazard * t) the (flat-hazard) survival probability and
df(t) = exp(-r * t) the SOFR-flat discount factor (rates.latest_rate).
Unilateral only — no DVA, per spec. Sign convention: CVA >= 0 whenever
EE >= 0 and parameters are sensible; it is the expected loss from the
COUNTERPARTY defaulting while in the money to us.

QuantLib benchmark (12-04): the matched case re-derives the same
discretization from QuantLib's own FlatHazardRate survival curve and
FlatForward discount curve. Same inputs, same grid — the benchmark
validates our hazard/discount CONVENTIONS against the library; the
pre-declared tolerance covers float noise only. (QuantLib's simulation
engines would benchmark the exposure model too, but with different
dynamics — out of scope here, see spec.)
"""

from __future__ import annotations

import numpy as np


def survival(t: float | np.ndarray, hazard: float) -> float | np.ndarray:
    """Flat-hazard survival probability S(t) = exp(-h t)."""
    return np.exp(-hazard * np.asarray(t, dtype=float))


def discount(t: float | np.ndarray, r: float) -> float | np.ndarray:
    """Flat SOFR discount factor df(t) = exp(-r t)."""
    return np.exp(-r * np.asarray(t, dtype=float))


def cva_unilateral(
    ee: np.ndarray, tenors: np.ndarray, hazard: float, lgd: float, r: float
) -> float:
    """Unilateral CVA (same units as ee) on a bucketed EE profile.

    ee[i] pairs with the bucket ENDING at tenors[i]; S starts at 1.
    """
    ee = np.asarray(ee, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    if ee.shape != tenors.shape or ee.ndim != 1 or len(ee) == 0:
        raise ValueError("ee and tenors must be same-length 1-D arrays")
    if (np.diff(tenors) <= 0).any() or tenors[0] <= 0:
        raise ValueError("tenors must be strictly increasing and positive")
    s_prev = np.concatenate([[1.0], survival(tenors[:-1], hazard)])
    ds = s_prev - survival(tenors, hazard)
    return float(lgd * np.sum(ee * discount(tenors, r) * ds))


def cva_quantlib(
    ee: np.ndarray, tenors: np.ndarray, hazard: float, lgd: float, r: float
) -> float:
    """Same buckets priced through QuantLib's FlatHazardRate/FlatForward
    curves. Convention benchmark, not a different model: identical
    discretisation, so agreement certifies OUR survival/discount/pairing
    conventions (notebook 04). Requires the QuantLib package.
    """
    try:
        import QuantLib as ql
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install QuantLib-Python to run the benchmark"
        ) from e
    ee = np.asarray(ee, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    ql.Settings.instance().evaluationDate = ql.Date(
        4, ql.September, 2026
    )  # frozen curve date
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    hzc = ql.FlatHazardRate(0, cal, ql.QuoteHandle(ql.SimpleQuote(hazard)), dc)
    dfc = ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(r)), dc)
    s_ql = np.array([hzc.survivalProbability(t) for t in tenors])
    df_ql = np.array([dfc.discount(t) for t in tenors])
    return float(lgd * np.sum(ee * df_ql * (np.r_[1.0, s_ql[:-1]] - s_ql)))
