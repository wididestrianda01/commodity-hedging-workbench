"""Margin-liquidity stress, FSB 2024 framing (ticket 10-08).

Scenarios apply a constant annualised price shock to the frozen KC=F path
from its as-of date forward (exp drift), then run the 10-05 variation-margin
engine. Scenario set: -1sigma / -2sigma / +2sigma (shocks from the Phase 2
GARCH working vol) plus a 'backwardation_widening' case — a steeper fall on
the back of the horizon, the crude stand-in for the front holding while
deferred prices sink.

Collar modelling assumption (stated): the long put's intrinsic gain is
assumed monetisable day-by-day, so the collar program's cumulative funding
loss is floored at the strike protection (f0 - put strike). Real clearing
treatment charges full futures VM daily and settles options separately —
this is a learning-model simplification, flagged for the notebook.

Peak funding need = largest drawdown of the margin balance below its
starting level; buffer = reference collateral - peak need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedging_workbench.conventions import DOLLARS_PER_CENT, INITIAL_MARGIN_PER_CONTRACT
from hedging_workbench.hedge import variation_margin


def stressed_path(
    base: pd.Series, shock_annual: float, horizon_days: int = 63
) -> pd.Series:
    """Business-day path from (and including) the frozen as-of date under a
    constant annualised shock. The base's last price anchors day 0 so the
    margin engine sees the full move from the as-of mark."""
    last = float(base.iloc[-1])
    idx = pd.DatetimeIndex([base.index[-1]]).append(
        pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    )
    t = np.arange(0, horizon_days + 1) / 252.0
    return pd.Series(last * np.exp(shock_annual * t), index=idx)


@dataclass
class StressResult:
    scenario: str
    shock_annual: float
    peak_need_usd: float  # futures program: max drawdown below start
    peak_date: pd.Timestamp
    collar_peak_need_usd: float | None = None
    buffer_usd: float | None = None  # vs reference collateral, futures peak


def run_scenario(
    contracts: float,
    base: pd.Series,
    shock_annual: float,
    scenario: str,
    collar=None,
    initial_margin: float = INITIAL_MARGIN_PER_CONTRACT,
    collateral_usd: float | None = None,
    horizon_days: int = 63,
) -> StressResult:
    """One stressed path -> margin profiles for futures (and collar)."""
    path = stressed_path(base, shock_annual, horizon_days)
    vm = variation_margin(contracts, path, initial_margin)
    start = initial_margin * abs(contracts)
    dd = start - vm["balance_usd"]
    peak_i = dd.idxmax()
    res = StressResult(
        scenario=scenario,
        shock_annual=shock_annual,
        peak_need_usd=float(dd.max()),
        peak_date=peak_i,
        buffer_usd=(
            collateral_usd - float(dd.max()) if collateral_usd is not None else None
        ),
    )
    if collar is not None:
        usd = contracts * DOLLARS_PER_CENT
        put_intrinsic = (collar.put_strike - path).clip(lower=0.0) * usd
        # premium paid upfront reduces the starting position; net balance
        # floored by the strike protection (see module docstring)
        collar_bal = vm["balance_usd"] + put_intrinsic
        floor = start - (collar.f0 - collar.put_strike) * usd
        collar_bal = collar_bal.clip(lower=floor)
        cdd = start - collar_bal
        res.collar_peak_need_usd = float(cdd.max())
    return res


def stress_report(
    contracts: float,
    base: pd.Series,
    sigma_annual: float,
    collar=None,
    collateral_usd: float | None = None,
    horizon_days: int = 63,
) -> list[StressResult]:
    """Full scenario set: -1s/-2s/+2s from the GARCH vol + backwardation widening."""
    shocks = [
        (-1.0 * sigma_annual, "down_1sigma"),
        (-2.0 * sigma_annual, "down_2sigma"),
        (+2.0 * sigma_annual, "up_2sigma"),
        (-1.5 * sigma_annual, "backwardation_widening"),
    ]
    return [
        run_scenario(
            contracts,
            base,
            s,
            name,
            collar,
            collateral_usd=collateral_usd,
            horizon_days=horizon_days,
        )
        for s, name in shocks
    ]
