"""Margin-liquidity stress, FSB 2024 framing (ticket 10-08).

Scenarios apply a constant annualised price shock to the frozen KC=F path
from its as-of date forward (exp drift), then run the 10-05 variation-margin
engine. Scenario set: -1sigma / -2sigma / +2sigma (shocks from the Phase 2
GARCH working vol) plus a 'backwardation_widening' case — the front keeps
drifting down at -1.5sigma, and from the first roll window on the drift is
steepened by a further -1.5sigma: the crude stand-in for rolling into a
sinking deferred leg while the front is held.

Assumptions stated (ticket 10-08): initial margin is the fixed per-contract
level in conventions.py; NO liquidity-spiral / procyclicality feedback is
modelled — margin levels and shocks are exogenous scenario inputs, not
endogenous to the book's own losses (FSB 2024 flags exactly this channel;
flagged as out of scope).

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
    base: pd.Series,
    shock_annual: float,
    horizon_days: int = 63,
    back_shock_annual: float | None = None,
    back_start_day: int = 21,
) -> pd.Series:
    """Business-day path from (and including) the frozen as-of date under
    a constant annualised shock. The base's last price anchors day 0 so
    the margin engine sees the full move from the as-of mark.

    back_shock_annual steepens the drift from `back_start_day` on — the
    backwardation-widening stand-in: the front is held while the deferred
    leg sinks, so rolling into the back costs progressively more.
    """
    last = float(base.iloc[-1])
    idx = pd.DatetimeIndex([base.index[-1]]).append(
        pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    )
    daily = np.full(horizon_days + 1, shock_annual / 252.0)
    daily[0] = 0.0  # the as-of mark itself carries no drift yet
    if back_shock_annual is not None:
        daily[back_start_day:] += back_shock_annual / 252.0
    return pd.Series(last * np.exp(np.cumsum(daily)), index=idx)


# Starbucks FY2025 10-K margin-collateral reference (notebook 03): the
# buffer the stress scenarios are measured against.
COLLATERAL_REFERENCE = 37.9e6


@dataclass
class StressResult:
    scenario: str
    shock_annual: float
    peak_need_usd: float  # futures program: max drawdown below start
    peak_date: pd.Timestamp
    collar_peak_need_usd: float | None = None
    buffer_usd: float | None = None  # vs reference collateral, futures peak
    profile: pd.DataFrame | None = None  # futures margin cash-flow profile
    collar_profile: pd.Series | None = None  # collar program balance


def run_scenario(
    contracts: float,
    base: pd.Series,
    shock_annual: float,
    scenario: str,
    collar=None,
    initial_margin: float = INITIAL_MARGIN_PER_CONTRACT,
    collateral_usd: float | None = None,
    horizon_days: int = 63,
    back_shock_annual: float | None = None,
    back_start_day: int = 21,
) -> StressResult:
    """One stressed path -> margin profiles for futures (and collar).

    Returns the full daily cash-flow profile on the result (ticket 10-08
    AC: 'margin cash-flow profile per program'), not just the peaks.
    """
    path = stressed_path(
        base, shock_annual, horizon_days, back_shock_annual, back_start_day
    )
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
        profile=vm,
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
        res.collar_profile = collar_bal
    return res


def stress_report(
    contracts: float,
    base: pd.Series,
    sigma_annual: float,
    collar=None,
    collateral_usd: float | None = None,
    horizon_days: int = 63,
) -> list[StressResult]:
    """Full scenario set: -1s/-2s/+2s from the GARCH vol, plus the
    backwardation-widening case — front drift steepened after the first
    roll window by a further -1.5 sigma (the deferred leg sinking while
    the front is held).
    """
    back = -1.5 * sigma_annual
    shocks = [
        (-1.0 * sigma_annual, "down_1sigma", None),
        (-2.0 * sigma_annual, "down_2sigma", None),
        (+2.0 * sigma_annual, "up_2sigma", None),
        (-1.5 * sigma_annual, "backwardation_widening", back),
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
            back_shock_annual=bs,
        )
        for s, name, bs in shocks
    ]
