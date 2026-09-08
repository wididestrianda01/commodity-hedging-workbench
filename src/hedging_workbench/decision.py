"""Hedged vs unhedged client outcome (Phase 6, ticket 13-02).

The decision-report numbers: for a roaster committed to buying
`volume_lb` of coffee at the terminal date, compare three strategies:

- UNHEDGED: cost = volume × f_T (unbounded both ways).
- FUTURES: long hedge locks the price — net cost = volume × f0
  regardless of f_T; the residual risk is margin funding, measured as
  the peak cumulative drawdown of the futures position along the path.
- COLLAR: futures + long put + short call — cost locked between the
  strikes. The strikes bound the HEDGE P&L, not the cost: below the
  put and above the call the net cost moves 1:1 with the market again
  (a crash below the put keeps helping the buyer, a spike above the
  call keeps hurting). Margin peak capped at the put protection
  (same monetisable-put convention as stress.py, ticket 10-08).
No new analytics — only the comparison table over the Phase 3 program
and the Phase 5 scenario engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedging_workbench.carry import load_curve
from hedging_workbench.conventions import DOLLARS_PER_CENT, lb_to_usd
from hedging_workbench.pricing import Collar, collar_vs_futures, zero_cost_collar
from hedging_workbench.sim import martingale_paths
from hedging_workbench.vol import vol_from_frozen


def net_costs(
    volume_lb: float, collar: Collar, f_terminal: np.ndarray
) -> pd.DataFrame:
    """Per-scenario net purchase cost (USD) for the three strategies.

    Physical cost volume×f_T is incurred under every strategy; each
    hedge's terminal P&L (collar_vs_futures) reduces it.
    """
    f_terminal = np.asarray(f_terminal, dtype=float)
    usd = lb_to_usd(volume_lb)
    pnl = np.array(
        [
            (
                collar_vs_futures(volume_lb, collar, f)["futures_usd"],
                collar_vs_futures(volume_lb, collar, f)["collar_usd"],
            )
            for f in f_terminal
        ]
    )
    cost = f_terminal * usd  # cents/lb × $/cent = USD
    return pd.DataFrame(
        {
            "unhedged": cost,
            "futures": cost - pnl[:, 0],  # volume × f0 for every scenario
            "collar": cost - pnl[:, 1],
        }
    )


def margin_peaks(
    contracts: float,
    paths: np.ndarray,
    f0: float,
    put_strike: float | None = None,
) -> dict[str, float]:
    """Peak variation-margin funding per strategy along each path.

    Long loses when price falls: peak = contracts × $375/ct × max(0, f0 −
    path min). Under the collar the cumulative funding loss is CAPPED at
    the put protection (f0 − put_strike): with the long put's gain
    assumed monetisable day-by-day (stress.py convention, ticket 10-08)
    further price declines stop draining margin. Reported as mean and
    95th percentile across paths.
    """
    drawdown = np.maximum(f0 - paths.min(axis=1), 0.0) * contracts * DOLLARS_PER_CENT
    out = {"futures_mean": float(drawdown.mean()),
           "futures_p95": float(np.quantile(drawdown, 0.95))}
    if put_strike is not None:
        capped = np.minimum(
            drawdown, max(f0 - put_strike, 0.0) * contracts * DOLLARS_PER_CENT
        )
        out["collar_mean"] = float(capped.mean())
        out["collar_p95"] = float(np.quantile(capped, 0.95))
    return out


def decision_table(
    volume_lb: float,
    contracts: float,
    f0: float,
    sigma: float,
    r: float,
    put_strike: float,
    t: float = 0.25,
    n_paths: int = 50_000,
    steps: int = 63,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per strategy: net-cost distribution + margin peak (USD)."""
    collar = zero_cost_collar(f0, put_strike, t, sigma, r)
    paths = martingale_paths(f0, sigma, t, steps, n_paths, seed)
    costs = net_costs(volume_lb, collar, paths[:, -1])
    peaks = margin_peaks(contracts, paths, f0, put_strike)

    rows = []
    for strat in ("unhedged", "futures", "collar"):
        s = costs[strat]
        rows.append(
            {
                "strategy": strat,
                "cost_mean": float(s.mean()),
                "cost_p95": float(np.quantile(s, 0.95)),
                "cost_worst": float(s.max()),
                "margin_peak_mean": peaks.get(f"{strat}_mean", 0.0),
                "margin_peak_p95": peaks.get(f"{strat}_p95", 0.0),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["collar"] = collar
    return out


def decision_from_frozen(
    volume_lb: float,
    contracts: float,
    put_strike: float,
    t: float = 0.25,
    n_paths: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Wire the frozen curve front price, GARCH vol and SOFR into
    decision_table — the single call the report and app make."""
    curve = load_curve("coffee")
    _, fit = vol_from_frozen()
    from hedging_workbench.data.frozen import latest_rate

    return decision_table(
        volume_lb=volume_lb,
        contracts=contracts,
        f0=float(curve["price"].iloc[0]),
        sigma=fit.garch_last / 100.0,
        r=latest_rate(),
        put_strike=put_strike,
        t=t,
        n_paths=n_paths,
        seed=seed,
    )
