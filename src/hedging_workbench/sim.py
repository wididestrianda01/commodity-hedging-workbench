"""Lognormal martingale paths for futures prices (deterministic rates).

Under Q with zero-cost carry a futures price is a martingale, so paths
walk

    F_{t+dt} = F_t * exp(-0.5*sigma^2*dt + sigma*sqrt(dt)*Z)

One generator serves every simulated payoff in the workbench: the
exposure book needs the full path grid; terminal-only pricing (the
participation note) is steps=1. sigma is the Phase 2 GARCH working vol
(decimal/yr).
"""

from __future__ import annotations

import numpy as np


def martingale_paths(
    f0: float,
    sigma: float,
    horizon_years: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
    antithetic: bool = False,
) -> np.ndarray:
    """(n_paths, steps+1) grid, column 0 = f0; E[F_t] = f0 by construction.

    antithetic: n_paths is the TOTAL path count (n/2 mirrored pairs),
    for variance reduction on terminal or path-integral payoffs.
    """
    if steps < 1 or n_paths < 1:
        raise ValueError("need at least one step and path")
    if antithetic and (n_paths < 2 or n_paths % 2):
        raise ValueError("n_paths must be even and >= 2 for antithetic")
    rng = np.random.default_rng(seed)
    dt = horizon_years / steps
    if antithetic:
        half = n_paths // 2
        w = rng.standard_normal((half, steps))
        z = np.concatenate([w, -w], axis=0)
    else:
        z = rng.standard_normal((n_paths, steps))
    incr = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
    log = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return f0 * np.exp(log)
