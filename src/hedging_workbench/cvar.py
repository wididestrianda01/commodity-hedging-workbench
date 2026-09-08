"""Discrete-CVaR optimal hedge ratio (ticket 10-09).

Method: MDPI Mathematics (2025) industrial hedging — pick the hedge ratio h
from a discrete grid minimising the CVaR of the hedged position's losses.

A coffee BUYER's net cost move per period is
    loss = Δspot - h · Δfutures        (long futures offsets price rises)
CVaR at level alpha = mean of the worst (1 - alpha) loss tail. Minimising
CVaR over a grid is transparent and hand-verifiable — no LP solver needed
for a 1-D search (choice documented).

Objective differs from the Phase 3 program (cash-flow smoothing via roll
schedule + collar); the comparison table states that, so the verdict is
evidence-based, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = np.round(np.linspace(0.0, 2.0, 201), 4)


@dataclass
class CvarResult:
    ratio: float  # optimal h (contracts per unit of exposure)
    cvar: float  # CVaR at the optimum
    cvar_unhedged: float  # CVaR at h = 0
    alpha: float


def cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Mean of the worst (1 - alpha) tail of losses."""
    l = np.sort(np.asarray(losses, dtype=float))
    n_tail = max(int(np.ceil(len(l) * (1 - alpha) - 1e-9)), 1)
    return float(l[-n_tail:].mean())


def cvar_optimal_ratio(
    item_moves, futures_moves, alpha: float = 0.95, grid=GRID
) -> CvarResult:
    """Grid search h minimising CVaR of (Δitem - h·Δfutures) losses."""
    d_item = np.asarray(item_moves, dtype=float)
    d_fut = np.asarray(futures_moves, dtype=float)
    best = min(grid, key=lambda h: cvar(d_item - h * d_fut, alpha))
    return CvarResult(
        ratio=float(best),
        cvar=cvar(d_item - best * d_fut, alpha),
        cvar_unhedged=cvar(d_item, alpha),
        alpha=alpha,
    )
