"""Seam 2 tests: discrete-CVaR hedge ratio (ticket 10-09)."""

import numpy as np
import pytest

from hedging_workbench.cvar import GRID, cvar, cvar_optimal_ratio

def test_cvar_tail_mean_hand_calculated():
    """Worst 5% of 100 sorted losses = worst 5 observations."""
    losses = np.arange(100, dtype=float)
    assert cvar(losses, 0.95) == pytest.approx(np.arange(95, 100).mean())

def test_perfect_hedge_recovers_ratio_one():
    """item == futures -> zero-variance portfolio at h = 1 exactly."""
    rng = np.random.default_rng(7)
    moves = rng.normal(0, 1.0, 2000)
    r = cvar_optimal_ratio(moves, moves)
    assert r.ratio == pytest.approx(1.0)
    assert r.cvar == pytest.approx(0.0, abs=1e-12)

def test_bounds_and_improvement():
    """Ratio stays on the 0-2 grid; hedging must not worsen the objective."""
    rng = np.random.default_rng(11)
    fut = rng.normal(0, 1.0, 3000)
    item = fut * 1.4 + rng.normal(0, 0.2, 3000)   # noisy over-reacting item
    r = cvar_optimal_ratio(item, fut)
    assert 0.0 <= r.ratio <= 2.0
    assert r.cvar <= r.cvar_unhedged + 1e-12
    assert GRID[0] == 0.0 and GRID[-1] == 2.0

def test_real_frozen_returns_smoke():
    """Self-hedge on frozen KC=F returns -> h = 1; cross-contract ratio sane."""
    from hedging_workbench.data.download import load_frozen
    kc = load_frozen(["KC=F"])["KC=F"].dropna()
    kcz = load_frozen(["KCZ26.NYB"])["KCZ26.NYB"].dropna()
    j = (kc.to_frame("kc")).join(kcz.rename("kcz"), how="inner").pct_change().dropna()
    r_self = cvar_optimal_ratio(j["kc"], j["kc"])
    assert r_self.ratio == pytest.approx(1.0)
    r_cross = cvar_optimal_ratio(j["kc"], j["kcz"])
    assert 0.0 < r_cross.ratio < 2.0
    assert r_cross.cvar < r_cross.cvar_unhedged
