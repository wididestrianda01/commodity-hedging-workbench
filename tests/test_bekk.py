"""Seam 2 tests: GARCH-BEKK(1,1), variance-targeting fit (ticket 10-10)."""

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.bekk import fit_bekk, simulate_bekk

A_TRUE = np.array([[0.15, 0.02], [0.03, 0.12]])
B_TRUE = np.array([[0.75, 0.04], [0.02, 0.80]])
C_TRUE = np.array([[0.9, 0.0], [0.1, 0.7]])

def test_synthetic_recovery():
    """On simulated BEKK data the average conditional covariance tracks the
    realized second moments, and the conditional correlation stays in [-1, 1]."""
    sim = simulate_bekk(A_TRUE, B_TRUE, C_TRUE, T=4000, seed=42)
    fit = fit_bekk(pd.DataFrame(sim))
    assert fit.stationary
    realized = np.cov(sim.T)
    avg_H = fit.H.mean(axis=0)
    assert avg_H[0, 0] == pytest.approx(realized[0, 0], rel=0.25)
    assert avg_H[1, 1] == pytest.approx(realized[1, 1], rel=0.25)
    assert avg_H[0, 1] == pytest.approx(realized[0, 1], rel=0.35)
    corr_t = fit.H[:, 0, 1] / np.sqrt(fit.H[:, 0, 0] * fit.H[:, 1, 1])
    assert np.all(np.abs(corr_t) <= 1.0 + 1e-9)

def test_hedge_ratio_sane():
    """Average ratio recovers the true unconditional ratio (S12/S22);
    ratio is finite and does not explode period to period. Sign flips are
    legitimate when H12 hovers near zero (small true cross-dynamics)."""
    sim = simulate_bekk(A_TRUE, B_TRUE, C_TRUE, T=2500, seed=1)
    fit = fit_bekk(pd.DataFrame(sim))
    h = fit.hedge_ratio()
    assert fit.stationary
    assert np.isfinite(h).all()
    true_ratio = np.cov(sim.T)[0, 1] / np.cov(sim.T)[1, 1]
    assert h.mean() == pytest.approx(true_ratio, abs=0.25)
    assert h.std() / h.mean() < 1.0

def test_real_frozen_returns_fit():
    """KC=F (item) vs KCZ26.NYB (futures): converges stationary, ratio > 0."""
    from hedging_workbench.data.download import load_frozen
    kc = load_frozen(["KC=F"])["KC=F"].dropna()
    kcz = load_frozen(["KCZ26.NYB"])["KCZ26.NYB"].dropna()
    j = kc.to_frame("kc").join(kcz.rename("kcz"), how="inner").pct_change().dropna()
    fit = fit_bekk(j)
    assert fit.stationary
    h = fit.hedge_ratio()
    assert 0.0 < h.mean() < 5.0
