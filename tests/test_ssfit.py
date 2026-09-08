"""Seam 2 tests: Schwartz–Smith synthetic recovery + real frozen fit."""

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.ssfit import fit_curve, fit_from_frozen


def ss_curve(prices_exp, ttms):
    """Curve DataFrame from model log-forward values."""
    return pd.DataFrame({"price": np.exp(prices_exp), "ttm": ttms})


def test_synthetic_recovery():
    """A curve generated from known parameters refits to them (exact, no noise)."""
    ttms = np.array([0.03, 0.28, 0.53, 0.78, 1.03, 1.28, 1.53, 1.78])
    true = dict(chi=0.08, xi=5.70, kappa=2.0, slope=-0.09)
    ln_f = (
        true["chi"] * np.exp(-true["kappa"] * ttms) + true["xi"] + true["slope"] * ttms
    )
    fit = fit_curve(ss_curve(ln_f, ttms))
    assert fit.converged
    assert fit.chi == pytest.approx(true["chi"], abs=1e-5)
    assert fit.xi == pytest.approx(true["xi"], abs=1e-5)
    assert fit.kappa == pytest.approx(true["kappa"], rel=1e-3)
    assert fit.slope == pytest.approx(true["slope"], abs=1e-5)
    assert fit.half_life == pytest.approx(np.log(2) / true["kappa"], rel=1e-3)


def test_real_fit_converges_and_is_backwardated():
    """Frozen coffee fit: converges, small log-space residuals, negative drift,
    positive long-run price; thin identification surfaced as wide kappa SE."""
    curve, fit, ts = fit_from_frozen("coffee")
    assert fit.converged
    assert fit.rmse < 0.05  # <5% log misfit on 8 points
    assert fit.slope < 0  # backwardation drift
    assert 0 < fit.long_run_price < 1e4  # cents/lb sanity
    assert fit.se["kappa"] > 0  # uncertainty reported
    assert len(curve) == 8 and len(ts) == 7


def test_half_life_positive_and_finite():
    curve, fit, _ = fit_from_frozen("coffee")
    assert np.isfinite(fit.half_life) and fit.half_life > 0
