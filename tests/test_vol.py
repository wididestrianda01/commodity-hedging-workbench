"""Seam 2 tests: vol layer — EWMA hand recursion, GARCH identity."""

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.vol import EWMA_LAMBDA, garch_vol, vol_from_frozen


def test_ewma_hand_recursion():
    """EWMA variance = the manual RiskMetrics recursion, to the last value."""
    r = pd.Series([1.0, -0.5, 2.0, 0.3, -1.2])
    lam = EWMA_LAMBDA
    manual = r.iloc[0] ** 2
    for x in r.iloc[1:]:
        manual = lam * manual + (1 - lam) * x ** 2
    ewma = float(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])
    assert ewma == pytest.approx(manual, rel=1e-10)


def test_garch_identity_and_plausibility():
    """Fitted GARCH long-run vol follows omega/(1-alpha-beta) annualised and
    is same order as the sample vol on the frozen returns."""
    rets, fit = vol_from_frozen()
    assert 0 < fit.persistence < 1
    expected = np.sqrt(252 * fit.omega / (1 - fit.persistence))
    assert fit.garch_longrun == pytest.approx(expected, rel=1e-12)
    sample_annual = np.sqrt(252 * rets.var())
    assert 0.25 * sample_annual < fit.garch_longrun < 4 * sample_annual
    assert fit.garch_last > 0
