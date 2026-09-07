"""Seam 2 tests: Black-76 + zero-cost collar (ticket 10-06).

Expectations are identities and hand-calculated numbers:
- put-call parity on futures: c - p = e^{-rT}(F - K), exact
- zero-cost identity: put premium = call premium
- collar payoff bounded by [-(f0-put), call-f0] in cents/lb
"""

import math

import numpy as np
import pytest

from hedging_workbench.pricing import (black76, collar_vs_futures,
                                       margin_relief, zero_cost_collar)

F0, R, T, SIG = 300.0, 0.0366, 0.25, 0.385

def test_black76_parity_identity():
    """c - p = e^{-rT}(F-K) must hold at every strike."""
    for k in (260.0, 280.0, 300.0, 320.0, 350.0):
        c = black76("call", F0, k, T, SIG, R)
        p = black76("put", F0, k, T, SIG, R)
        assert c - p == pytest.approx(math.exp(-R * T) * (F0 - k), abs=1e-9)

def test_black76_t0_intrinsic():
    """No time left -> premiums are intrinsic values."""
    assert black76("call", 300, 280, 0.0, SIG, R) == pytest.approx(20.0)
    assert black76("put", 300, 320, 0.0, SIG, R) == pytest.approx(20.0)

def test_zero_cost_identity():
    collar = zero_cost_collar(F0, put_strike=280.0, t=T, sigma=SIG, r=R)
    assert collar.premium_gap < 1e-8
    assert collar.put_strike < collar.f0 < collar.call_strike

def test_collar_payoff_bounded():
    collar = zero_cost_collar(F0, put_strike=280.0, t=T, sigma=SIG, r=R)
    floor, cap = -(F0 - collar.put_strike), collar.call_strike - F0
    grid = np.linspace(200.0, 400.0, 41)
    payoffs = [collar.payoff(f) for f in grid]
    assert min(payoffs) == pytest.approx(floor, abs=1e-9)
    assert max(payoffs) == pytest.approx(cap, abs=1e-9)

def test_collar_vs_futures_hand_calculated():
    """3,750,000 lb -> $37,500 per cent. At 260: futures -1.5M, collar -0.75M."""
    collar = zero_cost_collar(F0, put_strike=280.0, t=T, sigma=SIG, r=R)
    out = collar_vs_futures(3_750_000, collar, f_terminal=260.0)
    assert out["futures_usd"] == pytest.approx(-40.0 * 37_500)
    assert out["collar_usd"] == pytest.approx(-20.0 * 37_500)

def test_margin_relief_direction():
    collar = zero_cost_collar(F0, put_strike=280.0, t=T, sigma=SIG, r=R)
    out = margin_relief(3_750_000, collar, shock_floor=240.0)
    assert out["futures_worst_usd"] == pytest.approx(60.0 * 37_500)
    assert out["collar_worst_usd"] == pytest.approx(20.0 * 37_500)
    assert out["collar_worst_usd"] < out["futures_worst_usd"]

def test_real_frozen_inputs():
    """Collar prices on the real frozen front price + GARCH vol, converges."""
    from hedging_workbench.carry import load_curve
    from hedging_workbench.vol import vol_from_frozen
    curve = load_curve("coffee")
    rets, fit = vol_from_frozen()
    sigma = fit.garch_last / 100.0
    collar = zero_cost_collar(float(curve["price"].iloc[0]),
                              put_strike=280.0, t=0.25, sigma=sigma, r=R)
    assert collar.premium_gap < 1e-8
    assert collar.call_strike > float(curve["price"].iloc[0])
