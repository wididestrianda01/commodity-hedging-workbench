"""Seam 2 tests: margin-liquidity stress (ticket 10-08).

Hand-calculated on constructed paths: the peak of the margin profile must
equal the worst day computed by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.conventions import DOLLARS_PER_CENT
from hedging_workbench.pricing import zero_cost_collar
from hedging_workbench.stress import run_scenario, stress_report, stressed_path


def _base(last=300.0):
    return pd.Series([295.0, 297.0, last], index=pd.date_range("2026-09-01", periods=3))


def test_stressed_path_drift():
    """Day 0 anchors at the as-of mark; day 1 = 300*exp(-0.5/252)."""
    p = stressed_path(_base(), -0.5, horizon_days=1)
    assert p.iloc[0] == pytest.approx(300.0)
    assert p.iloc[1] == pytest.approx(300.0 * math.exp(-0.5 / 252))
    assert p.index[-1] > pd.Timestamp("2026-09-03")  # after last base date


def test_backwardation_widening_steepens_after_roll_window():
    """Back shock leaves the front path alone until back_start_day, then
    compounds a steeper fall (ticket 10-08: rolling into a sinking
    deferred leg). Also: run_scenario returns the full profiles.
    """
    base = _base()
    p = stressed_path(
        base, -0.5, horizon_days=40, back_shock_annual=-0.5, back_start_day=21
    )
    flat = stressed_path(base, -0.5, horizon_days=40)
    assert p.iloc[20] == pytest.approx(flat.iloc[20])
    assert p.iloc[-1] < flat.iloc[-1]
    r = run_scenario(2.0, base, -0.5, "down", horizon_days=5)
    assert list(r.profile.columns) == ["flow_usd", "balance_usd"]
    assert r.collar_profile is None


def test_peak_equals_hand_computed_worst_day():
    """2 contracts, down shock: every day loses k*375*2 vs prior day, so the
    peak drawdown is the last day — cumulative drift k*t/252."""
    contracts, shock = 2.0, -0.5
    base = _base(300.0)
    path = stressed_path(base, shock, horizon_days=3)
    r = run_scenario(contracts, base, shock, "down", horizon_days=3)
    expected_last = DOLLARS_PER_CENT * contracts * (300.0 - path.iloc[-1])
    assert r.peak_need_usd == pytest.approx(expected_last, rel=1e-9)
    assert r.peak_date == path.index[-1]


def test_collar_caps_funding_on_downside():
    """Shock far past the put floor: collar peak = (f0-put)*usd, < futures peak."""
    contracts = 4.0
    base = _base(300.0)
    collar = zero_cost_collar(300.0, put_strike=280.0, t=0.25, sigma=0.385, r=0.0366)
    r = run_scenario(contracts, base, -1.5, "backwardation_widening", collar=collar)
    assert r.collar_peak_need_usd == pytest.approx(
        (300.0 - 280.0) * DOLLARS_PER_CENT * contracts, rel=1e-6
    )
    assert r.collar_peak_need_usd < r.peak_need_usd


def test_up_scenario_no_margin_call():
    """Long futures under a price RISE: balance only grows, peak need 0."""
    base = _base(300.0)
    r = run_scenario(2.0, base, +0.5, "up")
    assert r.peak_need_usd == pytest.approx(0.0, abs=1e-9)


def test_report_scenario_set_and_buffer():
    base = _base(300.0)
    out = stress_report(2.666, base, 0.385, collateral_usd=500_000.0)
    names = [r.scenario for r in out]
    assert names == [
        "down_1sigma",
        "down_2sigma",
        "up_2sigma",
        "backwardation_widening",
    ]
    assert all(r.buffer_usd is not None for r in out)
    assert out[1].peak_need_usd > out[0].peak_need_usd  # -2s worse than -1s


def test_real_frozen_inputs_smoke():
    """Full report on the frozen path with GARCH vol and real program size."""
    from hedging_workbench.data.frozen import load
    from hedging_workbench.vol import vol_from_frozen

    _, fit = vol_from_frozen()
    kc = load(["KC=F"])["KC=F"].dropna()
    out = stress_report(2.6667, kc, fit.garch_last / 100.0, collateral_usd=37.9e6)
    assert len(out) == 4
    assert all(np.isfinite(r.peak_need_usd) for r in out)
