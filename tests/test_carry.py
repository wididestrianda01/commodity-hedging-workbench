"""Seam 2 tests: cost-of-carry identity, curve state, real frozen curve."""

import math

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.carry import (
    curve_state,
    expiry,
    implied_carry,
    implied_yield,
    load_curve,
    yield_term_structure,
)
from hedging_workbench.data.frozen import latest_rate


def synth_curve(prices, start_year=2026):
    """Synthetic chain as a curve DataFrame (expiry, price, ttm)."""
    expiries = pd.date_range(
        f"{start_year}-09-15", periods=len(prices), freq="MS"
    ) + pd.Timedelta(days=14)
    ttm = (expiries - expiries[0]).days / 365.25
    return pd.DataFrame(
        {
            "price": prices,
            "expiry": expiries,
            "ttm": ttm,
            "symbol": [f"C{i}" for i in range(len(prices))],
            "label": [f"m{i}" for i in range(len(prices))],
        }
    )


def test_carry_identity_exact():
    """F2 = F1*exp((r + d - y)*tau) round-trips exactly through the API."""
    f1, f2, tau, r, d = 100.0, 102.0, 0.5, 0.03, 0.01
    y = implied_yield(f1, f2, tau, r, storage=d)
    c = implied_carry(f1, f2, tau)
    assert c == pytest.approx(math.log(1.02) / 0.5)
    assert y == pytest.approx(r + d - c)
    assert f1 * math.exp((r + d - y) * tau) == pytest.approx(f2)


def test_curve_state_labels():
    up = yield_term_structure(synth_curve([100, 101, 102]), r=0.03)
    down = yield_term_structure(synth_curve([102, 101, 100]), r=0.03)
    mixed = yield_term_structure(synth_curve([100, 101, 100]), r=0.03)
    assert curve_state(up) == "contango"
    assert curve_state(down) == "backwardation"
    assert curve_state(mixed) == "mixed"


def test_real_coffee_curve_backwardated():
    """Frozen coffee curve: 8 contracts, monotonic expiries, negative carry
    direction (backwardated, convenience yield above the rate)."""
    curve = load_curve("coffee")
    assert len(curve) == 8
    assert curve["expiry"].is_monotonic_increasing
    assert curve["ttm"].iloc[0] > 0
    ts = yield_term_structure(curve, r=latest_rate())
    assert len(ts) == 7
    assert ts["carry"].mean() < 0  # phase-1 finding: backwardation
    assert (ts["implied_yield"] > ts.attrs["r"]).all()
    assert curve_state(ts) == "backwardation"


def test_gold_curve_loads():
    """Fallback universe produces a valid curve."""
    curve = load_curve("gold")
    assert len(curve) == 5
    ts = yield_term_structure(curve, r=0.03)
    assert np.isfinite(ts["carry"]).all()


def test_expiry_mapping():
    assert expiry("KCH27.NYB") == pd.Timestamp(2027, 3, 15)
    assert expiry("KCZ26.NYB") == pd.Timestamp(2026, 12, 15)
    assert expiry("KC=F") is None
