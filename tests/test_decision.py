"""Seam 2 tests: hedged vs unhedged client outcome (13-02).

Hand-computed on a synthetic collar (zero premium gap), plus a seeded
smoke run of the full decision table.
"""

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.decision import decision_table, margin_peaks, net_costs
from hedging_workbench.pricing import Collar


def _collar() -> Collar:
    return Collar(
        f0=320.0,
        put_strike=280.0,
        call_strike=350.0,
        put_premium=1.0,
        call_premium=1.0,  # gap = 0: zero-cost by construction
        t=0.25,
        sigma=0.4,
        r=0.04,
    )


VOL_LB = 100_000.0  # $1000 per cent/lb


def test_net_costs_hand_computed():
    """volume 100k lb, f0=320, put 280, call 350, zero gap:
    futures net cost locked at 320k for every scenario.
    Collar net cost: 240k below put (f_T−put_k+f0), 320k between,
    350k at f_T=380 (capped at call). Unhedged = volume×f_T.
    """
    f = np.array([200.0, 300.0, 350.0, 380.0])
    out = net_costs(VOL_LB, _collar(), f)
    assert out["unhedged"].tolist() == pytest.approx(
        [200_000.0, 300_000.0, 350_000.0, 380_000.0]
    )
    assert out["futures"].tolist() == pytest.approx([320_000.0] * 4)
    assert out["collar"].tolist() == pytest.approx(
        [240_000.0, 320_000.0, 320_000.0, 350_000.0]
    )


def test_margin_peaks_hand_computed():
    """contracts=2 → $750/cent. Path mins 300 and 200 → futures drawdowns
    20×750=15k and 120×750=90k. Collar funding loss capped at the put
    protection (320−280)×750=30k → capped drawdowns [15k, 30k]."""
    paths = np.array([[320.0, 340.0, 300.0], [320.0, 310.0, 200.0]])
    out = margin_peaks(2.0, paths, f0=320.0, put_strike=280.0)
    assert out["futures_mean"] == pytest.approx(52_500.0)
    assert out["futures_p95"] == pytest.approx(86_250.0)
    assert out["collar_mean"] == pytest.approx(22_500.0)
    assert out["collar_p95"] == pytest.approx(29_250.0)


def test_decision_table_strategy_structure():
    f0 = 320.0
    tbl = decision_table(
        volume_lb=VOL_LB,
        contracts=2.0,
        f0=f0,
        sigma=0.385,
        r=0.0366,
        put_strike=280.0,
        n_paths=2_000,
        steps=63,
        seed=42,
    )
    by = tbl.set_index("strategy")
    # futures locks the price: mean AND worst equal volume × f0
    assert by.loc["futures", "cost_mean"] == pytest.approx(VOL_LB * f0 / 100.0)
    assert by.loc["futures", "cost_worst"] == pytest.approx(VOL_LB * f0 / 100.0)
    # above the call the collar is unhedged again: p95 sits exactly one
    # (call_k − f0) below the unhedged p95 (p95 f_T is in the call region)
    call_k = tbl.attrs["collar"].call_strike
    usd = VOL_LB / 100.0
    assert by.loc["collar", "cost_p95"] == pytest.approx(
        by.loc["unhedged", "cost_p95"] - (call_k - f0) * usd, rel=1e-6
    )
    # margin: futures peak ≥ collar peak (put floors the funding loss)
    assert by.loc["futures", "margin_peak_p95"] >= by.loc["collar", "margin_peak_p95"]
