"""Seam 2 tests: hedge program core (ticket 10-05).

Hand-calculated expectations, not implementation echoes:
- exposure -> contracts conversion is exact division by 37,500 lb
- selection rule picks the earliest chain expiry at/after month end
- VM flow = contracts * $375 * price diff (cents/lb path)
"""

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.carry import load_curve
from hedging_workbench.hedge import (CONTRACT_LB, DOLLARS_PER_CENT,
                                     build_program, exposure_schedule,
                                     roll_schedule, select_contracts,
                                     variation_margin)

def _synth_curve():
    """Two-contract chain: Dec 2026 and Mar 2027."""
    return pd.DataFrame({
        "symbol": ["KCZ26.NYB", "KCH27.NYB"],
        "label": ["Dec 2026", "Mar 2027"],
        "expiry": [pd.Timestamp(2026, 12, 15), pd.Timestamp(2027, 3, 15)],
        "price": [295.6, 287.4],
        "ttm": [0.28, 0.53],
    })

def test_exposure_contracts_exact():
    exp = exposure_schedule(37_500, "2026-10-01", 3)
    assert len(exp) == 3
    exp = select_contracts(exp, _synth_curve())
    assert np.allclose(exp["contracts"], 1.0)
    assert np.allclose(exp["volume_lb"], CONTRACT_LB)

def test_selection_rule_earliest_expiry_in_or_after_month():
    exp = select_contracts(
        exposure_schedule(37_500, "2026-10-01", 4), _synth_curve())
    # Oct/Nov/Dec 2026 -> Dec 2026 delivery contract; Jan 2027 -> Mar 2027
    assert (exp["symbol"].iloc[:3] == "KCZ26.NYB").all()
    assert exp["symbol"].iloc[3] == "KCH27.NYB"
    assert not exp["gap"].any()

def test_selection_gap_flagged_past_last_contract():
    exp = select_contracts(
        exposure_schedule(37_500, "2027-04-01", 2), _synth_curve())
    assert exp["gap"].all()
    assert (exp["symbol"] == "KCH27.NYB").all()   # last contract fallback

def test_roll_schedule_transitions_and_dates():
    exp = select_contracts(
        exposure_schedule(37_500, "2026-10-01", 6), _synth_curve())
    rolls = roll_schedule(exp)
    assert len(rolls) == 1
    r = rolls.iloc[0]
    assert r["front_symbol"] == "KCZ26.NYB"
    assert r["back_symbol"] == "KCH27.NYB"
    assert r["roll_date"] == pd.Timestamp(2026, 12, 5)   # expiry - 10 days
    assert r["roll_date"] < r["front_expiry"]

def test_variation_margin_hand_calculated():
    path = pd.Series([100.0, 102.0, 99.0],
                     index=pd.date_range("2026-10-05", periods=3))
    vm = variation_margin(contracts=2.0, price_path=path,
                          initial_margin=8_000.0)
    # flow = 2 contracts * $375/cent * cents diff
    assert vm["flow_usd"].iloc[0] == pytest.approx(0.0)
    assert vm["flow_usd"].iloc[1] == pytest.approx(2 * 375 * 2)      # +1500
    assert vm["flow_usd"].iloc[2] == pytest.approx(2 * 375 * (-3))   # -2250
    assert vm["balance_usd"].tolist() == pytest.approx(
        [16_000.0, 17_500.0, 15_250.0])

def test_variation_margin_fractional_contracts():
    """Fractional ratio scales flows linearly: 0.5 contract, +1 cent -> +$187.50."""
    path = pd.Series([300.0, 301.0])
    vm = variation_margin(0.5, path)
    assert vm["flow_usd"].iloc[1] == pytest.approx(DOLLARS_PER_CENT / 2)

def test_real_coffee_program():
    """Frozen coffee chain: monthly 100k-lb roaster, 12 months from Sep 2026."""
    prog = build_program(100_000, "2026-09-01", 12, universe="coffee")
    assert np.allclose(prog.exposure["contracts"], 100_000 / CONTRACT_LB)
    # every roll exits before its front expiry; each month hedged in-or-after
    assert (prog.rolls["roll_date"] < prog.rolls["front_expiry"]).all()
    assert (prog.exposure["expiry"] >= prog.exposure["month"]).all()
    assert not prog.exposure["gap"].any()   # 12 months fit the 8-contract chain

def test_gold_universe_same_code_path():
    prog = build_program(37_500, "2027-02-01", 4, universe="gold")
    assert len(prog.exposure) == 4
    assert np.isfinite(prog.exposure["contracts"]).all()
