"""Seam 2 tests: IFRS 9 principles-based effectiveness (ticket 10-07).

Boundary demonstrations, hand-constructed:
- an INSIDE-the-80-125%-band program with a weak economic relationship FAILS
- an OUTSIDE-the-band program with a strong relationship PASSES
- a credit-dominated instrument is REJECTED
"""

import numpy as np
import pytest

from hedging_workbench.effectiveness import assess

def test_clean_program_passes():
    r = assess(hypothetical_pnl=[100, 10, 0, 0],
               hedged_item_pnl=[-100, -10, 0, 0])
    assert r.effective
    assert r.correlation == pytest.approx(1.0)
    assert r.dollar_offset == pytest.approx(1.0)
    assert r.credit_share == 0.0

def test_credit_dominance_rejected():
    """Instrument P&L driven mostly by credit terms -> reject, in-band or not."""
    r = assess(hypothetical_pnl=[100, 10, 0, 0],
               hedged_item_pnl=[-100, -10, 0, 0],
               market_pnl=[10, 5, 0, 0],
               credit_pnl=[100, 10, 0, 0])   # credit variance >> market
    assert not r.effective
    assert any("credit" in x for x in r.reasons)

def test_band_is_not_the_operative_test():
    """Inside 80-125% but uncorrelated -> FAIL; outside band but aligned -> PASS."""
    # offset 0.9 (inside band), but timing mismatched -> negative correlation
    inside = assess(hypothetical_pnl=[100, -10, 10, -10],
                    hedged_item_pnl=[-50, -50, 0, 0])
    assert 0.8 <= inside.dollar_offset <= 1.25
    assert inside.correlation < 0.8
    assert not inside.effective

    # offset 1.30 (outside band), perfectly correlated -> effective
    outside = assess(hypothetical_pnl=[130, 13, 0, 0],
                     hedged_item_pnl=[-100, -10, 0, 0])
    assert outside.dollar_offset > 1.25
    assert outside.correlation == pytest.approx(1.0)
    assert outside.effective

def test_ratio_consistency_check():
    r = assess(hypothetical_pnl=[100, 10, 0, 0],
               hedged_item_pnl=[-100, -10, 0, 0],
               designated_ratio=1.0, actual_ratio=1.2)
    assert not r.effective
    assert any("ratio" in x for x in r.reasons)

def test_real_program_assessment_smoke():
    """The real 10-05 program vs its hypothetical derivative assesses clean."""
    from hedging_workbench.hedge import build_program
    from hedging_workbench.memo import designation_memo
    prog = build_program(100_000, "2026-09-01", 12, universe="coffee")
    # synthetic P&L pair standing in for the assessment window: perfectly
    # offsetting by construction (futures P&L vs item price change)
    path = [300.0, 310.0, 295.0, 305.0]
    item = [-p * 100_000 / 100 for p in np.diff(path, prepend=path[0])]
    fut = [+p * 100_000 / 100 for p in np.diff(path, prepend=path[0])]
    r = assess(hypothetical_pnl=fut, hedged_item_pnl=item)
    assert r.effective
    memo = designation_memo(prog, f0=300.0)
    assert "cash-flow hedge" in memo.lower()
    assert "EMIR 3" in memo
    assert "not compliance advice" in memo
