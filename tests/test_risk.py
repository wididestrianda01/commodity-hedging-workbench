"""Seam 2 tests: historical VaR/ES (ticket 12-01).

Hand-computed on a constructed discrete distribution plus invariants.
"""

import pandas as pd
import pytest

from hedging_workbench.risk import filtered_var, var_es


def test_hand_computed_discrete_distribution():
    """P&L {-10, -4, +1, +5, +8} equally likely. pandas linear quantile:
    Q_0.05 = -10 + 0.2*6 = -8.8 -> tail {pnl <= -8.8} = {-10},
    so VaR95 = 8.8, ES95 = 10.0. VaR99: Q_0.01 = -9.76, same tail:
    VaR99 = 9.76, ES99 = 10.0.
    """
    pnl = pd.Series([-10.0, -4.0, 1.0, 5.0, 8.0])
    out = var_es(pnl, levels=(0.95, 0.99))
    assert out.loc[0.95, "var"] == pytest.approx(8.8)
    assert out.loc[0.95, "es"] == pytest.approx(10.0)
    assert out.loc[0.99, "var"] == pytest.approx(9.76)
    assert out.loc[0.99, "es"] == pytest.approx(10.0)


def test_es_strictly_exceeds_var_when_tail_is_fat():
    """A distribution with a fat tail: ES must strictly exceed VaR."""
    pnl = pd.Series([-100.0, -20.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    out = var_es(pnl, levels=(0.80,))
    assert out.loc[0.80, "es"] > out.loc[0.80, "var"]
    # hand check: Q_0.20 = -20*0.8 + (-1)*0.2? quantile(0.2) with n=10 ->
    # idx 0.2*9=1.8 -> -20 + 0.8*(-1+20) = -4.8; tail <= -4.8 = {-100, -20}
    assert out.loc[0.80, "var"] == pytest.approx(4.8)
    assert out.loc[0.80, "es"] == pytest.approx(60.0)


def test_es_ge_var_invariant_on_real_data():
    """Frozen KC=F daily returns: invariant holds, numbers plausible."""
    from hedging_workbench.vol import vol_from_frozen

    rets, _ = vol_from_frozen()
    out = var_es(rets / 100.0)
    assert (out["es"] >= out["var"]).all()
    # 99% daily ES on coffee should be a loss of a few %, not 50%
    assert 0.001 < out.loc[0.99, "es"] < 0.50


def test_empty_and_bad_level_fail_loud():
    with pytest.raises(ValueError):
        var_es(pd.Series(dtype=float))
    with pytest.raises(ValueError):
        var_es(pd.Series([1.0, -1.0]), levels=(1.0,))


def test_filtered_var_hand_computed():
    """Prices [100, 101, 103, 102] → returns [1%, 1.9802%, −0.9709%].
    No look-ahead: VaR at day t uses the EWMA variance over returns
    ≤ t−1 and price(t−1). At the LAST day (t4) the usable variance is
    over {1%, 1.9802%}: weights 0.06/1.00, var = 0.054381/0.11321
    = 0.48039 %² → sd 0.69310 %/day.
    VaR(95) = 1.6449 × 0.0069310 × price(t−1)=103 × 1000 = 1,174.26.
    First three days have no usable lagged variance → NaN."""
    px = pd.Series([100.0, 101.0, 103.0, 102.0], index=pd.date_range("2026-01-01", periods=4, freq="B"))
    out = filtered_var(px, multiplier=1_000.0, level=0.95)
    assert out.iloc[:3].isna().all()
    assert out.iloc[-1] == pytest.approx(1_174.26, rel=1e-3)


def test_filtered_var_no_lookahead():
    """A future outlier must not move earlier VaR values."""
    px = pd.Series(
        [100.0, 101.0, 103.0, 102.0, 300.0], index=pd.date_range("2026-01-01", periods=5, freq="B")
    )
    out = filtered_var(px, multiplier=1_000.0)
    base = filtered_var(px.iloc[:4], multiplier=1_000.0)
    assert out.iloc[:4].dropna().tolist() == pytest.approx(base.dropna().tolist())
    with pytest.raises(ValueError):
        filtered_var(px, level=1.0)
