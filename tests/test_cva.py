"""Seam 2 tests: unilateral CVA + QuantLib benchmark (12-03, 12-04).

Hand-computed closed forms on constructed cases; the benchmark re-derives
the same discretization from QuantLib's own curves.
"""

import math

import numpy as np
import pytest

from hedging_workbench.cva import cva_unilateral, discount, survival

ql = pytest.importorskip("QuantLib", reason="QuantLib not installed")

HAZ, LGD, R = 0.1, 0.6, 0.04


def test_single_bucket_hand_value():
    """EE=c over [0,1]: CVA = LGD * c * (1-e^{-lam}) * e^{-r}."""
    c = 1_000_000.0
    hand = LGD * c * (1 - math.exp(-HAZ)) * math.exp(-R)
    got = cva_unilateral([c], [1.0], HAZ, LGD, R)
    assert got == pytest.approx(hand, rel=1e-12)


def test_two_bucket_hand_value():
    """Constant EE, two buckets — numbers worked out by hand:

    dS1 = 1 - e^{-0.05} = 0.04877058, df1 = e^{-0.02} = 0.98019867
    dS2 = e^{-0.05} - e^{-0.10} = 0.04639219, df2 = e^{-0.04} = 0.96078944
    CVA = 0.6 * 1e6 * (0.04780387 + 0.04457460) = 55,427.08
    """
    got = cva_unilateral([1e6, 1e6], [0.5, 1.0], HAZ, LGD, R)
    assert got == pytest.approx(55_427.08, rel=1e-4)


def test_conventions_survival_and_discount():
    assert survival(2.0, HAZ) == pytest.approx(math.exp(-0.2))
    assert discount(2.0, R) == pytest.approx(math.exp(-0.08))
    with pytest.raises(ValueError):
        cva_unilateral([1.0, 2.0], [1.0, 0.5], HAZ, LGD, R)  # not increasing
    with pytest.raises(ValueError):
        cva_unilateral([1.0], [], HAZ, LGD, R)


def test_ql_benchmark_matched_case():
    """Same EE grid re-priced with QuantLib's FlatHazardRate survival and
    FlatForward discount. Identical discretization -> tolerance is float
    noise only (1e-9 relative), pre-declared in the module docstring.
    """
    ref = ql.Date(15, ql.January, 2026)
    ql.Settings.instance().evaluationDate = ref
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    hazard_curve = ql.FlatHazardRate(0, cal, ql.QuoteHandle(ql.SimpleQuote(HAZ)), dc)
    df_curve = ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(R)), dc)

    tenors = np.array([0.25, 0.5, 1.0, 2.0])
    ee = np.array([80_000.0, 95_000.0, 110_000.0, 90_000.0])

    # component check: QuantLib's own S(t) and df(t) match our closed forms
    s_ql = np.array([hazard_curve.survivalProbability(t) for t in tenors])
    df_ql = np.array([df_curve.discount(t) for t in tenors])
    assert np.allclose(s_ql, survival(tenors, HAZ), rtol=1e-9)
    assert np.allclose(df_ql, discount(tenors, R), rtol=1e-9)

    # assembled benchmark: same discretization, QuantLib inputs
    s_prev = np.concatenate([[1.0], s_ql[:-1]])
    ql_cva = LGD * np.sum(ee * df_ql * (s_prev - s_ql))
    ours = cva_unilateral(ee, tenors, HAZ, LGD, R)
    assert ours == pytest.approx(ql_cva, rel=1e-9)
