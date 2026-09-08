"""Seam 2 tests: SR 26-2 validation harness (13-01).

Hand-computed expectations on constructed cases; the frozen-data
backtests assert structure and pre-declared sanity bands. QuantLib is
skipped only where actually needed (the CVA benchmark test).
"""

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.carry import curve_state, yield_term_structure
from hedging_workbench.data.universe import UNIVERSES
from hedging_workbench.validate import (
    LookaheadError,
    asof_value,
    assert_no_lookahead,
    curve_stability_backtest,
    cva_ql_benchmark,
    ewma_forecast,
    model_inventory,
    note_mc_check,
    rolling_vol_backtest,
    validation_findings,
    var_coverage_backtest,
)


# -- no-look-ahead gate -------------------------------------------------------


def _dated(values, start="2026-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


def test_gate_rejects_post_asof_observation():
    s = _dated([1.0, 2.0, 3.0])
    assert_no_lookahead(s, s.index[-1])  # as-of = last observation: clean
    with pytest.raises(LookaheadError, match="after as-of"):
        assert_no_lookahead(s, s.index[-2])  # last obs poisons the window


def test_gate_rejects_all_nan_window():
    """All-NaN window at/below as-of: gate passes, asof_value refuses."""
    s = pd.Series([np.nan, np.nan], index=pd.date_range("2026-01-01", periods=2, freq="B"))
    with pytest.raises(LookaheadError, match="no data"):
        asof_value(s, s.index[-1])


def test_gate_empty_series_fails_closed():
    with pytest.raises(LookaheadError, match="cannot gate"):
        assert_no_lookahead(pd.Series(dtype=float), pd.Timestamp("2026-01-01"))


# -- EWMA forecast: independent recomputation from the definition -------------


def test_ewma_forecast_hand_computed():
    """pandas ewm(adjust=True) variance = Σw(x−x̄)²/(Σw − Σw²/Σw) with
    w_t = (1−α)^(T−t). Recomputed here explicitly, not via pandas."""
    x = np.array([1.0, 2.0, 3.0])
    alpha = 0.06  # lambda = 0.94
    w = (1 - alpha) ** np.arange(len(x) - 1, -1, -1)  # 1, 0.94, 0.8836
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / (np.sum(w) - np.sum(w**2) / np.sum(w))
    expected = np.sqrt(252 * var)

    s = _dated(x)
    got = ewma_forecast(s)
    assert got == pytest.approx(expected, rel=1e-12)
    # hand number: var ≈ 0.9993624, annualised ≈ 15.869 %/yr
    assert got == pytest.approx(15.869, rel=1e-3)


# -- rolling vol backtest ------------------------------------------------------


def test_rolling_vol_backtest_structure():
    rng = np.random.default_rng(7)
    n = 3 * 126 + 10
    rets = pd.Series(
        rng.normal(0, 1.2, n), index=pd.date_range("2025-01-01", periods=n, freq="B")
    )
    out = rolling_vol_backtest(rets * 100, window=126)
    assert list(out.columns[:5]) == [
        "fit_end", "realized", "garch", "garch_longrun", "ewma",
    ]
    assert len(out) == 2  # 3 windows: fit(0-126)+real(126-252), fit(126-252)+real(252-378)
    for m in ("garch", "ewma"):
        assert out[f"err_{m}"].notna().all()
    assert set(out.attrs["rmse"]) == {"garch", "ewma"}
    # EWMA column must equal the standalone forecast on the same fit window
    w = rets.iloc[126:252] * 100
    assert out["ewma"].iloc[1] == pytest.approx(ewma_forecast(w))


def test_rolling_vol_backtest_needs_two_windows():
    with pytest.raises(ValueError, match="returns"):
        rolling_vol_backtest(_dated(np.zeros(200)), window=126)


# -- VaR coverage: fully hand-computed -----------------------------------------


def test_var_coverage_hand_computed():
    """Two 5-point windows at level 0.8, worked by hand:
    W1 hist [-5..-1]: VaR = |q(0.2)| = 4.2; next [-2..2] -> 0 breaches.
    W2 hist [-2..2]:  VaR = |q(0.2)| = 1.2; next [-10,1,1,1,1] -> 1 breach.
    Expected breaches per window = 0.2*5 = 1.
    """
    pnl = np.array([-5.0, -4, -3, -2, -1, -2, -1, 0, 1, 2, -10, 1, 1, 1, 1])
    idx = pd.date_range("2026-01-01", periods=len(pnl), freq="B")
    out = var_coverage_backtest(pd.Series(pnl, index=idx), level=0.8, window=5)
    assert len(out) == 2
    assert out["var"].tolist() == pytest.approx([4.2, 1.2])
    assert out["breaches"].tolist() == [0, 1]
    assert out["expected"].tolist() == pytest.approx([1.0, 1.0])


def test_var_coverage_needs_two_windows():
    with pytest.raises(ValueError, match="P&L"):
        var_coverage_backtest(_dated(np.zeros(50)), window=30)


# -- curve stability: hand case + frozen chain structure ------------------------


def test_curve_stability_hand_flat_curve():
    """Hand case: every chain contract priced IDENTICALLY at the as-of
    → carry = ln(1)/tau = 0 on every spread → implied yield = r exactly
    (y = r + 0 − 0), and the state is 'contango' (f2 == f1 is not
    backwardation). Exercises the same _curve_asof → yield_term_structure
    path the frozen-data backtest drives."""
    from hedging_workbench.data.frozen import load
    from hedging_workbench.validate import _curve_asof

    symbols = [s for s in UNIVERSES["coffee"] if not s.endswith("=F")]
    raw = load(symbols)
    as_of = min(s.dropna().index.max() for s in raw.values()) - pd.Timedelta(days=5)
    flat = pd.Series(
        100.0, index=pd.date_range("2026-01-01", periods=30, freq="B")
    )
    series = {s: flat for s in symbols}
    curve = _curve_asof(symbols, series, as_of, "coffee")
    ts = yield_term_structure(curve, r=0.04)
    assert ts["implied_yield"].tolist() == pytest.approx([0.04] * len(ts))
    assert curve_state(ts) == "contango"


def test_curve_stability_frozen_chain():
    stab = curve_stability_backtest(n_dates=4)
    assert len(stab) == 4
    assert (stab["n_pairs"] >= 1).all()
    assert np.isfinite(stab["front_yield"]).all()
    assert np.isfinite(stab["back_yield"]).all()
    # as-of dates are strictly increasing and in the past
    assert stab["as_of"].is_monotonic_increasing
    # states come from carry.curve_state's vocabulary
    assert set(stab["state"]).issubset({"backwardation", "contango", "mixed"})


# -- note MC vs closed form ------------------------------------------------------


def test_note_mc_within_standard_error():
    mc = note_mc_check()
    assert mc["abs_diff"] <= 2 * mc["se"]


# -- CVA vs QuantLib (same discretization, 1e-9 pre-declared) ---------------------


def test_cva_ql_benchmark():
    pytest.importorskip("QuantLib", reason="QuantLib not installed")
    ee = np.array([80_000.0, 95_000.0, 110_000.0, 90_000.0])
    tenors = np.array([0.25, 0.5, 1.0, 2.0])
    res = cva_ql_benchmark(ee, tenors, hazard=0.1, lgd=0.6, r=0.04)
    assert res is not None
    assert res["rel_diff"] <= 1e-9


# -- inventory and findings -------------------------------------------------------


def test_model_inventory_complete():
    inv = model_inventory()
    assert len(inv) >= 13  # all Phase 1–5 models, incl. Phase 3 studies
    names = {c.name for c in inv}
    assert len(names) == len(inv)  # unique
    modules = {c.module for c in inv}
    assert {
        "hedging_workbench.effectiveness",
        "hedging_workbench.cvar",
        "hedging_workbench.bekk",
        "hedging_workbench.stress",
    }.issubset(modules)
    for c in inv:
        assert all(getattr(c, f) for f in ("module", "inputs", "assumptions", "benchmark"))
        assert c.review_status


def test_validation_findings_runs_all_checks():
    f = validation_findings()
    assert len(f) >= 6
    assert set(f["status"]).issubset({"PASS", "FLAG"})
    assert (f["detail"].str.len() > 0).all()
