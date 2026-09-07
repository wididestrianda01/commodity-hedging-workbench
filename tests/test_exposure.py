"""Seam 2 tests: EE/EPE/PFE exposure sim + netting/collateral (12-02, 12-05)."""

import numpy as np
import pytest

from hedging_workbench.exposure import (FuturesBook, _b76, collar_book_mtm,
                                        collateralized, epe, netted_mtm,
                                        profile, simulate_front)
from hedging_workbench.pricing import black76, zero_cost_collar

F0, SIG = 300.0, 0.385
CONTRACTS = 10.0
BOOK = FuturesBook(CONTRACTS, F0)


def test_vectorized_black76_matches_scalar():
    """The path-pricing duplicate is the same function as pricing.black76."""
    rng = np.random.default_rng(0)
    f = F0 * np.exp(rng.normal(0, 0.2, 1000))
    for kind in ("call", "put"):
        vec = _b76(kind, f, 290.0, 0.25, SIG, 0.0366)
        scal = np.array([black76(kind, x, 290.0, 0.25, SIG, 0.0366)
                         for x in f[:20]])
        assert np.allclose(vec[:20], scal)


def test_martingale_mean_is_preserved():
    """E[F_t] = f0 at every grid time — the Q-measure premise."""
    f = simulate_front(F0, SIG, horizon_years=0.25, steps=60,
                       n_paths=200_000, seed=11)
    assert f.shape == (200_000, 61)
    assert np.allclose(f[:, 0], F0)
    assert np.allclose(f.mean(axis=0), F0, rtol=0.02)


def test_deterministic_path_ee_matches_analytic():
    """sigma=0 -> no variance: EE(t) is the exact analytic forward MtM."""
    entry = 280.0
    analytic = CONTRACTS * 375.0 * (F0 - entry)
    f = simulate_front(F0, 0.0, 1.0, 10, 50)
    mtm = FuturesBook(CONTRACTS, entry).mtm(f)
    prof = profile(mtm)
    assert np.allclose(prof["ee"], analytic)
    assert epe(prof) == pytest.approx(analytic)
    assert (prof["pfe"] >= prof["ee"]).all()


def test_ee_le_pfe_invariant_on_stochastic_paths():
    f = simulate_front(F0, SIG, 0.5, 100, 20_000, seed=3)
    prof = profile(BOOK.mtm(f))
    assert (prof["ee"] <= prof["pfe"] + 1e-9).all()
    assert (prof["ee"][1:] > 0).all()  # t=0 MtM is 0 by construction
    assert epe(prof) > 0


def test_epe_stabilizes_with_path_count():
    """Convergence: EPE drift shrinks as paths grow (mean-stable)."""
    eps = []
    for n in (20_000, 80_000):
        f = simulate_front(F0, SIG, 0.25, 50, n, seed=5)
        eps.append(epe(profile(BOOK.mtm(f))))
    assert abs(eps[0] - eps[1]) / eps[1] < 0.05


# -- 12-05 -------------------------------------------------------------------

def _same_shape(sigma=SIG):
    return simulate_front(F0, sigma, 0.25, 50, 30_000, seed=9)


def test_netting_benefit_direction():
    """Netted EE <= sum of standalones; equality for same-sign books."""
    m1 = BOOK.mtm(_same_shape())
    m2 = FuturesBook(5.0, F0 * 1.05).mtm(_same_shape())
    ee_ind = profile(m1)["ee"] + profile(m2)["ee"]
    ee_net = profile(netted_mtm([m1, m2]))["ee"]
    assert (ee_net <= ee_ind + 1e-9).all()
    # perfectly correlated same-sign books: equality (no diversification)
    ee_same = profile(netted_mtm([m1, m1.copy()]))["ee"]
    assert np.allclose(ee_same, 2 * profile(m1)["ee"])


def test_collateral_reduces_ee_monotonically():
    mtm = BOOK.mtm(_same_shape())
    ee0 = profile(mtm)["ee"]
    for th, ia in ((0.0, 0.0), (50_000.0, 0.0), (50_000.0, 100_000.0)):
        eec = profile(collateralized(mtm, th, ia))["ee"]
        assert (eec <= ee0 + 1e-9).all()
        ee0 = eec  # monotone in (threshold, ia)


def test_collar_wraps_futures_tail_and_starts_flat():
    """Zero-cost premiums cancel at t=0; the floor cuts the left tail."""
    col = zero_cost_collar(F0, put_strike=270.0, t=0.25, sigma=SIG, r=0.0366)
    assert col.premium_gap < 0.05  # zero-cost identity (cents/lb)
    f = _same_shape()
    t_ttm = np.full(f.shape[1], 0.25)
    mtm_c = collar_book_mtm(CONTRACTS, col, f, t_ttm, SIG, 0.0366)
    mtm_n = BOOK.mtm(f)
    assert abs(mtm_c[0, 0]) < CONTRACTS * 375.0 * 0.05
    assert mtm_c.min() > mtm_n.min()


def test_netting_set_validation():
    with pytest.raises(ValueError):
        netted_mtm([])
    with pytest.raises(ValueError):
        netted_mtm([np.zeros((3, 3)), np.zeros((4, 3))])
