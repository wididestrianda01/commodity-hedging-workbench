"""Seam 2 tests: EE/EPE/PFE exposure sim + netting/collateral (12-02, 12-05)."""

import numpy as np
import pytest

from hedging_workbench.exposure import (
    FuturesBook,
    collar_book_mtm,
    collateralized,
    epe,
    netted_mtm,
    profile,
    simulate_front,
)
from hedging_workbench.pricing import zero_cost_collar

F0, SIG = 300.0, 0.385
CONTRACTS = 10.0
BOOK = FuturesBook(CONTRACTS, F0)


def test_martingale_mean_is_preserved():
    """E[F_t] = f0 at every grid time — the Q-measure premise."""
    f = simulate_front(F0, SIG, horizon_years=0.25, steps=60, n_paths=200_000, seed=11)
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
    f = _same_shape()
    m1 = BOOK.mtm(f)
    m2 = FuturesBook(CONTRACTS, F0 * 1.02).mtm(f)
    ee_net = profile(netted_mtm([m1, m2]))["ee"]
    ee_sum = profile(m1)["ee"] + profile(m2)["ee"]
    assert (ee_net <= ee_sum + 1e-9).all()
    # perfectly correlated same-sign book: netting is a no-op
    ee_same = profile(netted_mtm([m1, m1]))["ee"]
    assert np.allclose(ee_same, 2 * profile(m1)["ee"])


def test_netting_cva_direction():
    """CVA-level direction (12-05): netted CVA <= sum of standalone CVAs;
    equality on a perfectly-correlated same-sign book; a mirrored book
    nets to zero exposure, hence zero CVA.
    """
    from hedging_workbench.cva import HAZARD, LGD, cva_unilateral

    f = _same_shape()
    m1 = BOOK.mtm(f)
    idx = np.linspace(0, m1.shape[1] - 1, 6).round().astype(int)[1:]
    ten = profile(m1)["t"].to_numpy()[idx]

    def cva(mtm):
        ee = profile(mtm)["ee"].to_numpy()[idx]
        return cva_unilateral(ee, ten, HAZARD, LGD, 0.0366)

    m2 = FuturesBook(CONTRACTS, F0 * 1.02).mtm(f)
    assert cva(netted_mtm([m1, m2])) <= cva(m1) + cva(m2) + 1e-6
    assert cva(netted_mtm([m1, m1])) == pytest.approx(2 * cva(m1), rel=1e-9)
    assert cva(netted_mtm([m1, -m1])) == pytest.approx(0.0, abs=1e-6)


def test_collateral_mta_rounds_transfers():
    """MTA mechanics (12-05): IA + threshold held in full; further
    transfers post in completed MTA round lots only, so the residual
    exposure is bounded by the MTA; mta=0 keeps the continuous block.
    """
    mtm = np.array([[80.0, 120.0, 175.0, 260.0]])
    ia, threshold, mta = 10.0, 100.0, 50.0
    ee = collateralized(mtm, threshold, ia, mta)
    # below threshold: fully collateralised
    assert ee[0, 0] == pytest.approx(0.0)
    # completed lots only: residual (V-threshold) mod MTA, net of IA
    assert np.allclose(ee, np.array([[0.0, 10.0, 15.0, 0.0]]))
    # no granularity (mta=0) reproduces the continuous threshold block
    assert np.allclose(
        collateralized(mtm, threshold, ia), np.maximum(mtm - (ia + threshold), 0.0)
    )


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
