"""Seam 2 tests: participation note — closed form, MC, greeks (11-01..11-03).

Expectations are identities and hand-computed numbers on a constructed
flat curve/vol, plus convergence of MC into the closed form.
"""

import math

import pytest

from hedging_workbench.note import ParticipationNote, render_governance

F0, R, T, SIG = 300.0, 0.0366, 1.0, 0.385
NOTE = ParticipationNote(notional=1_000_000, f0=F0, tenor=T, participation=0.6)


def test_closed_form_decomposition():
    """Price = bond + p * Black-76 call / F0, hand-decomposed."""
    cf = NOTE.closed_form(SIG, R)
    bond = math.exp(-R * T)
    assert cf["bond"] == pytest.approx(bond)
    assert cf["price"] == pytest.approx(bond + 0.6 * cf["call"])
    # and the call fraction matches pricing.black76 directly
    from hedging_workbench.pricing import black76

    assert cf["call"] == pytest.approx(black76("call", F0, F0, T, SIG, R) / F0)


def test_participation_zero_is_pure_bond():
    """p=0 -> principal protection only, no option value."""
    n = ParticipationNote(1_000_000, F0, T, 0.0)
    assert n.closed_form(SIG, R)["price"] == pytest.approx(math.exp(-R * T))


def test_full_participation_zero_strike_is_double_discount():
    """Conservation, p=1 with K=0 (ticket 11-01 AC): call = e^{-rT}*F0, so
    price = bond + call/F0 = 2 * e^{-rT} — pure bond plus the whole
    forward, hand-computed.
    """
    n = ParticipationNote(1_000_000, F0, T, 1.0, strike=0.0)
    assert n.closed_form(SIG, R)["price"] == pytest.approx(2 * math.exp(-R * T))


def test_mc_converges_to_closed_form():
    """Closed-form price inside MC 95% CI; se shrinks ~ 1/sqrt(n)."""
    cf = NOTE.closed_form(SIG, R)["price"]
    ses = []
    for n in (20_000, 80_000, 320_000):
        mc = NOTE.mc_price(SIG, R, n_paths=n, seed=7)
        lo, hi = mc["ci95"]
        assert lo < cf < hi, f"n={n}: closed form {cf:.6f} outside {lo:.6f},{hi:.6f}"
        ses.append(mc["se"])
    # variance reduction working: 4x paths -> ~2x se shrink (allow slack)
    assert ses[0] / ses[2] > 3.0


def test_mc_deterministic_seed():
    a = NOTE.mc_price(SIG, R, n_paths=10_000, seed=3)
    b = NOTE.mc_price(SIG, R, n_paths=10_000, seed=3)
    assert a["price"] == b["price"]


def test_greeks_analytic_and_direction():
    """Issuer is short the call: delta < 0, vega < 0; both match FD on the
    closed form (hand-checkable analytic-vs-FD, ticket 11-03).
    """
    g = NOTE.greeks(SIG, R)
    assert g["delta"] < 0 and g["vega"] < 0
    h = 1e-3
    up = ParticipationNote(1, F0 * (1 + h), T, 0.6, strike=F0).closed_form(SIG, R)[
        "price"
    ]
    dn = ParticipationNote(1, F0 * (1 - h), T, 0.6, strike=F0).closed_form(SIG, R)[
        "price"
    ]
    # issuer delta = -d(price)/df0; FD over the bumped closed form
    assert g["delta"] == pytest.approx(-(up - dn) / (2 * h * F0), rel=1e-4)
    up = NOTE.closed_form(SIG + h, R)["price"]
    dn = NOTE.closed_form(SIG - h, R)["price"]
    assert -g["vega"] == pytest.approx((up - dn) / (2 * h), rel=1e-4)


def test_mc_greeks_validate_analytic():
    """MC bump-and-reprice lands near the analytic greeks (bump noise bound)."""
    g = NOTE.greeks(SIG, R)
    mg = NOTE.mc_greeks(SIG, R, n_paths=200_000, seed=11)
    assert g["delta"] == pytest.approx(mg["delta"], abs=2e-4)
    assert g["vega"] == pytest.approx(mg["vega"], abs=0.05)


def test_governance_renders_note_parameters():
    """A changed note visibly flows through; disclaimer present."""
    doc = render_governance(NOTE, SIG, R)
    assert "60%" in doc and "1.00 yr" in doc and "300.00" in doc and "KC=F" in doc
    assert "not a KID" in doc
    other = render_governance(ParticipationNote(5_000_000, 250.0, 2.0, 0.4), 0.3, R)
    assert "40%" in other and "250.00" in other and other != doc
