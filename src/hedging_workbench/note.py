"""Capital-protected participation note on KC=F (Phase 4, tickets 11-01..11-03).

Payoff at T per unit notional N:
    N * (1 + p * max(F_T - K, 0) / F0)

i.e. principal repaid in full, plus p (participation rate) times the
call payoff expressed as a fraction of the reference price. Closed form
decomposes into a discounted zero + p * Black-76 call / F0 (premium in
cents/lb from hedging_workbench.pricing, divided by F0 to make a
fraction payoff). MC simulates F_T as a risk-neutral martingale
(E[F_T] = F0 for a futures forward under deterministic rates), which is
exactly the distribution Black-76 prices — the two-way match is the
trust mechanism (spec story 9).

Conventions follow pricing.py: F, K in cents/lb, sigma decimal, T years,
r decimal (SOFR). Antithetic variates for variance reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from hedging_workbench.pricing import black76, d1
from hedging_workbench.sim import martingale_paths


@dataclass(frozen=True)
class ParticipationNote:
    notional: float  # USD principal, repaid at maturity
    f0: float  # reference price, cents/lb (frozen curve)
    tenor: float  # years
    participation: float  # fraction of upside paid to the holder
    strike: float | None = None  # default ATM

    def __post_init__(self):
        object.__setattr__(
            self, "strike", self.f0 if self.strike is None else self.strike
        )
        if not (0 <= self.participation <= 1):
            raise ValueError(f"participation {self.participation} outside [0,1]")
        if self.f0 <= 0 or self.tenor <= 0 or self.notional <= 0:
            raise ValueError("notional, f0, tenor must be positive")

    def closed_form(self, sigma: float, r: float) -> dict:
        """Price per unit notional, decomposed: bond + participation call."""
        k, t = self.strike, self.tenor
        bond = math.exp(-r * t)
        call = black76("call", self.f0, k, t, sigma, r)
        upside = self.participation * call / self.f0
        return {
            "bond": bond,
            "call": call / self.f0,
            "participation_value": upside,
            "price": bond + upside,
        }

    # -- greeks, ISSUER side (issuer is short the call: delta/vega < 0) --
    def greeks(self, sigma: float, r: float, h: float = 1e-4) -> dict:
        """Issuer delta (per $1 of F0) and vega (per 1.0 vol) per unit notional.

        Analytic on the closed form: c_F = exp(-rT) N(d1), c_sigma =
        exp(-rT) F0 phi(d1) sqrt(T); note upside is p * c / F0. The
        HOLDER is long the call (delta/vega > 0); the issuer's book is
        the negative — reported from the issuer's side, matching the
        hedge problem.
        """
        k, t, f0 = self.strike, self.tenor, self.f0
        d1v = d1(f0, k, t, sigma)
        df = math.exp(-r * t)
        p = self.participation
        delta = -(
            p * df * norm.cdf(d1v) / f0
            - p * black76("call", f0, k, t, sigma, r) / f0**2
        )
        vega = -p * df * norm.pdf(d1v) * math.sqrt(t)
        return {"delta": delta, "vega": vega}

    def mc_price(
        self, sigma: float, r: float, n_paths: int = 100_000, seed: int = 42
    ) -> dict:
        """MC price per unit notional with standard error and 95% CI.

        Antithetic: n_paths is the TOTAL path count (n/2 pairs). F_T
        lognormal martingale, payoff discounted at r.
        """
        ft = martingale_paths(
            self.f0,
            sigma,
            self.tenor,
            steps=1,
            n_paths=n_paths,
            seed=seed,
            antithetic=True,
        )[:, -1]
        payoff = np.maximum(ft - self.strike, 0.0) / self.f0
        disc = math.exp(-r * self.tenor)
        bond = disc
        ups = disc * self.participation * payoff
        est = ups.mean()
        se = ups.std(ddof=1) / math.sqrt(n_paths)
        return {
            "price": bond + est,
            "upside": est,
            "se": se,
            "ci95": (bond + est - 1.96 * se, bond + est + 1.96 * se),
        }

    def mc_greeks(
        self,
        sigma: float,
        r: float,
        n_paths: int = 100_000,
        seed: int = 42,
        rel_bump: float = 0.01,
    ) -> dict:
        """Bump-and-reprice MC greeks, ISSUER side — validates `greeks`."""

        def price(f0: float, s: float) -> float:
            n = ParticipationNote(
                self.notional, f0, self.tenor, self.participation, self.strike
            )
            return n.mc_price(s, r, n_paths=n_paths, seed=seed)["price"]

        f0 = self.f0
        d0, d1_ = price(f0 * (1 + rel_bump), sigma), price(f0 * (1 - rel_bump), sigma)
        u0, u1_ = price(f0, sigma + 0.01), price(f0, sigma - 0.01)
        return {"delta": -(d0 - d1_) / (2 * f0 * rel_bump), "vega": -(u0 - u1_) / 0.02}


def render_governance(
    note: ParticipationNote,
    sigma: float,
    r: float,
    underlying: str = "ICE Coffee C futures (KC=F)",
) -> str:
    """MiFID II product-governance / PRIIPs-KID-style page (ticket 11-04).

    Rendered from the note's own parameters so a changed note regenerates
    its disclosure. Regulation is MAPPED, not asserted satisfied.
    """
    cf = note.closed_form(sigma, r)
    return "\n".join(
        [
            "# Participation Note — Product Governance & Disclosure Page",
            "",
            f"**Underlying:** {underlying} (reference price {note.f0:.2f} ¢/lb)",
            f"**Notional:** ${note.notional:,.0f} · **Tenor:** {note.tenor:.2f} yr "
            f"· **Participation rate:** {note.participation:.0%} "
            f"· **Strike:** {note.strike:.2f} ¢/lb (ATM)",
            f"**Structure:** 100% principal protection + {note.participation:.0%} "
            "of the underlying's upside (call payoff / reference price). "
            f"Per unit notional: bond leg {cf['bond']:.4f}, "
            f"option leg {cf['participation_value']:.4f} at the pricing vol.",
            "",
            "## Target market (MiFID II product governance template)",
            "",
            "- Investors seeking commodity exposure with defined downside "
            "(principal protected at maturity) and accepting capped-in-structure "
            "participation.",
            "- NOT for investors needing liquidity before maturity, income, or "
            "unlimited participation.",
            "",
            "## Risk factors (not hedged by the structure)",
            "",
            "- **Issuer credit risk:** principal protection is only as good as "
            "the issuer — a CVA/PFE exercise (Phase 5) quantifies this.",
            "- **Liquidity:** no secondary market; value before maturity is a "
            "model mark, not a tradeable price.",
            "- **Gap risk:** daily-close observation only; an overnight move "
            "through the strike is captured differently than an intraday path.",
            "- **Opportunity cost:** flat/down markets return principal only — "
            f"the {note.participation:.0%} participation rate is the issuer's "
            "margin lever: it prices the option leg against the funding "
            "advantage of the zero-coupon structure.",
            "",
            "## Costs",
            "",
            "- Explicit: none itemised (wholesale structure). Implicit: the "
            "difference between the Black-76 option value and the participation "
            "actually granted, plus issuer funding spread over SOFR discounting.",
            "",
            "## Honest-framing note",
            "",
            "This page maps MiFID II product-governance and PRIIPs KID *concepts* "
            "to a learning artifact. It is not a KID, not an offer document, and "
            "asserts no compliance status.",
            "",
        ]
    )
