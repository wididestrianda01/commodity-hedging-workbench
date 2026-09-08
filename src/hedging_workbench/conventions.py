"""ICE Coffee C contract conventions — single source for units.

Quote unit is US cents per pound; one contract is 37,500 lb, so a
1-cent move = $375 per contract. Every module converting between
cents/lb and USD imports from here: re-deriving 375 (or dividing by
100.0) in a second place is how unit bugs ship in this domain.
"""

CONTRACT_LB = 37_500
DOLLARS_PER_CENT = CONTRACT_LB / 100.0  # $375 per cent per contract
INITIAL_MARGIN_PER_CONTRACT = 8_000.0  # assumed; ICE levels change —
# stated learner assumption (see hedge.py), not an exchange feed.


def lb_to_usd(volume_lb: float) -> float:
    """Multiplier turning a cents/lb P&L into USD for a volume in pounds."""
    return volume_lb / 100.0
