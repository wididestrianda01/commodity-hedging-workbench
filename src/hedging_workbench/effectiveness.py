"""IFRS 9 principles-based hedge effectiveness (ticket 10-07).

Three-pronged assessment per IFRS 9 (deliberately NOT the IAS 39 80-125%
mechanical band, which could pass an ineffective hedge and fail an
effective one — the test suite demonstrates both):

1. Economic relationship: values of hedging instrument and hedged item move
   together — Pearson correlation of P&L series >= CORR_MIN.
2. Credit dominance: P&L from credit-related terms must not dominate the
   instrument's cash flows — credit-variance share <= CREDIT_MAX_SHARE.
3. Hedge-ratio consistency: actual volume weighted against the designated
   ratio stays within RATIO_TOL (rebalancing limited to genuine volume
   changes, not mechanical resets).

All thresholds are stated learner judgment calls (principles-based
standards permit them), pre-declared in constants, not fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CORR_MIN = 0.8
CREDIT_MAX_SHARE = 0.5
RATIO_TOL = 0.05


@dataclass
class Effectiveness:
    correlation: float
    dollar_offset: float
    credit_share: float
    ratio_deviation: float
    effective: bool
    reasons: list[str] = field(default_factory=list)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def assess(hypothetical_pnl, hedged_item_pnl, market_pnl=None,
           credit_pnl=None, designated_ratio: float = 1.0,
           actual_ratio: float = 1.0) -> Effectiveness:
    """Run the three IFRS 9 principles on one assessment window.

    hypothetical_pnl / hedged_item_pnl: P&L series of the hypothetical
    derivative and the hedged item (same periods, same units). An offsetting
    hedge shows raw correlation ≈ -1 (instrument gains as the item loses);
    relationship strength is measured as |corr|.
    market_pnl / credit_pnl: decomposition of the HEDGING instrument's P&L
    into price-driven and credit-driven parts (credit test; None -> 0).
    designated_ratio / actual_ratio: designated hedge ratio vs the volume-
    weighted ratio actually maintained.
    """
    x = np.asarray(hypothetical_pnl, dtype=float)
    y = np.asarray(hedged_item_pnl, dtype=float)
    corr = abs(_pearson(x, y))
    denom = abs(y.sum())
    offset = abs(x.sum()) / denom if denom > 0 else np.inf

    if market_pnl is None:
        credit_share = 0.0
    else:
        m = np.asarray(market_pnl, dtype=float)
        c = np.asarray(credit_pnl, dtype=float)
        vm, vc = float(np.var(m)), float(np.var(c))
        credit_share = vc / (vm + vc) if (vm + vc) > 0 else 0.0

    ratio_dev = abs(actual_ratio - designated_ratio) / designated_ratio

    reasons = []
    if corr < CORR_MIN:
        reasons.append(f"economic relationship weak (corr {corr:.2f} < {CORR_MIN})")
    if credit_share > CREDIT_MAX_SHARE:
        reasons.append(f"credit dominance (share {credit_share:.2f} > {CREDIT_MAX_SHARE})")
    if ratio_dev > RATIO_TOL:
        reasons.append(f"hedge ratio drifted {ratio_dev:.0%} from designation")
    return Effectiveness(correlation=corr, dollar_offset=offset,
                         credit_share=credit_share, ratio_deviation=ratio_dev,
                         effective=not reasons, reasons=reasons)
