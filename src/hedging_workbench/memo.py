"""IFRS 9 cash-flow-hedge designation memo builder (ticket 10-07).

Renders the 1-page memo from REAL program numbers (10-05 exposure, 10-06
collar) so the artifact can never drift from the data. Mirrors the
Starbucks FY2025 10-K hedge disclosure structure. Regulation is MAPPED,
not asserted as satisfied — the memo says so.
"""

from __future__ import annotations

from hedging_workbench.conventions import CONTRACT_LB


def designation_memo(prog, f0: float, collar=None, usd_eur: float = 1.08) -> str:
    """Render the designation memo. prog: HedgeProgram; f0 in cents/lb."""
    exp = prog.exposure
    total_lb = float(exp["volume_lb"].sum())
    notional_usd = total_lb * f0 / 100.0
    notional_eur = notional_usd / usd_eur
    emir_threshold_eur = 3e9
    instrument = "long coffee C futures (ICE KC)" + (
        " with zero-cost collar (long put / short call)" if collar else ""
    )
    lines = [
        "# IFRS 9 Cash-Flow Hedge Designation Memo",
        "",
        "**Archetype:** coffee roaster/buyer (Starbucks FY2025 10-K template)",
        f"**Program horizon:** {exp['month'].iloc[0]:%b %Y} – {exp['month'].iloc[-1]:%b %Y} "
        f"({len(exp)} monthly purchases, {total_lb:,.0f} lb total, "
        f"{total_lb / CONTRACT_LB:.2f} contract-equivalents)",
        "**Hedged item:** highly probable forecast green-coffee purchases "
        "(variable price; 'C' price risk via ICE Coffee C futures)",
        f"**Hedging instrument:** {instrument}",
        "",
        "## Designation",
        "",
        "- **Type:** cash-flow hedge of forecast transactions (IFRS 9.6.3.1).",
        "- **Risk designated:** coffee C price risk only; basis between the "
        "roaster's physical origin differentials and the exchange C price is "
        "NOT designated and remains in P&L.",
        "- **AOCI treatment:** effective portions of the instrument's gains/"
        "losses go to OCI and accumulate in AOCI; reclassified to P&L in the "
        "period the forecast purchases affect earnings (IFRS 9.6.3.2-6.3.3).",
        "- **Margin collateral:** variation margin posted to the clearing house "
        "is a receivable/liability, not a hedge-account item (10-K treatment: "
        "collateral shown separately; Starbucks discloses $37.9M margin deposits).",
        "",
        "## Effectiveness (principles-based, IFRS 9.6.3.2 / B6.3.5)",
        "",
        "Assessed on three principles — economic relationship, credit "
        "dominance, hedge-ratio consistency — NOT the retired IAS 39 80-125% "
        "dollar-offset band. Quantitative outputs come from "
        "`hedging_workbench.effectiveness.assess` on the program's P&L "
        "series; forward-starting assessments run each reporting period.",
        "",
        "## EMIR 3 mapping (mapped, not compliance advice)",
        "",
        f"- Annual commodity derivative notional ≈ €{notional_eur / 1e6:.1f}M "
        f"(USD {notional_usd / 1e6:.1f}M at {usd_eur}).",
        f"- ESMA NFC uncleared threshold for commodity derivatives: €3B "
        f"(ESMA FR 2026-02-25). Position is "
        f"{notional_eur / emir_threshold_eur:.4%} of threshold → below-threshold "
        f"NFC: clearing obligation does not apply; risk-mitigation techniques "
        f"(variance margin) apply only above thresholds.",
        "- Hedging exemption (RTS 21a criteria): positions that objectively "
        "reduce commercial risk are exempt from position-limit and certain "
        "reporting considerations — the program's designation memo and "
        "purchase schedule are the objective evidence.",
        "",
        "## Honest-framing note",
        "",
        "This memo maps regulatory *concepts* to a learning artifact. It is "
        "not legal, accounting, or compliance advice and asserts no "
        "regulatory status for any real entity.",
        "",
    ]
    return "\n".join(lines)
