# IFRS 9 Cash-Flow Hedge Designation Memo

**Archetype:** coffee roaster/buyer (Starbucks FY2025 10-K template)
**Program horizon:** Sep 2026 – Aug 2027 (12 monthly purchases, 1,200,000 lb total, 32.00 contract-equivalents)
**Hedged item:** highly probable forecast green-coffee purchases (variable price; 'C' price risk via ICE Coffee C futures)
**Hedging instrument:** long coffee C futures (ICE KC) with zero-cost collar (long put / short call)

## Designation

- **Type:** cash-flow hedge of forecast transactions (IFRS 9.6.3.1).
- **Risk designated:** coffee C price risk only; basis between the roaster's physical origin differentials and the exchange C price is NOT designated and remains in P&L.
- **AOCI treatment:** effective portions of the instrument's gains/losses go to OCI and accumulate in AOCI; reclassified to P&L in the period the forecast purchases affect earnings (IFRS 9.6.3.2-6.3.3).
- **Margin collateral:** variation margin posted to the clearing house is a receivable/liability, not a hedge-account item (10-K treatment: collateral shown separately; Starbucks discloses $37.9M margin deposits).

## Effectiveness (principles-based, IFRS 9.6.3.2 / B6.3.5)

Assessed on three principles — economic relationship, credit dominance, hedge-ratio consistency — NOT the retired IAS 39 80-125% dollar-offset band. Quantitative outputs come from `hedging_workbench.effectiveness.assess` on the program's P&L series; forward-starting assessments run each reporting period.

## EMIR 3 mapping (mapped, not compliance advice)

- Annual commodity derivative notional ≈ €3.6M (USD 3.9M at 1.08).
- ESMA NFC uncleared threshold for commodity derivatives: €3B (ESMA FR 2026-02-25). Position is 0.1201% of threshold → below-threshold NFC: clearing obligation does not apply; risk-mitigation techniques (variance margin) apply only above thresholds.
- Hedging exemption (RTS 21a criteria): positions that objectively reduce commercial risk are exempt from position-limit and certain reporting considerations — the program's designation memo and purchase schedule are the objective evidence.

## Honest-framing note

This memo maps regulatory *concepts* to a learning artifact. It is not legal, accounting, or compliance advice and asserts no regulatory status for any real entity.
