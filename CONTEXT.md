# Commodity Hedging & Structuring Workbench — domain model

Learning workbench replicating practitioner commodity-hedging and
structuring workflows on a coffee roaster archetype (Starbucks FY2025
10-K template), anchored to frozen market data. Terms below are the
ubiquitous language; use them in code, notebooks, and docs.

## Domain terms

- **Frozen data** — committed market snapshots in `data/frozen/`
  (per-symbol CSVs + SHA-256 manifests), the only inputs analytics
  may read. No live data is read anywhere. Adapters: commodity
  universes (Yahoo) and SOFR (FRED), served by `data/frozen.py`.
- **Data gates** — count/completeness/sanity assertions on a universe
  (`data/gates.py`). Coffee primary, gold fail-closed fallback: if
  both fail, no curve and everything stops.
- **Universe** — a contract chain plus the continuous front symbol
  (`KC=F` coffee, `GC=F` gold); per-contract tickers are discovered
  empirically, not via a chain API (`data/universe.py`).
- **Curve** — contract-level snapshot (symbol, label, expiry, price,
  ttm) built from a frozen universe; carries the as-of date in attrs.
- **Implied carry / convenience yield** — annualised log spread between
  consecutive curve points; `y = r + storage - carry`
  (`carry.py`). Backwardation = negative carry.
- **Schwartz–Smith fit** — two-factor (OU deviation + GBM level)
  reduced-form fit to the curve snapshot; anchors the curve LEVEL,
  carries no vol parameters (`ssfit.py`).
- **Working vol** — GARCH(1,1) conditional vol, annualised %/yr, the
  single vol input to pricing/simulation (`vol.py`).
- **Hedge program** — exposure schedule → contract selection (in-or-
  after-month rule) → roll schedule (exit 10 days pre-expiry), the
  Phase 3 deep module (`hedge.py`).
- **Variation margin** — daily cash flows on the long futures position:
  contracts × $375/cent × Δsettle, plus the margin balance (`hedge.py`).
- **Zero-cost collar** — long put funded by a short call solved so the
  premiums cancel exactly (`pricing.py`).
- **Black-76** — the pricing formula for European options on futures;
  implemented once, scalar-or-array, in `pricing.py` (with `d1`).
- **Hedge effectiveness** — IFRS 9 principles-based three-prong
  assessment (economic relationship, credit dominance, ratio
  consistency), explicitly NOT the IAS 39 80-125% band
  (`effectiveness.py`).
- **Participation note** — capital-protected structured note: principal
  plus participation × ATM call/F0; priced closed-form AND by MC with
  the two-way match as the trust mechanism (`note.py`).
- **Exposure profile** — EE(t)/EPE/PFE from simulated MtM paths of the
  book; netting and collateral transforms on the same paths
  (`exposure.py`).
- **Martingale paths** — the shared lognormal Q-measure path generator
  (`sim.py`); futures are martingales under zero-cost carry.
- **Unilateral CVA** — expected loss from counterparty default, flat
  hazard + SOFR-flat discount, QuantLib-benchmarked conventions
  (`cva.py`).
- **Contract conventions** — ICE Coffee C facts (37,500 lb, $375 per
  cent, assumed $8k initial margin) with the single cents/lb→USD
  conversion, `conventions.py`. Never re-derive 375.

## Architecture decisions

- One Black-76 implementation (`pricing.black76`, scalar or array);
  path repricing and closed forms share it.
- Frozen data conventions (paths, slugs, SHA-256, manifest naming)
  live only in `data/frozen.py`; quant modules learn
  freeze/load/verify/latest_rate and keep an explicit `frozen_dir`
  parameter for test overrides.
- One MC engine (`sim.martingale_paths`); terminal-only pricing is
  steps=1, antithetic variates opt-in.
- Contract constants live in `conventions.py`; USD-outcome helpers
  stay in `pricing`, margin mechanics in `hedge`.
- Pure quant cores take Series/DataFrames; `*_from_frozen` conveniences
  are the only places that touch the data layer.
