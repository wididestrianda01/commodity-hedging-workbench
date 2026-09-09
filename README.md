# Commodity hedging & structuring workbench

A coffee roaster must buy 1.2 million pounds of green coffee over the next
twelve months. Coffee can move 40% in a year; the roaster's margin cannot.
This workbench prices and hedges that purchase program on real frozen market
data. It extends the [Citi Markets Quantitative Analysis Forage
simulation](https://www.theforage.com/simulations/citi/global-quantitative-analysis-analyst-6b4m)
— an interview-practice simulation — onto a live-shaped desk workflow: frozen
futures curves, carry calibration, a hedge program, a structured note, and a
counterparty-risk layer, every number regenerated from gated public data.

Not affiliated with or endorsed by Citi. Educational project only; no trading
or investment advice.

## Results

All numbers below are regenerated in one run by
`report/make_numbers.py` from the frozen data, and the full analysis is in
[`report/report.pdf`](report/report.pdf).

| Question | Result |
|---|---|
| What does the frozen curve say? | Backwardated at every consecutive spread as of 2026-09-04; implied convenience yield 40.8%/yr at the front decaying to 6.3%/yr on the back |
| Futures or collar? | A zero-cost collar (280 put / 382.06 call) locks the annual cost at $3.891M between the strikes while cutting the 95th-percentile margin peak from $103,797 to $44,250 — 58% lower |
| What does going unhedged cost? | 95th-percentile annual cost $5.29M, simulated worst case $10.3M |
| Which volatility model? | GARCH(1,1) beats EWMA on rolling forecasts: RMSE 4.01 vs 6.80 %/yr over 4 windows |
| Does the structured note price two ways? | $1M-notional capital-protected participation note: closed form $1.05169 vs Monte Carlo $1.051913 ± 0.000452 |
| Does the CVA model match a benchmark? | Bucketed unilateral CVA $583.79 agrees with a QuantLib re-derivation to 1.9e-16 relative |
| Does VaR cover? | Static historical 95% VaR under-covers (38 breaches vs 25.2 expected) — kept as a flag; a filtered EWMA estimator restores coverage to ratio 1.10 |

## Data

| Series | Source | Role |
|---|---|---|
| ICE Coffee C futures (KC=F + 8-contract chain) | Yahoo Finance daily bars (unofficial source, documented) | primary universe |
| Gold futures (GC=F) | Yahoo Finance daily bars | pre-declared fail-closed fallback |
| SOFR | FRED | discounting and carry |

As-of date 2026-09-04, frozen with SHA-256 manifests. Exchange-data licensing
forbids redistributing the price bars, so the repo commits manifests, gate
reports and executed notebooks — the bars regenerate locally:

```bash
python -m hedging_workbench.data.frozen --universe coffee --start 2024-01-01
python -m hedging_workbench.data.frozen --universe gold   --start 2024-01-01
python -m hedging_workbench.data.frozen --rates
python -m hedging_workbench.data.frozen --verify   # checksums against manifests
```

The data pipeline is fail-closed: a coffee gate failure switches to the gold
fallback, a gold failure stops everything, and a no-look-ahead gate rejects
any calibration window that sees past the as-of date.

## Methodology

| Phase | Method | Deliverable |
|---|---|---|
| 1. Data gates + MQA replication | SHA-256 manifest freeze, count/completeness gates; the Forage Task-3 methodology re-run on the real curve | `notebooks/00_mqa_replication.ipynb` |
| 2. Carry & curve calibration | Cost-of-carry identity gives the implied convenience-yield term structure; Schwartz–Smith (2000) two-factor fit (κ = 5.97/yr, half-life ≈ 1.4 months); GARCH(1,1) working volatility | `notebooks/01_carry_calibration.ipynb` |
| 3. Roaster hedging program | Contract selection with roll schedule; futures vs zero-cost collar; IFRS 9 principles-based effectiveness testing; EMIR 3 mapping | `notebooks/02_hedge_program.ipynb` |
| 4. Structured note | Black-76 closed form vs antithetic Monte Carlo (100k paths); bump-and-reprice greeks | `notebooks/03_structured_note.ipynb` |
| 5. Risk gates + junior xVA | Historical VaR/ES; EE/EPE/PFE engine; bucketed unilateral CVA; netting and collateral transforms | `notebooks/04_book_risk_xva.ipynb` |
| 6. Validation + decision report | SR 26-2-aligned harness: 13-model inventory, no-look-ahead gate, rolling outcomes analysis; hedged-vs-unhedged decision | `report/report.pdf` |
| 7. Streamlit app | The frozen curve, hedge program, note and book risk in one local UI | `app/` |
| 8. Cloud pipeline | Nightly fail-closed refresh, Terraform-managed against a local AWS emulator | `pipeline/` |

Validation verdict: six of seven outcome checks pass. The one flag — static
historical VaR coverage — is kept deliberately as recorded evidence and
remediated with the filtered estimator. Every headline check is also
reproducible by hand: the test suite carries the arithmetic in docstrings.

## Run it

```bash
# refresh the frozen data first (bars are not committed; see Data above)
pip install -e ".[dev]"
pytest                                     # gate, fallback and verification tests
python -m hedging_workbench.data.gates     # run the data gates
jupyter lab notebooks/
```

One-command demo:

```bash
docker compose up
```

Opens the Streamlit workbench on http://localhost:8501: the frozen coffee
curve with implied convenience yields, the roaster hedge program with a
zero-cost collar, the participation note priced two ways, and book-risk xVA.
Alongside it, the [floci](https://github.com/floci-io/floci) local AWS
emulator answers on http://localhost:4566 as the gate-test fixture for the
Phase 8 pipeline. Nothing in the demo leaves the machine.

Teardown: `docker compose down` (add `-v` to drop floci state).

## Cloud pipeline

The nightly refresh is IaC-managed against the floci emulator — each run
re-runs the data gates and lands a SHA-256-verified snapshot bundle in
emulated S3; if the gates fail, no snapshot is promoted. The full runbook is
[`pipeline/terraform/README.md`](pipeline/terraform/README.md). Deploying the
same Terraform to real AWS is an explicit, separately-gated step behind a
zero-spend guardrail.

## Layout

```
├── src/hedging_workbench/   # the package (single code seam)
│   └── data/                # universe, throttled download + SHA-256 freeze, gates
├── tests/                   # pytest: gates, fallback, verification, invariants
├── notebooks/               # per-phase deliverables (executed outputs committed)
├── report/                  # validation & decision report (LaTeX + PDF)
├── app/                     # Streamlit workbench
├── pipeline/                # refresh job, Docker, Terraform + runbook
└── data/frozen/             # manifests, gate reports, SOFR (bars regenerate locally)
```
