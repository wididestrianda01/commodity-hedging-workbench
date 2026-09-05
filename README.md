# Commodity Hedging & Structuring Workbench

Personal learning project extending the [Citi Markets Quantitative Analysis Forage simulation](https://www.theforage.com/simulations/citi/global-quantitative-analysis-analyst-6b4m) into a real-data commodity-desk workflow: frozen futures curves, carry calibration, a roaster hedging program, a structured note, and junior-scope xVA — every number from gated public data.

Not affiliated with or endorsed by Citi. Educational project only; no trading or investment advice.

## Structure

```
├── src/hedging_workbench/   # the package (single code seam)
│   └── data/                # universe, throttled download + SHA-256 freeze, gates
├── tests/                   # pytest: gates pass/fail, fallback, manifest tamper
├── notebooks/               # per-phase deliverables (executed outputs committed)
├── data/frozen/             # frozen market data + manifests (committed)
└── .scratch/                # local planning artifacts (NOT in the repo)
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                                        # 6 gate tests
python -m hedging_workbench.data.gates                        # run data gates
jupyter nbconvert --to notebook --execute notebooks/00_mqa_replication.ipynb
```

Refresh frozen data (throttled Yahoo download):

```bash
python -m hedging_workbench.data.download --universe coffee --start 2024-01-01
python -m hedging_workbench.data.download --universe gold   --start 2024-01-01
```

## Data policy

Coffee `KC=F` + per-contract chain is primary; gold `GC=F` is the pre-declared fail-closed fallback. Yahoo Finance is an unofficial source — accepted and documented; manifests carry SHA-256 per file and are verified before use. Frozen snapshots are committed for reproducibility.

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 1. Data gates + MQA replication | `notebooks/00_mqa_replication.ipynb` | ✅ |
| 2. Carry & curve calibration | `notebooks/01_carry_calibration.ipynb` | next |
| 3. Roaster hedging program | `notebooks/02_hedge_program.ipynb` | — |
| 4. Structured participation note | `notebooks/03_structured_note.ipynb` | — |
| 5. Risk gates (book + junior xVA) | `notebooks/04_book_risk_xva.ipynb` | — |
| 6. Validation + decision report | `report/report.pdf` | — |
| 7. Local Streamlit app (Docker) | `app/` | — |
| 8. Cloud pipeline (floci → AWS optional) | Terraform + runbook | — |
