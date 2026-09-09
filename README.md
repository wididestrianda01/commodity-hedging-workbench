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
python -m hedging_workbench.data.frozen --universe coffee --start 2024-01-01
python -m hedging_workbench.data.frozen --universe gold   --start 2024-01-01
python -m hedging_workbench.data.frozen --rates                # SOFR snapshot
python -m hedging_workbench.data.frozen --verify               # checksums
```

## Demo (one command, local)

```bash
docker compose up
```

Opens the Streamlit workbench on http://localhost:8501: the frozen coffee
curve with implied convenience yields, the roaster hedge program with a
zero-cost collar, the participation note priced two ways, and book-risk
xVA. Alongside it, the [floci](https://github.com/floci-io/floci) local AWS
emulator answers on http://localhost:4566 as the permanent gate-test fixture
for the Phase 8 Terraform pipeline — nothing in this demo ever leaves the
machine.

Teardown:

```bash
docker compose down        # add -v to drop any floci state
```

## Refresh pipeline (emulator-based)

The nightly pipeline is IaC-managed against the floci emulator — see
`pipeline/terraform/README.md` for the full runbook:

```bash
docker build -f pipeline/Dockerfile.lambda -t workbench-refresh-lambda:latest .
cd pipeline/terraform && tofu init && tofu apply
# scheduled (EventBridge Scheduler, rate(1 day)) or manual:
curl -s -X POST "http://localhost:4566/2015-03-31/functions/workbench-refresh/invocations" -d '{}'
tofu destroy   # teardown
```

Each run re-runs the data gates and lands a SHA-256-verified snapshot bundle
(universe CSVs + manifests + gate report + SOFR) in emulated S3. Fail-closed
holds end-to-end: if the gates fail, the invocation errors and **no snapshot
is promoted** — the failure is visible in the Lambda's CloudWatch logs.

Cloud work is emulator-based by design — labeled emulator-based (floci),
never production AWS. Deploying the same Terraform to a
real AWS account is an explicit, separately-gated step (zero-spend
guardrail: free-tier verification, budget alarm, immediate teardown).
The app displays frozen data locally only — no redistribution of ICE data.

## Data policy

Coffee `KC=F` + per-contract chain is primary; gold `GC=F` is the pre-declared fail-closed fallback. Yahoo Finance is an unofficial source — accepted and documented; manifests carry SHA-256 per file and are verified before use. Frozen snapshots are committed for reproducibility.

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 1. Data gates + MQA replication | `notebooks/00_mqa_replication.ipynb` | done |
| 2. Carry & curve calibration | `notebooks/01_carry_calibration.ipynb` | done |
| 3. Roaster hedging program | `notebooks/02_hedge_program.ipynb` | done |
| 4. Structured participation note | `notebooks/03_structured_note.ipynb` | done |
| 5. Risk gates (book + junior xVA) | `notebooks/04_book_risk_xva.ipynb` | done |
| 6. Validation + decision report | `report/report.pdf` | done |
| 7. Local Streamlit app (Docker) | `app/` + `docker compose up` | done |
| 8. Cloud pipeline (floci → AWS optional) | `pipeline/` Terraform + runbook | done (emulator-based) |
