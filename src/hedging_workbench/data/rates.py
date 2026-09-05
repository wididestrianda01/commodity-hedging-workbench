"""Risk-free rate layer: FRED SOFR (primary) with frozen snapshot + manifest.

SOFR is an overnight rate, published as an annualised percentage. It is the
Phase-1 discounting anchor; a term structure is Phase 2+ work.

CLI:  python -m hedging_workbench.data.rates --start 2024-01-01
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

FROZEN_DIR = Path(__file__).resolve().parents[3] / "data" / "frozen"
SOFR_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
RATES_MANIFEST = "manifest_rates.json"


class RateError(RuntimeError):
    """SOFR unavailable and no frozen snapshot exists."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_sofr(start: str = "2024-01-01") -> pd.Series:
    df = pd.read_csv(f"{SOFR_URL}&cosd={start}", index_col="observation_date",
                     parse_dates=True)
    s = df["SOFR"].dropna().astype(float)
    if s.empty:
        raise RateError("FRED returned no SOFR observations")
    return s


def freeze_rates(start: str = "2024-01-01",
                 frozen_dir: Path = FROZEN_DIR) -> Path:
    """Fetch SOFR, freeze CSV, write rates manifest. Returns manifest path."""
    frozen_dir.mkdir(parents=True, exist_ok=True)
    s = fetch_sofr(start)
    path = frozen_dir / "SOFR.csv"
    s.rename("sofr").to_csv(path, index_label="date")
    manifest = {
        "name": "rates",
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "start": start,
        "files": {"SOFR": {"path": "SOFR.csv",
                           "sha256": _sha256_file(path)}},
    }
    out = frozen_dir / RATES_MANIFEST
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def load_sofr(frozen_dir: Path = FROZEN_DIR) -> pd.Series:
    path = frozen_dir / "SOFR.csv"
    if not path.exists():
        raise RateError("no frozen SOFR — run `python -m hedging_workbench.data.rates`")
    return pd.read_csv(path, index_col="date", parse_dates=True)["sofr"].astype(float)


def latest_rate(frozen_dir: Path = FROZEN_DIR) -> float:
    """Latest published SOFR as a decimal (e.g. 0.0366)."""
    return float(load_sofr(frozen_dir).iloc[-1]) / 100.0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-01-01")
    args = ap.parse_args()
    out = freeze_rates(args.start)
    print(f"frozen rates: {out}  latest SOFR {latest_rate():.4f}")


if __name__ == "__main__":
    main()
