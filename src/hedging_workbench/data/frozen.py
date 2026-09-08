"""Frozen data seam: fetch, freeze, load, verify — one interface.

All artifacts live in data/frozen/ as per-symbol CSVs plus SHA-256
manifests (manifest_coffee.json / manifest_gold.json / manifest_rates.json).
Paths, symbol slugs, hashing, and manifest naming are internal — callers
learn freeze() / load() / verify() / latest_rate() and nothing else.

Two frozen kinds, two adapters at the same seam:
- commodity curves: coffee / gold universes (data.universe), Yahoo source
- rates: FRED SOFR snapshot

CLI:  python -m hedging_workbench.data.frozen --universe coffee
      python -m hedging_workbench.data.frozen --universe gold
      python -m hedging_workbench.data.frozen --rates
      python -m hedging_workbench.data.frozen --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from hedging_workbench.data.universe import UNIVERSES

FROZEN_DIR = Path(__file__).resolve().parents[3] / "data" / "frozen"
THROTTLE_SECONDS = 1.2
SOFR_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"


class DownloadError(RuntimeError):
    """A symbol returned no usable data."""


class RateError(RuntimeError):
    """SOFR unavailable and no frozen snapshot exists."""


def slug(symbol: str) -> str:
    return symbol.replace("=", "_").replace(".", "_")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(name: str) -> str:
    return f"manifest_{name}.json"


def _load_manifest(frozen_dir: Path, name: str) -> dict:
    return json.loads((frozen_dir / _manifest_path(name)).read_text())


def load(symbols: list[str], frozen_dir: Path = FROZEN_DIR) -> dict[str, pd.Series]:
    """Load frozen closes for the given symbols."""
    out = {}
    for symbol in symbols:
        p = frozen_dir / f"{slug(symbol)}.csv"
        df = pd.read_csv(p, index_col="date", parse_dates=True)
        out[symbol] = df["close"].astype(float)
    return out


def verify(name: str, frozen_dir: Path = FROZEN_DIR) -> list[str]:
    """Return symbols whose frozen file is missing or checksum-tampered."""
    manifest = _load_manifest(frozen_dir, name)
    bad = []
    for symbol, entry in manifest["files"].items():
        p = frozen_dir / entry["path"]
        if not p.exists() or sha256_file(p) != entry["sha256"]:
            bad.append(symbol)
    return bad


def download_series(
    symbols: list[str],
    start: str,
    end: str | None = None,
    throttle: float = THROTTLE_SECONDS,
) -> dict[str, pd.Series]:
    """Download adjusted closes, one symbol at a time, throttled."""
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for i, symbol in enumerate(symbols):
        if i:
            time.sleep(throttle)
        df = yf.download(
            symbol, start=start, end=end, progress=False, auto_adjust=False
        )
        if df is None or df.empty:
            raise DownloadError(f"no data for {symbol}")
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        series = close.dropna().astype(float)
        if series.empty:
            raise DownloadError(f"no usable closes for {symbol}")
        out[symbol] = series
    return out


def _write_manifest(
    frozen_dir: Path, name: str, start: str, files: dict[str, Path]
) -> Path:
    manifest = {
        "name": name,
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "start": start,
        "files": {
            symbol: {"path": path.name, "sha256": sha256_file(path)}
            for symbol, path in files.items()
        },
    }
    out = frozen_dir / _manifest_path(name)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def freeze(
    universe: str,
    start: str,
    frozen_dir: Path = FROZEN_DIR,
    throttle: float = THROTTLE_SECONDS,
) -> Path:
    """Download a universe, write per-symbol CSVs, write manifest."""
    frozen_dir.mkdir(parents=True, exist_ok=True)
    series = download_series(list(UNIVERSES[universe]), start, throttle=throttle)
    files = {}
    for symbol, s in series.items():
        path = frozen_dir / f"{slug(symbol)}.csv"
        s.rename("close").to_csv(path, index_label="date")
        files[symbol] = path
    return _write_manifest(frozen_dir, universe, start, files)


# -- rates adapter -----------------------------------------------------------


def fetch_sofr(start: str = "2024-01-01") -> pd.Series:
    df = pd.read_csv(
        f"{SOFR_URL}&cosd={start}", index_col="observation_date", parse_dates=True
    )
    s = df["SOFR"].dropna().astype(float)
    if s.empty:
        raise RateError("FRED returned no SOFR observations")
    return s


def freeze_rates(start: str = "2024-01-01", frozen_dir: Path = FROZEN_DIR) -> Path:
    """Fetch SOFR, freeze CSV, write rates manifest. Returns manifest path."""
    frozen_dir.mkdir(parents=True, exist_ok=True)
    s = fetch_sofr(start)
    path = frozen_dir / "SOFR.csv"
    s.rename("sofr").to_csv(path, index_label="date")
    return _write_manifest(frozen_dir, "rates", start, {"SOFR": path})


def load_sofr(frozen_dir: Path = FROZEN_DIR) -> pd.Series:
    path = frozen_dir / "SOFR.csv"
    if not path.exists():
        raise RateError(
            "no frozen SOFR — run `python -m hedging_workbench.data.frozen --rates`"
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)["sofr"].astype(float)


def latest_rate(frozen_dir: Path = FROZEN_DIR) -> float:
    """Latest published SOFR as a decimal (e.g. 0.0366)."""
    return float(load_sofr(frozen_dir).iloc[-1]) / 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=sorted(UNIVERSES))
    ap.add_argument("--rates", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--start", default="2024-01-01")
    args = ap.parse_args()
    if args.verify:
        bad = verify("coffee") + verify("gold")
        if bad:
            raise SystemExit(f"manifest checksum mismatch for {bad}")
        print("manifests verified: coffee, gold")
        return
    if args.rates:
        out = freeze_rates(args.start)
        print(f"frozen rates: {out}  latest SOFR {latest_rate():.4f}")
        return
    if args.universe:
        out = freeze(args.universe, args.start)
        print(f"frozen {args.universe}: {out}")
        return
    ap.error("nothing to do — pass --universe, --rates, or --verify")


if __name__ == "__main__":
    main()
