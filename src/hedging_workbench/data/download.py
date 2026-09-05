"""Throttled Yahoo download, freeze to CSV, SHA-256 manifest.

CLI:  python -m hedging_workbench.data.download --universe coffee --start 2024-01-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

FROZEN_DIR = Path(__file__).resolve().parents[3] / "data" / "frozen"
THROTTLE_SECONDS = 1.2


class DownloadError(RuntimeError):
    """A symbol returned no usable data."""


def slug(symbol: str) -> str:
    return symbol.replace("=", "_").replace(".", "_")


def manifest_path(name: str) -> str:
    return f"manifest_{name}.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(frozen_dir: Path, name: str) -> dict:
    return json.loads((frozen_dir / manifest_path(name)).read_text())


def verify_manifest(frozen_dir: Path, name: str) -> list[str]:
    """Return symbols whose frozen file is missing or checksum-tampered."""
    manifest = load_manifest(frozen_dir, name)
    bad = []
    for symbol, entry in manifest["files"].items():
        p = frozen_dir / entry["path"]
        if not p.exists() or sha256_file(p) != entry["sha256"]:
            bad.append(symbol)
    return bad


def download_series(symbols: list[str], start: str, end: str | None = None,
                    throttle: float = THROTTLE_SECONDS) -> dict[str, pd.Series]:
    """Download adjusted closes, one symbol at a time, throttled."""
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for i, symbol in enumerate(symbols):
        if i:
            time.sleep(throttle)
        df = yf.download(symbol, start=start, end=end, progress=False,
                         auto_adjust=False)
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


def freeze(symbols: list[str], start: str, frozen_dir: Path = FROZEN_DIR,
           name: str = "default", throttle: float = THROTTLE_SECONDS) -> Path:
    """Download, write per-symbol CSVs, write manifest. Returns manifest path."""
    frozen_dir.mkdir(parents=True, exist_ok=True)
    series = download_series(symbols, start, throttle=throttle)
    for symbol, s in series.items():
        s.rename("close").to_csv(frozen_dir / f"{slug(symbol)}.csv",
                                 index_label="date")
    manifest = {
        "name": name,
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "start": start,
        "files": {
            symbol: {
                "path": f"{slug(symbol)}.csv",
                "sha256": sha256_file(frozen_dir / f"{slug(symbol)}.csv"),
            }
            for symbol in symbols
        },
    }
    out = frozen_dir / manifest_path(name)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def load_frozen(symbols: list[str], frozen_dir: Path = FROZEN_DIR) -> dict[str, pd.Series]:
    """Load frozen closes for the given symbols."""
    out = {}
    for symbol in symbols:
        p = frozen_dir / f"{slug(symbol)}.csv"
        df = pd.read_csv(p, index_col="date", parse_dates=True)
        out[symbol] = df["close"].astype(float)
    return out


def main() -> None:
    from hedging_workbench.data.universe import UNIVERSES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=sorted(UNIVERSES), required=True)
    ap.add_argument("--start", default="2024-01-01")
    args = ap.parse_args()
    manifest = freeze(list(UNIVERSES[args.universe]), args.start, name=args.universe)
    print(f"frozen {args.universe}: {manifest}")


if __name__ == "__main__":
    main()
