"""Refresh job: re-run data gates, freeze the passing universe + SOFR into a
gated snapshot artifact bundle — the packaged job the Phase 8 Lambda runs.

Fail-closed: gate failure raises and writes nothing. Gold is downloaded only
if coffee fails its gates.

CLI:  python -m hedging_workbench.refresh --out /out
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hedging_workbench.data import frozen, gates
from hedging_workbench.data.universe import UNIVERSES


def refresh(start: str = "2024-01-01", out_dir: str | Path = "/out") -> Path:
    """Gate the universes, then write the artifact bundle into `out_dir`:
    passing-universe CSVs + manifest, SOFR + rates manifest, gate report.
    Returns the out dir. Raises (nothing written) on gate/download failure.
    """
    coffee = frozen.download_series(list(UNIVERSES["coffee"]), start)
    grabbed: dict[str, dict] = {}

    def gold() -> dict:
        grabbed["series"] = frozen.download_series(list(UNIVERSES["gold"]), start)
        return grabbed["series"]

    report = gates.evaluate(coffee, gold)
    series = coffee if report.universe == "coffee" else grabbed["series"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for symbol, s in series.items():
        path = out / f"{frozen.slug(symbol)}.csv"
        s.rename("close").to_csv(path, index_label="date")
        files[symbol] = path
    frozen._write_manifest(out, report.universe, start, files)
    (out / "gate_report.json").write_text(
        json.dumps(
            {"summary": report.summary(), **asdict(report)}, indent=2, default=str
        )
        + "\n"
    )
    rates_path = out / "SOFR.csv"
    frozen.fetch_sofr(start).rename("sofr").to_csv(rates_path, index_label="date")
    frozen._write_manifest(out, "rates", start, {"SOFR": rates_path})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--out", default="/out")
    args = ap.parse_args()
    out = refresh(args.start, args.out)
    print(f"gated snapshot bundle: {out}")
    print((out / "gate_report.json").read_text())


if __name__ == "__main__":
    main()
