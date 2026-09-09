"""Refresh job: gates -> gated snapshot bundle, fail-closed. Container-level
behavior is exercised by the 8-01 smoke; these tests pin the contract."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hedging_workbench.data import frozen, gates
from hedging_workbench.refresh import refresh

from hedging_workbench.data.universe import UNIVERSES


def _real_series(universe: str) -> dict[str, pd.Series]:
    """Frozen series as a passing fixture (already gate-clean in repo)."""
    bad = frozen.verify(universe)
    assert not bad, f"frozen {universe} tampered: {bad}"
    return frozen.load(list(UNIVERSES[universe]))


def _short(series: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Same universe, failing gates (too few rows)."""
    return {k: s.iloc[:10] for k, s in series.items()}


@pytest.fixture()
def sofr(monkeypatch):
    s = frozen.load_sofr()
    monkeypatch.setattr(frozen, "fetch_sofr", lambda start="2024-01-01": s)


def test_coffee_pass_bundle(tmp_path, monkeypatch, sofr):
    coffee = _real_series("coffee")
    downloads = []
    monkeypatch.setattr(
        frozen,
        "download_series",
        lambda symbols, start, **kw: (downloads.append(symbols[0]), coffee)[1],
    )
    out = refresh(out_dir=tmp_path / "bundle")
    assert out.exists()
    bad = frozen.verify("coffee", out)
    assert not bad, bad
    report = json.loads((out / "gate_report.json").read_text())
    assert report["universe"] == "coffee" and report["ok"]
    assert frozen.load_sofr(out).index.is_monotonic_increasing
    assert downloads == ["KC=F"], "gold must not be downloaded when coffee passes"


def test_gold_fallback(tmp_path, monkeypatch, sofr):
    coffee, gold = _short(_real_series("coffee")), _real_series("gold")
    monkeypatch.setattr(
        frozen,
        "download_series",
        lambda symbols, start, **kw: coffee if symbols[0].startswith("KC") else gold,
    )
    out = refresh(out_dir=tmp_path / "bundle")
    bad = frozen.verify("gold", out)
    assert not bad, bad
    report = json.loads((out / "gate_report.json").read_text())
    assert report["universe"] == "gold" and report["fallback_used"]


def test_fail_closed_writes_nothing(tmp_path, monkeypatch, sofr):
    coffee, gold = _short(_real_series("coffee")), _short(_real_series("gold"))
    monkeypatch.setattr(
        frozen,
        "download_series",
        lambda symbols, start, **kw: coffee if symbols[0].startswith("KC") else gold,
    )
    out = tmp_path / "bundle"
    with pytest.raises(gates.GateError):
        refresh(out_dir=out)
    assert not out.exists(), "fail-closed: no artifact dir on gate failure"
