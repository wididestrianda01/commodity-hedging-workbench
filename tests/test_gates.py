"""Seam 1 + 2 tests: manifest checksums, gate pass/fail, fallback trigger."""

import json

import numpy as np
import pandas as pd
import pytest

from hedging_workbench.data import gates as G
from hedging_workbench.data.download import sha256_file, verify_manifest


def make_series(n=300, price=300.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(price + rng.normal(0, 1, n).cumsum() * 0.1, index=idx)


def test_gates_pass_on_clean_coffee():
    report = G.run_gates({"KC=F": make_series(), "KCH27.NYB": make_series(seed=1)},
                         "coffee")
    assert report.ok and not report.fallback_used and not report.failures


def test_gate_fails_on_short_series():
    report = G.run_gates({"KC=F": make_series(), "KCH27.NYB": make_series(n=10)},
                         "coffee")
    assert not report.ok
    assert any("KCH27.NYB" in f and "rows" in f for f in report.failures)


def test_gate_fails_on_corrupt_price():
    s = make_series()
    s.iloc[5] = -1.0
    report = G.run_gates({"KC=F": s}, "coffee")
    assert not report.ok
    assert any("non-positive" in f for f in report.failures)


def test_fallback_to_gold_on_coffee_gap():
    coffee = {"KC=F": make_series(), "KCH27.NYB": make_series(n=3)}
    gold = {"GC=F": make_series(price=4500.0), "GCZ27.CMX": make_series(price=4600.0)}
    report = G.evaluate(coffee, gold)
    assert report.ok and report.fallback_used and report.universe == "gold"
    assert "coffee failed" in report.reason


def test_fail_closed_when_both_universes_fail():
    with pytest.raises(G.GateError) as exc:
        G.evaluate({"KC=F": make_series(n=2)}, {"GC=F": make_series(n=2)})
    assert "fail closed" in str(exc.value)


def test_manifest_tamper_detected(tmp_path):
    f = tmp_path / "KC_F.csv"
    f.write_text("date,close\n2024-01-02,300.0\n")
    (tmp_path / "manifest_coffee.json").write_text(json.dumps(
        {"files": {"KC=F": {"path": "KC_F.csv",
                            "sha256": sha256_file(f)}}}))
    assert verify_manifest(tmp_path, "coffee") == []
    f.write_text("date,close\n2024-01-02,999.0\n")   # tamper
    assert verify_manifest(tmp_path, "coffee") == ["KC=F"]
