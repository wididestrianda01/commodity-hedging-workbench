"""App prep layer: headless data prep and chart builders (ticket 7-01)."""

import matplotlib

matplotlib.use("Agg")

import prep
from prep import curve_chart, curve_snapshot, yield_chart

M = prep.market_inputs()


def test_hedge_snapshot_matches_notebook():
    s = prep.hedge_snapshot(100_000, "2026-09-01", 12, 280.0, 0.25)
    c = s["collar"]
    # zero-cost by construction; notebook reference strikes
    assert c.premium_gap < 1e-8
    assert abs(c.call_strike - 382.06) < 0.01
    assert len(s["program"].exposure) == 12
    assert not s["program"].exposure["gap"].any()
    # collar floors the down-move in every scenario
    for sc in s["stress"]:
        assert sc.collar_peak_need_usd <= sc.peak_need_usd + 1e-6


def test_note_snapshot_two_way_agreement():
    n = prep.note_snapshot(0.4, M["f0"], M["sigma"], M["r"])
    lo, hi = n["mc"]["ci95"]
    assert lo <= n["cf"]["price"] <= hi
    assert n["cf"]["bond"] > 0 and n["cf"]["participation_value"] > 0
    # issuer is short the call: both greeks negative
    assert n["greeks"]["delta"] < 0 and n["greeks"]["vega"] < 0


def test_book_snapshot_and_benchmark():
    b = prep.book_snapshot()
    assert b["cva_rel"] < 1e-9  # QuantLib convention benchmark
    assert (b["var_tab"]["es"] >= b["var_tab"]["var"]).all()
    assert b["epe"] > 0
    # collateral strictly reduces the exposure profile
    col = prep.collat_profile(b, 50_000.0, 100_000.0)
    assert (col["ee"] <= b["prof"]["ee"]).all()
    # netting the offsetting short never adds exposure
    assert (b["net_prof"]["ee"] <= b["prof"]["ee"]).all()


def test_curve_snapshot_shape():
    snap = curve_snapshot()
    assert len(snap["curve"]) == 8
    assert len(snap["ts"]) == 7
    assert snap["state"] in {"contango", "backwardation", "mixed"}
    assert snap["front_price"] > 0
    assert isinstance(snap["r"], float)
    assert snap["curve"].attrs["universe"] == "coffee"


def test_curve_snapshot_ts_consistent_with_curve():
    snap = curve_snapshot()
    # implied_yield satisfies the carry identity: y = r + storage - carry
    r = snap["r"]
    for row in snap["ts"].itertuples():
        assert abs((r - row.carry) - row.implied_yield) < 1e-12


def test_charts_build_figures():
    snap = curve_snapshot()
    fig1 = curve_chart(snap["curve"])
    fig2 = yield_chart(snap["ts"])
    assert fig1.axes and fig2.axes
    assert len(fig2.axes[0].patches) == 7


def test_gate_snapshot_all_pass():
    g = prep.gate_snapshot()
    assert {k: v["ok"] for k, v in g.items()} == {
        "coffee": True,
        "gold": True,
        "rates": True,
    }
    assert len(g["coffee"]["files"]) == 9
    assert all(len(f["sha256"]) == 12 for v in g.values() for f in v["files"])
