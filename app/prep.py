"""Headless prep layer for the app: package calls only, no Streamlit.

Every view section gets a prep function returning plain pandas objects /
figures so unit tests run without a Streamlit runtime. Rendering (app.py)
is thin. Parameter defaults mirror the notebooks so the app reproduces
their numbers at default inputs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hedging_workbench.carry import curve_state, load_curve, yield_term_structure
from hedging_workbench.cva import cva_quantlib, cva_unilateral
from hedging_workbench.data.frozen import latest_rate, load, manifest, verify
from hedging_workbench.exposure import (
    FuturesBook,
    collateralized,
    epe,
    profile,
    simulate_front,
)
from hedging_workbench.hedge import build_program
from hedging_workbench.note import ParticipationNote
from hedging_workbench.pricing import collar_vs_futures, zero_cost_collar
from hedging_workbench.risk import var_es
from hedging_workbench.conventions import lb_to_usd
from hedging_workbench.cva import HAZARD, LGD
from hedging_workbench.stress import COLLATERAL_REFERENCE, stress_report
from hedging_workbench.vol import vol_from_frozen


def gate_snapshot() -> dict:
    """Data-gate status per universe: verified files + tampered symbols."""

    out = {}
    for name in ("coffee", "gold", "rates"):
        try:
            mf = manifest(name)
            bad = verify(name)
            out[name] = {
                "ok": not bad,
                "bad": bad,
                "files": [
                    {
                        "symbol": sym,
                        "path": entry["path"],
                        "sha256": entry["sha256"][:12],
                    }
                    for sym, entry in mf["files"].items()
                ],
            }
        except FileNotFoundError:
            out[name] = {"ok": False, "bad": ["manifest missing"], "files": []}
    return out


def curve_snapshot(universe: str = "coffee") -> dict:
    """Frozen curve + implied yields + headline numbers for the curve view."""
    curve = load_curve(universe)
    ts = yield_term_structure(curve)
    front = curve.iloc[0]
    return {
        "curve": curve,
        "ts": ts,
        "state": curve_state(ts),
        "front_price": float(front["price"]),
        "front_label": front["label"],
        "curve_date": curve.attrs["curve_date"],
        "r": ts.attrs["r"],
        "universe": universe,
    }


def market_inputs() -> dict:
    """Frozen marks shared by every view: f0, GARCH vol, SOFR, KC=F path."""
    curve = load_curve("coffee")
    _, vol_fit = vol_from_frozen()
    return {
        "f0": float(curve["price"].iloc[0]),
        "sigma": float(vol_fit.garch_last) / 100.0,
        "r": latest_rate(),
        "kc": load(["KC=F"])["KC=F"].dropna(),
    }


def hedge_snapshot(
    monthly_lb: float,
    start: str,
    periods: int,
    put_strike: float,
    t_years: float,
) -> dict:
    """Roaster hedge program + collar + P&L grid + margin stress."""
    m = market_inputs()
    program = build_program(monthly_lb, start, periods, universe="coffee")
    collar = zero_cost_collar(m["f0"], put_strike, t_years, m["sigma"], m["r"])
    grid = np.linspace(200.0, 400.0, 201)
    pnl = pd.DataFrame(
        {
            "f": grid,
            "futures_usd": (grid - m["f0"]) * lb_to_usd(monthly_lb),
            "collar_usd": [
                collar_vs_futures(monthly_lb, collar, x)["collar_usd"] for x in grid
            ],
        }
    )
    stress = stress_report(
        program.exposure["contracts"].sum(),
        m["kc"],
        m["sigma"],
        collar=collar,
        collateral_usd=COLLATERAL_REFERENCE,
    )
    return {
        "program": program,
        "collar": collar,
        "pnl": pnl,
        "stress": stress,
        "f0": m["f0"],
        "sigma": m["sigma"],
        "r": m["r"],
        "monthly_lb": monthly_lb,
    }


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#EAEAEA")
    ax.tick_params(colors="#787774", labelsize=9)
    ax.grid(True, axis="y", color="#EAEAEA", linewidth=1)
    ax.set_axisbelow(True)


def curve_chart(curve) -> plt.Figure:
    """Futures prices along the frozen chain, ink line, no chartjunk."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    ax.plot(
        curve["label"],
        curve["price"],
        color="#2F3437",
        linewidth=1.5,
        marker="o",
        markersize=5,
        markerfacecolor="#FBFBFA",
        markeredgecolor="#2F3437",
    )
    _style(ax)
    ax.set_ylabel("¢/lb", color="#787774", fontsize=9)
    fig.tight_layout()
    return fig


def yield_chart(ts) -> plt.Figure:
    """Implied convenience yield per spread pair; signed bars in pastels."""
    colors = ["#346538" if v >= 0 else "#9F2F2D" for v in ts["implied_yield"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    ax.bar(range(len(ts)), ts["implied_yield"], color=colors, alpha=0.75, width=0.55)
    _style(ax)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(
        [
            f"{n.split()[0]}{n[-2:]}–{f.split()[0]}{f[-2:]}"
            for n, f in zip(ts["near"], ts["far"])
        ],
        fontsize=8,
    )
    ax.set_ylabel("y (annualised)", color="#787774", fontsize=9)
    fig.tight_layout()
    return fig


def pnl_chart(pnl: pd.DataFrame, collar) -> plt.Figure:
    """Terminal P&L: long futures vs collar, strikes marked."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    ax.plot(
        pnl["f"],
        pnl["futures_usd"],
        color="#2F3437",
        linewidth=1.5,
        label="long futures",
    )
    ax.plot(
        pnl["f"],
        pnl["collar_usd"],
        color="#1F6C9F",
        linewidth=1.5,
        label="zero-cost collar",
    )
    for x in (collar.put_strike, collar.call_strike):
        ax.axvline(x, color="#EAEAEA", linewidth=1)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor="#787774")
    ax.set_xlabel("terminal price, ¢/lb", color="#787774", fontsize=9)
    ax.set_ylabel("P&L, USD", color="#787774", fontsize=9)
    fig.tight_layout()
    return fig


def stress_chart(stress: list) -> plt.Figure:
    """Peak funding need per scenario: futures vs collar bars."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    x = np.arange(len(stress))
    w = 0.38
    ax.bar(
        x - w / 2,
        [s.peak_need_usd for s in stress],
        w,
        color="#9F2F2D",
        alpha=0.75,
        label="long futures",
    )
    ax.bar(
        x + w / 2,
        [s.collar_peak_need_usd for s in stress],
        w,
        color="#346538",
        alpha=0.75,
        label="collar",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s.scenario for s in stress], fontsize=8)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor="#787774")
    ax.set_ylabel("peak funding need, USD", color="#787774", fontsize=9)
    fig.tight_layout()
    return fig


def note_snapshot(
    participation: float, f0: float, sigma: float, r: float, tenor: float = 1.0
) -> dict:
    """Participation note priced two ways + issuer greeks + payoff curve."""
    note = ParticipationNote(1_000_000, f0, tenor, participation)
    cf = note.closed_form(sigma, r)
    mc = note.mc_price(sigma, r, n_paths=100_000)
    greeks = note.greeks(sigma, r)
    grid = np.linspace(0.6 * f0, 1.6 * f0, 201)
    payoff = pd.DataFrame(
        {
            "f": grid,
            "payoff": [
                1.0 + participation * max(x - note.strike, 0.0) / f0 for x in grid
            ],
        }
    )
    return {
        "note": note,
        "cf": cf,
        "mc": mc,
        "greeks": greeks,
        "payoff": payoff,
        "f0": f0,
        "tenor": tenor,
    }


def payoff_chart(payoff: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    ax.plot(payoff["f"], payoff["payoff"], color="#2F3437", linewidth=1.5)
    _style(ax)
    ax.set_xlabel("F_T, ¢/lb", color="#787774", fontsize=9)
    ax.set_ylabel("return per $1 notional", color="#787774", fontsize=9)
    fig.tight_layout()
    return fig


def book_snapshot(seed: int = 42) -> dict:
    """Book risk: VaR/ES, EE/PFE profile, CVA + QuantLib benchmark.

    Book = one month's program size marked on the front price (notebook
    04's junior-view simplification).
    """
    m = market_inputs()
    program = build_program(100_000, "2026-09-01", 12, universe="coffee")
    contracts = float(program.exposure["contracts"].iloc[0])
    book = FuturesBook(contracts, m["f0"])

    rets, _ = vol_from_frozen()
    pnl_book = FuturesBook(contracts, m["f0"]).pnl_from_returns(rets)
    var_tab = var_es(pnl_book)

    H, STEPS, N = 1.0, 252, 50_000
    f_paths = simulate_front(m["f0"], m["sigma"], H, STEPS, N, seed=seed)
    mtm = book.mtm(f_paths)
    prof = profile(mtm)

    idx = np.linspace(0, STEPS, 13).round().astype(int)[1:]
    tenors = prof["t"].to_numpy()[idx]
    ee_grid = prof["ee"].to_numpy()[idx]
    cva_pkg = cva_unilateral(ee_grid, tenors, HAZARD, LGD, m["r"])
    cva_ql = cva_quantlib(ee_grid, tenors, HAZARD, LGD, m["r"])

    # netting set: roaster long + the Phase 4 issuer's offsetting short
    short = FuturesBook(contracts, m["f0"]).mtm(f_paths)
    net_prof = profile(mtm - short)

    return {
        "var_tab": var_tab,
        "prof": prof,
        "net_prof": net_prof,
        "epe": epe(prof),
        "cva_pkg": cva_pkg,
        "cva_ql": cva_ql,
        "cva_rel": abs(cva_pkg - cva_ql) / cva_ql,
        "tenors": tenors,
        "ee_grid": ee_grid,
        "mtm": mtm,
        "contracts": contracts,
        "f0": m["f0"],
    }


def collat_profile(snap: dict, threshold: float, ia: float) -> pd.DataFrame:
    """EE/PFE after collateral for the given threshold + IA."""
    return profile(collateralized(snap["mtm"], threshold, ia))


def exposure_chart(
    prof: pd.DataFrame, net_prof: pd.DataFrame | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    ax.plot(prof["t"] * 12, prof["ee"], color="#2F3437", linewidth=1.5, label="EE")
    ax.plot(
        prof["t"] * 12,
        prof["pfe"],
        color="#9F2F2D",
        linewidth=1.5,
        alpha=0.8,
        label="PFE 95%",
    )
    if net_prof is not None:
        ax.plot(
            net_prof["t"] * 12,
            net_prof["ee"],
            color="#346538",
            linewidth=1.5,
            alpha=0.8,
            label="EE netted",
        )
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor="#787774")
    ax.set_xlabel("months ahead", color="#787774", fontsize=9)
    ax.set_ylabel("USD exposure", color="#787774", fontsize=9)
    fig.tight_layout()
    return fig
