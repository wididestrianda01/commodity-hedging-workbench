"""Hedging Workbench — Streamlit app. Thin UI over hedging_workbench.

Design discipline: .scratch/workbench-expansion/design-app.md (minimalist-ui
adapted to Streamlit). Analytics live in the package and app/prep.py;
this module renders and caches only. Frozen data, local display only
(ICE licensing).
"""

from __future__ import annotations

import pandas as pd
import prep
import streamlit as st
from theme import STATE_GLOSSARY, badge_kind, inject_css

st.set_page_config(page_title="Hedging Workbench", page_icon=None, layout="wide")
inject_css()


def _kv_table(d: dict) -> pd.DataFrame:
    """Flat key/value table (shaping only, no analytics)."""
    return pd.DataFrame({"item": list(d), "value": list(d.values())})


curve = prep.curve_snapshot()
m = prep.market_inputs()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("**Hedging Workbench**")
    st.markdown(
        f"snapshot {curve['curve_date'].date().isoformat()} · universe "
        f"{curve['universe']} · frozen data, local display only"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("### Hedge program")
    monthly_lb = st.number_input(
        "Monthly purchase volume, lb", 10_000, 1_000_000, 100_000, step=10_000
    )
    periods = st.slider("Horizon, months", 3, 12, 12)
    put_strike = st.slider("Collar put strike, ¢/lb", 240.0, 310.0, 280.0, 5.0)
    collar_t = st.slider("Collar tenor, months", 1, 12, 3) / 12.0

    st.markdown("### Participation note")
    participation = st.slider("Participation rate", 0.0, 1.0, 0.4, 0.05)
    note_tenor = st.slider("Note tenor, years", 0.5, 3.0, 1.0, 0.25)

    st.markdown("### Collateral (risk view)")
    threshold = st.number_input("Threshold, USD", 0, 500_000, 50_000, step=25_000)
    ia = st.number_input("Independent amount, USD", 0, 500_000, 100_000, step=50_000)

    st.markdown("### Data gates")
    gates = prep.gate_snapshot()
    for name, g in gates.items():
        kind = "pos" if g["ok"] else "neg"
        label = "pass" if g["ok"] else "FAIL"
        n = len(g["files"])
        st.markdown(
            f"<div class='gate-row'>{name} <span class='badge badge-{kind}'>"
            f"{label}</span> <span class='gate-n'>{n} files</span></div>",
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner="computing program…")
def _hedge(monthly_lb, periods, put_strike, t_years):
    return prep.hedge_snapshot(monthly_lb, "2026-09-01", periods, put_strike, t_years)


@st.cache_data(show_spinner="pricing note…")
def _note(participation, tenor):
    return prep.note_snapshot(participation, m["f0"], m["sigma"], m["r"], tenor)


@st.cache_data(show_spinner="simulating book…")
def _book():
    return prep.book_snapshot()


snap_h = _hedge(monthly_lb, periods, put_strike, collar_t)
snap_n = _note(participation, note_tenor)
snap_b = _book()
col_prof = prep.collat_profile(snap_b, threshold, ia)


def snap_meta() -> str:
    return (
        f"<div class='snap-meta'>frozen snapshot "
        f"{curve['curve_date'].date().isoformat()} · Yahoo unofficial source · "
        "local display only — no redistribution</div>"
    )


# ================================================================ curve tab
tab_c, tab_h, tab_n, tab_r = st.tabs(
    ["Curve", "Hedge program", "Participation note", "Book risk"]
)

with tab_c:
    st.markdown("## Coffee futures curve")
    st.markdown(
        "Frozen ICE Coffee C chain with the implied convenience-yield term "
        "structure — carried from the cost-of-carry identity between "
        "consecutive maturities. Reproduces the carry layer of the workbench, "
        "not the Forage toy inputs."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Front {curve['front_label']} · ¢/lb", f"{curve['front_price']:,.2f}")
    c2.markdown(
        f"<div class='stat-card stat-{badge_kind(curve['state'])}'>"
        "<div class='stat-label'>Curve state</div>"
        f"<div class='stat-value'>{curve['state']}</div>"
        f"<div class='stat-note'>{STATE_GLOSSARY[curve['state']]}</div></div>",
        unsafe_allow_html=True,
    )
    c3.metric("SOFR (frozen)", f"{curve['r']:.2%}")
    c4.metric("Spread pairs", f"{len(curve['ts'])}")
    st.markdown("")
    with st.container(border=True):
        st.markdown("### Futures chain")
        st.pyplot(prep.curve_chart(curve["curve"]))
    with st.container(border=True):
        st.markdown("### Implied convenience yield")
        st.markdown(
            "y = r + storage − carry per consecutive pair, storage reported at 0 "
            "(gross of outlay). Sign is the state of that spread."
        )
        st.pyplot(prep.yield_chart(curve["ts"]))
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Chain")
            st.dataframe(
                curve["curve"][["label", "expiry", "ttm", "price"]].style.format(
                    {"ttm": "{:.2f}", "price": "{:.2f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("#### Yields")
            st.dataframe(
                curve["ts"][
                    ["near", "far", "carry", "implied_yield", "state"]
                ].style.format({"carry": "{:.2%}", "implied_yield": "{:.2%}"}),
                hide_index=True,
                use_container_width=True,
            )
    st.markdown(snap_meta(), unsafe_allow_html=True)

# ================================================================ hedge tab
with tab_h:
    p = snap_h["program"]
    c = snap_h["collar"]
    st.markdown("## Roaster hedge program")
    st.markdown(
        "Long-futures hedge for a monthly green-coffee purchase schedule, "
        "wrapped in a zero-cost collar. Mirrors the Starbucks FY2025 10-K "
        "structure: lock the C-price with futures, cap the give-up with the "
        "short call, keep the protection with the long put."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volume hedged, lb/mo", f"{monthly_lb:,.0f}")
    c2.metric(
        "Contracts, avg",
        f"{p.exposure['contracts'].mean():.2f}",
        help="37,500 lb per contract",
    )
    c3.metric("Put strike", f"{c.put_strike:,.2f} ¢/lb")
    c4.metric("Call strike (solved)", f"{c.call_strike:,.2f} ¢/lb")

    with st.container(border=True):
        st.markdown("### Collar P&L vs long futures")
        st.markdown(
            "Terminal outcome on one month's volume. Premiums cancel by "
            f"construction (gap {c.premium_gap:.1e} ¢/lb)."
        )
        st.pyplot(prep.pnl_chart(snap_h["pnl"], c))
    with st.container(border=True):
        st.markdown("### Margin-liquidity stress")
        st.markdown(
            "Peak variation-margin funding need per scenario; the collar's "
            "put floors the down-move. Collateral reference $37.9M (10-K)."
        )
        st.pyplot(prep.stress_chart(snap_h["stress"]))
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Exposure schedule")
            st.dataframe(
                p.exposure[
                    ["month", "volume_lb", "contracts", "label", "expiry", "gap"]
                ]
                .assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
                .style.format({"volume_lb": "{:,.0f}", "contracts": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("#### Roll schedule")
            rolls = p.rolls.assign(
                roll_date=lambda d: d["roll_date"].dt.strftime("%Y-%m-%d")
            )
            if len(rolls):
                st.dataframe(rolls, hide_index=True, use_container_width=True)
            else:
                st.markdown("_single contract — no rolls_")
    st.markdown(snap_meta(), unsafe_allow_html=True)

# ================================================================ note tab
with tab_n:
    n = snap_n
    cf, mc, g = n["cf"], n["mc"], n["greeks"]
    st.markdown("## Participation note")
    st.markdown(
        "Capital-protected note on KC=F: principal repaid in full, plus the "
        "participation rate times the ATM call payoff. Priced two ways — the "
        "closed form and a martingale Monte Carlo whose agreement is the "
        "trust mechanism. Issuer-side greeks: the issuer is short the call."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closed form / $1", f"{cf['price']:.4f}")
    c2.metric(
        "Monte Carlo / $1",
        f"{mc['price']:.4f}",
        help=f"95% CI [{mc['ci95'][0]:.4f}, {mc['ci95'][1]:.4f}], se {mc['se']:.2e}",
    )
    c3.metric(
        "Issuer delta", f"{g['delta']:.5f}", help="per $1 move in F0, per $1 notional"
    )
    c4.metric(
        "Issuer vega", f"{g['vega']:.4f}", help="per 1.00 vol point, per $1 notional"
    )

    agree = abs(cf["price"] - mc["price"]) < 3 * mc["se"]
    st.markdown(
        f"<span class='badge badge-{'pos' if agree else 'warn'}'>"
        f"{'closed form inside MC 95% CI' if agree else 'closed form OUTSIDE MC CI — inspect'}</span>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("### Payoff at maturity")
        st.pyplot(prep.payoff_chart(n["payoff"]))
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Decomposition")
            dec = {
                "bond (discounted zero)": cf["bond"],
                "participation value": cf["participation_value"],
                "price": cf["price"],
            }
            st.dataframe(_kv_table(dec), hide_index=True, use_container_width=True)
        with right:
            st.markdown("#### Disclosure")
            st.markdown(
                "Target market: institutional or professional investors with "
                "a directional view on coffee and capacity to bear full "
                "principal risk in the issuer's default. Priorities: capital "
                "protection is the ISSUER's credit, not a guarantee. "
                "MiFID II / PRIIPs context — mapped, not asserted satisfied."
            )
    st.markdown(snap_meta(), unsafe_allow_html=True)

# ================================================================ risk tab
with tab_r:
    st.markdown("## Book risk & junior xVA")
    st.markdown(
        "The hedge book seen as counterparty exposure: historical VaR/ES on "
        "USD daily P&L, EE/PFE from the martingale simulation, unilateral CVA "
        "with the QuantLib convention benchmark, and what netting + collateral "
        "buy back."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95% (daily)", f"${snap_b['var_tab'].iloc[0]['var']:,.0f}")
    c2.metric("ES 99% (daily)", f"${snap_b['var_tab'].iloc[1]['es']:,.0f}")
    c3.metric("EPE (1y)", f"${snap_b['epe']:,.0f}")
    c4.metric(
        "CVA (unilateral)",
        f"${snap_b['cva_pkg']:,.2f}",
        help=f"QuantLib benchmark ${snap_b['cva_ql']:,.2f}, "
        f"rel diff {snap_b['cva_rel']:.1e}",
    )

    with st.container(border=True):
        st.markdown("### Exposure profile")
        st.markdown(
            f"Collateral haircut applied: threshold ${threshold:,.0f}, IA "
            f"${ia:,.0f}. Netted line: roaster long vs the Phase 4 issuer's "
            "offsetting short in one netting set."
        )
        st.pyplot(prep.exposure_chart(col_prof, snap_b["net_prof"]))
    with st.container(border=True):
        st.dataframe(
            snap_b["var_tab"].style.format("{:,.2f}"),
            use_container_width=True,
        )
    st.markdown(snap_meta(), unsafe_allow_html=True)
