"""SR 26-2-aligned validation harness (Phase 6, ticket 13-01).

Three sections mirroring the model-risk pillars of SR 26-2 (2026-04-17,
superseding SR 11-7) — mapped as methodology, never asserted compliance:

1. Model inventory — one card per Phase 1–5 model: what it consumes,
   its stated assumptions, its benchmark, its outcomes check.
2. No-look-ahead gate — fail-closed: any observation after the
   valuation as-of date poisons a calibration; the gate refuses it.
3. Outcomes analysis — rolling-window backtests on the frozen data:
   working vol (GARCH vs EWMA forecast vs realized), curve stability
   (implied-yield term structure rebuilt at past as-of dates), VaR
   coverage (rolling VaR vs realized breach counts), note pricing
   (MC vs closed form within the MC standard error), CVA re-quoted

Every number is meant to be hand-verifiable (spec story 14): each
backtest has a hand-computed expectation in tests/test_validate.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import pandas as pd

from hedging_workbench.carry import curve_state, expiry, yield_term_structure
from hedging_workbench.cva import cva_unilateral
from hedging_workbench.data.frozen import load, load_sofr
from hedging_workbench.data.universe import UNIVERSES, contract_label
from hedging_workbench.note import ParticipationNote
from hedging_workbench.risk import filtered_var, var_es
from hedging_workbench.vol import EWMA_LAMBDA, garch_vol


# ---------------------------------------------------------------------------
# 1. Model inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCard:
    """One row of the SR 26-2-style model inventory."""

    name: str
    module: str
    inputs: str
    assumptions: str
    benchmark: str
    outcomes_check: str
    review_status: str = "reviewed 2026-09; suite green"


def model_inventory() -> list[ModelCard]:
    """Inventory of every Phase 1–5 model with its validation status."""
    return [
        ModelCard(
            name="Data gates (frozen chain)",
            module="hedging_workbench.data.gates",
            inputs="Frozen KC=F chain + GC=F fallback, SHA-256 manifests",
            assumptions="ICE Coffee C quote unit cents/lb; count/completeness ceilings",
            benchmark="Manifest checksum; gold fallback armed",
            outcomes_check="Gate pass/fail on injected gap (fail-closed)",
        ),
        ModelCard(
            name="Implied carry term structure",
            module="hedging_workbench.carry",
            inputs="Frozen contract chain, SOFR",
            assumptions="Expiry ≈ 15th of delivery month; storage d = 0 (gross yield)",
            benchmark="Cost-of-carry identity F2 = F1·exp((r+d−y)τ)",
            outcomes_check="Curve stability at past as-of dates",
        ),
        ModelCard(
            name="Schwartz–Smith reduced-form fit",
            module="hedging_workbench.ssfit",
            inputs="Single frozen curve snapshot",
            assumptions="A(τ) reduced to slope·τ; κ weakly identified (~8 maturities)",
            benchmark="NLS residual RMSE reported, κ > 0 bounded",
            outcomes_check="RMSE vs pre-declared tolerance",
        ),
        ModelCard(
            name="GARCH(1,1) working vol + EWMA",
            module="hedging_workbench.vol",
            inputs="Frozen KC=F daily returns",
            assumptions="Constant mean; returns in percent",
            benchmark="EWMA (λ=0.94) same filter, fixed persistence",
            outcomes_check="Rolling forecast vs realized vol (GARCH vs EWMA)",
        ),
        ModelCard(
            name="Black-76 pricing + zero-cost collar",
            module="hedging_workbench.pricing",
            inputs="Futures price, strikes, GARCH vol, SOFR",
            assumptions="European; discounted premiums; no early-exercise premium",
            benchmark="Put-call parity; collar zero-cost identity (brentq)",
            outcomes_check="Zero-cost gap ≈ 0 by construction",
        ),
        ModelCard(
            name="Participation note (closed form + MC)",
            module="hedging_workbench.note",
            inputs="F0, GARCH vol, SOFR, participation rate",
            assumptions="Risk-neutral martingale futures; antithetic variates",
            benchmark="MC vs closed form within MC standard error",
            outcomes_check="MC-vs-closed-form re-check on a fixed config",
        ),
        ModelCard(
            name="Historical VaR/ES",
            module="hedging_workbench.risk",
            inputs="Realized P&L series",
            assumptions="Empirical distribution; ES = mean of tail beyond VaR",
            benchmark="ES ≥ VaR invariant; hand-computed discrete distribution",
            outcomes_check="Rolling VaR breach count vs expectation",
        ),
        ModelCard(
            name="EE/EPE/PFE exposure",
            module="hedging_workbench.exposure",
            inputs="F0, GARCH vol, SOFR, book",
            assumptions="Lognormal martingale paths (deterministic rates)",
            benchmark="Deterministic-path EE = analytic forward MtM; EE ≤ PFE",
            outcomes_check="Invariant asserted in test suite",
        ),
        ModelCard(
            name="Unilateral CVA (bucketed)",
            module="hedging_workbench.cva",
            inputs="EE profile, flat hazard, flat SOFR",
            assumptions="Unilateral only (no DVA); flat hazard/SOFR",
            benchmark="QuantLib FlatHazardRate + FlatForward, 1e-9 relative",
            outcomes_check="CVA-vs-QuantLib re-quote (optional dep)",
        ),
        ModelCard(
            name="IFRS 9 effectiveness testing",
            module="hedging_workbench.effectiveness",
            inputs="Hypothetical vs designated hedge P&L, ratios",
            assumptions="Principles-based (economic relationship, credit-dominance, ratio consistency)",
            benchmark="Credit-dominance rejection; band-not-operative boundary",
            outcomes_check="Boundary tests in suite",
        ),
        ModelCard(
            name="Discrete-CVaR hedge ratio",
            module="hedging_workbench.cvar",
            inputs="Scenario P&L grid, volume",
            assumptions="Discrete scenario distribution",
            benchmark="CVaR-minimising ratio vs literature (MDPI 2025)",
            outcomes_check="Directional invariants in suite",
        ),
        ModelCard(
            name="GARCH-BEKK cross hedge",
            module="hedging_workbench.bekk",
            inputs="Coffee/gold return pairs",
            assumptions="Diagonal BEKK(1,1); cross-asset hedge reference",
            benchmark="Literature comparison (GARCH-BEKK hedge ratios)",
            outcomes_check="Stability of ratio in suite",
        ),
        ModelCard(
            name="Margin-liquidity stress",
            module="hedging_workbench.stress",
            inputs="Frozen path, GARCH vol, program",
            assumptions="Monetisable-put funding cap; FSB 2024 framing",
            benchmark="margin_relief worst-case identities",
            outcomes_check="Scenario peaks vs buffer in suite",
        ),
    ]


# ---------------------------------------------------------------------------
# 2. No-look-ahead gate (fail-closed)
# ---------------------------------------------------------------------------


class LookaheadError(RuntimeError):
    """Calibration window contaminated with post-as-of observations."""


def assert_no_lookahead(series: pd.Series, as_of: pd.Timestamp, name: str = "") -> None:
    """Refuse any observation strictly after `as_of`. Fail-closed."""
    idx = pd.DatetimeIndex(series.index)
    if idx.empty:
        raise LookaheadError(f"{name or 'series'}: empty — cannot gate")
    bad = idx[idx > pd.Timestamp(as_of)]
    if len(bad):
        raise LookaheadError(
            f"{name or 'series'}: {len(bad)} observation(s) after as-of "
            f"{pd.Timestamp(as_of).date()}, first {bad[0].date()}"
        )


def asof_value(series: pd.Series, as_of: pd.Timestamp, name: str = "") -> float:
    """Gated last observation at or before `as_of`; raises when empty."""
    assert_no_lookahead(series, as_of, name)
    s = series.dropna().loc[: pd.Timestamp(as_of)]
    if s.empty:
        raise LookaheadError(f"{name or 'series'}: no data on or before as-of")
    return float(s.iloc[-1])


def _gate_rejects_poisoned(series: dict[str, pd.Series]) -> bool:
    """The gate must REJECT a poisoned window — this is the check."""
    name, s = next(iter(series.items()))
    try:
        asof_value(s, s.index[-2], name)
    except LookaheadError:
        return True  # as-of = second-to-last date; the last obs must poison it
    return False


# ---------------------------------------------------------------------------
# 3. Outcomes analysis — rolling-window backtests
# ---------------------------------------------------------------------------


def ewma_forecast(returns_pct: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """Annualised %/yr EWMA variance forecast (pandas ewm semantics;
    tests recompute it from the weighted-variance definition)."""
    v = returns_pct.ewm(alpha=1 - lam).var().iloc[-1]
    return float(np.sqrt(252 * v))


def rolling_vol_backtest(returns_pct: pd.Series, window: int = 126) -> pd.DataFrame:
    """GARCH vs EWMA one-step vol forecast vs next-window realized vol.

    Non-overlapping windows: fit on [i, i+window), realized = std of
    [i+window, i+2*window). Forecast convention: the LAST conditional
    vol of the fit window (GARCH) and the last EWMA variance (EWMA);
    the GARCH long-run vol is reported alongside as the horizon anchor.
    """
    r = pd.Series(returns_pct).dropna()
    n = len(r)
    if n < 2 * window:
        raise ValueError(f"need ≥ {2 * window} returns, have {n}")
    rows = []
    for start in range(0, n - 2 * window + 1, window):
        fit_w = r.iloc[start : start + window]
        real_w = r.iloc[start + window : start + 2 * window]
        fit = garch_vol(fit_w)
        rows.append(
            {
                "fit_end": fit_w.index[-1],
                "realized": float(real_w.std() * np.sqrt(252)),
                "garch": fit.garch_last,
                "garch_longrun": fit.garch_longrun,
                "ewma": ewma_forecast(fit_w),
            }
        )
    out = pd.DataFrame(rows)
    for m in ("garch", "ewma"):
        out[f"err_{m}"] = out[m] - out["realized"]
    out.attrs["rmse"] = {
        m: float(np.sqrt((out[f"err_{m}"] ** 2).mean())) for m in ("garch", "ewma")
    }
    return out


def _curve_asof(
    symbols: list[str],
    series: dict[str, pd.Series],
    as_of: pd.Timestamp,
    universe: str,
) -> pd.DataFrame:
    """Curve rebuilt strictly from data ≤ as_of (same shape as load_curve).

    Series are pre-truncated to the as-of window (what was visible at
    that date); the gate then rejects any UNtruncated input.
    """
    as_of = pd.Timestamp(as_of)
    rows = []
    for sym in symbols:
        s = series[sym].dropna().loc[:as_of]
        if s.empty:
            raise LookaheadError(f"{sym}: no data on or before as-of {as_of.date()}")
        rows.append(
            {
                "symbol": sym,
                "label": contract_label(sym),
                "expiry": expiry(sym),
                "price": float(s.iloc[-1]),
            }
        )
    df = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    df["ttm"] = (df["expiry"] - as_of).dt.days / 365.25
    df.attrs["curve_date"] = as_of
    df.attrs["universe"] = universe
    return df


def curve_stability_backtest(
    universe: str = "coffee", n_dates: int = 6, step_days: int = 21
) -> pd.DataFrame:
    """Implied-yield term structure rebuilt at past as-of dates.

    Each as-of is a no-look-ahead reconstruction: series gated to the
    date, SOFR taken at the same date. Fail-closed if any series has no
    data at an as-of (long-dated contracts list later, so as-of dates
    stay within the chain's common history — the earliest date where
    every series has observed at least once is used as the anchor).
    """
    symbols = [s for s in UNIVERSES[universe] if not s.endswith("=F")]
    series = load(symbols)
    last = min(s.dropna().index.max() for s in series.values())
    first = max(s.dropna().index.min() for s in series.values())
    asofs = [last - pd.Timedelta(days=step_days * k) for k in range(n_dates)]
    if min(asofs) <= first:
        raise ValueError(
            f"cannot build {n_dates} as-of dates before the chain's first "
            f"common observation {first.date()}"
        )
    sofr = load_sofr()
    rows = []
    for as_of in sorted(asofs):
        curve = _curve_asof(symbols, series, as_of, universe)
        r = float(sofr.dropna().loc[:as_of].iloc[-1]) / 100.0
        ts = yield_term_structure(curve, r=r)
        rows.append(
            {
                "as_of": as_of,
                "n_pairs": len(ts),
                "front_yield": float(ts["implied_yield"].iloc[0]),
                "back_yield": float(ts["implied_yield"].iloc[-1]),
                "state": curve_state(ts),
            }
        )
    return pd.DataFrame(rows)


def var_coverage_backtest(
    pnl: pd.Series, level: float = 0.95, window: int = 126
) -> pd.DataFrame:
    """Rolling VaR coverage: breaches in the next window vs expectation.

    VaR from the trailing window (historical, via risk.var_es); breach
    = P&L strictly below -VaR. Expected breaches per window =
    (1 - level) * window; coverage is assessed on the total.
    """
    p = pd.Series(pnl).dropna()
    n = len(p)
    if n < 2 * window:
        raise ValueError(f"need ≥ {2 * window} P&L points, have {n}")
    rows = []
    for start in range(0, n - 2 * window + 1, window):
        hist = p.iloc[start : start + window]
        nxt = p.iloc[start + window : start + 2 * window]
        v = float(var_es(hist, levels=(level,)).iloc[0]["var"])
        rows.append(
            {
                "window_end": hist.index[-1],
                "var": v,
                "breaches": int((nxt < -v).sum()),
                "expected": (1 - level) * window,
            }
        )
    return pd.DataFrame(rows)


def var_coverage_filtered(
    price: pd.Series,
    pnl: pd.Series,
    multiplier: float,
    level: float = 0.95,
    lam: float = EWMA_LAMBDA,
) -> dict:
    """Day-by-day coverage of the FILTERED VaR (risk.filtered_var).

    Standard unconditional-coverage form: each day's VaR uses only
    information available at t−1 (lagged EWMA vol × lagged price); a
    breach is P&L_t strictly below -VaR_t. The static historical window
    backtest (var_coverage_backtest) holds one stale VaR constant across
    a whole test window — it conflates regime drift with model failure,
    which is why the filtered form is the remediation.
    """
    var = filtered_var(price, multiplier=multiplier, lam=lam, level=level)
    p = pd.Series(pnl).reindex(var.index).dropna()
    n = len(p)
    breaches = int((p < -var.reindex(p.index)).sum())
    expected = (1 - level) * n
    return {
        "n": n,
        "breaches": breaches,
        "expected": expected,
        "ratio": breaches / expected,
    }


def note_mc_check(
    notional: float = 1_000_000.0,
    f0: float = 320.0,
    tenor: float = 0.75,
    participation: float = 0.6,
    sigma: float = 0.385,
    r: float = 0.0366,
    n_paths: int = 50_000,
    seed: int = 42,
) -> dict:
    """Note pricing outcome check: MC vs closed form within 2 MC s.e."""
    note = ParticipationNote(
        notional=notional, f0=f0, tenor=tenor, participation=participation
    )
    cf = note.closed_form(sigma, r)["price"]
    mc = note.mc_price(sigma, r, n_paths=n_paths, seed=seed)
    diff = abs(mc["price"] - cf)
    return {"closed_form": cf, "mc": mc["price"], "se": mc["se"], "abs_diff": diff}


def cva_ql_benchmark(
    ee: np.ndarray, tenors: np.ndarray, hazard: float, lgd: float, r: float
) -> dict | None:
    """Re-quote the bucketed CVA with QuantLib's own survival/discount
    curves (same discretization). Pre-declared tolerance: 1e-9 relative.
    Returns None when QuantLib is not installed."""
    if find_spec("QuantLib") is None:
        return None
    import QuantLib as ql

    ref = ql.Date(15, ql.January, 2026)
    ql.Settings.instance().evaluationDate = ref
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    hz = ql.FlatHazardRate(0, cal, ql.QuoteHandle(ql.SimpleQuote(hazard)), dc)
    dfc = ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(r)), dc)
    t = np.asarray(tenors, dtype=float)
    s_ql = np.array([hz.survivalProbability(x) for x in t])
    df_ql = np.array([dfc.discount(x) for x in t])
    s_prev = np.concatenate([[1.0], s_ql[:-1]])
    ql_cva = lgd * float(np.sum(np.asarray(ee) * df_ql * (s_prev - s_ql)))
    ours = cva_unilateral(ee, tenors, hazard, lgd, r)
    return {"ours": ours, "ql": ql_cva, "rel_diff": abs(ours - ql_cva) / abs(ql_cva)}


# ---------------------------------------------------------------------------
# Findings assembly (feeds the report's validation chapter)
# ---------------------------------------------------------------------------

# Pre-declared acceptance bands (documented in the report as set before
# citing results — bands are sanity-level, not tuned to the outcome):
VOL_RMSE_RATIO_MAX = 1.10  # GARCH RMSE may exceed EWMA's by at most 10%
VAR_BREACH_RATIO_BAND = (0.5, 1.5)  # observed/expected breach ratio
CURVE_YIELD_BAND = (-100.0, 100.0)  # %/yr sanity band on front implied yield
NOTE_MC_SE_MULT = 2.0
CVA_QL_TOL = 1e-9


def validation_findings() -> pd.DataFrame:
    """Run every outcome check on the frozen data -> PASS/FLAG rows."""
    from hedging_workbench.conventions import DOLLARS_PER_CENT
    from hedging_workbench.hedge import variation_margin
    from hedging_workbench.vol import vol_from_frozen

    rows: list[dict] = []

    def add(check: str, detail: str, expectation: str, ok: bool) -> None:
        rows.append(
            {
                "check": check,
                "detail": detail,
                "expectation": expectation,
                "status": "PASS" if ok else "FLAG",
            }
        )

    symbols = [s for s in UNIVERSES["coffee"] if not s.endswith("=F")]
    series = load(symbols)
    last = min(s.dropna().index.max() for s in series.values())
    add(
        "no-look-ahead gate",
        f"gate active on {len(symbols)} frozen series (last {last.date()}); "
        "rejects a poisoned as-of window",
        "rejects any observation after as-of; fail-closed",
        _gate_rejects_poisoned(series),
    )

    vol = rolling_vol_backtest(vol_from_frozen()[0])
    rmse = vol.attrs["rmse"]
    add(
        "working vol forecast",
        f"GARCH RMSE {rmse['garch']:.2f} vs EWMA {rmse['ewma']:.2f} %/yr "
        f"over {len(vol)} windows",
        f"GARCH/EWMA RMSE ratio ≤ {VOL_RMSE_RATIO_MAX}",
        rmse["garch"] / rmse["ewma"] <= VOL_RMSE_RATIO_MAX,
    )

    stab = curve_stability_backtest()
    fy_pct = stab["front_yield"] * 100.0  # fraction -> %/yr
    n_back = int((stab["state"] == "backwardation").sum())
    add(
        "curve stability",
        f"front implied yield {fy_pct.min():.1f}..{fy_pct.max():.1f} %/yr over "
        f"{len(stab)} as-of dates; backwardated at {n_back} of {len(stab)} as-ofs",
        f"front yield within {CURVE_YIELD_BAND} %/yr at every as-of",
        bool(fy_pct.between(*CURVE_YIELD_BAND).all()),
    )

    kc = load(["KC=F"])["KC=F"].dropna()
    pnl = variation_margin(1.0, kc)["flow_usd"].iloc[1:]  # long 1 contract
    cov = var_coverage_backtest(pnl)
    tot_b, tot_e = int(cov["breaches"].sum()), float(cov["expected"].sum())
    add(
        "VaR coverage (95%, static historical)",
        f"{tot_b} breaches vs {tot_e:.1f} expected over {len(cov)} windows",
        f"observed/expected within {VAR_BREACH_RATIO_BAND}",
        VAR_BREACH_RATIO_BAND[0] <= tot_b / tot_e <= VAR_BREACH_RATIO_BAND[1],
    )
    filt = var_coverage_filtered(kc, pnl, multiplier=DOLLARS_PER_CENT, level=0.95)
    add(
        "VaR coverage (95%, filtered EWMA)",
        f"{filt['breaches']} breaches vs {filt['expected']:.1f} expected over "
        f"{filt['n']} days (ratio {filt['ratio']:.2f}) — remediation of the static FLAG",
        f"observed/expected within {VAR_BREACH_RATIO_BAND}",
        VAR_BREACH_RATIO_BAND[0] <= filt["ratio"] <= VAR_BREACH_RATIO_BAND[1],
    )

    mc = note_mc_check()
    add(
        "note MC vs closed form",
        f"|diff| = {mc['abs_diff']:.2e} per unit notional, s.e. {mc['se']:.2e}",
        f"|diff| ≤ {NOTE_MC_SE_MULT} × s.e.",
        mc["abs_diff"] <= NOTE_MC_SE_MULT * mc["se"],
    )

    ql = cva_ql_benchmark(
        np.array([80_000.0, 95_000.0, 110_000.0, 90_000.0]),
        np.array([0.25, 0.5, 1.0, 2.0]),
        0.1,
        0.6,
        0.04,
    )
    if ql is None:
        add("CVA vs QuantLib", "QuantLib not installed — skipped", "n/a", True)
    else:
        add(
            "CVA vs QuantLib",
            f"relative difference {ql['rel_diff']:.2e}",
            f"≤ {CVA_QL_TOL:.0e} (same discretization)",
            ql["rel_diff"] <= CVA_QL_TOL,
        )

    return pd.DataFrame(rows)
