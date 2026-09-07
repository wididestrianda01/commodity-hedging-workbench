"""Roaster hedge core: exposure -> contract selection -> roll schedule ->
variation-margin cash flows (ticket 10-05, Phase 3 deep module).

Archetype: a coffee roaster/buyer with a monthly green-coffee purchase
schedule (Starbucks FY2025 10-K template: futures/collars on the "C" price
as cash-flow hedges). The module maps a volume schedule onto the frozen
contract chain and makes margin liquidity observable from day one.

Conventions and assumptions (all pre-declared):
- ICE Coffee C contract = 37,500 lb; frozen prices are US cents/lb, so a
  1 cent move = $375 per contract (DOLLARS_PER_CENT).
- Exposure months between chain delivery months map to the earliest chain
  contract expiring at or after month end; exposures past the last chain
  contract fall back to the last contract and are flagged `gap=True`.
- Roll convention: exit a contract 10 calendar days before its expiry
  (ICE last trading day is ~8 business days before expiry; the buffer keeps
  the schedule honest without modelling the exchange calendar).
- Contracts are kept FRACTIONAL (volume / 37,500) so the learner sees the
  exact hedge ratio; round at execution, not in the model.
- Initial margin is an assumed $8,000/contract (ICE levels change; the
  number is a stated assumption, not an exchange feed).
- Frozen-snapshot as-of date comes from the curve attrs (2026-09-05 chain);
  no live data is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hedging_workbench.carry import load_curve
from hedging_workbench.data.rates import latest_rate

CONTRACT_LB = 37_500
DOLLARS_PER_CENT = CONTRACT_LB / 100.0     # $375 per cent per contract
INITIAL_MARGIN_PER_CONTRACT = 8_000.0      # assumed, see docstring
ROLL_BUFFER_DAYS = 10                      # exit N calendar days pre-expiry


@dataclass
class HedgeProgram:
    """End-to-end long-futures program for a purchase schedule."""
    exposure: pd.DataFrame   # month, volume_lb, contracts, symbol, label, expiry, gap
    rolls: pd.DataFrame      # front symbol/label/expiry, back symbol/label, roll_date


def exposure_schedule(monthly_lb: float, start: str | pd.Timestamp,
                      periods: int) -> pd.DataFrame:
    """Monthly purchase volumes -> exposure table (month, volume_lb, contracts)."""
    months = pd.date_range(start, periods=periods, freq="MS")
    return pd.DataFrame({"month": months, "volume_lb": float(monthly_lb)})


def select_contracts(exposure: pd.DataFrame,
                     curve: pd.DataFrame) -> pd.DataFrame:
    """Earliest chain contract expiring in or after each exposure month.

    A month's purchase is hedged with that month's delivery contract when
    one exists (lock the C-price, offset before last trading day); months
    without a chain contract map forward to the next expiry. Exposures past
    the last chain contract get the last contract and are flagged gap=True.
    """
    month_start = pd.to_datetime(exposure["month"])
    picks = []
    for ms in month_start:
        ok = curve[curve["expiry"] >= ms]
        row = ok.iloc[0] if len(ok) else curve.iloc[-1]
        picks.append((row["symbol"], row["label"], row["expiry"],
                      len(ok) == 0))
    picks = pd.DataFrame(picks, columns=["symbol", "label", "expiry", "gap"],
                         index=exposure.index)
    out = pd.concat([exposure, picks], axis=1)
    out["contracts"] = out["volume_lb"] / CONTRACT_LB
    return out


def roll_schedule(program_exposure: pd.DataFrame) -> pd.DataFrame:
    """Rows where the selected contract changes: exit front 10 days pre-expiry."""
    rows = []
    prev_sym = None
    for _, r in program_exposure.iterrows():
        if prev_sym is not None and r["symbol"] != prev_sym:
            front = program_exposure.loc[_prev_idx]
            rows.append({
                "roll_date": front["expiry"] - pd.Timedelta(days=ROLL_BUFFER_DAYS),
                "front_symbol": front["symbol"], "front_label": front["label"],
                "front_expiry": front["expiry"],
                "back_symbol": r["symbol"], "back_label": r["label"],
            })
        prev_sym, _prev_idx = r["symbol"], r.name
    return pd.DataFrame(rows)


def build_program(monthly_lb: float, start: str | pd.Timestamp, periods: int,
                  universe: str = "coffee") -> HedgeProgram:
    """Exposure -> selection -> rolls in one call, on a frozen universe."""
    exposure = select_contracts(exposure_schedule(monthly_lb, start, periods),
                                load_curve(universe))
    return HedgeProgram(exposure=exposure, rolls=roll_schedule(exposure))


def variation_margin(contracts: float, price_path: pd.Series,
                     initial_margin: float = INITIAL_MARGIN_PER_CONTRACT,
                     ) -> pd.DataFrame:
    """Daily variation-margin flows and balance for a LONG position.

    Price path in cents/lb; flow_t = contracts * $375 * (P_t - P_{t-1}).
    The first day contributes no flow (no prior settle).
    """
    diff = price_path.diff().fillna(0.0)
    flow = contracts * DOLLARS_PER_CENT * diff
    return pd.DataFrame({
        "flow_usd": flow,
        "balance_usd": initial_margin * abs(contracts) + flow.cumsum(),
    })
