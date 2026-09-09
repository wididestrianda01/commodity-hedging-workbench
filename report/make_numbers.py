"""Regenerate report/report_numbers.json — every number the report cites,
computed in one run from the frozen data (rerun before touching the .tex)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hedging_workbench.carry import curve_state, load_curve
from hedging_workbench.cva import cva_unilateral
from hedging_workbench.data.frozen import latest_rate, load
from hedging_workbench.decision import decision_from_frozen
from hedging_workbench.exposure import FuturesBook, epe, profile, simulate_front
from hedging_workbench.hedge import build_program
from hedging_workbench.note import ParticipationNote
from hedging_workbench.risk import var_es
from hedging_workbench.ssfit import fit_from_frozen
from hedging_workbench.validate import validation_findings
from hedging_workbench.vol import vol_from_frozen
from hedging_workbench.conventions import DOLLARS_PER_CENT
from hedging_workbench.hedge import variation_margin

OUT = Path(__file__).resolve().parent / "report_numbers.json"

curve = load_curve("coffee")
_, fit_ss, ts = fit_from_frozen()
_, vf = vol_from_frozen()
r = latest_rate()

out: dict = {}
out["curve"] = {
    "date": str(curve.attrs["curve_date"].date()),
    "front_label": curve["label"].iloc[0],
    "front_price": float(curve["price"].iloc[0]),
    "n_contracts": len(curve),
    "state": curve_state(ts),
    "front_yield": float(ts["implied_yield"].iloc[0]),
    "back_yield": float(ts["implied_yield"].iloc[-1]),
}
out["ss"] = {
    "chi": fit_ss.chi,
    "xi": fit_ss.xi,
    "kappa": fit_ss.kappa,
    "half_life_yr": float(np.log(2) / fit_ss.kappa),
}
out["vol"] = {
    "garch_last": vf.garch_last,
    "garch_longrun": vf.garch_longrun,
    "ewma": vf.ewma_annual,
    "persistence": vf.persistence,
}
out["sofr"] = r

note = ParticipationNote(
    notional=1_000_000, f0=out["curve"]["front_price"], tenor=0.75, participation=0.6
)
cf = note.closed_form(vf.garch_last / 100, r)
mc = note.mc_price(vf.garch_last / 100, r, n_paths=100_000, seed=42)
out["note"] = {
    "cf_price": cf["price"],
    "bond": cf["bond"],
    "call": cf["call"],
    "participation_value": cf["participation_value"],
    "mc": mc["price"],
    "mc_se": mc["se"],
}

prog = build_program(100_000, "2026-09-01", 12)
contracts = float(prog.exposure["contracts"].mean())
f0 = out["curve"]["front_price"]
F = simulate_front(f0, vf.garch_last / 100, 1.0, steps=252, n_paths=100_000, seed=42)
book = FuturesBook(contracts=contracts, entry=f0)
prof = profile(book.mtm(F))
out["book"] = {
    "contracts": contracts,
    "epe": epe(prof),
    "pfe_1y": float(prof["pfe"].iloc[-1]),
}
ee = prof["ee"].to_numpy()
tn = np.linspace(1 / 252, 1.0, len(ee))
out["cva"] = {"ours": cva_unilateral(ee, tn, 0.03, 0.6, r)}

kc = load(["KC=F"])["KC=F"].dropna()
pnl = variation_margin(contracts, kc)["flow_usd"].iloc[1:]
ve = var_es(pnl)
out["var_es"] = {
    str(i): {"var": float(row["var"]), "es": float(row["es"])}
    for i, row in ve.iterrows()
}

tbl = decision_from_frozen(volume_lb=1_200_000, contracts=contracts, put_strike=280.0)
out["decision"] = tbl.to_dict("records")
c = tbl.attrs["collar"]
out["collar"] = {
    "f0": c.f0,
    "put_strike": c.put_strike,
    "call_strike": c.call_strike,
    "put_premium": c.put_premium,
    "call_premium": c.call_premium,
    "premium_gap": c.premium_gap,
}
out["findings"] = validation_findings().to_dict("records")

OUT.write_text(json.dumps(out, indent=1, default=str))
print(f"wrote {OUT}")
