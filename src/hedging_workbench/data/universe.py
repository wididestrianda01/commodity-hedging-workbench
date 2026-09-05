"""Futures universes. Per-contract tickers discovered empirically on Yahoo
(2026-09-05); no chain-discovery API exists, so the chain is hardcoded and
covered by the data gates. Coffee month letters: H Mar, K May, N Jul,
U Sep, Z Dec (ICE Coffee C regular months)."""

COFFEE_CHAIN = (
    "KC=F",        # continuous front-month
    "KCU26.NYB",   # Sep 2026
    "KCZ26.NYB",   # Dec 2026
    "KCH27.NYB",   # Mar 2027
    "KCK27.NYB",   # May 2027
    "KCN27.NYB",   # Jul 2027
    "KCU27.NYB",   # Sep 2027
    "KCZ27.NYB",   # Dec 2027
    "KCH28.NYB",   # Mar 2028
)

GOLD_CHAIN = (
    "GC=F",        # continuous front-month
    "GCG27.CMX",   # Feb 2027
    "GCJ27.CMX",   # Apr 2027
    "GCK27.CMX",   # May 2027
    "GCN27.CMX",   # Jul 2027
    "GCZ27.CMX",   # Dec 2027
)

UNIVERSES = {"coffee": COFFEE_CHAIN, "gold": GOLD_CHAIN}

_MONTHS = {"H": "Mar", "K": "May", "N": "Jul", "U": "Sep", "Z": "Dec",
           "G": "Feb", "J": "Apr", "F": "Jan"}


def contract_label(symbol: str) -> str:
    """'KCH27.NYB' -> 'Mar 2027'; continuous symbols -> 'continuous'."""
    if symbol.endswith("=F"):
        return "continuous"
    root = symbol.split(".")[0][2:]
    return f"{_MONTHS[root[0]]} 20{root[1:]}"
