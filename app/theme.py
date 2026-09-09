"""Streamlit theme: config-driven palette + one injected CSS block.

Discipline: .scratch/workbench-expansion/design-app.md (minimalist-ui adapted).
Palette is exhaustive there; nothing outside it appears here.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

/* ---- palette ---- */
:root {
  --bone: #FBFBFA; --card: #FFFFFF; --line: #EAEAEA;
  --ink: #2F3437; --muted: #787774;
  --pos-bg: #EDF3EC; --pos: #346538;
  --neg-bg: #FDEBEC; --neg: #9F2F2D;
  --info-bg: #E1F3FE; --info: #1F6C9F;
  --warn-bg: #FBF3DB; --warn: #956400;
}

/* ---- typography ---- */
html, body, [class*="stApp"] { background: var(--bone); color: var(--ink);
  font-family: system-ui, 'Helvetica Neue', sans-serif; line-height: 1.6; }
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Newsreader', 'Playfair Display', Georgia, serif;
  letter-spacing: -0.02em; line-height: 1.1; font-weight: 600; }
[data-testid="stSidebar"] {
  background: var(--bone); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
  color: var(--muted); font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.05em; }

/* ---- mono numerals everywhere data appears ---- */
[data-testid="stMetricValue"], .stDataFrame, td, .stCode {
  font-family: 'SF Mono', 'JetBrains Mono', ui-monospace, monospace;
  font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"] { color: var(--ink); }
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div {
  color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.05em; }

/* ---- cards: flat, 1px line, no shadow ---- */
[data-testid="stMetric"] { background: var(--card);
  border: 1px solid var(--line); border-radius: 8px; padding: 20px 24px;
  transition: box-shadow 150ms ease; }
[data-testid="stMetric"]:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
[data-testid="stVerticalBlock"] > div:has([data-testid="stDataFrame"]) {
  background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 24px 28px; }
.stDataFrame { border: none; }
.stDataFrame [data-testid="stDataFrameResizable"] { border: none; }

/* ---- state badges ---- */
.badge { display: inline-block; padding: 2px 12px; border-radius: 9999px;
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; vertical-align: middle; }
.badge-info { background: var(--info-bg); color: var(--info); }
.badge-pos  { background: var(--pos-bg);  color: var(--pos); }
.badge-neg  { background: var(--neg-bg);  color: var(--neg); }
.badge-warn { background: var(--warn-bg); color: var(--warn); }

/* ---- tables: thin row lines, mono, header micro-labels ---- */
[data-testid="stDataFrame"] th div {
  text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.7rem;
  color: var(--muted); }

/* ---- buttons: solid charcoal, no pill, no shadow ---- */
.stButton > button {
  background: #111111; color: #FFFFFF; border: none; border-radius: 5px;
  transition: transform 100ms ease; }
.stButton > button:hover { background: #333333; }
.stButton > button:active { transform: scale(0.98); }

/* ---- footer meta ---- */
.snap-meta { color: var(--muted); font-family: 'SF Mono', 'JetBrains Mono',
  ui-monospace, monospace; font-size: 0.72rem; margin-top: 48px; }

/* ---- chrome: hide Deploy/toolbar ---- */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu {
  visibility: hidden; }
/* metric values: fit long words like "backwardation" */
[data-testid="stMetricValue"] { font-size: 1.35rem; white-space: normal;
  overflow-wrap: anywhere; }
.stat-card { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: 20px 24px; height: 100%;
  transition: box-shadow 150ms ease; }
.stat-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.stat-label { color: var(--muted); font-size: 0.72rem; text-transform:
  uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.stat-value { font-family: 'SF Mono', 'JetBrains Mono', ui-monospace;
  font-variant-numeric: tabular-nums; font-size: 1.35rem; color: var(--ink); }
.stat-pos .stat-value { color: var(--pos); }
.stat-neg .stat-value { color: var(--neg); }
.stat-warn .stat-value { color: var(--warn); }
.stat-note { color: var(--muted); font-size: 0.68rem; margin-top: 6px;
  line-height: 1.4; font-family: system-ui, sans-serif; }
.gate-row { font-size: 0.78rem; color: var(--ink); margin: 6px 0;
  display: flex; gap: 8px; align-items: center; }
.gate-n { color: var(--muted); font-family: 'SF Mono', 'JetBrains Mono',
  ui-monospace, monospace; font-size: 0.68rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    """kind: info | pos | neg | warn (design palette only)."""
    return f'<span class="badge badge-{kind}">{text}</span>'


def badge_kind(state: str) -> str:
    return {"contango": "info", "backwardation": "neg", "mixed": "warn"}.get(
        state, "warn"
    )


STATE_GLOSSARY = {
    "contango": "far contracts above near — carry positive, storage abundant",
    "backwardation": "far contracts below near — prompt scarcity premium",
    "mixed": "state varies along the curve — read per spread",
}
