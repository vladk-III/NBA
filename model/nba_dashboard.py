#!/usr/bin/env python3
"""
NBA Bayesian Betting Dashboard
==============================
The NBA version of the NFL dashboard: three forecasting models, Bayesian Predictive Synthesis
(McAlinn, Aastveit, Nakajima & West 2018) through a Rao-Blackwellized particle filter, Brier
skill against the de-vigged market, confusion matrices, and the Money Ball credible-interval
rule.

Difference from the NFL setup: every model (A pace x efficiency, B team strength, C market-
anchored) forecasts BOTH the total and the home margin, and BPS blends all three in each market.

Data comes from this repo's data/ folder (local clone) or GitHub raw:
  historical_nba_cumulative_stats.csv  pre-game team stats + final scores
  odds_snapshots.csv                   morning and closing lines from every US book

Run in Colab:
  !git clone https://github.com/vladk-III/NBA.git
  %cd NBA
  %run model/nba_dashboard.py
Daily: 1 Train (after the overnight data run) -> 2 Optimize BPS -> 3 Predict.
"""

GITHUB_OWNER = "vladk-III"
GITHUB_REPO = "NBA"
GITHUB_BRANCH = "main"
DATA_SUBDIR = "data"

DRAWS, TUNE, CHAINS = 1000, 1000, 4
NUTS_SAMPLER = "auto"            # "auto" uses nutpie when installed, else PyMC's sampler
TRAIN_SEED = 20261021
BPS_SEED = 20261022
JUPYTER_MODE = "external"
CSV_CACHE_TTL_SEC = 300
MAX_STORED_DRAWS = 1000          # per game per market in predictions_history.csv (file-size guard)
AGENTS = ["A", "B", "C"]
AGENT_NAMES = {"A": "Mod A (Pace×Eff)", "B": "Mod B (Team)", "C": "Mod C (Market)"}

import io
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _ensure(pkg: str, import_name: str | None = None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


for _pkg, _imp in [("pymc", "pymc"), ("arviz", "arviz"), ("netCDF4", "netCDF4"), ("requests", "requests"),
                   ("dash", "dash"), ("plotly", "plotly"), ("dash[diskcache]", "diskcache"),
                   ("multiprocess", "multiprocess"), ("nutpie", "nutpie")]:
    try:
        _ensure(_pkg, _imp)
    except subprocess.CalledProcessError:
        pass  # nutpie is optional; the rest will fail loudly on import below

import numpy as np
import pandas as pd
import requests
import arviz as az
import diskcache
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash_table, DiskcacheManager, no_update

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("nba_dashboard")
EASTERN = ZoneInfo("America/New_York")


def _load_model_modules():
    """Import nba_model_core / nba_models from next to this file, the repo's model/ folder,
    or (when this file was pasted into a notebook) download them from GitHub."""
    here = Path(globals().get("__file__", Path.cwd() / "model" / "x.py")).resolve().parent
    for d in (here, Path.cwd() / "model", Path.cwd()):
        if (d / "nba_model_core.py").exists():
            sys.path.insert(0, str(d))
            break
    else:
        d = Path.cwd() / "_nba_model_src"
        d.mkdir(exist_ok=True)
        for name in ("nba_model_core.py", "nba_models.py"):
            url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/model/{name}"
            (d / name).write_text(requests.get(url, timeout=30).text)
        sys.path.insert(0, str(d))
    import nba_model_core as core
    import nba_models as models
    return core, models


if "BPSFilter" in globals() and "model_a" in globals():
    # Single-notebook build (NBA_Dashboard.ipynb): core and model code already live in this namespace.
    core = models = sys.modules[__name__]
else:
    core, models = _load_model_modules()

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_BASE = Path("/content/drive/MyDrive/Stonks") / "Basketball/NBA/2026-27"
except Exception:
    DRIVE_BASE = Path("./NBA_Betting_Engine")
STATE_DIR = DRIVE_BASE / "nba_model_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_HISTORY_PATH = DRIVE_BASE / "predictions_history.csv"
BUNDLE_PATH = STATE_DIR / "model_bundle.pkl"
BPS_BUNDLE_PATH = STATE_DIR / "bps_bundle.pkl"


def model_path(key: str) -> Path:
    """Saved posterior for model a, b or c."""
    return STATE_DIR / f"model_{key}.pkl"

# =============================================================================
# THEME & UI STYLES (dark scoreboard: black slate strip, charcoal panels, lime + violet accents)
# =============================================================================
theme = {
    "page_bg": "#060708", "shell_bg": "#0F1013", "panel_bg": "#16171B", "card_bg": "#1F2126",
    "border": "#2A2D34", "text_main": "#F2F3F5", "text_axis": "#A3A8B3", "text_muted": "#7A808B",
    "grid": "#2A2D34", "accent": "#C6F432", "accent2": "#9D6BFF",
    "colors": ["#C6F432", "#9D6BFF", "#3DD6F5", "#FF8A3D", "#F2F3F5"],
    "positive": "#C6F432", "negative": "#FF5A5F",
    "font": "'Barlow', 'Segoe UI', 'Helvetica Neue', sans-serif",
    "display": "'Barlow Condensed', 'Arial Narrow', sans-serif",
}
AGENT_COLORS = {"A": "#3DD6F5", "B": "#9D6BFF", "C": "#FF8A3D", "BPS": "#C6F432"}
CHAIN_COLORS = ["#C6F432", "#9D6BFF", "#3DD6F5", "#FF8A3D", "#F2F3F5", "#FF5A5F"]
HEATMAP_SCALE = [[0.0, "#23252B"], [0.45, "#5A3EA8"], [0.7, "#9D6BFF"], [1.0, "#C6F432"]]
HIGHLIGHT = "rgba(198, 244, 50, 0.08)"

FONTS_URL = ("https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700"
             "&family=Barlow+Condensed:ital,wght@0,600;0,700;1,700;1,800&display=swap")
APP_CSS = """
:root {
  --bg: #060708; --shell: #0F1013; --panel: #16171B; --card: #1F2126; --line: #2A2D34;
  --text: #F2F3F5; --muted: #7A808B; --soft: #A3A8B3; --lime: #C6F432; --violet: #9D6BFF; --red: #FF5A5F;
  --display: 'Barlow Condensed', 'Arial Narrow', sans-serif; --body: 'Barlow', 'Segoe UI', sans-serif;
  /* Dash 4 component colors */
  --Dash-Fill-Inverse-Strong: #1F2126; --Dash-Fill-Interactive-Strong: #C6F432; --Dash-Fill-Interactive-Weak: rgba(198,244,50,.08);
  --Dash-Fill-Primary-Hover: rgba(255,255,255,.05); --Dash-Fill-Primary-Active: rgba(198,244,50,.12);
  --Dash-Fill-Disabled: rgba(255,255,255,.06); --Dash-Stroke-Strong: #3A3E47; --Dash-Stroke-Weak: #2A2D34;
  --Dash-Text-Strong: #F2F3F5; --Dash-Text-Primary: #F2F3F5; --Dash-Text-Weak: #A3A8B3; --Dash-Text-Disabled: #7A808B;
  --Dash-Tooltip-Background-Color: #0F1013; --Dash-Tooltip-Border-Color: #2A2D34;
  color-scheme: dark;
}
html, body { background: var(--bg); margin: 0 !important; padding: 0 !important; color: var(--text); font-family: var(--body); }
* { box-sizing: border-box; }
.shell { max-width: 1520px; margin: 20px auto; background: var(--shell); border: 1px solid var(--line); border-radius: 28px; padding: 22px 26px 30px; }
.topbar { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; margin-bottom: 22px; }
.logo { font-family: var(--display); font-style: italic; font-weight: 800; font-size: 34px; letter-spacing: .5px; color: var(--text); line-height: 1; }
.logo span { color: var(--lime); }
.nav { flex: 1; min-width: 0; overflow-x: auto; }
.nav-tab { background: transparent !important; border: none !important; color: var(--soft) !important; font-weight: 600 !important;
  font-size: 16px !important; padding: 10px 18px !important; position: relative; flex: 0 0 auto !important; width: auto !important; white-space: nowrap; }
.nav-tab--selected { color: var(--text) !important; }
.nav-tab--selected::after { content: ""; position: absolute; left: 50%; bottom: 0; width: 6px; height: 6px; border-radius: 50%; background: var(--lime); transform: translateX(-50%); }
.top-chips { display: flex; gap: 10px; align-items: center; }
.top-chip { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 8px 12px; font-size: 13px; color: var(--soft); font-weight: 600; }
.top-chip b { color: var(--text); }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--lime); margin-right: 6px; vertical-align: middle; }
.dot--red { background: var(--red); }
.frame { display: flex; gap: 20px; align-items: flex-start; }
.rail { display: flex; flex-direction: column; gap: 12px; position: sticky; top: 16px; }
.rail-btn { width: 64px; height: 64px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); color: var(--text);
  display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; gap: 2px; font-family: var(--display); }
.rail-btn:hover { border-color: var(--lime); }
.rail-btn--primary { background: var(--lime); color: #0F1013; border-color: var(--lime); }
.rail-btn:disabled { background: var(--lime); color: #0F1013; animation: pulse 1.4s ease-in-out infinite; cursor: progress; }
.rail-num { font-size: 22px; font-weight: 800; font-style: italic; line-height: 1; }
.rail-label { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
@keyframes pulse { 50% { opacity: .55; } }
@media (prefers-reduced-motion: reduce) { .rail-btn:disabled { animation: none; } }
.main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 20px; }
.status-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.status-chip { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 10px 14px; min-width: 0; }
.status-label { font-family: var(--display); font-weight: 700; font-style: italic; text-transform: uppercase; letter-spacing: .06em; color: var(--lime); font-size: 14px; }
.status-text { color: var(--soft); font-size: 13px; margin-top: 2px; overflow-wrap: anywhere; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 16px 16px 18px; min-width: 0; }
.panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 0 6px 12px; }
.panel-title { font-family: var(--display); font-style: italic; font-weight: 800; text-transform: uppercase; font-size: 24px; letter-spacing: .02em; margin: 0; }
.panel-sub { color: var(--muted); font-size: 13px; margin: 2px 0 0; }
.inner { background: var(--card); border-radius: 16px; padding: 14px; min-width: 0; }
.grid-2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.6fr); gap: 20px; }
.grid-even { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; }
@media (max-width: 1100px) { .grid-2 { grid-template-columns: minmax(0, 1fr); } .frame { flex-direction: column; align-items: stretch; }
  .rail { flex-direction: row; position: static; } .main { width: 100%; } }
.strip { background: #000; border-radius: 22px; padding: 14px 16px 16px; display: flex; gap: 16px; overflow-x: auto; }
.strip-section { display: flex; flex-direction: column; gap: 10px; flex: none; }
.strip-title { font-family: var(--display); font-style: italic; font-weight: 800; text-transform: uppercase; font-size: 24px; color: #fff; display: flex; align-items: center; gap: 8px; }
.strip-cards { display: flex; gap: 12px; }
.strip-divider { width: 1px; background: #2A2D34; margin: 40px 2px 0; flex: none; }
.game-card { background: var(--card); border-radius: 14px; padding: 10px 12px 8px; width: 238px; flex: none; position: relative; }
.card-top { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 12.5px; margin-bottom: 4px; gap: 6px; }
.chip-time { background: var(--shell); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 2px 7px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.team-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; }
.team-logo { width: 38px; height: 38px; border-radius: 10px; background: #fff; padding: 4px; object-fit: contain; flex: none; }
.team-name { font-weight: 600; font-size: 15px; flex: 1; }
.score { font-family: var(--display); font-style: italic; font-weight: 800; font-size: 26px; font-variant-numeric: tabular-nums; }
.score--dim { color: var(--muted); }
.card-foot { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.pill { font-family: var(--display); font-weight: 700; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--soft); white-space: nowrap; }
.pill--good { color: #0F1013; background: var(--lime); border-color: var(--lime); }
.pill--bad { color: var(--red); border-color: var(--red); }
.pill--warn { color: #FFB547; border-color: #FFB547; }
.pill--violet { color: var(--violet); border-color: var(--violet); }
.strip-empty { color: var(--muted); font-size: 14px; padding: 18px 4px; max-width: 360px; }
.bets { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.bet-card { background: var(--card); border-radius: 16px; padding: 14px; display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.bet-game { font-weight: 600; font-size: 15px; }
.bet-call { font-family: var(--display); font-style: italic; font-weight: 800; font-size: 22px; text-transform: uppercase; }
.bet-meta { color: var(--muted); font-size: 12.5px; }
.bet-ev { font-family: var(--display); font-style: italic; font-weight: 800; font-size: 34px; color: var(--lime); line-height: 1; font-variant-numeric: tabular-nums; }
.bet-ev--pass { color: var(--muted); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.kpi { background: var(--card); border-radius: 16px; padding: 14px 16px; min-width: 0; }
.kpi-label { font-family: var(--display); font-weight: 700; text-transform: uppercase; letter-spacing: .08em; font-size: 13px; color: var(--muted); }
.kpi-value { font-family: var(--display); font-style: italic; font-weight: 800; font-size: 32px; color: var(--lime); font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 12.5px; color: var(--soft); }
.controls { display: flex; flex-wrap: wrap; gap: 14px 20px; align-items: flex-end; }
.field-wrap { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-family: var(--display); font-weight: 700; text-transform: uppercase; letter-spacing: .08em; font-size: 12.5px; color: var(--muted); }
.field { background: var(--card); border: 1px solid var(--line); border-radius: 10px; color: var(--text); padding: 8px 10px; font-weight: 600; font-size: 14px; font-family: var(--body); }
.field:focus, .btn:focus-visible, .rail-btn:focus-visible { outline: 2px solid var(--lime); outline-offset: 2px; }
.btn { background: var(--lime); color: #0F1013; border: none; border-radius: 12px; padding: 9px 16px; font-weight: 700; font-family: var(--body); font-size: 14px; cursor: pointer; }
.btn--ghost { background: var(--card); color: var(--text); border: 1px solid var(--line); }
.note { color: var(--soft); font-size: 13px; max-width: 900px; margin: 0; line-height: 1.5; }
.sub-head { font-family: var(--display); font-weight: 700; text-transform: uppercase; letter-spacing: .06em; font-size: 15px; color: var(--soft); margin: 4px 0 8px; }
.empty { background: var(--card); border-radius: 16px; padding: 28px 20px; text-align: center; }
.empty-title { font-weight: 700; font-size: 15px; margin: 0 0 4px; }
.empty-sub { color: var(--muted); font-size: 13px; margin: 0; }
.error { background: var(--card); border-radius: 16px; padding: 16px; border-left: 4px solid var(--red); }
.health { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.health-card { background: var(--card); border-radius: 16px; padding: 14px; display: flex; flex-direction: column; gap: 8px; }
.health-name { font-family: var(--display); font-style: italic; font-weight: 800; font-size: 20px; text-transform: uppercase; }
.health-grid { display: grid; grid-template-columns: auto 1fr auto; gap: 6px 10px; align-items: center; font-size: 13.5px; }
.health-grid .val { font-variant-numeric: tabular-nums; text-align: right; font-weight: 600; }
.legend-row { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12.5px; color: var(--soft); }
.legend-row i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
input[type=radio] { accent-color: #C6F432; }
.Select-control, .Select-menu-outer, .VirtualizedSelectOption { background: var(--card) !important; color: var(--text) !important; border-color: var(--line) !important; }
.Select-value-label, .Select-placeholder { color: var(--text) !important; }
@media (max-width: 600px) { .shell { margin: 8px; padding: 14px 12px 20px; border-radius: 20px; } .topbar { gap: 12px; }
  .bets { grid-template-columns: minmax(0, 1fr); } .grid-even { grid-template-columns: minmax(0, 1fr); } }
"""

_ESPN_CODES = {"GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah", "WAS": "wsh"}


def team_logo(abbr):
    code = _ESPN_CODES.get(abbr, str(abbr).lower())
    return html.Img(src=f"https://a.espncdn.com/i/teamlogos/nba/500/{code}.png", alt=abbr, className="team-logo")


def panel(title, children, right=None, sub=None, extra_class=""):
    head = [html.Div([html.H2(title, className="panel-title")] + ([html.P(sub, className="panel-sub")] if sub else []))]
    if right is not None:
        head.append(right)
    return html.Section([html.Div(head, className="panel-head"), html.Div(children)], className=f"panel {extra_class}".strip())


def kpi_card(label, value, sub=None):
    children = [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")]
    if sub:
        children.append(html.Div(sub, className="kpi-sub"))
    return html.Div(children, className="kpi")


def section_header(title, subtext=None):
    children = [html.Div(title, className="sub-head")]
    if subtext:
        children.append(html.P(subtext, className="note", style={"marginBottom": "10px"}))
    return html.Div(children)


def empty_state(message, sub=None):
    children = [html.P(message, className="empty-title")]
    if sub:
        children.append(html.P(sub, className="empty-sub"))
    return html.Div(children, className="empty")


def error_state(title, detail):
    return html.Div([html.Div(f"⚠ {title}", style={"color": theme["negative"], "fontWeight": "700", "fontSize": "15px", "marginBottom": "8px"}),
                     html.Pre(detail, style={"fontSize": "12px", "color": theme["text_axis"], "whiteSpace": "pre-wrap", "margin": "0"})],
                    className="error")


def apply_plotly_theme(fig, title):
    fig.update_layout(title=dict(text=title, font=dict(color=theme["text_main"], family=theme["display"], size=17)),
                      plot_bgcolor=theme["card_bg"], paper_bgcolor=theme["card_bg"],
                      font=dict(color=theme["text_axis"], family=theme["font"], size=12),
                      margin=dict(t=50, b=50, l=50, r=24),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor=theme["shell_bg"], bordercolor=theme["border"], font=dict(color=theme["text_main"])))
    fig.update_xaxes(gridcolor=theme["grid"], zerolinecolor=theme["border"], linecolor=theme["border"])
    fig.update_yaxes(gridcolor=theme["grid"], zerolinecolor=theme["border"], linecolor=theme["border"])
    return fig


TABLE_STYLE_CELL = {"fontFamily": theme["font"], "fontSize": "14px", "padding": "11px 14px", "backgroundColor": theme["card_bg"],
                    "color": theme["text_main"], "border": "none", "borderBottom": f"1px solid {theme['border']}", "textAlign": "left"}
TABLE_STYLE_HEADER = {"fontWeight": "600", "backgroundColor": theme["card_bg"], "color": theme["text_muted"], "border": "none",
                      "borderBottom": f"1px solid {theme['border']}", "padding": "12px 14px", "textAlign": "left", "fontSize": "13px"}


def table(df, page_size=15, cond=None):
    return html.Div(dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
        style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER, sort_action="native", page_size=page_size,
        style_data_conditional=cond or [], style_table={"overflowX": "auto"},
        css=[{"selector": ".previous-next-container, .page-number", "rule": f"color: {theme['text_axis']};"},
             {"selector": "input.current-page", "rule": f"background: {theme['card_bg']}; color: {theme['text_main']};"}]),
        className="inner", style={"padding": "4px 6px"})


# =============================================================================
# DATA
# =============================================================================
_CSV_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def _load_csv(filename: str, max_age: float = CSV_CACHE_TTL_SEC) -> pd.DataFrame:
    """data/<file> from a local clone, else GitHub raw (cached for max_age seconds)."""
    for local in (Path(DATA_SUBDIR) / filename, Path("..") / DATA_SUBDIR / filename):
        if local.exists():
            return pd.read_csv(local, dtype={"game_id": str})
    url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{DATA_SUBDIR}/{filename}"
    hit = _CSV_CACHE.get(url)
    if hit is not None and max_age > 0 and time.time() - hit[0] < max_age:
        return hit[1].copy()
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text), dtype={"game_id": str})
            _CSV_CACHE[url] = (time.time(), df)
            return df.copy()
    except requests.RequestException:
        pass
    return hit[1].copy() if hit is not None else pd.DataFrame()


def load_data(fresh: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    age = 0 if fresh else CSV_CACHE_TTL_SEC
    hist = _load_csv("historical_nba_cumulative_stats.csv", age)
    if hist.empty:
        raise RuntimeError("historical_nba_cumulative_stats.csv is empty or unreachable.")
    return hist, _load_csv("odds_snapshots.csv", age)


def load_kalshi(fresh: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(totals, spreads) Kalshi snapshots."""
    age = 0 if fresh else CSV_CACHE_TTL_SEC
    return _load_csv("kalshi_snapshots.csv", age), _load_csv("kalshi_spread_snapshots.csv", age)


def _morning_first(lines_fn, snaps) -> pd.DataFrame:
    morning, closing = lines_fn(snaps, "morning"), lines_fn(snaps, "closing")
    return pd.concat([morning, closing[~closing["game_id"].isin(morning["game_id"])]], ignore_index=True)


def attach_lines(feats: pd.DataFrame, odds: pd.DataFrame,
                 kalshi: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Morning consensus lines and prices (closing snapshot as fallback for games with no morning one),
    plus the total and home margin implied by Kalshi's markets (inputs to Model C)."""
    out = feats.merge(_morning_first(core.market_lines, odds), on="game_id", how="left")
    out["line_implied_home_margin"] = -out["line_spread_home"]
    totals, spreads = kalshi if kalshi is not None else (None, None)
    out = out.merge(_morning_first(core.kalshi_lines, totals), on="game_id", how="left")
    spread_lines = lambda snaps, label: core.kalshi_lines(snaps, label, "spread")  # noqa: E731
    return out.merge(_morning_first(spread_lines, spreads), on="game_id", how="left")


def format_spread_val(val, home, away):
    """Home-margin number -> book notation naming the favorite (e.g. 'BOS -4.5')."""
    if val is None or pd.isna(val):
        return "—"
    v = float(val)
    if abs(v) < 0.05:
        return "EVEN"
    return f"{home} -{abs(v):.1f}" if v > 0 else f"{away} -{abs(v):.1f}"


def fmt_price(p):
    return "—" if p is None or pd.isna(p) else (f"+{int(p)}" if p > 0 else f"{int(p)}")


def _clean(value, default, lo, hi, cast=float):
    try:
        v = cast(round(float(value)) if cast is int else float(value))
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(v) else max(lo, min(hi, v))


# =============================================================================
# PIPELINES
# =============================================================================
def run_training_pipeline(set_progress):
    set_progress(f"Loading historical stats and odds... ({models.library_versions()})")
    hist, odds = load_data(fresh=True)
    final = hist["home_points"].notna()
    prior = core.league_prior(hist[final])
    feats = attach_lines(core.build_features(hist, prior), odds, load_kalshi())
    train = feats[feats["total_points"].notna()].reset_index(drop=True)
    upcoming = feats[feats["total_points"].isna()]
    if train.empty:
        raise RuntimeError("No completed games to train on.")
    train_c = train.dropna(subset=["line_total", "line_implied_home_margin"]).reset_index(drop=True)

    tot_std = models.Standardizer(models.TOTAL_COVS).fit(train)
    mar_std = models.Standardizer(models.MARGIN_COVS).fit(train)
    center = float(train["exp_total"].mean())
    # Every team of the latest season gets a rating slot, so tomorrow's games can be predicted
    # without retraining even when a team has not appeared in an upcoming row yet.
    latest = sorted(feats["season"].unique())[-1]
    teams = sorted(set(feats["home_team"]) | set(feats["away_team"]))
    all_latest = pd.DataFrame({"season": latest, "home_team": teams, "away_team": teams})
    index = models.team_index([train, upcoming, all_latest])
    sampler = models.resolve_sampler(NUTS_SAMPLER)

    set_progress(f"Training Model A (pace × efficiency) on {len(train)} games with {sampler} NUTS...")
    idata_a = models.sample(models.model_a(train, tot_std, mar_std, center), DRAWS, TUNE, CHAINS, TRAIN_SEED, NUTS_SAMPLER)
    models.save_idata(idata_a, str(model_path("a")))
    set_progress(f"Training Model B (team strength, {len(index['keys'])} team-seasons)...")
    idata_b = models.sample(models.model_b(train, index), DRAWS, TUNE, CHAINS, TRAIN_SEED + 1, NUTS_SAMPLER)
    models.save_idata(idata_b, str(model_path("b")))
    set_progress(f"Training Model C (market-anchored) on {len(train_c)} games with lines"
                 + (" — priors only until graded lines exist" if train_c.empty else "") + "...")
    idata_c = models.sample(models.model_c(train_c, tot_std, mar_std), DRAWS, TUNE, CHAINS, TRAIN_SEED + 2, NUTS_SAMPLER)
    models.save_idata(idata_c, str(model_path("c")))

    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump({"prior": prior, "tot_std": tot_std.get_state(), "mar_std": mar_std.get_state(),
                     "center": center, "index": index, "n_train": len(train), "n_train_c": len(train_c),
                     "trained_at": pd.Timestamp.now(tz="UTC").isoformat()}, f)
    set_progress(f"Training complete: {len(train)} games (A, B), {len(train_c)} with lines (C). {models.library_versions()}")
    return {"ok": True}


def load_bps(set_progress, n_particles, delta, beta):
    if BPS_BUNDLE_PATH.exists():
        try:
            with open(BPS_BUNDLE_PATH, "rb") as f:
                b = pickle.load(f)
            ft = core.BPSFilter.from_state(b["total"]["state"], seed=BPS_SEED + 1)
            fm = core.BPSFilter.from_state(b["margin"]["state"], seed=BPS_SEED + 2)
            tt, tm = b["total"]["theta_mean"], b["margin"]["theta_mean"]
            set_progress(f"Loaded BPS (p={ft.n}). Totals θ=[{tt[0]:.2f}, A={tt[1]:.2f}, B={tt[2]:.2f}, C={tt[3]:.2f}] "
                         f"Margins θ=[{tm[0]:.2f}, A={tm[1]:.2f}, B={tm[2]:.2f}, C={tm[3]:.2f}]")
            return ft, fm
        except Exception as e:
            LOG.warning("Could not load BPS bundle (%s); using prior filters.", e)
    set_progress(f"No fitted BPS yet — equal-weight (⅓ each) synthesis with p={n_particles}. "
                 "Run step 2 once predicted games are graded.")
    return (core.BPSFilter(3, n_particles, delta, beta, core.BPS_FALLBACK_OBS_VAR["total"], BPS_SEED + 1),
            core.BPSFilter(3, n_particles, delta, beta, core.BPS_FALLBACK_OBS_VAR["margin"], BPS_SEED + 2))


def run_bps_pipeline(set_progress, n_particles, delta, beta):
    n_particles = _clean(n_particles, core.BPS_DEFAULTS["particles"], 50, 20000, int)
    delta = _clean(delta, core.BPS_DEFAULTS["delta"], 0.5, 1.0)
    beta = _clean(beta, core.BPS_DEFAULTS["beta"], 0.5, 1.0)
    if not PREDICTIONS_HISTORY_PATH.exists():
        raise RuntimeError("No predictions_history.csv yet. Run predictions for a few days and let the games finish.")
    hist, odds = load_data(fresh=True)
    graded = prepare_graded_history(pd.read_csv(PREDICTIONS_HISTORY_PATH, dtype={"game_id": str}), hist, odds)
    if graded is None or graded.empty:
        raise RuntimeError("None of the logged predictions have final scores yet.")
    set_progress(f"{len(graded)} graded predictions. Filtering totals (p={n_particles}, δ={delta:g}, β={beta:g})...")
    tot = core.fit_bps(graded, [f"{a}_total_mean" for a in AGENTS], [f"{a}_total_std" for a in AGENTS],
                       "actual_total", n_particles, delta, beta, BPS_SEED + 1)
    set_progress("Filtering margins...")
    mar = core.fit_bps(graded, [f"{a}_margin_mean" for a in AGENTS], [f"{a}_margin_std" for a in AGENTS],
                       "actual_margin", n_particles, delta, beta, BPS_SEED + 2)
    with open(BPS_BUNDLE_PATH, "wb") as f:
        pickle.dump({"total": tot, "margin": mar, "n_particles": n_particles, "delta": delta, "beta": beta}, f)
    tt, tm = tot["theta_mean"], mar["theta_mean"]
    set_progress(f"BPS complete. Totals θ: int={tt[0]:.2f}, A={tt[1]:.2f}, B={tt[2]:.2f}, C={tt[3]:.2f}, "
                 f"σ={tot['vol_sd']:.1f} (n={tot['n_obs']}, ESS={tot['ess']:.0f}) | Margins θ: int={tm[0]:.2f}, "
                 f"A={tm[1]:.2f}, B={tm[2]:.2f}, C={tm[3]:.2f}, σ={mar['vol_sd']:.1f} (n={mar['n_obs']})")
    return {"ok": True}


def edge_tier(z):
    if pd.isna(z):
        return "No Line"
    a = abs(z)
    return "★★★ Strong" if a >= 1.5 else "★★☆ Moderate" if a >= 0.75 else "★☆☆ Slight"


HISTORY_COLS = (["game_id", "game_date", "season", "home_team", "away_team"]
                + [f"{a}_{m}_{s}" for a in AGENTS for m in ("total", "margin") for s in ("mean", "std")]
                + ["combined_total_mean", "combined_total_median", "combined_total_std",
                   "combined_margin_mean", "combined_margin_median", "combined_margin_std",
                   "line_total", "line_spread_home", "line_implied_home_margin",
                   "over_price", "under_price", "best_over_price", "best_under_price",
                   "home_price", "away_price", "best_home_price", "best_away_price", "kalshi_total",
                   "kalshi_home_margin", "edge_total", "edge_margin", "edge_total_z", "edge_margin_z", "total_tier", "spread_tier",
                   "model_prob_over", "model_prob_home_cover", "total_draws_json", "margin_draws_json", "predicted_at"])


def run_predictions_pipeline(set_progress, game_date, n_samples, n_particles, delta, beta):
    n_samples = _clean(n_samples, core.BPS_DEFAULTS["samples"], 200, 100000, int)
    n_particles = _clean(n_particles, core.BPS_DEFAULTS["particles"], 50, 20000, int)
    delta = _clean(delta, core.BPS_DEFAULTS["delta"], 0.5, 1.0)
    beta = _clean(beta, core.BPS_DEFAULTS["beta"], 0.5, 1.0)
    if not BUNDLE_PATH.exists() or not all(model_path(k).exists() for k in "abc"):
        raise RuntimeError("No trained models yet. Run step 1 first.")
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    tot_std = models.Standardizer.from_state(bundle["tot_std"])
    mar_std = models.Standardizer.from_state(bundle["mar_std"])

    set_progress("Loading today's matchups and lines...")
    hist, odds = load_data(fresh=True)
    feats = attach_lines(core.build_features(hist, bundle["prior"]), odds, load_kalshi())
    open_games = feats[feats["total_points"].isna()]
    today = datetime.now(EASTERN).strftime("%Y-%m-%d")
    target = str(game_date)[:10] if game_date else today
    slate = open_games[open_games["game_date"] == target]
    if slate.empty:
        later = sorted(d for d in open_games["game_date"].unique() if d >= today)
        if not later:
            raise RuntimeError(f"No unplayed games on {target}. The data pipeline adds each day's games overnight.")
        target = later[0]
        slate = open_games[open_games["game_date"] == target]
    slate = slate.reset_index(drop=True)
    known = {tuple(k) for k in bundle["index"]["keys"]}
    unknown = [f"{a}@{h}" for s, h, a in zip(slate["season"], slate["home_team"], slate["away_team"])
               if (s, h) not in known or (s, a) not in known]
    if unknown:
        raise RuntimeError(f"Teams not in the trained model ({', '.join(unknown)}). Re-run step 1.")

    rng = np.random.default_rng(BPS_SEED + int(pd.Timestamp(target).strftime("%Y%m%d")))
    set_progress(f"Forecasting {len(slate)} games on {target} with Models A, B and C...")
    draws = {
        "A": models.predict_a(load_idata("a"), slate, tot_std, mar_std, bundle["center"], rng),
        "B": models.predict_b(load_idata("b"), slate, bundle["index"], rng),
        "C": models.predict_c(load_idata("c"), slate, tot_std, mar_std, rng),
    }
    out = slate[["game_id", "game_date", "season", "home_team", "away_team", "line_total", "line_spread_home",
                 "line_implied_home_margin", "over_price", "under_price", "best_over_price", "best_under_price",
                 "home_price", "away_price", "best_home_price", "best_away_price", "kalshi_total", "kalshi_home_margin"]].copy()
    for a, (tot, mar) in draws.items():
        out[f"{a}_total_mean"], out[f"{a}_total_std"] = tot.mean(0), tot.std(0)
        out[f"{a}_margin_mean"], out[f"{a}_margin_std"] = mar.mean(0), mar.std(0)

    flt_t, flt_m = load_bps(set_progress, n_particles, delta, beta)
    set_progress(f"Synthesizing BPS draws ({n_samples:,} per game per market)...")
    recs = []
    for i, r in out.iterrows():
        mu_t = [r[f"{a}_total_mean"] for a in AGENTS]
        mu_m = [r[f"{a}_margin_mean"] for a in AGENTS]
        rec = {}
        if np.all(np.isfinite(mu_t)) and np.all(np.isfinite(mu_m)):
            pool_t = flt_t.predict(mu_t, [r[f"{a}_total_std"] for a in AGENTS], n_samples)["draws"]
            pool_m = flt_m.predict(mu_m, [r[f"{a}_margin_std"] for a in AGENTS], n_samples)["draws"]
            for name, pool, line in (("total", pool_t, r["line_total"]), ("margin", pool_m, r["line_implied_home_margin"])):
                rec[f"combined_{name}_mean"] = float(pool.mean())
                rec[f"combined_{name}_median"] = float(np.median(pool))
                rec[f"combined_{name}_std"] = float(pool.std())
                rec[f"edge_{name}"] = float(pool.mean() - line) if pd.notna(line) else np.nan
                rec[f"edge_{name}_z"] = rec[f"edge_{name}"] / float(pool.std()) if pd.notna(line) else np.nan
                rec[f"{name}_draws_json"] = json.dumps(np.round(pool[:MAX_STORED_DRAWS], 1).tolist())
            rec["model_prob_over"] = float((pool_t > r["line_total"]).mean()) if pd.notna(r["line_total"]) else np.nan
            rec["model_prob_home_cover"] = (float((pool_m > r["line_implied_home_margin"]).mean())
                                            if pd.notna(r["line_implied_home_margin"]) else np.nan)
        recs.append(rec)
    out = pd.concat([out, pd.DataFrame(recs, index=out.index)], axis=1)
    for c in HISTORY_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out["total_tier"] = out["edge_total_z"].apply(edge_tier)
    out["spread_tier"] = out["edge_margin_z"].apply(edge_tier)
    out["predicted_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    set_progress("Saving prediction history...")
    new = out[HISTORY_COLS]
    if PREDICTIONS_HISTORY_PATH.exists():
        old = pd.read_csv(PREDICTIONS_HISTORY_PATH, dtype={"game_id": str})
        new = pd.concat([old, new], ignore_index=True).sort_values("predicted_at").drop_duplicates("game_id", keep="last")
    new.to_csv(PREDICTIONS_HISTORY_PATH, index=False)
    n_blend = int(out["combined_total_mean"].notna().sum())
    set_progress(f"Predictions complete for {target}: {len(out)} games, {n_blend} with a line (BPS blend needs Model C, which needs the line).")
    return {"ok": True, "date": target, "data": out.to_json(orient="records")}


# =============================================================================
# GRADING
# =============================================================================
_GRADED_CACHE: dict = {}


def _sorted_draws(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or raw == "":
        return None
    try:
        a = np.asarray(json.loads(raw), dtype=float)
    except (ValueError, TypeError):
        return None
    a = a[np.isfinite(a)]
    return np.sort(a) if a.size else None


def prepare_graded_history(history: pd.DataFrame, hist: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame | None:
    """Prediction log + final scores + closing lines + results, with draws parsed once."""
    if history is None or history.empty:
        return None
    key = (int(pd.util.hash_pandas_object(history, index=False).sum()), len(hist), len(odds))
    if key in _GRADED_CACHE:
        return _GRADED_CACHE[key]
    finals = hist.loc[hist["home_points"].notna(), ["game_id", "home_points", "away_points"]].copy()
    finals["game_id"] = finals["game_id"].astype(str)
    close = core.market_lines(odds, "closing")[["game_id", "line_total", "line_spread_home"]]
    close = close.rename(columns={"line_total": "close_total", "line_spread_home": "close_spread_home"})
    h = history.copy()
    h["game_id"] = h["game_id"].astype(str)
    g = h.merge(finals, on="game_id", how="inner").merge(close, on="game_id", how="left")
    if g.empty:
        _GRADED_CACHE[key] = None
        return None
    g["close_implied_home_margin"] = -g["close_spread_home"]
    g["actual_total"] = g["home_points"] + g["away_points"]
    g["actual_margin"] = g["home_points"] - g["away_points"]
    g["total_result"] = [core.total_result(a, l) for a, l in zip(g["actual_total"], g["line_total"])]
    g["spread_result"] = [core.spread_result(a, l) for a, l in zip(g["actual_margin"], g["line_spread_home"])]
    g["_total_sorted"] = g["total_draws_json"].map(_sorted_draws)
    g["_margin_sorted"] = g["margin_draws_json"].map(_sorted_draws)
    g = g.reset_index(drop=True)
    if len(_GRADED_CACHE) > 8:
        _GRADED_CACHE.clear()
    _GRADED_CACHE[key] = g
    return g


def directional_confusion(df, pred_col, actual_col, line_col):
    """2x2 of predicted side vs actual side of the book line; pushes and on-the-line excluded."""
    if any(c not in df.columns for c in (pred_col, actual_col, line_col)):
        return {"cm": np.zeros((2, 2), dtype=int), "n": 0, "pushes": 0, "no_call": 0}
    v = df.dropna(subset=[pred_col, actual_col, line_col])
    pushes = int((v[actual_col] == v[line_col]).sum())
    v = v[v[actual_col] != v[line_col]]
    no_call = int((v[pred_col] == v[line_col]).sum())
    v = v[v[pred_col] != v[line_col]]
    p_hi, a_hi = (v[pred_col] > v[line_col]).to_numpy(), (v[actual_col] > v[line_col]).to_numpy()
    cm = np.array([[np.sum(a_hi & p_hi), np.sum(a_hi & ~p_hi)], [np.sum(~a_hi & p_hi), np.sum(~a_hi & ~p_hi)]], dtype=int)
    return {"cm": cm, "n": int(cm.sum()), "pushes": pushes, "no_call": no_call}


def confusion_heatmap(cm, x_labels, y_labels, title, x_title="Model call", y_title="Actual result"):
    cm = np.asarray(cm, dtype=float)
    col_tot = cm.sum(axis=0, keepdims=True)
    col_pct = np.divide(cm, col_tot, out=np.full_like(cm, np.nan), where=col_tot > 0) * 100
    zmax = max(cm.max(), 1.0)
    fig = go.Figure(go.Heatmap(z=cm, x=x_labels, y=y_labels, zmin=0, zmax=zmax, colorscale=HEATMAP_SCALE,
                               showscale=False, xgap=2, ygap=2, customdata=col_pct,
                               hovertemplate="Actual: %{y}<br>Call: %{x}<br><b>%{z:.0f} games</b><br>%{customdata:.1f}% of this call<extra></extra>"))
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            dark = cm[i, j] / zmax > 0.7   # light lime/violet cells take dark text
            pct = f"{col_pct[i, j]:.0f}% of {x} calls" if np.isfinite(col_pct[i, j]) else "—"
            fig.add_annotation(x=x, y=y, showarrow=False, text=f"<b>{int(cm[i, j])}</b><br><span style='font-size:11px'>{pct}</span>",
                               font=dict(size=18, color=theme["card_bg"] if dark else theme["text_main"], family=theme["font"]))
    apply_plotly_theme(fig, title)
    fig.update_layout(height=340, margin=dict(t=56, b=56, l=110, r=24))
    fig.update_xaxes(title_text=x_title, showgrid=False, type="category")
    fig.update_yaxes(title_text=y_title, autorange="reversed", showgrid=False, type="category")
    return fig


CM_MODEL_OPTIONS = {
    "bps_median": ("BPS Blend (Median)", "combined_total_median", "combined_margin_median"),
    "bps_mean": ("BPS Blend (Mean)", "combined_total_mean", "combined_margin_mean"),
    **{f"agent_{a}": (AGENT_NAMES[a], f"{a}_total_mean", f"{a}_margin_mean") for a in AGENTS},
}


def confusion_section(graded, model_key):
    label, tot_col, mar_col = CM_MODEL_OPTIONS.get(model_key, CM_MODEL_OPTIONS["bps_median"])
    if graded is None or graded.empty:
        return empty_state("No graded games yet.", "Confusion matrices appear once predicted games have final scores.")
    panels = []
    for res, title, labels in [
        (directional_confusion(graded, tot_col, "actual_total", "line_total"), "Totals — Over / Under", ["Over", "Under"]),
        (directional_confusion(graded, mar_col, "actual_margin", "line_implied_home_margin"), "Spreads — Against the Spread", ["Home covers", "Away covers"]),
    ]:
        cm, n = res["cm"], res["n"]
        hits = cm[0, 0] + cm[1, 1]
        summary = f"Hit rate {hits / n * 100:.1f}% of {n}" if n else "No graded games with a line."
        if res["pushes"] or res["no_call"]:
            summary += f" · excluded: {res['pushes']} pushes, {res['no_call']} on the line"
        panels.append(html.Div([dcc.Graph(figure=confusion_heatmap(cm, labels, labels, title), config={"displayModeBar": False}),
                                html.P(summary, className="note", style={"margin": "4px 0 0 0"})], className="inner"))
    return panel(f"Calls vs. book · {label}", html.Div(panels, className="grid-even"),
                 sub="Which side of the morning line each forecast landed on, against the result")


def brier_skill(graded, p_col, result_col, yes, no, price_yes, price_no):
    """Brier skill of the model's probability vs the de-vigged market from that game's own prices."""
    if graded is None or graded.empty or p_col not in graded.columns:
        return np.nan, 0
    g = graded[graded[result_col].isin([yes, no])].copy()
    g["p_mkt"] = core.devig_pair(g[price_yes].fillna(-110), g[price_no].fillna(-110)).values
    g = g.dropna(subset=[p_col, "p_mkt"])
    if g.empty:
        return np.nan, 0
    y = (g[result_col] == yes).astype(float)
    base = float(((g["p_mkt"] - y) ** 2).mean())
    return (1 - float(((g[p_col] - y) ** 2).mean()) / base if base > 0 else np.nan), len(g)


def moneyball_sweep(prep, rule="interval"):
    """Backtest across interval levels (interval rule) or EV thresholds (EV rule)."""
    rows = []
    levels = np.arange(0.50, 0.8001, 0.025) if rule == "interval" else np.arange(0.0, 0.1001, 0.01)
    for lvl in levels:
        res = (core.backtest(prep, float(lvl)) if rule == "interval"
               else core.backtest(prep, rule="ev", min_ev=float(lvl)))
        if res.get("empty"):
            continue
        rows.append({"Level": lvl * 100,
                     **{f"{m.title()} {k}": res[m][key] for m in ("total", "spread")
                        for k, key in (("Selectivity", "selectivity"), ("Win Rate", "win_rate"),
                                       ("ROI", "roi"), ("Breakeven", "breakeven"))}})
    return pd.DataFrame(rows)


def sweep_figure(sweep, market, level, rule="interval"):
    m = market.title()
    fig = go.Figure()
    for col, name, color in ((f"{m} Selectivity", "Selectivity (% of games bet)", theme["accent2"]),
                             (f"{m} Win Rate", "Win rate (%)", theme["accent"]),
                             (f"{m} ROI", "ROI (%)", theme["colors"][2])):
        fig.add_trace(go.Scatter(x=sweep["Level"], y=sweep[col], name=name, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=7)))
    be = sweep[f"{m} Breakeven"].dropna()
    fig.add_hline(y=float(be.mean()) if len(be) else 52.38, line_dash="dot", line_color=theme["negative"],
                  annotation_text="Breakeven win rate at the prices taken")
    fig.add_hline(y=0, line_color=theme["border"])
    fig.add_vline(x=level * 100, line_dash="dash", line_color=theme["text_axis"], annotation_text="Current")
    x_title = "Credible interval level (%)" if rule == "interval" else "Minimum EV per unit staked (%)"
    fig.update_layout(xaxis_title=x_title, yaxis=dict(title="%"), hovermode="x unified")
    what = "width" if rule == "interval" else "EV threshold"
    return apply_plotly_theme(fig, f"{'Totals' if market == 'total' else 'Spreads'} · results by {what}").update_layout(height=360)


# =============================================================================
# DASH APP
# =============================================================================
app = Dash(__name__, background_callback_manager=DiskcacheManager(diskcache.Cache(str(Path("/tmp/nba_dash_cache")))),
           suppress_callback_exceptions=True, title="NBA Projections")
app.index_string = f"""<!DOCTYPE html><html><head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS_URL}"><style>{APP_CSS}</style></head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""


def _num_input(id_, value, lo, hi, step, width="110px"):
    return dcc.Input(id=id_, type="number", min=lo, max=hi, step=step, value=value, debounce=True,
                     className="field", style={"width": width})


def field(label, control):
    return html.Label([html.Span(label, className="field-label"), control], className="field-wrap")


def rail_button(id_, num, label, tip, primary=False):
    return html.Button([html.Span(num, className="rail-num"), html.Span(label, className="rail-label")], id=id_, n_clicks=0,
                       title=tip, className="rail-btn rail-btn--primary" if primary else "rail-btn")


def status_chip(id_, label):
    return html.Div([html.Div(label, className="status-label"), html.Div("Not run this session", id=id_, className="status-text")],
                    className="status-chip")


tab1_layout = html.Div([
    html.Div(id="slate-strip"),
    html.Div([panel("Best bets", html.Div(id="best-bets"), sub="Top expected value at the best price across books"),
              panel("Edge board", html.Div(id="edge-board"), sub="BPS forecast minus the book line, in points")],
             className="grid-2"),
    html.Div(id="dash-results", style={"display": "flex", "flexDirection": "column", "gap": "20px"}),
    panel("Run settings", html.Div([
        html.Div([html.Div("2 · BPS", className="sub-head"),
                  html.Div([field("Particles (p)", _num_input("bps-particles", core.BPS_DEFAULTS["particles"], 50, 20000, 50)),
                            field("State discount (δ)", _num_input("bps-delta", core.BPS_DEFAULTS["delta"], 0.5, 1.0, 0.005)),
                            field("Volatility discount (β)", _num_input("bps-beta", core.BPS_DEFAULTS["beta"], 0.5, 1.0, 0.005))],
                           className="controls")], className="inner"),
        html.Div([html.Div("3 · Predict", className="sub-head"),
                  html.Div([field("Game date", dcc.Input(id="pred-date", type="text", placeholder="today (YYYY-MM-DD)", debounce=True,
                                                         className="field", style={"width": "180px"})),
                            field("Draws per game", _num_input("pred-samples", core.BPS_DEFAULTS["samples"], 200, 100000, 100)),
                            field("Projection basis", dcc.RadioItems(id="proj-basis-toggle", value="median", inline=True,
                                                                     options=[{"label": " Median", "value": "median"},
                                                                              {"label": " Mean", "value": "mean"}],
                                                                     labelStyle={"marginRight": "16px", "fontWeight": "600"}))],
                           className="controls")], className="inner"),
    ], className="grid-even"), sub="BPS settings are used by step 2; Predict only falls back on them before BPS has been fitted"),
])

tab2_layout = html.Div([
    panel("Performance ledger", html.Div([
        html.Button("Refresh", id="refresh-history-btn", n_clicks=0, className="btn btn--ghost"),
        field("Confusion matrix model", dcc.Dropdown(id="cm-model-dropdown", value="bps_median", clearable=False, searchable=False,
                                                     options=[{"label": v[0], "value": k} for k, v in CM_MODEL_OPTIONS.items()],
                                                     style={"width": "280px"}))], className="controls"),
          sub="Every logged prediction graded against final scores, the de-vigged market and the closing line"),
    dcc.Loading(html.Div(id="performance-results", style={"display": "flex", "flexDirection": "column", "gap": "20px"}),
                color=theme["accent"]),
], style={"display": "flex", "flexDirection": "column", "gap": "20px"})

tab3_layout = html.Div([
    panel("Money Ball", html.Div([
        field("Bet rule", dcc.RadioItems(id="moneyball-rule", value="ev", inline=True,
                                         options=[{"label": " Expected value (price-aware)", "value": "ev"},
                                                  {"label": " Credible interval (original)", "value": "interval"}],
                                         labelStyle={"marginRight": "20px", "fontWeight": "600"})),
        html.Div(field("Minimum EV per unit staked", dcc.Slider(id="moneyball-ev-slider", min=0, max=10, step=0.5, value=3,
                                                                marks={v: f"{v}%" for v in range(0, 11, 2)},
                                                                tooltip={"placement": "bottom", "always_visible": True})),
                 id="ev-slider-box", style={"minWidth": "320px", "flex": "1"}),
        html.Div(field("Credible interval width", dcc.Slider(id="moneyball-ci-slider", min=50, max=80, step=2.5, value=55,
                                                             marks={v: f"{v}%" for v in range(50, 81, 5)},
                                                             tooltip={"placement": "bottom", "always_visible": True})),
                 id="ci-slider-box", style={"display": "none"}),
    ], className="controls"), sub="Which bets the rule takes today, and how it would have done on graded history"),
    dcc.Loading(html.Div(id="tab3-results", style={"display": "flex", "flexDirection": "column", "gap": "20px"}), color=theme["accent"]),
], style={"display": "flex", "flexDirection": "column", "gap": "20px"})

tab4_layout = html.Div([
    panel("Sampler health", dcc.Loading(html.Div(id="diag-health"), color=theme["accent"]),
          right=html.Button("Refresh", id="diag-refresh-btn", n_clicks=0, className="btn btn--ghost"),
          sub="Convergence of each model's last training run: R̂ near 1, plenty of effective draws, no divergences"),
    panel("Trace & mixing", html.Div([
        html.Div([field("Model", dcc.Dropdown(id="diag-model", value="a", clearable=False, searchable=False, style={"width": "240px"},
                                              options=[{"label": "Model A · Pace × efficiency", "value": "a"},
                                                       {"label": "Model B · Team strength", "value": "b"},
                                                       {"label": "Model C · Market-anchored", "value": "c"}])),
                  field("Parameter", dcc.Dropdown(id="diag-param", clearable=False, style={"width": "300px"}))], className="controls"),
        dcc.Loading(html.Div(id="diag-trace", style={"marginTop": "14px"}), color=theme["accent"])]),
          sub="Chains should overlap like fuzzy caterpillars, their densities should agree, and rank histograms should be flat"),
    panel("Parameters", dcc.Loading(html.Div(id="diag-table"), color=theme["accent"]),
          sub="Posterior mean, 89% interval and convergence for every parameter; long vectors are summarized by their worst element"),
    panel("BPS mixture", dcc.Loading(html.Div(id="diag-bps"), color=theme["accent"]),
          sub="How the blend's trust in each model moved as graded games came in, and the particle filter's health"),
], style={"display": "flex", "flexDirection": "column", "gap": "20px"})

TABS = [("tab1", "Forecast"), ("tab2", "Skill"), ("tab3", "Money Ball"), ("tab4", "Diagnostics")]
app.layout = html.Div([
    html.Header([
        html.Div(["NBA", html.Span("X")], className="logo"),
        dcc.Tabs(id="tabs", value="tab1", className="nav", style={"borderBottom": "none"}, mobile_breakpoint=0,
                 children=[dcc.Tab(label=lbl, value=v, className="nav-tab", selected_className="nav-tab--selected",
                                   style={"border": "none"}, selected_style={"border": "none"}) for v, lbl in TABS]),
        html.Div([html.Div([html.Span(className="dot"), "Data ", html.B(f"{GITHUB_OWNER}/{GITHUB_REPO}")], className="top-chip"),
                  html.Div(["Season ", html.B("2026-27")], className="top-chip")], className="top-chips"),
    ], className="topbar"),
    html.Div([
        html.Aside([rail_button("train-btn", "1", "Train", "Train Models A, B and C on every final game"),
                    rail_button("bps-btn", "2", "BPS", "Fit the BPS blend on graded predictions"),
                    rail_button("predict-btn", "3", "Predict", "Forecast the slate and log it", primary=True)], className="rail"),
        html.Main([
            html.Div([status_chip("train-status", "1 · Train"), status_chip("bps-status", "2 · BPS"),
                      status_chip("predict-status", "3 · Predict")], className="status-row"),
            html.Div(tab1_layout, id="tab1-page"),
            html.Div(tab2_layout, id="tab2-page", style={"display": "none"}),
            html.Div(tab3_layout, id="tab3-page", style={"display": "none"}),
            html.Div(tab4_layout, id="tab4-page", style={"display": "none"}),
            dcc.Store(id="train-store", storage_type="session"),
            dcc.Store(id="bps-store", storage_type="session"),
            dcc.Store(id="predictions-store", storage_type="session"),
        ], className="main"),
    ], className="frame"),
], className="shell")


@app.callback([Output(f"{t}-page", "style") for t, _ in TABS], Input("tabs", "value"))
def toggle_tabs(tab):
    return [{"display": "block" if tab == t else "none"} for t, _ in TABS]


def _background(fn, label):
    def run(set_progress, *args):
        try:
            return fn(lambda msg: set_progress([msg]), *args)
        except Exception as e:
            LOG.exception("%s failed", label)
            set_progress([f"⚠ {label} failed: {e} ({models.library_versions()})"])
            return {"ok": False, "error": str(e)}
    return run


@app.callback(Output("train-store", "data"), Input("train-btn", "n_clicks"), background=True,
              running=[(Output("train-btn", "disabled"), True, False)], progress=[Output("train-status", "children")],
              prevent_initial_call=True)
def train_callback(set_progress, _):
    return _background(run_training_pipeline, "Training")(set_progress)


@app.callback(Output("bps-store", "data"), Input("bps-btn", "n_clicks"),
              State("bps-particles", "value"), State("bps-delta", "value"), State("bps-beta", "value"),
              background=True, running=[(Output("bps-btn", "disabled"), True, False)],
              progress=[Output("bps-status", "children")], prevent_initial_call=True)
def bps_callback(set_progress, _, p, d, b):
    return _background(run_bps_pipeline, "BPS")(set_progress, p, d, b)


@app.callback(Output("predictions-store", "data"), Input("predict-btn", "n_clicks"),
              State("pred-date", "value"), State("pred-samples", "value"), State("bps-particles", "value"),
              State("bps-delta", "value"), State("bps-beta", "value"),
              background=True, running=[(Output("predict-btn", "disabled"), True, False)],
              progress=[Output("predict-status", "children")], prevent_initial_call=True)
def predict_callback(set_progress, _, gdate, n, p, d, b):
    return _background(run_predictions_pipeline, "Prediction")(set_progress, gdate, n, p, d, b)


# ----------------------------------------------------------------------------- Forecast tab
def _tip_times() -> dict:
    """game_id -> tip-off in ET, e.g. '7:30 PM'."""
    games = _load_csv("games.csv")
    if games.empty or "tipoff_utc" not in games:
        return {}
    tips = pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce").dt.tz_convert(EASTERN)
    return {g: t.strftime("%I:%M %p").lstrip("0") for g, t in zip(games["game_id"].astype(str), tips) if pd.notna(t)}


def _num(v, fmt="{:.1f}"):
    return "—" if v is None or pd.isna(v) else fmt.format(v)


def team_row(abbr, value, dim=False):
    return html.Div([team_logo(abbr), html.Span(abbr, className="team-name"),
                     html.Span(value, className="score score--dim" if dim else "score")], className="team-row")


def slate_card(r, basis, tips):
    t, m = r.get(f"combined_total_{basis}"), r.get(f"combined_margin_{basis}")
    home = (t + m) / 2 if pd.notna(t) and pd.notna(m) else np.nan
    away = (t - m) / 2 if pd.notna(t) and pd.notna(m) else np.nan
    meta = f"O/U {_num(r['line_total'])} · {format_spread_val(r['line_implied_home_margin'], r['home_team'], r['away_team'])}"
    return html.Div([html.Div([html.Span(meta), html.Span(tips.get(str(r["game_id"]), r["game_date"]), className="chip-time")],
                              className="card-top"),
                     team_row(r["away_team"], _num(away), dim=pd.notna(m) and m > 0),
                     team_row(r["home_team"], _num(home), dim=pd.notna(m) and m < 0),
                     html.Div([html.Span(f"Total {_num(t)}", className="pill pill--violet"),
                               html.Span(format_spread_val(m, r["home_team"], r["away_team"]), className="pill")],
                              className="card-foot")], className="game-card")


def _hit_pill(label, pred, actual, line):
    if any(v is None or pd.isna(v) for v in (pred, actual, line)) or actual == line or pred == line:
        return html.Span(f"{label} —", className="pill")
    hit = (pred > line) == (actual > line)
    return html.Span(f"{label} {'✓' if hit else '✗'}", className="pill pill--good" if hit else "pill pill--bad")


def result_card(r):
    won_home = r["home_points"] > r["away_points"]
    return html.Div([html.Div([html.Span(f"Model {_num(r['combined_total_median'])} · Line {_num(r['line_total'])}"),
                               html.Span(str(r["game_date"])[5:], className="chip-time")], className="card-top"),
                     team_row(r["away_team"], _num(r["away_points"], "{:.0f}"), dim=won_home),
                     team_row(r["home_team"], _num(r["home_points"], "{:.0f}"), dim=not won_home),
                     html.Div([_hit_pill("Total", r["combined_total_median"], r["actual_total"], r["line_total"]),
                               _hit_pill("ATS", r["combined_margin_median"], r["actual_margin"], r["line_implied_home_margin"])],
                              className="card-foot")], className="game-card")


def slate_strip(df, store, basis):
    tips = _tip_times()
    if df is not None and not df.empty:
        cards = html.Div([slate_card(r, basis, tips) for _, r in df.iterrows()], className="strip-cards")
        title = f"Slate · {store.get('date', '')}"
    else:
        cards = html.Div("Press 3 · Predict to load the next slate. Each card shows the BPS projected score and the book line.",
                         className="strip-empty")
        title = "Slate"
    sections = [html.Div([html.Div([title, html.Span(className="dot dot--red")], className="strip-title"), cards], className="strip-section")]
    graded = None
    if PREDICTIONS_HISTORY_PATH.exists():
        hist, odds = load_data(fresh=False)
        graded = prepare_graded_history(pd.read_csv(PREDICTIONS_HISTORY_PATH, dtype={"game_id": str}), hist, odds)
    if graded is not None and not graded.empty:
        recent = graded.sort_values(["game_date", "game_id"], ascending=False).head(6)
        sections += [html.Div(className="strip-divider"),
                     html.Div([html.Div("Recent results", className="strip-title"),
                               html.Div([result_card(r) for _, r in recent.iterrows()], className="strip-cards")],
                              className="strip-section")]
    return html.Div(sections, className="strip")


def _bet_text(r, market, side):
    if market == "total":
        return f"{'Over' if side == 'HIGH' else 'Under'} {_num(r['line_total'])}"
    home_line = r["line_spread_home"]
    team = r["home_team"] if side == "HIGH" else r["away_team"]
    line = home_line if side == "HIGH" else -home_line
    return f"{team} {line:+.1f}".replace("+0.0", "PK")


def best_bets(df, min_ev=0.03):
    if df is None or df.empty:
        return empty_state("No slate loaded.", "The four best prices by expected value appear here after step 3.")
    picks = []
    for _, r in df.iterrows():
        for market, line, hi, lo in (("total", r["line_total"], r["best_over_price"], r["best_under_price"]),
                                     ("margin", r["line_implied_home_margin"], r["best_home_price"], r["best_away_price"])):
            c = core.ev_choice(_sorted_draws(r.get(f"{market}_draws_json")), line, hi, lo)
            if c is not None and np.isfinite(c["ev"]):
                picks.append((c["ev"], r, market, c))
    if not picks:
        return empty_state("No priced lines yet.", "Best bets need the morning odds snapshot.")
    cards = []
    for ev, r, market, c in sorted(picks, key=lambda x: -x[0])[:4]:
        bet = ev >= min_ev
        cards.append(html.Div([
            html.Div(f"{r['away_team']} @ {r['home_team']}", className="bet-game"),
            html.Div([html.Span("Total" if market == "total" else "Spread", className="pill pill--violet"), " ",
                      html.Span("Bet" if bet else "Pass", className="pill pill--good" if bet else "pill")]),
            html.Div(_bet_text(r, market, c["side"]), className="bet-call"),
            html.Div("EV per unit", className="bet-meta"),
            html.Div(f"{ev * 100:+.1f}%", className="bet-ev" if bet else "bet-ev bet-ev--pass"),
            html.Div(f"P(win) {c['p_win'] * 100:.1f}% · {fmt_price(c['price'])} · ¼ Kelly {c['kelly'] * 100:.1f}%", className="bet-meta"),
        ], className="bet-card"))
    return html.Div(cards, className="bets")


def edge_board(df, basis):
    if df is None or df.empty:
        return empty_state("No slate loaded.", "Lime bars: total edge (up = over). Violet bars: spread edge (up = home).")
    games = df["away_team"] + " @ " + df["home_team"]
    tot = df[f"combined_total_{basis}"] - df["line_total"]
    spr = df[f"combined_margin_{basis}"] - df["line_implied_home_margin"]
    fig = go.Figure([go.Bar(x=games, y=tot, name="Total edge (+ = over)", marker_color=theme["accent"], width=0.32,
                            hovertemplate="%{x}<br>Total edge %{y:+.1f} pts<extra></extra>"),
                     go.Bar(x=games, y=spr, name="Spread edge (+ = home)", marker_color=theme["accent2"], width=0.32,
                            hovertemplate="%{x}<br>Spread edge %{y:+.1f} pts<extra></extra>")])
    apply_plotly_theme(fig, "")
    fig.update_layout(barmode="group", bargap=0.35, height=340, margin=dict(t=30, b=60, l=50, r=10), yaxis_title="Points vs line")
    fig.update_xaxes(showgrid=False)
    return html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}), className="inner", style={"padding": "6px"})


@app.callback(Output("slate-strip", "children"), Output("best-bets", "children"), Output("edge-board", "children"),
              Output("dash-results", "children"), Input("predictions-store", "data"), Input("proj-basis-toggle", "value"))
def render_results(store, basis):
    df = None
    if store and store.get("ok"):
        df = pd.read_json(io.StringIO(store["data"]), orient="records", dtype={"game_id": str})
    strip = slate_strip(df, store or {}, basis)
    if store and not store.get("ok"):
        return strip, "", "", error_state("Prediction pipeline failed", store.get("error", "Unknown error"))
    if df is None or df.empty:
        return strip, best_bets(None), edge_board(None, basis), ""
    df["Game"] = df["away_team"] + " @ " + df["home_team"]
    tot_col, mar_col = f"combined_total_{basis}", f"combined_margin_{basis}"
    tot = pd.DataFrame({"Game": df["Game"],
                        **{AGENT_NAMES[a]: df[f"{a}_total_mean"].round(1) for a in AGENTS},
                        f"BPS ({basis.title()})": df[tot_col].round(1),
                        "Line": df["line_total"],
                        "Kalshi": df["kalshi_total"].round(1),
                        "P(Over)": (df["model_prob_over"] * 100).round(1),
                        "Best Over / Under": [f"{fmt_price(o)} / {fmt_price(u)}" for o, u in zip(df["best_over_price"], df["best_under_price"])],
                        "Edge": df["total_tier"]})
    fs = lambda col: [format_spread_val(v, h, w) for v, h, w in zip(df[col], df["home_team"], df["away_team"])]  # noqa: E731
    spr = pd.DataFrame({"Game": df["Game"],
                        **{AGENT_NAMES[a]: fs(f"{a}_margin_mean") for a in AGENTS},
                        f"BPS ({basis.title()})": fs(mar_col),
                        "Line": fs("line_implied_home_margin"),
                        "Kalshi": fs("kalshi_home_margin"),
                        "P(Home Covers)": (df["model_prob_home_cover"] * 100).round(1),
                        "Best Home / Away": [f"{fmt_price(o)} / {fmt_price(u)}" for o, u in zip(df["best_home_price"], df["best_away_price"])],
                        "Edge": df["spread_tier"]})
    bold = [{"if": {"column_id": f"BPS ({basis.title()})"}, "fontWeight": "bold", "color": theme["accent"], "backgroundColor": HIGHLIGHT}]
    tables = [panel("Totals", table(tot, cond=bold), sub=f"Every model's forecast, the blend and the market · {store.get('date', '')}"),
              panel("Spreads", table(spr, cond=bold), sub="Spreads name the favorite: BOS -4.5 = Boston by 4.5")]
    return strip, best_bets(df), edge_board(df, basis), tables


# ----------------------------------------------------------------------------- Skill tab
def _accuracy_rows(graded, market):
    actual, line = ("actual_total", "line_total") if market == "total" else ("actual_margin", "line_implied_home_margin")
    rows = []
    for name, col in [(AGENT_NAMES[a], f"{a}_{market}_mean") for a in AGENTS] + \
                     [("BPS Blend (Mean)", f"combined_{market}_mean"), ("BPS Blend (Median)", f"combined_{market}_median")]:
        v = graded.dropna(subset=[col, actual]) if col in graded.columns else graded.iloc[0:0]
        res = directional_confusion(graded, col, actual, line)
        win = (res["cm"][0, 0] + res["cm"][1, 1]) / res["n"] * 100 if res["n"] else np.nan
        rows.append({"Model": name,
                     "MAE": round(float((v[col] - v[actual]).abs().mean()), 2) if len(v) else "—",
                     "Bias": round(float((v[col] - v[actual]).mean()), 2) if len(v) else "—",
                     "Win % vs line": f"{win:.1f}%" if pd.notna(win) else "—", "Games": len(v)})
    return pd.DataFrame(rows)


@app.callback(Output("performance-results", "children"), Input("refresh-history-btn", "n_clicks"), Input("cm-model-dropdown", "value"))
def render_performance(n_clicks, cm_model):
    if not PREDICTIONS_HISTORY_PATH.exists():
        return empty_state("No prediction history yet.", "It builds up each time you run predictions.")
    history = pd.read_csv(PREDICTIONS_HISTORY_PATH, dtype={"game_id": str})
    hist, odds = load_data(fresh=(n_clicks or 0) > 0)
    graded = prepare_graded_history(history, hist, odds)
    if graded is None or graded.empty:
        return empty_state("No graded games yet.", f"{len(history)} predictions logged; they're graded once final scores land.")
    bss_t, n_t = brier_skill(graded, "model_prob_over", "total_result", "over", "under", "over_price", "under_price")
    bss_s, n_s = brier_skill(graded, "model_prob_home_cover", "spread_result", "home_cover", "away_cover", "home_price", "away_price")
    bold = [{"if": {"filter_query": '{Model} contains "BPS"'}, "fontWeight": "bold", "color": theme["accent"], "backgroundColor": HIGHLIGHT}]
    arch = graded.sort_values("game_date", ascending=False)
    fs = lambda col: [format_spread_val(v, h, a) for v, h, a in zip(arch[col], arch["home_team"], arch["away_team"])]  # noqa: E731
    archive = pd.DataFrame({
        "Date": arch["game_date"], "Game": arch["away_team"] + " @ " + arch["home_team"],
        "Book Total": arch["line_total"], "Model Total": arch["combined_total_median"].round(1),
        "Actual Total": arch["actual_total"], "Close Total": arch["close_total"],
        "Book Line": fs("line_implied_home_margin"), "Model Line": fs("combined_margin_median"), "Actual": fs("actual_margin")})
    return [
        html.Div([kpi_card("Totals Brier skill", f"{bss_t:.4f}" if pd.notna(bss_t) else "—",
                           f"P(over) vs de-vigged market · {n_t} games"),
                  kpi_card("Spreads Brier skill", f"{bss_s:.4f}" if pd.notna(bss_s) else "—",
                           f"P(home covers) vs de-vigged market · {n_s} games"),
                  kpi_card("Graded games", f"{len(graded)}", f"{len(history)} predictions logged")], className="kpis"),
        html.P("Brier skill above 0 means the model's probabilities beat the market's own, with the vig removed using both sides' prices.",
               className="note"),
        confusion_section(graded, cm_model),
        html.Div([panel("Totals accuracy", table(_accuracy_rows(graded, "total"), cond=bold)),
                  panel("Spreads accuracy", table(_accuracy_rows(graded, "margin"), cond=bold))], className="grid-even"),
        panel("Game-by-game archive", table(archive)),
    ]


# ----------------------------------------------------------------------------- Money Ball tab
def moneyball_live(out, market, rule, conf, min_ev):
    """Today's calls under the selected rule. HIGH = over / home covers; LOW = under / away."""
    lo_pct, hi_pct = core.credible_interval_bounds(conf)
    total = market == "total"
    rows = []
    for _, r in out.iterrows():
        draws = _sorted_draws(r.get(f"{market}_draws_json"))
        line = r["line_total"] if total else r["line_implied_home_margin"]
        px_hi, px_lo = (r["best_over_price"], r["best_under_price"]) if total else (r["best_home_price"], r["best_away_price"])
        fmt = (lambda v: round(v, 1) if pd.notna(v) else None) if total else \
              (lambda v, h=r["home_team"], a=r["away_team"]: format_spread_val(v, h, a))
        row = {"Game": f"{r['away_team']} @ {r['home_team']}", "Book Line": fmt(line)}
        if rule == "ev":
            c = core.ev_choice(draws, line, px_hi, px_lo)
            side = None if c is None else (c["side"] if c["ev"] >= min_ev else "NO_BET")
            lean = None if c is None else c["side"]
            row.update({"Lean": "—" if lean is None else (("OVER" if lean == "HIGH" else "UNDER") if total
                                                           else (r["home_team"] if lean == "HIGH" else r["away_team"])),
                        "P(Win)": f"{c['p_win'] * 100:.1f}%" if c else "—",
                        "Breakeven": f"{c['breakeven'] * 100:.1f}%" if c else "—",
                        "EV / unit": f"{c['ev'] * 100:+.1f}%" if c else "—",
                        "Best Price": fmt_price(c["price"]) if c else "—",
                        "¼ Kelly stake": f"{c['kelly'] * 100:.1f}% of bankroll" if c and side not in (None, "NO_BET") else "—"})
        else:
            lo = core.quantile_sorted(draws, lo_pct) if draws is not None else np.nan
            hi = core.quantile_sorted(draws, hi_pct) if draws is not None else np.nan
            side = core.side_vs_interval(line, lo, hi)
            row.update({f"{conf * 100:.0f}% CrI Lower": fmt(lo), f"{conf * 100:.0f}% CrI Upper": fmt(hi)})
        if side is None:
            call, price = "No Line", np.nan
        elif side == "NO_BET":
            call, price = "NO BET", np.nan
        else:
            call = ("OVER" if side == "HIGH" else "UNDER") if total else (r["home_team"] if side == "HIGH" else r["away_team"])
            price = px_hi if side == "HIGH" else px_lo
        row["Call"] = call
        if rule != "ev":
            row["Best Price"] = fmt_price(price)
        rows.append(row)
    return pd.DataFrame(rows)


@app.callback(Output("ev-slider-box", "style"), Output("ci-slider-box", "style"), Input("moneyball-rule", "value"))
def toggle_rule_controls(rule):
    show, hide = {"minWidth": "320px", "flex": "1"}, {"display": "none"}
    return (show, hide) if rule == "ev" else (hide, show)


@app.callback(Output("tab3-results", "children"), Input("moneyball-rule", "value"), Input("moneyball-ci-slider", "value"),
              Input("moneyball-ev-slider", "value"), Input("predictions-store", "data"))
def render_moneyball(rule, ci_pct, ev_pct, store):
    conf, min_ev = ci_pct / 100.0, ev_pct / 100.0
    lo_pct, hi_pct = core.credible_interval_bounds(conf)
    if rule == "ev":
        explain = (f"For each side, the BPS draws give the chance of winning, losing and pushing at the morning line. "
                   f"EV per unit = P(win) × payout − P(loss), at the best price across books. The rule bets the better side "
                   f"when its EV is at least {ev_pct:g}%. A side needs a win chance above its breakeven (52.4% at −110) just to "
                   "reach 0%. The quarter-Kelly stake is a sizing guide, capped at 5% of bankroll. EV is only as good as the "
                   "model's probabilities, so check the Brier skill on the Skill tab before trusting it.")
        rule_label, current = f"EV ≥ {ev_pct:g}%", min_ev
    else:
        explain = (f"The {ci_pct:.0f}% interval spans the {lo_pct:.2f}th to {hi_pct:.2f}th percentile of the BPS draws. "
                   "Line below it: OVER / home side; above it: UNDER / away side; inside: NO BET. NBA game-to-game noise "
                   "makes this interval wide, so this rule bets rarely.")
        rule_label, current = f"{ci_pct:.0f}% interval", conf
    call_style = [{"if": {"filter_query": '{Call} = "NO BET"'}, "color": theme["text_muted"]},
                  {"if": {"filter_query": '{Call} != "NO BET" && {Call} != "No Line"'}, "fontWeight": "bold", "color": theme["positive"]}]
    if not store or not store.get("ok"):
        calls = empty_state("No live predictions loaded.", "Press 3 · Predict.")
    else:
        out = pd.read_json(io.StringIO(store["data"]), orient="records", dtype={"game_id": str})
        calls = html.Div([html.Div([html.Div(label, className="sub-head"), table(moneyball_live(out, market, rule, conf, min_ev), 10, call_style)])
                          for market, label in (("total", "Totals"), ("margin", "Spreads"))],
                         style={"display": "flex", "flexDirection": "column", "gap": "14px"})
    sections = [panel("Today's calls", [html.P(explain, className="note", style={"marginBottom": "14px"}), calls],
                      sub=f"{rule_label} vs. the morning line")]

    def backtest_panel(children):
        return panel("Historical backtest", children, sub=f"The {rule_label} rule on graded history, flat 1-unit bets")

    if not PREDICTIONS_HISTORY_PATH.exists():
        return sections + [backtest_panel(empty_state("No prediction history yet."))]
    hist, odds = load_data(fresh=False)
    prep = prepare_graded_history(pd.read_csv(PREDICTIONS_HISTORY_PATH, dtype={"game_id": str}), hist, odds)
    bt = core.backtest(prep, conf, rule=rule, min_ev=min_ev)
    if bt.get("empty"):
        return sections + [backtest_panel(empty_state("No graded games yet.", "Results appear once predicted games are final."))]
    cards = []
    for m, label in (("total", "Totals"), ("spread", "Spreads")):
        r = bt[m]
        ev_note = f" · avg expected {r['avg_ev']:+.1f}%" if rule == "ev" and pd.notna(r["avg_ev"]) else ""
        cards += [kpi_card(f"{label} · Bets", f"{r['n_bets']} / {r['n_games']}", f"Selectivity {r['selectivity']:.1f}%"),
                  kpi_card(f"{label} · Win rate", f"{r['win_rate']:.1f}%" if pd.notna(r["win_rate"]) else "—",
                           f"breakeven {r['breakeven']:.1f}% at prices taken" if pd.notna(r["breakeven"]) else ""),
                  kpi_card(f"{label} · Units", f"{r['units']:+.2f}", (f"ROI {r['roi']:+.1f}%" if pd.notna(r["roi"]) else "") + ev_note),
                  kpi_card(f"{label} · CLV", f"{r['avg_clv']:+.2f} pts" if pd.notna(r["avg_clv"]) else "—",
                           f"interval misses {r['outside_pct']:.0f}% vs {bt['nominal_outside_pct']:.0f}% nominal"
                           if rule == "interval" and pd.notna(r["outside_pct"]) else "")]
    calls_df = bt["calls"]
    figs = []
    for m, (hi, lo), (c_hi, c_lo), ylab in (("total", ("over", "under"), ("OVER", "UNDER"), ["Over", "Under"]),
                                             ("spread", ("home_cover", "away_cover"), ("Bet home", "Bet away"), ["Home covered", "Away covered"])):
        d = calls_df[(calls_df["market"] == m) & calls_df["result"].isin([hi, lo])]
        cm = np.array([[np.sum((d["call"] == s) & (d["result"] == res)) for s in ("HIGH", "NO_BET", "LOW")] for res in (hi, lo)])
        figs.append(confusion_heatmap(cm, [c_hi, "NO BET", c_lo], ylab, f"{'Totals' if m == 'total' else 'Spreads'} · {rule_label} vs result",
                                      x_title="Money Ball call"))
    body = [html.Div(cards, className="kpis"),
            html.P("CLV = points the line moved toward your bet between the morning and closing snapshots. Consistently positive CLV is "
                   "the earliest sign of a real edge, well before win rate settles. "
                   + ("With the EV rule, compare realized ROI to the average expected EV: if ROI keeps landing well below it, "
                      "the model's probabilities are overconfident." if rule == "ev" else ""), className="note",
                   style={"margin": "12px 0"}),
            html.Div([html.Div(dcc.Graph(figure=f, config={"displayModeBar": False}), className="inner") for f in figs], className="grid-even")]
    sections.append(backtest_panel(body))
    sweep = moneyball_sweep(prep, rule)
    if not sweep.empty:
        sections.append(panel("Finding the right setting",
                              html.Div([html.Div(dcc.Graph(figure=sweep_figure(sweep, m, current, rule)), className="inner")
                                        for m in ("total", "spread")], className="grid-even"),
                              sub="Selectivity, win rate and ROI across " + ("EV thresholds" if rule == "ev" else "interval levels")))
    return sections


# ----------------------------------------------------------------------------- Diagnostics tab
DIAG_SKIP = {"z_off", "z_def", "nu_raw", "nu_t_raw", "nu_m_raw"}   # raw sampler coordinates; the derived values carry the meaning
DIAG_EXPAND = 8                                                    # vectors up to this length get one row per element
_IDATA_CACHE: dict = {}


def load_idata(key):
    path = model_path(key)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    hit = _IDATA_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    idata = models.read_idata(str(path))
    _IDATA_CACHE[key] = (mtime, idata)
    return idata


def _team_labels():
    try:
        with open(BUNDLE_PATH, "rb") as f:
            return [f"{s} {t}" for s, t in pickle.load(f)["index"]["keys"]]
    except Exception:
        return None


def _param_elements(idata, key):
    """(label, var, flat index, chain × draw array) for every monitored scalar."""
    labels = _team_labels() if key == "b" else None
    out = []
    for var in idata.posterior.data_vars:
        if var in DIAG_SKIP:
            continue
        arr = np.asarray(idata.posterior[var].values)
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for i in range(flat.shape[2]):
            if flat.shape[2] == 1:
                label = var
            elif labels is not None and var in ("off", "def") and flat.shape[2] == len(labels):
                label = f"{var}[{labels[i]}]"
            else:
                label = f"{var}[{i}]"
            out.append((label, var, i, flat[:, :, i]))
    return out


def _diag_stat(fn, idata, **kw):
    """Per-variable values from az.rhat / az.ess, flattened; works with ArviZ 0.x Datasets and 1.x DataTrees."""
    try:
        res = fn(idata, **kw)
        return {v: np.asarray(res[v].values, dtype=float).ravel() for v in idata.posterior.data_vars if v not in DIAG_SKIP}
    except Exception as e:
        LOG.warning("Diagnostic %s failed: %s", getattr(fn, "__name__", fn), e)
        return {}


def _sampler_stats(idata):
    if hasattr(idata, "children"):          # ArviZ 1.x DataTree
        ss = idata["sample_stats"] if "sample_stats" in idata.children else None
    else:                                    # ArviZ 0.x InferenceData
        ss = getattr(idata, "sample_stats", None)
    out = {"divergences": np.nan, "accept": np.nan, "bfmi": np.nan}
    if ss is None:
        return out
    if "diverging" in ss:
        out["divergences"] = int(np.asarray(ss["diverging"].values).sum())
    for name in ("mean_tree_accept", "acceptance_rate"):
        if name in ss:
            out["accept"] = float(np.nanmean(np.asarray(ss[name].values)))
            break
    if "energy" in ss:
        e = np.asarray(ss["energy"].values, dtype=float)
        bfmi = np.square(np.diff(e, axis=1)).mean(axis=1) / e.var(axis=1)
        out["bfmi"] = float(np.nanmin(bfmi))
    return out


_DIAG_CACHE: dict = {}


def model_diagnostics(key):
    idata = load_idata(key)
    if idata is None:
        return None
    mtime = model_path(key).stat().st_mtime
    if _DIAG_CACHE.get(key, (None,))[0] == mtime:
        return _DIAG_CACHE[key][1]
    rhat = _diag_stat(az.rhat, idata)
    ess_b = _diag_stat(az.ess, idata, method="bulk")
    ess_t = _diag_stat(az.ess, idata, method="tail")
    rows = []
    elements = _param_elements(idata, key)
    by_var: dict = {}
    for label, var, i, arr in elements:
        by_var.setdefault(var, []).append((label, i, arr))
    def pick(stat, var, i):
        v = stat.get(var)
        return float(v[i]) if v is not None and len(v) > i else np.nan

    for var, items in by_var.items():
        if len(items) <= DIAG_EXPAND:
            for label, i, arr in items:
                rows.append({"Parameter": label, "Mean": float(arr.mean()), "SD": float(arr.std()),
                             "Low": float(np.quantile(arr, 0.055)), "High": float(np.quantile(arr, 0.945)),
                             "R̂": pick(rhat, var, i), "ESS bulk": pick(ess_b, var, i), "ESS tail": pick(ess_t, var, i)})
        else:   # long vector (e.g. one rating per team-season): mean/SD/interval describe the spread of its elements' means
            means = np.array([arr.mean() for _, _, arr in items])
            rows.append({"Parameter": f"{var} ({len(items)} values)", "Mean": float(means.mean()), "SD": float(means.std()),
                         "Low": float(np.quantile(means, 0.055)), "High": float(np.quantile(means, 0.945)),
                         "R̂": float(np.nanmax(rhat.get(var, [np.nan]))), "ESS bulk": float(np.nanmin(ess_b.get(var, [np.nan]))),
                         "ESS tail": float(np.nanmin(ess_t.get(var, [np.nan])))})
    params = pd.DataFrame(rows)
    post = idata.posterior
    first = next(iter(post.data_vars))
    n_chains, n_draws = post[first].shape[0], post[first].shape[1]
    all_r = np.concatenate([v for v in rhat.values()]) if rhat else np.array([np.nan])
    all_eb = np.concatenate([v for v in ess_b.values()]) if ess_b else np.array([np.nan])
    all_et = np.concatenate([v for v in ess_t.values()]) if ess_t else np.array([np.nan])
    res = {"params": params, "chains": int(n_chains), "draws": int(n_draws), "max_rhat": float(np.nanmax(all_r)),
           "min_ess_bulk": float(np.nanmin(all_eb)), "min_ess_tail": float(np.nanmin(all_et)), **_sampler_stats(idata),
           "elements": [label for label, _, _, _ in elements]}
    _DIAG_CACHE[key] = (mtime, res)
    return res


def _grade(value, good, warn, higher_is_better=True):
    if value is None or not np.isfinite(value):
        return "pill", "—"
    ok = value >= good if higher_is_better else value <= good
    meh = value >= warn if higher_is_better else value <= warn
    return ("pill pill--good", "Good") if ok else (("pill pill--warn", "Check") if meh else ("pill pill--bad", "Poor"))


MODEL_TITLES = {"a": "Model A · Pace × efficiency", "b": "Model B · Team strength", "c": "Model C · Market-anchored"}


def health_card(key, d):
    if d is None:
        return html.Div([html.Div(MODEL_TITLES[key], className="health-name"),
                         html.P("Not trained yet. Press 1 · Train.", className="empty-sub")], className="health-card")
    n = d["chains"] * d["draws"]
    div_rate = d["divergences"] / n * 100 if np.isfinite(d["divergences"]) else np.nan
    rows = [("Max R̂", f"{d['max_rhat']:.3f}", _grade(d["max_rhat"], 1.01, 1.05, False)),
            ("Min bulk ESS", f"{d['min_ess_bulk']:.0f}", _grade(d["min_ess_bulk"], 400, 100)),
            ("Min tail ESS", f"{d['min_ess_tail']:.0f}", _grade(d["min_ess_tail"], 400, 100)),
            ("Divergences", "—" if not np.isfinite(d["divergences"]) else f"{d['divergences']} ({div_rate:.2f}%)",
             _grade(div_rate, 0.0, 0.5, False)),
            ("Min E-BFMI", f"{d['bfmi']:.2f}" if np.isfinite(d["bfmi"]) else "—", _grade(d["bfmi"], 0.3, 0.2)),
            ("Mean accept", f"{d['accept']:.2f}" if np.isfinite(d["accept"]) else "—", ("pill", "info"))]
    grid = []
    for name, val, (cls, word) in rows:
        grid += [html.Span(name), html.Span(val, className="val"), html.Span(word, className=cls)]
    return html.Div([html.Div(MODEL_TITLES[key], className="health-name"),
                     html.Div(f"{d['chains']} chains × {d['draws']} draws", className="kpi-sub"),
                     html.Div(grid, className="health-grid")], className="health-card")


def trace_figure(arr, label):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, column_widths=[0.52, 0.26, 0.22], horizontal_spacing=0.06,
                        subplot_titles=("Trace by chain", "Density by chain", "Rank histogram"))
    ranks = np.argsort(np.argsort(arr.ravel())).reshape(arr.shape)
    bins = np.linspace(0, arr.size, 21)
    lo, hi = np.quantile(arr, [0.001, 0.999])
    grid = np.linspace(lo, hi, 41)
    for c in range(arr.shape[0]):
        color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
        fig.add_trace(go.Scattergl(y=arr[c], mode="lines", line=dict(color=color, width=1), opacity=0.75, name=f"Chain {c + 1}",
                                   legendgroup=str(c)), row=1, col=1)
        dens, edges = np.histogram(arr[c], bins=grid, density=True)
        fig.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=dens, mode="lines", line=dict(color=color, width=2, shape="spline"),
                                 name=f"Chain {c + 1}", legendgroup=str(c), showlegend=False), row=1, col=2)
        counts, _ = np.histogram(ranks[c], bins=bins)
        fig.add_trace(go.Bar(x=np.arange(20) + c / (arr.shape[0] + 1), y=counts, marker_color=color, width=1 / (arr.shape[0] + 1),
                             name=f"Chain {c + 1}", legendgroup=str(c), showlegend=False), row=1, col=3)
    fig.add_hline(y=arr.shape[1] / 20, line_dash="dot", line_color=theme["text_axis"], row=1, col=3)
    apply_plotly_theme(fig, label)
    fig.update_layout(height=380, bargap=0, margin=dict(t=90, b=40, l=50, r=20),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.2, yanchor="bottom"))
    fig.update_annotations(font=dict(color=theme["text_axis"], size=12))
    fig.update_xaxes(title_text="Draw", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Rank bin", showticklabels=False, row=1, col=3)
    return fig


def bps_figure(fit, market):
    from plotly.subplots import make_subplots
    h = fit.get("history")
    title = "Totals blend" if market == "total" else "Spreads blend"
    if not h or not h.get("theta"):
        fig = go.Figure(go.Bar(x=[AGENT_NAMES[a] for a in AGENTS], y=fit["theta_mean"][1:],
                               marker_color=[AGENT_COLORS[a] for a in AGENTS]))
        return apply_plotly_theme(fig, f"{title} · final weights (re-run step 2 to record their path)").update_layout(height=320)
    theta = np.asarray(h["theta"])
    x = np.arange(1, len(theta) + 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.08,
                        subplot_titles=("Weight on each model", "Particle effective sample size"))
    for j, a in enumerate(AGENTS, start=1):
        fig.add_trace(go.Scatter(x=x, y=theta[:, j], mode="lines", name=AGENT_NAMES[a], line=dict(color=AGENT_COLORS[a], width=2.5),
                                 customdata=h["game_date"], hovertemplate="Game %{x} · %{customdata}<br>θ %{y:.3f}<extra></extra>"),
                      row=1, col=1)
    fig.add_hline(y=1 / 3, line_dash="dot", line_color=theme["text_muted"], row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=h["ess"], mode="lines", name="ESS", line=dict(color=theme["accent"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(198,244,50,0.12)", showlegend=False), row=2, col=1)
    apply_plotly_theme(fig, title)
    fig.update_layout(height=430, margin=dict(t=80, b=40, l=50, r=20))
    fig.update_annotations(font=dict(color=theme["text_axis"], size=12))
    fig.update_xaxes(title_text="Graded games, in date order", row=2, col=1)
    return fig


@app.callback(Output("diag-health", "children"), Output("diag-table", "children"), Output("diag-bps", "children"),
              Input("tabs", "value"), Input("diag-refresh-btn", "n_clicks"))
def render_diagnostics(tab, _):
    if tab != "tab4":
        return no_update, no_update, no_update
    diags = {k: model_diagnostics(k) for k in ("a", "b", "c")}
    health = html.Div([health_card(k, d) for k, d in diags.items()], className="health")
    frames = [d["params"].assign(Model=k.upper()) for k, d in diags.items() if d is not None]
    if frames:
        p = pd.concat(frames, ignore_index=True)
        tbl = pd.DataFrame({"Model": p["Model"], "Parameter": p["Parameter"], "Mean": p["Mean"].round(3), "SD": p["SD"].round(3),
                            "89% interval": [f"{lo:.2f} to {hi:.2f}" for lo, hi in zip(p["Low"], p["High"])],
                            "R̂": p["R̂"].round(3), "ESS bulk": p["ESS bulk"].round(0), "ESS tail": p["ESS tail"].round(0)})
        cond = [{"if": {"filter_query": "{R̂} > 1.01", "column_id": "R̂"}, "color": "#FFB547", "fontWeight": "bold"},
                {"if": {"filter_query": "{R̂} > 1.05", "column_id": "R̂"}, "color": theme["negative"], "fontWeight": "bold"},
                {"if": {"filter_query": "{ESS bulk} < 400", "column_id": "ESS bulk"}, "color": "#FFB547", "fontWeight": "bold"},
                {"if": {"filter_query": "{ESS tail} < 400", "column_id": "ESS tail"}, "color": "#FFB547", "fontWeight": "bold"}]
        params = table(tbl, page_size=20, cond=cond)
    else:
        params = empty_state("No trained models yet.", "Press 1 · Train, then come back here.")
    if BPS_BUNDLE_PATH.exists():
        with open(BPS_BUNDLE_PATH, "rb") as f:
            b = pickle.load(f)
        chips = []
        for market, label in (("total", "Totals"), ("margin", "Spreads")):
            fit = b[market]
            t = fit["theta_mean"]
            chips.append(kpi_card(f"{label} weights", f"A {t[1]:.2f} · B {t[2]:.2f} · C {t[3]:.2f}",
                                  f"intercept {t[0]:+.1f} · noise σ {fit['vol_sd']:.1f} · ESS {fit['ess']:.0f} of {b['n_particles']} · {fit['n_obs']} games"))
        bps = [html.Div(chips, className="kpis"),
               html.P(f"Fitted with p = {b['n_particles']}, δ = {b['delta']:g}, β = {b['beta']:g}. Weights start at ⅓ each (dotted line). "
                      "A weight that keeps drifting means that model's usefulness is changing; an ESS that collapses toward 1 means "
                      "the particles have degenerated, so try more particles.", className="note", style={"margin": "12px 0"}),
               html.Div([html.Div(dcc.Graph(figure=bps_figure(b[m], m), config={"displayModeBar": False}), className="inner")
                         for m in ("total", "margin")], className="grid-even")]
    else:
        bps = empty_state("BPS not fitted yet.", "Press 2 · BPS once a few predicted games have final scores.")
    return health, params, bps


@app.callback(Output("diag-param", "options"), Output("diag-param", "value"), Input("diag-model", "value"),
              Input("tabs", "value"), Input("diag-refresh-btn", "n_clicks"), State("diag-param", "value"))
def diag_param_options(key, tab, _, current):
    if tab != "tab4":
        return no_update, no_update
    d = model_diagnostics(key)
    if d is None:
        return [], None
    opts = [{"label": e, "value": e} for e in d["elements"]]
    return opts, current if current in d["elements"] else (d["elements"][0] if d["elements"] else None)


@app.callback(Output("diag-trace", "children"), Input("diag-model", "value"), Input("diag-param", "value"))
def diag_trace(key, label):
    idata = load_idata(key)
    if idata is None:
        return empty_state("This model hasn't been trained yet.", "Press 1 · Train.")
    for lbl, _, _, arr in _param_elements(idata, key):
        if lbl == label:
            return html.Div(dcc.Graph(figure=trace_figure(arr, f"{MODEL_TITLES[key]} · {lbl}"), config={"displayModeBar": False}),
                            className="inner", style={"padding": "6px"})
    return empty_state("Pick a parameter.")


if __name__ == "__main__":
    app.run(jupyter_mode=JUPYTER_MODE, debug=False) if "ipykernel" in sys.modules else app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
