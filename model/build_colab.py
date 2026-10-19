#!/usr/bin/env python3
"""Build model/NBA_Dashboard.ipynb: the core module, the three models and the dashboard in one
Colab notebook. The .py files stay the source of truth; re-run this after changing them:

    python model/build_colab.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "NBA_Dashboard.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def module_body(name):
    src = (HERE / name).read_text()
    src = src.replace("from __future__ import annotations\n", "")
    return src.lstrip()


def split_sections(src):
    """Split at the '# ====' banner comments so each section gets its own cell."""
    parts, current = [], []
    lines = src.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if lines[i].startswith("# ====") and i + 2 < len(lines) and lines[i + 2].startswith("# ===="):
            if "".join(current).strip():
                parts.append("".join(current))
            current = lines[i:i + 3]
            i += 3
            continue
        current.append(lines[i])
        i += 1
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def build(write=True):
    dash_src = module_body("nba_dashboard.py")
    dash_src = re.sub(r'^#!/usr/bin/env python3\n', "", dash_src)
    cells = [
        md("""
# 🏀 NBA Projections & Insight Ledger

Three models, each forecasting **both** the total and the home margin, blended by Bayesian Predictive
Synthesis, with Brier skill, confusion matrices and the Money Ball bet rules (expected value or credible
interval). Data is read from the [vladk-III/NBA](https://github.com/vladk-III/NBA) repo, which the GitHub
Actions pipeline updates every day.

**How to use:** Runtime → Run all. The last cell opens the dashboard. Daily, after the overnight data run:
**1 Train → 2 Optimize BPS → 3 Predict**. Model state and `predictions_history.csv` are saved to Google Drive
under `Stonks/Basketball/NBA/2026-27`.

*Generated from `model/*.py` by `model/build_colab.py`; edit the .py files and rebuild rather than editing
this notebook.*
"""),
        code("!pip install -q pymc nutpie arviz netCDF4 dash plotly diskcache multiprocess requests"),
        md("## Core: features, Bayesian Predictive Synthesis, odds and grading"),
        code(module_body("nba_model_core.py")),
        md("## Models A, B and C (each forecasts total and margin)"),
        code(module_body("nba_models.py")),
        md("## Dashboard: data, pipelines and UI"),
    ]
    cells += [code(part) for part in split_sections(dash_src)]
    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if write:
        OUT.write_text(text)
    return text, len(cells)


if __name__ == "__main__":
    _, n = build()
    print(f"Wrote {OUT} ({n} cells)")
