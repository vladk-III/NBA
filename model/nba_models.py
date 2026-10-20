"""
The three NBA forecasting models. Each predicts BOTH the game total and the home margin, so
Bayesian Predictive Synthesis can blend all three in each market.

  A  Pace x efficiency   total and margin regressed on the possessions x points-per-possession
                         expectation built from both teams' shrunk pace and ratings, plus rest.
  B  Team strength       every (season, team) gets an offense and defense rating; ratings carry
                         over between seasons with an AR(1) pull toward average. Home and away
                         points share a game-level pace effect (bivariate normal).
  C  Market-anchored     the morning line plus small learned adjustments for how far the pace x
                         efficiency expectation sits from it, and for rest. With no graded lines
                         yet it falls back to its priors: the line itself, with typical NBA error.

Posterior predictive draws are computed in numpy from the saved posterior samples, so the same
code serves any slate of games without rebuilding the PyMC models.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

LOG = logging.getLogger("nba_models")

TOTAL_COVS = ["home_b2b", "away_b2b", "home_rest", "away_rest"]
MARGIN_COVS = ["home_b2b", "away_b2b", "rest_diff"]


class Standardizer:
    def __init__(self, cols):
        self.cols, self.mean_, self.std_ = list(cols), None, None

    def fit(self, df):
        self.mean_ = df[self.cols].mean()
        self.std_ = df[self.cols].std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, df):
        return ((df[self.cols] - self.mean_) / self.std_).fillna(0.0).to_numpy(float)

    def get_state(self):
        return {"cols": self.cols, "mean_": self.mean_.to_dict(), "std_": self.std_.to_dict()}

    @classmethod
    def from_state(cls, s):
        obj = cls(s["cols"])
        obj.mean_, obj.std_ = pd.Series(s["mean_"]), pd.Series(s["std_"])
        return obj


def _post(idata, name):
    """Posterior samples flattened over chains: shape (draws, ...)."""
    v = idata.posterior[name].values
    return v.reshape((-1,) + v.shape[2:])


def _student_t(rng, nu, size):
    """Standard Student-t draws with per-draw degrees of freedom nu (shape (S,))."""
    z = rng.standard_normal(size)
    g = rng.chisquare(np.broadcast_to(nu[:, None], size))
    return z / np.sqrt(g / nu[:, None])


# =============================================================================
# MODEL A — pace x efficiency
# =============================================================================
def model_a(train: pd.DataFrame, tot_std: Standardizer, mar_std: Standardizer, center: float):
    Zt, Zm = tot_std.transform(train), mar_std.transform(train)
    exp_t = train["exp_total"].to_numpy(float) - center
    exp_m = train["exp_margin"].to_numpy(float)
    with pm.Model() as m:
        a_t = pm.Normal("a_t", 0.0, 5.0)
        b_t = pm.Normal("b_t", 1.0, 0.5)
        g_t = pm.Normal("g_t", 0.0, 2.0, shape=Zt.shape[1])
        a_m = pm.Normal("a_m", 2.0, 2.0)          # home court
        b_m = pm.Normal("b_m", 1.0, 0.5)
        g_m = pm.Normal("g_m", 0.0, 2.0, shape=Zm.shape[1])
        s_t = pm.Gamma("sigma_t", mu=18.0, sigma=5.0)
        s_m = pm.Gamma("sigma_m", mu=12.5, sigma=4.0)
        nu_t = pm.Deterministic("nu_t", 2.0 + pm.Gamma("nu_t_raw", 2.0, 0.1))
        nu_m = pm.Deterministic("nu_m", 2.0 + pm.Gamma("nu_m_raw", 2.0, 0.1))
        pm.StudentT("total", nu=nu_t, mu=center + a_t + b_t * exp_t + pt.dot(Zt, g_t), sigma=s_t,
                    observed=train["total_points"].to_numpy(float))
        pm.StudentT("margin", nu=nu_m, mu=a_m + b_m * exp_m + pt.dot(Zm, g_m), sigma=s_m,
                    observed=train["margin"].to_numpy(float))
    return m


def predict_a(idata, df, tot_std, mar_std, center, rng):
    Zt, Zm = tot_std.transform(df), mar_std.transform(df)
    exp_t = df["exp_total"].to_numpy(float) - center
    exp_m = df["exp_margin"].to_numpy(float)
    mu_t = center + _post(idata, "a_t")[:, None] + _post(idata, "b_t")[:, None] * exp_t + _post(idata, "g_t") @ Zt.T
    mu_m = _post(idata, "a_m")[:, None] + _post(idata, "b_m")[:, None] * exp_m + _post(idata, "g_m") @ Zm.T
    tot = mu_t + _post(idata, "sigma_t")[:, None] * _student_t(rng, _post(idata, "nu_t"), mu_t.shape)
    mar = mu_m + _post(idata, "sigma_m")[:, None] * _student_t(rng, _post(idata, "nu_m"), mu_m.shape)
    return tot, mar


# =============================================================================
# MODEL B — team strength with season-to-season carry-over
# =============================================================================
def team_index(frames: list[pd.DataFrame]) -> dict:
    """(season, team) keys across training and prediction games, ordered by season, plus each
    key's previous-season key (or -1) for the AR(1) carry-over."""
    keys = set()
    for f in frames:
        for side in ("home_team", "away_team"):
            keys.update(zip(f["season"].astype(str), f[side]))
    seasons = sorted({s for s, _ in keys})
    ordered = sorted(keys, key=lambda k: (seasons.index(k[0]), k[1]))
    pos = {k: i for i, k in enumerate(ordered)}
    prev = []
    for s, t in ordered:
        i = seasons.index(s)
        prev.append(pos.get((seasons[i - 1], t), -1) if i > 0 else -1)
    return {"keys": [list(k) for k in ordered], "seasons": seasons, "prev": prev}


def _key_idx(index, df, side):
    pos = {tuple(k): i for i, k in enumerate(index["keys"])}
    return np.array([pos[(str(s), t)] for s, t in zip(df["season"], df[f"{side}_team"])], dtype=int)


def model_b(train: pd.DataFrame, index: dict):
    keys, seasons, prev = index["keys"], index["seasons"], np.array(index["prev"])
    season_of = np.array([seasons.index(k[0]) for k in keys])
    h_idx, a_idx = _key_idx(index, train, "home"), _key_idx(index, train, "away")
    y = np.column_stack([train["home_points"].to_numpy(float), train["away_points"].to_numpy(float)])
    with pm.Model() as m:
        mu = pm.Normal("mu", 115.0, 5.0)
        hca = pm.Normal("hca", 2.0, 1.5)
        rho = pm.Beta("rho", 6.0, 3.0)
        sd_o = pm.HalfNormal("sd_off", 5.0)
        sd_d = pm.HalfNormal("sd_def", 5.0)
        z_o = pm.Normal("z_off", 0.0, 1.0, shape=len(keys))
        z_d = pm.Normal("z_def", 0.0, 1.0, shape=len(keys))
        off, dfn = pt.zeros(len(keys)), pt.zeros(len(keys))
        for s in range(len(seasons)):
            idx = np.where(season_of == s)[0]
            if s == 0:
                off = pt.set_subtensor(off[idx], sd_o * z_o[idx])
                dfn = pt.set_subtensor(dfn[idx], sd_d * z_d[idx])
                continue
            has_prev = prev[idx] >= 0
            p_idx = np.where(has_prev, prev[idx], 0)
            carry_o = pt.switch(has_prev, rho * off[p_idx], 0.0)
            carry_d = pt.switch(has_prev, rho * dfn[p_idx], 0.0)
            scale = pt.sqrt(1.0 - rho ** 2)
            off = pt.set_subtensor(off[idx], carry_o + scale * sd_o * z_o[idx])
            dfn = pt.set_subtensor(dfn[idx], carry_d + scale * sd_d * z_d[idx])
        off = pm.Deterministic("off", off)
        dfn = pm.Deterministic("def", dfn)
        s_e = pm.HalfNormal("sigma_team", 15.0)
        s_g = pm.HalfNormal("sigma_game", 10.0)
        mu_h = mu + hca / 2 + off[h_idx] - dfn[a_idx]
        mu_a = mu - hca / 2 + off[a_idx] - dfn[h_idx]
        var = s_e ** 2 + s_g ** 2
        cov = pt.stack([pt.stack([var, s_g ** 2]), pt.stack([s_g ** 2, var])])
        pm.MvNormal("pts", mu=pt.stack([mu_h, mu_a], axis=1), cov=cov, observed=y)
    return m


def predict_b(idata, df, index, rng):
    h_idx, a_idx = _key_idx(index, df, "home"), _key_idx(index, df, "away")
    off, dfn = _post(idata, "off"), _post(idata, "def")
    mu, hca = _post(idata, "mu")[:, None], _post(idata, "hca")[:, None]
    s_e, s_g = _post(idata, "sigma_team")[:, None], _post(idata, "sigma_game")[:, None]
    mu_h = mu + hca / 2 + off[:, h_idx] - dfn[:, a_idx]
    mu_a = mu - hca / 2 + off[:, a_idx] - dfn[:, h_idx]
    g = s_g * rng.standard_normal(mu_h.shape)
    h = mu_h + g + s_e * rng.standard_normal(mu_h.shape)
    a = mu_a + g + s_e * rng.standard_normal(mu_h.shape)
    return h + a, h - a


# =============================================================================
# MODEL C — market-anchored
# =============================================================================
KALSHI_GAP_CAP = {"total": 6.0, "margin": 4.0}   # points; a bigger gap is a thin or stale market, not information
KALSHI_COLS = {"total": ("kalshi_total", "line_total"), "margin": ("kalshi_home_margin", "line_implied_home_margin")}


def kalshi_gap(df: pd.DataFrame, market: str = "total") -> np.ndarray:
    """Kalshi's implied total (or home margin) minus the books' consensus, capped; 0 where Kalshi has no market."""
    kcol, lcol = KALSHI_COLS[market]
    if kcol not in df:
        return np.zeros(len(df))
    gap = df[kcol].to_numpy(float) - df[lcol].to_numpy(float)
    cap = KALSHI_GAP_CAP[market]
    return np.nan_to_num(np.clip(gap, -cap, cap), nan=0.0)


def model_c(train: pd.DataFrame, tot_std: Standardizer, mar_std: Standardizer):
    """train needs line_total and line_implied_home_margin; kalshi_total and kalshi_home_margin are
    optional. Works with zero rows (priors only)."""
    Zt, Zm = tot_std.transform(train), mar_std.transform(train)
    line_t = train["line_total"].to_numpy(float)
    line_m = train["line_implied_home_margin"].to_numpy(float)
    gap_t = train["exp_total"].to_numpy(float) - line_t
    gap_m = train["exp_margin"].to_numpy(float) - line_m
    gap_k = kalshi_gap(train, "total")
    gap_km = kalshi_gap(train, "margin")
    with pm.Model() as m:
        d0 = pm.Normal("d0", 0.0, 1.5)
        d1 = pm.Normal("d1", 0.0, 0.2)
        g_t = pm.Normal("g_t", 0.0, 0.75, shape=Zt.shape[1])
        k1 = pm.Normal("k1", 0.0, 0.3)   # weight on where Kalshi's total sits relative to the books' line
        e0 = pm.Normal("e0", 0.0, 1.0)
        e1 = pm.Normal("e1", 0.0, 0.2)
        g_m = pm.Normal("g_m", 0.0, 0.75, shape=Zm.shape[1])
        k2 = pm.Normal("k2", 0.0, 0.3)   # the same for Kalshi's spread vs the books' spread
        s_t = pm.Gamma("sigma_t", mu=18.0, sigma=3.0)
        s_m = pm.Gamma("sigma_m", mu=12.5, sigma=2.5)
        nu = pm.Deterministic("nu", 2.0 + pm.Gamma("nu_raw", 4.0, 0.2))
        if len(train):
            pm.StudentT("total", nu=nu, mu=line_t + d0 + d1 * gap_t + k1 * gap_k + pt.dot(Zt, g_t), sigma=s_t,
                        observed=train["total_points"].to_numpy(float))
            pm.StudentT("margin", nu=nu, mu=line_m + e0 + e1 * gap_m + k2 * gap_km + pt.dot(Zm, g_m), sigma=s_m,
                        observed=train["margin"].to_numpy(float))
    return m


def predict_c(idata, df, tot_std, mar_std, rng):
    """NaN for games without a line."""
    Zt, Zm = tot_std.transform(df), mar_std.transform(df)
    line_t = df["line_total"].to_numpy(float)
    line_m = df["line_implied_home_margin"].to_numpy(float)
    gap_t = df["exp_total"].to_numpy(float) - line_t
    gap_m = df["exp_margin"].to_numpy(float) - line_m
    # A state saved before the Kalshi inputs were added lacks k1/k2: treat them as zero weight.
    zero = np.zeros_like(_post(idata, "d0"))
    k1 = _post(idata, "k1") if "k1" in idata.posterior else zero
    k2 = _post(idata, "k2") if "k2" in idata.posterior else zero
    mu_t = (line_t + _post(idata, "d0")[:, None] + _post(idata, "d1")[:, None] * gap_t
            + k1[:, None] * kalshi_gap(df, "total") + _post(idata, "g_t") @ Zt.T)
    mu_m = (line_m + _post(idata, "e0")[:, None] + _post(idata, "e1")[:, None] * gap_m
            + k2[:, None] * kalshi_gap(df, "margin") + _post(idata, "g_m") @ Zm.T)
    nu = _post(idata, "nu")
    tot = mu_t + _post(idata, "sigma_t")[:, None] * _student_t(rng, nu, mu_t.shape)
    mar = mu_m + _post(idata, "sigma_m")[:, None] * _student_t(rng, nu, mu_m.shape)
    return tot, mar


# =============================================================================
# SAMPLING
# =============================================================================
def library_versions() -> str:
    """'pymc 5.28.5 · arviz 0.23.4 · nutpie 0.16.11 · xarray 2026.9.0', for status lines and bug reports."""
    import importlib
    out = []
    for name in ("pymc", "arviz", "nutpie", "xarray"):
        try:
            out.append(f"{name} {importlib.import_module(name).__version__}")
        except Exception:
            out.append(f"{name} —")
    return " · ".join(out)


def resolve_sampler(choice: str = "auto") -> str:
    if choice == "auto":
        return "nutpie" if importlib.util.find_spec("nutpie") is not None else "pymc"
    return choice


def sample(model, draws: int, tune: int, chains: int, seed: int, sampler: str = "auto"):
    sampler = resolve_sampler(sampler)
    kw = dict(model=model, draws=draws, tune=tune, chains=chains, progressbar=False, random_seed=seed)
    if sampler != "pymc":
        try:
            return pm.sample(nuts_sampler=sampler, **kw)
        except Exception as e:  # fall back rather than fail a daily run
            LOG.warning("%s sampler failed (%s); falling back to PyMC NUTS.", sampler, e)
    return pm.sample(cores=1, **kw)


def _netcdf_attrs(attrs: dict) -> dict:
    """netCDF stores only strings, numbers and flat arrays as attributes; anything else becomes JSON text."""
    def plain(v):
        return isinstance(v, (str, bytes, int, float, np.number, np.ndarray)) and not isinstance(v, (bool, np.bool_))
    out = {}
    for k, v in attrs.items():
        if plain(v) or (isinstance(v, (list, tuple)) and all(plain(x) for x in v)):
            out[k] = v
        else:
            out[k] = json.dumps(v, default=str)
    return out


SAVED_GROUPS = ("posterior", "sample_stats", "observed_data", "constant_data")


def _name(n):
    """netCDF names must be strings; some library combinations hand back ('posterior', 'a_m')."""
    return n if isinstance(n, str) else str(n[-1] if isinstance(n, tuple) and n else n)


def _clean_dataset(ds, group: str):
    """A fresh Dataset with one plain-named copy of each variable. Some library combinations hand back
    a group whose names carry a group prefix, ('posterior', 'a_m'), next to the plain dims, or that
    holds other groups' variables too; prefixed names from another group are dropped."""
    import xarray as xr

    def keep(name):
        return not (isinstance(name, tuple) and len(name) > 1 and name[0] != group)

    coords, data_vars = {}, {}
    for name, var in ds.coords.items():
        if keep(name) and _name(name) not in coords:
            coords[_name(name)] = (tuple(_name(d) for d in var.dims), np.asarray(var.values), _netcdf_attrs(var.attrs))
    for name, var in ds.data_vars.items():
        n = _name(name)
        if keep(name) and n not in coords and n not in data_vars:
            data_vars[n] = (tuple(_name(d) for d in var.dims), np.asarray(var.values), _netcdf_attrs(var.attrs))
    return xr.Dataset(data_vars, coords=coords, attrs=_netcdf_attrs(ds.attrs))


def save_idata(idata, path: str) -> None:
    """Pickle the groups the dashboard reads (posterior, sample_stats, observed_data, constant_data) as
    plain xarray Datasets with string names and attributes. Pickle instead of netCDF because netCDF
    rejects names or attributes that some ArviZ / nutpie / xarray / netCDF-library combinations
    produce (dict attributes, tuple names, characters the bundled netCDF library won't accept)."""
    if hasattr(idata, "children"):   # ArviZ 1.x DataTree
        groups = {g: idata[g].to_dataset() for g in SAVED_GROUPS if g in idata.children}
    else:                            # ArviZ 0.x InferenceData
        groups = {g: getattr(idata, g) for g in SAVED_GROUPS if g in idata.groups()}
    if "posterior" not in groups:
        raise ValueError("The sampler returned no posterior to save.")
    clean = {g: _clean_dataset(ds, g) for g, ds in groups.items()}
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(clean, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)   # a half-written file never replaces a good one


def read_idata(path: str):
    """What save_idata wrote, as the installed ArviZ's own type: InferenceData on 0.x, DataTree on 1.x."""
    import arviz as az
    import xarray as xr
    with open(path, "rb") as f:
        groups = pickle.load(f)
    if int(az.__version__.split(".")[0]) < 1:
        return az.InferenceData(**groups)
    return xr.DataTree.from_dict(groups)
