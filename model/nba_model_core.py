"""
Pure numpy/pandas pieces of the NBA modeler (no PyMC, no Dash), so they can be unit-tested
and reused: features, Bayesian Predictive Synthesis filter, odds handling and grading.

Everything here works on the files the data pipeline commits to data/:
  historical_nba_cumulative_stats.csv  pre-game season-to-date team stats, one row per game
  odds_snapshots.csv                   every US book's lines, morning and closing snapshots
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

# =============================================================================
# FEATURES
# =============================================================================
# League per-game averages used when no earlier season is available to estimate them.
LEAGUE_DEFAULTS = {"pts": 115.0, "poss": 99.5}
# Early-season shrinkage: a team's average after g games is blended with the league average
# as (g * team + k * league) / (g + k). With k = 10, a team's own numbers get half the weight
# after 10 games and about 85% by game 55.
SHRINK_GAMES = 10.0
REST_CAP = 4  # rest days beyond this add nothing


def side_stats(hist: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per-game averages for one side, derived only from the original 50 columns so every
    season (including the 2025-26 rows from before the pipeline) gets the same features.

    opp_pts = PTS/G - plus,  oreb = REB/G - D_REB/G,  poss = FGA - OREB + TOV + 0.44 * FTA.
    All are averages of per-game values, and the formulas are linear, so they are exact.
    """
    p = f"{side}_"
    out = pd.DataFrame(index=hist.index)
    out["games"] = pd.to_numeric(hist[p + "games"], errors="coerce").fillna(0)
    out["pts"] = hist[p + "PTS/G"]
    out["opp_pts"] = hist[p + "PTS/G"] - hist[p + "plus"]
    oreb = hist[p + "REB/G"] - hist[p + "D_REB/G"]
    out["poss"] = hist[p + "FGA/G"] - oreb + hist[p + "TO/G"] + 0.44 * hist[p + "FTA/G"]
    fga = hist[p + "FGA/G"].where(hist[p + "FGA/G"] > 0)
    out["fg3a_rate"] = hist[p + "FG3A/G"] / fga
    out["ft_rate"] = hist[p + "FTA/G"] / fga
    return out.astype(float)


def league_prior(hist: pd.DataFrame, min_games: int = 20) -> dict:
    """League-average points and possessions per team-game from rows where teams have played
    at least min_games (so early-season noise doesn't set the prior). Falls back to defaults."""
    vals = {"pts": [], "poss": [], "fg3a_rate": [], "ft_rate": []}
    for side in ("home", "away"):
        s = side_stats(hist, side)
        s = s[(s["games"] >= min_games) & (s["poss"] > 0)]
        for k in vals:
            vals[k].append(s[k])
    prior = {}
    for k, parts in vals.items():
        series = pd.concat(parts) if parts else pd.Series(dtype=float)
        prior[k] = float(series.mean()) if len(series) else np.nan
    prior["pts"] = prior["pts"] if np.isfinite(prior["pts"]) else LEAGUE_DEFAULTS["pts"]
    prior["poss"] = prior["poss"] if np.isfinite(prior["poss"]) else LEAGUE_DEFAULTS["poss"]
    prior["fg3a_rate"] = prior["fg3a_rate"] if np.isfinite(prior["fg3a_rate"]) else 0.42
    prior["ft_rate"] = prior["ft_rate"] if np.isfinite(prior["ft_rate"]) else 0.26
    prior["rtg"] = prior["pts"] / prior["poss"] * 100.0
    return prior


def shrink(value, games, prior_value, k: float = SHRINK_GAMES):
    value = pd.Series(value, dtype=float)
    games = pd.Series(games, dtype=float).fillna(0).clip(lower=0)
    filled = value.where(games > 0, prior_value).fillna(prior_value)
    return (games * filled + k * prior_value) / (games + k)


def rest_days_from_schedule(hist: pd.DataFrame) -> pd.DataFrame:
    """Days since each side's previous game in the same season, computed from the history file
    itself (so 2025-26 rows get it too). First game of a season -> NaN."""
    rows = []
    for side in ("home", "away"):
        rows.append(pd.DataFrame({"idx": hist.index, "season": hist["season"].values,
                                  "team": hist[f"{side}_team_abbreviation"].values,
                                  "date": pd.to_datetime(hist["game_date"]).values, "side": side}))
    long = pd.concat(rows, ignore_index=True).sort_values(["season", "team", "date"])
    long["prev"] = long.groupby(["season", "team"])["date"].shift(1)
    long["rest"] = (long["date"] - long["prev"]).dt.days
    wide = long.pivot_table(index="idx", columns="side", values="rest", aggfunc="first")
    return wide.reindex(hist.index).rename(columns={"home": "home_rest", "away": "away_rest"})


def build_features(hist: pd.DataFrame, prior: dict | None = None, k: float = SHRINK_GAMES) -> pd.DataFrame:
    """One row per game: shrunk pace/efficiency for each side, the pace x efficiency
    expectation for the score, rest, and the actual result when the game is final."""
    hist = hist.reset_index(drop=True)
    if "season" not in hist.columns:
        hist = hist.assign(season="unknown")
    prior = prior or league_prior(hist)
    df = pd.DataFrame({
        "game_id": hist["game_id"].astype(str),
        "game_date": hist["game_date"].astype(str).str[:10],
        "season": hist["season"].astype(str),
        "home_team": hist["home_team_abbreviation"],
        "away_team": hist["away_team_abbreviation"],
        "home_points": pd.to_numeric(hist["home_points"], errors="coerce"),
        "away_points": pd.to_numeric(hist["away_points"], errors="coerce"),
    })
    for side in ("home", "away"):
        s = side_stats(hist, side)
        g = s["games"]
        pace = shrink(s["poss"], g, prior["poss"], k)
        pts = shrink(s["pts"], g, prior["pts"], k)
        opp = shrink(s["opp_pts"], g, prior["pts"], k)
        df[f"{side}_games"] = g.values
        df[f"{side}_pace"] = pace.values
        df[f"{side}_ortg"] = (pts / pace * 100).values
        df[f"{side}_drtg"] = (opp / pace * 100).values
        df[f"{side}_fg3a_rate"] = shrink(s["fg3a_rate"], g, prior["fg3a_rate"], k).values
        df[f"{side}_ft_rate"] = shrink(s["ft_rate"], g, prior["ft_rate"], k).values

    rest = rest_days_from_schedule(hist)
    for side in ("home", "away"):
        r = rest[f"{side}_rest"]
        df[f"{side}_b2b"] = (r == 1).astype(float).values
        df[f"{side}_rest"] = r.fillna(REST_CAP).clip(upper=REST_CAP).values
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # Pace x efficiency expectation (no home court; models learn that themselves).
    lg_pace, lg_rtg = prior["poss"], prior["rtg"]
    exp_poss = df["home_pace"] * df["away_pace"] / lg_pace
    df["exp_poss"] = exp_poss
    df["exp_home"] = exp_poss * df["home_ortg"] * df["away_drtg"] / lg_rtg / 100.0
    df["exp_away"] = exp_poss * df["away_ortg"] * df["home_drtg"] / lg_rtg / 100.0
    df["exp_total"] = df["exp_home"] + df["exp_away"]
    df["exp_margin"] = df["exp_home"] - df["exp_away"]

    df["total_points"] = df["home_points"] + df["away_points"]
    df["margin"] = df["home_points"] - df["away_points"]
    return df


# =============================================================================
# ODDS
# =============================================================================
def american_to_prob(odds):
    o = pd.to_numeric(pd.Series(odds), errors="coerce")
    return pd.Series(np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0)), index=o.index).where(o.notna())


def american_to_decimal(odds):
    o = pd.to_numeric(pd.Series(odds), errors="coerce")
    return pd.Series(np.where(o < 0, 1.0 + 100.0 / -o, 1.0 + o / 100.0), index=o.index).where(o.notna())


def devig_pair(price_a, price_b):
    """Fair probability of side A from the two prices of one market (multiplicative de-vig)."""
    pa, pb = american_to_prob(price_a), american_to_prob(price_b)
    return pa / (pa + pb)


KALSHI_SIGMA = 18.0         # spread of NBA totals around the line; turns P(over X) into an implied median
KALSHI_SPREAD_SIGMA = 12.5  # the same for the home margin around the spread
KALSHI_MAX_SPREAD = 15      # cents; a wider bid/ask is too stale to price from, so fall back to the last trade


def kalshi_lines(kalshi: pd.DataFrame, label: str = "morning", market: str = "total") -> pd.DataFrame:
    """One row per game_id: the number Kalshi's markets imply.

    market="total" (KXNBATOTAL, "over X points") gives kalshi_total. market="spread" (KXNBASPREAD,
    "TEAM wins by over X points", with a team column) gives kalshi_home_margin.

    Each line X with price p (bid/ask mid, else last trade, in cents) implies a median of X + sigma * z(p),
    for the total or for that team's margin (negated for the away team). Lines are averaged with weights
    p(1 - p) * (volume + 1), so liquid lines near 50/50 count most and lines at 2 or 98 cents barely count."""
    spread = market == "spread"
    name, sigma = ("kalshi_home_margin", KALSHI_SPREAD_SIGMA) if spread else ("kalshi_total", KALSHI_SIGMA)
    prefix = "kalshi_spread" if spread else "kalshi"
    cols = ["game_id", name, f"{prefix}_volume", f"{prefix}_n_lines"]
    if kalshi is None or kalshi.empty:
        return pd.DataFrame(columns=cols)
    k = kalshi[(kalshi["snapshot_label"] == label) & kalshi["game_id"].notna()].copy()
    if k.empty:
        return pd.DataFrame(columns=cols)
    k["game_id"] = k["game_id"].astype(str)
    k = k[k["snapshot_utc"] == k.groupby("game_id")["snapshot_utc"].transform("max")]
    for c in ["line", "yes_bid", "yes_ask", "last_price", "volume"]:
        k[c] = pd.to_numeric(k[c], errors="coerce")
    quoted = (k["yes_bid"] > 0) & (k["yes_ask"] < 100) & (k["yes_ask"] - k["yes_bid"]).between(0, KALSHI_MAX_SPREAD)
    k["p"] = np.where(quoted, (k["yes_bid"] + k["yes_ask"]) / 2, k["last_price"]) / 100.0
    k = k[k["line"].notna() & k["p"].between(0.02, 0.98)]
    if spread:
        home = k["team"] == k["home_team_abbreviation"]
        away = k["team"] == k["away_team_abbreviation"]
        k = k[home | away].assign(side=np.where(home[home | away], 1.0, -1.0))
    if k.empty:
        return pd.DataFrame(columns=cols)
    z = NormalDist().inv_cdf
    k["implied"] = k["line"] + sigma * k["p"].map(z)
    if spread:
        k["implied"] *= k["side"]
    k["w"] = k["p"] * (1 - k["p"]) * (k["volume"].fillna(0).clip(lower=0) + 1)
    rows = [{"game_id": gid, name: float(np.average(g["implied"], weights=g["w"])),
             f"{prefix}_volume": float(g["volume"].fillna(0).sum()), f"{prefix}_n_lines": len(g)}
            for gid, g in k.groupby("game_id")]
    return pd.DataFrame(rows, columns=cols)


def market_lines(odds: pd.DataFrame, label: str = "morning") -> pd.DataFrame:
    """One row per game_id from a snapshot label: consensus (median) total and home spread,
    the median price at that consensus line, and the best price any book offers at it."""
    cols = ["game_id", "line_total", "line_spread_home", "over_price", "under_price",
            "best_over_price", "best_under_price", "home_price", "away_price",
            "best_home_price", "best_away_price", "n_books"]
    if odds is None or odds.empty:
        return pd.DataFrame(columns=cols)
    o = odds[(odds["snapshot_label"] == label) & odds["game_id"].notna()].copy()
    if o.empty:
        return pd.DataFrame(columns=cols)
    o["game_id"] = o["game_id"].astype(str)
    last = o.groupby("game_id")["snapshot_utc"].transform("max")
    o = o[o["snapshot_utc"] == last]
    for c in ["total", "spread_home", "over_price", "under_price", "spread_home_price", "spread_away_price"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")

    rows = []
    for gid, g in o.groupby("game_id"):
        row = {"game_id": gid, "n_books": int(g["bookmaker"].nunique())}
        t = g.dropna(subset=["total"])
        row["line_total"] = float(t["total"].median()) if len(t) else np.nan
        at = t[t["total"] == t["total"].median()] if len(t) else t
        if at.empty and len(t):  # median falls between two lines: use the most common one
            at = t[t["total"] == t["total"].mode().iloc[0]]
            row["line_total"] = float(at["total"].iloc[0])
        row["over_price"] = float(at["over_price"].median()) if len(at) else np.nan
        row["under_price"] = float(at["under_price"].median()) if len(at) else np.nan
        row["best_over_price"] = _best(at["over_price"])
        row["best_under_price"] = _best(at["under_price"])
        s = g.dropna(subset=["spread_home"])
        row["line_spread_home"] = float(s["spread_home"].median()) if len(s) else np.nan
        ats = s[s["spread_home"] == s["spread_home"].median()] if len(s) else s
        if ats.empty and len(s):
            ats = s[s["spread_home"] == s["spread_home"].mode().iloc[0]]
            row["line_spread_home"] = float(ats["spread_home"].iloc[0])
        row["home_price"] = float(ats["spread_home_price"].median()) if len(ats) else np.nan
        row["away_price"] = float(ats["spread_away_price"].median()) if len(ats) else np.nan
        row["best_home_price"] = _best(ats["spread_home_price"])
        row["best_away_price"] = _best(ats["spread_away_price"])
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _best(prices: pd.Series) -> float:
    """The price that pays most (highest decimal odds), as American odds."""
    p = pd.to_numeric(prices, errors="coerce").dropna()
    if p.empty:
        return np.nan
    return float(p.iloc[int(np.argmax(american_to_decimal(p).to_numpy()))])


# =============================================================================
# BAYESIAN PREDICTIVE SYNTHESIS — Rao-Blackwellized particle filter, K agents
# =============================================================================
# Same model as the NFL dashboard (McAlinn, Aastveit, Nakajima & West 2018), generalized from
# 2 agents to K. Synthesis: y_t = theta_t' x_t + nu_t,  x_t = (1, x_1t, ..., x_Kt), one draw
# from each agent's forecast density N(mean_k, sd_k). theta_t is a random walk discounted by
# delta (R_t = C_{t-1}/delta); the residual variance is discounted by beta. Each particle keeps
# an analytic Kalman state for theta; particles are reweighted by the one-step predictive
# likelihood and systematically resampled when the ESS drops.
BPS_DEFAULTS = {"particles": 1000, "samples": 2000, "delta": 0.99, "beta": 0.99}
BPS_PRIOR_DOF = 7.0
BPS_ESS_FRAC = 0.5
BPS_PRIOR_INTERCEPT_SD = 5.0
BPS_PRIOR_WEIGHT_SD = 0.25
BPS_FALLBACK_OBS_VAR = {"total": 300.0, "margin": 150.0}


class BPSFilter:
    def __init__(self, n_agents: int, n_particles: int, delta: float = 0.99, beta: float = 0.99,
                 obs_var0: float = 300.0, seed: int = 20261021):
        self.k = int(n_agents)
        self.dim = self.k + 1
        self.n = int(n_particles)
        self.delta, self.beta = float(delta), float(beta)
        self.rng = np.random.default_rng(seed)
        m0 = np.concatenate([[0.0], np.full(self.k, 1.0 / self.k)])
        self.m = np.tile(m0, (self.n, 1))
        C0 = np.diag([BPS_PRIOR_INTERCEPT_SD ** 2] + [BPS_PRIOR_WEIGHT_SD ** 2] * self.k)
        self.C = np.tile(C0, (self.n, 1, 1))
        self.h = np.full(self.n, BPS_PRIOR_DOF)
        self.D = np.full(self.n, BPS_PRIOR_DOF * float(obs_var0))
        self.w = np.full(self.n, 1.0 / self.n)

    def _draw_x(self, mu, sd, n):
        z = self.rng.standard_normal((n, self.k))
        X = np.ones((n, self.dim))
        X[:, 1:] = np.asarray(mu, float) + np.asarray(sd, float) * z
        return X

    @property
    def ess(self) -> float:
        return float(1.0 / np.sum(self.w ** 2))

    def theta_mean(self) -> np.ndarray:
        return self.w @ self.m

    def vol_mean(self) -> float:
        return float(np.sqrt(self.w @ (self.D / self.h)))

    def update(self, mu, sd, y: float) -> None:
        X = self._draw_x(mu, sd, self.n)
        R = self.C / self.delta
        f = np.einsum("ni,ni->n", X, self.m)
        RX = np.einsum("nij,nj->ni", R, X)
        S = self.D / self.h
        Q = np.einsum("ni,ni->n", X, RX) + S
        e = y - f
        loglik = -0.5 * (np.log(2 * np.pi * Q) + e ** 2 / Q)
        logw = np.log(self.w + 1e-300) + loglik
        logw -= logw.max()
        w = np.exp(logw)
        self.w = w / w.sum()

        K = RX / Q[:, None]
        self.m = self.m + K * e[:, None]
        C = R - K[:, :, None] * RX[:, None, :]
        self.C = 0.5 * (C + C.transpose(0, 2, 1))
        self.h = self.beta * self.h + 1.0
        self.D = self.beta * self.D + S * e ** 2 / Q

        if self.ess < BPS_ESS_FRAC * self.n:
            pos = (self.rng.random() + np.arange(self.n)) / self.n
            cs = np.cumsum(self.w)
            cs[-1] = 1.0
            idx = np.searchsorted(cs, pos)
            self.m, self.C, self.h, self.D = self.m[idx], self.C[idx], self.h[idx], self.D[idx]
            self.w = np.full(self.n, 1.0 / self.n)

    def predict(self, mu, sd, n_draws: int) -> dict:
        idx = self.rng.choice(self.n, size=int(n_draws), p=self.w / self.w.sum())
        X = self._draw_x(mu, sd, int(n_draws))
        R = self.C[idx] / self.delta
        f = np.einsum("di,di->d", X, self.m[idx])
        Q = np.einsum("di,dij,dj->d", X, R, X) + self.D[idx] / self.h[idx]
        draws = f + np.sqrt(Q) * self.rng.standard_normal(int(n_draws))
        return {"draws": draws, "mean": float(f.mean()), "var": float(Q.mean() + f.var())}

    def get_state(self) -> dict:
        return {"k": self.k, "n": self.n, "delta": self.delta, "beta": self.beta,
                "m": self.m, "C": self.C, "h": self.h, "D": self.D, "w": self.w}

    @classmethod
    def from_state(cls, state: dict, seed: int = 20261021) -> "BPSFilter":
        obj = cls(state["k"], state["n"], state["delta"], state["beta"], seed=seed)
        obj.m, obj.C, obj.h, obj.D, obj.w = state["m"], state["C"], state["h"], state["D"], state["w"]
        return obj


def fit_bps(df: pd.DataFrame, mu_cols: list[str], sd_cols: list[str], actual_col: str,
            n_particles: int, delta: float, beta: float, seed: int) -> dict:
    """Run the filter forward through graded games in date order."""
    d = df.dropna(subset=mu_cols + [actual_col]).copy()
    if len(d) < 3:
        raise RuntimeError(f"Only {len(d)} graded game(s) with forecasts from every model; BPS needs at least 3.")
    d = d.sort_values(["game_date", "game_id"])
    y = d[actual_col].to_numpy(float)
    MU = d[mu_cols].to_numpy(float)
    fallback_sd = np.sqrt(np.mean((MU - y[:, None]) ** 2, axis=0))
    SD = np.column_stack([
        np.where(np.isfinite(v := d[c].to_numpy(float)) & (v > 0), v, fallback_sd[i])
        for i, c in enumerate(sd_cols)])
    obs_var0 = max(0.5 * float(np.var(y - MU.mean(axis=1))), 1e-3)
    flt = BPSFilter(len(mu_cols), n_particles, delta, beta, obs_var0, seed)
    for t in range(len(y)):
        flt.update(MU[t], SD[t], y[t])
    return {"state": flt.get_state(), "n_obs": int(len(y)), "theta_mean": flt.theta_mean().tolist(),
            "vol_sd": flt.vol_mean(), "ess": flt.ess}


# =============================================================================
# GRADING
# =============================================================================
def total_result(actual_total, line_total):
    if pd.isna(actual_total) or pd.isna(line_total):
        return None
    if actual_total == line_total:
        return "push"
    return "over" if actual_total > line_total else "under"


def spread_result(actual_margin, line_spread_home):
    """line_spread_home is the home team's spread (e.g. -4.5 when home is favored)."""
    if pd.isna(actual_margin) or pd.isna(line_spread_home):
        return None
    net = actual_margin + line_spread_home
    if net == 0:
        return "push"
    return "home_cover" if net > 0 else "away_cover"


def bet_profit(won, price) -> float:
    """Profit in units for a 1-unit stake: won True/False, or None for a push."""
    if won is None or pd.isna(price):
        return 0.0 if won is None else np.nan
    return float(american_to_decimal([price]).iloc[0] - 1.0) if won else -1.0


def clv_points(market: str, side: str, bet_line: float, close_line: float) -> float:
    """Points of closing line value: positive means the line moved toward the bet.
    Totals use the total; spreads use the implied home margin (= -home spread)."""
    if pd.isna(bet_line) or pd.isna(close_line):
        return np.nan
    if market == "total":
        return close_line - bet_line if side == "over" else bet_line - close_line
    return close_line - bet_line if side == "home_cover" else bet_line - close_line


def credible_interval_bounds(conf_level: float) -> tuple[float, float]:
    lower = (100.0 - conf_level * 100.0) / 2.0
    return lower, 100.0 - lower


def side_vs_interval(market_val, lower, upper):
    """HIGH: the line is below the interval (bet over / home); LOW: above (under / away)."""
    if pd.isna(market_val) or pd.isna(lower) or pd.isna(upper):
        return None
    if market_val < lower:
        return "HIGH"
    if market_val > upper:
        return "LOW"
    return "NO_BET"


def quantile_sorted(s: np.ndarray, pct: float) -> float:
    pos = pct / 100.0 * (s.size - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, s.size - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def prob_above(sorted_draws: np.ndarray | None, line: float) -> float:
    if sorted_draws is None or pd.isna(line):
        return np.nan
    return float(1.0 - np.searchsorted(sorted_draws, line, side="right") / sorted_draws.size)


def prob_below(sorted_draws: np.ndarray | None, line: float) -> float:
    if sorted_draws is None or pd.isna(line):
        return np.nan
    return float(np.searchsorted(sorted_draws, line, side="left") / sorted_draws.size)


KELLY_FRACTION = 0.25   # quarter Kelly
KELLY_CAP = 0.05        # never suggest more than 5% of bankroll on one bet


def ev_choice(sorted_draws: np.ndarray | None, line: float, price_hi, price_lo) -> dict | None:
    """Expected value per 1-unit stake of each side of a line, from the model's draws.

    HIGH = over (totals) / home covers (spreads, line on the implied-home-margin scale);
    LOW = under / away covers. A push returns the stake, so
        EV = P(win) * (decimal - 1) - P(loss).
    Returns the better side with its EV, win probability, the breakeven win rate at that
    price, and a quarter-Kelly stake (fraction of bankroll, capped at KELLY_CAP)."""
    if sorted_draws is None or pd.isna(line):
        return None
    p_hi, p_lo = prob_above(sorted_draws, line), prob_below(sorted_draws, line)
    best = None
    for side, p_win, p_loss, price in (("HIGH", p_hi, p_lo, price_hi), ("LOW", p_lo, p_hi, price_lo)):
        price = price if pd.notna(price) else -110.0
        b = float(american_to_decimal([price]).iloc[0]) - 1.0
        ev = p_win * b - p_loss
        q = p_win / (p_win + p_loss) if p_win + p_loss > 0 else 0.0   # win chance excluding pushes
        kelly = max(0.0, (b * q - (1 - q)) / b) * KELLY_FRACTION if b > 0 else 0.0
        cand = {"side": side, "ev": float(ev), "p_win": float(p_win), "price": float(price),
                "breakeven": float(american_to_prob([price]).iloc[0]), "kelly": float(min(kelly, KELLY_CAP))}
        if best is None or cand["ev"] > best["ev"]:
            best = cand
    return best


def backtest(prep: pd.DataFrame, conf_level: float = 0.55, use_best_price: bool = True,
             rule: str = "interval", min_ev: float = 0.03) -> dict:
    """Money Ball on graded history. Two bet rules:
      interval  bet when the morning line is outside the BPS credible interval (conf_level)
      ev        bet the side whose expected value at the available price is at least min_ev
                (0.03 = 3 cents per unit staked)
    Profit is graded at the price you could have taken (best book by default) and CLV against
    the closing line. prep needs: _total_sorted, _margin_sorted (sorted draws), line_total,
    line_implied_home_margin, actual_total, actual_margin, total_result, spread_result, price
    columns, close_total, close_implied_home_margin."""
    if prep is None or prep.empty:
        return {"empty": True}
    lo_pct, hi_pct = credible_interval_bounds(conf_level)
    out = {"empty": False, "nominal_outside_pct": (1 - conf_level) * 100, "rule": rule}
    rows = []
    markets = [
        ("total", "_total_sorted", "line_total", "actual_total", "total_result", ("over", "under"),
         ("best_over_price", "best_under_price") if use_best_price else ("over_price", "under_price"),
         "close_total"),
        ("spread", "_margin_sorted", "line_implied_home_margin", "actual_margin", "spread_result",
         ("home_cover", "away_cover"),
         ("best_home_price", "best_away_price") if use_best_price else ("home_price", "away_price"),
         "close_implied_home_margin"),
    ]
    for market, draws_col, line_col, actual_col, result_col, (hi_side, lo_side), (hi_px, lo_px), close_col in markets:
        bets, n_usable, outside, n_cov = [], 0, 0, 0
        for _, r in prep.iterrows():
            s = r.get(draws_col)
            if s is None or not isinstance(s, np.ndarray) or s.size == 0:
                continue
            lo, hi = quantile_sorted(s, lo_pct), quantile_sorted(s, hi_pct)
            if pd.notna(r.get(actual_col)):
                n_cov += 1
                outside += int(r[actual_col] < lo or r[actual_col] > hi)
            if rule == "ev":
                choice = ev_choice(s, r.get(line_col), r.get(hi_px), r.get(lo_px))
                call = None if choice is None else (choice["side"] if choice["ev"] >= min_ev else "NO_BET")
                ev = choice["ev"] if choice is not None and call != "NO_BET" else np.nan
            else:
                call, ev = side_vs_interval(r.get(line_col), lo, hi), np.nan
            if call is None:
                continue
            n_usable += 1
            rows.append({"market": market, "game_id": r["game_id"], "call": call, "result": r.get(result_col)})
            if call == "NO_BET":
                continue
            side = hi_side if call == "HIGH" else lo_side
            price = r.get(hi_px if call == "HIGH" else lo_px)
            price = price if pd.notna(price) else -110.0
            result = r.get(result_col)
            won = None if result == "push" else (result == side)
            bets.append({"won": won, "price": price, "profit": bet_profit(won, price), "ev": ev,
                         "clv": clv_points(market, side, r.get(line_col), r.get(close_col)),
                         "breakeven": float(american_to_prob([price]).iloc[0])})
        b = pd.DataFrame(bets)
        graded = b[b["won"].notna()] if len(b) else b
        out[market] = {
            "n_games": n_usable, "n_bets": len(b),
            "selectivity": len(b) / n_usable * 100 if n_usable else 0.0,
            "n_graded": len(graded),
            "win_rate": float(graded["won"].astype(float).mean() * 100) if len(graded) else np.nan,
            "breakeven": float(graded["breakeven"].mean() * 100) if len(graded) else np.nan,
            "units": float(b["profit"].sum()) if len(b) else 0.0,
            "roi": float(b["profit"].sum() / len(b) * 100) if len(b) else np.nan,
            "avg_clv": float(b["clv"].mean()) if len(b) and b["clv"].notna().any() else np.nan,
            "avg_ev": float(b["ev"].mean() * 100) if len(b) and b["ev"].notna().any() else np.nan,
            "outside_pct": outside / n_cov * 100 if n_cov else np.nan,
        }
    out["calls"] = pd.DataFrame(rows, columns=["market", "game_id", "call", "result"])
    return out
