import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
import nba_model_core as core  # noqa: E402

HISTORICAL = Path(__file__).parent.parent / "data" / "historical_nba_cumulative_stats.csv"


@pytest.fixture(scope="module")
def hist():
    h = pd.read_csv(HISTORICAL, dtype={"game_id": str})
    return h[h["season"] == "2025-26"].reset_index(drop=True)


def test_side_stats_derivations(hist):
    s = core.side_stats(hist, "home")
    row = hist.iloc[200]
    oreb = row["home_REB/G"] - row["home_D_REB/G"]
    assert s.loc[200, "opp_pts"] == pytest.approx(row["home_PTS/G"] - row["home_plus"])
    assert s.loc[200, "poss"] == pytest.approx(row["home_FGA/G"] - oreb + row["home_TO/G"] + 0.44 * row["home_FTA/G"])


def test_shrinkage_and_first_game(hist):
    prior = core.league_prior(hist)
    assert 95 < prior["poss"] < 110 and 105 < prior["pts"] < 125
    f = core.build_features(hist, prior)
    first = f[f["home_games"] == 0].iloc[0]
    # A team with no games gets exactly the league average, not the CSV's zeros.
    assert first["home_pace"] == pytest.approx(prior["poss"])
    assert first["home_ortg"] == pytest.approx(prior["rtg"])
    shrunk = core.shrink([120.0], [10], 110.0, k=10)
    assert shrunk.iloc[0] == pytest.approx(115.0)


def test_rest_days_from_schedule(hist):
    f = core.build_features(hist)
    assert f["home_rest"].between(1, core.REST_CAP).all()
    assert f["home_b2b"].isin([0, 1]).all()
    assert 0.1 < f["home_b2b"].mean() < 0.3  # NBA teams play roughly 14-18 back-to-backs a season


def test_devig_and_prices():
    assert core.devig_pair([-110], [-110]).iloc[0] == pytest.approx(0.5)
    p = core.devig_pair([-150], [130]).iloc[0]
    assert 0.55 < p < 0.6
    assert core.american_to_decimal([150]).iloc[0] == pytest.approx(2.5)
    assert core.bet_profit(True, -110) == pytest.approx(100 / 110)
    assert core.bet_profit(False, 150) == -1.0
    assert core.bet_profit(None, -110) == 0.0


def _snap(gid, book, total, over, under, spread, label="morning", ts="2026-10-21T15:23:00Z"):
    return {"snapshot_utc": ts, "snapshot_label": label, "game_id": gid, "bookmaker": book,
            "total": total, "over_price": over, "under_price": under, "spread_home": spread,
            "spread_home_price": -110, "spread_away_price": -110}


def test_market_lines_consensus_and_best_price():
    odds = pd.DataFrame([_snap("g1", "dk", 224.5, -110, -110, -4.5), _snap("g1", "fd", 224.5, -105, -115, -4.5),
                         _snap("g1", "mgm", 225.5, -110, -110, -5.5),
                         _snap("g1", "dk", 230.5, -110, -110, -6.5, ts="2026-10-21T10:00:00Z")])  # older, ignored
    m = core.market_lines(odds).set_index("game_id").loc["g1"]
    assert m["line_total"] == 224.5 and m["line_spread_home"] == -4.5 and m["n_books"] == 3
    assert m["best_over_price"] == -105 and m["best_under_price"] == -110
    assert core.market_lines(odds, "closing").empty


def test_results_and_clv():
    assert core.total_result(225, 224.5) == "over" and core.total_result(224, 224) == "push"
    assert core.spread_result(5, -4.5) == "home_cover" and core.spread_result(4, -4.5) == "away_cover"
    assert core.clv_points("total", "over", 224.5, 226.5) == 2.0
    assert core.clv_points("total", "under", 224.5, 226.5) == -2.0
    assert core.clv_points("spread", "home_cover", 4.5, 6.0) == 1.5   # implied home margin moved toward home


def test_bps_learns_the_better_agent():
    rng = np.random.default_rng(0)
    truth = rng.normal(225, 15, 400)
    good = truth + rng.normal(0, 3, 400)
    bad = 225 + rng.normal(0, 15, 400)
    meh = truth + rng.normal(0, 12, 400)
    df = pd.DataFrame({"game_id": range(400), "game_date": pd.date_range("2026-10-21", periods=400).astype(str),
                       "y": truth, "a": good, "b": bad, "c": meh, "sa": 3.0, "sb": 15.0, "sc": 12.0})
    fit = core.fit_bps(df, ["a", "b", "c"], ["sa", "sb", "sc"], "y", 300, 0.99, 0.99, seed=1)
    theta = fit["theta_mean"]
    assert theta[1] > theta[2] and theta[1] > theta[3]
    flt = core.BPSFilter.from_state(fit["state"])
    pred = flt.predict([230, 225, 228], [3, 15, 12], 4000)
    assert abs(pred["mean"] - 230) < 4


def test_backtest_profit_and_clv():
    draws = np.sort(np.random.default_rng(1).normal(230, 5, 2000))
    prep = pd.DataFrame([{
        "game_id": "g1", "_total_sorted": draws, "_margin_sorted": None,
        "line_total": 220.5, "line_implied_home_margin": np.nan, "actual_total": 231, "actual_margin": 3,
        "total_result": "over", "spread_result": None, "best_over_price": 100, "best_under_price": -120,
        "close_total": 222.5, "close_implied_home_margin": np.nan,
        "best_home_price": np.nan, "best_away_price": np.nan}])
    bt = core.backtest(prep, 0.6)
    t = bt["total"]
    assert t["n_bets"] == 1 and t["win_rate"] == 100.0
    assert t["units"] == pytest.approx(1.0)      # +100 pays even money
    assert t["avg_clv"] == pytest.approx(2.0)    # line moved 220.5 -> 222.5 toward the over
    assert t["breakeven"] == pytest.approx(50.0)
    assert bt["spread"]["n_bets"] == 0
