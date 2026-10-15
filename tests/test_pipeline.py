import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_pipeline import features, kalshi, merge, nba_cdn, odds
from nba_pipeline.legacy import recover_game_logs

FIXTURES = Path(__file__).parent / "fixtures"
HISTORICAL = Path(__file__).parent.parent / "data" / "historical_nba_cumulative_stats.csv"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def legacy_history():
    h = pd.read_csv(HISTORICAL, dtype={"game_id": str})
    return h[h["season"] == "2025-26"].reset_index(drop=True)


def test_builder_reproduces_legacy_historical_csv():
    """Rebuilding 2025-26 from recovered game logs gives back the original CSV."""
    h = legacy_history()
    logs = recover_game_logs(h)
    logs["season"] = "2025-26"
    for col in ("oreb",):
        logs[col] = np.nan
    games = h[["game_id", "game_date", "home_team_abbreviation", "away_team_abbreviation",
               "home_points", "away_points"]].assign(
        season="2025-26", game_type="regular", game_status=3, postponed=False, tipoff_utc=None)
    rebuilt = features.build_rows(games, logs, "2026-12-31").set_index("game_id")
    original = h.set_index("game_id").loc[rebuilt.index]

    text = ["game_date", "home_team_abbreviation", "away_team_abbreviation"]
    pd.testing.assert_frame_equal(rebuilt[text], original[text], check_dtype=False)
    exact = [c for c in features.LEGACY_COLUMNS[1:] if c not in text and not c.endswith("_plus")]
    pd.testing.assert_frame_equal(rebuilt[exact].astype(float), original[exact].astype(float),
                                  check_dtype=False, atol=1e-9, rtol=0)
    # The old source reported a fractional plus/minus in 3 games; we use the exact margin.
    for side in ("home", "away"):
        diff = (rebuilt[f"{side}_plus"] - original[f"{side}_plus"]).abs()
        assert (diff < 0.5).all()
        assert (diff > 1e-9).mean() < 0.05


def test_rest_days_and_new_columns():
    h = legacy_history().head(60)
    logs = recover_game_logs(h).assign(season="2025-26", oreb=10.0)
    games = h[["game_id", "game_date", "home_team_abbreviation", "away_team_abbreviation",
               "home_points", "away_points"]].assign(
        season="2025-26", game_type="regular", game_status=3, postponed=False, tipoff_utc=None)
    rows = features.build_rows(games, logs, "2026-12-31")
    first = rows.iloc[0]
    assert np.isnan(first["home_rest_days"]) and first["home_games"] == 0
    later = rows[rows["home_games"] > 0].iloc[0]
    assert later["home_rest_days"] >= 1
    assert later["home_back_to_back"] == int(later["home_rest_days"] == 1)
    assert later["home_OREB/G"] == 10.0


def test_merge_into_history_keeps_other_seasons_untouched():
    h = pd.read_csv(HISTORICAL, dtype={"game_id": str})
    new = pd.DataFrame([{**{c: 0.0 for c in features.COLUMNS}, "game_id": "0022600001",
                         "game_date": "2026-10-20", "season": "2026-27", "game_type": "regular"}])
    out = features.merge_into_history(h, new)
    assert len(out) == len(h) + 1
    pd.testing.assert_frame_equal(out[out["season"] == "2025-26"].reset_index(drop=True), h, check_dtype=False)
    # Rebuilding the same season replaces rather than duplicates.
    again = features.merge_into_history(out, new)
    assert len(again) == len(out)


def test_schedule_filters_preseason_and_tbd(monkeypatch):
    monkeypatch.setattr(nba_cdn, "get_json", lambda *a, **k: (load("schedule.json"), None))
    season, games = nba_cdn.fetch_schedule()
    assert season == "2026-27"
    assert [g["game_id"] for g in games] == ["0022600001", "0022600002"]
    assert games[0]["home_points"] == 120 and games[1]["home_points"] is None
    assert games[0]["game_date"] == "2026-10-20"


def test_parse_boxscore():
    rows = nba_cdn.parse_boxscore(load("boxscore.json")["game"], "2026-27")
    home, away = rows
    assert home["team"] == "BOS" and home["is_home"] and home["win"] == 1
    assert home["reb"] == 45 and home["tov"] == 13 and home["overtime_periods"] == 1
    assert away["tov"] == 14  # no turnoversTotal: players + team
    assert away["game_date"] == "2026-10-20"
    not_final = dict(load("boxscore.json")["game"], gameStatus=2)
    assert nba_cdn.parse_boxscore(not_final, "2026-27") is None


def test_kalshi_ticker_parsing():
    assert kalshi.date_code("2026-10-21") == "26OCT21"
    assert kalshi.split_teams("LALGSW") == ("LAL", "GSW")
    assert kalshi.split_teams("GSNY") == ("GSW", "NYK")
    assert kalshi.split_teams("XXXYYY") is None
    assert kalshi._cents({"yes_bid_dollars": "0.5500"}, "yes_bid") == 55.0
    assert kalshi._line({"ticker": "KXNBATOTAL-26OCT21LALGSW-219"}) == 219.0


def _games():
    return pd.DataFrame([{"game_id": "0022600002", "game_date": "2026-10-21", "home_team_abbreviation": "NYK",
                          "away_team_abbreviation": "BOS", "tipoff_utc": "2026-10-21T23:30:00Z", "game_status": 1}])


def test_kalshi_snapshot(monkeypatch):
    markets = {"markets": [
        {"ticker": "KXNBATOTAL-26OCT21BOSNYK-220", "event_ticker": "KXNBATOTAL-26OCT21BOSNYK",
         "floor_strike": 219.5, "yes_bid": 48, "yes_ask": 50, "last_price": 49, "volume": 5000},
        {"ticker": "KXNBATOTAL-26OCT21BOSNYK-225", "event_ticker": "KXNBATOTAL-26OCT21BOSNYK",
         "floor_strike": 224.5, "last_price": 30, "volume": 900},
        {"ticker": "KXNBATOTAL-26OCT22LALGSW-230", "event_ticker": "KXNBATOTAL-26OCT22LALGSW", "volume": 1}],
        "cursor": ""}
    trades = {"trades": [{"created_time": "2026-10-21T15:00:00Z", "yes_price": 49},
                         {"created_time": "2026-10-20T15:00:00Z", "yes_price": 44}], "cursor": ""}
    monkeypatch.setattr(kalshi, "get_json",
                        lambda url, **k: (trades if url.endswith("/trades") else markets, None))
    now = datetime(2026, 10, 21, 15, 30, tzinfo=timezone.utc)
    df = kalshi.fetch_snapshot("morning", "2026-10-21", _games(), now)
    assert len(df) == 2
    top = df[df["is_top_volume"] == True].iloc[0]  # noqa: E712
    assert top["line"] == 219.5 and top["open_price"] == 44 and top["price_change"] == 5
    assert top["game_id"] == "0022600002"


def test_odds_snapshot(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test")
    events = [{"id": "ev1", "commence_time": "2026-10-21T23:30:00Z",
               "home_team": "New York Knicks", "away_team": "Boston Celtics",
               "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "New York Knicks", "price": 120},
                                               {"name": "Boston Celtics", "price": -140}]},
                   {"key": "spreads", "outcomes": [{"name": "New York Knicks", "price": -110, "point": 3.5},
                                                   {"name": "Boston Celtics", "price": -110, "point": -3.5}]},
                   {"key": "totals", "outcomes": [{"name": "Over", "price": -108, "point": 221.5},
                                                  {"name": "Under", "price": -112, "point": 221.5}]}]}]}]

    class Resp:
        headers = {"x-requests-remaining": "497", "x-requests-used": "3"}

    seen = {}

    def fake_get(url, params=None, **k):
        seen.update(params)
        return events, Resp()

    monkeypatch.setattr(odds, "get_json", fake_get)
    now = datetime(2026, 10, 21, 15, 30, tzinfo=timezone.utc)
    df, quota = odds.fetch_snapshot("morning", "2026-10-21", _games(), now)
    assert quota["remaining"] == "497"
    assert seen["commenceTimeFrom"] == "2026-10-21T15:30:00Z"
    assert seen["commenceTimeTo"] == "2026-10-22T04:00:00Z"
    r = df.iloc[0]
    assert r["game_id"] == "0022600002" and r["ml_home"] == 120 and r["spread_home"] == 3.5
    assert r["total"] == 221.5 and r["under_price"] == -112


def test_merged_results_and_pushes():
    hist = pd.DataFrame([
        {"game_id": "1", "season": "2026-27", "home_points": 115, "away_points": 110},
        {"game_id": "2", "season": "2026-27", "home_points": 110, "away_points": 111},
        {"game_id": "3", "season": "2026-27", "home_points": np.nan, "away_points": np.nan},
        {"game_id": "4", "season": "2025-26", "home_points": 100, "away_points": 100}])
    def snap(gid, total, spread):
        row = {c: None for c in odds.COLUMNS}
        row.update({"snapshot_utc": "2026-10-21T15:30:00Z", "snapshot_label": "morning",
                    "game_id": gid, "bookmaker": "draftkings", "total": total, "spread_home": spread})
        return row
    o = pd.DataFrame([snap("1", 220.5, -3.5), snap("2", 221, 0.5), snap("3", 220, -2)])
    out = merge.build_merged(hist, o, pd.DataFrame(columns=kalshi.COLUMNS), {"2026-27"}).set_index("game_id")
    assert list(out.index) == ["1", "2", "3"]
    assert out.at["1", "Over Hit (DraftKings)"] == 1 and out.at["1", "Home Covered (DraftKings)"] == 1
    assert np.isnan(out.at["2", "Over Hit (DraftKings)"])  # 221 total on 221: push
    assert out.at["2", "Home Covered (DraftKings)"] == 0
    assert np.isnan(out.at["3", "Over Hit (DraftKings)"])  # not played yet
    assert out.at["1", "Consensus - Over/Under Total"] == 220.5


def test_refuses_to_rebuild_legacy_season(tmp_path, monkeypatch):
    import collect_nba
    hist = tmp_path / "hist.csv"
    pd.read_csv(HISTORICAL, dtype={"game_id": str}).to_csv(hist, index=False)
    monkeypatch.setattr(collect_nba, "HISTORICAL_FILE", hist)
    games = pd.DataFrame(columns=nba_cdn.GAME_COLUMNS)
    with pytest.raises(RuntimeError):
        collect_nba.rebuild_history(games, pd.DataFrame(columns=nba_cdn.LOG_COLUMNS), "2025-26", "2026-02-20")
