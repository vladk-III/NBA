"""The Odds API: one call per snapshot for every US book's moneyline, spread and total.

Cost per call is 3 credits (3 markets x 1 region). Two snapshots per game day is
about 6 credits/day, well under the free tier's 500/month.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import EASTERN, to_abbr
from .http import get_json

LOG = logging.getLogger("nba_collector")

ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

COLUMNS = [
    "snapshot_utc", "snapshot_label", "game_date", "game_id", "odds_event_id", "commence_time_utc",
    "home_team_abbreviation", "away_team_abbreviation", "bookmaker", "bookmaker_title", "last_update",
    "ml_home", "ml_away", "spread_home", "spread_home_price", "spread_away", "spread_away_price",
    "total", "over_price", "under_price",
]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_snapshot(label: str, game_date: str, games: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, dict]:
    """Odds rows for games on game_date (ET) that have not tipped off yet."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is not set")

    day_start = datetime.fromisoformat(game_date).replace(tzinfo=EASTERN)
    day_end = day_start + timedelta(days=1)
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": _iso(max(now, day_start)),
        "commenceTimeTo": _iso(day_end),
    }
    events, resp = get_json(ODDS_URL, params=params, attempts=3)
    quota = {}
    if resp is not None:
        quota = {"remaining": resp.headers.get("x-requests-remaining"),
                 "used": resp.headers.get("x-requests-used")}
        LOG.info("Odds API credits: used=%s remaining=%s", quota["used"], quota["remaining"])
    if events is None:
        raise RuntimeError("The Odds API request failed")

    game_ids = {(r.game_date, r.home_team_abbreviation, r.away_team_abbreviation): r.game_id
                for r in games.itertuples(index=False)}
    snapshot_utc = _iso(now)
    rows = []
    for ev in events:
        home, away = to_abbr(ev.get("home_team")), to_abbr(ev.get("away_team"))
        commence = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        ev_date = commence.astimezone(EASTERN).strftime("%Y-%m-%d")
        if commence <= now or ev_date != game_date:
            continue
        gid = game_ids.get((ev_date, home, away))
        if gid is None:
            LOG.warning("Odds event %s @ %s on %s has no NBA schedule match", away, home, ev_date)
        for book in ev.get("bookmakers", []):
            row = {c: None for c in COLUMNS}
            row.update({
                "snapshot_utc": snapshot_utc, "snapshot_label": label, "game_date": ev_date,
                "game_id": gid, "odds_event_id": ev["id"], "commence_time_utc": ev["commence_time"],
                "home_team_abbreviation": home, "away_team_abbreviation": away,
                "bookmaker": book["key"], "bookmaker_title": book.get("title"),
                "last_update": book.get("last_update"),
            })
            for market in book.get("markets", []):
                for o in market.get("outcomes", []):
                    name = o.get("name", "")
                    side = "home" if to_abbr(name) == home else "away" if to_abbr(name) == away else None
                    if market["key"] == "h2h" and side:
                        row[f"ml_{side}"] = o.get("price")
                    elif market["key"] == "spreads" and side:
                        row[f"spread_{side}"] = o.get("point")
                        row[f"spread_{side}_price"] = o.get("price")
                    elif market["key"] == "totals":
                        if name.lower() == "over":
                            row["total"] = o.get("point")
                            row["over_price"] = o.get("price")
                        elif name.lower() == "under":
                            row["under_price"] = o.get("price")
            rows.append(row)
    LOG.info("Odds snapshot '%s': %d book rows across %d games", label, len(rows),
             len({r["odds_event_id"] for r in rows}))
    return pd.DataFrame(rows, columns=COLUMNS), quota
