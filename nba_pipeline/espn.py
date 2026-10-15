"""ESPN's public site API: NBA schedule/scores and per-game team box scores (no key).

Used instead of NBA.com's CDN, which refuses requests from GitHub Actions runners.
ESPN's team box score fields match the original historical CSV's source exactly:
rebounds = totalRebounds, turnovers = turnovers (players only), fouls = fouls.
"""

import logging
from datetime import date, datetime, timedelta

from .config import EASTERN, season_label, to_abbr
from .http import get_json

LOG = logging.getLogger("nba_collector")

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SCOREBOARD_URL = f"{BASE}/scoreboard"
SUMMARY_URL = f"{BASE}/summary"

# ESPN season.type -> our game_type. 1 preseason and 4 off-season are excluded.
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}

GAME_COLUMNS = [
    "game_id", "season", "game_type", "game_date", "tipoff_utc",
    "home_team_abbreviation", "away_team_abbreviation",
    "game_status", "home_points", "away_points", "game_label", "game_sub_label", "postponed",
]

LOG_COLUMNS = [
    "game_id", "season", "game_type", "game_date", "team", "opponent", "is_home",
    "pts", "opp_pts", "win", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "overtime_periods",
]


def season_start_year(day: date) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _et_date(iso_utc: str) -> str:
    return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(EASTERN).strftime("%Y-%m-%d")


def _status(comp: dict) -> tuple[int, bool]:
    """(1 scheduled / 2 live / 3 final, postponed-or-canceled)."""
    st = (comp.get("status") or {}).get("type") or {}
    name = str(st.get("name", "")).upper()
    if "POSTPONED" in name or "CANCELED" in name or "SUSPENDED" in name:
        return 1, True
    if st.get("completed") or st.get("state") == "post":
        return 3, False
    return (2 if st.get("state") == "in" else 1), False


def parse_event(ev: dict) -> dict | None:
    """One schedule row from a scoreboard event, or None if it is not a counted NBA game."""
    season = ev.get("season") or {}
    gtype = SEASON_TYPES.get(int(season.get("type") or 0))
    if gtype is None:
        return None
    comp = (ev.get("competitions") or [{}])[0]
    notes = " ".join(n.get("headline", "") for n in comp.get("notes") or [])
    if "Cup" in notes and ("Championship" in notes or "Final" in notes):
        return None  # NBA Cup final: does not count toward team records
    sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
    home, away = sides.get("home"), sides.get("away")
    if not home or not away:
        return None
    home_abbr, away_abbr = to_abbr(home["team"].get("abbreviation")), to_abbr(away["team"].get("abbreviation"))
    if not home_abbr or not away_abbr:
        return None  # All-Star teams, TBD playoff slots
    status, postponed = _status(comp)
    final = status == 3
    tip = ev.get("date") or comp.get("date")
    return {
        "game_id": str(ev["id"]),
        "season": season_label(int(season["year"]) - 1),
        "game_type": gtype,
        "game_date": _et_date(tip),
        "tipoff_utc": tip,
        "home_team_abbreviation": home_abbr,
        "away_team_abbreviation": away_abbr,
        "game_status": status,
        "home_points": int(home["score"]) if final and home.get("score") not in (None, "") else None,
        "away_points": int(away["score"]) if final and away.get("score") not in (None, "") else None,
        "game_label": notes,
        "game_sub_label": "",
        "postponed": postponed,
    }


def fetch_games(start: date, end: date) -> list[dict]:
    """Counted games with an Eastern date between start and end, inclusive."""
    games, seen = [], set()
    # Ask for one extra day each side: the API's date range is not in Eastern time.
    cur, stop = start - timedelta(days=1), end + timedelta(days=1)
    while cur <= stop:
        chunk_end = min(cur + timedelta(days=30), stop)
        params = {"dates": f"{cur:%Y%m%d}-{chunk_end:%Y%m%d}", "limit": 1000}
        data, _ = get_json(SCOREBOARD_URL, params=params)
        if data is None:
            raise RuntimeError(f"Could not download the ESPN scoreboard for {params['dates']}")
        for ev in data.get("events", []):
            row = parse_event(ev)
            if row and row["game_id"] not in seen and start.isoformat() <= row["game_date"] <= end.isoformat():
                seen.add(row["game_id"])
                games.append(row)
        cur = chunk_end + timedelta(days=1)
    LOG.info("ESPN schedule %s to %s: %d counted games", start, end, len(games))
    return games


def fetch_boxscore(game_id: str) -> dict | None:
    """Raw ESPN game summary (box score + header), or None if unavailable."""
    data, _ = get_json(SUMMARY_URL, params={"event": game_id}, allow_404=True, attempts=3)
    return data


def _stat(stats: dict, name: str) -> float | None:
    v = stats.get(name)
    if v in (None, "", "-", "--"):
        return None
    return float(v)


def _pair(stats: dict, name: str) -> tuple[float | None, float | None]:
    v = stats.get(name)
    if not v or "-" not in v:
        return None, None
    made, att = v.split("-", 1)
    return float(made), float(att)


def parse_boxscore(summary: dict, season: str) -> list[dict] | None:
    """Two team-game log rows from a final game summary; None if not final."""
    header = summary.get("header") or {}
    comp = (header.get("competitions") or [{}])[0]
    status, _ = _status(comp)
    if status != 3:
        return None
    gid = str(header.get("id") or comp.get("id"))
    gtype = SEASON_TYPES.get(int((header.get("season") or {}).get("type") or 0))
    game_date = _et_date(comp["date"])
    scores = {c["homeAway"]: c for c in comp.get("competitors") or []}
    period = max((len(c.get("linescores") or []) for c in scores.values()), default=0)
    period = period or int(((comp.get("status") or {}).get("period")) or 4)
    teams = {t["homeAway"]: t for t in (summary.get("boxscore") or {}).get("teams") or []}
    if set(teams) != {"home", "away"}:
        return None

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        t = teams[side]
        s = {x["name"]: x.get("displayValue") for x in t.get("statistics") or []}
        fgm, fga = _pair(s, "fieldGoalsMade-fieldGoalsAttempted")
        fg3m, fg3a = _pair(s, "threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        ftm, fta = _pair(s, "freeThrowsMade-freeThrowsAttempted")
        pts, opp_pts = float(scores[side]["score"]), float(scores[other]["score"])
        rows.append({
            "game_id": gid,
            "season": season,
            "game_type": gtype,
            "game_date": game_date,
            "team": to_abbr(t["team"].get("abbreviation")),
            "opponent": to_abbr(teams[other]["team"].get("abbreviation")),
            "is_home": side == "home",
            "pts": pts,
            "opp_pts": opp_pts,
            "win": int(pts > opp_pts),
            "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
            "oreb": _stat(s, "offensiveRebounds"),
            "dreb": _stat(s, "defensiveRebounds"),
            "reb": _stat(s, "totalRebounds"),
            "ast": _stat(s, "assists"),
            "stl": _stat(s, "steals"),
            "blk": _stat(s, "blocks"),
            "tov": _stat(s, "turnovers"),
            "pf": _stat(s, "fouls"),
            "overtime_periods": max(period - 4, 0),
        })
    return rows
