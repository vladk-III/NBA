"""NBA.com static CDN: season schedule and per-game team box scores (no API key)."""

import logging
import re
from datetime import datetime

from .config import GAME_TYPES, to_abbr
from .http import get_json

LOG = logging.getLogger("nba_collector")

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

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


def game_type(game_id: str) -> str | None:
    gid = str(game_id).zfill(10)
    return GAME_TYPES.get(gid[2])


def fetch_schedule() -> tuple[str, list[dict]]:
    """Return (season label, list of game dicts) for counted game types only."""
    data, _ = get_json(SCHEDULE_URL)
    if not data:
        raise RuntimeError("Could not download the NBA CDN schedule")
    league = data["leagueSchedule"]
    season = league.get("seasonYear")
    games = []
    for day in league.get("gameDates", []):
        for g in day.get("games", []):
            gid = str(g.get("gameId", "")).zfill(10)
            gtype = game_type(gid)
            if gtype is None:
                continue
            home, away = g.get("homeTeam") or {}, g.get("awayTeam") or {}
            home_abbr, away_abbr = to_abbr(home.get("teamTricode")), to_abbr(away.get("teamTricode"))
            if not home_abbr or not away_abbr:
                # Placeholder playoff slots ("TBD") until the matchup is set.
                continue
            status = int(g.get("gameStatus") or 1)
            games.append({
                "game_id": gid,
                "season": season,
                "game_type": gtype,
                "game_date": _et_date(g),
                "tipoff_utc": g.get("gameDateTimeUTC"),
                "home_team_abbreviation": home_abbr,
                "away_team_abbreviation": away_abbr,
                "game_status": status,
                "home_points": home.get("score") if status == 3 else None,
                "away_points": away.get("score") if status == 3 else None,
                "game_label": g.get("gameLabel") or "",
                "game_sub_label": g.get("gameSubLabel") or "",
                "postponed": str(g.get("postponedStatus") or "").upper() == "P",
            })
    LOG.info("Schedule %s: %d counted games", season, len(games))
    return season, games


def _et_date(g: dict) -> str:
    # gameDateEst is the Eastern calendar date with a fake "Z" suffix, e.g. 2025-10-21T00:00:00Z.
    raw = g.get("gameDateEst") or g.get("gameDateTimeEst") or ""
    if raw:
        return raw[:10]
    return datetime.strptime(g["gameDate"][:10], "%m/%d/%Y").strftime("%Y-%m-%d")


def fetch_boxscore(game_id: str) -> dict | None:
    """Raw box score JSON 'game' object, or None if not published / unreachable."""
    data, _ = get_json(BOXSCORE_URL.format(game_id=game_id), allow_404=True, attempts=3)
    if not data:
        return None
    return data.get("game")


def _num(stats: dict, *keys: str) -> float | None:
    for k in keys:
        v = stats.get(k)
        if v is not None:
            return float(v)
    return None


def team_turnovers(stats: dict) -> float | None:
    """Total team turnovers (player + team), matching NBA.com team game logs."""
    if stats.get("turnoversTotal") is not None:
        return float(stats["turnoversTotal"])
    tov = _num(stats, "turnovers")
    if tov is None:
        return None
    return tov + float(stats.get("turnoversTeam") or 0)


def parse_boxscore(game: dict, season: str) -> list[dict] | None:
    """Two team-game log rows from a final box score; None if the game is not final."""
    if int(game.get("gameStatus") or 0) != 3:
        return None
    gid = str(game["gameId"]).zfill(10)
    game_date = _box_et_date(game)
    rows = []
    sides = [("homeTeam", "awayTeam", True), ("awayTeam", "homeTeam", False)]
    for side, other, is_home in sides:
        team, opp = game[side], game[other]
        s = team.get("statistics") or {}
        pts, opp_pts = float(team["score"]), float(opp["score"])
        periods = team.get("periods") or []
        ot = sum(1 for p in periods if str(p.get("periodType", "")).upper() == "OVERTIME")
        if not periods:
            ot = max(int(game.get("period") or 4) - 4, 0)
        rows.append({
            "game_id": gid,
            "season": season,
            "game_type": game_type(gid),
            "game_date": game_date,
            "team": to_abbr(team.get("teamTricode")),
            "opponent": to_abbr(opp.get("teamTricode")),
            "is_home": is_home,
            "pts": pts,
            "opp_pts": opp_pts,
            "win": int(pts > opp_pts),
            "fgm": _num(s, "fieldGoalsMade"),
            "fga": _num(s, "fieldGoalsAttempted"),
            "fg3m": _num(s, "threePointersMade"),
            "fg3a": _num(s, "threePointersAttempted"),
            "ftm": _num(s, "freeThrowsMade"),
            "fta": _num(s, "freeThrowsAttempted"),
            "oreb": _num(s, "reboundsOffensive"),
            "dreb": _num(s, "reboundsDefensive"),
            "reb": _num(s, "reboundsTotal"),
            "ast": _num(s, "assists"),
            "stl": _num(s, "steals"),
            "blk": _num(s, "blocks"),
            "tov": team_turnovers(s),
            "pf": _num(s, "foulsPersonal"),
            "overtime_periods": ot,
        })
    return rows


def _box_et_date(game: dict) -> str:
    # gameEt is Eastern local time, e.g. 2025-10-21T19:30:00-04:00 or ...Z (fake UTC).
    et = game.get("gameEt")
    if et:
        return et[:10]
    code = game.get("gameCode", "")
    m = re.match(r"(\d{4})(\d{2})(\d{2})/", code)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    raise ValueError(f"No game date in box score {game.get('gameId')}")
