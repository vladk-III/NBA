"""Paths, constants and team mappings shared across the pipeline."""

from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

DATA_DIR = Path("data")
GAMES_FILE = DATA_DIR / "games.csv"
TEAM_LOGS_FILE = DATA_DIR / "team_game_logs.csv"
HISTORICAL_FILE = DATA_DIR / "historical_nba_cumulative_stats.csv"
ODDS_FILE = DATA_DIR / "odds_snapshots.csv"
KALSHI_FILE = DATA_DIR / "kalshi_snapshots.csv"
MERGED_FILE = DATA_DIR / "nba_merged_data.csv"

# Game ID digit 3 encodes the game type: 1 preseason, 2 regular season, 3 All-Star,
# 4 playoffs, 5 play-in, 6 NBA Cup final. Preseason, All-Star and the Cup final
# (which does not count toward team records) are excluded.
GAME_TYPES = {"2": "regular", "4": "playoffs", "5": "play-in"}

TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Full-name spellings seen across providers -> NBA tricode.
FULL_NAME_TO_ABBR = {name: abbr for abbr, name in TEAMS.items()}
FULL_NAME_TO_ABBR.update({"LA Clippers": "LAC", "LA Lakers": "LAL"})

# Non-NBA tricodes some providers use -> NBA tricode.
ABBR_ALIASES = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "NOR": "NOP", "SA": "SAS", "PHO": "PHX",
    "UTAH": "UTA", "UTH": "UTA", "WSH": "WAS", "BRK": "BKN", "BRO": "BKN", "CHO": "CHA",
}


def to_abbr(name: str | None) -> str | None:
    """Normalize a full team name or provider tricode to the NBA tricode."""
    if not name:
        return None
    name = str(name).strip()
    if name in FULL_NAME_TO_ABBR:
        return FULL_NAME_TO_ABBR[name]
    code = name.upper()
    if code in TEAMS:
        return code
    return ABBR_ALIASES.get(code)


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"
