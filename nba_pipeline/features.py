"""Pre-game cumulative team stats, one row per game, in the historical CSV's format.

Every row only uses games completed before that game's date, so a row never contains
information from its own result. Averages are season-to-date over counted games
(regular season, play-in, playoffs). Shooting percentages are total makes over total
attempts, matching how the original historical CSV was built.
"""

import numpy as np
import pandas as pd

# The original 50 columns, in their original order.
LEGACY_COLUMNS = [
    "game_id", "game_date", "month_played", "home_team_abbreviation", "away_team_abbreviation",
    "home_points", "away_points", "outcome", "home_games", "away_games",
    "home_wins_current", "away_wins_current", "home_win_percentage", "away_win_percentage",
    "home_PTS/G", "away_PTS/G", "home_REB/G", "away_REB/G", "home_D_REB/G", "away_D_REB/G",
    "home_TO/G", "away_TO/G", "home_block", "away_block", "home_plus", "away_plus",
    "home_steal", "away_steal", "home_foul", "away_foul", "home_assist", "away_assist",
    "home_FTM/G", "away_FTM/G", "home_FTA/G", "away_FTA/G", "home_FGM/G", "away_FGM/G",
    "home_FGA/G", "away_FGA/G", "home_FG3M/G", "away_FG3M/G", "home_FG3A/G", "away_FG3A/G",
    "home_FG%", "away_FG%", "home_FG3%", "away_FG3%", "home_FT%", "away_FT%",
]

# Columns added by this pipeline. Rows from before the pipeline existed leave the
# stat columns blank (their game logs were not collected).
NEW_COLUMNS = [
    "season", "game_type",
    "home_OREB/G", "away_OREB/G", "home_OPP_PTS/G", "away_OPP_PTS/G",
    "home_POSS/G", "away_POSS/G", "home_rest_days", "away_rest_days",
    "home_back_to_back", "away_back_to_back",
]

COLUMNS = LEGACY_COLUMNS + NEW_COLUMNS

# Whole-number columns, kept as integers (blank when unknown) so the CSV never shows "125.0".
INT_COLUMNS = [
    "month_played", "home_points", "away_points", "outcome", "home_games", "away_games",
    "home_wins_current", "away_wins_current", "home_rest_days", "away_rest_days",
    "home_back_to_back", "away_back_to_back",
]

# Column suffix -> per-game log column averaged over prior games.
MEAN_STATS = {
    "PTS/G": "pts", "REB/G": "reb", "D_REB/G": "dreb", "TO/G": "tov", "block": "blk",
    "plus": "margin", "steal": "stl", "foul": "pf", "assist": "ast",
    "FTM/G": "ftm", "FTA/G": "fta", "FGM/G": "fgm", "FGA/G": "fga",
    "FG3M/G": "fg3m", "FG3A/G": "fg3a",
    "OREB/G": "oreb", "OPP_PTS/G": "opp_pts", "POSS/G": "poss",
}
RATIO_STATS = {"FG%": ("fgm", "fga"), "FG3%": ("fg3m", "fg3a"), "FT%": ("ftm", "fta")}


def season_from_date(game_date: str) -> str:
    year, month = int(game_date[:4]), int(game_date[5:7])
    start = year if month >= 8 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def team_snapshot(prior: pd.DataFrame) -> dict:
    """Season-to-date stats for one team from its prior game logs (may be empty)."""
    n = len(prior)
    out = {"games": n, "wins_current": int(prior["win"].sum()) if n else 0}
    out["win_percentage"] = out["wins_current"] / n if n else 0.0
    for col, src in MEAN_STATS.items():
        out[col] = float(prior[src].mean()) if n else 0.0
    for col, (made, att) in RATIO_STATS.items():
        attempts = prior[att].sum() if n else 0
        out[col] = float(prior[made].sum() / attempts) if attempts else 0.0
    return out


def build_rows(games: pd.DataFrame, logs: pd.DataFrame, through_date: str) -> pd.DataFrame:
    """Rows for every non-postponed game dated on or before through_date (ET)."""
    games = games[(games["game_date"] <= through_date)].copy()
    games = games[~(games["postponed"].astype(bool) & (games["game_status"] != 3))]

    logs = logs.copy()
    logs["margin"] = logs["pts"] - logs["opp_pts"]
    logs["poss"] = logs["fga"] - logs["oreb"] + logs["tov"] + 0.44 * logs["fta"]
    by_team = {t: df.sort_values("game_date") for t, df in logs.groupby(["season", "team"])}

    played = games[["season", "game_date", "home_team_abbreviation", "away_team_abbreviation"]]
    schedule_dates = {}
    for side in ("home_team_abbreviation", "away_team_abbreviation"):
        for (season, team), df in played.groupby(["season", side]):
            schedule_dates.setdefault((season, team), set()).update(df["game_date"])

    rows = []
    for g in games.sort_values(["game_date", "game_id"]).itertuples(index=False):
        final = g.game_status == 3 and pd.notna(g.home_points) and pd.notna(g.away_points)
        row = {
            "game_id": g.game_id,
            "game_date": g.game_date,
            "month_played": int(g.game_date[5:7]),
            "home_team_abbreviation": g.home_team_abbreviation,
            "away_team_abbreviation": g.away_team_abbreviation,
            "home_points": int(g.home_points) if final else np.nan,
            "away_points": int(g.away_points) if final else np.nan,
            "outcome": int(g.home_points > g.away_points) if final else np.nan,
            "season": g.season,
            "game_type": g.game_type,
        }
        for prefix, team in (("home", g.home_team_abbreviation), ("away", g.away_team_abbreviation)):
            team_logs = by_team.get((g.season, team))
            prior = team_logs[team_logs["game_date"] < g.game_date] if team_logs is not None else logs.iloc[0:0]
            for key, val in team_snapshot(prior).items():
                row[f"{prefix}_{key}"] = val
            earlier = [d for d in schedule_dates.get((g.season, team), ()) if d < g.game_date]
            if earlier:
                rest = (pd.Timestamp(g.game_date) - pd.Timestamp(max(earlier))).days
                row[f"{prefix}_rest_days"] = rest
                row[f"{prefix}_back_to_back"] = int(rest == 1)
            else:
                row[f"{prefix}_rest_days"] = np.nan
                row[f"{prefix}_back_to_back"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def merge_into_history(history: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Replace the rebuilt seasons in the history file; keep every other season untouched."""
    history = history.copy()
    for col in COLUMNS:
        if col not in history.columns:
            history[col] = np.nan
    for col in ("season", "game_type"):
        history[col] = history[col].astype(object)
    missing_season = history["season"].isna()
    history.loc[missing_season, "season"] = history.loc[missing_season, "game_date"].map(season_from_date)
    missing_type = history["game_type"].isna()
    history.loc[missing_type, "game_type"] = "regular"

    seasons = set(rebuilt["season"].dropna())
    kept = history[~history["season"].isin(seasons)]
    extra = [c for c in history.columns if c not in COLUMNS]
    frames = [f for f in (kept, rebuilt) if not f.empty]
    if not frames:
        return history[COLUMNS + extra]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    for col in INT_COLUMNS:
        combined[col] = pd.to_numeric(combined[col]).round().astype("Int64")
    return combined[COLUMNS + extra]
